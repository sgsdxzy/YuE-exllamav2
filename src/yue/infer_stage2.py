import copy
import os
from collections import Counter

import numpy as np
import torch
from codecmanipulator import CodecManipulator
from common import BlockTokenRangeProcessor, load_exl2_model, parser
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from mmtokenizer import _MMSentencePieceTokenizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LogitsProcessorList
from transformers.cache_utils import StaticCache
import math


def align(n, m):
    return ((n + m - 1) // m) * m


def split_bsz(bsz, maxbsz):
    n_sub_batches = math.ceil(bsz / maxbsz)
    base_size = bsz // n_sub_batches
    remainder = bsz % n_sub_batches
    sub_batch_sizes = [base_size + 1] * remainder + [base_size] * (n_sub_batches - remainder)
    indices = []
    start = 0
    for size in sub_batch_sizes:
        end = start + size
        indices.append((start, end))
        start = end
    return indices


class Stage2Pipeline:

    def __init__(
        self,
        device: torch.device,
    ):
        self.device = device

        self.codec_tool = CodecManipulator("xcodec", 0, 1)
        self.codec_tool_stage2 = CodecManipulator("xcodec", 0, 8)

        # Load tokenizer
        self.mmtokenizer = _MMSentencePieceTokenizer(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "mm_tokenizer_v0.2_hf",
                "tokenizer.model",
            )
        )


    def get_codec_ids(self, prompt: np.array):
        codec_ids = self.codec_tool.unflatten(prompt, n_quantizer=1)
        codec_ids = self.codec_tool.offset_tok_ids(
            codec_ids,
            global_offset = self.codec_tool.global_offset,
            codebook_size = self.codec_tool.codebook_size,
            num_codebooks = self.codec_tool.num_codebooks,
        ).astype(np.int32)
        return codec_ids


    def fix_output(self, output):
        # Fix invalid codes (a dirty solution, which may harm the quality of audio)
        # We are trying to find better one
        fixed_output = copy.deepcopy(output)
        for i, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequant = sorted(counter.items(), key = lambda x: x[1], reverse = True)[0][0]
                    fixed_output[i, j] = most_frequant
        return fixed_output


    def save(
        self,
        output_dir: str,
        outputs,
    ):
        for output_name, output in outputs.items():
            # save output
            stage2_output_dir = os.path.join(output_dir, "stage2")
            os.makedirs(stage2_output_dir, exist_ok = True)
            output_filename = os.path.join(stage2_output_dir, output_name)
            np.save(output_filename, output)


    def get_stage1_prompt(self, output_dir: str, output_name: str):
        stage1_output_dir = os.path.join(output_dir, "stage1")
        prompt = np.load(os.path.join(stage1_output_dir, output_name)).astype(np.int32)
        return prompt


    def prepare_prompt_batch(self, prompt: np.array, batch_size: int):

        codec_ids = self.get_codec_ids(prompt)

        # Prepare prompt_ids based on batch size or single input
        if batch_size > 1:
            codec_list = []
            for i in range(batch_size):
                idx_begin = i * 300
                idx_end = (i + 1) * 300
                codec_list.append(codec_ids[:, idx_begin:idx_end])

            codec_ids = np.concatenate(codec_list, axis=0)
            prompt_ids = np.concatenate(
                [
                    np.tile([self.mmtokenizer.soa, self.mmtokenizer.stage_1], (batch_size, 1)),
                    codec_ids,
                    np.tile([self.mmtokenizer.stage_2], (batch_size, 1)),
                ],
                axis=1,
            )
        else:
            prompt_ids = np.concatenate(
                [
                    np.array([self.mmtokenizer.soa, self.mmtokenizer.stage_1]),
                    codec_ids.flatten(),  # Flatten the 2D array to 1D
                    np.array([self.mmtokenizer.stage_2]),
                ]
            ).astype(np.int32)
            prompt_ids = prompt_ids[np.newaxis, ...]

        codec_ids = torch.as_tensor(codec_ids, dtype=torch.long)
        prompt_ids = torch.as_tensor(prompt_ids, dtype=torch.long)
        return codec_ids, prompt_ids


class Stage2Pipeline_HF(Stage2Pipeline):

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        batch_size: int,
    ):
        super().__init__(device)
        self.batch_size = batch_size

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype = torch.float16,
            attn_implementation = "sdpa"
        )
        self.model.to(device)
        self.model.eval()


    def generate_batch(
        self,
        prompt: np.array,
        batch_size: int,
    ):
        codec_ids, prompt_ids = self.prepare_prompt_batch(prompt, batch_size)
        len_prompt = prompt_ids.shape[-1]

        # Teacher forcing generate loop
        codec_ids = codec_ids.to(self.device)
        prompt_ids = prompt_ids.to(self.device)
        block_list = LogitsProcessorList([
            BlockTokenRangeProcessor(0, 46358),
            BlockTokenRangeProcessor(53526, self.mmtokenizer.vocab_size)]
        )
        past_key_values = StaticCache(
            self.model.config,
            max_batch_size = batch_size,
            max_cache_len = prompt_ids.shape[1] + codec_ids.shape[1] * 8,
            device = self.model.device,
            dtype = self.model.dtype
        )
        for frames_idx in range(codec_ids.shape[1]):
            cb0 = codec_ids[:, frames_idx : frames_idx + 1]
            prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
            input_ids = prompt_ids

            stage2_output = self.model.generate(
                input_ids = input_ids,
                min_new_tokens = 7,
                max_new_tokens = 7,
                eos_token_id = self.mmtokenizer.eoa,
                pad_token_id = self.mmtokenizer.eoa,
                logits_processor = block_list,
                past_key_values = past_key_values,
            )

            assert stage2_output.shape[1] - prompt_ids.shape[1] == 7, \
                f"output new tokens={stage2_output.shape[1]-prompt_ids.shape[1]}"
            prompt_ids = stage2_output

        # Return output based on batch size
        if batch_size > 1:
            output = prompt_ids.cpu().numpy()[:, len_prompt:]
            output_list = [output[i] for i in range(batch_size)]
            output = np.concatenate(output_list, axis=0)
        else:
            output = prompt_ids[0].cpu().numpy()[len_prompt:]

        return output


    def generate(
        self,
        output_dir: str,
    ) -> dict[str, np.array]:
        outputs = {}
        for output_name in tqdm(["cot_vocal.npy", "cot_instrumental.npy"]):
            # Load the prompt
            prompt = self.get_stage1_prompt(output_dir, output_name)

            # Only accept 6s segments
            output_duration = prompt.shape[-1] // 50 // 6 * 6
            num_batch = output_duration // 6

            if num_batch <= self.batch_size:
                # If num_batch is less than or equal to batch_size, we can infer the entire prompt at once
                output = self.generate_batch(prompt[:, : output_duration * 50], batch_size = num_batch)
            else:
                # If num_batch is greater than batch_size, process in chunks of batch_size
                segments = []
                num_segments = (num_batch // self.batch_size) + (1 if num_batch % self.batch_size != 0 else 0)

                for seg in range(num_segments):
                    start_idx = seg * self.batch_size * 300
                    # Ensure the end_idx does not exceed the available length
                    end_idx = min((seg + 1) * self.batch_size * 300, output_duration * 50)  # Adjust the last segment
                    current_batch_size = self.batch_size if seg != num_segments - 1 or num_batch % self.batch_size == 0 else num_batch % self.batch_size
                    segment = self.generate_batch(prompt[:, start_idx:end_idx], batch_size = current_batch_size)
                    segments.append(segment)

                # Concatenate all the segments
                output = np.concatenate(segments, axis = 0)

            # Process the ending part of the prompt
            if output_duration * 50 != prompt.shape[-1]:
                ending = self.generate_batch(prompt[:, output_duration * 50:], batch_size = 1)
                output = np.concatenate([output, ending], axis = 0)

            output = self.codec_tool_stage2.ids2npy(output)

            output = self.fix_output(output)
            outputs[output_name] = output
        return outputs


class Stage2Pipeline_EXL2(Stage2Pipeline):

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        cache_size: int,
    ):
        super().__init__(device)

        self.cache_size = cache_size

        assert device != "cpu", \
            "ExLlamaV2 does not support CPU inference."

        # Load EXL2 model
        device_idx = self.device.index
        gpu_split = [0] * torch.cuda.device_count()
        gpu_split[device_idx] = 9999
        exl2_config = ExLlamaV2Config(model_path)
        self.model = ExLlamaV2(exl2_config)
        self.model.load(gpu_split)

        # Move embedding layer to GPU to avoid CPU sync during argmax gen loop
        self.model.modules[0].device_idx = self.model.modules[1].device_idx
        self.model.modules[0].reload()

        # Load tokenizer (only needed for vocab size in disallow_tokens)
        self.tokenizer = ExLlamaV2Tokenizer(exl2_config)


    def generate(
        self,
        output_dir: str,
    ) -> dict[str, np.array]:

        parts = ["cot_vocal.npy", "cot_instrumental.npy"]
        full_batch = []

        # Collect up to 300 token (6s) segments for all parts
        for output_idx, output_name in tqdm(enumerate(parts)):
            prompt = self.get_stage1_prompt(output_dir, output_name)
            prompt = self.get_codec_ids(prompt)
            prompt = torch.as_tensor(prompt, dtype = torch.long)

            segs = torch.split(prompt, 300, dim = -1)

            for seg_idx, seg in enumerate(segs):
                seg_len = seg.shape[-1]
                full_batch.append((seg_len, seg_idx, output_idx, seg))

        # Prepare segments
        prefix = torch.tensor([[self.mmtokenizer.soa, self.mmtokenizer.stage_1]], dtype = torch.long)
        suffix = torch.tensor([[self.mmtokenizer.stage_2]], dtype = torch.long)
        for i in range(len(full_batch)):
            seg_len, seg_idx, output_idx, codec_ids = full_batch[i]
            prompt_ids = torch.cat((prefix, codec_ids, suffix), dim = -1)
            full_batch[i] = (seg_len, seg_idx, output_idx, codec_ids, prompt_ids)

        # Group prompts by length
        batches = {}
        for seq in full_batch:
            if not seq[0] in batches:
                batches[seq[0]] = []
            batches[seq[0]].append(seq)

        # Split into on minibatches
        split_batch = []
        for idx, (seg_len, batch) in enumerate(batches.items()):
            b_seg_order = [b[1] for b in batch]
            b_part_order = [b[2] for b in batch]
            b_codec_ids = torch.cat([b[3] for b in batch], dim = 0)
            b_prompt_ids = torch.cat([b[4] for b in batch], dim = 0)

            max_bsz = self.cache_size // align(b_prompt_ids.shape[1] + b_codec_ids.shape[1] * 8, 32)
            assert max_bsz > 0
            for (a, b) in split_bsz(b_prompt_ids.shape[0], max_bsz):
                split_batch.append((
                    b_seg_order[a:b],
                    b_part_order[a:b],
                    b_codec_ids[a:b],
                    b_prompt_ids[a:b]
                ))

        # Inference
        output_parts = []
        for _ in parts:
            output_parts.append([])

        for (seg_order, part_order, codec_ids, prompt_ids) in tqdm(split_batch):
            codec_ids = codec_ids.to(self.device)
            prompt_ids = prompt_ids.to(self.device)
            batch_size, len_prompt = prompt_ids.shape

            cache = ExLlamaV2Cache(
                self.model,
                batch_size = batch_size,
                max_seq_len = align(prompt_ids.shape[1] + codec_ids.shape[1] * 8, 32)
            )
            output_ids = torch.empty((batch_size, 0), dtype = torch.long, device = self.device)

            for frames_idx in tqdm(range(codec_ids.shape[1])):
                cb0 = codec_ids[:, frames_idx : frames_idx + 1]

                # Append the initial prompt to the first codec frame
                if frames_idx == 0:
                    cb0 = torch.cat([prompt_ids, cb0], dim = -1)

                # Forward prompt
                output_ids = torch.cat((output_ids, cb0), dim = -1)
                logits = self.model.forward(
                    cb0,
                    cache = cache,
                    last_id_only = True
                )

                for i in range(7):

                    # Slice logits instead of biasing start and end of distribution
                    first_logit = 46358
                    last_logit = 53526
                    logits = logits[:, :, first_logit : last_logit]

                    # Greedy sampling
                    sample = logits.argmax(dim = -1) + first_logit
                    output_ids = torch.cat((output_ids, sample), dim = -1)

                    # TODO: Here, original asserts that we didn't sample mmtokenizer.eoa (can we just mask it out?)

                    # Forward sample
                    logits = self.model.forward(
                        sample,
                        cache = cache,
                    )

            # Trim prompt
            output_ids = output_ids[:, len_prompt:]

            # Split outputs
            for i in range(batch_size):
                output_parts[part_order[i]].append((seg_order[i], output_ids[i:i+1, :]))

        # Unshuffle and recombine output parts
        output = {}
        for i, p in enumerate(output_parts):
            p = sorted(p, key = lambda x: x[0])
            part_o = torch.cat([pp[1] for pp in p], dim = -1).flatten().cpu().numpy()
            part_o = self.codec_tool_stage2.ids2npy(part_o)
            part_o = self.fix_output(part_o)
            output[parts[i]] = part_o

        return output


def main():
    args = parser.parse_args()
    device = torch.device(f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")

    if args.stage2_use_exl2:
        pipeline = Stage2Pipeline_EXL2(
            model_path = args.stage2_model,
            device = device,
            cache_size = args.stage2_cache_size,
        )
        pass
    else:
        pipeline = Stage2Pipeline_HF(
            model_path = args.stage2_model,
            device = device,
            batch_size = args.stage2_batch_size,
        )

    outputs = pipeline.generate(
        output_dir = args.output_dir,
    )

    pipeline.save(
        output_dir = args.output_dir,
        outputs = outputs
    )


if __name__ == "__main__":
    # enable inference mode globally
    torch.autograd.grad_mode._enter_inference_mode(True)
    torch.autograd.set_grad_enabled(False)
    main()
import copy
import os
from collections import Counter

import numpy as np
import torch
from codecmanipulator import CodecManipulator
from common import BlockTokenRangeProcessor, load_exl2_model, parser
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
from mmtokenizer import _MMSentencePieceTokenizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LogitsProcessorList
from transformers.cache_utils import StaticCache


def stage2_generate(
    model: ExLlamaV2DynamicGenerator | AutoModelForCausalLM,
    prompt: np.ndarray,
    mmtokenizer: _MMSentencePieceTokenizer,
    codec_tool: CodecManipulator,
    batch_size: int = 16,
):
    codec_ids = codec_tool.unflatten(prompt, n_quantizer=1)
    codec_ids = codec_tool.offset_tok_ids(
        codec_ids,
        global_offset=codec_tool.global_offset,
        codebook_size=codec_tool.codebook_size,
        num_codebooks=codec_tool.num_codebooks,
    ).astype(np.int32)

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
                np.tile([mmtokenizer.soa, mmtokenizer.stage_1], (batch_size, 1)),
                codec_ids,
                np.tile([mmtokenizer.stage_2], (batch_size, 1)),
            ],
            axis=1,
        )
    else:
        prompt_ids = np.concatenate(
            [
                np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
                codec_ids.flatten(),  # Flatten the 2D array to 1D
                np.array([mmtokenizer.stage_2]),
            ]
        ).astype(np.int32)
        prompt_ids = prompt_ids[np.newaxis, ...]

    codec_ids = torch.as_tensor(codec_ids, dtype=torch.long)
    prompt_ids = torch.as_tensor(prompt_ids, dtype=torch.long)
    len_prompt = prompt_ids.shape[-1]

    # Teacher forcing generate loop
    if isinstance(model, ExLlamaV2DynamicGenerator):
        gen_settings = ExLlamaV2Sampler.Settings(top_k=1)
        gen_settings.disallow_tokens(model.tokenizer, list(range(0, 46358)) + list(range(53526, mmtokenizer.vocab_size)))
    else:
        codec_ids = codec_ids.to(model.device)
        prompt_ids = prompt_ids.to(model.device)
        block_list = LogitsProcessorList([BlockTokenRangeProcessor(0, 46358), BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size)])
        past_key_values = StaticCache(
            model.config, max_batch_size=batch_size, max_cache_len=prompt_ids.shape[1] + codec_ids.shape[1] * 8, device=model.device, dtype=model.dtype
        )
    for frames_idx in range(codec_ids.shape[1]):
        cb0 = codec_ids[:, frames_idx : frames_idx + 1]
        prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
        input_ids = prompt_ids

        if isinstance(model, ExLlamaV2DynamicGenerator):
            split_ids = list(input_ids.split(1, dim=0))
            stage2_output = split_ids
            for idx, input_id in enumerate(input_ids):
                job = ExLlamaV2DynamicJob(input_ids=input_id.unsqueeze(0), min_new_tokens=7, max_new_tokens=7, gen_settings=gen_settings, identifier=idx)
                model.enqueue(job)
            while model.num_remaining_jobs():
                results = model.iterate()
                for result in results:
                    if result["stage"] != "streaming":
                        continue
                    new_token_ids = result.get("token_ids", None)
                    if new_token_ids is not None:
                        idx = result["identifier"]
                        stage2_output[idx] = torch.cat((stage2_output[idx], new_token_ids), dim=-1)
            stage2_output = torch.cat(stage2_output, dim=0)
        else:
            stage2_output = model.generate(
                input_ids=input_ids,
                min_new_tokens=7,
                max_new_tokens=7,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=block_list,
                past_key_values=past_key_values,
            )

        assert stage2_output.shape[1] - prompt_ids.shape[1] == 7, f"output new tokens={stage2_output.shape[1]-prompt_ids.shape[1]}"
        prompt_ids = stage2_output

    # Return output based on batch size
    if batch_size > 1:
        output = prompt_ids.cpu().numpy()[:, len_prompt:]
        output_list = [output[i] for i in range(batch_size)]
        output = np.concatenate(output_list, axis=0)
    else:
        output = prompt_ids[0].cpu().numpy()[len_prompt:]

    return output


def stage2_save(
    model: ExLlamaV2DynamicGenerator | AutoModelForCausalLM,
    mmtokenizer: _MMSentencePieceTokenizer,
    codec_tool: CodecManipulator,
    codec_tool_stage2: CodecManipulator,
    output_dir: str,
    batch_size: int = 4,
):
    stage1_output_dir = os.path.join(output_dir, "stage1")
    for output_name in tqdm(["cot_vocal.npy", "cot_instrumental.npy"]):
        # Load the prompt
        prompt = np.load(os.path.join(stage1_output_dir, output_name)).astype(np.int32)

        # Only accept 6s segments
        output_duration = prompt.shape[-1] // 50 // 6 * 6
        num_batch = output_duration // 6

        if num_batch <= batch_size:
            # If num_batch is less than or equal to batch_size, we can infer the entire prompt at once
            output = stage2_generate(model, prompt[:, : output_duration * 50], mmtokenizer, codec_tool, batch_size=num_batch)
        else:
            # If num_batch is greater than batch_size, process in chunks of batch_size
            segments = []
            num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)

            for seg in range(num_segments):
                start_idx = seg * batch_size * 300
                # Ensure the end_idx does not exceed the available length
                end_idx = min((seg + 1) * batch_size * 300, output_duration * 50)  # Adjust the last segment
                current_batch_size = batch_size if seg != num_segments - 1 or num_batch % batch_size == 0 else num_batch % batch_size
                segment = stage2_generate(model, prompt[:, start_idx:end_idx], mmtokenizer, codec_tool, batch_size=current_batch_size)
                segments.append(segment)

            # Concatenate all the segments
            output = np.concatenate(segments, axis=0)

        # Process the ending part of the prompt
        if output_duration * 50 != prompt.shape[-1]:
            ending = stage2_generate(model, prompt[:, output_duration * 50 :], mmtokenizer, codec_tool, batch_size=1)
            output = np.concatenate([output, ending], axis=0)
        output = codec_tool_stage2.ids2npy(output)

        # Fix invalid codes (a dirty solution, which may harm the quality of audio)
        # We are trying to find better one
        fixed_output = copy.deepcopy(output)
        for i, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequant = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                    fixed_output[i, j] = most_frequant
        # save output
        stage2_output_dir = os.path.join(output_dir, "stage2")
        os.makedirs(stage2_output_dir, exist_ok=True)
        output_filename = os.path.join(stage2_output_dir, output_name)
        np.save(output_filename, fixed_output)


def main():
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
    if args.stage2_use_exl2:
        model = load_exl2_model(args.stage2_model, args.stage2_cache_size)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.stage2_model, torch_dtype=torch.float16, attn_implementation="sdpa")
        model.to(device)
        model.eval()

    mmtokenizer = _MMSentencePieceTokenizer(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "mm_tokenizer_v0.2_hf",
            "tokenizer.model",
        )
    )
    codec_tool = CodecManipulator("xcodec", 0, 1)
    codec_tool_stage2 = CodecManipulator("xcodec", 0, 8)

    stage2_save(model, mmtokenizer, codec_tool, codec_tool_stage2, args.output_dir, args.stage2_batch_size)


if __name__ == "__main__":
    # enable inference mode globally
    torch.autograd.grad_mode._enter_inference_mode(True)
    torch.autograd.set_grad_enabled(False)
    main()

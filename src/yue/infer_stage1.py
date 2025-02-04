import os
import random
import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from codecmanipulator import CodecManipulator
from common import BlockTokenRangeProcessor, parser, seed_everything, get_cache_class
from einops import rearrange
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2Sampler
from mmtokenizer import _MMSentencePieceTokenizer
from models.soundstream_hubert_new import SoundStream
from omegaconf import OmegaConf
from torchaudio.transforms import Resample
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LogitsProcessorList
from transformers.cache_utils import StaticCache


@dataclass
class SampleSettings:
    # Here is suggested decoding config
    top_p = 0.93
    temperature = 1
    repetition_penalty = 1.0
    guidance_scale_seg0 = 1.5  # None to disable cfg
    guidance_scale = 1.2  # None to disable cfg

    def __init__(self, use_guidance: bool = True):
        if not use_guidance:
            self.guidance_scale_seg0 = None
            self.guidance_scale = None


def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio


def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
    if len(audio_prompt.shape) < 3:
        audio_prompt.unsqueeze_(0)
    with torch.no_grad():
        raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
    raw_codes = raw_codes.transpose(0, 1)
    raw_codes = raw_codes.cpu().numpy().astype(np.int16)
    return raw_codes


class Stage1Pipeline:

    def __init__(self, device: torch.device, basic_model_config: str, resume_path: str):
        self.device = device
        self.codec_tool = CodecManipulator("xcodec", 0, 1)
        self.basic_model_config = basic_model_config
        self.resume_path = resume_path
        self.codec_model = None

        # Load tokenizer
        self.mmtokenizer = _MMSentencePieceTokenizer(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mm_tokenizer_v0.2_hf", "tokenizer.model"))
        self.start_of_segment = self.mmtokenizer.tokenize("[start_of_segment]")
        self.end_of_segment = self.mmtokenizer.tokenize("[end_of_segment]")

    def load_codec_model(self):
        if self.codec_model is not None:
            return
        model_config = OmegaConf.load(self.basic_model_config)
        assert model_config.generator.name == "SoundStream"
        self.codec_model = SoundStream(**model_config.generator.config).to(self.device)
        parameter_dict = torch.load(self.resume_path, map_location=self.device, weights_only=False)
        self.codec_model.load_state_dict(parameter_dict["codec_model"])
        self.codec_model.eval()

    def get_prompt_texts(self, genres: str, lyrics: str):
        def split_lyrics(lyrics):
            pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
            segments = re.findall(pattern, lyrics, re.DOTALL)
            structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
            return structured_lyrics

        lyrics = split_lyrics(lyrics)
        full_lyrics = "\n".join(lyrics)
        prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
        prompt_texts += lyrics
        return lyrics, prompt_texts

    def get_audio_prompt_ids(
        self,
        use_dual_tracks_prompt: bool,
        vocal_track_prompt_path: str,
        instrumental_track_prompt_path: str,
        use_audio_prompt: bool,
        audio_prompt_path: str,
        prompt_start_time: int,
        prompt_end_time: int,
    ):
        self.load_codec_model()
        if use_dual_tracks_prompt:
            vocals_ids = load_audio_mono(vocal_track_prompt_path)
            instrumental_ids = load_audio_mono(instrumental_track_prompt_path)
            vocals_ids = encode_audio(self.codec_model, vocals_ids, self.device, target_bw=0.5)
            instrumental_ids = encode_audio(self.codec_model, instrumental_ids, self.device, target_bw=0.5)
            vocals_ids = self.codec_tool.npy2ids(vocals_ids[0])
            instrumental_ids = self.codec_tool.npy2ids(instrumental_ids[0])
            ids_segment_interleaved = rearrange([np.array(vocals_ids), np.array(instrumental_ids)], "b n -> (n b)")
            audio_prompt_codec = ids_segment_interleaved[int(prompt_start_time * 50 * 2) : int(prompt_end_time * 50 * 2)]
            audio_prompt_codec = audio_prompt_codec.tolist()
        elif use_audio_prompt:
            audio_prompt = load_audio_mono(audio_prompt_path)
            raw_codes = encode_audio(self.codec_model, audio_prompt, self.device, target_bw=0.5)
            # Format audio prompt
            code_ids = self.codec_tool.npy2ids(raw_codes[0])
            audio_prompt_codec = code_ids[int(prompt_start_time * 50) : int(prompt_end_time * 50)]  # 50 is tps of xcodec
        audio_prompt_codec_ids = [self.mmtokenizer.soa] + self.codec_tool.sep_ids + audio_prompt_codec + [self.mmtokenizer.eoa]
        sentence_ids = self.mmtokenizer.tokenize("[start_of_reference]") + audio_prompt_codec_ids + self.mmtokenizer.tokenize("[end_of_reference]")
        return sentence_ids

    def get_first_segment_prompt(
        self,
        segment_p: str,
        prompt_text_0: str,
        use_dual_tracks_prompt: bool,
        vocal_track_prompt_path: str,
        instrumental_track_prompt_path: str,
        use_audio_prompt: bool,
        audio_prompt_path: str,
        prompt_start_time: int,
        prompt_end_time: int,
    ):
        section_text = segment_p.replace("[start_of_segment]", "").replace("[end_of_segment]", "")
        head_id = self.mmtokenizer.tokenize(prompt_text_0)
        if use_dual_tracks_prompt or use_audio_prompt:
            head_id += self.get_audio_prompt_ids(
                use_dual_tracks_prompt,
                vocal_track_prompt_path,
                instrumental_track_prompt_path,
                use_audio_prompt,
                audio_prompt_path,
                prompt_start_time,
                prompt_end_time,
            )
        return head_id + self.start_of_segment + self.mmtokenizer.tokenize(section_text) + [self.mmtokenizer.soa] + self.codec_tool.sep_ids

    def get_segment_prompt(self, segment_p: str):
        section_text = segment_p.replace("[start_of_segment]", "").replace("[end_of_segment]", "")
        return self.end_of_segment + self.start_of_segment + self.mmtokenizer.tokenize(section_text) + [self.mmtokenizer.soa] + self.codec_tool.sep_ids

    def save(self, raw_output: torch.Tensor, output_dir: str, use_audio_prompt: bool, use_dual_tracks_prompt: bool):
        # save raw output and check sanity
        ids = raw_output[0].cpu().numpy()
        soa_idx = np.where(ids == self.mmtokenizer.soa)[0].tolist()
        eoa_idx = np.where(ids == self.mmtokenizer.eoa)[0].tolist()
        if len(soa_idx) != len(eoa_idx):
            raise ValueError(f"invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}")

        vocals = []
        instrumentals = []
        range_begin = 1 if use_audio_prompt or use_dual_tracks_prompt else 0
        for i in range(range_begin, len(soa_idx)):
            codec_ids = ids[soa_idx[i] + 1 : eoa_idx[i]]
            if codec_ids[0] == 32016:
                codec_ids = codec_ids[1:]
            codec_ids = codec_ids[: 2 * (codec_ids.shape[0] // 2)]
            vocals_ids = self.codec_tool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
            vocals.append(vocals_ids)
            instrumentals_ids = self.codec_tool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
            instrumentals.append(instrumentals_ids)
        vocals = np.concatenate(vocals, axis=1)
        instrumentals = np.concatenate(instrumentals, axis=1)
        stage1_output_dir = os.path.join(output_dir, "stage1")
        os.makedirs(stage1_output_dir, exist_ok=True)
        vocal_save_path = os.path.join(stage1_output_dir, "vtrack.npy")
        inst_save_path = os.path.join(stage1_output_dir, "itrack.npy")
        np.save(vocal_save_path, vocals)
        np.save(inst_save_path, instrumentals)


class Stage1Pipeline_HF(Stage1Pipeline):

    def __init__(self, model_path: str, device: torch.device, cache_size: int, **kwargs):
        super().__init__(device, **kwargs)

        # Load HF model
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation="sdpa", device_map=self.device)
        self.model.eval()
        if torch.__version__ >= "2.0.0":
            self.model = torch.compile(self.model)
        self.cache_size = cache_size

    def generate(
        self,
        use_dual_tracks_prompt: bool,
        vocal_track_prompt_path: str,
        instrumental_track_prompt_path: str,
        use_audio_prompt: bool,
        audio_prompt_path: str,
        genres: str,
        lyrics: str,
        run_n_segments: int,
        max_new_tokens: int,
        prompt_start_time: int,
        prompt_end_time: int,
        sample_settings: SampleSettings,
    ) -> torch.Tensor:

        lyrics, prompt_texts = self.get_prompt_texts(genres, lyrics)
        run_n_segments = min(run_n_segments, len(lyrics))

        for i in tqdm(range(run_n_segments)):

            # Get prompt
            if i == 0:
                prompt_ids = self.get_first_segment_prompt(
                    prompt_texts[1],
                    prompt_texts[0],
                    use_dual_tracks_prompt,
                    vocal_track_prompt_path,
                    instrumental_track_prompt_path,
                    use_audio_prompt,
                    audio_prompt_path,
                    prompt_start_time,
                    prompt_end_time,
                )
            else:
                prompt_ids = self.get_segment_prompt(prompt_texts[i + 1])
            prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(self.device)
            input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 0 else prompt_ids

            # Use window slicing in case output sequence exceeds the context of model
            max_context = self.cache_size - max_new_tokens - 1
            if input_ids.shape[-1] > max_context:
                print(f"Section {i}: output length {input_ids.shape[-1]} exceeding context length {max_context}, " f"now using the last {max_context} tokens.")
                input_ids = input_ids[:, -max_context:]

            past_key_values = StaticCache(
                self.model.config, max_batch_size=1, max_cache_len=input_ids.shape[-1] + max_new_tokens, device=self.model.device, dtype=self.model.dtype
            )

            processors = LogitsProcessorList([BlockTokenRangeProcessor(0, 32002), BlockTokenRangeProcessor(32016, 32016)])

            output_seq = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=100,
                do_sample=True,
                top_p=sample_settings.top_p,
                temperature=sample_settings.temperature,
                repetition_penalty=sample_settings.repetition_penalty,
                eos_token_id=self.mmtokenizer.eoa,
                pad_token_id=self.mmtokenizer.eoa,
                logits_processor=processors,
                guidance_scale=sample_settings.guidance_scale_seg0 if i == 0 else sample_settings.guidance_scale,
                past_key_values=past_key_values,
            )

            if output_seq[0][-1].item() != self.mmtokenizer.eoa:
                tensor_eoa = torch.tensor([[self.mmtokenizer.eoa]], dtype=torch.long, device=output_seq.device)
                output_seq = torch.cat((output_seq, tensor_eoa), dim=1)
            if i > 0:
                raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1] :]], dim=1)
            else:
                raw_output = output_seq
        return raw_output


class Stage1Pipeline_EXL2(Stage1Pipeline):

    def __init__(self, model_path: str, device: torch.device, cache_size: int, cache_mode: str, **kwargs):
        super().__init__(device, **kwargs)

        assert device != "cpu", "ExLlamaV2 does not support CPU inference."

        # Load EXL2 model
        device_idx = self.device.index
        gpu_split = [0] * torch.cuda.device_count()
        gpu_split[device_idx] = 9999
        exl2_config = ExLlamaV2Config(model_path)
        exl2_config.no_sdpa = True  # TODO: Figure out why SDPA slows to a crawl when given custom attn mask
        self.model = ExLlamaV2(exl2_config)
        self.model.load(gpu_split)

        # Load tokenizer (only needed for vocab size in disallow_tokens)
        self.tokenizer = ExLlamaV2Tokenizer(exl2_config)

        # Define cache
        self.cache_size = cache_size
        self.cache_mode = get_cache_class(cache_mode)

        # TODO: Output layer could be trimmed here to avoid masking out the first 32k tokens during generation

    def generate(
        self,
        use_dual_tracks_prompt: bool,
        vocal_track_prompt_path: str,
        instrumental_track_prompt_path: str,
        use_audio_prompt: bool,
        audio_prompt_path: str,
        genres: str,
        lyrics: str,
        run_n_segments: int,
        max_new_tokens: int,
        prompt_start_time: int,
        prompt_end_time: int,
        sample_settings: SampleSettings,
    ) -> torch.Tensor:

        if sample_settings.guidance_scale_seg0 is None:
            bsz = 1
            cfg = False
            position_offsets = None
            input_mask = None
        else:
            bsz = 2
            cfg = True

        lyrics, prompt_texts = self.get_prompt_texts(genres, lyrics)
        run_n_segments = min(run_n_segments, len(lyrics))

        # Cache for the whole output sequence
        cache = self.cache_mode(self.model, batch_size=bsz, max_seq_len=self.cache_size)

        # Collect output here
        seq = torch.empty((bsz, 0), dtype=torch.long)

        # Sample settings
        gen_settings = ExLlamaV2Sampler.Settings(
            top_k=0, top_p=sample_settings.top_p, token_repetition_penalty=sample_settings.repetition_penalty, temperature=sample_settings.temperature
        )
        gen_settings.allow_tokens(self.tokenizer, [32002] + list(range(45334, 56722)))

        # RNG for sampling, could seed here
        rng = random.Random()

        for i in tqdm(range(run_n_segments)):

            # Get prompt for this segment
            if i == 0:
                prompt_ids = self.get_first_segment_prompt(
                    prompt_texts[1],
                    prompt_texts[0],
                    use_dual_tracks_prompt,
                    vocal_track_prompt_path,
                    instrumental_track_prompt_path,
                    use_audio_prompt,
                    audio_prompt_path,
                    prompt_start_time,
                    prompt_end_time,
                )
            else:
                prompt_ids = self.get_segment_prompt(prompt_texts[i + 1])
            prompt_ids = torch.tensor([prompt_ids] * bsz, dtype=torch.long)

            # Accept prompt tokens
            seq = torch.cat((seq, prompt_ids), dim=-1)

            # Use window slicing in case output sequence exceeds the context of model
            max_context = self.cache_size - max_new_tokens - 1
            if seq.shape[-1] > max_context:
                print(f"Section {i}: output length {seq.shape[-1]} exceeding context length {max_context}, " f"now using the last {max_context} tokens.")
                cache.current_seq_len = 0
                full_ids = seq[:, -max_context:]
                incremental_ids = full_ids
            else:
                full_ids = seq
                incremental_ids = prompt_ids

            # For the unconditional context, mask out all but the last token
            if cfg:
                mask_len = full_ids.shape[-1] - 1
                full_mask = torch.zeros((2, cache.max_seq_len), dtype=torch.half, device=self.device)
                full_mask[1, :mask_len] = -65504.0
                position_offsets = torch.tensor([[0], [-mask_len]], dtype=torch.int)
                input_mask = full_mask[:, : full_ids.shape[-1]]

            # Forward prompt
            logits = self.model.forward(incremental_ids[:, :], cache=cache, input_mask=input_mask, position_offsets=position_offsets, last_id_only=True)

            # Generate until EOS or max_new_tokens
            for new_tokens in tqdm(range(max_new_tokens)):

                # Transformers-equiv. CFG
                if cfg:
                    cfg_scale = sample_settings.guidance_scale_seg0 if i == 0 else sample_settings.guidance_scale
                    logits = logits.float()
                    logits = F.log_softmax(logits, dim=-1)
                    logits = cfg_scale * logits[0] + (1 - cfg_scale) * logits[1]
                    logits = logits.unsqueeze(0)

                # Sample
                logits = logits.float().cpu()
                sample, _, _, _, _ = ExLlamaV2Sampler.sample(logits, gen_settings, full_ids[:1], rng.random(), self.tokenizer)
                if cfg:
                    sample = torch.cat((sample, sample), dim=0)

                # Accept token
                full_ids = torch.cat((full_ids, sample), dim=-1)
                seq = torch.cat((seq, sample), dim=-1)

                # Get next logits (update cache even if sample is EOA and we don't need next logits)
                if cfg:
                    input_mask = full_mask[:, : full_ids.shape[-1]]
                logits = self.model.forward(sample, cache=cache, input_mask=input_mask, position_offsets=position_offsets)

                # End on EOA
                if sample[0].item() == self.mmtokenizer.eoa:
                    break

            # Make sure sequence ends with EOA if we reached max_new_tokens
            else:
                sample = torch.tensor([[self.mmtokenizer.eoa]] * bsz, dtype=torch.long)
                seq = torch.cat((seq, sample), dim=-1)
                # Update cache with forced token
                self.model.forward(sample, cache=cache)

        raw_output = seq[:1, :]
        return raw_output


def main():
    args = parser.parse_args()
    if args.use_audio_prompt and not args.audio_prompt_path:
        raise FileNotFoundError("Please offer audio prompt filepath using '--audio_prompt_path', when you enable 'use_audio_prompt'!")
    if args.use_dual_tracks_prompt and not args.vocal_track_prompt_path and not args.instrumental_track_prompt_path:
        raise FileNotFoundError(
            "Please offer dual tracks prompt filepath using '--vocal_track_prompt_path' and '--inst_decoder_path', when you enable '--use_dual_tracks_prompt'!"
        )
    if args.seed is not None:
        seed_everything(args.seed)

    device = torch.device(f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")

    with open(args.genre_txt) as f:
        genres = f.read().strip()
    with open(args.lyrics_txt) as f:
        lyrics = f.read().strip()

    if args.stage1_use_exl2:
        pipeline = Stage1Pipeline_EXL2(
            model_path=args.stage1_model,
            device=device,
            basic_model_config=args.basic_model_config,
            resume_path=args.resume_path,
            cache_size=args.stage1_cache_size,
            cache_mode=args.stage1_cache_mode,
        )
    else:
        pipeline = Stage1Pipeline_HF(
            model_path=args.stage1_model,
            device=device,
            basic_model_config=args.basic_model_config,
            resume_path=args.resume_path,
            cache_size=args.stage1_cache_size,
        )

    # Load tokenizer and models
    raw_output = pipeline.generate(
        use_dual_tracks_prompt=args.use_dual_tracks_prompt,
        vocal_track_prompt_path=args.vocal_track_prompt_path,
        instrumental_track_prompt_path=args.instrumental_track_prompt_path,
        use_audio_prompt=args.use_audio_prompt,
        audio_prompt_path=args.audio_prompt_path,
        genres=genres,
        lyrics=lyrics,
        run_n_segments=args.run_n_segments,
        max_new_tokens=args.max_new_tokens,
        prompt_start_time=args.prompt_start_time,
        prompt_end_time=args.prompt_end_time,
        sample_settings=SampleSettings(use_guidance=not args.stage1_no_guidance),
    )

    # Save result
    pipeline.save(raw_output, args.output_dir, args.use_audio_prompt, args.use_dual_tracks_prompt)


if __name__ == "__main__":
    # enable inference mode globally
    torch.autograd.grad_mode._enter_inference_mode(True)
    torch.autograd.set_grad_enabled(False)
    main()

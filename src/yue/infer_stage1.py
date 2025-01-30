import os
import re

import numpy as np
import torch
import torchaudio
from codecmanipulator import CodecManipulator
from common import BlockTokenRangeProcessor, load_exl2_model, parser
from einops import rearrange
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
from mmtokenizer import _MMSentencePieceTokenizer
from models.soundstream_hubert_new import SoundStream
from omegaconf import OmegaConf
from torchaudio.transforms import Resample
from tqdm import tqdm
from transformers import AutoModelForCausalLM, LogitsProcessorList
from transformers.cache_utils import StaticCache


def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio


def split_lyrics(lyrics):
    pattern = r"\[(\w+)\](.*?)\n(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
    return structured_lyrics


def stage1_generate(
    mmtokenizer: _MMSentencePieceTokenizer,
    run_n_segments: int,
    lyrics: list[str],
    prompt_texts: list[str],
    device: torch.device,
    max_new_tokens: int,
    model: ExLlamaV2DynamicGenerator | AutoModelForCausalLM,
    use_audio_prompt: bool,
    codec_tool: CodecManipulator,
    codec_model: SoundStream | None,
    audio_prompt_path: str,
    prompt_start_time: float,
    prompt_end_time: float,
) -> torch.Tensor:
    output_seq = None
    # Here is suggested decoding config
    top_p = 0.93
    temperature = 1.0
    repetition_penalty = 1.2
    if isinstance(model, ExLlamaV2DynamicGenerator):
        gen_settings = ExLlamaV2Sampler.Settings(token_presence_penalty=repetition_penalty, temperature=temperature, top_k=0, top_p=top_p)
        gen_settings.disallow_tokens(model.tokenizer, list(range(0, 32002)) + [32016])
    # special tokens
    start_of_segment = mmtokenizer.tokenize("[start_of_segment]")
    end_of_segment = mmtokenizer.tokenize("[end_of_segment]")
    # Format text prompt
    run_n_segments = min(run_n_segments + 1, len(lyrics))
    for i, p in enumerate(tqdm(prompt_texts[1:run_n_segments])):
        section_text = p.replace("[start_of_segment]", "").replace("[end_of_segment]", "")
        guidance_scale = 1.5 if i == 0 else 1.2
        if i == 0:
            if use_audio_prompt:
                audio_prompt = load_audio_mono(audio_prompt_path)
                audio_prompt.unsqueeze_(0)
                raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=0.5)
                raw_codes = raw_codes.transpose(0, 1)
                raw_codes = raw_codes.cpu().numpy().astype(np.int16)
                # Format audio prompt
                code_ids = codec_tool.npy2ids(raw_codes[0])
                audio_prompt_codec = code_ids[int(prompt_start_time * 50) : int(prompt_end_time * 50)]  # 50 is tps of xcodec
                audio_prompt_codec_ids = [mmtokenizer.soa] + codec_tool.sep_ids + audio_prompt_codec + [mmtokenizer.eoa]
                sentence_ids = mmtokenizer.tokenize("[start_of_reference]") + audio_prompt_codec_ids + mmtokenizer.tokenize("[end_of_reference]")
                head_id = mmtokenizer.tokenize(prompt_texts[0]) + sentence_ids
            else:
                head_id = mmtokenizer.tokenize(prompt_texts[0])
            prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codec_tool.sep_ids
        else:
            prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codec_tool.sep_ids

        prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)
        input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 0 else prompt_ids
        # Use window slicing in case output sequence exceeds the context of model
        max_context = 16384 - max_new_tokens - 1
        if input_ids.shape[-1] > max_context:
            print(f"Section {i}: output length {input_ids.shape[-1]} exceeding context length {max_context}, now using the last {max_context} tokens.")
            input_ids = input_ids[:, -(max_context):]

        if isinstance(model, ExLlamaV2DynamicGenerator):
            # TODO: guidance_scale
            input_ids = input_ids.cpu()
            job = ExLlamaV2DynamicJob(
                input_ids=input_ids,
                min_new_tokens=100,
                max_new_tokens=max_new_tokens,
                stop_conditions=[mmtokenizer.eoa],  # stop on EOS token
                gen_settings=gen_settings,
            )
            model.enqueue(job)
            output_seq = input_ids
            while model.num_remaining_jobs():
                results = model.iterate()
                for result in results:
                    if result["stage"] != "streaming":
                        continue
                    new_token_ids = result.get("token_ids", None)
                    if new_token_ids is not None:
                        output_seq = torch.cat((output_seq, new_token_ids), dim=-1)
            output_seq = output_seq.to(device)
        else:
            past_key_values = StaticCache(
                model.config, max_batch_size=1, max_cache_len=input_ids.shape[-1] + max_new_tokens, device=model.device, dtype=model.dtype
            )
            output_seq = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                min_new_tokens=100,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=LogitsProcessorList([BlockTokenRangeProcessor(0, 32002), BlockTokenRangeProcessor(32016, 32016)]),
                guidance_scale=guidance_scale,
                past_key_values=past_key_values,
            )

        if output_seq[0][-1].item() != mmtokenizer.eoa:
            tensor_eoa = torch.tensor([[mmtokenizer.eoa]], dtype=torch.long, device=output_seq.device)
            output_seq = torch.cat((output_seq, tensor_eoa), dim=1)
        if i > 0:
            raw_output = torch.cat([raw_output, prompt_ids, output_seq[:, input_ids.shape[-1] :]], dim=1)
        else:
            raw_output = output_seq
    return raw_output


def stage1_save(raw_output: torch.Tensor, mmtokenizer: _MMSentencePieceTokenizer, codec_tool: CodecManipulator, output_dir: str, use_audio_prompt: bool):
    # save raw output and check sanity
    ids = raw_output[0].cpu().numpy()
    soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
    eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()
    if len(soa_idx) != len(eoa_idx):
        raise ValueError(f"invalid pairs of soa and eoa, Num of soa: {len(soa_idx)}, Num of eoa: {len(eoa_idx)}")

    vocals = []
    instrumentals = []
    range_begin = 1 if use_audio_prompt else 0
    for i in range(range_begin, len(soa_idx)):
        codec_ids = ids[soa_idx[i] + 1 : eoa_idx[i]]
        if codec_ids[0] == 32016:
            codec_ids = codec_ids[1:]
        codec_ids = codec_ids[: 2 * (codec_ids.shape[0] // 2)]
        vocals_ids = codec_tool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
        vocals.append(vocals_ids)
        instrumentals_ids = codec_tool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
        instrumentals.append(instrumentals_ids)
    vocals = np.concatenate(vocals, axis=1)
    instrumentals = np.concatenate(instrumentals, axis=1)
    stage1_output_dir = os.path.join(output_dir, "stage1")
    os.makedirs(stage1_output_dir, exist_ok=True)
    vocal_save_path = os.path.join(stage1_output_dir, "cot_vocal.npy")
    inst_save_path = os.path.join(stage1_output_dir, "cot_instrumental.npy")
    np.save(vocal_save_path, vocals)
    np.save(inst_save_path, instrumentals)


def main():
    args = parser.parse_args()
    if args.use_audio_prompt and not args.audio_prompt_path:
        raise FileNotFoundError("Please offer audio prompt filepath using '--audio_prompt_path', when you enable 'use_audio_prompt'!")

    # load tokenizer and model
    device = torch.device(f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
    if args.stage1_use_exl2:
        model = load_exl2_model(args.stage1_model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.stage1_model, torch_dtype=torch.float16, attn_implementation="sdpa")
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
    if args.use_audio_prompt:
        model_config = OmegaConf.load(args.basic_model_config)
        assert model_config.generator.name == "SoundStream"
        codec_model = SoundStream(**model_config.generator.config).to(device)
        parameter_dict = torch.load(args.resume_path, map_location=device, weights_only=False)
        codec_model.load_state_dict(parameter_dict["codec_model"])
        codec_model.eval()
    else:
        codec_model = None

    # Tips:
    # genre tags support instrumental，genre，mood，vocal timbr and vocal gender
    # all kinds of tags are needed
    with open(args.genre_txt) as f:
        genres = f.read().strip()
    with open(args.lyrics_txt) as f:
        lyrics = split_lyrics(f.read())
    # intruction
    full_lyrics = "\n".join(lyrics)
    prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
    prompt_texts += lyrics

    raw_output = stage1_generate(
        mmtokenizer,
        args.run_n_segments,
        lyrics,
        prompt_texts,
        device,
        args.max_new_tokens,
        model,
        args.use_audio_prompt,
        codec_tool,
        codec_model,
        args.audio_prompt_path,
        args.prompt_start_time,
        args.prompt_end_time,
    )
    stage1_save(raw_output, mmtokenizer, codec_tool, args.output_dir, args.use_audio_prompt)


if __name__ == "__main__":
    # enable inference mode globally
    torch.autograd.grad_mode._enter_inference_mode(True)
    torch.autograd.set_grad_enabled(False)
    main()

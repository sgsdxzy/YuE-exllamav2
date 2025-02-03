import argparse

from exllamav2 import (
    ExLlamaV2Cache,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Cache_Q8,
)
from transformers import LogitsProcessor
import torch
import random
import numpy as np

parser = argparse.ArgumentParser()
# Model Configuration:
parser.add_argument("--stage1_model", type=str, default="m-a-p/YuE-s1-7B-anneal-en-cot", help="The model checkpoint path or identifier for the Stage 1 model.")
parser.add_argument("--stage2_model", type=str, default="m-a-p/YuE-s2-1B-general", help="The model checkpoint path or identifier for the Stage 2 model.")
parser.add_argument("--max_new_tokens", type=int, default=3000, help="The maximum number of new tokens to generate in one pass during text generation.")
parser.add_argument("--run_n_segments", type=int, default=2, help="The number of segments to process during the generation.")
parser.add_argument("--stage1_use_exl2", action="store_true", help="Use exllamav2 to load and run stage 1 model.")
parser.add_argument("--stage2_use_exl2", action="store_true", help="Use exllamav2 to load and run stage 2 model.")
parser.add_argument("--stage2_batch_size", type=int, default=4, help="The non-exl2 batch size used in Stage 2 inference.")
parser.add_argument("--stage1_cache_size", type=int, default=16384, help="The cache size used in Stage 1 inference.")
parser.add_argument("--stage2_cache_size", type=int, default=8192, help="The exl2 cache size used in Stage 2 inference.")
parser.add_argument("--stage1_cache_mode", type=str, default="FP16", help="The cache mode used in Stage 1 inference (FP16, Q8, Q6, Q4). Quantized k/v cache will save VRAM at the cost of some speed and precision.")
parser.add_argument("--stage2_cache_mode", type=str, default="FP16", help="The cache mode used in Stage 2 inference (FP16, Q8, Q6, Q4). Quantized k/v cache will save VRAM at the cost of some speed and precision.")
parser.add_argument("--stage1_no_guidance", action="store_true", help="Disable classifier-free guidance for stage 1")
# Prompt
parser.add_argument(
    "--genre_txt",
    type=str,
    required=True,
    help="The file path to a text file containing genre tags that describe the musical style or characteristics (e.g., instrumental, genre, mood, vocal timbre, vocal gender). This is used as part of the generation prompt.",
)
parser.add_argument(
    "--lyrics_txt",
    type=str,
    required=True,
    help="The file path to a text file containing the lyrics for the music generation. These lyrics will be processed and split into structured segments to guide the generation process.",
)
parser.add_argument(
    "--use_audio_prompt",
    action="store_true",
    help="If set, the model will use an audio file as a prompt during generation. The audio file should be specified using --audio_prompt_path.",
)
parser.add_argument(
    "--audio_prompt_path", type=str, default="", help="The file path to an audio file to use as a reference prompt when --use_audio_prompt is enabled."
)
parser.add_argument("--prompt_start_time", type=float, default=0.0, help="The start time in seconds to extract the audio prompt from the given audio file.")
parser.add_argument("--prompt_end_time", type=float, default=30.0, help="The end time in seconds to extract the audio prompt from the given audio file.")
parser.add_argument(
    "--use_dual_tracks_prompt",
    action="store_true",
    help="If set, the model will use dual tracks as a prompt during generation. The vocal and instrumental files should be specified using --vocal_track_prompt_path and --instrumental_track_prompt_path.",
)
parser.add_argument(
    "--vocal_track_prompt_path",
    type=str,
    default="",
    help="The file path to a vocal track file to use as a reference prompt when --use_dual_tracks_prompt is enabled.",
)
parser.add_argument(
    "--instrumental_track_prompt_path",
    type=str,
    default="",
    help="The file path to an instrumental track file to use as a reference prompt when --use_dual_tracks_prompt is enabled.",
)
# Output
parser.add_argument("--output_dir", type=str, default="./output", help="The directory where generated outputs will be saved.")
parser.add_argument("--keep_intermediate", action="store_true", help="If set, intermediate outputs will be saved during processing.")
parser.add_argument("--disable_offload_model", action="store_true", help="If set, the model will not be offloaded from the GPU to CPU after Stage 1 inference.")
parser.add_argument("--cuda_idx", type=int, default=0)
parser.add_argument("--seed", type=int, default=None, help="An integer value to reproduce generation.")
# Config for xcodec and upsampler
parser.add_argument("--basic_model_config", default="./xcodec_mini_infer/final_ckpt/config.yaml", help="YAML files for xcodec configurations.")
parser.add_argument("--resume_path", default="./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth", help="Path to the xcodec checkpoint.")
parser.add_argument("--config_path", type=str, default="./xcodec_mini_infer/decoders/config.yaml", help="Path to Vocos config file.")
parser.add_argument("--vocal_decoder_path", type=str, default="./xcodec_mini_infer/decoders/decoder_131000.pth", help="Path to Vocos decoder weights.")
parser.add_argument("--inst_decoder_path", type=str, default="./xcodec_mini_infer/decoders/decoder_151000.pth", help="Path to Vocos decoder weights.")
parser.add_argument("-r", "--rescale", action="store_true", help="Rescale output to avoid clipping.")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cache_class(cache_mode: str):
    match cache_mode:
        case "Q4":
            return ExLlamaV2Cache_Q4
        case "Q6":
            return ExLlamaV2Cache_Q6
        case "Q8":
            return ExLlamaV2Cache_Q8
        case _:
            return ExLlamaV2Cache


class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores

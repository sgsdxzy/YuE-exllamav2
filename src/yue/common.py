import argparse

from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Config, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2DynamicGenerator
from transformers import LogitsProcessor

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
# Output
parser.add_argument("--output_dir", type=str, default="./output", help="The directory where generated outputs will be saved.")
parser.add_argument("--keep_intermediate", action="store_true", help="If set, intermediate outputs will be saved during processing.")
parser.add_argument("--disable_offload_model", action="store_true", help="If set, the model will not be offloaded from the GPU to CPU after Stage 1 inference.")
parser.add_argument("--cuda_idx", type=int, default=0)
# Config for xcodec and upsampler
parser.add_argument("--basic_model_config", default="./xcodec_mini_infer/final_ckpt/config.yaml", help="YAML files for xcodec configurations.")
parser.add_argument("--resume_path", default="./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth", help="Path to the xcodec checkpoint.")
parser.add_argument("--config_path", type=str, default="./xcodec_mini_infer/decoders/config.yaml", help="Path to Vocos config file.")
parser.add_argument("--vocal_decoder_path", type=str, default="./xcodec_mini_infer/decoders/decoder_131000.pth", help="Path to Vocos decoder weights.")
parser.add_argument("--inst_decoder_path", type=str, default="./xcodec_mini_infer/decoders/decoder_151000.pth", help="Path to Vocos decoder weights.")
parser.add_argument("-r", "--rescale", action="store_true", help="Rescale output to avoid clipping.")


def load_exl2_model(model_path: str, max_seq_len: int = -1, paged: bool = True):
    exl2_config = ExLlamaV2Config(model_path)
    model = ExLlamaV2(exl2_config)
    model.load()
    tokenizer = ExLlamaV2Tokenizer(exl2_config)
    cache = ExLlamaV2Cache(model, max_seq_len=max_seq_len)
    generator = ExLlamaV2DynamicGenerator(model, cache, tokenizer, paged=paged)

    return generator


class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores

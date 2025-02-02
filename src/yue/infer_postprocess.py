import os

import numpy as np
import soundfile as sf
import torch
import torchaudio
from common import parser, seed_everything
from models.soundstream_hubert_new import SoundStream
from omegaconf import OmegaConf
from post_process_audio import replace_low_freq_with_energy_matched
from vocoder import build_codec_model, process_audio


# convert audio tokens to audio
def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding="PCM_S", bits_per_sample=16)


def post_process(
    codec_model: SoundStream, device: torch.device, output_dir: str, config_path: str, vocal_decoder_path: str, inst_decoder_path: str, rescale: bool
):
    # reconstruct tracks
    recons_output_dir = os.path.join(output_dir, "recons")
    recons_mix_dir = os.path.join(recons_output_dir, "mix")
    os.makedirs(recons_mix_dir, exist_ok=True)
    stage2_result = [os.path.join(output_dir, "stage2", filename) for filename in ["vtrack.npy", "itrack.npy"]]
    tracks = []
    for npy in stage2_result:
        codec_result = np.load(npy)
        decodec_rlt = []
        decoded_waveform = codec_model.decode(torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long).unsqueeze(0).permute(1, 0, 2).to(device))
        decoded_waveform = decoded_waveform.cpu().squeeze(0)
        decodec_rlt.append(torch.as_tensor(decoded_waveform))
        decodec_rlt = torch.cat(decodec_rlt, dim=-1)
        save_path = os.path.join(recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3")
        tracks.append(save_path)
        save_audio(decodec_rlt, save_path, 16000)
    # mix tracks
    for inst_path in tracks:
        try:
            if (inst_path.endswith(".wav") or inst_path.endswith(".mp3")) and "itrack" in inst_path:
                # find pair
                vocal_path = inst_path.replace("itrack", "vtrack")
                if not os.path.exists(vocal_path):
                    continue
                # mix
                recons_mix = os.path.join(
                    recons_mix_dir,
                    os.path.basename(inst_path).replace("itrack", "mixed"),
                )
                vocal_stem, sr = sf.read(inst_path)
                instrumental_stem, _ = sf.read(vocal_path)
                mix_stem = (vocal_stem + instrumental_stem) / 1
                sf.write(recons_mix, mix_stem, sr)
        except Exception as e:
            print(e)

    # vocoder to upsample audios
    vocal_decoder, inst_decoder = build_codec_model(config_path, vocal_decoder_path, inst_decoder_path)
    vocoder_output_dir = os.path.join(output_dir, "vocoder")
    vocoder_stems_dir = os.path.join(vocoder_output_dir, "stems")
    vocoder_mix_dir = os.path.join(vocoder_output_dir, "mix")
    os.makedirs(vocoder_mix_dir, exist_ok=True)
    os.makedirs(vocoder_stems_dir, exist_ok=True)
    for npy in stage2_result:
        if "itrack" in npy:
            # Process instrumental
            instrumental_output = process_audio(npy, os.path.join(vocoder_stems_dir, "itrack.mp3"), rescale, device, inst_decoder, codec_model)
        else:
            # Process vocal
            vocal_output = process_audio(npy, os.path.join(vocoder_stems_dir, "vtrack.mp3"), rescale, device, vocal_decoder, codec_model)
    # mix tracks
    try:
        mix_output = instrumental_output + vocal_output
        vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix))
        save_audio(mix_output, vocoder_mix, 44100, rescale)
        print(f"Created mix: {vocoder_mix}")
    except RuntimeError as e:
        print(e)
        print(f"mix {vocoder_mix} failed! inst: {instrumental_output.shape}, vocal: {vocal_output.shape}")

    # Post process
    replace_low_freq_with_energy_matched(
        a_file=recons_mix, b_file=vocoder_mix, c_file=os.path.join(output_dir, os.path.basename(recons_mix)), cutoff_freq=5500.0  # 16kHz  # 48kHz
    )


def main():
    args = parser.parse_args()
    if args.seed is not None:
        seed_everything(args.seed)

    device = torch.device(f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu")
    model_config = OmegaConf.load(args.basic_model_config)
    assert model_config.generator.name == "SoundStream"
    codec_model = SoundStream(**model_config.generator.config).to(device)
    parameter_dict = torch.load(args.resume_path, map_location=device, weights_only=False)
    codec_model.load_state_dict(parameter_dict["codec_model"])
    codec_model.eval()

    post_process(codec_model, device, args.output_dir, args.config_path, args.vocal_decoder_path, args.inst_decoder_path, args.rescale)


if __name__ == "__main__":
    # enable inference mode globally
    torch.autograd.grad_mode._enter_inference_mode(True)
    torch.autograd.set_grad_enabled(False)
    main()

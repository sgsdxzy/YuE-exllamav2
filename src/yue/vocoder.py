import argparse
import os
import sys
import typing as tp
from pathlib import Path
from time import time

import numpy as np
import torch
import torchaudio
from omegaconf import OmegaConf
from tqdm import tqdm
from vocos import VocosDecoder


def build_soundstream_model(config):
    model = eval(config.generator.name)(**config.generator.config)
    return model

def build_codec_model(config_path, vocal_decoder_path, inst_decoder_path):
    vocal_decoder = VocosDecoder.from_hparams(config_path=config_path)
    vocal_decoder.load_state_dict(torch.load(vocal_decoder_path))
    inst_decoder = VocosDecoder.from_hparams(config_path=config_path)
    inst_decoder.load_state_dict(torch.load(inst_decoder_path))
    return vocal_decoder, inst_decoder

def save_audio(wav: torch.Tensor, path: tp.Union[Path, str], sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    
    path = str(Path(path).with_suffix('.mp3'))
    torchaudio.save(path, wav, sample_rate=sample_rate)

def process_audio(input_file, output_file, rescale, args, decoder, soundstream):
    compressed = np.load(input_file, allow_pickle=True).astype(np.int16)
    print(f"Processing {input_file}")
    print(f"Compressed shape: {compressed.shape}")
    
    args.bw = float(4)
    compressed = torch.as_tensor(compressed, dtype=torch.long).unsqueeze(1)
    compressed = soundstream.get_embed(compressed.to(f"cuda:{args.cuda_idx}"))
    compressed = torch.tensor(compressed).to(f"cuda:{args.cuda_idx}")
    
    start_time = time()
    with torch.no_grad():
        decoder.eval()
        decoder = decoder.to(f"cuda:{args.cuda_idx}")
        out = decoder(compressed)
        out = out.detach().cpu()
    duration = time() - start_time
    rtf = (out.shape[1] / 44100.0) / duration
    print(f"Decoded in {duration:.2f}s ({rtf:.2f}x RTF)")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_audio(out, output_file, 44100, rescale=rescale)
    print(f"Saved: {output_file}")
    return out

def find_matching_pairs(input_folder):
    if str(input_folder).endswith('.lst'):  # Convert to string
        with open(input_folder, 'r') as file:
            files = [line.strip() for line in file if line.strip()]
    else:
        files = list(Path(input_folder).glob('*.npy'))
    print(f"found {len(files)} npy.")
    instrumental_files = {}
    vocal_files = {}
    
    for file in files:
        if not isinstance(file, Path):
            file = Path(file)
        name = file.stem
        if 'instrumental' in name.lower():
            base_name = name.lower().replace('instrumental', '')#.strip('_')
            instrumental_files[base_name] = file
        elif 'vocal' in name.lower():
            # base_name = name.lower().replace('vocal', '').strip('_')
            last_index = name.lower().rfind('vocal')
            if last_index != -1:
                # Create a new string with the last 'vocal' removed
                base_name = name.lower()[:last_index] + name.lower()[last_index + len('vocal'):]
            else:
                base_name = name.lower()
            vocal_files[base_name] = file
    
    # Find matching pairs
    pairs = []
    for base_name in instrumental_files.keys():
        if base_name in vocal_files:
            pairs.append((
                instrumental_files[base_name],
                vocal_files[base_name],
                base_name
            ))
    
    return pairs

def main():
    parser = argparse.ArgumentParser(description='High fidelity neural audio codec using Vocos decoder.')
    parser.add_argument('--input_folder', type=Path, required=True, help='Input folder containing NPY files.')
    parser.add_argument('--output_base', type=Path, required=True, help='Base output folder.')
    parser.add_argument('--resume_path', type=str, default='./final_ckpt/ckpt_00360000.pth', help='Path to model checkpoint.')
    parser.add_argument('--config_path', type=str, default='./config.yaml', help='Path to Vocos config file.')
    parser.add_argument('--vocal_decoder_path', type=str, default='/aifs4su/mmcode/codeclm/xcodec_mini_infer_newdecoder/decoders/decoder_131000.pth', help='Path to Vocos decoder weights.')
    parser.add_argument('--inst_decoder_path', type=str, default='/aifs4su/mmcode/codeclm/xcodec_mini_infer_newdecoder/decoders/decoder_151000.pth', help='Path to Vocos decoder weights.')
    parser.add_argument('-r', '--rescale', action='store_true', help='Rescale output to avoid clipping.')
    args = parser.parse_args()

    # Validate inputs
    if not args.input_folder.exists():
        sys.exit(f"Input folder {args.input_folder} does not exist.")
    if not os.path.isfile(args.config_path):
        sys.exit(f"{args.config_path} file does not exist.")
    # if not os.path.isfile(args.decoder_path):
    #     sys.exit(f"{args.decoder_path} file does not exist.")

    # Create output directories
    mix_dir = args.output_base / 'mix'
    stems_dir = args.output_base / 'stems'
    os.makedirs(mix_dir, exist_ok=True)
    os.makedirs(stems_dir, exist_ok=True)

    # Initialize models
    config_ss = OmegaConf.load("./final_ckpt/config.yaml")
    soundstream = build_soundstream_model(config_ss)
    parameter_dict = torch.load(args.resume_path)
    soundstream.load_state_dict(parameter_dict['codec_model'])
    soundstream.eval()
    
    vocal_decoder, inst_decoder = build_codec_model(args.config_path, args.vocal_decoder_path, args.inst_decoder_path)
    
    # Find and process matching pairs
    pairs = find_matching_pairs(args.input_folder)
    print(f"Found {len(pairs)} matching pairs")
    pairs = [p for p in pairs if not os.path.exists(mix_dir / f'{p[2]}.mp3')]
    print(f"{len(pairs)} to reconstruct...")
    
    for instrumental_file, vocal_file, base_name in tqdm(pairs):
        print(f"\nProcessing pair: {base_name}")
        # Create stems directory for this song
        song_stems_dir = stems_dir / base_name
        os.makedirs(song_stems_dir, exist_ok=True)
        
        try:
            # Process instrumental
            instrumental_output = process_audio(
                instrumental_file,
                song_stems_dir / 'instrumental.mp3',
                args.rescale,
                args,
                inst_decoder,
                soundstream
            )
            
            # Process vocal
            vocal_output = process_audio(
                vocal_file,
                song_stems_dir / 'vocal.mp3',
                args.rescale,
                args,
                vocal_decoder,
                soundstream
            )
        except IndexError as e:
            print(e)
            continue
        
        # Create and save mix
        try:
            mix_output = instrumental_output + vocal_output
            save_audio(mix_output, mix_dir / f'{base_name}.mp3', 44100, args.rescale)
            print(f"Created mix: {mix_dir / f'{base_name}.mp3'}")
        except RuntimeError as e:
            print(e)
            print(f"mix {base_name} failed! inst: {instrumental_output.shape}, vocal: {vocal_output.shape}")

if __name__ == '__main__':
    main()

    # Example Usage
    # python reconstruct_separately.py --input_folder test_samples --output_base test

import os
import re
import sys

import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T


def replace_low_freq_with_energy_matched(
    a_file: str,
    b_file: str,
    c_file: str,
    cutoff_freq: float = 5500.0,
    eps: float = 1e-10
):
    """
    1. Load a_file (16kHz) and b_file (48kHz).
    2. Resample 'a' to 48kHz if needed.
    3. Match the low-frequency energy of 'a' to that of 'b'.
    4. Replace the low-frequency of 'b' with the matched low-frequency of 'a'.
    5. Save the result to c_file.
    
    Args:
        a_file (str): Path to a.mp3 (16kHz).
        b_file (str): Path to b.mp3 (48kHz).
        c_file (str): Output path for combined result.
        cutoff_freq (float): Cutoff frequency for low/highpass filters.
        eps (float): Small value to avoid division-by-zero.
    """

    # ----------------------------------------------------------
    # 1. Load the two files
    # ----------------------------------------------------------
    wave_a, sr_a = torchaudio.load(a_file)
    wave_b, sr_b = torchaudio.load(b_file)
    
    # If 'a' doesn't match 'b' sample rate, resample it
    if sr_a != sr_b:
        resampler = T.Resample(orig_freq=sr_a, new_freq=sr_b)
        wave_a = resampler(wave_a)
        sr_a = sr_b  # Now they match
    
    # ----------------------------------------------------------
    # 2. Low-pass both signals to isolate low-frequency content
    # ----------------------------------------------------------
    wave_a_low = F.lowpass_biquad(
        wave_a,
        sample_rate=sr_b,
        cutoff_freq=cutoff_freq
    )
    wave_b_low = F.lowpass_biquad(
        wave_b,
        sample_rate=sr_b,
        cutoff_freq=cutoff_freq
    )
    
    # ----------------------------------------------------------
    # 3. Compute RMS of low-frequency portions
    # ----------------------------------------------------------
    # We'll do a simple global RMS (across channels & time)
    # If you need per-channel matching, handle each channel separately.
    a_rms = wave_a_low.pow(2).mean().sqrt().item() + eps
    b_rms = wave_b_low.pow(2).mean().sqrt().item() + eps
    
    # ----------------------------------------------------------
    # 4. Scale 'a_low' so its energy matches 'b_low'
    # ----------------------------------------------------------
    scale_factor = b_rms / a_rms
    wave_a_low_matched = wave_a_low * scale_factor
    
    # ----------------------------------------------------------
    # 5. High-pass 'b' to isolate high-frequency content
    # ----------------------------------------------------------
    wave_b_high = F.highpass_biquad(
        wave_b,
        sample_rate=sr_b,
        cutoff_freq=cutoff_freq
    )
    
    # ----------------------------------------------------------
    # 6. Combine: (scaled a_low) + (b_high)
    # ----------------------------------------------------------
    if wave_a_low_matched.size(1)!=wave_b_high.size(1):
        print(f"Original lengths: a_low={wave_a_low_matched.size()}, b_high={wave_b_high.size()}")
        min_length = min(wave_a_low_matched.size(1), wave_b_high.size(1))
        wave_a_low_matched = wave_a_low_matched[:, :min_length]
        wave_b_high = wave_b_high[:, :min_length]
        
        print(f"After truncation: a_low={wave_a_low_matched.size()}, b_high={wave_b_high.size()}")
        print(f"Samples truncated: {max(wave_a_low_matched.size(1), wave_b_high.size(1)) - min_length}")
    
    wave_combined = wave_a_low_matched + wave_b_high
    
    # (Optional) Normalize if needed to avoid clipping
    # wave_combined /= max(wave_combined.abs().max(), 1.0)
    
    # ----------------------------------------------------------
    # 7. Save to c.mp3
    # ----------------------------------------------------------
    torchaudio.save(c_file, wave_combined, sample_rate=sr_b)
    
    print(f"Successfully created '{os.path.basename(c_file)}' with matched low-frequency energy.")

if __name__ == "__main__":
    stage2_output_dir = sys.argv[1]
    recons_dir = os.path.join(stage2_output_dir, "recons", "mix")
    vocoder_dir = os.path.join(stage2_output_dir, "vocoder", "mix")
    save_dir = os.path.join(stage2_output_dir, "post_process")
    os.makedirs(save_dir, exist_ok=True)
    
    # Create dictionaries mapping IDs to filenames
    recons_files = {}
    vocoder_files = {}

    pattern = r"mixed_([a-f0-9-]+)_xcodec_16k\.mp3$"
    
    # Map IDs to filenames for recons/mix
    for filename in os.listdir(recons_dir):
        match = re.search(pattern, filename)
        if match:
            recons_files[(match.group(1)).lower()] = filename
    
    print(recons_files)

    pattern = r"__([a-f0-9-]+)\.mp3$"
    # Map IDs to filenames for vocoder/mix
    for filename in os.listdir(vocoder_dir):
        match = re.search(pattern, filename)
        if match:
            vocoder_files[(match.group(1)).lower()] = filename
    
    # Find common IDs
    common_ids = set(recons_files.keys()) & set(vocoder_files.keys())
    print(f"Found {len(common_ids)} matching file pairs")
    
    # Create matched file lists
    a_list = []
    b_list = []
    for id in common_ids:
        a_list.append(os.path.join(recons_dir, recons_files[id]))
        b_list.append(os.path.join(vocoder_dir, vocoder_files[id]))

    # Process only matching pairs
    for a, b in zip(a_list, b_list):
        if os.path.exists(os.path.join(save_dir, os.path.basename(b))):
            continue

        replace_low_freq_with_energy_matched(
            a_file=a,     # 16kHz
            b_file=b,     # 48kHz
            c_file=os.path.join(save_dir, os.path.basename(b)),
            cutoff_freq=5500.0
        )

# YuE-exllamav2

Optimized implementation of [multimodal-art-projection/YuE](https://github.com/multimodal-art-projection/YuE) in exllamav2.

## Benchmark
### Inference time on RTX 4090 24gb:
- bf16 model, Flash Attention 2 enabled, CFG enabled
- default example prompt for genre and lyrics
- 2 segments in length
- default batch size 4 was used during stage 2 inference with the original pipeline, and cache size was set to 64k for exllamav2

| Stage  | Original           | Exllamav2          |
| ------ | ------------------ | ------------------ |
|    1   |        282s        |        125s        |
|    2   |        666s        |         49s        |
| Total  |        948s        |        174s        |

On this hardware configuration, this implmentation achieves over 500% end-to-end speedup over baseline. There may be some variance in results due to track length variability between generations, although during this test there was only a 3 second difference in track length (0:58 vs 0:55).

### Inference time on RTX 3060 mobile 6gb:

Quantized models are available for inference with lower VRAM usage: [Stage 1](https://huggingface.co/Doctor-Shotgun/YuE-s1-7B-anneal-en-cot-exl2) [Stage 2](https://huggingface.co/Doctor-Shotgun/YuE-s2-1B-general-exl2)

- 4.25bpw-h6 stage 1 with Q4 cache
- 8.0bpw-h8 stage 2 with Q8 cache
- Flash Attention 2 enabled, CFG enabled
- default example prompt for genre and lyrics
- 2 segments in length

| Stage  | Original           | Exllamav2          |
| ------ | ------------------ | ------------------ |
|    1   |        OOM         |        317s        |
|    2   |        OOM         |        350s        |
| Total  |        OOM         |        667s        |

This produces a coherent result despite aggressive quantization, although there are no metrics for now to measure quality loss.

## Usage

### 1. Prerequisites
- Git
- Conda or Python 3.10+ with pip

### 2. Setup environment
We recommend using conda to create a new environment:
```bash
conda create -n yue python=3.12 # 3.12 >= Python >= 3.10 is recommended.
conda activate yue
```
Or alternatively, you can create a native Python virtualenv:
```bash
# You can create a Python virtualenv
python -m venv venv

# Linux
source ./venv/bin/activate
# Windows Powershell
./venv/scripts/activate
# Windows cmd
call ./venv/scripts/activate
```

### 3. Download the inference code and codec model
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
# if you don't have root, see https://github.com/git-lfs/git-lfs/issues/4134#issuecomment-1635204943
sudo apt update
sudo apt install git-lfs
git lfs install
git clone https://github.com/sgsdxzy/YuE-exllamav2.git
cd YuE-exllamav2
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
pip install -r requirements.txt
```

### 4. Run inference via CLI
```bash
python src/yue/infer.py --stage1_use_exl2 --stage2_use_exl2 --stage2_cache_size 32768 [original args]
```


## Original README of multimodal-art-projection/YuE
<p align="center">
    <img src="./assets/logo/白底.png" width="400" />
</p>

<p align="center">
    <a href="https://map-yue.github.io/">Demo 🎶</a> &nbsp;|&nbsp; 📑 <a href="">Paper (coming soon)</a>
    <br>
    <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-cot">YuE-s1-7B-anneal-en-cot 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-icl">YuE-s1-7B-anneal-en-icl 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-jp-kr-cot">YuE-s1-7B-anneal-jp-kr-cot 🤗</a>
    <br>
    <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-jp-kr-icl">YuE-s1-7B-anneal-jp-kr-icl 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-cot">YuE-s1-7B-anneal-zh-cot 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-s1-7B-anneal-zh-icl">YuE-s1-7B-anneal-zh-icl 🤗</a>
    <br>
    <a href="https://huggingface.co/m-a-p/YuE-s2-1B-general">YuE-s2-1B-general 🤗</a> &nbsp;|&nbsp; <a href="https://huggingface.co/m-a-p/YuE-upsampler">YuE-upsampler 🤗</a>
</p>

---
Our model's name is **YuE (乐)**. In Chinese, the word means "music" and "happiness." Some of you may find words that start with Yu hard to pronounce. If so, you can just call it "yeah." We wrote a song with our model's name, see [here](assets/logo/yue.mp3).

YuE is a groundbreaking series of open-source foundation models designed for music generation, specifically for transforming lyrics into full songs (lyrics2song). It can generate a complete song, lasting several minutes, that includes both a catchy vocal track and accompaniment track. YuE is capable of modeling diverse genres/languages/vocal techniques. Please visit the [**Demo Page**](https://map-yue.github.io/) for amazing vocal performance.

## News and Updates

* **2025.01.30 🔥 Inference Update**: We now support dual-track ICL mode! You can prompt the model with a reference song, and it will generate a new song in a similar style (voice cloning, music style transfer, etc.). Try it out! See [demo video by @abrakjamson on X](https://x.com/abrakjamson/status/1885932885406093538). 🔥🔥🔥

* **2025.01.30 🔥 Announcement: A New Era Under Apache 2.0 🔥**: We are thrilled to announce that, in response to overwhelming requests from our community, **YuE** is now officially licensed under the **Apache 2.0** license. We sincerely hope this marks a watershed moment—akin to what Stable Diffusion and LLaMA have achieved in their respective fields—for music generation and creative AI. 🎉🎉🎉

* **2025.01.29 🎉**: We have updated the license description. we **ENCOURAGE** artists and content creators to sample and incorporate outputs generated by our model into their own works, and even monetize them. The only requirement is to credit our name: **YuE by HKUST/M-A-P** (alphabetic order).
* **2025.01.28 🫶**: Thanks to Fahd for creating a tutorial on how to quickly get started with YuE. Here is his [demonstration](https://www.youtube.com/watch?v=RSMNH9GitbA).
* **2025.01.26 🔥**: We have released the **YuE** series.

<br>

---
## TODOs📋
- [ ] Example finetune code for enabling BPM control using 🤗 Transformers.
- [ ] Support stemgen mode https://github.com/multimodal-art-projection/YuE/issues/21
- [ ] Support llama.cpp https://github.com/ggerganov/llama.cpp/issues/11467
- [ ] Support gradio interface. https://github.com/multimodal-art-projection/YuE/issues/1
- [ ] Online serving on huggingface space.
- [ ] Support transformers tensor parallel. https://github.com/multimodal-art-projection/YuE/issues/7
- [x] Support dual-track ICL mode.
- [x] Fix "instrumental" naming bug in output files. https://github.com/multimodal-art-projection/YuE/pull/26
- [x] Support seeding https://github.com/multimodal-art-projection/YuE/issues/20

---

## Hardware and Performance

### **GPU Memory**
YuE requires significant GPU memory for generating long sequences. Below are the recommended configurations:

- **For GPUs with 24GB memory or less**: Run **up to 2 sessions** concurrently to avoid out-of-memory (OOM) errors. You could try this [YuEGP](https://github.com/deepbeepmeep/YuEGP) to see if it helps reduce VRAM usage or improve speed.
- **For full song generation** (many sessions, e.g., 4 or more): Use **GPUs with at least 80GB memory**. i.e. H800, A100, or multiple RTX4090s with tensor parallel.

To customize the number of sessions, the interface allows you to specify the desired session count. By default, the model runs **2 sessions** (1 verse + 1 chorus) to avoid OOM issue.

### **Execution Time**
On an **H800 GPU**, generating 30s audio takes **150 seconds**.
On an **RTX 4090 GPU**, generating 30s audio takes approximately **360 seconds**. 

---

## Quickstart
Quick start **VIDEO TUTORIAL** by Fahd: [Link here](https://www.youtube.com/watch?v=RSMNH9GitbA). We recommend watching this video if you are not familiar with machine learning or the command line.

### 1. Install environment and dependencies
Make sure properly install flash attention 2 to reduce VRAM usage. 
```bash
# We recommend using conda to create a new environment.
conda create -n yue python=3.8 # Python >=3.8 is recommended.
conda activate yue
# install cuda >= 11.8
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
pip install -r <(curl -sSL https://raw.githubusercontent.com/multimodal-art-projection/YuE/main/requirements.txt)

# For saving GPU memory, FlashAttention 2 is mandatory. 
# Without it, long audio may lead to out-of-memory (OOM) errors.
# Be careful about matching the cuda version and flash-attn version
pip install flash-attn --no-build-isolation
```

### 2. Download the infer code and tokenizer
```bash
# Make sure you have git-lfs installed (https://git-lfs.com)
# if you don't have root, see https://github.com/git-lfs/git-lfs/issues/4134#issuecomment-1635204943
sudo apt update
sudo apt install git-lfs
git lfs install
git clone https://github.com/multimodal-art-projection/YuE.git

cd YuE/inference/
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
```

### 3. Run the inference
Now generate music with **YuE** using 🤗 Transformers. Make sure your step [1](#1-install-environment-and-dependencies) and [2](#2-download-the-infer-code-and-tokenizer) are properly set up. 

Note:
- Set `--run_n_segments` to the number of lyric sections if you want to generate a full song. Additionally, you can increase `--stage2_batch_size` based on your available GPU memory.

- You may customize the prompt in `genre.txt` and `lyrics.txt`. See prompt engineering guide [here](#prompt-engineering-guide).

- You can increase `--stage2_batch_size` to speed up the inference, but be careful for OOM.

- LM ckpts will be automatically downloaded from huggingface. 


```bash
# This is the CoT mode.
cd YuE/inference/
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000
```

We also support music in-context-learning (provide a reference song), there are 2 types: single-track (mix/vocal/instrumental) and dual-track. 

Note: 
- ICL requires a different ckpt, e.g. `m-a-p/YuE-s1-7B-anneal-en-icl`.

- Music ICL generally requires a 30s audio segment. The model will write new songs with similar style of the provided audio, and may improve musicality.

- Dual-track ICL works better in general, requiring both vocal and instrumental tracks.

- For single-track ICL, you can provide a mix, vocal, or instrumental track.

- You can separate the vocal and instrumental tracks using [python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator) or [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui).

```bash
# This is the dual-track ICL mode.
# To turn on dual-track mode, enable `--use_dual_tracks_prompt`
# and provide `--vocal_track_prompt_path`, `--instrumental_track_prompt_path`, 
# `--prompt_start_time`, and `--prompt_end_time`
# The ref audio is taken from GTZAN test set.
cd YuE/inference/
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-icl \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --use_dual_tracks_prompt \
    --vocal_track_prompt_path ../prompt_egs/pop.00001.Vocals.mp3 \
    --instrumental_track_prompt_path ../prompt_egs/pop.00001.Instrumental.mp3 \
    --prompt_start_time 0 \
    --prompt_end_time 30 
```

```bash
# This is the single-track (mix/vocal/instrumental) ICL mode.
# To turn on single-track ICL, enable `--use_audio_prompt`, 
# and provide `--audio_prompt_path` , `--prompt_start_time`, and `--prompt_end_time`. 
# The ref audio is taken from GTZAN test set.
cd YuE/inference/
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-icl \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --use_audio_prompt \
    --audio_prompt_path ../prompt_egs/pop.00001.mp3 \
    --prompt_start_time 0 \
    --prompt_end_time 30 
```
---
 
## Prompt Engineering Guide
The prompt consists of three parts: genre tags, lyrics, and ref audio.

### Genre Tagging Prompt
1. An example genre tagging prompt can be found [here](prompt_egs/genre.txt).

2. A stable tagging prompt usually consists of five components: genre, instrument, mood, gender, and timbre. All five should be included if possible, separated by space (space delimiter).

3. Although our tags have an open vocabulary, we have provided the top 200 most commonly used [tags](./top_200_tags.json). It is recommended to select tags from this list for more stable results.

3. The order of the tags is flexible. For example, a stable genre tagging prompt might look like: "inspiring female uplifting pop airy vocal electronic bright vocal vocal."

4. Additionally, we have introduced the "Mandarin" and "Cantonese" tags to distinguish between Mandarin and Cantonese, as their lyrics often share similarities.

### Lyrics Prompt
1. An example lyric prompt can be found [here](prompt_egs/lyrics.txt).

2. We support multiple languages, including but not limited to English, Mandarin Chinese, Cantonese, Japanese, and Korean. The default top language distribution during the annealing phase is revealed in [issue 12](https://github.com/multimodal-art-projection/YuE/issues/12#issuecomment-2620845772). A language ID on a specific annealing checkpoint indicates that we have adjusted the mixing ratio to enhance support for that language.

3. The lyrics prompt should be divided into sessions, with structure labels (e.g., [verse], [chorus], [bridge], [outro]) prepended. Each session should be separated by 2 newline character "\n\n".

4. **DONOT** put too many words in a single segment, since each session is around 30s (`--max_new_tokens 3000` by default).

5. We find that [intro] label is less stable, so we recommend starting with [verse] or [chorus].

6. For generating music with no vocal, see [issue 18](https://github.com/multimodal-art-projection/YuE/issues/18).


### Audio Prompt

1. Audio prompt is optional. Providing ref audio for ICL usually increase the good case rate, and result in less diversity since the generated token space is bounded by the ref audio. CoT only (no ref) will result in a more diverse output.

2. We find that dual-track ICL mode gives the best musicality and prompt following. 

3. Use the chorus part of the music as prompt will result in better musicality.

4. Around 30s audio is recommended for ICL.

---

## License Agreement \& Disclaimer  
- The YuE model (including its weights) is now released under the **Apache License, Version 2.0**. We do not make any profit from this model, and we hope it can be used for the betterment of human creativity.
- **Use & Attribution**: 
    - We encourage artists and content creators to freely incorporate outputs generated by YuE into their own works, including commercial projects. 
    - We encourage attribution to the model’s name (“YuE by HKUST/M-A-P”), especially for public and commercial use. 
- **Originality & Plagiarism**: It is the sole responsibility of creators to ensure that their works, derived from or inspired by YuE outputs, do not plagiarize or unlawfully reproduce existing material. We strongly urge users to perform their own due diligence to avoid copyright infringement or other legal violations.
- **Recommended Labeling**: When uploading works to streaming platforms or sharing them publicly, we **recommend** labeling them with terms such as: “AI-generated”, “YuE-generated", “AI-assisted” or “AI-auxiliated”. This helps maintain transparency about the creative process.
- **Disclaimer of Liability**: 
    - We do not assume any responsibility for the misuse of this model, including (but not limited to) illegal, malicious, or unethical activities. 
    - Users are solely responsible for any content generated using the YuE model and for any consequences arising from its use. 
    - By using this model, you agree that you understand and comply with all applicable laws and regulations regarding your generated content.

---

## Acknowledgements
The project is co-lead by HKUST and M-A-P (alphabetic order). Also thanks moonshot.ai, bytedance, 01.ai, and geely for supporting the project.
A friendly link to HKUST Audio group's [huggingface space](https://huggingface.co/HKUSTAudio). 

We deeply appreciate all the support we received along the way. Long live open-source AI!

---

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :)

```BibTeX
@misc{yuan2025yue,
  title={YuE: Open Music Foundation Models for Full-Song Generation},
  author={Ruibin Yuan and Hanfeng Lin and Shawn Guo and Ge Zhang and Jiahao Pan and Yongyi Zang and Haohe Liu and Xingjian Du and Xeron Du and Zhen Ye and Tianyu Zheng and Yinghao Ma and Minghao Liu and Lijun Yu and Zeyue Tian and Ziya Zhou and Liumeng Xue and Xingwei Qu and Yizhi Li and Tianhao Shen and Ziyang Ma and Shangda Wu and Jun Zhan and Chunhui Wang and Yatian Wang and Xiaohuan Zhou and Xiaowei Chi and Xinyue Zhang and Zhenzhu Yang and Yiming Liang and Xiangzhou Wang and Shansong Liu and Lingrui Mei and Peng Li and Yong Chen and Chenghua Lin and Xie Chen and Gus Xia and Zhaoxiang Zhang and Chao Zhang and Wenhu Chen and Xinyu Zhou and Xipeng Qiu and Roger Dannenberg and Jiaheng Liu and Jian Yang and Stephen Huang and Wei Xue and Xu Tan and Yike Guo}, 
  howpublished={\url{https://github.com/multimodal-art-projection/YuE}},
  year={2025},
  note={GitHub repository}
}
```
<br>

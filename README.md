# XTTS-Voice-Studio

A complete voice cloning and guided audio production studio powered by **XTTS v2** (Coqui TTS).  
Generate professional narrations — meditations, audiobooks, podcasts, voice-overs — from simple text scripts, with per-voice acoustic tuning and automatic parameter analysis.

---

## ✨ Features

- 🎤 **Voice cloning** from any WAV reference file (6s minimum)
- 🌍 **17 languages** supported (FR, EN, ES, DE, IT, PT, PL, TR, RU, NL, CS, AR, ZH-CN, HU, KO, JA, HI)
- 🎭 **Multi-voice** support — switch voices mid-script
- 🎛️ **Per-sentence XTTS parameters** (temperature, top_k, top_p, trim, fades)
- 🎚️ **Full audio processing pipeline** (EQ, noise reduction, de-esser, compression, rubberband)
- 🎵 **Ambient music** looping + **punctual sound** injection
- ⏸️ **Smart pauses** — fixed or adjusted to speech duration
- 🔬 **Voice analyser** — automatically finds optimal parameters for any voice
- 🎙️ **Audio transcriber** — converts any MP3/WAV to a generator-ready `.txt` with auto-detected pauses (Whisper)
- 🎲 **Reproducible generation** via seed control

---

## 📁 Recommended Directory Structure

All files should be organised under a single `~/XTTS/` root folder.  
**Clone this repo into `~/XTTS/`** and create the additional directories below:

```
~/XTTS/                                ← root folder (this repo)
│
├── Python_Scripting/                  ← scripts (from this repo)
│   ├── guided_meditation_generator_v20.py
│   └── voice_analyser.py
│
├── Prompts/                           ← your .txt script files (from this repo)
│
├── Punctual_sounds/                   ← short WAV sounds triggered inline (from this repo)
│
├── Song_to_TXT_with_Pauses/          ← utility scripts (from this repo)
│
├── Voices_Cloning/                    ← ⚠️ create this manually — your WAV voice references
│
├── Ambient_Musics/                    ← ⚠️ create this manually — background music WAV files
│                                           (filename must contain "ambiance" or "ambient")
│
└── Output_Song_files/                 ← ⚠️ create this manually — generated WAV output
```

Create the missing directories in one command:

```bash
mkdir -p ~/XTTS/Voices_Cloning ~/XTTS/Ambient_Musics ~/XTTS/Output_Song_files
```

---

- Ubuntu 20.04 or later (tested on Ubuntu 24.04)
- NVIDIA GPU with CUDA support (recommended — CPU mode works but is slow)
- 8 GB RAM minimum, 16 GB recommended
- ~5 GB disk space for the XTTS v2 model

---

## 📦 Installation

### 1. Install Miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the prompts, then restart your terminal or:
source ~/.bashrc
```

### 2. Install NVIDIA drivers and CUDA toolkit

Check your GPU first:
```bash
lspci | grep -i nvidia
```

Install the NVIDIA driver (if not already installed):
```bash
sudo apt update
sudo apt install -y nvidia-driver-535   # or latest available
sudo reboot
```

Verify after reboot:
```bash
nvidia-smi
```

Install CUDA toolkit:
```bash
sudo apt install -y nvidia-cuda-toolkit
nvcc --version   # verify
```

### 3. Create the conda environment

```bash
conda create -n xtts python=3.10 -y
conda activate xtts
```

### 4. Install PyTorch with CUDA support

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

Verify CUDA is available:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0))"
```

### 5. Install Coqui TTS (XTTS v2)

```bash
pip install coqui-tts
```

> **Note:** The original `coqui-ai/TTS` repository is archived. Use `coqui-tts` (community-maintained fork) which supports XTTS v2.

Download the XTTS v2 model (first run only, ~2 GB):
```bash
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
```

### 6. Install audio processing dependencies

```bash
# pydub and ffmpeg
pip install pydub
sudo apt install -y ffmpeg

# rubberband (time-stretching without pitch change)
sudo apt install -y rubberband-cli

# Voice analyser dependencies
pip install librosa numpy scipy
```

### 7. Clone this repository

```bash
git clone https://github.com/ZFEbHVUE/XTTS-Voice-Studio.git
cd XTTS-Voice-Studio
```

---

## 📁 Project Structure

```
XTTS-Voice-Studio/
├── Python_Scripting/
│   ├── guided_meditation_generator_v20.py      # Main generator
│   ├── voice_analyser.py                       # Voice parameter analyser
│   └── transcribeSong2txt_with_pause.py        # Audio → TXT transcriber (Whisper)
├── Prompts/                                    # Example script files (.txt)
├── Punctual_sounds/                            # Example punctual audio files
├── Song_to_TXT_with_Pauses/                    # Output folder for transcribed .txt files
└── README.md
```

You will need to create these directories locally (not tracked by git):
```bash
mkdir -p Voices_Cloning Output_Song_files Ambient_Musics
```

---

## 🚀 Quick Start

### Step 1 — Analyse your voice reference

```bash
conda activate xtts

# With language (recommended)
python Python_Scripting/voice_analyser.py Voices_Cloning/my_voice.wav FR

# Multiple voices at once
python Python_Scripting/voice_analyser.py voice1.wav FR voice2.wav EN

# Language defaults to FR if omitted
python Python_Scripting/voice_analyser.py Voices_Cloning/my_voice.wav
```

This outputs two ready-to-paste brackets for your script:
```
{1, 0, 89, 314, 150, 300, 0.55, 30, 0.75}
[1, FR, 0.85, +1, -5, +1, -3, 90, 8000, 0.3, 0.4, 0.2]
```

### Step 2 — Write your script

Create a `.txt` file in `Prompts/`. See the syntax section below.

### Step 3 — Generate

```bash
python Python_Scripting/guided_meditation_generator_v20.py \
    Prompts/my_script.txt \
    Output_Song_files/output.wav \
    Voices_Cloning/my_voice.wav \
    Ambient_Musics/ambient.wav \
    Punctual_sounds/bell.wav
```

---

## 📝 Script Syntax

### XTTS params block `{...}` — per-voice XTTS settings

```
{N, seed, trim_start, trim_end, fade_in, fade_out, temperature, top_k, top_p}
```

| Position | Parameter | Value 0 means |
|----------|-----------|---------------|
| 1 | Voice number | — |
| 2 | Seed | No seed (random) |
| 3 | trim_start (ms) | No trim |
| 4 | trim_end (ms) | No trim |
| 5 | fade_in (ms) | No fade |
| 6 | fade_out (ms) | No fade |
| 7 | temperature | Keep default |
| 8 | top_k | Keep default |
| 9 | top_p | Keep default |

### Audio bracket `[...]` — per-voice audio processing

```
[N, LANG, speed, vol(dB), eq_low, eq_mid, eq_high, hp, lp, NR, comp, de-ess]
```

`LANG` is optional (FR, EN, ES, DE, IT, ...). All values after N are optional.

### Pauses

```
[pause=2s]           → 2 second silence
[pause=4s,start]     → adjusted pause: total (speech + silence) = 4s
```

### Punctual music

```
# Declare at top of file
musique_1=30s,-15    → file #1: 30s, -15 dB
musique_2=-10        → file #2: full duration, -10 dB

# Trigger inline
[musique=1]
```

### Ambient music

```
volume_ambiance=-12  → loops throughout at -12 dB
```
The ambient file must contain `ambiance` or `ambiant` in its filename.

### Complete example

```
# ── Global settings ──────────────────────────────────────────────────────
volume_ambiance=-18        # ambient music at -18 dB throughout
musique_1=5s,-10           # punctual sound #1: 5 seconds, -10 dB

# ── Voice 1 (female narrator, English) ───────────────────────────────────
{1, 42, 110, 255, 150, 300, 0.65, 50, 0.85}
[1, EN, 0.85, +1, -5, +1, -1, 90, 9000, 0.3, 0.4, 0.2]
Welcome. [pause=4s,start]
Close your eyes, and take a slow, deep breath. [pause=5s,start]
Feel the weight of your body becoming heavier with each exhale. [pause=5s,start]

# ── Trigger punctual sound (e.g. a singing bowl) ─────────────────────────
[musique=1]

# ── Voice 2 (male narrator, English, different cloning reference) ─────────
{2, 42, 60, 339, 150, 300, 0.60, 40, 0.80}
[2, EN, 0.90, +3, -3, +2, -2, 80, 8000, 0.5, 0.4, 0.2]
Now, let your thoughts drift away like clouds across the sky. [pause=4s,start]
There is nothing to do, nowhere to be. [pause=5s,start]
Just breathe. [pause=6s,start]
```

---

## 🔬 Voice Analyser

```bash
# Basic usage — language defaults to FR
python Python_Scripting/voice_analyser.py Voices_Cloning/voice.wav

# With language code (recommended)
python Python_Scripting/voice_analyser.py Voices_Cloning/voice.wav EN

# Precise mode (30-90s, pyin algorithm)
python Python_Scripting/voice_analyser.py --precise Voices_Cloning/voice.wav FR

# Multiple voices at once
python Python_Scripting/voice_analyser.py voice1.wav FR voice2.wav EN voice3.wav DE
```

Supported languages: `FR EN ES DE IT PT PL TR RU NL CS AR ZH-CN HU KO JA HI`

The analyser measures F0, RMS level, SNR, crest factor, sibilance, and spectral balance to recommend optimal EQ, compression, noise reduction, XTTS temperature, and trim values for your specific voice.

---

## 🎙️ Audio Transcriber

`transcribeSong2txt_with_pause.py` converts any audio file (MP3, WAV) into a `.txt` script ready for the generator, with **automatic pause detection** based on Whisper word-level timestamps.

```bash
# Basic usage (medium model, 0.7s minimum pause)
python Python_Scripting/transcribeSong2txt_with_pause.py audio.mp3 Song_to_TXT_with_Pauses/output.txt

# Specify model and minimum pause duration
python Python_Scripting/transcribeSong2txt_with_pause.py audio.mp3 Song_to_TXT_with_Pauses/output.txt medium
python Python_Scripting/transcribeSong2txt_with_pause.py audio.mp3 Song_to_TXT_with_Pauses/output.txt small 0.5
python Python_Scripting/transcribeSong2txt_with_pause.py audio.mp3 Song_to_TXT_with_Pauses/output.txt large-v3 1.0
```

**Available Whisper models:**

| Model | Speed | Quality | Recommended for |
|-------|-------|---------|-----------------|
| `tiny` | Very fast | Basic | Quick tests |
| `base` | Fast | Decent | Short clips |
| `small` | Medium | Good | GTX 1650 / 4GB GPU |
| `medium` | Slow | Very good | Default |
| `large` | Very slow | Excellent | High quality |
| `large-v3` | Very slow | Best | Non-English audio |

**Minimum pause parameter:**

| Value | Effect |
|-------|--------|
| `0.5s` | Captures all small pauses |
| `0.7s` | Significant pauses only (recommended) |
| `1.0s` | Long pauses only |

**Install Whisper** (if not already installed):
```bash
pip install openai-whisper
```

The output `.txt` is placed in `Song_to_TXT_with_Pauses/` and is directly usable with the generator — just add your `{...}` and `[...]` brackets at the top.

---

## ⚙️ Requirements Summary

```
# conda environment
python=3.10
pytorch (with CUDA 12.1)

# pip packages
coqui-tts
pydub
librosa
numpy
scipy

# system packages
ffmpeg
rubberband-cli
nvidia-driver-535 (or newer)
nvidia-cuda-toolkit
```

Or install all pip dependencies at once:
```bash
pip install coqui-tts pydub librosa numpy scipy
```

---

## 🌍 Supported Languages

| Code | Language | Code | Language |
|------|----------|------|----------|
| `fr` | French | `ru` | Russian |
| `en` | English | `nl` | Dutch |
| `es` | Spanish | `cs` | Czech |
| `de` | German | `ar` | Arabic |
| `it` | Italian | `zh-cn` | Chinese |
| `pt` | Portuguese | `hu` | Hungarian |
| `pl` | Polish | `ko` | Korean |
| `tr` | Turkish | `ja` | Japanese |
| `hi` | Hindi | | |

---

## 📋 Voice Reference Guidelines

For best cloning quality:
- **Duration**: 10-30 seconds (6s minimum)
- **Format**: WAV, 22050 Hz or 44100 Hz, mono or stereo
- **Content**: Clear speech, no background music, no reverb
- **Quality**: SNR > 30 dB recommended (use `voice_analyser.py` to check)
- **Emotion**: Neutral to slightly warm tone works best for narration

---

## 🙏 Credits

- [Coqui TTS](https://github.com/coqui-ai/TTS) — XTTS v2 model
- [coqui-tts](https://pypi.org/project/coqui-tts/) — community-maintained fork
- [rubberband](https://breakfastquay.com/rubberband/) — time-stretching
- [pydub](https://github.com/jiaaro/pydub) — audio manipulation
- [librosa](https://librosa.org/) — voice analysis

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

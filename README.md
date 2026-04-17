# XTTS Voice Studio

A complete toolkit for voice cloning, guided meditation generation, song transcription, and audio processing — built around Coqui XTTS v2 with a Tkinter GUI that unifies every script under one roof.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey)

---

## Overview

XTTS Voice Studio is a personal production suite for:

- **Cloning voices** from short audio samples (via XTTS v2)
- **Generating guided meditations** with multi-voice narration, ambient music, punctual sound cues, and per-voice fine-tuning
- **Transcribing songs or speech** into XTTS-compatible scripts with pause detection and optional pitch annotation
- **Separating vocals** from instrumental tracks
- **Applying pitch correction** to cloned voices
- **Converting video files** to MP3 for further processing

Every tool is accessible through a single Tkinter interface (`xtts_studio.py`) or directly from the command line.

![XTTS Voice Studio GUI](docs/gui_main.png)

---

## Features

### GUI (`xtts_studio.py`)

A unified Tkinter interface with one tab per tool. Every tab shares a consistent live progress indicator and a frozen final timer that displays `[HH:MM:SS] done` until the next run.

| Tab | Script / Command | Purpose |
|-----|-----------------|---------|
| **[Gen] Generator** | `guided_meditation_generator_v22.py` | Build multi-voice guided meditations with inline XTTS parameters, ambient music, punctual sounds, and parallel voice overlay |
| **[Ana] Analyser** | `voice_analyser.py` | Analyse one or more voices and produce ready-to-paste XTTS parameter blocks |
| **[Txt] Transcription** | `transcribeSong2txt_with_pause.py` (audio) or `video2txt.py` (video) | Transcribe audio or video to XTTS-ready text with pause markers and optional per-word pitch annotation |
| **[Vox] Voice sep.** | `extract_voices.py` | Separate vocal stems from a mixed track |
| **[Pit] Pitch** | `apply_pitch_to_clone.py` | Apply pitch correction to a cloned voice |
| **[Vid] Video->MP3** | `ffmpeg` (system command) | Extract an MP3 audio track from any video file |

---

### Guided Meditation Generator

Produces long-form guided meditation WAV files from a simple text script. Supports:

- **Inline XTTS parameters per voice** using curly-brace syntax — 9 values (v20, still valid) or 11 values (v21/v22):
  ```
  {N, seed, trim_start, trim_end, fade_in, fade_out, temperature, top_k, top_p}
  {N, seed, trim_start, trim_end, fade_in, fade_out, temperature, top_k, top_p, rep_pen, len_pen}
  ```
- **Per-voice audio configuration** using bracket syntax:
  ```
  [N, LANG, speed, volume, eq_low, eq_mid, eq_high, hp, lp, noise_reduction, compression, de-esser]
  ```
- **Parallel voice overlay** (v22): mix two or more voices simultaneously with optional time offset:
  ```
  [parallel, offset=1.5s]
  {1, ...} [1, FR, ...] Voice one speaks here.
  {2, ...} [2, FR, ...] Voice two speaks something different at the same time.
  [/parallel]
  ```
- **Smart pause handling**: `[pause=2s]` for fixed silences, `[pause=4s,start]` to ensure total sentence+silence duration
- **Ambient tracks** that loop throughout the entire meditation at a chosen volume
- **Punctual music cues** triggered at specific positions (e.g. bells, chimes, metronomes)
- **Multi-language support**: FR, EN, ES, DE, IT, PT, PL, TR, RU, NL, CS, AR, ZH-CN, HU, KO, JA, HI
- **Rubberband-based speed adjustment** without pitch distortion
- **Live progress tracking** via hidden `[PROGRESS=n/N]` markers consumed by the GUI
- **Acoustic parameter derivation**: use `voice_analyser.py` to generate ready-to-paste `{}` and `[]` blocks from any reference WAV

---

### Song Transcription (v21)

Fast and accurate audio-to-text transcription powered by `faster-whisper`:

- 2-4× faster than the previous `openai-whisper` backend, with identical quality
- Word-level timestamps for precise pause detection
- Optional per-word pitch annotation (`[p:+2]`, `[p:-1]`, `[p:0]`, `[p:?]` for unvoiced) using single-pass YIN F0 extraction
- Shared audio buffer between `librosa` (F0) and Whisper (transcription) — audio is loaded exactly once
- Apostrophe and hyphen-aware tokenisation: merges `qu'on`, `Saint-Michel`, etc. before annotation
- Silero VAD pre-filtering (`--vad`) to skip long silences and accelerate further
- Automatic CUDA → CPU fallback if the GPU is unavailable
- Output is drop-in compatible with the Guided Meditation Generator

---

## Installation

### Requirements

- Python 3.10
- Miniconda or Anaconda (recommended for environment isolation)
- CUDA 12.x + compatible GPU (optional but highly recommended for `faster-whisper` and XTTS)
- `ffmpeg` (for audio format conversion)

### Step 1 — Clone the repository

```bash
cd ~
git clone https://github.com/ZFEbHVUE/XTTS-Voice-Studio.git
cd XTTS-Voice-Studio
```

### Step 2 — Create the conda environment

```bash
conda create -n xtts python=3.10
conda activate xtts
```

### Step 3 — Install dependencies

```bash
# Core dependencies
pip install TTS torch torchaudio
pip install faster-whisper
pip install librosa pydub numpy
pip install pyrubberband

# System libraries (Ubuntu/Debian)
sudo apt install ffmpeg rubberband-cli
```

### Step 4 — Verify CUDA (optional)

```bash
python -c "from faster_whisper import WhisperModel; m = WhisperModel('tiny', device='cuda', compute_type='float16'); print('CUDA OK')"
```

If you see `CUDA OK`, everything is ready. If not, the scripts will automatically fall back to CPU.

---

## Directory Structure

```
XTTS-Voice-Studio/
├── Python_Scripting/              # All executable scripts
│   ├── xtts_studio.py             # Tkinter GUI (main entry point)
│   ├── guided_meditation_generator_v22.py
│   ├── transcribeSong2txt_with_pause.py   # v21 (faster-whisper) — audio transcription
│   ├── video2txt.py                        # video transcription (extracts audio + transcribes)
│   ├── voice_analyser.py
│   ├── extract_voices.py
│   └── apply_pitch_to_clone.py
├── Prompts/                       # Text scripts for meditation generation
├── Voices_Cloning/                # Voice samples for XTTS cloning (.wav, 6-30s)
├── Ambient_Musics/                # Background ambient loops
├── Punctual_sounds/               # One-shot audio cues (bells, chimes)
├── MP3toTXT/                      # Audio/video sources for transcription
├── Output_Song_files/             # Generated meditation WAVs
├── Song_to_TXT_with_Pauses/       # Transcribed text files
└── README.md
```

The root directory is auto-detected by `xtts_studio.py` based on the script's location. You can move or rename the clone anywhere — no hardcoded paths.

---

## Usage

### Launching the GUI

```bash
conda activate xtts
python ~/XTTS-Voice-Studio/Python_Scripting/xtts_studio.py
```

Every tab has a **Browse** button that opens in the appropriate default directory (`Prompts/`, `Voices_Cloning/`, etc.) and a live timer that freezes on completion so you can see how long the run took.

### Generating a guided meditation

Create a text file in `Prompts/` using this syntax:

```
{1, 42, 110, 255, 150, 300, 0.65, 50, 0.85, 5.0, 1.0}
[1, FR, 0.85, +1, -5, +1, -1, 90, 9000, 0.3, 0.4, 0.2]
Welcome to this guided meditation session.
[pause=3s]
Take a deep breath in through your nose.
[pause=5s,start]
And slowly exhale through your mouth.
[music=1]
[pause=4s]
```

The last two values in the `{}` block are `rep_pen` (repetition penalty, default 5.0) and `len_pen` (length penalty, default 1.0). They can be omitted for backward compatibility with v20 scripts.

### Using parallel voice overlay (v22)

Two or more voices can speak simultaneously — each with its own text and full XTTS/audio parameter set. All parallel blocks support `{N,...}`, `[N,...]`, and `[pause=Xs]` syntax identically to normal blocks.

```
# Example 1 — two voices start simultaneously
[parallel]
{1, 42, 110, 255, 150, 300, 0.65, 50, 0.85, 5.0, 1.0}
[1, FR, 0.85, +1, -5, +1, -1, 90, 9000, 0.3, 0.4, 0.2]
The main narrator speaks this phrase slowly.
{2, 42, 60, 339, 150, 300, 0.60, 40, 0.80, 5.0, 1.0}
[2, FR, 0.90, -6, -5, +1, -1, 90, 9000, 0.3, 0.4, 0.2]
A second voice whispers something different underneath.
[/parallel]

# Example 2 — three voices in a staggered canon (1s offset between each)
[parallel, offset=1s]
{1, ...} [1, FR, ...] First voice begins the phrase.
{2, ...} [2, FR, ...] Second voice echoes one second later.
{3, ...} [3, FR, ...] Third voice joins another second after that.
[/parallel]
```

The `offset=` parameter delays each successive voice relative to the one declared before it. The total block duration equals the longest track including its offset. With no `offset=`, all voices start at exactly the same time.

The number of voices is unlimited — the only hard constraint is having a matching WAV reference file for each voice number used.

### Transcribing a song

From the **Transcription** tab, pick your audio file and output path. Choose the Whisper model (`small` is recommended for a GTX 1650), set a minimum pause threshold, pick a language, and optionally enable pitch annotation.

Or from the command line:

```bash
python Python_Scripting/transcribeSong2txt_with_pause.py \
    audio.mp3 output.txt small 0.7 fr --pitch --device cuda --vad
```

---

## Recent changes

### Version 22 (April 2026)

#### `guided_meditation_generator_v22.py` *(new script)*

- **Parallel voice overlay** — new `[parallel, offset=Xs]` / `[/parallel]` block syntax allows two or more voices to speak simultaneously, each with independent text and full `{N,...}` / `[N,...]` / `[pause=Xs]` parameter control
- Each voice track inside a parallel block is generated independently and mixed into a single segment via `pydub.overlay()`
- The `offset=` parameter (default `0s`) staggers voice entry times — voice 1 at 0s, voice 2 at offset, voice 3 at 2×offset, and so on
- No limit on the number of simultaneous voices — requires one WAV reference file per voice number used
- New `generate_sentence_audio()` helper extracted from the main generation loop — shared by normal and parallel modes, ensuring identical audio processing in both paths
- Progress bar correctly counts sentences inside parallel blocks
- Fully backward-compatible: all v20 and v21 scripts run unchanged

#### `xtts_studio.py`

- Generator tab now calls `guided_meditation_generator_v22.py`

---

### Version 21 (April 2026)

#### `guided_meditation_generator_v21.py`

- `{}` XTTS block extended from 9 to **11 values** — fully backward-compatible with v20 scripts
- `repetition_penalty` and `length_penalty` are now **configurable per voice** in the `{}` block
- Both parameters respect the value-0 = "keep default" convention already used by `temperature`, `top_k`, `top_p`
- Console log on voice switch now displays `rep_pen` and `len_pen`

#### `voice_analyser.py`

- **Two new derived parameters**: `repetition_penalty` and `length_penalty` computed from acoustic analysis
  - `repetition_penalty` derived from F0 jitter: monotone voices → 6.0–7.0; expressive voices → 4.0
  - `length_penalty` derived from `voiced_ratio`: dense/fast speakers → 0.9; slow/breathy speakers → 1.1
- Output `{}` block updated to 11 values — ready to paste directly into v21/v22 scripts

#### `transcribeSong2txt_with_pause.py`

- Backend switched from `openai-whisper` to `faster-whisper` (2-4× speedup at identical quality)
- Audio loaded once with `librosa`, shared with Whisper — no redundant ffmpeg decode
- Unvoiced words tagged `[p:?]` instead of misleading `[p:0]`
- Optional Silero VAD pre-filter via `--vad`
- Automatic CUDA → CPU fallback
- All emojis replaced with ASCII tags — fixes Windows console cp1252 crashes
- Hard exit with `os._exit(0)` prevents shutdown logs from polluting the final output

#### `xtts_studio.py`

- Generator tab now calls `guided_meditation_generator_v21.py`

---

### Version 20 (earlier 2026)

#### `guided_meditation_generator_v20.py`

- Hidden `[PROGRESS=n/N]` markers emitted after each sentence and each post-processing phase so the GUI progress bar reaches 100% only when the WAV is written
- All emojis replaced with ASCII tags
- Final execution time displayed alongside audio length
- Hard exit prevents XTTS teardown logs from polluting output

#### `xtts_studio.py`

- `XTTS_ROOT` auto-detected from the script's location — no more hardcoded paths
- Live progress label `[HH:MM:SS] [n/total] [pct%]` during runs
- Frozen final timer `[HH:MM:SS] [n/total] [100%] done` remains visible until the next run
- Applies to all tabs: Generator, Analyser, Transcription, Voice Separation, Pitch, Video→MP3

---

## Troubleshooting

**`faster-whisper` not installed**
```bash
conda activate xtts
pip install faster-whisper
```

**CUDA errors about missing `cudnn_ops_infer.so`**
```bash
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
```
*(Replace `cu12` with `cu11` if your CUDA is 11.x — check with `nvidia-smi`.)*

**XTTS asks for terms of service agreement on first run**
Edit `~/.local/share/tts/` to pre-accept, or answer `y` the first time.

**Windows console shows `?` or crashes on Unicode characters**
This should no longer happen as of v21 — all scripts output pure ASCII. If you still see issues, verify you are running the latest versions from `main`.

---

## Credits

- **XTTS v2** by [Coqui](https://github.com/coqui-ai/TTS)
- **faster-whisper** by [SYSTRAN](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend)
- **librosa** for F0 analysis
- **pydub** and **Rubberband** for audio processing
- Guided meditation concept and multi-voice orchestration: personal project

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

[ZFEbHVUE](https://github.com/ZFEbHVUE) — GitHub

> Note: the username `ZFEbHVUE` reads as `STEPHANE` when mirrored vertically.

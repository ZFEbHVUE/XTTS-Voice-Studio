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
- **Converting video files** to audio for further processing

Every tool is accessible through a single Tkinter interface (`xtts_studio.py`) or directly from the command line.

![XTTS Voice Studio GUI](docs/gui_main.png)

---

## Installation

### Requirements

- Python 3.10
- Miniconda or Anaconda (recommended)
- CUDA 12.x + compatible GPU (optional but recommended)
- `ffmpeg` and `rubberband-cli` (system packages)

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
pip install TTS torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install faster-whisper librosa pydub numpy soundfile pyrubberband
pip install torchcrepe demucs noisereduce nara-wpe deepfilternet
pip install praat-parselmouth

sudo apt install ffmpeg rubberband-cli
```

### Step 4 — Launch the GUI

```bash
conda activate xtts
python ~/XTTS-Voice-Studio/Python_Scripting/xtts_studio.py
```

The `XTTS_ROOT` is auto-detected from the script location — no hardcoded paths. You can clone the repo anywhere or rename the directory freely.

---

## Directory Structure

```
XTTS-Voice-Studio/
├── Python_Scripting/
│   ├── xtts_studio.py                      # Tkinter GUI (main entry point)
│   ├── guided_meditation_generator_v23.py  # Meditation generator
│   ├── voice_analyser.py                   # Acoustic analysis → XTTS params
│   ├── extract_voices.py                   # Vocal separation
│   ├── transcribeSong2txt_with_pause.py    # Audio transcription
│   ├── video2txt.py                        # Video transcription
│   └── apply_pitch_to_clone.py             # Pitch correction
├── Prompts/                                # Text scripts for meditation generation
├── Voices_Cloning/                         # Voice reference samples (.wav, 6–60s)
├── Ambient_Musics/                         # Background ambient loops
├── Punctual_sounds/                        # One-shot audio cues (bells, chimes)
├── MP3toTXT/                               # Audio/video sources for transcription
├── Output_Song_files/                      # Generated meditation WAVs
├── Song_to_TXT_with_Pauses/               # Transcribed text files
└── README.md
```

---

## Guided Meditation Generator

Produces long-form guided meditation audio files from a simple text script.

### XTTS parameter block `{}` — 14 values (v23)

```
{N, seed, trim_start, trim_end, fade_in, fade_out, temp, top_k, top_p, rep_pen, len_pen, gpt_cond_len, gpt_cond_chunk_len, sound_norm_refs}
```

| Pos | Parameter | Default | Description |
|-----|-----------|---------|-------------|
| 1 | `N` | — | Voice number |
| 2 | `seed` | 0 | Random seed (0 = none) |
| 3 | `trim_start` | 0 | Trim from start of generated audio (ms) |
| 4 | `trim_end` | 0 | Trim from end (ms) |
| 5 | `fade_in` | 100 | Fade-in (ms) |
| 6 | `fade_out` | 250 | Fade-out (ms) |
| 7 | `temperature` | 0.72 | GPT sampling temperature |
| 8 | `top_k` | 50 | GPT top-k |
| 9 | `top_p` | 0.85 | GPT top-p |
| 10 | `rep_pen` | 5.0 | Repetition penalty |
| 11 | `len_pen` | 1.0 | Length penalty |
| 12 | `gpt_cond_len` | 30 | Reference WAV seconds used for cloning (up to 60) |
| 13 | `gpt_cond_chunk_len` | 4 | GPT conditioning chunk size |
| 14 | `sound_norm_refs` | 0 | Normalise reference before cloning (0/1) |

Fully backward-compatible — v20/v21/v22 scripts (9–13 values) run unchanged.

### Audio parameter block `[]` — 16 values (v23)

```
[N, LANG, speed, vol, eq_low, eq_mid, eq_high, hp, lp, NR, comp, de-ess, reverb, noise_gate, pan, limiter]
```

| Pos | Parameter | Default | Description |
|-----|-----------|---------|-------------|
| 1 | `N` | — | Voice number |
| 2 | `LANG` | FR | Language code (FR, EN, ES, DE, IT, PT...) |
| 3 | `speed` | 1.0 | Rubberband speed factor |
| 4 | `vol` | 0 | Volume adjustment (dB) |
| 5 | `eq_low` | -2 | Low EQ 80–300 Hz (dB) |
| 6 | `eq_mid` | +3 | Mid EQ 300–3000 Hz (dB) |
| 7 | `eq_high` | -4 | High EQ 3000–8000 Hz (dB) |
| 8 | `hp` | 90 | Highpass filter (Hz) |
| 9 | `lp` | 8000 | Lowpass filter (Hz) |
| 10 | `NR` | 0.5 | Noise reduction strength (0–2) |
| 11 | `comp` | 0.4 | Compression strength (0–1) |
| 12 | `de-ess` | 0.3 | De-esser strength (0–1) |
| 13 | `reverb` | 0 | Reverb wet level (0–1) |
| 14 | `noise_gate` | 0 | Noise gate threshold (dB, 0=off, e.g. -40) |
| 15 | `pan` | 0 | Stereo pan (-1.0 left / 0 centre / +1.0 right) |
| 16 | `limiter` | 0 | Output limiter (0=off, 1=on) |

Processing order: Trim → Filters → EQ → NR → De-esser → Compression → Noise gate → Reverb → Fades → Pan → Limiter

### Per-voice config persistence

Short blocks inherit the last full config for that voice — only write what changes:

```
# First full config — memorised for voice 1
[1, FR, 0.9, 6, -5, 1, -2, 75, 8000, 0.35, 0.5, 0.5, 0, 0, 0, 1]
Première phrase.
[pause=2s]

# Only speed and volume change — rest inherited
[1, FR, 0.8, 3]
Deuxième phrase.
[pause=2s]

# Only language changes
[1, EN]
Third sentence in English.
```

Config resets at the start of each generation run.

### Multi-reference voices

Pass multiple reference WAV files per voice separated by spaces in the GUI. XTTS averages the speaker embeddings for a more robust clone.

In the CLI, voice groups are separated by `--`:

```bash
generator.py script.txt output.wav \
    ref1.wav ref2.wav \
    -- \
    hollie.wav \
    --mp3-bitrate 192 --mp3-mode cbr
```

### Parallel voice overlay

```
[parallel, offset=1s,5s]
{1, 42, ...} [1, FR, ...] First voice begins immediately.
{2, 42, ...} [2, FR, ...] Second voice enters after 1 second.
{3, 42, ...} [3, FR, ...] Third voice joins at 5 seconds.
[/parallel]
```

Voice 1 always starts at 0s. `offset=` values are absolute start times for voices 2, 3, 4...

### Other syntax

```
[pause=2s]           # fixed silence
[pause=4s,start]     # pad sentence+silence to 4s total
[music=1]            # trigger punctual music cue #1
ambient_volume=-18   # set ambient track volume (dB)
music_1=5s,-10       # music cue 1: offset 5s, volume -10dB
```

---

## Voice Analyser

Analyses one or more reference audio files and produces ready-to-paste `{N,...}` and `[N,...]` parameter blocks.

### Acoustic measurements

| Measurement | Tool | Derived parameter |
|-------------|------|------------------|
| F0 median, std, jitter | YIN / torchcrepe / pyin | Voice type, hp/lp, speed, len_pen |
| HNR | Praat | noise_reduction |
| Shimmer APQ5 | Praat | compression |
| Shimmer + jitter score | Praat | temperature, rep_pen |
| Formants F1/F2 | Praat | highpass refinement |
| Syllable tempo | Praat | speed refinement |
| Voiced RMS | librosa | volume |
| Sibilance | librosa | de-esser |
| Duration | — | gpt_cond_len |

### Per-voice F0 engine selector

Each voice row in the Analyser tab has an independent engine selector:

```
V [1] [ voice.wav titi.wav ] [Browse] [FR▼] Seed:[42] ☐ Prec [none]   [X]
V [2] [ hollie.wav         ] [Browse] [FR▼] Seed:[0 ] ☑ Prec [pyin▼]  [X]
```

- **Prec unchecked** → fast YIN, `none` label shown (read-only)
- **Prec checked** → precise mode, choose `auto / crepe / pyin`

**Recommended by voice type:**
- Bass/baritone → `pyin` (torchcrepe detects harmonics on low voices)
- Soprano/high → `auto` or `crepe`
- Meditation voice → `none` (fast, Praat does the real work)

### Multi-reference averaging

When multiple files are passed for the same voice, each is analysed separately and all numeric parameters are averaged into a single representative `{}[]` block:

```bash
# Analyse two references for voice 1, average the results
voice_analyser.py --precise --f0-engine pyin --start-num 1 toto.wav titi.wav FR
```

---

## Video→Audio Tab

Extracts audio from any video file:

- **Formats**: WAV, MP3 (CBR/VBR), FLAC, OGG
- **Channels**: stereo or mono
- **Sample rates**: 16000, 22050, 44100, 48000 Hz
- **XTTS preset**: forces WAV + mono + 22050 Hz (optimal for XTTS reference files)

---

## Optimal reference audio for XTTS

| Property | Recommended |
|----------|-------------|
| Format | WAV (lossless) |
| Channels | Mono |
| Sample rate | 22050 Hz |
| Duration | 20–60 seconds |
| Content | Clean speech only — no music, no echo, no noise |

Use the **XTTS preset** button in the Video→Audio tab to extract reference audio in the correct format directly from a video.

---

## TTS Backend Comparison

| Model | French | Cloning | GPU | Verdict |
|-------|--------|---------|-----|---------|
| **XTTS v2** | ✅ native | ✅ excellent | 4GB+ | **Best for French** |
| Chatterbox Multilingual | ✅ correct | ✅ good | 8GB+ | Acceptable, XTTS better |
| F5-TTS | ❌ English accent | ✅ timbre | 8GB+ | Not usable for French |
| IndexTTS2 | ❌ English accent | ✅ timbre | 8GB+ | Not usable for French |

XTTS v2 remains the best model for French voice cloning without fine-tuning.

---

## Troubleshooting

**`No module named 'librosa'` or similar**
Launch the GUI with `conda activate xtts` first.

**`CUDA out of memory` with demucs**
Switch from `mdx_extra` to `htdemucs_ft` or use `--device cpu`.

**torchcrepe returns F0=1900Hz or voiced=0% on bass/baritone voices**
Use `pyin` engine instead — torchcrepe can detect harmonics rather than the fundamental on low voices.

**`[!] File not found: --mp3-bitrate`**
You are running an old version of the generator. Replace with the latest `guided_meditation_generator_v23.py`.

**WPE dereverberation is very slow**
WPE is inherently single-threaded. Use `deepfilter` for GPU-accelerated dereverberation of similar quality.

**GUI opens in wrong directory on Browse**
Make sure you launch `xtts_studio.py` from its own directory (or use the correct path). `XTTS_ROOT` is auto-detected from the script location.

---

## Credits

- **XTTS v2** by [Coqui](https://github.com/coqui-ai/TTS)
- **faster-whisper** by [SYSTRAN](https://github.com/SYSTRAN/faster-whisper)
- **demucs** by [Meta Research](https://github.com/facebookresearch/demucs)
- **parselmouth / Praat** for acoustic analysis
- **torchcrepe** for GPU F0 estimation
- **librosa**, **pydub**, **rubberband** for audio processing

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

[ZFEbHVUE](https://github.com/ZFEbHVUE) — GitHub

> The username `ZFEbHVUE` reads as `STEPHANE` when mirrored vertically.

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
- **Validating XTTS parameters** empirically by generating and comparing multiple audio variations

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
│   ├── voice_validator.py                  # Empirical parameter validation
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

## GUI Tabs

### [Gen] Generator

Generates guided meditation audio from a text script.

- Per-voice rows with multi-reference browse (space-separated WAV files)
- `+ Add voice` button — new rows appear above the button
- MP3 output options (bitrate, CBR/VBR)
- Ambient music and punctual sound cue inputs

### [Ana] Analyser

Analyses voice reference files and produces ready-to-paste XTTS parameter blocks.

- Per-voice rows with multi-file browse (space-separated files averaged)
- Per-voice F0 engine selector: `none` (fast YIN) / `auto` / `crepe` / `pyin`
- Prec checkbox — when checked, shows F0 engine combobox; when unchecked, shows `none` label
- FINAL SUMMARY — complete 14-value `{}` and 16-value `[]` blocks ready to paste

### [Txt] Transcription

Transcribes audio or video files to XTTS-ready text with pause markers.

- faster-whisper backend (2–4× faster than openai-whisper)
- Optional VAD pre-filter, pitch annotation, CUDA/CPU auto-detection
- Separate Video and Audio file browsers

### [Vox] Voice sep.

Separates vocal stems from mixed audio.

- demucs music removal (htdemucs, htdemucs_ft, mdx_extra)
- Dereverberation: none / noisereduce / wpe / deepfilter
- F0-based separation (female / male / overlap / vocals only)
- Multi-format output (WAV/MP3/FLAC/OGG)

### [Pit] Pitch

Applies pitch correction to cloned voice audio.

### [Vid] Video->Audio

Extracts audio from any video file.

- Output formats: WAV, MP3 (CBR/VBR), FLAC, OGG
- Channels: stereo or mono
- Sample rates: 16000, 22050, 44100, 48000 Hz
- **XTTS preset** button: forces WAV + mono + 22050 Hz (optimal for XTTS references)

### [Val] Validator

Empirically validates XTTS parameters by generating multiple variations and combining them into a single audio file for A/B listening.

See the [Validator](#validator) section below for full details.

---

## Guided Meditation Generator

### XTTS parameter block `{}` — 14 values (v23)

```
{N, seed, trim_start, trim_end, fade_in, fade_out, temp, top_k, top_p, rep_pen, len_pen, gpt_cond_len, gpt_cond_chunk_len, sound_norm_refs}
```

| Pos | Parameter | Default | Description |
|-----|-----------|---------|-------------|
| 1 | `N` | — | Voice number |
| 2 | `seed` | 0 | Random seed (0 = none). **The seed interacts with the voice embedding — test several values per voice reference.** |
| 3 | `trim_start` | 0 | Trim from start of generated audio (ms) |
| 4 | `trim_end` | 0 | Trim from end (ms) |
| 5 | `fade_in` | 100 | Fade-in (ms) |
| 6 | `fade_out` | 250 | Fade-out (ms) |
| 7 | `temperature` | 0.72 | GPT sampling temperature |
| 8 | `top_k` | 50 | GPT top-k |
| 9 | `top_p` | 0.85 | GPT top-p |
| 10 | `rep_pen` | 5.0 | Repetition penalty |
| 11 | `len_pen` | 1.0 | Length penalty |
| 12 | `gpt_cond_len` | 30 | Reference WAV seconds used for cloning (up to 60). Set to actual WAV duration. |
| 13 | `gpt_cond_chunk_len` | 4 | GPT conditioning chunk size |
| 14 | `sound_norm_refs` | 0 | Normalise reference before cloning (0/1) |

Fully backward-compatible — v20/v21/v22 scripts run unchanged.

### Audio parameter block `[]` — 16 values (v23)

```
[N, LANG, speed, vol, eq_low, eq_mid, eq_high, hp, lp, NR, comp, de-ess, reverb, noise_gate, pan, limiter]
```

| Pos | Parameter | Default | Description |
|-----|-----------|---------|-------------|
| 1 | `N` | — | Voice number |
| 2 | `LANG` | FR | Language code (FR, EN, ES, DE, IT, PT, PL, TR, RU, NL, CS, AR, ZH-CN, HU, KO, JA, HI) |
| 3 | `speed` | 1.0 | Rubberband speed factor (0.5–2.0) |
| 4 | `vol` | 0 | Volume adjustment (dB) |
| 5 | `eq_low` | 0 | Low EQ 80–300 Hz (dB) |
| 6 | `eq_mid` | 0 | Mid EQ 300–3000 Hz (dB) |
| 7 | `eq_high` | 0 | High EQ 3000–8000 Hz (dB) |
| 8 | `hp` | 0 | Highpass filter (Hz, 0=off) |
| 9 | `lp` | 0 | Lowpass filter (Hz, 0=off) |
| 10 | `NR` | 0 | Noise reduction strength (0=off, 0.5=moderate, 2=aggressive) |
| 11 | `comp` | 0 | Compression strength (0=off, 0.5=moderate) |
| 12 | `de-ess` | 0 | De-esser strength (0=off, 0.5=moderate) |
| 13 | `reverb` | 0 | Reverb wet level (0=off, 0.3=subtle) |
| 14 | `noise_gate` | 0 | Noise gate threshold (dB, 0=off, e.g. -40=gentle) |
| 15 | `pan` | 0 | Stereo pan (-1.0=left, 0=centre, +1.0=right) |
| 16 | `limiter` | 0 | Output limiter (0=off, 1=on) |

Processing order: Trim → Filters → EQ → NR → De-esser → Compression → Noise gate → Reverb → Fades → Pan → Limiter

### Per-voice config persistence

Short blocks inherit the last full config for that voice — only write what changes:

```
# First full config — memorised for voice 1
[1, FR, 0.9, 6, -5, 1, -2, 95, 8000, 0.35, 0.35, 0.5, 0, 0, 0, 1]
Première phrase.
[pause=2s]

# Only speed changes — rest inherited
[1, FR, 0.8]
Deuxième phrase.
[pause=2s]

# Only language changes
[1, EN]
Third sentence in English.
```

Config resets at the start of each generation run.

### Multi-reference voices

Pass multiple reference WAV files per voice in the GUI (space-separated). XTTS averages the speaker embeddings for a more robust clone. In the CLI, voice groups are separated by `--`:

```bash
generator.py script.txt output.wav \
    ref1.wav ref2.wav ref3.wav \
    -- \
    hollie.wav \
    --mp3-bitrate 192 --mp3-mode cbr
```

**Using multiple references significantly improves cloning quality** — XTTS has more context to reconstruct the voice faithfully.

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
| F0 median, jitter | YIN / torchcrepe / pyin | Voice type, hp/lp, len_pen |
| HNR | Praat | noise_reduction |
| Shimmer APQ5 | Praat | compression |
| Shimmer + jitter score | Praat | temperature, rep_pen |
| Formants F1/F2 | Praat | highpass refinement |
| Syllable tempo | Praat | speed refinement |
| Voiced RMS | librosa | volume |
| Sibilance | librosa | de-esser |
| Duration | — | gpt_cond_len |

### F0 engine selection

Each voice row has its own F0 engine selector:

- **Prec unchecked** → fast YIN, `none` label (read-only)
- **Prec checked** → precise mode, choose `auto / crepe / pyin`

Recommended:
- Bass/baritone voices → `pyin` (torchcrepe detects harmonics on low voices)
- Soprano/high voices → `auto` or `crepe`
- Meditation voices → `none` (fast, Praat does the real work)

### Multi-reference averaging

When multiple files are passed for the same voice, each is analysed separately and all numeric parameters are averaged into a single `{}[]` block:

```bash
voice_analyser.py --precise --f0-engine pyin --start-num 1 ref1.wav ref2.wav ref3.wav FR
```

---

## Validator

The Validator generates multiple audio variations with different parameter values, concatenates them into a single WAV file (each preceded by a spoken label), and lets you listen and choose the best value.

### How to use

1. Select your voice reference file(s)
2. Choose language and fill mode (Default / Zero)
3. Click **Fill** to auto-populate the XTTS and Audio blocks — or paste directly from voice_analyser
4. Add one or more parameters to test with `+ Add parameter`
5. When you select a parameter, the Values field auto-fills with the current value from the blocks
6. Edit the values to test (space-separated)
7. Click **Generate** → listen to the output WAV in the Player

### Parameter types

**XTTS params** (`seed`, `temp`, `top_k`, `top_p`, `rep_pen`, `len_pen`, `gpt_cond_len`):
Each combination requires a full XTTS generation — slower but necessary since these change the GPT output.

**Audio params** (`speed`, `vol`, `eq_low`, `eq_mid`, `eq_high`, `hp`, `lp`, `NR`, `comp`, `de-ess`, `reverb`, `noise_gate`, `pan`):
XTTS generates **once**, then each filter value is applied in milliseconds. Very fast for audio calibration.

### Cartesian product

Add multiple parameters to test all combinations:

```
[ seed  ] [ 0  42  100  ]     → 3 values
[ lp    ] [ 7000  9000  ]     → 2 values
                               Total: 6 combinations
```

The validator generates all 6 combinations (seed=0/lp=7000, seed=0/lp=9000, seed=42/lp=7000, etc.)

### Fill modes

- **Default** → fills `{}` and `[]` with standard XTTS defaults
- **Zero** → fills with zeros where valid (minimum values used where 0 would be invalid: temp=0.01, top_k=1, rep_pen=1.0, gpt_cond_len=6)

### Important note on seed

**The seed interacts with the voice embedding in an unpredictable way.** A seed that works perfectly for one voice reference may produce an accent or degraded quality with another. Always test several seeds (0, 7, 13, 42, 100, 200) for each new voice reference file. Use seed=0 for random generation.

### CLI usage

```bash
# Test seeds
python voice_validator.py Elo.wav Elo2.wav FR \
    --param seed --values 0 7 13 42 100 \
    --xtts-block "{1, 0, 0, 200, 100, 250, 0.72, 55, 0.88, 4.5, 1, 60, 4, 0}" \
    --audio-block "[1, FR, 0.9, 6, -5, 1, -2, 95, 8000, 0.35, 0.35, 0.5, 0, 0, 0, 1]" \
    --output validation_seed.wav

# Test audio params (fast — single XTTS generation)
python voice_validator.py Elo.wav FR \
    --param hp --values 60 80 95 120 \
    --param lp --values 7000 8000 9000 \
    --output validation_hp_lp.wav
```

---

## Voice Separation

```bash
extract_voices.py input.mp3 output.wav \
    --keep "vocals only" \
    --remove-music --demucs-model htdemucs_ft \
    --dereverberate deepfilter \
    --device cuda \
    --silence auto --min-silence 0.3
```

| Option | Values | Description |
|--------|--------|-------------|
| `--keep` | `vocals only`, `female`, `male`, `all` | Stems to keep |
| `--demucs-model` | `htdemucs`, `htdemucs_ft`, `mdx_extra` | Model (htdemucs_ft = best quality/VRAM balance) |
| `--dereverberate` | `none`, `noisereduce`, `wpe`, `deepfilter` | Dereverberation |
| `--device` | `cuda`, `cpu` | Processing device |

**Dereverberation comparison:**

| Engine | Speed | Quality | Notes |
|--------|-------|---------|-------|
| `none` | ★★★ | — | No processing |
| `noisereduce` | ★★★ | ★★ | Good for static noise |
| `wpe` | ★ | ★★ | Single-threaded, very slow |
| `deepfilter` | ★★★ | ★★★ | GPU recommended |

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

**Using multiple reference files** (2–3 clips of the same voice) significantly improves cloning quality — XTTS averages the embeddings for a more complete voice representation.

---

## Recommended workflow

1. **Extract reference audio** → Video→Audio tab with XTTS preset (WAV, mono, 22050 Hz)
2. **Clean the reference** → Vox tab with demucs + deepfilter if needed
3. **Analyse the voice** → Analyser tab, add multiple reference files, choose `pyin` for low voices
4. **Validate key parameters** → Validator tab, test `seed` first (most impactful), then audio params
5. **Generate the meditation** → Generator tab, paste the validated `{}` and `[]` blocks

---

## TTS Backend Comparison (tested April 2026)

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

**Voice has an English accent or sounds wrong with a single reference**
Try multiple reference files (2–3 clips). Also test different seeds with the Validator — seed=42 may produce poor results for some voices.

**torchcrepe returns F0=1900Hz or voiced=0% on bass voices**
Use `pyin` engine — torchcrepe can detect harmonics instead of the fundamental on low voices.

**`[!] File not found: --mp3-bitrate`**
You are running an old version of the generator. Replace with the latest `guided_meditation_generator_v23.py`.

**WPE dereverberation is very slow**
WPE is inherently single-threaded and sequential. Use `deepfilter` instead.

**GUI opens Browse in wrong directory**
Launch `xtts_studio.py` with the correct path — `XTTS_ROOT` is auto-detected from the script location.

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

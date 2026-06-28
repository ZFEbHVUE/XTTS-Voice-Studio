# XTTS Voice Studio

A complete toolkit for voice cloning, guided meditation generation, audio analysis, parameter validation and automated optimisation — built around Coqui XTTS v2 with a unified Tkinter GUI.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![License](https://img.shields.io/badge/License-MIT-green) ![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20WSL-lightgrey)

---

## Overview

XTTS Voice Studio is a personal production suite for:

- **Cloning voices** from short audio samples via XTTS v2
- **Generating guided meditations** with multi-voice narration, ambient music and punctual sound cues
- **Analysing voices** acoustically (Praat + librosa) to derive optimal XTTS parameters automatically
- **Validating parameters** empirically by generating and comparing audio variations
- **Comparing clones** to their reference voice spectrally and optimising audio parameters automatically via iterative feedback
- **Transcribing** audio and video to XTTS-ready scripts
- **Separating voices** by gender from mixed audio sources
- **Converting video** to audio in various formats

Every tool is accessible through a single Tkinter interface (`xtts_studio.py`) or directly from the command line.

![XTTS Voice Studio GUI](docs/gui_main.png)

---

## Installation

### Requirements

- Python 3.10+
- Miniconda or Anaconda (recommended)
- CUDA 12.x + compatible GPU (optional but significantly faster)
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
pip install faster-whisper librosa pydub numpy soundfile pyrubberband scipy
pip install torchcrepe demucs noisereduce nara-wpe deepfilternet
pip install praat-parselmouth
pip install speechbrain          # ECAPA-TDNN speaker identity (comparator)

sudo apt install ffmpeg rubberboard-cli
```

### Step 4 — Launch the GUI

```bash
conda activate xtts
python ~/XTTS-Voice-Studio/Python_Scripting/xtts_studio.py
```

`XTTS_ROOT` is auto-detected from the script location — no hardcoded paths. You can rename or move the directory freely.

---

## Directory Structure

```
XTTS-Voice-Studio/
├── Python_Scripting/
│   ├── xtts_studio.py                      # Tkinter GUI (main entry point)
│   ├── guided_meditation_generator_v23.py  # Meditation generator
│   ├── voice_analyser.py                   # Acoustic analysis → XTTS params
│   ├── voice_validator.py                  # Empirical parameter validation
│   ├── voice_comparator.py                 # Identity (ECAPA) + LS EQ optimisation
│   ├── speaker_identity.py                 # ECAPA-TDNN speaker embedding / cosine
│   ├── ltas_match.py                       # LTAS least-squares EQ fit (exact RBJ responses)
│   ├── extract_voices.py                   # Vocal separation by gender
│   ├── transcribeSong2txt_with_pause.py    # Audio transcription
│   └── video2txt.py                        # Video transcription
├── Prompts/                                # Text scripts for meditation generation
├── Voices_Cloning/                         # Voice reference samples (.wav)
├── Ambient_Musics/                         # Background ambient loops
├── Punctual_sounds/                        # One-shot audio cues (bells, chimes)
├── MP3toTXT/                               # Audio/video sources for transcription
├── Output_Song_files/                      # Generated meditation WAVs
└── README.md
```

---

## GUI Tabs

### [Gen] Generator

Generates guided meditation audio from a text script with pause markers and voice blocks.

- Per-voice rows with multi-reference browse (space-separated WAV files)
- `+ Add voice` button
- Ambient music and punctual sound cue inputs
- MP3/FLAC/OGG output options

### [Ana] Analyser

Analyses voice reference files and produces ready-to-paste XTTS parameter blocks.

- Per-voice rows with multi-file browse (multiple files averaged per voice)
- Per-voice **Praat / Librosa** selector — Praat gives more accurate results
- Per-voice F0 engine selector: `none` (fast YIN) / `auto` / `crepe` / `pyin` (only in Prec mode)
- FINAL SUMMARY with all voices in ready-to-paste format including analysis options

### [Txt] Transcription

Transcribes audio or video files to XTTS-ready text with pause markers.

### [Vox] Voice sep.

Separates vocal stems from mixed audio or video sources.

- Accepts audio (WAV, MP3, FLAC) and video (MP4, MKV, AVI, MOV, WEBM...)
- F0-based male/female separation
- demucs music removal (htdemucs, htdemucs_ft, mdx_extra)
- Dereverberation: none / noisereduce / wpe / deepfilter

### [Pit] Pitch

Applies pitch correction to cloned voice audio.

### [Vid] Video→Audio

Extracts audio from any video file with format, channel and sample rate options. Includes an **XTTS preset** button (WAV + mono + 22050 Hz).

### [Val] Validator

Empirically validates XTTS parameters by generating multiple variations and combining them into a single audio file for A/B listening.

- Multi-parameter rows with dynamic combobox filtering (used params excluded from other rows)
- Cartesian product of all parameter combinations
- Auto-fill Values field from XTTS/Audio blocks when selecting a parameter
- **Default** / **Raw** fill buttons
- Single XTTS generation for audio params (fast), N generations for XTTS params

### [Cmp] Comparator

Optimises a clone in two principled stages instead of spectral-band heuristics.

- **Stage 1 — identity (seed search):** generates the clone for several seeds and keeps the one whose **ECAPA-TDNN** speaker embedding is closest (cosine) to the reference. Text-independent — this is what "same voice" actually measures. Automates the manual seed hunt.
- **Stage 2 — tone (least squares):** fits the generator's exact 3-band peaking EQ + volume so the clone's long-term average spectrum (LTAS) matches the reference, by bounded least squares on the real RBJ filter responses.
- Reports an identity cosine per seed and the in-band LTAS residual (dB) before/after.
- `[]` Audio params field updates automatically with the fitted block.

---

## Guided Meditation Generator

### XTTS parameter block `{}` — 14 values

```
{N, seed, trim_start, trim_end, fade_in, fade_out, temp, top_k, top_p, rep_pen, len_pen, gpt_cond_len, gpt_cond_chunk_len, sound_norm_refs}
```

| Pos | Parameter | Default | Description |
|-----|-----------|---------|-------------|
| 1 | `N` | — | Voice number |
| 2 | `seed` | 0 | Random seed (0 = random). **Test several seeds per voice — the seed interacts with the embedding in an unpredictable way.** |
| 3 | `trim_start` | 0 | Trim from start (ms) |
| 4 | `trim_end` | 0 | Trim from end (ms) |
| 5 | `fade_in` | 100 | Fade-in (ms) |
| 6 | `fade_out` | 250 | Fade-out (ms) |
| 7 | `temperature` | 0.65 | GPT sampling temperature |
| 8 | `top_k` | 50 | GPT top-k |
| 9 | `top_p` | 0.85 | GPT top-p |
| 10 | `rep_pen` | 5.0 | Repetition penalty |
| 11 | `len_pen` | 1.0 | Length penalty |
| 12 | `gpt_cond_len` | 30 | Reference WAV seconds used for cloning (up to 60) |
| 13 | `gpt_cond_chunk_len` | 4 | GPT conditioning chunk size |
| 14 | `sound_norm_refs` | 0 | Normalise reference before cloning (0/1) |

### Audio parameter block `[]` — 16 values

```
[N, LANG, speed, vol, eq_low, eq_mid, eq_high, hp, lp, NR, comp, de-ess, reverb, noise_gate, pan, limiter]
```

| Pos | Parameter | Default | Description |
|-----|-----------|---------|-------------|
| 1 | `N` | — | Voice number |
| 2 | `LANG` | FR | Language code |
| 3 | `speed` | 1.0 | Rubberband speed factor (0.5–2.0) |
| 4 | `vol` | 0 | Volume adjustment (dB) |
| 5 | `eq_low` | 0 | Low EQ 80–300 Hz (dB) |
| 6 | `eq_mid` | 0 | Mid EQ 300–3000 Hz (dB) |
| 7 | `eq_high` | 0 | High EQ 3000–8000 Hz (dB) |
| 8 | `hp` | 0 | Highpass filter (Hz, 0=off) |
| 9 | `lp` | 0 | Lowpass filter (Hz, 0=off) |
| 10 | `NR` | 0 | Noise reduction (0=off) |
| 11 | `comp` | 0 | Compression (0=off) |
| 12 | `de-ess` | 0 | De-esser (0=off) |
| 13 | `reverb` | 0 | Reverb wet level (0=off) |
| 14 | `noise_gate` | 0 | Noise gate threshold dB (0=off) |
| 15 | `pan` | 0 | Stereo pan (-1.0=left, 0=centre, +1.0=right) |
| 16 | `limiter` | 0 | Output limiter (0=off, 1=on) |

### Per-voice config persistence

Short blocks inherit the last full config for that voice:

```
{1, 42, 0, 420, 100, 250, 0.68, 50, 0.85, 5, 1.05, 60, 4, 0}
[1, FR, 0.9, 5, -3, 0, -4, 95, 8500, 0.35, 0.35, 0.5, 0, 0, 0, 1]
Première phrase.
[pause=2s]

[1, FR, 0.8]
Deuxième phrase plus lente.
[pause=2s]
```

### Multi-reference voices

Multiple reference files per voice significantly improve cloning quality — XTTS averages the speaker embeddings:

```bash
generator.py script.txt output.wav ref1.wav ref2.wav ref3.wav -- hollie.wav
```

### Parallel voice overlay

```
[parallel, offset=1s,5s]
{1, 42, ...} [1, FR, ...] First voice starts immediately.
{2, 0, ...}  [2, FR, ...] Second voice enters at 1s.
{3, 0, ...}  [3, FR, ...] Third voice enters at 5s.
[/parallel]
```

---

## Voice Analyser

Analyses one or more reference files and produces ready-to-paste parameter blocks.

### Analysis modes

| Mode | Engine | Speed | Accuracy |
|------|--------|-------|----------|
| Fast (default) | YIN + Librosa | ~8s | Good |
| Fast + Praat | YIN + Praat | ~8s | **Better** |
| Precise + Praat | pyin/crepe + Praat | 30–90s | Best for atypical voices |

**Conclusion from testing:** Praat vs Librosa is the meaningful choice — the F0 engine (none/pyin/crepe) has negligible impact on typical voices. Praat gives more accurate `NR`, `comp`, `hp` and `temp` via HNR, APQ5 and formant analysis.

### Multi-reference averaging

Multiple files per voice are averaged into a single block — more reference material = better cloning:

```bash
voice_analyser.py --precise ref1.wav ref2.wav ref3.wav FR
```

### Analysis output

```
# Voice 1 [FR]  soprano / high voice  214 Hz
# Analysis: Praat + pyin | seed=42
{1, 42, 0, 200, 100, 250, 0.72, 55, 0.88, 4.5, 1, 60, 4, 0}
[1, FR, 0.9, 0, 0, 0, 0, 95, 8000, 0.3, 0.3, 0.3, 0, 0, 0, 1]
```

`eq_low/eq_mid/eq_high` and `vol` are emitted as `0` **by design** — they are
fitted empirically by the Comparator (least-squares LTAS match on the actual
clone). Deriving EQ from the reference alone double-counts the spectral balance
the clone already inherits through cloning. `NR`, `comp` and `de-ess` are gentle
starting points (capped low); confirm by ear. The Analyser owns the generation
priors (the `{}` block + safe `hp`/`lp`); the Comparator owns the tonal match.

---

## Validator

Generates multiple audio variations with different parameter values, concatenates them with spoken labels, and lets you listen and choose.

### Workflow

1. Paste `{}` and `[]` blocks from voice_analyser (or use Default/Raw fill)
2. Select a parameter — Values field auto-fills with the current value
3. Edit values around the current value to test
4. Add more parameters with `+ Add parameter` — cartesian product is generated
5. Listen to the output WAV — each variation is preceded by a spoken label

### Parameter types

- **Audio params** (`hp`, `lp`, `eq_*`, `NR`, `comp`...) → single XTTS generation + N filter applications (fast)
- **XTTS params** (`seed`, `temp`, `rep_pen`...) → N separate XTTS generations (slower)

### Important note on seed

The seed interacts with the voice embedding unpredictably. A seed that works perfectly for one reference may produce accent degradation with another. Always test several seeds (0, 7, 13, 42, 100, 200) for each voice. Use seed=0 for random generation.

---

## Comparator

Optimises a clone in two principled stages. Post-processing can only colour the
tone of a clone — it cannot change *who* the clone sounds like. So identity and
tone are handled separately, with the right tool for each.

### Stage 1 — identity (seed search via ECAPA-TDNN)

The voice identity (timbre, accent) is decided by XTTS at generation time, set by
the interaction between the reference embedding and the seed. The comparator
generates the clone for each candidate seed and scores it against the reference
with **ECAPA-TDNN speaker-verification cosine similarity** (SpeechBrain
`spkrec-ecapa-voxceleb`). This metric is text-independent — it answers "same
speaker?" regardless of what was said. The best seed is kept automatically,
replacing the manual `0 7 13 42 100 200` hunt.

| Cosine | Interpretation |
|--------|----------------|
| > 0.75 | same speaker |
| 0.55–0.75 | close |
| < 0.55 | different — reference likely too short/noisy, or wrong language |

ECAPA runs on CPU by default (a few short clips score in ~1 s) so all VRAM stays
with XTTS — relevant on 4 GB cards.

### Stage 2 — tone (least-squares LTAS EQ fit)

On the winning clone, the generator's exact 3-band peaking EQ + volume are fit by
**bounded least squares** so the clone's long-term average spectrum (LTAS)
matches the reference:

1. Compute the LTAS (dB, log-frequency) of reference and clone, peak-normalised (shape only — level handled separately by `vol` from the RMS gap).
2. Target correction `D(f) = LTAS_ref(f) − LTAS_clone(f)`.
3. Model the EQ as the **exact** FFmpeg `equalizer` RBJ peaking responses (low f0=200/BW=200, mid f0=1500/BW=2000, high f0=5000/BW=3000) plus a nuisance offset.
4. Solve `min ‖EQ(g) − D‖²` over the 80 Hz–8 kHz speech band with gains bounded to ±6 dB (`scipy.optimize.least_squares`, perceptually weighted).
5. `hp`/`lp` derived from the LTAS roll-off at the extremes.

Because the modelled responses are the real filters the generator applies, the
fit corrects the actual EQ rather than an approximation. The reported residual is
the weighted in-band RMS error (dB) before vs after — a meaningful, homogeneous
number, unlike the old composite score that summed dB + Hz + percentages.

### Why this matters

The old comparator iteratively nudged EQ to match spectral-band *percentages* of
two **different** utterances (reference text ≠ clone text), so it chased phonetic
differences, not voice differences — and its score mixed incommensurable units.
It could never improve identity because EQ is post-processing. The rework targets
identity directly (ECAPA) and solves tone as a well-posed least-squares problem.

### Tips

- For the sharpest tonal match, generate the clone on the **reference's own transcript** (`--text-file`) so the LTAS compares like-for-like phonetics. Over 20–60 s of speech the LTAS still approximates the speaker envelope on arbitrary text, but matched text is cleaner.
- The `{}` XTTS params other than `seed` (temp, rep_pen…) are not searched here — use the Validator for those, or the Analyser for sensible priors.
- More / cleaner reference audio raises the achievable cosine more than any post-processing.

---

## Recommended Workflow

1. **Extract reference audio** → Video→Audio tab with XTTS preset (WAV, mono, 22050 Hz)
2. **Clean if needed** → Vox tab with demucs + deepfilter
3. **Analyse** → Analyser tab, Praat mode, multiple reference files (gives `{}`/`[]` priors)
4. **Find best seed + tone** → Comparator tab: ECAPA seed search picks the identity-best seed and least-squares fits the EQ in one run
5. **Validate other XTTS params** → Validator tab, test `temp`, `rep_pen` if needed
6. **Generate** → Generator tab with the final `{}` and `[]` blocks

---

## TTS Backend Comparison (tested 2026)

| Model | French | Cloning | GPU | Verdict |
|-------|--------|---------|-----|---------|
| **XTTS v2** | ✅ native | ✅ excellent | 4GB+ | **Best for French** |
| Chatterbox | ✅ correct | ✅ good | 8GB+ | Acceptable |
| F5-TTS | ❌ English accent | ✅ timbre | 8GB+ | Not usable for French |
| IndexTTS2 | ❌ English accent | ✅ timbre | 8GB+ | Not usable for French |

---

## Optimal reference audio for XTTS

| Property | Recommended |
|----------|-------------|
| Format | WAV (lossless) |
| Channels | Mono |
| Sample rate | 22050 Hz |
| Duration | 20–60 seconds |
| Content | Clean speech only — no music, no echo, no noise |

Using **multiple reference files** (2–3 clips) significantly improves cloning quality.

---

## Troubleshooting

**Voice sounds like a Canadian/English accent on French text**
Test different seeds with the Validator. A single reference file may also be insufficient — use 2–3 reference files and average them.

**torchcrepe returns aberrant F0 on bass voices**
Use `pyin` engine — torchcrepe can detect harmonics instead of the fundamental on low voices.

**CUDA out of memory with demucs**
Switch from `mdx_extra` to `htdemucs_ft` or use `--device cpu`.

**Stop button does not kill the process**
Some GPU operations ignore SIGKILL until the current CUDA kernel completes. Wait a moment after pressing Stop.

**`eq_high` takes non-integer values like -3.775**
Fixed in recent version — values are now rounded to 1 decimal place.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

[ZFEbHVUE](https://github.com/ZFEbHVUE) — the username reads as `STEPHANE` when mirrored vertically.

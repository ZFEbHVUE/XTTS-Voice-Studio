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
# CUDA torch first (pick your CUDA version at pytorch.org):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Everything else, pinned and documented:
pip install -r requirements.txt

# System tools (ffmpeg for audio/video, rubberband for pitch-preserving tempo):
sudo apt install ffmpeg rubberband-cli
```

Run this in the `xtts` env on EVERY machine you clone the repo to — git syncs
the code, never the Python environment (the recurring `ModuleNotFoundError:
librosa/speechbrain` on a freshly pulled machine is exactly this).

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
│   ├── voice_validator.py                  # Param sweep + accent/identity scoring
│   ├── voice_comparator.py                 # Closed-loop EQ/vol fit (LTAS, least squares)
│   ├── speaker_identity.py                 # ECAPA speaker embedding + identity CLI (ref vs clones)
│   ├── pron_score.py                       # French accent scoring via faster-whisper
│   ├── ltas_match.py                       # LTAS least-squares EQ fit (exact RBJ responses)
│   ├── xtts_clone.py                        # Low-level XTTS gen honouring all cloning knobs
│   ├── xtts_optimize.py                     # Coordinate-descent search of sampling params
│   ├── xtts_pipeline.py                     # One-shot: curate -> analyse -> optimise -> fit
│   ├── curate_reference.py                  # Curate a clean reference by ECAPA coherence
│   ├── prepare_rvc_dataset.py               # Build an RVC training dataset (10+ min utterances)
│   ├── rvc_convert.py                       # Timbre conversion via a trained Applio/RVC model
│   ├── chatterbox_ab.py                     # A/B sample vs Chatterbox (own env, see below)
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

### [Auto] Pipeline

One-shot mode: add your voices (each with its own language), press Run, and the
whole chain executes per voice — curation → analysis (Praat/pyin priors) →
optimisation (seed screen + least-squares temp surface) → closed-loop tone fit
(auto-text) — then prints the final numbered `{}` / `[]` blocks ready to paste
into the generator prompt. Also writes a `*_pipeline_clone.wav` per voice:
**listen to it before generating** — the scores don't hear naturalness.
CLI equivalent: `xtts_pipeline.py --voice lea.wav FR --voice john.wav EN`.

### [Cur] Curation

Builds a clean reference before anything else: windows the raw recording, embeds
each window with ECAPA, and keeps only the most speaker-coherent ~45 s (breaths,
noise, reverb tails, off-voice segments are dropped). Output auto-named
`<ref>_curated.wav`; hand-off buttons push it straight into the Analyser or
Optimiser. Run the whole downstream pipeline on the curated file — this is the
single biggest quality lever (Lea: identity 0.69 → 0.85).

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
- Speaker separation methods: `f0` (pitch threshold/k-means), **`ecapa`**
  (ECAPA-TDNN embedding clustering — separates by WHO speaks, not by pitch,
  so two same-register speakers are separable; F0 methods fail there by
  construction), `sepformer`, `pyannote`
- demucs music removal (htdemucs, htdemucs_ft, mdx_extra) with `--demucs-shifts`
  (test-time augmentation: 1=fast, 2=default, 5=cleanest vocal stem)
- Dereverberation: none / noisereduce / wpe / deepfilter — with a measured
  voice-health guard that warns when denoising eats the voice's high band
  (dulls timbre, hurts cloning identity; never stack denoisers)
- Pitch-preserving tempo via rubberband (R3 `--fine` engine when rubberband≥3
  is installed; loud warning if only the metallic librosa fallback is available)

### [Pit] Pitch

Applies pitch correction to cloned voice audio.

### [Vid] Video→Audio

Extracts audio from any video file with format, channel and sample rate options. Includes an **XTTS preset** button (WAV + mono + 22050 Hz).

### [Val] Validator

Sweeps XTTS/audio parameters, generates a variation per value for A/B listening, and **scores** each variation to rank them and produce the winning `{}` block.

- Multi-parameter rows with dynamic combobox filtering (used params excluded from other rows)
- Cartesian product of all parameter combinations
- Auto-fill Values field from XTTS/Audio blocks when selecting a parameter
- **Default** / **Raw** fill buttons
- Single XTTS generation for audio params (fast), N generations for XTTS params
- **Scoring** (when a `{}` param is swept): accent (faster-whisper → `french`/WER) + identity (ECAPA cosine vs reference) per variant, a ranked table, and the best `{}` to paste into the Comparator. `--no-score` to disable.

### [Opt] Optimiser

Automated search of the sampling parameters against a measured objective
(`w_accent`·french + `w_identity`·identity). RSM method: seed screen →
least-squares temp surface per kept seed → inertness probe (rep_pen/top_p) →
Pareto front. Fields: voice ref(s), starting `{}` (hand-off from the Analyser),
seeds to screen, budget, weights, whisper model, device. `--probe-beams`
optionally tests beam-search/greedy decoding on the winner. Prints the winning
`{}` block; **→ Comparator** pushes it to the next stage.

### [Cmp] Comparator

Takes a **frozen** `{}` block (seed/temp chosen in the Validator) and fits only the post-processing tone, in a closed loop against the reference.

- Generates the clone once on the reference text, then iteratively fits the generator's exact 3-band peaking EQ + volume so the clone's long-term average spectrum (LTAS) matches the reference (bounded least squares on the real RBJ responses).
- Each pass renders the full chain (hp/lp/NR/comp/limiter), re-measures, and adds the LS correction — closing the loop on the post-chain colouring.
- No seed search, no accent/identity scoring (that is the Validator's job).
- `[]` Audio params field updates automatically with the fitted block.

---

### [RVC] Timbre

The stage past the zero-shot ceiling: XTTS clones top out around identity
0.7–0.85 ("close", not "the person"). An RVC model TRAINED on the target
(Applio) re-voices the XTTS output with their actual timbre. The tab covers:
1) building the training dataset (ECAPA-coherent 3–10 s utterances, 10+ min
needed — `prepare_rvc_dataset.py`), 2) converting any XTTS output through the
trained model (`rvc_convert.py`, runs Applio's own env), 3) measuring identity
before/after against the real reference. Training itself stays in Applio's UI
— see docs/RVC_GUIDE.md for settings and the A/B protocol.

---

## Guided Meditation Generator

### XTTS parameter block `{}` — 14 values

```
{N, seed, trim_start, trim_end, fade_in, fade_out, temp, top_k, top_p, rep_pen, len_pen, gpt_cond_len, gpt_cond_chunk_len, sound_norm_refs}

An optional 15th value, `num_beams`, switches the GPT decode to beam search
(default 1 = sampling). `len_pen` only has an effect when `num_beams > 1` —
XTTS forwards it to HF `generate()`, which ignores it in sampling mode (the
runtime warning about `num_beams`/`length_penalty` says exactly this). The
optimiser's `--probe-beams` tests beam/greedy decoding and emits the 15-value
block if it wins.
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

> **⚠ Important — XTTS cloning knobs and `tts_to_file`.** In TTS 0.22.0, the
> high-level `tts_to_file()` path (`Xtts.synthesize`) **overrides**
> `gpt_cond_len`, `gpt_cond_chunk_len`, `max_ref_len` and `sound_norm_refs` with
> the `XttsConfig` defaults (**12 / 4 / 10 / False**) *after* applying your
> kwargs — so passing them has no effect, and `max_ref_len` is never exposed.
> The speaker embedding therefore only ever uses the first **10 s** of reference
> and the GPT conditioning only **12 s**, however much clean reference you give.
>
> To make them bite, generate via the low-level path (`get_conditioning_latents`
> + `inference`) — implemented in **`xtts_clone.py`**. The **Comparator already
> uses it** (and exposes `--max-ref-len`, default 30 s). The Generator still goes
> through `tts_to_file`; until it is switched over (set the four values on
> `model.config` before generating, or use `xtts_clone`), its final renders use
> the crippled 12 s / 10 s conditioning regardless of the `{}` block.

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

Sweeps parameters, generates one variation per value (concatenated with spoken
labels for A/B listening), and **scores** each variation so you can pick — and
hand off — the winning `{}` block objectively, not just by ear.

### Workflow

1. Paste `{}` and `[]` blocks from the Analyser (or use Default/Raw fill)
2. Select a parameter — Values field auto-fills with the current value
3. Edit values around the current value to test
4. Add more parameters with `+ Add parameter` — cartesian product is generated
5. Listen to the output WAV (each variation has a spoken label) **and** read the ranked score table
6. Paste the printed winning `{}` block into the Comparator

### Scoring (the upgrade)

When a generation (`{}`) param is swept, each variant is scored on two axes —
the same signals used to judge a clone:

- **Accent / pronunciation** — faster-whisper transcription → a `french` score in [0,1] from the detected language (heard as English → heavy penalty), `avg_logprob` (low = mispronounced), and WER vs the known text. This is the axis that catches the foreign-accent problem.
- **Identity / timbre** — ECAPA-TDNN cosine vs the reference.

The variants are ranked (`--prioritise accent` by default; `identity` or
`balanced` available), and the best `{}` block is printed ready to paste. Audio-
only sweeps (`eq_*`, `hp`…) are not scored — every variant is the same clone.

> The scores are ASR/embedding proxies, not calibrated meters — keep your ear as
> the final judge. But they *move* with accent and timbre, so they rank reliably.

### Important note on seed

The seed interacts with the voice embedding unpredictably. A seed that works for
one reference can degrade the accent on another. Sweep several (0, 7, 13, 42, 100,
200) and let the `french` score rank them. `seed=0` = random.

### Generation path

The Validator generates via `xtts_clone.py`, so swept cloning knobs
(`gpt_cond_len`, `max_ref_len`, `sound_norm_refs`) actually take effect — and the
winning `{}` is produced under the same conditioning the Comparator will use.
`--max-ref-len` (default 30 s) controls the speaker-embedding reference length.

---

## Comparator

Takes a **frozen** `{}` block (seed, temp, rep_pen… already chosen in the
Validator) and fits only the post-processing tone, in a closed loop against the
reference. Post-processing can colour the tone but cannot change *who* the clone
sounds like — that was decided upstream — so the Comparator does one thing well.

### How it works

1. Generates the clone **once** on the reference text (via `xtts_clone.py`, honouring the cloning knobs).
2. Renders the full post chain (hp/lp/NR/comp/limiter) with the current `[]`.
3. Measures the rendered clone's LTAS (dB, log-frequency) against the reference.
4. Fits the generator's **exact** 3-band peaking EQ (FFmpeg `equalizer` RBJ responses: low f0=200/BW=200, mid 1500/2000, high 5000/3000) + volume by bounded least squares (`scipy.optimize.least_squares`, ±6 dB, speech-band weighted), `hp`/`lp` from the LTAS roll-off.
5. Adds the correction and repeats (`--iterations`, default 3) until the largest change falls below `--conv-threshold` dB (default 0.5) — closing the loop on the post-chain colouring, not just the raw clone.

Level (`vol`) comes from the RMS gap; EQ shape from the LS fit on peak-normalised
LTAS. The reported residual is the weighted in-band RMS error (dB) before vs after
— a homogeneous, meaningful number.

### Tips

- For the sharpest tonal match, use `--auto-text`: the reference is transcribed (faster-whisper, CPU) and the clone is generated on its own words, so the LTAS compares like-for-like phonetics. (`--text-file` does the same with a transcript you provide.)
- The LS fit is perceptually weighted (A-weighting): errors are penalised where the ear hears them (~1–5 kHz) rather than uniformly across the spectrum.
- `--target-dbfs -20` switches the volume fit to an absolute production level instead of matching the reference. Use it whenever the reference is quiet: matching a -33 dBFS reference produces an unusable meditation level (the fitted `vol` was the honest match, not a good render level).
- comp and de-ess are now fitted by measurement too (crest factor and 5–8 kHz sibilance vs the reference), each kept only if the measured gap actually shrinks — `--no-fit-dynamics` disables it. NR, reverb, noise_gate and pan stay at 0 by design: no measurable target, and the gate chops speech.
- `eq_*` and `vol` in the input `[]` are overwritten by the fit; `NR`/`comp`/`de-ess`/`hp`/`lp` are kept from the block (Analyser priors).
- The `{}` is never modified — fix seed/temp in the Validator first.

---

## Advanced — input curation & sampling search

Two optional tools that go beyond per-parameter tuning. Both reuse the same
scorers (ECAPA identity, faster-whisper accent) — no new dependencies.

### Reference curation (`curate_reference.py`)

Optimises the **input** to cloning rather than the seed. It windows the
reference, embeds each window with ECAPA, builds a robust (outlier-trimmed)
centroid, scores each window's coherence, and concatenates the most consistent
windows (up to a target duration) into a clean reference — dropping breaths,
reverberant tails, noise, or a stray second voice that would dilute the speaker
embedding.

```bash
python curate_reference.py raw_ref.wav -o curated.wav --keep-seconds 45
```

Feed `curated.wav` to the Validator / Comparator / Generator.

### Sampling optimiser (`xtts_optimize.py`)

`--probe-beams` adds an optional decode-mode stage: after the winner is found it
probes beam search (`num_beams=3`, which finally activates `length_penalty`) and
greedy decoding (`do_sample=False`), adopting either only if the measured score
improves — listen before trusting a beam/greedy win, they can sound flatter.

The generator itself now renders through the same low-level XTTS path as the
rest of the chain (latents cached per voice), so `gpt_cond_len`,
`gpt_cond_chunk_len`, `sound_norm_refs` and a 30 s speaker embedding
(`max_ref_len`) actually apply to the final meditation — `tts_to_file` silently
capped the embedding at 10 s, which quietly degraded every final render.

The automated form of the Validator. The sampling knobs (`temp`, `top_k`,
`top_p`, `rep_pen`) don't describe the speaker, so they can't be predicted from
the reference — they must be searched against an observable objective. This runs
a coordinate-descent search maximising `w_accent·french + w_identity·identity`,
~15–25 generations, deterministic per seed.

```bash
python xtts_optimize.py curated.wav FR --xtts-block "{1, 42, ...}" \
    --seeds "0 42 100" --budget 25 --w-accent 0.6 --w-identity 0.4
```

It prints a ranked table and the winning `{}` block to paste into the Comparator.
Use it instead of hand-sweeping in the Validator when you want the optimum rather
than an A/B listen.

---

## Recommended Workflow

One-shot: the **[Auto] Pipeline** tab runs steps 3–6 automatically per voice.
Manual chain:

1. **Extract reference audio** → Video→Audio tab with XTTS preset (WAV, mono, 22050 Hz)
2. **Clean if needed** → Vox tab with demucs + deepfilter
3. **Curate the reference** → [Cur] Curation tab (or `curate_reference.py`) to keep only the most coherent segments — then use the curated file everywhere downstream
4. **Analyse** → Analyser tab, Praat mode, multiple reference files (gives `{}`/`[]` priors)
5. **Find best seed + params** → Validator tab (manual A/B + scores) **or** `xtts_optimize.py` (automated search) → winning `{}` block
6. **Fit the tone** → Comparator tab: paste the frozen `{}`; closed-loop `eq`/`vol`/`hp`/`lp` by least squares
7. **Generate** → Generator tab with the final `{}` and `[]` blocks
8. *(optional)* **RVC timbre conversion** → [RVC] tab, when zero-shot identity
   (~0.7–0.85) is not enough — requires a model trained on 10+ min of the voice

---

## TTS Backend Comparison (tested 2026)

| Model | French | Cloning | GPU | Verdict |
|-------|--------|---------|-----|---------|
| **XTTS v2** | ✅ native | ✅ excellent | 4GB+ | **Best for French** |
| Chatterbox V2 | ✅ correct | ✅ good | CPU ok / 4GB | Competitive (see note) |
| F5-TTS | ❌ English accent | ✅ timbre | 8GB+ | Not usable for French |
| IndexTTS2 | ❌ English accent | ✅ timbre | 8GB+ | Not usable for French |

Measured (same curated reference, same sentence, ECAPA identity): XTTS pipeline
0.710 vs Chatterbox V2 0.725 — same zero-shot ceiling, no migration justified;
the identity gap is RVC territory. `chatterbox_ab.py` regenerates this A/B
(own conda env — its deps clash with TTS 0.22.0). V3 (better similarity per
Resemble) not yet on PyPI; retest when released.

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
This is the seed × reference interaction. Run the **Validator** sweeping `seed` —
it scores each variant for French pronunciation (faster-whisper) and ranks them,
so you pick a clean-accent seed (`--prioritise accent`, the default). If even the
best seed is accented, the reference is the limit: use 2–3 cleaner reference
files, widen the seed list, or lower `temp`. Timbre matching (ECAPA / the
Comparator's EQ) will NOT fix accent.

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

# RVC post-conversion — proof-of-concept guide

**Why**: XTTS zero-shot cloning tops out around ECAPA identity 0.7–0.85 — a
voice "in the direction of" the target, not one a familiar listener recognises.
RVC attacks exactly that gap: a small voice-conversion model TRAINED on the
target person re-voices the XTTS output. XTTS keeps doing the French and the
calm prosody (your whole pipeline stays valid); RVC replaces the timbre.

**Pipeline position**: `... -> generator (XTTS) -> RVC convert -> final audio`

---

## 1. Build the training dataset

RVC wants **10+ minutes** of clean single-speaker utterances (3–10 s each).
Your 45 s curated reference is NOT enough — go back to the raw source:

```bash
conda activate xtts
cd Python_Scripting
python prepare_rvc_dataset.py ../Voices_Cloning/Lea_voice.wav \
    -o ../RVC_datasets/lea --keep-minutes 12 --device cuda
```

This reuses the ECAPA coherence selection (minutes-scale) then splits into
40 kHz mono utterances. Heed the warning if it finds < 5 coherent minutes:
**more source material beats every training trick.** Lea (17 min raw) is the
right first candidate; elo (30 s) is not trainable until you find more audio.

## 2. Install Applio (asteria recommended)

```bash
git clone https://github.com/IAHispano/Applio ~/Applio
cd ~/Applio && ./run-install.sh      # creates its own env — do NOT use the xtts env
./run-applio.sh                      # opens the web UI
```

Applio bundles RVC training + inference with a sane UI. Keep it in its own
environment: its torch/deps differ from TTS 0.22.0.

## 3. Train

In Applio → **Train** tab:

| Setting            | Value                                             |
|--------------------|---------------------------------------------------|
| Model name         | `lea`                                             |
| Dataset path       | `.../RVC_datasets/lea`                            |
| Sample rate        | **40k** (matches the dataset)                     |
| F0 method          | **rmvpe**                                         |
| Epochs             | **250–300** for 10–15 min of data                 |
| Batch size         | **8–16** on the A4500 (20 GB); 2–4 on the 1650    |
| Save every N epochs| 50 (keep checkpoints, pick by ear later)          |
| Train index        | **yes** (feature index = retrieval of target timbre) |

A4500: roughly 1–2 h. GTX 1650: feasible overnight, batch 2–4.
Overfitting on small data shows as raspy/artefacted output — if 300 epochs
sounds worse than 200, use the 200 checkpoint (that's why you save every 50).

## 4. Convert the XTTS output

Applio → **Inference** tab:

| Setting        | Value                                                  |
|----------------|--------------------------------------------------------|
| Voice model    | `lea` + its index                                      |
| Input audio    | `Lea_..._pipeline_clone.wav` (the XTTS output)         |
| Pitch (semitones)| **0** (same-register conversion)                     |
| F0 method      | rmvpe                                                  |
| Index rate     | **0.6–0.75** (higher = more target timbre, more artefacts) |
| Protect        | **0.33** (guards consonants/breaths from conversion)   |

## 5. The A/B that decides everything

Make three files listenable side by side: the **real reference**, the **XTTS
clone**, and the **XTTS+RVC conversion**. Then the two-part test:

1. Measure: `python speaker_identity.py ref.wav clone.wav` vs
   `python speaker_identity.py ref.wav clone_rvc.wav` — the RVC one should
   jump well past the XTTS ceiling (0.85+ typical when training data is good).
2. **Play it to someone who knows the voice** — the only test that matters,
   and the one XTTS was failing. Expectation honesty: RVC is the biggest
   available jump, but fooling intimates 100 % of the time is beyond any
   current open tool; "clearly her voice, slightly smoothed" is a win.

If the PoC convinces your ears → we wire an `rvc_convert` stage into
`xtts_pipeline.py` + a GUI tab, so every meditation renders through it.

---

## When to consider XTTS fine-tuning instead (step 3 of the plan)

Fine-tuning XTTS on the speaker beats RVC on integration (one model, prosody
AND timbre learned together) but costs more: 10–20+ min data, the Coqui
training recipe, A4500-class VRAM, and one model per voice to manage.
Criterion: do it only if one voice (e.g. Lea) becomes THE permanent voice of
your meditations and RVC's result still leaves you wanting. Otherwise
XTTS + RVC is the better effort/quality trade.

# Fidelity search (3.10.0)

3.10.0 puts ChatGPT's fidelity extension (written on top of 3.8.1) on top of
3.9.0, and corrects what its review found wrong: the changes are listed in the
[changelog](CHANGELOG.md). What it adds is real search machinery: new
conditioning controls, a joint search of the decoding settings, and a new
criterion for the audio treatments `[]`. It does **not** promise a better
resemblance on a voice nobody has tried it on. Nothing here has been run
against real XTTS, Whisper or ECAPA models, and nothing has been judged by ear.

## Using it

**GUI**

1. In **[Auto]**, choose the reference and the language as usual, open the
   advanced settings and click **Fidelity setup + auto budgets**.
2. Tick **Never overwrite** to keep the results of the older method, then run
   the pipeline.

The button selects the `{}` objective `both`, the Resemblance priority, the new
conditioning controls, twelve joint decoding candidates, the greedy/beam probes
(width 2), the `de` audio search with the **fidelity** target, an EQ limit of
8 dB, and computes the budgets from the fields and references shown.

It does **not** tick *Include speed, pan and limiter*. The pace of a meditation,
a pan and a limiter are editorial choices that neither ECAPA nor the acoustic
profile can judge, so they keep following that box (off by default). Tick it
yourself if you want them searched.

Elsewhere:

- **[Cmp]**: *Audio search* `nelder` or `de`, then the *Search target* menu:
  `fidelity` (the default), `identity` (ECAPA only, the 3.9 search), `timbre`
  or `both` (the 3.9 search with a timbre term). The menu is only sent to a search
  that runs.
- **[Opt]**: the *Speaker cap / ref trim* and *Joint sampling (12)* boxes switch
  the new native searches on.
- **[Val]**: the three new `{}` fields can be tried one by one.

**Command line** (from the scripts folder)

```sh
python xtts_pipeline.py --voice ../Voices_Cloning/my_voice.wav FR --fidelity \
    --target-dbfs -20 --keep-preset-history
```

`--fidelity` sets `--objective both` and `--audio-objective fidelity`, adds
`--advanced-conditioning`, `--probe-beams` (width 2), twelve joint candidates
and the `de` audio search if none was asked, an EQ limit of 8 dB, the weights
0.3 / 0.7, and sizes the budgets. Every one of those that you pass yourself is
kept (`--budget 150` and `--budget=150` both work, and so do `--objective`,
`--audio-objective` and `--optimise-audio none`; an option written in
abbreviated form, `--obj`, is refused). It leaves *speed, pan and limiter* alone; add `--search-all-fields` to free them. A budget is a ceiling:
convergence or a cache can stop the work before it.

The older searches remain: `--audio-objective identity` is the 3.9 search
(ECAPA, or the timbre term of `--objective`), `--optimise-audio none` is the tone
fit alone. The Comparator's own default stays `identity`, so an old call to
`voice_comparator.py` does what it did; the pipeline and the GUI choose
`fidelity` when an audio search is on.

## 1. The `{}` block: three optional trailing fields

The 17 historical positions are unchanged. The three new fields are appended
only when they differ from their defaults, so an old preset still reads and a
block that does not use them stays short.

| Position | Field | Role |
|---:|---|---|
| 1 | N | Voice number |
| 2 | seed | Reproducible draw; 0 is a real seed |
| 3–4 | trim_start, trim_end | Cut of the render, in ms |
| 5–6 | fade_in, fade_out | Fades of the render, in ms |
| 7 | temp | Decoding temperature |
| 8 | top_k | Token filter; 0 turns it off |
| 9 | top_p | Nucleus sampling |
| 10 | rep_pen | Repetition penalty |
| 11 | len_pen | Length penalty, which matters in beam search |
| 12–13 | gpt_cond_len, gpt_cond_chunk_len | Length and chunking of the GPT conditioning |
| 14 | sound_norm_refs | Reference normalisation |
| 15–16 | num_beams, do_sample | Decoding mode |
| 17 | max_ref_len | The historical reference cap, shared by both paths |
| 18 | speaker_ref_len | A cap of its own for the speaker identity; 0 keeps the shared one |
| 19 | ref_trim_db | Threshold for cutting silence at the edges of each reference; 0 is off |
| 20 | typical_p | Typical sampling; 1 is off, and it only acts with `do_sample=1` |

**Separate conditioning.** Coqui loads each reference with `max_ref_length`, then
uses the result for the GPT conditioning and for the speaker embedding alike, so
shortening the reference to look for a better identity also shortened the
material the prosody comes from. With `speaker_ref_len` the GPT keeps its latent
while a second call of the public API gives the speaker embedding over another
length (its unused GPT part is limited to one second). No private attribute of
the model and no mixing of speakers is involved.

`ref_trim_db` is the low-level `librosa_trim_db`, applied on the CPU (some Coqui
versions move the tensor to the GPU before librosa sees it). Each reference is
capped, then trimmed, in that order; the temporary files are removed.

The optimiser tries the lengths 0, 10, 20 and 45 s, the thresholds 0, 30, 45 and
60 dB, and their combinations (stages C4b, C4c, C4d, with
`--advanced-conditioning`). The existing confirmation of the conditioning still
applies. `test_max_ref_len.py` still isolates the effect of the embedding length
on purpose, by keeping the GPT latent fixed.

**Joint decoding search (stage S5, `--joint-sampling N`).** A Sobol sequence
proposes combinations of temperature, repetition penalty, `top_k`, `top_p` and
`typical_p`, which reaches the repetition-penalty region below 4 that the old
grid never visited. The two best candidates are compared with the previous
setting on **the same three seeds and the same sentences**; the decision is made
on paired differences with identity and intelligibility checked, so a wide
spread between seeds no longer stands in for that comparison. A confirmation
that is incomplete keeps the previous setting. Twelve bounded proposals by
default, not an exhaustive search; the results and any budget cut are in the
optimiser's JSON. Trims and fades stay editing settings: nothing here claims to
optimise them for recognition. `len_pen` is only tried with beam search, where
it acts, and the sampling filters are not presented as active in deterministic
decoding.

## 2. The `[]` treatments

The order is unchanged:

```text
[N, LANG, speed, vol, eq_low, eq_mid, eq_high, hp, lp, NR, comp, de-ess, reverb, noise_gate, pan, limiter]
```

`lp=0` turns the low-pass off; NR, compression, de-esser, reverb and gate at 0
are off; `pan=0` is centred. With `--target-dbfs` the gain is computed after the
effects instead of being chosen by the identity score.

### The criterion

`fidelity` is half the ECAPA cosine and half an acoustic score. The acoustic
score compares, on the voiced frames, the spectral envelope, the pitch, a
periodicity measure, the sibilance, the dynamics and the level of the quiet
passages (weights 0.45, 0.15, 0.15, 0.10, 0.10 and 0.05). It uses NumPy and SciPy
and no new neural model, and ignores a plain change of gain. Its periodicity is
**not** the Praat HNR: the two keep different names and are never mixed. The
weights, scales and limits are written in the JSON report. They are engineering
criteria: neither a listening score nor a probability.

What was measured (`python -I fidelity_vs_timbre.py`, which reads
`Voices_Cloning/`), on the four reference voices of the repository (three 12 s
excerpts of each, other words every time): the score put every pair of excerpts of
one speaker above every pair of excerpts of two speakers (area under the curve
1.00; 0.988 for the timbre score of 3.9), and it moves less between excerpts of
one speaker (standard deviation 0.031 against 0.133). Measured in those
standard deviations, a +4 dB high shelf moves it by 2.2 (the timbre score: 0.6)
and a 6 kHz low-pass by 4.0 (timbre: 0.2): it sees what an EQ or a filter does,
which ECAPA does not. That makes it a steadier thing to search on. It does not
say that the clone it prefers sounds closer.

### What competes, and what is kept

**The tone fit is the baseline.** The closed-loop spectral fit runs first, as it
always did; everything below is measured against it, and it is what you get if
nothing is proven better. A refused search never undoes the tone fit.

Beside the trials of every free field (off, on and the grid values), the
combined winners and the joint refinement, two whole blocks are scored as they
stand, **the input `[]`** and **a neutral block** (EQ, filters, noise reduction,
compression, de-esser, reverb and gate at zero). They are there so that a tone
fit that hurts cannot shelter itself behind a baseline nothing beats. Their
speed, pan and limiter are the tone fit's, like every block of the search. A
block the fit budget cannot pay for is reported as `not tried (fit budget)`.

### Selection on other sentences

Up to three finalists (`--audio-finalists`) are checked on **three distinct other
sentences**: the pruned joint winner, the best one-field alternatives and the two
anchors above, best fit score first, a block never twice. The candidate with the
best score on the fitting sentence can lose to another that transfers better.
The raw and baseline renders of those sentences, and their transcripts, are
cached from one finalist to the next.

A finalist is eligible when, against the baseline and on the three sentences:

- the mean fidelity gain exceeds the minimum (`--audio-min-gain`, 0.005) and one
  paired standard error (descriptive, not a confidence interval);
- no sentence loses more than 0.02;
- identity stays within tolerance (no sentence below −0.015, mean not below
  −0.005);
- the word error rate does not rise by more than 0.02, the ASR score does not fall
  by more than 0.03, the detected language does not change, and no clipping is
  added.

Among the eligible finalists the one with the largest paired mean gain is
delivered; if none is eligible the tone fit is. These margins are explicit
choices, not a calibrated interval.

One more reserved sentence, not one of the earlier searches' texts, is generated
for diagnosis only. It does not rank the finalists; its WAVs, scores and
transcripts are kept and a regression is reported. The sentence comes from a
list of its own (French, English, Spanish, German, Italian, Portuguese, Dutch);
for another language give `--unseen-text`. Your own selection sentences
(`--audio-holdout-file`) come first, and the built-in ones of the language fill
up to three when they exist. This list is used by the fidelity search only: the
3.9 search keeps the sentence it always had.

**An effect may still end at zero.** The report gives its criterion, its trials,
the failures and the decision on each finalist. A non-zero value is kept only for
the benefit measured with these checks.

## 3. The WAVs and their blocks

For a pipeline output `my_voice_pipeline_clone.wav` the fidelity mode keeps:

| File | Content |
|---|---|
| `my_voice_pipeline_clone_raw.wav` | Raw XTTS, without `[]` or the `{}` trims and fades |
| `my_voice_pipeline_clone_base.wav` | The input `[]` |
| `my_voice_pipeline_clone_tone.wav` | The tone fit alone, before the fidelity selection |
| `my_voice_pipeline_clone.wav` | The result, after validation |
| `my_voice_pipeline_clone_audio_search.json` | Criterion, trials, decisions, and the blocks of the WAVs |
| `*_audio_holdout<n>_raw/base/candidate.wav` | Each finalist against the baseline, on the selection sentences |
| `*_fidelity_unseen_raw/base/final.wav` | The same on the last reserved sentence |

In the JSON, `render_blocks` ties each render to its `{}` and `[]`. The raw one
carries `post_processing_applied=false` and no `[]`. The adaptive trimming keeps
its rule that protects the last words. The baseline of the `holdout` and
`unseen` files is the tone fit; their candidate is the finalist. When a target
level is asked for, the validation renders use the gain computed for it.

## 4. Compatibility and what was checked

`--audio-objective identity` reproduces the 3.9 search, `--optimise-audio none`
keeps the tone fit alone, and old presets and blocks read as before.

The tests (see the [README](README.md#tests)) check the new fields all the way
to the engines, the caches, the joint search, the refusals on an ASR or identity
degradation, and the adoption of an acoustic gain when ECAPA stays constant. They
use synthetic signals and real signal processing; XTTS, Whisper and ECAPA are
replaced by stand-ins in the tests that drive the scripts.

**No real XTTS generation on a reference voice was run.** Whether the criterion's
weights are right, and whether the clone it prefers is closer, has to be judged
on your voices: compare several sentences and the saved WAVs, blind.

## Sources the patch cites

- [XTTS source: conditioning and inference](https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/models/xtts.py)
- [XTTS GPT source: passing the Transformers options](https://github.com/coqui-ai/TTS/blob/dev/TTS/tts/layers/xtts/gpt.py)
- [XTTS documentation and inference parameters](https://github.com/coqui-ai/TTS/blob/dev/docs/source/models/xtts.md)
- [Transformers 4.33.3: TypicalLogitsWarper](https://huggingface.co/docs/transformers/v4.33.3/en/internal/generation_utils#transformers.TypicalLogitsWarper)

The patch gave them for the interfaces it calls; they were not re-read for
3.10.0, whose tests replace the model with a stand-in, so a Coqui release that
changes `get_conditioning_latents` or `inference` would only show in a real run.
They say nothing of the perceptual performance of the criterion proposed here.

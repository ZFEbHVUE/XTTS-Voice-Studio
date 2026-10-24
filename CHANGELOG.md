# Changelog

What changed in the scripts of `Python_Scripting/`, why, and what was
measured. The [README](README.md) describes how the tools work now.

> **What these corrections do not do.** None of them raises the identity
> ceiling of zero-shot XTTS. They make what is measured, what is tuned and
> what is rendered the same thing, and make the tools say when a result is
> within the noise. Judge every voice by ear, blind, on sentences it was not
> tuned on.

---

## 3.5 — 2026-10-24 — priority, tone-fit A/B and production level

Asked for: auto buttons for EQ weighting, Optimise audio, Target dBFS, w_acc,
w_id, Objective and gcl s. Only gcl s can be computed from the audio; the
others are choices, so they got a menu, an A/B or a default instead, and
Optimise audio stays as it was (`none` is the only setting that does not tune
the audio block to please ECAPA).

### [Auto] Pipeline

- **Priority menu** for w_acc, w_id and Objective together: Resemblance
  (0.3 / 0.7, `both`), now the tab's default, or Balanced (0.6 / 0.4,
  `identity`, the earlier default). A field edited by hand shows Custom;
  typing a preset's values back finds it again.
- **gcl s** moved to the grid of conditioning lengths, with an auto button:
  the 20 / 30 / 45 / 60 ladder up to what XTTS reads of the shortest file,
  which is the last value (from 10 s for a file shorter than 20 s). No value
  is ignored by the search or reads the same audio twice. The 20 s floor is
  the one the tab already kept, after one Paris-French voice came out
  Canadian at 6 s.
- **Target dBFS −20** by default. Matching each reference gave the voices of
  one meditation different levels and drove a loud reference into the
  limiter. An empty field still matches.
- **EQ weighting `a+blend`**, the new default, renders two tone fits of each
  voice (below).

### Two tone fits of the same clone

- `xtts_pipeline.py --eq-weighting` takes several weightings (`a+blend`,
  `a,blend`): the first gives the preset, each further one another tone fit,
  written next to it (`*_pipeline_clone_blend.wav`) and stored as the preset
  `<name> blend`. The recap gives its `[]` as a comment, so a pasted recap
  still holds one `[]` per voice, and the command for a blind test with
  `listening_test.py`.
- The further fit works on the very same material: the first fit's untouched
  render and the same stretch of the reference, so only the weighting
  differs. The comparator gained `--raw-clone WAV` (fit on an existing
  render) and `--reference-interval BEGIN END` (the stretch to compare
  with), and prints its fit input (`[*] Fit input: {...}`) for a later run
  to reuse. On a reused clone it loads XTTS only if the unseen-text identity
  check or the audio optimisation needs it; the further fit needs neither.
- Why an A/B rather than a rule: in this fit the A-weighting is floored at a
  quarter of its 1 kHz weight, reached at about 180 Hz. A bass at 120 Hz and
  a low female voice at 170 Hz get the same weight on their fundamental, a
  voice at 220 Hz barely more (0.32): pitch does not decide between `a` and
  `blend`. The pipeline's help no longer says that `voice` is better on a
  deep voice.

### Tests

- 9 new tests: the Priority menu, the defaults reaching the pipeline, the
  gcl s auto and its ladder, the comparator refitting a given clone without
  XTTS on the given stretch, its argument checks, the pipeline's second fit
  (same raw clone, same text and stretch, commented recap, second preset
  without the first fit's unseen identity), a failed second fit keeping the
  first, and the weighting lists.
- Each addition was checked by putting a plausible bug back (12 more cases,
  30 in all): every one makes at least one test fail. All 150 tests pass.

### Upgrading to 3.5

Install the changed files together: `xtts_studio.py`, `xtts_pipeline.py`,
`voice_comparator.py` (the pipeline's second fit needs `--raw-clone`),
`search_plan.py`, and the three test files.

---

## 3.4 — 2026-10-24 — safe budgets in the [Auto] tab

Asked for: a button that sets a safe value next to the budget and
conditioning fields of the [Auto] Pipeline tab.

### Auto buttons ([Auto] Pipeline)

- **One source for the schedule.** `search_plan.py` (new) holds the search
  grids, the noise probe, the C2b reserve and the arithmetic of both plans.
  `xtts_optimize.py` takes its constants from there, so the search and its
  plan cannot drift apart, and the tab uses the same functions before any
  model loads.
- **Budget**: the whole sampling plan for the seeds on screen, beam/greedy
  probe included, in the longest case. The analysis decides the starting
  temperature, and off the S2 grid it costs 6 generations more than at 0.55
  or 0.65. 5 seeds: 60; 9 seeds: 68; 9 seeds with the beam/greedy probe: 80.
- **Conditioning budget**: an upper bound over every path the conditioning
  stages can take, whichever candidates win, plus the noise probe and the
  reserve C2b leaves for C3. Checked by running the optimiser under
  simulation with every gain adopted, so that the noise sends each run down
  a different path: no stage cut, C5 complete, never more generations than
  the plan. Defaults: 21 (19 used). With reference lengths `20 30 45`: 45
  (43 used).
- **Max reference s**: the whole number of seconds that reads all the audio
  XTTS gets: the curated file the run will make, an existing curated file it
  will reuse, or the reference itself without curation. The longest voice
  decides, at most 60 s. XTTS cuts every file to this length before the
  speaker embedding and the GPT conditioning alike (TTS 0.22.0,
  `get_conditioning_latents`).
- **Reference lengths to test**: the 20 / 30 / 45 / 60 ladder below the
  shortest of those files, plus the length that reads all of it.
- **Chunk lengths**: the 4 / 6 / 8 / 10 ladder up to the shortest usable
  gcl s, so every chunk pairs with every GPT length.
- **Keep s** auto only lowers the value. It used to set it to 80 % of the
  usable speech of the shortest reference, which raised it past 100 s on a
  long recording: curation then kept almost everything, and XTTS read only
  the first Max reference s of it.
- Before a run, the tab warns when a budget is below its plan. An empty
  field counts as the pipeline default in every computation, as it does in
  the run.

### Re-curate

- **Re-curate** checkbox, passing `--recurate`. Without it the pipeline
  reuses an existing `<ref>_curated.wav`, whatever Keep s says now; the
  voice readout now says so, with the file's length.

### Curation

- **The curated file is exactly Keep s long.** Segments are joined with a
  10 ms crossfade, and the selection stopped as soon as the segments added
  up to Keep s. Windows on a 2 s grid often add up exactly, so an even Keep s
  came out 10 ms per join short: 29.95 s for 30 with 6 segments. The
  analyser starts `gpt_cond_len` at the file's whole seconds, so the search
  started from 29 and tested it against 30, two near-identical settings, and
  a conditioning budget computed for 30 no longer covered the run. The
  selection now counts the crossfades. Files curated before 3.4 keep their
  length; the auto buttons read it.

### Optimiser

- The header prints the conditioning plan next to the sampling plan
  (`Conditioning plan: up to N generations for a budget of B`), with the
  `--cond-budget` that runs every stage in full.

### Documentation

- `max_ref_len` caps each file for the GPT conditioning too, not only for
  the speaker embedding; `--probe-texts` is not in the GUI; the 5 % of the
  adoption rule holds per stage and only if the noise were known exactly
  (see [Known limits](#known-limits)).

### Tests

- `test_budget_plan_regressions.py`: 14 new tests. The conditioning plan
  against the real optimiser run down many paths; the sampling budgets
  60 / 68 / 80 against the optimiser's own header; the assumptions the plans
  take from the analyser and the pipeline; the [Auto] tab's buttons,
  Re-curate, empty fields and the pre-run warnings, with stand-in widgets;
  the curated file's length. The tab was also driven for real, Tk under a
  virtual display.
- Each fix was checked by putting its bug back (18 cases): every one makes
  at least one test fail. All 141 tests pass with Python 3.11.

### Upgrading to 3.4

Install the changed files together: `search_plan.py` (new),
`xtts_optimize.py` and `xtts_studio.py` (both import it),
`curate_reference.py` and `test_budget_plan_regressions.py` (new).

---

## 3.3 — 2026-10-24 — render chain and search statistics

A review of 3.2.1. It fixes the render chain (clipping, limiter order, EQ overs,
limiter delay) and the optimiser's statistics (decisions taken on noise, stages
cut silently by the budget).

### Render chain

- **Volume before the limiter, in one floating-point pass.** The limiter
  used to run inside the effects stage, before the speed change and before
  the volume. pydub applies a gain to the integer samples and saturates
  them, so any positive volume clipped. Measured on a synthetic voice-like
  signal peaking at −1 dBFS: +6 dB clipped 9.1 % of the samples; now none.
  Volume and limiter are now one FFmpeg filter graph (`volume`, then
  `alimiter`, limit 0.99 = −0.09 dBFS). With `vol = 0` the render is the
  same audio as in 3.2.1: bit-identical with the limiter off, identical
  apart from the removed 5 ms delay with it on.
- **EQ boosts can no longer clip.** The EQ stage writes 16-bit audio, so a
  boost could push the peaks past full scale before the volume and limiter
  could act. Measured on the same signal: +4/+4/+4 dB (the 3.2.1 bound)
  clipped 2.0 % of the samples, an earlier fitted EQ of +3.6/+8.3/+9.3 dB
  3.0 %. A later negative volume cannot repair that distortion. A boosted
  EQ is now followed by the limiter; below full scale it changes nothing,
  and cut-only EQ is untouched.
- **No more 5 ms shift.** Without its `latency` option, FFmpeg's `alimiter`
  delays the audio by its 5 ms look-ahead and drops the last 5 ms of every
  sentence (measured with FFmpeg 6.1). The option is used when the
  installed FFmpeg has it (probed once per run), otherwise the old
  behaviour is kept.
- **One render function.** The generator, comparator, validator and
  `test_truncation.py` used to rebuild the chain each on their own, which
  is how they drifted apart (speed missing in one, volume in another,
  limiter misplaced in all). They now share `render_voice_chain()`.
- **The chain says what it does:**

  ```text
  [*]  Limiter: peaks reduced by 5.1 dB              (adds "heavy limiting is audible" above 6 dB)
  [*]  EQ boost reached full scale: peaks limited (they used to clip)
  [!]  Volume +6.0 dB on peaks at -1.0 dBFS (limiter off): they reach +5.0 dBFS and are CLIPPED.
  ```

### Optimiser statistics

#### Noise-aware adoption

Every candidate is scored on one or two sentences with one random draw. Two
settings that sound identical can differ by several hundredths by luck
alone, and 3.2.1 adopted any higher score, even +0.001. Settings walked away
from their starting values on noise.

- **The noise is measured, not assumed.** Conditioning stage: the base
  settings are rendered with 3 more seeds (when `--cond-budget` is at least
  10 generations per conditioning sentence). Sampling stage: the seed screen
  itself (at least 3 seeds). The standard deviation of those scores is the
  draw-to-draw noise of one evaluation.
- **A challenger must clear a bar.** It replaces the incumbent only when its
  gain exceeds `z(1 − α/m) × √2 × sd`, where `m` is the number of challengers
  in the stage (Bonferroni: a stage adopts pure noise with probability about
  α). Floor `sd = 0.005`; `0.010` is assumed when nothing could be measured,
  and the log says so.
- **Seeds versus settings.** Seeds are interchangeable random draws: the
  best one is simply taken. A setting (temperature, rep_pen, top_p, top_k,
  conditioning, decoding) must beat the incumbent beyond the noise.
- **`--adopt-alpha`** (default `0.05`; `1` restores the old "any gain
  wins", for comparison).

Simulated effect (20 full runs, simulated scores in which only the
temperature has a small real effect): 3.2.1 moved the chunk length in 17
runs, the temperature in 12 and `gpt_cond_len` in 9, all on noise. 3.3
moved each of them at most twice in 20.

**Consequence:** with 2 search sentences and a draw noise around 0.03 (the
level of the simulations), a setting needs roughly +0.07 to +0.10 to be
told apart from luck, so the optimiser usually keeps the analyser's values.
That is the honest outcome of a 2-sentence measurement; every run prints
its own measured figure. The bar falls as 1/√n: `--probe-texts 4` divides
it by about 1.4.

#### Budget reporting

- The sampling plan is printed before any model loads, with the budget
  needed to run it in full. With the defaults (5 seeds, 3 kept,
  2 sentences) it can reach exactly 60, the default budget: no margin.
- A stage cut short by its budget says so when it happens, in the summary,
  in the final block and in the result JSON (`truncated_stages`). The
  pipeline repeats it in its recap. Before, `evaluate()` returned in
  silence and the summary read as if the whole schedule had run.
- C5 fills whatever conditioning budget is left by design: reaching the end
  of the budget there is reported as information, not as a cut. C2b, when
  skipped to leave budget to C3/C4, is reported as skipped.
- The result JSON also records `noise` (`conditioning_sd`, `sampling_sd`,
  `adopt_alpha`).

### Tests

- `test_render_search_regressions.py`: 18 new tests. The render tests drive the
  generator's own functions through the real FFmpeg — no clipped sample at
  +6 dB, bursts limited rather than flat-topped, no 5 ms shift, EQ overs
  limited, bit depth kept; the optimiser tests run `xtts_optimize.main()` end to
  end with simulated models — gains within the noise refused, real gains
  adopted, `--adopt-alpha 1` restoring the old rule, cut stages reported live,
  in the summary and in the JSON; the pipeline recap repeats them.
- Four 3.2.1 tests simulated the effects, the speed and the gain as separate
  calls; they now go through the single render function (one of them runs the
  real `render_voice_chain()` and checks its order).
- Each fix was checked by putting its bug back (8 cases): every one makes at
  least one test fail. All 127 tests pass with Python 3.11 and FFmpeg 6.1.

### Upgrading to 3.3

Install the changed files together and keep them next to the other scripts.
The comparator, the validator and `test_truncation.py` render through the 3.3
generator's `render_voice_chain()` and stop with an error against an older one.

**Behaviour changes to expect after upgrading**

- Each sentence is now rendered with the block's seed. A meditation that
  relied on the random state running on from one sentence to the next will
  sound different.
- The limiter now runs last and, when the installed FFmpeg supports it, no
  longer shifts each sentence by 5 ms.
- The optimiser keeps its starting values far more often (see
  [Noise-aware adoption](#noise-aware-adoption)).

`voice_comparator_old.py`, `xtts_studio_old.py` and the generators v20 to
v22 are kept for reference only. The 3.2.1 notes (`LISEZ_MOI_*.md`,
`VALIDATION.txt`, `MANIFEST.json`) and the one-off replay scripts
`recompare_elo.sh` / `recompare_fanny.sh` are superseded by the README and
this changelog, and can be deleted.

---

## 3.2.1 — 2026-10-22 — settings, search and separation

### Shared settings and the generation engine

- `xtts_settings.py`: one lossless format for the GUI, optimiser,
  validator, comparator and generator. `num_beams`, `do_sample` and
  `max_ref_len` survive every export and render.
- Generator: the seed is re-applied for every sentence, as the test tools
  do; long-text splitting is on; conditioning latents are cached per
  reference files and conditioning settings; no silent fallback when
  `xtts_clone.py` is missing.
- `xtts_clone.py`: inputs validated, output sample rate read from the
  model, sampled beam search disabled.

### Optimiser (`xtts_optimize.py`, [Opt])

- Conditioning search before sampling: C0 `max_ref_len` (`--max-ref-lens`),
  C1 `gpt_cond_len`, C2 `gpt_cond_chunk_len`, C2b pairs the sequential
  search cannot reach, C3 `sound_norm_refs`, C4 reference subsets, C5 a
  bounded joint grid. Each combination is checked against its own
  reference cap; requested GPT lengths come back after a short cap wins.
- Sampling: S1 seed screen, S2 temperature response surface, S3
  `rep_pen` / `top_p` / `top_k` combining gains across axes, S4 optional
  beam / greedy probe. The `coord` method no longer mixes one-sentence
  conditioning scores with search scores.
- Hold-out validation on unseen sentences, with its own budget.
- `--result-json` writes the winning block, the selected references in
  order, the texts and the scores; the pipeline reads it instead of parsing
  console text. The winning reference subset follows through to the
  comparator and the presets, and the GUI can restore it.
- `--probe-text-file`, `--holdout-text-file`, `--text`: custom sentences, one
  per line. No English fallback is used to score another language.
- GUI: GPT durations, chunk sizes, conditioning budget, sentence count,
  objective (`identity` / `timbre` / `both`), method (`rsm` / `coord`),
  reference durations and the beam probe are exposed in [Opt]; chunks,
  conditioning budget and reference durations in [Auto].

### Comparator (`voice_comparator.py`, [Cmp])

- `--auto-text` now measures the stretch of the reference that matches the
  transcribed words (word timestamps; zero-duration words grouped with a
  neighbour; complete-segment fallback). It used to compare a 240-character
  clone with the whole reference. An empty transcription is an error, never
  a silent default sentence.
- EQ responses are modelled at the clone's sample rate (24 kHz), not the
  reference's. The 44.1 kHz model is the likely cause of fits such as
  +3.6/+8.3/+9.3 dB.
- `--eq-limit` (default 4 dB) bounds the cumulative gains. Each pass is
  rendered and measured, and rejected if the real spectrum got worse.
- Exports the untouched XTTS output (`--output-raw`) next to the base and
  fitted renders: listen to the raw clone first. If the person is not
  recognisable there, no EQ will make them so.
- `--clone-refs` separates the comparison target from the cloning
  references; `--unseen-text` and `--holdout-text` choose the check
  sentences.
- Refuses to score a clone when the generator's chain cannot be loaded.
- `ltas_match.py`: the three bell gains are no longer re-centred (bells do
  not add up to a constant gain), the residual reported is that of the
  rounded gains actually exported, and the high band is measured from 6 kHz.

### Validator (`voice_validator.py`, [Val])

- New sweep axes: `gpt_cond_chunk_len`, `sound_norm_refs`, `num_beams`,
  `do_sample`, `max_ref_len`, trims, fades, limiter (and every audio
  parameter).
- Invalid combinations are rejected before the models load.
- Blind listening (`--blind`) available from the GUI. The "accent" ranking
  keeps its historical label but is a Whisper intelligibility and
  language-consistency score, not a regional-accent measure.

### Voice separation (`extract_voices.py`, [Vox])

- Solo examples of each person (`--female-ref`, `--male-ref`) take
  precedence over pitch guesses. In the GUI, choosing one switches the method
  to ECAPA; on the command line `f0` refuses them. Thresholds
  `--match-minimum 0.35` and `--match-margin 0.08` are heuristics, not
  probabilities.
- Ambiguous passages go to `_review.wav` instead of the male output; the
  split no longer forces two groups or assumes the higher voice is female;
  F0 sorting follows short stretches of the contour, so a speaker change
  needs no pause.
- `--method sepformer` separates every speech block, mixed ones included,
  in one pass per block. The model works at 8 kHz (about 4 kHz of useful
  band). `--restore-high-band` rebuilds the upper band from the mixture,
  off by default because it can bring the other voice back. For a cloning
  reference, prefer passages where the person speaks alone.
- `--method pyannote` (`--diarization-model`, `HF_TOKEN`) locates turns and
  overlaps; overlapping speech goes to the review file, never to both
  outputs.
- Outputs `_female.wav`, `_male.wav`, `_review.wav` (when needed) and
  `_separation.json`. Existing outputs are never overwritten. Music removal
  no longer forces "vocals only".
- Audio from a video is kept at its native rate (it used to be resampled
  to 16 kHz, which removed everything above 8 kHz).

### Curation, identity and utilities

- `curate_reference.py`: windows never straddle two reference files;
  silence no longer counts as speech; an error replaces the silent "keep
  the single best window"; the output must differ from every reference.
- `speaker_identity.py`: invalid embeddings and cosines are rejected; the
  split-half value is a descriptive anchor, not a ceiling or a percentage
  of fidelity; one ECAPA cache per device and model folder.
- `pron_score.py`: normalised apostrophes, correct WER on an empty expected
  text, no transcription scores 0; described as intelligibility, not accent.
- `vowel_pitch.py`: the MFA cache depends on the audio, the words, the
  command and the model. `apply_pitch_to_clone.py`: annotations apply to
  each occurrence of a word.
- `test_truncation.py` and `test_max_ref_len.py`: per-sentence reports,
  `--xtts-block`, `--keep`; the latter isolates the speaker embedding.
- `prepare_rvc_dataset.py` refuses a non-empty dataset folder and writes a
  manifest; `rvc_convert.py` normalises Applio paths and checks the output.
- `brainwave_studio.py`: segment values validated; the streamed mix is
  scaled from a peak bound and the export aborts rather than clip; empty
  music is rejected (it could hang the render).
- `video2txt.py`: CPU by default, as its help says; the Stop button's
  forced kill works again. `listening_test.py`: input checks; the sign test
  is labelled descriptive for one listener.

### Earlier fixes kept in 3.2.1

The limiter is read from the audio config (it was never applied in the
comparator and validator); speed is applied in the comparator and volume in
the validator; comparator scores reach the presets; `gpt_cond_chunk_len`
defaults to 4 everywhere; the GUI splits path fields tolerantly (spaces in
names, `;` separators).

---

## New command-line options

| Script | Option | Default | Meaning |
| --- | --- | --- | --- |
| `xtts_optimize.py` | `--adopt-alpha A` | 0.05 | chance of adopting pure noise per stage; 1 = any gain wins **(3.3)** |
| | `--max-ref-lens "20 30 45"` | none | search the per-file reference cap |
| | `--result-json FILE` | none | complete winner, references, texts, scores, cuts, noise |
| | `--probe-text-file` / `--holdout-text-file` | built-in | custom sentences, one per line |
| | `--cond-budget`, `--cond-probe-texts` | 24, 1 | conditioning-stage budget and sentences |
| | `--gpt-cond-lens`, `--gpt-chunks` | "6 10 15 20 30", "4 6 8 10" | conditioning candidates |
| `xtts_pipeline.py` | `--adopt-alpha` | optimiser's | passed to the optimiser **(3.3)** |
| | `--eq-weighting a+blend` | `a` | several weightings: further tone fits of the same clone **(3.5)** |
| | `--cond-budget`, `--gpt-chunks`, `--max-ref-len(s)`, `--probe-texts`, `--text`, `--probe-text-file`, `--holdout-text-file` | | passed to the optimiser / comparator |
| `voice_comparator.py` | `--eq-limit DB` | 4 | bound on cumulative EQ gains |
| | `--output-raw FILE` | none | untouched XTTS render |
| | `--raw-clone WAV` | none | fit on an existing untouched render instead of generating one **(3.5)** |
| | `--reference-interval BEGIN END` | whole reference | stretch of the reference to compare with, for `--text` **(3.5)** |
| | `--clone-refs ...` | target | cloning references distinct from the target |
| | `--unseen-text`, `--holdout-text` | built-in | sentences for the final and optimisation checks |
| `extract_voices.py` | `--female-ref`, `--male-ref` | none | solo examples of each person |
| | `--match-minimum`, `--match-margin` | 0.35, 0.08 | ECAPA routing thresholds |
| | `--method pyannote`, `--diarization-model` | 3.1 model | turn-based routing |
| | `--restore-high-band` | off | rebuild SepFormer's upper band (may reintroduce voices) |

`--adopt-alpha` is not in the GUI: GUI runs use the default.

---

## Known limits

- **Zero-shot ceiling.** On the voices tried so far with short references,
  ECAPA identity on new sentences has stayed around 0.6 to 0.7 whatever the
  settings, and the comparator's screening found that post-processing
  changes the tone, not the identity. More clean audio per voice and
  fine-tuning (XTTS, or RVC on top) are the levers left; none of these
  scripts' searches replaces them.
- **Scores are proxies.** ECAPA forgives pitch shifts; the "accent" score
  does not hear a regional accent. Use them to screen, then listen blind
  (`voice_validator.py --blind`, `listening_test.py`).
- **Two sentences are a coarse instrument.** Expect the optimiser to keep
  its starting values unless an effect is large; add search sentences to
  see smaller ones.
- **The adoption bar is optimistic.** It treats the draw-to-draw noise as
  known, but estimates it from 4 renders at the conditioning base, and its
  5 % holds per stage, not per run. In a simulation where no conditioning
  setting has any effect, 50 runs out of 200 still changed at least one of
  them with the default settings, and 60 with reference lengths `20 30 45`
  and a conditioning budget of 48. A Student quantile in place of the normal
  one brought the default case down to 17. Two runs that end on different
  conditioning may differ by chance: judge them by ear.
- **Matching a loud reference.** Level-matching a heavily compressed
  reference (a loudness-normalised web video) drives the limiter hard
  instead of clipping. Watch for "heavy limiting is audible", or set the
  level with `--target-dbfs`.

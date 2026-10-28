# Changelog

What changed in the scripts of `Python_Scripting/`, why, and what was
measured. The [README](README.md) describes how the tools work now.

> **What these corrections do not do.** None of them raises the identity
> ceiling of zero-shot XTTS. They make what is measured, what is tuned and
> what is rendered the same thing, and make the tools say when a result is
> within the noise. Judge every voice by ear, blind, on sentences it was not
> tuned on.

---

## 3.7.5 — 2026-10-28 — a small reverb renders

Prompted by the audio-search report of a real run (Vivienne, every field
searched): 5 of its 240 trials failed, all with `reverb` between 0.002 and
0.02 and the same FFmpeg error, exit status 234.

- `apply_reverb` builds `aecho=0.8:<out>:25|50|80:<d1>|<d2>|<d3>` with the
  decays 0.45, 0.30 and 0.18 times `wet`, rounded to two decimals, and
  `aecho` accepts decays in (0, 1] only ("decay[2]: 0.000000 is out of
  allowed range"). Any `wet` below 0.028 rounded one of them to 0: the effect
  failed, the generator printed "Reverb error" and rendered the voice dry. The
  search counted these trials as failed, as it should; a `[]` written by hand
  with such a value lost its reverb in the meditation.
- The decays now stop at 0.01, an echo 40 dB down. From 0.028 on, the filter
  string is the one rendered before, so existing blocks sound the same.

### Tests

- 2 new tests (302 in all): small reverbs render through the real FFmpeg
  without an error; the filter is unchanged from 0.028 on. Both fail without
  the fix.

---

## 3.7.4 — 2026-10-28 — a review of 3.7.3 (ChatGPT): one adoption bar, budgets checked first

ChatGPT's corrige6 was computed against 3.7.3 and applies to it without a
conflict; its code is taken unchanged, as the first commit of this version.
It changes four files; every other one is 3.7.3's, byte for byte, and the 19
checks of 3.7.3 still fail when their bug is put back.

### From ChatGPT's corrige6

- **One adoption bar.** A searched block needs a mean gain on the two other
  sentences above both 0.02 and twice the measured tolerance, with or without
  `--screen-audio`. In 3.7.3 the diagnostic changed the decision: with it the
  bar was twice the screening's tolerance, down to 0.008; without it, 0.02
  whatever the tolerance. The JSON records the rule.
- **Budgets checked before anything runs.** The pipeline checks at its start
  the budget a search needs with the fields it may change: 68 evaluations at
  most with every field and no target level, 64 with one. With
  `--search-all-fields` and a budget of 64 to 67, 3.7.3 found out at the tone
  fit, after the curation, the analysis and the optimiser, and stopped.
- `worst_case_budget()` probes the EQ at half its limit (beyond the limit when
  rounding to the exported precision would put that on the grid). At an EQ
  limit of exactly 0.5 dB, 3.7.3's probe sat on the grid and the bound came out
  3 evaluations short (65 for 68).
- **Include speed, pan and limiter** in [Auto] and [Cmp] passes
  `--search-all-fields`; ticking it selects `nelder` when the search was off.
  Both tabs check the budget the same way.
- A split-half identity score that is not finite is refused instead of
  falling back on the floor (`max(0.004, nan)` gives 0.004); the search then
  fails and the tone fit is kept. The docstring now says the halves differ by
  their words too: the tolerance is a heuristic, not a confidence interval.

### Added here

- Two tests corrige6 did not have: the [Auto] tab refusing a full search's
  budget of 64 without a target level (and taking it with one), and a block
  fitted with the default limit, then searched with a tiny one, whose EQ
  values are off the grid.
- README for the box, the adoption bar and the budget check. The two French
  delivery notes of corrige6 are folded into this entry, as in 3.7.3.

### Tests

- 300 tests: 3.7.3's 288, corrige6's 10, the 2 above. corrige6's changes were
  checked by putting each bug back, 14 cases: 13 make at least one test fail,
  two of them only with the tests added here; the fourteenth changes nothing
  the code can reach (the EQ probe at 0.5 dB, which the rounding fallback
  covers).

### Upgrading to 3.7.4

Install `audio_parameter_search.py`, `voice_comparator.py`, `xtts_pipeline.py`
and `xtts_studio.py` together, with the test files. A searched block is now
held to a bar at least as high as before, higher when the clip's two halves
disagree: compare searches of 3.7.4 with each other.

---

## 3.7.3 — 2026-10-27 — the search of every `[]` field (ChatGPT), pruned and held

ChatGPT's corrige5 was computed against 3.7.2 and applies to it without a
conflict: it is the first commit of this version, unchanged. The second one
changes what is listed under "Changed here".

### From ChatGPT's corrige5

- `audio_parameter_search.py` (new) maps each `[]` field to a [0, 1]
  coordinate. hp, lp and the gate are off below 0.2 and active above it, the
  limiter is on or off: a simplex can no longer turn `lp 0` into a 6 kHz
  low-pass by clipping it into a range. Every free field is tried on its own,
  both states of every switch included; the one-factor winners are combined,
  then refined with Nelder-Mead, or differential evolution and Nelder-Mead,
  within a strict budget. A failed trial is recorded and never selected; a
  baseline that cannot be scored stops the search.
- Comparator, `--optimise-audio`: two selection sentences instead of one; the
  candidate needs a gain on each, and a mean gain above 40 % of the fitting
  gain and above 0.02 (twice the tolerance after `--screen-audio`). Their
  renders are kept with `*_audio_search.json`. With `--target-dbfs`, each trial is
  rendered, its RMS measured over both channels and `vol` corrected until it
  is within 0.35 dB (holding `vol` did not hold the level once compression,
  NR, reverb or the gate changed it). An effect error printed by the generator
  fails the trial. The screening uses the same trial values. The dynamics fit
  says it keeps the block's value where it printed "-> 0" while keeping it, a
  bug of 3.7.2.
- Pipeline and tabs: method and budget reach every EQ fit; [Cmp] gets the
  search, its budget and a target level.

### Changed here

- **Pruning.** The block is adopted or rejected as a whole, so a field that
  does nothing for the score kept whatever value the optimiser left it at and
  rode along with the fields that do. After the search, each changed field
  goes back to its starting value when the block without its change still
  scores within the tolerance of the best found (the heuristic tolerance of
  the comparator: half the identity spread between the two halves of the
  render, 0.004 at least). The floor is fixed, so the total loss is one
  tolerance, not one per field; the fields whose one-factor trials moved the
  score least are tried first. One evaluation per field is reserved for it
  within the budget, so the minimum budget grows by that many (55 instead of
  41 for a block of all 14 fields; the default search needs 54 at most, under
  the tabs' 64).
- **Speed, pan and limiter are held** unless `--search-all-fields` (comparator
  and pipeline). The speed is the meditation's pace, which the Analyser derives
  from the measured syllable rate and a target pace; corrige5 searched it from
  0.75 to 1.25 against a score that does not measure pace. ECAPA averages the
  two channels, so pan is only a gain to it. The limiter protects against
  clipping. The screening still sweeps all three: it is a diagnostic.
- **Off by default** in the pipeline, [Auto] and [Cmp] (corrige5 turned it
  on). Measured here on 8 s of voice through the generator's chain, one render
  takes 1.25 s (EQ, comp, de-ess, limiter) to 1.6 s (with NR and reverb) on two
  cores, two renders per trial when a target level must be corrected; 400
  trials per EQ fit, two fits per voice with `a+blend`.
- **A failed search keeps the tone fit.** corrige5 exited with status 1, which
  stopped the whole pipeline at that voice; a failure after an adoption could
  leave the half-adopted block. Now the block is the tone fit, exactly as
  without the search, the run succeeds (and saves its preset), and the
  pipeline's recap lists each failed search with its report.
- `EQ error` and `Limiter error`, the generator's own wording when FFmpeg fails
  there and returns the audio unprocessed, now fail a trial too.
- The two French delivery notes of corrige5 (`AMELIORATIONS_XTTS.md`,
  `LISEZ_MOI_OPTIMISATION_AUDIO.md`) are folded into this entry: they
  described its defaults.

### Measured

A simulated ECAPA where only `eq_mid` matters (+0.036 from −4 to −1 dB), every
other field moving the score by render noise only (σ = 3·10⁻⁴), on a
tone-fitted block with speed 0.9; the search and the adoption rule are the
real ones, 20 draws, `vol` held by a target level:

| | corrige5 | 3.7.3 |
|---|---|---|
| Blocks adopted | 20 / 20 | 20 / 20 |
| Fields changed | 13 free fields; speed, pan, reverb, NR every time | `eq_mid` only |
| Final speed (start 0.9) | 0.76 to 1.23 | 0.9 |
| Largest pan | 0.96 | 0 |
| Final `eq_mid` (best −1) | −1.12 to −0.76 | −1.14 to −0.81 |

With a hypothetical penalty for the Rubber Band stretch added, corrige5 set
the speed to 1.0 in the 20 draws.

### Tests

- 15 new tests (288 in all): the inert fields that rode along with a real
  gain, now pruned; the total loss of one tolerance; pruning within the strict
  budget; held fields kept in every render, freed on request; the budget floor
  covering the default search; a failed search, before and after an adoption,
  keeping the tone fit with a successful run and its preset; the pipeline going
  on, with the failures in its recap; the EQ and limiter errors; the defaults;
  a hold-out that regresses rejects even with a good mean; the dynamics fit's
  message. corrige5's tests follow the new defaults and minimum budget.
- The changes of this version were checked by putting the old behaviour back
  (19 cases), and corrige5's additions the same way (12 cases): each makes at
  least one test fail, except two of corrige5's that change nothing the code
  can reach (two hold-outs are always rendered; the optimisers stop on their
  own counts before the budget guard).

### Upgrading to 3.7.3

Install `audio_parameter_search.py`, `voice_comparator.py`, `xtts_pipeline.py`
and `xtts_studio.py` together, with the test files. The audio search is off:
choose `nelder` under **Audio search** in [Auto] or [Cmp], or pass
`--optimise-audio nelder`, and listen to the two validation renders
(`*_audio_holdout1_base.wav` / `_candidate.wav`) before keeping a searched
block.

---

## 3.7.2 — 2026-10-26 — the Comparator compares smoothed spectra; a review of 3.7.1 merged

Prompted by a pipeline run on a 16 kHz reference. The tone fit moved the shape
error from 5.71 to 5.70 dB and raised `eq_low` by 1.9 dB, while its own final
table showed the clone with 67.7 % of its energy below 300 Hz against 52.7 %
for the reference, and 30.2 % against 45.4 % from 300 Hz to 3 kHz.

### Comparator: spectra compared over bands

- Below about 1 kHz the long-term spectrum of a voice is a comb: peaks at the
  harmonics of F0, hollows between them, almost nothing below F0. Where the
  teeth fall depends on the melody of the sentence, which a clone never
  repeats. Compared point by point on 256 log-spaced frequencies, more than
  half of them below 1 kHz, the teeth made the error: on a synthetic 204 Hz
  voice, 4.7 dB whether the clone had the reference's fundamental or 4 dB too
  much of it. No EQ can follow the teeth, and the least squares in dB, pulled
  by the hollows, boosted `eq_low`.
- `ltas_match.grid_db` reads the mean power of a band around each frequency,
  max(250 Hz, a third of an octave) wide: 250 Hz spans one harmonic spacing up
  to that F0, and the third octave takes over above 1.1 kHz, where the
  harmonics merge. `compute_ltas` uses it (`floor_hz=0` reads point by point).
  On the voice above: 0.27 dB for the same voice, 1.31 dB with the stronger
  fundamental.
- `ltas_match.fit_eq_psd` fits the EQ on the render's own power spectrum: the
  gains it was rendered with are taken out bin by bin, the candidate gains and
  any filter change put in, and the result is smoothed like the measurement.
  Every linear stage is exact this way; through the generator's own chain the
  predicted error held to 0.02 dB.
- The gains are fitted as they are exported, within `--eq-limit`. 3.7.1
  fitted a step per band and clipped the sum, so a band at its bound left the
  others fitted for a move it could not make (on a test case, 0.32 dB of error
  where the bounded fit leaves 0.22).

Synthetic harmonic voices, 8 s, a 16 kHz reference, through the generator's own
chain from `hp` 95 and `lp` 9000; share of the energy below 300 Hz:

| Clone | Reference | Start | 3.7.1 | 3.7.2 |
|---|---|---|---|---|
| Same voice, other intonation | 27.2 % | 24.9 % | 42.1 % (EQ +4.0/−2.5/−3.1) | 27.2 % (+1.4/−0.1/−0.1) |
| Fundamental 4 dB too strong | 27.2 % | 45.4 % | 53.4 % (+0.9/−1.6/−2.3) | 31.9 % (−4.0/+0.3/+0.1) |

`eq_low` is a bell at 120 Hz: at its −4 dB bound it reaches a 204 Hz
fundamental only in part.

### From a review of 3.7.1 (ChatGPT)

Taken:

- [Cmp] tab: 3 passes by default (1 before; the script and the pipeline run
  3, and a filter change needs pass 2), and the threshold named for what it is,
  "Stop if EQ/volume change <" (it said "score delta").
- A C6 confirmation cut by the budget before one complete comparison keeps the
  base, and the log names the screening winner it does not adopt. 3.7 kept
  that winner, unconfirmed; a winner of single draws is too often a lucky one
  (in 3.6.1, 22 % of the runs on pure noise changed a setting).
- French numbers in the WER: `pron_score` matches a number Whisper writes in
  digits with the French cardinal of the text, both ways (« 48 » and
  « quarante-huit », « vingt et un », « quatre-vingts », « deux cent un »,
  hyphenated or not). A different number, a missing one or an added word still
  counts; dates, decimals, signed numbers, ordinals and leading zeros are left
  alone. Results keep the raw WER (`wer_raw`), and the hold-out report shows
  both when they differ. On the real log, the sentence « Quarante-huit
  personnes … numéro trois » goes from 4 errors in 9 words to 2: the
  « c'est toi » Whisper heard at the end.
- The result JSON records `score_definition` (objective, weights, WER
  normalisation).
- The Comparator no longer calls the identity of one unseen sentence "what
  your generated meditation will resemble".

Redone:

- Language. The review stopped any run whose command and `[]` disagreed, but
  `parse_audio_block` fills in FR for a block that names no language, and the
  Validator falls back to FR when the command names none: an English command
  with a `[]` without a language, or a Validator run without one on a
  `[1, EN, …]` block, were refused for a mismatch nobody asked for. Now only
  two named languages can disagree; otherwise the one named wins, FR when
  neither is. A `[]` without a language no longer imposes FR on an English
  command, which 3.7.1 rendered and scored in French.
- Timbre. The review stopped the whole search when one take's timbre could not
  be measured, which is what a failed render (no usable voiced audio) gives: in
  a simulation, the search stopped at its third render. Such a take now counts
  with the worst timbre, 0, in the same formula as the others (3.7.1 used its
  identity alone, on another scale), and the log says so. When most takes fail,
  from the third on, it is the reference or the tools, and the search stops
  with that message.

### Tests

- 29 tests: the bands and the PSD fit (a known bell found exactly, a band kept
  at its bound and the others fitted knowing it); harmonic voices (point by
  point the comb hides a real difference, another intonation is no tone to
  correct, a too strong fundamental is cut with its prediction within
  0.02 dB); the review's 13 number tests and 2 language tests, and 3 more on
  the language (only two named languages can disagree, the Comparator keeps an
  asked EN, the Validator takes the block's EN); a failed take scored with the
  worst timbre, a systematic failure that stops; the result's score
  definition; the tab and the message. 259 tests in all.
- Each change was checked by putting its bug back (17 cases): every one makes
  at least one test fail. The checks of 3.7.1 still hold, 19 of 21; the other
  two were on code 3.7.2 replaces.

### Upgrading to 3.7.2

Install `ltas_match.py`, `voice_comparator.py`, `voice_validator.py`,
`pron_score.py`, `xtts_optimize.py` and `xtts_studio.py` together, with the test
files. Tone fits and their shape errors change: compare the errors of 3.7.2
runs with each other, not with older ones. French scores of sentences with
numbers change too: run a search again to rank its candidates with the new
WER.

---

## 3.7.1 — 2026-10-26 — the low-pass is a treble tilt; the Comparator can switch it off

Prompted by a Comparator `[]` that no longer moved. In four fits of one voice,
the first pass was kept, the second rejected by 0.03 to 0.05 dB, and `eq_high`
ended at its +4 dB bound, the clone still short of treble (5–8 kHz sibilance
−17.1 dB against −14.4 dB for the reference).

The cause is the low-pass. The generator applies pydub's first-order RC
filter, and at the 24 kHz of an XTTS render it is a tilt of the whole upper
band, not a cut above `lp` (dB, within 0.01 dB of pydub measured through a
render):

| `lp` | 4 kHz | 6 kHz | 8 kHz | 11.9 kHz |
|---|---|---|---|---|
| 7000 | −2.7 | −4.3 | −5.5 | −6.4 |
| 9000 | −2.1 | −3.4 | −4.5 | −5.3 |
| 12000 | −1.5 | −2.6 | −3.5 | −4.3 |
| 16000 | −1.1 | −2.0 | −2.8 | −3.4 |

The Analyser wrote 7000 to 9000 Hz by voice type, as a "hiss guard". The
Comparator, finding the clone short of highs, opened it by 1000 Hz (lp 10000
still takes 4.1 dB at 8 kHz) and fitted the EQ as if the filter had not moved,
so `eq_high` spent its range giving back what the filter took. Nothing in the
fit modelled the filters: `ltas_match` had analogue one-pole formulas, unused,
which put lp 9000 at −2.5 dB at 8 kHz.

### Comparator

- `ltas_match.rc_lowpass_db` and `rc_highpass_db` are pydub's exact
  recursions at the render rate; `filter_change_db` gives what a change of
  `hp`/`lp` does to a spectrum.
- Pass 1 weighs up to three filter settings: what `derive_hp_lp` proposes,
  the low-pass off (whenever one is on), and the block's own. For each, the
  EQ is fitted on the curve those filters will give, and the shape error it
  leaves is predicted. The low-pass off is rendered first when its prediction
  is better, the block's own filters last, and the first render that improves
  the measured error is kept. A filter change the render does not show is not
  kept, like any other change.
- `derive_hp_lp` switches the low-pass off when the clone lacks highs, instead
  of opening it by 1000 Hz; a clone with too much treble still gets a lower
  one. Its rule alone (more than 3 dB missing above 6 kHz) misses a block
  pasted back with `eq_high` +4, where the bell hides part of the filter's cut
  (−2.9 dB there). The prediction does not miss it.
- A filter change in pass 1 earns pass 2, even when the EQ hardly moved.
- Each pass prints its predicted error and the filter change:
  `pass 1: rendered shape error 1.38 -> 0.68 dB (predicted 0.68); EQ=(+1.1/-0.4/-0.2), volume=+0.0 dB; low-pass 9000 Hz -> off; kept`
- `--optimise-audio` holds a filter that is off (clipped into 6000–16000 Hz,
  lp 0 started the search at a 6 kHz low-pass), and `vol` under
  `--target-dbfs` (the search traded the target level for ECAPA cosine). It
  prints what it holds.
- `--screen-audio` tries the low-pass off. `--fit-identity` tries
  "low-pass off", when one is on, instead of lp 16000 (−2.8 dB at 8 kHz).
- `--fit-identity` with `--no-fit-dynamics` stopped on a NameError after the
  identity search.

### Analyser

- `lp` is 0 (off) for every voice type, instead of 7000 to 9000 Hz. `hp` is
  unchanged.

### Generator

- Values unchanged, so existing presets and scripts sound the same. The
  comments said "cut everything above 8 kHz"; they now say what the filters do.

### Measured

Voice-like noise (a speech spectrum, syllabic modulation, 6 s, `hp` 90)
through the generator's own chain (pydub filters, FFmpeg EQ, limiter); the
reference is another draw of the same spectrum at 44.1 kHz. Weighted shape
error, dB, 80 Hz to 8 kHz; 0.68 is about what the two draws differ by.

| Clone, filters | Start | 3.7 | 3.7.1 |
|---|---|---|---|
| Same timbre, lp 9000 | 1.38 | 0.96 (lp 10000, `eq_high` +1.9) | 0.68 (lp off, −0.2) |
| 3 dB short of treble (2.5 to 7 kHz), lp 9000 | 2.29 | 1.40 (lp 10000, +3.8) | 0.89 (lp off, +1.6) |
| Pasted back: `eq_high` +4, lp 9000 | 1.13 | 1.00 (lp 9000, +2.0) | 0.68 (lp off, 0.0) |
| 4 dB too bright (3 to 7 kHz), lp 9000 | 0.80 | 0.78 (lp 9000) | 0.78 (lp 9000) |
| 4 dB too bright, lp off | 1.51 | 0.78 (lp 10000) | 0.78 (lp 10000) |
| Same timbre, lp off | 0.76 | 0.68 | 0.68 |

The predicted errors matched the rendered ones to 0.01 dB. The check covers
80 Hz to 8 kHz, like the whole fit: what switching lp 9000 off adds above
8 kHz (5 dB at 10 to 12 kHz) is not measured, so listen to the top end.

### Tests

- The filter models against pydub through a real render (5 settings, within
  0.05 dB), `filter_change_db`, and `derive_hp_lp` switching off rather than
  opening.
- The tone fit through an exact model of the chain: a low-pass that only dulls
  the clone is switched off with a flat `eq_high`; a pasted-back block lets its
  boost go, although `derive_hp_lp` alone keeps the filter; a render worse than
  predicted falls back to the block's filters; a change the render does not
  show is not kept; a bright clone keeps its low-pass and the low-pass off is
  not even rendered; a filter change earns pass 2. Then once through the
  generator's own chain.
- `identity_axes`, and `main()` with a stub encoder: the optimiser holds an
  off low-pass and a target level, the screening and the identity search try
  the low-pass off. The Analyser on four synthetic voices (skipped without
  librosa). 230 tests in all.
- Each change was checked by putting its bug back (21 cases): every one makes
  at least one test fail.

### Upgrading to 3.7.1

Install `ltas_match.py` and `voice_comparator.py` together (the Comparator
uses the new models), with `voice_analyser.py`, the generator and the test
files. A preset keeps its `lp`; the Comparator proposes switching it off on its
next run, if the fit is better without it. A short `[]` that stops before `lp`
still takes the voice's previous `[]` or, without one, the generator's default,
8000: write `0` there to switch it off.

---

## 3.7 — 2026-10-26 — the conditioning search decides on fresh seeds

Prompted by two real pipeline logs of one voice: a 45 s reference with three
GPT lengths and three reference caps, then a 52 s one with four of each. The
conditioning search judged every candidate on one render, with a draw-to-draw
spread of 0.026 to 0.033 per render: a lone challenger needed +0.061 to +0.076
to be adopted, and the joint grid of C5 grew to 87 candidates (bar 0.146), so
nothing was ever adopted. On the second run, three of the four best draws of
C5 (0.751, 0.730, 0.729 against 0.635 for the base) came from one GPT pair,
gcl 20 s with chunk 8, under three reference caps: the same GPT conditioning
three times, so very likely the same tokens — one draw of evidence, which no
stage could check.

### C6 — confirmation on fresh seeds

- C0 to C5 still screen one render per candidate, and what they adopt still
  steers the next stages, but no longer decides the result. C6 renders the
  stages' winner and the best other screened candidates (3 in all,
  `CONFIRM_FINALISTS`) on the noise probe's 4 seeds and compares each with the
  base on those same seeds, mean against mean. The base's renders there are the
  probe's own, so they cost nothing. The best mean replaces the base only if it
  beats it by more than `t(1 − α/F, df) · sd · √(2/4)`: one-sided Student t
  for the pooled spread of every render of the comparison, Bonferroni over the
  `F` candidates, `α` = `--adopt-alpha`. Otherwise the base stays, and a
  screening winner that fresh seeds do not confirm is named.
- Fresh seeds matter: the best of twenty single draws is mostly the luckiest
  (winner's curse), and the screening draws that chose the candidates are not
  part of their means. A candidate whose renders are identical to an earlier
  one's (`= same audio as`) is that one again and is not confirmed twice.
- A budget too small for one complete comparison leaves the screening's
  winner, unconfirmed, and says so; `--adopt-alpha 1` (the old "any gain
  wins") skips C6.
- The t quantile comes from the Cornish-Fisher expansion (no scipy): within
  0.2 % of the exact value from 6 degrees of freedom (C6 has 6 to 12 on 4
  seeds), within 1 % from 4.

### C5 — capped, GPT changes first

- C5 renders at most 8 new combinations (`C5_SCREEN`), nearest first, and
  never the renders C6 needs: below the plan, C5 is cut first. Combinations an
  earlier stage rendered cost nothing and count as screened.
- At the same distance, a change of the GPT settings goes before a change of
  the reference cap. With one reference and no normalisation, a cap at or above
  `gpt_cond_len` leaves the GPT conditioning (its first `gpt_cond_len` seconds,
  which decide the tokens) as it is: XTTS uses the cap for the speaker
  embedding only. With 8 renders, the grid of the second run reaches the gcl
  20 s / chunk 8 pair this way, and not in plain distance order.

### Budgets

- The noise probe renders the base with 4 extra seeds (3 before), the seeds C6
  reuses.
- Plans: 34 for the pipeline's default fields (21 before), 46 for the fields of
  the first run (53), 48 for the second (93). `--cond-budget` defaults to 34 in
  `xtts_optimize.py`, `xtts_pipeline.py` and both GUI tabs (24 before).

### Measured on simulations

Scores simulated with a known true effect per setting and a random spread per
render; the search itself runs unchanged.

| Configuration, spread | Real gain | 3.6.1 found | 3.7 found |
|---|---|---|---|
| Second run (52 s, 4 GPT lengths, 4 caps), 0.03 | +0.04 on chunk 8 | 33 % | 47 % |
| same | +0.06 on chunk 8 | 52 % | 82 % |
| same | +0.06 on gcl 20 / chunk 8 only | 20 % | 51 % |
| same | none: another GPT setting chosen | 33 % | 2.5 % |
| Pipeline defaults, 0.018 | none: any conditioning change | 45 / 200 | 9 / 200 |

Renders per run on the second run's configuration: 89 in 3.6.1, 41 in 3.7. A
gain of +0.02 for a spread of 0.03 is found in 4 to 11 % of the runs: that
small, it is out of reach of 4 seeds, and keeping the base is the honest
result.

### Result and recap

- Result JSON: `conditioning_confirmation` (seeds, base and its mean, screening
  winner, each candidate's screening draw and mean, bar, sd, degrees of
  freedom, verdict). The conditioning winner block prints the confirmed mean.
- The pipeline recap prints C6's verdict, and the screening winner it did not
  confirm.

### Tests

- `test_confirmation_regressions.py`, 23 tests: the decision (bar, Bonferroni,
  pooled sd, floor, `α` = 1) and the t quantile against scipy; then runs where
  each setting has a known true effect — a real gain one draw cannot show is
  adopted, a lucky draw of the same size is not, a single-draw adoption that
  fresh seeds do not confirm goes back to the base, a real one is confirmed,
  the probe renders are reused, identical audio is confirmed once, C5 leaves
  C6 its renders, a cut confirmation says so and reverts nothing, and the
  structure of the second run reaches gcl 20 / chunk 8.
- The plan tests run the noise-aware rule too (it is the one that fills C6),
  and the joint-search tests keep their purpose within C5's cap; one more test
  shows the cap's price, a pure three-way interaction left unscreened. 213
  tests in all.
- Each change was checked by putting its bug back (28 cases): every one makes
  at least one test fail.

### Upgrading to 3.7

Install `xtts_optimize.py`, `search_plan.py`, `xtts_pipeline.py` and
`xtts_studio.py` together (the plan arithmetic is shared), with the test
files. A command with an older `--cond-budget` keeps working: 93 is more than
the new plan and is not all spent; 24 or 21 is less, the header says so, and
C5 is cut first, then C6. In the [Auto] tab, **auto** (or **auto all**) sets
the plan of the fields on screen.

---

## 3.6.1 — 2026-10-25 — the Validator renders what the script renders; `{}` per voice

Prompted by a listening test: the same sentence ("Salut Stéphane. c'est
élodie") made a liaison with seed 42 and not with seed 80. Changing the seed of
one sentence is the fix, and it needs the Validator to reproduce the script.

### Validator

- The text goes through the Generator's own `clean_text()` before XTTS: the
  final punctuation and the pitch marks are removed, as in the script. The raw
  text was rendered, so the same sentence, `{}` and seed gave a different take
  in the Validator and in the script, and a seed chosen in the Validator did not
  reproduce. The console prints the text as the generator renders it when it
  differs, and the accent score compares Whisper with that text.
- With a generator older than this Validator (no `clean_text`), the text is
  rendered as typed and the console says so.

### Generator: `{}` blocks

- A short `{}` block takes the fields it leaves out from the voice it names. It
  took them from the voice in use: with two voices, `{2, 17}` met while voice 1
  spoke gave voice 2 the settings of voice 1 with seed 17. One voice was not
  affected.
- Inside `[parallel]`, the `{}` written before each `[N, ...]` — the
  documented layout — were discarded: the `[]` reloaded the settings from
  outside the block. They now apply to their voice's text, stay local to the
  block as before, and a short one inherits from the last `{}` of its voice in
  the block.

### Tests

- 7 new tests (189 in all): the Validator's text with the real `clean_text`,
  pitch marks, an older generator; short blocks per voice and from the globals;
  the script loop's use of the helper; a parallel block in the documented
  layout, with a short block and nothing leaking out.
- Each change was checked by putting its bug back (8 cases): every one makes at
  least one test fail.

### Upgrading to 3.6.1

Install `voice_validator.py` and `guided_meditation_generator_v23.py` together
(the Validator reads `clean_text` from the generator), with the new test file.

---

## 3.6 — 2026-10-24 — timbre on voiced frames, readable hold-out, EQ headroom

Prompted by a real pipeline log (one voice, Resemblance priority, 9 seeds).
Nothing was adopted, and decomposed over the 9 seeds of S1 the timbre term
carried 86 % of the score's draw-to-draw noise (sd 0.058, against 0.015 for
ECAPA and 0.021 for the accent). The mean accent on the two unseen sentences
fell from 0.946 to 0.495 while the report said "no large score drop", and the
renders had been deleted. The loudest peaks of the fitted clone were clamped,
then lowered to −2.1 dBFS.

### Timbre term of the objective (`both`, `timbre`)

- Pitch, brightness and HNR are measured on the **voiced frames** only, each
  weighted by pyin's voicing probability. They were averaged over the whole
  file: a frame of silence weighed as much as a vowel in the centroid, and the
  "HNR" was the harmonic/percussive energy ratio of everything, consonants and
  pauses included (−1.3 dB on a reference that Praat puts at 15.7 dB).
- Pitch: level-weighted mean in semitones, tracking errors more than 7 st from
  the centre left out (the plain median varied by as much as 0.66 st, sd, on
  the same audio). Brightness: centroid of the voiced magnitude spectrum. HNR: Praat's
  (frames above 0 dB, median), as in the Analyser and the Comparator; without
  parselmouth, the harmonic/percussive ratio of the voiced frames, on another
  scale. The method is recorded, the clones are measured with the reference's,
  and two files measured differently are not compared.
- Measured on four real voices (`Voices_Cloning/`), with only the pauses and
  breaths around the same 4 s of speech changed: the distance moved by
  0.52–0.76 (sd) the old way, 0.06–0.10 now; a 4 dB EQ change on the highs
  still moves the centroid by about 70 Hz. In the optimiser's own terms, on a
  simulation where the seeds only move pauses and breaths around real speech:
  draw-to-draw noise 0.015 → 0.007, adoption bar 0.034 → 0.016, and a 4 dB
  brightness change went from 0.3 to 1.2 times the bar. Real XTTS renders also
  vary in prosody and voice quality, so the gain on a real run will be smaller;
  the noise line of the next run says by how much.
- Not fixed: the reference says other words than the search sentences, so each
  sentence keeps its own offset from it. It is the same for every candidate of
  a search; the Comparator, which fits on the matching stretch of the
  reference, stays the place where brightness is matched.
- Scores of `both` and `timbre` from 3.5 and 3.6 are not comparable.

### Hold-out report

- Each unseen sentence is reported: accent, identity, score, WER, the language
  Whisper heard, the text and the transcript.
- A sentence is flagged when Whisper heard another language or nothing, or
  when its accent is more than 0.2 below the search's (spelled-out numbers
  written as digits cost about 0.1). The closing line then says that the
  score drop alone does not show it, instead of "no large score drop", and
  tells how to tell a draw from the voice: the same sentence with two other
  seeds in the Validator.
- The "one sentence is carrying the mean" check compared the median with the
  mean, which are equal for two sentences: it now runs from three.
- `--holdout-wav-prefix` keeps the unseen renders (raw XTTS, as scored). The
  pipeline names them after the clone (`*_pipeline_clone_holdout1.wav`), lists
  them in the recap with the flagged sentences, and the result JSON carries
  `holdout_detail` and `holdout_alerts`.

### Identical renders

- A setting can leave the render bit-identical: in the log, rep_pen 4, 7 and
  10 and top_k 70 gave exactly the base render. Not a bug — the value reaches
  XTTS — but once repeated tokens fall outside the top_p nucleus, a stronger
  penalty changes nothing, and top_k changes nothing above the size of that
  nucleus. Such a render is now recognised: its line says
  `= same audio as [S  6]` and it is not scored again. It still counts in the
  budget; skipping it before generating would not be safe, since a stronger
  penalty can shift the nucleus by one token.

### EQ headroom in the render chain

- The EQ stage writes 16-bit audio, so a boost reaching full scale was limited
  there — before the volume step. On the real fit (eq_mid +3.3, eq_high +4,
  volume −2), the loudest peaks were clamped at −0.1 dBFS and came out at
  −2.1 dBFS. The EQ now lowers the audio by the sum of its boosts, in the same
  filter graph, and the volume step gives it back, in floating point, before
  the final limiter.
- Only when nothing after the EQ depends on the absolute level and the limiter
  is on: noise reduction, compression and the gate work on thresholds in dBFS,
  so with any of them on the chain is unchanged. The Comparator renders through
  the same chain, so its fit measures the corrected audio.

### [Auto] Pipeline

- **auto all** runs every auto in the order they depend on each other: Keep s,
  Max reference s, Reference lengths to test, gcl s, Chunk lengths,
  Conditioning budget, Budget. An empty Reference lengths stays empty. Pressed
  one by one in another order, the conditioning budget was computed before the
  fields it counts: 21 instead of 53 in the log.
- **Search sentences file** and **Unseen sentences file** pass
  `--probe-text-file` and `--holdout-text-file`; the first two lines of each
  are used. With a search file and no `--text`, the optimiser now searches on
  the file's sentences: the built-in default sentence used to come first, so a
  two-sentence search used a single line of the file.
- Before a run, a seed like `160180` among `0 42 … 200` is flagged as two seeds
  typed without a space. It still runs, as one seed.
- **Stop** is reported as `[STOP] Stopped.`: it kills the child (code −9),
  which was reported as `[ERR] Error (code -9)`. Same in the [Ana] tab, which
  also printed its stop message in French and "Analysis complete" after a stop.

### Analyser

- The meditation-pace line said that speed 0.8 gives 3.5 syllables/s for a
  7.5/s voice. The speed is clamped at 0.8, which gives 6.0/s: the line now
  says what the speed gives, what the target would need, and why it stops at
  0.8.

### Tests

- 32 new tests (182 in all): the voiced-frame timbre on synthetic voices
  (pauses, an unvoiced burst, brightness, octave errors, breathiness, silence,
  forced HNR method), the clones measured like the reference, the hold-out
  report and its kept renders end to end, the pipeline's prefix and recap,
  identical renders, the EQ headroom decision and its effect through the real
  FFmpeg chain, joined seeds, the pace message, custom search sentences, and
  the GUI's real `run_cmd` stopped and failing.
- Each change was checked by putting its bug back (20 cases): every one makes
  at least one test fail. The timbre tests need librosa, and parselmouth for
  the Praat case; they are skipped without.

### Upgrading to 3.6

Install the changed files together: `xtts_optimize.py`, `xtts_pipeline.py`
(it passes `--holdout-wav-prefix`, which an older optimiser rejects),
`guided_meditation_generator_v23.py`, `voice_analyser.py`, `xtts_studio.py`,
`search_plan.py`, and the four new test files.

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

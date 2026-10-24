#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Search-plan arithmetic shared by xtts_optimize.py and the [Auto] Pipeline tab.

How many generations the optimiser's schedule can ask for, and which settings
keep it coherent with the audio XTTS will actually read. No model and no audio
here, only numbers: the GUI computes a safe budget before anything loads, and
the optimiser prints the same figures in its header.

The schedule constants live here so that the search (xtts_optimize.py) and its
plan cannot drift apart.
"""
import math
import os
import re

# ── Sampling schedule ─────────────────────────────────────────────────────────
# Coordinate-descent grid (--method coord).
GRID = {
    "temp":    [0.50, 0.55, 0.60, 0.65, 0.70],
    "rep_pen": [4.0, 5.0, 7.0, 10.0],
    "top_p":   [0.80, 0.85, 0.90],
    "top_k":   [30, 50, 70],
}
AXIS_ORDER = ["temp", "rep_pen", "top_p", "top_k"]

# RSM schedule (--method rsm, the one the pipeline uses).
S2_TEMPS = [0.45, 0.55, 0.65, 0.75, 0.85]
S3_AXES = (("rep_pen", [4., 5., 7., 10.]),
           ("top_p", [.80, .85, .90]),
           ("top_k", [30, 50, 70]))
S4_LENGTH_PENALTIES = [0.8, 0.9, 1.0, 1.1, 1.2]

# ── Conditioning schedule ─────────────────────────────────────────────────────
NOISE_PROBES = 3         # extra seeds rendered at the base conditioning
NOISE_PROBE_MARGIN = 6   # the probe runs only if the budget leaves this many
                         # evaluations for the stages themselves
C2B_RESERVE = 2          # generations C2b leaves untouched, so C3 can still run

# ── What the pipeline hands the optimiser ─────────────────────────────────────
# voice_analyser.py always writes these sampling values into the {} block;
# only the temperature depends on the voice (0.55 to 0.72), so it is the one
# unknown before the analysis has run. test_budget_plan_regressions checks
# that the analyser still writes them.
ANALYSER_SAMPLING = {"rep_pen": 5.0, "top_p": 0.85, "top_k": 50}
ANALYSER_CHUNK = 4
# xtts_pipeline.py defaults that the [Auto] tab never changes.
PIPELINE_PROBE_TEXTS = 2     # --probe-texts
PIPELINE_KEEP_SEEDS = 3      # optimiser --keep-seeds, not passed by the pipeline
PIPELINE_COND_TEXTS = 1      # optimiser --cond-probe-texts, not passed either

# ── Ladders the auto buttons choose from ──────────────────────────────────────
CHUNK_LADDER = (4, 6, 8, 10)          # the pipeline's default chunk lengths
REF_LENGTH_LADDER = (20, 30, 45, 60)
GCL_LADDER = (10, 15, 20, 30, 45, 60)
# The [Auto] tab's gcl s starts at 20 s: allowed down to 6 s, the search once
# picked 6 s on a Paris-French voice and the clone came out Canadian. One voice,
# not a general threshold, but no score in the loop hears a regional accent.
GCL_FLOOR = 20
LONGEST_REFERENCE = 60   # the analyser never proposes a longer conditioning


def _unique(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def find_ints(text):
    """Integers in a field, the way the optimiser reads --gpt-cond-lens and
    --gpt-chunks: anything that is not a digit separates them."""
    return [int(x) for x in re.findall(r"\d+", text or "")]


def parse_ints(text):
    """Integers separated by spaces or commas, the way the optimiser reads
    --seeds and --max-ref-lens. Anything else is an error, as it is there."""
    return [int(x) for x in (text or "").replace(",", " ").split()]


# ── Sampling ──────────────────────────────────────────────────────────────────

def sampling_plan(n_seeds, keep_seeds, n_texts, start, method="rsm",
                  rounds=2, probe_beams=False):
    """Upper bound on the sampling generations the schedule can ask for.

    Values equal to the incumbent are cached and cost nothing, so they are
    not counted; the parabola vertex (one per kept seed) is counted as if it
    were always tried. A start value that is not a number (NaN) matches no
    grid value: that is the longest case.
    """
    def differs(values, current):
        return sum(1 for v in values if round(float(v), 4) != round(float(current), 4))
    if method == "coord":
        per_round = sum(len(GRID[axis]) for axis in AXIS_ORDER)
        candidates = (n_seeds if n_seeds > 1 else 1) + rounds * per_round
    else:
        keep = max(1, min(keep_seeds, n_seeds))
        candidates = (n_seeds
                      + keep * (differs(S2_TEMPS, start["temp"]) + 1)
                      + sum(differs(values, start[axis]) for axis, values in S3_AXES)
                      + (1 + len(S4_LENGTH_PENALTIES) if probe_beams else 0))
    return candidates * n_texts


def pipeline_sampling_budget(seeds, probe_beams=False):
    """--budget that runs the pipeline's whole sampling plan, whatever the voice.

    The analyser's temperature is not known before the run, so it is taken as
    off the S2 grid, the longest case (6 generations more than a temperature
    of 0.55 or 0.65 needs).
    """
    n = len(_unique(list(seeds)))
    if n < 1:
        raise ValueError("At least one seed is required")
    start = dict(ANALYSER_SAMPLING, temp=float("nan"))
    return sampling_plan(n, PIPELINE_KEEP_SEEDS, PIPELINE_PROBE_TEXTS, start,
                         "rsm", 2, probe_beams)


# ── Conditioning ──────────────────────────────────────────────────────────────

def analyser_gcl_prior(seconds):
    """gpt_cond_len that voice_analyser.py writes for a file this long."""
    return int(min(60, max(6, int(seconds))))


def _closure(gcl_lens, chunks, ref_lengths, base_gcl, base_chunk, base_max_ref,
             n_refs, extra_gcl=()):
    """Values each axis can take, clamps included (see conditioning_candidates)."""
    lengths = _unique([int(base_max_ref)] + [int(d) for d in ref_lengths if int(d) >= 1])
    g0 = max(1, min(int(base_gcl), int(base_max_ref) * n_refs))
    c0 = max(1, min(int(base_chunk), g0))
    g_base = _unique([g for g in gcl_lens if g >= 1] + [g0] + list(extra_gcl))
    # test_cond() cuts gpt_cond_len to max_ref_len x references, so a reference
    # length shorter than a requested value brings in that shorter value.
    gcls = set(g_base) | {d * n_refs for d in lengths if d * n_refs < max(g_base)}
    c_base = set([c for c in chunks if c >= 1] + [c0])
    # ... and it cuts the chunk to gpt_cond_len, which brings in those values.
    chunk_set = c_base | {g for g in gcls if g < max(c_base)}
    return gcls, chunk_set, lengths


def conditioning_candidates(gcl_lens, chunks, ref_lengths, base_gcl, base_chunk,
                            base_max_ref, n_refs=1):
    """Every (gpt_cond_len, chunk, max_ref_len, norm) the conditioning search can
    render with all the references, whichever candidates win along the way.

    C0 to C3 only ever move one axis from the current winner, C2b pairs
    values already listed, and C5 combines the requested values: all of them
    stay inside this set. Clamping can bring in two kinds of value that were
    never typed: a reference length times the number of references (when
    max_ref_len cuts gpt_cond_len) and a gpt_cond_len (when it cuts the
    chunk).
    """
    gcls, chunk_set, lengths = _closure(gcl_lens, chunks, ref_lengths, base_gcl,
                                        base_chunk, base_max_ref, n_refs)
    return {(g, c, d, n) for g in gcls for c in chunk_set for d in lengths
            for n in (0, 1) if c <= g <= d * n_refs}


def conditioning_plan(gcl_lens, chunks, ref_lengths, base_gcl, base_chunk, base_max_ref,
                      n_refs=1, n_texts=1, ref_sets=0):
    """--cond-budget that runs every conditioning stage in full, whatever wins.

    An upper bound on the generations (every candidate the stages can render,
    plus the noise probe), plus the reserve C2b keeps for C3, and never below
    what the noise probe needs. ref_sets is the number of reference subsets
    C4 tries (0 when it does not run); C5 may then work on the winning subset,
    whose candidates are counted separately.
    """
    total = len(conditioning_candidates(gcl_lens, chunks, ref_lengths, base_gcl,
                                        base_chunk, base_max_ref, n_refs))
    if ref_sets and n_refs > 1:
        lengths = _unique([int(base_max_ref)] + [int(d) for d in ref_lengths if int(d) >= 1])
        # gpt_cond_len cut to any reference length x any subset size.
        cuts = [d * r for d in lengths for r in range(1, n_refs + 1)]
        gcls, chunk_set, _ = _closure(gcl_lens, chunks, ref_lengths, base_gcl, base_chunk,
                                      base_max_ref, n_refs, extra_gcl=cuts)
        subset = max(len({(g, c, d, n) for g in gcls for c in chunk_set for d in lengths
                          for n in (0, 1) if c <= g <= d * r})
                     for r in range(1, n_refs + 1))
        # C5 on one subset never exceeds its own grid: the winner's value or a
        # requested one on each axis, both normalisations.
        grid = ((len(_unique([g for g in gcl_lens if g >= 1])) + 1)
                * (len(_unique([c for c in chunks if c >= 1])) + 1) * (len(lengths) + 1) * 2)
        total += ref_sets + min(subset, grid)
    need = (total + NOISE_PROBES) * n_texts + C2B_RESERVE
    return max(need, (1 + NOISE_PROBES + NOISE_PROBE_MARGIN) * n_texts)


# ── What the pipeline will hand XTTS ──────────────────────────────────────────

def pipeline_reference_seconds(paths, curate, recurate, keep_seconds, seconds_of):
    """Seconds of audio each voice will hand XTTS, as xtts_pipeline.py builds it.

    With curation on, an existing <ref>_curated.wav is reused unless
    re-curation is asked for, so its own length counts, not Keep s.
    Returns (path, seconds, what) for each voice.
    """
    out = []
    for path in paths:
        if curate:
            curated = os.path.splitext(path)[0] + '_curated.wav'
            if os.path.exists(curated) and not recurate:
                out.append((path, float(seconds_of(curated)), 'existing curated file'))
            else:
                out.append((path, float(keep_seconds), 'curation to Keep s'))
        else:
            out.append((path, float(seconds_of(path)), 'reference as it is'))
    return out


def whole_file_seconds(seconds):
    """Smallest whole number of seconds that reads all of a file this long.

    max_ref_len is a whole number of seconds: 30 reads every sample of a
    29.95 s file, 29 would drop the last 0.95 s. The tolerance absorbs the
    rounding of frames / sample rate.
    """
    return max(1, int(math.ceil(float(seconds) - 1e-6)))


def max_reference_for(seconds):
    """Max reference s that reads ALL the audio of every voice, capped at 60 s.

    XTTS cuts EVERY file to max_ref_len before the speaker embedding and the
    GPT conditioning alike: anything beyond is never heard. One value serves
    every voice of the run, so the longest file decides; a shorter file is
    simply read in full.
    """
    return min(LONGEST_REFERENCE, whole_file_seconds(max(seconds)))


def reference_lengths_for(seconds):
    """Reference lengths worth testing on a file this long (the shortest of
    the run): the ladder below it, and the length that reads all of it. A
    longer value reads the same audio again."""
    top = min(LONGEST_REFERENCE, whole_file_seconds(seconds))
    return sorted({x for x in REF_LENGTH_LADDER if x < top} | {top})


def gcl_candidates(seconds, max_ref, n_refs=1):
    """gcl s worth testing: the ladder from 20 s (from 10 s when the file is
    shorter than that) up to what XTTS reads of the shortest file, which is the
    last value. A longer value reads the same audio again; the analysis's own
    starting value is tested in any case."""
    top = min(LONGEST_REFERENCE, int(max_ref) * n_refs, whole_file_seconds(min(seconds)))
    floor = GCL_FLOOR if top >= GCL_FLOOR else min(GCL_LADDER)
    return sorted({g for g in GCL_LADDER if floor <= g < top} | {top})


def chunk_candidates(gcl_lens, max_ref, n_refs=1):
    """Chunk lengths every usable gpt_cond_len can take: the ladder up to the
    shortest one. A chunk longer than gpt_cond_len is ignored."""
    usable = [g for g in gcl_lens if 1 <= g <= max_ref * n_refs] or [max_ref * n_refs]
    shortest = min(usable)
    return [c for c in CHUNK_LADDER if c <= shortest] or [shortest]


def pipeline_conditioning_budget(gcl_lens, chunks, ref_lengths, max_ref, seconds):
    """--cond-budget that runs every conditioning stage for every voice.

    The pipeline gives the optimiser one reference per voice, and the
    analyser's gpt_cond_len (the search's starting point) follows each file's
    length, so each voice has its own plan: the largest one decides.
    """
    return max(conditioning_plan(gcl_lens, chunks, ref_lengths, analyser_gcl_prior(s),
                                 ANALYSER_CHUNK, max_ref, 1, PIPELINE_COND_TEXTS)
               for s in seconds)

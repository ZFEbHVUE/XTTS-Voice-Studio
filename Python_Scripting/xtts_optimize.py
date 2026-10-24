#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xtts_optimize_enhanced.py
=========================

Drop-in replacement for xtts_optimize.py.

Adds a conditioning search BEFORE the existing XTTS sampling search:

  1) gpt_cond_len
  2) gpt_cond_chunk_len
  3) sound_norm_refs
  4) reference-set choice (when several refs are supplied)
  5) seed + temperature + rep_pen + top_p/top_k
  6) optional beam/greedy probe
  7) unseen-sentence hold-out validation

The normal pipeline command remains compatible:
    python xtts_optimize.py ref.wav FR --xtts-block "{...}" ...

If this file is renamed to xtts_optimize.py, xtts_pipeline.py does not need
to be modified.

Important:
- --budget remains the SAMPLING generation budget.
- --cond-budget is a separate cap for conditioning-search generations.
- Conditioning search is ON by default. Disable with --no-search-conditioning.
- A stage adopts a candidate only when its gain exceeds the draw-to-draw
  noise measured on the same settings with several seeds (--adopt-alpha;
  1 restores the old "any gain wins").
- A stage cut short by a budget says so: live, in the summary, and in the
  result JSON ("truncated_stages"). The plan size is printed up front.
"""

try:
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
import json
from xtts_settings import parse_block, format_block, validate
import itertools
import os
import re
import tempfile
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, SCRIPT_DIR)

LANGS = {"FR","EN","ES","DE","IT","PT","PL","TR","RU","NL","CS","AR",
         "ZH-CN","HU","KO","JA","HI"}

# The search grids and schedule constants live in search_plan.py, shared with
# the plan arithmetic the [Auto] Pipeline tab uses to propose safe budgets.
from search_plan import (GRID, AXIS_ORDER, S2_TEMPS, S3_AXES, S4_LENGTH_PENALTIES,
                         NOISE_PROBES, NOISE_PROBE_MARGIN, C2B_RESERVE,
                         sampling_plan, conditioning_plan)

# Noise-aware adoption. A candidate is scored on one or two sentences with
# one random draw: two settings that sound the same can differ by several
# hundredths by luck alone, and taking every such "gain" walks the settings
# away from their priors on noise. The draw-to-draw spread is measured (same
# settings, several seeds) and a challenger replaces the incumbent only when
# its gain is larger than that spread can explain.
SD_FLOOR = 0.005        # a measured spread is never trusted below this
SD_FALLBACK = 0.010     # assumed when nothing could be measured


def adoption_bar(sd, n_challengers, alpha):
    """Smallest score gain a stage's best challenger needs to be adopted.

    The challenger and the incumbent are each one noisy draw, so their
    difference has sd*sqrt(2). The more challengers a stage tries, the more
    likely one of them looks better by chance; the one-sided quantile is
    Bonferroni-corrected so that a stage adopts pure noise with probability
    about alpha. alpha >= 1 restores the old rule: any gain is adopted.
    """
    if alpha >= 1 or n_challengers < 1:
        return 0.0
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1.0 - alpha / n_challengers)
    return max(0.0, z) * float(np.sqrt(2.0)) * max(SD_FLOOR, float(sd))


def parse_xtts_block(block):
    return parse_block(block)


def fmt(v):
    v = float(v)
    return str(int(v)) if v == int(v) else str(round(v, 3))


def format_xtts_block(seed, p):
    return format_block(dict(p, seed=seed))


def read_text(args, lang="en"):
    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, encoding="utf-8") as fh:
            raw = fh.read()
        lines = []
        for ln in raw.splitlines():
            s = ln.strip()
            if not s or s.startswith("{") or s.startswith("["):
                continue
            s = re.sub(r"\[[^\]]*\]", " ", s).strip()
            if s:
                lines.append(s)
        t = " ".join(lines).strip()
        if t:
            return t
    if args.text:
        return args.text
    from probe_texts import default_text as _dt
    return _dt(lang)



# ── Timbre distance ──────────────────────────────────────────────────────────
# Why this exists: ECAPA cosine is what every score here has been built on, and
# it is robust to pitch shifts BY DESIGN -- it has to recognise someone who
# speaks higher today than yesterday. Measured on a real case, a clone 33 Hz
# above the reference with a spectral centroid 900 Hz brighter scored 0.6465
# while an XTTS clone 24 Hz below and 9 Hz from the reference's centroid scored
# 0.6039. The ear ranked them the other way round, and unambiguously.
#
# So this measures what the ear notices and ECAPA forgives: how far the voice
# sits in pitch, how bright it is, and how clean. Each term is divided by a gap
# that is audible on its own, so a distance of 1.0 in any one of them is about
# as wrong as 1.0 in another.
TIMBRE_SCALE = {
    'f0_semitones': 2.0,    # 2 st off is a different-sounding voice
    'centroid_hz': 300.0,   # 300 Hz of brightness is plainly audible
    'hnr_db': 3.0,          # 3 dB of harmonic-to-noise is a change of texture
}


def timbre_features(path, sr_target=22050):
    """Median F0, spectral centroid and HNR of a file -- the three the ear uses.

    Returns None when the file holds no usable voiced material, so a failed
    generation is skipped rather than scored as a perfect match.
    """
    try:
        import numpy as _np
        import librosa as _lb
        y, sr = _lb.load(path, sr=sr_target, mono=True)
        if len(y) < sr // 4:
            return None
        f0, voiced, _ = _lb.pyin(y, fmin=60, fmax=500, sr=sr,
                                 frame_length=2048, hop_length=512)
        fv = f0[~_np.isnan(f0)]
        if len(fv) < 8:
            return None
        cen = float(_np.mean(_lb.feature.spectral_centroid(y=y, sr=sr)))
        # HNR from the harmonic/percussive split: cheap, and consistent between
        # the two files being compared, which is all that matters here.
        h, p = _lb.effects.hpss(y)
        eh = float(_np.mean(h ** 2)) + 1e-12
        ep = float(_np.mean(p ** 2)) + 1e-12
        return {'f0': float(_np.median(fv)),
                'centroid': cen,
                'hnr': float(10.0 * _np.log10(eh / ep))}
    except Exception:
        return None


def timbre_distance(ref, clone):
    """0 = same timbre. Each term normalised by an audible gap; see TIMBRE_SCALE."""
    import numpy as _np
    if not ref or not clone:
        return None
    st = abs(12.0 * _np.log2(max(clone['f0'], 1e-6) / max(ref['f0'], 1e-6)))
    d_f0 = st / TIMBRE_SCALE['f0_semitones']
    d_cen = abs(clone['centroid'] - ref['centroid']) / TIMBRE_SCALE['centroid_hz']
    d_hnr = abs(clone['hnr'] - ref['hnr']) / TIMBRE_SCALE['hnr_db']
    return float(d_f0 + d_cen + d_hnr)


def timbre_score(ref, clone):
    """Turn the distance into something to MAXIMISE, like every other score."""
    d = timbre_distance(ref, clone)
    return None if d is None else float(1.0 / (1.0 + d))


def _unique(seq):
    out = []
    for x in seq:
        if x not in out:
            out.append(x)
    return out


def _reference_sets(paths, max_sets=8):
    """Useful reference combinations without combinatorial explosion."""
    paths = _unique([os.path.abspath(p) for p in paths])
    if len(paths) <= 1:
        return [tuple(paths)]

    sets = []
    # all references first: usually the strongest baseline
    sets.append(tuple(paths))
    # each single reference
    sets.extend((p,) for p in paths)
    # pairs are useful when there are only a few references
    if len(paths) <= 4:
        sets.extend(tuple(c) for c in itertools.combinations(paths, 2))

    # dedupe, preserve order, cap
    out = []
    for s in sets:
        if s not in out:
            out.append(s)
        if len(out) >= max_sets:
            break
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Optimise XTTS conditioning + sampling vs accent+identity")
    ap.add_argument("reference")
    ap.add_argument("voice_refs", nargs="*")
    ap.add_argument("--xtts-block", required=True)
    ap.add_argument("--text", default=None)
    ap.add_argument("--text-file", default=None)
    ap.add_argument("--probe-text-file", help="One custom search sentence per line")
    ap.add_argument("--holdout-text-file", help="Unseen validation sentences, one per line")
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--method", default="rsm", choices=["rsm", "coord"])
    ap.add_argument("--keep-seeds", type=int, default=3)
    ap.add_argument("--probe-texts", type=int, default=2)
    ap.add_argument("--budget", type=int, default=60,
                    help="Sampling-search generation budget (default 60)")
    ap.add_argument("--objective", choices=["identity", "timbre", "both"],
                    default="identity",
                    help="What the second term of the score measures. "
                         "identity (default): ECAPA cosine, as before. "
                         "timbre: closeness in pitch, brightness and HNR -- "
                         "what the ear notices and ECAPA forgives by design. "
                         "both: the mean of the two.")
    ap.add_argument("--w-accent", type=float, default=0.6)
    ap.add_argument("--w-identity", type=float, default=0.4)
    ap.add_argument("--adopt-alpha", type=float, default=0.05,
                    help="A stage replaces its incumbent only when the best gain "
                         "exceeds the measured draw-to-draw noise, with about "
                         "this probability of adopting pure noise per stage "
                         "(default 0.05). 1 restores the old rule: any gain wins.")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--holdout-texts", type=int, default=2)
    ap.add_argument("--no-holdout", action="store_true")
    ap.add_argument("--seed-robust", action="store_true", default=True)
    ap.add_argument("--seed-mean", dest="seed_robust", action="store_false")
    ap.add_argument("--beam-width", type=int, default=3)
    ap.add_argument("--probe-beams", action="store_true")
    ap.add_argument("--max-ref-len", type=int, default=None)
    ap.add_argument("--max-ref-lens", default="",
                    help="Optional reference-duration search, e.g. '20 30 45'")
    ap.add_argument("--result-json", default=None,
                    help="Write the complete winner, references and validation scores")
    ap.add_argument("--whisper-model", default="small")
    ap.add_argument("--whisper-device", default="cpu")
    ap.add_argument("--device", default=None)

    # New conditioning controls. Search ON by default so this remains a true
    # drop-in upgrade when xtts_pipeline.py invokes it unchanged.
    ap.add_argument("--search-conditioning", dest="search_conditioning",
                    action="store_true", default=True,
                    help="Search speaker-conditioning settings (default ON)")
    ap.add_argument("--no-search-conditioning", dest="search_conditioning",
                    action="store_false")
    ap.add_argument("--cond-budget", type=int, default=24,
                    help="Generation cap for conditioning search (default 24)")
    ap.add_argument("--cond-probe-texts", type=int, default=1,
                    help="Sentences per conditioning candidate (1-2, default 1)")
    ap.add_argument("--gpt-cond-lens", default="6 10 15 20 30",
                    help='Candidate gpt_cond_len values, e.g. "6 10 15 20 30"')
    ap.add_argument("--gpt-chunks", default="4 6 8 10",
                    help='Candidate gpt_cond_chunk_len values, e.g. "4 6 8 10"')
    ap.add_argument("--search-ref-sets", dest="search_ref_sets",
                    action="store_true", default=True,
                    help="If multiple refs are supplied, test useful subsets (default ON)")
    ap.add_argument("--no-search-ref-sets", dest="search_ref_sets",
                    action="store_false")
    args = ap.parse_args()

    # Backward-compatible positional parsing:
    #   ref.wav FR
    #   ref1.wav ref2.wav FR
    extras = list(args.voice_refs)
    lang = "FR"
    if extras and extras[-1].upper() in LANGS:
        lang = extras[-1].upper()
        extras = extras[:-1]

    # FIX vs the old script: primary reference is never silently dropped when
    # extra same-speaker refs are supplied.
    all_refs = [args.reference] + extras
    for ref in all_refs:
        if not os.path.isfile(ref):
            ap.error(f"Reference not found: {ref}")
    all_refs = _unique([os.path.abspath(r) for r in all_refs])
    if not all_refs:
        raise SystemExit("[ERR] No valid reference file found.")

    xtts = parse_xtts_block(args.xtts_block)
    if args.max_ref_len is not None:
        xtts["max_ref_len"] = args.max_ref_len
    try:
        xtts = validate(xtts)
    except ValueError as exc:
        ap.error(str(exc))
    args.max_ref_len = xtts["max_ref_len"]
    try:
        text = read_text(args, lang)
    except ValueError as exc:
        ap.error(str(exc))
    block_seed = xtts.get("seed")
    try:
        seeds = ([int(v) for v in args.seeds.replace(',', ' ').split()]
                 if args.seeds is not None else [block_seed if block_seed is not None else 0])
        seeds = _unique(seeds)
        if not seeds:
            raise ValueError('At least one seed is required')
        for seed in seeds:
            validate({'seed': seed})
        ref_lengths = [int(v) for v in args.max_ref_lens.replace(',', ' ').split()]
        if any(v <= 0 for v in ref_lengths):
            raise ValueError('Reference durations must be positive')
        if args.w_accent < 0 or args.w_identity < 0 or not np.isfinite(
                args.w_accent + args.w_identity) or args.w_accent + args.w_identity <= 0:
            raise ValueError('Score weights must be finite, nonnegative, and not both zero')
        if not (np.isfinite(args.adopt_alpha) and 0 < args.adopt_alpha <= 1):
            raise ValueError('--adopt-alpha must be in (0, 1]')
    except ValueError as exc:
        ap.error(str(exc))

    from probe_texts import probe_texts as _probes, holdout_texts as _holdout
    from probe_texts import supported as _supported
    def custom_sentences(path):
        with open(path, encoding='utf-8-sig') as handle:
            result = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith('#')]
        if not result:
            raise ValueError(f"No sentences in {path}")
        return result
    known = lang.lower().split('-')[0] in _supported()
    try:
        if args.probe_text_file:
            PROBES = _unique([text] + custom_sentences(args.probe_text_file))
        elif known:
            PROBES = [text] + [t for t in _probes(lang, 5) if t != text]
        elif args.probe_texts == 1 and args.cond_probe_texts == 1:
            PROBES = [text]
        else:
            raise ValueError("This language needs --probe-text-file (or --probe-texts 1 --cond-probe-texts 1)")
        HOLDOUT = ([] if args.no_holdout else custom_sentences(args.holdout_text_file)
                   if args.holdout_text_file else _holdout(lang, 3))
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    n_probe = max(1, min(5, args.probe_texts))
    probe_texts = PROBES[:n_probe]
    cond_probe_texts = PROBES[:max(1, min(2, args.cond_probe_texts))]
    holdout_texts = [] if args.no_holdout else [
        t for t in HOLDOUT if t.strip().casefold() not in
        {q.strip().casefold() for q in probe_texts}][:max(1, min(3, args.holdout_texts))]
    if not args.no_holdout and not holdout_texts:
        ap.error("Hold-out sentences must differ from search sentences")
    if args.budget < len(probe_texts):
        ap.error('Sampling budget must cover at least one complete candidate')
    if args.search_conditioning and args.cond_budget < len(cond_probe_texts):
        ap.error('Conditioning budget must cover at least one complete candidate')
    if args.rounds < 0:
        ap.error('--rounds must be nonnegative')

    print("=" * 72)
    print("  XTTS Enhanced Optimiser — conditioning + sampling")
    print("=" * 72)
    print(f"  Language   : {lang}")
    print(f"  References : {len(all_refs)}")
    for i, r in enumerate(all_refs, 1):
        print(f"    {i}. {os.path.basename(r)}")
    _second = {'identity': 'identity', 'timbre': 'timbre',
               'both': 'identity+timbre'}[args.objective]
    print(f"  Objective  : {args.w_accent:.2f}.accent + "
          f"{args.w_identity:.2f}.{_second}")
    print(f"  Budgets    : conditioning {args.cond_budget} gens + sampling {args.budget} gens")
    print(f"  Search text: {len(probe_texts)} sentence(s); hold-out {len(holdout_texts)}")
    plan_start = {a: float(xtts.get(a, GRID[a][len(GRID[a]) // 2])) for a in GRID}
    planned = sampling_plan(len(seeds), args.keep_seeds, len(probe_texts), plan_start,
                            args.method, args.rounds, args.probe_beams)
    print(f"  Sampling plan: up to {planned} generations for a budget of {args.budget}")
    if planned > args.budget:
        print(f"  [!] The budget cannot run the whole plan: its last stages will be "
              f"cut short. --budget {planned} runs it in full.")
    if args.search_conditioning:
        _sets = (len(_reference_sets(all_refs))
                 if args.search_ref_sets and len(all_refs) > 1 else 0)
        cond_planned = conditioning_plan(
            [int(x) for x in re.findall(r"\d+", args.gpt_cond_lens)],
            [int(x) for x in re.findall(r"\d+", args.gpt_chunks)], ref_lengths,
            int(xtts.get("gpt_cond_len", 30)), int(xtts.get("gpt_cond_chunk_len", 4)),
            args.max_ref_len, len(all_refs), len(cond_probe_texts), _sets)
        print(f"  Conditioning plan: up to {cond_planned} generations for a budget of "
              f"{args.cond_budget}")
        if cond_planned > args.cond_budget:
            print(f"  [!] The conditioning budget cannot try every candidate: its last "
                  f"stages will try fewer. --cond-budget {cond_planned} runs them all.")
    if args.adopt_alpha >= 1:
        print("  Adoption   : any score gain (old rule, --adopt-alpha 1)")
    else:
        print(f"  Adoption   : only gains above the measured noise "
              f"(--adopt-alpha {args.adopt_alpha:g} per stage)")
    print("=" * 72)

    import torch
    import xtts_clone as XC
    from pron_score import PronScorer
    from speaker_identity import SpeakerEncoder

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Loading XTTS on {device}...")
    _tts, model = XC.load_xtts(device)
    print(f"[*] Loading faster-whisper '{args.whisper_model}' on {args.whisper_device}...")
    pron = PronScorer(device=args.whisper_device, model=args.whisper_model)
    print("[*] Loading ECAPA-TDNN...")
    enc = SpeakerEncoder(device="cpu" if device == "cuda" else device)

    # Robust identity target: average all same-speaker reference embeddings.
    ref_embs = [enc.embed(r) for r in all_refs]
    ref_emb = np.mean(np.stack(ref_embs), axis=0)
    ref_emb = ref_emb / (np.linalg.norm(ref_emb) + 1e-12)
    print("[OK] Ready\n")

    with tempfile.TemporaryDirectory(prefix='xtts_work_') as tmpdir:
        wa, wi = float(args.w_accent), float(args.w_identity)
        # Measured once: the reference does not change during the search.
        ref_timbre = None
        if args.objective in ('timbre', 'both'):
            # Averaged over every reference, like the identity target above: one
            # take can sit a little higher or brighter than the speaker usually is.
            feats = [f for f in (timbre_features(r) for r in all_refs) if f]
            if feats:
                ref_timbre = {k: float(np.mean([f[k] for f in feats]))
                              for k in ('f0', 'centroid', 'hnr')}
            if ref_timbre:
                print(f"  Reference timbre: F0 {ref_timbre['f0']:.0f} Hz, "
                      f"centroid {ref_timbre['centroid']:.0f} Hz, "
                      f"HNR {ref_timbre['hnr']:.1f} dB")
            else:
                print("  [!] Could not measure the reference timbre -- "
                      "falling back to identity")
                args.objective = 'identity'

        sample_evals = [0]
        cond_evals = [0]
        holdout_evals = [0]
        score_cache = {}
        latent_cache = {}

        base_cond = {
            "gpt_cond_len": int(xtts.get("gpt_cond_len", 30)),
            "gpt_cond_chunk_len": int(xtts.get("gpt_cond_chunk_len", 4)),
            "sound_norm_refs": int(bool(xtts.get("sound_norm_refs", 0))),
            "refs": tuple(all_refs),
            "max_ref_len": args.max_ref_len,
        }
        base_cond["gpt_cond_len"] = max(1, min(base_cond["gpt_cond_len"], args.max_ref_len * len(all_refs)))
        base_cond["gpt_cond_chunk_len"] = max(
            1, min(base_cond["gpt_cond_chunk_len"], base_cond["gpt_cond_len"]))

        def cond_key(c):
            return (int(c["gpt_cond_len"]),
                    int(c["gpt_cond_chunk_len"]),
                    int(bool(c["sound_norm_refs"])),
                    tuple(c["refs"]), int(c["max_ref_len"]))

        def get_latents(c):
            k = cond_key(c)
            if k not in latent_cache:
                refs = list(c["refs"])
                speaker_wav = refs if len(refs) > 1 else refs[0]
                latent_cache[k] = XC.compute_latents(
                    model, speaker_wav,
                    gpt_cond_len=int(c["gpt_cond_len"]),
                    gpt_cond_chunk_len=int(c["gpt_cond_chunk_len"]),
                    max_ref_len=int(c["max_ref_len"]),
                    sound_norm_refs=bool(c["sound_norm_refs"]),
                )
            return latent_cache[k]

        start = {
            a: float(xtts.get(a, GRID[a][len(GRID[a]) // 2]))
            for a in GRID
        }
        start["len_pen"] = float(xtts.get("len_pen", 1.0))
        start["num_beams"] = int(xtts.get("num_beams", 1))
        start["do_sample"] = bool(xtts.get("do_sample", 1))

        # Budget bookkeeping. evaluate() used to return None in silence when a
        # budget ran out: the stage's remaining candidates were skipped and the
        # summary still read as if the whole schedule had run (seen on a real
        # run: the sampling budget ran out just as the rep_pen/top_p/top_k
        # probe began). Every skipped candidate is now counted against its stage.
        current_stage = {"name": None}
        cut_stages = {}
        budget_name = {"cond": "conditioning budget (--cond-budget)",
                       "sample": "sampling budget (--budget)",
                       "holdout": "hold-out budget"}

        def note_cut(phase, used, limit):
            name = current_stage["name"] or phase
            if name not in cut_stages:
                print(f"  [!] {budget_name[phase]} exhausted ({used}/{limit}): "
                      f"'{name}' is cut short, its remaining candidates are NOT evaluated")
            cut_stages[name] = cut_stages.get(name, 0) + 1

        def sample_key(seed, prm, texts, c):
            return (
                cond_key(c), seed,
                round(float(prm["temp"]), 4),
                int(prm["top_k"]),
                round(float(prm["top_p"]), 4),
                round(float(prm["rep_pen"]), 4),
                round(float(prm.get("len_pen", 1.0)), 4),
                int(prm.get("num_beams", 1)),
                bool(prm.get("do_sample", True)),
                tuple(texts),
            )

        def evaluate(seed, prm, texts, c, phase="sample"):
            k = (phase, sample_key(seed, prm, texts, c))
            if k in score_cache:
                return score_cache[k]

            counter = {"cond": cond_evals, "sample": sample_evals, "holdout": holdout_evals}[phase]
            limit = {"cond": args.cond_budget, "sample": args.budget,
                     "holdout": len(holdout_texts)}[phase]
            if counter[0] + len(texts) > limit:
                note_cut(phase, counter[0], limit)
                return None

            latents = get_latents(c)
            frs, ids, per, tms = [], [], [], []
            for txt in texts:
                if counter[0] >= limit:
                    note_cut(phase, counter[0], limit)
                    return None
                counter[0] += 1
                wav = os.path.join(tmpdir, f"{phase}_{counter[0]}_{len(score_cache)}.wav")
                XC.generate(
                    model, txt, lang, latents, wav,
                    temperature=float(prm["temp"]),
                    length_penalty=float(prm.get("len_pen", 1.0)),
                    repetition_penalty=float(prm["rep_pen"]),
                    top_k=int(prm["top_k"]),
                    top_p=float(prm["top_p"]),
                    speed=1.0, seed=int(seed),
                    num_beams=int(prm.get("num_beams", 1)),
                    do_sample=bool(prm.get("do_sample", True)),
                )
                ps = pron.score(wav, lang=lang, target_text=txt)
                ident = enc.cosine(ref_emb, enc.embed(wav))
                # The second term of the objective, whichever it is. A file whose
                # timbre cannot be measured keeps its identity score rather than
                # being scored as perfect.
                if args.objective == 'identity' or ref_timbre is None:
                    second = float(ident)
                else:
                    ts = timbre_score(ref_timbre, timbre_features(wav))
                    if ts is None:
                        second = float(ident)
                    elif args.objective == 'timbre':
                        second = ts
                    else:
                        second = 0.5 * float(ident) + 0.5 * ts
                frs.append(float(ps["score"]))
                ids.append(float(ident))
                tms.append(second)
                per.append(wa * frs[-1] + wi * second)
                if device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

            fr = float(np.mean(frs))
            identity = float(np.mean(ids))
            second_m = float(np.mean(tms)) if tms else identity
            score = wa * fr + wi * second_m
            # The median alongside the mean: one exceptional generation can carry
            # a mean past two mediocre ones, and on a meditation of a hundred
            # sentences the typical result matters more than the best.
            median = float(np.median(per)) if per else score
            sd = float(np.std(per, ddof=1)) if len(per) > 1 else 0.0
            sem = sd / np.sqrt(len(per)) if len(per) > 1 else 0.0
            rec = {
                "score": score, "french": fr, "identity": identity,
                "phase": phase, "texts": tuple(texts),
                "timbre": second_m if args.objective != 'identity' else None,
                "seed": int(seed), "per_text": per, "sd": sd, "sem": sem,
                "median": median,
                "worst": min(per), "cond": dict(c),
                "temp": float(prm["temp"]), "rep_pen": float(prm["rep_pen"]),
                "top_p": float(prm["top_p"]), "top_k": int(prm["top_k"]),
                "len_pen": float(prm.get("len_pen", 1.0)),
                "num_beams": int(prm.get("num_beams", 1)),
                "do_sample": bool(prm.get("do_sample", True)),
            }
            score_cache[k] = rec

            ctag = (f"gcl={c['gpt_cond_len']:>2} chunk={c['gpt_cond_chunk_len']:>2} "
                    f"norm={int(bool(c['sound_norm_refs']))} refs={len(c['refs'])} "
                    f"max_ref={c['max_ref_len']}")
            print(f"  [{phase[0].upper()}{counter[0]:>3}] {ctag} | "
                  f"seed={seed:>3} t={prm['temp']:.3f} -> "
                  f"score={score:.3f} (acc={fr:.3f} id={identity:.3f})")
            return rec

        # Draw-to-draw noise per evaluation, by phase. Measured whenever the
        # schedule allows it (see adoption_bar), assumed only otherwise.
        draw_noise = {"cond": None, "sample": None}

        def noise_sd(phase):
            sd = draw_noise.get(phase)
            if sd is None and phase == "sample" and draw_noise["cond"] is not None:
                # Measured on the conditioning sentences: a mean over more
                # sentences is proportionally steadier.
                sd = draw_noise["cond"] * float(np.sqrt(len(cond_probe_texts) / len(probe_texts)))
            return SD_FALLBACK if sd is None else max(SD_FLOOR, sd)

        def measure_noise(phase, records, what):
            scores = [r["score"] for r in records if r]
            if len(scores) < 3:
                return
            draw_noise[phase] = float(np.std(scores, ddof=1))
            lone = adoption_bar(draw_noise[phase], 1, args.adopt_alpha)
            print(f"  [*] Draw-to-draw noise ({what}, {len(scores)} seeds): sd "
                  f"{draw_noise[phase]:.3f} per evaluation -> a lone challenger needs "
                  f"> {lone:+.3f} to be adopted")
            if phase == "sample":
                n = len(probe_texts)
                print(f"      (scored on {n} sentence(s); the bar falls as 1/sqrt(n): "
                      f"--probe-texts {2 * n} would bring it to about {lone / np.sqrt(2):.3f})")

        def describe(r):
            if r["phase"] == "cond":
                c = r["cond"]
                return (f"gcl={c['gpt_cond_len']} chunk={c['gpt_cond_chunk_len']} "
                        f"norm={int(bool(c['sound_norm_refs']))} refs={len(c['refs'])} "
                        f"max_ref={c['max_ref_len']}")
            decode = (f"beam{r['num_beams']}" if r["num_beams"] > 1
                      else "sampling" if r["do_sample"] else "greedy")
            return (f"seed={r['seed']} temp={r['temp']:.3f} rep={r['rep_pen']:g} "
                    f"top_p={r['top_p']:g} top_k={r['top_k']} {decode} "
                    f"len_pen={r['len_pen']:g}")

        def adopt(incumbent, challengers, phase, stage):
            """The stage's best challenger if it beats the incumbent beyond the
            measured noise; otherwise the incumbent stays. Replaces the old
            'any higher score wins', which adopted gains of a thousandth."""
            pool, seen = [], {id(incumbent)}
            for r in challengers:
                if r is not None and id(r) not in seen:
                    seen.add(id(r))
                    pool.append(r)
            if incumbent is None:
                return max(pool, key=lambda r: r["score"]) if pool else None
            if not pool:
                return incumbent
            top = max(pool, key=lambda r: r["score"])
            gain = top["score"] - incumbent["score"]
            bar = adoption_bar(noise_sd(phase), len(pool), args.adopt_alpha)
            if gain > bar:
                print(f"  -> {stage}: ADOPTED {describe(top)} ({gain:+.3f}, "
                      f"bar {bar:.3f} for {len(pool)} candidate(s))")
                return top
            if gain > 0:
                print(f"  -> {stage}: kept {describe(incumbent)} -- best gain "
                      f"{gain:+.3f} is within the noise (bar {bar:.3f} for "
                      f"{len(pool)} candidate(s))")
            else:
                print(f"  -> {stage}: kept {describe(incumbent)} (no candidate scored higher)")
            return incumbent

        # ------------------------------------------------------------------
        # STAGE C — speaker-conditioning search
        # ------------------------------------------------------------------
        best_cond = dict(base_cond)
        if args.search_conditioning:
            print("\n" + "-" * 72)
            print("  STAGE C — speaker-conditioning search")
            print("-" * 72)
            fixed_seed = seeds[0]
            fixed_prm = dict(start)

            def test_cond(c, seed=None):
                c = dict(c)
                c["gpt_cond_len"] = max(1, min(int(c["gpt_cond_len"]), int(c["max_ref_len"]) * len(c["refs"])))
                c["gpt_cond_chunk_len"] = max(
                    1, min(int(c["gpt_cond_chunk_len"]), int(c["gpt_cond_len"])))
                return evaluate(fixed_seed if seed is None else seed, fixed_prm,
                                cond_probe_texts, c, phase="cond")

            def run_stage(stage, conds):
                """All of a stage's candidates against the stage's incumbent."""
                nonlocal best_r, best_cond
                current_stage["name"] = stage
                results = [test_cond(c) for c in conds]
                current_stage["name"] = None
                win = adopt(best_r, results, "cond", stage)
                if win is not None and win is not best_r:
                    best_r, best_cond = win, dict(win["cond"])

            current_stage["name"] = "C base"
            base_r = test_cond(best_cond)
            current_stage["name"] = None
            best_r = base_r

            # Noise of this stage: the base settings rendered with a few other
            # seeds. Worth its cost only when the budget leaves room for the
            # stages themselves.
            probe_seeds = [s for s in _unique(list(seeds[1:]) + [1, 2, 3, 4])
                           if s != fixed_seed][:NOISE_PROBES]
            if base_r is not None and args.cond_budget >= (1 + NOISE_PROBES + NOISE_PROBE_MARGIN) * len(cond_probe_texts):
                current_stage["name"] = "C noise probe"
                reps = [base_r] + [test_cond(base_r["cond"], seed=s) for s in probe_seeds]
                current_stage["name"] = None
                measure_noise("cond", reps, "base conditioning")
            if draw_noise["cond"] is None:
                print(f"  [*] Conditioning noise not measured (--cond-budget too small "
                      f"for the probe): assuming sd {SD_FALLBACK:.3f}")

            # Reference embedding length is per file, GPT conditioning is total.
            if ref_lengths:
                print("\n  C0 — max_ref_len")
                run_stage("C0 max_ref_len",
                          [dict(best_cond, max_ref_len=d) for d in _unique(ref_lengths)])

            # C1: gpt_cond_len
            requested_lens = [int(x) for x in re.findall(r"\d+", args.gpt_cond_lens)]
            lens = _unique([x for x in requested_lens
                            if 1 <= x <= best_cond["max_ref_len"] * len(best_cond["refs"])] +
                           [best_cond["gpt_cond_len"]])
            print("\n  C1 — gpt_cond_len")
            run_stage("C1 gpt_cond_len",
                      [dict(best_cond, gpt_cond_len=gcl,
                            gpt_cond_chunk_len=min(best_cond["gpt_cond_chunk_len"], gcl))
                       for gcl in lens])

            # C2: chunk length under winning gpt_cond_len
            chunks = [int(x) for x in re.findall(r"\d+", args.gpt_chunks)]
            chunks = _unique([x for x in chunks
                              if 1 <= x <= best_cond["gpt_cond_len"]] +
                             [best_cond["gpt_cond_chunk_len"]])
            print("\n  C2 — gpt_cond_chunk_len")
            run_stage("C2 gpt_cond_chunk_len",
                      [dict(best_cond, gpt_cond_chunk_len=ch) for ch in chunks])

            # C2b: the two are searched one after the other, so the pairs that were
            # never tried are exactly the ones where the winner of C1 is not the
            # best partner for the winner of C2 -- 30/8 is unreachable once 15 won
            # on chunk=4. Rather than a full grid, revisit the neighbourhood of both
            # winners: a handful of pairs, and only when there is budget left.
            gw, cw = best_cond["gpt_cond_len"], best_cond["gpt_cond_chunk_len"]
            near_g = _unique([x for x in lens if 0.5 * gw <= x <= 2.0 * gw])[:3]
            near_c = _unique([x for x in chunks if 0.5 * cw <= x <= 2.0 * cw])[:3]
            pairs = [(g, c) for g in near_g for c in near_c
                     if c <= g and (g, c) != (gw, cw)]
            if pairs:
                print("\n  C2b — gpt_cond_len x chunk (pairs the sequential "
                      "search cannot reach)")
                if cond_evals[0] < args.cond_budget - C2B_RESERVE:
                    run_stage("C2b gpt_cond_len x chunk",
                              [dict(best_cond, gpt_cond_len=g, gpt_cond_chunk_len=ch)
                               for g, ch in pairs])
                else:
                    # Skipped on purpose, to leave the budget to C3/C4 -- but said.
                    cut_stages["C2b gpt_cond_len x chunk"] = len(pairs)
                    print(f"  [!] conditioning budget nearly spent ({cond_evals[0]}/"
                          f"{args.cond_budget}): C2b skipped, {len(pairs)} pair(s) "
                          f"NOT evaluated")

            # C3: reference normalization
            print("\n  C3 — sound_norm_refs")
            run_stage("C3 sound_norm_refs",
                      [dict(best_cond, sound_norm_refs=sn) for sn in (0, 1)])

            # C4: useful reference subsets, only when several same-speaker refs exist
            if args.search_ref_sets and len(all_refs) > 1:
                print("\n  C4 — reference-set search")
                run_stage("C4 reference sets",
                          [dict(best_cond, refs=tuple(rs)) for rs in _reference_sets(all_refs)])

            # Joint bounded grid: later GPT/chunk winners can change which
            # reference cap or normalization is best. Nearest choices go first.
            from itertools import product
            anchor = dict(best_cond)
            # Earlier short-cap winners may have filtered out longer requested
            # GPT/chunk lengths. Rebuild from the original requests here; each
            # combination is checked against its OWN cap below.
            joint_lens = [x for x in requested_lens if x >= 1]
            joint_chunks = [int(x) for x in re.findall(r"\d+", args.gpt_chunks) if int(x) >= 1]
            choices = [_unique([anchor['gpt_cond_len']] + joint_lens),
                       _unique([anchor['gpt_cond_chunk_len']] + joint_chunks),
                       _unique([anchor['max_ref_len']] + ref_lengths),
                       [int(anchor['sound_norm_refs']), 1 - int(anchor['sound_norm_refs'])]]
            names = ('gpt_cond_len', 'gpt_cond_chunk_len', 'max_ref_len', 'sound_norm_refs')
            combinations = [values for values in product(*choices)
                            if values[1] <= values[0] <= values[2] * len(anchor['refs'])]
            combinations.sort(key=lambda values: sum(value != anchor[name]
                                                     for name, value in zip(names, values)))
            if cond_evals[0] + len(cond_probe_texts) <= args.cond_budget:
                print('\n  C5 — joint GPT/chunk/reference-cap/normalization checks')
                # This stage fills whatever budget is left, by design: running
                # out here is expected, so it is reported as information.
                current_stage["name"] = "C5 joint checks"
                results, tried = [], 0
                for values in combinations:
                    if cond_evals[0] + len(cond_probe_texts) > args.cond_budget:
                        break
                    results.append(test_cond(dict(anchor, **dict(zip(names, values)))))
                    tried += 1
                current_stage["name"] = None
                if tried < len(combinations):
                    print(f"  [*] C5 used the remaining budget: {tried} of "
                          f"{len(combinations)} combinations tried")
                win = adopt(best_r, results, "cond", "C5 joint checks")
                if win is not None and win is not best_r:
                    best_r, best_cond = win, dict(win["cond"])

            if best_r:
                print("\n  Conditioning winner:")
                print(f"    gpt_cond_len       = {best_cond['gpt_cond_len']}")
                print(f"    gpt_cond_chunk_len = {best_cond['gpt_cond_chunk_len']}")
                print(f"    sound_norm_refs    = {int(bool(best_cond['sound_norm_refs']))}")
                print(f"    refs               = {len(best_cond['refs'])}")
                for r in best_cond["refs"]:
                    print(f"      - {os.path.basename(r)}")
                print(f"    score              = {best_r['score']:.3f} "
                      f"(acc {best_r['french']:.3f}, id {best_r['identity']:.3f})")
        else:
            print("\n[*] Conditioning search disabled; using input block values.")

        # ------------------------------------------------------------------
        # SAMPLING SEARCH, using the winning conditioning
        # ------------------------------------------------------------------
        cwin = best_cond

        def P(temp, rep=None, tp=None, tk=None, lp=None):
            return {
                "temp": float(temp),
                "rep_pen": start["rep_pen"] if rep is None else float(rep),
                "top_k": start["top_k"] if tk is None else int(tk),
                "top_p": start["top_p"] if tp is None else float(tp),
                "len_pen": start["len_pen"] if lp is None else float(lp),
                "num_beams": 1,
                "do_sample": True,
            }

        def screen_seeds(params, stage):
            """Every seed at the same settings: the candidates of the seed
            choice AND the measurement of the sampling noise."""
            current_stage["name"] = stage
            recs = [evaluate(s, params, probe_texts, cwin, phase="sample") for s in seeds]
            current_stage["name"] = None
            recs = [r for r in recs if r]
            measure_noise("sample", recs, "start settings")
            if draw_noise["sample"] is None:
                note_unmeasured()
            return recs

        def note_unmeasured():
            how = ("scaled from the conditioning noise" if draw_noise["cond"] is not None
                   else f"assuming sd {SD_FALLBACK:.3f}")
            print(f"  [*] Sampling noise not measured (fewer than 3 seeds): {how}; "
                  f"pass more --seeds to measure it")

        print("\n" + "=" * 72)
        print("  SAMPLING SEARCH")
        print("=" * 72)

        if args.method == "coord":
            if len(seeds) > 1:
                print("\n  Seed screen")
                ss = screen_seeds(start, "seed screen")
                seed = max(ss, key=lambda r: r["score"])["seed"] if ss else seeds[0]
            else:
                seed = seeds[0]
                note_unmeasured()

            cur = dict(start)
            current_stage["name"] = "coord start"
            best = evaluate(seed, cur, probe_texts, cwin, phase="sample")
            current_stage["name"] = None
            for rnd in range(1, args.rounds + 1):
                improved = False
                for axis in AXIS_ORDER:
                    stage = f"coord round {rnd} {axis}"
                    current_stage["name"] = stage
                    trials = [evaluate(seed, dict(cur, **{axis: v}), probe_texts, cwin,
                                       phase="sample") for v in GRID[axis]]
                    current_stage["name"] = None
                    win = adopt(best, trials, "sample", stage)
                    if win is not None and win is not best:
                        best, improved = win, True
                        cur[axis] = win[axis]
                if not improved:
                    break
            if best is None:
                raise SystemExit("[ERR] No complete sampling candidate was evaluated")

        else:
            # Stage S1 — robust seed screen
            rank_mode = "worst-case" if args.seed_robust else "mean"
            print(f"\n  S1 — seed screen ({rank_mode})")
            screen = screen_seeds(P(start["temp"]), "S1 seed screen")
            screen.sort(key=lambda r: -(r["worst"] if args.seed_robust else r["score"]))
            keep = [r["seed"] for r in screen[:max(1, args.keep_seeds)]]
            print(f"  kept seeds: {keep}")

            # Stage S2 — temperature surface. Seeds are interchangeable draws,
            # so the best one is simply taken; a temperature is a setting, and
            # must beat its seed's start temperature beyond the noise.
            print("\n  S2 — temperature response surface")
            per_seed_best = []
            for s in keep:
                stage = f"S2 temperature (seed {s})"
                current_stage["name"] = stage
                incumbent = evaluate(s, P(start["temp"]), probe_texts, cwin, phase="sample")
                pts = [r for r in (evaluate(s, P(t), probe_texts, cwin, phase="sample")
                                   for t in S2_TEMPS) if r]
                fit = list({id(r): r for r in pts + ([incumbent] if incumbent else [])}.values())
                if len(fit) >= 3:
                    ts = np.array([r["temp"] for r in fit])
                    ss = np.array([r["score"] for r in fit])
                    a, b, _c = np.polyfit(ts, ss, 2)
                    if a < -1e-6:
                        tstar = float(np.clip(-b / (2 * a), 0.45, 0.85))
                        r = evaluate(s, P(tstar), probe_texts, cwin, phase="sample")
                        if r:
                            pts.append(r)
                current_stage["name"] = None
                win = adopt(incumbent, pts, "sample", stage)
                if win:
                    per_seed_best.append(win)

            if not per_seed_best:
                raise SystemExit("[ERR] Sampling budget exhausted before a winner was measured.")
            best = max(per_seed_best, key=lambda r: r["score"])

            # Stage S3 — rep_pen + top_p + top_k, each against the incumbent.
            print("\n  S3 — sampling-axis probe")
            for axis, values in S3_AXES:
                stage = f"S3 {axis}"
                current_stage["name"] = stage
                trials = []
                for value in values:
                    trial = {key: best[key] for key in
                             ("temp", "rep_pen", "top_p", "top_k", "len_pen",
                              "num_beams", "do_sample")}
                    trial[axis] = value
                    trials.append(evaluate(best["seed"], trial, probe_texts, cwin,
                                           phase="sample"))
                current_stage["name"] = None
                best = adopt(best, trials, "sample", stage)

            # Stage S4 — optional deterministic decode + length_penalty search.
            if args.probe_beams:
                print("\n  S4 — beam/greedy + length_penalty")
                stage = "S4 beam/greedy"
                current_stage["name"] = stage
                bw = max(2, int(args.beam_width))
                decode_cands = []

                # Greedy
                pg = dict(P(best["temp"], rep=best["rep_pen"],
                            tp=best["top_p"], tk=best["top_k"]),
                          num_beams=1, do_sample=False)
                try:
                    decode_cands.append(evaluate(best["seed"], pg, probe_texts, cwin,
                                                 phase="sample"))
                except Exception as e:
                    print(f"  greedy skipped: {type(e).__name__}: {e}")

                # Beam: len_pen is meaningful here.
                for lp in S4_LENGTH_PENALTIES:
                    pb = dict(P(best["temp"], rep=best["rep_pen"],
                                tp=best["top_p"], tk=best["top_k"], lp=lp),
                              num_beams=bw, do_sample=False)
                    try:
                        decode_cands.append(evaluate(best["seed"], pb, probe_texts, cwin,
                                                     phase="sample"))
                    except Exception as e:
                        msg = str(e).lower()
                        if "out of memory" in msg:
                            print(f"  beam({bw}) skipped: not enough VRAM")
                            break
                        print(f"  beam({bw}) lp={lp}: skipped ({type(e).__name__}: {e})")
                current_stage["name"] = None
                best = adopt(best, decode_cands, "sample", stage)

        # ------------------------------------------------------------------
        # Ranking and hold-out
        # ------------------------------------------------------------------
        candidates = [
            r for r in score_cache.values()
            if cond_key(r["cond"]) == cond_key(cwin)
            and r["phase"] == "sample" and r["texts"] == tuple(probe_texts)
            and r.get("seed") is not None
        ]
        candidates.sort(key=lambda r: -r["score"])

        print(f"\n{'='*72}")
        print(f"  TOP SAMPLING RESULTS ({sample_evals[0]} generations)")
        print("=" * 72)
        print(f"  {'score':>6}{'acc':>8}{'ident':>8}{'seed':>6}{'temp':>7}"
              f"{'rep':>6}{'top_p':>7}{'top_k':>7}{'beam':>6}{'lp':>6}")
        for r in candidates[:8]:
            print(f"  {r['score']:>6.3f}{r['french']:>8.3f}{r['identity']:>8.3f}"
                  f"{r['seed']:>6}{r['temp']:>7.3f}{r['rep_pen']:>6.1f}"
                  f"{r['top_p']:>7.2f}{r['top_k']:>7}{r['num_beams']:>6}"
                  f"{r['len_pen']:>6.2f}")

        print(f"  Evaluations: conditioning {cond_evals[0]}/{args.cond_budget}, sampling {sample_evals[0]}/{args.budget} (budgets are caps; candidate schedules may finish earlier).")
        if cut_stages:
            print("\n  [!] THE BUDGET CUT THESE STAGES SHORT -- their remaining "
                  "candidates were never evaluated:")
            for name, skipped in cut_stages.items():
                print(f"      {name}: {skipped} candidate(s) skipped")
            print("      Their settings stayed where the search had left them. Raise "
                  "--budget / --cond-budget to run the whole schedule.")
        print(f"\n  Best on search: {best['score']:.3f} "
              f"(accent {best['french']:.3f}, identity {best['identity']:.3f})")

        held = None
        if holdout_texts:
            print(f"\n{'-'*72}")
            print(f"  HOLD-OUT VALIDATION ({len(holdout_texts)} unseen sentence(s))")
            print("-" * 72)

            # Hold-out should never be starved by the normal sampling budget.
            hp = {
                "temp": best["temp"], "rep_pen": best["rep_pen"],
                "top_k": best["top_k"], "top_p": best["top_p"],
                "len_pen": best["len_pen"],
                "num_beams": best["num_beams"],
                "do_sample": best["do_sample"],
            }
            current_stage["name"] = "hold-out"
            held = evaluate(best["seed"], hp, holdout_texts, cwin, phase="holdout")
            current_stage["name"] = None
            if held:
                drop = best["score"] - held["score"]
                noise = max(0.01, np.hypot(best.get("sem", 0), held.get("sem", 0)))
                print(f"  HELD-OUT score {held['score']:.3f} ±{held['sd']:.3f} "
                      f"(accent {held['french']:.3f}, identity {held['identity']:.3f})")
                if len(held.get("per_text", [])) > 1:
                    gap = held["median"] - held["score"]
                    print(f"  HELD-OUT median {held['median']:.3f} "
                          f"({gap:+.3f} vs the mean)")
                    # A median well below the mean means one sentence came out much
                    # better than the rest: the settings suit that sentence, not the
                    # voice. Worth saying, since the mean alone hides it.
                    if gap < -2 * noise:
                        print(f"  [!] One sentence is carrying the mean "
                              f"(per-sentence: "
                              f"{', '.join(f'{x:.3f}' for x in held['per_text'])}). "
                              f"Judge on the median.")
                print(f"  Search -> hold-out drop {drop:+.3f}; noise ~{noise:.3f}")
                if drop > 2 * noise:
                    print("  [!] Some overfitting remains; use more probe texts before raising budget.")
                else:
                    print("  [INFO] No large score drop on these held-out texts; this is not proof of voice quality or a calibrated significance test.")

        # ------------------------------------------------------------------
        # Final block: conditioning winner + sampling winner.
        # ------------------------------------------------------------------
        win = dict(xtts)
        for a in GRID:
            win[a] = best[a]
        win["len_pen"] = best["len_pen"]
        win["num_beams"] = best["num_beams"]
        win["do_sample"] = int(best["do_sample"])
        win["max_ref_len"] = int(cwin["max_ref_len"])
        win["gpt_cond_len"] = int(cwin["gpt_cond_len"])
        win["gpt_cond_chunk_len"] = int(cwin["gpt_cond_chunk_len"])
        win["sound_norm_refs"] = int(bool(cwin["sound_norm_refs"]))

        print(f"\n{'='*72}")
        print("  FINAL WINNER")
        print("=" * 72)
        print(f"  conditioning : gpt_cond_len={win['gpt_cond_len']}  "
              f"chunk={win['gpt_cond_chunk_len']}  "
              f"sound_norm_refs={win['sound_norm_refs']}")
        print(f"  references   : {len(cwin['refs'])}")
        for r in cwin["refs"]:
            print(f"    - {os.path.basename(r)}")
        print(f"  sampling     : seed={best['seed']} temp={best['temp']:.3f} "
              f"rep={best['rep_pen']:.1f} top_p={best['top_p']:.2f} "
              f"top_k={best['top_k']} beams={best['num_beams']} "
              f"len_pen={best['len_pen']:.2f}")
        if held:
            print(f"  HELD-OUT score {held['score']:.3f} "
                  f"(accent {held['french']:.3f}, identity {held['identity']:.3f})")
        if cut_stages:
            print(f"  [!] found with a cut schedule: {', '.join(cut_stages)}")

        block = format_xtts_block(best['seed'], win)
        result = {
            "schema_version": 1, "xtts_block": block,
            "params": validate(dict(win, seed=best['seed'])),
            "selected_refs": list(cwin['refs']), "all_refs": all_refs,
            "target_reference": os.path.abspath(args.reference), "language": lang,
            "search_texts": probe_texts, "holdout_texts": holdout_texts,
            "scores": {"search": best['score'],
                       "held_out": held['score'] if held else None,
                       "french": (held or best)['french'],
                       "identity": (held or best)['identity']},
            "generations": {"conditioning": cond_evals[0],
                            "sampling": sample_evals[0], "holdout": holdout_evals[0]},
            "truncated_stages": dict(cut_stages),
            "noise": {"conditioning_sd": draw_noise["cond"],
                      "sampling_sd": draw_noise["sample"],
                      "adopt_alpha": args.adopt_alpha},
        }
        if args.result_json:
            result_path = os.path.abspath(args.result_json)
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            with open(result_path + '.tmp', 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            os.replace(result_path + '.tmp', result_path)
        print('XTTS_REFS_JSON: ' + json.dumps(result['selected_refs']))
        print("\n  Paste into the Validator/Comparator:")
        print(f"  {format_xtts_block(best['seed'], win)}")
        print("[OK] Done.")


if __name__ == "__main__":
    main()

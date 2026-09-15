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
"""

try:
    import sys as _sys
    if hasattr(_sys.stdout, "reconfigure"):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import argparse
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

GRID = {
    "temp":    [0.50, 0.55, 0.60, 0.65, 0.70],
    "rep_pen": [4.0, 5.0, 7.0, 10.0],
    "top_p":   [0.80, 0.85, 0.90],
    "top_k":   [30, 50, 70],
}
AXIS_ORDER = ["temp", "rep_pen", "top_p", "top_k"]


def parse_xtts_block(block):
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", block)]
    keys = ["seed","trim_start","trim_end","fade_in","fade_out","temp","top_k",
            "top_p","rep_pen","len_pen","gpt_cond_len","gpt_cond_chunk_len",
            "sound_norm_refs","num_beams"]
    out = {}
    for i, k in enumerate(keys):
        if i + 1 < len(nums):
            out[k] = nums[i + 1]
    return out


def fmt(v):
    v = float(v)
    return str(int(v)) if v == int(v) else str(round(v, 3))


def format_xtts_block(seed, p):
    order = [
        1, seed,
        int(p.get("trim_start", 0)), int(p.get("trim_end", 0)),
        int(p.get("fade_in", 100)), int(p.get("fade_out", 250)),
        p["temp"], int(p["top_k"]), p["top_p"], p["rep_pen"],
        p.get("len_pen", 1.0),
        int(p.get("gpt_cond_len", 30)),
        int(p.get("gpt_cond_chunk_len", 6)),
        int(p.get("sound_norm_refs", 0)),
    ]
    if int(p.get("num_beams", 1)) != 1:
        order.append(int(p["num_beams"]))
    return "{" + ", ".join(fmt(v) for v in order) + "}"


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
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--method", default="rsm", choices=["rsm", "coord"])
    ap.add_argument("--keep-seeds", type=int, default=3)
    ap.add_argument("--probe-texts", type=int, default=2)
    ap.add_argument("--budget", type=int, default=60,
                    help="Sampling-search generation budget (default 60)")
    ap.add_argument("--w-accent", type=float, default=0.6)
    ap.add_argument("--w-identity", type=float, default=0.4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--holdout-texts", type=int, default=2)
    ap.add_argument("--no-holdout", action="store_true")
    ap.add_argument("--seed-robust", action="store_true", default=True)
    ap.add_argument("--seed-mean", dest="seed_robust", action="store_false")
    ap.add_argument("--beam-width", type=int, default=3)
    ap.add_argument("--probe-beams", action="store_true")
    ap.add_argument("--max-ref-len", type=int, default=30)
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
    all_refs = _unique([r for r in all_refs if os.path.exists(r)])
    if not all_refs:
        raise SystemExit("[ERR] No valid reference file found.")

    xtts = parse_xtts_block(args.xtts_block)
    text = read_text(args, lang)
    block_seed = int(xtts.get("seed", 0))
    seeds = ([int(s) for s in re.findall(r"-?\d+", args.seeds)]
             if args.seeds else [block_seed])

    from probe_texts import probe_texts as _probes, holdout_texts as _holdout
    PROBES = [text] + [t for t in _probes(lang, 5) if t != text]
    HOLDOUT = _holdout(lang, 3)
    n_probe = max(1, min(5, args.probe_texts))
    probe_texts = PROBES[:n_probe]
    cond_probe_texts = PROBES[:max(1, min(2, args.cond_probe_texts))]
    holdout_texts = [] if args.no_holdout else HOLDOUT[:max(1, min(3, args.holdout_texts))]

    print("=" * 72)
    print("  XTTS Enhanced Optimiser — conditioning + sampling")
    print("=" * 72)
    print(f"  Language   : {lang}")
    print(f"  References : {len(all_refs)}")
    for i, r in enumerate(all_refs, 1):
        print(f"    {i}. {os.path.basename(r)}")
    print(f"  Objective  : {args.w_accent:.2f}.accent + {args.w_identity:.2f}.identity")
    print(f"  Budgets    : conditioning {args.cond_budget} gens + sampling {args.budget} gens")
    print(f"  Search text: {len(probe_texts)} sentence(s); hold-out {len(holdout_texts)}")
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

    tmpdir = tempfile.mkdtemp(prefix="xtts_opt_")
    wa, wi = float(args.w_accent), float(args.w_identity)

    sample_evals = [0]
    cond_evals = [0]
    score_cache = {}
    latent_cache = {}

    base_cond = {
        "gpt_cond_len": int(xtts.get("gpt_cond_len", 30)),
        "gpt_cond_chunk_len": int(xtts.get("gpt_cond_chunk_len", 6)),
        "sound_norm_refs": int(bool(xtts.get("sound_norm_refs", 0))),
        "refs": tuple(all_refs),
    }
    base_cond["gpt_cond_len"] = max(1, min(base_cond["gpt_cond_len"], args.max_ref_len))
    base_cond["gpt_cond_chunk_len"] = max(
        1, min(base_cond["gpt_cond_chunk_len"], base_cond["gpt_cond_len"]))

    def cond_key(c):
        return (int(c["gpt_cond_len"]),
                int(c["gpt_cond_chunk_len"]),
                int(bool(c["sound_norm_refs"])),
                tuple(c["refs"]))

    def get_latents(c):
        k = cond_key(c)
        if k not in latent_cache:
            refs = list(c["refs"])
            speaker_wav = refs if len(refs) > 1 else refs[0]
            latent_cache[k] = XC.compute_latents(
                model, speaker_wav,
                gpt_cond_len=int(c["gpt_cond_len"]),
                gpt_cond_chunk_len=int(c["gpt_cond_chunk_len"]),
                max_ref_len=int(args.max_ref_len),
                sound_norm_refs=bool(c["sound_norm_refs"]),
            )
        return latent_cache[k]

    start = {
        a: float(xtts.get(a, GRID[a][len(GRID[a]) // 2]))
        for a in GRID
    }
    start["len_pen"] = float(xtts.get("len_pen", 1.0))
    start["num_beams"] = int(xtts.get("num_beams", 1))
    start["do_sample"] = True

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
        k = sample_key(seed, prm, texts, c)
        if k in score_cache:
            return score_cache[k]

        counter = cond_evals if phase == "cond" else sample_evals
        limit = args.cond_budget if phase == "cond" else args.budget
        if counter[0] >= limit:
            return None

        latents = get_latents(c)
        frs, ids, per = [], [], []
        for txt in texts:
            if counter[0] >= limit:
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
            frs.append(float(ps["score"]))
            ids.append(float(ident))
            per.append(wa * frs[-1] + wi * ids[-1])
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        fr = float(np.mean(frs))
        identity = float(np.mean(ids))
        score = wa * fr + wi * identity
        sd = float(np.std(per, ddof=1)) if len(per) > 1 else 0.0
        sem = sd / np.sqrt(len(per)) if len(per) > 1 else 0.0
        rec = {
            "score": score, "french": fr, "identity": identity,
            "seed": int(seed), "per_text": per, "sd": sd, "sem": sem,
            "worst": min(per), "cond": dict(c),
            "temp": float(prm["temp"]), "rep_pen": float(prm["rep_pen"]),
            "top_p": float(prm["top_p"]), "top_k": int(prm["top_k"]),
            "len_pen": float(prm.get("len_pen", 1.0)),
            "num_beams": int(prm.get("num_beams", 1)),
            "do_sample": bool(prm.get("do_sample", True)),
        }
        score_cache[k] = rec

        ctag = (f"gcl={c['gpt_cond_len']:>2} chunk={c['gpt_cond_chunk_len']:>2} "
                f"norm={int(bool(c['sound_norm_refs']))} refs={len(c['refs'])}")
        print(f"  [{phase[0].upper()}{counter[0]:>3}] {ctag} | "
              f"seed={seed:>3} t={prm['temp']:.3f} -> "
              f"score={score:.3f} (acc={fr:.3f} id={identity:.3f})")
        return rec

    def tie(a, b):
        noise = max(0.01, np.hypot(a.get("sem", 0.0), b.get("sem", 0.0)))
        return abs(a["score"] - b["score"]) <= noise

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

        def test_cond(c):
            c = dict(c)
            c["gpt_cond_len"] = max(1, min(int(c["gpt_cond_len"]), int(args.max_ref_len)))
            c["gpt_cond_chunk_len"] = max(
                1, min(int(c["gpt_cond_chunk_len"]), int(c["gpt_cond_len"])))
            return evaluate(fixed_seed, fixed_prm, cond_probe_texts, c, phase="cond")

        base_r = test_cond(best_cond)
        best_r = base_r

        # C1: gpt_cond_len
        lens = [int(x) for x in re.findall(r"\d+", args.gpt_cond_lens)]
        lens = _unique([x for x in lens if 1 <= x <= args.max_ref_len] +
                       [best_cond["gpt_cond_len"]])
        print("\n  C1 — gpt_cond_len")
        for gcl in lens:
            if cond_evals[0] >= args.cond_budget:
                break
            c = dict(best_cond)
            c["gpt_cond_len"] = gcl
            c["gpt_cond_chunk_len"] = min(c["gpt_cond_chunk_len"], gcl)
            r = test_cond(c)
            if r and (best_r is None or r["score"] > best_r["score"]):
                best_r, best_cond = r, dict(c)

        # C2: chunk length under winning gpt_cond_len
        chunks = [int(x) for x in re.findall(r"\d+", args.gpt_chunks)]
        chunks = _unique([x for x in chunks
                          if 1 <= x <= best_cond["gpt_cond_len"]] +
                         [best_cond["gpt_cond_chunk_len"]])
        print("\n  C2 — gpt_cond_chunk_len")
        for ch in chunks:
            if cond_evals[0] >= args.cond_budget:
                break
            c = dict(best_cond)
            c["gpt_cond_chunk_len"] = ch
            r = test_cond(c)
            if r and (best_r is None or r["score"] > best_r["score"]):
                best_r, best_cond = r, dict(c)

        # C3: reference normalization
        print("\n  C3 — sound_norm_refs")
        for sn in [0, 1]:
            if cond_evals[0] >= args.cond_budget:
                break
            c = dict(best_cond)
            c["sound_norm_refs"] = sn
            r = test_cond(c)
            if r and (best_r is None or r["score"] > best_r["score"]):
                best_r, best_cond = r, dict(c)

        # C4: useful reference subsets, only when several same-speaker refs exist
        if args.search_ref_sets and len(all_refs) > 1:
            print("\n  C4 — reference-set search")
            for rs in _reference_sets(all_refs):
                if cond_evals[0] >= args.cond_budget:
                    break
                c = dict(best_cond)
                c["refs"] = tuple(rs)
                r = test_cond(c)
                if r and (best_r is None or r["score"] > best_r["score"]):
                    best_r, best_cond = r, dict(c)

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

    print("\n" + "=" * 72)
    print("  SAMPLING SEARCH")
    print("=" * 72)

    if args.method == "coord":
        if len(seeds) > 1:
            print("\n  Seed screen")
            ss = []
            for s in seeds:
                r = evaluate(s, start, probe_texts, cwin, phase="sample")
                if r:
                    ss.append(r)
            seed = max(ss, key=lambda r: r["score"])["seed"] if ss else seeds[0]
        else:
            seed = seeds[0]

        cur = dict(start)
        base = evaluate(seed, cur, probe_texts, cwin, phase="sample")
        best_score = base["score"] if base else -1.0
        for rnd in range(1, args.rounds + 1):
            improved = False
            for axis in AXIS_ORDER:
                for v in GRID[axis]:
                    if sample_evals[0] >= args.budget:
                        break
                    trial = dict(cur)
                    trial[axis] = v
                    r = evaluate(seed, trial, probe_texts, cwin, phase="sample")
                    if r and r["score"] > best_score:
                        best_score = r["score"]
                        cur[axis] = v
                        improved = True
                if sample_evals[0] >= args.budget:
                    break
            if not improved or sample_evals[0] >= args.budget:
                break
        vals = [r for r in score_cache.values() if cond_key(r["cond"]) == cond_key(cwin)]
        best = max(vals, key=lambda r: r["score"])

    else:
        # Stage S1 — robust seed screen
        rank_mode = "worst-case" if args.seed_robust else "mean"
        print(f"\n  S1 — seed screen ({rank_mode})")
        screen = []
        for s in seeds:
            r = evaluate(s, P(start["temp"]), probe_texts, cwin, phase="sample")
            if r:
                screen.append(r)
        screen.sort(key=lambda r: -(r["worst"] if args.seed_robust else r["score"]))
        keep = [r["seed"] for r in screen[:max(1, args.keep_seeds)]]
        print(f"  kept seeds: {keep}")

        # Stage S2 — temperature surface
        print("\n  S2 — temperature response surface")
        temps = [0.45, 0.55, 0.65, 0.75, 0.85]
        per_seed_best = []
        for s in keep:
            pts = []
            for t in temps:
                r = evaluate(s, P(t), probe_texts, cwin, phase="sample")
                if r:
                    pts.append(r)
            if len(pts) >= 3 and sample_evals[0] < args.budget:
                ts = np.array([r["temp"] for r in pts])
                ss = np.array([r["score"] for r in pts])
                a, b, _c = np.polyfit(ts, ss, 2)
                if a < -1e-6:
                    tstar = float(np.clip(-b / (2 * a), 0.45, 0.85))
                    r = evaluate(s, P(tstar), probe_texts, cwin, phase="sample")
                    if r:
                        pts.append(r)
            if pts:
                per_seed_best.append(max(pts, key=lambda r: r["score"]))

        if not per_seed_best:
            raise SystemExit("[ERR] Sampling budget exhausted before a winner was measured.")
        best = max(per_seed_best, key=lambda r: r["score"])
        seed, tw = best["seed"], best["temp"]

        # Stage S3 — rep_pen + top_p + top_k.
        # top_k is now actually probed too; the old RSM path froze it.
        print("\n  S3 — sampling-axis probe")
        cand = [best]
        for rp in [4.0, 5.0, 7.0, 10.0]:
            r = evaluate(seed, P(tw, rep=rp), probe_texts, cwin, phase="sample")
            if r:
                cand.append(r)
            if sample_evals[0] >= args.budget:
                break
        for tp in [0.80, 0.85, 0.90]:
            if sample_evals[0] >= args.budget:
                break
            r = evaluate(seed, P(tw, rep=best["rep_pen"], tp=tp),
                         probe_texts, cwin, phase="sample")
            if r:
                cand.append(r)
        for tk in [30, 50, 70]:
            if sample_evals[0] >= args.budget:
                break
            r = evaluate(seed, P(tw, rep=best["rep_pen"],
                                 tp=best["top_p"], tk=tk),
                         probe_texts, cwin, phase="sample")
            if r:
                cand.append(r)
        best = max(cand, key=lambda r: r["score"])

        # Stage S4 — optional deterministic decode + length_penalty search.
        if args.probe_beams and sample_evals[0] < args.budget:
            print("\n  S4 — beam/greedy + length_penalty")
            bw = max(2, int(args.beam_width))
            decode_cands = [best]

            # Greedy
            pg = dict(P(best["temp"], rep=best["rep_pen"],
                        tp=best["top_p"], tk=best["top_k"]),
                      num_beams=1, do_sample=False)
            try:
                r = evaluate(best["seed"], pg, probe_texts, cwin, phase="sample")
                if r:
                    decode_cands.append(r)
            except Exception as e:
                print(f"  greedy skipped: {type(e).__name__}: {e}")

            # Beam: len_pen is meaningful here.
            for lp in [0.8, 0.9, 1.0, 1.1, 1.2]:
                if sample_evals[0] >= args.budget:
                    break
                pb = dict(P(best["temp"], rep=best["rep_pen"],
                            tp=best["top_p"], tk=best["top_k"], lp=lp),
                          num_beams=bw, do_sample=False)
                try:
                    r = evaluate(best["seed"], pb, probe_texts, cwin, phase="sample")
                    if r:
                        decode_cands.append(r)
                except Exception as e:
                    msg = str(e).lower()
                    if "out of memory" in msg:
                        print(f"  beam({bw}) skipped: not enough VRAM")
                        break
                    print(f"  beam({bw}) lp={lp}: skipped ({type(e).__name__}: {e})")
            best = max(decode_cands, key=lambda r: r["score"])

    # ------------------------------------------------------------------
    # Ranking and hold-out
    # ------------------------------------------------------------------
    candidates = [
        r for r in score_cache.values()
        if cond_key(r["cond"]) == cond_key(cwin)
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

    print(f"\n  Best on search: {best['score']:.3f} "
          f"(accent {best['french']:.3f}, identity {best['identity']:.3f})")

    held = None
    if holdout_texts:
        print(f"\n{'-'*72}")
        print(f"  HOLD-OUT VALIDATION ({len(holdout_texts)} unseen sentence(s))")
        print("-" * 72)

        # Hold-out should never be starved by the normal sampling budget.
        old_budget = args.budget
        args.budget += len(holdout_texts) + 2
        hp = {
            "temp": best["temp"], "rep_pen": best["rep_pen"],
            "top_k": best["top_k"], "top_p": best["top_p"],
            "len_pen": best["len_pen"],
            "num_beams": best["num_beams"],
            "do_sample": best["do_sample"],
        }
        held = evaluate(best["seed"], hp, holdout_texts, cwin, phase="sample")
        args.budget = old_budget
        if held:
            drop = best["score"] - held["score"]
            noise = max(0.01, np.hypot(best.get("sem", 0), held.get("sem", 0)))
            print(f"  HELD-OUT score {held['score']:.3f} ±{held['sd']:.3f} "
                  f"(accent {held['french']:.3f}, identity {held['identity']:.3f})")
            print(f"  Search -> hold-out drop {drop:+.3f}; noise ~{noise:.3f}")
            if drop > 2 * noise:
                print("  [!] Some overfitting remains; use more probe texts before raising budget.")
            else:
                print("  [OK] Winner generalises within the measured noise.")

    # ------------------------------------------------------------------
    # Final block: conditioning winner + sampling winner.
    # ------------------------------------------------------------------
    win = dict(xtts)
    for a in GRID:
        win[a] = best[a]
    win["len_pen"] = best["len_pen"]
    win["num_beams"] = best["num_beams"]
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

    print("\n  Paste into the Validator/Comparator:")
    print(f"  {format_xtts_block(best['seed'], win)}")
    print("[OK] Done.")


if __name__ == "__main__":
    main()

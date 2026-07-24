#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xtts_optimize.py — search the XTTS sampling parameters against a measured objective.

The sampling knobs (temperature, top_k, top_p, repetition_penalty) do NOT describe
the speaker, so predicting them from the reference acoustics (Praat → temp) has no
causal basis. They must be SEARCHED against something observable on the output.
This is the automated form of the Validator: instead of sweeping by hand, it runs
a coordinate-descent search that maximises

    score = w_accent · french(Whisper)  +  w_identity · cosine(ECAPA)

evaluated on a generated clip. Deterministic for a fixed seed, so no averaging.

Coordinate descent: start from the {} block, optimise one axis at a time over a
small candidate grid (temp → rep_pen → top_p → top_k), repeat for a couple of
rounds. Cheap (~15–25 generations), explainable, dependency-free. Optionally
screens a few seeds first and optimises the best one.

Usage:
  python xtts_optimize.py ref.wav [refs...] FR --xtts-block "{1, 42, ...}" \\
      [--text-file probe.txt] [--seeds "0 42 100"] [--budget 25] \\
      [--w-accent 0.6 --w-identity 0.4] [--max-ref-len 30] [--device cuda]
"""

import os
import re
import argparse
import tempfile
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, SCRIPT_DIR)

LANGS = {'FR','EN','ES','DE','IT','PT','PL','TR','RU','NL','CS','AR','ZH-CN','HU','KO','JA','HI'}

# Candidate grids per axis (impact-ordered). Ranges kept in diction-safe
# territory: rep_pen below ~4 makes XTTS babble/slur, and temp below ~0.55 over-
# flattens — both raise the identity score while wrecking articulation, which the
# accent/identity proxies do not penalise. So we don't let the search go there.
GRID = {
    'temp':    [0.50, 0.55, 0.60, 0.65, 0.70],
    'rep_pen': [4.0, 5.0, 7.0, 10.0],
    'top_p':   [0.80, 0.85, 0.90],
    'top_k':   [30, 50, 70],
}
AXIS_ORDER = ['temp', 'rep_pen', 'top_p', 'top_k']


def parse_xtts_block(block):
    nums = [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?', block)]
    keys = ['seed','trim_start','trim_end','fade_in','fade_out','temp','top_k',
            'top_p','rep_pen','len_pen','gpt_cond_len','gpt_cond_chunk_len','sound_norm_refs']
    out = {}
    for i, k in enumerate(keys):
        if i + 1 < len(nums):
            out[k] = nums[i + 1]
    return out


def fmt(v):
    v = float(v)
    return str(int(v)) if v == int(v) else str(round(v, 3))


def format_xtts_block(seed, p):
    order = [1, seed, int(p.get('trim_start', 0)), int(p.get('trim_end', 0)),
             int(p.get('fade_in', 100)), int(p.get('fade_out', 250)),
             p['temp'], int(p['top_k']), p['top_p'], p['rep_pen'],
             p.get('len_pen', 1.0), int(p.get('gpt_cond_len', 30)),
             int(p.get('gpt_cond_chunk_len', 6)), int(p.get('sound_norm_refs', 0))]
    if int(p.get('num_beams', 1)) != 1:
        order.append(int(p['num_beams']))   # optional 15th field (v24 generator+)
    return '{' + ', '.join(fmt(v) for v in order) + '}'


def read_text(args):
    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, encoding='utf-8') as fh:
            raw = fh.read()
        lines = []
        for ln in raw.splitlines():
            s = ln.strip()
            if not s or s.startswith('{') or s.startswith('['):
                continue
            s = re.sub(r'\[[^\]]*\]', ' ', s).strip()
            if s:
                lines.append(s)
        t = ' '.join(lines).strip()
        if t:
            return t
    return args.text


def main():
    p = argparse.ArgumentParser(description='Optimise XTTS sampling params vs accent+identity')
    p.add_argument('reference')
    p.add_argument('voice_refs', nargs='*')
    p.add_argument('--xtts-block', required=True)
    p.add_argument('--text', default="Bonjour, ceci est une phrase de test pour régler la voix avec soin.")
    p.add_argument('--text-file', default=None)
    p.add_argument('--seeds', default=None, help='Optional seeds to screen, e.g. "0 42 100"')
    p.add_argument('--method', default='rsm', choices=['rsm', 'coord'],
                   help="rsm = seed screen + least-squares response surface on temp "
                        "(matches the problem structure); coord = old coordinate descent")
    p.add_argument('--keep-seeds', type=int, default=3,
                   help='RSM: how many top seeds to refine on temp (default: 3)')
    p.add_argument('--probe-texts', type=int, default=2,
                   help='RSM: sentences averaged per score to cut phonetic noise (1-3, default: 2)')
    p.add_argument('--budget', type=int, default=60, help='Max generations (default: 60)')
    p.add_argument('--w-accent', type=float, default=0.6)
    p.add_argument('--w-identity', type=float, default=0.4)
    p.add_argument('--rounds', type=int, default=2, help='coord only: descent rounds (default: 2)')
    p.add_argument('--holdout-texts', type=int, default=2,
                   help='Sentences reserved for validating the winner (1-2, default: 2). '
                        'Never used during the search — the reported score comes from them.')
    p.add_argument('--no-holdout', action='store_true',
                   help='Skip hold-out validation (faster, but the reported score is '
                        'then the optimised-on score and is optimistically biased)')
    p.add_argument('--seed-robust', action='store_true', default=True,
                   help='Rank seeds on worst-case across sentences instead of the mean, '
                        'so a seed that is great on one phrase and poor on others loses')
    p.add_argument('--seed-mean', dest='seed_robust', action='store_false',
                   help='Rank seeds on the mean score (legacy behaviour)')
    p.add_argument('--probe-beams', action='store_true',
                   help='After the winner is found, probe beam-search decoding '
                        '(num_beams=3, activates length_penalty) and greedy '
                        '(do_sample=False); adopt if the score improves')
    p.add_argument('--max-ref-len', type=int, default=30)
    p.add_argument('--whisper-model', default='small')
    p.add_argument('--whisper-device', default='cpu',
                   help="Device for Whisper accent scoring (default: cpu — frees "
                        "VRAM for XTTS on small cards; use 'cuda' on big GPUs)")
    p.add_argument('--device', default=None)
    args = p.parse_args()

    refs = list(args.voice_refs); lang = 'FR'
    if refs and refs[-1].upper() in LANGS:
        lang = refs[-1].upper(); refs = refs[:-1]
    if not refs:
        refs = [args.reference]
    speaker_wav = refs if len(refs) > 1 else refs[0]

    xtts = parse_xtts_block(args.xtts_block)
    text = read_text(args)
    block_seed = int(xtts.get('seed', 0))
    seeds = [int(s) for s in re.findall(r'-?\d+', args.seeds)] if args.seeds else [block_seed]

    # Probe sentences: varied phonetics so the score reflects the voice, not one
    # sentence's sounds. XTTS is deterministic given (seed, params), so averaging
    # across sentences is the only real variance reduction available.
    # Search sentences vs HELD-OUT sentences. The winner is picked on the search
    # set, then re-scored on sentences it has never seen: that hold-out score is
    # the honest one, and the gap search->holdout measures how much the seed was
    # overfitted to the test phrase (the main methodological hole of a 1-2
    # sentence search).
    PROBES = [
        text,
        "Le soleil se couche doucement derrière les collines lointaines.",
        "Respire profondément et laisse partir toutes les tensions.",
    ]
    HOLDOUT = [
        "Chaque expiration relâche un peu plus les épaules et la mâchoire.",
        "Quarante-huit personnes attendaient déjà sur le quai numéro trois.",
    ]
    n_probe = max(1, min(3, args.probe_texts))
    probe_texts = PROBES[:n_probe]
    holdout_texts = [] if args.no_holdout else HOLDOUT[:max(1, min(2, args.holdout_texts))]

    print("=" * 64)
    print(f"  XTTS Sampling Optimiser — method: {args.method}  (accent + identity)")
    print("=" * 64)
    print(f"  Reference : {os.path.basename(args.reference)}   lang {lang}")
    print(f"  Objective : {args.w_accent:.2f}·french + {args.w_identity:.2f}·identity")
    print(f"  Budget    : {args.budget} gens   seeds {seeds}   probe-texts {max(1, min(3, args.probe_texts))}")
    print("=" * 64)

    import torch
    import xtts_clone as XC
    from pron_score import PronScorer
    from speaker_identity import SpeakerEncoder
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n[*] Loading XTTS on {device}...")
    tts, model = XC.load_xtts(device)
    print(f"[*] Loading faster-whisper '{args.whisper_model}' on {args.whisper_device}...")
    pron = PronScorer(device=args.whisper_device, model=args.whisper_model)
    print("[*] Loading ECAPA-TDNN...")
    enc = SpeakerEncoder(device='cpu' if device == 'cuda' else device)
    ref_emb = enc.embed(args.reference)
    print("[OK] Ready\n")

    tmpdir = tempfile.mkdtemp()
    gpt_cl  = int(xtts.get('gpt_cond_len', 30))
    gpt_ccl = int(xtts.get('gpt_cond_chunk_len', 6))
    snorm   = bool(xtts.get('sound_norm_refs', 0))
    latents = {s: None for s in seeds}
    lat_common = XC.compute_latents(model, speaker_wav, gpt_cond_len=gpt_cl,
                                    gpt_cond_chunk_len=gpt_ccl,
                                    max_ref_len=args.max_ref_len, sound_norm_refs=snorm)

    cache, evals = {}, [0]
    wa, wi = args.w_accent, args.w_identity

    def key(seed, prm, texts):
        # The text SET must be part of the key: keying on len(texts) alone made
        # a hold-out evaluation collide with a search evaluation of the same
        # size, silently returning the cached search result (the hold-out then
        # reported a fake +0.000 drop).
        return (seed, round(prm['temp'], 3), int(prm['top_k']),
                round(prm['top_p'], 3), round(prm['rep_pen'], 2),
                tuple(texts),
                int(prm.get('num_beams', 1)), bool(prm.get('do_sample', True)))

    def evaluate(seed, prm, texts=None):
        """Score = mean over `texts` of (wa*french + wi*identity). Deterministic
        per (seed, params, text), so cached."""
        texts = texts or [text]
        k = key(seed, prm, texts)
        if k in cache:
            return cache[k]
        if evals[0] >= args.budget:
            return None
        frs, ids = [], []
        for ti, txt in enumerate(texts):
            evals[0] += 1
            wav = os.path.join(tmpdir, f"o_{evals[0]}.wav")
            XC.generate(model, txt, lang, lat_common, wav,
                        temperature=prm['temp'], length_penalty=prm.get('len_pen', 1.0),
                        repetition_penalty=prm['rep_pen'], top_k=int(prm['top_k']),
                        top_p=prm['top_p'], speed=1.0, seed=seed,
                        num_beams=int(prm.get('num_beams', 1)),
                        do_sample=bool(prm.get('do_sample', True)))
            pr = pron.score(wav, lang=lang, target_text=txt)
            co = enc.cosine(ref_emb, enc.embed(wav))
            if device == 'cuda':
                try: torch.cuda.empty_cache()
                except Exception: pass
            frs.append(pr['score']); ids.append(co)
        per = [wa * a + wi * b for a, b in zip(frs, ids)]   # score per sentence
        fr = sum(frs) / len(frs); co = sum(ids) / len(ids)
        sc = wa * fr + wi * co
        # Keep the dispersion: a mean alone cannot say whether two candidates
        # differ. sem = standard error of the mean -> the noise floor below
        # which two scores must be called a tie.
        sd = float(np.std(per, ddof=1)) if len(per) > 1 else 0.0
        sem = sd / np.sqrt(len(per)) if len(per) > 1 else 0.0
        rec = dict(score=sc, french=fr, identity=co, wer=None, detected='',
                   seed=seed, num_beams=int(prm.get('num_beams', 1)),
                   per_text=per, sd=sd, sem=sem, worst=min(per),
                   **{a: prm[a] for a in GRID})
        cache[k] = rec
        disp = f" ±{sd:.3f}" if len(per) > 1 else ""
        print(f"  [{evals[0]:>3}] seed={seed:>3} temp={prm['temp']:.2f} rep={prm['rep_pen']:.1f} "
              f"top_p={prm['top_p']:.2f} top_k={int(prm['top_k'])} (n={len(texts)})  ->  "
              f"score={sc:.3f}{disp} (fr={fr:.3f} id={co:.3f})")
        return rec

    def tie(a, b):
        """True when two candidates are statistically indistinguishable:
        their difference is within the combined noise of the measurements."""
        noise = max(0.01, np.hypot(a.get('sem', 0.0), b.get('sem', 0.0)))
        return abs(a['score'] - b['score']) <= noise

    start = {a: float(xtts.get(a, GRID[a][len(GRID[a]) // 2])) for a in GRID}
    start['len_pen'] = float(xtts.get('len_pen', 1.0))

    if args.method == 'coord':
        # ── Legacy coordinate descent ─────────────────────────────────────────
        if len(seeds) > 1:
            print("-" * 64 + "\n  Seed screen (block params)\n" + "-" * 64)
            best_seed, best_sc = seeds[0], -1
            for s in seeds:
                r = evaluate(s, start)
                if r and r['score'] > best_sc:
                    best_sc, best_seed = r['score'], s
            seed = best_seed
            print(f"  -> seed {seed}\n")
        else:
            seed = seeds[0]
        print("-" * 64 + "\n  Coordinate descent\n" + "-" * 64)
        cur = dict(start)
        best_score = (evaluate(seed, cur) or dict(score=-1))['score']
        for rnd in range(1, args.rounds + 1):
            improved = False
            for axis in AXIS_ORDER:
                best_v, best_r = cur[axis], None
                for v in GRID[axis]:
                    trial = dict(cur); trial[axis] = v
                    r = evaluate(seed, trial)
                    if r is None:
                        break
                    if r['score'] > best_score + 1e-6:
                        best_score, best_v, best_r = r['score'], v, r
                if best_r is not None and best_v != cur[axis]:
                    cur[axis] = best_v; improved = True
                    print(f"   round {rnd}: {axis} -> {best_v}  (score {best_score:.3f})")
                if evals[0] >= args.budget:
                    break
            if not improved or evals[0] >= args.budget:
                break
        best = sorted(cache.values(), key=lambda r: -r['score'])[0]

    else:
        # ── RSM: seed screen + least-squares response surface on temp ─────────
        rep0, topk0, topp0 = start['rep_pen'], start['top_k'], start['top_p']
        def P(temp, rep=None, tp=None):
            return {'temp': temp, 'rep_pen': rep0 if rep is None else rep,
                    'top_k': topk0, 'top_p': topp0 if tp is None else tp,
                    'len_pen': start['len_pen']}

        # Stage 1 — seed screen. The seed is chaotic, so we brute-screen it; but
        # screening on ONE sentence overfits the seed to that phrase. We screen
        # on the probe set and rank by WORST-CASE sentence, so a seed that is
        # brilliant once and mediocre twice cannot win.
        rank_mode = 'worst-case' if args.seed_robust else 'mean'
        print("-" * 64 + f"\n  Stage 1 — seed screen ({len(probe_texts)} sentence(s), "
              f"ranked on {rank_mode})\n" + "-" * 64)
        screen = [r for s in seeds for r in [evaluate(s, P(start['temp']), probe_texts)] if r]
        screen.sort(key=lambda r: -(r['worst'] if args.seed_robust else r['score']))
        keep = [r['seed'] for r in screen[:max(1, args.keep_seeds)]]
        for r in screen[:max(1, args.keep_seeds)]:
            print(f"    seed {r['seed']:>3}: mean {r['score']:.3f} ±{r['sd']:.3f}  "
                  f"worst {r['worst']:.3f}")
        print(f"  kept seeds (top {len(keep)}): {keep}")

        # Stage 2 — least-squares parabola of score vs temp, per kept seed.
        # temp is the one smooth continuous axis, so this is where LS belongs.
        print("\n" + "-" * 64 + "\n  Stage 2 — least-squares temp surface (avg of "
              f"{len(probe_texts)} sentences)\n" + "-" * 64)
        temps = [0.45, 0.55, 0.65, 0.75, 0.85]
        per_seed_best = []
        for s in keep:
            pts = [r for t in temps for r in [evaluate(s, P(t), probe_texts)] if r]
            if len(pts) >= 3:
                ts = np.array([r['temp'] for r in pts])
                ss = np.array([r['score'] for r in pts])
                a, b, c = np.polyfit(ts, ss, 2)          # least-squares quadratic
                if a < -1e-6:                            # concave => interior maximum
                    t_star = float(np.clip(-b / (2 * a), 0.45, 0.85))
                    print(f"  seed {s}: LS vertex temp* = {t_star:.3f}")
                    rs = evaluate(s, P(t_star), probe_texts)
                    if rs:
                        pts.append(rs)
                else:
                    print(f"  seed {s}: surface not concave -> best sampled")
            bs = max(pts, key=lambda r: r['score'])
            per_seed_best.append(bs)
            print(f"  seed {s}: best temp {bs['temp']:.3f}  score {bs['score']:.3f} "
                  f"(fr {bs['french']:.3f} id {bs['identity']:.3f})")
        best = max(per_seed_best, key=lambda r: r['score'])
        seed = best['seed']; tw = best['temp']

        # Stage 3 — inertness probe: confirm rep_pen / top_p don't move the output
        # here (your runs showed they often don't); adopt any that genuinely helps.
        print("\n" + "-" * 64 + "\n  Stage 3 — inertness probe (rep_pen, top_p)\n" + "-" * 64)
        base = best['score']; probes = []
        for rp in [4.0, 10.0]:
            r = evaluate(seed, P(tw, rep=rp), probe_texts)
            if r: probes.append((('rep_pen', rp), r))
        for tp in [0.80, 0.90]:
            r = evaluate(seed, P(tw, tp=tp), probe_texts)
            if r: probes.append((('top_p', tp), r))
        spread = max((abs(r['score'] - base) for _, r in probes), default=0.0)
        # Compare against the MEASURED noise (standard error), not an arbitrary
        # constant: a difference smaller than the measurement noise is not a
        # difference. This is what keeps the optimiser from chasing 0.01s.
        (axval, rbest) = (max(probes, key=lambda pr: pr[1]['score'])
                          if probes else ((None, None), None))
        if rbest is None or tie(rbest, best):
            noise = max(0.01, np.hypot(best.get('sem', 0.0),
                                       rbest.get('sem', 0.0) if rbest else 0.0))
            print(f"  rep_pen/top_p inert here (max Δ {spread:.3f} <= noise {noise:.3f}) "
                  f"-> frozen at rep_pen={rep0}, top_p={topp0}")
        else:
            print(f"  sensitivity {spread:.3f} (above noise); best probe "
                  f"{axval[0]}={axval[1]} score {rbest['score']:.3f}")
            if rbest['score'] > best['score']:
                best = rbest

        # ── Stage 4 (optional) — decode-mode probe: beam search & greedy ──────
        # num_beams>1 switches the GPT decode to beam search (length_penalty
        # finally becomes active); do_sample=False is greedy. Both are decoding
        # modes XTTS exposes but nobody sweeps. Scores don't hear naturalness:
        # LISTEN before adopting a beam/greedy winner.
        if args.probe_beams:
            print("\n" + "-" * 64 + "\n  Stage 4 — decode-mode probe (beam search / greedy)\n" + "-" * 64)
            b0 = best['score']
            pb = dict(P(best['temp']), num_beams=3)
            r = evaluate(best['seed'], pb, probe_texts)
            if r:
                print(f"  beam(3): score {r['score']:.3f} (fr {r['french']:.3f} id {r['identity']:.3f})")
                if r['score'] > b0 + 1e-6:
                    best = r
            pg = dict(P(best['temp']), do_sample=False)
            rg = evaluate(best['seed'], pg, probe_texts)
            if rg:
                print(f"  greedy : score {rg['score']:.3f} (fr {rg['french']:.3f} id {rg['identity']:.3f})")
                if rg['score'] > best['score'] + 1e-6:
                    print("  (greedy wins on score — verify by ear, greedy can sound flat)")
                    best = rg

        # Pareto front: french vs identity, so you can pick the tradeoff yourself
        vals = list(cache.values())
        def dom(r):
            return any(o['french'] >= r['french'] and o['identity'] >= r['identity']
                       and (o['french'] > r['french'] or o['identity'] > r['identity'])
                       for o in vals)
        pareto = sorted([r for r in vals if not dom(r)], key=lambda r: -r['identity'])
        print(f"\n{'='*64}\n  PARETO FRONT (accent vs identity — pick your tradeoff)\n{'='*64}")
        print(f"  {'french':>8}{'ident':>7}{'  seed':>6}{'temp':>6}")
        for r in pareto[:10]:
            print(f"  {r['french']:>8.3f}{r['identity']:>7.3f}{r['seed']:>6}{r['temp']:>6.2f}")

    # ── Shared report ─────────────────────────────────────────────────────────
    ranked = sorted(cache.values(), key=lambda r: -r['score'])
    print(f"\n{'='*64}\n  TOP RESULTS ({evals[0]} generations)\n{'='*64}")
    print(f"  {'score':>6}{'sd':>7}{'french':>8}{'ident':>7}{'  seed':>6}{'temp':>6}"
          f"{'rep':>5}{'top_p':>7}{'top_k':>6}")
    top = ranked[0]
    for r in ranked[:8]:
        mark = ' =' if (r is not top and tie(r, top)) else '  '
        print(f"  {r['score']:>6.3f}{r['sd']:>7.3f}{r['french']:>8.3f}{r['identity']:>7.3f}"
              f"{r['seed']:>6}{r['temp']:>6.2f}{r['rep_pen']:>5.1f}{r['top_p']:>7.2f}"
              f"{int(r['top_k']):>6}{mark}")
    n_tied = sum(1 for r in ranked[:8] if r is not top and tie(r, top))
    if n_tied:
        print(f"  ('=' marks the {n_tied} candidate(s) statistically indistinguishable "
              f"from the best — the ranking between them is noise, pick by ear)")

    # Winner block: inherit ALL non-searched fields (trim/fade/gpt_cond_len/...)
    # from the input {} — only the searched axes are overwritten.
    win = dict(xtts); win.update({a: best[a] for a in GRID})
    win['num_beams'] = int(best.get('num_beams', 1))
    print(f"\n  Best on the search sentences: score {best['score']:.3f}  "
          f"(french {best['french']:.3f}, identity {best['identity']:.3f})")

    # ── Hold-out validation: sentences never used during the search ───────────
    # Validate the winner AND every candidate that tied with it: their ranking
    # on the search sentences is noise, so the tie-break that matters is which
    # one still holds up on unseen text. That is also what makes the overfitting
    # warning actionable — otherwise it points at candidates nobody measured.
    if holdout_texts:
        cands = [best] + [r for r in ranked[:8]
                          if r is not best and tie(r, best)][:3]
        print(f"\n{'-'*64}\n  Hold-out validation ({len(holdout_texts)} unseen sentence(s), "
              f"{len(cands)} candidate(s))\n{'-'*64}")
        args.budget += len(holdout_texts) * (len(cands) + 1) + 2   # never starve it
        results = []
        for c in cands:
            pw = {'temp': c['temp'], 'rep_pen': c['rep_pen'], 'top_k': c['top_k'],
                  'top_p': c['top_p'], 'len_pen': start['len_pen'],
                  'num_beams': int(c.get('num_beams', 1))}
            hv = evaluate(c['seed'], pw, holdout_texts)
            if hv:
                results.append((c, hv))
        if results:
            print(f"\n  {'seed':>5}{'temp':>6}{'search':>9}{'held-out':>10}{'drop':>8}")
            for c, hv in results:
                print(f"  {c['seed']:>5}{c['temp']:>6.2f}{c['score']:>9.3f}"
                      f"{hv['score']:>10.3f}{c['score'] - hv['score']:>+8.3f}")
            # Pick on the HELD-OUT score: that is the unbiased estimate.
            c, hv = max(results, key=lambda p: p[1]['score'])
            drop = c['score'] - hv['score']
            # Compare the drop to the MEASURED noise, not to a magic constant.
            noise = max(0.01, np.hypot(c.get('sem', 0.0), hv.get('sem', 0.0)))
            if c is not best:
                print(f"\n  Winner CHANGED: seed {c['seed']} temp {c['temp']:.2f} "
                      f"generalises better than seed {best['seed']} temp {best['temp']:.2f}.")
                best = c
                win = dict(xtts); win.update({a: best[a] for a in GRID})
                win['num_beams'] = int(best.get('num_beams', 1))
            print(f"\n  HELD-OUT score {hv['score']:.3f} ±{hv['sd']:.3f}  "
                  f"(french {hv['french']:.3f}, identity {hv['identity']:.3f})"
                  f"  <- the honest figures")
            if drop > 2 * noise:
                print(f"  [!] Drops {drop:.3f} (noise {noise:.3f}): still partly overfitted "
                      f"to the search sentences.")
                print(f"  [!] Re-run with --probe-texts 3 for a more robust pick.")
            elif drop > noise:
                print(f"  Drop {drop:+.3f} is around the measurement noise ({noise:.3f}) "
                      f"— weak evidence of overfitting.")
            else:
                print(f"  Generalises well (drop {drop:+.3f} within noise {noise:.3f}).")

    print(f"  Paste into the Validator/Comparator:")
    print(f"  {format_xtts_block(best['seed'], win)}")
    print("[OK] Done.")


if __name__ == '__main__':
    main()

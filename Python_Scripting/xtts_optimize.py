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
    p.add_argument('--budget', type=int, default=25, help='Max generations (default: 25)')
    p.add_argument('--w-accent', type=float, default=0.6)
    p.add_argument('--w-identity', type=float, default=0.4)
    p.add_argument('--rounds', type=int, default=2, help='Coordinate-descent rounds (default: 2)')
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

    print("=" * 64)
    print("  XTTS Sampling Optimiser — coordinate descent (accent + identity)")
    print("=" * 64)
    print(f"  Reference : {os.path.basename(args.reference)}   lang {lang}")
    print(f"  Objective : {args.w_accent:.2f}·french + {args.w_identity:.2f}·identity")
    print(f"  Budget    : {args.budget} generations   seeds {seeds}")
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

    def key(seed, prm):
        return (seed, round(prm['temp'], 3), int(prm['top_k']),
                round(prm['top_p'], 3), round(prm['rep_pen'], 2))

    def evaluate(seed, prm):
        k = key(seed, prm)
        if k in cache:
            return cache[k]
        if evals[0] >= args.budget:
            return None
        evals[0] += 1
        wav = os.path.join(tmpdir, f"o_{evals[0]}.wav")
        XC.generate(model, text, lang, lat_common, wav,
                    temperature=prm['temp'], length_penalty=prm.get('len_pen', 1.0),
                    repetition_penalty=prm['rep_pen'], top_k=int(prm['top_k']),
                    top_p=prm['top_p'], speed=1.0, seed=seed)
        pr = pron.score(wav, lang=lang, target_text=text)
        co = enc.cosine(ref_emb, enc.embed(wav))
        if device == 'cuda':
            try: torch.cuda.empty_cache()
            except Exception: pass
        sc = wa * pr['score'] + wi * co
        rec = dict(score=sc, french=pr['score'], identity=co, wer=pr['wer'],
                   detected=pr['detected'], seed=seed, **{a: prm[a] for a in GRID})
        cache[k] = rec
        print(f"  [{evals[0]:>2}] seed={seed} temp={prm['temp']:.2f} rep={prm['rep_pen']:.1f} "
              f"top_p={prm['top_p']:.2f} top_k={int(prm['top_k'])}  ->  "
              f"score={sc:.3f} (fr={pr['score']:.3f} id={co:.3f})")
        return rec

    # ── Optional seed screen at block params ──────────────────────────────────
    start = {a: float(xtts.get(a, GRID[a][len(GRID[a]) // 2])) for a in GRID}
    start['len_pen'] = float(xtts.get('len_pen', 1.0))
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

    # ── Coordinate descent ────────────────────────────────────────────────────
    print("-" * 64 + "\n  Coordinate descent\n" + "-" * 64)
    cur = dict(start)
    base = evaluate(seed, cur) or dict(score=-1)
    best_score = base['score']
    for rnd in range(1, args.rounds + 1):
        improved = False
        for axis in AXIS_ORDER:
            cands = GRID[axis]
            best_v, best_r = cur[axis], None
            for v in cands:
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

    # ── Report ────────────────────────────────────────────────────────────────
    ranked = sorted(cache.values(), key=lambda r: -r['score'])
    print(f"\n{'='*64}\n  TOP RESULTS ({evals[0]} generations)\n{'='*64}")
    print(f"  {'score':>6}{'french':>8}{'ident':>7}{'  seed':>6}{'temp':>6}{'rep':>5}{'top_p':>7}{'top_k':>6}")
    for r in ranked[:8]:
        print(f"  {r['score']:>6.3f}{r['french']:>8.3f}{r['identity']:>7.3f}{r['seed']:>6}"
              f"{r['temp']:>6.2f}{r['rep_pen']:>5.1f}{r['top_p']:>7.2f}{int(r['top_k']):>6}")

    best = ranked[0]
    win = dict(start); win.update({a: best[a] for a in GRID})
    print(f"\n  Best: score {best['score']:.3f}  (french {best['french']:.3f}, "
          f"identity {best['identity']:.3f})")
    print(f"  Paste into the Validator/Comparator:")
    print(f"  {format_xtts_block(best['seed'], win)}")
    print("[OK] Done.")


if __name__ == '__main__':
    main()

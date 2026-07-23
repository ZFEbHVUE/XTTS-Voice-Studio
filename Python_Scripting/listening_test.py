#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
listening_test.py — blind A/B listening test with a binomial significance test.

Everything else in this toolkit measures PROXIES (ECAPA cosine, ASR score).
They cannot hear diction, naturalness, or "is this really her" — the two
failures this project hit (rep_pen=2 scoring well while sounding broken; clones
at 0.72 that nobody recognises). This is the missing subjective evaluation, in
the cheapest defensible form: forced-choice A/B, order RANDOMISED and labels
HIDDEN, repeated N times, concluded with a two-sided sign test.

Blinding matters: knowing which file is "the new one" biases the answer. Here
you only hear A and B, in a random order each trial.

Usage:
  python listening_test.py candidate1.wav candidate2.wav [--trials 12] \\
      [--reference real_voice.wav] [--player ffplay]

  # example: does RVC actually beat the raw XTTS clone?
  python listening_test.py Lea_pipeline_clone.wav Lea_pipeline_clone_rvc.wav \\
      --reference Lea_curated.wav --trials 12

Report: preference counts, the binomial p-value, and a plain verdict
("X preferred, p=0.02 — significant" / "not conclusive at this sample size").
"""

import os
import sys
import random
import argparse
import subprocess
from math import comb


def sign_test_p(k, n):
    """Two-sided exact binomial p-value under H0: p = 0.5 (no preference)."""
    if n == 0:
        return 1.0
    k = max(k, n - k)                      # count of the more-preferred option
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def find_player(explicit=None):
    for p in ([explicit] if explicit else []) + ['ffplay', 'paplay', 'aplay', 'afplay']:
        if p and __import__('shutil').which(p):
            return p
    return None


def play(player, path):
    if player == 'ffplay':
        cmd = [player, '-nodisp', '-autoexit', '-loglevel', 'quiet', path]
    else:
        cmd = [player, path]
    subprocess.run(cmd)


def main():
    p = argparse.ArgumentParser(description='Blind A/B listening test with a sign test')
    p.add_argument('file_a')
    p.add_argument('file_b')
    p.add_argument('--trials', type=int, default=12,
                   help='Number of A/B trials (default: 12; below ~10 nothing can '
                        'reach significance)')
    p.add_argument('--reference', default=None,
                   help='Real voice, played before each trial as the anchor')
    p.add_argument('--player', default=None, help='ffplay / paplay / aplay')
    p.add_argument('--question', default=None,
                   help='What to judge (default: closeness to the reference, or '
                        'overall quality when no reference is given)')
    args = p.parse_args()

    for f in (args.file_a, args.file_b, args.reference):
        if f and not os.path.exists(f):
            sys.exit(f"[ERR] Not found: {f}")
    player = find_player(args.player)
    if player is None:
        sys.exit("[ERR] No audio player found — install ffmpeg (ffplay) or pass --player")

    question = args.question or (
        "Which one sounds MORE LIKE the reference voice?" if args.reference
        else "Which one sounds better overall?")

    print("=" * 64)
    print("  Blind A/B listening test")
    print("=" * 64)
    print(f"  Trials    : {args.trials}   (order randomised, labels hidden)")
    print(f"  Question  : {question}")
    print("  Answer 'a' or 'b'; 'r' replays, 'x' skips, 'q' quits early.\n")

    counts = {args.file_a: 0, args.file_b: 0}
    done = 0
    rng = random.Random()
    for t in range(1, args.trials + 1):
        pair = [args.file_a, args.file_b]
        rng.shuffle(pair)                        # blinding: A/B swapped at random
        while True:
            print(f"  --- trial {t}/{args.trials} ---")
            if args.reference:
                print("  [reference]"); play(player, args.reference)
            print("  [A]"); play(player, pair[0])
            print("  [B]"); play(player, pair[1])
            ans = input("  a / b / r(eplay) / x(skip) / q(uit) > ").strip().lower()
            if ans == 'r':
                continue
            break
        if ans == 'q':
            break
        if ans == 'x':
            continue
        if ans in ('a', 'b'):
            counts[pair[0 if ans == 'a' else 1]] += 1
            done += 1

    print(f"\n{'='*64}\n  RESULT ({done} scored trials)\n{'='*64}")
    ka, kb = counts[args.file_a], counts[args.file_b]
    print(f"  {os.path.basename(args.file_a)} : {ka}")
    print(f"  {os.path.basename(args.file_b)} : {kb}")
    if done == 0:
        print("  No trials scored."); return
    pv = sign_test_p(ka, done)
    winner = args.file_a if ka > kb else args.file_b
    print(f"  two-sided sign test: p = {pv:.3f}")
    if ka == kb:
        print("  VERDICT: perfect tie — no audible preference.")
    elif pv < 0.05:
        print(f"  VERDICT: {os.path.basename(winner)} is preferred, significant (p<0.05).")
    else:
        print(f"  VERDICT: {os.path.basename(winner)} leads but NOT conclusive at this "
              f"sample size.")
        need = 0
        for n in range(done, 61):
            if sign_test_p(int(round(n * (max(ka, kb) / done))), n) < 0.05:
                need = n; break
        if need:
            print(f"  At the same preference rate, ~{need} trials would be needed.")
    print("\n  Note: this is YOUR ear. For 'is it really her', run it again with "
          "someone\n  who knows the voice — that is the only test the metrics cannot do.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_rvc_dataset.py — build an RVC training dataset from raw reference audio.

RVC (the timbre post-conversion stage that lifts cloning past XTTS's identity
ceiling) trains best on 10+ minutes of clean, single-speaker utterances of a
few seconds each. This tool bridges from your raw reference to that format:

  1. ECAPA coherence selection (same engine as curate_reference, but keeping
     MINUTES instead of ~45 s): drops noise, breaths, off-voice regions.
  2. Splits the kept audio into utterances on silences (3–10 s each).
  3. Resamples to the RVC training rate (default 40 kHz) mono, light peak
     normalisation, writes dataset/NNN.wav + a summary.

Usage:
  python prepare_rvc_dataset.py Lea_voice.wav -o rvc_dataset_lea \\
      [--keep-minutes 12] [--sr 40000] [--min-utt 3] [--max-utt 10] [--device cuda]

Then point Applio's training at the output folder (see docs/RVC_GUIDE.md).
"""

import os
import argparse
import numpy as np

def _resample(y, sr, target):
    if sr == target:
        return y.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(sr), int(target))
    return resample_poly(y, target // g, sr // g).astype(np.float32)


def split_utterances(y, sr, min_utt=3.0, max_utt=10.0, top_db=35):
    """Split on silences into utterances between min_utt and max_utt seconds."""
    import librosa
    intervals = librosa.effects.split(y, top_db=top_db)
    # Merge short gaps, then pack into 3–10 s utterances
    utts, cur_s, cur_e = [], None, None
    for s, e in intervals:
        if cur_s is None:
            cur_s, cur_e = s, e
            continue
        if (e - cur_s) / sr <= max_utt:
            cur_e = e                       # extend current utterance
        else:
            utts.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    if cur_s is not None:
        utts.append((cur_s, cur_e))
    # Enforce bounds: drop too-short, hard-split too-long
    out = []
    for s, e in utts:
        dur = (e - s) / sr
        if dur < min_utt:
            continue
        if dur <= max_utt:
            out.append((s, e))
        else:
            n = int(np.ceil(dur / max_utt))
            step = (e - s) // n
            for k in range(n):
                a, b = s + k * step, min(e, s + (k + 1) * step)
                if (b - a) / sr >= min_utt:
                    out.append((a, b))
    return out


def main():
    p = argparse.ArgumentParser(description='Build an RVC training dataset from reference audio')
    p.add_argument('refs', nargs='+', help='Raw reference WAV/MP3 file(s)')
    p.add_argument('-o', '--out-dir', required=True, help='Output dataset folder')
    p.add_argument('--keep-minutes', type=float, default=12.0,
                   help='Minutes of most-coherent audio to keep (default: 12; '
                        'RVC wants 10+ for a solid model)')
    p.add_argument('--sr', type=int, default=40000,
                   help='Dataset sample rate (default: 40000 — RVC v2 40k)')
    p.add_argument('--min-utt', type=float, default=3.0)
    p.add_argument('--max-utt', type=float, default=10.0)
    p.add_argument('--top-db', type=float, default=35.0,
                   help='Silence threshold for utterance splitting (default: 35)')
    p.add_argument('--no-curate', action='store_true',
                   help='Skip ECAPA coherence selection (use the raw audio as-is)')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    import soundfile as sf
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 64)
    print("  RVC dataset builder")
    print("=" * 64)

    # ── 1. Coherence selection (minutes-scale curation) ──────────────────────
    if args.no_curate:
        sigs, sr0 = [], None
        for r in args.refs:
            y, sr = sf.read(r)
            if y.ndim > 1:
                y = y.mean(axis=1)
            if sr0 is None:
                sr0 = sr
            elif sr != sr0:
                y = _resample(y, sr, sr0)
            sigs.append(y.astype(np.float32))
        clean = np.concatenate(sigs)
        print(f"[*] Curation skipped — {len(clean)/sr0/60:.1f} min raw")
    else:
        from curate_reference import curate
        tmp = os.path.join(args.out_dir, '_curated_long.wav')
        curate(args.refs, tmp, keep_seconds=args.keep_minutes * 60.0,
               device=args.device)
        clean, sr0 = sf.read(tmp)
        if clean.ndim > 1:
            clean = clean.mean(axis=1)
        clean = clean.astype(np.float32)

    kept_min = len(clean) / sr0 / 60
    if kept_min < 5:
        print(f"[!] Only {kept_min:.1f} min of coherent audio — RVC will underfit.")
        print(f"[!] Aim for 10+ min: find more source material for this voice.")

    # ── 2. Utterance split + 3. export at RVC rate ────────────────────────────
    utts = split_utterances(clean, sr0, args.min_utt, args.max_utt, args.top_db)
    if not utts:
        raise SystemExit("[ERR] No utterances found — lower --top-db or --min-utt.")

    total = 0.0
    for k, (s, e) in enumerate(utts, 1):
        seg = clean[s:e]
        seg = _resample(seg, sr0, args.sr)
        peak = float(np.max(np.abs(seg)) + 1e-9)
        if peak > 0.95:
            seg = seg * (0.95 / peak)       # light peak safety only — no processing
        sf.write(os.path.join(args.out_dir, f"{k:04d}.wav"), seg, args.sr)
        total += len(seg) / args.sr

    print(f"\n[OK] Dataset: {len(utts)} utterances, {total/60:.1f} min "
          f"@ {args.sr} Hz mono -> {args.out_dir}/")
    print("     Next: train in Applio on this folder (see docs/RVC_GUIDE.md).")


if __name__ == '__main__':
    main()

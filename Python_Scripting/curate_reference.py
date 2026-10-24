#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
curate_reference.py — build a clean XTTS reference by ECAPA embedding coherence.

Instead of gambling on the seed, optimise the INPUT to cloning. Long or imperfect
reference audio often contains segments that drag the speaker embedding off-target
— breaths, laughter, reverberant tails, a second voice, noise. XTTS averages the
embedding over the whole reference, so those segments dilute the identity.

This tool windows the reference, embeds each window with ECAPA-TDNN, computes a
robust centroid (trimmed to reject outliers), scores each window by cosine to that
centroid, and concatenates the most coherent windows (up to a target duration)
into a curated reference WAV. Coherence is not a noise/music/second-speaker
detector: review the selected audio by ear. Feed it to the Validator / Comparator.

Usage:
  python curate_reference.py ref1.wav [ref2.wav ...] -o curated.wav \\
      [--keep-seconds 45] [--window 4] [--hop 2] [--min-score auto] [--device cpu]
"""

import os
import argparse
import numpy as np
from audio_checks import mono_audio

SR16 = 16000


def _load_mono(path):
    import soundfile as sf
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return mono_audio(y, sr).astype(np.float32), sr


def _resample(y, sr, target):
    y = mono_audio(y, sr)
    if not isinstance(target, (int, np.integer)) or target <= 0:
        raise ValueError("Target sample rate must be a positive integer")
    if sr == target:
        return y.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(sr), int(target))
    return resample_poly(y, target // g, sr // g).astype(np.float32)


def _voiced_fraction(seg16, thr_ratio=0.06):
    """Energy-active frames; this gate cannot distinguish speech from noise/music."""
    fl = 512
    if len(seg16) < fl:
        return 0.0
    n = len(seg16) // fl
    frames = seg16[:n * fl].reshape(n, fl)
    if not np.isfinite(frames).all():
        return 0.0
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    gate = max(float(rms.max()) * thr_ratio, 1e-5)
    return float((rms > gate).mean())


def curate(refs, out_path, keep_seconds=45.0, window=4.0, hop=2.0,
           min_score='auto', device=None, verbose=True):
    if not refs:
        raise ValueError("At least one reference is required")
    for name, value in (("keep_seconds", keep_seconds), ("window", window), ("hop", hop)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if hop > window:
        raise ValueError("hop must be <= window to avoid unexamined gaps")
    if min_score != 'auto' and (not np.isfinite(float(min_score)) or not -1 <= float(min_score) <= 1):
        raise ValueError("min_score must be auto or a cosine in [-1, 1]")
    for path in refs:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        if os.path.realpath(path) == os.path.realpath(out_path):
            raise ValueError("Output must differ from every reference")

    # Concatenate all references (original sr from the first file)
    sigs, sr0, boundaries, offset = [], None, [], 0
    for r in refs:
        y, sr = _load_mono(r)
        if sr0 is None:
            sr0 = sr
        elif sr != sr0:
            y = _resample(y, sr, sr0); sr = sr0
        sigs.append(y)
        boundaries.append((offset, offset + len(y)))
        offset += len(y)
    orig = np.concatenate(sigs)
    sig16 = _resample(orig, sr0, SR16)

    win_o, hop_o = int(window * sr0), int(hop * sr0)
    win_16       = int(window * SR16)
    if min(win_o, hop_o, win_16) < 1:
        raise ValueError("Window and hop must cover at least one sample")
    from speaker_identity import SpeakerEncoder
    enc = SpeakerEncoder(device=device)

    # ── Window + embed ────────────────────────────────────────────────────────
    wins = []   # (start_orig, end_orig, emb, voiced)
    for begin, end in boundaries:
        if end - begin < win_o:
            continue
        starts = list(range(begin, end - win_o + 1, hop_o))
        if starts[-1] != end - win_o:
            starts.append(end - win_o)
        for pos in starts:
            s16 = round(pos / sr0 * SR16)
            seg16 = sig16[s16:s16 + win_16]
            vf = _voiced_fraction(seg16)
            if vf >= 0.5:
                emb = enc.embed(seg16, sr=SR16)
                wins.append([pos, pos + win_o, emb, vf])

    if len(wins) < 2:
        raise RuntimeError("Not enough voiced material to curate "
                           f"({len(wins)} window(s)). Lower --window or check the file.")

    embs = np.stack([w[2] for w in wins])

    # ── Robust centroid (trim worst 25%, recompute) ──────────────────────────
    def centroid(E):
        c = E.mean(axis=0); return c / (np.linalg.norm(c) + 1e-12)
    c0 = centroid(embs)
    cos0 = embs @ c0
    keep = cos0 >= np.percentile(cos0, 25)
    cref = centroid(embs[keep]) if keep.sum() >= 2 else c0
    cos = embs @ cref   # coherence score per window

    for w, s in zip(wins, cos):
        w[3] = float(s)

    # ── Selection threshold ───────────────────────────────────────────────────
    if min_score == 'auto':
        med = float(np.median(cos))
        mad = float(np.median(np.abs(cos - med))) + 1e-9
        thr = med - 1.0 * 1.4826 * mad      # drop clear outliers only
    else:
        thr = float(min_score)

    order = np.argsort(-cos)                  # best first
    # Windows overlap (hop < window), so naive concatenation would DUPLICATE the
    # overlapped audio. Select by net (union) contribution and merge intervals.
    def _merge(iv):
        iv = sorted(iv)
        out = [list(iv[0])]
        for s, e in iv[1:]:
            if s <= out[-1][1]:
                out[-1][1] = max(out[-1][1], e)
            else:
                out.append([s, e])
        return out

    # Segments are joined with a 10 ms crossfade, which overlaps them by xf
    # samples: the selection has to count that loss. It used to stop as soon as
    # the segments ADDED UP to keep_seconds, and windows on a 2 s grid often add
    # up exactly, so an even Keep s came out short by 10 ms per join (29.95 s
    # for 30): the analyser then started gpt_cond_len at 29 instead of 30.
    xf = int(0.01 * sr0)
    chosen, unions, total = [], [], 0.0
    for idx in order:
        if cos[idx] < thr:
            break
        s, e = wins[idx][0], wins[idx][1]
        covered = sum(max(0, min(e, ue) - max(s, us)) for us, ue in unions)
        if (e - s) - covered <= 0:
            continue                          # fully covered already — no new audio
        chosen.append(idx)
        unions = _merge(unions + [[s, e]])
        total = (sum(ue - us for us, ue in unions) - xf * (len(unions) - 1)) / sr0
        if total >= keep_seconds:
            break
    if not chosen:
        raise RuntimeError("No window meets min_score; lower the threshold or check the references")

    # ── Concatenate merged segments with short crossfades ─────────────────────
    out = np.zeros(0, dtype=np.float32)
    for k, (s, e) in enumerate(unions):
        seg = orig[s:e].copy()
        if k > 0 and len(out) >= xf and len(seg) >= xf:
            ramp = np.linspace(0, 1, xf, dtype=np.float32)
            out[-xf:] = out[-xf:] * (1 - ramp) + seg[:xf] * ramp
            seg = seg[xf:]
        out = np.concatenate([out, seg])

    out = out[:max(1, int(keep_seconds * sr0))]
    if not out.size:
        raise RuntimeError("Curation selected no audio")
    total = len(out) / sr0
    import soundfile as sf
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, out, sr0)

    if verbose:
        kept_cos = cos[chosen]
        print(f"  Reference: {len(orig)/sr0:.1f}s @ {sr0} Hz  ->  "
              f"{len(wins)} voiced windows")
        print(f"  Centroid coherence: mean {cos.mean():.3f}  "
              f"min {cos.min():.3f}  max {cos.max():.3f}")
        print(f"  Threshold ({min_score}): {thr:.3f}")
        print(f"  Kept {len(chosen)}/{len(wins)} windows -> {len(unions)} merged "
              f"segment(s), {total:.1f}s net (no duplicated overlap), "
              f"mean coherence {kept_cos.mean():.3f}")
        dropped = [i for i in range(len(wins)) if i not in set(chosen)]
        if dropped:
            worst = sorted(dropped, key=lambda i: cos[i])[:5]
            tags = ', '.join(f"{wins[i][0]/sr0:.0f}s:{cos[i]:.2f}" for i in worst)
            print(f"  Dropped (worst): {tags}")
        print(f"  Curated -> {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description='Curate an XTTS reference by ECAPA coherence')
    p.add_argument('refs', nargs='+', help='Reference WAV file(s)')
    p.add_argument('-o', '--out', required=True, help='Curated output WAV')
    p.add_argument('--keep-seconds', type=float, default=45.0,
                   help='Target duration of the curated reference (default: 45)')
    p.add_argument('--window', type=float, default=4.0, help='Window seconds (default: 4)')
    p.add_argument('--hop', type=float, default=2.0, help='Hop seconds (default: 2)')
    p.add_argument('--min-score', default='auto',
                   help="Coherence threshold: 'auto' (median−MAD) or a float in [-1,1]")
    p.add_argument('--device', default=None, help='cpu or cuda (default: auto)')
    args = p.parse_args()

    for r in args.refs:
        if not os.path.exists(r):
            raise SystemExit(f"[ERR] not found: {r}")

    print("=" * 60)
    print("  Reference curation by ECAPA coherence")
    print("=" * 60)
    curate(args.refs, args.out, keep_seconds=args.keep_seconds,
           window=args.window, hop=args.hop, min_score=args.min_score,
           device=args.device)
    print("[OK] Done.")


if __name__ == '__main__':
    main()

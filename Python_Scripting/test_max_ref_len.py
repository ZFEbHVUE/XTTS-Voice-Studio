#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_max_ref_len.py — does the speaker embedding benefit from more reference?

XTTS builds TWO conditioning objects from the reference:
  * gpt_cond_len   -> GPT conditioning latents (prosody, style)
  * max_ref_len    -> speaker embedding (HiFiGAN speaker encoder) = TIMBRE

The second one is capped at 30 s by XttsConfig and is NOT part of the {} block,
so it has never been searched: giving 45 s of conditioning still builds the
timbre vector from 30 s only. This script measures whether lifting that cap
changes ECAPA identity — the metric it should affect most directly.

Method: one model load, one set of latents per max_ref_len value, the SAME
sentences and the SAME seed everywhere, so the only variable is the cap.
A noise floor is measured first (two halves of one render) so a difference can
be called real or not.

Usage:
  python test_max_ref_len.py Voices_Cloning/Fanny_Ardant_bis_deep_curated.wav FR \\
      --seed 180 --temp 0.85 --gpt-cond-len 45 --values 20 30 45 60
"""

import os
import sys
import argparse
import tempfile


def main():
    p = argparse.ArgumentParser(description='Measure the effect of max_ref_len on identity')
    p.add_argument('reference')
    p.add_argument('lang', nargs='?', default='FR')
    p.add_argument('--values', type=int, nargs='+', default=[20, 30, 45, 60],
                   help='max_ref_len values to compare (default: 20 30 45 60)')
    p.add_argument('--gpt-cond-len', type=int, default=45)
    p.add_argument('--gpt-cond-chunk-len', type=int, default=4)
    p.add_argument('--seed', type=int, default=180)
    p.add_argument('--temp', type=float, default=0.85)
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--top-p', type=float, default=0.85)
    p.add_argument('--rep-pen', type=float, default=5.0)
    p.add_argument('--device', default=None)
    p.add_argument('--keep', default=None,
                   help='Directory to keep the rendered wavs for listening')
    args = p.parse_args()

    import numpy as np
    import soundfile as sf
    import xtts_clone as XC
    from speaker_identity import SpeakerEncoder

    if not os.path.exists(args.reference):
        sys.exit(f"[ERR] reference not found: {args.reference}")

    # Several sentences: one sentence is not a measurement, it is an anecdote.
    TEXTS = [
        "Respire lentement et laisse les épaules redescendre, sans forcer.",
        "Chaque expiration relâche un peu plus la mâchoire et le front.",
        "Installe-toi confortablement et laisse le silence prendre sa place.",
    ]

    device = args.device or ('cuda' if _cuda() else 'cpu')
    print("=" * 66)
    print("  max_ref_len — does a longer speaker embedding help identity?")
    print("=" * 66)
    print(f"  reference     : {os.path.basename(args.reference)}")
    print(f"  fixed         : seed={args.seed} temp={args.temp} "
          f"gpt_cond_len={args.gpt_cond_len}s")
    print(f"  varying       : max_ref_len = {args.values}")
    print(f"  sentences     : {len(TEXTS)}\n")

    ref_dur = sf.info(args.reference).duration
    print(f"  [*] reference is {ref_dur:.1f}s long")
    usable = [v for v in args.values if v <= ref_dur + 1]
    if len(usable) < len(args.values):
        skipped = [v for v in args.values if v not in usable]
        print(f"  [!] skipping {skipped}: longer than the reference itself "
              f"(the cap would do nothing)")
    if not usable:
        sys.exit("[ERR] no usable value — the reference is too short.")

    print(f"  [*] loading XTTS on {device}...")
    # load_xtts returns (TTS api wrapper, raw model) — compute_latents and
    # generate need the RAW model, i.e. the second element.
    _tts, model = XC.load_xtts(device)
    enc = SpeakerEncoder(device=device)
    ref_emb = enc.embed(args.reference)
    print("  [OK] ready\n")

    tmpdir = args.keep or tempfile.mkdtemp(prefix='mrl_')
    os.makedirs(tmpdir, exist_ok=True)

    def render(mrl, idx, text):
        lat = XC.compute_latents(model, args.reference,
                                 gpt_cond_len=args.gpt_cond_len,
                                 gpt_cond_chunk_len=args.gpt_cond_chunk_len,
                                 max_ref_len=mrl, sound_norm_refs=False)
        out = os.path.join(tmpdir, f"mrl{mrl}_{idx}.wav")
        XC.generate(model, text, args.lang, lat, out,
                    temperature=args.temp, top_k=args.top_k, top_p=args.top_p,
                    repetition_penalty=args.rep_pen, seed=args.seed)
        return out

    # ── Noise floor: the two halves of one render are the same voice, so their
    #    spread is this setup's measurement variability.
    first = render(usable[0], 0, TEXTS[0])
    y, sr = sf.read(first)
    if getattr(y, 'ndim', 1) > 1:
        y = y.mean(axis=1)
    h = len(y) // 2
    noise = max(0.004, abs(float(enc.cosine(ref_emb, enc.embed(y[:h], sr=sr))) -
                           float(enc.cosine(ref_emb, enc.embed(y[h:], sr=sr)))) / 2.0)
    print(f"  noise floor: +/-{noise:.4f}  "
          f"(differences below this are not differences)\n")

    results = []
    for mrl in usable:
        vals = []
        for i, t in enumerate(TEXTS):
            w = first if (mrl == usable[0] and i == 0) else render(mrl, i, t)
            vals.append(float(enc.cosine(ref_emb, enc.embed(w))))
        mean = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        results.append((mrl, mean, sd, min(vals)))
        print(f"  max_ref_len {mrl:>3}s : identity {mean:.4f} +/-{sd:.4f}  "
              f"(worst {min(vals):.4f})")

    base = next((m for v, m, _, _ in results if v == 30), results[0][1])
    best_v, best_m, best_sd, _ = max(results, key=lambda r: r[1])
    print(f"\n{'-'*66}")
    gain = best_m - base
    combined = max(noise, best_sd / np.sqrt(len(TEXTS)))
    if best_v == 30 or gain <= combined:
        print(f"  VERDICT: no gain beyond the noise ({gain:+.4f} vs {combined:.4f}).")
        print(f"  The 30 s cap is not what limits identity here — leave it alone.")
    else:
        print(f"  VERDICT: max_ref_len {best_v}s gains {gain:+.4f} over 30s "
              f"(noise {combined:.4f}).")
        print(f"  Worth adding to the {{}} block and searching. Confirm by EAR "
              f"before adopting:")
        print(f"    python listening_test.py {tmpdir}/mrl30_0.wav "
              f"{tmpdir}/mrl{best_v}_0.wav --reference {args.reference} --trials 12")
    if args.keep:
        print(f"\n  renders kept in {tmpdir}/")


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == '__main__':
    main()

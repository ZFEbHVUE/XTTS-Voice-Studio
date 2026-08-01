#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_truncation.py — does this {} block cut the end of LONG sentences?

Why this exists: the optimiser searches on short probe sentences (9-12 words)
and the hold-out validates on short ones too, yet a meditation script is full
of 25-40 word sentences. XTTS is known to emit its end-of-sequence token early
at low temperature, and the effect grows with sentence length — so a block that
scores well on the search can truncate exactly where you use it.

Method: the SAME sentences at several temperatures, everything else frozen.
For each render we compare the transcription with the text we asked for and
report the fraction of the ending that is missing. No listening required to see
truncation: missing words at the end are missing words.

Usage:
  python test_truncation.py Voices_Cloning/anna_curated.wav FR \\
      --seed 180 --temps 0.45 0.55 0.65 0.75 --gpt-cond-len 45
"""

import os
import re
import sys
import argparse
import tempfile


# Deliberately long, in the register of a guided meditation.
LONG_TEXTS = {
    'fr': ["Installe-toi confortablement, laisse tes épaules redescendre le long "
           "du dossier, et prends le temps de sentir l'air qui entre par le nez "
           "avant de ressortir lentement par la bouche.",
           "À chaque expiration, tu peux relâcher un peu plus la mâchoire, le "
           "front, la nuque, et laisser le poids de ton corps se déposer "
           "complètement dans le siège qui te porte."],
    'en': ["Settle in comfortably, let your shoulders come down along the back of "
           "the chair, and take the time to feel the air entering through your "
           "nose before it slowly leaves through your mouth.",
           "With every breath out you can release the jaw a little more, the "
           "forehead, the back of the neck, and let the weight of your body "
           "settle completely into the seat that carries you."],
}


def main():
    p = argparse.ArgumentParser(description='Detect end-of-sentence truncation vs temperature')
    p.add_argument('reference')
    p.add_argument('lang', nargs='?', default='FR')
    p.add_argument('--temps', type=float, nargs='+', default=[0.45, 0.55, 0.65, 0.75])
    p.add_argument('--seed', type=int, default=180)
    p.add_argument('--seeds', type=int, nargs='+', default=None,
                   help='Vary the SEED instead of the temperature (same test, other axis)')
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--top-p', type=float, default=0.85)
    p.add_argument('--rep-pen', type=float, default=5.0)
    p.add_argument('--temp-fixed', type=float, default=0.65,
                   help='Temperature held constant when varying the seed')
    p.add_argument('--gpt-cond-len', type=int, default=45)
    p.add_argument('--trim-start', type=int, default=0,
                   help='trim_start (ms) from the {} block, applied with --audio-block')
    p.add_argument('--trim-end', type=int, default=200,
                   help='trim_end (ms) from the {} block — a prime suspect for cut endings')
    p.add_argument('--fade-in', type=int, default=200)
    p.add_argument('--fade-out', type=int, default=400)
    p.add_argument('--audio-block', default=None, metavar='BLOCK',
                   help='Also apply this [] block after generation (trim/fades/'
                        'speed act there) and measure the RESULT, not the raw wav')
    p.add_argument('--whisper-model', default='small')
    p.add_argument('--device', default=None)
    p.add_argument('--keep', default=None, help='Directory to keep the renders in')
    args = p.parse_args()

    import numpy as np
    import soundfile as sf
    import xtts_clone as XC
    # Transcribe here rather than through PronScorer: its 'detected' field is
    # the detected LANGUAGE, not the text, and truncation can only be seen by
    # comparing the words actually spoken with the words asked for.
    from faster_whisper import WhisperModel

    if not os.path.exists(args.reference):
        sys.exit(f"[ERR] reference not found: {args.reference}")
    key = args.lang.lower().split('-')[0]
    texts = LONG_TEXTS.get(key, LONG_TEXTS['en'])
    if key not in LONG_TEXTS:
        print(f"  [!] No long sentences for '{args.lang}' — using English ones; "
              f"the accent score will not be meaningful, only the LENGTH is.")

    args.vary = 'seed' if args.seeds else 'temp'
    values = args.seeds if args.seeds else args.temps
    device = args.device or ('cuda' if _cuda() else 'cpu')
    print("=" * 68)
    print(f"  End-of-sentence truncation vs {args.vary.upper()}")
    print("=" * 68)
    print(f"  reference : {os.path.basename(args.reference)}")
    _fixed = (f"temp={args.temp_fixed}" if args.seeds else f"seed={args.seed}")
    print(f"  fixed     : {_fixed} top_k={args.top_k} top_p={args.top_p} "
          f"rep_pen={args.rep_pen}")
    print(f"  varying   : {args.vary} = {values}")
    print(f"  sentences : {len(texts)} long ones "
          f"({min(len(t.split()) for t in texts)}-{max(len(t.split()) for t in texts)} words)\n")

    print(f"  [*] loading XTTS on {device}...")
    _tts, model = XC.load_xtts(device)
    lat = XC.compute_latents(model, args.reference, gpt_cond_len=args.gpt_cond_len)
    asr = WhisperModel(args.whisper_model, device='cpu', compute_type='int8')
    print("  [OK] ready\n")

    def transcribe(path):
        segs, _ = asr.transcribe(path, language=key, beam_size=1)
        return ' '.join(s.text for s in segs).strip()

    tmpdir = args.keep or tempfile.mkdtemp(prefix='trunc_')
    os.makedirs(tmpdir, exist_ok=True)

    # Optional: run the [] block through the generator's own process_audio, so
    # the measurement covers what you actually hear.
    audio_cfg = None
    _apply_block = None
    if args.audio_block:
        import importlib.util, glob as _glob
        gens = sorted(_glob.glob(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), 'guided_meditation_generator_v*.py')))
        if not gens:
            print("  [!] No generator found — --audio-block ignored.")
        else:
            _spec = importlib.util.spec_from_file_location('gmg', gens[-1])
            _gmg = importlib.util.module_from_spec(_spec)
            try:
                _spec.loader.exec_module(_gmg)
                _pa = getattr(_gmg, 'process_audio', None)
            except Exception as _e:
                _pa, _gmg = None, None
                print(f"  [!] Generator not importable ({_e}) — --audio-block ignored.")
            if _pa:
                nums = [float(x) for x in re.findall(r'-?[\d.]+', args.audio_block)]
                keys = ['speed', 'volume', 'eq_low', 'eq_mid', 'eq_high',
                        'highpass', 'lowpass', 'noise_reduction', 'compression',
                        'deesser', 'reverb', 'noise_gate', 'pan', 'limiter']
                vals = nums[1:]          # drop the leading voice number
                audio_cfg = {k: (vals[j] if j < len(vals) else 0)
                             for j, k in enumerate(keys)}
                print(f"  [*] audio block applied after generation: "
                      f"speed={audio_cfg['speed']} vol={audio_cfg['volume']:+.0f} "
                      f"hp={audio_cfg['highpass']:.0f} lp={audio_cfg['lowpass']:.0f}")
                print(f"  [*] plus trim {args.trim_start}/{args.trim_end} ms, "
                      f"fades {args.fade_in}/{args.fade_out} ms")

                def _apply_block(src_wav, dst_wav, cfg, xp):
                    # Real signature: process_audio(audio_segment, config,
                    # xtts_params) — trim_start/trim_end and the fades live in
                    # the {} block, so they must be passed too, otherwise the
                    # very steps suspected of cutting the endings never run.
                    from pydub import AudioSegment
                    seg = AudioSegment.from_wav(src_wav)
                    seg = _pa(seg, cfg, xp)
                    seg.export(dst_wav, format='wav')

    def tail_kept(said, asked):
        """Fraction of the ASKED words still present at the end of SAID.
        Truncation shows up as a missing tail, which a global WER dilutes."""
        aw = [w.strip('.,;:!?').lower() for w in asked.split() if w.strip('.,;:!?')]
        sw = {w.strip('.,;:!?').lower() for w in said.split()}
        last = aw[-8:] if len(aw) >= 8 else aw
        return sum(1 for w in last if w in sw) / max(1, len(last))

    # One axis at a time: everything else stays frozen so the difference is
    # attributable.
    rows = []
    for val in values:
        tp = args.temp_fixed if args.seeds else val
        sd = val if args.seeds else args.seed
        durs, keeps, words = [], [], []
        for i, txt in enumerate(texts):
            wav = os.path.join(tmpdir, f"{args.vary}{val}_{i}.wav")
            XC.generate(model, txt, args.lang, lat, wav,
                        temperature=tp, top_k=args.top_k, top_p=args.top_p,
                        repetition_penalty=args.rep_pen, seed=sd)
            # The raw render is only half the chain. If an audio block is
            # given, apply it exactly as the generator does and measure THAT:
            # trim_end, fades and speed all act after generation, and a word
            # lost there is invisible to a test that stops at the raw wav.
            if audio_cfg is not None:
                post = os.path.join(tmpdir, f"{args.vary}{val}_{i}_post.wav")
                _apply_block(wav, post, audio_cfg,
                             {'trim_start': args.trim_start,
                              'trim_end': args.trim_end,
                              'fade_in': args.fade_in,
                              'fade_out': args.fade_out})
                wav = post
            y, sr = sf.read(wav)
            durs.append(len(y) / sr)
            said = transcribe(wav)
            keeps.append(tail_kept(said, txt))
            words.append(len(said.split()) / max(1, len(txt.split())))
        rows.append((float(val), float(np.mean(durs)), float(np.mean(keeps)),
                     float(np.mean(words))))
        print(f"  {args.vary} {float(val):.2f} : {np.mean(durs):6.2f}s   ending kept "
              f"{np.mean(keeps)*100:5.0f}%   words spoken "
              f"{np.mean(words)*100:5.0f}% of asked")

    print(f"\n{'-'*68}")
    # Truncation is a WORD-level fact: words asked for that were not spoken.
    # Duration alone cannot show it — a shorter render may simply be a faster
    # delivery. An earlier version compared each duration to the LONGEST one and
    # flagged everything below 85% of it, which turned a single abnormally slow
    # render into the reference and produced false alarms on complete sentences.
    bad = [r for r in rows if r[2] < 0.75 or r[3] < 0.85]
    med = sorted(r[1] for r in rows)[len(rows) // 2]
    if bad:
        print("  TRUNCATION at: " + ', '.join(f"{args.vary} {r[0]:.2f}" for r in bad))
        ok = [r for r in rows if r not in bad]
        if ok:
            print("  Complete at : " + ', '.join(f"{args.vary} {r[0]:.2f}" for r in ok))
        print("  A short-sentence search cannot see this — add a long sentence to")
        print("  the probe set so the optimiser is penalised for truncating.")
    else:
        print("  NO truncation at any tested value: every word asked for was spoken.")
        print("  The cut endings you hear come from something else — the next")
        print("  suspects are the seed (try --vary seed), then trim_end / fade_out")
        print("  in the {} block, then the generator's own segment assembly.")
    outl = [r for r in rows if abs(r[1] - med) > 0.25 * med]
    if outl:
        print(f"\n  Note: duration varies a lot ({min(r[1] for r in rows):.1f}-"
              f"{max(r[1] for r in rows):.1f}s, median {med:.1f}s) — "
              + ', '.join(f"{args.vary} {r[0]:.2f}" for r in outl) +
              " differ by more than 25%.")
        print("  Same words, different pacing: that is delivery, not truncation.")
    if args.keep:
        print(f"\n  renders kept in {tmpdir}/ — listen to confirm")


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == '__main__':
    main()

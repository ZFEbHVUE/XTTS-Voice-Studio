#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
chatterbox_ab.py — generate a French A/B sample with Chatterbox Multilingual,
to compare against the XTTS pipeline clone on the SAME reference and text.

Chatterbox (Resemble AI, MIT) is the 2026 zero-shot contender: 0.5B params,
23 languages incl. French, V3 improves speaker similarity. This script is
STANDALONE — it must run in its own env (its deps are incompatible with
TTS 0.22.0):

  conda create -n cbx python=3.11 -y && conda activate cbx
  pip install chatterbox-tts
  python chatterbox_ab.py Lea_curated.wav -o lea_chatterbox.wav \\
      --text "Bonjour, ceci est une phrase de test pour comparer les voix."

Then, back in the xtts env, measure both candidates against the reference:
  python speaker_identity.py Lea_curated.wav lea_chatterbox.wav
  python speaker_identity.py Lea_curated.wav Lea_pipeline_clone.wav
...and above all: LISTEN. Tips from Resemble: keep the reference in the same
language as the text; if pacing is too fast, lower --cfg to ~0.3.
"""

import argparse


def main():
    p = argparse.ArgumentParser(description='Chatterbox Multilingual A/B sample (French)')
    p.add_argument('reference', help='Reference voice WAV (e.g. the curated file)')
    p.add_argument('-o', '--out', default='chatterbox_ab.wav')
    p.add_argument('--text', default="Bonjour, ceci est une phrase de test pour comparer les voix.")
    p.add_argument('--lang', default='fr')
    p.add_argument('--v3', action='store_true', default=True,
                   help='Use the V3 multilingual checkpoint (better similarity; default)')
    p.add_argument('--v2', dest='v3', action='store_false',
                   help='Use the legacy V2 multilingual checkpoint')
    p.add_argument('--exaggeration', type=float, default=0.5)
    p.add_argument('--cfg', type=float, default=0.5,
                   help='cfg_weight; ~0.3 slows pacing if the reference speaks fast')
    p.add_argument('--device', default=None)
    args = p.parse_args()

    import torch
    import torchaudio as ta
    # Chatterbox instantiates resemble-perth's watermarker unconditionally; on
    # some installs its import fails silently and the symbol is None -> crash.
    # For a private A/B the watermark is irrelevant: inject a pass-through.
    try:
        import perth
        if getattr(perth, 'PerthImplicitWatermarker', None) is None:
            class _NoWatermark:
                def apply_watermark(self, wav, sample_rate=None, **kw):
                    return wav
                def get_watermark(self, *a, **kw):
                    return None
            perth.PerthImplicitWatermarker = _NoWatermark
            print("[!] perth watermarker unavailable -> generating WITHOUT watermark "
                  "(fine for a private A/B; pip install -U resemble-perth to restore it).")
    except Exception:
        pass
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Loading Chatterbox Multilingual ({'V3' if args.v3 else 'V2'}) on {device}...")
    if args.v3:
        try:
            model = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
        except TypeError:
            print("[!] Installed chatterbox-tts predates the V3 checkpoint — falling back to V2.")
            print("[!] For V3 (better speaker similarity): pip install -U chatterbox-tts")
            print("[!] (or, if PyPI lags: pip install -U git+https://github.com/resemble-ai/chatterbox.git)")
            model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    else:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    print(f"[*] Generating [{args.lang}]: {args.text[:60]}...")
    wav = model.generate(args.text, language_id=args.lang,
                         audio_prompt_path=args.reference,
                         exaggeration=args.exaggeration, cfg_weight=args.cfg)
    ta.save(args.out, wav, model.sr)
    print(f"[OK] {args.out}  ({model.sr} Hz)")
    print("    Now, in the xtts env:")
    print(f"      python speaker_identity.py {args.reference} {args.out}")
    print("    ...and listen against the XTTS pipeline clone.")


if __name__ == '__main__':
    main()

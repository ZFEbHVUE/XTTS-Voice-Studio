#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_comparator.py — closed-loop EQ/vol fit for an XTTS clone.

The {} block is FROZEN input (seed, temp, rep_pen… are chosen upstream by the
Validator, which scores accent + identity). The comparator's only job is the
post-processing tone: it generates the clone once on the reference text, then
iteratively fits the generator's exact 3-band peaking EQ + volume by least
squares so the rendered clone's long-term average spectrum (LTAS) matches the
reference. Each pass renders the full chain (hp/lp/NR/comp/limiter), re-measures,
and adds the LS correction — closing the loop on the post-chain colouring.

It does NOT search seeds or score pronunciation/identity — that belongs to the
Validator. Generation goes through xtts_clone.py so the cloning knobs
(gpt_cond_len / gpt_cond_chunk_len / max_ref_len / sound_norm_refs) actually take
effect, consistent with the Validator and the final render.

Usage:
  python voice_comparator.py ref.wav [extra_refs...] FR \\
      --xtts-block "{1, 42, ...}" --audio-block "[1, FR, ...]" \\
      [--text-file script.txt | --text "Phrase de test."] \\
      [--iterations 3] [--conv-threshold 0.5] \\
      [--max-ref-len 30] [--output clone.wav] [--output-optimised clone_opt.wav]
"""

import os
import sys
import re
import argparse
import tempfile
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

LANGS = {'FR','EN','ES','DE','IT','PT','PL','TR','RU','NL','CS','AR','ZH-CN','HU','KO','JA','HI'}


# ── Block parsing / formatting ────────────────────────────────────────────────

def parse_xtts_block(block):
    nums = [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?', block)]
    keys = ['seed', 'trim_start', 'trim_end', 'fade_in', 'fade_out',
            'temp', 'top_k', 'top_p', 'rep_pen', 'len_pen',
            'gpt_cond_len', 'gpt_cond_chunk_len', 'sound_norm_refs',
            'num_beams']   # optional 15th (v24)
    out = {}
    for i, k in enumerate(keys):
        if i + 1 < len(nums):
            out[k] = nums[i + 1]
    return out


def parse_audio_block(block):
    m = re.search(r'\b(FR|EN|ES|DE|IT|PT|PL|TR|RU|NL|CS|AR|HU|KO|JA|HI)\b', block, re.I)
    lang = m.group(1).upper() if m else 'FR'
    nums = [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?', block)][1:]
    keys = ['speed', 'vol', 'eq_low', 'eq_mid', 'eq_high', 'hp', 'lp',
            'NR', 'comp', 'de-ess', 'reverb', 'noise_gate', 'pan', 'limiter']
    out = {'lang': lang}
    for i, k in enumerate(keys):
        if i < len(nums):
            out[k] = nums[i]
    return out


def fmt(v):
    f = float(v)
    if f == int(f):
        return str(int(f))
    return str(round(f, 3))


def format_blocks(N, xtts, audio, lang):
    xa = [N, int(xtts.get('seed', 0)), int(xtts.get('trim_start', 0)),
          int(xtts.get('trim_end', 0)), int(xtts.get('fade_in', 100)),
          int(xtts.get('fade_out', 250)), xtts.get('temp', 0.65),
          int(xtts.get('top_k', 50)), xtts.get('top_p', 0.85),
          xtts.get('rep_pen', 5.0), xtts.get('len_pen', 1.0),
          int(xtts.get('gpt_cond_len', 30)), int(xtts.get('gpt_cond_chunk_len', 4)),
          int(xtts.get('sound_norm_refs', 0))]
    aa = [audio.get('speed', 1.0), audio.get('vol', 0), audio.get('eq_low', 0),
          audio.get('eq_mid', 0), audio.get('eq_high', 0), audio.get('hp', 0),
          audio.get('lp', 0), audio.get('NR', 0), audio.get('comp', 0),
          audio.get('de-ess', 0), audio.get('reverb', 0), audio.get('noise_gate', 0),
          audio.get('pan', 0), audio.get('limiter', 1)]
    xtts_str  = '{' + ', '.join(fmt(v) for v in xa) + '}'
    audio_str = f"[{N}, {lang}, " + ', '.join(fmt(v) for v in aa) + ']'
    return xtts_str, audio_str


# ── Text ──────────────────────────────────────────────────────────────────────

def clean_text(raw):
    """Strip pause/parameter markup, keep prose for a short test clip."""
    lines = []
    for ln in raw.splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith('{') or s.startswith('['):
            continue
        s = re.sub(r'\[pause[^\]]*\]', ' ', s, flags=re.I)
        s = re.sub(r'\[/?parallel[^\]]*\]', ' ', s, flags=re.I)
        s = s.strip()
        if s:
            lines.append(s)
    text = ' '.join(lines).strip()
    return text or "Bonjour, ceci est un court test de comparaison de voix."


# ── Audio I/O ─────────────────────────────────────────────────────────────────

def load_mono(path, target_sr=None):
    import soundfile as sf
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if target_sr and sr != target_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), int(target_sr))
        y = resample_poly(y, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    return y, sr


# ── Post-processing ───────────────────────────────────────────────────────────

def apply_post(raw_path, out_path, audio, xtts, lang, gen_path):
    """Apply the generator's process_audio chain to a raw clone."""
    from pydub import AudioSegment
    seg = AudioSegment.from_wav(raw_path)
    process_audio = None
    if gen_path and os.path.exists(gen_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location('gmg', gen_path)
        gmg = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(gmg)
            process_audio = getattr(gmg, 'process_audio', None)
        except Exception:
            pass
    if process_audio:
        config = {
            'speed': float(audio.get('speed', 1.0)), 'volume': int(audio.get('vol', 0)),
            'eq_low': float(audio.get('eq_low', 0)), 'eq_mid': float(audio.get('eq_mid', 0)),
            'eq_high': float(audio.get('eq_high', 0)), 'highpass': float(audio.get('hp', 0)),
            'lowpass': float(audio.get('lp', 0)), 'noise_reduction': float(audio.get('NR', 0)),
            'compression': float(audio.get('comp', 0)), 'deesser': float(audio.get('de-ess', 0)),
            'reverb': float(audio.get('reverb', 0)), 'noise_gate': float(audio.get('noise_gate', 0)),
            'pan': float(audio.get('pan', 0)), 'language': lang.lower(),
        }
        xtts_fx = {
            'trim_start': int(xtts.get('trim_start', 0)), 'trim_end': int(xtts.get('trim_end', 0)),
            'fade_in': int(xtts.get('fade_in', 100)), 'fade_out': int(xtts.get('fade_out', 250)),
            'limiter': int(audio.get('limiter', 1)),
        }
        seg = process_audio(seg, config, xtts_fx)
        # In the generator, volume is applied AFTER process_audio (in
        # generate_sentence_audio) — process_audio itself never touches it.
        # Mirror that here, otherwise the closed loop can never level-match.
        vol = int(audio.get('vol', 0))
        if vol:
            seg = seg + vol
    seg.export(out_path, format='wav')
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Identity-driven XTTS clone optimiser')
    p.add_argument('reference')
    p.add_argument('voice_refs', nargs='*')
    p.add_argument('--xtts-block', required=True)
    p.add_argument('--audio-block', required=True)
    p.add_argument('--text', default="Bonjour, ceci est un court test de comparaison de voix.")
    p.add_argument('--text-file', default=None)
    p.add_argument('--auto-text', action='store_true',
                   help='Transcribe the reference (faster-whisper, CPU) and fit on '
                        'its own words — like-for-like phonetics for the LTAS match')
    p.add_argument('--whisper-model', default='small',
                   help='faster-whisper model for --auto-text (default: small)')
    p.add_argument('--output', default=None)
    p.add_argument('--output-optimised', default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--max-ref-len', type=int, default=30,
                   help='Seconds of reference used for the SPEAKER EMBEDDING '
                        '(XTTS default is only 10; higher = better identity).')
    p.add_argument('--gpt-cond-len', type=int, default=None,
                   help='Override seconds of reference for GPT conditioning '
                        '(default: value from the {} block, or 30)')
    p.add_argument('--iterations', type=int, default=3,
                   help='Max closed-loop EQ/vol refinement passes (default: 3)')
    p.add_argument('--conv-threshold', type=float, default=0.5,
                   help='Stop when the largest EQ/vol change in a pass falls '
                        'below this many dB (default: 0.5)')
    args = p.parse_args()

    # Separate trailing lang from refs
    refs = list(args.voice_refs)
    lang = 'FR'
    if refs and refs[-1].upper() in LANGS:
        lang = refs[-1].upper(); refs = refs[:-1]
    if not refs:
        refs = [args.reference]
    speaker_wav = refs if len(refs) > 1 else refs[0]

    xtts  = parse_xtts_block(args.xtts_block)
    audio = parse_audio_block(args.audio_block)
    lang  = audio.get('lang', lang)

    if args.auto_text:
        # Fit on the reference's own words: same phonemes on both sides of the
        # LTAS comparison, so the fit corrects the voice, not the text.
        print(f"[*] Transcribing reference with faster-whisper '{args.whisper_model}' (cpu)...")
        from faster_whisper import WhisperModel
        wm = WhisperModel(args.whisper_model, device='cpu', compute_type='int8')
        segs, _info = wm.transcribe(args.reference, language=lang.lower().split('-')[0],
                                    beam_size=5, vad_filter=True)
        parts, total = [], 0
        for s in segs:                       # whole segments up to XTTS's fr limit
            t = s.text.strip()
            if not t:
                continue
            if total + len(t) + 1 > 240:
                break
            parts.append(t); total += len(t) + 1
        auto = clean_text(' '.join(parts))
        if len(auto) >= 30:
            text = auto
            print(f"[OK] Fitting on: \"{text[:70]}{'...' if len(text) > 70 else ''}\"")
        else:
            print("[!] Transcription too short — keeping the provided text.")
            text = clean_text(args.text)
    elif args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, encoding='utf-8') as fh:
            text = clean_text(fh.read())
    else:
        text = clean_text(args.text)

    seed = int(xtts.get('seed', 0))

    print("=" * 64)
    print("  XTTS Voice Comparator — closed-loop EQ/vol fit (LTAS, least squares)")
    print("=" * 64)
    print(f"  Reference : {os.path.basename(args.reference)}")
    print(f"  Language  : {lang}")
    print(f"  Text      : {text[:58]}{'...' if len(text) > 58 else ''}")
    print(f"  Seed      : {seed} (frozen — set it with the Validator)")
    print("=" * 64)

    gen_path = os.path.join(SCRIPT_DIR, 'guided_meditation_generator_v23.py')
    tmpdir   = tempfile.mkdtemp()

    # Load XTTS once, via the low-level path that actually honours the cloning
    # knobs (tts_to_file silently overrides gpt_cond_len/chunk/max_ref_len/
    # sound_norm_refs with config defaults 12/4/10/False).
    import torch
    import xtts_clone as XC
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[*] Loading XTTS on {device}...")
    tts, model = XC.load_xtts(device)
    print("[OK] XTTS ready")

    # Precompute conditioning latents ONCE (independent of seed) with the real
    # cloning parameters — this is where max_ref_len/gpt_cond_len finally bite.
    gpt_cl  = int(args.gpt_cond_len if args.gpt_cond_len is not None
                  else xtts.get('gpt_cond_len', 30))
    gpt_ccl = int(xtts.get('gpt_cond_chunk_len', 6))
    snorm   = bool(xtts.get('sound_norm_refs', 0))
    print(f"[*] Conditioning: gpt_cond_len={gpt_cl}s  chunk={gpt_ccl}s  "
          f"max_ref_len={args.max_ref_len}s  sound_norm_refs={snorm}")
    latents = XC.compute_latents(model, speaker_wav, gpt_cond_len=gpt_cl,
                                 gpt_cond_chunk_len=gpt_ccl,
                                 max_ref_len=args.max_ref_len, sound_norm_refs=snorm)

    import ltas_match as L

    # ── Generate the clone ONCE (the {} block is frozen, seed included) ───────
    print(f"\n[*] Generating clone (seed {seed})...")
    raw = os.path.join(tmpdir, 'clone_raw.wav')
    XC.generate(model, text, lang, latents, raw,
                temperature=xtts.get('temp', 0.65),
                length_penalty=xtts.get('len_pen', 1.0),
                repetition_penalty=xtts.get('rep_pen', 5.0),
                top_k=int(xtts.get('top_k', 50)),
                top_p=xtts.get('top_p', 0.85),
                num_beams=int(xtts.get('num_beams', 1)),
                speed=1.0, seed=seed)
    print("[OK] Clone generated")

    ref_y, ref_sr  = load_mono(args.reference)
    f_grid, ref_db = L.compute_ltas(ref_y, ref_sr)
    ref_shape      = ref_db - ref_db.mean()

    # ── Closed-loop EQ/vol fit vs the reference (multi-pass) ──────────────────
    # Each pass renders the full chain (incl. hp/lp/NR/comp/limiter), measures the
    # rendered clone's LTAS against the reference, and adds the least-squares EQ
    # correction. Iterating closes the loop on the post-chain colouring, not just
    # the raw clone. Converges in 1–3 passes.
    print(f"\n{'-'*64}\n  Closed-loop EQ/vol fit (LTAS, least squares)\n{'-'*64}")
    opt = dict(audio)
    opt['vol'] = 0; opt['eq_low'] = 0; opt['eq_mid'] = 0; opt['eq_high'] = 0
    cand = os.path.join(tmpdir, 'cand.wav')
    prev_step = float('inf')
    for it in range(1, max(1, args.iterations) + 1):
        apply_post(raw, cand, opt, xtts, lang, gen_path)
        cy, _ = load_mono(cand, target_sr=ref_sr)
        _, cd = L.compute_ltas(cy, ref_sr)
        fit  = L.fit_eq_ls(f_grid, ref_db, cd, fs=ref_sr,
                           cur_hp=opt.get('hp', 0), cur_lp=opt.get('lp', 0))
        dvol = L.level_match_db(ref_y, cy, cur_vol=0)
        step = max(abs(fit['eq_low']), abs(fit['eq_mid']), abs(fit['eq_high']), abs(dvol))
        print(f"  pass {it}: residual {fit['residual_before']:.2f}->{fit['residual_after']:.2f} dB"
              f"  | Δeq=({fit['eq_low']:+.1f},{fit['eq_mid']:+.1f},{fit['eq_high']:+.1f}) Δvol={dvol:+d}")
        if it > 1 and step >= prev_step:
            print("  no further improvement — stopping")
            break
        opt['eq_low']  = round(opt['eq_low']  + fit['eq_low'], 1)
        opt['eq_mid']  = round(opt['eq_mid']  + fit['eq_mid'], 1)
        opt['eq_high'] = round(opt['eq_high'] + fit['eq_high'], 1)
        opt['hp'] = fit['hp']; opt['lp'] = fit['lp']
        opt['vol'] = int(np.clip(opt['vol'] + dvol, -18, 18))
        prev_step = step
        if step < args.conv_threshold:
            print(f"  converged (largest change {step:.2f} dB < {args.conv_threshold} dB)")
            break

    print(f"  Final []: vol={opt['vol']:+d}  eq_low={opt['eq_low']:+.1f}  "
          f"eq_mid={opt['eq_mid']:+.1f}  eq_high={opt['eq_high']:+.1f}  "
          f"hp={opt['hp']:.0f}  lp={opt['lp']:.0f}")

    xtts_str, opt_audio_str = format_blocks(1, xtts, opt, lang)

    # GUI hook: this exact prefix updates the Audio [] field in xtts_studio.py
    print(f"  Next []    : {opt_audio_str}")

    # ── Render outputs ────────────────────────────────────────────────────────
    out_clone = args.output or os.path.join(tmpdir, 'clone.wav')
    out_opt   = args.output_optimised or os.path.join(tmpdir, 'clone_optimised.wav')
    print(f"\n[*] Rendering base clone (original [])...")
    apply_post(raw, out_clone, audio, xtts, lang, gen_path)
    print(f"[*] Rendering optimised clone (fitted [])...")
    apply_post(raw, out_opt, opt, xtts, lang, gen_path)

    by, _ = load_mono(out_clone, target_sr=ref_sr)
    _, bd = L.compute_ltas(by, ref_sr)
    oy, _ = load_mono(out_opt, target_sr=ref_sr)
    _, od = L.compute_ltas(oy, ref_sr)
    res_base = float(np.sqrt(np.mean((ref_shape - (bd - bd.mean())) ** 2)))
    res_opt  = float(np.sqrt(np.mean((ref_shape - (od - od.mean())) ** 2)))
    print(f"  LTAS residual vs reference: base {res_base:.2f} dB -> optimised {res_opt:.2f} dB")

    print(f"\n{'='*64}\n  RESULT\n{'='*64}")
    print(f"  {xtts_str}")
    print(f"  {opt_audio_str}")
    print(f"\n  Base clone      : {out_clone}")
    print(f"  Optimised clone : {out_opt}")
    print("[OK] Done.")


if __name__ == '__main__':
    main()

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

def clean_text(raw, lang='en'):
    """Strip pause/parameter markup, keep prose for a short test clip.
    Falls back to a built-in sentence IN THE TARGET LANGUAGE when empty."""
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
    from probe_texts import default_text as _dt
    return text or _dt(lang)


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

def _quiet_post(raw_path, out_path, audio, xtts, lang, gen_path):
    """apply_post with the generator's per-render logging suppressed.
    The screening and the optimiser call it hundreds of times; without this,
    one useful line drowns in thousands of '[*] Audio: EQ(...)' lines."""
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        return apply_post(raw_path, out_path, audio, xtts, lang, gen_path)


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
    p.add_argument('--text', default=None,
                   help='Sentence to fit on (default: a built-in sentence in the '
                        'target language — see probe_texts.py)')
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
    p.add_argument('--target-dbfs', type=float, default=None,
                   help='Production level mode: fit the volume toward this absolute '
                        'RMS target (e.g. -20) instead of matching the reference. '
                        'Use it when the reference is quiet — matching a -33 dBFS '
                        'reference yields an unusable meditation level.')
    p.add_argument('--no-fit-dynamics', dest='fit_dynamics', action='store_false',
                   help='Do not fit comp / de-ess by measurement (crest factor and '
                        '5-8 kHz sibilance vs the reference)')
    p.add_argument('--fit-identity', action='store_true',
                   help='After the tone fit, search the remaining post-processing '
                        'settings for one that measurably raises ECAPA identity '
                        '(the tone fit optimises spectrum, never identity). Keeps a '
                        'candidate only if it beats the measured noise floor.')
    p.add_argument('--screen-audio', action='store_true',
                   help='Sensitivity screening BEFORE optimising (the Optimetrics '
                        'reflex): sweep each audio parameter alone over its full range '
                        'and report how much ECAPA identity moves, so you know which '
                        'knobs matter for this voice and which are measurably inert.')
    p.add_argument('--optimise-audio', default='none', choices=['none', 'nelder', 'de'],
                   help="Derivative-free optimisation of the whole audio block against "
                        "ECAPA identity. 'nelder' = local simplex (fast), 'de' = "
                        "differential evolution then simplex polish (global, slower). "
                        "Any result is validated on a clone of an unseen sentence and "
                        "discarded if the gain does not transfer.")
    p.add_argument('--optimise-budget', type=int, default=400,
                   help='Max objective evaluations for --optimise-audio (default: 400; '
                        'each one is just post-processing + an embedding)')
    p.add_argument('--no-identity-check', action='store_true',
                   help='Skip the extra generation that measures identity on UNSEEN '
                        'text (the figure that matches real usage)')
    p.set_defaults(fit_dynamics=True)
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
        auto = clean_text(' '.join(parts), lang)
        if len(auto) >= 30:
            text = auto
            print(f"[OK] Fitting on: \"{text[:70]}{'...' if len(text) > 70 else ''}\"")
        else:
            print("[!] Transcription too short — keeping the provided text.")
            text = clean_text(args.text, lang)
    elif args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, encoding='utf-8') as fh:
            text = clean_text(fh.read(), lang)
    else:
        text = clean_text(args.text, lang)

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
        if args.target_dbfs is not None:
            dvol = L.level_to_target_db(cy, args.target_dbfs, cur_vol=0)
        else:
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
        if it == 1:
            # hp/lp are guards, set once from the first measurement. Re-deriving
            # them every pass let them walk away cumulatively while the EQ was
            # already correcting the same imbalance.
            opt['hp'] = fit['hp']; opt['lp'] = fit['lp']
        opt['vol'] = int(np.clip(opt['vol'] + dvol, -18, 18))
        prev_step = step
        if step < args.conv_threshold:
            print(f"  converged (largest change {step:.2f} dB < {args.conv_threshold} dB)")
            break

    # ── Measured fit of de-ess and compression ────────────────────────────────
    # The tail of the [] block used to stay at zero because nothing measured it.
    # Sibilance (5-8 kHz vs 300-3 kHz) and crest factor ARE measurable against
    # the reference, so they get the same treatment as the EQ: propose a value,
    # render, re-measure, and KEEP IT ONLY IF THE GAP SHRINKS. NR/reverb/gate/pan
    # stay at zero on purpose (no measurable target; the gate chops speech).
    if args.fit_dynamics:
        print(f"\n{'-'*64}\n  Measured de-ess / compression fit\n{'-'*64}")
        ref_sib, ref_crest = L.sibilance_db(ref_y, ref_sr), L.crest_db(ref_y)
        apply_post(raw, cand, opt, xtts, lang, gen_path)
        cy, _ = load_mono(cand, target_sr=ref_sr)
        sib_gap = L.sibilance_db(cy, ref_sr) - ref_sib
        crest_gap = L.crest_db(cy) - ref_crest
        left_at_zero = []
        print(f"  reference: sibilance {ref_sib:+.1f} dB   crest {ref_crest:.1f} dB")
        print(f"  clone gap: sibilance {sib_gap:+.1f} dB   crest {crest_gap:+.1f} dB")

        for name, gap, thr, scale, cap, what in (
                ('de-ess', sib_gap, 1.5, 0.18, 0.5, 'sibilance'),
                ('comp', crest_gap, 2.0, 0.12, 0.6, 'dynamics')):
            if gap <= thr:
                if gap < 0:
                    # Expected for synthetic speech: XTTS output is smoother than
                    # a real recording. A de-esser/compressor can only REDUCE, so
                    # a negative gap is not correctable here — 0 is the right value.
                    print(f"  {name}: clone has LESS {what} than the reference "
                          f"({gap:+.1f} dB) — a {name} can only reduce it, nothing "
                          f"to do -> 0")
                else:
                    print(f"  {name}: gap {gap:+.1f} dB within tolerance ({thr}) -> 0")
                left_at_zero.append(name)
                continue
            trial = dict(opt)
            trial[name] = round(min(cap, (gap - thr) * scale), 2)
            apply_post(raw, cand, trial, xtts, lang, gen_path)
            ty, _ = load_mono(cand, target_sr=ref_sr)
            new_gap = ((L.sibilance_db(ty, ref_sr) - ref_sib) if name == 'de-ess'
                       else (L.crest_db(ty) - ref_crest))
            if abs(new_gap) < abs(gap) - 0.2:
                opt[name] = trial[name]
                print(f"  {name} -> {trial[name]}  (gap {gap:+.1f} -> {new_gap:+.1f} dB)")
            else:
                print(f"  {name} {trial[name]} did not help ({gap:+.1f} -> {new_gap:+.1f}) "
                      f"-> reverted to 0")

    # ── Identity-driven post-processing search ────────────────────────────────
    # Everything above optimises TONE (LTAS residual). This stage asks a
    # different question, the one that actually matters: can post-processing
    # raise the ECAPA IDENTITY of the clone? Nobody had ever measured it — the
    # remaining [] fields stayed at 0 because no criterion ever asked them to
    # move. Each candidate is rendered and scored; only a gain above the
    # measured noise floor is kept.
    if args.fit_identity:
        print(f"\n{'-'*64}\n  Identity-driven post-processing search (ECAPA)\n{'-'*64}")
        try:
            from speaker_identity import SpeakerEncoder
            enc = SpeakerEncoder(device=args.device)
            ref_emb = enc.embed(args.reference)

            def ident_of(block):
                apply_post(raw, cand, block, xtts, lang, gen_path)
                return float(enc.cosine(ref_emb, enc.embed(cand))), cand

            base_id, _ = ident_of(opt)
            # Noise floor: identity measured on each half of the same clone.
            # Two halves of one render are the same voice, so their spread is
            # the measurement variability for this clip.
            cy_all, csr = load_mono(cand, target_sr=16000)   # ECAPA rate
            h = len(cy_all) // 2
            halves = [float(enc.cosine(ref_emb, enc.embed(cy_all[:h], sr=csr))),
                      float(enc.cosine(ref_emb, enc.embed(cy_all[h:], sr=csr)))]
            noise = max(0.005, abs(halves[0] - halves[1]) / 2.0)
            print(f"  baseline (tone-fitted) identity {base_id:.4f}   "
                  f"noise floor ±{noise:.4f}")

            trials = [('no post-processing at all',
                       dict(opt, vol=0, eq_low=0, eq_mid=0, eq_high=0, hp=0, lp=0)),
                      ('EQ off (keep vol/filters)', dict(opt, eq_low=0, eq_mid=0, eq_high=0)),
                      ('brighter (eq_high +2)', dict(opt, eq_high=opt['eq_high'] + 2)),
                      ('darker (eq_high -2)',   dict(opt, eq_high=opt['eq_high'] - 2)),
                      ('full band (lp 16k)',    dict(opt, lp=16000)),
                      ('narrow band (lp 8k)',   dict(opt, lp=8000)),
                      ('NR 0.2',                dict(opt, NR=0.2)),
                      ('comp 0.3',              dict(opt, comp=0.3)),
                      ('de-ess 0.3',            {**opt, 'de-ess': 0.3})]
            results = [('baseline', opt, base_id)]
            for label, blk in trials:
                try:
                    v, _ = ident_of(blk)
                except Exception as e:
                    print(f"  {label:26s} failed ({e})"); continue
                results.append((label, blk, v))
                d = v - base_id
                flag = '  <-- above noise' if d > noise else ''
                print(f"  {label:26s} identity {v:.4f}  ({d:+.4f}){flag}")

            lbl, blk, best_id = max(results, key=lambda r: r[2])
            if best_id > base_id + noise:
                print(f"\n  ADOPTED '{lbl}': identity {base_id:.4f} -> {best_id:.4f} "
                      f"(+{best_id - base_id:.4f}, above the {noise:.4f} noise floor)")
                opt = dict(blk)
            else:
                print(f"\n  No post-processing setting raises identity beyond the noise "
                      f"floor (best {best_id:.4f} vs baseline {base_id:.4f}).")
                print(f"  Conclusion for this voice: post-processing shapes TONE, not "
                      f"IDENTITY — the zero fields are zero because nothing they can do")
                print(f"  makes the clone measurably more like the person.")
            apply_post(raw, cand, opt, xtts, lang, gen_path)   # restore chosen block
        except Exception as e:
            print(f"  [!] Identity search unavailable ({e})")

        if left_at_zero and not args.screen_audio:
            print(f"  ({', '.join(left_at_zero)} left at 0 because the measured gap "
                  f"points the wrong way for these tools.")
            print(f"   To find out whether ANY audio parameter can move identity for "
                  f"this voice, re-run with --screen-audio.)")

    # ── Sensitivity screening (which knobs matter at all?) ────────────────────
    # The Optimetrics reflex: screen BEFORE optimising. One factor at a time
    # across its full musical range, measuring how much ECAPA identity actually
    # moves. This answers, per parameter and with numbers, the question the tool
    # could never answer before: does this knob do anything for THIS voice?
    if args.screen_audio:
        print(f"\n{'-'*64}\n  Sensitivity screening (one factor at a time, ECAPA identity)\n{'-'*64}")
        try:
            from speaker_identity import SpeakerEncoder
            enc_s = SpeakerEncoder(device=args.device)
            ref_s = enc_s.embed(args.reference)

            def ident_blk(blk):
                _quiet_post(raw, cand, blk, xtts, lang, gen_path)
                return float(enc_s.cosine(ref_s, enc_s.embed(cand)))

            base_s = ident_blk(opt)
            # Noise floor from the two halves of the same render (same voice,
            # so their spread is the measurement variability for this clip).
            cy_h, csr_h = load_mono(cand, target_sr=16000)
            hh = len(cy_h) // 2
            noise_s = max(0.004, abs(
                float(enc_s.cosine(ref_s, enc_s.embed(cy_h[:hh], sr=csr_h))) -
                float(enc_s.cosine(ref_s, enc_s.embed(cy_h[hh:], sr=csr_h)))) / 2.0)
            print(f"  baseline identity {base_s:.4f}   noise floor ±{noise_s:.4f}\n")

            GRID = [('vol',    [-12, -6, 0, 6, 12]),
                    ('eq_low', [-6, -3, 0, 3, 6]),
                    ('eq_mid', [-6, -3, 0, 3, 6]),
                    ('eq_high',[-6, -3, 0, 3, 6]),
                    ('hp',     [40, 65, 95, 150]),
                    ('lp',     [6000, 8000, 11000, 16000]),
                    ('NR',     [0, 0.15, 0.3, 0.5]),
                    ('comp',   [0, 0.2, 0.4, 0.6]),
                    ('de-ess', [0, 0.15, 0.3, 0.5])]
            rows = []
            for axis, values in GRID:
                ids = []
                for v in values:
                    blk = dict(opt); blk[axis] = v
                    ids.append((v, ident_blk(blk)))
                swing = max(i for _, i in ids) - min(i for _, i in ids)
                bestv, besti = max(ids, key=lambda t: t[1])
                rows.append((axis, swing, bestv, besti, besti - base_s))
            rows.sort(key=lambda r: -r[1])

            print(f"  {'axis':>8}{'swing':>9}{'best value':>12}{'identity':>10}{'vs base':>9}")
            for axis, swing, bestv, besti, delta in rows:
                verdict = ('MATTERS' if swing > 3 * noise_s else
                           'marginal' if swing > noise_s else 'INERT')
                print(f"  {axis:>8}{swing:>9.4f}{str(bestv):>12}{besti:>10.4f}"
                      f"{delta:>+9.4f}   {verdict}")
            live = [r[0] for r in rows if r[1] > 3 * noise_s]
            dead = [r[0] for r in rows if r[1] <= noise_s]
            print(f"\n  Knobs that move identity for this voice: "
                  f"{', '.join(live) if live else 'NONE'}")
            if dead:
                print(f"  Knobs measurably inert here: {', '.join(dead)}")
            print(f"  (screening only — use --optimise-audio to search the live axes "
                  f"jointly, interactions included)")
        except Exception as e:
            print(f"  [!] Screening unavailable ({e})")

    # ── Derivative-free optimisation of the whole audio block ─────────────────
    # Evaluations here are CHEAP (one clone, then post-processing + one ECAPA
    # embed each), so unlike the XTTS search we can afford hundreds of them —
    # which is exactly the regime where derivative-free global optimisers earn
    # their keep. Least squares does not apply (ECAPA is a non-differentiable
    # black box, there is no design matrix); bisection does not either (12
    # interacting axes, non-monotonic). Differential evolution + Nelder-Mead
    # polish is the method that matches this structure.
    #
    # DANGER, handled below: optimising 12 knobs against a neural metric is
    # metric hacking. Every result is therefore validated on a SECOND clone
    # generated from a different sentence, and rejected if it does not transfer.
    if args.optimise_audio != 'none':
        print(f"\n{'-'*64}\n  Audio-block optimisation ({args.optimise_audio}) — "
              f"target: identity\n{'-'*64}")
        try:
            import numpy as _np
            from scipy.optimize import differential_evolution, minimize
            from speaker_identity import SpeakerEncoder
            enc = SpeakerEncoder(device=args.device)
            ref_emb = enc.embed(args.reference)

            # Bounds kept musically sane: outside these the result stops being
            # a listenable voice regardless of what the metric says.
            AXES = [('vol', -18, 18), ('eq_low', -6, 6), ('eq_mid', -6, 6),
                    ('eq_high', -6, 6), ('hp', 40, 150), ('lp', 6000, 16000),
                    ('NR', 0, 0.5), ('comp', 0, 0.6), ('de-ess', 0, 0.5)]
            x0 = _np.array([float(opt.get(k, 0)) for k, _, _ in AXES])
            bounds = [(lo, hi) for _, lo, hi in AXES]

            evals = [0]
            def identity_of(x, wav_in, wav_out):
                blk = dict(opt)
                for (k, _, _), v in zip(AXES, x):
                    blk[k] = round(float(v)) if k in ('vol', 'hp', 'lp') else float(v)
                _quiet_post(wav_in, wav_out, blk, xtts, lang, gen_path)
                return float(enc.cosine(ref_emb, enc.embed(wav_out))), blk

            def neg_identity(x):
                evals[0] += 1
                v, _ = identity_of(x, raw, cand)
                return -v

            base_id = -neg_identity(x0)
            print(f"  start (tone-fitted block): identity {base_id:.4f}")

            if args.optimise_audio == 'de':
                # Seed the population with the tone-fitted block. Without it,
                # differential evolution never evaluates the starting point and
                # can return something WORSE after hundreds of evaluations —
                # which is exactly what happens on a flat, noisy landscape.
                try:
                    res = differential_evolution(
                        neg_identity, bounds, maxiter=max(5, args.optimise_budget // 90),
                        popsize=10, tol=1e-4, seed=0, polish=False, init='sobol', x0=x0)
                except TypeError:            # older scipy has no x0=
                    res = differential_evolution(
                        neg_identity, bounds, maxiter=max(5, args.optimise_budget // 90),
                        popsize=10, tol=1e-4, seed=0, polish=False, init='sobol')
                xb = res.x
                print(f"  differential evolution: {evals[0]} evaluations")
                r2 = minimize(neg_identity, xb, method='Nelder-Mead',
                              options={'maxiter': 120, 'xatol': 1e-3, 'fatol': 1e-5})
                if r2.fun < res.fun:
                    xb = r2.x
                print(f"  + Nelder-Mead polish: {evals[0]} total evaluations")
            else:
                r = minimize(neg_identity, x0, method='Nelder-Mead',
                             options={'maxiter': args.optimise_budget,
                                      'xatol': 1e-3, 'fatol': 1e-5})
                xb = _np.clip(r.x, [b[0] for b in bounds], [b[1] for b in bounds])
                print(f"  Nelder-Mead: {evals[0]} evaluations")

            best_id, best_blk = identity_of(xb, raw, cand)
            if best_id < base_id:
                # A search that ends below its own starting point has found
                # nothing; say so plainly instead of reporting a "result".
                print(f"  search ended BELOW the starting point "
                      f"({best_id:.4f} < {base_id:.4f}) — the landscape is flat "
                      f"and the optimiser is sampling noise.")
                print(f"  Keeping the tone-fitted block (run --screen-audio to see "
                      f"which axes, if any, actually move identity for this voice).")
                apply_post(raw, cand, opt, xtts, lang, gen_path)
                raise StopIteration
            print(f"  optimised identity {best_id:.4f}  ({best_id - base_id:+.4f})")

            # ── Hold-out: does it transfer to a clone of a DIFFERENT sentence? ──
            HOLD_TEXT = ("Respire lentement et laisse les épaules redescendre, "
                         "sans forcer.")
            print(f"\n  Hold-out check on an unseen sentence "
                  f"(one extra generation)...")
            raw2 = os.path.join(tmpdir, 'clone_raw_holdout.wav')
            cand2 = os.path.join(tmpdir, 'cand_holdout.wav')
            XC.generate(model, HOLD_TEXT, lang, latents, raw2,
                        temperature=xtts.get('temp', 0.65),
                        length_penalty=xtts.get('len_pen', 1.0),
                        repetition_penalty=xtts.get('rep_pen', 5.0),
                        top_k=int(xtts.get('top_k', 50)),
                        top_p=xtts.get('top_p', 0.85),
                        num_beams=int(xtts.get('num_beams', 1)),
                        speed=1.0, seed=seed)
            h_base, _ = identity_of(x0, raw2, cand2)
            h_best, _ = identity_of(xb, raw2, cand2)
            print(f"  held-out clone: {h_base:.4f} -> {h_best:.4f} "
                  f"({h_best - h_base:+.4f})")

            gain_fit = best_id - base_id
            gain_out = h_best - h_base
            if gain_out > 0.01 and gain_out > 0.4 * gain_fit:
                for k, _, _ in AXES:
                    opt[k] = (round(best_blk[k]) if k in ('vol', 'hp', 'lp')
                              else round(best_blk[k], 2))
                print(f"\n  ADOPTED: the gain transfers to unseen speech "
                      f"({gain_out:+.4f}).")
            else:
                print(f"\n  REJECTED: the gain does NOT transfer "
                      f"(fit {gain_fit:+.4f} vs held-out {gain_out:+.4f}).")
                print(f"  That is metric hacking — settings tuned to flatter ECAPA on "
                      f"one clip, not to make the voice more like the person.")
                print(f"  Keeping the tone-fitted block.")
            apply_post(raw, cand, opt, xtts, lang, gen_path)
        except StopIteration:
            pass          # search ended below its start; message already printed
        except Exception as e:
            print(f"  [!] Audio optimisation unavailable ({e})")

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

    # ── Identity on UNSEEN text — the figure that matches real usage ──────────
    # With --auto-text the fit runs on the reference's OWN words, and identity
    # measured there is systematically optimistic: the clone repeats sentences
    # the speaker actually said, with matching prosody. Measured on new text —
    # i.e. on the meditation you will actually generate — it drops noticeably
    # (0.80 vs 0.69 on one measured voice). Report both so the number you read
    # is the one that applies.
    if not args.no_identity_check:
        try:
            from speaker_identity import SpeakerEncoder
            _enc = SpeakerEncoder(device=args.device)
            _ref = _enc.embed(args.reference)
            _fit_id = float(_enc.cosine(_ref, _enc.embed(out_opt)))
            UNSEEN = ("Installe-toi confortablement et laisse le silence "
                      "prendre sa place.")
            _raw2 = os.path.join(tmpdir, 'clone_unseen_raw.wav')
            _out2 = os.path.join(tmpdir, 'clone_unseen.wav')
            XC.generate(model, UNSEEN, lang, latents, _raw2,
                        temperature=xtts.get('temp', 0.65),
                        length_penalty=xtts.get('len_pen', 1.0),
                        repetition_penalty=xtts.get('rep_pen', 5.0),
                        top_k=int(xtts.get('top_k', 50)),
                        top_p=xtts.get('top_p', 0.85),
                        num_beams=int(xtts.get('num_beams', 1)),
                        speed=1.0, seed=seed)
            apply_post(_raw2, _out2, opt, xtts, lang, gen_path)
            _unseen_id = float(_enc.cosine(_ref, _enc.embed(_out2)))
            print(f"  Identity: {_fit_id:.4f} on the fitted sentence, "
                  f"{_unseen_id:.4f} on UNSEEN text ({_unseen_id - _fit_id:+.4f})")
            if _fit_id - _unseen_id > 0.05:
                print(f"  ^ the fitted figure is optimistic; {_unseen_id:.4f} is what "
                      f"your generated meditation will resemble.")
        except Exception as _e:
            print(f"  [!] Unseen-text identity check skipped ({_e})")

    print(f"\n{'='*64}\n  RESULT\n{'='*64}")
    print(f"  {xtts_str}")
    print(f"  {opt_audio_str}")
    print(f"\n  Base clone      : {out_clone}")
    print(f"  Optimised clone : {out_opt}")
    print("[OK] Done.")


if __name__ == '__main__':
    main()

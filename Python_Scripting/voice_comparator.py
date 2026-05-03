#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_comparator.py — Compare a reference voice with a XTTS clone and
automatically suggest improved audio parameters.

Usage:
    python voice_comparator.py <reference.wav> <voice_ref.wav> <lang> \\
        --xtts-block "{1, 42, ...}" --audio-block "[1, FR, ...]" \\
        [--text "Short test sentence."] [--output result.wav]

The script:
  1. Generates a short clone using the provided parameters
  2. Analyses both reference and clone spectrally
  3. Computes the gap in RMS, centroid, eq bands
  4. Suggests optimised [] audio params
  5. Optionally generates an improved clone to confirm
"""

import os
import sys
import re
import time
import argparse
import tempfile
import numpy as np

# ── Spectral analysis ────────────────────────────────────────────────────────

def spectral(y, sr):
    from scipy.signal import welch
    freqs, psd = welch(y, sr, nperseg=1024)
    total = np.sum(psd) + 1e-10
    rms_db    = 20 * np.log10(np.sqrt(np.mean(y**2)) + 1e-10)
    peak      = float(np.max(np.abs(y)) + 1e-10)
    crest_db  = 20 * np.log10(peak) - rms_db
    centroid  = float(np.sum(freqs * psd) / total)
    e_low     = float(np.sum(psd[freqs < 300]) / total * 100)
    e_mid     = float(np.sum(psd[(freqs >= 300) & (freqs < 3000)]) / total * 100)
    e_high    = float(np.sum(psd[(freqs >= 3000) & (freqs < 8000)]) / total * 100)
    return dict(rms=rms_db, crest=crest_db, centroid=centroid, low=e_low, mid=e_mid, high=e_high)


def load_mono(path, target_sr=None):
    import soundfile as sf
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if target_sr and sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr, target_sr)
        y = resample_poly(y, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    return y, sr


# ── Block parsing ─────────────────────────────────────────────────────────────

def parse_xtts_block(block):
    nums = [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?', block)]
    keys = ['seed', 'trim_start', 'trim_end', 'fade_in', 'fade_out',
            'temp', 'top_k', 'top_p', 'rep_pen', 'len_pen',
            'gpt_cond_len', 'gpt_cond_chunk_len', 'sound_norm_refs']
    result = {}
    for i, k in enumerate(keys):
        if i + 1 < len(nums):
            result[k] = nums[i + 1]  # skip N
    return result


def parse_audio_block(block):
    # Extract lang
    lang_match = re.search(r'\b(FR|EN|ES|DE|IT|PT|PL|TR|RU|NL|CS|AR|HU|KO|JA|HI)\b',
                           block, re.IGNORECASE)
    lang = lang_match.group(1).upper() if lang_match else 'FR'
    all_nums = [float(x) for x in re.findall(r'[-+]?\d+(?:\.\d+)?', block)]
    nums = all_nums[1:]  # skip N
    keys = ['speed', 'vol', 'eq_low', 'eq_mid', 'eq_high',
            'hp', 'lp', 'NR', 'comp', 'de-ess',
            'reverb', 'noise_gate', 'pan', 'limiter']
    result = {'lang': lang}
    for i, k in enumerate(keys):
        if i < len(nums):
            result[k] = nums[i]
    return result


def format_block(N, xtts, audio, lang):
    def fmt(v):
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(round(v, 3))

    xtts_arr = [N,
                int(xtts.get('seed', 0)),
                int(xtts.get('trim_start', 0)),
                int(xtts.get('trim_end', 0)),
                int(xtts.get('fade_in', 100)),
                int(xtts.get('fade_out', 250)),
                xtts.get('temp', 0.72),
                int(xtts.get('top_k', 50)),
                xtts.get('top_p', 0.85),
                xtts.get('rep_pen', 5.0),
                xtts.get('len_pen', 1.0),
                int(xtts.get('gpt_cond_len', 30)),
                int(xtts.get('gpt_cond_chunk_len', 4)),
                int(xtts.get('sound_norm_refs', 0))]
    xtts_str = ', '.join(fmt(v) for v in xtts_arr)

    audio_arr = [audio.get('speed', 1.0),
                 audio.get('vol', 0),
                 audio.get('eq_low', 0),
                 audio.get('eq_mid', 0),
                 audio.get('eq_high', 0),
                 audio.get('hp', 0),
                 audio.get('lp', 0),
                 audio.get('NR', 0),
                 audio.get('comp', 0),
                 audio.get('de-ess', 0),
                 audio.get('reverb', 0),
                 audio.get('noise_gate', 0),
                 audio.get('pan', 0),
                 audio.get('limiter', 1)]
    audio_str = f"{N}, {lang}, " + ', '.join(fmt(v) for v in audio_arr)

    return f"{{{xtts_str}}}", f"[{audio_str}]"


# ── Score ─────────────────────────────────────────────────────────────────────

def compute_score(ref, clone):
    """Lower = closer to reference."""
    return (abs(clone['rms'] - ref['rms']) +
            abs(clone['centroid'] - ref['centroid']) / 10 +
            abs(clone['low'] - ref['low']) +
            abs(clone['mid'] - ref['mid']) +
            abs(clone['high'] - ref['high']))


# ── Optimise audio params ─────────────────────────────────────────────────────

def optimise_audio(ref, clone, audio, clone_dur=None, ref_dur=None):
    """Compute improved audio params based on spectral gaps."""
    opt = dict(audio)

    # ── Volume — match RMS directly ──────────────────────────────────────────
    rms_gap = ref['rms'] - clone['rms']
    # Apply full correction in one step, clamp to ±12dB
    opt['vol'] = float(np.clip(round(audio.get('vol', 0) + rms_gap), -6, 12))

    # ── EQ high — centroid proxy, proportional step ───────────────────────────
    centroid_gap = ref['centroid'] - clone['centroid']
    if abs(centroid_gap) > 50:
        step = np.clip(centroid_gap / 100.0, -3, 3)
        opt['eq_high'] = float(round(np.clip(audio.get('eq_high', 0) + step, -9, 6), 1))

    # ── EQ low — bass balance, proportional step ──────────────────────────────
    low_gap = ref['low'] - clone['low']
    if abs(low_gap) > 1:
        step = np.clip(low_gap / 3.0, -3, 3)
        opt['eq_low'] = float(round(np.clip(audio.get('eq_low', 0) + step, -9, 6), 1))

    # ── EQ mid ────────────────────────────────────────────────────────────────
    mid_gap = ref['mid'] - clone['mid']
    if abs(mid_gap) > 2:
        step = np.clip(mid_gap / 4.0, -2, 2)
        opt['eq_mid'] = float(round(np.clip(audio.get('eq_mid', 0) + step, -6, 6), 1))

    # ── Compression — match crest factor ─────────────────────────────────────
    if 'crest' in ref and 'crest' in clone:
        crest_gap = ref['crest'] - clone['crest']
        if crest_gap > 2:
            opt['comp'] = float(round(np.clip(audio.get('comp', 0) - 0.05, 0, 1), 2))
        elif crest_gap < -2:
            opt['comp'] = float(round(np.clip(audio.get('comp', 0) + 0.05, 0, 1), 2))

    # NOTE: speed NOT optimised here — reference and clone have different texts

    # ── Highpass — bidirectional ──────────────────────────────────────────────
    if clone['low'] - ref['low'] > 5:
        opt['hp'] = float(np.clip(audio.get('hp', 0) + 10, 0, 200))
    elif ref['low'] - clone['low'] > 5 and audio.get('hp', 0) > 0:
        opt['hp'] = float(np.clip(audio.get('hp', 0) - 10, 0, 200))

    # ── Lowpass — bidirectional ───────────────────────────────────────────────
    high_gap = ref['high'] - clone['high']
    if clone['high'] - ref['high'] > 2:
        current_lp = audio.get('lp', 0) or 12000
        opt['lp'] = float(np.clip(current_lp - 500, 3000, 12000))
    elif high_gap > 2 and audio.get('lp', 0) > 0:
        opt['lp'] = float(np.clip(audio.get('lp', 0) + 500, 3000, 16000))

    return opt


# ── TTS generation ────────────────────────────────────────────────────────────

def set_seed(seed):
    import torch, random
    seed = int(seed or 0)
    if seed > 0:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def generate_clone(tts, speaker_wav, lang, text, xtts, audio_params, output_path,
                   gen_path=None):
    """Generate clone then apply audio processing via generator's process_audio."""
    import tempfile

    # Set seed
    set_seed(xtts.get('seed', 0))

    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()

    tts.tts_to_file(
        text=text,
        file_path=tmp.name,
        language=lang.lower().split('-')[0],
        speaker_wav=speaker_wav,
        temperature=float(xtts.get('temp', 0.72)),
        length_penalty=float(xtts.get('len_pen', 1.0)),
        repetition_penalty=float(xtts.get('rep_pen', 5.0)),
        top_k=int(xtts.get('top_k', 50)),
        top_p=float(xtts.get('top_p', 0.85)),
        gpt_cond_len=int(xtts.get('gpt_cond_len', 30)),
        speed=1.0,
    )

    # Apply audio processing
    _process_audio = None
    if gen_path and os.path.exists(gen_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location('gmg', gen_path)
        gmg  = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(gmg)
            _process_audio = getattr(gmg, 'process_audio', None)
        except Exception:
            pass

    from pydub import AudioSegment
    audio = AudioSegment.from_wav(tmp.name)
    os.unlink(tmp.name)

    if _process_audio:
        config = {
            'speed':           float(audio_params.get('speed', 1.0)),
            'volume':          int(audio_params.get('vol', 0)),
            'eq_low':          float(audio_params.get('eq_low', 0)),
            'eq_mid':          float(audio_params.get('eq_mid', 0)),
            'eq_high':         float(audio_params.get('eq_high', 0)),
            'highpass':        float(audio_params.get('hp', 0)),
            'lowpass':         float(audio_params.get('lp', 0)),
            'noise_reduction': float(audio_params.get('NR', 0)),
            'compression':     float(audio_params.get('comp', 0)),
            'deesser':         float(audio_params.get('de-ess', 0)),
            'reverb':          float(audio_params.get('reverb', 0)),
            'noise_gate':      float(audio_params.get('noise_gate', 0)),
            'pan':             float(audio_params.get('pan', 0)),
            'language':        lang.lower(),
        }
        xtts_fx = {
            'trim_start': int(xtts.get('trim_start', 0)),
            'trim_end':   int(xtts.get('trim_end', 0)),
            'fade_in':    int(xtts.get('fade_in', 100)),
            'fade_out':   int(xtts.get('fade_out', 250)),
            'limiter':    int(audio_params.get('limiter', 1)),
        }
        audio = _process_audio(audio, config, xtts_fx)

    audio.export(output_path, format='wav')
    return output_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='XTTS Voice Comparator')
    parser.add_argument('reference',   help='Reference voice WAV/MP3 — used for comparison AND XTTS cloning')
    parser.add_argument('voice_refs',  nargs='*', help='Additional voice refs (optional) + lang code last')
    parser.add_argument('--xtts-block',  required=True, help='XTTS {} block')
    parser.add_argument('--audio-block', required=True, help='Audio [] block')
    parser.add_argument('--text', default="Bonjour, voici un court test de comparaison de voix.",
                        help='Test sentence to generate')
    parser.add_argument('--text-file', default=None,
                        help='Text file with pause syntax support')
    parser.add_argument('--output', default=None, help='Output WAV for clone')
    parser.add_argument('--output-optimised', default=None,
                        help='Output WAV for optimised clone')
    parser.add_argument('--device', default=None)
    parser.add_argument('--no-generate', action='store_true',
                        help='Skip TTS generation (analyse only)')
    parser.add_argument('--iterations', type=int, default=1,
                        help='Number of optimisation iterations (default: 1)')
    parser.add_argument('--conv-threshold', type=float, default=0.5,
                        help='Stop if score improvement < this value (default: 0.5)')
    args = parser.parse_args()

    # Separate lang from voice refs
    voice_refs = args.voice_refs or []
    lang = 'FR'
    LANGS = {'FR','EN','ES','DE','IT','PT','PL','TR','RU','NL','CS','AR','HU','KO','JA','HI'}
    if voice_refs and voice_refs[-1].upper() in LANGS:
        lang = voice_refs[-1].upper()
        voice_refs = voice_refs[:-1]

    # If no extra voice refs — use reference itself as XTTS source
    if not voice_refs:
        voice_refs = [args.reference]

    speaker_wav = voice_refs if len(voice_refs) > 1 else voice_refs[0]

    # Parse blocks
    xtts  = parse_xtts_block(args.xtts_block)
    audio = parse_audio_block(args.audio_block)
    lang  = audio.get('lang', lang)

    # Text source
    if args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, encoding='utf-8') as fh:
            text_content = fh.read().strip()
        use_generator = True
    else:
        text_content = args.text
        use_generator = False

    print(f"\n{'='*62}")
    print(f"  XTTS Voice Comparator")
    print(f"{'='*62}")
    print(f"  Reference  : {os.path.basename(args.reference)}")
    print(f"  Language   : {lang}")
    print(f"  Text       : {text_content[:60]}{'...' if len(text_content)>60 else ''}")
    print(f"{'='*62}\n")

    # ── Iteration loop ────────────────────────────────────────────────────────
    tmpdir     = tempfile.mkdtemp()
    clone_path = args.output or os.path.join(tmpdir, 'clone.wav')
    xtts_str, _ = format_block(1, xtts, audio, lang)
    current_audio = dict(audio)
    best_score    = float('inf')
    prev_audio_str = None

    for iteration in range(1, args.iterations + 1):
        print(f"\n{'─'*62}")
        print(f"  Iteration {iteration}/{args.iterations}")
        print(f"{'─'*62}")

        _, current_audio_str = format_block(1, xtts, current_audio, lang)
        iter_clone = clone_path  # always overwrite clone
        print(f"  Current [] : {current_audio_str}")

        # Generate clone
        if not args.no_generate:
            gen_path = os.path.join(os.path.dirname(__file__), 'guided_meditation_generator_v23.py')
            print(f"[*] Generating clone...")
            if use_generator and os.path.exists(gen_path):
                import subprocess, tempfile as _tf2
                _p = _tf2.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
                _p.write(f"{xtts_str}\n{current_audio_str}\n{text_content}\n")
                _p.close()
                _refs = speaker_wav if isinstance(speaker_wav, list) else [speaker_wav]
                subprocess.run([sys.executable, gen_path, _p.name, iter_clone] + _refs, check=True)
                os.unlink(_p.name)
            else:
                if iteration == 1:
                    import torch
                    from TTS.api import TTS
                    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
                    print(f"[*] Loading XTTS on {device}...")
                    tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to(device)
                    print(f"[OK] TTS ready")
                generate_clone(tts, speaker_wav, lang, text_content, xtts, current_audio, iter_clone, gen_path)
            print(f"[OK] Clone: {iter_clone}")

        # Analyse
        print(f"\n[*] Analysing...")
        ref_y, sr = load_mono(args.reference)
        cln_y, _  = load_mono(iter_clone, target_sr=sr)
        ref_dur   = len(ref_y) / sr
        cln_dur   = len(cln_y) / sr
        ref_s = spectral(ref_y, sr)
        cln_s = spectral(cln_y, sr)
        score = compute_score(ref_s, cln_s)

        print(f"\n{'Metric':<14} {'Reference':>12} {'Clone':>12} {'Gap':>10}")
        print(f"{'':->52}")
        print(f"{'RMS (dB)':<14} {ref_s['rms']:>12.1f} {cln_s['rms']:>12.1f} {cln_s['rms']-ref_s['rms']:>+10.1f}")
        print(f"{'Crest (dB)':<14} {ref_s['crest']:>12.1f} {cln_s['crest']:>12.1f} {cln_s['crest']-ref_s['crest']:>+10.1f}")
        print(f"{'Centroid (Hz)':<14} {ref_s['centroid']:>12.0f} {cln_s['centroid']:>12.0f} {cln_s['centroid']-ref_s['centroid']:>+10.0f}")
        print(f"{'Low <300Hz (%)':<14} {ref_s['low']:>12.1f} {cln_s['low']:>12.1f} {cln_s['low']-ref_s['low']:>+10.1f}")
        print(f"{'Mid 300-3k (%)':<14} {ref_s['mid']:>12.1f} {cln_s['mid']:>12.1f} {cln_s['mid']-ref_s['mid']:>+10.1f}")
        print(f"{'Hi 3-8kHz (%)':<14} {ref_s['high']:>12.1f} {cln_s['high']:>12.1f} {cln_s['high']-ref_s['high']:>+10.1f}")
        print(f"\n  Score: {score:.1f}  (prev best: {best_score:.1f})")

        # Convergence checks
        improvement = best_score - score
        if iteration > 1:
            if improvement < args.conv_threshold:
                print(f"\n[*] Score improvement {improvement:.2f} < threshold {args.conv_threshold} → converged, stopping.")
                break
            if current_audio_str == prev_audio_str:
                print(f"\n[*] Audio params unchanged → converged, stopping.")
                break

        best_score    = score
        prev_audio_str = current_audio_str

        # Optimise for next iteration
        opt_audio = optimise_audio(ref_s, cln_s, current_audio, cln_dur, ref_dur)
        _, opt_audio_str = format_block(1, xtts, opt_audio, lang)

        changes = []
        for k in ['vol','eq_low','eq_mid','eq_high','hp','lp','comp']:
            b = current_audio.get(k, 0); a = opt_audio.get(k, 0)
            if round(b, 3) != round(a, 3):
                changes.append(f"{k}: {b}→{a}")
        if changes:
            print(f"  Changes    : {', '.join(changes)}")
            _, next_audio_str = format_block(1, xtts, opt_audio, lang)
            print(f"  Next []    : {next_audio_str}")
        else:
            print(f"  No changes needed.")

        current_audio = opt_audio

    # Final optimised output — always generate with best params found
    print(f"\n{'='*62}")
    print(f"  FINAL RESULT  (after {iteration} iteration{'s' if iteration>1 else ''})")
    print(f"{'='*62}")

    opt_path = args.output_optimised or os.path.join(tmpdir, 'clone_optimised.wav')
    _, final_audio_str = format_block(1, xtts, current_audio, lang)

    if not args.no_generate:
        print(f"\n[*] Generating final optimised clone...")
        gen_path = os.path.join(os.path.dirname(__file__), 'guided_meditation_generator_v23.py')
        if use_generator and os.path.exists(gen_path):
            import subprocess as _sp2, tempfile as _tf3
            _p3 = _tf3.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8')
            _p3.write(f"{xtts_str}\n{final_audio_str}\n{text_content}\n")
            _p3.close()
            _refs = speaker_wav if isinstance(speaker_wav, list) else [speaker_wav]
            _sp2.run([sys.executable, gen_path, _p3.name, opt_path] + _refs, check=True)
            os.unlink(_p3.name)
        else:
            generate_clone(tts, speaker_wav, lang, text_content, xtts, current_audio, opt_path, gen_path)
        opt_y, _ = load_mono(opt_path, target_sr=sr)
        opt_s    = spectral(opt_y, sr)
        final_score = compute_score(ref_s, opt_s)
        print(f"[OK] Optimised clone: {opt_path}")
        print(f"  Score final  : {final_score:.1f}  {'✅ improved' if final_score < best_score else '⚠️ same'}")

    xtts_out, audio_out = format_block(1, xtts, current_audio, lang)
    print(f"\n  OPTIMISED PARAMS:")
    print(f"  {xtts_out}")
    print(f"  {audio_out}")
    print(f"\n[OK] Done.")



if __name__ == '__main__':
    main()

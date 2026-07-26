#!/usr/bin/env python3
"""
voice_validator.py — XTTS parameter validation script.

Generates multiple variations of a single sentence with different parameter
values, concatenates them with silence and spoken labels, so you can listen
and pick the best combination for a given voice.

Usage:
    python voice_validator.py <voice_ref.wav> <lang> [options]

Examples:
    # Test different seeds
    python voice_validator.py Elo3.wav FR --param seed --values 0 7 13 42 100 200

    # Test different temperatures
    python voice_validator.py Elo3.wav FR --param temp --values 0.5 0.65 0.72 0.85 1.0

    # Test different rep_pen values
    python voice_validator.py Elo3.wav FR --param rep_pen --values 3.0 4.5 6.0 8.0 10.0

    # Test multiple refs (space-separated)
    python voice_validator.py Elo.wav Elo2.wav Elo3.wav FR --param seed --values 0 7 42
"""

import argparse
import os
import sys
import tempfile
import time

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_TEXT = "Bonjour, ceci est un test de validation de la voix."
SILENCE_MS   = 800   # silence between variations
LABEL_PAUSE  = 400   # silence after spoken label

DEFAULT_PARAMS = {
    # XTTS params
    'seed'        : 42,
    'temp'        : 0.72,
    'top_k'       : 50,
    'top_p'       : 0.85,
    'rep_pen'     : 5.0,
    'len_pen'     : 1.0,
    'gpt_cond_len': 30,
    # Audio params
    'speed'       : 1.0,
    'vol'         : 0,
    'eq_low'      : 0,
    'eq_mid'      : 0,
    'eq_high'     : 0,
    'hp'          : 0,
    'lp'          : 0,
    'NR'          : 0,
    'comp'        : 0,
    'de-ess'      : 0,
    'reverb'      : 0,
    'noise_gate'  : 0,
    'pan'         : 0,
}

PARAM_ALIASES = {
    'temperature'       : 'temp',
    'repetition_penalty': 'rep_pen',
    'length_penalty'    : 'len_pen',
}

# Audio params — applied post-processing (not passed to XTTS directly)
AUDIO_PARAMS = {'speed','vol','eq_low','eq_mid','eq_high','hp','lp','NR','comp','de-ess','reverb','noise_gate','pan'}


# ── Helpers ───────────────────────────────────────────────────────────────────

def format_xtts_block(p):
    """Build a 14-value {} block string from a params dict (to paste into the comparator)."""
    def f(v):
        v = float(v)
        return str(int(v)) if v == int(v) else str(round(v, 3))
    order = [1, p.get('seed', 0), p.get('trim_start', 0), p.get('trim_end', 0),
             p.get('fade_in', 100), p.get('fade_out', 250), p.get('temp', 0.65),
             p.get('top_k', 50), p.get('top_p', 0.85), p.get('rep_pen', 5.0),
             p.get('len_pen', 1.0), p.get('gpt_cond_len', 30),
             p.get('gpt_cond_chunk_len', 6), p.get('sound_norm_refs', 0)]
    return '{' + ', '.join(f(v) for v in order) + '}'


def speak_label(tts, speaker_wav, language, label, output_path):
    """Generate a spoken label for a variation."""
    tts.tts_to_file(
        text=label,
        file_path=output_path,
        language=language.lower().split('-')[0],
        speaker_wav=speaker_wav,
        temperature=0.5,
        repetition_penalty=5.0,
        top_k=30,
        top_p=0.8,
        speed=1.0,
    )


def apply_speed_rubberband(audio, speed):
    """Apply speed change via rubberband if speed != 1.0."""
    if abs(speed - 1.0) < 0.01:
        return audio
    from pydub import AudioSegment
    import subprocess, tempfile as tf
    with tf.NamedTemporaryFile(suffix='_in.wav', delete=False) as fin:
        audio.export(fin.name, format='wav')
        fin_path = fin.name
    fout_path = fin_path.replace('_in.wav', '_out.wav')
    subprocess.run(
        ['rubberband', '-t', str(1.0/speed), fin_path, fout_path],
        capture_output=True
    )
    result = AudioSegment.from_wav(fout_path)
    os.unlink(fin_path)
    os.unlink(fout_path)
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='XTTS parameter validator')
    parser.add_argument('voice_refs', nargs='+',
                        help='Voice reference WAV file(s). Last arg before --param must be language code.')
    parser.add_argument('--param', action='append', dest='params',
                        help='Parameter to test (can be repeated)')
    parser.add_argument('--values', action='append', dest='values_list', nargs='+',
                        help='Values for the corresponding --param (can be repeated)')
    parser.add_argument('--text', default=DEFAULT_TEXT,
                        help=f'Sentence to generate (default: "{DEFAULT_TEXT}")')
    parser.add_argument('--output', default=None,
                        help='Output WAV file (default: validation_<param>.wav)')
    parser.add_argument('--lang', default=None,
                        help='Language code (FR, EN, etc.) — can also be last positional arg')
    parser.add_argument('--device', default=None,
                        help='cuda or cpu (auto-detected if not specified)')
    parser.add_argument('--gpt-cond-len', type=int, default=30,
                        help='GPT conditioning length in seconds (default: 30)')
    parser.add_argument('--xtts-block', default=None,
                        help='XTTS {} block from voice_analyser (e.g. "{1, 42, 0, 200, ...}")')
    parser.add_argument('--audio-block', default=None,
                        help='Audio [] block from voice_analyser (e.g. "[1, FR, 0.9, 6, ...]")')
    parser.add_argument('--blind', action='store_true',
                        help='Blind audition: randomise variant order and announce only '
                             '"Variante N" instead of the settings, so the label cannot '
                             'bias the ear. The key is printed after the ranking.')
    parser.add_argument('--blind-seed', type=int, default=0,
                        help='Seed for the blind shuffle (change it to re-randomise)')
    parser.add_argument('--no-labels', action='store_true',
                        help='Do not prepend spoken labels to each variation')
    parser.add_argument('--no-score', action='store_true',
                        help='Disable accent/identity scoring of the variants')
    parser.add_argument('--whisper-model', default='small',
                        help='faster-whisper model for accent scoring (default: small)')
    parser.add_argument('--whisper-device', default='cpu',
                        help="Device for Whisper scoring (default: cpu — frees VRAM)")
    parser.add_argument('--max-ref-len', type=int, default=30,
                        help='Seconds of reference for the speaker embedding (XTTS default 10)')
    parser.add_argument('--prioritise', default='accent',
                        choices=['accent', 'identity', 'balanced'],
                        help='Which score ranks the winning {} (default: accent)')

    args = parser.parse_args()

    # Separate language from voice refs
    voice_refs = args.voice_refs
    lang = args.lang
    if lang is None:
        # Last positional arg might be a language code
        if voice_refs and voice_refs[-1].upper() in (
            'FR','EN','ES','DE','IT','PT','PL','TR','RU','NL','CS','AR','ZH-CN','HU','KO','JA','HI'
        ):
            lang = voice_refs[-1].upper()
            voice_refs = voice_refs[:-1]
        else:
            lang = 'FR'
            print(f"[!] No language specified, defaulting to FR")

    # Validate voice refs
    for ref in voice_refs:
        if not os.path.exists(ref):
            print(f"[ERR] Voice reference not found: {ref}")
            sys.exit(1)

    speaker_wav = voice_refs if len(voice_refs) > 1 else voice_refs[0]

    # Build list of (param_key, [values]) pairs
    if not args.params or not args.values_list:
        print("[ERR] At least one --param and --values required.")
        sys.exit(1)

    param_value_pairs = []
    for p, vals in zip(args.params, args.values_list):
        param_key = PARAM_ALIASES.get(p, p)
        if param_key not in DEFAULT_PARAMS:
            print(f"[ERR] Unknown parameter '{p}'.")
            sys.exit(1)
        param_value_pairs.append((param_key, vals))  # store raw, resolve after base_params

    # Cartesian product of all param combinations
    output_file = args.output or f"validation_{'_'.join(p for p, _ in param_value_pairs)}.wav"

    # Parse XTTS {} block if provided
    base_params = DEFAULT_PARAMS.copy()
    if args.xtts_block:
        try:
            import re as _re
            nums = [float(x) for x in _re.findall(r'[-+]?\d+(?:\.\d+)?', args.xtts_block)]
            keys_xtts = ['_N','seed','trim_start','trim_end','fade_in','fade_out',
                         'temp','top_k','top_p','rep_pen','len_pen','gpt_cond_len',
                         'gpt_cond_chunk_len','sound_norm_refs']
            for i, k in enumerate(keys_xtts):
                if i < len(nums) and k != '_N':
                    base_params[k] = nums[i]
            print(f"[*] XTTS block parsed: {len(nums)} values")
        except Exception as e:
            print(f"[!] Could not parse XTTS block: {e}")

    if args.audio_block:
        try:
            import re as _re
            # Extract lang
            lang_match = _re.search(r'\b(FR|EN|ES|DE|IT|PT|PL|TR|RU|NL|CS|AR|HU|KO|JA|HI)\b',
                                    args.audio_block, _re.IGNORECASE)
            if lang_match:
                lang = lang_match.group(1).upper()
            nums = [float(x) for x in _re.findall(r'[-+]?\d+(?:\.\d+)?', args.audio_block)]
            keys_audio = ['speed','vol','eq_low','eq_mid','eq_high',
                          'hp','lp','NR','comp','de-ess','reverb','noise_gate','pan','limiter']
            # regex finds all numbers — first one is N, skip it
            all_nums = [float(x) for x in _re.findall(r'[-+]?\d+(?:\.\d+)?', args.audio_block)]
            numeric_vals = all_nums[1:]  # skip N
            for i, k in enumerate(keys_audio):
                if i < len(numeric_vals) and k != 'limiter':
                    base_params[k] = numeric_vals[i]
            print(f"[*] Audio block parsed: {len(nums)} values, lang={lang}")
        except Exception as e:
            print(f"[!] Could not parse audio block: {e}")

    # Resolve _base sentinel and convert values to float
    param_value_pairs = [(pk, [base_params.get(pk, DEFAULT_PARAMS.get(pk, 0))] if vals == ['_base']
                          else [float(v) for v in vals])
                         for pk, vals in param_value_pairs]
    param_keys  = [p for p, _ in param_value_pairs]
    import itertools
    all_combos  = list(itertools.product(*[vals for _, vals in param_value_pairs]))

    print(f"\n{'='*60}")
    print(f"  XTTS Voice Validator")
    print(f"{'='*60}")
    print(f"  Voice refs : {[os.path.basename(r) for r in voice_refs] if isinstance(voice_refs, list) else os.path.basename(voice_refs)}")
    print(f"  Language   : {lang}")
    for p, vals in param_value_pairs:
        print(f"  {p:<12}: {vals}")
    print(f"  Combinations: {len(all_combos)}")
    print(f"  Text       : {args.text}")
    print(f"  Output     : {output_file}")
    print(f"{'='*60}\n")

    # Show base params
    _bp_str = '  '.join(f'{p}={base_params.get(p, "N/A")}' for p in param_keys)
    print(f"[*] Base params: {_bp_str}")
    import torch
    import xtts_clone as XC
    from pydub import AudioSegment

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Loading XTTS on {device}...")
    tts, model = XC.load_xtts(device)   # tts kept for spoken labels; model for generation
    print(f"[OK] TTS ready\n")

    # Latents honour the cloning knobs (tts_to_file would ignore them); cached so
    # they are recomputed only when a cloning param (gpt_cond_len/chunk/
    # sound_norm_refs) is the one being swept.
    _lat_cache = {}
    def _latents_for(params):
        key = (int(params.get('gpt_cond_len', args.gpt_cond_len)),
               int(params.get('gpt_cond_chunk_len', 6)),
               int(args.max_ref_len),
               bool(int(params.get('sound_norm_refs', 0))))
        if key not in _lat_cache:
            _lat_cache[key] = XC.compute_latents(
                model, speaker_wav, gpt_cond_len=key[0], gpt_cond_chunk_len=key[1],
                max_ref_len=key[2], sound_norm_refs=key[3])
        return _lat_cache[key]

    def gen_variant(params, out_path):
        XC.generate(model, args.text, lang, _latents_for(params), out_path,
                    temperature=float(params.get('temp', 0.65)),
                    length_penalty=float(params.get('len_pen', 1.0)),
                    repetition_penalty=float(params.get('rep_pen', 5.0)),
                    top_k=int(params.get('top_k', 50)),
                    top_p=float(params.get('top_p', 0.85)),
                    num_beams=int(params.get('num_beams', 1)),
                    speed=1.0, seed=int(params.get('seed', 0) or 0))

    silence = AudioSegment.silent(duration=SILENCE_MS)
    label_pause = AudioSegment.silent(duration=LABEL_PAUSE)
    final_audio = AudioSegment.silent(duration=500)  # intro silence

    tmpdir = tempfile.mkdtemp()

    # Import process_audio from generator
    gen_path = os.path.join(os.path.dirname(__file__), 'guided_meditation_generator_v23.py')
    _process_audio = None
    _apply_speed   = None
    if os.path.exists(gen_path):
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location('gmg', gen_path)
        _gmg  = _ilu.module_from_spec(_spec)
        try:
            _spec.loader.exec_module(_gmg)
            _process_audio = getattr(_gmg, 'process_audio', None)
            _apply_speed   = getattr(_gmg, 'apply_speed_rubberband', None)
            if _process_audio:
                print("[*] process_audio imported from generator")
        except Exception as e:
            print(f"[!] Could not import generator: {e}")

    def apply_audio(audio, params):
        """Apply audio post-processing using base_params + current variation."""
        if _apply_speed:
            spd = float(params.get('speed', 1.0))
            if abs(spd - 1.0) > 0.01:
                audio = _apply_speed(audio, spd)
        if _process_audio:
            # Build config and xtts_params dicts as expected by process_audio
            config = {
                'speed':          float(params.get('speed', 1.0)),
                'volume':         int(params.get('vol', 0)),
                'eq_low':         float(params.get('eq_low', 0)),
                'eq_mid':         float(params.get('eq_mid', 0)),
                'eq_high':        float(params.get('eq_high', 0)),
                'highpass':       float(params.get('hp', 0)),
                'lowpass':        float(params.get('lp', 0)),
                'noise_reduction':float(params.get('NR', 0)),
                'compression':    float(params.get('comp', 0)),
                'deesser':        float(params.get('de-ess', 0)),
                'reverb':         float(params.get('reverb', 0)),
                'noise_gate':     float(params.get('noise_gate', 0)),
                'pan':            float(params.get('pan', 0)),
                'language':       lang.lower(),
            }
            xtts_p = {
                'trim_start': int(params.get('trim_start', 0)),
                'trim_end':   int(params.get('trim_end', 0)),
                'fade_in':    int(params.get('fade_in', 100)),
                'fade_out':   int(params.get('fade_out', 250)),
                'limiter':    1,
            }
            audio = _process_audio(audio, config, xtts_p)
        else:
            # Fallback — basic volume only
            vol = int(params.get('vol', 0))
            if vol != 0:
                audio = audio + vol
        return audio

    # For audio-only params: generate base audio once
    audio_only_params = AUDIO_PARAMS
    all_audio = all(p in audio_only_params for p in param_keys)

    # Audio-only sweeps ARE scorable: the clone is identical, but post-processing
    # changes the rendered spectrum, and ECAPA identity is measured on that.
    # (Accent may also move if heavy processing hurts intelligibility — useful.)
    do_score = not args.no_score
    pron = None; enc = None; ref_emb = None; scores = []
    if do_score:
        ref0 = voice_refs[0] if isinstance(voice_refs, list) else voice_refs
        try:
            from pron_score import PronScorer
            print(f"[*] Loading faster-whisper '{args.whisper_model}' on {args.whisper_device} (accent)...")
            pron = PronScorer(device=args.whisper_device, model=args.whisper_model)
        except Exception as e:
            print(f"[!] Accent scoring off ({e})")
        try:
            from speaker_identity import SpeakerEncoder
            print("[*] Loading ECAPA-TDNN (identity)...")
            enc = SpeakerEncoder(device='cpu' if device == 'cuda' else device)
            ref_emb = enc.embed(ref0)
        except Exception as e:
            print(f"[!] Identity scoring off ({e})")
        if pron is None and enc is None:
            do_score = False

    if all_audio:
        print(f"[*] All audio params — generating base audio once\n")
        base_wav = os.path.join(tmpdir, 'base.wav')
        gen_variant(base_params, base_wav)
        base_audio_raw = AudioSegment.from_wav(base_wav)
        print(f"[OK] Base audio: {len(base_audio_raw)/1000:.1f}s\n")

    # Blind mode: hearing "N R zero point three" just before a clip biases the
    # judgement. Randomise the order, drop the spoken labels, and reveal the key
    # only at the end — so the ear decides before the label does.
    order = list(range(len(all_combos)))
    if args.blind:
        import random as _rnd
        _rnd.Random(args.blind_seed).shuffle(order)
        print(f"[*] BLIND mode: {len(order)} variants in random order, no spoken "
              f"labels. The key is printed at the end.\n")
    blind_key = []
    noise_floor = [None]

    for pos, i in enumerate(order):
        combo = all_combos[i]
        params = base_params.copy()
        for pk, pv in zip(param_keys, combo):
            params[pk] = pv

        combo_str = '  '.join(f"{pk}={pv}" for pk, pv in zip(param_keys, combo))
        blind_key.append((pos + 1, combo_str))
        if args.blind:
            print(f"[{pos+1}/{len(all_combos)}] (hidden)")
        else:
            print(f"[{pos+1}/{len(all_combos)}] {combo_str}")

        # Spoken label ("variant N" only in blind mode, so you can still count)
        if args.blind:
            label_path = os.path.join(tmpdir, f'label_{i}.wav')
            try:
                speak_label(tts, speaker_wav, lang, f"Variante {pos+1}.", label_path)
                final_audio = final_audio + AudioSegment.from_wav(label_path) + label_pause
            except Exception as e:
                print(f"   [!] Label failed: {e}")
        elif not args.no_labels:
            label_text = '  '.join(f"{pk} {pv}" for pk, pv in zip(param_keys, combo)) + '.'
            label_path = os.path.join(tmpdir, f'label_{i}.wav')
            try:
                speak_label(tts, speaker_wav, lang, label_text, label_path)
                label_audio = AudioSegment.from_wav(label_path)
                final_audio = final_audio + label_audio + label_pause
            except Exception as e:
                print(f"   [!] Label failed: {e}")

        t0 = time.time()
        try:
            if all_audio:
                audio = apply_audio(base_audio_raw, params)
            else:
                out_path = os.path.join(tmpdir, f'var_{i}.wav')
                gen_variant(params, out_path)
                if device == 'cuda':
                    try: torch.cuda.empty_cache()
                    except Exception: pass
                audio = AudioSegment.from_wav(out_path)
                audio = apply_audio(audio, params)

            # Score the audio you will actually HEAR — i.e. AFTER post-processing.
            # Scoring the raw clone meant the [] parameters could never influence
            # the ranking, and audio-only sweeps were left unscored entirely.
            # ECAPA reads the processed spectrum, so EQ/NR/filters DO move identity.
            if do_score:
                scored_path = os.path.join(tmpdir, f'scored_{i}.wav')
                audio.export(scored_path, format='wav')
                rec = {'combo': list(zip(param_keys, combo)),
                       'fr': None, 'detected': '', 'wer': None, 'cos': None}
                if pron is not None:
                    pr = pron.score(scored_path, lang=lang, target_text=args.text)
                    rec['fr'] = pr['score']; rec['detected'] = pr['detected']; rec['wer'] = pr['wer']
                if enc is not None:
                    rec['cos'] = enc.cosine(ref_emb, enc.embed(scored_path))
                    # Noise floor, measured once: identity of each half of the
                    # same render. Same voice, so their spread is this clip's
                    # measurement variability — the bar a difference must clear
                    # before it means anything.
                    if noise_floor[0] is None:
                        try:
                            import soundfile as _sf
                            _y, _sr = _sf.read(scored_path)
                            if getattr(_y, 'ndim', 1) > 1:
                                _y = _y.mean(axis=1)
                            _h = len(_y) // 2
                            noise_floor[0] = max(0.004, abs(
                                float(enc.cosine(ref_emb, enc.embed(_y[:_h], sr=_sr))) -
                                float(enc.cosine(ref_emb, enc.embed(_y[_h:], sr=_sr)))) / 2.0)
                            print(f"   [noise floor] identity differences below "
                                  f"±{noise_floor[0]:.4f} are not meaningful")
                        except Exception:
                            noise_floor[0] = 0.004
                scores.append(rec)
                sc = (f"french={rec['fr']:.3f} " if rec['fr'] is not None else "") + \
                     (f"identity={rec['cos']:.4f}" if rec['cos'] is not None else "")
                print(f"   [score] {sc}")

            elapsed = time.time() - t0
            final_audio = final_audio + audio + silence
            print(f"   [OK] {len(audio)/1000:.1f}s in {elapsed:.1f}s")
        except Exception as e:
            print(f"   [ERR] {e}")
            import traceback; traceback.print_exc()

    # ── Ranked scores + winning {} block to paste into the comparator ─────────
    if scores:
        def rankkey(r):
            fr = r['fr'] if r['fr'] is not None else -1.0
            co = r['cos'] if r['cos'] is not None else -1.0
            if args.prioritise == 'identity': return (co, fr)
            if args.prioritise == 'balanced': return (0.6 * fr + 0.4 * ((co + 1) / 2),)
            return (fr, co)   # accent first
        ranked = sorted(scores, key=rankkey, reverse=True)
        print(f"\n{'='*60}\n  RANKING (by {args.prioritise})\n{'='*60}")
        hdr = '  '.join(f"{k:>10}" for k in param_keys)
        print(f"  {hdr}{'french':>10}{'wer':>7}{'identity':>11}")
        nf = noise_floor[0] or 0.004
        top = ranked[0]
        n_tied = 0
        for r in ranked:
            vals = '  '.join(f"{str(v):>10}" for _, v in r['combo'])
            fr = f"{r['fr']:.3f}" if r['fr'] is not None else "  -  "
            we = f"{r['wer']:.2f}" if r['wer'] is not None else "  -  "
            co = f"{r['cos']:.4f}" if r['cos'] is not None else "   -   "
            tie = ''
            if (r is not top and r['cos'] is not None and top['cos'] is not None
                    and abs(top['cos'] - r['cos']) <= nf):
                tie = ' ='; n_tied += 1
            print(f"  {vals}{fr:>10}{we:>7}{co:>11}{tie}")
        if n_tied:
            print(f"\n  '=' marks {n_tied} variant(s) whose identity is within the "
                  f"±{nf:.4f} noise floor of the best —")
            print(f"  the measurement cannot separate them. THIS is where your ear "
                  f"decides, not the ranking.")

        winner = base_params.copy()
        for k, v in ranked[0]['combo']:
            winner[k] = v
        print(f"\n  Best {args.prioritise}: " +
              '  '.join(f"{k}={v}" for k, v in ranked[0]['combo']))
        swept_audio = [k for k in param_keys if k in AUDIO_PARAMS]
        if swept_audio:
            # Audio params live in the [] block, not the {} one.
            a = dict(base_params)
            for k, v in ranked[0]['combo']:
                if k in AUDIO_PARAMS:
                    a[k] = v
            order = ['speed', 'vol', 'eq_low', 'eq_mid', 'eq_high', 'hp', 'lp',
                     'NR', 'comp', 'de-ess', 'reverb', 'noise_gate', 'pan', 'limiter']
            def _f(x):
                x = float(x)
                return str(int(x)) if abs(x - round(x)) < 1e-9 else f"{x:g}"
            vals = ', '.join(_f(a.get(k, 1 if k in ('speed', 'limiter') else 0))
                             for k in order)
            print(f"  Paste into the generator / comparator (audio block):")
            print(f"  [1, {lang}, {vals}]")
        if any(k not in AUDIO_PARAMS for k in param_keys):
            print(f"  Paste into the comparator (frozen seed/temp/...):")
            print(f"  {format_xtts_block(winner)}")

    if args.blind and blind_key:
        print(f"\n{'='*60}\n  BLIND KEY (read AFTER listening)\n{'='*60}")
        for n, desc in blind_key:
            print(f"  Variante {n}: {desc}")

    # Save
    print(f"\n[*] Saving {output_file}...")
    final_audio.export(output_file, format='wav')
    duration = len(final_audio) / 1000
    print(f"[OK] Done — {duration:.1f}s total ({len(all_combos)} combinations)")
    print(f"[*] Output: {os.path.abspath(output_file)}")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()

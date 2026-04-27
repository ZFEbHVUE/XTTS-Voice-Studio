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

def set_seed(tts_model, seed):
    import torch, random, numpy as np
    if seed and seed > 0:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def generate_one(tts, speaker_wav, language, text, params, output_path, gpt_cond_len=30):
    """Generate a single sentence with given params."""
    seed = int(params.get('seed', 0) or 0)
    set_seed(tts, seed)

    tts.tts_to_file(
        text=text,
        file_path=output_path,
        language=language.lower().split('-')[0],
        speaker_wav=speaker_wav,
        temperature=float(params.get('temp', 0.72)),
        length_penalty=float(params.get('len_pen', 1.0)),
        repetition_penalty=float(params.get('rep_pen', 5.0)),
        top_k=int(params.get('top_k', 50)),
        top_p=float(params.get('top_p', 0.85)),
        gpt_cond_len=int(params.get('gpt_cond_len', gpt_cond_len)),
        speed=1.0,  # always 1.0 — rubberband applied below if needed
    )


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
    parser.add_argument('--no-labels', action='store_true',
                        help='Do not prepend spoken labels to each variation')

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
        param_value_pairs.append((param_key, [float(v) for v in vals]))

    # Cartesian product of all param combinations
    import itertools
    all_combos = list(itertools.product(*[vals for _, vals in param_value_pairs]))
    param_keys = [p for p, _ in param_value_pairs]

    output_file = args.output or f"validation_{'_'.join(param_keys)}.wav"

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

    # Show base params
    _bp_str = '  '.join(f'{p}={base_params.get(p, "N/A")}' for p in param_keys)
    print(f"[*] Base params: {_bp_str}")
    import torch
    from TTS.api import TTS
    from pydub import AudioSegment

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Loading XTTS on {device}...")
    tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to(device)
    print(f"[OK] TTS ready\n")

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

    if all_audio:
        print(f"[*] All audio params — generating base audio once\n")
        base_wav = os.path.join(tmpdir, 'base.wav')
        generate_one(tts, speaker_wav, lang, args.text, base_params, base_wav,
                     gpt_cond_len=int(args.gpt_cond_len))
        base_audio_raw = AudioSegment.from_wav(base_wav)
        print(f"[OK] Base audio: {len(base_audio_raw)/1000:.1f}s\n")

    for i, combo in enumerate(all_combos):
        params = base_params.copy()
        for pk, pv in zip(param_keys, combo):
            params[pk] = pv

        combo_str = '  '.join(f"{pk}={pv}" for pk, pv in zip(param_keys, combo))
        print(f"[{i+1}/{len(all_combos)}] {combo_str}")

        # Spoken label
        if not args.no_labels:
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
                generate_one(tts, speaker_wav, lang, args.text, params, out_path,
                             gpt_cond_len=int(args.gpt_cond_len))
                audio = AudioSegment.from_wav(out_path)
                audio = apply_audio(audio, params)

            elapsed = time.time() - t0
            final_audio = final_audio + audio + silence
            print(f"   [OK] {len(audio)/1000:.1f}s in {elapsed:.1f}s")
        except Exception as e:
            print(f"   [ERR] {e}")
            import traceback; traceback.print_exc()

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

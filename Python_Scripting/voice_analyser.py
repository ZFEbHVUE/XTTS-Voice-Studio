#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Voice Analyser - XTTS parameter optimiser for guided_meditation_generator_v23.py
==================================================================================
Analyses acoustic features of a reference WAV file and outputs ready-to-paste
parameters for both bracket types:

  {N, seed, trim_start, trim_end, fade_in, fade_out, temp, top_k, top_p, rep_pen, len_pen, gpt_cond_len, gpt_cond_chunk_len, sound_norm_refs}
  [N, LANG, speed, vol, eq_low, eq_mid, eq_high, hp, lp, NR, comp, de-ess, reverb, noise_gate, pan, limiter]

Usage:
  python voice_analyser.py voice.wav LANG [voice2.wav LANG2 ...]
  python voice_analyser.py --precise voice.wav FR
  python voice_analyser.py --start-num 3 voice.wav FR

  LANG is optional -- defaults to FR if omitted.
  Supported: FR EN ES DE IT PT PL TR RU NL CS AR ZH-CN HU KO JA HI

Examples:
  python voice_analyser.py Monique.wav FR
  python voice_analyser.py Monique.wav FR John.wav EN
  python voice_analyser.py --precise FannyArdant.wav FR

Estimated duration (30s file, standard CPU):
  Default mode  : 3-10s  (yin algorithm)
  --precise     : 30-90s (pyin algorithm, much slower)

Dependencies:
  pip install librosa numpy scipy
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
except ImportError:
    sys.exit("[!] numpy missing -> pip install numpy")

try:
    import librosa
except ImportError:
    sys.exit("[!] librosa missing -> pip install librosa")

try:
    from scipy.stats import kurtosis as sp_kurtosis
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# -----------------------------------------------------------------------------
# Progress helper
# -----------------------------------------------------------------------------

def step(msg, t0=None):
    """Print a step label with elapsed time when t0 is provided."""
    if t0 is not None:
        print(f"   [OK] [{time.time()-t0:5.1f}s]  {msg}", flush=True)
    else:
        print(f"   [*]  {msg} ...", end=' ', flush=True)


# -----------------------------------------------------------------------------
# Spectral helpers
# -----------------------------------------------------------------------------

def band_energy(S, freqs, f_low, f_high):
    """Mean amplitude energy in a frequency band."""
    mask = (freqs >= f_low) & (freqs <= f_high)
    return float(np.mean(S[mask, :])) if mask.any() else 0.0


def fmt(v):
    """Format a scalar without trailing zeros."""
    if isinstance(v, int):
        return str(v)
    if float(v) == int(float(v)):
        return str(int(float(v)))
    return f"{v:.2f}".rstrip('0').rstrip('.')


# -----------------------------------------------------------------------------
# Core analysis
# -----------------------------------------------------------------------------

def analyse_voice(wav_file, fast=True):
    """
    Analyse an XTTS reference WAV file.

    fast=True  -> yin algorithm  (default, 3-10s)
    fast=False -> pyin algorithm (--precise, 30-90s, more accurate)

    Returns (params_dict, stats_dict).
    """
    t0 = time.time()

    print(f"\n[*] Analysing -> {os.path.basename(wav_file)}")
    print(f"   {'[fast] yin algorithm' if fast else '[precise] pyin algorithm (may take several minutes)'}")
    print("-" * 62)

    # -- Load --------------------------------------------------------------
    step("Loading audio")
    y, sr = librosa.load(wav_file, sr=None, mono=True)
    duration = len(y) / sr
    step(f"Loaded  ({duration:.1f}s, {sr} Hz)", t0)

    # -- 1. XTTS artifact detection -> trim_start / trim_end ---------------
    step("Detecting silence / XTTS artifacts")
    frame_ms  = 10
    frame_len = int(sr * frame_ms / 1000)
    hop_len   = frame_len // 2

    rms       = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    threshold = rms.max() * 0.04   # 4 % of peak = silence / artifact

    start_frame = next((i for i, r in enumerate(rms) if r > threshold), 0)
    end_frame   = next((i for i, r in enumerate(reversed(rms)) if r > threshold), 0)

    natural_start_ms = int(start_frame * hop_len / sr * 1000)
    natural_end_ms   = int(end_frame   * hop_len / sr * 1000)

    # XTTS adds 30-80 ms artifact at start, 150-400 ms tail at end
    trim_start = int(np.clip(natural_start_ms + 40,  60, 160))
    trim_end   = int(np.clip(natural_end_ms   + 120, 200, 420))

    print(f"   Natural silence: start {natural_start_ms}ms  end {natural_end_ms}ms")
    step(f"Silence  -> trim_start={trim_start}ms  trim_end={trim_end}ms", t0)

    # -- 2. F0 estimation -> voice type, hp, lp ----------------------------
    step("F0 estimation  (longest step...)")
    if fast:
        f0_raw      = librosa.yin(y,
                                   fmin=librosa.note_to_hz('C2'),
                                   fmax=librosa.note_to_hz('C6'),
                                   sr=sr, hop_length=hop_len)
        # Energy gate: discard frames below 8% of peak RMS to avoid
        # spurious YIN F0 estimates in silence/noise frames inflating voiced_ratio
        rms_f0      = librosa.feature.rms(y=y, frame_length=frame_len*2,
                                           hop_length=hop_len)[0]
        # Align lengths (YIN may produce slightly different frame count)
        min_len     = min(len(f0_raw), len(rms_f0))
        f0_raw      = f0_raw[:min_len]
        rms_f0      = rms_f0[:min_len]
        energy_gate = rms_f0 > (rms_f0.max() * 0.08)
        voiced_flag = (energy_gate &
                       (f0_raw > librosa.note_to_hz('C2')) &
                       (f0_raw < librosa.note_to_hz('C6')))
        f0 = np.where(voiced_flag, f0_raw, np.nan)
    else:
        f0, voiced_flag, _ = librosa.pyin(y,
                                           fmin=librosa.note_to_hz('C2'),
                                           fmax=librosa.note_to_hz('C6'),
                                           sr=sr, hop_length=hop_len)

    f0_voiced = f0[voiced_flag & ~np.isnan(f0)]
    if len(f0_voiced) < 5:
        print(f"\n   [!] Very few voiced frames detected ({len(f0_voiced)}) "
              f"-- file may be too short or mostly unvoiced. "
              f"Falling back to neutral defaults (150 Hz / tenor).")
        f0_median, f0_std, f0_jitter, voiced_ratio = 150.0, 20.0, 0.15, 0.5
    else:
        f0_median    = float(np.median(f0_voiced))
        f0_std       = float(np.std(f0_voiced))
        f0_jitter    = f0_std / f0_median if f0_median > 0 else 0.15
        voiced_ratio = float(np.mean(voiced_flag))
    step(f"F0={f0_median:.0f}Hz  std={f0_std:.0f}Hz  jitter={f0_jitter:.3f}  voiced={voiced_ratio:.0%}", t0)

    # Voice type classification
    if f0_median < 110:
        voice_type = "bass / deep male"
        highpass, lowpass = 55, 7000
    elif f0_median < 155:
        voice_type = "baritone / contralto"
        highpass, lowpass = 65, 7500
    elif f0_median < 210:
        voice_type = "tenor / mezzo-soprano"
        highpass, lowpass = 75, 8000
    else:
        voice_type = "soprano / high voice"
        highpass, lowpass = 90, 9000

    print(f"   Voice type: {voice_type}")

    # -- 3. Spectral analysis -> EQ -----------------------------------------
    step("Spectral analysis (EQ, sibilance, breathiness)")
    S     = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    e_low      = band_energy(S, freqs,   80,  300)
    e_bm       = band_energy(S, freqs,  300,  800)   # body-mid (warmth)
    e_mid      = band_energy(S, freqs,  800, 3000)   # presence (used for eq_mid)
    e_high     = band_energy(S, freqs, 3000, 8000)

    ref          = e_mid if e_mid > 0 else 1e-9
    ratio_low    = e_low  / ref
    ratio_high   = e_high / ref
    ratio_bm     = e_bm   / ref

    # EQ lows (80-300 Hz)
    if   ratio_low > 1.6: eq_low = -5
    elif ratio_low > 1.2: eq_low = -3
    elif ratio_low > 0.8: eq_low = -2
    elif ratio_low > 0.5: eq_low = -1
    else:                 eq_low = +1

    # EQ mids (presence band 800-3000 Hz) -- compare against body-mid (300-800 Hz)
    # High ratio_bm means warm/boomy mid -> less boost needed on presence
    if   ratio_bm > 1.5: eq_mid = +1
    elif ratio_bm > 1.0: eq_mid = +2
    else:                eq_mid = +3

    # EQ highs (3000-8000 Hz)
    if   ratio_high > 1.3: eq_high = -6
    elif ratio_high > 1.0: eq_high = -5
    elif ratio_high > 0.7: eq_high = -3
    elif ratio_high > 0.4: eq_high = -2
    else:                  eq_high = -1

    # -- 6b. Breathiness -> spectral flatness -------------------------------
    # Spectral flatness near 1.0 = noise-like / breathy
    # Spectral flatness near 0.0 = tonal / clear voice
    # Computed on voiced frames only (via energy gate)
    S_power       = S ** 2
    geom_mean     = np.exp(np.mean(np.log(S_power + 1e-10), axis=0))
    arith_mean    = np.mean(S_power, axis=0)
    flatness_per_frame = geom_mean / (arith_mean + 1e-10)
    # Use only frames with sufficient energy
    rms_stft      = np.sqrt(np.mean(S_power, axis=0))
    energy_mask   = rms_stft > (rms_stft.max() * 0.05)
    spectral_flatness = float(np.mean(flatness_per_frame[energy_mask])) \
                        if energy_mask.any() else 0.1
    # Breathy voice has high flatness (>0.15) -- needs more NR and compression
    breathiness = spectral_flatness   # stored for NR/compression adjustment below

    # -- 4. Sibilance -> de-esser -------------------------------------------
    e_sibi  = band_energy(S, freqs, 5000, 9000)
    e_ref2  = band_energy(S, freqs, 1000, 5000)
    sib_ratio = e_sibi / (e_ref2 if e_ref2 > 0 else 1e-9)

    if   sib_ratio > 0.80: deesser = 0.7
    elif sib_ratio > 0.55: deesser = 0.5
    elif sib_ratio > 0.35: deesser = 0.3
    else:                  deesser = 0.2

    # -- 5. Level -> volume boost -------------------------------------------
    rms_global  = float(np.sqrt(np.mean(y ** 2)))
    rms_db      = 20.0 * np.log10(rms_global + 1e-10)
    target_db   = -18.0   # meditation target level (leaves headroom for ambient mix)
    volume      = int(np.clip(round(target_db - rms_db), -6, +12))

    # -- 6. Noise floor -> noise reduction ----------------------------------
    rms_sorted  = np.sort(rms)
    n_quiet     = max(1, len(rms_sorted) // 10)
    noise_floor = float(np.mean(rms_sorted[:n_quiet]))
    snr         = 20.0 * np.log10(rms_global / (noise_floor + 1e-10))

    if   snr < 18: noise_reduction = 1.0
    elif snr < 28: noise_reduction = 0.7
    elif snr < 38: noise_reduction = 0.5
    else:          noise_reduction = 0.3

    # Breathy voice boost: high spectral flatness -> noisier signal -> more NR
    if breathiness > 0.20:
        noise_reduction = min(1.2, noise_reduction + 0.2)
        print(f"   [*]  Breathy voice detected (flatness={breathiness:.3f}) "
              f"-> NR boosted to {noise_reduction:.1f}")

    # -- 7. Dynamics -> compression -----------------------------------------
    peak            = float(np.max(np.abs(y)))
    crest_factor_db = 20.0 * np.log10(peak / (rms_global + 1e-10))

    if   crest_factor_db > 22: compression = 0.7
    elif crest_factor_db > 16: compression = 0.5
    elif crest_factor_db > 11: compression = 0.4
    else:                      compression = 0.3

    # Breathy voice has uneven dynamics -> more compression needed
    if breathiness > 0.20:
        compression = min(0.8, compression + 0.1)

    step(f"Levels   RMS={rms_db:.1f}dBFS  SNR={snr:.0f}dB  crest={crest_factor_db:.0f}dB  sib={sib_ratio:.2f}", t0)

    # -- 8. Speed -> rubberband ---------------------------------------------
    if   voiced_ratio > 0.65: speed = 0.85
    elif voiced_ratio > 0.45: speed = 0.90
    else:                     speed = 0.93

    # -- 9. XTTS params ----------------------------------------------------
    # F0 jitter = voice expressiveness
    # Stable/monotone voice -> low temperature but high repetition_penalty
    # Expressive voice     -> higher temperature, lower repetition_penalty
    if   f0_jitter < 0.10:
        temperature, top_k, top_p = 0.50, 25, 0.70
        repetition_penalty        = 7.0   # monotone -> high anti-repetition
    elif f0_jitter < 0.18:
        temperature, top_k, top_p = 0.55, 30, 0.75
        repetition_penalty        = 6.0
    elif f0_jitter < 0.30:
        temperature, top_k, top_p = 0.60, 40, 0.80
        repetition_penalty        = 5.0
    else:
        temperature, top_k, top_p = 0.65, 50, 0.85
        repetition_penalty        = 4.0   # expressive -> natural variation reduces repetition risk

    # length_penalty: based on voiced_ratio (speech density)
    # Dense speech (high voiced_ratio) -> slightly push the model toward longer output
    # Sparse speech (lots of pauses)   -> neutral / slight shortening tendency
    if   voiced_ratio > 0.65: length_penalty = 0.9   # fast / dense speaker
    elif voiced_ratio > 0.45: length_penalty = 1.0   # neutral
    else:                     length_penalty = 1.1   # slow / breathy speaker

    # gpt_cond_len: use as much of the reference WAV as available, up to 60s
    # Longer reference = better voice fidelity. Cap at 60s (XTTS limit).
    gpt_cond_len       = int(min(60, max(6, int(duration))))
    gpt_cond_chunk_len = 4    # default is optimal for most voices
    sound_norm_refs    = 0    # off by default; enable if reference is very quiet/loud

    step(f"XTTS     temp={temperature}  top_k={top_k}  top_p={top_p}  "
         f"rep_pen={repetition_penalty}  len_pen={length_penalty}  "
         f"gpt_cond_len={gpt_cond_len}s", t0)

    # -- 10b. New audio params (v23) -----------------------------------------
    reverb     = 0.0    # user decides; 0=off by default
    # noise_gate: auto-suggest a gentle gate for noisy voices, else off
    noise_gate = int(-50 + (noise_reduction * 10)) if snr < 28 else 0
    pan        = 0.0    # center by default
    limiter    = 1      # always on -- prevents clipping after all processing

    # -- 10. Fades -- derived from voice attack / decay ---------------------
    # Attack: median rise time across voiced segments (fast attack -> shorter fade_in)
    # Approximated via the natural_start_ms already measured
    # Soft voices (low RMS, breathy) benefit from longer fades to mask clicks
    if breathiness > 0.20 or rms_db < -30:
        fade_in  = 200
        fade_out = 400
    elif crest_factor_db > 18:   # punchy / dynamic voice
        fade_in  = 100
        fade_out = 250
    else:                         # standard
        fade_in  = 150
        fade_out = 300

    step(f"Analysis complete  ({duration:.1f}s audio processed)", t0)

    params = dict(
        speed              = speed,
        volume             = volume,
        eq_low             = eq_low,
        eq_mid             = eq_mid,
        eq_high            = eq_high,
        highpass           = highpass,
        lowpass            = lowpass,
        noise_reduction    = noise_reduction,
        compression        = compression,
        deesser            = deesser,
        reverb             = reverb,
        noise_gate         = noise_gate,
        pan                = pan,
        limiter            = limiter,
        trim_start         = trim_start,
        trim_end           = trim_end,
        fade_in            = fade_in,
        fade_out           = fade_out,
        temperature        = temperature,
        top_k              = top_k,
        top_p              = top_p,
        repetition_penalty = repetition_penalty,
        length_penalty     = length_penalty,
        gpt_cond_len       = gpt_cond_len,
        gpt_cond_chunk_len = gpt_cond_chunk_len,
        sound_norm_refs    = sound_norm_refs,
    )

    stats = dict(
        f0_median       = f0_median,
        voice_type      = voice_type,
        rms_db          = rms_db,
        snr             = snr,
        crest_factor_db = crest_factor_db,
        voiced_ratio    = voiced_ratio,
        f0_jitter       = f0_jitter,
        sib_ratio       = sib_ratio,
        breathiness     = breathiness,
        duration        = duration,
    )

    return params, stats


# -----------------------------------------------------------------------------
# Display results
# -----------------------------------------------------------------------------

def display_results(params, stats, voice_num=1, wav_file=None, language='FR', seed=0):
    p    = params
    s    = stats
    N    = voice_num
    lang = language.upper()

    # Build bracket strings -- 14 values for XTTS block (v23)
    xtts_arr = [N, seed,
                p['trim_start'], p['trim_end'],
                p['fade_in'], p['fade_out'],
                p['temperature'], p['top_k'], p['top_p'],
                p['repetition_penalty'], p['length_penalty'],
                p['gpt_cond_len'], p['gpt_cond_chunk_len'], p['sound_norm_refs']]
    xtts_str = ', '.join(fmt(v) for v in xtts_arr)

    # Audio bracket includes LANG in position 2 -- 14 audio values (v23)
    audio_vals = [p['speed'], p['volume'],
                  p['eq_low'], p['eq_mid'], p['eq_high'],
                  p['highpass'], p['lowpass'],
                  p['noise_reduction'], p['compression'], p['deesser'],
                  p['reverb'], p['noise_gate'], p['pan'], p['limiter']]
    audio_str = f"{N}, {lang}, " + ', '.join(fmt(v) for v in audio_vals)

    print(f"\n{'='*62}")
    if wav_file:
        print(f"  [*] RESULTS -> voice #{N} [{lang}]  :  {os.path.basename(wav_file)}")
    print(f"{'='*62}")

    print(f"""
  ACOUSTIC ANALYSIS
     Voice type     : {s['voice_type']}
     F0 median      : {s['f0_median']:.0f} Hz
     F0 jitter      : {s['f0_jitter']:.3f}  (expressiveness)
     RMS level      : {s['rms_db']:.1f} dBFS
     Estimated SNR  : {s['snr']:.1f} dB
     Crest factor   : {s['crest_factor_db']:.1f} dB
     Voiced ratio   : {s['voiced_ratio']:.0%}
     Sibilance      : {s['sib_ratio']:.3f}
     Breathiness    : {s['breathiness']:.3f}  (spectral flatness -- 0=tonal, 1=noisy)
""")

    print(f"""  [*] READY TO PASTE  (voice #{N} / {lang})
  ==================================================================

  # Voice {N} [{lang}]  {s['voice_type']}  {int(s['f0_median'])} Hz
  # -- XTTS params  {{N, seed, trim_start, trim_end, fade_in, fade_out, temp, top_k, top_p, rep_pen, len_pen}}
  {{{xtts_str}}}

  # -- Audio params  [N, LANG, speed, vol(dB), eq_low, eq_mid, eq_high, hp, lp, NR, comp, de-ess]
  [{audio_str}]

  ==================================================================""")

    print(f"""  [*] DETAIL
  +--------------------------------------------------------------+
  |  XTTS PARAMS  {{{xtts_str}}}
  |  [{N}]    voice number
  |  [0]    seed               = 0  (none)
  |  [{p['trim_start']}]  trim_start         = {p['trim_start']} ms
  |  [{p['trim_end']}]  trim_end           = {p['trim_end']} ms
  |  [{p['fade_in']}]  fade_in            = {p['fade_in']} ms
  |  [{p['fade_out']}]  fade_out           = {p['fade_out']} ms
  |  [{p['temperature']}]  temperature        = {p['temperature']}
  |  [{p['top_k']:.0f}]   top_k              = {p['top_k']:.0f}
  |  [{p['top_p']}]  top_p              = {p['top_p']}
  |  [{p['repetition_penalty']}]  repetition_penalty = {p['repetition_penalty']}  (anti-begaiement)
  |  [{p['length_penalty']}]  length_penalty     = {p['length_penalty']}  (duree phrase)
  +--------------------------------------------------------------+
  |  AUDIO PARAMS  [{audio_str}]
  |  [{p['speed']}]  speed        = {p['speed']}  (rubberband)
  |  [{p['volume']:+d}]   volume       = {p['volume']:+d} dB
  |  [{p['eq_low']:+d}]   eq_low       = {p['eq_low']:+d} dB  (80-300 Hz)
  |  [{p['eq_mid']:+d}]   eq_mid       = {p['eq_mid']:+d} dB  (300-3000 Hz)
  |  [{p['eq_high']:+d}]   eq_high      = {p['eq_high']:+d} dB  (3000-8000 Hz)
  |  [{p['highpass']:.0f}]   highpass     = {p['highpass']:.0f} Hz
  |  [{p['lowpass']:.0f}]  lowpass      = {p['lowpass']:.0f} Hz
  |  [{p['noise_reduction']}]  noise redu.  = {p['noise_reduction']}
  |  [{p['compression']}]  compression  = {p['compression']}
  |  [{p['deesser']}]  de-esser     = {p['deesser']}
  +--------------------------------------------------------------+""")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    VALID_LANGUAGES = {'fr','en','es','de','it','pt','pl','tr','ru',
                       'nl','cs','ar','zh-cn','hu','ko','ja','hi'}

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    args     = sys.argv[1:]
    precise  = '--precise' in args
    args     = [a for a in args if a not in ('--precise', '--fast')]
    fast     = not precise

    # --start-num N : numero de la premiere voix (defaut: 1)
    start_num = 1
    if '--start-num' in args:
        idx_sn = args.index('--start-num')
        if idx_sn + 1 < len(args):
            start_num = int(args[idx_sn + 1])
            args = [a for i, a in enumerate(args) if i != idx_sn and i != idx_sn + 1]

    # --seed N[,N2,...] : seed(s) par voix (defaut: 0)
    seeds = []
    if '--seed' in args:
        idx_sd = args.index('--seed')
        if idx_sd + 1 < len(args):
            seeds = [int(x) for x in args[idx_sd + 1].split(',')]
            args = [a for i, a in enumerate(args) if i != idx_sd and i != idx_sd + 1]

    def get_seed(voice_idx):
        """Retourne le seed pour la voix voice_idx (0-based)."""
        if not seeds:
            return 0
        if voice_idx < len(seeds):
            return seeds[voice_idx]
        return seeds[-1]  # dernier seed repete

    if not args:
        print(__doc__)
        sys.exit(0)

    # Parse pairs: voice.wav [LANG] voice2.wav [LANG2] ...
    # If an arg looks like a language code -> attach to previous wav
    voice_list = []   # list of (wav_path, language)
    i = 0
    while i < len(args):
        token = args[i]
        # Is it a language code?
        if token.lower() in VALID_LANGUAGES:
            if voice_list:
                wav, _ = voice_list[-1]
                voice_list[-1] = (wav, token.upper())
            i += 1
        else:
            # Treat as a WAV path, default language FR
            voice_list.append((token, 'FR'))
            i += 1

    results = []

    for idx, (wav_file, lang) in enumerate(voice_list, start_num):
        if not os.path.exists(wav_file):
            print(f"[!] File not found: {wav_file}")
            continue
        try:
            params, stats = analyse_voice(wav_file, fast=fast)
            seed = get_seed(idx - start_num)
            display_results(params, stats, voice_num=idx,
                            wav_file=wav_file, language=lang, seed=seed)
            results.append((wav_file, lang, params, stats, seed))
        except Exception as e:
            print(f"[!] Error analysing {wav_file}: {e}")
            import traceback
            traceback.print_exc()

    # Multi-voice summary
    if len(results) > 1:
        print(f"\n{'='*62}")
        print(f"  [*] MULTI-VOICE SUMMARY -- ready to paste")
        print(f"{'='*62}")
        for idx, (wav, lang, p, s, seed) in enumerate(results, start_num):
            xtts_arr = [idx, seed,
                        p['trim_start'], p['trim_end'],
                        p['fade_in'], p['fade_out'],
                        p['temperature'], p['top_k'], p['top_p'],
                        p['repetition_penalty'], p['length_penalty']]
            xtts_str  = ', '.join(fmt(v) for v in xtts_arr)
            audio_vals = [p['speed'], p['volume'],
                          p['eq_low'], p['eq_mid'], p['eq_high'],
                          p['highpass'], p['lowpass'],
                          p['noise_reduction'], p['compression'], p['deesser']]
            audio_str = f"{idx}, {lang}, " + ', '.join(fmt(v) for v in audio_vals)
            print(f"\n  # Voice {idx} [{lang}]  {s['voice_type']:<22} {s['f0_median']:.0f} Hz  {os.path.basename(wav)}")
            print(f"  {{{xtts_str}}}")
            print(f"  [{audio_str}]")
        print()


if __name__ == "__main__":
    main()

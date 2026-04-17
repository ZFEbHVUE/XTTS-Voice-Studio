#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Guided Meditation Generator - VERSION 22 - PARALLEL VOICE OVERLAY
============================================================================
- NEW v22: [parallel, offset=Xs] / [/parallel] blocks — two or more voices
           generated independently and mixed via pydub.overlay(). Each voice
           track supports the full {N,...} / [N,...] / [pause=Xs] syntax.
           offset= (default 0s) delays every non-first voice relative to
           the first voice in the block.
- v21: repetition_penalty and length_penalty now configurable in {} block.
       {} block extended from 9 to 11 values (rétro-compatible : 0 = défaut).
- v20: Inline XTTS params in curly-brace block {N, seed, trim_start, trim_end,
       fade_in, fade_out, temp, top_k, top_p}  — auto-detected, no ambiguity
       with audio brackets [ ].  Both syntaxes coexist.
- v19: Per-voice global XTTS params (multi-line blocks)
- v18: trim_start/end, fade_in/out, temperature, top_k, top_p configurable
- v16: seed=value for full reproducibility

Usage: python guided_meditation_generator_v22.py script.txt output.wav voice1.wav [voice2.wav ...] [music...]

Command for two voices, an ambient music and 2 punctual songs:

        You need to do under ~/XTTS/ : conda activate xtts before using this command

    python ~/XTTS/Python_Scripting/guided_meditation_generator_v22.py \\
               ~/XTTS/Prompts/prompt.txt \\
               ~/XTTS/Output_Song_files/output.wav \\
               ~/XTTS/Cloning_Voices/voice1.wav \\
               ~/XTTS/Cloning_Voices/voice2.wav \\
               ~/XTTS/Ambient_Musics/forest.wav \\
               ~/XTTS/Punctual_Sounds/bell1.wav \\
               ~/XTTS/Punctual_Sounds/bell2.wav

TWO bracket formats in the text file:

  1) AUDIO bracket (1 to 12 values):
     [N, LANG, speed, vol, eq_low, eq_mid, eq_high, hp, lp, NR, comp, de-ess]
      N  lang  spd   dB    lows   mids     highs   hp  lp   NR  comp  ds

    N means the number of the voice

    By default :
    'speed': 0.90,            # playback speed (rubberband)
    'volume': +3,             # voice volume boost in dB (files are often quiet)
    'eq_low': -2,             # 80-300 Hz  (reduce rumble)
    'eq_mid': +3,             # 300-3000 Hz (boost presence)
    'eq_high': -4,            # 3000-8000 Hz (cut artifacts, keep clarity)
    'highpass': 90,           # cut everything below 90 Hz
    'lowpass': 8000,          # cut everything above 8 kHz
    'noise_reduction': 0.5,   # noise reduction strength (0=off, 2=aggressive)
    'compression': 0.4,       # dynamic compression (0=off, 2=heavy)
    'deesser': 0.3,           # de-esser strength (0=off, 1=max)
    'language': 'fr'          # XTTS language code

     you can also do: [N, LANG, 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    LANG is optional (FR, EN, ES, DE, IT, PT, PL, TR, RU, NL, CS, AR, ZH-CN, HU, KO, JA, HI)

  2) XTTS-PARAMS block (curly braces, 9 OR 11 values — rétro-compatible):

     9 values (v20 format — still valid):
     {N, seed, trim_start, trim_end, fade_in, fade_out, temp, top_k, top_p}

     11 values (v21/v22 format — adds rep_pen and len_pen):
     {N, seed, trim_start, trim_end, fade_in, fade_out, temp, top_k, top_p, rep_pen, len_pen}
      N   seed  trim_ms    trim_ms    ms       ms       0-1   int   0-1    float     float

       Value 0 meaning ({N, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}) :

       seed               -> no seed (None)
       trim_start         -> 0 ms (no trim)
       trim_end           -> 0 ms (no trim)
       fade_in            -> 0 ms (no fade)
       fade_out           -> 0 ms (no fade)
       temperature        -> keep default (0 is invalid for XTTS)
       top_k              -> keep default
       top_p              -> keep default
       rep_pen            -> keep default
       len_pen            -> keep default

  3) PARALLEL block (NEW in v22):

     [parallel]                    -> all voices start simultaneously (offset=0)
     [parallel, offset=1.5s]       -> every non-first voice starts 1.5s after voice 1
     ...voice lines with {N} / [N,...] / text / [pause=Xs] ...
     [/parallel]

     - Inside the block, {N,...} and [N,...] switch the active voice track.
     - Each voice track is generated independently, then mixed via overlay().
     - The resulting mixed segment is inserted into the main timeline as one block.
     - All existing parameters ({}, [], [pause=]) work identically inside a parallel block.
     - The total duration of the block = duration of the longest voice track.

     Example:
       [parallel, offset=1s]
       {1, 42, 110, 255, 150, 300, 0.65, 50, 0.85, 5.0, 1.0}
       [1, FR, 0.85, +1, -5, +1, -1, 90, 9000, 0.3, 0.4, 0.2]
       La voix principale dit quelque chose posément.
       {2, 42, 60, 339, 150, 300, 0.60, 40, 0.80, 5.0, 1.0}
       [2, FR, 0.90, -4, -5, +1, -1, 90, 9000, 0.3, 0.4, 0.2]
       Et la deuxième voix murmure autre chose en même temps.
       [/parallel]

Typical usage (both brackets together):
  {1, 42, 110, 255, 150, 300, 0.65, 50, 0.85, 5.0, 1.0}        <- XTTS params voice 1 (v21)
  [1, FR, 0.85, +1, -5, +1, -1, 90, 9000, 0.3, 0.4, 0.2]       <- audio params voice 1

  {2, 42, 60, 339, 150, 300, 0.60, 40, 0.80, 6.0, 0.9}          <- XTTS params voice 2 (v21)
  [2, EN, 0.85, +8, -5, +1, -1, 90, 9000, 0.5, 0.5, 0.2]        <- audio params voice 2

  rep_pen  guidelines:
    4.0  -> expressive voice, natural variation
    5.0  -> standard (default)
    6.0-7.0 -> monotone voice, prevent robotic repetitions

  len_pen  guidelines:
    0.9  -> dense/fast speaker  (push model toward longer output)
    1.0  -> neutral (default)
    1.1  -> breathy/slow speaker

  ==============================================================================================
  To get good values for each voice you can use in the same folder : voice_analyser.py
  ==============================================================================================

Pause syntax:
  [pause=2s]           -> 2 second silence
  [pause=4s,start]     -> silence adjusted so total (speech + pause) = 4s

 ====================================================================
 To get from an mp3 song directly the txt and the according pauses :
 you can use in the same folder : transcribe_to_txt.py
 ====================================================================

Punctual music syntax (declare volume/duration at top, trigger inline):
  music_1=192s,-15   -> file #1: 192s duration, -15 dB attenuation
  music_2=-10        -> file #2: full duration, -10 dB
  [music=1]          -> trigger music #1 at this position

Ambient music:
  ambient_vol=-12  -> loops throughout the whole track at -12 dB
  (ambient file auto-detected: full path must contain "ambient" -- e.g. ~/XTTS/Ambient_Musics/forest.wav)
"""

from TTS.api import TTS
from pydub import AudioSegment
from pydub.effects import normalize, compress_dynamic_range
import subprocess
import re
import sys
import os
import time
from datetime import timedelta
import tempfile
import random
import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Default audio config (optimised for voice cloning, anti-artifact)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_AUDIO_CONFIG = {
    'speed': 0.90,          # playback speed (rubberband)
    'volume': +3,           # voice volume boost in dB (files are often quiet)
    'eq_low': -2,           # 80-300 Hz  (reduce rumble)
    'eq_mid': +3,           # 300-3000 Hz (boost presence)
    'eq_high': -4,          # 3000-8000 Hz (cut artifacts, keep clarity)
    'highpass': 90,         # cut everything below 90 Hz
    'lowpass': 8000,        # cut everything above 8 kHz
    'noise_reduction': 0.5, # noise reduction strength (0=off, 2=aggressive)
    'compression': 0.4,     # dynamic compression (0=off, 2=heavy)
    'deesser': 0.3,         # de-esser strength (0=off, 1=max)
    'language': 'fr'        # XTTS language code
}

# Alias for backward compatibility
CONFIG_AUDIO_DEFAUT = DEFAULT_AUDIO_CONFIG

# ─────────────────────────────────────────────────────────────────────────────
# Default XTTS params
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_XTTS_PARAMS = {
    'seed': None,
    'trim_start': 100,          # trim start ms  (XTTS artifact at beginning)
    'trim_end': 500,            # trim end ms    (XTTS tail artifact)
    'fade_in': 150,             # fade-in ms
    'fade_out': 300,            # fade-out ms
    'temperature': 0.65,
    'top_k': 50,
    'top_p': 0.85,
    'repetition_penalty': 5.0, # anti-bégaiement / anti-répétition
    'length_penalty': 1.0,     # <1 = plus court, >1 = plus long
}

# Alias used internally
PARAMS_GLOBAUX_DEFAUT = DEFAULT_XTTS_PARAMS


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text):
    """Strip trailing punctuation so XTTS does not vocalise it."""
    text = re.sub(r'[.!?]+\s*$', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Audio processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_eq(audio_segment, graves_db=0, mediums_db=0, aigus_db=0):
    """
    Apply a 3-band equaliser via FFmpeg.
      lows   : 80-300 Hz
      mids   : 300-3000 Hz
      highs  : 3000-8000 Hz
    """
    if graves_db == 0 and mediums_db == 0 and aigus_db == 0:
        return audio_segment

    temp_in  = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_out = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)

    try:
        audio_segment.export(temp_in.name, format='wav')

        filters = []
        if graves_db  != 0: filters.append(f"equalizer=f=200:t=h:width=200:g={graves_db}")
        if mediums_db != 0: filters.append(f"equalizer=f=1500:t=h:width=2000:g={mediums_db}")
        if aigus_db   != 0: filters.append(f"equalizer=f=5000:t=h:width=3000:g={aigus_db}")

        subprocess.run(
            ['ffmpeg', '-i', temp_in.name, '-af', ','.join(filters), '-y', temp_out.name],
            check=True, capture_output=True
        )
        return AudioSegment.from_wav(temp_out.name)
    except subprocess.CalledProcessError as e:
        print(f"  [!]  EQ error: {e}")
        return audio_segment
    finally:
        for f in (temp_in.name, temp_out.name):
            if os.path.exists(f): os.unlink(f)


def apply_filters(audio_segment, highpass_hz=0, lowpass_hz=0):
    """Apply high-pass and low-pass filters."""
    if highpass_hz > 0: audio_segment = audio_segment.high_pass_filter(highpass_hz)
    if lowpass_hz  > 0: audio_segment = audio_segment.low_pass_filter(lowpass_hz)
    return audio_segment


def apply_noise_reduction(audio_segment, force=0.8):
    """
    Noise reduction via FFmpeg afftdn filter.
    force: 0.0=off  1.0=normal  2.0=aggressive
    """
    if force <= 0:
        return audio_segment

    temp_in  = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_out = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)

    try:
        audio_segment.export(temp_in.name, format='wav')
        noise_floor = max(-80, min(-20, -60 + (force * 20)))
        subprocess.run(
            ['ffmpeg', '-i', temp_in.name, '-af', f'afftdn=nf={noise_floor}:tn=1', '-y', temp_out.name],
            check=True, capture_output=True
        )
        return AudioSegment.from_wav(temp_out.name)
    except subprocess.CalledProcessError as e:
        print(f"  [!]  Noise reduction error: {e}")
        return audio_segment
    finally:
        for f in (temp_in.name, temp_out.name):
            if os.path.exists(f): os.unlink(f)


def apply_compression(audio_segment, ratio=0.5):
    """
    Dynamic range compression.
    ratio: 0=off  0.5=light  1.0=medium  2.0=heavy
    """
    if ratio <= 0:
        return audio_segment
    threshold_db = -20 - (ratio * 10)
    return compress_dynamic_range(
        audio_segment,
        threshold=threshold_db,
        ratio=4.0, attack=5.0, release=50.0
    )


def apply_deesser(audio_segment, force=0.3):
    """
    De-esser: attenuate sibilance (5-8 kHz) via FFmpeg.
    force: 0=off  1=max
    """
    if force <= 0:
        return audio_segment

    temp_in  = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_out = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)

    try:
        audio_segment.export(temp_in.name, format='wav')
        intensity = min(1.0, force)
        subprocess.run(
            ['ffmpeg', '-i', temp_in.name,
             '-af', f'equalizer=f=6500:t=h:width=2000:g=-{intensity*10}',
             '-y', temp_out.name],
            check=True, capture_output=True
        )
        return AudioSegment.from_wav(temp_out.name)
    except subprocess.CalledProcessError as e:
        print(f"  [!]  De-esser error: {e}")
        return audio_segment
    finally:
        for f in (temp_in.name, temp_out.name):
            if os.path.exists(f): os.unlink(f)


def process_audio(audio_segment, config, xtts_params):
    """
    Full audio processing pipeline (per sentence):
    Trim -> Filters -> EQ -> Noise reduction -> De-esser -> Compression -> Fades
    """
    print(f"      [*]  Audio: EQ({config['eq_low']:+.0f}/{config['eq_mid']:+.0f}/{config['eq_high']:+.0f}dB) "
          f"Filters({config['highpass']}-{config['lowpass']}Hz) "
          f"NR({config['noise_reduction']:.1f}) Comp({config['compression']:.1f}) DS({config['deesser']:.1f})")

    # 1. Trim XTTS artifacts
    trim_start = int(xtts_params['trim_start'])
    trim_end   = int(xtts_params['trim_end'])

    if trim_start == 0 and trim_end == 0:
        print(f"      [*]  No trim (trim_start=0, trim_end=0)")
    elif trim_start > 0 and trim_end > 0:
        if len(audio_segment) > (trim_start + trim_end + 50):
            audio_segment = audio_segment[trim_start:-trim_end]
            print(f"      [*]  Trimmed: {trim_start}ms start + {trim_end}ms end")
        else:
            print(f"      [*]  Segment too short to trim ({len(audio_segment)}ms), skipped")
    elif trim_start > 0:
        if len(audio_segment) > trim_start + 50:
            audio_segment = audio_segment[trim_start:]
            print(f"      [*]  Trimmed: {trim_start}ms start only")
    elif trim_end > 0:
        if len(audio_segment) > trim_end:
            audio_segment = audio_segment[:-trim_end]
            print(f"      [*]  Trimmed: {trim_end}ms end only")

    # 2. High-pass / low-pass filters
    audio_segment = apply_filters(audio_segment,
                                      highpass_hz=config['highpass'],
                                      lowpass_hz=config['lowpass'])

    # 3. 3-band EQ
    audio_segment = apply_eq(audio_segment,
                                  graves_db=config['eq_low'],
                                  mediums_db=config['eq_mid'],
                                  aigus_db=config['eq_high'])

    # 4. Noise reduction
    audio_segment = apply_noise_reduction(audio_segment, force=config['noise_reduction'])

    # 5. De-esser
    audio_segment = apply_deesser(audio_segment, force=config['deesser'])

    # 6. Dynamic compression
    audio_segment = apply_compression(audio_segment, ratio=config['compression'])

    # 7. Fades (anti-click between sentences)
    fade_in  = int(xtts_params['fade_in'])
    fade_out = int(xtts_params['fade_out'])
    dur_ms   = len(audio_segment)

    if fade_in > 0 or fade_out > 0:
        # Safety: fades cannot exceed 1/3 of segment length each
        fi_safe = min(fade_in,  dur_ms // 3)
        fo_safe = min(fade_out, dur_ms // 3)
        if fi_safe > 0: audio_segment = audio_segment.fade_in(fi_safe)
        if fo_safe > 0: audio_segment = audio_segment.fade_out(fo_safe)
        note = f"  (reduced, segment={dur_ms}ms)" if (fi_safe != fade_in or fo_safe != fade_out) else ""
        print(f"      [*]  Fades: in={fi_safe}ms  out={fo_safe}ms{note}")
    else:
        print(f"      [*]  Fades: disabled (0/0)")

    return audio_segment


def apply_speed_rubberband(audio_segment, speed,
                               temp_input="/tmp/rb_in.wav",
                               temp_output="/tmp/rb_out.wav"):
    """Time-stretch audio using rubberband (preserves pitch)."""
    audio_segment.export(temp_input, format="wav")
    time_stretch = 1.0 / speed
    try:
        subprocess.run(
            ["rubberband", "-t", str(time_stretch), "-c", "1", temp_input, temp_output],
            check=True, capture_output=True
        )
        result = AudioSegment.from_wav(temp_output)
        for f in (temp_input, temp_output):
            if os.path.exists(f): os.remove(f)
        return result
    except subprocess.CalledProcessError as e:
        print(f"  [!]  Rubberband error: {e}")
        return audio_segment


# ─────────────────────────────────────────────────────────────────────────────
# Bracket parsers
# ─────────────────────────────────────────────────────────────────────────────

def parse_voice_config(config_str):
    """
    Parse audio bracket:
      [N, speed, vol, eq_low, eq_mid, eq_high, hp, lp, NR, comp, de-ess]
      [N, LANG, speed, vol, eq_low, eq_mid, eq_high, hp, lp, NR, comp, de-ess]

    LANG (FR, EN, ES, ...) is auto-detected in position 2 if non-numeric.
    Returns: (voice_num, config_dict)
    """
    VALID_LANGUAGES = {'fr','en','es','de','it','pt','pl','tr','ru',
                       'nl','cs','ar','zh-cn','hu','ko','ja','hi'}

    config_str = config_str.strip('[]').strip()
    parts = [p.strip() for p in config_str.split(',')]

    try:
        voice_num = int(parts[0])
    except (ValueError, IndexError):
        return None, None

    config = DEFAULT_AUDIO_CONFIG.copy()

    # Detect optional language in position 1
    offset = 1
    if len(parts) > 1 and parts[1] and not parts[1].lstrip('+-').replace('.','',1).isdigit():
        lang = parts[1].lower().strip()
        if lang in VALID_LANGUAGES:
            config['language'] = lang
            print(f"  [*] Language: {lang.upper()}")
        else:
            print(f"  [!]  Unknown language '{parts[1]}', ignored (default: {config['language']})")
        offset = 2

    param_names = ['speed', 'volume', 'eq_low', 'eq_mid', 'eq_high',
                   'highpass', 'lowpass', 'noise_reduction', 'compression', 'deesser']

    for i, param in enumerate(param_names):
        idx = i + offset
        if idx < len(parts) and parts[idx]:
            try:
                config[param] = float(parts[idx])
            except ValueError:
                pass

    print(f"  [OK] Voice {voice_num} [{config['language'].upper()}] "
          f"speed={config['speed']} vol={config['volume']:+.0f}dB")
    return voice_num, config


def is_params_xtts_bracket(segment):
    """
    Returns True if segment is an XTTS-PARAMS block: {N, seed, ...}
    Curly braces are unambiguous vs audio brackets [ ].
    """
    s = segment.strip()
    return s.startswith('{') and s.endswith('}')


def parse_xtts_params(segment, base_params):
    """
    Parse XTTS-PARAMS block — v21: 9 OR 11 values (rétro-compatible).

    9 values (v20 format):
    {N, seed, trim_start, trim_end, fade_in, fade_out, temperature, top_k, top_p}

    11 values (v21/v22 format):
    {N, seed, trim_start, trim_end, fade_in, fade_out, temperature, top_k, top_p, rep_pen, len_pen}

    Value 0 semantics:
      seed               -> no seed (None)
      trim_start         -> 0 ms  (no trim)
      trim_end           -> 0 ms  (no trim)
      fade_in            -> 0 ms  (no fade)
      fade_out           -> 0 ms  (no fade)
      temperature        -> keep default (0 is invalid for XTTS)
      top_k              -> keep default
      top_p              -> keep default
      rep_pen            -> keep default
      len_pen            -> keep default

    Returns: (voice_num, params_dict)
    """
    contenu = segment.strip('{}').strip()
    parts   = [p.strip() for p in contenu.split(',')]

    try:
        voice_num = int(parts[0])
    except (ValueError, IndexError):
        return None, None

    # Fields where 0 is a valid literal value (= "do not trim / do not fade")
    ZERO_VALID = {'trim_start', 'trim_end', 'fade_in', 'fade_out'}

    keys = ['seed', 'trim_start', 'trim_end', 'fade_in', 'fade_out',
            'temperature', 'top_k', 'top_p',
            'repetition_penalty', 'length_penalty']    # v21: 2 new keys
    p = base_params.copy()

    for i, key in enumerate(keys):
        idx = i + 1
        if idx < len(parts) and parts[idx] != '':
            try:
                val = float(parts[idx])
                if key == 'seed':
                    p['seed'] = int(val) if val != 0 else None
                elif key in ZERO_VALID:
                    p[key] = val           # 0 = "no trim/fade" -> valid
                elif val != 0:
                    p[key] = val           # 0 = invalid -> keep default
            except ValueError:
                pass

    return voice_num, p


# ─────────────────────────────────────────────────────────────────────────────
# XTTS params extraction (multi-line block syntax)
# ─────────────────────────────────────────────────────────────────────────────

def extract_per_voice_params(text):
    """
    Extract per-voice XTTS params from multi-line block syntax (v19 style).

    Strategy: scan line by line.
    Param lines (seed=, trim_start=, ...) preceding a [N,...] are assigned to voice N.
    Params before the first [N,...] are treated as globals (voice 0 = fallback).

    Returns: dict { 0: {global params},
                    1: {voice 1 params},
                    2: {voice 2 params}, ... }
    Missing values are filled with DEFAULT_XTTS_PARAMS.
    """
    import re as _re

    XTTS_KEYS = {'seed', 'trim_start', 'trim_end', 'fade_in',
                 'fade_out', 'temperature', 'top_k', 'top_p',
                 'repetition_penalty', 'length_penalty'}    # v21: 2 new keys

    RE_PARAM = _re.compile(
        r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+]?[0-9.]+(?:s)?(?:,[-+]?[0-9.]+)?)\s*$'
    )
    RE_VOICE = _re.compile(r'^\s*\[(\d+)(?:\s*,.*?)?\]\s*$')

    xtts_params_by_voice = {}
    pending = {}

    for line in text.split('\n'):
        if '#' in line:
            line = line.split('#')[0]
        line = line.strip()
        if not line:
            continue

        m_voice = RE_VOICE.match(line)
        if m_voice:
            num = int(m_voice.group(1))
            if pending:
                xtts_params_by_voice[num] = pending.copy()
                pending = {}
            continue

        if '=' in line and not line.startswith('['):
            m = RE_PARAM.match(line)
            if m:
                key, val_str = m.group(1), m.group(2)
                if key in XTTS_KEYS:
                    pending[key] = val_str

    if pending:
        xtts_params_by_voice[0] = pending

    result = {}
    base = DEFAULT_XTTS_PARAMS.copy()
    if 0 in xtts_params_by_voice:
        base = _apply_raw_params(base, xtts_params_by_voice[0])
    result[0] = base.copy()

    for num, raw in xtts_params_by_voice.items():
        if num == 0:
            continue
        p = base.copy()
        result[num] = _apply_raw_params(p, raw)

    return result


def _apply_raw_params(params, raw):
    """Apply a dict of raw string values onto a params dict."""
    p = params.copy()
    for key, val_str in raw.items():
        try:
            p[key] = int(val_str) if key == 'seed' else float(val_str)
        except ValueError:
            pass
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Configuration extraction (music, ambient)
# ─────────────────────────────────────────────────────────────────────────────

def extract_config(text):
    """
    Extract ambient_vol, music configs, and clean text.
    XTTS per-voice params are handled by extract_per_voice_params().
    """
    music_configs = {}
    ambient_vol = None

    # Strip comments
    clean_lines = []
    for line in text.split('\n'):
        if '#' in line:
            line = line.split('#')[0]
        line = line.rstrip()
        if line:
            clean_lines.append(line)

    text_out = []
    for line in clean_lines:
        stripped = line.strip()

        if '=' in stripped and not stripped.startswith('['):
            match = re.match(
                r'^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([-+]?[0-9.]+(?:s)?(?:,[-+]?[0-9.]+)?)\s*$',
                stripped
            )
            if match:
                name     = match.group(1)
                val_str  = match.group(2)

                if name == 'ambient_volume':
                    ambient_vol = float(val_str)
                    print(f"   [*] Ambient volume: {ambient_vol} dB")

                elif name.startswith('music_'):
                    try:
                        idx = int(name.replace('music_', ''))
                        if ',' in val_str:
                            parts   = val_str.split(',')
                            dur_str = parts[0].strip()
                            vol     = float(parts[1].strip())
                            dur_sec = float(dur_str[:-1]) if dur_str.endswith('s') else float(dur_str)
                            music_configs[idx] = (dur_sec, vol)
                            print(f"   [*] Music #{idx}: duration {dur_sec}s, volume {vol} dB")
                        else:
                            vol = float(val_str)
                            music_configs[idx] = (None, vol)
                            print(f"   [*] Music #{idx}: full duration, volume {vol} dB")
                    except ValueError:
                        print(f"   [!]  Invalid music config: {name}={val_str}")

                continue

        text_out.append(line)

    return ambient_vol, music_configs, '\n'.join(text_out)


# ─────────────────────────────────────────────────────────────────────────────
# Audio file parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_audio_files(args):
    """
    Classify audio files based on their parent folder name (case-insensitive):
      - Path contains "voices_cloning"  -> voice file
      - Path contains "ambient"         -> ambient track
      - Anything else                   -> punctual music file

    Examples:
      ~/XTTS/Voices_Cloning/Monique.wav          -> voice
      ~/XTTS/Ambient_Musics/forest.wav           -> ambient
      ~/XTTS/Punctual_Sounds/bell.wav            -> punctual music
      ~/XTTS/Punctual_Sounds/metronome.wav       -> punctual music

    Returns: (voice_files, ambient_file, punctual_music_files)
    """
    voice_files  = []
    ambient_file = None
    music_files  = []

    for fpath in args:
        if not os.path.exists(fpath):
            print(f"[!]  File not found: {fpath}")
            continue

        path_lower = fpath.lower()

        if "voices_cloning" in path_lower:
            voice_files.append(fpath)
            print(f"   [*] Voice #{len(voice_files)}: {os.path.basename(fpath)}")

        elif "ambient" in path_lower:
            ambient_file = fpath
            print(f"   [*] Ambient file: {os.path.basename(fpath)}")

        else:
            music_files.append(fpath)
            print(f"   [*] Music #{len(music_files)}: {os.path.basename(fpath)}")

    return voice_files, ambient_file, music_files


# ─────────────────────────────────────────────────────────────────────────────
# Sentence audio generator (helper — shared by main loop and parallel blocks)
# ─────────────────────────────────────────────────────────────────────────────

def generate_sentence_audio(clean, voice_num, voice_files, tts_instances,
                             config, xtts_params, temp_file):
    """
    Generate and fully process audio for a single clean sentence.
    Returns AudioSegment on success, None on error.
    Pipeline: TTS -> trim -> EQ -> NR -> de-ess -> compress -> fades -> rubberband -> volume
    """
    if voice_num not in tts_instances:
        print(f"  [!]  TTS instance for voice {voice_num} not found")
        return None
    if voice_num > len(voice_files):
        print(f"  [!]  Voice #{voice_num} not available (only {len(voice_files)} voices loaded)")
        return None

    speaker_wav = voice_files[voice_num - 1]
    try:
        tts_instances[voice_num].tts_to_file(
            text=clean,
            file_path=temp_file,
            language=config.get('language', 'fr'),
            speaker_wav=speaker_wav,
            temperature=xtts_params['temperature'],
            length_penalty=xtts_params['length_penalty'],
            repetition_penalty=xtts_params['repetition_penalty'],
            top_k=int(xtts_params['top_k']),
            top_p=xtts_params['top_p'],
            speed=1.0      # speed applied later via rubberband
        )
        audio = AudioSegment.from_wav(temp_file)
        audio = process_audio(audio, config, xtts_params)
        audio = apply_speed_rubberband(audio, config['speed'])
        audio = audio + config['volume']
        return audio
    except Exception as e:
        print(f"  [!]  Error generating sentence audio: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Parallel block processor (NEW in v22)
# ─────────────────────────────────────────────────────────────────────────────

def process_parallel_block(inner_segments, voice_files, tts_instances,
                            xtts_params_by_voice, offset_ms, temp_file):
    """
    Process a [parallel] ... [/parallel] block.

    inner_segments : list of segment strings between [parallel] and [/parallel]
    offset_ms      : milliseconds to delay each successive voice track
                     (voice 1 starts at 0, voice 2 at offset_ms, voice 3 at 2*offset_ms, ...)
    temp_file      : path for TTS temp WAV output

    Returns: (mixed AudioSegment, total_duration_ms, sentences_generated_count)

    Each voice track is built from the segments attributed to it (via {N} / [N,...] switches).
    All tracks are then mixed via pydub.overlay() into a single segment.
    The total duration equals that of the longest track (accounting for offsets).
    """
    print(f"\n  [PARALLEL] Starting parallel block (offset={offset_ms}ms, "
          f"{len(inner_segments)} inner segments)")

    # ── Parse inner segments into per-voice tracks ────────────────────────────
    # Each track is a list of ('text', clean_text, config, xtts_params)
    #                     or  ('pause', duration_ms)
    tracks = {}          # {voice_num: [items]}
    track_order = []     # keeps insertion order of voice numbers

    # Initial state inherits from the outer context defaults
    cur_voice = 1
    cur_config = DEFAULT_AUDIO_CONFIG.copy()
    cur_xtts   = xtts_params_by_voice.get(1, xtts_params_by_voice.get(0, DEFAULT_XTTS_PARAMS.copy()))

    def ensure_track(vn):
        if vn not in tracks:
            tracks[vn] = []
            track_order.append(vn)

    for seg in inner_segments:
        seg = seg.strip()
        if not seg:
            continue

        # XTTS params block  {N, ...}
        if is_params_xtts_bracket(seg):
            vn, new_params = parse_xtts_params(
                seg,
                xtts_params_by_voice.get(cur_voice, xtts_params_by_voice.get(0, DEFAULT_XTTS_PARAMS.copy()))
            )
            if vn and vn <= len(voice_files):
                cur_voice = vn
                cur_xtts  = new_params
                print(f"  [PARALLEL] XTTS params -> voice {vn}")

        # Audio bracket  [N, ...]
        elif seg.startswith('[') and not seg.startswith('[pause=') and \
             not seg.startswith('[music=') and \
             ((',' in seg) or (len(seg) > 1 and seg[1].isdigit())):
            vn, cfg = parse_voice_config(seg)
            if vn and vn <= len(voice_files):
                cur_voice  = vn
                cur_config = cfg
                cur_xtts   = xtts_params_by_voice.get(vn, xtts_params_by_voice.get(0, DEFAULT_XTTS_PARAMS.copy()))

        # Pause
        elif seg.startswith('[pause='):
            m = re.search(r'\[pause=([\d.]+)s?(?:,start)?\]', seg)
            if m:
                dur_ms = int(float(m.group(1)) * 1000)
                ensure_track(cur_voice)
                tracks[cur_voice].append(('pause', dur_ms))
                print(f"  [PARALLEL] Voice {cur_voice}: pause {float(m.group(1))}s")

        # Text sentence
        elif not seg.startswith('[') and not seg.startswith('{'):
            clean = clean_text(seg)
            if clean:
                ensure_track(cur_voice)
                tracks[cur_voice].append(('text', clean, cur_config.copy(), cur_xtts.copy()))
                print(f"  [PARALLEL] Voice {cur_voice}: \"{clean[:50]}{'...' if len(clean)>50 else ''}\"")

    if not tracks:
        print(f"  [PARALLEL] Empty block — skipped")
        return AudioSegment.empty(), 0, 0

    # ── Generate audio for each voice track ──────────────────────────────────
    track_audios = {}   # {voice_num: AudioSegment}
    sentences_count = 0

    for vn in track_order:
        items = tracks[vn]
        track_audio = AudioSegment.empty()
        print(f"\n  [PARALLEL] Generating track for voice {vn} ({len(items)} items)...")

        for item in items:
            if item[0] == 'pause':
                dur_ms = item[1]
                track_audio += AudioSegment.silent(duration=dur_ms)
                print(f"      [*] Voice {vn}: silence {dur_ms}ms")

            elif item[0] == 'text':
                _, clean, cfg, xp = item
                print(f"  [*] [Voice {vn}] [parallel] "
                      f"[{cfg['speed']}x, {cfg['volume']:+.0f}dB] "
                      f"{clean[:40]}{'...' if len(clean) > 40 else ''}")

                t_start = time.time()
                audio = generate_sentence_audio(clean, vn, voice_files, tts_instances,
                                                cfg, xp, temp_file)
                if audio is not None:
                    track_audio += audio
                    track_audio += AudioSegment.silent(duration=150)
                    t_sent = time.time() - t_start
                    sentences_count += 1
                    print(f"      [*] Duration: {len(audio)/1000:.2f}s  ({t_sent:.1f}s gen)")

        track_audios[vn] = track_audio
        print(f"  [PARALLEL] Track voice {vn}: total {len(track_audio)/1000:.2f}s")

    # ── Mix all tracks with offsets ───────────────────────────────────────────
    voices_ordered = track_order  # preserves declaration order in the block

    # Start with an empty base long enough to hold all tracks
    max_needed = 0
    for idx, vn in enumerate(voices_ordered):
        end_ms = idx * offset_ms + len(track_audios[vn])
        if end_ms > max_needed:
            max_needed = end_ms

    mixed = AudioSegment.silent(duration=max_needed)

    for idx, vn in enumerate(voices_ordered):
        start_ms = idx * offset_ms
        mixed = mixed.overlay(track_audios[vn], position=start_ms)
        print(f"  [PARALLEL] Mixed voice {vn} at position {start_ms/1000:.2f}s "
              f"(duration {len(track_audios[vn])/1000:.2f}s)")

    print(f"  [PARALLEL] Block complete — total duration {len(mixed)/1000:.2f}s, "
          f"{sentences_count} sentences generated\n")

    return mixed, len(mixed), sentences_count


# ─────────────────────────────────────────────────────────────────────────────
# Parallel block syntax helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_parallel_offset(segment):
    """
    Parse [parallel] or [parallel, offset=1.5s] -> offset in milliseconds.
    Returns offset_ms (int), default 0.
    """
    m = re.search(r'offset\s*=\s*([\d.]+)\s*s?', segment, re.IGNORECASE)
    if m:
        return int(float(m.group(1)) * 1000)
    return 0


def count_total_sentences(segments):
    """
    Count all text sentences across normal segments and parallel block inner segments.
    Used to compute the progress bar total.
    """
    count = 0
    in_parallel = False
    for seg in segments:
        s = seg.strip()
        if not s:
            continue
        if re.match(r'^\[parallel[^\]]*\]$', s, re.IGNORECASE):
            in_parallel = True
            continue
        if re.match(r'^\[/parallel\]$', s, re.IGNORECASE):
            in_parallel = False
            continue
        # Count text lines (not brackets, not curly blocks)
        if not s.startswith('{') and not s.startswith('['):
            count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Main generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_meditation(text, output_file, voice_files, ambient_file, music_files):
    print("[*] Guided Meditation Generator - VERSION 22 - PARALLEL VOICE OVERLAY")
    print(f"   [*] Available voices: {len(voice_files)}")
    for i, v in enumerate(voice_files, 1):
        print(f"      Voice {i}: {os.path.basename(v)}")

    # Extract per-voice XTTS params BEFORE text cleaning
    xtts_params_by_voice = extract_per_voice_params(text)

    # Summary
    for num in sorted(k for k in xtts_params_by_voice if k > 0):
        p = xtts_params_by_voice[num]
        print(f"   [*]  Voice {num}: trim={p['trim_start']:.0f}/{p['trim_end']:.0f}ms "
              f"fade={p['fade_in']:.0f}/{p['fade_out']:.0f}ms "
              f"temp={p['temperature']} top_k={p['top_k']:.0f} top_p={p['top_p']} "
              f"rep_pen={p['repetition_penalty']} len_pen={p['length_penalty']} "
              f"seed={p['seed']}")

    # Initial seed (voice 1 or global fallback)
    params_voice1  = xtts_params_by_voice.get(1, xtts_params_by_voice.get(0, DEFAULT_XTTS_PARAMS))
    sval = params_voice1.get('seed')
    if sval is not None:
        random.seed(sval)
        np.random.seed(sval)
        torch.manual_seed(sval)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sval)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        print(f"[*] Initial seed = {sval} (voice 1)")

    ambient_vol, music_configs, clean_content = extract_config(text)

    # Single shared TTS instance — XTTS v2 is a voice cloning model, so
    # the same model handles every voice. The actual voice switch happens
    # later via the speaker_wav argument of tts_to_file(). This saves
    # ~2 GB of VRAM per additional voice (critical on 4 GB GPUs).
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] TTS device: {_device}")
    shared_tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(_device)
    tts_instances = {i: shared_tts for i in range(1, len(voice_files) + 1)}
    for i in tts_instances:
        print(f"   [OK] TTS ready for voice {i}")

    # Split text into segments — v22: also captures [parallel...] and [/parallel]
    # NOTE: [parallel...] and [/parallel] patterns must come BEFORE [\d+...] in the
    # alternation to avoid being swallowed by the generic bracket pattern.
    raw_segments = re.split(
        r'(\{[^}]+\}'
        r'|\[pause=[\d.]+s?(?:,start)?\]'
        r'|\[parallel[^\]]*\]'
        r'|\[/parallel\]'
        r'|\[\d+(?:\s*,\s*[^]]*?)?\]'
        r'|\[music=\d+(?:,(?:music_\d+|[\d.]+s?,)?[-+]?\d+)?\])',
        clean_content
    )
    segments = [s for s in raw_segments if s and s.strip()]

    voice_audio        = AudioSegment.empty()
    temp_file          = "/tmp/temp_tts_v22.wav"
    music_to_apply     = []
    position_ms        = 0

    print("\n[*] Generating...\n")

    sentence_count  = 0
    current_voice   = 1
    current_config  = DEFAULT_AUDIO_CONFIG.copy()
    current_xtts_params = xtts_params_by_voice.get(1, xtts_params_by_voice.get(0, DEFAULT_XTTS_PARAMS.copy()))

    # Count all sentences (including inside parallel blocks) for progress bar
    total_sentences = count_total_sentences(segments)

    # Total work units = sentences + post-processing phases
    # (mix music + ambient + save = 3 phases). Used for the GUI progress bar
    # so that 100% only fires when the WAV is actually written to disk.
    POST_PHASES = 3
    total_steps = total_sentences + POST_PHASES
    step_done   = 0

    times_per_sentence = []
    start_time         = time.time()
    last_audio         = None
    last_sentence_pos  = 0

    i = 0
    while i < len(segments):
        segment = segments[i]
        if not segment or not segment.strip():
            i += 1
            continue
        segment = segment.strip()

        # ── [parallel, offset=Xs] block ───────────────────────────────────────
        if re.match(r'^\[parallel[^\]]*\]$', segment, re.IGNORECASE):
            offset_ms = parse_parallel_offset(segment)
            print(f"  [*] [parallel] block detected (offset={offset_ms}ms)")

            # Collect all inner segments until [/parallel]
            inner_segments = []
            j = i + 1
            while j < len(segments):
                s = segments[j].strip()
                if re.match(r'^\[/parallel\]$', s, re.IGNORECASE):
                    break
                inner_segments.append(s)
                j += 1

            # j now points at [/parallel] (or end of list if missing — graceful)
            if j < len(segments):
                i = j + 1  # skip past [/parallel]
            else:
                print(f"  [!]  [parallel] block without [/parallel] — processing to end of script")
                i = j

            # Generate and mix parallel voices
            t_parallel_start = time.time()
            mixed_audio, block_dur_ms, n_sentences = process_parallel_block(
                inner_segments, voice_files, tts_instances,
                xtts_params_by_voice, offset_ms, temp_file
            )

            if block_dur_ms > 0:
                voice_audio += mixed_audio
                position_ms += block_dur_ms
                last_audio        = mixed_audio
                last_sentence_pos = position_ms - block_dur_ms

                sentence_count += n_sentences
                elapsed_total   = time.time() - start_time
                pct = int(100 * sentence_count / total_sentences) if total_sentences else 0
                t_parallel = time.time() - t_parallel_start
                print(f"      [*] [{pct:3d}%] [{sentence_count}/{total_sentences}] "
                      f"{int(t_parallel)}s | "
                      f"Elapsed: {timedelta(seconds=int(elapsed_total))}")
                step_done += n_sentences
                print(f"[PROGRESS={step_done}/{total_steps}]")

            continue  # i already advanced past [/parallel]

        # ── Pause ────────────────────────────────────────────────────────────
        elif segment.startswith('[pause='):
            match = re.search(r'\[pause=([\d.]+)s?(?:,start)?\]', segment)
            if match:
                duration    = float(match.group(1))
                is_adjusted = ',start' in segment

                if is_adjusted and last_audio is not None:
                    elapsed      = (position_ms - last_sentence_pos) / 1000.0
                    adjusted_pause = duration - elapsed
                    if adjusted_pause > 0:
                        voice_audio += AudioSegment.silent(duration=int(adjusted_pause * 1000))
                        position_ms += int(adjusted_pause * 1000)
                        print(f"  [*]  Adjusted pause: speech={elapsed:.2f}s + silence={adjusted_pause:.2f}s = TOTAL {duration:.2f}s [OK]")
                    else:
                        print(f"  [*]  Adjusted pause: speech already >= {duration}s")
                else:
                    voice_audio += AudioSegment.silent(duration=int(duration * 1000))
                    position_ms += int(duration * 1000)
                    print(f"  [*]  Pause {duration}s")

        # ── XTTS params block  {N, seed, ...}  ───────────────────────────────
        elif segment.startswith('{') and segment.endswith('}'):
            voice_num, new_params = parse_xtts_params(
                segment,
                xtts_params_by_voice.get(current_voice, xtts_params_by_voice.get(0, DEFAULT_XTTS_PARAMS.copy()))
            )
            if voice_num and voice_num <= len(voice_files):
                xtts_params_by_voice[voice_num] = new_params
                if voice_num == current_voice:
                    current_xtts_params = new_params
                    sv = new_params.get('seed')
                    if sv is not None:
                        random.seed(sv); np.random.seed(sv); torch.manual_seed(sv)
                        if torch.cuda.is_available(): torch.cuda.manual_seed_all(sv)
                print(f"  [*]  XTTS params voice {voice_num}: "
                      f"seed={new_params['seed']} "
                      f"trim={new_params['trim_start']:.0f}/{new_params['trim_end']:.0f}ms "
                      f"fade={new_params['fade_in']:.0f}/{new_params['fade_out']:.0f}ms "
                      f"temp={new_params['temperature']} "
                      f"top_k={new_params['top_k']:.0f} top_p={new_params['top_p']} "
                      f"rep_pen={new_params['repetition_penalty']} "
                      f"len_pen={new_params['length_penalty']}")
            elif voice_num:
                print(f"  [!]  Voice #{voice_num} not available")

        # ── Audio bracket  [N, ...]  ──────────────────────────────────────────
        elif segment.startswith('[') and ',' in segment or \
             (segment.startswith('[') and len(segment) > 1 and segment[1].isdigit()):
            voice_num, config = parse_voice_config(segment)
            if voice_num:
                if voice_num <= len(voice_files):
                    current_voice   = voice_num
                    current_config  = config
                    current_xtts_params = xtts_params_by_voice.get(
                        voice_num, xtts_params_by_voice.get(0, DEFAULT_XTTS_PARAMS.copy())
                    )
                    sv = current_xtts_params.get('seed')
                    if sv is not None:
                        random.seed(sv); np.random.seed(sv); torch.manual_seed(sv)
                        if torch.cuda.is_available(): torch.cuda.manual_seed_all(sv)
                    print(f"  [*] Voice {current_voice}: speed={config['speed']}x "
                          f"vol={config['volume']:+.0f}dB "
                          f"lang={config['language'].upper()} | "
                          f"trim={current_xtts_params['trim_start']:.0f}/"
                          f"{current_xtts_params['trim_end']:.0f}ms "
                          f"fade={current_xtts_params['fade_in']:.0f}/"
                          f"{current_xtts_params['fade_out']:.0f}ms "
                          f"rep_pen={current_xtts_params['repetition_penalty']} "
                          f"len_pen={current_xtts_params['length_penalty']} "
                          f"seed={sv}")
                else:
                    print(f"  [!]  Voice #{voice_num} not available")

        # ── Punctual music  [music=N]  ──────────────────────────────────────
        elif segment.startswith('[music='):
            match = re.search(r'\[music=(\d+)\]', segment)
            if match:
                idx = int(match.group(1))
                if idx <= len(music_files) and idx in music_configs:
                    fpath   = music_files[idx - 1]
                    dur_sec, vol_db = music_configs[idx]
                    music_to_apply.append((position_ms, fpath, dur_sec, vol_db))
                    dur_str = f"{dur_sec}s" if dur_sec else "full"
                    print(f"  [*] Music #{idx} at {position_ms/1000:.1f}s: "
                          f"{os.path.basename(fpath)} ({dur_str}, {vol_db} dB)")

        # ── Text sentence ─────────────────────────────────────────────────────
        elif not segment.startswith('['):
            sentence_count += 1
            t_sentence_start = time.time()
            last_sentence_pos = position_ms

            clean = clean_text(segment)
            print(f"  [*] [Voice {current_voice}] "
                  f"[{current_config['speed']}x, {current_config['volume']:+.0f}dB] "
                  f"{clean[:40]}{'...' if len(clean) > 40 else ''}")

            audio = generate_sentence_audio(
                clean, current_voice, voice_files, tts_instances,
                current_config, current_xtts_params, temp_file
            )
            if audio is not None:
                last_audio   = audio
                voice_audio += audio
                voice_audio += AudioSegment.silent(duration=150)
                position_ms += len(audio) + 150

                t_sentence = time.time() - t_sentence_start
                times_per_sentence.append(t_sentence)
                avg_time      = sum(times_per_sentence) / len(times_per_sentence)
                remaining_est = avg_time * (total_sentences - sentence_count)
                elapsed_total = time.time() - start_time
                pct = int(100 * sentence_count / total_sentences) if total_sentences else 0

                # Human-readable progress (sentence-based)
                print(f"      [*] [{pct:3d}%] [{sentence_count}/{total_sentences}] "
                      f"{int(t_sentence)}s | "
                      f"Elapsed: {timedelta(seconds=int(elapsed_total))} | "
                      f"ETA: {timedelta(seconds=int(remaining_est))}")
                print(f"      [*] Duration: {len(audio)/1000:.2f}s")

                # GUI progress marker (covers ALL phases incl. post-processing)
                step_done += 1
                print(f"[PROGRESS={step_done}/{total_steps}]")

        i += 1

    if os.path.exists(temp_file):
        os.remove(temp_file)

    # ── Mix music ────────────────────────────────────────────────────────────
    print(f"\n[*] Mixing music...")
    final_audio = voice_audio

    if music_to_apply:
        for pos_ms, fpath, dur_sec, vol_db in music_to_apply:
            try:
                music = AudioSegment.from_file(fpath) + vol_db
                if dur_sec:
                    music = music[:int(dur_sec * 1000)]
                final_audio = final_audio.overlay(music, position=pos_ms)
                dur_str = f"{dur_sec}s" if dur_sec else f"{len(music)/1000:.1f}s"
                print(f"   [OK] {os.path.basename(fpath)} at {pos_ms/1000:.1f}s ({dur_str})")
            except Exception as e:
                print(f"   [!]  Error: {e}")
    step_done += 1
    print(f"[PROGRESS={step_done}/{total_steps}]")

    if ambient_file and ambient_vol is not None:
        print(f"   [*] Ambient track: {os.path.basename(ambient_file)}...")
        try:
            ambient = AudioSegment.from_file(ambient_file) + ambient_vol
            total_dur = len(final_audio)
            looped    = AudioSegment.empty()
            while len(looped) < total_dur:
                looped += ambient
            looped      = looped[:total_dur]
            final_audio = looped.overlay(final_audio)
            print(f"   [OK] Applied ({len(looped)/1000:.1f}s, {ambient_vol} dB)")
        except Exception as e:
            print(f"   [!]  Error: {e}")
    step_done += 1
    print(f"[PROGRESS={step_done}/{total_steps}]")

    print(f"\n[*] Saving...")
    final_audio.export(output_file, format="wav")
    step_done += 1
    print(f"[PROGRESS={step_done}/{total_steps}]")

    duration = len(final_audio) / 1000
    total_elapsed = time.time() - start_time
    print(f"\n[OK] Done - VERSION 22 - PARALLEL VOICE OVERLAY")
    print(f"[*] {output_file}")
    print(f"[*] Audio length   : {int(duration // 60)}min {int(duration % 60)}s")
    print(f"[*] Total elapsed  : {timedelta(seconds=int(total_elapsed))}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 4:
        print("Usage: python guided_meditation_generator_v22.py script.txt output.wav voice1.wav [voice2.wav ...] [music...]")
        print("\nVERSION 22 - Parallel voice overlay + rep_pen/len_pen configurables")
        print("\nAUDIO bracket syntax:")
        print("  [N, LANG, speed, vol, eq_low, eq_mid, eq_high, hp, lp, NR, comp, de-ess]")
        print("  [1, FR, 0.85, +1, -5, +1, -1, 90, 9000, 0.3, 0.4, 0.2]")
        print("  [2, EN, 0.90, -3]                   # only speed and volume")
        print("  [1]                                  # just switch voice")
        print("\nXTTS params block (curly braces) — 9 or 11 values:")
        print("  {N, seed, trim_start, trim_end, fade_in, fade_out, temp, top_k, top_p}           <- v20 (still valid)")
        print("  {N, seed, trim_start, trim_end, fade_in, fade_out, temp, top_k, top_p, rep_pen, len_pen}  <- v21/v22")
        print("  {1, 42, 110, 255, 150, 300, 0.65, 50, 0.85, 5.0, 1.0}")
        print("  {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}   # all defaults")
        print("\nPARALLEL block syntax (NEW v22):")
        print("  [parallel]                          # simultaneous, no offset")
        print("  [parallel, offset=1.5s]             # voice 2 starts 1.5s after voice 1")
        print("  {1, ...} [1, FR, ...] Phrase voice 1.")
        print("  {2, ...} [2, FR, ...] Phrase voice 2.")
        print("  [/parallel]")
        print("\nDefault XTTS config:")
        for k, v in DEFAULT_XTTS_PARAMS.items():
            print(f"  {k}: {v}")
        print("\nDefault audio config:")
        for k, v in DEFAULT_AUDIO_CONFIG.items():
            print(f"  {k}: {v}")
        sys.exit(1)

    input_file  = sys.argv[1]
    output_file = sys.argv[2]
    audio_files = sys.argv[3:]

    if not audio_files:
        print("[!] At least one voice file is required!")
        sys.exit(1)

    print("[*] Parsing audio files...")
    voice_files, ambient_file, music_files = parse_audio_files(audio_files)

    if not voice_files:
        print("[!] No voice files detected!")
        sys.exit(1)

    # Read input text (try UTF-8, then fallback encodings)
    text = None
    for encoding in ('utf-8', 'windows-1252', 'latin-1'):
        try:
            with open(input_file, 'r', encoding=encoding) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            print(f"[!]  {encoding} failed, trying next encoding...")
        except FileNotFoundError:
            print(f"[!] Input file not found: {input_file}")
            sys.exit(1)

    if text is None:
        print("[!] Could not read input file (encoding issue).")
        sys.exit(1)

    generate_meditation(text, output_file, voice_files, ambient_file, music_files)

    # Flush stdout and exit hard: os._exit(0) skips Python's normal teardown,
    # which prevents XTTS/torch/CUDA shutdown logs from scrolling past the
    # final "Total elapsed" line in the GUI console.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

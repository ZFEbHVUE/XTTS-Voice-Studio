#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio -> Text transcription with pause detection + per-word pitch annotation
VERSION 21 - faster-whisper backend, shared audio buffer, ASCII output

Uses faster-whisper (CTranslate2) for word-level timestamps to detect
pauses between words. Uses librosa to extract F0 (pitch) per word and
annotate semitone deviations.

Output is compatible with guided_meditation_generator_v20.py

Usage:
  python transcribeSong2txt_with_pause.py audio.mp3 output.txt [model] [min_pause] [language] [--pitch] [--device cpu|cuda] [--vad]

Models (fastest -> most accurate):
  tiny      Very fast, basic quality
  base      Fast, decent quality
  small     Medium speed, good quality       (recommended for GTX 1650)
  medium    Slow, very good quality          (recommended default)
  large-v2  Very slow, excellent quality
  large-v3  Very slow, best quality for non-English

Parameters:
  min_pause : Minimum pause duration in seconds to insert (default: 0.7s)
              0.5s -> all small pauses
              0.7s -> significant pauses  (recommended)
              1.0s -> long pauses only
  language  : Language code (e.g. 'fr', 'en') or omit for auto-detect
  --pitch   : Enable per-word pitch annotation (requires librosa)
  --device  : 'cpu' (default) or 'cuda'. Auto-fallback to CPU if CUDA unavailable.
  --vad     : Enable Silero VAD pre-filtering (skips long silences, faster)

Examples:
  python transcribeSong2txt_with_pause.py audio.mp3 output.txt
  python transcribeSong2txt_with_pause.py audio.mp3 output.txt medium
  python transcribeSong2txt_with_pause.py audio.mp3 output.txt small 0.5
  python transcribeSong2txt_with_pause.py audio.mp3 output.txt medium 0.7 fr --pitch
  python transcribeSong2txt_with_pause.py audio.mp3 output.txt large-v3 1.0 en --pitch --device cuda --vad

Output format WITHOUT --pitch (compatible with guided_meditation_generator_v20.py):
  Hello, welcome to this session.
  [pause=3.2s]
  Take a deep breath.
  [pause=2.1s]

Output format WITH --pitch (each word annotated with semitone deviation):
  Hello[p:0] welcome[p:+2] to[p:+1] this[p:-1] session[p:-2].
  [pause=3.2s]
  Take[p:+1] a[p:0] deep[p:-2] breath[p:-3].
  [pause=2.1s]

Pitch annotation meaning:
  [p:0]   -> word spoken at median pitch of the audio
  [p:?]   -> unvoiced word (whispered, percussive) - no pitch detected
  [p:+2]  -> word spoken 2 semitones above median (rising intonation)
  [p:-3]  -> word spoken 3 semitones below median (falling intonation)

Note: Pitch annotations are intended for post-processing with rubberband.
      XTTS v2 itself does not use them during generation.

CHANGES vs v20:
  - Backend: openai-whisper -> faster-whisper (2-4x faster, same quality)
  - Audio loaded ONCE with librosa at native sr, resampled to 16k for whisper
  - All emojis replaced with ASCII tags ([OK]/[!]/[*])
  - CUDA -> CPU automatic fallback if GPU unavailable
  - Optional Silero VAD pre-filter (--vad)
  - Targeted warning filter instead of global ignore
  - Unvoiced words tagged [p:?] instead of misleading [p:0]
"""

import sys
import os
import time
import warnings

# Targeted filter: silence the noisy whisper/torch warnings only,
# not our own UserWarnings.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='ctranslate2')


# ─────────────────────────────────────────────────────────────────────────────
# Audio loading (librosa - shared between F0 analysis and Whisper input)
# ─────────────────────────────────────────────────────────────────────────────

def load_audio_shared(audio_file):
    """
    Load audio ONCE with librosa.
    Returns (y_native, sr_native, y_16k, librosa_module) or (None, None, None, None).

    y_native is kept at the file's native sample rate for accurate F0 analysis.
    y_16k is resampled to 16 kHz for Whisper input (avoids double-loading via ffmpeg).
    """
    try:
        import librosa
        import numpy as np
        print("   [*] Loading audio (single pass, shared with Whisper)...")
        y_native, sr_native = librosa.load(audio_file, sr=None, mono=True)

        if sr_native != 16000:
            y_16k = librosa.resample(y_native, orig_sr=sr_native, target_sr=16000)
        else:
            y_16k = y_native

        # Whisper expects float32 mono in [-1, 1]
        y_16k = y_16k.astype(np.float32)
        return y_native, sr_native, y_16k, librosa
    except ImportError:
        print("   [!] librosa not installed")
        print("       Install with: pip install librosa")
        return None, None, None, None
    except Exception as e:
        print(f"   [!] Audio load error: {e}")
        return None, None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# F0 / pitch analysis (yin, single pass over the whole file)
# ─────────────────────────────────────────────────────────────────────────────

def compute_f0_full(y, sr, librosa):
    """
    Compute F0 curve ONCE over the whole audio using yin (fast).
    Returns (f0_array, f0_median, times_array) or (None, None, None).
    yin is ~10x faster than pyin and sufficient for semitone-level annotation.
    """
    import numpy as np

    hop_length = 512
    try:
        print("   [*] Computing F0 over full audio (one pass)...")
        f0_raw = librosa.yin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'),
            sr=sr,
            hop_length=hop_length
        )
        # Mark unvoiced frames as NaN
        voiced = (f0_raw > librosa.note_to_hz('C2')) & \
                 (f0_raw < librosa.note_to_hz('C6'))
        f0 = np.where(voiced, f0_raw, np.nan)

        f0_voiced = f0[~np.isnan(f0)]
        if len(f0_voiced) < 5:
            print("   [!] Not enough voiced frames - pitch tags will be [p:?]")
            return None, None, None

        f0_median = float(np.median(f0_voiced))
        times = librosa.frames_to_time(
            np.arange(len(f0)), sr=sr, hop_length=hop_length
        )
        print(f"   [*] F0 median: {f0_median:.0f} Hz  "
              f"({len(f0_voiced)} voiced frames / {len(f0)} total)")
        return f0, f0_median, times

    except Exception as e:
        print(f"   [!] F0 computation error: {e}")
        return None, None, None


def get_word_pitch_semitones(f0, f0_median, times, t_start, t_end):
    """
    Read pre-computed F0 frames for a word time range and return
    deviation in semitones relative to f0_median.
    Returns None if no voiced frames in range (unvoiced word).
    """
    import numpy as np

    if f0 is None or f0_median is None or f0_median <= 0:
        return None

    mask = (times >= t_start) & (times <= t_end)
    f0_segment = f0[mask]
    f0_voiced  = f0_segment[~np.isnan(f0_segment)]

    if len(f0_voiced) == 0:
        return None  # unvoiced

    f0_mean   = float(np.mean(f0_voiced))
    semitones = 12.0 * np.log2(f0_mean / f0_median)
    return int(round(semitones))


def format_pitch_tag(semitones):
    """Format semitone value as [p:+2], [p:-1], [p:0], or [p:?] for unvoiced."""
    if semitones is None:
        return "[p:?]"
    if semitones == 0:
        return "[p:0]"
    elif semitones > 0:
        return f"[p:+{semitones}]"
    else:
        return f"[p:{semitones}]"


def annotate_word(word_text, semitones):
    """
    Attach pitch annotation to a word, preserving trailing punctuation.
    Example: "breath." -> "breath[p:-2]."
    """
    import re
    m = re.match(r'^(.*?)([.,!?;:]+)$', word_text)
    if m:
        core  = m.group(1)
        punct = m.group(2)
        return f"{core}{format_pitch_tag(semitones)}{punct}"
    else:
        return f"{word_text}{format_pitch_tag(semitones)}"


# ─────────────────────────────────────────────────────────────────────────────
# Device handling (CUDA -> CPU fallback)
# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(requested):
    """
    Validate the requested device and fall back gracefully.
    Returns (device, compute_type) suitable for faster-whisper.
    """
    if requested == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                print("   [!] CUDA requested but no GPU available - falling back to CPU")
                return "cpu", "int8"
            return "cuda", "float16"
        except ImportError:
            print("   [!] torch not available for CUDA check - falling back to CPU")
            return "cpu", "int8"
    return "cpu", "int8"


# ─────────────────────────────────────────────────────────────────────────────
# Main transcription
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_with_pauses(audio_file, output_file,
                            model_name="medium", min_pause=0.7,
                            language=None, pitch=False,
                            device="cpu", vad=False):
    """
    Transcribe an audio file and insert pause markers between words.
    Optionally annotates each word with its pitch deviation in semitones.
    """

    print(f"[*] Transcribing with model '{model_name}' (faster-whisper)")
    print(f"   [*] Minimum pause threshold : {min_pause}s")
    print(f"   [*] Language               : {language if language else 'auto-detect'}")
    print(f"   [*] Pitch annotation       : {'enabled' if pitch else 'disabled (use --pitch to enable)'}")
    print(f"   [*] VAD pre-filter         : {'enabled' if vad else 'disabled'}")

    device, compute_type = resolve_device(device)
    print(f"   [*] Device                 : {device} ({compute_type})")
    print(f"   (This may take several minutes depending on file size)")

    t_global = time.time()

    # ── Load audio ONCE (shared between F0 + Whisper) ─────────────────────
    f0_array  = None
    f0_median = None
    f0_times  = None
    whisper_input = audio_file  # default: let faster-whisper read the file itself

    if pitch:
        t = time.time()
        y_native, sr_native, y_16k, lib = load_audio_shared(audio_file)
        if y_native is not None:
            f0_array, f0_median, f0_times = compute_f0_full(y_native, sr_native, lib)
            print(f"   [*] F0 analysis done in {time.time()-t:.1f}s")
            if f0_median is None:
                print("   [!] Pitch analysis disabled (no voiced frames)")
                pitch = False
            # Reuse the resampled buffer for Whisper to skip a redundant ffmpeg load
            whisper_input = y_16k
        else:
            print("   [!] Pitch analysis disabled due to audio load failure")
            pitch = False

    # ── Load faster-whisper model ─────────────────────────────────────────
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[!] faster-whisper not installed")
        print("    Install with: pip install faster-whisper")
        sys.exit(1)

    t = time.time()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    print(f"   [*] Whisper model loaded in {time.time()-t:.1f}s")

    # ── Transcribe with word-level timestamps ──────────────────────────────
    print("   [*] Analysing word-level timestamps...")
    t = time.time()

    transcribe_kwargs = dict(
        word_timestamps=True,
        beam_size=5,
    )
    if language:
        transcribe_kwargs['language'] = language
    if vad:
        transcribe_kwargs['vad_filter']     = True
        transcribe_kwargs['vad_parameters'] = dict(min_silence_duration_ms=500)

    segments_gen, info = model.transcribe(whisper_input, **transcribe_kwargs)

    # faster-whisper returns a generator - materialise it (we need 2 passes:
    # one for segment-to-segment pause detection, and one for word loops).
    segments = list(segments_gen)
    detected = info.language
    print(f"   [OK] Transcription done in {time.time()-t:.1f}s - language: {detected}")
    print(f"[*] Detecting pauses between words (threshold >= {min_pause}s)...")

    # ── Build output with pauses (and optional pitch) ──────────────────────
    lines    = []
    n_pauses = 0
    n_pitch  = 0
    n_unvoiced = 0

    for seg_idx, segment in enumerate(segments):
        # faster-whisper Segment.words is a list of Word(start, end, word, probability)
        words = segment.words or []

        if not words:
            text = (segment.text or '').strip()
            if text:
                lines.append(text)
            continue

        # ── Fuse tokens split by apostrophe or hyphen ─────────────────────
        # Whisper splits "qu'on" -> ["qu", "'on"] and "Saint-Michel" -> ["Saint", "-Michel"]
        fused = []
        for w in words:
            raw = w.word  # keep original spacing
            stripped = raw.strip()
            if stripped and stripped[0] in ("'", "\u2019", "-") and fused:
                prev = fused[-1]
                prev['word'] = prev['word'].rstrip() + raw.rstrip()
                prev['end']  = w.end if w.end is not None else prev['end']
            else:
                fused.append({
                    'word' : raw,
                    'start': w.start if w.start is not None else 0.0,
                    'end'  : w.end   if w.end   is not None else 0.0,
                })

        # ── Word-level processing on fused tokens ─────────────────────────
        for word_idx, word in enumerate(fused):
            word_text = word['word'].strip()

            if word_text:
                if pitch and f0_array is not None:
                    t_start   = word['start']
                    t_end     = word['end'] if word['end'] > t_start else t_start + 0.1
                    semitones = get_word_pitch_semitones(
                        f0_array, f0_median, f0_times, t_start, t_end
                    )
                    word_text = annotate_word(word_text, semitones)
                    if semitones is None:
                        n_unvoiced += 1
                    else:
                        n_pitch += 1

                if lines and not lines[-1].startswith('[pause='):
                    lines[-1] += " " + word_text
                else:
                    lines.append(word_text)

            if word_idx < len(fused) - 1:
                gap = fused[word_idx + 1]['start'] - word['end']
                if gap >= min_pause:
                    lines.append(f"[pause={gap:.1f}s]")
                    n_pauses += 1

        # ── Pause between segments ─────────────────────────────────────────
        if seg_idx < len(segments) - 1:
            gap = segments[seg_idx + 1].start - segment.end
            if gap >= min_pause:
                lines.append(f"[pause={gap:.1f}s]")
                n_pauses += 1

    # ── Save ──────────────────────────────────────────────────────────────
    output_path = os.path.abspath(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    n_text = len([l for l in lines if not l.startswith('[pause=')])

    print(f"\n[OK] Saved: {output_path}")
    print(f"   [*] Text lines       : {n_text}")
    print(f"   [*] Pauses           : {n_pauses}  (>= {min_pause}s)")
    print(f"   [*] Words annotated  : {n_pitch}")
    if pitch:
        print(f"   [*] Unvoiced words   : {n_unvoiced}  (tagged [p:?])")
    print(f"   [*] Ratio            : {n_pauses / max(n_text, 1):.1f} pauses per line")
    print(f"   [*] Total time       : {time.time()-t_global:.1f}s")

    if pitch:
        print(f"\n[*] Pitch tags [p:+/-N] = semitone deviation from median pitch")
        print(f"    Use with rubberband post-processing in guided_meditation_generator_v20.py")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(0)

    args        = sys.argv[1:]
    pitch_mode  = '--pitch' in args
    args        = [a for a in args if a != '--pitch']

    vad_mode    = '--vad' in args
    args        = [a for a in args if a != '--vad']

    # --device cpu|cuda
    device = "cpu"
    if '--device' in args:
        idx = args.index('--device')
        if idx + 1 < len(args):
            device = args[idx + 1]
            args   = [a for i, a in enumerate(args) if i != idx and i != idx + 1]

    audio_file  = args[0]
    output_file = args[1]
    model_name  = args[2] if len(args) > 2 else "medium"
    min_pause   = float(args[3]) if len(args) > 3 else 0.7
    language    = args[4] if len(args) > 4 else None

    if not os.path.exists(audio_file):
        print(f"[!] File not found: {audio_file}")
        sys.exit(1)

    transcribe_with_pauses(
        audio_file,
        output_file,
        model_name = model_name,
        min_pause  = min_pause,
        language   = language,
        pitch      = pitch_mode,
        device     = device,
        vad        = vad_mode,
    )

    # Hard exit to skip torch/ctranslate2 teardown logs that would scroll past
    # the final timing report in the GUI console.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

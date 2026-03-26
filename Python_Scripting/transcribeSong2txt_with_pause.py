#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio → Text transcription with pause detection
Uses OpenAI Whisper word-level timestamps to detect pauses between words.
Output is compatible with guided_meditation_generator_v20.py

Usage:
  python transcribe_to_txt.py audio.mp3 output.txt [model] [min_pause]

Models (fastest → most accurate):
  tiny      Very fast, basic quality
  base      Fast, decent quality
  small     Medium speed, good quality       (recommended for GTX 1650)
  medium    Slow, very good quality          (recommended default)
  large     Very slow, excellent quality
  large-v3  Very slow, best quality for non-English

Parameters:
  min_pause : Minimum pause duration in seconds to insert (default: 0.7s)
              0.5s → all small pauses
              0.7s → significant pauses  (recommended)
              1.0s → long pauses only

Examples:
  python transcribe_to_txt.py audio.mp3 output.txt
  python transcribe_to_txt.py audio.mp3 output.txt medium
  python transcribe_to_txt.py audio.mp3 output.txt small 0.5
  python transcribe_to_txt.py audio.mp3 output.txt large-v3 1.0

Output format (compatible with guided_meditation_generator_v20.py):
  Hello, welcome to this session.
  [pause=3.2s]
  Take a deep breath.
  [pause=2.1s]
  ...
"""

import whisper
import sys
import os


def transcribe_with_pauses(audio_file, output_file,
                           model_name="medium", min_pause=0.7,
                           language=None):
    """
    Transcribe an audio file and insert pause markers between words.

    Args:
        audio_file   : Path to audio file (mp3, wav, flac, m4a, ...)
        output_file  : Path to output text file
        model_name   : Whisper model name (tiny/base/small/medium/large/large-v3)
        min_pause    : Minimum pause duration in seconds to insert (default: 0.7)
        language     : Language code (e.g. 'fr', 'en') or None for auto-detect
    """

    print(f"🎤  Transcribing with model '{model_name}'...")
    print(f"   ⏸️  Minimum pause threshold : {min_pause}s")
    print(f"   🌐  Language               : {language if language else 'auto-detect'}")
    print(f"   (This may take several minutes depending on file size)")

    # ── Load model ────────────────────────────────────────────────────────
    model = whisper.load_model(model_name, device="cpu")

    # ── Transcribe with word-level timestamps ─────────────────────────────
    print("   🔄  Analysing word-level timestamps...")
    transcribe_opts = dict(
        word_timestamps=True,   # key feature: pause detection at word level
        verbose=False,
    )
    if language:
        transcribe_opts['language'] = language

    result = model.transcribe(audio_file, **transcribe_opts)

    detected_language = result.get('language', 'unknown')
    print(f"   ✅  Detected language: {detected_language}")
    print(f"🔇  Detecting pauses between words (threshold ≥ {min_pause}s)...")

    # ── Build output with pauses ──────────────────────────────────────────
    lines   = []
    n_pauses = 0
    segments = result['segments']

    for seg_idx, segment in enumerate(segments):
        words = segment.get('words', [])

        if not words:
            # No word timestamps available — add segment as-is
            text = segment['text'].strip()
            if text:
                lines.append(text)
            continue

        # ── Word-level pause detection ────────────────────────────────────
        for word_idx, word in enumerate(words):
            word_text = word['word'].strip()

            if word_text:
                # Append to current line or start a new one
                if lines and not lines[-1].startswith('[pause='):
                    lines[-1] += " " + word_text
                else:
                    lines.append(word_text)

            # Pause between this word and the next (within segment)
            if word_idx < len(words) - 1:
                gap = words[word_idx + 1].get('start', 0) - word.get('end', 0)
                if gap >= min_pause:
                    lines.append(f"[pause={gap:.1f}s]")
                    n_pauses += 1

        # ── Pause between this segment and the next ───────────────────────
        if seg_idx < len(segments) - 1:
            gap = segments[seg_idx + 1]['start'] - segment['end']
            if gap >= min_pause:
                lines.append(f"[pause={gap:.1f}s]")
                n_pauses += 1

    # ── Save ──────────────────────────────────────────────────────────────
    output_path = os.path.abspath(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    n_text = len([l for l in lines if not l.startswith('[pause=')])

    print(f"\n✅  Saved: {output_path}")
    print(f"   📊  Text lines : {n_text}")
    print(f"   ⏸️  Pauses     : {n_pauses}  (≥ {min_pause}s)")
    print(f"   📈  Ratio      : {n_pauses / max(n_text, 1):.1f} pauses per line")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(0)

    audio_file  = sys.argv[1]
    output_file = sys.argv[2]
    model_name  = sys.argv[3] if len(sys.argv) > 3 else "medium"
    min_pause   = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
    language    = sys.argv[5] if len(sys.argv) > 5 else None

    if not os.path.exists(audio_file):
        print(f"❌  File not found: {audio_file}")
        sys.exit(1)

    transcribe_with_pauses(
        audio_file,
        output_file,
        model_name=model_name,
        min_pause=min_pause,
        language=language,
    )


if __name__ == "__main__":
    main()

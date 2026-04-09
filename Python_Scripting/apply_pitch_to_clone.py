#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply pitch corrections to an existing cloned voice WAV file.
Uses annotated .txt (with [p:±N] tags) + Whisper timestamps + rubberband.

Usage:
  python apply_pitch_to_clone.py clone.wav annotated.txt output.wav [options]

Options:
  --global-shift N   Apply a global pitch shift of N semitones first (default: 0)
                     Use this to match the original's median pitch.
                     Example: --global-shift +4
  --lang LANG        Language for Whisper (default: fr)
  --model MODEL      Whisper model: tiny/base/small/medium (default: small)
  --no-per-word      Skip per-word pitch, only apply global shift

Examples:
  # Global shift only
  python apply_pitch_to_clone.py clone.wav script.txt output.wav --global-shift +4

  # Global shift + per-word pitch
  python apply_pitch_to_clone.py clone.wav script.txt output.wav --global-shift +4 --lang fr

  # Per-word only (no global shift)
  python apply_pitch_to_clone.py clone.wav script.txt output.wav

Dependencies:
  pip install pydub openai-whisper
  apt install rubberband-cli
"""

import sys
import os
import re
import subprocess
import tempfile
import argparse
import time

try:
    from pydub import AudioSegment
except ImportError:
    sys.exit("❌  pydub missing → pip install pydub")

try:
    import whisper
except ImportError:
    sys.exit("❌  whisper missing → pip install openai-whisper")


# ─────────────────────────────────────────────────────────────────────────────
# Pitch tag parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_pitch_tags(text):
    """
    Extract per-word pitch annotations from text.
    Returns (clean_text, pitch_map)
      clean_text : text without [p:N] tags
      pitch_map  : {word: semitones}  — only words with semitones != 0
    """
    pitch_map = {}
    RE_WORD_TAG = re.compile(r'(\S+?)\[p:([+-]?\d+)\]')
    RE_TAG      = re.compile(r'\[p:[+-]?\d+\]')

    for m in RE_WORD_TAG.finditer(text):
        word_raw  = m.group(1)
        semitones = int(m.group(2))
        key = re.sub(r'[.,!?;:]+$', '', word_raw)
        if key and semitones != 0:
            pitch_map[key] = semitones

    clean = RE_TAG.sub('', text).strip()
    return clean, pitch_map


def load_annotated_txt(txt_file):
    """
    Read annotated .txt and return list of (clean_sentence, pitch_map).
    Ignores [pause=...], {XTTS params}, [audio params], # comments.
    """
    sentences = []
    with open(txt_file, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                continue
            if line.startswith('[pause') or line.startswith('[music') or \
               line.startswith('[ambiance') or line.startswith('{') or \
               (line.startswith('[') and ',' in line and not '[p:' in line):
                continue
            # Remove inline [pause=...] from text lines
            line = re.sub(r'\[pause=[^\]]+\]', '', line).strip()
            if not line:
                continue
            clean, pitch_map = parse_pitch_tags(line)
            clean = clean.strip()
            if clean:
                sentences.append((clean, pitch_map))
    return sentences


# ─────────────────────────────────────────────────────────────────────────────
# Rubberband helpers
# ─────────────────────────────────────────────────────────────────────────────

def rubberband_pitch(audio_segment, semitones, tmp_dir='/tmp'):
    """Apply pitch shift of N semitones using rubberband."""
    if semitones == 0:
        return audio_segment

    tmp_in  = os.path.join(tmp_dir, 'rb_pitch_in.wav')
    tmp_out = os.path.join(tmp_dir, 'rb_pitch_out.wav')

    audio_segment.export(tmp_in, format='wav')
    result = subprocess.run(
        ['rubberband', '--pitch', str(semitones), '-c', '1', tmp_in, tmp_out],
        capture_output=True
    )
    if result.returncode != 0:
        print(f"  ⚠️  rubberband error: {result.stderr.decode()[:200]}")
        return audio_segment

    shifted = AudioSegment.from_wav(tmp_out)
    for f in (tmp_in, tmp_out):
        if os.path.exists(f): os.remove(f)
    return shifted


# ─────────────────────────────────────────────────────────────────────────────
# Per-word pitch via Whisper timestamps
# ─────────────────────────────────────────────────────────────────────────────

def apply_pitch_per_word(audio_segment, pitch_map, sentence_text,
                          whisper_model, language='fr', tmp_dir='/tmp'):
    """
    Apply per-word pitch using Whisper timestamps on the audio segment.
    """
    if not pitch_map:
        return audio_segment, 0

    tmp_in = os.path.join(tmp_dir, 'pw_in.wav')
    audio_segment.export(tmp_in, format='wav')

    result = whisper_model.transcribe(
        tmp_in,
        word_timestamps=True,
        language=language,
        verbose=False
    )

    # Build word → (start_ms, end_ms) map
    word_times = {}
    for seg in result['segments']:
        for w in seg.get('words', []):
            raw = w['word'].strip()
            # Fuse apostrophe/hyphen fragments
            key = re.sub(r'[.,!?;:]+$', '', raw).lower()
            if key:
                word_times[key] = (
                    int(w['start'] * 1000),
                    int(w['end']   * 1000)
                )

    n_applied = 0
    tmp_seg_in  = os.path.join(tmp_dir, 'seg_in.wav')
    tmp_seg_out = os.path.join(tmp_dir, 'seg_out.wav')

    for word, semitones in pitch_map.items():
        if semitones == 0:
            continue
        key = re.sub(r'[.,!?;:]+$', '', word).lower()
        if key not in word_times:
            # Try partial match
            matches = [k for k in word_times if k in key or key in k]
            if matches:
                key = matches[0]
            else:
                continue

        t_start_ms, t_end_ms = word_times[key]
        if t_end_ms <= t_start_ms:
            continue

        seg_audio = audio_segment[t_start_ms:t_end_ms]
        seg_audio.export(tmp_seg_in, format='wav')

        rb = subprocess.run(
            ['rubberband', '--pitch', str(semitones), '-c', '1',
             tmp_seg_in, tmp_seg_out],
            capture_output=True
        )
        if rb.returncode != 0:
            continue

        seg_pitched = AudioSegment.from_wav(tmp_seg_out)
        audio_segment = (
            audio_segment[:t_start_ms] +
            seg_pitched +
            audio_segment[t_end_ms:]
        )
        n_applied += 1
        print(f"      🎵  {word}: {semitones:+d}st")

        for f in (tmp_seg_in, tmp_seg_out):
            if os.path.exists(f): os.remove(f)

    if os.path.exists(tmp_in): os.remove(tmp_in)
    return audio_segment, n_applied


# ─────────────────────────────────────────────────────────────────────────────
# Sentence alignment via Whisper on full file
# ─────────────────────────────────────────────────────────────────────────────

def align_sentences_to_audio(audio, sentences, whisper_model, language, tmp_dir):
    """
    Use Whisper on full audio to find approximate start/end of each sentence.
    Returns list of (start_ms, end_ms, sentence_idx).
    Simple approach: match Whisper segments to sentence list by order.
    """
    print("  🔄  Aligning sentences to audio with Whisper...")
    tmp_full = os.path.join(tmp_dir, 'full_align.wav')
    audio.export(tmp_full, format='wav')

    result = whisper_model.transcribe(
        tmp_full,
        word_timestamps=True,
        language=language,
        verbose=False
    )

    if os.path.exists(tmp_full): os.remove(tmp_full)

    # Group words into segments approximating our sentences
    segments = result['segments']
    alignments = []

    # Simple mapping: one Whisper segment ≈ one sentence
    for i, (clean, _) in enumerate(sentences):
        if i < len(segments):
            seg = segments[i]
            start_ms = int(seg['start'] * 1000)
            end_ms   = int(seg['end']   * 1000)
            alignments.append((start_ms, end_ms, i))

    return alignments


# ─────────────────────────────────────────────────────────────────────────────
# Main processing
# ─────────────────────────────────────────────────────────────────────────────

def process(clone_wav, txt_file, output_wav,
            global_shift=0, language='fr',
            whisper_model_name='small', per_word=True):

    t0 = time.time()
    print(f"\n🎵  Pitch correction pipeline")
    print(f"   Clone    : {os.path.basename(clone_wav)}")
    print(f"   Script   : {os.path.basename(txt_file)}")
    print(f"   Output   : {os.path.basename(output_wav)}")
    print(f"   Global   : {global_shift:+d} semitones")
    print(f"   Per-word : {'enabled (Whisper ' + whisper_model_name + ')' if per_word else 'disabled'}")

    # Load audio
    print("\n📂  Loading audio...")
    audio = AudioSegment.from_file(clone_wav)
    dur   = len(audio) / 1000
    print(f"   Duration : {dur:.1f}s  SR:{audio.frame_rate}Hz  Ch:{audio.channels}")

    tmp_dir = tempfile.mkdtemp()

    # ── Step 1: Global pitch shift ─────────────────────────────────────────
    if global_shift != 0:
        print(f"\n🎚️  Applying global pitch shift: {global_shift:+d} semitones...")
        t = time.time()
        audio = rubberband_pitch(audio, global_shift, tmp_dir)
        print(f"   ⏱️  Done in {time.time()-t:.1f}s")

    # ── Step 2: Per-word pitch ─────────────────────────────────────────────
    if per_word:
        print(f"\n📝  Loading annotated script...")
        sentences = load_annotated_txt(txt_file)
        has_pitch = [(c, p) for c, p in sentences if p]
        print(f"   {len(sentences)} sentences  —  {len(has_pitch)} with pitch annotations")

        if has_pitch:
            print(f"\n🔬  Loading Whisper '{whisper_model_name}'...")
            t = time.time()
            wmodel = whisper.load_model(whisper_model_name, device="cpu")
            print(f"   ⏱️  Loaded in {time.time()-t:.1f}s")

            print(f"\n🎵  Applying per-word pitch (full audio + Whisper alignment)...")
            t = time.time()

            # Transcribe full audio to get word timestamps
            tmp_full = os.path.join(tmp_dir, 'full.wav')
            audio.export(tmp_full, format='wav')
            result = wmodel.transcribe(
                tmp_full,
                word_timestamps=True,
                language=language,
                verbose=False
            )
            if os.path.exists(tmp_full): os.remove(tmp_full)
            print(f"   ⏱️  Whisper done in {time.time()-t:.1f}s")

            # Build global word→time map
            word_times = {}
            for seg in result['segments']:
                for w in seg.get('words', []):
                    raw = w['word'].strip()
                    key = re.sub(r'[.,!?;:]+$', '', raw).lower()
                    if key:
                        word_times[key] = (
                            int(w['start'] * 1000),
                            int(w['end']   * 1000)
                        )

            # Collect all pitch corrections
            all_corrections = {}
            for clean, pitch_map in sentences:
                for word, semitones in pitch_map.items():
                    if semitones != 0:
                        key = re.sub(r'[.,!?;:]+$', '', word).lower()
                        all_corrections[key] = semitones

            # Apply all corrections
            print(f"   Applying {len(all_corrections)} unique word corrections...")
            tmp_seg_in  = os.path.join(tmp_dir, 'seg_in.wav')
            tmp_seg_out = os.path.join(tmp_dir, 'seg_out.wav')
            n_applied = 0

            for key, semitones in sorted(all_corrections.items(),
                                          key=lambda x: word_times.get(
                                              re.sub(r'[.,!?;:]+$','',x[0]).lower(),
                                              (0,0))[0]):
                if key not in word_times:
                    matches = [k for k in word_times if k in key or key in k]
                    if matches:
                        key = matches[0]
                    else:
                        continue

                t_start_ms, t_end_ms = word_times[key]
                if t_end_ms <= t_start_ms or t_end_ms > len(audio):
                    continue

                seg_audio = audio[t_start_ms:t_end_ms]
                seg_audio.export(tmp_seg_in, format='wav')

                rb = subprocess.run(
                    ['rubberband', '--pitch', str(semitones), '-c', '1',
                     tmp_seg_in, tmp_seg_out],
                    capture_output=True
                )
                if rb.returncode != 0:
                    continue

                seg_pitched = AudioSegment.from_wav(tmp_seg_out)
                audio = (
                    audio[:t_start_ms] +
                    seg_pitched +
                    audio[t_end_ms:]
                )
                n_applied += 1

                for f in (tmp_seg_in, tmp_seg_out):
                    if os.path.exists(f): os.remove(f)

            print(f"   ✅  {n_applied} word corrections applied")

    # ── Save ──────────────────────────────────────────────────────────────
    print(f"\n💾  Saving → {output_wav}")
    audio.export(output_wav, format='wav')
    print(f"   ⏱️  Total time: {time.time()-t0:.1f}s")
    print(f"✅  Done")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Apply pitch corrections to a cloned voice WAV file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('clone_wav',   help='Input clone WAV file')
    parser.add_argument('txt_file',    help='Annotated .txt with [p:±N] tags')
    parser.add_argument('output_wav',  help='Output WAV file')
    parser.add_argument('--global-shift', type=int, default=0,
                        help='Global pitch shift in semitones (default: 0)')
    parser.add_argument('--lang',    default='fr',
                        help='Language for Whisper (default: fr)')
    parser.add_argument('--model',   default='small',
                        choices=['tiny','base','small','medium','large'],
                        help='Whisper model (default: small)')
    parser.add_argument('--no-per-word', action='store_true',
                        help='Skip per-word pitch, only apply global shift')

    args = parser.parse_args()

    if not os.path.exists(args.clone_wav):
        sys.exit(f"❌  File not found: {args.clone_wav}")
    if not os.path.exists(args.txt_file):
        sys.exit(f"❌  File not found: {args.txt_file}")

    process(
        clone_wav         = args.clone_wav,
        txt_file          = args.txt_file,
        output_wav        = args.output_wav,
        global_shift      = args.global_shift,
        language          = args.lang,
        whisper_model_name= args.model,
        per_word          = not args.no_per_word,
    )


if __name__ == '__main__':
    main()

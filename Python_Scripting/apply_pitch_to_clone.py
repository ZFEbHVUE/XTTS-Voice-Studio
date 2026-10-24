#!/usr/bin/env python3
"""Apply [p:+N] word pitch marks to an existing clone using Whisper/Rubberband.

Corrections belong to token occurrences, including repeated words. Exact ordered
ASR matches are required; unmatched marks are reported and skipped. Pitch-only
edits preserve the input duration and use Rubberband formant preservation.

Dependencies: pydub, openai-whisper, rubberband-cli.
"""
import argparse
import math
import os
import re
import subprocess
import tempfile
from text_alignment import words, ordered_matches


def parse_pitch_tags(text):
    """Return clean text and ordered records (word positions, word, semitones)."""
    clean_parts, corrections, offset = [], [], 0
    for raw in text.split():
        tags = re.findall(r'\[p:([+-]?\d+)\]', raw)
        clean = re.sub(r'\[p:[+-]?\d+\]', '', raw)
        pieces = words(clean)
        if len(tags) > 1:
            raise ValueError(f"Multiple pitch tags on one token: {raw}")
        if tags and pieces and int(tags[0]):
            shift = int(tags[0])
            if abs(shift) > 24:
                raise ValueError("Per-word pitch must be in [-24, 24] semitones")
            corrections.append(dict(start=offset, end=offset + len(pieces),
                                    word=clean, semitones=shift))
        clean_parts.append(clean)
        offset += len(pieces)
    return ' '.join(clean_parts).strip(), corrections


def load_annotated_txt(txt_file):
    sentences = []
    with open(txt_file, encoding='utf-8-sig') as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(('#', '{', '[pause', '[music', '[ambiance')):
                continue
            if line.startswith('[') and ',' in line and '[p:' not in line:
                continue
            line = re.sub(r'\[pause=[^\]]+\]', ' ', line)
            clean, corrections = parse_pitch_tags(line)
            if words(clean):
                sentences.append((clean, corrections))
    return sentences


def combine_annotations(sentences):
    texts, corrections, offset = [], [], 0
    for text, marks in sentences:
        texts.append(text)
        corrections.extend(dict(mark, start=mark['start'] + offset,
                                end=mark['end'] + offset) for mark in marks)
        offset += len(words(text))
    return ' '.join(texts), corrections


def aligned_corrections(sentence_text, corrections, result, duration_ms):
    observed, timings = [], []
    for segment in result.get('segments', []):
        for word in segment.get('words', []):
            start, end = float(word['start']), float(word['end'])
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                continue
            for part in words(word.get('word', '')):
                observed.append(part)
                timings.append((max(0, round(start * 1000)), min(duration_ms, round(end * 1000))))
    mapping = ordered_matches(words(sentence_text), observed)
    edits, previous_end = [], 0
    for mark in corrections:
        indices = list(range(mark['start'], mark['end']))
        if not indices or not all(i in mapping for i in indices):
            print(f"[!] Unmatched pitch mark skipped: {mark['word']}")
            continue
        found = [mapping[i] for i in indices]
        if found != list(range(found[0], found[-1] + 1)):
            continue
        begin, end = timings[found[0]][0], timings[found[-1]][1]
        if begin < previous_end or end <= begin:
            print(f"[!] Overlapping/invalid pitch interval skipped: {mark['word']}")
            continue
        edits.append((begin, end, mark['semitones'], mark['word']))
        previous_end = end
    return edits


def rubberband_pitch(audio_segment, semitones, tmp_dir=None):
    from pydub import AudioSegment
    if not math.isfinite(semitones) or abs(semitones) > 24:
        raise ValueError("Pitch must be finite and in [-24, 24] semitones")
    if semitones == 0 or not len(audio_segment):
        return audio_segment
    with tempfile.TemporaryDirectory(prefix='xtts_pitch_', dir=tmp_dir) as work:
        source, target = os.path.join(work, 'in.wav'), os.path.join(work, 'out.wav')
        audio_segment.export(source, format='wav')
        subprocess.run(['rubberband', '--pitch', str(semitones), '--formant',
                        '-c', '1', source, target], check=True, capture_output=True)
        shifted = AudioSegment.from_wav(target).set_frame_rate(audio_segment.frame_rate)
        shifted = shifted.set_channels(audio_segment.channels).set_sample_width(audio_segment.sample_width)
        # Exact frame count prevents later word timestamps drifting after each edit.
        wanted = len(audio_segment.raw_data)
        raw = shifted.raw_data[:wanted].ljust(wanted, b'\0')
        return audio_segment._spawn(raw)


def _apply_edits(audio, edits, tmp_dir):
    count = 0
    for start, end, shift, word in reversed(edits):
        segment = audio[start:end]
        shifted = rubberband_pitch(segment, shift, tmp_dir)
        # Splice raw frames at the same sample boundaries, without ms rounding.
        first = int(audio.frame_count(ms=start)) * audio.frame_width
        last = first + len(segment.raw_data)
        audio = audio._spawn(audio.raw_data[:first] + shifted.raw_data + audio.raw_data[last:])
        print(f"  {word}: {shift:+d} semitones ({start}-{end} ms)")
        count += 1
    return audio, count


def apply_pitch_per_word(audio_segment, pitch_map, sentence_text,
                         whisper_model, language='fr', tmp_dir=None):
    if not pitch_map:
        return audio_segment, 0
    with tempfile.TemporaryDirectory(prefix='xtts_align_', dir=tmp_dir) as work:
        source = os.path.join(work, 'audio.wav')
        audio_segment.export(source, format='wav')
        result = whisper_model.transcribe(source, word_timestamps=True,
                                          language=language, verbose=False)
        edits = aligned_corrections(sentence_text, pitch_map, result, len(audio_segment))
        return _apply_edits(audio_segment, edits, work)


def process(clone_wav, txt_file, output_wav, global_shift=0, language='fr',
            whisper_model_name='small', per_word=True):
    from pydub import AudioSegment
    if os.path.realpath(clone_wav) == os.path.realpath(output_wav):
        raise ValueError("Output must differ from source")
    audio = AudioSegment.from_file(clone_wav)
    if not len(audio):
        raise ValueError("Input audio is empty")
    with tempfile.TemporaryDirectory(prefix='xtts_pitch_run_') as work:
        audio = rubberband_pitch(audio, global_shift, work)
        if per_word:
            text, corrections = combine_annotations(load_annotated_txt(txt_file))
            if corrections:
                import whisper
                model = whisper.load_model(whisper_model_name, device='cpu')
                audio, count = apply_pitch_per_word(audio, corrections, text, model, language, work)
                print(f"Applied {count}/{len(corrections)} occurrence-specific pitch marks")
        os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
        audio.export(output_wav, format='wav')
    print(f"[OK] {output_wav}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clone_wav')
    parser.add_argument('txt_file')
    parser.add_argument('output_wav')
    parser.add_argument('--global-shift', type=int, default=0)
    parser.add_argument('--lang', default='fr')
    parser.add_argument('--model', default='small', choices=['tiny', 'base', 'small', 'medium', 'large'])
    parser.add_argument('--no-per-word', action='store_true')
    args = parser.parse_args()
    if abs(args.global_shift) > 24:
        parser.error("global-shift must be in [-24, 24]")
    for name in ('clone_wav', 'txt_file'):
        if not os.path.isfile(getattr(args, name)):
            parser.error(f"File not found: {getattr(args, name)}")
    process(args.clone_wav, args.txt_file, args.output_wav, args.global_shift,
            args.lang, args.model, not args.no_per_word)


if __name__ == '__main__':
    main()

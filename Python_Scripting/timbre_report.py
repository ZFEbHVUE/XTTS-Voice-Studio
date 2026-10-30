#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Which part of the timbre score costs: pitch, brightness or HNR?

    python timbre_report.py REFERENCE.wav CLONE.wav [CLONE.wav ...]

The timbre score of the search (--objective timbre / both) is
1 / (1 + distance), the distance being the sum of three terms, each divided by
a gap that is audible on its own: pitch (2 semitones), brightness (300 Hz of
spectral centroid) and HNR (3 dB). A score of 0.30 is a distance of 2.3, and
says nothing about where it comes from. This prints the three terms for each
clone, measured on the voiced frames like the search does, with no model, no
GPU and no generation: it works on any WAV already on disk (a *_pipeline_clone,
a *_raw, a render from the Generator).

Pitch cannot be moved by the [] block at all, brightness by the EQ and the
filters, HNR hardly by anything it holds: the largest term says what a better
search could still change and what it could not.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    p = argparse.ArgumentParser(description='Timbre distance of clones, term by term')
    p.add_argument('reference', help='the voice the clones should resemble')
    p.add_argument('clones', nargs='+', help='one or more clone WAV files')
    args = p.parse_args(argv)

    for path in [args.reference] + args.clones:
        if not os.path.isfile(path):
            p.error(f'File not found: {path}')

    from xtts_optimize import timbre_features, timbre_lines, timbre_score

    ref = timbre_features(args.reference)
    if ref is None:
        print(f"[ERR] No voiced audio could be measured in {args.reference} "
              "(librosa and parselmouth are needed, and the file must hold speech).")
        return 1
    print(f"Reference {os.path.basename(args.reference)}: F0 {ref['f0']:.0f} Hz, "
          f"centroid {ref['centroid']:.0f} Hz, HNR {ref['hnr']:.1f} dB "
          f"({'Praat' if ref['hnr_method'] == 'praat' else 'HPSS fallback, no parselmouth'})")

    scores, failed = [], 0
    for path in args.clones:
        clone = timbre_features(path, hnr_method=ref['hnr_method'])
        print(f"\n{os.path.basename(path)}")
        if clone is None:
            failed += 1
            print("  no measurable voiced audio")
            scores.append((path, None))
            continue
        for line in timbre_lines(ref, clone):
            print(line)
        scores.append((path, timbre_score(ref, clone)))

    if len(args.clones) > 1:
        print("\nTimbre score (1 = same timbre):")
        for path, score in scores:
            print(f"  {score:.2f}  {os.path.basename(path)}" if score is not None
                  else f"   n/a  {os.path.basename(path)}")
    return 1 if failed == len(args.clones) else 0


if __name__ == '__main__':
    sys.exit(main())

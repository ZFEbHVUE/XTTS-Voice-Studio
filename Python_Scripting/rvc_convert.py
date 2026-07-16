#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rvc_convert.py — convert an XTTS output through a trained RVC model via Applio.

This is the timbre stage that lifts cloning past the zero-shot ceiling: XTTS
does the French and the prosody, the RVC model (trained on the target person,
see docs/RVC_GUIDE.md) re-voices it with their actual timbre.

Runs Applio's CLI (`core.py infer`) with Applio's OWN environment python
(.venv inside the Applio folder), so it works from the xtts env / the GUI
without any dependency mixing.

Usage:
  python rvc_convert.py input.wav -o output_rvc.wav \\
      --model ~/Applio/logs/lea/lea.pth --index ~/Applio/logs/lea/lea.index \\
      [--applio-dir ~/Applio] [--pitch 0] [--index-rate 0.75] \\
      [--protect 0.33] [--f0-method rmvpe]
"""

import os
import sys
import argparse
import subprocess


def find_applio_python(applio_dir):
    """Locate the python of Applio's own environment."""
    for rel in ('.venv/bin/python', 'env/bin/python', '.venv/Scripts/python.exe'):
        p = os.path.join(applio_dir, rel)
        if os.path.exists(p):
            return p
    return None


def main():
    p = argparse.ArgumentParser(description='RVC timbre conversion via Applio')
    p.add_argument('input', help='Input WAV (the XTTS clone / meditation)')
    p.add_argument('-o', '--output', default=None,
                   help='Output WAV (default: <input>_rvc.wav)')
    p.add_argument('--model', required=True, help='Trained RVC model .pth')
    p.add_argument('--index', required=True, help='Matching .index file')
    p.add_argument('--applio-dir', default=os.path.expanduser('~/Applio'))
    p.add_argument('--pitch', type=int, default=0,
                   help='Pitch shift in semitones (0 = same register)')
    p.add_argument('--index-rate', type=float, default=0.75,
                   help='Target-timbre retrieval strength (0.6-0.75 typical)')
    p.add_argument('--protect', type=float, default=0.33,
                   help='Protect consonants/breaths from conversion')
    p.add_argument('--f0-method', default='rmvpe')
    args = p.parse_args()

    applio = os.path.expanduser(args.applio_dir)
    core = os.path.join(applio, 'core.py')
    if not os.path.exists(core):
        sys.exit(f"[ERR] Applio not found at {applio} (no core.py). "
                 f"Install: git clone https://github.com/IAHispano/Applio && ./run-install.sh")
    py = find_applio_python(applio)
    if py is None:
        sys.exit(f"[ERR] Applio env python not found under {applio}/.venv — "
                 f"run Applio's ./run-install.sh first.")
    for f, tag in [(args.input, 'input'), (args.model, 'model'), (args.index, 'index')]:
        if not os.path.exists(os.path.expanduser(f)):
            sys.exit(f"[ERR] {tag} not found: {f}")

    out = args.output or (os.path.splitext(args.input)[0] + '_rvc.wav')
    cmd = [py, core, 'infer',
           '--pitch', str(args.pitch),
           '--index_rate', str(args.index_rate),
           '--protect', str(args.protect),
           '--f0_method', args.f0_method,
           '--input_path', os.path.abspath(os.path.expanduser(args.input)),
           '--output_path', os.path.abspath(os.path.expanduser(out)),
           '--pth_path', os.path.abspath(os.path.expanduser(args.model)),
           '--index_path', os.path.abspath(os.path.expanduser(args.index))]

    print(f"[*] RVC convert via Applio ({os.path.basename(args.model)}, "
          f"index_rate={args.index_rate}, protect={args.protect})...")
    proc = subprocess.Popen(cmd, cwd=applio, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            env=dict(os.environ, PYTHONUNBUFFERED='1'))
    for line in proc.stdout:
        print(line, end='', flush=True)
    proc.wait()
    if proc.returncode != 0 or not os.path.exists(out):
        sys.exit(f"[ERR] Conversion failed (code {proc.returncode})")
    print(f"[OK] {out}")
    print(f"    Measure: python speaker_identity.py <reference.wav> {args.input} {out}")


if __name__ == '__main__':
    main()

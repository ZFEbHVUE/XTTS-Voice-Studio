#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xtts_pipeline.py — one-shot pipeline: curation -> analysis -> optimisation ->
closed-loop tone fit, for one or several voices, each with its own language.

Runs the existing validated tools as subprocesses (no logic duplicated) and
parses their outputs, exactly like the GUI hand-offs do:

  1. curate_reference.py   raw ref        -> <ref>_curated.wav
  2. voice_analyser.py     curated        -> {} priors + [] neutral
  3. xtts_optimize.py      {} priors      -> {} winner (RSM: seed screen +
                                             least-squares temp surface)
  4. voice_comparator.py   {} winner + [] -> [] fitted (A-weighted LS, auto-text)
                                          -> <ref>_pipeline_clone.wav to LISTEN to

Final output: the numbered {} and [] blocks per voice, ready to paste into the
generator prompt. The scores are proxies — ALWAYS listen to the clone before
generating the full meditation.

Usage:
  python xtts_pipeline.py --voice lea.wav FR --voice john.wav EN \\
      [--seeds "0 42 100 180 200"] [--budget 60] [--keep-seconds 45] \\
      [--no-curate] [--no-auto-text] [--start-num 1] [--device cuda]
"""

import os
import re
import sys
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LANGS = {'FR','EN','ES','DE','IT','PT','PL','TR','RU','NL','CS','AR','ZH-CN','HU','KO','JA','HI'}

RE_XTTS  = re.compile(r'^\s*(\{\s*\d[^}]*\})\s*$', re.M)   # "  {1, 42, ...}"
RE_AUDIO = re.compile(r'^\s*(\[\s*\d[^\]]*\])\s*$', re.M)  # "  [1, FR, ...]"
RE_NEXT  = re.compile(r'Next \[\]\s*:\s*(\[[^\]]*\])')


def run_stream(cmd, tag):
    """Run a tool, stream its output live (so the GUI console shows progress),
    return (returncode, full_output)."""
    print(f"\n{'='*70}\n  [{tag}] {os.path.basename(cmd[1])}\n{'='*70}", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1,
                            env=dict(os.environ, PYTHONUNBUFFERED='1'))
    lines = []
    for line in proc.stdout:
        print(line, end='', flush=True)
        lines.append(line)
    proc.wait()
    return proc.returncode, ''.join(lines)


def renumber(block, n):
    """Set the leading voice number of a {}/[] block to n."""
    return re.sub(r'^([\{\[])\s*\d+', lambda m: m.group(1) + str(n), block.strip())


def set_lang(audio_block, lang):
    """Force the language field (2nd slot) of an [] block."""
    return re.sub(r'^(\[\s*\d+\s*,\s*)[A-Za-z-]+', lambda m: m.group(1) + lang, audio_block.strip())


def main():
    p = argparse.ArgumentParser(description='One-shot XTTS voice pipeline (curate -> analyse -> optimise -> fit)')
    p.add_argument('--voice', nargs=2, action='append', metavar=('WAV', 'LANG'),
                   required=True, help='Voice reference + language; repeatable')
    p.add_argument('--start-num', type=int, default=1, help='First voice number (default: 1)')
    p.add_argument('--seeds', default='0 42 100 180 200',
                   help='Seeds screened by the optimiser (default: "0 42 100 180 200")')
    p.add_argument('--budget', type=int, default=60, help='Optimiser generation budget (default: 60)')
    p.add_argument('--w-accent', type=float, default=0.6)
    p.add_argument('--w-identity', type=float, default=0.4)
    p.add_argument('--keep-seconds', type=float, default=45.0, help='Curated reference length (default: 45)')
    p.add_argument('--no-curate', action='store_true', help='Skip curation (use the raw reference)')
    p.add_argument('--recurate', action='store_true', help='Redo curation even if the curated file exists')
    p.add_argument('--no-auto-text', action='store_true',
                   help='Comparator: do not transcribe the reference (use its default text)')
    p.add_argument('--target-dbfs', type=float, default=None,
                   help='Comparator: fit the volume toward this absolute RMS level '
                        '(e.g. -20) instead of matching a quiet reference')
    p.add_argument('--fit-identity', action='store_true',
                   help='Comparator: also search post-processing settings that raise '
                        'ECAPA identity (not just spectral tone)')
    p.add_argument('--screen-audio', action='store_true',
                   help='Comparator: sensitivity screening of each audio parameter '
                        '(which knobs move identity for this voice)')
    p.add_argument('--optimise-audio', default='none', choices=['none','nelder','de'],
                   help='Comparator: derivative-free optimisation of the audio block '
                        'against ECAPA identity (validated on an unseen sentence)')
    p.add_argument('--probe-beams', action='store_true',
                   help='Optimiser: also probe beam-search/greedy decoding on the winner')
    p.add_argument('--whisper-model', default='small')
    p.add_argument('--device', default=None, help='cpu or cuda (default: auto)')
    args = p.parse_args()

    py = sys.executable
    S = lambda name: os.path.join(SCRIPT_DIR, name)
    results = []

    for idx, (ref, lang) in enumerate(args.voice):
        n = args.start_num + idx
        lang = lang.upper()
        if lang not in LANGS:
            sys.exit(f"[ERR] Unknown language '{lang}' for voice {n}")
        if not os.path.exists(ref):
            sys.exit(f"[ERR] Not found: {ref}")

        print(f"\n{'#'*70}\n#  VOICE {n}  [{lang}]  {os.path.basename(ref)}\n{'#'*70}")

        # ── 1. Curation ───────────────────────────────────────────────────────
        if args.no_curate:
            work = ref
            print("[*] Curation skipped (--no-curate)")
        else:
            base, _ = os.path.splitext(ref)
            work = base + '_curated.wav'
            if os.path.exists(work) and not args.recurate:
                print(f"[*] Curated file already exists -> reusing {os.path.basename(work)} "
                      f"(--recurate to redo)")
            else:
                cmd = [py, S('curate_reference.py'), ref, '-o', work,
                       '--keep-seconds', str(args.keep_seconds)]
                if args.device:
                    cmd += ['--device', args.device]
                rc, _out = run_stream(cmd, f'V{n} 1/4 CURATE')
                if rc != 0 or not os.path.exists(work):
                    sys.exit(f"[ERR] Curation failed for voice {n}")

        # ── 2. Analysis (priors) ──────────────────────────────────────────────
        cmd = [py, S('voice_analyser.py'), '--precise', '--f0-engine', 'pyin',
               '--start-num', str(n), work, lang]
        rc, out = run_stream(cmd, f'V{n} 2/4 ANALYSE')
        xb = RE_XTTS.findall(out)
        ab = RE_AUDIO.findall(out)
        if rc != 0 or not xb or not ab:
            sys.exit(f"[ERR] Analysis failed for voice {n} (no blocks found)")
        xtts_prior, audio_block = xb[-1], ab[-1]
        print(f"\n[pipeline] priors: {xtts_prior}")
        print(f"[pipeline] audio : {audio_block}")

        # ── 3. Optimisation (seed + temp by measured objective) ───────────────
        cmd = [py, S('xtts_optimize.py'), work, lang,
               '--xtts-block', xtts_prior, '--seeds', args.seeds,
               '--budget', str(args.budget), '--method', 'rsm',
               '--w-accent', str(args.w_accent), '--w-identity', str(args.w_identity),
               '--whisper-model', args.whisper_model]
        if args.probe_beams:
            cmd += ['--probe-beams']
        if args.device:
            cmd += ['--device', args.device]
        rc, out = run_stream(cmd, f'V{n} 3/4 OPTIMISE')
        xb = RE_XTTS.findall(out)
        if rc != 0 or not xb:
            sys.exit(f"[ERR] Optimisation failed for voice {n}")
        xtts_win = xb[-1]
        print(f"\n[pipeline] winner: {xtts_win}")

        # ── 4. Closed-loop tone fit ───────────────────────────────────────────
        base, _ = os.path.splitext(ref)
        clone_out = base + '_pipeline_clone.wav'
        cmd = [py, S('voice_comparator.py'), work, lang,
               '--xtts-block', xtts_win, '--audio-block', audio_block,
               '--output-optimised', clone_out,
               '--whisper-model', args.whisper_model]
        if not args.no_auto_text:
            cmd += ['--auto-text']
        if args.target_dbfs is not None:
            cmd += ['--target-dbfs', str(args.target_dbfs)]
        if args.fit_identity:
            cmd += ['--fit-identity']
        if args.screen_audio:
            cmd += ['--screen-audio']
        if args.optimise_audio != 'none':
            cmd += ['--optimise-audio', args.optimise_audio]
        if args.device:
            cmd += ['--device', args.device]
        rc, out = run_stream(cmd, f'V{n} 4/4 TONE FIT')
        m = RE_NEXT.search(out)
        if rc != 0 or not m:
            sys.exit(f"[ERR] Comparator failed for voice {n}")
        audio_fit = m.group(1)

        results.append(dict(n=n, lang=lang, ref=ref, clone=clone_out,
                            xtts=renumber(xtts_win, n),
                            audio=set_lang(renumber(audio_fit, n), lang)))

    # ── Final recap ───────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}\n  [*] PIPELINE COMPLETE — READY TO PASTE\n{'='*70}")
    for r in results:
        print(f"\n  # Voice {r['n']} [{r['lang']}]  {os.path.basename(r['ref'])}")
        print(f"  {r['xtts']}")
        print(f"  {r['audio']}")
        print(f"  # Listen before generating: {r['clone']}")
    print(f"\n  Scores are proxies (accent/identity) — they don't hear naturalness.")
    print(f"  LISTEN to each *_pipeline_clone.wav; if diction/naturalness is off,")
    print(f"  sweep temp ±1 step around the winner in the Validator and trust your ear.")
    print("[OK] Done.")


if __name__ == '__main__':
    main()

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


# Windows consoles default to cp1252 and raise UnicodeEncodeError on any
# non-ASCII character in output — a run that computed correctly then died on a
# print. Force UTF-8 where the interpreter allows it.
try:
    import sys as _sys
    if hasattr(_sys.stdout, 'reconfigure'):
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import os
import re
import sys
import argparse
import json
import shlex
import tempfile
from xtts_settings import parse_block, validate
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LANGS = {'FR','EN','ES','DE','IT','PT','PL','TR','RU','NL','CS','AR','ZH-CN','HU','KO','JA','HI'}

RE_XTTS  = re.compile(r'^\s*(\{\s*\d[^}]*\})\s*$', re.M)   # "  {1, 42, ...}"
RE_AUDIO = re.compile(r'^\s*(\[\s*\d[^\]]*\])\s*$', re.M)  # "  [1, FR, ...]"
RE_NEXT  = re.compile(r'Next \[\]\s*:\s*(\[[^\]]*\])')
RE_FIT   = re.compile(r'^\[\*\] Fit input: (\{.*\})\s*$', re.M)
EQ_WEIGHTINGS = ('a', 'voice', 'blend')


def eq_weightings(text):
    """'a', 'blend', or several such as 'a+blend' / 'a,blend': the first one
    gives the preset, each further one another tone fit of the same clone."""
    items = [w for w in re.split(r'[+,\s]+', str(text).strip()) if w]
    if not items or any(w not in EQ_WEIGHTINGS for w in items) or len(set(items)) < len(items):
        raise argparse.ArgumentTypeError(
            "one or more of a, voice, blend without repeats, e.g. a+blend")
    return items


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



def _free_name(path):
    """First unused name in the <stem>_2, _3 ... series.

    Underscore, not "(2)": a parenthesised name has to be quoted on every
    command line and trips the path splitting elsewhere in the project, and
    these files are handled by hand far more often than presets are.
    """
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    return f"{stem}_{n}{ext}"


def main():
    p = argparse.ArgumentParser(description='One-shot XTTS voice pipeline (curate -> analyse -> optimise -> fit)')
    p.add_argument('--voice', nargs=2, action='append', metavar=('WAV', 'LANG'),
                   required=True, help='Voice reference + language; repeatable')
    p.add_argument('--start-num', type=int, default=1, help='First voice number (default: 1)')
    p.add_argument('--seeds', default='0 42 100 180 200',
                   help='Seeds screened by the optimiser (default: "0 42 100 180 200")')
    p.add_argument('--budget', type=int, default=60, help='Optimiser generation budget (default: 60)')
    p.add_argument('--objective', choices=['identity', 'timbre', 'both'],
                   default='identity',
                   help="What the optimiser's second term measures: ECAPA "
                        "identity (default), timbre closeness (pitch, "
                        "brightness, HNR), or both.")
    p.add_argument('--gpt-cond-lens', default='20 30',
                   help="Optimiser: candidate gpt_cond_len values in seconds "
                        "(default: '20 30'). The GPT conditioning carries prosody "
                        "and ACCENT, and neither score hears a regional accent "
                        "(Whisper rates Quebec French as fluent French, ECAPA "
                        "ignores it): left free to go down to 6 s, the search "
                        "picked 6 s on a Paris-French voice and the clone came "
                        "out Canadian. Short values are only for references "
                        "too short to supply more.")
    p.add_argument('--cond-budget', type=int, default=24)
    p.add_argument('--gpt-chunks', default='4 6 8 10')
    p.add_argument('--max-ref-lens', default='')
    p.add_argument('--max-ref-len', type=int, default=30)
    p.add_argument('--probe-texts', type=int, default=2)
    p.add_argument('--adopt-alpha', type=float, default=None,
                   help='Optimiser: chance of adopting pure noise per search stage '
                        '(default: the optimiser\'s 0.05; 1 = any gain wins, the old rule)')
    p.add_argument('--text', help='Search sentence in the selected language')
    p.add_argument('--probe-text-file', help='Custom search sentences, one per line')
    p.add_argument('--holdout-text-file', help='Custom unseen sentences, one per line')
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
    p.add_argument('--no-save-preset', action='store_true',
                   help='Do not store the final blocks in Voice_Presets/')
    p.add_argument('--preset-name', action='append', default=None, metavar='NAME',
                   help="Preset name for the corresponding --voice, in order. "
                        "Repeat it once per voice; pass '-' to skip that voice. "
                        "Default: derived from each reference file.")
    p.add_argument('--keep-preset-history', action='store_true',
                   help="Never overwrite: a re-run of the same voice keeps the "
                        "previous preset, curated reference and clone, numbering "
                        "the new ones _2, _3 ... Stored as "
                        "'<name> (2)', '(3)'... so two runs can be compared. "
                        "Without it, the previous preset for that name is replaced.")
    p.add_argument('--precise', action='store_true', default=True,
                   help='Analyser: pyin precise mode (default). --fast disables it.')
    p.add_argument('--fast', dest='precise', action='store_false',
                   help='Analyser: quick F0 estimation instead of the precise pass')
    p.add_argument('--f0-engine', default='pyin', choices=['auto', 'crepe', 'pyin'],
                   help='Analyser: F0 estimator (default: pyin)')
    p.add_argument('--analysis', default='praat', choices=['praat', 'librosa'],
                   help='Analyser: Praat metrics (HNR/shimmer/jitter/formants) or '
                        'librosa only')
    p.add_argument('--emit-suggestions', action='store_true',
                   help='Analyser: put the measured NR/comp/de-ess into the [] '
                        'instead of zeros (unvalidated heuristics — test them)')
    p.add_argument('--eq-weighting', default=['a'], type=eq_weightings,
                   help="Comparator: where EQ errors count. 'a' = equal-loudness "
                        "(default; below ~180 Hz every frequency gets a floor weight), "
                        "'voice' = the reference's own spectrum, 'blend' = both. "
                        "Several, e.g. a+blend: the first gives the preset, each "
                        "further one another tone fit of the SAME clone, written next "
                        "to it and stored as '<preset> <weighting>' for a blind A/B.")
    p.add_argument('--screen-audio', action='store_true',
                   help='Comparator: sensitivity screening of each audio parameter '
                        '(which knobs move identity for this voice)')
    p.add_argument('--optimise-audio', default='none', choices=['none','nelder','de'],
                   help='Comparator: derivative-free optimisation of the audio block '
                        'against ECAPA identity (validated on an unseen sentence)')
    p.add_argument('--beam-width', type=int, default=3,
                   help='Beam width for --probe-beams (2 on a 4 GB card)')
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
            if args.keep_preset_history and args.recurate:
                # Only when re-curating: without --recurate an existing curated
                # file is REUSED on purpose, and numbering it would silently
                # curate again every run.
                work = _free_name(work)
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
        cmd = [py, S('voice_analyser.py'), '--f0-engine', args.f0_engine,
               '--start-num', str(n)]
        if args.precise:
            cmd += ['--precise']
        if args.analysis == 'librosa':
            cmd += ['--no-praat']
        if args.emit_suggestions:
            cmd += ['--emit-suggestions']
        cmd += [work, lang]
        rc, out = run_stream(cmd, f'V{n} 2/4 ANALYSE')
        # Same one-line voice description the analyser writes into a preset, so
        # a pipeline-saved preset identifies the speaker just as well.
        def _grab(pat, cast=str, default=None):
            m = re.search(pat, out)
            try:
                return cast(m.group(1)) if m else default
            except Exception:
                return default
        _vt = _grab(r'Voice type\s*:\s*(.+)')
        _f0 = _grab(r'F0 median\s*:\s*([\d.]+)', float)
        _vr = _grab(r'Voiced ratio\s*:\s*(\d+)%', int)
        _rm = _grab(r'RMS level\s*:\s*(-?[\d.]+)', float)
        _sn = _grab(r'Estimated SNR\s*:\s*([\d.]+)', float)
        _acoustics = None
        if _vt:
            _bits = [_vt.strip()]
            if _f0 is not None: _bits.append(f"F0 {_f0:.0f} Hz")
            if _vr is not None: _bits.append(f"{_vr}% voiced")
            if _rm is not None: _bits.append(f"{_rm:.1f} dBFS")
            if _sn is not None: _bits.append(f"SNR {_sn:.0f} dB")
            _acoustics = ' | '.join(_bits)
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
               '--whisper-model', args.whisper_model,
               '--cond-budget', str(args.cond_budget),
               '--gpt-chunks', args.gpt_chunks,
               '--max-ref-len', str(args.max_ref_len),
               '--probe-texts', str(args.probe_texts)]
        if args.max_ref_lens.strip():
            cmd += ['--max-ref-lens', args.max_ref_lens]
        if args.objective != 'identity':
            cmd += ['--objective', args.objective]
        if args.gpt_cond_lens and args.gpt_cond_lens.strip():
            cmd += ['--gpt-cond-lens', args.gpt_cond_lens.strip()]
        for option, value in (('--text', args.text), ('--probe-text-file', args.probe_text_file),
                              ('--holdout-text-file', args.holdout_text_file)):
            if value:
                cmd += [option, value]
        if args.probe_beams:
            cmd += ['--probe-beams', '--beam-width', str(args.beam_width)]
        if args.adopt_alpha is not None:
            cmd += ['--adopt-alpha', str(args.adopt_alpha)]
        if args.device:
            cmd += ['--device', args.device]
        with tempfile.TemporaryDirectory(prefix='xtts_result_') as result_dir:
            result_path = os.path.join(result_dir, 'winner.json')
            cmd += ['--result-json', result_path]
            rc, out = run_stream(cmd, f'V{n} 3/4 OPTIMISE')
            if rc != 0:
                sys.exit(f"[ERR] Optimisation failed for voice {n}")
            try:
                with open(result_path, encoding='utf-8') as f:
                    result = json.load(f)
                if result.get('schema_version') != 1:
                    raise ValueError('Unsupported optimiser result format')
                xtts_win = result['xtts_block']
                validate(parse_block(xtts_win))
                clone_refs = result['selected_refs']
                if not clone_refs or not all(isinstance(r, str) and os.path.isfile(r)
                                             for r in clone_refs):
                    raise ValueError('Missing winning cloning references')
                _sc = result['scores']
                # Stages the budget cut short: the recap repeats them, since
                # the optimiser's own warning scrolls away in a long run.
                _cut = dict(result.get('truncated_stages') or {})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                sys.exit(f"[ERR] Cannot read optimiser winner: {exc}")
        print(f"\n[pipeline] winner: {xtts_win}")

        # ── 4. Closed-loop tone fit ───────────────────────────────────────────
        base, _ = os.path.splitext(ref)
        clone_out = base + '_pipeline_clone.wav'
        if args.keep_preset_history:
            # The clone is what gets listened to, and comparing two runs by ear
            # is the only way to judge them -- overwriting it destroys the very
            # thing the comparison needs.
            clone_out = _free_name(clone_out)
        cmd = [py, S('voice_comparator.py'), work, lang,
               '--clone-refs', *clone_refs,
               '--xtts-block', xtts_win, '--audio-block', audio_block,
               '--output-optimised', clone_out,
               '--output', os.path.splitext(clone_out)[0]+'_base.wav',
               '--output-raw', os.path.splitext(clone_out)[0]+'_raw.wav',
               '--whisper-model', args.whisper_model]
        if args.text:
            cmd += ['--text', args.text]
        if args.holdout_text_file:
            with open(args.holdout_text_file, encoding='utf-8-sig') as handle:
                candidates = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith('#')]
            chosen = next((t for t in candidates if t.casefold() != (args.text or '').strip().casefold()), None)
            if chosen:
                cmd += ['--holdout-text', chosen]
        if not args.no_auto_text:
            cmd += ['--auto-text']
        # What every tone fit of this voice shares, the further ones included.
        holdout = (cmd[cmd.index('--holdout-text'):cmd.index('--holdout-text') + 2]
                   if '--holdout-text' in cmd else [])
        shared = []
        if args.target_dbfs is not None:
            shared += ['--target-dbfs', str(args.target_dbfs)]
        if args.fit_identity:
            shared += ['--fit-identity']
        if args.optimise_audio != 'none':
            shared += ['--optimise-audio', args.optimise_audio]
        if args.device:
            shared += ['--device', args.device]
        cmd += shared
        if args.eq_weighting[0] != 'a':
            cmd += ['--eq-weighting', args.eq_weighting[0]]
        if args.screen_audio:
            cmd += ['--screen-audio']
        rc, out = run_stream(cmd, f'V{n} 4/4 TONE FIT')
        m = RE_NEXT.search(out)
        if rc != 0 or not m:
            sys.exit(f"[ERR] Comparator failed for voice {n}")
        audio_fit = m.group(1)

        # Further EQ weightings: the SAME clone fitted again, against the same
        # stretch of the reference, so that only the weighting differs. No rule
        # says which one suits a voice; a blind A/B does.
        alternatives = []
        fit_input = RE_FIT.search(out)
        for weighting in args.eq_weighting[1:]:
            if not fit_input:
                print(f"[!] The comparator did not report what it fitted on: "
                      f"no '{weighting}' fit for voice {n}")
                break
            fit = json.loads(fit_input.group(1))
            alt_out = os.path.splitext(clone_out)[0] + f'_{weighting}.wav'
            if args.keep_preset_history:
                alt_out = _free_name(alt_out)
            with tempfile.TemporaryDirectory(prefix='xtts_fit_') as alt_dir:
                alt = [py, S('voice_comparator.py'), work, lang,
                       '--clone-refs', *clone_refs,
                       '--xtts-block', xtts_win, '--audio-block', audio_block,
                       '--raw-clone', os.path.splitext(clone_out)[0] + '_raw.wav',
                       '--text', fit['text'], '--eq-weighting', weighting,
                       '--no-identity-check', '--output-optimised', alt_out,
                       '--output', os.path.join(alt_dir, 'base.wav'),
                       '--output-raw', os.path.join(alt_dir, 'raw.wav'),
                       '--whisper-model', args.whisper_model, *shared, *holdout]
                if fit.get('interval'):
                    alt += ['--reference-interval', *[repr(float(x)) for x in fit['interval']]]
                rc_alt, out_alt = run_stream(alt, f'V{n} 4/4 TONE FIT ({weighting})')
            m_alt = RE_NEXT.search(out_alt)
            if rc_alt != 0 or not m_alt:
                print(f"[!] The '{weighting}' tone fit failed for voice {n}: "
                      f"only '{args.eq_weighting[0]}' is kept")
                continue
            alternatives.append(dict(weighting=weighting, clone=alt_out,
                                     audio=set_lang(renumber(m_alt.group(1), n), lang)))

        # Identity on UNSEEN text, if the comparator measured it: that is the
        # figure matching real usage, so it belongs in the preset.
        _mu = re.search(r'([\d.]+) on UNSEEN text', out)
        if _mu:
            _sc['identity_unseen'] = float(_mu.group(1))
        results.append(dict(n=n, lang=lang, ref=ref, clone=clone_out,
                            xtts=renumber(xtts_win, n),
                            audio=set_lang(renumber(audio_fit, n), lang),
                            scores=_sc, acoustics=_acoustics, references=clone_refs,
                            cut=_cut, weighting=args.eq_weighting[0],
                            alternatives=alternatives))

    # ── Final recap ───────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}\n  [*] PIPELINE COMPLETE — READY TO PASTE\n{'='*70}")
    for r in results:
        print(f"\n  # Voice {r['n']} [{r['lang']}]  {os.path.basename(r['ref'])}")
        print(f"  {r['xtts']}")
        print(f"  {r['audio']}")
        print('# Clone references: ' + json.dumps(r['references']))
        if r.get('cut'):
            print(f"  # [!] Optimiser stages cut short by the budget: "
                  f"{', '.join(r['cut'])} -- raise --budget / --cond-budget")
        print(f"  # Listen before generating: {r['clone']}")
        for a in r.get('alternatives') or []:
            # Commented out: pasting the recap must give one [] per voice.
            print(f"  # Same clone, tone fit with EQ weighting '{a['weighting']}' instead "
                  f"of '{r['weighting']}': {a['clone']}")
            print(f"  # {a['audio']}")
            print(f"  # Blind A/B: python listening_test.py {shlex.quote(r['clone'])} "
                  f"{shlex.quote(a['clone'])} --reference {shlex.quote(r['ref'])}")
    # ── Save each voice into the preset library ──────────────────────────────
    # This is the natural moment: the blocks are complete and the scores are
    # known. Without it the result lives only in a console you will close.
    if not args.no_save_preset:
        try:
            import voice_presets as VP
            print(f"\n  Saved to {os.path.basename(VP.PRESET_DIR)}/:")
            names = args.preset_name or []
            for i, r in enumerate(results):
                # One name per voice, in --voice order. '-' skips that voice,
                # so a multi-voice run stores exactly the ones you named.
                nm = names[i].strip() if i < len(names) else ''
                if nm == '-':
                    continue
                if not nm:
                    nm = VP.name_from_reference(r['ref'])
                saved = VP.save(nm, r['xtts'], r['audio'], reference=r['ref'],
                                source='pipeline', scores=r.get('scores'),
                                acoustics=r.get('acoustics'), references=r['references'],
                                overwrite=not args.keep_preset_history)
                print(f"    '{saved}'  {VP.describe(saved)}")
                for a in r.get('alternatives') or []:
                    # The identity on unseen text was measured with the first fit.
                    scores = {k: v for k, v in (r.get('scores') or {}).items()
                              if k != 'identity_unseen'}
                    alt_saved = VP.save(
                        f"{nm} {a['weighting']}", r['xtts'], a['audio'], reference=r['ref'],
                        source='pipeline', scores=scores, acoustics=r.get('acoustics'),
                        references=r['references'], overwrite=not args.keep_preset_history,
                        note=(f"same clone as '{saved}', tone fit with EQ weighting "
                              f"'{a['weighting']}' instead of '{r['weighting']}'"))
                    print(f"    '{alt_saved}'  {VP.describe(alt_saved)}")
            print(f"    (insert them in the [Gen] editor via the Voice preset bar)")
        except Exception as e:
            print(f"  [!] Preset save failed: {e}")

    print(f"\n  Scores are proxies (accent/identity) -- they don't hear naturalness.")
    print(f"  LISTEN to each *_pipeline_clone.wav; if diction/naturalness is off,")
    print(f"  sweep temp +/-1 step around the winner in the Validator and trust your ear.")
    print("[OK] Done.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Regression tests: gain/limiter order, noise-aware adoption, budget reporting.

    python -m unittest -v test_render_search_regressions

The render tests drive the generator's REAL functions through FFmpeg (skipped
when FFmpeg or pydub is missing). The optimiser tests run xtts_optimize.main()
end to end with simulated XTTS / Whisper / ECAPA, like the other suites.
"""
import ast
import contextlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import xtts_optimize as optimizer
import xtts_pipeline as pipeline
import xtts_settings as settings

try:
    from pydub import AudioSegment
    from pydub.effects import normalize, compress_dynamic_range
except Exception:          # pydub needs audioop (removed in Python 3.13)
    AudioSegment = None
HAVE_FFMPEG = shutil.which('ffmpeg') is not None
needs_audio = unittest.skipUnless(AudioSegment is not None and HAVE_FFMPEG,
                                  'pydub and ffmpeg are required')

GENERATOR = 'guided_meditation_generator_v23.py'
CHAIN = {'_trailing_silence_ms', 'apply_filters', 'apply_eq', 'apply_noise_reduction',
         'apply_deesser', 'apply_compression', 'apply_noise_gate', 'apply_reverb',
         'apply_pan', 'apply_limiter', 'process_audio', 'apply_speed_rubberband',
         'apply_gain_and_limit', '_pydub_gain', 'render_voice_chain', '_limiter_filter',
         'LIMITER_FILTER', 'LIMIT_DBFS', '_PCM_CODECS', '_LIMITER_LATENCY'}


def generator_namespace(names=CHAIN, **overrides):
    """The generator's own functions and constants, without importing TTS."""
    tree = ast.parse((ROOT / GENERATOR).read_text(encoding='utf-8'))
    nodes = [n for n in tree.body
             if (isinstance(n, ast.FunctionDef) and n.name in names)
             or (isinstance(n, ast.Assign)
                 and any(getattr(t, 'id', None) in names for t in n.targets))]
    printed = []
    ns = dict(AudioSegment=AudioSegment, subprocess=subprocess, tempfile=tempfile, os=os,
              math=math, np=np, print=lambda *a, **k: printed.append(' '.join(map(str, a))))
    if AudioSegment is not None:
        ns.update(normalize=normalize, compress_dynamic_range=compress_dynamic_range)
    ns.update(overrides)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), GENERATOR, 'exec'), ns)
    ns['printed'] = printed
    return ns


def voice_like(peak_dbfs=-1.0, seconds=2.0, sr=24000, width=2):
    """Harmonic, syllable-modulated test signal at an exact sample peak."""
    t = np.arange(int(sr * seconds)) / sr
    phase = 2 * np.pi * np.cumsum(180 * (1 + 0.05 * np.sin(2 * np.pi * 0.7 * t))) / sr
    y = sum(np.sin(k * phase) / k ** 1.2 for k in range(1, 30))
    y = y * np.clip(np.sin(2 * np.pi * 2.5 * t), 0, None) ** 0.6
    y = 10 ** (peak_dbfs / 20) * y / np.max(np.abs(y))
    full = 2 ** (8 * width - 1) - 1
    raw = (y * full).astype('<i4' if width == 4 else '<i2')
    return AudioSegment(raw.tobytes(), frame_rate=sr, sample_width=width, channels=1)


def spiky(base_dbfs=-14.0, peak_dbfs=-1.0, seconds=2.0, sr=24000):
    """A steady tone with rare 2 ms bursts reaching the peak, like the high
    crest factor of real speech: a limiter only touches the bursts."""
    t = np.arange(int(sr * seconds)) / sr
    y = 10 ** (base_dbfs / 20) * np.sin(2 * np.pi * 220 * t)
    for start in range(sr // 8, len(t) - sr // 100, sr // 4):
        n = sr // 500
        y[start:start + n] = 10 ** (peak_dbfs / 20) * np.sin(2 * np.pi * 1000 * np.arange(n) / sr + np.pi / 4)
    return AudioSegment((y * 32767).astype('<i2').tobytes(), frame_rate=sr, sample_width=2, channels=1)


def burst_crest(seg, sr=24000):
    """Lowest peak/RMS ratio over the bursts of spiky(): sqrt(2) for an intact
    sine, lower once the tops have been flattened (clipped, even if limited
    afterwards)."""
    a = np.array(seg.get_array_of_samples(), dtype=float)
    n = sr // 500
    ratios = []
    for start in range(sr // 8, len(a) - sr // 100, sr // 4):
        b = a[start + n // 4:start + 3 * n // 4]
        ratios.append(np.max(np.abs(b)) / np.sqrt(np.mean(b ** 2)))
    return min(ratios)


def clipped(seg):
    a = np.abs(np.array(seg.get_array_of_samples(), dtype=np.int64))
    return int(np.sum(a >= seg.max_possible_amplitude - 1))


def rms_db(seg):
    a = np.array(seg.get_array_of_samples(), dtype=float)
    return 20 * np.log10(np.sqrt(np.mean(a ** 2)) / seg.max_possible_amplitude)


class RenderChainTests(unittest.TestCase):
    @needs_audio
    def test_positive_volume_is_limited_instead_of_clipped(self):
        ns = generator_namespace()
        seg = spiky()
        self.assertGreater(clipped(seg + 6), 50)            # the old order: pydub saturates
        out = ns['apply_gain_and_limit'](seg, 6.0, limiter=True)
        self.assertEqual(clipped(out), 0)
        self.assertLessEqual(out.max_dBFS, ns['LIMIT_DBFS'] + 0.01)
        self.assertGreater(rms_db(out) - rms_db(seg), 5.0)  # below the bursts: the full gain
        self.assertGreater(burst_crest(out), 1.3)             # bursts limited, not flat-topped
        self.assertLess(burst_crest(seg + 6), 1.3)            # (what clipping does to them)
        self.assertTrue(any('peaks reduced by' in line for line in ns['printed']))

    @needs_audio
    def test_limiter_does_not_shift_or_truncate_when_ffmpeg_can_compensate(self):
        ns = generator_namespace()
        if 'latency' not in ns['_limiter_filter']():
            self.skipTest('this FFmpeg has no alimiter latency option')
        sr = 24000
        y = np.zeros(sr, dtype='<i2')
        y[2400] = y[-48] = 16000                       # 100 ms in, 2 ms before the end
        seg = AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=1)
        out = np.array(ns['apply_gain_and_limit'](seg, 0.0, limiter=True).get_array_of_samples())
        self.assertEqual(len(out), len(y))
        self.assertEqual(np.flatnonzero(np.abs(out) > 8000).tolist(), [2400, sr - 48])

    @needs_audio
    def test_bit_depth_is_preserved(self):
        # pydub holds 24-bit audio as 32-bit; FFmpeg used to write it back as 16-bit.
        ns = generator_namespace()
        out = ns['apply_gain_and_limit'](voice_like(-6.0, width=4), 3.0, limiter=True)
        self.assertEqual(out.sample_width, 4)
        self.assertEqual(ns['apply_eq'](voice_like(-6.0, width=4), 2, 0, 0).sample_width, 4)

    @needs_audio
    def test_gain_and_limiter_share_one_float_filter_graph(self):
        commands = []
        def run(cmd, **kw):
            commands.append(cmd)
            if '-h' in cmd:
                return types.SimpleNamespace(stdout='latency', stderr='')
            shutil.copyfile(cmd[cmd.index('-i') + 1], cmd[-1])
            return types.SimpleNamespace(returncode=0)
        ns = generator_namespace(subprocess=types.SimpleNamespace(
            run=run, SubprocessError=subprocess.SubprocessError,
            CalledProcessError=subprocess.CalledProcessError))
        seg = voice_like(-12.0)
        ns['apply_gain_and_limit'](seg, 6.0, limiter=True)
        ns['apply_gain_and_limit'](seg, 0.0, limiter=True)
        graphs = [c[c.index('-af') + 1] for c in commands if '-af' in c]
        self.assertTrue(graphs[0].startswith('volume=6dB,alimiter='))
        self.assertTrue(graphs[1].startswith('alimiter='))      # no gain: limiter alone
        self.assertTrue(all(g.endswith(':latency=1') for g in graphs))

    @needs_audio
    def test_limiter_off_applies_gain_and_warns_when_it_clips(self):
        ns = generator_namespace()
        seg = voice_like(-1.0)
        out = ns['apply_gain_and_limit'](seg, 6.0, limiter=False)
        self.assertGreater(clipped(out), 0)
        self.assertTrue(any('CLIPPED' in line for line in ns['printed']))

    @needs_audio
    def test_missing_ffmpeg_keeps_the_volume(self):
        def run(cmd, **kw):
            raise FileNotFoundError('ffmpeg')
        ns = generator_namespace(subprocess=types.SimpleNamespace(
            run=run, SubprocessError=subprocess.SubprocessError,
            CalledProcessError=subprocess.CalledProcessError))
        seg = voice_like(-12.0)
        out = ns['apply_gain_and_limit'](seg, 6.0, limiter=True)
        self.assertAlmostEqual(out.max_dBFS, seg.max_dBFS + 6.0, places=1)
        self.assertTrue(any('limiter skipped' in line for line in ns['printed']))

    @needs_audio
    def test_eq_boost_is_limited_instead_of_clipped(self):
        ns = generator_namespace()
        seg = voice_like(-1.0)
        out = ns['apply_eq'](seg, 4, 4, 4)
        self.assertEqual(clipped(out), 0)
        self.assertTrue(any('EQ boost reached full scale' in line for line in ns['printed']))
        quiet = voice_like(-12.0)                       # no overs: limiter transparent
        self.assertLess(ns['apply_eq'](quiet, 4, 4, 4).max_dBFS, -3.0)

    def test_eq_cut_only_adds_no_limiter(self):
        commands = []
        def run(cmd, **kw):
            commands.append(cmd)
            raise subprocess.CalledProcessError(1, cmd)
        ns = generator_namespace(subprocess=types.SimpleNamespace(
            run=run, SubprocessError=subprocess.SubprocessError,
            CalledProcessError=subprocess.CalledProcessError))
        class Seg:
            sample_width = 2
            def export(self, path, **kw): Path(path).write_bytes(b'')
        ns['apply_eq'](Seg(), -3, 0, -2)
        graph = commands[-1][commands[-1].index('-af') + 1]
        self.assertNotIn('alimiter', graph)

    def test_render_chain_runs_the_limiter_last(self):
        events = []
        ns = generator_namespace({'render_voice_chain'},
            process_audio=lambda a, cfg, xp, limiter=True: events.append(('effects', limiter)) or a,
            apply_speed_rubberband=lambda a, speed: events.append(('speed', speed)) or a,
            apply_gain_and_limit=lambda a, gain, limiter=True: events.append(('gain', gain, limiter)) or a)
        ns['render_voice_chain'](object(), dict(speed=.9, volume=4.5, limiter=1), {})
        ns['render_voice_chain'](object(), dict(speed=1, volume=-2, limiter=0), {})
        self.assertEqual(events, [('effects', False), ('speed', .9), ('gain', 4.5, True),
                                  ('effects', False), ('speed', 1), ('gain', -2, False)])

    def test_comparator_renders_through_the_generator_chain(self):
        import voice_comparator as comparator
        calls = []
        class Seg:
            def export(self, path, **kw): calls.append(('export', path))
        fake = types.SimpleNamespace(from_wav=lambda path: Seg())
        chain = lambda seg, cfg, xp: calls.append(('chain', cfg, xp)) or seg
        with patch.dict(sys.modules, {'pydub': types.SimpleNamespace(AudioSegment=fake)}), \
             patch.object(comparator, '_generator_chain', return_value=chain):
            comparator.apply_post('in.wav', 'out.wav',
                                  dict(vol=6.0, speed=0.9, limiter=1, eq_mid=2.0),
                                  dict(trim_end=200), 'FR', 'gen.py')
        (tag, cfg, xp), export = calls
        self.assertEqual((cfg['volume'], cfg['speed'], cfg['limiter'], cfg['eq_mid']), (6.0, .9, 1, 2.0))
        self.assertEqual(xp['trim_end'], 200)
        self.assertEqual(export, ('export', 'out.wav'))


SEED_OFFSET = {0: 0.0, 42: 0.05, 100: -0.05, 180: 0.03, 200: -0.03}


class OptimiserSearchTests(unittest.TestCase):
    def run_optimizer(self, options, accent):
        """accent(rec) -> accent score of one simulated render."""
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / 'a.wav'
            ref.touch()
            meta = {}
            def latents(model, paths, **kw): return dict(kw)
            def generate(model, text, lang, lat, path, **kw): meta[path] = dict(kw, text=text, lat=lat)
            class Pron:
                def __init__(self, **kw): pass
                def score(self, path, **kw): return {'score': accent(meta[path])}
            class Encoder:
                def __init__(self, **kw): pass
                def embed(self, path): return np.array([1., 0.])
                def cosine(self, a, b): return .7
            mocks = {'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
                     'xtts_clone': types.SimpleNamespace(load_xtts=lambda d: (None, None),
                                                        compute_latents=latents, generate=generate),
                     'pron_score': types.SimpleNamespace(PronScorer=Pron),
                     'speaker_identity': types.SimpleNamespace(SpeakerEncoder=Encoder)}
            out = Path(td) / 'winner.json'
            argv = ['xtts_optimize.py', str(ref), 'FR', '--xtts-block',
                    settings.format_block(dict(seed=0, temp=.62, rep_pen=5, top_p=.85, top_k=50)),
                    '--text', 'TEST', '--seeds', '0 42 100 180 200', '--device', 'cpu',
                    '--no-search-conditioning', '--no-holdout', '--result-json', str(out)] + options
            log = io.StringIO()
            with patch.dict(sys.modules, mocks), patch.object(sys, 'argv', argv), \
                 contextlib.redirect_stdout(log):
                optimizer.main()
            return json.loads(out.read_text()), log.getvalue()

    @staticmethod
    def accent(rec, rep_effect=.01, top_p_effect=.2):
        """Seeds shift the score (pure draw noise); rep_pen 7 helps a tiny bit,
        top_p .9 a lot; temperature does nothing."""
        return (.6 + SEED_OFFSET[rec['seed']]
                + rep_effect * (rec['repetition_penalty'] == 7)
                + top_p_effect * (abs(rec['top_p'] - .9) < 1e-9))

    def test_gain_within_measured_noise_is_not_adopted(self):
        result, log = self.run_optimizer(['--budget', '100'], self.accent)
        self.assertEqual(result['params']['rep_pen'], 5)          # +0.006: noise-sized
        self.assertAlmostEqual(result['params']['top_p'], .9)     # +0.12: real
        self.assertAlmostEqual(result['params']['temp'], .62)     # no effect: prior kept
        self.assertAlmostEqual(result['noise']['sampling_sd'],
                               .6 * np.std(list(SEED_OFFSET.values()), ddof=1), places=6)
        self.assertIn('is within the noise', log)

    def test_old_rule_is_available(self):
        result, _ = self.run_optimizer(['--budget', '100', '--adopt-alpha', '1'], self.accent)
        self.assertEqual(result['params']['rep_pen'], 7)
        self.assertAlmostEqual(result['params']['top_p'], .9)

    def test_budget_cut_is_reported_live_in_summary_and_json(self):
        result, log = self.run_optimizer(['--budget', '40'], self.accent)
        self.assertIn('[!] The budget cannot run the whole plan', log)
        self.assertIn("'S3 rep_pen' is cut short", log)
        self.assertIn('THE BUDGET CUT THESE STAGES SHORT', log)
        self.assertEqual(set(result['truncated_stages']), {'S3 rep_pen', 'S3 top_p', 'S3 top_k'})
        self.assertAlmostEqual(result['params']['top_p'], .85)    # never searched: prior kept

    def test_full_plan_reports_no_cut(self):
        result, log = self.run_optimizer(['--budget', '60'], self.accent)
        self.assertEqual(result['truncated_stages'], {})
        self.assertNotIn('cut short', log)

    def test_invalid_adopt_alpha_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.run_optimizer(['--adopt-alpha', '0'], self.accent)


class PlanAndBarTests(unittest.TestCase):
    START = dict(temp=.62, rep_pen=5., top_p=.85, top_k=50)

    def test_default_plan_exactly_fills_the_default_budget(self):
        self.assertEqual(optimizer.sampling_plan(5, 3, 2, self.START), 60)
        self.assertEqual(optimizer.sampling_plan(5, 3, 2, dict(self.START, temp=.65)), 54)
        self.assertEqual(optimizer.sampling_plan(5, 3, 2, dict(self.START, top_p=.88)), 62)
        self.assertEqual(optimizer.sampling_plan(5, 3, 2, self.START, probe_beams=True), 72)

    def test_bar_rises_with_candidates_and_has_a_floor(self):
        bar = optimizer.adoption_bar
        self.assertEqual(bar(.02, 4, 1), 0.0)
        self.assertLess(bar(.02, 1, .05), bar(.02, 6, .05))
        self.assertAlmostEqual(bar(.02, 1, .05), 1.6449 * np.sqrt(2) * .02, places=4)
        self.assertEqual(bar(0., 3, .05), bar(optimizer.SD_FLOOR, 3, .05))


class PipelineRecapTests(unittest.TestCase):
    def test_cut_stages_are_repeated_in_the_recap(self):
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / 'a.wav'
            ref.touch()
            xb = settings.format_block(dict(seed=0))
            ab = '[1, FR, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]'
            def run(cmd, tag):
                if 'voice_analyser.py' in cmd[1]:
                    return 0, xb + '\n' + ab
                if 'xtts_optimize.py' in cmd[1]:
                    Path(cmd[cmd.index('--result-json') + 1]).write_text(json.dumps(dict(
                        schema_version=1, xtts_block=xb, selected_refs=[str(ref)],
                        scores=dict(held_out=.8), truncated_stages={'S3 top_k': 2})))
                    return 0, ''
                return 0, 'Next []    : ' + ab
            output = io.StringIO()
            with patch.object(pipeline, 'run_stream', run), \
                 patch.object(sys, 'argv', ['xtts_pipeline.py', '--voice', str(ref), 'FR',
                                            '--no-curate', '--no-save-preset']), \
                 contextlib.redirect_stdout(output):
                pipeline.main()
        self.assertIn('Optimiser stages cut short by the budget: S3 top_k', output.getvalue())

    def run_pipeline(self, td, extra, comparator_reply):
        """The pipeline with every tool simulated; returns (commands, saves, recap)."""
        ref = Path(td) / 'a.wav'
        ref.touch()
        xb = settings.format_block(dict(seed=0))
        ab = '[1, FR, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]'
        commands, saves = [], []

        def run(cmd, tag):
            commands.append(cmd)
            if 'voice_analyser.py' in cmd[1]:
                return 0, xb + '\n' + ab
            if 'xtts_optimize.py' in cmd[1]:
                Path(cmd[cmd.index('--result-json') + 1]).write_text(json.dumps(dict(
                    schema_version=1, xtts_block=xb, selected_refs=[str(ref)],
                    scores=dict(held_out=.8))))
                return 0, ''
            return comparator_reply(cmd)
        vp = types.SimpleNamespace(PRESET_DIR=td, name_from_reference=lambda p: 'Voice',
                                   save=lambda name, *a, **kw: saves.append((name, kw)) or name,
                                   describe=lambda n: '')
        output = io.StringIO()
        with patch.object(pipeline, 'run_stream', run), patch.dict(sys.modules, {'voice_presets': vp}), \
             patch.object(sys, 'argv', ['xtts_pipeline.py', '--voice', str(ref), 'FR',
                                        '--no-curate', *extra]), \
             contextlib.redirect_stdout(output):
            pipeline.main()
        return [c for c in commands if 'voice_comparator.py' in c[1]], saves, output.getvalue()

    def test_a_second_weighting_refits_the_same_clone(self):
        fitted = {'a': '[1, FR, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]',
                  'blend': '[1, FR, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]'}

        def reply(cmd):
            weighting = cmd[cmd.index('--eq-weighting') + 1] if '--eq-weighting' in cmd else 'a'
            text = 'Next []    : ' + fitted[weighting]
            if '--raw-clone' not in cmd:
                text += ('\n[*] Fit input: {"text": "Bonjour Elo.", "interval": [0.5, 1.5]}'
                         '\n  Identity 0.700 on UNSEEN text')
            return 0, text
        with tempfile.TemporaryDirectory() as td:
            fits, saves, recap = self.run_pipeline(td, ['--eq-weighting', 'a+blend',
                                                        '--target-dbfs', '-20'], reply)
        first, second = fits
        self.assertNotIn('--eq-weighting', first)                     # 'a' is the default
        self.assertEqual(second[second.index('--raw-clone') + 1],
                         first[first.index('--output-raw') + 1])       # the very same clone
        for option, value in (('--text', 'Bonjour Elo.'), ('--eq-weighting', 'blend'),
                              ('--target-dbfs', '-20.0')):
            self.assertEqual(second[second.index(option) + 1], value)
        self.assertEqual(second[second.index('--reference-interval') + 1:
                                second.index('--reference-interval') + 3], ['0.5', '1.5'])
        self.assertIn('--no-identity-check', second)
        self.assertNotIn('--auto-text', second)
        self.assertTrue(second[second.index('--output-optimised') + 1]
                        .endswith('a_pipeline_clone_blend.wav'))
        # The recap stays one [] per voice: the other fit is a comment.
        self.assertIn("Same clone, tone fit with EQ weighting 'blend' instead of 'a'", recap)
        self.assertIn('  # ' + fitted['blend'], recap)
        self.assertIn('Blind A/B: python listening_test.py', recap)
        self.assertEqual([name for name, _kw in saves], ['Voice', 'Voice blend'])
        self.assertEqual(saves[0][1]['scores']['identity_unseen'], .7)
        self.assertNotIn('identity_unseen', saves[1][1]['scores'])
        self.assertIn("same clone as 'Voice'", saves[1][1]['note'])

    def test_a_failed_second_fit_keeps_the_first(self):
        def reply(cmd):
            if '--raw-clone' in cmd:
                return 1, 'Traceback'
            return 0, ('Next []    : [1, FR, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]\n'
                       '[*] Fit input: {"text": "Bonjour Elo.", "interval": null}')
        with tempfile.TemporaryDirectory() as td:
            fits, saves, recap = self.run_pipeline(td, ['--eq-weighting', 'a,blend'], reply)
        self.assertNotIn('--reference-interval', fits[1])              # whole reference, as the first
        self.assertIn("The 'blend' tone fit failed for voice 1: only 'a' is kept", recap)
        self.assertEqual([name for name, _kw in saves], ['Voice'])

    def test_eq_weighting_lists(self):
        self.assertEqual(pipeline.eq_weightings('a+blend'), ['a', 'blend'])
        self.assertEqual(pipeline.eq_weightings('blend, a'), ['blend', 'a'])
        self.assertEqual(pipeline.eq_weightings('voice'), ['voice'])
        for bad in ('', 'a+a', 'x', 'a+loud'):
            with self.subTest(bad=bad), self.assertRaises(Exception):
                pipeline.eq_weightings(bad)


if __name__ == '__main__':
    unittest.main()

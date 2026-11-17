"""3.10.6: one voice from several short clips of the same speaker.

Asked from use: XTTS-v2 takes a list of reference files, and three or four
clean clips of 6 to 10 s are said to clone better than one long file. The
[Auto] tab took one file per voice (two paths on a row gave "Pick a reference
first.": the row was read as one path), and xtts_pipeline.py's --voice took
exactly one file and a language.

Now a row of the [Auto] tab, and one --voice, take several files of one
speaker, then the language:
  - curation scores the windows of all the clips against one centroid and
    keeps Keep s in all, one <clip>_curated.wav per clip
    (curate_reference.py --outputs);
  - XTTS gets the clips as separate references, and the optimiser's C4 stage
    tries subsets of them, as it already did for several references;
  - the analyser and the tone fit read the clips joined (<first>_clips.wav);
  - the [Auto] tab's autos count what XTTS reads of each clip, and the
    conditioning budget counts the C4 stage.

    python -m unittest test_3106_multi_clip_voice -v
"""
import ast
import contextlib
import io
import json
import math
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
# Loaded before any patch.dict(sys.modules): leaving one drops what was
# imported under it, and numpy.fft cannot be loaded twice in a process.
import numpy.fft  # noqa: F401
try:
    import scipy.signal  # noqa: F401
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import search_plan as SP
import xtts_optimize as optimizer
import xtts_pipeline as pipeline
import voice_presets as VP
import xtts_settings as settings
from test_budget_plan_regressions import Var, Widget, SyncThread, run_optimizer

try:
    import soundfile as sf
except ImportError:
    sf = None

SR = 16000


def tone(path, seconds, freq, sr=SR, channels=1, level=0.3):
    """A clip whose content says where it came from: its frequency."""
    t = np.arange(int(round(seconds * sr))) / sr
    y = (level * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    if channels > 1:
        y = np.stack([y] * channels, axis=1)
    sf.write(str(path), y, sr)
    return str(path)


def peak_frequency(y, sr=SR):
    y = np.asarray(y, dtype=np.float64)
    spectrum = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return float(np.argmax(spectrum) * sr / len(y))


class ToneSpeakers:
    """ECAPA stand-in: windows at 1 kHz are another speaker, the rest one."""
    def __init__(self, **kw):
        pass

    def embed(self, y, sr):
        return np.array([0.0, 1.0, 0.0]) if abs(peak_frequency(y, sr) - 1000) < 50 \
            else np.array([1.0, 0.0, 0.0])


def curate_module():
    import curate_reference
    return curate_reference


# ── The command line: --voice WAV [WAV ...] LANG ─────────────────────────────

class VoiceSpecTests(unittest.TestCase):
    def test_files_then_the_language(self):
        self.assertEqual(pipeline._voice_spec(['a.wav', 'FR']), (['a.wav'], 'FR'))
        self.assertEqual(pipeline._voice_spec(['a.wav', 'b.wav', 'c.wav', 'fr']),
                         (['a.wav', 'b.wav', 'c.wav'], 'FR'))

    def test_what_is_refused(self):
        for values, said in ((['a.wav'], 'at least one file and a language'),
                             (['a.wav', 'b.wav'], 'the language comes last'),
                             (['FR', 'a.wav'], 'the language comes last'),
                             (['a.wav', './a.wav', 'FR'], 'the same file is given twice')):
            with self.subTest(values=values):
                with self.assertRaises(ValueError) as caught:
                    pipeline._voice_spec(values)
                self.assertIn(said, str(caught.exception))


# ── What the plan counts ─────────────────────────────────────────────────────

class ReferenceSetTests(unittest.TestCase):
    def test_the_count_is_the_optimisers_own(self):
        # C4 tries all the clips, each one alone and, up to four, every pair;
        # with two clips the one pair is all of them.
        for n in range(1, 10):
            with self.subTest(clips=n):
                sets = optimizer._reference_sets([f'/r{i}.wav' for i in range(n)])
                self.assertEqual(SP.reference_set_count(n), len(sets) if n > 1 else 0)
        self.assertEqual([SP.reference_set_count(n) for n in (2, 3, 4, 5)], [3, 7, 8, 6])


class VoiceAudioTests(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.dir = Path(td.name)
        self.lengths = {}
        for name, seconds in (('a', 8.0), ('b', 6.0), ('c', 10.0), ('solo', 40.0)):
            path = str(self.dir / f'{name}.wav')
            Path(path).touch()
            self.lengths[path] = seconds
        self.clips = [str(self.dir / f'{n}.wav') for n in 'abc']
        self.solo = str(self.dir / 'solo.wav')

    def audio(self, voices, curate, recurate=False, keep=45.0):
        return SP.pipeline_voice_audio(voices, curate, recurate, keep, lambda p: self.lengths[p])

    def test_one_file_is_what_it_was(self):
        for curate, recurate in ((False, False), (True, False), (True, True)):
            with self.subTest(curate=curate):
                old = SP.pipeline_reference_seconds([self.solo], curate, recurate, 30,
                                                    lambda p: self.lengths[p])[0]
                for voice in ([self.solo], self.solo):          # a bare path is one file
                    new = self.audio([voice], curate, recurate, 30)[0]
                    self.assertEqual((new['label'], new['seconds'], new['joined'], new['what']),
                                     (old[0], [old[1]], old[1], old[2]))

    def test_clips_as_they_are(self):
        v = self.audio([self.clips], curate=False)[0]
        self.assertEqual(v['files'], self.clips)
        self.assertEqual(v['seconds'], [8.0, 6.0, 10.0])
        # What the analyser reads: the clips end to end, 0.2 s apart.
        self.assertAlmostEqual(v['joined'], 24.0 + 2 * pipeline.JOIN_GAP_S)
        self.assertEqual(SP.JOIN_GAP_S, pipeline.JOIN_GAP_S)
        self.assertIn('3 clips as they are', v['what'])

    def test_clips_to_curate_count_keep_seconds_in_all(self):
        v = self.audio([self.clips], curate=True, keep=7.0)[0]
        self.assertEqual(v['seconds'], [7.0, 6.0, 7.0])       # none longer than Keep s
        self.assertAlmostEqual(v['joined'], 7.0 + 0.4)        # Keep s in all
        v = self.audio([self.clips], curate=True, keep=45.0)[0]
        self.assertAlmostEqual(v['joined'], 24.0 + 0.4)       # no more than the clips hold

    def test_existing_curated_clips_count_unless_recurated(self):
        for name, seconds in (('a', 5.0), ('b', 4.0), ('c', 6.0)):
            path = str(self.dir / f'{name}_curated.wav')
            Path(path).touch()
            self.lengths[path] = seconds
        v = self.audio([self.clips], curate=True, keep=45.0)[0]
        self.assertEqual(v['seconds'], [5.0, 4.0, 6.0])
        self.assertIn('existing curated', v['what'])
        v = self.audio([self.clips], curate=True, recurate=True, keep=45.0)[0]
        self.assertEqual(v['seconds'], [8.0, 6.0, 10.0])
        os.remove(self.dir / 'b_curated.wav')                 # one missing: curated again
        v = self.audio([self.clips], curate=True, keep=45.0)[0]
        self.assertEqual(v['seconds'], [8.0, 6.0, 10.0])


class BudgetTests(unittest.TestCase):
    GCL, CHUNKS = [20, 24], [4, 6, 8, 10]

    def test_one_file_per_voice_is_unchanged(self):
        args = ([20, 30], [4, 6, 8, 10], [20, 30], 30, [30.0, 45.0])
        self.assertEqual(SP.pipeline_conditioning_budget(*args),
                         SP.pipeline_conditioning_budget(*args, n_refs=[1, 1]))
        self.assertEqual(SP.pipeline_conditioning_budget([20, 30], [4, 6, 8, 10], [], 30, [45.0]), 34)

    def test_several_clips_add_the_reference_sets(self):
        joined = 24.0 + 0.4
        one = SP.pipeline_conditioning_budget(self.GCL, self.CHUNKS, [], 8, [joined])
        three = SP.pipeline_conditioning_budget(self.GCL, self.CHUNKS, [], 8, [joined], n_refs=[3])
        self.assertEqual(three, SP.conditioning_plan(self.GCL, self.CHUNKS, [], 24, 4, 8, 3, 1, 7))
        self.assertGreater(three, one)
        with self.assertRaises(ValueError):
            SP.pipeline_conditioning_budget(self.GCL, self.CHUNKS, [], 8, [joined], n_refs=[3, 1])

    def test_the_optimiser_runs_every_stage_within_it(self):
        # Three 8 s clips as they are: what the [Auto] tab's autos give, run
        # through the optimiser with its models simulated.
        clips = [8.0, 8.0, 8.0]
        joined = sum(clips) + 2 * SP.JOIN_GAP_S
        max_ref = SP.max_reference_for(clips)
        gcl = SP.gcl_candidates([sum(min(s, max_ref) for s in clips)], max_ref, n_refs=3)
        chunks = SP.chunk_candidates(gcl, max_ref, n_refs=3)
        self.assertEqual((max_ref, gcl, chunks), (8, [20, 24], [4, 6, 8, 10]))
        plan = SP.pipeline_conditioning_budget(gcl, chunks, [], max_ref, [joined], n_refs=[3])
        cfg = dict(gcl='20 24', chunks='4 6 8 10', max_ref=max_ref,
                   prior=SP.analyser_gcl_prior(joined), n_refs=3)
        for alpha in ('1', '0.05'):
            for trial in range(4):
                with self.subTest(alpha=alpha, trial=trial):
                    result, log = run_optimizer(dict(cfg, alpha=alpha), plan, trial)
                    self.assertEqual(int(re.search(r'Conditioning plan: up to (\d+)', log).group(1)),
                                     plan)
                    used = int(re.search(r'Evaluations: conditioning (\d+)/', log).group(1))
                    self.assertLessEqual(used, plan)
                    self.assertFalse([k for k in result['truncated_stages'] if k.startswith('C')])


# ── Curation of several clips, one file each ─────────────────────────────────

@unittest.skipIf(sf is None, 'soundfile is needed')
class CurationTests(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.dir = Path(td.name)
        # Three clips of the speaker, one of another speaker (1 kHz), and one
        # shorter than the 4 s window.
        self.clips = [tone(self.dir / 'a.wav', 8, 200), tone(self.dir / 'b.wav', 6, 300),
                      tone(self.dir / 'c.wav', 10, 400), tone(self.dir / 'other.wav', 6, 1000),
                      tone(self.dir / 'short.wav', 3, 500)]
        self.outputs = [str(self.dir / f'{Path(c).stem}_curated.wav') for c in self.clips]
        p = patch.dict(sys.modules, {'speaker_identity': types.SimpleNamespace(
            SpeakerEncoder=ToneSpeakers)})
        p.start()
        self.addCleanup(p.stop)

    def curate(self, keep, refs=None, outputs=None):
        with contextlib.redirect_stdout(io.StringIO()) as printed:
            written = curate_module().curate(refs or self.clips, None, keep_seconds=keep,
                                             outputs=outputs or self.outputs)
        return written, printed.getvalue()

    def test_each_clip_keeps_its_own_audio(self):
        written, printed = self.curate(45)
        self.assertEqual(written, self.outputs[:3])
        for path, seconds, freq in zip(written, (8, 6, 10), (200, 300, 400)):
            y, sr = sf.read(path)
            self.assertEqual((sr, len(y)), (SR, seconds * SR))      # all of it: Keep s is more
            self.assertAlmostEqual(peak_frequency(y), freq, delta=2)
        # The other speaker is scored out, the short clip has no window.
        self.assertFalse(os.path.exists(self.outputs[3]) or os.path.exists(self.outputs[4]))
        self.assertIn('other.wav: 0.0s kept -> dropped', printed)
        self.assertIn('short.wav: 0.0s kept -> dropped', printed)
        self.assertIn('Kept 9/11 windows', printed)

    def test_keep_seconds_is_the_total(self):
        # 19 s or more of the 24 s cannot be found without all three clips.
        for keep in (19, 20, 22):
            with self.subTest(keep=keep):
                written, _ = self.curate(keep, self.clips[:3], self.outputs[:3])
                self.assertEqual(len(written), 3)
                self.assertEqual(sum(sf.info(w).frames for w in written), keep * SR)

    def test_the_excess_comes_off_the_longest_clips(self):
        trim = curate_module()._trim_from_the_longest
        clips = lambda: [[[0, 8 * SR]], [[8 * SR, 14 * SR]], [[14 * SR, 24 * SR]]]
        lengths = lambda per_ref: [sum(e - s for s, e in segs) / SR for segs in per_ref]
        step = int(0.05 * SR)
        self.assertEqual(lengths(trim(clips(), 22 * SR, 160, step)), [8, 6, 8])
        self.assertEqual(lengths(trim(clips(), 20 * SR, 160, step)), [7, 6, 7])
        # Levelled from the top: the longest is never the one that goes.
        self.assertEqual(lengths(trim(clips(), 12 * SR, 160, step)), [4, 4, 4])
        # Joins inside a clip count their crossfade.
        two = [[[0, 3 * SR], [4 * SR, 8 * SR]], [[8 * SR, 10 * SR]]]
        trim(two, 6 * SR, 160, step)
        self.assertEqual(sum(curate_module()._net(segs, 160) for segs in two), 6 * SR)

    def test_nothing_kept_is_an_error(self):
        with self.assertRaises(RuntimeError):         # no clip as long as the window
            self.curate(45, [self.clips[4], tone(self.dir / 'z.wav', 2, 200)],
                        [self.outputs[4], str(self.dir / 'z_c.wav')])
        with self.assertRaises(RuntimeError):
            # One window of the speaker per clip, 4 s each, and 1 s asked: a
            # clip left under 2 s is dropped, and here that is all of them.
            self.curate(1, [tone(self.dir / 'x.wav', 4, 200), tone(self.dir / 'y.wav', 4, 300)],
                        [str(self.dir / 'x_c.wav'), str(self.dir / 'y_c.wav')])

    def test_outputs_are_checked(self):
        mod = curate_module()
        for outputs in ([self.outputs[0]], [self.outputs[0], self.outputs[0]],
                        [self.clips[0], self.outputs[1]]):
            with self.subTest(outputs=outputs), self.assertRaises(ValueError):
                mod.curate(self.clips[:2], None, outputs=outputs, verbose=False)

    def test_the_command_line_lists_what_it_wrote(self):
        argv = ['curate_reference.py', *self.clips[:4], '--outputs', *self.outputs[:4],
                '--keep-seconds', '45']
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()) as out:
            curate_module().main()
        line = re.search(r'^CURATED_JSON: (\[.*\])$', out.getvalue(), re.M)
        self.assertEqual(json.loads(line.group(1)), [os.path.abspath(o) for o in self.outputs[:3]])
        for bad in (['--outputs', *self.outputs[:3]],                       # one short
                    ['-o', str(self.dir / 'all.wav'), '--outputs', *self.outputs[:4]]):
            with self.subTest(bad=bad), \
                    patch.object(sys, 'argv', ['curate_reference.py', *self.clips[:4], *bad]), \
                    contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaises(SystemExit):
                curate_module().main()

    def test_one_output_file_is_unchanged(self):
        out = str(self.dir / 'all.wav')
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(curate_module().curate(self.clips[:3], out, keep_seconds=20), out)
        self.assertEqual(sf.info(out).frames, 20 * SR)


@unittest.skipIf(sf is None, 'soundfile is needed')
class JoinTests(unittest.TestCase):
    def test_clips_end_to_end_mono_at_the_first_rate(self):
        with tempfile.TemporaryDirectory() as td:
            a = tone(Path(td, 'a.wav'), 2, 200)
            b = tone(Path(td, 'b.wav'), 1, 300, sr=22050, channels=2)
            out = pipeline.join_clips([a, b], str(Path(td, 'sub', 'a_clips.wav')))
            y, sr = sf.read(out)
            self.assertEqual((sr, y.ndim), (SR, 1))
            gap = int(round(pipeline.JOIN_GAP_S * SR))
            self.assertEqual(len(y), 2 * SR + gap + SR)
            self.assertTrue(np.all(y[2 * SR:2 * SR + gap] == 0))
            self.assertAlmostEqual(peak_frequency(y[:2 * SR]), 200, delta=2)
            self.assertAlmostEqual(peak_frequency(y[2 * SR + gap:]), 300, delta=2)


# ── The pipeline with stand-ins for its four tools ───────────────────────────

@unittest.skipIf(sf is None, 'soundfile is needed')
class PipelineTests(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.dir = Path(td.name)
        p = patch.object(VP, 'PRESET_DIR', str(self.dir / 'presets'))
        p.start()
        self.addCleanup(p.stop)
        self.clips = [tone(self.dir / 'a.wav', 8, 200), tone(self.dir / 'b.wav', 6, 300),
                      tone(self.dir / 'c.wav', 10, 400)]
        self.xb = settings.format_block(dict(seed=100, temp=.64, max_ref_len=10))
        self.ab = '[1, FR, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]'
        self.fitted = '[1, FR, 1, -1, -4, 4, 1.5, 95, 0, 0, 0, 0, 0, 0, 0, 1]'
        self.drop = 0               # clips the curation stand-in drops, from the end

    def run_pipeline(self, *voice_args, extra=()):
        self.commands = []

        def run(cmd, tag):
            tool = os.path.basename(cmd[1])
            self.commands.append(list(cmd))
            if tool == 'curate_reference.py':
                if '--outputs' in cmd:
                    refs = cmd[2:cmd.index('--outputs')]
                    outs = cmd[cmd.index('--outputs') + 1:cmd.index('--keep-seconds')]
                    kept = outs[:len(outs) - self.drop]
                    for ref, out in zip(refs, kept):
                        y, sr = sf.read(ref)
                        sf.write(out, y[:4 * sr], sr)
                    return 0, 'CURATED_JSON: ' + json.dumps(kept) + '\n[OK] Done.\n'
                y, sr = sf.read(cmd[2])
                sf.write(cmd[cmd.index('-o') + 1], y[:5 * sr], sr)
                return 0, ''
            if tool == 'voice_analyser.py':
                return 0, 'Voice type     : female\n' + self.xb + '\n' + self.ab
            if tool == 'xtts_optimize.py':
                refs = cmd[2:cmd.index('--xtts-block') - 1]
                Path(cmd[cmd.index('--result-json') + 1]).write_text(json.dumps(dict(
                    schema_version=1, xtts_block=self.xb, selected_refs=refs[:2],
                    scores=dict(held_out=.8, identity=.7), generations=dict(conditioning=30))))
                return 0, ''
            return 0, ('[*] Fit input: ' + json.dumps({'text': 'Bonjour.', 'interval': [0.0, 5.0]})
                       + '\nNext []    : ' + self.fitted)
        argv = ['xtts_pipeline.py', *voice_args, '--eq-weighting', 'a', *extra]
        with patch.object(pipeline, 'run_stream', run), patch.object(sys, 'argv', argv), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            pipeline.main()
        return out.getvalue()

    def tool(self, name, n=0):
        return [c for c in self.commands if os.path.basename(c[1]) == name][n]

    def test_clips_are_curated_together_and_handed_to_xtts_separately(self):
        self.drop = 1
        out = self.run_pipeline('--voice', *self.clips, 'FR')
        curate = self.tool('curate_reference.py')
        curated = [str(self.dir / f'{n}_curated.wav') for n in 'abc']
        self.assertEqual(curate[2:], [*self.clips, '--outputs', *curated, '--keep-seconds', '45.0'])
        self.assertIn('1 clip(s) dropped by the curation', out)
        joined = str(self.dir / 'a_clips.wav')
        self.assertEqual(self.tool('voice_analyser.py')[-2:], [joined, 'FR'])
        self.assertEqual(sf.info(joined).frames, 8 * SR + int(0.2 * SR))   # 2 kept clips of 4 s
        optimise = self.tool('xtts_optimize.py')
        self.assertEqual(optimise[2:optimise.index('--xtts-block')], [*curated[:2], 'FR'])
        tone_fit = self.tool('voice_comparator.py')
        self.assertEqual(tone_fit[2:4], [joined, 'FR'])
        self.assertEqual(tone_fit[tone_fit.index('--clone-refs') + 1:][:2], curated[:2])
        # The preset: the joined file as its reference, the clips XTTS uses.
        e = VP.get('a')
        self.assertEqual(e['reference'], joined)
        self.assertEqual(e['references'], [os.path.abspath(c) for c in curated[:2]])
        cur = ' '.join(e['info']['Curation'])
        self.assertIn('3 of one speaker: a.wav, b.wav, c.wav', cur)
        self.assertIn('2 kept: a_curated.wav, b_curated.wav', cur)
        self.assertIn('VOICE 1  [FR]  a.wav  (+2 clip(s))', out)
        self.assertIn('# Voice 1 [FR]  a.wav  (+2 clip(s))', out)

    def test_existing_curated_clips_are_reused_unless_recurated(self):
        for clip in self.clips:
            tone(Path(clip[:-4] + '_curated.wav'), 5, 250)
        self.run_pipeline('--voice', *self.clips, 'FR')
        self.assertEqual([c for c in self.commands if 'curate_reference.py' in c[1]], [])
        optimise = self.tool('xtts_optimize.py')
        self.assertEqual(optimise[2:5], [c[:-4] + '_curated.wav' for c in self.clips])
        self.run_pipeline('--voice', *self.clips, 'FR', extra=('--recurate',))
        self.assertIn('--outputs', self.tool('curate_reference.py'))

    def test_no_curation_hands_the_clips_as_they_are(self):
        self.run_pipeline('--voice', *self.clips, 'FR', extra=('--no-curate',))
        optimise = self.tool('xtts_optimize.py')
        self.assertEqual(optimise[2:optimise.index('--xtts-block')], [*self.clips, 'FR'])
        joined = self.tool('voice_analyser.py')[-2]
        self.assertEqual(sf.info(joined).frames, 24 * SR + 2 * int(0.2 * SR))
        self.assertIn('curation  : skipped (--no-curate)', ' '.join(VP.get('a')['info']['Curation']))

    def test_one_file_voices_are_unchanged_beside_a_clip_voice(self):
        solo = tone(self.dir / 'solo.wav', 30, 220)
        self.run_pipeline('--voice', *self.clips[:2], 'FR', '--voice', solo, 'en')
        second = self.tool('xtts_optimize.py', 1)
        curated = str(self.dir / 'solo_curated.wav')
        self.assertEqual(second[2:second.index('--xtts-block')], [curated, 'EN'])
        self.assertEqual(self.tool('voice_analyser.py', 1)[-2:], [curated, 'EN'])
        self.assertEqual(self.tool('curate_reference.py', 1)[2:5], [solo, '-o', curated])
        self.assertEqual(VP.get('solo')['reference'], solo)

    def test_bad_voices_stop_before_anything_runs(self):
        for voice, said in ((['--voice', *self.clips[:2]], 'the language comes last'),
                            (['--voice', self.clips[0], str(self.dir / 'gone.wav'), 'FR'],
                             'Not found')):
            with self.subTest(said=said):
                err = io.StringIO()
                with self.assertRaises(SystemExit) as caught, contextlib.redirect_stderr(err):
                    self.run_pipeline(*voice)
                self.assertIn(said, err.getvalue() + str(caught.exception.code))
                self.assertEqual(self.commands, [])


# ── The [Auto] tab ───────────────────────────────────────────────────────────

@unittest.skipIf(sf is None, 'soundfile is needed')
class AutoTabTests(unittest.TestCase):
    """tab_auto with fake widgets, as test_budget_plan_regressions builds it."""

    def build(self):
        Widget.entries, Widget.variables, Widget.checks, Widget.buttons = {}, [], {}, {}
        Widget.created = []
        Widget.last_label = ''
        self.logs, self.commands, callbacks = [], [], {}
        self.chosen = []
        tk = types.SimpleNamespace(StringVar=Var, BooleanVar=Var, IntVar=Var, Frame=Widget,
                                   LabelFrame=Widget, Label=Widget, Entry=Widget, Button=Widget,
                                   Checkbutton=Widget)
        dialog = types.SimpleNamespace(askopenfilename=lambda **k: '',
                                       askopenfilenames=lambda **k: self.chosen.pop(0))
        ns = dict(tk=tk, ttk=types.SimpleNamespace(Combobox=Widget), os=os, sys=sys, SP=SP,
                  math=math, threading=types.SimpleNamespace(Thread=SyncThread),
                  filedialog=dialog, scrollable_tab=lambda *a: (Widget(), Widget()),
                  add_console=lambda *a: Widget(), log=lambda console, text: self.logs.append(text),
                  make_btn=lambda parent, text, callback, row: callbacks.update({text: callback}),
                  run_cmd=lambda cmd, *a, **kw: self.commands.append(cmd),
                  LANGS=['FR'], DEFAULT_LANG='FR', DIR_VOICES='/voices', SCRIPTS_DIR=str(ROOT))
        tree = ast.parse((ROOT / 'xtts_studio.py').read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('tab_auto', 'split_paths')]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'xtts_studio.py', 'exec'), ns)
        ns['tab_auto'](None)
        self.run_pipeline = lambda: callbacks['>  Run full pipeline'](None, None)
        return Widget.entries, Widget.checks, Widget.buttons

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.dir = Path(td.name)
        self.clips = [tone(self.dir / f'clip{i}.wav', 8, 200 + 100 * i) for i in range(3)]

    def readout(self):
        return [str(v.get()) for v in Widget.variables if 'clips of one speaker' in str(v.get())]

    def command_named(self, name):
        found = [w.kw['command'] for w in Widget.created
                 if getattr(w.kw.get('command'), '__name__', '') == name]
        self.assertEqual(len(found), 1)
        return found[0]

    def test_a_row_of_clips_runs_as_one_voice(self):
        entries, checks, buttons = self.build()
        entries['V1'].set(' ; '.join(self.clips))
        self.run_pipeline()
        cmd = self.commands[-1]
        i = cmd.index('--voice')
        self.assertEqual(cmd[i:i + 5], ['--voice', *self.clips, 'FR'])
        self.assertEqual(cmd.count('--voice'), 1)

    def test_clips_separated_by_spaces_as_typed(self):
        # The report: two paths typed on V1 gave "Pick a reference first.".
        entries, checks, buttons = self.build()
        entries['V1'].set(' '.join(self.clips[:2]))
        buttons['auto_maxref']()
        self.assertFalse([m for m in self.logs if 'Pick a reference first' in m])
        self.assertEqual(entries['Max reference s'].get(), '8')
        self.run_pipeline()
        cmd = self.commands[-1]
        self.assertEqual(cmd[cmd.index('--voice'):][:4], ['--voice', *self.clips[:2], 'FR'])

    def test_the_autos_count_what_xtts_reads_of_every_clip(self):
        entries, checks, buttons = self.build()
        checks['Curate reference'].set(False)                    # the clips as they are
        entries['V1'].set(' ; '.join(self.clips))
        self.assertTrue(self.readout())
        self.assertIn('3 clips of one speaker, 24.0s in all', self.readout()[0])
        buttons['auto_all']()
        # Each clip read in full; the GPT conditioning reads them end to end.
        self.assertEqual(entries['Max reference s'].get(), '8')
        self.assertEqual(entries['gcl s'].get(), '20 24')
        self.assertEqual(entries['Chunk lengths'].get(), '4 6 8 10')
        joined = 24 + 2 * SP.JOIN_GAP_S
        want = SP.pipeline_conditioning_budget([20, 24], [4, 6, 8, 10], [], 8, [joined], n_refs=[3])
        self.assertEqual(entries['Conditioning budget'].get(), str(want))
        self.assertGreater(want, SP.pipeline_conditioning_budget([20, 24], [4, 6, 8, 10], [], 8,
                                                                 [joined]))
        log = '\n'.join(self.logs)
        self.assertIn('every clip in full', log)
        self.assertIn('its clips together', log)

    def test_keep_auto_and_the_readout_warn_about_short_clips(self):
        entries, checks, buttons = self.build()
        short = tone(self.dir / 'short.wav', 3, 600)
        entries['Keep s'].set('45')
        entries['V1'].set(' ; '.join([*self.clips, short]))        # curation on by default
        text = self.readout()[0]
        self.assertIn('shorter than the 4 s curation window', text)
        self.assertIn('short.wav', text)
        self.assertIn('Keep s is 45 for all the clips', text)
        self.command_named('_auto_keep')()
        self.assertLess(float(entries['Keep s'].get()), 27)          # what the clips hold

    def test_browse_adds_clips_to_the_row(self):
        entries, checks, buttons = self.build()
        self.chosen = [tuple(self.clips[:2]), (self.clips[2],), ()]
        browse = self.command_named('browse')
        browse()
        self.assertEqual(entries['V1'].get(), ' ; '.join(self.clips[:2]))
        browse()
        self.assertEqual(entries['V1'].get(), ' ; '.join(self.clips))
        browse()                                                     # cancelled
        self.assertEqual(entries['V1'].get(), ' ; '.join(self.clips))

    def test_missing_or_repeated_clips_are_refused(self):
        entries, checks, buttons = self.build()
        gone = str(self.dir / 'gone.wav')
        entries['V1'].set(f'{self.clips[0]} ; {gone}')
        self.assertIn('not found: gone.wav', str([v.get() for v in Widget.variables]))
        buttons['auto_maxref']()
        self.assertIn(f'[ERR] Not found: {gone}', self.logs[-1])
        self.run_pipeline()
        self.assertEqual(self.commands, [])
        entries['V1'].set(f'{self.clips[0]} ; {self.clips[0]}')
        self.run_pipeline()
        self.assertEqual(self.commands, [])
        self.assertIn('The same file is twice', self.logs[-1])


def _display():
    if not os.environ.get('DISPLAY'):
        return False
    try:
        import tkinter as tk
        tk.Tk().destroy()
        return True
    except Exception:
        return False


@unittest.skipUnless(_display() and sf is not None, 'a display is needed (run under xvfb-run)')
class RealAutoTabTests(unittest.TestCase):
    """The tab built with real Tk widgets: the row, the autos and the run."""

    @classmethod
    def setUpClass(cls):
        import tkinter as tk
        from tkinter import ttk
        import xtts_studio
        cls.tk, cls.ttk, cls.s = tk, ttk, xtts_studio
        cls.root = tk.Tk()
        cls.root.geometry('900x700+20+20')

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.dir = Path(td.name)
        self.clips = [tone(self.dir / f'clip {i}.wav', 8, 200 + 100 * i) for i in range(2)]
        self.logs, self.commands = [], []
        for name, value in (('_root', self.root),
                            ('log', lambda console, text: self.logs.append(text)),
                            ('run_cmd', lambda cmd, *a, **kw: self.commands.append(cmd)),
                            ('threading', types.SimpleNamespace(Thread=SyncThread))):
            p = patch.object(self.s, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.nb = self.ttk.Notebook(self.root)
        self.nb.pack(fill='both', expand=True)
        self.addCleanup(self.nb.destroy)
        self.s.tab_auto(self.nb)
        self.root.update()

    def widgets(self, kind=None, text=None, name=None):
        out, todo = [], [self.nb]
        while todo:
            w = todo.pop()
            todo.extend(w.winfo_children())
            if ((kind is None or isinstance(w, kind))
                    and (text is None or ('text' in w.keys() and str(w.cget('text')) == text))
                    and (name is None or w.winfo_name() == name)):
                out.append(w)
        return out

    def field(self, label_text):
        label = self.widgets(self.tk.Label, label_text)[0]
        row = label.grid_info()['row']
        return [w for w in label.master.grid_slaves(row=row, column=1)
                if isinstance(w, self.tk.Entry)][0]

    def test_two_clips_with_spaces_in_their_names(self):
        row = self.widgets(self.tk.Label, 'V1')[0].master
        entry = [w for w in row.winfo_children() if isinstance(w, self.tk.Entry)][0]
        entry.insert(0, ' ; '.join(self.clips))
        self.root.update()
        self.assertTrue(any('2 clips of one speaker' in str(self.root.getvar(str(w.cget('textvariable'))))
                            for w in self.widgets(self.tk.Label) if str(w.cget('textvariable'))))
        self.widgets(name='auto_all')[0].invoke()
        self.assertEqual(self.field('Max reference s').get(), '8')
        self.assertFalse([m for m in self.logs if m.startswith('[ERR]')])
        self.widgets(self.tk.Button, '>  Run full pipeline')[0].invoke()
        cmd = self.commands[-1]
        lang = [w for w in row.winfo_children() if isinstance(w, self.ttk.Combobox)][0].get()
        self.assertEqual(cmd[cmd.index('--voice'):][:4], ['--voice', *self.clips, lang])
        self.assertEqual(cmd[cmd.index('--max-ref-len') + 1], '8')


if __name__ == '__main__':
    unittest.main(verbosity=2)

"""The [] search scores what the {} search scores, and stops when it cannot pay (3.9.0).

Until 3.8 the audio-block search judged every candidate with ECAPA identity
alone. On a real voice the effects moved ECAPA by 0.004 at most (below its own
noise), 240 evaluations went to a joint refinement of one sentence, and the
candidate was rejected on two others. The timbre score the {} search maximises
(pitch, brightness and HNR of the voiced frames) sat at about 0.30 whatever
was searched, and nobody could tell which of its three terms cost.

    python -m unittest test_bracket_criterion_regressions -v
"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audio_parameter_search as APS
import voice_presets as VP
import xtts_optimize as X
import timbre_report
from test_timbre_regressions import SR, floor, needs_librosa, sf, voiced
# The modules, not their classes: the loader would collect (and run) the tests of
# a class imported by name a second time.
import test_tone_fit_regressions as tone_tests
import test_preset_info_regressions as preset_tests
import test_clonage_regressions as gui_tests

START = dict(speed=1., vol=0., eq_low=0., eq_mid=0., eq_high=0., hp=0, lp=0, NR=0.,
             comp=0., reverb=0., noise_gate=0., pan=0., limiter=1, **{'de-ess': 0.})

REF = {'f0': 200., 'centroid': 1300., 'hnr': 16., 'hnr_method': 'praat', 'voiced_frames': 90}


def clone_features(pitch_st=0., centroid=0., hnr=0.):
    """A clone measured `pitch_st` semitones, `centroid` Hz and `hnr` dB from REF."""
    return dict(REF, f0=REF['f0'] * 2 ** (pitch_st / 12.), centroid=REF['centroid'] + centroid,
                hnr=REF['hnr'] + hnr)


# ── the timbre, term by term ──────────────────────────────────────────────────

class BreakdownTests(unittest.TestCase):
    def test_the_terms_add_up_to_the_distance_and_the_largest_is_named(self):
        clone = clone_features(pitch_st=-3.4, centroid=-150., hnr=-1.5)
        b = X.timbre_breakdown(REF, clone)
        self.assertAlmostEqual(b['terms']['pitch'], 1.7, places=6)
        self.assertAlmostEqual(b['terms']['brightness'], .5, places=6)
        self.assertAlmostEqual(b['terms']['hnr'], .5, places=6)
        self.assertAlmostEqual(b['distance'], X.timbre_distance(REF, clone), places=9)
        self.assertAlmostEqual(b['distance'], sum(b['terms'].values()), places=9)
        self.assertAlmostEqual(b['score'], X.timbre_score(REF, clone), places=9)
        self.assertEqual(b['largest'], 'pitch')
        # signed, clone minus reference
        self.assertAlmostEqual(b['gaps']['pitch'], -3.4, places=6)
        self.assertAlmostEqual(b['gaps']['brightness'], -150., places=6)
        self.assertAlmostEqual(b['gaps']['hnr'], -1.5, places=6)

    def test_each_term_can_be_the_largest(self):
        for kwargs, expected in ((dict(pitch_st=2.), 'pitch'), (dict(centroid=400.), 'brightness'),
                                 (dict(hnr=-6.), 'hnr')):
            with self.subTest(expected=expected):
                self.assertEqual(X.timbre_breakdown(REF, clone_features(**kwargs))['largest'], expected)

    def test_files_measured_another_way_are_not_compared(self):
        other = dict(clone_features(), hnr_method='hpss')
        self.assertIsNone(X.timbre_breakdown(REF, other))
        self.assertIsNone(X.timbre_breakdown(None, clone_features()))
        self.assertIsNone(X.timbre_breakdown(REF, None))
        self.assertIn('not measured the same way', '\n'.join(X.timbre_lines(REF, other)))

    def test_the_printed_table_names_the_three_terms_and_what_moves_the_largest(self):
        text = '\n'.join(X.timbre_lines(REF, clone_features(pitch_st=-3.4, centroid=-150., hnr=-1.5)))
        for part in ('pitch (F0)', 'brightness (centroid)', 'HNR', '-3.4 st', '-150.0 Hz', '-1.5 dB',
                     'distance 2.70', 'timbre score 0.27',
                     'largest cost: pitch (F0) (1.70 of 2.70)',
                     'what moves it: nothing in the [] block'):
            self.assertIn(part, text)
        bright = '\n'.join(X.timbre_lines(REF, clone_features(centroid=400.)))
        self.assertIn('largest cost: brightness (centroid)', bright)
        self.assertIn('the EQ and the low/high-pass filters', bright)
        noisy = '\n'.join(X.timbre_lines(REF, clone_features(hnr=-6.)))
        self.assertIn('hardly anything', noisy)

    def test_a_clone_that_matches_has_no_largest_cost(self):
        text = '\n'.join(X.timbre_lines(REF, dict(REF)))
        self.assertIn('no term costs anything', text)
        self.assertNotIn('largest cost', text)
        self.assertNotIn('what moves it', text)


@needs_librosa
class MeasureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def wav(self, y, name):
        path = os.path.join(self.tmp.name, name + '.wav')
        sf.write(path, y, SR, subtype='FLOAT')
        return path

    def test_the_array_entry_point_gives_what_the_file_gives(self):
        import librosa
        path = self.wav(voiced(3.0, noise=.1), 'v')
        y, sr = librosa.load(path, sr=SR, mono=True)
        a, b = X.timbre_features(path), X.timbre_features_array(y, sr)
        for key in ('f0', 'centroid', 'hnr', 'hnr_method', 'voiced_frames'):
            self.assertEqual(a[key], b[key], key)

    def test_a_signal_at_another_rate_is_resampled_like_a_file(self):
        import librosa
        y = voiced(3.0)
        a = X.timbre_features_array(y, SR)
        b = X.timbre_features_array(librosa.resample(y, orig_sr=SR, target_sr=44100), 44100)
        self.assertAlmostEqual(a['f0'], b['f0'], delta=1.5)
        self.assertAlmostEqual(a['centroid'], b['centroid'], delta=60.)

    def test_nothing_voiced_is_none_not_a_perfect_match(self):
        self.assertIsNone(X.timbre_features_array(floor(2.0), SR))
        self.assertIsNone(X.timbre_features_array(np.zeros(100, np.float32), SR))

    def test_the_meter_agrees_with_a_full_measure_of_a_coloured_render(self):
        from scipy.signal import lfilter
        base = voiced(4.0, tilt_db_per_khz=-8.0, noise=.2, seed=2)
        ref = X.timbre_features(self.wav(voiced(4.0, tilt_db_per_khz=-3.0, noise=.05, seed=1), 'ref'))
        meter = X.TimbreMeter(ref, self.wav(base, 'base'))
        self.assertIsNotNone(meter.voicing)
        for name, y in (('same', base),
                        ('duller', lfilter([.3], [1, -.7], base).astype(np.float32)),
                        ('louder', np.clip(2.0 * base, -1, 1).astype(np.float32))):
            with self.subTest(name):
                path = self.wav(y, name)
                full = X.timbre_score(ref, X.timbre_features(path, hnr_method=ref['hnr_method']))
                self.assertAlmostEqual(meter.score(path), full, delta=.01)

    def test_the_meter_follows_the_effect_it_is_asked_about(self):
        # The point of scoring timbre in the [] search: a darker render is a
        # different number, which ECAPA (blind to it) would not tell.
        from scipy.signal import lfilter
        ref = X.timbre_features(self.wav(voiced(4.0, tilt_db_per_khz=-3.0, seed=1), 'ref'))
        base = voiced(4.0, tilt_db_per_khz=-3.0, seed=2)
        meter = X.TimbreMeter(ref, self.wav(base, 'base'))
        dull = self.wav(lfilter([.15], [1, -.85], base).astype(np.float32), 'dull')
        self.assertGreater(meter.score(self.wav(base, 'again')), meter.score(dull) + .05)

    def test_a_shorter_render_is_measured_from_scratch(self):
        base = voiced(4.0, seed=2)
        ref = X.timbre_features(self.wav(voiced(4.0, seed=1), 'ref'))
        meter = X.TimbreMeter(ref, self.wav(base, 'base'))
        short = self.wav(base[:int(len(base) * .6)], 'short')
        with patch.object(X, 'timbre_features', wraps=X.timbre_features) as spy:
            got = meter.features(short)
        self.assertEqual(spy.call_count, 1)                      # the fallback
        self.assertEqual(got, X.timbre_features.__wrapped__(short, SR, ref['hnr_method'])
                         if hasattr(X.timbre_features, '__wrapped__') else got)
        self.assertIsNotNone(got)

    def test_a_render_laid_out_like_the_baseline_is_measured_against_its_voicing(self):
        # The EQ, the filters, the gate, the pan and the level do not move a
        # sample, a reverb adds its last echo (80 ms) at the end: the baseline's
        # pitch tracking holds for all of them, and no pitch is tracked again.
        base = voiced(4.0, seed=2)
        ref = X.timbre_features(self.wav(voiced(4.0, seed=1), 'ref'))
        meter = X.TimbreMeter(ref, self.wav(base, 'base'))
        for name, y in (('same length', base),
                        ('an echo tail', np.concatenate([base, np.zeros(int(.08 * SR), np.float32)]))):
            with self.subTest(name), patch.object(X, 'timbre_features') as scratch, \
                 patch.object(X, 'timbre_voicing', wraps=X.timbre_voicing) as tracking:
                self.assertIsNotNone(meter.features(self.wav(y, 'candidate')))
                scratch.assert_not_called()
                tracking.assert_not_called()

    def test_a_render_stretched_by_a_speed_change_is_measured_from_scratch(self):
        # Its voiced frames are not where the baseline's are: the baseline's
        # voicing would weigh the wrong frames, so the file is measured whole.
        base = voiced(4.0, seed=2)
        ref = X.timbre_features(self.wav(voiced(4.0, seed=1), 'ref'))
        meter = X.TimbreMeter(ref, self.wav(base, 'base'))
        stretched = np.interp(np.linspace(0, len(base) - 1, int(len(base) * 1.1)),
                              np.arange(len(base)), base).astype(np.float32)
        for name, y in (('10 % longer', stretched),
                        ('10 % shorter', base[:int(len(base) * .9)]),
                        ('0.3 s longer', np.concatenate([base, np.zeros(int(.3 * SR), np.float32)])),
                        ('10 ms shorter', base[:-int(.01 * SR)])):
            with self.subTest(name), patch.object(X, 'timbre_features',
                                                  wraps=X.timbre_features) as scratch:
                path = self.wav(y, 'candidate')
                got = meter.features(path)
                self.assertEqual(scratch.call_count, 1)
                self.assertEqual(scratch.call_args.args[2], ref['hnr_method'])
                self.assertIsNotNone(got)

    def test_the_alignment_window_is_the_one_documented(self):
        base = voiced(4.0, seed=2)
        ref = X.timbre_features(self.wav(voiced(4.0, seed=1), 'ref'))
        meter = X.TimbreMeter(ref, self.wav(base, 'base'))
        n = len(base)
        self.assertEqual(meter.length, n)
        self.assertTrue(meter.aligned(n))
        self.assertTrue(meter.aligned(n + int(.12 * SR)))
        self.assertFalse(meter.aligned(n + int(.13 * SR)))
        self.assertFalse(meter.aligned(n - 1))             # fewer frames: nothing to measure against
        self.assertFalse(X.TimbreMeter(ref, self.wav(floor(2.0), 'silence')).aligned(n))

    def test_a_signal_shorter_than_its_voicing_is_not_measured_with_it(self):
        base = voiced(4.0, seed=2)
        voicing = X.timbre_voicing(base, SR)
        self.assertIsNotNone(X.timbre_measure(base, SR, voicing))
        self.assertIsNone(X.timbre_measure(base[:len(base) // 2], SR, voicing))

    def test_the_meter_measures_like_the_reference_whichever_method_that_was(self):
        # A reference measured without parselmouth (harmonic/percussive HNR) is
        # only comparable with clones measured the same way.
        ref = X.timbre_features(self.wav(voiced(4.0, seed=1), 'ref'), hnr_method='hpss')
        self.assertEqual(ref['hnr_method'], 'hpss')
        base = voiced(4.0, seed=2)
        meter = X.TimbreMeter(ref, self.wav(base, 'base'))
        for name, y in (('aligned', base), ('stretched', base[:int(len(base) * .8)])):
            with self.subTest(name):
                self.assertEqual(meter.features(self.wav(y, name))['hnr_method'], 'hpss')
                self.assertIsNotNone(meter.score(self.wav(y, name)))

    def test_a_baseline_too_short_to_track_leaves_nothing_to_measure_against(self):
        ref = X.timbre_features(self.wav(voiced(3.0, seed=1), 'ref'))
        meter = X.TimbreMeter(ref, self.wav(voiced(.2, seed=2), 'blip'))
        self.assertIsNone(meter.voicing)
        self.assertIsNone(meter.length)
        self.assertFalse(meter.aligned(int(.2 * SR)))

    def test_a_baseline_without_voice_falls_back_to_full_measures(self):
        ref = X.timbre_features(self.wav(voiced(3.0, seed=1), 'ref'))
        meter = X.TimbreMeter(ref, self.wav(floor(2.0), 'silence'))
        self.assertIsNone(meter.voicing)
        path = self.wav(voiced(3.0, seed=3), 'v')
        self.assertAlmostEqual(meter.score(path),
                               X.timbre_score(ref, X.timbre_features(path, hnr_method=ref['hnr_method'])),
                               places=9)

    def test_an_unmeasurable_render_scores_none(self):
        ref = X.timbre_features(self.wav(voiced(3.0, seed=1), 'ref'))
        meter = X.TimbreMeter(ref, self.wav(voiced(3.0, seed=2), 'base'))
        self.assertIsNone(meter.score(self.wav(floor(2.0), 'silence')))


@needs_librosa
class TimbreReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def wav(self, y, name):
        path = os.path.join(self.tmp.name, name + '.wav')
        sf.write(path, y, SR)
        return path

    def run_report(self, *paths):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = timbre_report.main(list(paths))
        return code, out.getvalue()

    def test_it_prints_the_terms_for_each_clone_and_a_summary(self):
        ref = self.wav(voiced(3.0, f0=210.), 'ref')
        same = self.wav(voiced(3.0, f0=210., seed=5), 'same')
        low = self.wav(voiced(3.0, f0=165., tilt_db_per_khz=-8., noise=.4, seed=3), 'low')
        code, out = self.run_report(ref, same, low)
        self.assertEqual(code, 0)
        self.assertIn('Reference ref.wav: F0 210 Hz', out)
        self.assertIn('same.wav', out)
        self.assertIn('no term costs anything', out)
        self.assertIn('largest cost', out)
        self.assertRegex(out, r'1\.00\s+same\.wav')
        self.assertRegex(out, r'0\.\d\d\s+low\.wav')

    def test_a_silent_clone_is_reported_and_all_silent_is_an_error(self):
        ref = self.wav(voiced(3.0), 'ref')
        quiet = self.wav(floor(2.0), 'quiet')
        code, out = self.run_report(ref, quiet)
        self.assertEqual(code, 1)
        self.assertIn('no measurable voiced audio', out)
        code, out = self.run_report(ref, self.wav(voiced(3.0, seed=4), 'ok'), quiet)
        self.assertEqual(code, 0)

    def test_the_clones_are_measured_like_the_reference(self):
        # A reference measured without parselmouth is only comparable with clones
        # measured the same way: the HNR method of the reference is imposed.
        ref, clone = self.wav(voiced(3.0), 'ref'), self.wav(voiced(3.0, seed=3), 'clone')
        calls = []
        real = X.timbre_features

        def spy(path, sr_target=22050, hnr_method=None):
            calls.append((os.path.basename(path), hnr_method))
            return real(path, sr_target, 'hpss' if os.path.basename(path) == 'ref.wav' else hnr_method)
        with patch.object(X, 'timbre_features', spy):
            code, out = self.run_report(ref, clone)
        self.assertEqual(code, 0)
        self.assertEqual(calls, [('ref.wav', None), ('clone.wav', 'hpss')])
        self.assertIn('HPSS fallback', out)

    def test_a_silent_reference_stops_the_report(self):
        code, out = self.run_report(self.wav(floor(2.0), 'quiet'), self.wav(voiced(3.0), 'clone'))
        self.assertEqual(code, 1)
        self.assertIn('[ERR]', out)

    def test_a_missing_file_is_a_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            timbre_report.main(['missing.wav', 'also_missing.wav'])
        self.assertEqual(cm.exception.code, 2)


# ── the search stops when it cannot pay ───────────────────────────────────────

def search(gain, tolerance=.01, checkpoint=None, budget=200, **kw):
    """search_audio on a synthetic score: .5 plus gain(block)."""
    def evaluate(block):
        return .5 + gain(block), block
    return APS.search_audio(START, evaluate, method='nelder', budget=budget, tolerance=tolerance,
                            held=APS.EDITORIAL_FIELDS, checkpoint=checkpoint, **kw)


LOWPASS = lambda size: (lambda b: size * (b['lp'] == 8000))


class EarlyStopTests(unittest.TestCase):
    def test_nothing_above_the_tolerance_stops_before_the_refinement(self):
        block, rep = search(LOWPASS(.004), tolerance=.01)
        self.assertEqual(rep['stopped']['rule'], 'no gain above the tolerance')
        self.assertEqual(rep['stopped']['stage'], 'trials')
        self.assertEqual(rep['stopped']['evaluations'], rep['evaluations'])
        self.assertAlmostEqual(rep['stopped']['gain'], .004, places=9)
        self.assertEqual(block, rep['baseline_block'])                # the baseline, whole
        self.assertEqual(rep['candidate_block'], rep['baseline_block'])
        self.assertEqual(rep['best_score'], rep['baseline_score'])
        self.assertAlmostEqual(rep['search_best_score'], .504, places=9)   # what was seen is kept
        self.assertEqual(rep['search_best_block']['lp'], 8000)
        self.assertEqual(rep['pruning']['steps'], [])
        # ... and it saved the refinement, which is most of the budget
        _, full = search(LOWPASS(.004), tolerance=.001)
        self.assertIsNone(full['stopped'])
        self.assertGreater(full['evaluations'], rep['evaluations'] + 20)

    def test_the_tolerance_is_the_line(self):
        for gain, stops in ((.0099, True), (.0101, False)):
            with self.subTest(gain=gain):
                _, rep = search(LOWPASS(gain), tolerance=.01)
                self.assertEqual(rep['stopped'] is not None, stops)

    def test_a_search_that_finds_nothing_at_all_stops_too(self):
        block, rep = search(lambda b: 0., tolerance=.004)
        self.assertEqual(rep['stopped']['rule'], 'no gain above the tolerance')
        self.assertEqual(rep['search_best_score'], rep['baseline_score'])
        self.assertEqual(block, rep['baseline_block'])

    def test_the_stop_says_so_to_the_progress_line(self):
        lines = []
        search(LOWPASS(.004), tolerance=.01, progress=lines.append)
        self.assertTrue(any('stopped before the joint refinement' in line for line in lines), lines)

    def test_a_checkpoint_that_refuses_returns_the_refused_candidate(self):
        seen = []

        def refuse(candidate, baseline):
            seen.append((candidate, baseline))
            return dict(proceed=False, reason='held-out gains -0.0100 and 0.0000',
                        validation=[{'gain': -.01}, {'gain': 0.}])
        block, rep = search(LOWPASS(.1), checkpoint=refuse)
        self.assertEqual(len(seen), 1)
        candidate, baseline = seen[0]
        self.assertEqual((candidate['lp'], baseline['lp']), (8000, 0))
        self.assertEqual(rep['stopped']['rule'], 'does not carry over to other sentences')
        self.assertEqual(rep['stopped']['reason'], 'held-out gains -0.0100 and 0.0000')
        self.assertEqual(rep['early_validation'], [{'gain': -.01}, {'gain': 0.}])
        self.assertEqual(block['lp'], 8000)                       # what was refused, for the caller
        self.assertEqual(rep['candidate_block']['lp'], 8000)
        self.assertAlmostEqual(rep['best_score'], .6, places=9)
        self.assertEqual(rep['pruning']['steps'], [])
        _, full = search(LOWPASS(.1))
        self.assertGreater(full['evaluations'], rep['evaluations'] + 20)

    def test_a_checkpoint_that_proceeds_leaves_the_search_as_it_was(self):
        calls = []
        block, rep = search(LOWPASS(.1), checkpoint=lambda c, b: (
            calls.append(1), dict(proceed=True, validation=[{'gain': .1}]))[1])
        plain_block, plain = search(LOWPASS(.1))
        self.assertEqual(len(calls), 1)
        self.assertEqual(block, plain_block)
        for key in ('evaluations', 'search_best_score', 'best_score', 'candidate_block',
                    'search_best_block', 'pruning', 'one_factor_best'):
            self.assertEqual(rep[key], plain[key], key)
        self.assertIsNone(rep['stopped'])
        self.assertEqual(rep['early_validation'], [{'gain': .1}])

    def test_the_checkpoint_is_not_asked_when_nothing_gained(self):
        block, rep = search(LOWPASS(.004), tolerance=.01,
                            checkpoint=lambda c, b: self.fail('asked about a change below the tolerance'))
        self.assertIsNotNone(rep['stopped'])

    def test_the_objective_is_named_in_the_report(self):
        self.assertEqual(search(LOWPASS(.1))[1]['objective'], 'ecapa_identity')
        self.assertEqual(search(LOWPASS(.1), objective='identity+timbre')[1]['objective'],
                         'identity+timbre')
        self.assertEqual(search(LOWPASS(.004), objective='timbre')[1]['objective'], 'timbre')

    def test_the_inner_objective_function_does_not_shadow_the_name(self):
        # The refinement defines a function of its own: the report must keep the string.
        _, rep = search(LOWPASS(.1), objective='identity+timbre')
        self.assertIsInstance(rep['objective'], str)
        json.dumps(rep)


class WordingTests(unittest.TestCase):
    def test_what_each_objective_says_in_a_preset(self):
        self.assertEqual(set(APS.OBJECTIVES), {'identity', 'timbre', 'both'})
        self.assertIn('objective ECAPA identity only', APS.objective_line('ecapa_identity'))
        self.assertIn('{} objective applies to XTTS generation', APS.objective_line('ecapa_identity'))
        self.assertIn('objective identity+timbre', APS.objective_line('identity+timbre'))
        self.assertIn('accent is not scored', APS.objective_line('identity+timbre'))
        self.assertIn('objective timbre', APS.objective_line('timbre'))
        self.assertIn('ECAPA identity only', APS.objective_line('anything else'))
        self.assertEqual(APS.score_label('ecapa_identity'), 'ECAPA')
        self.assertEqual(APS.score_label(None), 'ECAPA')
        self.assertEqual(APS.score_label('identity+timbre'), 'score')

    def test_the_stop_line(self):
        self.assertIsNone(APS.stop_line(None))
        self.assertIsNone(APS.stop_line({'stopped': None}))
        line = APS.stop_line({'budget': 400, 'stopped': {'evaluations': 77, 'reason': 'no gain'}})
        self.assertEqual(line, 'stopped after 77 of 400 evaluations, before the joint refinement: no gain')


# ── the comparator ────────────────────────────────────────────────────────────

BLOCK = '[1, FR, 1, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 1]'


class Harness(unittest.TestCase):
    run_main = tone_tests.ComparatorStageTests.run_main

    def setUp(self):
        # See test_tone_fit_regressions: main() runs with stand-ins for soundfile,
        # and no real audio analysis may start under them. The tests that need a
        # timbre bring a fake (self.timbre) on top of these.
        for name in ('timbre_features', 'timbre_features_array'):
            patcher = patch.object(X, name, lambda *a, **kw: None)
            patcher.start()
            self.addCleanup(patcher.stop)

    def timbre(self, rule, reference=REF, clones=None, halves=None):
        """Patches giving the comparator a timbre: `rule(block)` is the score of the
        render made last. `clones` maps a file name to the features measured there;
        `halves` is a list of features for the two halves of the tolerance render."""
        seen = []

        class Meter:
            def __init__(self, ref, baseline_path, sr_target=22050):
                self.ref, self.sentence = ref, self.sentence_of(baseline_path)

            @staticmethod
            def sentence_of(path):
                # 'cand.wav' is the sentence of the trials, '*_holdout_N.wav' the others
                match = re.search(r'holdout_(\d)', Path(path).name)
                return match.group(1) if match else 'fit'

            def score(self, path):
                # The pitch of one sentence is tracked once: a meter built on one
                # sentence must never be asked about another.
                assert self.sentence_of(path) == self.sentence, (self.sentence, path)
                return rule(seen[-1])
        got = list(halves or [])

        def features(path, sr_target=22050, hnr_method=None):
            name = Path(path).name
            if clones is not None and name in clones:
                return clones[name]
            return reference if name == 'reference.wav' else None

        def array(y, sr, sr_target=22050, hnr_method=None):
            return got.pop(0) if got else dict(reference)
        patches = [patch.object(X, 'timbre_features', features),
                   patch.object(X, 'timbre_features_array', array),
                   patch.object(X, 'TimbreMeter', Meter)]
        return seen, patches

    def run_with(self, patches, block, extra, identity, return_report=True, **kw):
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            return self.run_main(block, extra, identity=identity, return_report=return_report, **kw)


class CriterionTests(Harness):
    def test_without_the_option_the_search_scores_identity_only(self):
        # --objective defaults to identity: whoever does not ask for the timbre
        # gets the search of 3.8, even when a timbre could be measured.
        seen, patches = self.timbre(lambda b: .3)

        def identity(block):
            seen.append(block)
            return .7
        out, _, report = self.run_with(patches, BLOCK,
            ['--optimise-audio', 'nelder', '--optimise-budget', '80'], identity)
        self.assertEqual(report['objective'], 'ecapa_identity')
        self.assertIn('target: ECAPA identity', out)
        self.assertNotIn('Reference timbre', out)

    def test_the_timbre_term_makes_an_effect_visible_that_ecapa_cannot_see(self):
        # ECAPA is flat; the timbre improves with the treble cut by 4 dB.
        seen, patches = self.timbre(lambda b: .3 + .2 * (b.get('eq_high') == -4))

        def identity(block):
            seen.append(block)
            return .7
        out, _, report = self.run_with(patches, BLOCK,
            ['--optimise-audio', 'nelder', '--objective', 'both', '--optimise-budget', '120'], identity)
        self.assertNotIn('Audio search failed', out)
        self.assertIn('target: identity + timbre', out)
        self.assertEqual(report['objective'], 'identity+timbre')
        self.assertTrue(report['adopted'])
        self.assertEqual(report['final_block']['eq_high'], -4)
        self.assertAlmostEqual(report['baseline_score'], .5 * .7 + .5 * .3, places=6)
        self.assertAlmostEqual(report['best_score'], .5 * .7 + .5 * .5, places=6)
        self.assertEqual(report['reference_timbre']['f0'], 200.)
        self.assertIn('ADOPTED', out)
        # ... which the identity objective, on the same renders, cannot find
        seen2, patches2 = self.timbre(lambda b: .3 + .2 * (b.get('eq_high') == -4))

        def identity2(block):
            seen2.append(block)
            return .7
        out, _, report = self.run_with(patches2, BLOCK,
            ['--optimise-audio', 'nelder', '--objective', 'identity', '--optimise-budget', '120'], identity2)
        self.assertEqual(report['objective'], 'ecapa_identity')
        self.assertFalse(report['adopted'])
        self.assertEqual(report['final_block']['eq_high'], 0)

    def test_the_objectives_weigh_identity_and_timbre_as_the_other_search_does(self):
        # Identity prefers the treble up by 0.1, timbre prefers it down by 0.2.
        def run(objective):
            seen, patches = self.timbre(lambda b: .3 + .2 * (b.get('eq_high') == -4))

            def identity(block):
                seen.append(block)
                return .7 + .1 * (block.get('eq_high') == 4)
            _, _, report = self.run_with(patches, BLOCK,
                ['--optimise-audio', 'nelder', '--objective', objective, '--optimise-budget', '120'],
                identity)
            return report
        self.assertEqual(run('identity')['final_block']['eq_high'], 4)       # ECAPA alone
        self.assertEqual(run('timbre')['final_block']['eq_high'], -4)        # timbre alone
        both = run('both')                                                   # .05 against .1
        self.assertEqual(both['final_block']['eq_high'], -4)
        self.assertAlmostEqual(both['baseline_score'], .5, places=6)
        self.assertAlmostEqual(both['best_score'], .6, places=6)

    def test_an_unmeasurable_reference_falls_back_to_identity_and_says_so(self):
        seen, patches = self.timbre(lambda b: 1., reference=None)

        def identity(block):
            seen.append(block)
            return .7
        out, _, report = self.run_with(patches, BLOCK,
            ['--optimise-audio', 'nelder', '--objective', 'both', '--optimise-budget', '120'], identity)
        self.assertIn('reference timbre could not be measured', out)
        self.assertIn('target: ECAPA identity', out)
        self.assertEqual(report['objective'], 'ecapa_identity')
        self.assertNotIn('reference_timbre', report)

    def test_a_render_without_measurable_voice_scores_the_worst_timbre(self):
        seen, patches = self.timbre(lambda b: None if b.get('lp') == 8000 else .5)

        def identity(block):
            seen.append(block)
            return .7 + .1 * (block.get('lp') == 8000)
        _, _, report = self.run_with(patches, BLOCK,
            ['--optimise-audio', 'nelder', '--objective', 'both', '--optimise-budget', '120'], identity)
        # ECAPA alone would have taken the low-pass (+0.1); its render has no voice
        self.assertEqual(report['final_block']['lp'], 0)
        self.assertNotEqual(report['search_best_block']['lp'], 8000)

    def test_the_tolerance_is_the_spread_of_the_score_the_search_maximises(self):
        # The halves of the tone-fitted render score 0.8 and 0.4 in timbre,
        # 0.7 in identity: a spread of 0.1 for the mean of the two.
        halves = [clone_features(centroid=75.), clone_features(centroid=450.)]   # d 0.25 and 1.5
        seen, patches = self.timbre(lambda b: .5, halves=list(halves))

        def identity(block):
            seen.append(block)
            return .7
        out, _, report = self.run_with(patches, BLOCK,
            ['--optimise-audio', 'nelder', '--objective', 'both', '--optimise-budget', '120'], identity)
        self.assertIn('pruning tolerance 0.1000 (score spread', out)
        self.assertAlmostEqual(report['pruning']['tolerance'], .1, places=6)
        # ... also when the screening has measured an ECAPA tolerance of its own
        seen, patches = self.timbre(lambda b: .5, halves=list(halves))
        _, _, report = self.run_with(patches, BLOCK,
            ['--optimise-audio', 'nelder', '--objective', 'both', '--optimise-budget', '120',
             '--screen-audio'], lambda block: (seen.append(block), .7)[1])
        self.assertAlmostEqual(report['pruning']['tolerance'], .1, places=6)
        self.assertAlmostEqual(report['adoption_tolerance'], .2, places=6)       # 2 x the tolerance
        seen, patches = self.timbre(lambda b: .5, halves=list(halves))
        _, _, report = self.run_with(patches, BLOCK,
            ['--optimise-audio', 'nelder', '--objective', 'timbre', '--optimise-budget', '120'],
            lambda block: (seen.append(block), .7)[1])
        self.assertAlmostEqual(report['pruning']['tolerance'], .2, places=6)     # |.8 - .4| / 2
        seen, patches = self.timbre(lambda b: .5, halves=list(halves))
        out, _, report = self.run_with(patches, BLOCK,
            ['--optimise-audio', 'nelder', '--objective', 'identity', '--optimise-budget', '120'],
            lambda block: (seen.append(block), .7)[1])
        self.assertLess(report['pruning']['tolerance'], .01)                     # ECAPA halves agree
        self.assertIn('pruning tolerance', out)
        self.assertIn('identity spread', out)

    def test_a_half_whose_timbre_cannot_be_measured_leaves_the_identity_spread(self):
        patch_array = patch.object(X, 'timbre_features_array', lambda *a, **kw: None)
        seen, patches = self.timbre(lambda b: .5)
        patches[1] = patch_array

        def identity(block):
            seen.append(block)
            return .7
        _, _, report = self.run_with(patches, BLOCK,
            ['--optimise-audio', 'nelder', '--objective', 'both', '--optimise-budget', '120'], identity)
        self.assertLess(report['pruning']['tolerance'], .01)


class EarlyStopHarnessTests(Harness):
    def test_a_search_that_finds_nothing_stops_and_never_generates_the_other_sentences(self):
        out, quiet, report = self.run_main(BLOCK,
            ['--optimise-audio', 'nelder', '--optimise-budget', '400'], return_report=True,
            identity=lambda block: .8)
        stop = report['stopped']
        self.assertEqual(stop['rule'], 'no gain above the tolerance')
        self.assertLess(report['evaluations'], 100)
        self.assertLess(report['evaluations'], report['budget'] // 3)
        self.assertEqual([n for n in self.generated if 'holdout' in n], [])
        self.assertEqual(report['validation'], [])
        self.assertEqual(report['validation_evaluations'], 0)
        self.assertIn(f"Stopped after {stop['evaluations']} of 400 evaluations, before the joint "
                      f"refinement", out)
        self.assertIn('REJECTED', out)
        self.assertNotIn('Checkpoint', out)
        self.assertFalse(report['adopted'])
        self.assertEqual(report['final_block'], report['baseline_block'])

    def test_a_change_that_does_not_carry_over_stops_at_the_checkpoint(self):
        # The case of the log: a gain on the sentence of the trials, none elsewhere.
        def identity(block):
            return .7 + .1 * (block.get('lp') == 8000) * (0 if 'holdout' in block['_source'] else 1)
        out, quiet, report = self.run_main(BLOCK,
            ['--optimise-audio', 'nelder', '--optimise-budget', '400'], return_report=True,
            identity=identity)
        stop = report['stopped']
        self.assertEqual(stop['rule'], 'does not carry over to other sentences')
        self.assertLess(report['evaluations'], 120)
        self.assertIn('Checkpoint:', out)
        self.assertEqual(out.count('held-out clone'), 2)          # not validated twice
        self.assertEqual(sorted(n for n in self.generated if 'holdout' in n),
                         ['clone_raw_holdout_1.wav', 'clone_raw_holdout_2.wav'])
        self.assertEqual(report['candidate_block']['lp'], 8000)   # what was refused
        self.assertEqual(report['final_block']['lp'], 0)
        self.assertEqual(report['validation'], report['early_validation'])
        self.assertEqual(report['validation_evaluations'], 4)
        self.assertEqual(len(report['validation_renders']), 2)
        self.assertIn('REJECTED: insufficient independent gains', out)
        self.assertIn('Stopped after', out)

    def test_a_change_that_carries_over_is_refined_and_validated_again(self):
        out, quiet, report = self.run_main(BLOCK,
            ['--optimise-audio', 'nelder', '--optimise-budget', '120'], return_report=True,
            identity=lambda block: .7 + .1 * (block.get('lp') == 8000))
        self.assertIsNone(report['stopped'])
        self.assertEqual(out.count('Checkpoint:'), 1)
        self.assertEqual(out.count('held-out clone'), 4)          # at the checkpoint, then at the end
        self.assertEqual(len(report['early_validation']), 2)
        self.assertEqual(len(report['validation']), 2)
        # The two other sentences are generated once and reused
        self.assertEqual([n for n in self.generated if 'holdout' in n].count('clone_raw_holdout_1.wav'), 1)
        self.assertEqual([n for n in self.generated if 'holdout' in n].count('clone_raw_holdout_2.wav'), 1)
        self.assertTrue(report['adopted'])
        self.assertGreater(report['evaluations'], 60)             # it did refine

    def test_the_baseline_of_each_other_sentence_is_rendered_once(self):
        # Rendered and scored when first needed (the checkpoint), then kept:
        # the candidate alone is rendered again for the final block.
        self.run_main(BLOCK, ['--optimise-audio', 'nelder', '--optimise-budget', '120'],
                      return_report=True, identity=lambda block: .7 + .1 * (block.get('lp') == 8000))
        for number in (1, 2):
            self.assertEqual(self.quiet_targets.count(f'base_holdout_{number}.wav'), 1)
            self.assertEqual(self.quiet_targets.count(f'cand_holdout_{number}.wav'), 2)

    def test_the_adoption_bar_includes_two_fifths_of_the_gain_where_it_was_found(self):
        # max(0.02, 2 x tolerance, 0.4 x fit gain): a change that gains 0.20 on the
        # sentence of the trials and 0.05 on the others clears the first bar, not the last (0.08).
        def run(held_out):
            def identity(block):
                if block.get('lp') != 8000:
                    return .7
                return .7 + (held_out if 'holdout' in block['_source'] else .2)
            return self.run_main(BLOCK, ['--optimise-audio', 'nelder', '--optimise-budget', '120'],
                                 return_report=True, identity=identity)
        for held_out, adopted in ((.05, False), (.07, False), (.09, True)):
            with self.subTest(held_out_gain=held_out):
                _, _, report = run(held_out)
                self.assertEqual(report['adopted'], adopted)
                self.assertAlmostEqual(report['adoption_tolerance'], .02)

    def test_the_checkpoint_asks_for_half_the_bar_on_both_sentences(self):
        # The bar is max(0.02, 2 x tolerance) = 0.02 here, half of it 0.01.
        def run(gain):
            def identity(block):
                if block.get('lp') != 8000:
                    return .7
                return .7 + (gain if 'holdout' in block['_source'] else .1)
            return self.run_main(BLOCK, ['--optimise-audio', 'nelder', '--optimise-budget', '120'],
                                 return_report=True, identity=identity)
        for gain, proceeds in ((.001, False), (.0099, False), (.0101, True), (.03, True)):
            with self.subTest(held_out_gain=gain):
                _, _, report = run(gain)
                self.assertEqual(report['stopped'] is None, proceeds)
        # A refined candidate still faces the old bar at the end: .0101 is not .04
        _, _, report = run(.0101)
        self.assertFalse(report['adopted'])
        self.assertEqual(len(report['validation']), 2)

    def test_one_negative_sentence_stops_it_whatever_the_mean(self):
        def identity(block):
            if block.get('lp') != 8000:
                return .7
            return .7 + {'clone_raw_holdout_1.wav': .3, 'clone_raw_holdout_2.wav': -.001}.get(block['_source'], .1)
        _, _, report = self.run_main(BLOCK, ['--optimise-audio', 'nelder', '--optimise-budget', '120'],
                                     return_report=True, identity=identity)
        self.assertEqual(report['stopped']['rule'], 'does not carry over to other sentences')

    def test_the_stop_is_in_the_preset_commentary(self):
        presets = []
        self.run_main(BLOCK, ['--optimise-audio', 'nelder', '--optimise-budget', '400', '--save-preset'],
                      return_report=True, identity=lambda block: .8, presets=presets)
        audio = ' '.join(self.preset_infos[0]['Audio search'])
        self.assertIn('stopped after', audio)
        self.assertIn('before the joint refinement', audio)
        self.assertIn('fit sentence ECAPA', audio)
        self.assertIn('objective ECAPA identity only', audio)

    def test_a_mixed_objective_is_named_in_the_preset_commentary(self):
        seen, patches = self.timbre(lambda b: .3 + .2 * (b.get('eq_high') == -4))

        def identity(block):
            seen.append(block)
            return .7
        presets = []
        self.run_with(patches, BLOCK, ['--optimise-audio', 'nelder', '--objective', 'both',
                                       '--optimise-budget', '120', '--save-preset'], identity,
                      presets=presets)
        audio = ' '.join(self.preset_infos[0]['Audio search'])
        self.assertIn('objective identity+timbre', audio)
        self.assertIn('fit sentence score 0.5000 -> 0.6000', audio)
        self.assertIn('result: adopted', audio)

    def timbre(self, rule, **kw):                 # same patches as CriterionTests
        return CriterionTests.timbre(self, rule, **kw)

    run_with = CriterionTests.run_with


class TimbreTableTests(Harness):
    def run_table(self, extra=(), presets=None):
        clones = {'raw.wav': clone_features(pitch_st=-3.4, centroid=-150., hnr=-1.5),
                  'base.wav': clone_features(pitch_st=-3.4, centroid=-120., hnr=-1.5),
                  'fit.wav': clone_features(pitch_st=-3.4, centroid=-100., hnr=-1.5)}
        _, patches = self.timbre(lambda b: .5, clones=clones)
        return self.run_with(patches, BLOCK, list(extra), lambda block: .8, presets=presets,
                             return_report=False)

    def timbre(self, rule, **kw):
        return CriterionTests.timbre(self, rule, **kw)

    run_with = CriterionTests.run_with

    def test_the_final_clone_is_broken_down_term_by_term(self):
        out, _ = self.run_table()
        self.assertIn('Timbre of the final clone, term by term', out)
        self.assertIn('pitch (F0)', out)
        self.assertIn('-3.4 st', out)
        self.assertIn('-100.0 Hz', out)                           # the final clone, not the raw one
        self.assertIn('largest cost: pitch (F0)', out)
        self.assertRegex(out, r'timbre score by stage: raw XTTS 0\.\d\d \| base \[\] 0\.\d\d \| final \[\] 0\.\d\d')

    def test_one_parsable_line_carries_it_to_the_pipeline(self):
        out, _ = self.run_table()
        line = re.search(r'^\[\*\] Timbre: (\{.*\})$', out, re.M).group(1)
        t = json.loads(line)
        self.assertEqual(t['largest'], 'pitch')
        self.assertAlmostEqual(t['gaps']['pitch'], -3.4, places=6)
        self.assertAlmostEqual(t['gaps']['brightness'], -100., places=6)
        self.assertAlmostEqual(t['distance'], 1.7 + 100 / 300 + .5, places=6)
        self.assertAlmostEqual(t['score'], 1 / (1 + t['distance']), places=6)
        self.assertEqual(t['reference']['f0'], 200.)
        self.assertEqual(set(t['stages']), {'raw XTTS', 'base []', 'final []'})
        self.assertLess(t['stages']['raw XTTS'], t['stages']['final []'])     # the [] helped here

    def test_it_is_written_in_the_preset(self):
        presets = []
        self.run_table(['--save-preset'], presets=presets)
        lines = ' '.join(self.preset_infos[0]['Timbre'])
        self.assertIn('reference: F0 200 Hz', lines)
        self.assertIn('pitch -3.4 st (1.70)', lines)
        self.assertIn('largest cost: pitch', lines)
        self.assertIn('timbre score by stage', lines)

    def test_no_reference_timbre_is_said_not_failed(self):
        _, patches = self.timbre(lambda b: .5, reference=None)
        out, _ = self.run_with(patches, BLOCK, [], lambda block: .8, return_report=False)
        self.assertIn('Timbre: not measured', out)
        self.assertNotIn('[*] Timbre:', out)
        self.assertIn('[OK] Done.', out)

    def test_an_unmeasurable_final_clone_is_said(self):
        _, patches = self.timbre(lambda b: .5, clones={'raw.wav': clone_features()})
        out, _ = self.run_with(patches, BLOCK, [], lambda block: .8, return_report=False)
        self.assertIn('final clone has no measurable voiced audio', out)
        self.assertNotIn('[*] Timbre:', out)


# ── the pipeline and the presets ──────────────────────────────────────────────

TIMBRE_LINE = {
    'reference': dict(REF), 'final': clone_features(pitch_st=-3.4, centroid=-150., hnr=-1.5),
    'stages': {'raw XTTS': .3, 'base []': .31, 'final []': .33},
    'terms': {'pitch': 1.7, 'brightness': .5, 'hnr': .5},
    'gaps': {'pitch': -3.4, 'brightness': -150., 'hnr': -1.5},
    'distance': 2.7, 'score': 1 / 3.7, 'largest': 'pitch'}


class PipelineTests(preset_tests.PipelineFixture):
    def comparator_commands(self):
        return [c for c in self.commands if os.path.basename(c[1]) == 'voice_comparator.py']

    def test_the_objective_reaches_every_comparator_run_when_the_search_is_on(self):
        # The identity search (3.9) follows --objective; the fidelity search
        # (3.10, the default) has its own criterion.
        self.run_pipeline(('--optimise-audio', 'nelder', '--audio-objective', 'identity',
                           '--objective', 'both', '--eq-weighting', 'a+blend'))
        commands = self.comparator_commands()
        self.assertEqual(len(commands), 2)
        for cmd in commands:
            self.assertEqual(cmd[cmd.index('--audio-objective') + 1], 'identity')
            self.assertEqual(cmd[cmd.index('--objective') + 1], 'both')

    def test_nothing_is_added_when_the_objective_is_identity_or_the_search_is_off(self):
        for extra in (('--optimise-audio', 'nelder', '--audio-objective', 'identity'),
                      ('--objective', 'both')):
            with self.subTest(extra=extra):
                self.run_pipeline(extra)
                for cmd in self.comparator_commands():
                    self.assertNotIn('--objective', cmd)

    def test_the_search_options_are_told_only_to_a_search_that_runs(self):
        self.run_pipeline(('--objective', 'both'))
        for cmd in self.comparator_commands():
            for flag in ('--audio-objective', '--audio-min-gain', '--audio-finalists',
                         '--exclude-unseen-texts-json', '--search-all-fields'):
                self.assertNotIn(flag, cmd)

    def test_the_sentences_the_optimiser_used_are_kept_from_the_final_check(self):
        # Only the fidelity search has a reserved sentence; the 3.9 search keeps its own.
        texts = dict(search_texts=['one', 'two'], conditioning_texts=['two', 'three'],
                     holdout_texts=['four'])
        self.run_pipeline(('--optimise-audio', 'nelder', '--eq-weighting', 'a+blend'),
                          optimiser_extra=texts)
        commands = self.comparator_commands()
        self.assertEqual(len(commands), 2)
        for cmd in commands:
            sent = json.loads(cmd[cmd.index('--exclude-unseen-texts-json') + 1])
            self.assertEqual(sent, ['one', 'two', 'three', 'four'])   # each once, in the order used
        self.run_pipeline(('--optimise-audio', 'nelder', '--audio-objective', 'identity'),
                          optimiser_extra=texts)
        for cmd in self.comparator_commands():
            self.assertNotIn('--exclude-unseen-texts-json', cmd)

    def test_the_fidelity_search_does_not_take_the_xtts_objective(self):
        self.run_pipeline(('--optimise-audio', 'nelder', '--objective', 'both',
                           '--eq-weighting', 'a+blend'))
        commands = self.comparator_commands()
        self.assertEqual(len(commands), 2)
        for cmd in commands:
            self.assertEqual(cmd[cmd.index('--audio-objective') + 1], 'fidelity')
            self.assertNotIn('--objective', cmd)           # it would be ignored there

    def test_the_timbre_of_the_comparator_is_recapped_and_written_in_the_preset(self):
        line = '[*] Timbre: ' + json.dumps(TIMBRE_LINE) + '\n'
        out, fitted, xb = self.run_pipeline(comparator_out=line)
        self.assertIn('# Timbre score 0.27 (raw XTTS 0.30 | base [] 0.31 | final [] 0.33); '
                      'largest cost: pitch -3.4 st (1.70 of 2.70)', out)
        info = VP.get('a')['info']
        self.assertIn('Timbre', info)
        text = ' '.join(info['Timbre'])
        self.assertIn('reference: F0 200 Hz', text)
        self.assertIn('largest cost: pitch', text)
        # the recap is commentary: the file still means two blocks
        file_text = Path(VP.path_of('a')).read_text(encoding='utf-8')
        self.assertEqual([l for l in file_text.splitlines() if l.strip() and not l.startswith('#')],
                         [xb, fitted])

    def test_the_recap_and_the_preset_name_the_term_that_costs_most(self):
        for largest, terms, gaps, expected in (
                ('brightness', {'pitch': .2, 'brightness': 1.8, 'hnr': .5}, {'pitch': -.4, 'brightness': -540., 'hnr': -1.5},
                 'largest cost: brightness -540.0 Hz (1.80 of 2.50)'),
                ('hnr', {'pitch': .2, 'brightness': .3, 'hnr': 2.0}, {'pitch': -.4, 'brightness': -90., 'hnr': -6.},
                 'largest cost: HNR -6.0 dB (2.00 of 2.50)')):
            with self.subTest(largest=largest):
                line = dict(TIMBRE_LINE, largest=largest, terms=terms, gaps=gaps, distance=2.5)
                out, _, _ = self.run_pipeline(comparator_out='[*] Timbre: ' + json.dumps(line) + '\n')
                self.assertIn(expected, out)
                self.assertIn('largest cost: ' + {'hnr': 'HNR'}.get(largest, largest),
                              ' '.join(VP.get('a')['info']['Timbre']))

    def test_without_a_timbre_line_nothing_is_invented(self):
        out, _, _ = self.run_pipeline()
        self.assertNotIn('Timbre score', out)
        self.assertNotIn('Timbre', VP.get('a')['info'])

    def test_a_garbled_timbre_line_is_ignored(self):
        for garbage in ('[*] Timbre: {not json}\n', '[*] Timbre: {"score": 1}\n', '[*] Timbre: {}\n',
                        # JSON with the key that makes it pass, and nothing else of a timbre:
                        '[*] Timbre: {"terms": {}}\n',
                        '[*] Timbre: {"terms": {"pitch": 1}, "largest": "pitch", "gaps": null, '
                        '"score": 0.3, "distance": 2}\n',
                        '[*] Timbre: {"terms": {"pitch": 1}, "largest": "pitch", "gaps": {"pitch": "x"}, '
                        '"score": 0.3, "distance": 2}\n'):
            with self.subTest(garbage=garbage):
                out, _, _ = self.run_pipeline(comparator_out=garbage)
                self.assertNotIn('Timbre score', out)
                self.assertIn('PIPELINE COMPLETE', out)
                info = VP.get('a')['info']
                self.assertNotIn('Timbre', info)
                self.assertIn('Run', info)                 # the rest of the commentary survives
                self.assertNotIn('Preset information skipped', out)

    def test_an_alternative_has_its_own_timbre(self):
        line = '[*] Timbre: ' + json.dumps(TIMBRE_LINE) + '\n'
        self.run_pipeline(('--eq-weighting', 'a+blend'), comparator_out=line)
        self.assertIn('Timbre', VP.get('a blend')['info'])

    def test_the_mixed_objective_is_in_the_audio_search_commentary(self):
        self.run_pipeline(('--optimise-audio', 'nelder', '--objective', 'both'),
                          search_report=dict(adopted=False, baseline_score=.5, best_score=.5,
                                             objective='identity+timbre',
                                             budget=400, stopped=dict(evaluations=80, reason='no gain')))
        audio = ' '.join(VP.get('a')['info']['Audio search'])
        self.assertIn('objective identity+timbre', audio)
        self.assertIn('fit sentence score 0.5000 -> 0.5000', audio)
        self.assertIn('stopped after 80 of 400 evaluations', audio)

    def test_the_old_reports_still_read_as_ecapa(self):
        # A report that names no objective is described by what was asked for.
        self.run_pipeline(('--optimise-audio', 'nelder', '--audio-objective', 'identity'),
                          search_report=dict(adopted=True, baseline_score=.6, best_score=.64))
        audio = ' '.join(VP.get('a')['info']['Audio search'])
        self.assertIn('objective ECAPA identity only', audio)
        self.assertIn('fit sentence ECAPA 0.6000 -> 0.6400', audio)

    def test_a_report_without_an_objective_is_described_by_the_search_that_was_asked_for(self):
        self.run_pipeline(('--optimise-audio', 'nelder'),
                          search_report=dict(adopted=True, baseline_score=.6, best_score=.64))
        audio = ' '.join(VP.get('a')['info']['Audio search'])
        self.assertIn('objective ECAPA identity + acoustic fidelity', audio)
        self.assertIn('fit sentence fidelity 0.6000 -> 0.6400', audio)
        self.assertNotIn('ECAPA identity only', audio)

    def bar_line(self, objective):
        self.run_pipeline(('--optimise-audio', 'nelder'),
                          search_report=dict(adopted=True, baseline_score=.6, best_score=.64,
                                             adoption_tolerance=.005, objective=objective))
        return ' '.join(VP.get('a')['info']['Audio search'])

    def test_the_bar_of_the_fidelity_search_is_said_to_be_on_three_paired_sentences(self):
        self.assertIn('bar on three paired sentences: 0.0050', self.bar_line('fidelity_v1'))

    def test_the_bar_of_the_identity_search_is_said_to_be_on_two_other_sentences(self):
        self.assertIn('bar on two other sentences: 0.0050', self.bar_line('ecapa_identity'))


class ExplainTimbreTests(preset_tests.Dir):
    def test_the_section_reads_without_the_code(self):
        text = '\n'.join(VP.explain_timbre(TIMBRE_LINE))
        for part in ('reference: F0 200 Hz | centroid 1300 Hz | HNR 16.0 dB',
                     'gap (cost): pitch -3.4 st (1.70) | brightness -150.0 Hz (0.50) | HNR -1.5 dB (0.50)',
                     'distance 2.70 -> timbre score 0.27; largest cost: pitch',
                     'timbre score by stage: raw XTTS 0.30 | base [] 0.31 | final [] 0.33'):
            self.assertIn(part, text)

    def test_it_names_the_term_that_costs_most_and_signs_the_gaps(self):
        line = dict(TIMBRE_LINE, largest='brightness', terms={'pitch': .2, 'brightness': 1.8, 'hnr': .5},
                    gaps={'pitch': .4, 'brightness': 540., 'hnr': 1.5}, distance=2.5)
        text = '\n'.join(VP.explain_timbre(line))
        self.assertIn('largest cost: brightness', text)
        self.assertIn('gap (cost): pitch +0.4 st (0.20) | brightness +540.0 Hz (1.80) | HNR +1.5 dB (0.50)', text)

    def test_a_stage_that_could_not_be_measured_is_marked(self):
        text = '\n'.join(VP.explain_timbre(dict(TIMBRE_LINE, stages={'raw XTTS': None, 'final []': .3})))
        self.assertIn('raw XTTS n/a | final [] 0.30', text)

    def test_it_survives_a_round_trip_through_the_file(self):
        VP.save('t', '{1, 100}', BLOCK, info={'Timbre': VP.explain_timbre(TIMBRE_LINE)})
        self.assertIn('largest cost: pitch', ' '.join(VP.get('t')['info']['Timbre']))


# ── the Comparator tab ────────────────────────────────────────────────────────

TARGET_MENU = ('fidelity', 'identity', 'timbre', 'both')     # the [Cmp] "Search target" values


class ComparatorTabTests(unittest.TestCase):
    """The tab run with stand-in widgets: the command it hands to run_cmd."""

    def run_tab(self, search='none', objective=None, all_fields=False):
        import math
        import types
        Fake = gui_tests.FakeWidget
        Fake.entries, Fake.labels, Fake.buttons = {}, {}, {}
        commands, callbacks, combos, checks, targets, logged = [], {}, {}, {}, {}, []

        class Variable:
            def __init__(self, value='', **kwargs):
                self.value, self.callbacks = value, []

            def get(self):
                return self.value

            def set(self, value):
                self.value = value
                for callback in self.callbacks:
                    callback()

            def trace_add(self, mode, callback):
                self.callbacks.append(callback)

        class Combo(Fake):
            def __init__(self, parent=None, **kw):
                super().__init__(parent, **kw)
                combos[tuple(kw.get('values', ()))] = kw.get('textvariable')

        class Check(Fake):
            def __init__(self, parent=None, **kw):
                super().__init__(parent, **kw)
                checks[kw.get('text', '')] = kw.get('variable')

        tk = types.SimpleNamespace(
            **{key: Fake for key in ('Frame', 'Label', 'Entry', 'Button', 'Spinbox')},
            Checkbutton=Check, StringVar=Variable, BooleanVar=Variable)
        with tempfile.TemporaryDirectory() as out:
            namespace = dict(
                tk=tk, ttk=types.SimpleNamespace(Combobox=Combo),
                scrolledtext=types.SimpleNamespace(ScrolledText=Fake),
                math=math, os=os, sys=sys, TARGETS=targets, DEFAULT_LANG='FR', LANGS=['FR', 'EN'],
                DIR_VOICES='/voices', DIR_OUTPUT=out, SCRIPTS_DIR='/scripts',
                scrollable_tab=lambda nb, title: (Fake(), Fake()),
                add_row=lambda *a, **kw: None, add_console=lambda *a: Fake(),
                make_btn=lambda parent, text, callback, row: callbacks.setdefault(text, callback),
                browse_file=lambda *a, **kw: None, _bind_default_text=lambda *a, **kw: None,
                _default_text=lambda lang: 'Bonjour.', _BUILTIN_TEXTS=set(),
                run_cmd=lambda command, *a, **kw: commands.append(command),
                log=lambda console, text: logged.append(text))
            gui_tests.extract_functions('xtts_studio.py', {'tab_comparator', 'split_paths'}, namespace)
            namespace['tab_comparator'](None)
            targets['cmp_reference'].set('/voices/reference.wav')
            targets['cmp_xtts'].set('{1, 0}')
            targets['cmp_audio'].set(BLOCK)
            combos[('none', 'nelder', 'de')].set(search)
            if objective is not None:
                combos[TARGET_MENU].set(objective)
            if all_fields:
                checks['Include speed, pan and limiter'].set(True)
            callbacks['>  Compare & Optimise'](None)
            self.assertEqual(logged, [])
            command = commands[0]
            temp = command[command.index('--text-file') + 1]
        with contextlib.suppress(OSError):
            os.remove(temp)
        return command, combos

    def test_one_menu_offers_the_four_targets_and_starts_on_fidelity(self):
        command, combos = self.run_tab()
        self.assertEqual(combos[TARGET_MENU].get(), 'fidelity')
        self.assertEqual([k for k in combos if 'fidelity' in k], [TARGET_MENU])   # no second selector
        self.assertNotIn('--objective', command)
        self.assertNotIn('--audio-objective', command)      # no search, nothing to tell it

    def test_the_target_is_forwarded_only_to_a_search_that_runs(self):
        off, _ = self.run_tab(search='none', objective='both')
        self.assertNotIn('--objective', off)
        self.assertNotIn('--audio-objective', off)
        default, _ = self.run_tab(search='nelder')
        self.assertEqual(default[default.index('--audio-objective') + 1], 'fidelity')
        self.assertNotIn('--objective', default)          # the fidelity search has its own criterion
        ecapa, _ = self.run_tab(search='nelder', objective='identity')
        self.assertEqual(ecapa[ecapa.index('--audio-objective') + 1], 'identity')   # said, not assumed
        self.assertNotIn('--objective', ecapa)
        for target in ('timbre', 'both'):
            with self.subTest(target=target):
                command, _ = self.run_tab(search='nelder', objective=target)
                self.assertEqual(command[command.index('--audio-objective') + 1], 'identity')
                self.assertEqual(command[command.index('--objective') + 1], target)
                self.assertEqual(command.count('--objective'), 1)
                self.assertEqual(command.count('--audio-objective'), 1)
                self.assertEqual(command[command.index('--optimise-audio') + 1], 'nelder')

    def test_it_comes_with_the_other_search_options_untouched(self):
        command, _ = self.run_tab(search='de', objective='both', all_fields=True)
        self.assertIn('--search-all-fields', command)
        self.assertEqual(command[command.index('--objective') + 1], 'both')
        self.assertEqual(command[command.index('--optimise-audio') + 1], 'de')

    def test_speed_pan_and_limiter_stay_held_unless_ticked_for_either_search(self):
        for target in ('fidelity', 'identity'):
            with self.subTest(target=target):
                held, _ = self.run_tab(search='nelder', objective=target)
                self.assertNotIn('--search-all-fields', held)
                free, _ = self.run_tab(search='nelder', objective=target, all_fields=True)
                self.assertIn('--search-all-fields', free)


if __name__ == '__main__':
    unittest.main()

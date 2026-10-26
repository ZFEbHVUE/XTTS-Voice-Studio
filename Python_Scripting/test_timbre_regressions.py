#!/usr/bin/env python3
"""Regression tests for the timbre term of the search objective (3.6).

The pitch, brightness and HNR used to be averaged over the whole file: silent
frames weighed as much as vowels, so the measure followed the pauses of each
render more than the voice. These tests pin the voiced-frame measure.
"""
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import xtts_optimize as X

import importlib.util
try:
    import soundfile as sf
except Exception:          # pragma: no cover - depends on the environment
    sf = None
if importlib.util.find_spec('librosa') is None:
    sf = None
HAVE_PRAAT = importlib.util.find_spec('parselmouth') is not None

SR = 22050
needs_librosa = unittest.skipIf(sf is None, 'librosa and soundfile are needed')


def voiced(seconds, f0=210.0, tilt_db_per_khz=-6.0, noise=0.0, seed=0):
    """A sung-vowel stand-in: harmonics under three formants, a slow pitch
    glide, syllable-like bursts. tilt sets the brightness, noise the HNR."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    pitch = f0 * 2 ** (0.5 * np.sin(2 * np.pi * 0.7 * t) / 12)
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    y = np.zeros_like(t)
    for k in range(1, 60):
        fk = k * f0
        if fk > 7000:
            break
        formants = sum(np.exp(-0.5 * ((fk - fc) / bw) ** 2)
                       for fc, bw in ((700, 150), (1250, 200), (2700, 300)))
        amp = (0.15 + formants) * 10 ** (tilt_db_per_khz * fk / 1000 / 20)
        y += amp * np.sin(k * phase)
    syll = (np.sin(2 * np.pi * 3.0 * t) > -0.6).astype(float)   # gaps between bursts
    ramp = np.convolve(syll, np.hanning(441) / np.hanning(441).sum(), mode='same')
    y = y * ramp
    if noise:
        y = y + noise * rng.normal(0, 1, len(y)) * ramp * np.std(y)
    return (0.3 * y / np.max(np.abs(y))).astype(np.float32)


def floor(seconds, seed=1):
    return np.random.default_rng(seed).normal(0, 10 ** (-65 / 20), int(seconds * SR)).astype(np.float32)


@needs_librosa
class VoicedTimbreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def wav(self, y, name):
        path = os.path.join(self.tmp.name, name + '.wav')
        sf.write(path, y, SR)
        return path

    def test_pauses_do_not_move_the_measure(self):
        core = voiced(3.0)
        a = X.timbre_features(self.wav(core, 'tight'))
        b = X.timbre_features(self.wav(np.concatenate([floor(0.4), core, floor(1.3, 2)]), 'padded'))
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertLess(abs(a['centroid'] - b['centroid']), 30.0)
        self.assertLess(abs(12 * np.log2(a['f0'] / b['f0'])), 0.1)
        self.assertLess(abs(a['hnr'] - b['hnr']), 1.0)
        # The old whole-file centroid moves by hundreds of Hz on the same pair.
        import librosa
        old = [float(np.mean(librosa.feature.spectral_centroid(y=librosa.load(p, sr=SR)[0], sr=SR)))
               for p in (self.wav(core, 'tight2'),
                         self.wav(np.concatenate([floor(0.4), core, floor(1.3, 2)]), 'padded2'))]
        self.assertGreater(abs(old[0] - old[1]), 5 * abs(a['centroid'] - b['centroid']))

    def test_an_unvoiced_burst_is_left_out(self):
        core = voiced(3.0)
        hiss = np.random.default_rng(3).normal(0, 0.03, int(0.25 * SR)).astype(np.float32)
        a = X.timbre_features(self.wav(core, 'clean'))
        b = X.timbre_features(self.wav(np.concatenate([core, hiss, floor(0.3)]), 'hiss'))
        self.assertLess(abs(a['centroid'] - b['centroid']), 60.0)

    def test_a_brighter_voice_reads_brighter_and_further(self):
        ref = X.timbre_features(self.wav(voiced(3.0), 'ref'))
        same = X.timbre_features(self.wav(voiced(3.0, seed=5), 'same'))
        bright = X.timbre_features(self.wav(voiced(3.0, tilt_db_per_khz=-2.0), 'bright'))
        self.assertGreater(bright['centroid'] - ref['centroid'], 150.0)
        self.assertGreater(X.timbre_score(ref, same), X.timbre_score(ref, bright))

    def test_pitch_follows_the_voice_and_ignores_octave_errors(self):
        y = voiced(3.0, f0=180.0)
        ok = X.timbre_features(self.wav(y, 'f0'))
        self.assertLess(abs(12 * np.log2(ok['f0'] / 180.0)), 0.3)
        import librosa
        real = librosa.pyin

        def doubled(*a, **k):                       # a tracker that jumps an octave
            f0, vf, vp = real(*a, **k)              # on one voiced frame in five
            f0 = f0.copy()
            idx = np.flatnonzero(np.isfinite(f0))[::5]
            f0[idx] *= 2.0
            return f0, vf, vp
        with mock.patch.object(librosa, 'pyin', doubled):
            bad = X.timbre_features(self.wav(y, 'f0b'))
        self.assertLess(abs(12 * np.log2(bad['f0'] / ok['f0'])), 0.1)

    def test_breathier_voice_has_lower_hnr(self):
        clean = X.timbre_features(self.wav(voiced(3.0), 'c'))
        breathy = X.timbre_features(self.wav(voiced(3.0, noise=0.5), 'b'))
        self.assertGreater(clean['hnr'] - breathy['hnr'], 3.0)

    def test_silence_is_not_a_perfect_match(self):
        self.assertIsNone(X.timbre_features(self.wav(floor(2.0), 'silence')))

    def test_clones_are_measured_like_the_reference(self):
        path = self.wav(voiced(3.0), 'v')
        with mock.patch.dict(sys.modules, {'parselmouth': None}):
            self.assertIsNone(X.timbre_features(path, hnr_method='praat'))
            fallback = X.timbre_features(path)
        self.assertEqual(fallback['hnr_method'], 'hpss')
        forced = X.timbre_features(path, hnr_method='hpss')
        self.assertEqual(forced['hnr_method'], 'hpss')
        if HAVE_PRAAT:
            self.assertEqual(X.timbre_features(path)['hnr_method'], 'praat')


class TimbreDistanceTests(unittest.TestCase):
    def test_different_hnr_methods_are_not_compared(self):
        ref = {'f0': 200.0, 'centroid': 1800.0, 'hnr': 16.0, 'hnr_method': 'praat'}
        clone = dict(ref, hnr=2.0, hnr_method='hpss')
        self.assertIsNone(X.timbre_distance(ref, clone))
        self.assertIsNone(X.timbre_score(ref, clone))
        self.assertEqual(X.timbre_distance(ref, dict(ref)), 0.0)

    def test_terms_use_their_audible_gaps(self):
        ref = {'f0': 200.0, 'centroid': 1800.0, 'hnr': 16.0, 'hnr_method': 'praat'}
        clone = {'f0': 200.0 * 2 ** (2 / 12), 'centroid': 2100.0, 'hnr': 13.0, 'hnr_method': 'praat'}
        self.assertAlmostEqual(X.timbre_distance(ref, clone), 3.0, places=6)

    def test_weighted_median(self):
        v = np.array([1.0, 2.0, 3.0, 100.0])
        self.assertEqual(X._weighted_median(v, np.array([1.0, 1.0, 5.0, 1.0])), 3.0)
        self.assertEqual(X._weighted_median(v, np.ones(4)), 2.0)


class SearchMeasuresClonesLikeTheReference(unittest.TestCase):
    def test_every_render_uses_the_reference_hnr_method(self):
        from test_holdout_report_regressions import run_optimizer
        calls = []

        def fake(path, sr_target=22050, hnr_method=None):
            calls.append((os.path.basename(path), hnr_method))
            return {'f0': 200.0, 'centroid': 1800.0, 'hnr': 15.0,
                    'hnr_method': hnr_method or 'hpss', 'voiced_frames': 50}
        with tempfile.TemporaryDirectory() as td, mock.patch.object(X, 'timbre_features', fake):
            run_optimizer(td, ['--objective', 'both', '--no-holdout', '--budget', '60'])
        reference, clones = calls[0], calls[1:]
        self.assertEqual(reference, ('a.wav', None))
        self.assertTrue(clones)
        self.assertEqual({method for _name, method in clones}, {'hpss'})


class FailedTakeTests(unittest.TestCase):
    """3.7.2: a take whose timbre cannot be measured (no usable voiced audio)."""
    MEASURED = {'f0': 200.0, 'centroid': 1800.0, 'hnr': 15.0, 'hnr_method': 'hpss',
                'voiced_frames': 50}

    def search(self, fails):
        from test_holdout_report_regressions import run_optimizer
        calls = []

        def fake(path, sr_target=22050, hnr_method=None):
            calls.append(path)
            take = len(calls) - 1              # call 0 measures the reference
            return None if take and fails(take) else dict(self.MEASURED)
        with tempfile.TemporaryDirectory() as td, mock.patch.object(X, 'timbre_features', fake):
            result, log, _ = run_optimizer(td, ['--objective', 'both', '--no-holdout',
                                                '--budget', '20'])
        return result, log, calls

    def test_one_failed_take_counts_as_the_worst_timbre_and_the_search_goes_on(self):
        result, log, calls = self.search(lambda take: take == 2)
        self.assertEqual(log.count('no measurable timbre in this take'), 1)
        self.assertGreater(len(calls), 10)                # every take after it was scored
        scores = [float(x) for x in re.findall(r'-> score=([\d.]+)', log)]
        # 0.6 accent + 0.4 x mean(0.5 id + 0.5 timbre) over the two sentences,
        # with timbre 0 for the failed one: 0.57 + 0.4 x (0.85 + 0.35) / 2. Its
        # identity alone (3.7.1) gave 0.88; the others score 0.91.
        self.assertAlmostEqual(min(scores), 0.81, places=3)
        self.assertEqual(sorted(scores)[1:], [0.91] * (len(scores) - 1))
        self.assertAlmostEqual(result['scores']['search'], 0.91, places=3)   # not the failed one

    def test_a_timbre_that_fails_on_most_takes_stops_the_search(self):
        # Not a take but the reference or the tools: say so instead of
        # ranking every take on a timbre of 0.
        with self.assertRaises(SystemExit) as stop:
            self.search(lambda take: True)
        self.assertIn('Timbre could not be measured on 3 of 3 takes', str(stop.exception))
        self.assertIn('--objective identity', str(stop.exception))


if __name__ == '__main__':
    unittest.main()

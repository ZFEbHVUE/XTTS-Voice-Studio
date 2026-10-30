"""3.10.1: what the review of 3.10.0 left open.

test_3100_review_regressions covers the objectives and the disabled search of
the --fidelity profile, and the speed routing of the timbre measure. This file
adds what the mutation check of that file found uncovered: the explicit values
of the other profile settings, the option names that argparse accepts in
abbreviated form, and the routing on a real signal instead of stand-in meters.
"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import test_3100_review_regressions as review


def _has(*modules):
    try:
        for name in modules:
            __import__(name)
    except Exception:
        return False
    return True


class ProfileKeepsExplicitValuesTests(unittest.TestCase):
    commands = staticmethod(review.ProfileOverrideTests.commands)
    option = staticmethod(review.ProfileOverrideTests.option)

    def test_profile_defaults_when_nothing_is_given(self):
        native, audio = self.commands([])
        self.assertEqual(self.option(native, '--w-accent'), '0.3')
        self.assertEqual(self.option(native, '--w-identity'), '0.7')
        self.assertEqual(self.option(native, '--cond-budget'), '80')
        self.assertEqual(self.option(native, '--joint-sampling'), '12')
        self.assertEqual(self.option(native, '--beam-width'), '2')
        self.assertEqual(self.option(audio, '--eq-limit'), '8.0')

    def test_explicit_values_survive_the_profile_in_both_spellings(self):
        cases = {'--w-accent': '0.5', '--w-identity': '0.5', '--cond-budget': '50',
                 '--joint-sampling': '6', '--beam-width': '3', '--budget': '77'}
        for name, value in cases.items():
            for flags in ([name, value], [f'{name}={value}']):
                with self.subTest(flags=flags):
                    native, _ = self.commands(flags)
                    self.assertEqual(self.option(native, name), value)

    def test_explicit_eq_limit_survives_the_profile_in_both_spellings(self):
        for flags in (['--eq-limit', '6'], ['--eq-limit=6']):
            with self.subTest(flags=flags):
                _, audio = self.commands(flags)
                self.assertEqual(float(self.option(audio, '--eq-limit')), 6.)

    def test_a_pair_of_explicit_weights_replaces_the_profile_pair(self):
        native, _ = self.commands(['--w-accent', '0.5', '--w-identity', '0.5'])
        self.assertEqual((self.option(native, '--w-accent'), self.option(native, '--w-identity')),
                         ('0.5', '0.5'))


class AbbreviatedNamesTests(unittest.TestCase):
    """argparse accepts '--obj' for '--objective'. The profile reads the names
    on the command line, so an abbreviation used to be parsed as the user's
    choice and then overwritten by the profile. It is an error now."""

    def rejected(self, flags):
        helper = review.render_tests.PipelineRecapTests('runTest')
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stderr(err), \
                self.assertRaises(SystemExit) as caught:
            helper.run_pipeline(td, ['--fidelity', *flags],
                                lambda command: (0, 'Next []    : [1, FR, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]'))
        self.assertEqual(caught.exception.code, 2)
        return err.getvalue()

    def test_a_shortened_option_name_is_refused(self):
        for flags in (['--obj', 'identity'], ['--optimise-a', 'none'], ['--audio-obj', 'identity'],
                      ['--eq-lim', '6'], ['--w-acc', '0.5']):
            with self.subTest(flags=flags):
                self.assertIn('unrecognized arguments', self.rejected(flags))

    def test_the_full_names_are_still_accepted(self):
        native, audio = review.ProfileOverrideTests.commands(
            ['--objective', 'identity', '--audio-objective', 'identity'])
        self.assertNotIn('--objective', native)
        self.assertEqual(review.ProfileOverrideTests.option(audio, '--audio-objective'), 'identity')


@unittest.skipUnless(_has('librosa', 'soundfile', 'scipy'), 'librosa, soundfile and scipy are needed')
class RealSignalRoutingTests(unittest.TestCase):
    """The speed routing on a real pitch track: a 1 % slowdown of 8 s lasts
    8.08 s, inside the 125 ms the meter allows for a reverb tail, so the
    length alone says 'aligned'. The block's speed is what tells otherwise."""

    SR = 22050

    @classmethod
    def voiced_signal(cls, seconds=8.0, seed=3):
        import numpy as np
        from scipy import signal
        rng = np.random.default_rng(seed)
        formants = [[(700, 90, 1.), (1200, 110, .6), (2600, 160, .3)],
                    [(300, 70, 1.), (2300, 150, .7), (3000, 180, .3)],
                    [(500, 80, 1.), (1700, 120, .5), (2500, 160, .3)]]

        def vowel(duration, f0a, f0b, bank):
            n = int(duration * cls.SR)
            phase = np.cumsum(2 * np.pi * np.linspace(f0a, f0b, n) / cls.SR)
            source = sum(np.sin(h * phase) / h ** 1.1 for h in range(1, int(5000 / min(f0a, f0b))))
            out = sum(g * signal.lfilter(*signal.iirpeak(fc / (cls.SR / 2), fc / bw), source)
                      for fc, bw, g in bank)
            return out * np.hanning(n) ** .35

        parts, elapsed = [], 0.
        while elapsed < seconds - .5:
            duration = float(rng.uniform(.14, .28))
            parts.append(vowel(duration, rng.uniform(180, 230), rng.uniform(180, 230),
                               formants[int(rng.integers(len(formants)))]))
            gap = float(rng.uniform(.04, .2))
            parts.append(np.zeros(int(gap * cls.SR)))
            elapsed += duration + gap
        y = np.concatenate(parts)[: int(seconds * cls.SR)]
        y = np.pad(y, (0, int(seconds * cls.SR) - len(y)))
        y = y + .003 * rng.standard_normal(len(y))
        return (.3 * y / np.max(np.abs(y))).astype('float32')

    def test_a_slowdown_inside_the_echo_window_is_measured_from_scratch(self):
        import soundfile as sf
        from scipy import signal
        import voice_comparator as comparator
        import xtts_optimize as optimizer
        with tempfile.TemporaryDirectory() as td:
            base, slow = Path(td, 'base.wav'), Path(td, 'slow.wav')
            y = self.voiced_signal()
            sf.write(base, y, self.SR)
            sf.write(slow, signal.resample_poly(y, 101, 100).astype('float32'), self.SR)
            reference = optimizer.timbre_features(str(base))
            if reference is None:
                self.skipTest('no voiced frames in the synthetic signal on this librosa')
            meter = optimizer.TimbreMeter(reference, str(base))
            self.assertTrue(meter.aligned(len(sf.read(slow)[0])),
                            'the stretched render must fall inside the echo window for this test to mean anything')
            routed = comparator._timbre_candidate_score(reference, meter, str(slow),
                                                        dict(speed=.99), dict(speed=1.))
            self.assertEqual(routed, optimizer.timbre_score(reference, optimizer.timbre_features(str(slow))))
            unchanged = comparator._timbre_candidate_score(reference, meter, str(slow),
                                                           dict(speed=1., eq_high=-3.), dict(speed=1.))
            self.assertEqual(unchanged, meter.score(str(slow)))


if __name__ == '__main__':
    unittest.main()

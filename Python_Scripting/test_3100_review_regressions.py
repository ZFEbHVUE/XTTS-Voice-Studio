"""Reproductions found while reviewing the merged 3.10.0.

The command and comparator tests run their real entry points with stand-in
models. No result here measures the perceptual quality of an XTTS clone.
"""
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import test_render_search_regressions as render_tests
import test_tone_fit_regressions as tone_tests
import voice_comparator as comparator
import xtts_optimize as optimizer


class ProfileOverrideTests(unittest.TestCase):
    @staticmethod
    def commands(flags):
        helper = render_tests.PipelineRecapTests('runTest')
        with tempfile.TemporaryDirectory() as td:
            commands, _, _ = helper.run_pipeline(td, ['--fidelity', *flags],
                lambda command: (0, 'Next []    : [1, FR, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]'),
                all_commands=True)
        return (next(c for c in commands if c[1].endswith('xtts_optimize.py')),
                next(c for c in commands if c[1].endswith('voice_comparator.py')))

    @staticmethod
    def option(command, name):
        return command[command.index(name) + 1] if name in command else None

    def test_explicit_native_objectives_survive_the_profile_in_both_spellings(self):
        for value in ('identity', 'timbre', 'both'):
            for flags in (['--objective', value], ['--objective=' + value]):
                with self.subTest(flags=flags):
                    native, _ = self.commands(flags)
                    self.assertEqual(self.option(native, '--objective') or 'identity', value)

    def test_explicit_audio_objective_survives_the_profile_in_both_spellings(self):
        for flags in (['--audio-objective', 'identity'], ['--audio-objective=identity']):
            with self.subTest(flags=flags):
                _, audio = self.commands(flags)
                self.assertEqual(self.option(audio, '--audio-objective'), 'identity')
                self.assertEqual(self.option(audio, '--objective'), 'both')

    def test_explicit_disabled_audio_search_stays_disabled_and_has_no_search_options(self):
        for flags in (['--optimise-audio', 'none'], ['--optimise-audio=none']):
            with self.subTest(flags=flags):
                _, audio = self.commands([*flags, '--optimise-budget', '1'])
                self.assertEqual(self.option(audio, '--optimise-audio') or 'none', 'none')
                for option in ('--audio-objective', '--audio-finalists', '--audio-min-gain',
                               '--objective', '--exclude-unseen-texts-json'):
                    self.assertNotIn(option, audio)

    def test_explicit_search_method_is_kept_and_unset_choices_keep_profile_defaults(self):
        for value in ('nelder', 'de'):
            with self.subTest(value=value):
                native, audio = self.commands(['--optimise-audio=' + value])
                self.assertEqual(self.option(audio, '--optimise-audio'), value)
                self.assertEqual(self.option(audio, '--audio-objective'), 'fidelity')
                self.assertEqual(self.option(native, '--objective'), 'both')


class SpeedCacheTests(unittest.TestCase):
    REFERENCE = dict(f0=200., centroid=1300., hnr=16., hnr_method='praat', voiced_frames=90)

    def test_a_one_percent_slowdown_uses_its_own_pitch_track_even_inside_the_echo_window(self):
        # Eight seconds become 8.081 s: the old length-only check accepted it
        # as a possible echo tail. The explicit speed proves it was stretched.
        meter = types.SimpleNamespace(score=lambda path: self.fail('reused shifted voiced frames'))
        with patch.object(optimizer, 'timbre_features', return_value=self.REFERENCE) as full:
            result = comparator._timbre_candidate_score(self.REFERENCE, meter, 'stretched.wav',
                dict(speed=.99), dict(speed=1.))
        self.assertEqual(result, 1.)
        full.assert_called_once_with('stretched.wav', hnr_method='praat')

    def test_unchanged_speed_keeps_the_fast_meter_for_colour_changes(self):
        meter = types.SimpleNamespace(score=lambda path: .72)
        with patch.object(optimizer, 'timbre_features', side_effect=AssertionError('unneeded pitch tracking')):
            self.assertEqual(comparator._timbre_candidate_score(self.REFERENCE, meter, 'eq.wav',
                dict(speed=.9, eq_high=-4.), dict(speed=.9)), .72)

    def test_a_stretched_render_without_measurable_voice_is_not_scored_as_the_baseline(self):
        meter = types.SimpleNamespace(score=lambda path: self.fail('used baseline pitch'))
        with patch.object(optimizer, 'timbre_features', return_value=None):
            self.assertIsNone(comparator._timbre_candidate_score(self.REFERENCE, meter, 'bad.wav',
                dict(speed=1.01), dict(speed=1.)))

    def test_real_comparator_routes_speed_changes_to_full_measurement_on_fit_and_checking_texts(self):
        # Every file in this harness has the same length. Length therefore
        # cannot detect the changed speed: the actual render blocks must steer
        # the measure on both the trials and the two checking sentences.
        helper = tone_tests.ComparatorStageTests('runTest')
        last, cached_shifted, full_shifted = [], [], []
        reference = self.REFERENCE

        def identity(block):
            last[:] = [block]
            return .7

        def features(path, sr_target=22050, hnr_method=None):
            if Path(path).name == 'reference.wav':
                return dict(reference, centroid=1600.)
            if last and last[0].get('speed') != 1.:
                full_shifted.append((Path(path).name, last[0]['speed']))
                return dict(reference)
            return dict(reference, centroid=2200.)

        class Meter:
            def __init__(self, ref, baseline_path, sr_target=22050):
                self.speed = 1.  # the configured tone baseline in this run

            def score(self, path):
                speed = last[0].get('speed', 1.)
                if speed != self.speed:
                    cached_shifted.append((Path(path).name, speed))
                # A small preference for .85 makes the old implementation
                # reach its checking texts, where it repeats the wrong reuse.
                return .75 if speed == .85 else 1. / 3.

        with patch.object(optimizer, 'timbre_features', side_effect=features), \
             patch.object(optimizer, 'timbre_features_array', return_value=reference), \
             patch.object(optimizer, 'TimbreMeter', Meter):
            log, _, report = helper.run_main(
                '[1, FR, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]',
                ['--optimise-audio', 'nelder', '--optimise-budget', '150',
                 '--objective', 'timbre', '--search-all-fields'],
                identity=identity, return_report=True)
        self.assertNotIn('Audio search failed', log)
        self.assertEqual(cached_shifted, [])
        self.assertTrue(any('holdout_' not in name for name, _ in full_shifted), full_shifted)
        self.assertTrue(any('holdout_' in name for name, _ in full_shifted), full_shifted)
        self.assertTrue(report['validation'])


if __name__ == '__main__':
    unittest.main()

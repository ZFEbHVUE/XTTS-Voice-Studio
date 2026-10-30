"""Audio provenance and text/audio alignment regressions for the latest archive."""
import re
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import voice_comparator as comparator
import voice_presets as presets
import test_preset_info_regressions as preset_tests
import test_tone_fit_regressions as tone_tests


class PresetProvenanceTests(unittest.TestCase):
    def test_blend_preset_names_its_own_wav(self):
        helper = preset_tests.PipelineFixture('runTest')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        helper.run_pipeline(('--eq-weighting', 'a+blend'))
        main = presets.get('a')['info']['Files']
        alternate = presets.get('a blend')['info']['Files']
        self.assertTrue(main[0].endswith('/a_pipeline_clone.wav'), main)
        self.assertTrue(alternate[0].endswith('/a_pipeline_clone_blend.wav'), alternate)


class ReferenceDiagnosticTests(unittest.TestCase):
    def test_final_acoustic_table_uses_the_fitted_reference_excerpt(self):
        # Speech outside 0.5--1.5 s is deliberately much louder. The EQ and
        # its diagnostic must both compare against the quiet paired excerpt.
        load = comparator.load_mono

        def changed_reference(path, target_sr=None):
            y, sr = load(path, target_sr=target_sr)
            if Path(path).name == 'reference.wav':
                y = y.copy()
                y[:round(.5 * sr)] *= 10
                y[round(1.5 * sr):] *= 10
            return y, sr

        helper = tone_tests.ComparatorStageTests('runTest')
        with patch.object(comparator, 'load_mono', side_effect=changed_reference):
            output, _ = helper.run_main(
                '[1, FR, 1, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 1]', [])
        row = re.search(r'^\s*RMS \(dBFS\)\s+(-?[\d.]+)', output, re.M)
        self.assertIsNotNone(row, output)
        expected = 20 * np.log10(.02 / np.sqrt(2))
        self.assertAlmostEqual(float(row.group(1)), expected, delta=.15)
        self.assertIn('reference interval 0.50-1.50 s', output)


class ElisionAlignmentTests(unittest.TestCase):
    @staticmethod
    def segment(tokens):
        return types.SimpleNamespace(words=[
            types.SimpleNamespace(word=text, start=start, end=end)
            for text, start, end in tokens])

    def test_character_limit_does_not_leave_a_split_french_word(self):
        segment = self.segment([
            ('En', 0., .2), ('train', .2, .4), ('pour', .4, .6),
            ('s', .6, .7), ("'adapter", .7, 1.)])
        self.assertEqual(comparator.transcript_excerpt([segment], limit=15),
                         ('En train pour', 0., .6))

    def test_complete_elision_fits_without_asr_token_spaces(self):
        segment = self.segment([
            ('Écoute', 0., 1.), ('l', 1., 1.2), ("'eau.", 1.2, 2.)])
        self.assertEqual(comparator.transcript_excerpt([segment], limit=len("Écoute l'eau.")),
                         ("Écoute l'eau.", 0., 2.))

    def test_split_elision_at_a_segment_boundary_keeps_matching_end_time(self):
        segments = [self.segment([('Il', 0., .2), ('faut', .2, .4), ('s', .4, .5)]),
                    self.segment([("'adapter", .5, .8)])]
        self.assertEqual(comparator.transcript_excerpt(iter(segments), limit=9),
                         ('Il faut', 0., .4))

    def test_apostrophe_as_a_separate_timed_token(self):
        segment = self.segment([
            ('Oui,', 0., .3), ('d', .3, .4), ("'", .4, .5), ('accord.', .5, 1.)])
        self.assertEqual(comparator.transcript_excerpt([segment]),
                         ("Oui, d'accord.", 0., 1.))

    def test_spaced_elision_inside_one_timed_unit_is_joined(self):
        # A zero-duration "s" opens the segment and is grouped with the next
        # token: the unit reads "s 'adapter" and must read "s'adapter".
        text, first, last = comparator.transcript_excerpt(
            [self.segment([('s', 0., 0.), ("'adapter", 0., .6), ('bien', .7, 1.)])], 240)
        self.assertEqual(text, "s'adapter bien")
        self.assertEqual((first, last), (0., 1.))

    def test_an_elision_opening_the_excerpt_keeps_its_prefix_audio(self):
        # Prefix and completion in two segments: the audio starts with the
        # prefix, at 0.4 s, not with the apostrophe word at 0.5 s.
        segments = [self.segment([('s', .4, .5)]), self.segment([("'adapter", .5, .8)])]
        self.assertEqual(comparator.transcript_excerpt(segments, 240), ("s'adapter", .4, .8))

    def test_other_languages_apostrophes_are_left_alone(self):
        text, _, _ = comparator.transcript_excerpt(
            [self.segment([('I', 0., .2), ("'m", .2, .4), ('here', .5, .9)])], 240)
        self.assertEqual(text, "I 'm here")


if __name__ == '__main__':
    unittest.main()

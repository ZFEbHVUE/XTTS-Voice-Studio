"""Regressions from reviewing the 28 October Claude audio-search changes."""
import contextlib
import io
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.io import wavfile
from audio_parameter_search import AudioSpace, EDITORIAL_FIELDS, worst_case_budget
import voice_comparator as comparator
import xtts_pipeline as pipeline
import test_tone_fit_regressions as tone_tests
import test_render_search_regressions as render_tests
import test_clonage_regressions as gui_tests
import test_budget_plan_regressions as budget_tests


class AdoptionToleranceTests(unittest.TestCase):
    BLOCK = '[1, FR, 1, 0, 0, 0, 0, 90, 0, 0, 0, 0, 0, 0, 0, 1]'

    def test_a_diagnostic_screen_does_not_change_the_adoption_decision(self):
        runner = tone_tests.ComparatorStageTests('runTest')
        reports = []
        for screen in ([], ['--screen-audio']):
            with patch.object(comparator, '_halves_tolerance', return_value=.04):
                _, _, report = runner.run_main(self.BLOCK,
                    ['--optimise-audio', 'nelder', '--optimise-budget', '100', *screen],
                    identity=lambda b: .7 + .05*(b['lp'] == 8000), return_report=True)
            reports.append(report)
        self.assertEqual([r['adoption_tolerance'] for r in reports], [.08, .08])
        self.assertEqual([r['adopted'] for r in reports], [False, False])
        self.assertTrue(all(r['candidate_block']['lp'] == 8000 for r in reports))

    def test_a_small_gain_still_has_to_clear_the_existing_minimum(self):
        runner = tone_tests.ComparatorStageTests('runTest')
        with patch.object(comparator, '_halves_tolerance', return_value=.004):
            _, _, report = runner.run_main(self.BLOCK,
                ['--optimise-audio', 'nelder', '--optimise-budget', '100', '--screen-audio'],
                identity=lambda b: .7 + .015*(b['lp'] == 8000), return_report=True)
        self.assertAlmostEqual(report['adoption_tolerance'], .02)
        self.assertFalse(report['adopted'])

    def test_nonfinite_split_scores_cannot_silently_use_the_floor(self):
        for bad in (float('nan'), float('inf'), -float('inf')):
            encoder = Mock()
            encoder.cosine.side_effect = [.7, bad]
            with self.subTest(bad=bad), \
                 patch.object(comparator, 'load_mono', return_value=(np.ones(100), 16000)):
                with self.assertRaisesRegex(ValueError, 'Non-finite split-half'):
                    comparator._halves_tolerance(encoder, np.ones(2), 'render.wav')

    def test_valid_split_scores_produce_the_expected_tolerance(self):
        encoder = Mock()
        encoder.cosine.side_effect = [.6, .68]
        with patch.object(comparator, 'load_mono', return_value=(np.ones(100), 16000)):
            value = comparator._halves_tolerance(encoder, np.ones(2), 'render.wav')
        self.assertAlmostEqual(value, .04)


class EarlyBudgetTests(unittest.TestCase):
    run_pipeline = render_tests.PipelineRecapTests.run_pipeline

    def test_preflight_covers_different_eq_limits_and_held_fields(self):
        for eq in (.0001, .001, .1, .5, 1., 4.):
            block = dict(speed=.95, vol=1., eq_low=eq/3, eq_mid=eq/3, eq_high=eq/3,
                         hp=60, lp=9000, NR=.05, comp=.05, reverb=.05,
                         noise_gate=-55, pan=.25, limiter=1, **{'de-ess': .05})
            for freeze in (False, True):
                for held in ({}, EDITORIAL_FIELDS):
                    with self.subTest(eq=eq, freeze=freeze, held=tuple(held)):
                        space = AudioSpace(block, eq, freeze, held)
                        actual = 1 + sum(len(v)-1 for v in space.grid().values()) + len(space.keys)
                        self.assertGreaterEqual(worst_case_budget(eq, freeze, held), actual)

    def test_a_start_outside_a_tiny_eq_range_is_covered_too(self):
        # A block fitted with the default 4 dB limit, searched with a tiny
        # one: its EQ values are off the grid, three more trials per field.
        for eq in (.0001, .001):
            block = dict(speed=.95, vol=1., eq_low=.5, eq_mid=-.5, eq_high=.5,
                         hp=60, lp=9000, NR=.05, comp=.05, reverb=.05,
                         noise_gate=-55, pan=.25, limiter=1, **{'de-ess': .05})
            with self.subTest(eq=eq):
                space = AudioSpace(block, eq, False, {})
                actual = 1 + sum(len(v)-1 for v in space.grid().values()) + len(space.keys)
                self.assertGreaterEqual(worst_case_budget(eq, False, {}), actual)

    def test_an_insufficient_full_budget_fails_before_any_pipeline_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp)/'ref.wav'
            ref.touch()
            error = io.StringIO()
            argv = ['xtts_pipeline.py', '--voice', str(ref), 'FR', '--optimise-audio', 'nelder',
                    '--search-all-fields', '--optimise-budget', '64']
            with patch.object(sys, 'argv', argv), patch.object(pipeline, 'run_stream') as run, \
                 contextlib.redirect_stderr(error):
                with self.assertRaises(SystemExit) as stopped:
                    pipeline.main()
            self.assertEqual(stopped.exception.code, 2)
            run.assert_not_called()
            self.assertIn('at least 68', error.getvalue())

    def test_target_volume_reduces_the_full_search_budget_requirement(self):
        def reply(cmd):
            return 0, 'Next [] : [1, FR, 1, 0, 0, 0, 0, 95, 0, 0, 0, 0, 0, 0, 0, 1]'
        with tempfile.TemporaryDirectory() as tmp:
            commands, _, _ = self.run_pipeline(tmp,
                ['--optimise-audio', 'nelder', '--search-all-fields',
                 '--optimise-budget', '64', '--target-dbfs', '-20'], reply)
        self.assertIn('--search-all-fields', commands[0])


class FullSearchGuiTests(unittest.TestCase):
    def comparator_ui(self):
        checks = {}
        original = gui_tests.FakeWidget
        class Widget(original):
            def __init__(self, parent=None, **kw):
                super().__init__(parent, **kw)
                if 'variable' in kw:
                    checks[kw.get('text', '')] = kw['variable']
        with patch.object(gui_tests, 'FakeWidget', Widget), \
             patch.object(gui_tests, 'FakeVar', budget_tests.Var):
            helper = gui_tests.GuiTests('runTest')
            ns, targets, commands, callbacks = helper.namespace('tab_comparator')
            fields = dict(Widget.entries)
        ns['math'] = math
        targets['cmp_reference'].set('/ref.wav')
        targets['cmp_xtts'].set('{1, 0}')
        targets['cmp_audio'].set('[1, FR, 1]')
        return fields, checks, commands, callbacks

    def test_full_search_is_reachable_from_the_comparator_gui(self):
        fields, checks, commands, callbacks = self.comparator_ui()
        self.assertEqual(fields['Audio search'].get(), 'none')
        checks['Include speed, pan and limiter'].set(True)
        self.assertEqual(fields['Audio search'].get(), 'nelder')
        callbacks['>  Compare & Optimise'](None)
        self.assertIn('--search-all-fields', commands[0])
        os.unlink(commands[0][commands[0].index('--text-file')+1])

    def test_full_search_is_reachable_from_the_pipeline_gui(self):
        helper = budget_tests.AutoTabTests('runTest')
        fields, checks, _ = helper.build()
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp)/'ref.wav'
            wavfile.write(ref, 16000, np.zeros(16000, np.float32))
            fields['V1'].set(str(ref))
            checks['Include speed, pan and limiter'].set(True)
            self.assertEqual(fields['Audio search'].get(), 'nelder')
            fields['Audio search'].set('de')
            helper.run_pipeline()
            self.assertIn('--search-all-fields', helper.commands[-1])
            self.assertEqual(helper.commands[-1][helper.commands[-1].index('--optimise-audio')+1], 'de')
            fields['Audio search'].set('none')
            helper.run_pipeline()
            self.assertNotIn('--search-all-fields', helper.commands[-1])

    def test_pipeline_gui_checks_the_full_budget_before_starting(self):
        helper = budget_tests.AutoTabTests('runTest')
        fields, checks, _ = helper.build()
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp)/'ref.wav'
            wavfile.write(ref, 16000, np.zeros(16000, np.float32))
            fields['V1'].set(str(ref))
            checks['Include speed, pan and limiter'].set(True)
            fields['Audio search budget (per EQ fit)'].set('64')
            fields['Target dBFS'].set('')
            started = len(helper.commands)
            helper.run_pipeline()
            self.assertEqual(len(helper.commands), started)
            self.assertIn('>= 68', helper.logs[-1])
            # With a target level the gain is computed: 64 is enough.
            fields['Target dBFS'].set('-20')
            helper.run_pipeline()
            self.assertEqual(len(helper.commands), started + 1)

    def test_comparator_checks_the_full_budget_before_creating_a_job(self):
        fields, checks, commands, callbacks = self.comparator_ui()
        checks['Include speed, pan and limiter'].set(True)
        fields['Audio budget'].set('64')
        callbacks['>  Compare & Optimise'](None)
        self.assertEqual(commands, [])
        # A target gain removes one search axis and its pruning reserve.
        fields['Target dBFS (optional)'].set('-20')
        callbacks['>  Compare & Optimise'](None)
        self.assertEqual(len(commands), 1)
        os.unlink(commands[0][commands[0].index('--text-file')+1])


if __name__ == '__main__':
    unittest.main()

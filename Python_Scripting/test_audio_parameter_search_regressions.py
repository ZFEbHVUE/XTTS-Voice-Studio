"""Search coverage, real switch activation, budgets and rendered RMS contracts."""
import contextlib
import hashlib
import io
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
from scipy.io import wavfile

from audio_parameter_search import (AUDIO_KEYS, EDITORIAL_FIELDS, AudioSpace,
                                    search_audio, worst_case_budget)
import voice_comparator as comparator
import test_render_search_regressions as render_tests
import test_clonage_regressions as gui_tests
import test_budget_plan_regressions as budget_tests


class AllAudioFieldTests(unittest.TestCase):
    def test_every_numeric_field_gets_an_explicit_nonbaseline_render(self):
        seen = []
        def evaluate(block):
            seen.append(dict(block))
            return .7, block
        best, report = search_audio({}, evaluate, budget=100)
        self.assertEqual(tuple(report['tested_values']), AUDIO_KEYS)
        for key in AUDIO_KEYS:
            self.assertGreater(len({b[key] for b in seen}), 1, key)
        self.assertTrue(report['all_free_fields_tried'])
        self.assertTrue(report['all_free_fields_scored'])
        # A flat metric keeps the exact baseline, rather than inventing changes.
        self.assertEqual(best, report['baseline_block'])

    def test_off_filters_gate_and_limiter_round_trip_without_clipping_to_active(self):
        space = AudioSpace({})
        for values in ({'hp': 0, 'lp': 0, 'noise_gate': 0, 'limiter': 1},
                       {'hp': 40, 'lp': 6000, 'noise_gate': -60, 'limiter': 0},
                       {'hp': 150, 'lp': 16000, 'noise_gate': -25, 'limiter': 1}):
            block = dict(space.initial, **values)
            actual = space.decode(space.encode(block))
            for key, expected in values.items():
                self.assertAlmostEqual(actual[key], expected, places=3)

    def test_nelder_can_activate_initially_zero_fields_and_combine_changes(self):
        def evaluate(block):
            return .5 + .1 * (block['lp'] == 8000) + .1 * (block['NR'] == .3), block
        best, report = search_audio({}, evaluate, budget=100)
        self.assertEqual((best['lp'], best['NR']), (8000, .3))
        self.assertAlmostEqual(report['best_score'], .7)

    def test_both_methods_obey_small_real_evaluation_budgets(self):
        # 55 = baseline + 40 grid trials + one pruning evaluation per field (14).
        with self.assertRaisesRegex(ValueError, 'at least 55'):
            search_audio({}, lambda b: (.7, b), budget=54)
        for method in ('nelder', 'de'):
            for budget in (55, 64, 79):
                seen = []
                def evaluate(block):
                    seen.append(block)
                    return .7 + .05 * math.sin(block['eq_low']), block
                with self.subTest(method=method, budget=budget):
                    _, report = search_audio({}, evaluate, method=method, budget=budget)
                    self.assertEqual(report['evaluations'], len(seen))
                    self.assertLessEqual(len(seen), budget)

    def test_insufficient_budget_fails_before_any_evaluation(self):
        seen = []
        with self.assertRaisesRegex(ValueError, 'try every free field'):
            search_audio({}, lambda b: seen.append(b), budget=16)
        self.assertEqual(seen, [])

    def test_target_volume_is_a_derived_quantity_and_other_fields_remain_free(self):
        def evaluate(block):
            rendered = dict(block, vol=2.125)
            return .7 + .1 * (block['reverb'] == .25), rendered
        best, report = search_audio({}, evaluate, freeze_vol=True, budget=100)
        self.assertNotIn('vol', report['tested_values'])
        self.assertIn('target-dbfs', report['held']['vol'])
        self.assertEqual(len(report['tested_values']), 13)
        self.assertEqual((best['vol'], best['reverb']), (2.125, .25))

    def test_a_failed_effect_is_recorded_without_being_selected(self):
        def evaluate(block):
            if block['reverb']:
                raise RuntimeError('ffmpeg reverb unavailable')
            return .7, block
        best, report = search_audio({}, evaluate, budget=80)
        self.assertEqual(best['reverb'], 0)
        self.assertTrue(report['failures'])
        self.assertIn('ffmpeg', report['failures'][0]['error'])
        self.assertFalse(report['all_free_fields_scored'])
        self.assertEqual(report['fields_without_scored_alternative'], ['reverb'])

    def test_an_unscorable_baseline_aborts_instead_of_becoming_a_winner(self):
        with self.assertRaisesRegex(RuntimeError, 'no encoder'):
            search_audio({}, lambda b: (_ for _ in ()).throw(RuntimeError('no encoder')), budget=64)


def render_noise(block, salt=0, sigma=3e-4):
    """Stand-in for the score difference between two renders of one clip:
    tiny, deterministic, different for every distinct block."""
    key = repr((salt, tuple(round(float(block[k]), 3) for k in AUDIO_KEYS)))
    u = (int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) + .5) / 16 ** 12
    v = (int(hashlib.sha256((key + '/').encode()).hexdigest()[:12], 16) + .5) / 16 ** 12
    return sigma * math.sqrt(-2 * math.log(u)) * math.cos(2 * math.pi * v)


# A tone-fitted block with a meditation pace, as the Analyser emits one.
TONE_FITTED = {'speed': .9, 'vol': 0., 'eq_low': -2.9, 'eq_mid': -4., 'eq_high': -4.,
               'hp': 75, 'lp': 0, 'NR': 0., 'comp': .2, 'de-ess': .15, 'reverb': 0.,
               'noise_gate': 0., 'pan': 0., 'limiter': 1}


def eq_mid_only(block, salt=0):
    """Only eq_mid moves the score for real: +.036 from -4 to -1."""
    return .672 - .004 * (block['eq_mid'] + 1) ** 2 + render_noise(block, salt)


def changed(block, start):
    return {k for k in AUDIO_KEYS if block[k] != start[k]}


class PruningTests(unittest.TestCase):
    def test_a_gain_elsewhere_does_not_carry_inert_fields_into_the_block(self):
        # Without pruning, the whole block rode along with eq_mid: over 20
        # draws, speed, pan, reverb and NR moved every time, the pace from 0.9
        # to anywhere in 0.76-1.23 and the pan up to 0.96, for render noise.
        for salt in range(3):
            with self.subTest(salt=salt):
                best, report = search_audio(TONE_FITTED, lambda b: (eq_mid_only(b, salt), b),
                                            budget=400, freeze_vol=True)
                start = report['baseline_block']
                self.assertGreater(len(changed(report['search_best_block'], start)), 3)
                self.assertEqual(changed(best, start), {'eq_mid'})
                self.assertAlmostEqual(best['eq_mid'], -1, delta=.5)
                self.assertGreaterEqual(report['best_score'], report['search_best_score']
                                        - report['pruning']['tolerance'])

    def test_the_total_loss_is_one_tolerance_not_one_per_field(self):
        # Five changes worth .003 each under a .004 tolerance: each revert
        # alone looks free, but reverting all of them would cost .015.
        good = {'eq_low': 4., 'NR': .3, 'comp': .4, 'de-ess': .3, 'reverb': .1}
        def evaluate(block):
            return .7 + .003 * sum(block[k] == v for k, v in good.items()), block
        best, report = search_audio({}, evaluate, budget=200, tolerance=.004)
        self.assertAlmostEqual(report['search_best_score'], .715)
        self.assertGreaterEqual(report['best_score'], .715 - .004 - 1e-12)
        self.assertEqual(report['pruning']['reverted'], ['eq_low'])
        self.assertEqual(changed(best, report['baseline_block']), set(good) - {'eq_low'})

    def test_pruning_stays_within_the_strict_budget(self):
        space = AudioSpace(TONE_FITTED)
        minimum = 1 + sum(len(v) - 1 for v in space.grid().values()) + len(space.keys)
        with self.assertRaisesRegex(ValueError, f'at least {minimum}'):
            search_audio(TONE_FITTED, lambda b: (eq_mid_only(b), b), budget=minimum - 1)
        for budget in (minimum, minimum + 12, 120):
            seen = []
            def evaluate(block):
                seen.append(block)
                return eq_mid_only(block), block
            with self.subTest(budget=budget):
                _, report = search_audio(TONE_FITTED, evaluate, budget=budget)
                self.assertLessEqual(len(seen), budget)
                self.assertEqual(report['evaluations'], len(seen))
                self.assertTrue(all(step['decision'] in ('reverted', 'kept')
                                    for step in report['pruning']['steps']))

    def test_held_fields_keep_their_value_in_every_render(self):
        seen = []
        def evaluate(block):
            # A metric that would take the pace to 1.0 and the voice hard right.
            seen.append(block)
            return .7 + .05 * (block['speed'] == 1) + .05 * (block['pan'] == 1), block
        best, report = search_audio(TONE_FITTED, evaluate, budget=200, held=EDITORIAL_FIELDS)
        self.assertEqual({(b['speed'], b['pan'], b['limiter']) for b in seen}, {(.9, 0, 1)})
        self.assertEqual((best['speed'], best['pan'], best['limiter']), (.9, 0, 1))
        self.assertEqual(set(report['held']), set(EDITORIAL_FIELDS))
        self.assertFalse(set(EDITORIAL_FIELDS) & set(report['tested_values']))

    def test_an_unknown_held_field_is_refused(self):
        with self.assertRaisesRegex(ValueError, 'tempo'):
            AudioSpace({}, held={'tempo': 'typo'})

    def test_the_64_floor_of_the_gui_and_pipeline_covers_the_default_search(self):
        for freeze_vol in (False, True):
            self.assertLessEqual(worst_case_budget(4., freeze_vol, EDITORIAL_FIELDS), 64)
        # The comparator's exact check: no start block needs more than the bound.
        for block in ({}, TONE_FITTED, {'speed': .95, 'vol': 1., 'lp': 9000, 'hp': 60}):
            for held in ({}, EDITORIAL_FIELDS):
                space = AudioSpace(block, 4., False, held)
                need = 1 + sum(len(v) - 1 for v in space.grid().values()) + len(space.keys)
                self.assertLessEqual(need, worst_case_budget(4., False, held))


class RenderedLevelTests(unittest.TestCase):
    def test_effect_gain_and_stereo_are_measured_before_level_correction(self):
        rate = 24000
        samples = .1 * np.sin(2*np.pi*200*np.arange(rate)/rate)
        # Deliberately unequal channels: their power, not their downmix, sets RMS.
        samples = np.column_stack([samples, samples*.25])
        with tempfile.TemporaryDirectory() as tmp:
            raw, out = str(Path(tmp)/'raw.wav'), str(Path(tmp)/'out.wav')
            wavfile.write(raw, rate, samples.astype(np.float32))
            def read(path, **kw):
                sr, audio = wavfile.read(path)
                return audio, sr
            def post(src, dst, block, *args):
                sr, audio = wavfile.read(src)
                # Reverb/compression/etc can alter gain before the volume step.
                gain = 5.25 + block.get('vol', 0)
                wavfile.write(dst, sr, (audio*10**(gain/20)).astype(np.float32))
            with patch.dict(sys.modules, {'soundfile': types.SimpleNamespace(read=read)}), \
                 patch.object(comparator, '_quiet_post', side_effect=post):
                result = comparator.fit_audio_level(raw, out, {'vol': 0}, {}, 'FR', '', -20)
                actual, _ = read(out)
            rms = 20*np.log10(np.sqrt(np.mean(actual.astype(float)**2)))
            self.assertAlmostEqual(rms, -20, delta=.01)
            self.assertNotEqual(result['vol'], 0)

    def test_silent_render_cannot_meet_an_rms_target(self):
        sf = types.SimpleNamespace(read=lambda *a, **k: (np.zeros(24000), 24000))
        with patch.dict(sys.modules, {'soundfile': sf}), patch.object(comparator, '_quiet_post'):
            with self.assertRaisesRegex(ValueError, 'silent'):
                comparator.fit_audio_level('in', 'out', {}, {}, 'FR', '', -20)

    def test_swallowed_generator_effect_errors_are_not_successful_trials(self):
        def post(*a, **kw):
            print('[!] Noise reduction error: ffmpeg failed')
        with patch.object(comparator, 'apply_post', side_effect=post):
            with self.assertRaisesRegex(RuntimeError, 'effect failed'):
                comparator._quiet_post('in', 'out', {}, {}, 'FR', '')

    def test_eq_and_limiter_failures_are_not_successful_trials_either(self):
        # The generator's own wording; both return the audio unprocessed.
        for message in ('  [!]  EQ error: Command ffmpeg returned non-zero exit status 1.',
                        '  [!]  Limiter error: Command ffmpeg returned non-zero exit status 1.'):
            def post(*a, **kw):
                print(message)
            with self.subTest(message=message), \
                 patch.object(comparator, 'apply_post', side_effect=post):
                with self.assertRaisesRegex(RuntimeError, 'effect failed'):
                    comparator._quiet_post('in', 'out', {}, {}, 'FR', '')

    def test_two_holdouts_and_final_check_are_disjoint(self):
        fit = 'Un texte de réglage.'
        holdouts = comparator.audio_holdouts('FR', fit)
        unseen = comparator.unseen_sentence('FR', fit, holdouts)
        self.assertEqual(len(set(holdouts)), 2)
        self.assertNotIn(unseen, [fit, *holdouts])


class PipelineAudioBudgetTests(unittest.TestCase):
    run_pipeline = render_tests.PipelineRecapTests.run_pipeline

    def test_audio_budget_and_method_reach_every_eq_alternative(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            def reply(cmd):
                block = '[1, FR, 1, 0, 0, 0, 0, 95, 0, 0, 0, 0, 0, 0, 0, 1]'
                fit = json.dumps({'text': 'Un test.', 'interval': [0, 1]})
                return 0, '[*] Fit input: ' + fit + '\nNext [] : ' + block
            commands, _, _ = self.run_pipeline(tmp,
                ['--optimise-audio', 'de', '--optimise-budget', '123', '--eq-weighting', 'a+blend'], reply)
        self.assertEqual(len(commands), 2)
        for cmd in commands:
            self.assertEqual(cmd[cmd.index('--optimise-audio')+1], 'de')
            self.assertEqual(cmd[cmd.index('--optimise-budget')+1], '123')

    def test_the_search_is_opt_in_and_all_fields_are_forwarded(self):
        def reply(cmd):
            return 0, 'Next []    : [1, FR, 1, 0, 0, 0, 0, 95, 0, 0, 0, 0, 0, 0, 0, 1]'
        with tempfile.TemporaryDirectory() as tmp:
            commands, _, _ = self.run_pipeline(tmp, [], reply)
        self.assertNotIn('--optimise-audio', commands[0])
        with tempfile.TemporaryDirectory() as tmp:
            commands, _, _ = self.run_pipeline(
                tmp, ['--optimise-audio', 'nelder', '--search-all-fields'], reply)
        self.assertIn('--search-all-fields', commands[0])

    def test_a_failed_search_does_not_stop_the_pipeline(self):
        import json
        def reply(cmd):
            block = '[1, FR, 1, 0, 0, 0, 0, 95, 0, 0, 0, 0, 0, 0, 0, 1]'
            fit = json.dumps({'text': 'Un test.', 'interval': [0, 1]})
            return 0, ('[*] Fit input: ' + fit + '\n  [!] Audio search failed: encoder lost\n'
                       '      The tone-fitted block is kept; details in the audio-search report.\n'
                       '  Next []    : ' + block + '\n'
                       '[*] Audio-search report: /out/v1_audio_search.json\n[OK] Done.\n')
        with tempfile.TemporaryDirectory() as tmp:
            commands, saves, recap = self.run_pipeline(
                tmp, ['--optimise-audio', 'nelder', '--eq-weighting', 'a+blend'], reply)
        self.assertEqual(len(commands), 2)            # the 'blend' fit still ran
        self.assertIn('PIPELINE COMPLETE', recap)
        self.assertIn("[!] Audio search failed ('a' fit)", recap)
        self.assertIn("[!] Audio search failed ('blend' fit)", recap)
        self.assertIn('/out/v1_audio_search.json', recap)
        self.assertEqual(len(saves), 2)               # the voice and its 'blend' A/B


class ComparatorGuiAudioTests(unittest.TestCase):
    namespace = gui_tests.GuiTests.namespace

    def test_method_budget_and_target_are_reachable_from_comparator_gui(self):
        import os
        ns, targets, commands, callbacks = self.namespace('tab_comparator')
        ns['math'] = math
        targets['cmp_reference'].set('/ref.wav')
        targets['cmp_xtts'].set('{1, 0}')
        targets['cmp_audio'].set('[1, FR, 1]')
        fields = gui_tests.FakeWidget.entries
        # Opt-in, as in [Auto]: hundreds of renders whose result has to be heard.
        self.assertEqual(fields['Audio search'].get(), 'none')
        fields['Audio search'].set('nelder')
        fields['Audio budget'].set('123')
        fields['Target dBFS (optional)'].set('-20')
        callbacks['>  Compare & Optimise'](None)
        cmd = commands[0]
        self.assertEqual(cmd[cmd.index('--optimise-audio')+1], 'nelder')
        self.assertEqual(cmd[cmd.index('--optimise-budget')+1], '123')
        self.assertEqual(cmd[cmd.index('--target-dbfs')+1], '-20')
        os.unlink(cmd[cmd.index('--text-file')+1])

    def test_pipeline_gui_defaults_to_no_search_and_passes_method_and_budget(self):
        harness = budget_tests.AutoTabTests('runTest')
        fields, _, _ = harness.build()
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp)/'ref.wav'
            wavfile.write(ref, 16000, np.zeros(16000, np.float32))
            fields['V1'].set(str(ref))
            harness.run_pipeline()
            cmd = harness.commands[-1]
            self.assertEqual(cmd[cmd.index('--optimise-audio')+1], 'none')
            self.assertNotIn('--optimise-budget', cmd)
            fields['Audio search'].set('de')
            fields['Audio search budget (per EQ fit)'].set('123')
            harness.run_pipeline()
            cmd = harness.commands[-1]
            self.assertEqual(cmd[cmd.index('--optimise-audio')+1], 'de')
            self.assertEqual(cmd[cmd.index('--optimise-budget')+1], '123')


if __name__ == '__main__':
    unittest.main()

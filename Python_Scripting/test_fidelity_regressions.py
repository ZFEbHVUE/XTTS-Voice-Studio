"""New fidelity behaviour: real signal processing, model/ASR stand-ins where noted."""
import ast
import contextlib
import io
import json
import math
import os
import re
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

import audio_parameter_search as APS
import voice_fidelity as fidelity
import fidelity_search
from fidelity_search import optimise_fidelity
from joint_sampling import run_joint_search, sampling_candidates
import xtts_settings as settings
import xtts_clone as clone
import voice_comparator as comparator
import test_tone_fit_regressions as tone_tests
import test_render_search_regressions as render_tests
import test_budget_plan_regressions as budget_tests
import test_audio_search_review_regressions as gui_tests
import test_clonage_regressions as clone_tests


def asr(wer=0., score=.95):
    return dict(wer=wer, score=score, detected='fr', text='Une phrase complète.')


def validation_rows(gain=.04, identity_gain=0., count=3):
    return [dict(text=f'Phrase distincte {i}',
        base_metrics=dict(score=.65, identity=.7, clipped_fraction=0.),
        candidate_metrics=dict(score=.65+gain, identity=.7+identity_gain, clipped_fraction=0.),
        base_asr=asr(), candidate_asr=asr()) for i in range(count)]


class AcousticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sr = 16000
        cls.voice = tone_tests.harmonic_voice(cls.sr, 3, 205, .2, noise_db=-45)
        cls.profile = fidelity.acoustic_profile(cls.voice, cls.sr)

    def test_volume_cannot_game_the_acoustic_score(self):
        for gain in (.2, .5, 1.5):
            measured = fidelity.acoustic_profile(self.voice*gain, self.sr)
            result = fidelity.compare_profiles(self.profile, measured, .7)
            self.assertAlmostEqual(result['acoustic_score'], 1., places=7)

    def test_pitch_noise_and_brightness_are_measured_even_with_flat_identity(self):
        variants = [tone_tests.harmonic_voice(self.sr, 3, 235, .2, noise_db=-45),
            self.voice+np.random.default_rng(8).normal(0, .01, len(self.voice)),
            tone_tests.exact_chain(self.voice, dict(eq_high=4), fs=self.sr, ratio=(1, 1))]
        for variant in variants:
            result = fidelity.compare_profiles(self.profile,
                fidelity.acoustic_profile(variant, self.sr), .7)
            self.assertLess(result['acoustic_score'], .99)
            self.assertEqual(result['identity'], .7)

    def test_silence_nonfinite_and_unvoiced_noise_are_not_perfect_matches(self):
        for audio in (np.zeros(16000), np.full(16000, np.nan),
                      np.random.default_rng(4).normal(0, .1, 16000)):
            with self.assertRaises(ValueError):
                fidelity.acoustic_profile(audio, 16000)


class AcousticScoreTests(unittest.TestCase):
    """The distance, part by part, on hand-made profiles: each part has the weight and the
    scale the documentation gives, counts in both directions, and the score is half identity."""

    @staticmethod
    def profile(**fields):
        base = dict(envelope=np.zeros(24), pitch_st=0., periodicity_db=10., sibilance_db=-20.,
                    dynamics_db=12., floor_db=-45., clipped_fraction=0.)
        return dict(base, **fields)

    PARTS = (('envelope', .45, 3.), ('pitch', .15, 2.), ('harmonicity', .15, 3.),
             ('sibilance', .10, 3.), ('dynamics', .10, 3.), ('floor', .05, 6.))
    ENTRY = dict(envelope='envelope', pitch='pitch_st', harmonicity='periodicity_db',
                 sibilance='sibilance_db', dynamics='dynamics_db', floor='floor_db')

    def moved(self, part, amount):
        reference = self.profile()
        if part == 'envelope':
            return reference, self.profile(envelope=np.full(24, amount))
        return reference, self.profile(**{self.ENTRY[part]: reference[self.ENTRY[part]]+amount})

    def test_the_weights_and_scales_are_the_documented_ones(self):
        self.assertEqual({k: (w, c) for k, w, c in self.PARTS},
                         {k: (fidelity.WEIGHTS[k], fidelity.SCALES[k]) for k in fidelity.WEIGHTS})
        self.assertAlmostEqual(sum(fidelity.WEIGHTS.values()), 1., places=12)

    def test_each_part_counts_by_its_weight_at_its_scale_in_both_directions(self):
        for part, weight, scale in self.PARTS:
            for sign in (1, -1):
                with self.subTest(part=part, sign=sign):
                    reference, candidate = self.moved(part, sign*scale)
                    result = fidelity.compare_profiles(reference, candidate, .7)
                    distance = weight*(2**.5-1)               # sqrt(1 + 1) - 1 at one scale
                    self.assertAlmostEqual(result['distance'], distance, places=9)
                    self.assertAlmostEqual(result['acoustic_score'], 1/(1+distance), places=9)
                    self.assertAlmostEqual(result['gaps'][part], scale, places=9)
                    others = [v for k, v in result['gaps'].items() if k != part]
                    self.assertEqual(others, [0.]*5)

    def test_the_score_is_half_identity_and_half_acoustic(self):
        reference, candidate = self.moved('pitch', 2.)
        acoustic = fidelity.compare_profiles(reference, candidate, .7)['acoustic_score']
        for identity in (.4, .7, .9):
            result = fidelity.compare_profiles(reference, candidate, identity)
            self.assertAlmostEqual(result['score'], .5*identity+.5*acoustic, places=12)
            self.assertEqual(result['identity'], identity)
        same = fidelity.compare_profiles(reference, reference, .8)
        self.assertEqual((same['acoustic_score'], same['distance']), (1., 0.))
        self.assertAlmostEqual(same['score'], .9, places=12)

    def test_the_clipping_of_the_candidate_is_handed_on_not_the_references(self):
        reference = self.profile(clipped_fraction=.5)
        result = fidelity.compare_profiles(reference, self.profile(clipped_fraction=.003), .7)
        self.assertEqual(result['clipped_fraction'], .003)

    def test_incompatible_or_non_finite_profiles_are_refused(self):
        good = self.profile()
        for bad in (self.profile(envelope=np.zeros(5)), self.profile(envelope=np.full(24, np.nan)),
                    self.profile(pitch_st=float('inf'))):
            with self.subTest(bad=str(sorted(bad))), self.assertRaises(ValueError):
                fidelity.compare_profiles(good, bad, .7)
        with self.assertRaises(ValueError):
            fidelity.compare_profiles(good, good, float('nan'))


class PairedGuardTests(unittest.TestCase):
    def test_final_text_is_disjoint_from_native_and_audio_selection_texts(self):
        from probe_texts import probe_texts,holdout_texts,final_check_texts
        native=probe_texts('fr',3)+holdout_texts('fr',2)
        final=comparator.unseen_sentence('FR','Texte original.',native,
            excluded_texts=native,dedicated=True)
        self.assertIn(final,final_check_texts('fr'))
        self.assertNotIn(final,native)
        with self.assertRaises(ValueError):
            comparator.unseen_sentence('FR','Texte original.',[],native[0],excluded_texts=native,dedicated=True)

    def test_two_custom_selection_texts_are_supplemented_before_models_load(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'sentences.txt'
            path.write_text('Une première phrase privée.\nUne autre phrase privée.\n',encoding='utf-8')
            chosen=comparator.audio_holdouts('FR','Référence originale.',count=3,text_file=str(path))
        self.assertEqual(len(chosen),3)
        self.assertTrue(chosen[0].startswith('Une première'))
        self.assertEqual(len(set(chosen)),3)
    def test_acoustic_gain_can_win_without_an_identity_gain(self):
        self.assertTrue(fidelity.paired_decision(validation_rows())['adopted'])

    def test_quality_and_identity_regressions_are_rejected(self):
        for fault in ('identity', 'wer', 'asr', 'language', 'clipping', 'nonfinite'):
            rows = validation_rows(gain=.2)
            bad = rows[1]
            if fault == 'identity': bad['candidate_metrics']['identity'] = .5
            if fault == 'wer': bad['candidate_asr']['wer'] = .3
            if fault == 'asr': bad['candidate_asr']['score'] = .4
            if fault == 'language': bad['candidate_asr']['detected'] = 'en'
            if fault == 'clipping': bad['candidate_metrics']['clipped_fraction'] = .01
            if fault == 'nonfinite': bad['candidate_asr']['wer'] = float('nan')
            with self.subTest(fault=fault):
                self.assertFalse(fidelity.paired_decision(rows)['adopted'])

    def test_insufficient_or_inconsistent_evidence_keeps_the_baseline(self):
        self.assertFalse(fidelity.paired_decision(validation_rows(count=2))['adopted'])
        self.assertFalse(fidelity.paired_decision(validation_rows(gain=.001))['adopted'])
        rows = validation_rows(gain=.03)
        rows[-1]['candidate_metrics']['score'] = .4
        self.assertFalse(fidelity.paired_decision(rows)['adopted'])

    def test_duplicate_texts_are_not_three_independent_checks(self):
        rows = validation_rows()
        for row in rows: row['text'] = 'La même phrase.'
        self.assertFalse(fidelity.paired_decision(rows)['adopted'])


class PairedDecisionBoundaryTests(unittest.TestCase):
    """Each number the documentation states about the paired decision, just inside it
    (adopted) and just beyond it (rejected, for that reason alone)."""

    @staticmethod
    def rows(gains=(.04, .04, .04), identity=(0., 0., 0.)):
        rows = validation_rows(count=len(gains))
        for row, gain, shift in zip(rows, gains, identity):
            row['candidate_metrics']['score'] = row['base_metrics']['score']+gain
            row['candidate_metrics']['identity'] = row['base_metrics']['identity']+shift
        return rows

    def verdict(self, rows, **kw):
        decision = fidelity.paired_decision(rows, **kw)
        return decision['adopted'], decision['reasons']

    def check(self, cases, reason):
        """cases: (rows, adopted); a rejection must give `reason` and nothing else."""
        for number, (rows, adopted) in enumerate(cases):
            with self.subTest(case=number, reason=reason):
                self.assertEqual(self.verdict(rows), (adopted, [] if adopted else [reason]))

    def test_the_minimum_gain(self):
        reason = 'paired fidelity gain too small or inconsistent'
        self.check([(self.rows((.006,)*3), True), (self.rows((.004,)*3), False)], reason)
        self.assertEqual(self.verdict(self.rows((.025,)*3), min_gain=.02), (True, []))
        self.assertEqual(self.verdict(self.rows((.015,)*3), min_gain=.02), (False, [reason]))

    def test_the_mean_must_beat_one_paired_standard_error(self):
        steady, noisy = self.rows((.027,)*3), self.rows((.10, -.019, 0.))     # the same mean .027
        self.check([(steady, True), (noisy, False)], 'paired fidelity gain too small or inconsistent')
        decision = fidelity.paired_decision(noisy)
        self.assertAlmostEqual(decision['mean_gain'], .027, places=9)
        self.assertAlmostEqual(decision['paired_stderr'], .03691, places=5)    # sd with n-1, over sqrt(3)

    def test_no_text_may_lose_more_than_two_hundredths(self):
        self.check([(self.rows((.10, .10, -.019)), True), (self.rows((.10, .10, -.021)), False)],
                   'large fidelity regression on one text')

    def test_identity_is_judged_on_each_text_and_on_average(self):
        reason = 'speaker identity decreased'
        self.check([(self.rows(identity=(-.014, .014, 0.)), True),      # one text down, the mean level
                    (self.rows(identity=(-.016, .016, 0.)), False),     # one text beyond -0.015
                    (self.rows(identity=(-.004,)*3), True),
                    (self.rows(identity=(-.009,)*3), False)], reason)   # each inside, the mean beyond -0.005
        self.assertEqual(self.verdict(self.rows(identity=(-.016, .016, 0.)), identity_tolerance=.03), (True, []))

    def test_the_word_error_rate_may_rise_by_two_hundredths(self):
        cases = []
        for before, after, adopted in ((0., .015, True), (0., .025, False), (.10, .115, True), (.10, .125, False)):
            rows = self.rows()
            rows[1]['base_asr']['wer'], rows[1]['candidate_asr']['wer'] = before, after
            cases.append((rows, adopted))
        self.check(cases, 'word error rate increased')

    def test_the_asr_score_may_fall_by_three_hundredths(self):
        cases = []
        for fall, adopted in ((.025, True), (.035, False)):
            rows = self.rows()
            rows[1]['candidate_asr']['score'] = rows[1]['base_asr']['score']-fall
            cases.append((rows, adopted))
        self.check(cases, 'ASR intelligibility decreased')

    def test_a_change_of_detected_language_is_a_rejection(self):
        rows = self.rows()
        rows[2]['candidate_asr']['detected'] = 'en'
        self.check([(rows, False)], 'ASR language changed')

    def test_clipping_may_not_grow_beyond_a_ten_thousandth(self):
        cases = []
        for before, after, adopted in ((0., .00005, True), (0., .0002, False),
                                       (.001, .0010999, True), (.001, .0012, False)):
            rows = self.rows()
            rows[1]['base_metrics']['clipped_fraction'] = before
            rows[1]['candidate_metrics']['clipped_fraction'] = after
            cases.append((rows, adopted))
        self.check(cases, 'clipping increased')

    def test_blank_or_repeated_texts_are_not_independent_checks(self):
        cases = []
        for texts in (['Une.', '', 'Trois.'], ['Une.', '   ', 'Trois.'], ['Une.', 'Deux.', ' UNE. ']):
            rows = self.rows()
            for row, text in zip(rows, texts):
                row['text'] = text
            cases.append((rows, False))
        self.check(cases, 'distinct nonempty validation texts required')

    def test_three_texts_are_the_least_that_can_be_decided_on(self):
        self.check([(self.rows((.04,)*2), False)], 'at least three paired texts required')
        self.assertTrue(self.verdict(self.rows((.04,)*4))[0])

    def test_a_missing_or_non_finite_number_is_a_rejection_not_a_guess(self):
        for place, key, bad, reason in (('candidate_asr', 'wer', None, 'missing or non-finite ASR validation'),
                                        ('base_asr', 'score', None, 'missing or non-finite ASR validation'),
                                        ('base_asr', 'wer', float('inf'), 'missing or non-finite ASR validation'),
                                        ('candidate_metrics', 'score', float('nan'), 'non-finite validation metric'),
                                        ('base_metrics', 'clipped_fraction', float('nan'), 'non-finite validation metric')):
            rows = self.rows()
            rows[0][place][key] = bad
            with self.subTest(place=place, key=key):
                self.assertEqual(self.verdict(rows), (False, [reason]))

    def test_the_decision_reports_what_it_decided_on(self):
        decision = fidelity.paired_decision(self.rows((.10, 0., .05), (.01, -.01, 0.)))
        for key, value in (('mean_gain', .05), ('paired_stderr', .028868), ('minimum_gain', .005),
                           ('identity_tolerance', .015)):
            self.assertAlmostEqual(decision[key], value, places=5, msg=key)
        self.assertEqual([round(g, 6) for g in decision['gains']], [.1, 0., .05])
        self.assertEqual([round(g, 6) for g in decision['identity_gains']], [.01, -.01, 0.])


class FidelitySearchTests(unittest.TestCase):
    @staticmethod
    def measure(block):
        improvement = {.0: 0., 6000.: .12, 8000.: .10}.get(float(block['lp']), 0.)
        acoustic = .6+improvement
        return dict(score=.5*.7+.5*acoustic, identity=.7, acoustic_score=acoustic,
                    clipped_fraction=0.), block

    def test_second_candidate_can_transfer_when_the_fit_winner_fails_asr(self):
        def validate(block, rank):
            rows = validation_rows(gain=.05)
            if block['lp'] == 6000:
                rows[0]['candidate_asr']['wer'] = .3
            return rows, []
        block, report, _ = optimise_fidelity({}, self.measure, validate,
            budget=100, held={'speed':'pace', 'pan':'placement', 'limiter':'protection'})
        self.assertTrue(report['adopted'])
        self.assertEqual(block['lp'], 8000)
        self.assertEqual(report['objective'], 'fidelity_v1')
        self.assertTrue(report['all_free_fields_scored'])
        self.assertLessEqual(report['evaluations'], 100)
        self.assertGreaterEqual(len(report['finalists']), 2)

    def test_no_fidelity_gain_keeps_zero_effects(self):
        def flat(block):
            return dict(score=.7, identity=.7, acoustic_score=.7, clipped_fraction=0.), block
        block, report, _ = optimise_fidelity({}, flat,
            lambda *a: self.fail('No transfer render needed for a flat fit'), budget=100)
        self.assertFalse(report['adopted'])
        self.assertEqual((block['NR'], block['comp'], block['reverb']), (0, 0, 0))

    def test_bad_tone_fit_cannot_protect_itself_as_the_reference_baseline(self):
        def measure(block):
            identity = .5 if block['eq_high'] else .7
            return dict(score=.5*identity+.4, identity=identity, acoustic_score=.8,
                        clipped_fraction=0.), block
        block, report, _ = optimise_fidelity({}, measure,
            lambda *a:self.fail('No improving configuration'), budget=100,
            anchors=[dict(eq_high=4.)])
        self.assertFalse(report['adopted'])
        self.assertEqual(block['eq_high'],0.)
        self.assertEqual(report['baseline_metrics']['identity'],.7)
        self.assertTrue(report['anchor_trials'])


class SearchGuardTests(unittest.TestCase):
    """While the fit is searched, a block that costs identity or clips cannot win on what it
    gains in acoustic score: it is scored below every honest block and never reaches a finalist."""
    ONLY_LP = {key: 'held for the test' for key in APS.AUDIO_KEYS if key != 'lp'}

    @staticmethod
    def measure_with(identity_drop=0., clipped=0.):
        def measure(block):
            if float(block['lp']) == 6000.:           # +0.12 acoustic, at a price
                acoustic, identity, clipping = .72, .7-identity_drop, clipped
            else:
                acoustic, identity, clipping = .6, .7, 0.
            return dict(score=.5*identity+.5*acoustic, identity=identity, acoustic_score=acoustic,
                        clipped_fraction=clipping), block
        return measure

    def reached_the_finalists(self, measure):
        block, report, _ = optimise_fidelity({}, measure, lambda block, rank: (validation_rows(gain=.05), []),
                                             budget=60, held=self.ONLY_LP)
        return any(row['block']['lp'] == 6000. for row in report['finalists']), block, report

    def test_an_identity_loss_beyond_fifteen_thousandths_keeps_a_block_out(self):
        for drop, expected in ((.012, True), (.018, False)):
            with self.subTest(identity_drop=drop):
                reached, block, _ = self.reached_the_finalists(self.measure_with(identity_drop=drop))
                self.assertEqual(reached, expected)
                self.assertEqual(block['lp'], 6000. if expected else 0.)

    def test_clipping_beyond_a_ten_thousandth_keeps_a_block_out(self):
        for clipped, expected in ((.00005, True), (.002, False)):
            with self.subTest(clipped=clipped):
                reached, block, _ = self.reached_the_finalists(self.measure_with(clipped=clipped))
                self.assertEqual(reached, expected)

    def test_the_guards_are_measured_against_the_baseline_block(self):
        # A baseline that already clips a little sets the bar: +0.0001 over it is tolerated.
        def measure(block):
            blocked = float(block['lp']) == 6000.
            return dict(score=.5*.7+.5*(.72 if blocked else .6), identity=.7,
                        acoustic_score=.72 if blocked else .6,
                        clipped_fraction=.0010999 if blocked else .001), block
        reached, _, _ = self.reached_the_finalists(measure)
        self.assertTrue(reached)
        _, _, worse = self.reached_the_finalists(lambda block: (dict(measure(block)[0], clipped_fraction=(
            .0012 if float(block['lp']) == 6000. else .001)), block))
        self.assertFalse(any(row['block']['lp'] == 6000. for row in worse['finalists']))


class FinalistSelectionTests(unittest.TestCase):
    """Who is sent to the other sentences: only a block that beats the baseline by more than half
    the minimum gain (renders are expensive), best fit score first, no block twice, at most
    `finalists` of them; the winner is the largest paired gain among the eligible."""
    ONLY_LP = SearchGuardTests.ONLY_LP

    @staticmethod
    def measure(gains):
        """gains: lp value -> gain in the fit score (half the acoustic gain, identity fixed)."""
        def measure(block):
            acoustic = .6+2*gains.get(float(block['lp']), 0.)
            return dict(score=.5*.7+.5*acoustic, identity=.7, acoustic_score=acoustic,
                        clipped_fraction=0.), block
        return measure

    def search(self, gains, validate=None, **kw):
        checked = []
        def default(block, rank):
            checked.append(block['lp'])
            return validation_rows(gain=.05), []
        block, report, _ = optimise_fidelity({}, self.measure(gains), validate or default,
                                             budget=80, held=self.ONLY_LP, **kw)
        return block, report, checked

    def test_a_gain_under_half_the_minimum_is_not_worth_a_validation(self):
        for gain, expected in ((.002, []), (.004, [6000.])):          # half of 0.005 is 0.0025
            with self.subTest(fit_gain=gain):
                _, report, checked = self.search({6000.: gain})
                self.assertEqual(checked, expected)
                self.assertEqual([row['block']['lp'] for row in report['finalists']], expected)

    def test_a_block_that_does_not_beat_the_baseline_is_never_a_finalist(self):
        _, report, checked = self.search({6000.: -.05, 8000.: -.01})
        self.assertEqual((checked, report['finalists'], report['adopted']), ([], [], False))

    def test_the_best_fit_scores_go_first_and_no_more_than_the_asked_number(self):
        gains = {16000.: .01, 6000.: .05, 8000.: .03, 11000.: .02}      # the values of the lp grid
        for asked, expected in ((1, [6000.]), (2, [6000., 8000.]), (3, [6000., 8000., 11000.])):
            with self.subTest(finalists=asked):
                _, report, checked = self.search(gains, finalists=asked)
                self.assertEqual(checked, expected)
                self.assertEqual([row['rank'] for row in report['finalists']], list(range(1, asked+1)))

    def test_the_winner_is_the_largest_paired_gain_among_the_eligible_not_the_first_or_the_best_fit(self):
        paired = {6000.: .02, 8000.: .06, 11000.: .04}              # the best fit gains least on other sentences
        def validate(block, rank):
            rows = validation_rows(gain=paired[block['lp']])
            if block['lp'] == 8000.:
                rows[0]['candidate_asr']['wer'] = .3                    # the strongest one fails its ASR check
            return rows, []
        block, report, _ = self.search({6000.: .05, 8000.: .04, 11000.: .03}, validate)
        self.assertEqual(block['lp'], 11000.)
        self.assertEqual(report['selected_finalist'], 3)
        self.assertTrue(report['adopted'])
        self.assertEqual([row['decision']['adopted'] for row in report['finalists']], [True, False, True])

    def test_a_validation_that_raises_rejects_that_finalist_and_the_search_goes_on(self):
        def validate(block, rank):
            if block['lp'] == 6000.:
                raise RuntimeError('render failed')
            return validation_rows(gain=.05), []
        block, report, _ = self.search({6000.: .05, 8000.: .03}, validate)
        self.assertEqual(block['lp'], 8000.)
        first = report['finalists'][0]
        self.assertFalse(first['decision']['adopted'])
        self.assertIn('validation failed: render failed', first['decision']['reasons'])
        self.assertEqual(first['validation'], [])

    def test_the_report_counts_the_validation_renders_and_says_what_was_adopted(self):
        block, report, _ = self.search({6000.: .05, 8000.: .03})
        self.assertEqual(report['validation_evaluations'], 2*3*2)      # base + candidate, 3 texts, 2 finalists
        self.assertTrue(report['adopted'])
        self.assertEqual(report['final_block'], block)
        self.assertEqual(report['validation'], report['finalists'][report['selected_finalist']-1]['validation'])
        _, none, _ = self.search({6000.: .05}, lambda block, rank: (validation_rows(gain=.0), []))
        self.assertFalse(none['adopted'])
        self.assertIsNone(none['selected_finalist'])
        self.assertEqual(none['validation'], [])

    def test_the_arguments_are_checked(self):
        measure = self.measure({})
        for kw in (dict(finalists=0), dict(finalists=6), dict(min_gain=0.), dict(min_gain=-.01),
                   dict(min_gain=float('nan'))):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                optimise_fidelity({}, measure, lambda *a: ([], []), budget=60, held=self.ONLY_LP, **kw)
        optimise_fidelity({}, measure, lambda *a: ([], []), budget=60, held=self.ONLY_LP, finalists=5)


# The 3.10 merge of this module with the 3.9 [] search: the tone fit is the baseline,
# the input [] and a neutral block compete against it, the report names its criterion.

HELD = {key: 'held for the test' for key in APS.AUDIO_KEYS if key not in ('eq_high', 'comp')}
TONE = dict(eq_high=2.5, comp=.1)          # off the trial grid: no one-field trial reaches a needle
INPUT = dict(eq_high=-1.5, comp=.3)
NEUTRAL = dict(eq_high=0., comp=0.)


def needles(heights):
    """A flat acoustic score of .6 with spikes at exact (eq_high, comp) points: only a
    block scored as it stands finds them, which is what an anchor is for."""
    def measure(block):
        acoustic = heights.get((block['eq_high'], block['comp']), .6)
        return dict(score=.5*.7+.5*acoustic, identity=.7, acoustic_score=acoustic,
                    clipped_fraction=0.), block
    return measure


class ToneFitBaselineTests(unittest.TestCase):
    ANCHORS = dict(anchors=[dict(INPUT), dict(NEUTRAL)], anchor_labels=['input []', 'neutral []'])

    def run_search(self, measure, validate, **kw):
        return optimise_fidelity(TONE, measure, validate, budget=60, held=HELD, **dict(self.ANCHORS, **kw))

    def test_the_input_and_the_neutral_block_compete_as_finalists_under_their_names(self):
        def validate(block, rank):
            # The block with the lower fit score transfers better to other sentences.
            return validation_rows(gain={0.: .03, -1.5: .05}[block['eq_high']]), []
        block, report, _ = self.run_search(needles({(0., 0.): .72, (-1.5, .3): .68}), validate)
        self.assertEqual([row['source'] for row in report['finalists']],
                         ['anchor: neutral []', 'anchor: input []'])
        self.assertTrue(report['adopted'])
        self.assertEqual(report['selected_finalist'], 2)
        self.assertEqual((block['eq_high'], block['comp']), (-1.5, .3))
        self.assertEqual([row['status'] for row in report['anchor_trials']], ['scored', 'scored'])

    def test_nothing_proven_gives_back_the_tone_fit_not_a_neutral_block(self):
        def validate(block, rank):
            rows = validation_rows(gain=.05)
            rows[0]['candidate_asr']['wer'] = .3
            return rows, []
        block, report, _ = self.run_search(needles({(0., 0.): .72, (-1.5, .3): .68}), validate)
        self.assertFalse(report['adopted'])
        self.assertEqual(len(report['finalists']), 2)                 # they were checked and refused
        self.assertEqual((block['eq_high'], block['comp']), (TONE['eq_high'], TONE['comp']))
        self.assertEqual(report['final_fit_metrics']['acoustic_score'],
                         report['baseline_metrics']['acoustic_score'])
        self.assertEqual(report['baseline_block']['eq_high'], TONE['eq_high'])

    def test_a_flat_landscape_never_replaces_the_tone_fit(self):
        block, report, renders = self.run_search(needles({}),
            lambda *a: self.fail('no finalist, nothing to render'))
        self.assertFalse(report['adopted'])
        self.assertEqual((block['eq_high'], block['comp']), (TONE['eq_high'], TONE['comp']))
        self.assertEqual((renders, report['finalists']), ([], []))

    def test_anchors_the_fit_budget_cannot_pay_for_are_said_not_guessed(self):
        # 10 is the least this space needs: the trials leave nothing for an anchor.
        block, report, _ = optimise_fidelity(TONE, needles({(0., 0.): .72}),
            lambda *a: self.fail('an anchor that was not scored cannot be a finalist'),
            budget=10, held=HELD, anchors=[dict(NEUTRAL)], anchor_labels=['neutral []'])
        self.assertEqual(report['anchor_trials'][0]['status'], 'not tried (fit budget)')
        self.assertFalse(report['adopted'])
        self.assertEqual(block['eq_high'], TONE['eq_high'])

    def test_an_unlabelled_anchor_is_still_named(self):
        block, report, _ = optimise_fidelity(TONE, needles({(0., 0.): .72}),
            lambda block, rank: (validation_rows(), []), budget=60, held=HELD, anchors=[dict(NEUTRAL)])
        self.assertEqual(report['finalists'][0]['source'], 'anchor: explicit block')

    def test_an_anchor_cannot_carry_a_held_field_away(self):
        # speed, pan and limiter stay what the tone fit has, whatever block is scored.
        held = {key: 'held for the test' for key in APS.AUDIO_KEYS if key not in ('eq_high', 'comp')}
        tone = dict(TONE, speed=1.1, pan=.2, limiter=0)
        seen = []
        def measure(block):
            seen.append(dict(block))
            return needles({})(block)
        optimise_fidelity(tone, measure, lambda *a: ([], []), budget=60, held=held,
                          anchors=[dict(NEUTRAL, speed=.8, pan=-1., limiter=1)])
        self.assertEqual({(b['speed'], b['pan'], b['limiter']) for b in seen}, {(1.1, .2, 0)})


class ObjectiveNameTests(unittest.TestCase):
    def test_the_search_and_the_metric_share_one_name(self):
        self.assertEqual(APS.FIDELITY, fidelity.METRIC_VERSION)

    def test_the_report_name_follows_the_search_that_ran(self):
        self.assertEqual(APS.report_objective('fidelity', 'both'), APS.FIDELITY)   # --objective is not its term
        self.assertEqual(APS.report_objective('identity', 'identity'), 'ecapa_identity')
        self.assertEqual(APS.report_objective('identity', 'timbre'), 'timbre')
        self.assertEqual(APS.report_objective('identity', 'both'), 'identity+timbre')
        self.assertEqual(APS.report_objective(), 'ecapa_identity')

    def test_each_criterion_has_its_own_sentence_and_label(self):
        names = ('ecapa_identity', 'timbre', 'identity+timbre', APS.FIDELITY)
        self.assertEqual(len({APS.objective_line(name) for name in names}), 4)
        self.assertIn('acoustic fidelity', APS.objective_line(APS.FIDELITY))
        self.assertIn('three other sentences', APS.objective_line(APS.FIDELITY))
        self.assertEqual([APS.score_label(name) for name in names], ['ECAPA', 'score', 'score', 'fidelity'])
        self.assertEqual(APS.score_label(None), 'ECAPA')

    def test_the_audit_keeps_every_trial_and_anchor_for_a_caller_that_validates(self):
        block, report = APS.search_audio(TONE, lambda b: (.5, b), budget=60, held=HELD,
                                         anchors=[dict(NEUTRAL)])
        self.assertEqual(len(report['anchor_trials']), 1)
        self.assertEqual(report['anchor_trials'][0]['block']['eq_high'], 0.)
        fields = {row['field'] for row in report['trial_candidates']}
        self.assertEqual(fields, {'eq_high', 'comp'})
        self.assertTrue(all('block' in row and row['score'] == .5 for row in report['trial_candidates']))
        plain, report = APS.search_audio(TONE, lambda b: (.5, b), budget=60, held=HELD)
        self.assertEqual(report['anchor_trials'], [])


class GenerationSitesTests(unittest.TestCase):
    def test_every_generation_of_every_script_forwards_typical_p(self):
        # One forgotten site would judge, or deliver, a sentence on a different
        # sampling than the {} block the user is shown.
        names = ('voice_comparator.py', 'xtts_optimize.py', 'voice_validator.py',
                 'guided_meditation_generator_v23.py')
        found = 0
        for name in names:
            tree = ast.parse((Path(__file__).parent/name).read_text(encoding='utf-8'))
            for call in ast.walk(tree):
                if isinstance(call, ast.Call) and ast.unparse(call.func) in ('XC.generate', '_XC.generate'):
                    found += 1
                    with self.subTest(script=name, line=call.lineno):
                        self.assertIn('typical_p', ast.unparse(call))
        self.assertGreaterEqual(found, 7)                    # 4 + 1 + 1 + 1


class ConditioningExtensionsTests(unittest.TestCase):
    def test_old_blocks_stay_17_fields_and_new_extensions_roundtrip(self):
        original = settings.format_block(dict(seed=0))
        self.assertEqual(len(original.strip('{}').split(',')), 17)
        extended = settings.format_block(dict(seed=0, speaker_ref_len=20, ref_trim_db=45))
        self.assertEqual(len(extended.strip('{}').split(',')), 19)
        values = settings.parse_block(extended)
        self.assertEqual((values['speaker_ref_len'], values['ref_trim_db']), (20, 45))
        typical = settings.format_block(dict(typical_p=.9))
        self.assertEqual(len(typical.strip('{}').split(',')), 20)
        self.assertEqual(settings.parse_block(typical)['typical_p'], .9)
        for values in (dict(speaker_ref_len=-1), dict(speaker_ref_len=.5), dict(ref_trim_db=-1)):
            with self.assertRaises(ValueError): settings.validate(values)

    def test_typical_sampling_reaches_hf_only_when_active(self):
        calls = []
        class Model:
            def inference(self, *args, **kwargs):
                calls.append(kwargs)
                return {'wav': np.ones(2400, np.float32)*.01}
        sound = types.SimpleNamespace(write=lambda *a, **kw: None)
        with patch.dict(sys.modules, {'soundfile':sound}):
            for kwargs in (dict(typical_p=.9), dict(typical_p=1.),
                           dict(typical_p=.9, do_sample=False)):
                clone.generate(Model(), 'Bonjour.', 'fr', (None,None), 'unused.wav', **kwargs)
        self.assertEqual(calls[0]['typical_p'], .9)
        self.assertNotIn('typical_p',calls[1])
        self.assertNotIn('typical_p',calls[2])

    def test_script_renderer_honours_extensions_and_invalidates_only_conditioning_cache(self):
        with tempfile.TemporaryDirectory() as td:
            ref=Path(td)/'ref.wav';ref.touch()
            calls=[]
            xc=types.SimpleNamespace(
                compute_latents=lambda *a,**kw:calls.append(('conditioning',kw)) or (None,None),
                generate=lambda *a,**kw:calls.append(('generate',kw)))
            class Audio:
                @classmethod
                def from_wav(cls,path): return cls()
                def __add__(self,value): return self
            ns=clone_tests.extract_functions('guided_meditation_generator_v23.py',{'generate_sentence_audio'},
                dict(os=os,_XC=xc,_LAT_CACHE={},AudioSegment=Audio,render_voice_chain=lambda a,*args:a))
            tts=types.SimpleNamespace(synthesizer=types.SimpleNamespace(tts_model=object()))
            for trim,typical in ((45,.9),(60,.9),(60,.8)):
                params=settings.generator_params(settings.format_block(
                    dict(speaker_ref_len=20,ref_trim_db=trim,typical_p=typical)),{})
                result=ns['generate_sentence_audio']('Bonjour',1,[[str(ref)]],{1:tts},
                    dict(language='fr'),params,str(Path(td)/'out.wav'))
                self.assertIsNotNone(result)
            latents=[kw for tag,kw in calls if tag=='conditioning']
            generated=[kw for tag,kw in calls if tag=='generate']
            self.assertEqual([(kw['speaker_ref_len'],kw['ref_trim_db']) for kw in latents],[(20,45),(20,60)])
            self.assertEqual([kw['typical_p'] for kw in generated],[.9,.9,.8])

    def test_speaker_cap_replaces_only_the_speaker_embedding(self):
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td)/'ref.wav'; ref.touch()
            calls = []
            class Model:
                def get_conditioning_latents(self, **kw):
                    calls.append(kw)
                    return ('gpt'+str(kw['max_ref_length']), 'speaker'+str(kw['max_ref_length']))
            result = clone.compute_latents(Model(), str(ref), gpt_cond_len=45,
                                          max_ref_len=45, speaker_ref_len=10)
            self.assertEqual(result, ('gpt45', 'speaker10'))
            self.assertEqual(calls[0]['gpt_cond_len'], 45)
            self.assertEqual(calls[1]['gpt_cond_len'], 1)

    def test_reference_trim_is_on_cpu_before_model_load_and_uses_each_cap(self):
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td)/'ref.wav'
            wavfile.write(ref, 22050, np.ones(4*22050, np.float32)*.1)
            trimmed, loaded = [], []
            def read(path, **kwargs):
                rate, audio = wavfile.read(path); return audio, rate
            def write(path, audio, rate, **kwargs): wavfile.write(path, rate, np.float32(audio))
            def trim(audio, top_db):
                trimmed.append((len(audio), top_db)); return audio[100:-100], None
            class Model:
                def get_conditioning_latents(self, **kw):
                    self.assertions = None
                    self_data = [wavfile.read(p)[1] for p in kw['audio_path']]
                    loaded.append((kw, [len(y) for y in self_data]))
                    return ('gpt', 'speaker')
            with patch.dict(sys.modules, {
                    'soundfile': types.SimpleNamespace(read=read, write=write),
                    'librosa': types.SimpleNamespace(effects=types.SimpleNamespace(trim=trim))}):
                clone.compute_latents(Model(), str(ref), gpt_cond_len=3, gpt_cond_chunk_len=1,
                    max_ref_len=3, speaker_ref_len=2, ref_trim_db=45)
            self.assertEqual(trimmed, [(3*22050,45), (2*22050,45)])
            self.assertEqual(loaded[0][1], [3*22050-200])
            self.assertTrue(all('librosa_trim_db' not in kw for kw, lengths in loaded))


class JointSamplingTests(unittest.TestCase):
    @staticmethod
    def incumbent():
        return dict(seed=0,temp=.65,rep_pen=5.,top_p=.85,top_k=50,len_pen=1.,
                    num_beams=1,do_sample=True,score=.65,identity=.7,
                    details=[dict(accent=.9)])

    def test_sobol_is_deterministic_and_covers_lower_repetition_region(self):
        base = self.incumbent()
        rows = sampling_candidates(base, 12)
        self.assertEqual(rows, sampling_candidates(base, 12))
        self.assertEqual(len(rows), 12)
        self.assertTrue(any(p['rep_pen'] < 3 for p in rows))
        self.assertTrue(any(p['typical_p'] < 1 for p in rows))
        for p in rows: settings.validate(p)

    def test_matched_seeds_cancel_seed_offsets_and_confirm_real_gain(self):
        base = self.incumbent()
        offsets = {0:0,42:.12,100:-.12}
        def evaluate(seed, params):
            gain = .03 if params['rep_pen'] == 2 else 0.
            return dict(base, **params, seed=seed, score=.65+offsets[seed]+gain)
        winner, report = run_joint_search(base, [0,42,100], evaluate, count=12, max_draws=18)
        self.assertTrue(report['adopted'])
        self.assertEqual(winner['rep_pen'], 2)
        self.assertAlmostEqual(report['finalists'][0]['mean_gain'], .03)
        self.assertLess(report['finalists'][0]['paired_stderr'], 1e-10)

    def test_missing_confirmation_or_identity_loss_keeps_incumbent(self):
        base = self.incumbent()
        for missing in (True, False):
            def evaluate(seed, params):
                if missing and seed != 0: return None
                row = dict(base, **params, seed=seed, score=.8)
                if not missing: row['identity'] = .5
                return row
            winner, report = run_joint_search(base, [0,42,100], evaluate)
            self.assertIs(winner, base)
            self.assertFalse(report['adopted'])

    def test_native_optimizer_uses_both_new_conditioning_axes(self):
        helper = render_tests.OptimiserSearchTests('runTest')
        result, log = helper.run_optimizer(
            ['--search-conditioning','--advanced-conditioning','--cond-budget','100',
             '--gpt-cond-lens','30','--gpt-chunks','4','--budget','100','--adopt-alpha','1'],
            lambda rec: .5+.25*(rec['lat'].get('speaker_ref_len') == 20)
                             +.15*(rec['lat'].get('ref_trim_db') == 45))
        values = settings.parse_block(result['xtts_block'])
        self.assertEqual((values['speaker_ref_len'], values['ref_trim_db']), (20,45))
        self.assertIn('C4b', log); self.assertIn('C4c', log)

    def test_native_optimizer_adopts_new_joint_sampling_and_preserves_budget(self):
        helper = render_tests.OptimiserSearchTests('runTest')
        result, log = helper.run_optimizer(['--joint-sampling','12','--budget','200'],
            lambda rec: .6+.10*(rec['repetition_penalty'] == 2))
        self.assertTrue(result['joint_sampling']['adopted'])
        self.assertEqual(result['params']['rep_pen'], 2)
        self.assertLessEqual(result['generations']['sampling'], 200)

    def test_native_optimizer_reproduces_selected_typical_sampling(self):
        helper = render_tests.OptimiserSearchTests('runTest')
        result, _ = helper.run_optimizer(['--joint-sampling','12','--budget','200'],
            lambda rec: .6+.15*(rec.get('typical_p',1.) < .99))
        self.assertTrue(result['joint_sampling']['adopted'])
        self.assertLess(settings.parse_block(result['xtts_block'])['typical_p'], 1.)

    @staticmethod
    def shortcut_commands(extra):
        helper = render_tests.PipelineRecapTests('runTest')
        with tempfile.TemporaryDirectory() as td:
            commands, _, _ = helper.run_pipeline(td, ['--fidelity', *extra],
                lambda cmd: (0, 'Next []    : [1, FR, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]'),
                all_commands=True)
        return (next(c for c in commands if c[1].endswith('xtts_optimize.py')),
                next(c for c in commands if c[1].endswith('voice_comparator.py')))

    def test_pipeline_fidelity_shortcut_forwards_searches_and_keeps_explicit_budgets(self):
        optimise, compare = self.shortcut_commands(['--cond-budget=99', '--budget=150'])
        self.assertIn('--advanced-conditioning',optimise)
        self.assertEqual(optimise[optimise.index('--joint-sampling')+1], '12')
        self.assertEqual(optimise[optimise.index('--cond-budget')+1], '99')
        self.assertEqual(optimise[optimise.index('--budget')+1], '150')
        self.assertEqual(compare[compare.index('--audio-objective')+1], 'fidelity')
        self.assertEqual(compare[compare.index('--eq-limit')+1], '8.0')
        self.assertIn('--probe-beams',optimise)
        self.assertEqual(optimise[optimise.index('--beam-width')+1], '2')

    def test_pipeline_fidelity_shortcut_holds_speed_pan_and_limiter_unless_asked(self):
        # ECAPA and the acoustic profile cannot judge a meditation's pace, a pan or a
        # limiter: the shortcut does not hand them to the search on its own.
        _, held = self.shortcut_commands([])
        self.assertNotIn('--search-all-fields', held)
        _, free = self.shortcut_commands(['--search-all-fields'])
        self.assertIn('--search-all-fields', free)


class GuiAndPipelineTests(unittest.TestCase):
    def press_fidelity_setup(self, tick_all_fields=False):
        helper = budget_tests.AutoTabTests('runTest')
        fields, checks, buttons = helper.build()
        with tempfile.TemporaryDirectory() as td:
            ref=Path(td)/'ref.wav'
            wavfile.write(ref,16000,np.float32(tone_tests.harmonic_voice(16000,12,205,.2)))
            def read(path,**kw):
                rate,y=wavfile.read(path);return y,rate
            def info(path):
                rate,y=wavfile.read(path)
                return types.SimpleNamespace(duration=len(y)/rate,samplerate=rate,frames=len(y),channels=1)
            with patch.dict(sys.modules,{'soundfile':types.SimpleNamespace(info=info,read=read)}):
                fields['V1'].set(str(ref))
                if tick_all_fields:
                    checks['Include speed, pan and limiter'].set(True)
                buttons['fidelity_setup']()
                helper.run_pipeline()
            self.assertTrue(helper.commands,helper.logs)
            return helper.commands[-1], checks

    def test_fidelity_setup_button_enables_the_search_and_sizes_budgets(self):
        import search_plan as plan
        command, _ = self.press_fidelity_setup()
        value=lambda flag:command[command.index(flag)+1]
        self.assertEqual(value('--objective'),'both')
        self.assertEqual(value('--audio-objective'),'fidelity')
        self.assertEqual(value('--optimise-audio'),'de')
        self.assertEqual(value('--eq-limit'),'8.0')
        self.assertIn('--probe-beams',command)
        expected=plan.pipeline_sampling_budget(plan.parse_ints(value('--seeds')),True)+plan.joint_sampling_plan(12,2)
        self.assertGreaterEqual(int(value('--budget')),expected)
        self.assertGreater(int(value('--cond-budget')),34)

    def test_fidelity_setup_button_leaves_speed_pan_and_limiter_to_the_checkbox(self):
        held, checks = self.press_fidelity_setup()
        self.assertNotIn('--search-all-fields', held)
        self.assertFalse(checks['Include speed, pan and limiter'].get())
        free, checks = self.press_fidelity_setup(tick_all_fields=True)
        self.assertIn('--search-all-fields', free)                  # the user's own choice survives
        self.assertTrue(checks['Include speed, pan and limiter'].get())

    def test_auto_forwards_new_search_modes(self):
        helper = budget_tests.AutoTabTests('runTest')
        fields, checks, _ = helper.build()
        with tempfile.TemporaryDirectory() as td:
            ref=Path(td)/'ref.wav'; wavfile.write(ref, 16000, np.zeros(16000,np.float32))
            fields['V1'].set(str(ref))
            fields['Audio search'].set('de')
            fields['Audio objective'].set('fidelity')
            checks['Advanced XTTS conditioning (speaker cap, silence trim)'].set(True)
            checks['Joint XTTS sampling (12 Sobol candidates, paired seeds)'].set(True)
            helper.run_pipeline()
            command = helper.commands[-1]
        self.assertIn('--advanced-conditioning',command)
        self.assertEqual(command[command.index('--joint-sampling')+1], '12')
        self.assertEqual(command[command.index('--audio-objective')+1], 'fidelity')

    def test_auto_tells_the_search_target_only_to_a_search_that_runs(self):
        helper = budget_tests.AutoTabTests('runTest')
        fields, checks, _ = helper.build()
        with tempfile.TemporaryDirectory() as td:
            ref=Path(td)/'ref.wav'; wavfile.write(ref, 16000, np.zeros(16000,np.float32))
            fields['V1'].set(str(ref))
            fields['Audio objective'].set('fidelity')
            helper.run_pipeline()
            off = helper.commands[-1]
            fields['Audio search'].set('nelder')
            helper.run_pipeline()
            on = helper.commands[-1]
        self.assertNotIn('--audio-objective', off)
        self.assertEqual(on[on.index('--audio-objective')+1], 'fidelity')

    def test_auto_identity_target_leaves_the_objective_menu_in_charge(self):
        helper = budget_tests.AutoTabTests('runTest')
        fields, checks, _ = helper.build()
        with tempfile.TemporaryDirectory() as td:
            ref=Path(td)/'ref.wav'; wavfile.write(ref, 16000, np.zeros(16000,np.float32))
            fields['V1'].set(str(ref))
            fields['Audio search'].set('nelder')
            fields['Audio objective'].set('identity')
            fields['Objective'].set('both')
            helper.run_pipeline()
            command = helper.commands[-1]
        self.assertEqual(command[command.index('--audio-objective')+1], 'identity')
        self.assertEqual(command[command.index('--objective')+1], 'both')

    def test_comparator_forwards_fidelity_objective(self):
        fields, checks, commands, callbacks = gui_tests.FullSearchGuiTests('runTest').comparator_ui()
        fields['Audio search'].set('nelder')
        callbacks['>  Compare & Optimise'](None)
        command = commands[-1]
        self.addCleanup(os.unlink, command[command.index('--text-file')+1])
        self.assertEqual(command[command.index('--audio-objective')+1], 'fidelity')


class ComparatorFidelityIntegrationTests(unittest.TestCase):
    @staticmethod
    def run_comparator(root, *flags, brightness=False, tone=None, reject=False, asr_of=None, search_fails=None,
                       audio_block='[1, FR, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]'):
        """voice_comparator.main() on real signals, with the model, the speaker encoder
        and the ASR replaced. Returns the keyword arguments of every generation and the log.
        The stand-in encoder says .7 for everything, or, with `brightness`, scores a render
        by how close its share of energy above 4.5 kHz is to the reference's. `tone` are the
        fields the (replaced) tone fit changes; `reject` makes every finalist fail its check;
        `asr_of(path)` is what the (replaced) ASR hears in a file; `search_fails` is the message of
        the error the fidelity search raises."""
        sr = 24000
        clean = tone_tests.harmonic_voice(sr, 2.5, 205, .2, noise_db=-55, top=3500)
        noise = sosfilt(butter(4, [5000,7900], btype='bandpass', fs=sr, output='sos'),
                        np.random.default_rng(9).normal(size=len(clean)))
        raw = np.float32(clean+.03*noise)
        ref, given = root/'reference.wav', root/'given.wav'
        wavfile.write(ref,sr,np.float32(clean)); wavfile.write(given,sr,raw)
        def read(path, **kw):
            rate, y = wavfile.read(path); return y, rate
        def info(path):
            rate, y = wavfile.read(path)
            return types.SimpleNamespace(samplerate=rate,duration=len(y)/rate)
        def post(source,destination,block,*args):
            rate, y = wavfile.read(source)
            result = tone_tests.exact_chain(y, block, fs=rate, ratio=(1,1))
            wavfile.write(destination,rate,np.float32(result))
        generations = []
        def generate(model,text,lang,lat,path,**kw):
            generations.append(kw)
            wavfile.write(path,sr,raw)
        class Encoder:
            def embed(self,audio=None,*a,sr=None,**kw):
                if not brightness: return np.array([1.,0.])
                if isinstance(audio,(str,os.PathLike)): rate, y = wavfile.read(audio)
                else: y, rate = audio, sr
                power = np.abs(np.fft.rfft(np.asarray(y, np.float64)))**2
                return np.array([power[np.fft.rfftfreq(len(y), 1/rate) > 4500].sum()/power.sum()])
            def cosine(self,a,b):
                return .9-4*abs(float(a[0])-float(b[0])) if brightness else .7
        class Pron:
            def __init__(self,**kw): pass
            def score(self,path,lang=None,target_text=None):
                return dict(asr_of(path) if asr_of else asr(), text=target_text)
        modules = dict(soundfile=types.SimpleNamespace(read=read,info=info),
            torch=types.SimpleNamespace(Tensor=type('FakeTensor', (), {})),
            pron_score=types.SimpleNamespace(PronScorer=Pron),
            xtts_clone=types.SimpleNamespace(load_xtts=lambda d:(None,None),
                compute_latents=lambda *a,**kw:(None,None),
                generate=generate))
        argv = ['voice_comparator.py',str(ref),'FR','--xtts-block',settings.format_block({'seed':0,'typical_p':.9}),
                '--audio-block',audio_block,
                '--text','Une phrase à régler.','--raw-clone',str(given),'--no-fit-dynamics',
                *flags,
                '--output',str(root/'base.wav'),'--output-optimised',str(root/'final.wav'),
                '--output-raw',str(root/'raw.wav'),'--device','cpu']
        log = io.StringIO()
        with patch.dict(sys.modules,modules), patch.object(sys,'argv',argv), \
             patch.object(comparator,'_encoder',return_value=Encoder()), \
             patch.object(comparator,'_quiet_post',side_effect=post), \
             patch.object(comparator,'apply_post',side_effect=post), \
             patch.object(comparator,'fit_rendered_tone',side_effect=lambda r,rs,cs,b,*a,**kw:(dict(b,**(tone or {})),[])), \
             tone_tests.no_real_timbre_analysis(), contextlib.redirect_stdout(log), \
             contextlib.ExitStack() as stack:
            if search_fails:
                stack.enter_context(patch.object(fidelity_search,'optimise_fidelity',
                    side_effect=RuntimeError(search_fails)))
            if reject:
                stack.enter_context(patch.object(fidelity_search,'paired_decision',
                    side_effect=lambda rows,min_gain=.005,**kw:dict(adopted=False,reasons=['forced rejection'])))
            comparator.main()
        return generations, log.getvalue()

    def test_real_profiles_with_model_standins_adopt_and_preserve_all_wav_reports(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            generations, log = self.run_comparator(root,
                '--audio-objective','fidelity','--optimise-audio','nelder','--optimise-budget','100')
            report=json.loads((root/'final_audio_search.json').read_text())
            self.assertNotIn('error',report,log)
            self.assertTrue(report['adopted'],log)
            self.assertEqual(report['final_fit_metrics']['identity'],.7)
            self.assertGreater(report['final_fit_metrics']['acoustic_score'],report['baseline_metrics']['acoustic_score'])
            self.assertGreaterEqual(len(report['finalists']),1)
            self.assertTrue(all(Path(p).is_file() for group in report['validation_renders'] for p in group.values()))
            self.assertFalse(report['unseen_check']['used_for_selection'])
            self.assertTrue(generations)
            self.assertTrue(all(g['typical_p'] == .9 for g in generations))
            self.assertTrue(all(Path(p).is_file() for p in report['unseen_check']['files'].values()))
            tone=next(row for row in report['render_blocks'] if row['role']=='tone fit before fidelity search')
            self.assertTrue(Path(tone['file']).is_file())
            self.assertTrue(tone['audio_block'].startswith('[1, FR,'))

    def test_a_finalist_is_judged_on_what_the_asr_hears_in_its_own_render(self):
        # The same search twice; only the ASR's opinion of the candidates' files differs.
        heard = lambda path: asr(wer=.3) if 'fidelity_candidate_' in os.path.basename(str(path)) else asr()
        for asr_of, adopted in ((None, True), (heard, False)):
            with self.subTest(asr_degrades_the_candidates=asr_of is not None), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _, log = self.run_comparator(root, '--audio-objective','fidelity','--optimise-audio',
                                             'nelder','--optimise-budget','100', asr_of=asr_of)
                report = json.loads((root/'final_audio_search.json').read_text())
                self.assertEqual(report['adopted'], adopted, log)
                self.assertTrue(report['finalists'])
                if not adopted:
                    for finalist in report['finalists']:
                        self.assertIn('word error rate increased', finalist['decision']['reasons'])
                    rows = report['finalists'][0]['validation']
                    self.assertEqual({row['candidate_asr']['wer'] for row in rows}, {.3})
                    self.assertEqual({row['base_asr']['wer'] for row in rows}, {0.})

    def test_a_search_that_fails_leaves_the_tone_fit_and_says_so(self):
        # Whatever the search was doing, the delivered [] is the tone fit: not the block the user
        # started from, not a half-adopted one.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, log = self.run_comparator(root, '--audio-objective','fidelity','--optimise-audio',
                'nelder','--optimise-budget','100', tone=dict(eq_high=3.), search_fails='the search exploded',
                audio_block='[1, FR, 1, 0, 1, -1, .5, 90, 9000, .1, .2, .15, .1, -50, .2, 1]')
            report=json.loads((root/'final_audio_search.json').read_text())
            self.assertFalse(report['adopted'])
            self.assertEqual(report['error'],'the search exploded')
            self.assertEqual(report['objective'],'fidelity_v1')
            self.assertEqual(report['final_block']['eq_high'],3.)          # the tone fit ...
            self.assertEqual(report['final_block']['eq_low'],1.)           # ... on top of the input's other fields
            self.assertIn('Audio search failed: the search exploded',log)
            self.assertIn('The tone-fitted block is kept',log)
            self.assertIn('Final []: vol=+0  eq_low=+1.0  eq_mid=-1.0  eq_high=+3.0  hp=90  lp=9000',log)

    def test_the_identity_search_generates_its_sentences_with_the_same_sampling(self):
        # The {} block's typical_p must reach every sentence the 3.9 search checks too: a
        # candidate judged on a different sampling than the one delivered proves nothing.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            generations, log = self.run_comparator(root, '--audio-objective','identity',
                '--optimise-audio','nelder','--optimise-budget','100', brightness=True)
            report=json.loads((root/'final_audio_search.json').read_text())
            self.assertNotIn('error',report,log)
            self.assertEqual(report['objective'],'ecapa_identity')
            self.assertIn('Audio hold-out 2',log)                  # the checkpoint did generate
            self.assertGreaterEqual(len(generations),3)            # two hold-outs and the unseen sentence
            self.assertTrue(all(g['typical_p'] == .9 for g in generations),generations)

    def test_the_input_and_a_neutral_block_are_scored_beside_the_tone_fit(self):
        # An input [] with every effect on (speed 1.1, a pan of .2); the tone fit then makes
        # the highs +3 dB.
        effects = dict(eq_low=1., eq_mid=-1., hp=90, lp=9000, NR=.1, comp=.2, reverb=.1, noise_gate=-50.)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, log = self.run_comparator(root, '--audio-objective','fidelity','--optimise-audio',
                'nelder','--optimise-budget','100', tone=dict(eq_high=3.), reject=True,
                audio_block='[1, FR, 1.1, 0, 1, -1, .5, 90, 9000, .1, .2, .15, .1, -50, .2, 1]')
            report=json.loads((root/'final_audio_search.json').read_text())
            self.assertEqual(report['baseline_block']['eq_high'],3.)              # the tone fit
            given, neutral = report['anchor_trials']
            self.assertEqual((given['status'],neutral['status']),('scored','scored'),log)
            for field, value in dict(effects, eq_high=.5, **{'de-ess': .15}).items():
                self.assertEqual(given['block'][field],value,field)               # the input, as given
            for field in ('eq_low','eq_mid','eq_high','hp','lp','NR','comp','de-ess','reverb','noise_gate'):
                self.assertEqual(neutral['block'][field],0,field)                 # no effect at all
            # speed, pan and limiter are held at the tone fit's values, for every block
            for anchor in (given,neutral):
                self.assertEqual((anchor['block']['speed'],anchor['block']['pan'],anchor['block']['limiter']),
                                 (1.1,.2,1))
            self.assertEqual(set(report['held']),{'speed','pan','limiter'})

    def test_a_search_that_proves_nothing_delivers_the_tone_fit(self):
        # The [] handed back is what the tone fit made of this voice, not the block the user
        # started from: refusing the search must not undo the fit that came before it.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, log = self.run_comparator(root, '--audio-objective','fidelity','--optimise-audio',
                'nelder','--optimise-budget','100', tone=dict(eq_high=3.), reject=True)
            report=json.loads((root/'final_audio_search.json').read_text())
            self.assertFalse(report['adopted'],log)
            self.assertEqual(report['baseline_block']['eq_high'],3.)       # the search's reference
            self.assertEqual(report['final_block']['eq_high'],3.)          # what comes out
            self.assertEqual(report['tone_fitted_block']['eq_high'],3.)
            self.assertEqual(report['input_audio_block'],'[1, FR, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]')
            self.assertIn('keeping the tone-fitted effects',log)
            # the finalists were checked, and refused for their reason
            self.assertTrue(report['finalists'])
            self.assertTrue(all(not row['decision']['adopted'] for row in report['finalists']))

    def test_the_dedicated_final_sentence_belongs_to_the_fidelity_search_only(self):
        from probe_texts import probe_texts, holdout_texts, final_check_texts
        shown = lambda log: re.search(r'Unseen sentence: (.*)', log).group(1).strip()
        self.assertFalse(set(final_check_texts('fr')) & set(holdout_texts('fr',2)+probe_texts('fr',3)))
        for flags, expected in (
                (('--audio-objective','fidelity'), final_check_texts('fr')),
                (('--audio-objective','identity'), holdout_texts('fr',2)+probe_texts('fr',3))):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as td:
                _, log = self.run_comparator(Path(td), *flags,'--optimise-audio','nelder','--optimise-budget','100')
                self.assertIn(shown(log), expected)
        for flags in ((), ('--audio-objective','fidelity')):    # no search: the 3.9 sentences,
            with self.subTest(search_off=flags), tempfile.TemporaryDirectory() as td:  # whatever the target
                _, log = self.run_comparator(Path(td), *flags, '--optimise-audio','none')
                self.assertIn(shown(log), holdout_texts('fr',2)+probe_texts('fr',3))

    def test_the_search_target_names_the_branch_in_the_report(self):
        for flags, objective in ((('--audio-objective','fidelity'), 'fidelity_v1'),
                                 (('--audio-objective','identity'), 'ecapa_identity')):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _, log = self.run_comparator(root, *flags,'--optimise-audio','nelder','--optimise-budget','100')
                report=json.loads((root/'final_audio_search.json').read_text())
                self.assertEqual(report['objective'],objective,log)


if __name__ == '__main__':
    unittest.main()

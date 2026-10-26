"""Offline regression tests for the conditioning confirmation (C6).

    python3 -m unittest -v test_confirmation_regressions

The optimiser runs end to end with XTTS, Whisper and ECAPA simulated, and a
scripted score: each test knows the true effect of every setting, so it can
check what one draw shows and what the confirmation concludes.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import xtts_optimize as optimizer
from search_plan import CONFIRM_FINALISTS, NOISE_PROBES
import test_clonage_regressions as clonage      # the module, so its tests are not run twice

try:
    from scipy import stats
except ImportError:
    stats = None

# Draw-to-draw noise of the accent score, by seed: sd 0.021 over seeds 0-4,
# 0.024 over the probe seeds 1-4. The objective is 0.6 accent + 0.4 identity,
# so both shrink by 0.6 in the score.
NOISE = {0: 0.0, 1: 0.02, 2: -0.02, 3: 0.03, 4: -0.01}
X = (30, 8, 0)          # gpt_cond_len, chunk, sound_norm_refs of the challenger


def scripted(effect_on_every_seed, extra_on_seed_0=0.0):
    """Accent 0.5 plus the seed's noise, the same for every setting; X adds
    its true effect on every seed, and its screening draw (seed 0) may be
    luckier still."""
    def score(record):
        c, seed = record['conditioning'], record['seed']
        value = 0.5 + NOISE[seed]
        if (c['gpt_cond_len'], c['gpt_cond_chunk_len'], int(c['sound_norm_refs'])) == X:
            value += effect_on_every_seed + (extra_on_seed_0 if seed == 0 else 0.0)
        return value
    return score


def run(score, budget=40, chunks='4 8', extra=(), gcl='30'):
    return clonage.OptimizerTests().run_optimizer([
        '--method', 'coord', '--rounds', '0', '--gpt-cond-lens', gcl, '--gpt-chunks', chunks,
        '--cond-budget', str(budget), '--budget', '2', '--no-holdout', *extra], score=score)


def chunk_of(result):
    return result['params']['gpt_cond_chunk_len']


def run_with_audio(score, audio_of, chunks):
    """Like run(), with each render written to its file: audio_of(record)
    gives its bytes, so two settings can give the same audio."""
    import json, tempfile, types
    from unittest.mock import patch
    import xtts_settings as settings
    with tempfile.TemporaryDirectory() as td:
        ref = Path(td) / 'a.wav'
        ref.touch()
        metadata = {}
        def generate(model, text, lang, lat, path, **kw):
            rec = dict(kw, text=text, conditioning=lat)
            metadata[path] = rec
            Path(path).write_bytes(audio_of(rec))
        class Pron:
            def __init__(self, **kw): pass
            def score(self, path, **kw): return {'score': score(metadata[path])}
        class Encoder:
            def __init__(self, **kw): pass
            def embed(self, path): return np.array([1., 0.])
            def cosine(self, a, b): return .8
        mocks = {'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
                 'xtts_clone': types.SimpleNamespace(load_xtts=lambda d: (None, None),
                                                    compute_latents=lambda model, paths, **kw: dict(kw),
                                                    generate=generate),
                 'pron_score': types.SimpleNamespace(PronScorer=Pron),
                 'speaker_identity': types.SimpleNamespace(SpeakerEncoder=Encoder)}
        output = Path(td) / 'winner.json'
        argv = [str(ref), 'FR', '--xtts-block', settings.format_block({'seed': 0}), '--text', 'TEST',
                '--seeds', '0', '--device', 'cpu', '--result-json', str(output),
                '--method', 'coord', '--rounds', '0', '--gpt-cond-lens', '30', '--gpt-chunks', chunks,
                '--cond-budget', '40', '--budget', '2', '--no-holdout']
        with patch.dict(sys.modules, mocks):
            log = clonage.invoke(optimizer, argv)
        return json.loads(output.read_text()), log


class DecisionTests(unittest.TestCase):
    BASE_SCORES = [0.80, 0.82, 0.78, 0.81]

    def shifted(self, gain):
        return [x + gain for x in self.BASE_SCORES]

    def test_a_gain_above_the_bar_is_adopted(self):
        won, gains, bar, sd, df = optimizer.confirmation_decision(
            self.BASE_SCORES, [('a', self.shifted(0.05))], 0.05)
        self.assertEqual((won, df), (0, 6))
        self.assertAlmostEqual(gains[0], 0.05)
        # Both groups spread alike, so the pooled sd is the base's own.
        self.assertAlmostEqual(sd, float(np.std(self.BASE_SCORES, ddof=1)))
        # One-sided t at 1 - 0.05 with 6 degrees of freedom, on a difference
        # of two means of 4: sd * sqrt(2/4).
        self.assertAlmostEqual(bar, 1.9432 * sd * np.sqrt(0.5), places=3)

    def test_a_gain_below_the_bar_is_kept(self):
        won, gains, bar, _sd, _df = optimizer.confirmation_decision(
            self.BASE_SCORES, [('a', self.shifted(0.02))], 0.05)
        self.assertIsNone(won)
        self.assertLess(gains[0], bar)

    def test_more_finalists_raise_the_bar(self):
        # +0.025 passes alone, not as the best of three (Bonferroni).
        alone = optimizer.confirmation_decision(self.BASE_SCORES, [('a', self.shifted(0.025))], 0.05)
        three = optimizer.confirmation_decision(
            self.BASE_SCORES, [('z1', self.shifted(0.0)), ('a', self.shifted(0.025)),
                             ('z2', self.shifted(0.0))], 0.05)
        self.assertEqual(alone[0], 0)
        self.assertIsNone(three[0])
        self.assertGreater(three[2], alone[2])
        self.assertEqual(three[4], 12)

    def test_the_best_mean_is_the_one_adopted(self):
        won, gains, _bar, _sd, _df = optimizer.confirmation_decision(
            self.BASE_SCORES, [('a', self.shifted(0.04)), ('b', self.shifted(0.06))], 0.05)
        self.assertEqual(won, 1)
        self.assertAlmostEqual(gains[1], 0.06)

    def test_alpha_one_adopts_any_gain(self):
        won, _gains, bar, _sd, _df = optimizer.confirmation_decision(
            self.BASE_SCORES, [('a', self.shifted(0.001))], 1.0)
        self.assertEqual((won, bar), (0, 0.0))

    def test_identical_renders_fall_back_on_the_floor(self):
        won, _gains, bar, sd, _df = optimizer.confirmation_decision(
            [0.8] * 4, [('a', [0.8] * 4)], 0.05)
        self.assertIsNone(won)
        self.assertEqual(sd, optimizer.SD_FLOOR)
        self.assertGreater(bar, 0)

    def test_no_finalist(self):
        self.assertEqual(optimizer.confirmation_decision(self.BASE_SCORES, [], 0.05)[:3],
                         (None, [], 0.0))

    @unittest.skipIf(stats is None, 'scipy is needed for the exact quantiles')
    def test_t_quantile_matches_scipy(self):
        for df in range(4, 120):
            for p in np.linspace(0.9, 0.999, 60):
                exact = stats.t.ppf(p, df)
                self.assertLess(abs(optimizer.t_quantile(p, df) / exact - 1),
                                0.01 if df < 6 else 0.002, (p, df))
        self.assertAlmostEqual(optimizer.t_quantile(0.95, 0), stats.norm.ppf(0.95))
        self.assertAlmostEqual(optimizer.t_quantile(0.95, float('inf')), stats.norm.ppf(0.95))

    def test_t_quantile_known_values(self):
        # Student's table, one-sided.
        for p, df, value in ((0.95, 6, 1.943), (0.975, 12, 2.179), (0.99, 9, 2.821),
                             (0.95, 30, 1.697)):
            self.assertAlmostEqual(optimizer.t_quantile(p, df), value, places=2)


class ConfirmationRunTests(unittest.TestCase):
    """The same screening, two truths: X screens +0.04 (accent) on seed 0
    either way. One draw cannot tell a real gain from a lucky draw; four
    fresh seeds can."""

    BASE = 'gcl=30 chunk=4 norm=0 refs=1 max_ref=30'
    X = 'gcl=30 chunk=8 norm=0 refs=1 max_ref=30'

    def test_a_real_gain_one_draw_cannot_show_is_adopted(self):
        result, log, _conditions, _generated = run(scripted(0.04))
        # Screening: X's +0.024 (score) is below the single-draw bar.
        self.assertIn(f'-> C2 gpt_cond_chunk_len: kept {self.BASE} '
                      '-- best gain +0.024 is within the noise (bar 0.029', log)
        self.assertIn('C6 — confirmation: 1 candidate(s) against the base, each on seeds '
                      '1 2 3 4', log)
        self.assertIn(f'-> C6 confirmation: ADOPTED {self.X} (+0.024 over the base, bar 0.020 '
                      'for 1 candidate(s) x 4 seeds', log)
        self.assertEqual(chunk_of(result), 8)
        conf = result['conditioning_confirmation']
        self.assertTrue(conf['adopted'])
        self.assertEqual(conf['seeds'], [1, 2, 3, 4])
        self.assertEqual(conf['df'], 6)
        self.assertEqual((conf['base'], conf['screening_winner'], conf['winner']),
                         (self.BASE, self.BASE, self.X))
        self.assertAlmostEqual(conf['mean'] - conf['base_mean'], 0.6 * 0.04)
        self.assertAlmostEqual(conf['finalists'][0]['screened'], 0.6 * 0.54 + 0.4 * 0.8)
        self.assertIn('confirmed mean     = ', log)

    def test_a_lucky_draw_is_rejected(self):
        result, log, _conditions, _generated = run(scripted(0.0, extra_on_seed_0=0.04))
        self.assertIn(f'-> C2 gpt_cond_chunk_len: kept {self.BASE} '
                      '-- best gain +0.024 is within the noise (bar 0.029', log)
        self.assertIn("-> C6 confirmation: kept the base -- no candidate's mean beat it on fresh "
                      "seeds (best +0.000", log)
        self.assertEqual(chunk_of(result), 4)
        self.assertFalse(result['conditioning_confirmation']['adopted'])

    def test_a_single_draw_adoption_that_fresh_seeds_do_not_confirm_goes_back_to_the_base(self):
        # X is luckier still on seed 0 (+0.08 accent): C2 adopts it on that
        # draw and the next stages work around it. On fresh seeds it is the
        # base again, so the base is the result.
        result, log, _conditions, _generated = run(scripted(0.0, extra_on_seed_0=0.08))
        self.assertIn(f'-> C2 gpt_cond_chunk_len: ADOPTED {self.X}', log)
        self.assertIn('C6 — confirmation: 1 candidate(s) against the base', log)   # once
        self.assertIn(f'     screening winner {self.X}: mean', log)
        self.assertIn("-> C6 confirmation: kept the base -- no candidate's mean beat it", log)
        self.assertIn(f'     the screening\'s {self.X} won on one draw and is not confirmed', log)
        self.assertEqual(chunk_of(result), 4)
        conf = result['conditioning_confirmation']
        self.assertEqual((conf['screening_winner'], conf['winner']), (self.X, self.BASE))
        self.assertEqual([f['cond'] for f in conf['finalists']], [self.X])

    def test_a_real_gain_adopted_by_a_stage_is_confirmed(self):
        result, log, _conditions, _generated = run(scripted(0.08))
        self.assertIn(f'-> C2 gpt_cond_chunk_len: ADOPTED {self.X}', log)
        self.assertIn(f'-> C6 confirmation: ADOPTED {self.X} (+0.048 over the base', log)
        self.assertEqual(chunk_of(result), 8)

    def test_the_old_rule_skips_the_confirmation(self):
        # --adopt-alpha 1 reproduces the old "any gain wins", lucky draw included.
        result, log, _conditions, _generated = run(scripted(0.0, extra_on_seed_0=0.04),
                                                   extra=('--adopt-alpha', '1'))
        self.assertNotIn('C6', log)
        self.assertIsNone(result['conditioning_confirmation'])
        self.assertEqual(chunk_of(result), 8)

    def test_the_probe_renders_are_the_bases_confirmation(self):
        result, _log, _conditions, generated = run(scripted(0.04))
        def renders(chunk, seed):
            return sum(1 for g in generated if g['seed'] == seed
                       and g['conditioning']['gpt_cond_chunk_len'] == chunk
                       and not g['conditioning']['sound_norm_refs'])
        for seed in (1, 2, 3, 4):
            self.assertEqual(renders(4, seed), 1)      # the probe, reused by C6
            self.assertEqual(renders(8, seed), 1)      # C6's renders of X
        # Base on 5 seeds, C2 (X), C3 (norm 1), C5 (chunk 8 with norm 1),
        # then X on the 4 probe seeds: nothing else.
        self.assertEqual(result['generations']['conditioning'], 1 + NOISE_PROBES + 3 + NOISE_PROBES)

    def test_no_screened_candidate_above_the_base_means_no_confirmation(self):
        result, log, _conditions, _generated = run(scripted(0.0))
        self.assertIn("C6 — confirmation: no screened candidate beat the base's draw; "
                      "nothing to confirm", log)
        self.assertIsNone(result['conditioning_confirmation'])
        self.assertEqual(result['generations']['conditioning'], 1 + NOISE_PROBES + 3)

    def test_c5_leaves_the_confirmation_its_renders(self):
        # 11 is the least that measures the noise: C5 gives up its render so
        # that C6 can still compare X on the four probe seeds.
        result, log, _conditions, _generated = run(scripted(0.04), budget=11)
        self.assertIn('[!] conditioning budget: C5 stops 1 render(s) short of its 8, the 4 '
                      'generation(s) left are kept for the confirmation (C6 needs up to '
                      f'{CONFIRM_FINALISTS * NOISE_PROBES})', log)
        self.assertEqual(result['truncated_stages'], {'C5 joint checks': 1})
        self.assertIn(f'-> C6 confirmation: ADOPTED {self.X}', log)
        self.assertEqual(result['generations']['conditioning'], 11)

    def test_a_confirmation_cut_by_the_budget_says_so(self):
        # One more C2 candidate (chunk 6) and the same 11: X gets 3 of its 4
        # renders, and nothing is decided on a partial comparison.
        result, log, _conditions, _generated = run(scripted(0.04), budget=11, chunks='4 6 8')
        self.assertIn("'C6 confirmation' is cut short", log)
        self.assertIn('-> C6 confirmation: cut by the conditioning budget before one complete '
                      'comparison; the base stays (--cond-budget ', log)
        self.assertIn('(C6 was cut by the budget: the base, nothing confirmed)', log)
        self.assertIn('C6 confirmation', result['truncated_stages'])
        self.assertIsNone(result['conditioning_confirmation'])
        self.assertEqual(chunk_of(result), 4)

    def test_a_cut_confirmation_keeps_the_base(self):
        # 3.7.2: unconfirmed, the stages' winner of single draws does not
        # replace the base; the log names what was not adopted.
        result, log, _conditions, _generated = run(scripted(0.0, extra_on_seed_0=0.08),
                                                   budget=11, chunks='4 6 8')
        self.assertIn(f"comparison; the base stays; the screening's {self.X} is not adopted", log)
        self.assertIn("(C6 was cut by the budget: the base, nothing confirmed)", log)
        self.assertEqual(chunk_of(result), 4)
        self.assertIsNone(result['conditioning_confirmation'])

    def test_at_most_the_configured_finalists_are_confirmed(self):
        # Four challengers screen above the base, by 0.01 to 0.04: the three
        # best are confirmed, on 4 seeds each.
        better = {(30, 6, 0): 0.01, (30, 8, 0): 0.04, (30, 10, 0): 0.03, (30, 12, 0): 0.02}
        def score(record):
            c, seed = record['conditioning'], record['seed']
            key = (c['gpt_cond_len'], c['gpt_cond_chunk_len'], int(c['sound_norm_refs']))
            return 0.5 + NOISE[seed] + (better.get(key, 0.0) if seed == 0 else 0.0)
        result, log, _conditions, _generated = run(score, budget=60, chunks='4 6 8 10 12')
        conf = result['conditioning_confirmation']
        self.assertEqual(len(conf['finalists']), CONFIRM_FINALISTS)
        self.assertEqual([f['cond'] for f in conf['finalists']],
                         [self.X, 'gcl=30 chunk=10 norm=0 refs=1 max_ref=30',
                          'gcl=30 chunk=12 norm=0 refs=1 max_ref=30'])
        self.assertFalse(conf['adopted'])               # lucky draws, all of them
        self.assertIn(f'C6 — confirmation: {CONFIRM_FINALISTS} candidate(s) against the base', log)

    def test_the_stages_winner_is_confirmed_first(self):
        # C2 adopts chunk 8 on its lucky draw. Three candidates around it then
        # screen higher still, within the noise (C2b, C3, C5): the stages'
        # winner is confirmed first all the same, with the two best of them.
        luck = {(30, 8, 0): 0.08, (20, 8, 0): 0.09, (30, 8, 1): 0.10, (20, 8, 1): 0.095}
        def score(record):
            c, seed = record['conditioning'], record['seed']
            key = (c['gpt_cond_len'], c['gpt_cond_chunk_len'], int(c['sound_norm_refs']))
            return 0.5 + NOISE[seed] + (luck.get(key, 0.0) if seed == 0 else 0.0)
        result, log, _conditions, _generated = run(score, gcl='20 30')
        self.assertIn(f'-> C2 gpt_cond_chunk_len: ADOPTED {self.X}', log)
        conf = result['conditioning_confirmation']
        self.assertEqual([f['cond'] for f in conf['finalists']],
                         [self.X, 'gcl=30 chunk=8 norm=1 refs=1 max_ref=30',
                          'gcl=20 chunk=8 norm=1 refs=1 max_ref=30'])
        self.assertEqual(conf['screening_winner'], self.X)
        self.assertEqual(chunk_of(result), 4)


    def test_a_setting_with_the_same_audio_is_confirmed_once(self):
        # chunk 10 renders exactly what chunk 8 renders (as a cap beyond the
        # audio XTTS gets would): one candidate, not two copies of it.
        score = scripted(0.0, extra_on_seed_0=0.04)
        def audio(record):
            c = record['conditioning']
            chunk = 8 if c['gpt_cond_chunk_len'] == 10 else c['gpt_cond_chunk_len']
            return f"{record['seed']} {chunk} {c['sound_norm_refs']}".encode()
        def scored(record):
            c = dict(record['conditioning'])
            if c['gpt_cond_chunk_len'] == 10:
                c['gpt_cond_chunk_len'] = 8
            return score(dict(record, conditioning=c))
        result, log = run_with_audio(scored, audio, '4 8 10')
        self.assertIn('= same audio as', log)
        conf = result['conditioning_confirmation']
        self.assertEqual([f['cond'] for f in conf['finalists']], [self.X])


    def test_c5_renders_new_gpt_pairs_before_other_reference_caps(self):
        # The structure of a real run: analysis gcl 51 s, gcl s 20 30 45 52,
        # caps 20 30 45 52. No sequential stage renders gcl 20 with chunk 8;
        # C5's grid holds 87 combinations, and 8 renders reach that pair only
        # if GPT changes go before cap changes (a cap at or above gcl repeats
        # the GPT conditioning of a render already screened).
        import xtts_settings as settings
        def score(record):
            c, seed = record['conditioning'], record['seed']
            value = 0.5 + NOISE[seed]
            if (c['gpt_cond_len'], c['gpt_cond_chunk_len'], int(c['sound_norm_refs'])) == (20, 8, 0):
                value += 0.06
            return value
        result, log, _conditions, _generated = clonage.OptimizerTests().run_optimizer([
            '--xtts-block', settings.format_block({'seed': 0, 'gpt_cond_len': 51}),
            '--method', 'coord', '--rounds', '0', '--gpt-cond-lens', '20 30 45 52',
            '--gpt-chunks', '4 6 8 10', '--max-ref-len', '52', '--max-ref-lens', '20 30 45 52',
            '--cond-budget', '48', '--budget', '2', '--no-holdout'], score=score)
        self.assertIn('C5 screened the nearest 19 of 88 combinations (8 new render(s), cap 8)', log)
        self.assertIn('-> C6 confirmation: ADOPTED gcl=20 chunk=8 norm=0 refs=1 max_ref=52', log)
        self.assertEqual((result['params']['gpt_cond_len'], chunk_of(result)), (20, 8))


if __name__ == '__main__':
    unittest.main()

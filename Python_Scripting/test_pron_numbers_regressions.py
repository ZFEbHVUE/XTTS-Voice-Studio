"""ASR numbers: tolerate spelling, retain wrong values and extra words."""
import types
import unittest

from pron_score import PronScorer, _french_cardinal, word_error_rate
from xtts_optimize import format_holdout_details


class FrenchNumberTests(unittest.TestCase):
    REFERENCE = 'Quarante-huit personnes attendaient déjà sur le quai numéro trois.'
    HEARD = "48 personnes attendaient déjà sur le quai numéro 3, c'est toi."

    def test_real_log_preserves_the_extra_words(self):
        self.assertAlmostEqual(word_error_rate(self.REFERENCE, self.HEARD), 4 / 9)
        self.assertAlmostEqual(word_error_rate(self.REFERENCE, self.HEARD, lang='fr'), 2 / 9)

    def test_only_formatting_changes_are_perfect(self):
        heard = '48 personnes attendaient déjà sur le quai numéro 3.'
        self.assertEqual(word_error_rate(self.REFERENCE, heard, lang='FR'), 0.)

    def test_equivalence_is_symmetric(self):
        self.assertEqual(word_error_rate('48 personnes', 'quarante-huit personnes', 'fr'), 0.)
        self.assertEqual(word_error_rate('quarante-huit personnes', '48 personnes', 'fr'), 0.)

    def test_hyphens_spaces_and_unicode_dashes(self):
        for spelling in ('quarante-huit', 'quarante huit', 'quarante‑huit', 'quarante–huit'):
            with self.subTest(spelling=spelling):
                self.assertEqual(word_error_rate(spelling, '48', 'fr'), 0.)

    def test_french_cardinal_boundaries(self):
        samples = {0: 'zéro', 17: 'dix-sept', 21: 'vingt et un',
                   71: 'soixante et onze', 80: 'quatre-vingts',
                   81: 'quatre-vingt-un', 99: 'quatre-vingt-dix-neuf',
                   100: 'cent', 200: 'deux cents', 201: 'deux cent un',
                   1000: 'mille', 2026: 'deux mille vingt-six',
                   80000: 'quatre-vingt mille', 200000: 'deux cent mille'}
        for n, spelling in samples.items():
            with self.subTest(n=n):
                self.assertEqual(_french_cardinal(n), spelling)
                self.assertEqual(word_error_rate(spelling, str(n), 'fr'), 0.)

    def test_distinct_values_remain_errors(self):
        self.assertGreater(word_error_rate('quarante-huit personnes', '43 personnes', 'fr'), 0.)
        self.assertGreater(word_error_rate('numéro trois', 'numéro 8', 'fr'), 0.)

    def test_larger_numbers_are_matched_before_their_components(self):
        self.assertEqual(word_error_rate('vingt et un puis un', '21 puis 1', 'fr'), 0.)

    def test_ordinary_words_and_other_languages_are_unchanged(self):
        pairs = [('Un vieux manteau neuf', 'Un manteau neuf'),
                 ('quarante-huit', 'quarante huit'), ('trois', '3')]
        for a, b in pairs:
            self.assertEqual(word_error_rate(a, b, 'en'), word_error_rate(a, b))
        self.assertEqual(word_error_rate('Un vieux manteau neuf', 'Un manteau neuf', 'fr'), 1 / 4)

    def test_unsupported_numeric_formats_are_not_guessed(self):
        for words, digits in [('trois', '-3'), ('trois', '+3'), ('trois', '3.14'),
                              ('trois', '3,14'), ('trois', '3/10'), ('trois', '3:10'),
                              ('sept', '007'), ('mille', '1 000'),
                              ('trois', '3e'), ('quarante-huit', 'A48')]:
            with self.subTest(digits=digits):
                self.assertGreater(word_error_rate(words, digits, 'fr'), 0.)

    def test_a_missing_number_is_still_missing(self):
        self.assertGreater(word_error_rate('3 personnes', 'personnes', 'fr'), 0.)

    def test_no_digit_anchor_preserves_the_historical_metric(self):
        self.assertEqual(word_error_rate('Un instant de paix', 'Un moment de paix', 'fr'), .25)

    def test_the_scorer_uses_the_language_and_keeps_both_metrics(self):
        segment = types.SimpleNamespace(text=self.HEARD, start=0., end=4., avg_logprob=-.1)
        scorer = PronScorer.__new__(PronScorer)
        scorer.model = types.SimpleNamespace(transcribe=lambda *a, **k: (
            iter([segment]), types.SimpleNamespace(language='fr', language_probability=1.)))
        result = scorer.score('unused.wav', lang='fr', target_text=self.REFERENCE)
        self.assertAlmostEqual(result['wer'], 2 / 9)
        self.assertAlmostEqual(result['wer_raw'], 4 / 9)
        self.assertLess(result['score'], 1.)
        self.assertEqual(result['text'], self.HEARD)

    def test_holdout_reports_raw_and_normalized_wer(self):
        lines = format_holdout_details([dict(accent=.8, identity=.7, score=.75,
            wer=2/9, wer_raw=4/9, detected='fr', text=self.REFERENCE, heard=self.HEARD)])
        self.assertIn('WER 0.22 (raw spelling 0.44; FR numbers normalised)', lines[0])


if __name__ == '__main__':
    unittest.main()

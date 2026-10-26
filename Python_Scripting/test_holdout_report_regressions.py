#!/usr/bin/env python3
"""Regression tests for the hold-out report and identical renders (3.6).

In a real run the mean accent fell from 0.946 on the search to 0.495 on two
unseen sentences (most likely one of them failed) while the report said 'no
large score drop': the accent weighs 0.3 and identity rose. The renders were
deleted with the work folder, so nobody could listen to them.

    python -m unittest -v test_holdout_report_regressions
"""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import xtts_optimize as optimizer
import xtts_pipeline as pipeline
import xtts_settings as settings

FR_UNSEEN = "Quarante-huit personnes attendaient déjà sur le quai numéro trois."


def run_optimizer(td, options, heard=None, audio_key=None):
    """xtts_optimize.main() with simulated XTTS / Whisper / ECAPA.

    heard(text) -> (score, transcript, language) for one render; audio_key(kw)
    -> the bytes the simulated XTTS writes, so two settings can give the very
    same file. Returns (result JSON, log, number of Whisper calls)."""
    ref = Path(td) / 'a.wav'
    ref.touch()
    meta, calls = {}, []
    heard = heard or (lambda text: (.95, text, 'fr'))
    audio_key = audio_key or (lambda kw: repr(sorted(kw.items())))

    def latents(model, paths, **kw):
        return dict(kw)

    def generate(model, text, lang, lat, path, **kw):
        meta[path] = dict(kw, text=text)
        Path(path).write_bytes(audio_key(dict(kw, text=text)).encode())

    class Pron:
        def __init__(self, **kw):
            pass

        def score(self, path, lang=None, target_text=None):
            calls.append(path)
            score, text, detected = heard(meta[path]['text'])
            return {'score': score, 'text': text, 'detected': detected,
                    'lang_prob': .9, 'wer': 0.0 if text == target_text else 1.0}

    class Encoder:
        def __init__(self, **kw):
            pass

        def embed(self, path):
            return np.array([1., 0.])

        def cosine(self, a, b):
            return .7
    mocks = {'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
             'xtts_clone': types.SimpleNamespace(load_xtts=lambda d: (None, None),
                                                compute_latents=latents, generate=generate),
             'pron_score': types.SimpleNamespace(PronScorer=Pron),
             'speaker_identity': types.SimpleNamespace(SpeakerEncoder=Encoder)}
    out = Path(td) / 'winner.json'
    argv = ['xtts_optimize.py', str(ref), 'FR', '--xtts-block',
            settings.format_block(dict(seed=0, temp=.62, rep_pen=5, top_p=.85, top_k=50)),
            '--text', 'Bonjour, ceci est un test.', '--seeds', '0 42 100', '--device', 'cpu',
            '--no-search-conditioning', '--result-json', str(out)] + options
    log = io.StringIO()
    with patch.dict(sys.modules, mocks), patch.object(sys, 'argv', argv), \
         contextlib.redirect_stdout(log):
        optimizer.main()
    return json.loads(out.read_text()), log.getvalue(), len(calls)


def unseen_fails(text):
    if text == FR_UNSEEN:
        return .07, 'Forty-eight people were already waiting on platform three.', 'en'
    return .93, text, 'fr'


class HoldoutReportTests(unittest.TestCase):
    def test_a_failed_unseen_sentence_is_named_kept_and_saved(self):
        with tempfile.TemporaryDirectory() as td:
            prefix = os.path.join(td, 'voice_pipeline_clone_holdout')
            result, log, _ = run_optimizer(td, ['--budget', '60',
                                                '--holdout-wav-prefix', prefix], unseen_fails)
            kept = [d['wav'] for d in result['holdout_detail']]
            self.assertTrue(all(kept) and all(os.path.isfile(p) for p in kept))
            self.assertEqual([os.path.basename(p) for p in kept],
                             ['voice_pipeline_clone_holdout1.wav', 'voice_pipeline_clone_holdout2.wav'])
        self.assertIn('[H2] accent 0.070', log)
        self.assertIn('heard: Forty-eight people', log)
        self.assertIn('[!] Unseen sentence 2: heard as EN, not FR (accent 0.07) -- listen to it', log)
        self.assertIn('[!] The score drop alone does not show it', log)
        self.assertNotIn('No large score drop', log)
        self.assertIn('unseen sentence(s) to listen to: 2', log)
        self.assertEqual(result['holdout_alerts'],
                         [{'sentence': 2, 'reason': 'heard as EN, not FR (accent 0.07)'}])
        self.assertEqual(result['holdout_detail'][1]['detected'], 'en')
        self.assertEqual(result['holdout_detail'][0]['text'],
                         'Chaque expiration relâche un peu plus les épaules et la mâchoire.')

    def test_good_unseen_sentences_raise_no_alert_and_keep_nothing_by_default(self):
        with tempfile.TemporaryDirectory() as td:
            result, log, _ = run_optimizer(td, ['--budget', '60'])
        self.assertEqual(result['holdout_alerts'], [])
        self.assertEqual([d['wav'] for d in result['holdout_detail']], [None, None])
        self.assertIn('No large score drop', log)
        self.assertNotIn('HELD-OUT median', log)       # two sentences: median == mean

    def test_the_result_says_how_it_was_scored(self):
        # From a review of 3.7.1: scores compare only under one definition, and
        # French numbers are matched since 3.7.2.
        with tempfile.TemporaryDirectory() as td:
            result, _, _ = run_optimizer(td, ['--budget', '60'])
        definition = result['score_definition']
        self.assertEqual(set(definition), {'objective', 'w_accent', 'w_identity', 'wer_normalization'})
        self.assertEqual(definition['wer_normalization'], 'fr_cardinal_equivalence_v1')
        self.assertTrue(result['holdout_detail'])
        self.assertTrue(all('wer_raw' in d for d in result['holdout_detail']))

    def test_identical_renders_are_recognised_and_not_scored_twice(self):
        def ignores_rep_pen(kw):                      # rep_pen has no effect on the audio
            return repr(sorted((k, v) for k, v in kw.items() if k != 'repetition_penalty'))
        with tempfile.TemporaryDirectory() as td:
            result, log, whisper = run_optimizer(td, ['--budget', '60', '--no-holdout'],
                                                 audio_key=ignores_rep_pen)
        lines = [l for l in log.splitlines() if '= same audio as [S' in l]
        self.assertGreaterEqual(len(lines), 2)         # rep_pen 4, 7, 10 at the least
        self.assertLess(whisper, result['generations']['sampling'])
        self.assertEqual(result['params']['rep_pen'], 5)


class HoldoutHelperTests(unittest.TestCase):
    def detail(self, accent, heard='ok', detected='fr'):
        return {'accent': accent, 'identity': .66, 'score': .6, 'text': 'ok',
                'heard': heard, 'detected': detected, 'wer': .0}

    def test_alert_rules(self):
        alerts = optimizer.holdout_alerts(.95, [
            self.detail(.85),                          # digits for spelled numbers: fine
            self.detail(.70),                          # 0.25 below the search
            self.detail(.07, detected='en'),
            self.detail(.0, heard='')], 'FR')
        self.assertEqual([k for k, _ in alerts], [2, 3, 4])
        self.assertIn('accent 0.70 against 0.95', alerts[0][1])
        self.assertIn('heard as EN, not FR', alerts[1][1])
        self.assertEqual(alerts[2][1], 'Whisper heard nothing')

    def test_details_show_text_and_transcript(self):
        lines = optimizer.format_holdout_details([dict(self.detail(.5), text='a ' * 100)])
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[1].rstrip().endswith('…'))
        self.assertIn('(Whisper: FR)', lines[0])

    def test_keep_copies_existing_renders_only(self):
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / 'h.wav'
            src.write_bytes(b'RIFF')
            kept = optimizer.keep_holdout_wavs([{'wav': str(src)}, {'wav': str(Path(td) / 'gone.wav')}],
                                               os.path.join(td, 'out', 'x_holdout'))
            self.assertEqual(Path(kept[0]).read_bytes(), b'RIFF')
            self.assertIsNone(kept[1])


class PipelineHoldoutTests(unittest.TestCase):
    def test_unseen_renders_sit_next_to_the_clone_and_alerts_reach_the_recap(self):
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / 'a.wav'
            ref.touch()
            xb = settings.format_block(dict(seed=0))
            ab = '[1, FR, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]'
            seen = {}

            def run(cmd, tag):
                if 'voice_analyser.py' in cmd[1]:
                    return 0, xb + '\n' + ab
                if 'xtts_optimize.py' in cmd[1]:
                    seen['prefix'] = cmd[cmd.index('--holdout-wav-prefix') + 1]
                    Path(cmd[cmd.index('--result-json') + 1]).write_text(json.dumps(dict(
                        schema_version=1, xtts_block=xb, selected_refs=[str(ref)],
                        scores=dict(held_out=.57),
                        holdout_detail=[{'wav': seen['prefix'] + '1.wav'}, {'wav': seen['prefix'] + '2.wav'}],
                        holdout_alerts=[{'sentence': 2, 'reason': 'heard as EN, not FR (accent 0.07)'}],
                        conditioning_confirmation=dict(
                            seeds=[42, 100, 180, 200], base='gcl=30 chunk=4', winner='gcl=30 chunk=4',
                            screening_winner='gcl=20 chunk=10', base_mean=.822, bar=.019, adopted=False,
                            finalists=[dict(cond='gcl=30 chunk=10', screened=.850, mean=.824),
                                       dict(cond='gcl=20 chunk=10', screened=.842, mean=.830)]))))
                    return 0, ''
                seen['clone'] = cmd[cmd.index('--output-optimised') + 1]
                return 0, 'Next []    : ' + ab
            output = io.StringIO()
            with patch.object(pipeline, 'run_stream', run), \
                 patch.object(sys, 'argv', ['xtts_pipeline.py', '--voice', str(ref), 'FR',
                                            '--no-curate', '--no-save-preset']), \
                 contextlib.redirect_stdout(output):
                pipeline.main()
        self.assertEqual(seen['prefix'], os.path.splitext(seen['clone'])[0] + '_holdout')
        recap = output.getvalue()
        self.assertIn('Unseen sentences of the hold-out (raw XTTS, as scored): '
                      + seen['prefix'] + '1.wav', recap)
        self.assertIn('# [!] Unseen sentence 2: heard as EN, not FR (accent 0.07)', recap)
        self.assertIn('# Conditioning confirmation on seeds 42 100 180 200: kept the base gcl=30 '
                      'chunk=4 (best candidate +0.008 over the base, bar 0.019)', recap)
        self.assertIn("# The screening's gcl=20 chunk=10 won on one draw only: not confirmed", recap)


if __name__ == '__main__':
    unittest.main()

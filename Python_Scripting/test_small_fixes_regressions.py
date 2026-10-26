#!/usr/bin/env python3
"""Regression tests for the small 3.6 fixes: joined seeds, the meditation-pace
message, custom search sentences, and the GUI's stop status.

    python -m unittest -v test_small_fixes_regressions
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import search_plan as SP
import xtts_optimize as optimizer
import xtts_settings as settings
from test_clonage_regressions import extract_functions


class JoinedSeedTests(unittest.TestCase):
    def test_the_real_typo_is_flagged(self):
        self.assertEqual(SP.joined_seeds([0, 42, 80, 100, 120, 140, 160180, 200]), [160180])
        self.assertEqual(SP.joined_seeds([7, 42, 4242]), [4242])

    def test_ordinary_seed_lists_are_not(self):
        for seeds in ([0, 42, 75, 100, 130, 155, 160, 180, 200], [1234, 5678],
                      [123456], [0, 1, 2, 5000], [100, 2000], []):
            self.assertEqual(SP.joined_seeds(seeds), [], seeds)


class PaceMessageTests(unittest.TestCase):
    def setUp(self):
        ns = extract_functions('voice_analyser.py', {'pace_speed'}, {'PACE_SPEED_FLOOR': .8})
        self.pace = ns['pace_speed']

    def test_a_fast_voice_is_not_promised_the_target(self):
        speed, reached, needed = self.pace(7.52, 3.5, floor=.8)
        self.assertEqual(speed, .8)
        self.assertAlmostEqual(reached, 6.016, places=3)
        self.assertAlmostEqual(needed, .465, places=3)

    def test_within_range_and_already_slow(self):
        speed, reached, _ = self.pace(4.0, 3.5, floor=.8)
        self.assertEqual(speed, .88)
        self.assertAlmostEqual(reached, 3.52, places=2)
        speed, reached, needed = self.pace(3.0, 3.5, floor=.8)
        self.assertEqual((speed, reached), (1.0, 3.0))
        self.assertGreater(needed, 1.0)

    def test_the_analyser_reports_what_the_speed_gives(self):
        source = (ROOT / 'voice_analyser.py').read_text(encoding='utf-8')
        self.assertNotIn('would give a 3.5/s meditation pace', source)
        self.assertIn('smears the voice', source)


class CustomSearchSentenceTests(unittest.TestCase):
    def search_texts(self, options):
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / 'a.wav'
            ref.touch()
            probe = Path(td) / 'probe.txt'
            probe.write_text('# my script\nPremière phrase du script.\nDeuxième phrase.\n',
                             encoding='utf-8')
            out = Path(td) / 'w.json'

            class Pron:
                def __init__(self, **kw):
                    pass

                def score(self, path, **kw):
                    return {'score': .9, 'text': kw.get('target_text'), 'detected': 'fr'}

            class Encoder:
                def __init__(self, **kw):
                    pass

                def embed(self, path):
                    return np.array([1., 0.])

                def cosine(self, a, b):
                    return .7
            mocks = {'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
                     'xtts_clone': types.SimpleNamespace(load_xtts=lambda d: (None, None),
                                                        compute_latents=lambda m, p, **kw: kw,
                                                        generate=lambda *a, **kw: None),
                     'pron_score': types.SimpleNamespace(PronScorer=Pron),
                     'speaker_identity': types.SimpleNamespace(SpeakerEncoder=Encoder)}
            argv = ['xtts_optimize.py', str(ref), 'FR', '--xtts-block',
                    settings.format_block(dict(seed=0, temp=.62)), '--seeds', '0 42',
                    '--device', 'cpu', '--no-search-conditioning', '--no-holdout',
                    '--budget', '60', '--probe-text-file', str(probe),
                    '--result-json', str(out)] + options
            with patch.dict(sys.modules, mocks), patch.object(sys, 'argv', argv), \
                 contextlib.redirect_stdout(io.StringIO()):
                optimizer.main()
            return json.loads(out.read_text())['search_texts']

    def test_the_file_gives_the_search_sentences(self):
        self.assertEqual(self.search_texts([]), ['Première phrase du script.', 'Deuxième phrase.'])

    def test_an_explicit_text_still_comes_first(self):
        self.assertEqual(self.search_texts(['--text', 'Bonjour.']),
                         ['Bonjour.', 'Première phrase du script.'])


class StopStatusTests(unittest.TestCase):
    """The GUI's real run_cmd, with the Tk calls replaced: Stop kills the
    child (SIGKILL, code -9) and that used to be reported as an error."""

    def run_and_stop(self, stop):
        try:
            import xtts_studio as studio
        except Exception as exc:                 # pragma: no cover - no tkinter
            self.skipTest(f'xtts_studio not importable: {exc}')
        import time
        messages = []
        button = types.SimpleNamespace(config=lambda **kw: None, _orig_text='Run')
        stopper = types.SimpleNamespace(config=lambda **kw: None)
        with patch.object(studio, '_root', None), \
             patch.object(studio, 'log', lambda console, text: messages.append(text)):
            studio.run_cmd([sys.executable, '-c', 'import time; time.sleep(20)'
                            if stop else 'raise SystemExit(3)'], None, button, stopper)
            deadline = time.time() + 15
            if stop:
                while not any(m.startswith('>') for m in messages) and time.time() < deadline:
                    time.sleep(0.05)
                time.sleep(0.3)
                stopper._stop_fn()
            while not any(('Stopped' in m or 'Error' in m) for m in messages) \
                    and time.time() < deadline:
                time.sleep(0.05)
        return '\n'.join(messages)

    def test_a_requested_stop_is_not_reported_as_an_error(self):
        log = self.run_and_stop(stop=True)
        self.assertIn('[STOP] Stopped.', log)
        self.assertNotIn('Error (code', log)

    def test_a_real_failure_still_is(self):
        self.assertIn('[ERR] Error (code 3)', self.run_and_stop(stop=False))


if __name__ == '__main__':
    unittest.main()

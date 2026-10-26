#!/usr/bin/env python3
"""Regression tests for 3.6.1: the Validator renders what the generator
renders, and {} blocks inherit from their own voice.

A seed chosen in the Validator did not reproduce in the script: the generator
strips the final punctuation before rendering, the Validator did not. And a
short {2, 17} took the settings of the voice in use instead of voice 2's;
inside [parallel], every {} written before its [] was discarded.

    python -m unittest -v test_script_blocks_regressions
"""
import importlib.machinery
from pathlib import Path
import re
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import voice_validator as validator
import xtts_settings as settings
from test_clonage_regressions import extract_functions, invoke
from test_render_search_regressions import generator_namespace

GENERATOR = 'guided_meditation_generator_v23.py'
real_clean_text = extract_functions(GENERATOR, {'clean_text'}, {'re': re})['clean_text']


class Audio:
    @classmethod
    def silent(cls, **kw): return cls()
    @classmethod
    def from_wav(cls, path): return cls()
    def __add__(self, other): return self
    def __len__(self): return 1000
    def export(self, path, **kw): Path(path).write_bytes(b'simulated audio')


def run_validator(text, generator_has_clean_text=True):
    """The Validator with XTTS simulated; returns (texts sent to XTTS, log)."""
    with tempfile.TemporaryDirectory() as td:
        ref = Path(td) / 'reference.wav'
        ref.touch()
        texts = []
        xc = types.SimpleNamespace(load_xtts=lambda d: (None, None),
                                   compute_latents=lambda *a, **kw: (None, None),
                                   generate=lambda model, text, *a, **kw: texts.append(text))
        mocks = {'xtts_clone': xc, 'pydub': types.SimpleNamespace(AudioSegment=Audio),
                 'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))}

        class Loader:
            def create_module(self, spec): return None
            def exec_module(self, module):
                module.render_voice_chain = lambda audio, *args: audio
                if generator_has_clean_text:
                    module.clean_text = real_clean_text
        spec = importlib.machinery.ModuleSpec('gmg', Loader())
        with patch.dict(sys.modules, mocks), \
             patch('importlib.util.spec_from_file_location', return_value=spec):
            log = invoke(validator, [str(ref), 'FR', '--param', 'seed', '--values', '17', '80',
                                     '--xtts-block', settings.format_block({'seed': 0}),
                                     '--text', text, '--no-score', '--no-labels',
                                     '--output', str(Path(td) / 'out.wav')])
        return texts, log


class ValidatorTextTests(unittest.TestCase):
    def test_the_validator_renders_the_text_the_generator_renders(self):
        texts, log = run_validator("Salut Stéphane. C'est Élodie.")
        self.assertEqual(texts, ["Salut Stéphane. C'est Élodie"] * 2)
        self.assertIn("Text as the generator renders it: Salut Stéphane. C'est Élodie", log)
        self.assertEqual(real_clean_text("Salut Stéphane. C'est Élodie."), texts[0])

    def test_pitch_marks_are_removed_as_in_the_script(self):
        texts, _ = run_validator("Respire [p:+2] profondément{+2}.")
        self.assertEqual(texts[0], real_clean_text("Respire [p:+2] profondément{+2}."))
        self.assertNotIn('[p:', texts[0])

    def test_an_older_generator_leaves_the_text_and_says_so(self):
        texts, log = run_validator("Bonjour.", generator_has_clean_text=False)
        self.assertEqual(texts[0], "Bonjour.")
        self.assertIn('The generator has no clean_text', log)


def block_namespace(**extra):
    return generator_namespace(
        {'xtts_block_voice', 'parse_xtts_params', 'xtts_block_params', 'DEFAULT_XTTS_PARAMS'},
        generator_params=settings.generator_params, **extra)


class ShortBlockTests(unittest.TestCase):
    V1 = dict(settings.generator_params('{1, 80, 0, 200, 100, 250, 0.65, 50, 0.85, 5, 1, 45, 4}', {}))
    V2 = dict(settings.generator_params('{2, 42, 0, 300, 100, 250, 0.72, 55, 0.88, 4.5, 1, 51, 6}', {}))

    def test_a_short_block_keeps_its_own_voice_settings(self):
        ns = block_namespace()
        voice, params = ns['xtts_block_params']('{2, 17}', {1: self.V1, 2: self.V2},
                                                ns['DEFAULT_XTTS_PARAMS'])
        self.assertEqual(voice, 2)
        self.assertEqual(params['seed'], 17)
        for key in ('temperature', 'top_k', 'repetition_penalty', 'gpt_cond_len',
                    'gpt_cond_chunk_len', 'trim_end'):
            self.assertEqual(params[key], self.V2[key], key)

    def test_a_voice_not_seen_yet_starts_from_the_globals(self):
        ns = block_namespace()
        default = ns['DEFAULT_XTTS_PARAMS']
        _voice, params = ns['xtts_block_params']('{3, 5}', {1: self.V1}, default)
        self.assertEqual(params['temperature'], default['temperature'])
        _voice, params = ns['xtts_block_params']('{3, 5}', {0: self.V2, 1: self.V1}, default)
        self.assertEqual(params['temperature'], self.V2['temperature'])

    def test_the_script_loop_uses_it(self):
        source = (ROOT / GENERATOR).read_text(encoding='utf-8')
        self.assertEqual(source.count('xtts_block_params('), 3)   # definition + 2 call sites
        self.assertNotIn('xtts_params_by_voice.get(current_voice, xtts_params_by_voice', source)
        self.assertNotIn('xtts_params_by_voice.get(cur_voice, xtts_params_by_voice', source)


class Seg:
    """Just enough of pydub.AudioSegment for the parallel mixer."""
    def __init__(self, ms=0): self.ms = ms
    @staticmethod
    def silent(duration=0): return Seg(duration)
    @staticmethod
    def empty(): return Seg(0)
    def __add__(self, other): return Seg(self.ms + other.ms)
    def __len__(self): return self.ms
    def overlay(self, other, position=0): return Seg(max(self.ms, position + other.ms))


class ParallelBlockTests(unittest.TestCase):
    def test_blocks_written_inside_apply_to_their_text(self):
        rendered = []

        def generate(clean, vn, voice_files, tts, cfg, xp, temp_file):
            rendered.append((clean, vn, xp['seed'], xp['temperature']))
            return Seg(1000)
        ns = generator_namespace(
            {'process_parallel_block', 'xtts_block_voice', 'parse_xtts_params',
             'xtts_block_params', 'parse_voice_config', 'is_params_xtts_bracket',
             'clean_text', 'DEFAULT_AUDIO_CONFIG', 'DEFAULT_XTTS_PARAMS'},
            AudioSegment=Seg, generate_sentence_audio=generate,
            generator_params=settings.generator_params, _voice_last_config={},
            time=time, re=re)
        outer = {1: ShortBlockTests.V1, 2: ShortBlockTests.V2}
        inner = ['{1, 7, 0, 200, 100, 250, 0.6}',            # the README's layout:
                 '[1, FR, 0.9, 0]',                          # {} first, then []
                 'Première voix.',
                 '{2, 9}',
                 '[2, FR, 0.9, 0]',
                 'Deuxième voix.',
                 '[1, FR, 0.9, 0]',
                 'Encore la première.',
                 '{1, 11}',                                  # short: from the block's {1}
                 'Toujours la première.']
        ns['process_parallel_block'](inner, ['a.wav', 'b.wav'], {}, outer, [1000], 'tmp.wav')
        self.assertEqual(rendered, [
            ('Première voix', 1, 7, 0.6),
            ('Encore la première', 1, 7, 0.6),                # the block's {} still holds
            ('Toujours la première', 1, 11, 0.6),
            ('Deuxième voix', 2, 9, ShortBlockTests.V2['temperature'])])
        self.assertEqual(outer[1]['seed'], 80)                # local to the block, as before


if __name__ == '__main__':
    unittest.main()

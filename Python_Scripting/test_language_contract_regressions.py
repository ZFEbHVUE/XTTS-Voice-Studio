"""Language contradictions must fail before any model is loaded (from a
review of 3.7.1), and only real ones: a language counts when it is given (3.7.2)."""
import contextlib
import io
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import voice_comparator
import voice_validator
from xtts_settings import format_block

class LanguageContractTests(unittest.TestCase):
    def check_conflict(self, module, extra):
        with tempfile.TemporaryDirectory() as folder:
            ref = Path(folder)/"ref.wav"
            ref.touch()
            argv = [module.__file__, str(ref), "FR", "--xtts-block", format_block({"seed":0}),
                    "--audio-block", "[1, EN, 1, 0, 0, 0, 0, 85, 7500, 0, 0, 0, 0, 0, 0, 1]", *extra]
            stderr = io.StringIO()
            with patch.object(sys, "argv", argv), contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    module.main()
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("Language mismatch", stderr.getvalue())
    def test_comparator(self):
        self.check_conflict(voice_comparator, [])
    def test_validator(self):
        self.check_conflict(voice_validator, ["--param", "seed", "--values", "0"])


class Stop(Exception):
    """Raised where XTTS would load: the language is settled by then."""


def stop(*args, **kwargs):
    raise Stop


class LanguageResolutionTests(unittest.TestCase):
    def test_only_two_given_languages_can_disagree(self):
        resolve = voice_comparator.resolve_language
        self.assertEqual(resolve('EN', '[1, 1, 0, 0]'), ('EN', None))   # the block names none
        self.assertEqual(resolve(None, '[1, EN, 1, 0]'), ('EN', None))
        self.assertEqual(resolve('en', '[1, EN, 1, 0]'), ('EN', None))
        self.assertEqual(resolve(None, '[1, 1, 0, 0]'), ('FR', None))
        self.assertEqual(resolve(None, None), ('FR', None))
        language, error = resolve('FR', '[1, EN, 1, 0]')
        self.assertIsNone(language)
        self.assertIn('Language mismatch', error)
        self.assertIsNone(voice_comparator.block_language('[1, 1, 0, 0]'))
        self.assertEqual(voice_comparator.block_language('[1, zh-cn, 1]'), 'ZH-CN')

    def language_used(self, module, tail):
        with tempfile.TemporaryDirectory() as folder:
            ref = Path(folder)/"ref.wav"
            ref.touch()
            modules = {'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
                       'xtts_clone': types.SimpleNamespace(load_xtts=stop),
                       'pydub': types.SimpleNamespace(AudioSegment=object)}
            out = io.StringIO()
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", [module.__file__, str(ref), *tail]), \
                 contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(Stop):
                    module.main()
        return re.search(r"Language\s*:\s*(\S+)", out.getvalue()).group(1), out.getvalue()

    def test_comparator_keeps_the_asked_language_when_the_block_names_none(self):
        # A [] without a language used to impose FR on an English request.
        language, _ = self.language_used(voice_comparator, [
            "EN", "--xtts-block", format_block({"seed": 0}), "--audio-block", "[1, 1, 0, 0, 0, 0, 95, 0]",
            "--text", "Hello there.", "--device", "cpu"])
        self.assertEqual(language, "EN")

    def test_validator_takes_the_block_language_when_none_is_asked(self):
        # Its FR default is no request: the review's check refused this block.
        language, log = self.language_used(voice_validator, [
            "--xtts-block", format_block({"seed": 0}), "--audio-block", "[1, EN, 1, 0, 0, 0, 0, 95, 0]",
            "--param", "seed", "--values", "0", "--text", "Hello there.", "--device", "cpu"])
        self.assertEqual(language, "EN")
        self.assertNotIn("defaulting to FR", log)

"""3.10.8: the coqui-tts fork (0.28) as a second environment.

Running the project's XTTS calls on TTS 0.22.0 and on coqui-tts 0.28 showed:
  - beam search (num_beams > 1) fails on 0.28 with its transformers 4.57: XTTS
    gives the generation config its longest audio sequence as vocab_size, and
    beam search now reshapes the scores with it. xtts_clone sizes it by the
    1026 codes the head scores;
  - coqui-tts no longer installs spacy (sentence splitting) and imports six
    without declaring it: check_engine stops early on the first,
    requirements-coqui-tts.txt lists both;
  - the model comes from Hugging Face, and a failed download deletes the folder
    TTS 0.22 keeps its own copy in: install.sh --coqui gives the new
    environment its own TTS_HOME.

The library tests build a tiny XTTS with random weights (nothing downloaded)
and run xtts_clone through the real XTTS code. They need torch and an XTTS
package, and are skipped without them: run them in both environments.

    python -m unittest test_3108_coqui_tts -v
"""
import argparse
import contextlib
import importlib
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import xtts_clone as XC
import engine_ab as AB
import voice_presets as VP

try:
    import soundfile as sf
except ImportError:
    sf = None


def _have(*modules):
    try:
        for m in modules:
            importlib.import_module(m)
        return True
    except Exception:
        return False


# ── The beam-search workaround, without XTTS ─────────────────────────────────

class FakeXtts:
    """An XTTS with the attributes _beam_vocabulary reads, and an inference
    that records the vocab_size it is called with."""
    def __init__(self, vocab=608, codes=1026):
        config = types.SimpleNamespace(vocab_size=vocab)
        self.gpt = types.SimpleNamespace(mel_head=types.SimpleNamespace(out_features=codes),
                                         gpt_inference=types.SimpleNamespace(config=config))
        self.config = types.SimpleNamespace(audio=types.SimpleNamespace(output_sample_rate=24000))
        self.seen = []

    def inference(self, text, lang, gpt_cond_latent, speaker_embedding, **kw):
        self.seen.append((self.gpt.gpt_inference.config.vocab_size, kw))
        return {'wav': np.full(2400, 0.01, dtype=np.float32)}


@unittest.skipIf(sf is None, 'soundfile is needed')
class BeamVocabularyTests(unittest.TestCase):
    def generate(self, model, **kw):
        with tempfile.TemporaryDirectory() as td:
            XC.generate(model, 'Bonjour.', 'fr', (None, None), str(Path(td, 'o.wav')), **kw)

    def test_beam_search_is_sized_by_the_audio_codes(self):
        model = FakeXtts()
        self.generate(model, num_beams=3)
        self.assertEqual(model.seen[-1][0], 1026)
        self.assertEqual(model.seen[-1][1]['num_beams'], 3)
        self.assertFalse(model.seen[-1][1]['do_sample'])        # beam search stays deterministic

    def test_sampling_is_left_alone(self):
        model = FakeXtts()
        self.generate(model)
        self.assertEqual(model.seen[-1][0], 608)

    def test_nothing_to_do_or_nothing_to_read(self):
        model = FakeXtts(vocab=1026)
        XC._beam_vocabulary(model)
        self.assertEqual(model.gpt.gpt_inference.config.vocab_size, 1026)
        XC._beam_vocabulary(types.SimpleNamespace())             # no gpt at all
        XC._beam_vocabulary(types.SimpleNamespace(gpt=types.SimpleNamespace(mel_head=None)))


class EngineTests(unittest.TestCase):
    def versions(self, installed):
        from importlib import metadata

        def version(name):
            if name in installed:
                return installed[name]
            raise metadata.PackageNotFoundError(name)
        with patch.object(metadata, 'version', version):
            return XC.engine_versions()

    def test_the_package_is_named(self):
        both = {'coqui-tts': '0.28.0', 'transformers': '4.57.6', 'torch': '2.5.1'}
        self.assertEqual(self.versions(both), dict(package='coqui-tts', version='0.28.0',
                                                   transformers='4.57.6', torch='2.5.1'))
        self.assertEqual(self.versions({'TTS': '0.22.0', 'transformers': '4.33.0'})['package'], 'TTS')
        none = self.versions({})
        self.assertEqual((none['package'], none['torch']), (None, None))
        self.assertEqual(XC.engine_label(dict(package='TTS', version='0.22.0',
                                              transformers='4.33.0', torch='2.5.1')),
                         'TTS 0.22.0 (transformers 4.33.0, torch 2.5.1)')

    def test_no_spacy_stops_before_the_model_loads(self):
        real = importlib.util.find_spec
        with patch('importlib.util.find_spec', lambda name, *a: None if name == 'spacy' else real(name, *a)):
            with self.assertRaises(RuntimeError) as caught:
                XC.check_engine(verbose=False)
        self.assertIn('pip install spacy', str(caught.exception))
        with patch('importlib.util.find_spec', lambda name, *a: object()), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            info = XC.check_engine()
        self.assertIn('[*] XTTS engine:', out.getvalue())
        self.assertIn('package', info)


# ── The real XTTS code, on a tiny model with random weights ──────────────────

def tiny_xtts(td):
    """XTTS-v2's structure (perceiver latents, 1026 audio codes) in miniature,
    with a character vocabulary written with tokenizers: nothing is downloaded."""
    import torch
    from tokenizers import Tokenizer, models
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
    vocab = {'[STOP]': 0, '[UNK]': 1, '[SPACE]': 2}
    for c in sorted(set("abcdefghijklmnopqrstuvwxyzàâäçéèêëîïôöùûüÿœæ'.,;:!?-0123456789")):
        vocab.setdefault(c, len(vocab))
    for t in ('[START]', '[fr]', '[en]'):
        vocab[t] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token='[UNK]'))
    tok.add_special_tokens(['[STOP]', '[UNK]', '[SPACE]', '[START]', '[fr]', '[en]'])
    path = os.path.join(td, 'vocab.json')
    tok.save(path)
    torch.manual_seed(0)
    config = XttsConfig()
    a = config.model_args
    a.gpt_layers, a.gpt_n_model_channels, a.gpt_n_heads = 1, 64, 2
    a.decoder_input_dim = 64
    a.gpt_max_audio_tokens = 40
    a.gpt_use_perceiver_resampler = True
    a.gpt_num_audio_tokens, a.gpt_start_audio_token, a.gpt_stop_audio_token = 1026, 1024, 1025
    model = Xtts(config)
    model.tokenizer = VoiceBpeTokenizer(vocab_file=path)
    model.init_models()
    model.gpt.init_gpt_for_inference(kv_cache=a.kv_cache, use_deepspeed=False)
    model.eval()                     # XTTS's own eval() returns nothing
    return model


@unittest.skipUnless(_have('torch', 'TTS.tts.models.xtts', 'tokenizers') and sf is not None,
                     'needs torch and an XTTS package (TTS 0.22 or coqui-tts)')
class RealXttsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.mkdtemp()
        cls.model = tiny_xtts(cls.td)
        sr = 22050
        t = np.arange(5 * sr) / sr
        cls.refs = []
        for i, f in enumerate((180, 240)):
            y = 0.3 * np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
            p = os.path.join(cls.td, f'ref{i}.wav')
            sf.write(p, y.astype(np.float32), sr)
            cls.refs.append(p)
        cls.latents = XC.compute_latents(cls.model, cls.refs, gpt_cond_len=6, gpt_cond_chunk_len=3,
                                         max_ref_len=4, sound_norm_refs=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td, ignore_errors=True)

    def render(self, name, **kw):
        out = os.path.join(self.td, name + '.wav')
        XC.generate(self.model, 'Bonjour. Respire doucement, et laisse aller.', 'fr',
                    self.latents, out, temperature=0.7, enable_text_splitting=True, **kw)
        y, sr = sf.read(out)
        self.assertEqual(sr, 24000)
        self.assertTrue(len(y) and np.isfinite(y).all())
        return y

    def test_conditioning(self):
        gpt, speaker = self.latents
        self.assertEqual(tuple(gpt.shape), (1, 32, 64))          # 32 perceiver latents
        self.assertEqual(tuple(speaker.shape), (1, 512, 1))
        capped = XC.compute_latents(self.model, self.refs, gpt_cond_len=6, gpt_cond_chunk_len=3,
                                    max_ref_len=4, speaker_ref_len=2, ref_trim_db=30)
        self.assertTrue(np.array_equal(capped[0].cpu().numpy(),
                                       XC.compute_latents(self.model, self.refs, gpt_cond_len=6,
                                                          gpt_cond_chunk_len=3, max_ref_len=4,
                                                          ref_trim_db=30)[0].cpu().numpy()))

    def test_a_seed_gives_the_same_audio_again(self):
        self.assertTrue(np.array_equal(self.render('a', seed=3), self.render('b', seed=3)))
        other = self.render('c', seed=4)
        self.assertFalse(np.array_equal(self.render('d', seed=3)[:len(other)], other[:len(other)]))

    def test_beam_greedy_and_typical_decoding(self):
        self.render('beam', seed=3, num_beams=2, do_sample=False)
        self.assertEqual(self.model.gpt.gpt_inference.config.vocab_size, 1026)
        self.render('greedy', seed=3, do_sample=False)
        self.render('typical', seed=3, typical_p=0.9)


# ── engine_ab.py, XTTS simulated ─────────────────────────────────────────────

class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = self.values.shape

    def detach(self):
        return self

    cpu = contiguous = detach

    def numpy(self):
        return self.values


class FakeEngine:
    """xtts_clone as engine_ab uses it: renders depend on text and seed, and
    on `shift` (another install that renders differently)."""
    def __init__(self, label='TTS', shift=0.0, weights=1.0):
        self.label, self.shift, self.weights, self.generated = label, shift, weights, []

    def engine_versions(self):
        return dict(package=self.label, version='9', transformers='4', torch='2')

    def engine_label(self, info=None):
        return XC.engine_label(info or self.engine_versions())

    def load_xtts(self, device):
        model = types.SimpleNamespace(state_dict=lambda: {
            'gpt.w': FakeTensor([1.0, 2.0]), 'decoder.w': FakeTensor([self.weights])})
        tts = types.SimpleNamespace(synthesizer=types.SimpleNamespace(tts_checkpoint='/m/model.pth'))
        return tts, model

    def compute_latents(self, model, refs, **kw):
        self.latents = (refs, kw)
        return ('gpt', 'speaker')

    def generate(self, model, text, lang, latents, out, seed=None, **kw):
        self.generated.append(dict(text=text, lang=lang, seed=seed, **kw))
        rng = np.random.default_rng(abs(hash((text, seed))) % 2 ** 32)
        y = 0.1 * rng.standard_normal(4800) + self.shift
        sf.write(out, y.astype(np.float32), 24000)


@unittest.skipIf(sf is None, 'soundfile is needed')
class EngineAbTests(unittest.TestCase):
    BLOCK = '{1, 7, 0, 200, 100, 250, 0.6, 40, 0.9, 4, 1, 24, 4, 0, 1, 1, 24, 0, 0, 1}'

    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.dir = Path(td.name)
        self.ref = str(self.dir / 'ref.wav')
        sf.write(self.ref, np.zeros(2400, dtype=np.float32), 24000)
        p = patch.dict(sys.modules, {'torch': types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False))})
        p.start()
        self.addCleanup(p.stop)

    def render(self, out, engine, *extra):
        args = argparse.Namespace(
            preset=None, xtts_block=self.BLOCK, refs=[self.ref], lang='FR', text_file=None,
            seeds=[1, 2], out=str(self.dir / out), device=None)
        for k, v in extra:
            setattr(args, k, v)
        with contextlib.redirect_stdout(io.StringIO()):
            return AB.render(args, XC=engine)

    def compare(self, a, b, identity=False, encoder=None):
        args = types.SimpleNamespace(a=str(self.dir / a), b=str(self.dir / b), identity=identity)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            result = AB.compare(args, encoder=encoder)
        return result, out.getvalue()

    def test_render_uses_the_block_its_references_and_unseen_sentences(self):
        engine = FakeEngine()
        manifest = self.render('a', engine)
        self.assertEqual(engine.latents[0], [self.ref])
        self.assertEqual(engine.latents[1]['gpt_cond_len'], 24)
        self.assertEqual(engine.latents[1]['max_ref_len'], 24)
        # The block's own seed first, then --seeds; four sentences of the language.
        self.assertEqual([g['seed'] for g in engine.generated[:3]], [7, 1, 2])
        import probe_texts as PT
        texts = PT.final_check_texts('FR') + PT.holdout_texts('FR', 2)
        self.assertEqual(sorted({g['text'] for g in engine.generated}), sorted(texts))
        self.assertEqual(len(manifest['files']), 12)
        self.assertEqual((engine.generated[0]['temperature'], engine.generated[0]['top_k']), (0.6, 40))
        saved = json.loads((self.dir / 'a' / 'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['engine']['package'], 'TTS')
        self.assertEqual(set(saved['weights']), {'gpt.w', 'decoder.w'})
        self.assertEqual(saved['checkpoint'], '/m/model.pth')

    def test_a_preset_brings_its_block_references_and_language(self):
        with patch.object(VP, 'PRESET_DIR', str(self.dir / 'presets')):
            VP.save('lea', self.BLOCK, '[1, EN, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]',
                    reference=self.ref, references=[self.ref])
            engine = FakeEngine()
            self.render('p', engine, ('preset', 'lea'), ('xtts_block', None), ('refs', None),
                        ('lang', None))
        self.assertEqual(engine.generated[0]['lang'], 'EN')
        self.assertEqual(engine.latents[0], [self.ref])

    def test_same_weights_and_audio(self):
        self.render('a', FakeEngine('TTS'))
        self.render('b', FakeEngine('coqui-tts'))
        result, out = self.compare('a', 'b')
        self.assertEqual(result['verdict'], 'same')
        self.assertIn('the migration changes nothing', out)
        self.assertNotIn('listening_test.py', out)

    def test_other_audio_or_other_weights(self):
        self.render('a', FakeEngine())
        self.render('b', FakeEngine(shift=0.01))
        self.render('c', FakeEngine(weights=2.0))
        result, out = self.compare('a', 'b')
        self.assertEqual(result['verdict'], 'audio')
        self.assertIn('listening_test.py', out)
        self.assertIn('--reference', out)
        result, out = self.compare('a', 'c')
        self.assertEqual(result['verdict'], 'weights')
        self.assertIn('1 different (first: decoder.w)', out)

    def test_identity_of_each_side(self):
        self.render('a', FakeEngine())
        self.render('b', FakeEngine(shift=0.01))
        encoder = types.SimpleNamespace(embed=lambda y, sr: np.array([1.0, float(np.mean(y)) * 10]))
        result, out = self.compare('a', 'b', identity=True, encoder=encoder)
        self.assertIn('Identity vs ref.wav: A ', out)
        self.assertIn('B - A', out)

    def test_bad_input_stops(self):
        with self.assertRaises(SystemExit):
            self.render('x', FakeEngine(), ('refs', [str(self.dir / 'gone.wav')]))
        with self.assertRaises(SystemExit):
            self.render('x', FakeEngine(), ('xtts_block', None))


# ── install.sh --coqui, with conda, pip and python simulated ─────────────────

@unittest.skipUnless(shutil.which('bash'), 'bash is needed')
class InstallScriptTests(unittest.TestCase):
    def run_install(self, *options):
        td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, td, True)
        log = os.path.join(td, 'calls.log')
        bin_dir = os.path.join(td, 'bin')
        base = os.path.join(td, 'conda')
        os.makedirs(bin_dir)
        os.makedirs(os.path.join(base, 'etc', 'profile.d'))
        Path(base, 'etc', 'profile.d', 'conda.sh').write_text(':\n')
        scripts = {
            'conda': f'''#!/bin/sh
echo "conda $*" >> "{log}"
if [ "$1 $2" = "info --base" ]; then echo "{base}"; fi
if [ "$1" = "--version" ]; then echo "conda 0"; fi
exit 0
''',
            'pip': f'#!/bin/sh\necho "pip $*" >> "{log}"\nexit 0\n',
            'python': f'''#!/bin/sh
if [ "$1" = "-" ]; then cat > /dev/null; echo "python-check $CHECK_MODULES" >> "{log}"; fi
if [ "$1" = "--version" ]; then echo "Python 3"; fi
exit 0
''',
        }
        for name, body in scripts.items():
            path = os.path.join(bin_dir, name)
            Path(path).write_text(body)
            os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        env = dict(os.environ, PATH=bin_dir + os.pathsep + '/usr/bin:/bin', HOME=td)
        proc = subprocess.run(['bash', str(ROOT.parent / 'install.sh'), *options],
                              capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        return Path(log).read_text(), td

    def test_coqui_makes_a_second_environment(self):
        calls, home = self.run_install('--coqui')
        self.assertIn('conda create -y -n xtts028 python=3.11', calls)
        self.assertIn('requirements-coqui-tts.txt', calls)
        self.assertIn('pip install --quiet torch==2.5.1 torchaudio==2.5.1 --index-url '
                      'https://download.pytorch.org/whl/cu121', calls)
        self.assertIn(f'conda env config vars set TTS_HOME={home}/.local/share/tts-coqui -n xtts028',
                      calls)
        self.assertIn('conda env config vars set COQUI_TOS_AGREED=1 -n xtts028', calls)
        self.assertIn('spacy six', calls)
        launcher = Path(home, '.local', 'bin', 'xtts-studio-coqui')
        self.assertIn('conda activate "xtts028"', launcher.read_text())
        self.assertFalse(Path(home, '.local', 'bin', 'xtts-studio').exists())
        desktop = Path(home, '.local', 'share', 'applications', 'xtts-studio-coqui.desktop')
        self.assertIn('Name=XTTS Voice Studio (coqui-tts)', desktop.read_text())

    def test_the_default_is_unchanged(self):
        calls, home = self.run_install()
        self.assertIn('conda create -y -n xtts python=3.10', calls)
        self.assertIn('/requirements.txt', calls)
        self.assertNotIn('TTS_HOME', calls)
        self.assertNotIn('spacy six', calls)
        self.assertTrue(Path(home, '.local', 'bin', 'xtts-studio').exists())


class RequirementsTests(unittest.TestCase):
    def test_the_two_requirement_files(self):
        coqui = (ROOT.parent / 'requirements-coqui-tts.txt').read_text(encoding='utf-8')
        for line in ('coqui-tts==0.28.0', 'transformers==4.57.6', 'spacy>=3.8,<3.9', 'six>=1.16'):
            self.assertIn('\n' + line + '\n', coqui)
        self.assertNotIn('\nTTS==', coqui)
        old = (ROOT.parent / 'requirements.txt').read_text(encoding='utf-8')
        self.assertIn('\nTTS==0.22.0\n', old)
        self.assertIn('\ntransformers==4.33.0\n', old)


if __name__ == '__main__':
    unittest.main(verbosity=2)

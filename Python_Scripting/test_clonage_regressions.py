"""Offline regression tests: python3 -m unittest -v test_clonage_regressions.

Real parsers, CLI entry points and GUI callbacks are exercised. Audio models
are simulated: these tests do not assess voice quality or GPU compatibility.
"""
import ast
import contextlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import xtts_settings as settings
import xtts_clone as clone
import xtts_optimize as optimizer
import voice_validator as validator
import voice_comparator as comparator
import xtts_pipeline as pipeline
import voice_presets as presets


def extract_functions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


def invoke(module, argv):
    output = io.StringIO()
    with patch.object(sys, 'argv', [module.__file__] + argv), contextlib.redirect_stdout(output):
        module.main()
    return output.getvalue()


class SettingsTests(unittest.TestCase):
    def test_historical_formats_and_roundtrip(self):
        full = '{1, 0, 0, 200, 100, 250, 0.65, 50, 0.85, 5, 1, 30, 4, 0, 3}'
        tokens = full.strip('{}').split(',')
        for count in (9, 11, 14, 15):
            with self.subTest(count=count):
                original = settings.parse_block('{' + ','.join(tokens[:count]) + '}')
                encoded = settings.format_block(original)
                self.assertEqual(settings.validate(original), settings.validate(settings.parse_block(encoded)))

    def test_exports_keep_decoder_and_reference_duration(self):
        p = settings.validate(dict(seed=0, num_beams=3, do_sample=0, max_ref_len=45))
        outputs = [optimizer.format_xtts_block(0, p), validator.format_xtts_block(p),
                   comparator.format_blocks(1, p, {}, 'FR')[0]]
        for output in outputs:
            self.assertEqual(settings.validate(settings.parse_block(output)), p)

    def test_greedy_and_sampling_have_different_exports(self):
        self.assertNotEqual(settings.format_block({'do_sample': 0}),
                            settings.format_block({'do_sample': 1}))

    def test_seed_zero_none_and_decimal_without_leading_zero(self):
        self.assertEqual(settings.parse_block('{1,0,,,,,.65}')['seed'], 0)
        self.assertEqual(settings.parse_block('{1,0,,,,,.65}')['temp'], .65)
        self.assertIsNone(settings.parse_block('{1,none}')['seed'])

    def test_invalid_values(self):
        for params in ({'seed': -1}, {'seed': 2**32}, {'top_p': 1.5},
                       {'do_sample': 2}, {'sound_norm_refs': .5},
                       {'temp': float('nan')}, {'max_ref_len': 0},
                       {'gpt_cond_len': 4, 'gpt_cond_chunk_len': 6}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                settings.validate(params)

    def test_generator_resets_boolean_and_keeps_new_fields(self):
        ns = extract_functions('guided_meditation_generator_v23.py',
                               {'parse_xtts_params', '_apply_raw_params'},
                               dict(generator_params=settings.generator_params,
                                    ALIASES=settings.ALIASES, LONG_NAMES=settings.LONG_NAMES,
                                    validate=settings.validate))
        base = dict(temperature=.7, repetition_penalty=5, length_penalty=1,
                    seed=123, sound_norm_refs=1)
        block = settings.format_block(dict(seed=0, sound_norm_refs=0,
                                           do_sample=0, max_ref_len=45))
        voice, parsed = ns['parse_xtts_params'](block, base)
        self.assertEqual(voice, 1)
        self.assertEqual([parsed[k] for k in ('seed', 'sound_norm_refs', 'do_sample', 'max_ref_len')],
                         [0, 0, 0, 45])
        raw = ns['_apply_raw_params'](base, {'seed': 'none', 'sound_norm_refs': '0',
                                             'num_beams': '3', 'max_ref_len': '45'})
        self.assertIsNone(raw['seed'])
        self.assertEqual(raw['do_sample'], 0)


class Model:
    def __init__(self):
        self.calls = []
        self.config = types.SimpleNamespace(audio={'output_sample_rate': 24000})
        self.wave = np.array([.1, -.1], dtype=np.float32)

    def get_conditioning_latents(self, **kw):
        self.calls.append(kw)
        return ('gpt', 'speaker')

    def inference(self, *args, **kw):
        self.calls.append((args, kw))
        return {'wav': self.wave}


class CoreTests(unittest.TestCase):
    def test_seed_zero_seeds_all_cpu_random_generators(self):
        calls = []
        torch = types.SimpleNamespace(manual_seed=calls.append,
                                      cuda=types.SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {'torch': torch}):
            clone.set_seed(0)
            first = (random.random(), np.random.random())
            clone.set_seed(0)
            self.assertEqual(first, (random.random(), np.random.random()))
            clone.set_seed(None)
        self.assertEqual(calls, [0, 0])

    def test_conditioning_passes_exact_reference_settings(self):
        with tempfile.TemporaryDirectory() as td:
            refs = [Path(td) / 'a.wav', Path(td) / 'b.wav']
            for ref in refs:
                ref.touch()
            model = Model()
            clone.compute_latents(model, refs, gpt_cond_len=60,
                                  gpt_cond_chunk_len=4, max_ref_len=45,
                                  sound_norm_refs=False)
            self.assertEqual(model.calls[0], dict(audio_path=list(map(str, refs)),
                gpt_cond_len=60, gpt_cond_chunk_len=4, max_ref_length=45, sound_norm_refs=False))
            with self.assertRaises(ValueError):
                clone.compute_latents(model, refs, gpt_cond_len=3, gpt_cond_chunk_len=4)
            with self.assertRaises(FileNotFoundError):
                clone.compute_latents(model, str(Path(td) / 'missing.wav'))

    def test_decode_mode_text_splitting_and_sample_rate(self):
        writes = []
        sf = types.SimpleNamespace(write=lambda *args: writes.append(args))
        model = Model()
        model.config.audio['output_sample_rate'] = 22050
        with patch.dict(sys.modules, {'soundfile': sf}):
            clone.generate(model, 'Bonjour', 'FR', ('gpt', 'speaker'), 'out.wav',
                           num_beams=3, do_sample=True)
        kw = model.calls[0][1]
        self.assertFalse(kw['do_sample'])
        self.assertEqual(kw['num_beams'], 3)
        self.assertTrue(kw['enable_text_splitting'])
        self.assertEqual(writes[0][2], 22050)

    def test_invalid_audio_is_rejected(self):
        with patch.dict(sys.modules, {'soundfile': types.SimpleNamespace(write=lambda *a: None)}):
            for wave in (np.array([]), np.array([np.nan, 0]), np.zeros((2, 3))):
                model = Model()
                model.wave = wave
                with self.subTest(shape=wave.shape), self.assertRaises(ValueError):
                    clone.generate(model, 'Bonjour', 'fr', (None, None), 'out.wav')

    def test_renderer_passes_same_seed_decode_and_reference_duration(self):
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / 'ref.wav'
            ref.touch()
            calls = []
            xc = types.SimpleNamespace(
                compute_latents=lambda *a, **kw: calls.append(('latents', kw)) or (None, None),
                generate=lambda *a, **kw: calls.append(('generate', kw)))
            class Audio:
                @classmethod
                def from_wav(cls, path): return cls()
                def __add__(self, value): return self
            ns = extract_functions('guided_meditation_generator_v23.py', {'generate_sentence_audio'},
                dict(os=os, _XC=xc, _LAT_CACHE={}, AudioSegment=Audio,
                     render_voice_chain=lambda a, *args: a))
            params = settings.generator_params(settings.format_block(
                dict(seed=0, do_sample=0, max_ref_len=45)), {})
            tts = types.SimpleNamespace(synthesizer=types.SimpleNamespace(tts_model=Model()))
            config = dict(language='fr', speed=1, volume=0)
            for text in ('Bonjour', 'Une autre phrase'):
                result = ns['generate_sentence_audio'](text, 1, [[str(ref)]], {1: tts},
                                                       config, params, str(Path(td) / 'out.wav'))
                self.assertIsNotNone(result)
            latents = [kw for tag, kw in calls if tag == 'latents']
            generated = [kw for tag, kw in calls if tag == 'generate']
            self.assertEqual(len(latents), 1)
            self.assertEqual(latents[0]['max_ref_len'], 45)
            self.assertEqual([(p['seed'], p['do_sample']) for p in generated], [(0, False), (0, False)])


class OptimizerTests(unittest.TestCase):
    def run_optimizer(self, options, score=None, extra_ref=False):
        score = score or (lambda rec: .95 if rec['text'] == 'TEST' else .05)
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / 'a.wav'
            other = Path(td) / 'b.wav'
            ref.touch(); other.touch()
            metadata, conditioning, generated = {}, [], []
            def latents(model, paths, **kw):
                rec = dict(kw, refs=paths if isinstance(paths, list) else [paths])
                conditioning.append(rec)
                return rec
            def generate(model, text, lang, lat, path, **kw):
                rec = dict(kw, text=text, conditioning=lat)
                metadata[path] = rec
                generated.append(rec)
            class Pron:
                def __init__(self, **kw): pass
                def score(self, path, **kw): return {'score': score(metadata[path])}
            class Encoder:
                def __init__(self, **kw): pass
                def embed(self, path): return np.array([1., 0.])
                def cosine(self, a, b): return .8
            mocks = {'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
                     'xtts_clone': types.SimpleNamespace(load_xtts=lambda d: (None, None),
                                                        compute_latents=latents, generate=generate),
                     'pron_score': types.SimpleNamespace(PronScorer=Pron),
                     'speaker_identity': types.SimpleNamespace(SpeakerEncoder=Encoder)}
            output = Path(td) / 'winner.json'
            argv = [str(ref)] + ([str(other)] if extra_ref else []) + ['FR',
                    '--xtts-block', settings.format_block({'seed': 0}), '--text', 'TEST',
                    '--seeds', '0', '--device', 'cpu', '--result-json', str(output)] + options
            with patch.dict(sys.modules, mocks):
                log = invoke(optimizer, argv)
            return json.loads(output.read_text()), log, conditioning, generated

    def test_coord_ranking_excludes_one_sentence_conditioning_scores(self):
        result, log, _, generated = self.run_optimizer([
            '--method', 'coord', '--rounds', '0', '--cond-budget', '1',
            '--budget', '2', '--no-holdout'])
        self.assertAlmostEqual(result['scores']['search'], .620)
        self.assertEqual(result['generations'], dict(conditioning=1, sampling=2, holdout=0))

    def test_holdout_has_separate_budget_and_unseen_texts(self):
        result, _, _, _ = self.run_optimizer([
            '--method', 'coord', '--rounds', '0', '--cond-budget', '1', '--budget', '2'])
        self.assertEqual(result['generations']['sampling'], 2)
        self.assertEqual(result['generations']['holdout'], 2)
        self.assertFalse(set(result['search_texts']) & set(result['holdout_texts']))
        self.assertIsNotNone(result['scores']['held_out'])

    def test_max_reference_duration_is_searched_and_exported(self):
        result, _, conditioning, _ = self.run_optimizer([
            '--method', 'coord', '--rounds', '0', '--cond-budget', '40', '--budget', '2',
            '--max-ref-lens', '20 45', '--no-holdout'],
            score=lambda r: .9 if r['conditioning']['max_ref_len'] == 45 else .1)
        # (A budget for C6 too: since 3.7.2 an unconfirmed winner stays out.)
        self.assertEqual(result['params']['max_ref_len'], 45)
        self.assertEqual(settings.parse_block(result['xtts_block'])['max_ref_len'], 45)
        self.assertIn(20, [c['max_ref_len'] for c in conditioning])

    def test_winning_reference_subset_is_exported(self):
        result, _, _, _ = self.run_optimizer([
            '--method', 'coord', '--rounds', '0', '--cond-budget', '30', '--budget', '2',
            '--gpt-cond-lens', '30', '--gpt-chunks', '4', '--no-holdout'], extra_ref=True,
            score=lambda r: .95 if len(r['conditioning']['refs']) == 1 and
                Path(r['conditioning']['refs'][0]).name == 'b.wav' else .1)
        self.assertEqual([Path(p).name for p in result['selected_refs']], ['b.wav'])

    def test_rsm_combines_sampling_axis_improvements(self):
        def score(rec):
            return .3 + .2 * (rec['repetition_penalty'] == 7) + .2 * (rec['top_p'] == .9)
        result, _, _, _ = self.run_optimizer([
            '--method', 'rsm', '--cond-budget', '1', '--budget', '100', '--no-holdout'], score=score)
        self.assertEqual(result['params']['rep_pen'], 7)
        self.assertEqual(result['params']['top_p'], .9)

    def test_too_small_budget_is_rejected_before_models_load(self):
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / 'a.wav'; ref.touch()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                invoke(optimizer, [str(ref), 'FR', '--xtts-block', settings.format_block({}), '--budget', '1'])
            self.assertEqual(cm.exception.code, 2)


class PipelineTests(unittest.TestCase):
    def test_pipeline_uses_json_scores_and_exact_cloning_subset(self):
        with tempfile.TemporaryDirectory() as td:
            ref, other = Path(td) / 'a.wav', Path(td) / 'b.wav'
            ref.touch(); other.touch()
            xb = settings.format_block(dict(seed=0, do_sample=0, max_ref_len=45))
            ab = '[1, FR, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]'
            commands, saves = [], []
            def run(cmd, tag):
                commands.append(cmd)
                if 'voice_analyser.py' in cmd[1]: return 0, xb + '\n' + ab
                if 'xtts_optimize.py' in cmd[1]:
                    target = cmd[cmd.index('--result-json') + 1]
                    Path(target).write_text(json.dumps(dict(schema_version=1, xtts_block=xb,
                        selected_refs=[str(other)], scores=dict(held_out=.8, identity=.7))))
                    return 0, 'Console wording may change completely'
                return 0, 'Next []    : ' + ab
            vp = types.SimpleNamespace(PRESET_DIR=td, name_from_reference=lambda p: 'Voice',
                save=lambda *a, **kw: saves.append((a, kw)) or 'Voice', describe=lambda n: '')
            with patch.object(pipeline, 'run_stream', run), patch.dict(sys.modules, {'voice_presets': vp}):
                invoke(pipeline, ['--voice', str(ref), 'FR', '--no-curate'])
            cmd = next(c for c in commands if 'voice_comparator.py' in c[1])
            self.assertEqual(cmd[2], str(ref))
            self.assertEqual(cmd[cmd.index('--clone-refs') + 1:cmd.index('--xtts-block')], [str(other)])
            self.assertEqual(saves[0][1]['references'], [str(other)])
            self.assertEqual(saves[0][1]['scores']['held_out'], .8)

    def test_presets_save_and_restore_exact_reference_order(self):
        with tempfile.TemporaryDirectory() as td, patch.object(presets, 'PRESET_DIR', td):
            refs = [os.path.join(td, 'b.wav'), os.path.join(td, 'a.wav')]
            name = presets.save('Voice', settings.format_block({}), '[1, FR, 1]', references=refs)
            self.assertEqual(presets.get(name)['references'], refs)


class ValidatorTests(unittest.TestCase):
    def test_new_validation_axes_reach_the_generation_engine(self):
        class Audio:
            @classmethod
            def silent(cls, **kw): return cls()
            @classmethod
            def from_wav(cls, path): return cls()
            def __add__(self, other): return self
            def __len__(self): return 1000
            def export(self, path, **kw): Path(path).write_bytes(b'simulated audio')
        for parameter, value in (('gpt_cond_chunk_len', '6'), ('sound_norm_refs', '1'),
                                 ('num_beams', '3'), ('do_sample', '0'), ('max_ref_len', '45'),
                                 ('speaker_ref_len', '20'), ('ref_trim_db', '45'), ('typical_p', '.9'),
                                 ('seed', 'none')):
            with self.subTest(parameter=parameter), tempfile.TemporaryDirectory() as td:
                ref = Path(td) / 'reference.wav'; ref.touch()
                latents, generated = [], []
                xc = types.SimpleNamespace(load_xtts=lambda d: (None, None),
                    compute_latents=lambda *a, **kw: latents.append(kw) or (None, None),
                    generate=lambda *a, **kw: generated.append(kw))
                mocks = {'xtts_clone': xc, 'pydub': types.SimpleNamespace(AudioSegment=Audio),
                         'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))}
                class Loader:
                    def create_module(self, spec): return None
                    def exec_module(self, module):
                        module.render_voice_chain = lambda audio, *args: audio
                spec = importlib.machinery.ModuleSpec('gmg', Loader())
                with patch.dict(sys.modules, mocks), patch('importlib.util.spec_from_file_location', return_value=spec):
                    invoke(validator, [str(ref), 'FR', '--param', parameter, '--values', value,
                        '--xtts-block', settings.format_block({'seed': 0}),
                        '--text', 'TEST', '--no-score', '--no-labels',
                        '--output', str(Path(td) / 'out.wav')])
                self.assertEqual(generated[0]['seed'], None if parameter == 'seed' else 0)
                target = latents[0] if parameter in ('gpt_cond_chunk_len', 'sound_norm_refs', 'max_ref_len',
                                                    'speaker_ref_len', 'ref_trim_db') else generated[0]
                self.assertEqual(target[parameter], None if value == 'none' else float(value))

    def test_invalid_cartesian_combination_is_rejected_before_models_load(self):
        with tempfile.TemporaryDirectory() as td:
            ref = Path(td) / 'reference.wav'; ref.touch()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                invoke(validator, [str(ref), 'FR', '--param', 'gpt_cond_len', '--values', '3',
                                   '--param', 'gpt_cond_chunk_len', '--values', '6'])
            self.assertEqual(cm.exception.code, 2)


class FakeVar:
    def __init__(self, value='', **kw): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value
    def trace_add(self, *args): pass


class FakeWidget:
    labels, entries, buttons = {}, {}, {}
    last_label = ''
    def __init__(self, parent=None, **kw):
        self.children, self.text = [], ''
        if isinstance(parent, FakeWidget): parent.children.append(self)
        if 'text' in kw:
            FakeWidget.last_label = kw['text']
            FakeWidget.labels[kw['text']] = self
        if 'textvariable' in kw:
            FakeWidget.entries[FakeWidget.last_label] = kw['textvariable']
        if 'command' in kw: FakeWidget.buttons[kw.get('text', '')] = kw['command']
    def grid(self, **kw): pass
    def pack(self, **kw): pass
    def grid_columnconfigure(self, *args, **kw): pass
    def config(self, **kw): pass
    def bind(self, *args): pass
    def winfo_children(self): return self.children
    def destroy(self): pass
    def get(self, *args): return self.text
    def delete(self, *args): self.text = ''
    def insert(self, *args): self.text += args[-1]
    def __setitem__(self, key, value): pass


class GuiTests(unittest.TestCase):
    def namespace(self, function):
        FakeWidget.buttons = {}; FakeWidget.entries = {}
        targets, commands, callbacks = {}, [], {}
        tk = types.SimpleNamespace(**{k: FakeWidget for k in ('Frame','Label','Entry','Button',
            'Checkbutton','Spinbox','LabelFrame')}, StringVar=FakeVar, IntVar=FakeVar, BooleanVar=FakeVar)
        def add_row(parent, label, var, row, *args, **kw): FakeWidget.entries[label] = var
        ns = dict(tk=tk, ttk=types.SimpleNamespace(Combobox=FakeWidget),
            scrolledtext=types.SimpleNamespace(ScrolledText=FakeWidget), os=os, sys=sys,
            json=json, math=math, parse_block=settings.parse_block, HANDOFF={}, TARGETS=targets,
            DEFAULT_LANG='FR', LANGS=['FR'], DIR_VOICES='/voices', DIR_OUTPUT='/tmp',
            SCRIPTS_DIR=str(ROOT), _BUILTIN_TEXTS=set(), _default_text=lambda lang: 'TEST',
            scrollable_tab=lambda *a: (FakeWidget(), FakeWidget()), add_row=add_row,
            _bind_default_text=lambda *a: None, add_console=lambda *a: FakeWidget(),
            split_paths=lambda s: [v.strip() for v in s.split(';') if v.strip()],
            log=lambda *a: None, browse_file=lambda *a, **kw: None,
            make_btn=lambda parent, text, callback, row: callbacks.update({text: callback}),
            run_cmd=lambda cmd, *a, **kw: commands.append(cmd),
            _handoff_set=lambda key, value: targets[key].set(value) if key in targets else False)
        extract_functions('xtts_studio.py', {function}, ns)
        ns[function](None)
        return ns, targets, commands, callbacks

    def test_optimizer_gui_passes_conditioning_controls(self):
        ns, targets, commands, callbacks = self.namespace('tab_optimize')
        targets['opt_voices'].set('/a.wav ; /b.wav')
        targets['opt_xtts'].set(settings.format_block({'seed': 0}))
        FakeWidget.entries['Reference lengths to test'].set('20 45')
        callbacks['>  Optimise'](None)
        cmd = commands[0]
        for option in ('--cond-budget', '--probe-texts', '--gpt-cond-lens', '--gpt-chunks',
                       '--objective', '--method', '--max-ref-lens'):
            self.assertIn(option, cmd)
        self.assertEqual(cmd[cmd.index('--max-ref-lens') + 1], '20 45')

    def test_comparator_gui_separates_target_and_clone_references(self):
        ns, targets, commands, callbacks = self.namespace('tab_comparator')
        targets['cmp_reference'].set('/target.wav')
        targets['cmp_clone_refs'].set('/winner b.wav ; /winner a.wav')
        targets['cmp_xtts'].set(settings.format_block({}))
        targets['cmp_audio'].set('[1, FR, 1]')
        callbacks['>  Compare & Optimise'](None)
        cmd = commands[0]
        self.assertEqual(cmd[2], '/target.wav')
        self.assertEqual(cmd[cmd.index('--clone-refs') + 1:cmd.index('--xtts-block')],
                         ['/winner b.wav', '/winner a.wav'])
        os.unlink(cmd[cmd.index('--text-file') + 1])


if __name__ == '__main__':
    unittest.main(verbosity=2)

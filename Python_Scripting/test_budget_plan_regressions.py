"""Offline regression tests for the search plans and the [Auto] auto buttons.

    python3 -m unittest -v test_budget_plan_regressions

The optimiser runs end to end with XTTS, Whisper and ECAPA simulated: these
tests check budgets, schedules and GUI wiring, not voice quality.
"""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
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

try:
    import soundfile as sf
except ImportError:            # the GUI tests read real WAV headers
    sf = None


def run_optimizer(cfg, cond_budget, trial=0, budget=60, seeds='0 42 100 180 200'):
    """xtts_optimize.main() with simulated models; returns (result, log)."""
    with tempfile.TemporaryDirectory() as td:
        refs = []
        for i in range(cfg.get('n_refs', 1)):
            ref = Path(td) / f'r{i}.wav'; ref.touch(); refs.append(str(ref))
        meta = {}

        def latents(model, paths, **kw):
            return dict(kw, n=len(paths) if isinstance(paths, list) else 1)

        def generate(model, text, lang, lat, path, **kw):
            meta[path] = dict(kw, text=text, lat=lat)

        def draw(rec):
            key = json.dumps([trial, rec['seed'], rec['temperature'], rec['repetition_penalty'],
                              rec['top_p'], rec['top_k'], rec['text'], rec.get('num_beams'),
                              sorted(rec['lat'].items())], default=str)
            return np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest()[:8], 16)).normal()

        class Pron:
            def __init__(self, **kw): pass
            def score(self, path, **kw):
                return {'score': 0.9 + 0.03 * draw(meta[path])}

        class Encoder:
            def __init__(self, **kw): pass
            def embed(self, path): return np.array([1., 0.])
            def cosine(self, a, b): return 0.70

        mocks = {'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
                 'xtts_clone': types.SimpleNamespace(load_xtts=lambda d: (None, None),
                                                    compute_latents=latents, generate=generate),
                 'pron_score': types.SimpleNamespace(PronScorer=Pron),
                 'speaker_identity': types.SimpleNamespace(SpeakerEncoder=Encoder)}
        out = Path(td) / 'winner.json'
        block = settings.format_block(dict(
            seed=0, temp=cfg.get('temp', 0.62), rep_pen=5, top_p=0.85, top_k=50,
            gpt_cond_len=cfg['prior'], gpt_cond_chunk_len=cfg.get('chunk', 4),
            sound_norm_refs=cfg.get('norm', 0)))
        argv = ['xtts_optimize.py', *refs, 'FR', '--xtts-block', block, '--seeds', seeds,
                '--budget', str(budget), '--cond-budget', str(cond_budget),
                '--gpt-chunks', cfg['chunks'], '--max-ref-len', str(cfg['max_ref']),
                '--gpt-cond-lens', cfg['gcl'], '--cond-probe-texts', str(cfg.get('texts', 1)),
                '--adopt-alpha', cfg.get('alpha', '1'), '--device', 'cpu',
                '--result-json', str(out), *cfg.get('extra', ())]
        if cfg.get('ref_lengths'):
            argv += ['--max-ref-lens', cfg['ref_lengths']]
        log = io.StringIO()
        with patch.dict(sys.modules, mocks), patch.object(sys, 'argv', argv), \
                contextlib.redirect_stdout(log):
            optimizer.main()
        return json.loads(out.read_text()), log.getvalue()


def plan_for(cfg):
    n_refs = cfg.get('n_refs', 1)
    sets = len(optimizer._reference_sets([f'/r{i}.wav' for i in range(n_refs)])) if n_refs > 1 else 0
    return SP.conditioning_plan(SP.find_ints(cfg['gcl']), SP.find_ints(cfg['chunks']),
                                SP.parse_ints(cfg.get('ref_lengths', '')), cfg['prior'],
                                cfg.get('chunk', 4), cfg['max_ref'], n_refs,
                                cfg.get('texts', 1), sets)


PIPELINE = dict(gcl='20 30', chunks='4 6 8 10', max_ref=30, prior=45)


class PlanTests(unittest.TestCase):
    def test_conditioning_plan_runs_every_stage_whatever_wins(self):
        # --adopt-alpha 1 lets every gain win, so the noise walks the search
        # down a different path on each trial: the plan must cover all of them.
        # It leaves C6 nothing to confirm (the incumbent is always the best
        # draw so far), so the noise-aware rule runs too: there the finalists
        # and a replaced base fill the confirmation.
        configs = {
            'pipeline defaults': PIPELINE,
            'reference lengths 20 30 45': dict(PIPELINE, ref_lengths='20 30 45'),
            'reference lengths that cut gcl': dict(PIPELINE, ref_lengths='15 25'),
            # A cut gpt_cond_len (15) that C5 then combines with longer caps.
            'cut gcl combined by C5': dict(PIPELINE, ref_lengths='15 25 45'),
            # A chunk cut to a cut gpt_cond_len (5): a value nobody typed.
            'chunk cut by a short reference': dict(gcl='6 20', chunks='4 8 10', max_ref=30,
                                                   prior=30, chunk=10, ref_lengths='5 30'),
            'chunk longer than gcl': dict(gcl='6 10 15 20 30', chunks='4 6 8 10 12',
                                          max_ref=30, prior=30),
            'chunk cut to a reference length': dict(gcl='8', chunks='4 6 8 10', max_ref=30,
                                                    prior=30, ref_lengths='5 10'),
            'short file': dict(PIPELINE, max_ref=12, prior=12),
            'two sentences per candidate': dict(PIPELINE, ref_lengths='20 45', texts=2),
            'normalised block': dict(PIPELINE, prior=30, norm=1),
            'three references': dict(PIPELINE, ref_lengths='10 20', n_refs=3),
        }
        confirmed = 0
        for name, cfg in configs.items():
            plan = plan_for(cfg)
            for alpha in ('1', '0.05'):
                for trial in range(10):
                    with self.subTest(config=name, alpha=alpha, trial=trial):
                        result, log = run_optimizer(dict(cfg, alpha=alpha), plan, trial)
                        used = int(re.search(r'Evaluations: conditioning (\d+)/', log).group(1))
                        self.assertLessEqual(used, plan)
                        self.assertEqual(int(re.search(r'Conditioning plan: up to (\d+)',
                                                       log).group(1)), plan)
                        self.assertFalse([k for k in result['truncated_stages']
                                          if k.startswith('C')])
                        self.assertNotIn('C5 stops', log)
                        self.assertNotIn('C6 confirmation: cut', log)
                        self.assertNotIn('Conditioning noise not measured', log)
                        confirmed += bool(result['conditioning_confirmation'])
        self.assertGreater(confirmed, 50)      # C6 did run, in most noise-aware trials

    def test_plan_is_tight_for_the_pipeline(self):
        self.assertEqual(plan_for(PIPELINE), 34)
        self.assertEqual(plan_for(dict(PIPELINE, ref_lengths='20 30 45')), 45)
        _result, log = run_optimizer(dict(PIPELINE, alpha='0.05'), 34)
        # 20 screening renders (the base on 5 seeds, C1 to C5), then 8 for
        # C6's 2 candidates on the 4 probe seeds: the base's renders there are
        # the probe's own.
        self.assertIn('Evaluations: conditioning 28/34', log)
        # Everything the schedule can ask for, nothing more: 3 candidates fill
        # C6 (12 renders), and the 2 left are the reserve that lets C2b run.
        result, log = run_optimizer(dict(PIPELINE, alpha='0.05'), 34, trial=2)
        self.assertEqual(len(result['conditioning_confirmation']['finalists']), 3)
        self.assertIn('Evaluations: conditioning 32/34', log)

    def test_header_warns_when_the_conditioning_budget_is_short(self):
        _result, log = run_optimizer(dict(PIPELINE, alpha='0.05'), 12)
        self.assertIn('--cond-budget 34 runs them all', log)

    def test_sampling_budget_is_the_optimiser_plan_in_the_longest_case(self):
        for seeds, beams, want in (('0 42 100 180 200', False, 60),
                                   ('0 42 75 100 130 155 160 180 200', False, 68),
                                   ('0 42 75 100 130 155 160 180 200', True, 80),
                                   ('0 0 42', False, 42)):   # a repeated seed counts once
            with self.subTest(seeds=seeds, beams=beams):
                need = SP.pipeline_sampling_budget(SP.parse_ints(seeds), beams)
                self.assertEqual(need, want)
                extra = ('--probe-beams', '--beam-width', '3') if beams else ()
                cfg = dict(PIPELINE, alpha='0.05', extra=extra)
                result, log = run_optimizer(cfg, plan_for(PIPELINE), budget=need, seeds=seeds)
                self.assertIn(f'Sampling plan: up to {need} generations', log)
                self.assertEqual(result['truncated_stages'], {})
        # A starting temperature on the S2 grid needs 6 fewer.
        _r, log = run_optimizer(dict(PIPELINE, alpha='0.05', temp=0.65), plan_for(PIPELINE))
        self.assertIn('Sampling plan: up to 54 generations', log)

    def test_the_pipeline_still_hands_the_optimiser_what_the_plan_assumes(self):
        analyser = (ROOT / 'voice_analyser.py').read_text(encoding='utf-8')
        self.assertIn('top_k, top_p, repetition_penalty = 50, 0.85, 5.0', analyser)
        self.assertEqual(SP.ANALYSER_SAMPLING, {'rep_pen': 5.0, 'top_p': 0.85, 'top_k': 50})
        self.assertIn('gpt_cond_chunk_len = 4', analyser)
        self.assertIn('gpt_cond_len       = int(min(60, max(6, int(duration))))', analyser)
        pipeline = (ROOT / 'xtts_pipeline.py').read_text(encoding='utf-8')
        self.assertIn("p.add_argument('--probe-texts', type=int, default=2)", pipeline)
        # The [Auto] tab passes no --gpt-cond-lens for an empty field: this
        # default is what its FALLBACK table has to mirror.
        self.assertIn("p.add_argument('--gpt-cond-lens', default='20 30',", pipeline)
        self.assertNotIn("'--keep-seeds'", pipeline)
        self.assertNotIn("'--cond-probe-texts'", pipeline)
        self.assertIn("work = base + '_curated.wav'", pipeline)
        source = (ROOT / 'xtts_optimize.py').read_text(encoding='utf-8')
        # The search and its plan share the schedule constants: no literal
        # may creep back into the stages the plan counts.
        self.assertIn('if cond_evals[0] < args.cond_budget - C2B_RESERVE:', source)
        self.assertIn('(1 + NOISE_PROBES + NOISE_PROBE_MARGIN) * len(cond_probe_texts)', source)
        self.assertIn('[:NOISE_PROBES]', source)
        self.assertIn('if not free and new >= C5_SCREEN:', source)
        self.assertIn('finalists = finalists[:CONFIRM_FINALISTS]', source)
        self.assertIn('c6_reserve = ((CONFIRM_FINALISTS + (0 if probed else 1)) * len(probe_seeds) '
                      '* n_txt', source)
        self.assertIn('ap.add_argument("--keep-seeds", type=int, default=3)', source)
        self.assertIn('ap.add_argument("--cond-probe-texts", type=int, default=1,', source)

    def test_length_helpers(self):
        # One Max reference s for every voice: the longest file decides, and
        # a shorter one is simply read in full.
        self.assertEqual(SP.max_reference_for([45.0, 30.2]), 45)
        self.assertEqual(SP.max_reference_for([200.0]), 60)
        # 30 reads every sample of a 29.95 s file (an older curated file), 29
        # would drop the last 0.95 s.
        self.assertEqual(SP.max_reference_for([29.95]), 30)
        self.assertEqual(SP.whole_file_seconds(30.0), 30)
        self.assertEqual(SP.whole_file_seconds(30.0 + 1e-9), 30)
        self.assertEqual(SP.whole_file_seconds(30.2), 31)
        self.assertEqual(SP.reference_lengths_for(30.0), [20, 30])
        self.assertEqual(SP.reference_lengths_for(29.95), [20, 30])
        self.assertEqual(SP.reference_lengths_for(45.0), [20, 30, 45])
        self.assertEqual(SP.reference_lengths_for(25.9), [20, 26])
        self.assertEqual(SP.reference_lengths_for(12.0), [12])
        self.assertEqual(SP.chunk_candidates([20, 30], 30), [4, 6, 8, 10])
        self.assertEqual(SP.chunk_candidates([6, 20], 30), [4, 6])
        self.assertEqual(SP.chunk_candidates([3], 30), [3])
        self.assertEqual(SP.chunk_candidates([45], 30), [4, 6, 8, 10])
        self.assertEqual(SP.analyser_gcl_prior(29.99), 29)
        self.assertEqual(SP.analyser_gcl_prior(3), 6)
        with self.assertRaises(ValueError):
            SP.parse_ints('0 42 x')
        self.assertEqual(SP.find_ints('20, 30;45'), [20, 30, 45])

    def test_gcl_candidates_stop_at_what_xtts_reads(self):
        self.assertEqual(SP.gcl_candidates([30.0], 30), [20, 30])
        self.assertEqual(SP.gcl_candidates([29.95], 30), [20, 30])
        self.assertEqual(SP.gcl_candidates([45.0], 45), [20, 30, 45])
        self.assertEqual(SP.gcl_candidates([45.0], 30), [20, 30])      # Max reference s cuts it
        self.assertEqual(SP.gcl_candidates([25.0, 60.0], 60), [20, 25])  # the shortest file decides
        self.assertEqual(SP.gcl_candidates([200.0], 90), [20, 30, 45, 60])
        self.assertEqual(SP.gcl_candidates([12.0], 30), [10, 12])      # below the 20 s floor
        self.assertEqual(SP.gcl_candidates([8.0], 30), [8])
        # Every value passes the optimiser's own C1 filter: none is ignored.
        for seconds, max_ref in (([30.0], 30), ([45.0], 30), ([12.0], 30), ([200.0], 90)):
            self.assertTrue(all(g <= max_ref for g in SP.gcl_candidates(seconds, max_ref)))

    def test_reference_seconds_follow_the_pipeline(self):
        with tempfile.TemporaryDirectory() as td:
            ref = os.path.join(td, 'voice.wav')
            open(ref, 'w').close()
            lengths = {ref: 40.0}
            seconds_of = lambda p: lengths[p]
            self.assertEqual(SP.pipeline_reference_seconds([ref], True, False, 30, seconds_of),
                             [(ref, 30.0, 'curation to Keep s')])
            cur = os.path.join(td, 'voice_curated.wav')
            open(cur, 'w').close()
            lengths[cur] = 45.0
            # Reused as it is: Keep s does not apply.
            self.assertEqual(SP.pipeline_reference_seconds([ref], True, False, 30, seconds_of)[0][1], 45.0)
            self.assertEqual(SP.pipeline_reference_seconds([ref], True, True, 30, seconds_of)[0][1], 30.0)
            self.assertEqual(SP.pipeline_reference_seconds([ref], False, False, 30, seconds_of)[0][1], 40.0)


# ── GUI: the [Auto] Pipeline tab with fake widgets ───────────────────────────

class Var:
    def __init__(self, value='', **kw):
        self.value, self.traces = value, []
    def get(self): return self.value
    def set(self, value):
        self.value = value
        for callback in list(self.traces):
            callback()
    def trace_add(self, mode, callback): self.traces.append(callback)


class Widget:
    def __init__(self, parent=None, **kw):
        self.kw = kw
        Widget.created.append(self)
        if 'text' in kw:
            Widget.last_label = kw['text']
        if 'textvariable' in kw:
            Widget.entries.setdefault(Widget.last_label, kw['textvariable'])
            Widget.variables.append(kw['textvariable'])
        if 'variable' in kw:
            Widget.checks[kw.get('text', '')] = kw['variable']
        if 'command' in kw and 'name' in kw:
            Widget.buttons[kw['name']] = kw['command']
    def __getattr__(self, name):
        return lambda *a, **k: None


class SyncThread:
    def __init__(self, target=None, **kw): self.target = target
    def start(self): self.target()


def speechlike(path, seconds, sr=16000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    env = (np.sin(2 * np.pi * 2.3 * t) > -0.3).astype(np.float32)
    sf.write(path, (0.1 * rng.standard_normal(t.size) * env).astype(np.float32), sr)


@unittest.skipIf(sf is None, 'soundfile is needed to read the WAV headers')
class AutoTabTests(unittest.TestCase):
    def build(self):
        Widget.entries, Widget.variables, Widget.checks, Widget.buttons = {}, [], {}, {}
        Widget.created = []
        Widget.last_label = ''
        self.logs, self.commands, callbacks = [], [], {}
        tk = types.SimpleNamespace(StringVar=Var, BooleanVar=Var, IntVar=Var, Frame=Widget,
                                   LabelFrame=Widget, Label=Widget, Entry=Widget, Button=Widget,
                                   Checkbutton=Widget)
        ns = dict(tk=tk, ttk=types.SimpleNamespace(Combobox=Widget), os=os, sys=sys, SP=SP,
                  threading=types.SimpleNamespace(Thread=SyncThread),
                  filedialog=types.SimpleNamespace(askopenfilename=lambda **k: ''),
                  scrollable_tab=lambda *a: (Widget(), Widget()), add_console=lambda *a: Widget(),
                  log=lambda console, text: self.logs.append(text),
                  make_btn=lambda parent, text, callback, row: callbacks.update({text: callback}),
                  run_cmd=lambda cmd, *a, **kw: self.commands.append(cmd),
                  LANGS=['FR'], DEFAULT_LANG='FR', DIR_VOICES='/voices', SCRIPTS_DIR=str(ROOT))
        tree = ast.parse((ROOT / 'xtts_studio.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'tab_auto')
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'xtts_studio.py', 'exec'), ns)
        ns['tab_auto'](None)
        self.run_pipeline = lambda: callbacks['>  Run full pipeline'](None, None)
        return Widget.entries, Widget.checks, Widget.buttons

    def test_budget_auto(self):
        entries, checks, buttons = self.build()
        entries['Seeds'].set('0 42 75 100 130 155 160 180 200')
        buttons['auto_budget']()
        self.assertEqual(entries['Budget'].get(), '68')
        checks['Probe beam/greedy decoding'].set(True)
        buttons['auto_budget']()
        self.assertEqual(entries['Budget'].get(), '80')
        entries['Seeds'].set('0 42 x')
        buttons['auto_budget']()
        self.assertEqual(entries['Budget'].get(), '80')
        self.assertIn('[ERR] Seeds must be whole numbers', self.logs[-1])

    def test_length_autos_follow_the_audio_xtts_gets(self):
        entries, checks, buttons = self.build()
        with tempfile.TemporaryDirectory() as td:
            ref = os.path.join(td, 'voice.wav')
            speechlike(ref, 40)
            entries['V1'].set(ref)
            entries['Keep s'].set('30')
            buttons['auto_maxref']()
            self.assertEqual(entries['Max reference s'].get(), '30')
            buttons['auto_reflens']()
            self.assertEqual(entries['Reference lengths to test (optional)'].get(), '20 30')
            buttons['auto_chunks']()
            self.assertEqual(entries['Chunk lengths'].get(), '4 6 8 10')
            buttons['auto_cond']()
            self.assertEqual(entries['Conditioning budget'].get(),
                             str(SP.pipeline_conditioning_budget([20, 30], [4, 6, 8, 10],
                                                                 [20, 30], 30, [30.0])))
            # An older curated file is reused, whatever Keep s says...
            speechlike(os.path.join(td, 'voice_curated.wav'), 45, seed=1)
            checks['Curate reference'].set(True)          # refreshes the readout
            buttons['auto_maxref']()
            self.assertEqual(entries['Max reference s'].get(), '45')
            notice = [v.get() for v in Widget.variables if 'exists' in str(v.get())]
            self.assertTrue(notice and 'reused as it is' in notice[0])
            # ... unless Re-curate is ticked.
            checks['Re-curate'].set(True)
            buttons['auto_maxref']()
            self.assertEqual(entries['Max reference s'].get(), '30')
            notice = [v.get() for v in Widget.variables if 'exists' in str(v.get())]
            self.assertIn('Re-curate will curate the reference again', notice[0])
            checks['Curate reference'].set(False)
            buttons['auto_maxref']()
            self.assertEqual(entries['Max reference s'].get(), '40')

    def command_named(self, name):
        found = [w.kw['command'] for w in Widget.created
                 if getattr(w.kw.get('command'), '__name__', '') == name]
        self.assertEqual(len(found), 1)
        return found[0]

    def test_keep_auto_only_lowers(self):
        entries, checks, buttons = self.build()
        keep_auto = self.command_named('_auto_keep')
        with tempfile.TemporaryDirectory() as td:
            ref = os.path.join(td, 'voice.wav')
            speechlike(ref, 40)
            entries['Keep s'].set('300')
            entries['V1'].set(ref)        # the readout measures what curation can keep
            keep_auto()
            self.assertLess(float(entries['Keep s'].get()), 40)
            self.assertIn('Keep s lowered to', self.logs[-1])
            entries['Keep s'].set('10')
            keep_auto()
            self.assertEqual(entries['Keep s'].get(), '10')
            self.assertIn('Keep s stays at 10', self.logs[-1])

    def test_run_passes_recurate_and_warns_on_short_budgets(self):
        entries, checks, buttons = self.build()
        with tempfile.TemporaryDirectory() as td:
            ref = os.path.join(td, 'voice.wav')
            speechlike(ref, 40)
            entries['V1'].set(ref)
            entries['Keep s'].set('30')
            entries['Seeds'].set('0 42 75 100 130 155 160 180 200')
            entries['Budget'].set('60')
            entries['Reference lengths to test (optional)'].set('20 30')
            checks['Re-curate'].set(True)
            self.run_pipeline()
            cmd = self.commands[-1]
            self.assertIn('--recurate', cmd)
            self.assertTrue(any('Budget 60 is below the whole sampling plan (68)' in m
                                for m in self.logs))
            self.assertTrue(any('Conditioning budget 34 is below' in m for m in self.logs))
            checks['Curate reference'].set(False)
            self.run_pipeline()
            self.assertNotIn('--recurate', self.commands[-1])
            self.assertIn('--no-curate', self.commands[-1])

    def test_an_empty_field_means_the_same_everywhere(self):
        # lancer() falls back to a default for an empty field; the auto
        # buttons and the pre-run check must compute for that same default.
        entries, checks, buttons = self.build()
        with tempfile.TemporaryDirectory() as td:
            ref = os.path.join(td, 'voice.wav')
            speechlike(ref, 40)
            entries['V1'].set(ref)
            for name in ('Seeds', 'Budget', 'Keep s', 'gcl s', 'Conditioning budget',
                         'Chunk lengths', 'Max reference s'):
                entries[name].set('')
            self.run_pipeline()
            cmd = self.commands[-1]
            for option, value in (('--seeds', '0 42 100 180 200'), ('--budget', '60'),
                                  ('--cond-budget', '34'), ('--gpt-chunks', '4 6 8 10'),
                                  ('--max-ref-len', '30'), ('--keep-seconds', '45')):
                self.assertEqual(cmd[cmd.index(option) + 1], value)
            self.assertNotIn('--gpt-cond-lens', cmd)       # the pipeline's '20 30' applies
            buttons['auto_budget']()
            self.assertEqual(entries['Budget'].get(), '60')
            entries['Conditioning budget'].set('')
            buttons['auto_cond']()
            # Curation to the default Keep s (45 s), searched with gcl s '20 30'.
            self.assertEqual(entries['Conditioning budget'].get(),
                             str(SP.pipeline_conditioning_budget([20, 30], [4, 6, 8, 10], [],
                                                                 30, [45.0])))
            buttons['auto_chunks']()
            self.assertEqual(entries['Chunk lengths'].get(), '4 6 8 10')

    def test_priority_menu_sets_what_the_optimiser_maximises(self):
        entries, checks, buttons = self.build()
        with tempfile.TemporaryDirectory() as td:
            ref = os.path.join(td, 'voice.wav')
            speechlike(ref, 40)
            entries['V1'].set(ref)
            # Resemblance by default.
            self.assertEqual(entries['Priority'].get(), 'Resemblance')
            self.run_pipeline()
            cmd = self.commands[-1]
            for option, value in (('--w-accent', '0.3'), ('--w-identity', '0.7'),
                                  ('--objective', 'both')):
                self.assertEqual(cmd[cmd.index(option) + 1], value)
            entries['Priority'].set('Balanced')
            self.assertEqual((entries['w_acc'].get(), entries['w_id'].get(),
                              entries['Objective'].get()), ('0.6', '0.4', 'identity'))
            self.run_pipeline()
            cmd = self.commands[-1]
            self.assertEqual(cmd[cmd.index('--w-accent') + 1], '0.6')
            self.assertNotIn('--objective', cmd)                  # identity is the default
            # A field edited by hand makes it a custom objective...
            entries['w_acc'].set('0.5')
            self.assertEqual(entries['Priority'].get(), 'Custom')
            self.assertEqual(entries['w_id'].get(), '0.4')         # nothing else moved
            # ... and typing a preset's values back finds it again.
            entries['w_acc'].set('0.30')
            entries['w_id'].set('0.7')
            entries['Objective'].set('both')
            self.assertEqual(entries['Priority'].get(), 'Resemblance')

    def test_level_and_weighting_defaults_reach_the_pipeline(self):
        entries, checks, buttons = self.build()
        with tempfile.TemporaryDirectory() as td:
            ref = os.path.join(td, 'voice.wav')
            speechlike(ref, 40)
            entries['V1'].set(ref)
            self.run_pipeline()
            cmd = self.commands[-1]
            self.assertEqual(cmd[cmd.index('--target-dbfs') + 1], '-20')
            self.assertEqual(cmd[cmd.index('--eq-weighting') + 1], 'a+blend')
            entries['Target dBFS'].set('')                        # match each reference
            entries['EQ weighting'].set('a')
            self.run_pipeline()
            self.assertNotIn('--target-dbfs', self.commands[-1])
            self.assertNotIn('--eq-weighting', self.commands[-1])

    def test_gcl_auto_follows_the_audio_xtts_reads(self):
        entries, checks, buttons = self.build()
        with tempfile.TemporaryDirectory() as td:
            ref = os.path.join(td, 'voice.wav')
            speechlike(ref, 40)
            entries['V1'].set(ref)
            entries['Keep s'].set('30')
            buttons['auto_gcl']()
            self.assertEqual(entries['gcl s'].get(), '20 30')
            self.assertIn('floor kept for the accent', self.logs[-1])
            entries['Keep s'].set('45')                           # the file to be curated
            entries['Max reference s'].set('45')
            buttons['auto_gcl']()
            self.assertEqual(entries['gcl s'].get(), '20 30 45')
            entries['Max reference s'].set('30')                  # XTTS reads 30 s of it
            buttons['auto_gcl']()
            self.assertEqual(entries['gcl s'].get(), '20 30')
            short = os.path.join(td, 'short.wav')
            speechlike(short, 12)
            entries['V1'].set(short)
            checks['Curate reference'].set(False)                 # the 12 s file as it is
            buttons['auto_gcl']()
            self.assertEqual(entries['gcl s'].get(), '10 12')
            self.assertIn('shorter than 20 s', self.logs[-1])
            buttons['auto_chunks']()
            self.assertEqual(entries['Chunk lengths'].get(), '4 6 8 10')

    def test_max_reference_reads_every_voice_in_full(self):
        entries, checks, buttons = self.build()
        self.command_named('add_auto_row')()               # a second voice row
        with tempfile.TemporaryDirectory() as td:
            short, long_ = os.path.join(td, 'a.wav'), os.path.join(td, 'b.wav')
            speechlike(short, 30)
            speechlike(long_, 45, seed=1)
            entries['V1'].set(short)
            entries['V2'].set(long_)
            checks['Curate reference'].set(False)          # each file as it is
            buttons['auto_maxref']()
            self.assertEqual(entries['Max reference s'].get(), '45')
            self.assertIn('longest (b.wav, 45.0s', self.logs[-2])
            self.assertIn('30 / 45', self.logs[-1])         # one starting point per voice
            buttons['auto_reflens']()                       # nothing past the shortest
            self.assertEqual(entries['Reference lengths to test (optional)'].get(), '20 30')
            buttons['auto_cond']()
            self.assertEqual(entries['Conditioning budget'].get(),
                             str(SP.pipeline_conditioning_budget([20, 30], [4, 6, 8, 10],
                                                                 [20, 30], 45, [30.0, 45.0])))


# ── Curation: the file XTTS gets is exactly Keep s long ─────────────────────

class ScatteredEncoder:
    """One speaker, directions scattered around it: the coherence ranking
    jumps around in time, so curation keeps several separate segments and
    joins them with crossfades."""
    def __init__(self, **kw): pass

    def embed(self, y, sr):
        seed = int(hashlib.md5(np.asarray(y[:2000], np.float32).tobytes()).hexdigest()[:8], 16)
        v = np.array([1.0, 0.0, 0.0]) + 0.3 * np.random.default_rng(seed).standard_normal(3)
        return v / np.linalg.norm(v)


@unittest.skipIf(sf is None, 'soundfile is needed to write and read the curated file')
class CurationLengthTests(unittest.TestCase):
    def test_curated_file_is_exactly_keep_seconds(self):
        # 16 kHz, the ECAPA rate: curation has nothing to resample. Windows on
        # a 2 s grid often add up to an even Keep s exactly, and every join
        # used to cost 10 ms of crossfade: 29.95 s for 30, so the analyser
        # started the search at gpt_cond_len 29 and the conditioning plan the
        # auto button computed for 30 no longer covered the run.
        import curate_reference as curate
        sr = 16000
        rng = np.random.default_rng(0)
        y = (0.1 * np.sin(np.arange(sr * 120) * 0.05)
             * (1 + 0.5 * rng.standard_normal(sr * 120))).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            ref, out = os.path.join(td, 'voice.wav'), os.path.join(td, 'voice_curated.wav')
            sf.write(ref, y, sr)
            for keep in (20, 30, 40):
                with self.subTest(keep=keep), \
                        patch.dict(sys.modules, {'speaker_identity': types.SimpleNamespace(
                            SpeakerEncoder=ScatteredEncoder)}), \
                        contextlib.redirect_stdout(io.StringIO()) as printed:
                    curate.curate([ref], out, keep_seconds=keep)
                    segments = int(re.search(r'-> (\d+) merged', printed.getvalue()).group(1))
                    self.assertGreater(segments, 1)          # joins were made
                    self.assertEqual(sf.info(out).frames, keep * sr)
                    self.assertEqual(SP.analyser_gcl_prior(sf.info(out).duration), keep)


if __name__ == '__main__':
    unittest.main(verbosity=2)

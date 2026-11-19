"""3.10.7: gcl s auto stops at Keep s for a row of clips to be curated.

Seen in use (3.10.6, first run with three clips): clips of 9.9, 9.9 and 14.7 s,
Curate and Re-curate ticked, Keep s 20. auto all set gcl s to "20 30 35",
adding up the clips (34.5 s), whereas curation keeps 20 s of them in all. The
optimiser then rendered gpt_cond_len 30 and 35 for nothing, both reported as
"= same audio as [C 1]": the GPT conditioning had read all 20 s at 20 already.
The conditioning budget counted them (51 instead of 49).

    python -m unittest test_3107_curated_clip_gcl -v
"""
import ast
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import search_plan as SP
from test_budget_plan_regressions import Var, Widget, SyncThread

try:
    import soundfile as sf
except ImportError:
    sf = None

SR = 16000


def tone(path, seconds, freq):
    t = np.arange(int(round(seconds * SR))) / SR
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), SR)
    return str(path)


class ReadableSecondsTests(unittest.TestCase):
    def setUp(self):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        self.dir = Path(td.name)
        self.lengths = {}
        self.clips = []
        for name, seconds in (('a', 9.9), ('b', 9.9), ('c', 14.7)):
            path = str(self.dir / f'{name}.wav')
            Path(path).touch()
            self.lengths[path] = seconds
            self.clips.append(path)

    def voice(self, curate, recurate=False, keep=20.0, files=None):
        return SP.pipeline_voice_audio([files or self.clips], curate, recurate, keep,
                                       lambda p: self.lengths[p])[0]

    def test_clips_to_curate_hold_keep_seconds_in_all(self):
        v = self.voice(curate=True, recurate=True)
        self.assertEqual(v['total'], 20.0)
        self.assertEqual(v['seconds'], [9.9, 9.9, 14.7])          # bounds, clip by clip
        self.assertEqual(SP.gpt_readable_seconds(v, 15), 20.0)
        # The reported run: 20 only, not 20 30 35.
        self.assertEqual(SP.gcl_candidates([SP.gpt_readable_seconds(v, 15)], 15, n_refs=3), [20])

    def test_other_voices_read_what_their_files_hold(self):
        v = self.voice(curate=False)
        self.assertAlmostEqual(v['total'], 34.5)
        self.assertAlmostEqual(SP.gpt_readable_seconds(v, 15), 34.5)
        self.assertAlmostEqual(SP.gpt_readable_seconds(v, 10), 29.8)   # each cut to 10 s
        for name, seconds in (('a', 6.0), ('b', 7.0), ('c', 7.0)):
            path = str(self.dir / f'{name}_curated.wav')
            Path(path).touch()
            self.lengths[path] = seconds
        v = self.voice(curate=True)                                    # reused as they are
        self.assertEqual((v['total'], SP.gpt_readable_seconds(v, 15)), (20.0, 20.0))
        one = self.voice(curate=True, recurate=True, keep=30.0, files=[self.clips[2]])
        self.assertEqual((one['total'], SP.gpt_readable_seconds(one, 30)), (30.0, 30.0))
        one = self.voice(curate=False, files=[self.clips[2]])
        self.assertEqual(SP.gpt_readable_seconds(one, 10), 10.0)


@unittest.skipIf(sf is None, 'soundfile is needed')
class AutoTabTests(unittest.TestCase):
    """The reported run in the [Auto] tab, with fake widgets."""

    def build(self):
        Widget.entries, Widget.variables, Widget.checks, Widget.buttons = {}, [], {}, {}
        Widget.created = []
        Widget.last_label = ''
        self.logs, self.commands, callbacks = [], [], {}
        tk = types.SimpleNamespace(StringVar=Var, BooleanVar=Var, IntVar=Var, Frame=Widget,
                                   LabelFrame=Widget, Label=Widget, Entry=Widget, Button=Widget,
                                   Checkbutton=Widget)
        ns = dict(tk=tk, ttk=types.SimpleNamespace(Combobox=Widget), os=os, sys=sys, SP=SP,
                  math=math, threading=types.SimpleNamespace(Thread=SyncThread),
                  filedialog=types.SimpleNamespace(askopenfilename=lambda **k: '',
                                                   askopenfilenames=lambda **k: ()),
                  scrollable_tab=lambda *a: (Widget(), Widget()), add_console=lambda *a: Widget(),
                  log=lambda console, text: self.logs.append(text),
                  make_btn=lambda parent, text, callback, row: callbacks.update({text: callback}),
                  run_cmd=lambda cmd, *a, **kw: self.commands.append(cmd),
                  LANGS=['FR'], DEFAULT_LANG='FR', DIR_VOICES='/voices', SCRIPTS_DIR=str(ROOT))
        tree = ast.parse((ROOT / 'xtts_studio.py').read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('tab_auto', 'split_paths')]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'xtts_studio.py', 'exec'), ns)
        ns['tab_auto'](None)
        self.run_pipeline = lambda: callbacks['>  Run full pipeline'](None, None)
        return Widget.entries, Widget.checks, Widget.buttons

    def test_the_reported_run(self):
        entries, checks, buttons = self.build()
        with tempfile.TemporaryDirectory() as td:
            clips = [tone(Path(td, 'sans titre.wav'), 9.9, 200),
                     tone(Path(td, 'sans titre1.wav'), 9.9, 300),
                     tone(Path(td, 'sans titre2.wav'), 14.7, 400)]
            entries['V1'].set(' ; '.join(clips))
            entries['Keep s'].set('20')
            checks['Re-curate'].set(True)                     # Curate is ticked by default
            entries['Reference lengths to test (optional)'].set('15')
            buttons['auto_all']()
            self.assertEqual(entries['Keep s'].get(), '20')
            self.assertEqual(entries['Max reference s'].get(), '15')
            self.assertEqual(entries['gcl s'].get(), '20')
            self.assertEqual(entries['Chunk lengths'].get(), '4 6 8 10')
            want = SP.pipeline_conditioning_budget([20], [4, 6, 8, 10], [15], 15,
                                                   [20 + 2 * SP.JOIN_GAP_S], n_refs=[3])
            self.assertEqual(entries['Conditioning budget'].get(), str(want))
            self.assertEqual(want, 49)
            # Without curation the three clips are read in full, 34.5 s.
            checks['Curate reference'].set(False)
            buttons['auto_gcl']()
            self.assertEqual(entries['gcl s'].get(), '20 30 35')
            self.run_pipeline()
            cmd = self.commands[-1]
            self.assertEqual(cmd[cmd.index('--voice'):][:5], ['--voice', *clips, 'FR'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

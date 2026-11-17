"""3.10.5: a save writes the type chosen in the list, and the prompt dialogs
open in the prompt's folder.

Reported from use: in the studio's file chooser, a name saved without an
extension did not take the type selected in the list (only the caller's
default extension was added, and the list always started on its first type);
and Open / Save / Save as of the prompt editor opened in whatever folder had
been used last for a .txt file.

The rule now: an extension typed in the name is kept when it is one of the
dialog's types; otherwise the extension of the type selected in the list is
added. Picking another type with the mouse changes the extension shown in the
name at once.
"""
import os
import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from tkinter import ttk
from unittest.mock import patch

import studio_file_dialog as D

AUDIO = D._filetypes([("WAV", "*.wav"), ("MP3", "*.mp3"), ("FLAC", "*.flac"),
                      ("OGG", "*.ogg"), ("All", "*.*")])
WAV, MP3, FLAC, OGG, ALL = range(5)


class SaveNameTests(unittest.TestCase):
    def test_no_extension_takes_the_selected_type(self):
        self.assertEqual(D._save_name('flûte', AUDIO, MP3), 'flûte.mp3')
        self.assertEqual(D._save_name('flûte', AUDIO, FLAC, '.wav'), 'flûte.flac')
        self.assertEqual(D._save_name('sous dossier/flûte', AUDIO, OGG), 'sous dossier/flûte.ogg')

    def test_a_typed_extension_of_any_listed_type_is_kept(self):
        self.assertEqual(D._save_name('flûte.wav', AUDIO, MP3), 'flûte.wav')
        self.assertEqual(D._save_name('flûte.WAV', AUDIO, MP3), 'flûte.WAV')
        self.assertEqual(D._save_name('flûte.flac', AUDIO, ALL), 'flûte.flac')

    def test_a_suffix_that_is_no_listed_type_is_part_of_the_name(self):
        self.assertEqual(D._save_name('prise.v2', AUDIO, MP3), 'prise.v2.mp3')
        self.assertEqual(D._save_name('flûte.', AUDIO, MP3), 'flûte.mp3')

    def test_all_files_keeps_the_name_or_uses_the_default_extension(self):
        self.assertEqual(D._save_name('prise.v2', AUDIO, ALL), 'prise.v2')
        self.assertEqual(D._save_name('flûte', AUDIO, ALL, '.mp3'), 'flûte.mp3')
        self.assertEqual(D._save_name('flûte', AUDIO, ALL, 'wav'), 'flûte.wav')
        self.assertEqual(D._save_name('flûte', AUDIO, ALL), 'flûte')

    def test_picking_a_type_changes_a_listed_extension_only(self):
        self.assertEqual(D._retyped_name('flûte.wav', AUDIO, MP3), 'flûte.mp3')
        self.assertEqual(D._retyped_name('flûte.WAV', AUDIO, FLAC), 'flûte.flac')
        self.assertEqual(D._retyped_name('flûte', AUDIO, MP3), 'flûte')
        self.assertEqual(D._retyped_name('prise.v2', AUDIO, MP3), 'prise.v2')
        self.assertEqual(D._retyped_name('flûte.wav', AUDIO, ALL), 'flûte.wav')
        self.assertEqual(D._retyped_name('', AUDIO, MP3), '')

    def test_the_list_starts_on_the_expected_type(self):
        f = D._initial_filter
        self.assertEqual(f(AUDIO), WAV)
        self.assertEqual(f(AUDIO, defaultextension='.flac'), FLAC)
        self.assertEqual(f(AUDIO, defaultextension='OGG'), OGG)
        self.assertEqual(f(AUDIO, initialfile='session.mp3', defaultextension='.wav'), MP3)
        self.assertEqual(f(AUDIO, chosen='FLAC', initialfile='session.mp3'), FLAC)
        self.assertEqual(f(AUDIO, defaultextension='.seg'), WAV)

    def test_patterns(self):
        self.assertEqual(D._pattern_extension('*.wav'), '.wav')
        self.assertIsNone(D._pattern_extension('*'))
        self.assertIsNone(D._pattern_extension('*.w?v'))
        self.assertEqual(D._filter_extension(('*.bowl', '*.txt')), '.bowl')


def _display():
    if not os.environ.get('DISPLAY'):
        return False
    try:
        tk.Tk().destroy()
        return True
    except tk.TclError:
        return False


class _GuiBase(unittest.TestCase):
    """Drives the real dialog: actions run from Tk timers while _choose waits."""

    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.geometry('600x400+20+20')
        ttk.Style(cls.root).theme_use('clam')

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.root.deiconify()
        self.root.update()
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.folder = Path(self.td.name)
        (self.folder / 'existant.wav').write_bytes(b'x')
        self.errors, self.jobs = [], []
        self.root.report_callback_exception = lambda *e: self.errors.append(e)
        for name, value in (('_PARENT', lambda: self.root), ('_SHORTCUTS', lambda: []),
                            ('_BEFORE_PREVIEW', lambda: None), ('_PREFS', {})):
            p = patch.object(D, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(D, '_preferences_path', return_value=self.folder / 'prefs.json')
        p.start()
        self.addCleanup(p.stop)

    def tearDown(self):
        for job in self.jobs:
            try:
                self.root.after_cancel(job)
            except tk.TclError:
                pass
        self.assertEqual(self.errors, [], 'a Tk callback raised an exception')
        self.assertIsNone(D._ACTIVE)

    def when_ready(self, action):
        def check():
            d = D._ACTIVE
            # The first scan is done (an empty list is fine: no file of the type).
            if d is not None and not d._loading and d._history:
                try:
                    action(d)
                except Exception as exc:          # reported by tearDown
                    self.errors.append(exc)
                    d.cancel()
            else:
                self.jobs.append(self.root.after(10, check))
        self.jobs.append(self.root.after(10, check))

        def timeout():
            if D._ACTIVE is not None:
                self.errors.append('timeout')
                D._ACTIVE.cancel()
        self.jobs.append(self.root.after(5000, timeout))

    @staticmethod
    def pick_type(d, index):
        """What a mouse choice in the list does: select, then the virtual event."""
        d.filter_box.current(index)
        d.filter_box.event_generate('<<ComboboxSelected>>')
        d.update()


@unittest.skipUnless(_display(), 'a display is needed (run under xvfb-run)')
class DialogTests(_GuiBase):
    def save(self, action, **options):
        self.when_ready(action)
        return D._choose('save', initialdir=str(self.folder),
                         filetypes=[("WAV", "*.wav"), ("MP3", "*.mp3"), ("FLAC", "*.flac"),
                                    ("OGG", "*.ogg"), ("All", "*.*")], **options)

    def test_the_type_picked_with_the_mouse_is_the_one_saved(self):
        def action(d):
            self.pick_type(d, MP3)
            d.name_var.set('flûte')
            d.accept()
        self.assertEqual(self.save(action), str(self.folder / 'flûte.mp3'))

    def test_a_typed_extension_wins_over_the_selected_type(self):
        def action(d):
            self.pick_type(d, MP3)
            d.name_var.set('flûte.wav')
            d.accept()
        self.assertEqual(self.save(action), str(self.folder / 'flûte.wav'))

    def test_picking_a_type_updates_the_name_shown(self):
        seen = {}

        def action(d):
            d.name_var.set('flûte.wav')
            self.pick_type(d, FLAC)
            seen['name'] = d.name_var.get()
            d.accept()
        self.assertEqual(self.save(action), str(self.folder / 'flûte.flac'))
        self.assertEqual(seen['name'], 'flûte.flac')

    def test_the_list_starts_on_the_callers_format(self):
        seen = {}

        def action(d):
            seen['type'] = d.filter_var.get()
            d.name_var.set('séance')
            d.accept()
        self.assertEqual(self.save(action, defaultextension='.ogg'), str(self.folder / 'séance.ogg'))
        self.assertTrue(seen['type'].startswith('OGG'), seen)

    def test_a_typevariable_is_honoured_and_updated(self):
        var = tk.StringVar(self.root, value='FLAC')
        seen = {}

        def action(d):
            seen['start'] = d.filter_var.get()
            self.pick_type(d, OGG)
            d.name_var.set('x')
            d.accept()
        self.assertEqual(self.save(action, typevariable=var), str(self.folder / 'x.ogg'))
        self.assertTrue(seen['start'].startswith('FLAC'))
        self.assertEqual(var.get(), 'OGG')

    def test_overwrite_is_asked_about_the_final_name(self):
        asked = []

        def action(d):
            self.pick_type(d, WAV)
            d.name_var.set('existant')
            d.accept()
        with patch.object(D.messagebox, 'askyesno', lambda title, msg, **kw: asked.append(msg) or True):
            self.assertEqual(self.save(action), str(self.folder / 'existant.wav'))
        self.assertEqual(len(asked), 1)
        self.assertIn('existant.wav', asked[0])


@unittest.skipUnless(_display(), 'a display is needed (run under xvfb-run)')
class StudioTests(_GuiBase):
    """The Meditation tab of the studio itself."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import xtts_studio
        cls.s = xtts_studio

    def setUp(self):
        super().setUp()
        self.prompts = self.folder / 'Prompts'
        self.output = self.folder / 'Output_Song_files'
        for name, value in (('_root', self.root), ('DIR_PROMPTS', str(self.prompts)),
                            ('DIR_OUTPUT', str(self.output)), ('XTTS_ROOT', str(self.folder))):
            p = patch.object(self.s, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.nb = ttk.Notebook(self.root)
        self.nb.pack()
        self.addCleanup(self.nb.destroy)
        self.s.tab_generator(self.nb)
        self.root.update()

    def widgets(self, kind, text=None):
        out, todo = [], [self.nb]
        while todo:
            w = todo.pop()
            todo.extend(w.winfo_children())
            if isinstance(w, kind) and (text is None or str(w.cget('text')) == text):
                out.append(w)
        return out

    def var(self, label_text):
        """The StringVar of the field on the row of a label."""
        label = self.widgets(tk.Label, label_text)[0]
        row = label.grid_info()['row']
        entry = [w for w in label.master.grid_slaves(row=row, column=1) if isinstance(w, tk.Entry)][0]
        return entry.cget('textvariable')

    def prompt_button(self, text):
        """A button of the Prompt Editor's toolbar."""
        found = [b for b in self.widgets(tk.Button, text)
                 if b.master.master.winfo_class() == 'TLabelframe'
                 and str(b.master.master.cget('text')) == 'Prompt Editor']
        self.assertEqual(len(found), 1, text)
        return found[0]

    def test_prompt_dialogs_open_in_prompts_then_in_the_prompts_folder(self):
        calls = []

        def fake_save(**kw):
            calls.append(('save', kw))
            return ''

        def fake_open(**kw):
            calls.append(('open', kw))
            return ''
        with patch.object(self.s.filedialog, 'asksaveasfilename', fake_save), \
             patch.object(self.s.filedialog, 'askopenfilename', fake_open):
            for text in ('Open', 'Save', 'Save as'):
                self.prompt_button(text).invoke()
            other = self.folder / 'ailleurs'
            other.mkdir()
            (other / 'mon prompt.txt').write_text('Bonjour.', encoding='utf-8')
            self.root.setvar(self.var('Script (.txt)'), str(other / 'mon prompt.txt'))
            self.prompt_button('Save as').invoke()
            self.prompt_button('Open').invoke()
        dirs = [kw.get('initialdir') for _k, kw in calls]
        self.assertEqual(dirs[:3], [str(self.prompts)] * 3)
        self.assertEqual(dirs[3:], [str(other)] * 2)
        self.assertEqual(calls[3][1].get('initialfile'), 'mon prompt.txt')
        self.assertTrue(self.prompts.is_dir(), 'Prompts/ is created when missing')

    def test_the_format_list_follows_the_output_name(self):
        fmt = [w for w in self.widgets(ttk.Combobox) if 'flac' in w.cget('values')][0]
        out_var = self.var('Output (wav/mp3)')
        self.root.setvar(out_var, '/o/séance.wav')
        self.assertEqual(self.root.getvar(fmt.cget('textvariable')), 'wav')
        self.root.setvar(out_var, '/o/séance.ogg')
        self.assertEqual(self.root.getvar(fmt.cget('textvariable')), 'ogg')
        self.root.setvar(out_var, '/o/séance')
        self.assertEqual(self.root.getvar(fmt.cget('textvariable')), 'ogg')

    def test_browse_output_through_the_real_chooser(self):
        """The reported case, end to end: Format says mp3, FLAC is picked in the
        chooser, a name is typed without extension."""
        self.output.mkdir(exist_ok=True)
        (self.output / 'déjà là.mp3').write_bytes(b'x')
        fmt = [w for w in self.widgets(ttk.Combobox) if 'flac' in w.cget('values')][0]
        out_var = self.var('Output (wav/mp3)')
        label = self.widgets(tk.Label, 'Output (wav/mp3)')[0]
        browse = [w for w in label.master.grid_slaves(row=label.grid_info()['row'], column=2)
                  if isinstance(w, tk.Button)][0]
        seen = {}

        def action(d):
            seen['start'] = d.filter_var.get()
            seen['dir'] = d.current_dir
            self.pick_type(d, FLAC)
            d.name_var.set('ma séance')
            d.accept()
        self.when_ready(action)
        browse.invoke()
        self.assertTrue(seen['start'].startswith('MP3'), seen)
        self.assertEqual(seen['dir'], str(self.output))
        self.assertEqual(self.root.getvar(out_var), str(self.output / 'ma séance.flac'))
        self.assertEqual(self.root.getvar(fmt.cget('textvariable')), 'flac')

    def test_the_annotated_prompt_field_opens_in_prompts(self):
        src = Path(self.s.__file__).read_text(encoding='utf-8')
        self.assertIn('add_row(f, "Annotated prompt (.txt)", v_txt,  1, [("Text","*.txt"),("All","*.*")], '
                      'initialdir=DIR_PROMPTS)', src)


if __name__ == '__main__':
    unittest.main()

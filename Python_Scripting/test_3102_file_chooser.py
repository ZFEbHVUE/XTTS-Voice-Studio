"""Real Tk tests for the integrated chooser. Run on a graphical display."""
import os
from pathlib import Path
import tempfile
import time
import tkinter as tk
from tkinter import ttk
import unittest
import wave
from unittest.mock import patch
import studio_file_dialog as D


def _display():
    return bool(os.environ.get('DISPLAY') or os.name=='nt' or __import__('sys').platform=='darwin')


@unittest.skipUnless(_display(), 'a graphical display is needed')
class ChooserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root=tk.Tk()
        except tk.TclError as exc:
            raise unittest.SkipTest(str(exc))
        cls.root.geometry('460x300+20+20')
        cls.root.title('XTTS Voice Studio — test')
        ttk.Style(cls.root).theme_use('clam')

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.root.deiconify();self.root.update()
        self.td=tempfile.TemporaryDirectory();self.addCleanup(self.td.cleanup)
        self.folder=Path(self.td.name)
        self.files=[self.folder/'Sonia été.wav',self.folder/'voix {douce} | 2.wav',self.folder/'notes.txt']
        for p in self.files:p.write_bytes(b'content')
        (self.folder/'sous dossier').mkdir()
        (self.folder/'.cache').touch()
        self.errors=[];self.jobs=[]
        self.root.report_callback_exception=lambda *error:self.errors.append(error)
        for name,value in [('_PARENT',lambda:self.root),('_SHORTCUTS',lambda:[]),
                           ('_BEFORE_PREVIEW',lambda:None),('_PREFS',{})]:
            p=patch.object(D,name,value);p.start();self.addCleanup(p.stop)
        p=patch.object(D,'_preferences_path',return_value=self.folder/'prefs.json');p.start();self.addCleanup(p.stop)
        self.addCleanup(self.cleanup_windows)
        self.timed_out=False

    def cleanup_windows(self):
        for job in self.jobs:
            try:self.root.after_cancel(job)
            except tk.TclError:pass
        for child in self.root.winfo_children():
            try:child.destroy()
            except tk.TclError:pass
        self.root.update()

    def tearDown(self):
        self.assertEqual(self.errors,[],'a Tk callback raised an exception')
        self.assertFalse(self.timed_out,'the chooser failed to complete')
        self.assertIsNone(D._ACTIVE)

    def after(self, delay, fn):
        self.jobs.append(self.root.after(delay,fn))

    def call_when(self, condition, action):
        def check():
            d=D._ACTIVE
            if d is not None and condition(d):
                try:action(d)
                except Exception as exc:
                    self.errors.append(exc);d.cancel()
            else:self.after(10,check)
        self.after(10,check)

    def choose(self, action, mode='open', **options):
        self.call_when(lambda d:not d._loading and bool(d._rows),action)
        def timeout():
            self.timed_out=True
            if D._ACTIVE is not None:D._ACTIVE.cancel()
        self.after(4000,timeout)
        return D._choose(mode,initialdir=str(self.folder),**options)

    def select(self,d,*names):
        ids=[iid for iid,e in d._rows.items() if e.name in names]
        self.assertEqual(len(ids),len(names))
        d.tree.selection_set(ids);d._selection_changed()

    def test_parent_stays_visible_and_the_dialog_shares_its_theme(self):
        seen={}
        def action(d):
            seen.update(mapped=self.root.winfo_ismapped(),state=self.root.state(),
                        transient=str(d.transient()),grab=self.root.grab_current(),theme=ttk.Style(d).theme_use())
            d.cancel()
        with patch.object(self.root,'withdraw',side_effect=AssertionError('parent must remain visible')), \
             patch.object(D.subprocess,'Popen',side_effect=AssertionError('no external chooser')):
            self.assertEqual(self.choose(action),'')
        self.assertEqual(seen['mapped'],1);self.assertEqual(seen['state'],'normal')
        self.assertEqual(seen['transient'],str(self.root));self.assertEqual(seen['theme'],'clam')
        self.assertIsInstance(seen['grab'],D.StudioFileDialog)
        self.assertIsNone(self.root.grab_current())

    def test_a_single_unicode_filename_is_returned(self):
        def action(d):self.select(d,self.files[0].name);d.accept()
        self.assertEqual(self.choose(action),str(self.files[0]))

    def test_multiple_names_preserve_spaces_braces_and_pipes(self):
        def action(d):
            self.select(d,self.files[0].name,self.files[1].name)
            self.assertEqual(len(d.tk.splitlist(d.name_var.get())),2)
            d.accept()
        self.assertEqual(set(self.choose(action,'multiple')),set(map(str,self.files[:2])))

    def test_a_typed_multi_filename_with_spaces_is_one_existing_file(self):
        def action(d):d.name_var.set(self.files[0].name);d.accept()
        self.assertEqual(self.choose(action,'multiple'),(str(self.files[0]),))

    def test_save_honors_initial_name_and_default_extension(self):
        def action(d):
            self.assertEqual(d.name_var.get(),'session méditation')
            d.accept()
        self.assertEqual(self.choose(action,'save',initialfile='session méditation',defaultextension='.seg'),
                         str(self.folder/'session méditation.seg'))

    def test_save_keeps_an_explicit_extension(self):
        def action(d):d.name_var.set('sortie.WAV');d.accept()
        self.assertEqual(self.choose(action,'save',defaultextension='.mp3'),str(self.folder/'sortie.WAV'))

    def test_overwrite_is_checked_after_adding_the_extension(self):
        existing=self.folder/'sortie.mp3';existing.write_bytes(b'old')
        with patch.object(D.messagebox,'askyesno',side_effect=[False,True]) as ask:
            def action(d):
                d.name_var.set('sortie');d.accept()
                self.assertFalse(d._closed)
                d.accept()
            self.assertEqual(self.choose(action,'save',defaultextension='.mp3'),str(existing))
        self.assertEqual(ask.call_count,2);self.assertEqual(existing.read_bytes(),b'old')

    def test_missing_input_is_rejected_without_closing_the_dialog(self):
        def action(d):
            d.name_var.set('missing.wav');d.accept()
            self.assertFalse(d._closed);self.assertIn('introuvable',d.status_var.get());d.cancel()
        self.assertEqual(self.choose(action),'')

    def test_escape_cancels_and_restores_the_editors_grab(self):
        editor=tk.Toplevel(self.root);editor.update();editor.grab_set()
        def action(d):
            self.assertEqual(str(d.transient()),str(editor))
            d.focus_force()
            self.after(30,lambda:d.event_generate('<Escape>'))
        self.assertEqual(self.choose(action,parent=editor),'')
        self.assertIs(self.root.grab_current(),editor)
        self.assertTrue(editor.winfo_ismapped())
        editor.grab_release()

    def test_closing_the_editor_cancels_its_child_without_hiding_the_gui(self):
        editor=tk.Toplevel(self.root);editor.update()
        self.assertEqual(self.choose(lambda d:editor.destroy(),parent=editor),'')
        self.assertTrue(self.root.winfo_ismapped())

    def test_reentry_raises_the_existing_dialog_without_a_second_dialog(self):
        def action(d):
            self.assertEqual(D.askopenfilename(parent=self.root),'')
            self.assertIs(D._ACTIVE,d)
            self.assertEqual(len([w for w in self.root.winfo_children() if isinstance(w,D.StudioFileDialog)]),1)
            d.cancel()
        self.assertEqual(self.choose(action),'')

    def test_filters_search_and_numeric_sort_actually_change_the_list(self):
        large=self.folder/'grand.WAV';large.write_bytes(b'a'*3000)
        def action(d):
            names={e.name for e in d._rows.values()}
            self.assertNotIn('notes.txt',names);self.assertNotIn('.cache',names)
            self.assertIn('grand.WAV',names);self.assertIn('sous dossier',names)
            d._sort('size');d._sort('size')
            files=[e for e in d._rows.values() if not e.is_dir]
            self.assertEqual(files[0].name,'grand.WAV')
            d.search_var.set('sonia')
            self.assertEqual([e.name for e in d._rows.values()],[self.files[0].name])
            d.cancel()
        self.choose(action,filetypes=[('Audio','*.wav *.mp3')])

    def test_browsing_a_directory_and_returning_keeps_history_correct(self):
        sub=self.folder/'sous dossier';(sub/'inside.txt').write_text('a')
        def first(d):
            d.navigate(str(sub))
            def second(dialog):
                self.assertEqual(dialog.current_dir,str(sub));dialog._history_move(-1)
                self.call_when(lambda x:not x._loading and x.current_dir==str(self.folder),lambda x:x.cancel())
            self.call_when(lambda x:not x._loading and x.current_dir==str(sub),second)
        self.choose(first)

    def test_directory_mode_can_choose_the_current_folder(self):
        self.assertEqual(self.choose(lambda d:d.accept(),'directory'),str(self.folder))

    def test_selected_wave_information_arrives_without_blocking_tk(self):
        voice=self.folder/'mesure.wav'
        with wave.open(str(voice),'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(b'\0'*48000)
        def first(d):
            self.select(d,voice.name)
            self.call_when(lambda x:'24 000 Hz' in x.details_var.get(),lambda x:x.cancel())
        self.choose(first)

    def test_preview_stops_when_the_dialog_is_closed(self):
        with patch.object(D,'PreviewPlayer') as cls:
            player=cls.return_value;player.poll.return_value=None;player.process=None
            def action(d):
                self.select(d,self.files[0].name);d._preview();d.cancel()
            self.choose(action)
            player.start.assert_called_once_with(str(self.files[0]));self.assertTrue(player.stop.called)

    def test_directory_scan_keeps_timers_alive_and_inserts_large_lists_in_batches(self):
        for n in range(650):(self.folder/f'fichier_{n}.txt').touch()
        original=D._scan_directory;ticks=[]
        def slow(*args):time.sleep(.2);original(*args)
        def tick():
            ticks.append(1)
            if D._ACTIVE is not None:self.after(20,tick)
        self.after(30,tick)
        with patch.object(D,'_scan_directory',side_effect=slow):
            self.choose(lambda d:self.call_when(lambda x:len(x._rows)>=654,lambda x:x.cancel()))
        self.assertGreaterEqual(len(ticks),5)

    def test_gui_integration_replaces_all_four_tk_dialogs(self):
        import xtts_studio as S
        self.assertIs(S.filedialog.askopenfilename,D.askopenfilename)
        self.assertIs(S.filedialog.askopenfilenames,D.askopenfilenames)
        self.assertIs(S.filedialog.asksaveasfilename,D.asksaveasfilename)
        self.assertIs(S.filedialog.askdirectory,D.askdirectory)


if __name__=='__main__':
    unittest.main()

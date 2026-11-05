"""Headless filesystem, filter, audio-header and naming tests for the Tk picker."""
import json
from pathlib import Path
import queue
import tempfile
import threading
import unittest
import wave
from unittest.mock import patch
import studio_file_dialog as D


class FileDialogLogicTests(unittest.TestCase):
    def test_numeric_size_sort_keeps_directories_first_in_both_directions(self):
        entries = [D.FileEntry('/a', 'small.wav', False, 100, 1),
                   D.FileEntry('/b', 'large.wav', False, 2000, 3),
                   D.FileEntry('/c', 'folder', True, 0, 2)]
        self.assertEqual([e.name for e in D._visible_entries(entries, ('*',), column='size')],
                         ['folder', 'small.wav', 'large.wav'])
        self.assertEqual([e.name for e in D._visible_entries(entries, ('*',), column='size', reverse=True)],
                         ['folder', 'large.wav', 'small.wav'])

    def test_filters_accept_uppercase_extensions_and_keep_directories(self):
        entries = [D.FileEntry('', 'Sonia.WAV', False, 0, 0), D.FileEntry('', 'notes.txt', False, 0, 0),
                   D.FileEntry('', 'sub', True, 0, 0), D.FileEntry('', '.secret.wav', False, 0, 0)]
        self.assertEqual([e.name for e in D._visible_entries(entries, ('*.wav',))], ['sub', 'Sonia.WAV'])
        self.assertEqual([e.name for e in D._visible_entries(entries, ('*.wav',), 'sonia')], ['Sonia.WAV'])
        self.assertEqual(len(D._visible_entries(entries, ('*.wav',), hidden=True)), 3)

    def test_all_files_matches_extensionless_names_and_tuple_patterns(self):
        filters = D._filetypes([('All', '*.*'), ('Audio', ('*.wav', '*.mp3'))])
        self.assertEqual(filters, [('All', ('*',)), ('Audio', ('*.wav', '*.mp3'))])
        self.assertEqual(len(D._visible_entries([D.FileEntry('', 'README', False, 0, 0)], filters[0][1])), 1)

    def test_default_extension_preserves_an_explicit_extension(self):
        self.assertEqual(D._with_extension('/o/flûte', '.mp3'), '/o/flûte.mp3')
        self.assertEqual(D._with_extension('/o/flûte.WAV', '.mp3'), '/o/flûte.WAV')
        self.assertEqual(D._with_extension('/o/a', 'wav'), '/o/a.wav')
        self.assertEqual(D._with_extension('/o/a', None), '/o/a')

    def test_directory_scan_preserves_spaces_braces_and_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td, 'voix {douce} | été.wav');p.write_bytes(b'a' * 4096)
            Path(td, 'folder').mkdir()
            results = queue.Queue()
            D._scan_directory(td, 7, threading.Event(), results)
            kind, token, path, entries, error = results.get_nowait()
            self.assertEqual((kind, token, path, error), ('scan', 7, td, None))
            entry = next(e for e in entries if not e.is_dir)
            self.assertEqual((entry.name, entry.path, entry.size), (p.name, str(p), 4096))

    def test_cancelled_scan_does_not_publish_results(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, 'a').touch()
            stop=threading.Event();stop.set();results=queue.Queue()
            D._scan_directory(td, 0, stop, results)
            self.assertTrue(results.empty())

    def test_scan_failure_is_reported_without_raising_on_the_ui_thread(self):
        results=queue.Queue()
        D._scan_directory('/this-directory-does-not-exist-3105', 3, threading.Event(), results)
        self.assertIsNotNone(results.get_nowait()[4])

    def test_wave_headers_give_duration_sample_rate_and_channels(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td,'voice.wav')
            with wave.open(str(p),'wb') as w:
                w.setnchannels(2);w.setsampwidth(2);w.setframerate(24000)
                w.writeframes(b'\0' * 24000 * 2 * 2)
            with patch.object(D.shutil,'which',side_effect=AssertionError('WAV needs no external tool')):
                self.assertEqual(D._audio_info(p), {'duration':1.0,'rate':24000,'channels':2})

    def test_non_audio_files_never_start_ffprobe(self):
        with patch.object(D.subprocess,'run') as run:
            self.assertIsNone(D._audio_info('/a/prompt.txt'))
        run.assert_not_called()

    def test_compressed_audio_uses_the_container_duration_when_stream_is_na(self):
        from types import SimpleNamespace
        response=SimpleNamespace(returncode=0,stdout=json.dumps({
            'streams':[{'duration':'N/A','sample_rate':'44100','channels':2}],
            'format':{'duration':'12.5'}}))
        with patch.object(D.shutil,'which',return_value='/usr/bin/ffprobe'), \
             patch.object(D.subprocess,'run',return_value=response) as run:
            self.assertEqual(D._audio_info('/a/voix été.mp3'),{'duration':12.5,'rate':44100,'channels':2})
        self.assertEqual(run.call_args.args[0][-1],'/a/voix été.mp3')

    def test_preferences_are_written_as_json_atomically(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td,'settings','file_dialog.json')
            with patch.object(D,'_preferences_path',return_value=p):
                D._save_preferences({'favorites':['/a/été'], 'size':[900,600]})
            self.assertEqual(json.loads(p.read_text()),{'favorites':['/a/été'],'size':[900,600]})
            self.assertEqual([x.name for x in p.parent.iterdir()],['file_dialog.json'])

    def test_invalid_preferences_do_not_prevent_opening_the_dialog(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td,'broken.json');p.write_text('{ broken')
            with patch.object(D,'_preferences_path',return_value=p),patch.object(D,'_PREFS',None):
                self.assertEqual(D._preferences(),{})


if __name__=='__main__':
    unittest.main()

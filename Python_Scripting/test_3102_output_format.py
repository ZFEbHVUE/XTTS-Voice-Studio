"""3.10.2: the generator writes the format its output name asks for.

Until 3.10.2 guided_meditation_generator_v23.py wrote WAV whatever the name:
an 'output.mp3' was a WAV with an .mp3 name (the size of a WAV), --mp3-bitrate
and --mp3-mode were parsed and never used, and a name without an extension
gave a file without one. The GUI's save dialog (zenity) did not add the
extension either. These tests run the real export with pydub and ffmpeg.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from test_clonage_regressions import extract_functions

GENERATOR = 'guided_meditation_generator_v23.py'


def _has(*modules):
    try:
        for name in modules:
            __import__(name)
    except Exception:
        return False
    return True


def generator_functions(extra=None):
    ns = {'os': os}
    ns.update(extra or {})
    return extract_functions(GENERATOR, {'output_format', 'resolve_output_path', 'export_final'}, ns)


def probe(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                          'format=format_name,bit_rate:stream=codec_name', '-of', 'json', str(path)],
                         capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return data['format']['format_name'], data['streams'][0]['codec_name'], int(data['format'].get('bit_rate', 0))


class OutputNameTests(unittest.TestCase):
    def setUp(self):
        self.g = generator_functions()

    def test_a_name_without_extension_gets_wav(self):
        self.assertEqual(self.g['resolve_output_path']('/out/flûte'), '/out/flûte.wav')

    def test_supported_extensions_are_kept_whatever_their_case(self):
        for name in ('a.wav', 'a.mp3', 'a.MP3', 'a.flac', 'a.ogg'):
            with self.subTest(name=name):
                self.assertEqual(self.g['resolve_output_path'](name), name)

    def test_an_unsupported_extension_stops_before_the_synthesis(self):
        with self.assertRaises(SystemExit) as caught:
            self.g['resolve_output_path']('a.m4a')
        self.assertIn('.m4a', str(caught.exception))


@unittest.skipUnless(_has('pydub') and shutil.which('ffmpeg') and shutil.which('ffprobe'),
                     'pydub, ffmpeg and ffprobe are needed')
class RealExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pydub.generators import Sine
        cls.audio = Sine(440).to_audio_segment(duration=4000).set_channels(2).set_frame_rate(44100) - 6
        cls.g = generator_functions()

    def export(self, name, *args):
        td = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, td)
        path = Path(td, name)
        label = self.g['export_final'](self.audio, str(path), *args)
        return path, label

    def test_each_extension_gives_its_own_format(self):
        expected = {'x.wav': ('wav', 'pcm_s16le'), 'x.mp3': ('mp3', 'mp3'),
                    'x.flac': ('flac', 'flac'), 'x.ogg': ('ogg', 'vorbis')}
        for name, (container, codec) in expected.items():
            with self.subTest(name=name):
                path, _ = self.export(name)
                fmt, got_codec, _ = probe(path)
                self.assertEqual((fmt, got_codec), (container, codec))

    def test_an_mp3_is_no_longer_the_size_of_a_wav(self):
        wav, _ = self.export('x.wav')
        mp3, _ = self.export('x.mp3')
        self.assertLess(mp3.stat().st_size, wav.stat().st_size / 4)

    def test_the_cbr_bit_rate_is_the_one_asked_for(self):
        for rate in (128, 320):
            with self.subTest(rate=rate):
                path, label = self.export('x.mp3', rate, 'cbr')
                self.assertAlmostEqual(probe(path)[2] / 1000, rate, delta=rate * 0.05)
                self.assertEqual(label, f'MP3 CBR {rate}k')

    def test_vbr_follows_the_quality_scale(self):
        small, label_small = self.export('x.mp3', 128, 'vbr')
        large, label_large = self.export('x.mp3', 320, 'vbr')
        self.assertEqual((label_small, label_large), ('MP3 VBR -q:a 6', 'MP3 VBR -q:a 0'))
        self.assertLess(small.stat().st_size, large.stat().st_size)


class MainWiringTests(unittest.TestCase):
    """main() checks the name first and hands the MP3 settings to the export."""

    def run_main(self, out_name, extra=()):
        calls = []
        fake_os = types.SimpleNamespace(path=os.path, _exit=lambda code: (_ for _ in ()).throw(SystemExit(code)))
        ns = generator_functions({'os': fake_os})
        ns.update(sys=sys,
                  parse_audio_files=lambda files: (files, None, []),
                  generate_meditation=lambda *a, **kw: calls.append((a, kw)))
        extract_functions(GENERATOR, {'main'}, ns)
        with tempfile.TemporaryDirectory() as td:
            script = Path(td, 'p.txt')
            script.write_text('Bonjour.', encoding='utf-8')
            argv = ['gen', str(script), str(Path(td, out_name)), 'voice.wav', *extra]
            with patch.object(sys, 'argv', argv), self.assertRaises(SystemExit):
                ns['main']()
            return calls, td

    def test_the_mp3_settings_reach_the_export(self):
        calls, _ = self.run_main('m.mp3', ['--mp3-bitrate', '320', '--mp3-mode', 'vbr'])
        (args, kw), = calls
        self.assertTrue(args[1].endswith('m.mp3'))
        self.assertEqual(kw, {'mp3_bitrate': 320, 'mp3_mode': 'vbr'})

    def test_the_extension_is_added_before_the_synthesis(self):
        calls, td = self.run_main('flûte')
        self.assertEqual(calls[0][0][1], str(Path(td, 'flûte.wav')))

    def test_an_unsupported_extension_never_starts_the_synthesis(self):
        calls, _ = self.run_main('m.m4a')
        self.assertEqual(calls, [])


@unittest.skipUnless(_has('tkinter'), 'tkinter is needed to import the GUI')
class StudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import xtts_studio
        except Exception as exc:                    # pragma: no cover
            raise unittest.SkipTest(f'xtts_studio not importable: {exc}')
        cls.s = xtts_studio

    def test_output_with_extension(self):
        f = self.s.output_with_extension
        self.assertEqual(f(' /o/flûte ', 'mp3'), ('/o/flûte.mp3', None))
        self.assertEqual(f('/o/a.WAV', 'mp3'), ('/o/a.WAV', None))     # the typed name wins
        path, problem = f('/o/a.m4a', 'mp3')
        self.assertIn('.m4a', problem)

    def test_a_single_type_field_gets_its_extension(self):
        f = self.s._single_extension
        self.assertEqual(f([("WAV", "*.wav")]), '.wav')
        self.assertEqual(f([("Text", "*.txt"), ("All", "*.*")]), '.txt')
        self.assertIsNone(f([("WAV", "*.wav"), ("MP3", "*.mp3"), ("All", "*.*")]))

    def browse(self, filetypes, **kw):
        seen, value = {}, []
        var = types.SimpleNamespace(set=value.append)

        def dialog(**kwargs):
            seen.update(kwargs)
            return '/o/chosen'
        with patch.object(self.s.filedialog, 'asksaveasfilename', dialog):
            self.s.browse_file(var, filetypes, True, None, **kw)
        return seen, value

    def test_browse_passes_the_extension_to_the_save_dialog(self):
        seen, _ = self.browse([("WAV", "*.wav")])
        self.assertEqual(seen.get('defaultextension'), '.wav')
        fmt = ['flac']
        seen, _ = self.browse([("WAV", "*.wav"), ("MP3", "*.mp3")], defaultextension=lambda: '.' + fmt[0])
        self.assertEqual(seen.get('defaultextension'), '.flac')     # read at click time
        seen, _ = self.browse([("WAV", "*.wav"), ("MP3", "*.mp3")])
        self.assertNotIn('defaultextension', seen)

    def test_custom_save_wrapper_passes_the_extension(self):
        import studio_file_dialog as dialog
        with patch.object(dialog, '_choose', return_value='/o/flûte.mp3') as choose:
            self.assertEqual(self.s._ask_save(filetypes=[('MP3', '*.mp3')], defaultextension='.mp3'),
                             '/o/flûte.mp3')
        choose.assert_called_once_with('save', filetypes=[('MP3', '*.mp3')], defaultextension='.mp3')

    def test_the_meditation_tab_uses_the_format_list(self):
        src = Path(self.s.__file__).read_text(encoding='utf-8')
        self.assertIn("defaultextension=lambda: '.' + v_gen_format.get()", src)
        self.assertIn("out, problem = output_with_extension(v_output.get(), v_gen_format.get())", src)
        self.assertIn('f"format      = {v_gen_format.get()}\\n"', src)


if __name__ == '__main__':
    unittest.main()

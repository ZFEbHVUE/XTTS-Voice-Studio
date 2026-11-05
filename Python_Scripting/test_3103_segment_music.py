"""3.10.3: a segment's music can play once, and a session sets its crossfade.

Reported from use: a spoken phrase put as a segment's music started again
before the segment ended, because the engine loops a file shorter than its
segment; and the crossfade (2 s) and fades (3 s) were fixed in the GUI, so
segments could not be joined end to end.

The engine tests count the clicks of a short file in the rendered session:
how many times it plays, and where.
"""
import os
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np

import brainwave_studio as B

SR = 8000


def write_click_file(path, seconds=1.0, at=0.1):
    """A file of silence with one loud click: each playback leaves one click."""
    n = int(seconds * SR)
    x = np.zeros(n)
    i = int(at * SR)
    x[i:i + 40] = 0.9
    pcm = (x * 32767).astype('<i2')
    with wave.open(str(path), 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def clicks(audio, threshold=0.2):
    """Start times (s) of the clicks in a rendered session."""
    m = np.max(np.abs(np.asarray(audio, dtype=float)), axis=1) if np.ndim(audio) == 2 else np.abs(audio)
    above = m > threshold
    starts = np.flatnonzero(above & ~np.r_[False, above[:-1]])
    return [round(i / SR, 3) for i in starts]


def read_wav(path):
    with wave.open(str(path), 'rb') as w:
        raw = w.readframes(w.getnframes())
        ch = w.getnchannels()
    return np.frombuffer(raw, dtype='<i2').astype(float).reshape(-1, ch) / 32768.0


class MusicStreamTests(unittest.TestCase):
    def test_loop_false_plays_once_then_silence(self):
        a = np.ones((100, 2)) * 0.5
        s = B._MusicStream(a, 2.0, SR, loop=False)
        out = np.concatenate([s.read(70), s.read(70), s.read(70)])
        np.testing.assert_allclose(out[:100], 1.0)
        np.testing.assert_allclose(out[100:], 0.0)

    def test_loop_true_is_unchanged(self):
        a = np.tile(np.linspace(-1, 1, 100)[:, None], (1, 2))
        s = B._MusicStream(a, 1.0, SR, xfade=0.0)
        out = s.read(250)
        np.testing.assert_allclose(out[100:200], out[:100])


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.music = Path(self.td.name, 'phrase.wav')
        write_click_file(self.music, seconds=1.0, at=0.1)
        self.tb = B.ToneToolbox(SR, 0.7)

    def seg(self, duration, loop=None, level=0.8):
        s = {'mode': 'isochronic', 'carrier': 180, 'beat': 6, 'duration': duration, 'duty': 0.65,
             'tone_level': 0.0, 'music': str(self.music), 'music_level': level}
        if loop is not None:
            s['music_loop'] = loop
        return B.validate_segment(s)

    def stream(self, segs, xfade, fade=0.0):
        out = Path(self.td.name, 'out.wav')
        self.tb.stream_session(segs, str(out), fade=fade, xfade=xfade)
        return read_wav(out)

    def test_a_file_shorter_than_its_segment_loops_by_default(self):
        # The loop seam is blended over 50 ms, so a 1 s file comes back every 0.95 s.
        self.assertEqual(clicks(self.stream([self.seg(4.0)], 0.0)), [0.1, 1.05, 2.0, 2.95, 3.9])

    def test_music_loop_false_plays_it_once(self):
        self.assertEqual(clicks(self.stream([self.seg(4.0, loop=False)], 0.0)), [0.1])

    def test_end_to_end_segments_start_where_the_previous_one_ends(self):
        segs = [self.seg(3.0, False), self.seg(3.0, False), self.seg(3.0, False)]
        audio = self.stream(segs, 0.0)
        self.assertEqual(len(audio), 9 * SR)
        self.assertEqual(clicks(audio), [0.1, 3.1, 6.1])

    def test_with_the_old_two_second_crossfade_they_overlap(self):
        # The second phrase starts 2 s early, inside the fade-in, at 5 % of its level.
        segs = [self.seg(3.0, False), self.seg(3.0, False)]
        audio = self.stream(segs, 2.0)
        self.assertEqual(len(audio), 4 * SR)
        self.assertEqual(clicks(audio, threshold=0.01), [0.1, 1.1])
        self.assertLess(np.max(np.abs(audio[int(1.05 * SR):int(1.2 * SR)])), 0.05)

    def test_the_in_memory_render_agrees(self):
        segs = [self.seg(3.0, False), self.seg(3.0, False)]
        audio = self.tb.build_session(segs, fade=0.0, xfade=0.0)
        self.assertEqual(clicks(audio), [0.1, 3.1])
        looped = self.tb.build_session([self.seg(3.0)], fade=0.0, xfade=0.0)
        self.assertEqual(clicks(looped), [0.1, 1.05, 2.0, 2.95])

    def test_the_single_segment_preview_agrees(self):
        music = B.load_music(str(self.music), SR)
        once = self.tb.render('isochronic', 180, 6, 3.0, fade=0.0, music=music, music_level=0.8,
                              tone_level=0.0, music_loop=False)
        self.assertEqual(clicks(once), [0.1])


class TextTests(unittest.TestCase):
    def test_music_loop_and_session_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            m = Path(td, 'p.wav')
            write_click_file(m)
            segs = [B.validate_segment({'mode': 'binaural', 'carrier': 200, 'beat': 6, 'duration': 10,
                                        'music': str(m), 'music_level': 0.2, 'music_loop': False}),
                    B.validate_segment({'mode': 'binaural', 'carrier': 200, 'beat': 6, 'duration': 10,
                                        'music': str(m), 'music_level': 0.2})]
            text = B.segments_to_text(segs, None, {'crossfade': 0.0, 'fade': 0.5})
            self.assertEqual(text.count('"music_loop": False'), 1)
            self.assertNotIn('"music_loop": True', text)
            self.assertIn('session = {"crossfade": 0, "fade": 0.5}', text)
            back = [B.validate_segment(s) for s in B.parse_segments(text)]
            self.assertEqual([s['music_loop'] for s in back], [False, True])
            self.assertEqual(B.parse_session_options(text), {'crossfade': 0.0, 'fade': 0.5})

    def test_default_session_writes_nothing(self):
        seg = B.validate_segment({'mode': 'binaural', 'carrier': 200, 'beat': 6, 'duration': 10})
        text = B.segments_to_text([seg], None, dict(B.SESSION_DEFAULTS))
        self.assertNotIn('session =', text)
        self.assertEqual(B.parse_session_options(text), {})

    def test_bad_values_are_refused(self):
        with self.assertRaises(ValueError):
            B.parse_session_options('segments = []\nsession = {"crossfade": -1}')
        with self.assertRaises(ValueError):
            B.validate_segment({'mode': 'binaural', 'carrier': 200, 'beat': 6, 'duration': 10,
                                'music': 'x.wav', 'music_loop': 'no'})

    def test_music_loop_needs_music(self):
        seg = B.validate_segment({'mode': 'binaural', 'carrier': 200, 'beat': 6, 'duration': 10,
                                  'music_loop': False})
        self.assertNotIn('music_loop', seg)


def _display():
    if not os.environ.get('DISPLAY'):
        return False
    try:
        import tkinter as tk
        tk.Tk().destroy()
        return True
    except Exception:
        return False


@unittest.skipUnless(_display(), 'a display is needed (run under xvfb-run)')
class GuiTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.addCleanup(self.root.destroy)
        self.app = B.BrainwaveStudio(self.root)
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.music = Path(self.td.name, 'phrase.wav')
        write_click_file(self.music)

    def load(self, text):
        p = Path(self.td.name, 's.seg')
        p.write_text(text, encoding='utf-8')
        with patch.object(B.filedialog, 'askopenfilename', lambda **kw: str(p)):
            self.app.load_session_file()

    def test_a_seg_brings_its_crossfade_fade_and_loop_flags(self):
        self.load(f'segments = [{{"mode": "isochronic", "carrier": 180, "beat": 6, "duration": 15, '
                  f'"duty": 0.65, "music": {str(self.music)!r}, "music_level": 0.1, "music_loop": False}}]\n'
                  'session = {"crossfade": 0, "fade": 0.5}\n')
        self.assertEqual((self.app.xfade_var.get(), self.app.fade_var.get()), ('0', '0.5'))
        self.assertEqual(self.app._session_dict(), {'crossfade': 0.0, 'fade': 0.5})
        self.app.seg_list.selection_set(0)
        self.app._load_segment()
        self.assertFalse(self.app.music_loop_var.get())
        self.assertIs(self.app._current_segment().get('music_loop'), False)

    def test_save_writes_them_back(self):
        self.load(f'segments = [{{"mode": "isochronic", "carrier": 180, "beat": 6, "duration": 15, '
                  f'"duty": 0.65, "music": {str(self.music)!r}, "music_level": 0.1, "music_loop": False}}]\n'
                  'session = {"crossfade": 0, "fade": 0.5}\n')
        out = Path(self.td.name, 'saved.seg')
        with patch.object(B.filedialog, 'asksaveasfilename', lambda **kw: str(out)):
            self.app.save_session_file()
        text = out.read_text(encoding='utf-8')
        self.assertIn('"music_loop": False', text)
        self.assertIn('session = {"crossfade": 0, "fade": 0.5}', text)

    def test_the_total_follows_the_crossfade(self):
        for _ in range(3):
            self.app.segments.append(B.validate_segment(
                {'mode': 'binaural', 'carrier': 200, 'beat': 6, 'duration': 10}))
        self.app.xfade_var.set('2')
        self.assertEqual(self.app.total_lbl.get(), 'Total: 0:26')
        self.app.xfade_var.set('0')
        self.assertEqual(self.app.total_lbl.get(), 'Total: 0:30')
        self.assertEqual(self.app._session_offset(2), 2 * 10 * self.app.SR)

    def test_a_field_being_typed_falls_back_to_the_default(self):
        self.app.xfade_var.set('')
        self.assertEqual(self.app._session_time('crossfade'), 2.0)
        self.app.fade_var.set('-1')
        self.assertEqual(self.app._session_time('fade'), 3.0)

    def test_update_keeps_the_loop_flag_of_kept_music(self):
        self.app.segments.append(B.validate_segment(
            {'mode': 'binaural', 'carrier': 200, 'beat': 6, 'duration': 10,
             'music': str(self.music), 'music_level': 0.1, 'music_loop': False}))
        self.app._refresh_list(select=0)
        self.app._clear_music(announce=False)          # the panel has no music
        self.app._music_removed = False
        self.app.update_segment()
        self.assertIs(self.app.segments[0].get('music_loop'), False)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""Regression tests: a boosting EQ no longer clamps peaks the volume would
have brought back under full scale (3.6).

A real fit (eq_mid +3.3, eq_high +4, volume -2) came out with its loudest
peaks at -2.1 dBFS: limited at -0.1 dBFS inside the EQ stage, then lowered by
2 dB.

    python -m unittest -v test_eq_headroom_regressions
"""
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from test_render_search_regressions import AudioSegment, CHAIN, generator_namespace, needs_audio


class HeadroomDecisionTests(unittest.TestCase):
    def chain(self, config):
        events = []

        def effects(a, cfg, xp, limiter=True, **kw):
            events.append(('effects', limiter, kw.get('eq_headroom_db')))
            return a
        ns = generator_namespace(
            {'render_voice_chain'}, process_audio=effects,
            apply_speed_rubberband=lambda a, speed: a,
            apply_gain_and_limit=lambda a, gain, limiter=True: events.append(('gain', gain, limiter)) or a)
        ns['render_voice_chain'](object(), dict(dict(speed=1, limiter=1), **config), {})
        return events

    def test_boost_is_taken_off_before_and_given_back_at_the_volume_step(self):
        events = self.chain(dict(eq_low=-1.2, eq_mid=3.3, eq_high=4.0, volume=-2))
        self.assertAlmostEqual(events[0][2], 7.3)
        self.assertEqual(events[1][0], 'gain')
        self.assertAlmostEqual(events[1][1], 5.3)
        self.assertTrue(events[1][2])

    def test_level_dependent_effects_or_no_limiter_keep_the_old_chain(self):
        boost = dict(eq_mid=4.0, volume=-2)
        for extra in (dict(noise_reduction=.25), dict(compression=.5),
                      dict(noise_gate=-40), dict(limiter=0)):
            events = self.chain(dict(boost, **extra))
            self.assertIsNone(events[0][2], extra)
            self.assertEqual(events[1][1], -2, extra)

    def test_cuts_only_need_no_headroom(self):
        events = self.chain(dict(eq_low=-3, eq_high=-2, volume=1.5))
        self.assertIsNone(events[0][2])
        self.assertEqual(events[1][1], 1.5)


@needs_audio
class HeadroomAudioTests(unittest.TestCase):
    """The generator's real chain through FFmpeg."""
    CONFIG = dict(speed=1.0, volume=-4.0, limiter=1, eq_low=0, eq_mid=4.0, eq_high=0,
                  highpass=0, lowpass=0, noise_reduction=0, deesser=0, compression=0,
                  noise_gate=0, reverb=0, pan=0)
    XP = dict(trim_start=0, trim_end=0, fade_in=0, fade_out=0)

    def render(self, **config):
        sr = 24000
        t = np.arange(int(1.5 * sr)) / sr
        y = 10 ** (-1 / 20) * np.sin(2 * np.pi * 1500 * t)      # peaks at -1 dBFS
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, 'in.wav')
            import soundfile as sf
            sf.write(src, y.astype(np.float32), sr, subtype='PCM_16')
            seg = AudioSegment.from_wav(src)
            printed = []
            ns = generator_namespace(CHAIN, print=lambda *a, **k: printed.append(' '.join(map(str, a))))
            out = ns['render_voice_chain'](seg, dict(self.CONFIG, **config), self.XP)
            return out.max_dBFS, '\n'.join(printed)

    def test_peaks_the_volume_brings_back_are_kept(self):
        # +4 dB on the tone then -4 dB: the true peak is -1 dBFS again.
        peak, log = self.render()
        self.assertAlmostEqual(peak, -1.0, delta=0.4)
        self.assertNotIn('EQ boost reached full scale', log)
        self.assertIn('EQ headroom: -4.0 dB', log)

    def test_the_old_chain_still_applies_with_a_level_dependent_effect(self):
        # A gate far below the tone changes nothing but the order: the EQ
        # stage limits at -0.09 dBFS, then -4 dB -> about -4.1 dBFS.
        peak, log = self.render(noise_gate=-90)
        self.assertAlmostEqual(peak, -4.1, delta=0.4)
        self.assertIn('EQ boost reached full scale', log)


if __name__ == '__main__':
    unittest.main()

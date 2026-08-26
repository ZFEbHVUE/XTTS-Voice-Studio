#!/usr/bin/env python3
"""
brainwave_studio.py — Standalone GUI: binaural / isochronic / monaural
tones + Tibetan bowl, band & chakra presets (3 tuning systems),
multi-segment SESSION mode, pink/white/brown noise, DRONE masking.

Single file: engine + Tkinter interface.

Dependencies: numpy (required), pygame (optional: audio preview).

Run: python brainwave_studio.py
"""

import ast
import math
import os
import wave
import shutil
import tempfile
import threading
import subprocess
import queue
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog

try:
    import pygame
    _HAS_PYGAME = True
except Exception:
    _HAS_PYGAME = False


BANDS = {"Delta": 2.0, "Theta": 6.0, "Alpha": 10.0, "Beta": 18.0, "Gamma": 40.0}

CHAKRAS = ["Root", "Sacral", "Plexus", "Heart", "Throat", "3rd eye", "Crown"]
TUNINGS = {
    "A=440 (notes)": [261.6, 293.7, 329.6, 349.2, 392.0, 440.0, 493.9],
    "C=256":         [256.0, 288.0, 320.0, 341.3, 384.0, 432.0, 480.0],
    "Solfeggio":     [396.0, 417.0, 528.0, 639.0, 741.0, 852.0, 963.0],
}


# ── Parameter plausibility ───────────────────────────────────────────────────
# These are physiological limits of the *perception*, not opinions about what
# sounds nice. A binaural beat is built by phase comparison in the brainstem;
# outside the range where that comparison works, there is simply no beat to
# entrain to, however carefully the file is rendered.

CARRIER_BEST = (300.0, 600.0)   # Oster 1973: beat clearest in this window
CARRIER_MAX  = 1000.0           # perception fades above; essentially gone by 1500
BEAT_BINAURAL_MAX = 30.0        # above this the two tones separate perceptually


def check_parameters(mode, carrier, beat, duration=None, headphones_ack=True):
    """Return a list of (level, message). level is 'error' | 'warn' | 'info'.

    Written to say what is KNOWN, not what is hoped: the carrier and beat
    limits come from the perception mechanism, the mode advice from which
    delivery method the published work on that band actually used.
    """
    out = []
    if mode == 'binaural':
        if carrier > CARRIER_MAX:
            out.append(('error',
                f"Carrier {carrier:.0f} Hz is above ~{CARRIER_MAX:.0f} Hz: the "
                f"brainstem phase comparison that creates a binaural beat stops "
                f"working there, so no beat is perceived at all. Lower the carrier "
                f"or switch to isochronic, which has no such limit."))
        elif not (CARRIER_BEST[0] <= carrier <= CARRIER_BEST[1]):
            out.append(('warn',
                f"Carrier {carrier:.0f} Hz is outside the {CARRIER_BEST[0]:.0f}-"
                f"{CARRIER_BEST[1]:.0f} Hz window where the beat is clearest "
                f"(Oster, 1973). It still works, less distinctly."))
        if beat > BEAT_BINAURAL_MAX:
            out.append(('error',
                f"A {beat:.0f} Hz binaural beat is past the point where the two "
                f"tones fuse into a beat -- they are heard as two separate tones. "
                f"Gamma-range work (40 Hz) uses click trains or flicker, i.e. "
                f"ISOCHRONIC delivery. Switch mode for this band."))
        if headphones_ack:
            out.append(('info',
                "Binaural REQUIRES headphones: each ear must receive its own tone. "
                "On speakers the two mix in the air and the effect is zero."))
    if mode in ('isochronic', 'monaural'):
        out.append(('info',
            f"{mode.capitalize()} works on speakers and has no carrier limit -- "
            f"the amplitude modulation is in the signal itself."))
    if beat <= 0:
        out.append(('error', "Beat frequency must be above 0 Hz."))
    if duration is not None and duration < 300:
        out.append(('warn',
            f"{duration:.0f}s is short: published protocols typically run "
            f"10-30 minutes, and effects are usually measured after several "
            f"minutes of exposure."))
    return out


HONESTY_NOTE = (
    "Technical honesty. This tool renders binaural, isochronic and monaural "
    "beats accurately -- the signal is what it claims to be. What it CANNOT "
    "claim is the effect.\n\n"
    "The entrainment hypothesis (brainwaves synchronising to the beat) remains "
    "contested: a 2023 systematic review of 14 studies found 5 supporting it, "
    "8 contradicting it, 1 mixed, and concluded the question cannot be settled "
    "yet. A 2025 randomised study across 16 configurations did find reliable "
    "entrainment, but benefits appeared only for particular combinations of "
    "frequency, carrier, masking noise and timing -- wrong settings gave no "
    "effect or reversed it.\n\n"
    "Clinical effects are better documented than the mechanism: a 2025 "
    "meta-analysis of 15 randomised trials (>1000 surgical patients) found "
    "reduced anxiety and post-operative pain versus non-binaural control audio. "
    "Whether that works through entrainment or through something else is not "
    "established.\n\n"
    "Chakra and Solfeggio frequencies are a tuning convention offered here for "
    "composition, with no measured physiological basis. Presets are starting "
    "points, not protocols."
)


BOWL_PARTIALS = [(1.0, 1.0), (2.75, 0.5), (5.18, 0.35), (8.16, 0.20), (11.66, 0.12)]

NOISE_COLORS = ["pink", "white", "brown"]

MODES = ("binaural", "isochronic", "monaural", "bowl")


def parse_segments(text):
    """Parse a 'segments' list pasted by the user.
    Accepts: a raw literal list, or a full snippet containing
    'segments = [...]'. Ramps (10, 6) are supported (tuples)."""
    text = text.strip()
    if not text:
        raise ValueError("Empty text.")
    # case 1: the text IS the list
    try:
        val = ast.literal_eval(text)
        if isinstance(val, list):
            return val
    except (ValueError, SyntaxError):
        pass
    # case 2: snippet containing 'segments = [...]'
    try:
        tree = ast.parse(text)
    except SyntaxError as e:
        raise ValueError(f"Invalid Python: {e}")
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(isinstance(t, ast.Name) and t.id == "segments" for t in node.targets):
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    raise ValueError(
                        "The 'segments' list is not a literal "
                        "(comprehensions/variables not supported). "
                        "Paste an explicit list of dictionaries.")
    raise ValueError("No 'segments' list found in the text.")


def validate_segment(seg):
    """Validate/normalize a segment. Raises ValueError if invalid."""
    seg = dict(seg)
    mode = seg.get("mode")
    if mode not in MODES:
        raise ValueError(f"invalid mode: {mode!r} (expected {MODES})")
    for k in ("carrier", "duration", "beat"):
        if k not in seg:
            raise ValueError(f"missing key: {k!r}")
    b = seg["beat"]
    if isinstance(b, list):
        b = tuple(b)
    if isinstance(b, tuple):
        if len(b) != 2:
            raise ValueError("a 'beat' ramp must have 2 values (start, end)")
        seg["beat"] = (float(b[0]), float(b[1]))
    elif isinstance(b, (int, float)):
        seg["beat"] = float(b)
    else:
        raise ValueError("'beat' must be a number or (start, end)")
    seg["carrier"] = float(seg["carrier"])
    seg["duration"] = float(seg["duration"])
    if mode == "isochronic":
        seg["duty"] = float(seg.get("duty", 0.5))
    # Beat level travels WITH the segment: a session that alternates a quiet
    # theta bed and a foreground isochronic cue needs a different balance in
    # each part, and a single global value cannot express that.
    seg["tone_level"] = float(seg.get("tone_level", 1.0))
    if seg.get("music"):
        seg["music"] = str(seg["music"])
        seg["music_level"] = float(seg.get("music_level", 0.25))
    else:
        seg.pop("music", None)
        seg.pop("music_level", None)
    return seg


def segments_to_text(segments):
    """Serialize a list of segments into copyable Python code."""
    lines = ["segments = ["]
    for s in segments:
        b = s["beat"]
        bs = f"({b[0]:g}, {b[1]:g})" if isinstance(b, tuple) else f"{b:g}"
        duty = f', "duty": {s["duty"]:g}' if s.get("mode") == "isochronic" else ""
        tl = (f', "tone_level": {s["tone_level"]:g}'
              if abs(float(s.get("tone_level", 1.0)) - 1.0) > 1e-9 else "")
        mus = (f', "music": {s["music"]!r}, "music_level": {s.get("music_level", 0.25):g}'
               if s.get("music") else "")
        lines.append(
            f'    {{"mode": "{s["mode"]}", "carrier": {s["carrier"]:g}, '
            f'"beat": {bs}, "duration": {s["duration"]:g}{duty}{tl}{mus}}},')
    lines.append("]")
    return "\n".join(lines)


# ======================================================================
#  ENGINE
# ======================================================================
class ToneToolbox:
    def __init__(self, sample_rate=44100, amplitude=0.7):
        self.sr = sample_rate
        self.amp = amplitude

    def _n(self, d):
        return int(round(self.sr * d))

    def _freq_array(self, value, n):
        if isinstance(value, (tuple, list)):
            return np.linspace(float(value[0]), float(value[1]), n)
        return np.full(n, float(value))

    def _phase(self, freq):
        return 2.0 * np.pi * np.cumsum(freq) / self.sr  # continuous phase

    # ---- tone generators ----
    def binaural(self, carrier, beat, duration):
        n = self._n(duration)
        car, bt = self._freq_array(carrier, n), self._freq_array(beat, n)
        left = np.sin(self._phase(car))
        right = np.sin(self._phase(car + bt))
        return self.amp * np.stack([left, right], axis=1)

    def isochronic(self, carrier, beat, duration, duty=0.5):
        n = self._n(duration)
        car, bt = self._freq_array(carrier, n), self._freq_array(beat, n)
        tone = np.sin(self._phase(car))
        frac = np.mod(np.cumsum(bt) / self.sr, 1.0)
        gate = np.where(frac < duty, np.sin(np.pi * frac / duty) ** 2, 0.0)
        sig = tone * gate
        return self.amp * np.stack([sig, sig], axis=1)

    def monaural(self, carrier, beat, duration):
        n = self._n(duration)
        car, bt = self._freq_array(carrier, n), self._freq_array(beat, n)
        sig = 0.5 * (np.sin(self._phase(car)) + np.sin(self._phase(car + bt)))
        return self.amp * np.stack([sig, sig], axis=1)

    def bowl(self, carrier, beat, duration):
        n = self._n(duration)
        car = self._freq_array(carrier, n)
        warble = self._freq_array(beat, n)
        out = np.zeros(n)
        for ratio, amp in BOWL_PARTIALS:
            f = car * ratio
            out += amp * np.sin(self._phase(f))
            out += amp * np.sin(self._phase(f + warble))
        out /= (np.max(np.abs(out)) or 1.0)
        sig = self.amp * out
        return np.stack([sig, sig], axis=1)

    # ---- beds ----
    def colored_noise(self, duration, level=0.1, color="pink"):
        n = self._n(duration)
        white = np.random.randn(n)
        if color == "white":
            out = white
        else:
            f = np.fft.rfftfreq(n, 1.0 / self.sr)
            f[0] = f[1] if len(f) > 1 else 1.0
            spec = np.fft.rfft(white)
            spec /= np.sqrt(f) if color == "pink" else f   # pink 1/f, brown 1/f^2
            out = np.fft.irfft(spec, n)
        out /= (np.max(np.abs(out)) or 1.0)
        return np.stack([out, out], axis=1) * level

    def drone(self, duration, root, level=0.2):
        """Harmonic drone (tanpura-style) to mask the bare carrier.
        3 slightly detuned voices per harmonic -> warmth/chorus."""
        n = self._n(duration)
        t = np.arange(n) / self.sr
        out = np.zeros(n)
        amps = [1.0, 0.5, 0.33, 0.25, 0.2]
        for i, h in enumerate((1, 2, 3, 4, 5)):
            f = root * h
            for d in (-0.15, 0.0, 0.15):
                out += amps[i] * np.sin(2.0 * np.pi * (f + d) * t)
        out /= (np.max(np.abs(out)) or 1.0)
        return np.stack([out, out], axis=1) * level

    # ---- envelope / mix ----
    def fade(self, audio, fin=2.0, fout=2.0):
        n = len(audio)
        env = np.ones(n)
        ni, no = min(self._n(fin), n // 2), min(self._n(fout), n // 2)
        if ni:
            env[:ni] = np.linspace(0.0, 1.0, ni)
        if no:
            env[-no:] = np.linspace(1.0, 0.0, no)
        return audio * env[:, None]

    @staticmethod
    def _mix(a, b):
        m = min(len(a), len(b))
        return a[:m] + b[:m]

    def _crossfade_concat(self, parts, xfade):
        nx = self._n(xfade)
        if nx <= 0 or len(parts) == 1:
            return np.concatenate(parts, axis=0)
        out = parts[0]
        ramp = np.linspace(0.0, 1.0, nx)[:, None]
        for nxt in parts[1:]:
            k = min(nx, len(out), len(nxt))
            if k <= 0:
                out = np.concatenate([out, nxt], axis=0)
                continue
            r = ramp[:k]
            blend = out[-k:] * (1 - r) + nxt[:k] * r
            out = np.concatenate([out[:-k], blend, nxt[k:]], axis=0)
        return out

    # ---- high-level renders ----
    def render(self, mode, carrier, beat, duration, duty=0.5, noise=0.0,
               noise_color="pink", drone=0.0, fade=2.0, music=None, music_level=0.25,
               duck=0.0, tone_level=1.0):
        gen = getattr(self, mode)
        kw = {"duty": duty} if mode == "isochronic" else {}
        audio = self.fade(gen(carrier, beat, duration, **kw), fade, fade)
        # The beat carries no loudness requirement. Commercial "theta meditation"
        # tracks sit it 18-25 dB UNDER the music, which is why you never hear it
        # as a test tone — the entrainment claim rests on its presence, not its
        # level. Rendering it at full scale is what makes a home-made file sound
        # like an audiometry session instead of music.
        if tone_level != 1.0:
            audio = audio * float(tone_level)
        tones = audio
        if drone > 0:
            audio = self._mix(audio, self.drone(duration, carrier, drone))
        if noise > 0:
            audio = self._mix(audio, self.colored_noise(duration, noise, noise_color))
        if music is not None and music_level > 0:
            mm = _MusicStream(music, music_level, self.sr).read(len(audio))
            if duck > 0:
                mm = mm * _Ducker(duck, self.sr).gains(tones)
            audio = self._mix(audio, mm)
        return audio

    def build_session(self, segments, noise=0.0, noise_color="pink",
                      drone=0.0, fade=3.0, xfade=2.0, music=None, music_level=0.25,
                      duck=0.0, tone_level=1.0):
        music_arrays = {}
        for s in segments:
            p = s.get("music")
            if p and p not in music_arrays:
                music_arrays[p] = load_music(p, self.sr)
        parts = []
        for seg in segments:
            gen = getattr(self, seg.get("mode", "binaural"))
            kw = {"duty": seg["duty"]} if (seg.get("mode") == "isochronic"
                                           and "duty" in seg) else {}
            part = gen(seg["carrier"], seg["beat"], seg["duration"], **kw)
            _tl = float(seg.get("tone_level", tone_level))
            if _tl != 1.0:
                part = part * _tl
            arr = music_arrays.get(seg.get("music"))
            if arr is not None:                       # this segment's own music
                mm = _MusicStream(arr, float(seg.get("music_level", 0.25)),
                                  self.sr).read(len(part))
                if duck > 0:
                    mm = mm * _Ducker(duck, self.sr).gains(part)
                part = part + mm
            parts.append(part)
        full = self._crossfade_concat(parts, xfade)
        dur = len(full) / self.sr
        tones = full
        if drone > 0:
            root = segments[0]["carrier"]
            full = self._mix(full, self.drone(dur, root, drone))
        if noise > 0:
            full = self._mix(full, self.colored_noise(dur, noise, noise_color))
        if music is not None and music_level > 0:
            mm = _MusicStream(music, music_level, self.sr).read(len(full))
            if duck > 0:
                mm = mm * _Ducker(duck, self.sr).gains(tones)
            full = self._mix(full, mm)
        return self.fade(full, fade, fade)

    def write_wav(self, audio, path, peak=0.9):
        # Only attenuate if the mix would clip. Normalising to peak rescales the
        # whole file, which silently undoes Beat level: a beat set 22 dB under
        # the music came back out at the original balance. The mix is already
        # what the user asked for; the only legitimate reason to touch it is to
        # keep it inside full scale.
        m = float(np.max(np.abs(audio)))
        if m > peak:
            print(f"   [*] Peak {20 * math.log10(m):+.1f} dBFS -> attenuated to "
                  f"{20 * math.log10(peak):+.1f} dBFS to avoid clipping")
            audio = audio * (peak / m)
        data = (audio * 32767.0).astype(np.int16)
        with wave.open(path, "w") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(data.tobytes())
        return path

    # ==================================================================
    #  STREAMING RENDER (constant memory - sessions of several hours)
    # ==================================================================
    BLOCK = 65536            # ~1.5 s per block at 44.1 kHz

    def _seg_stream(self, seg, block=None):
        """Yield stereo float blocks for ONE segment with continuous phase.
        Same math as the in-memory generators, but stateful per block."""
        block = block or self.BLOCK
        mode = seg.get("mode", "binaural")
        n = self._n(seg["duration"])
        carrier, beat = seg["carrier"], seg["beat"]
        duty = float(seg.get("duty", 0.5))
        two_pi = 2.0 * np.pi

        def freqs(value, i0, i1):
            # slice of the whole-segment linear ramp (or constant)
            if isinstance(value, (tuple, list)):
                a, b = float(value[0]), float(value[1])
                if n > 1:
                    idx = np.arange(i0, i1)
                    return a + (b - a) * idx / (n - 1)
                return np.full(i1 - i0, a)
            return np.full(i1 - i0, float(value))

        if mode == "bowl":
            nppart = len(BOWL_PARTIALS)
            ph = np.zeros(2 * nppart)
            norm = 1.0 / (2.0 * sum(a for _, a in BOWL_PARTIALS))
            for i0 in range(0, n, block):
                i1 = min(n, i0 + block)
                car = freqs(carrier, i0, i1)
                war = freqs(beat, i0, i1)
                out = np.zeros(i1 - i0)
                for j, (ratio, amp) in enumerate(BOWL_PARTIALS):
                    p1 = ph[2 * j] + two_pi * np.cumsum(car * ratio) / self.sr
                    p2 = ph[2 * j + 1] + two_pi * np.cumsum(car * ratio + war) / self.sr
                    ph[2 * j] = p1[-1] % two_pi
                    ph[2 * j + 1] = p2[-1] % two_pi
                    out += amp * (np.sin(p1) + np.sin(p2))
                sig = self.amp * out * norm
                yield np.stack([sig, sig], axis=1)
            return

        ph1 = ph2 = 0.0
        cyc = 0.0                                   # isochronic gate cycle accumulator
        for i0 in range(0, n, block):
            i1 = min(n, i0 + block)
            car = freqs(carrier, i0, i1)
            bt = freqs(beat, i0, i1)
            if mode == "binaural":
                p1 = ph1 + two_pi * np.cumsum(car) / self.sr
                p2 = ph2 + two_pi * np.cumsum(car + bt) / self.sr
                ph1, ph2 = p1[-1] % two_pi, p2[-1] % two_pi
                yield self.amp * np.stack([np.sin(p1), np.sin(p2)], axis=1)
            elif mode == "monaural":
                p1 = ph1 + two_pi * np.cumsum(car) / self.sr
                p2 = ph2 + two_pi * np.cumsum(car + bt) / self.sr
                ph1, ph2 = p1[-1] % two_pi, p2[-1] % two_pi
                sig = self.amp * 0.5 * (np.sin(p1) + np.sin(p2))
                yield np.stack([sig, sig], axis=1)
            else:                                   # isochronic
                p1 = ph1 + two_pi * np.cumsum(car) / self.sr
                ph1 = p1[-1] % two_pi
                frac = np.mod(cyc + np.cumsum(bt) / self.sr, 1.0)
                cyc = (cyc + np.sum(bt) / self.sr) % 1.0
                gate = np.where(frac < duty, np.sin(np.pi * frac / duty) ** 2, 0.0)
                sig = self.amp * np.sin(p1) * gate
                yield np.stack([sig, sig], axis=1)

    class _Reader:
        """Buffered reader over a block generator: .read(m) returns exactly m
        samples (fewer only at end of stream)."""
        def __init__(self, gen):
            self.gen, self.buf = gen, np.empty((0, 2))

        def read(self, m):
            while len(self.buf) < m:
                try:
                    nxt = next(self.gen)
                except StopIteration:
                    break
                self.buf = np.concatenate([self.buf, nxt]) if len(self.buf) else nxt
            out, self.buf = self.buf[:m], self.buf[m:]
            return out

    class _DroneStream:
        """Continuous-phase harmonic drone served in arbitrary chunk sizes."""
        def __init__(self, root, level, sr):
            self.sr, self.level = sr, level
            self.f, self.a = [], []
            amps = [1.0, 0.5, 0.33, 0.25, 0.2]
            for i, h in enumerate((1, 2, 3, 4, 5)):
                for d in (-0.15, 0.0, 0.15):
                    self.f.append(root * h + d)
                    self.a.append(amps[i])
            self.f = np.array(self.f)
            self.a = np.array(self.a)
            self.ph = np.zeros(len(self.f))
            self.norm = 1.0 / np.sum(self.a)

        def read(self, m):
            t = np.arange(1, m + 1) / self.sr
            p = self.ph[:, None] + 2.0 * np.pi * self.f[:, None] * t[None, :]
            self.ph = p[:, -1] % (2.0 * np.pi)
            out = (self.a[:, None] * np.sin(p)).sum(axis=0) * self.norm * self.level
            return np.stack([out, out], axis=1)

    class _NoiseStream:
        """Colored noise served in chunks; successive FFT-shaped blocks are
        joined with a short crossfade (inaudible for noise)."""
        def __init__(self, level, color, sr, block=1 << 17, xf=4096):
            self.level, self.color, self.sr = level, color, sr
            self.block, self.xf = block, xf
            self.buf = np.empty(0)
            self.tail = None

        def _make(self):
            n = self.block
            white = np.random.randn(n)
            if self.color == "white":
                out = white
            else:
                f = np.fft.rfftfreq(n, 1.0 / self.sr)
                f[0] = f[1] if len(f) > 1 else 1.0
                spec = np.fft.rfft(white)
                spec /= np.sqrt(f) if self.color == "pink" else f
                out = np.fft.irfft(spec, n)
            out /= (np.max(np.abs(out)) or 1.0)
            if self.tail is not None:                 # stitch with previous block
                r = np.linspace(0.0, 1.0, self.xf)
                out[:self.xf] = self.tail * (1 - r) + out[:self.xf] * r
            self.tail = out[-self.xf:].copy()
            return out[:-self.xf]                     # keep tail for next stitch

        def read(self, m):
            while len(self.buf) < m:
                self.buf = np.concatenate([self.buf, self._make()])
            out, self.buf = self.buf[:m], self.buf[m:]
            s = out * self.level
            return np.stack([s, s], axis=1)

    def _session_layout(self, lens, nx):
        """Emitted length of each segment and the total (mirrors the pipeline)."""
        L, out, tail = len(lens), [], 0
        for k, n in enumerate(lens):
            h = min(nx, n) if k > 0 else 0
            keep = min(nx, n - h) if k < L - 1 else 0
            out.append(h + max(0, n - keep - h))
            tail = keep
        return out, sum(out)

    def _seg_stream_ex(self, seg, block, music_arrays, duck_depth):
        """Segment tone stream, plus the segment's own music (looped) if any,
        ducked by the tones. Mixing before the session crossfade means two
        segments' musics blend naturally at the boundary."""
        base = self._seg_stream(seg, block)
        # The segment's beat level must be applied HERE too. The streaming path
        # (used by long exports) never received it, so a session saved this way
        # came out with every beat at full scale while the preview was correct.
        tl = float(seg.get("tone_level", 1.0))
        path = seg.get("music")
        arr = music_arrays.get(path) if path else None
        if arr is None:
            if tl == 1.0:
                yield from base
            else:
                for tones in base:
                    yield tones * tl
            return
        mus = _MusicStream(arr, float(seg.get("music_level", 0.25)), self.sr)
        duck = _Ducker(duck_depth, self.sr) if duck_depth > 0 else None
        for tones in base:
            if tl != 1.0:
                tones = tones * tl
            m = mus.read(len(tones))
            if duck is not None:
                m = m * duck.gains(tones)
            yield tones + m

    def _session_blocks(self, segments, xfade, block=None, seg_gen=None):
        """Yield the crossfaded session tone stream, block by block."""
        block = block or self.BLOCK
        if seg_gen is None:
            seg_gen = lambda seg: self._seg_stream(seg, block)
        nx = self._n(xfade)
        L = len(segments)
        tail = None
        for k, seg in enumerate(segments):
            n = self._n(seg["duration"])
            rd = self._Reader(seg_gen(seg))
            pos = 0
            if k > 0 and nx > 0:
                head = rd.read(min(nx, n))
                pos += len(head)
                m = min(len(tail) if tail is not None else 0, len(head))
                if m:
                    r = np.linspace(0.0, 1.0, m)[:, None]
                    yield tail[:m] * (1 - r) + head[:m] * r
                if len(head) > m:
                    yield head[m:]
            keep = min(nx, n - pos) if k < L - 1 else 0
            stop = max(pos, n - keep)
            while pos < stop:
                chunk = rd.read(min(block, stop - pos))
                if not len(chunk):
                    break
                pos += len(chunk)
                yield chunk
            tail = rd.read(n - pos) if keep else None

    def stream_session(self, segments, path, noise=0.0, noise_color="pink",
                       drone=0.0, fade=3.0, xfade=2.0, music=None, music_level=0.25,
                       duck=0.0, progress=None, cancel=None, block=None):
        """Stream a whole session straight to a 16-bit WAV file.
        Constant memory: hours-long sessions are fine. progress(done, total)
        is called per block; cancel is a threading.Event to abort.
        music: optional stereo float array (loop-mixed under the tones);
        duck: 0..0.9, dips any music when the tones are loud;
        segments may carry their own {"music": path, "music_level": x}."""
        if not segments:
            raise ValueError("empty session")
        block = block or self.BLOCK
        nx = self._n(xfade)
        lens = [self._n(s["duration"]) for s in segments]
        _, total = self._session_layout(lens, nx)
        nf = min(self._n(fade), total // 2)
        # load per-segment musics once (path -> array), reused across segments
        music_arrays = {}
        seg_mlev = 0.0
        for s in segments:
            p = s.get("music")
            if p:
                if p not in music_arrays:
                    music_arrays[p] = load_music(p, self.sr)
                seg_mlev = max(seg_mlev, float(s.get("music_level", 0.25)))
        # fixed, clip-safe gain (streaming cannot normalize after the fact)
        mlev = music_level if music is not None else 0.0
        gain = 0.9 / max(self.amp + drone + noise + mlev + seg_mlev, 1e-9)
        dr = self._DroneStream(segments[0]["carrier"], drone, self.sr) if drone > 0 else None
        nz = self._NoiseStream(noise, noise_color, self.sr) if noise > 0 else None
        mus = _MusicStream(music, mlev, self.sr) if music is not None else None
        gduck = _Ducker(duck, self.sr) if (duck > 0 and mus is not None) else None
        seg_gen = (lambda seg: self._seg_stream_ex(seg, block, music_arrays, duck)) \
            if music_arrays else None
        written = 0
        try:
            with wave.open(path, "w") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(self.sr)
                for chunk in self._session_blocks(segments, xfade, block, seg_gen=seg_gen):
                    if cancel is not None and cancel.is_set():
                        raise KeyboardInterrupt
                    m = len(chunk)
                    tones = chunk                      # sidechain source for ducking
                    if dr is not None:
                        chunk = chunk + dr.read(m)
                    if nz is not None:
                        chunk = chunk + nz.read(m)
                    if mus is not None:
                        mm = mus.read(m)
                        if gduck is not None:
                            mm = mm * gduck.gains(tones)
                        chunk = chunk + mm
                    # global fade in/out by absolute position
                    if nf:
                        idx = np.arange(written, written + m)
                        env = np.ones(m)
                        lo = idx < nf
                        if lo.any():
                            env[lo] = idx[lo] / nf
                        hi = idx > (total - nf)
                        if hi.any():
                            env[hi] = np.maximum(0.0, (total - idx[hi]) / nf)
                        chunk = chunk * env[:, None]
                    data = np.clip(chunk * gain, -1.0, 1.0)
                    w.writeframes((data * 32767.0).astype(np.int16).tobytes())
                    written += m
                    if progress is not None:
                        progress(written, total)
        except KeyboardInterrupt:
            try:
                os.remove(path)
            except OSError:
                pass
            return None
        return path


def convert_audio(src_wav, dst):
    """Convert a WAV to the format implied by dst's extension, via ffmpeg."""
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("Exporting to this format needs ffmpeg installed.")
    subprocess.run([exe, "-y", "-i", src_wav, dst], capture_output=True, check=True)
    return dst


def load_music(path, sr=44100):
    """Load a music file (any format) as a stereo float array at sr.
    Order: soundfile (WAV/FLAC/OGG/AIFF...) -> ffmpeg (mp3/m4a/anything)."""
    try:
        import soundfile as sf
        a, fr = sf.read(path, always_2d=True, dtype="float64")
    except Exception:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("To load this format, install 'soundfile' or ffmpeg.")
        tmp = tempfile.mktemp(suffix=".wav")
        try:
            subprocess.run([exe, "-y", "-i", path, "-ac", "2", "-ar", str(sr),
                            "-c:a", "pcm_s16le", tmp], capture_output=True, check=True)
            with wave.open(tmp, "rb") as w:
                fr = w.getframerate()
                raw = w.readframes(w.getnframes())
            a = np.frombuffer(raw, dtype="<i2").astype(np.float64).reshape(-1, 2) / 32768.0
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    if a.ndim == 1:
        a = np.stack([a, a], axis=1)
    if a.shape[1] == 1:
        a = np.repeat(a, 2, axis=1)
    if fr != sr:                                    # linear resample
        t = np.linspace(0, len(a) / fr, int(len(a) / fr * sr), endpoint=False)
        xp = np.arange(len(a)) / fr
        a = np.stack([np.interp(t, xp, a[:, 0]), np.interp(t, xp, a[:, 1])], axis=1)
    peak = np.max(np.abs(a)) or 1.0
    return (a / peak).astype(np.float64)            # normalized stereo


class _MusicStream:
    """Serve a stereo music buffer in arbitrary chunks, looped seamlessly."""
    def __init__(self, audio, level, sr, xfade=0.05):
        self.a = np.ascontiguousarray(audio, dtype=np.float64)
        self.level, self.pos, self.n = level, 0, len(self.a)
        nx = min(int(xfade * sr), self.n // 4)
        if nx > 0 and self.n > 2 * nx:              # pre-blend the loop seam once
            r = np.linspace(0.0, 1.0, nx)[:, None]
            head = self.a[:nx].copy()
            self.a[:nx] = self.a[-nx:] * (1 - r) + head * r
            self.a = self.a[:-nx]
            self.n = len(self.a)

    def read(self, m):
        out = np.empty((m, 2))
        i = 0
        while i < m:
            take = min(m - i, self.n - self.pos)
            out[i:i + take] = self.a[self.pos:self.pos + take]
            self.pos = (self.pos + take) % self.n
            i += take
        return out * self.level


class _Ducker:
    """Sidechain gain: dips the music when the tones are loud. RMS is measured
    per small window then smoothed with a slow one-pole (tau ~0.35 s) so pulsed
    modes (isochronic) breathe instead of pumping. Stateful across blocks."""
    def __init__(self, depth, sr, ref=0.5, win=1024, tau=0.35):
        self.depth = min(0.9, max(0.0, depth))
        self.ref, self.win = ref, win
        self.alpha = 1.0 - np.exp(-win / (sr * tau))
        self.env = 0.0

    def gains(self, tones):
        """Return an (n,1) gain curve in [1-depth, 1] following the tone level."""
        n = len(tones)
        if self.depth <= 0.0 or n == 0:
            return 1.0
        mono = np.abs(tones[:, 0]) if tones.ndim == 2 else np.abs(tones)
        nw = int(np.ceil(n / self.win))
        g = np.empty(nw)
        for k in range(nw):
            w = mono[k * self.win:(k + 1) * self.win]
            r = float(np.sqrt(np.mean(w * w))) if len(w) else 0.0
            self.env += self.alpha * (r - self.env)
            g[k] = 1.0 - self.depth * min(1.0, self.env / self.ref)
        x = np.arange(nw) * self.win + self.win * 0.5
        return np.interp(np.arange(n), x, g)[:, None]


# ======================================================================
#  INTERFACE
# ======================================================================
class BrainwaveStudio:
    SR = 44100

    def __init__(self, root):
        self.root = root
        self.tb = ToneToolbox(self.SR)
        self.segments = []
        self.audio_ok = self._init_audio()
        # `root` may be a Tk window (standalone) or a notebook tab (embedded in
        # XTTS Voice Studio). Window-only calls are skipped in the second case;
        # everything else — Frame, Toplevel, after — works on both.
        if isinstance(root, (tk.Tk, tk.Toplevel)):
            root.title("Brainwave Studio")
            root.resizable(False, False)
        self._build()

    def _init_audio(self):
        if not _HAS_PYGAME:
            self.audio_err = "pygame is not installed"
            return False
        # WSLg (Windows/WSL): route SDL to the WSLg PulseAudio server automatically
        if os.path.exists("/mnt/wslg/PulseServer"):
            os.environ.setdefault("SDL_AUDIODRIVER", "pulseaudio")
            os.environ.setdefault("PULSE_SERVER", "unix:/mnt/wslg/PulseServer")
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=self.SR, size=-16, channels=2)
            self.audio_err = ""
            return True
        except Exception as e:
            self.audio_err = str(e)
            return False

    def _need_audio(self):
        """Ensure audio is available; retry once, else explain instead of staying mute."""
        if self.audio_ok:
            return True
        self.audio_ok = self._init_audio()      # a device may have appeared since launch
        if self.audio_ok:
            self.status.set("Audio ready.")
            return True
        messagebox.showinfo(
            "No audio",
            "Audio preview unavailable: " + (self.audio_err or "unknown reason") +
            "\n\nFor preview:  pip install pygame\n"
            "Export / Generate audio still work without it.")
        return False

    # ---------------- build ----------------
    def _build(self):
        cont = ttk.Frame(self.root, padding=10)
        cont.grid()
        left = ttk.Frame(cont)
        left.grid(row=0, column=0, sticky="n")
        right = ttk.LabelFrame(cont, text="Session (segments)", padding=8)
        right.grid(row=0, column=1, sticky="n", padx=(12, 0))
        self._build_controls(left)
        self._build_session(right)
        # Traces are registered HERE, after _build_controls created the
        # variables: attaching them earlier raises NameError on the first fire.
        for _v in (self.mode, self.carrier, self.beat, self.duration):
            _v.trace_add('write', self._update_advice)

        self.status = tk.StringVar()
        ttk.Label(cont, textvariable=self.status, foreground="#555").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # Live plausibility line: says when the current settings cannot produce
        # the effect they claim, before minutes are spent rendering them.
        self.advice = tk.StringVar()
        self.advice_lbl = tk.Label(cont, textvariable=self.advice, justify="left",
                                   anchor="w", wraplength=980,
                                   font=("Arial", 8), fg="#8a6d00")
        self.advice_lbl.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Button(cont, text="About the evidence",
                   command=self._show_honesty).grid(row=4, column=0,
                                                    sticky="w", pady=(6, 0))
        # progress row (hidden until a generation is running)
        self.pframe = ttk.Frame(cont)
        self.pframe.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        self.pbar = ttk.Progressbar(self.pframe, length=430, mode="determinate")
        self.pbar.grid(row=0, column=0, padx=(0, 8))
        ttk.Button(self.pframe, text="Cancel", command=self._cancel_job).grid(row=0, column=1)
        self.pframe.grid_remove()
        if not self.audio_ok:
            self.status.set(f"Audio preview unavailable ({self.audio_err or 'no device'}) "
                            "— export still works.")
        else:
            self.status.set("Ready. Binaural -> use headphones.")
        self._update_advice()
        self._on_mode()


    def _show_honesty(self):
        """What this tool can and cannot claim. Same spirit as the TB-303 note:
        say what it is before saying what it does."""
        win = tk.Toplevel(self.root)
        win.title("Brainwave Studio -- what the evidence supports")
        win.transient(self.root)
        txt = tk.Text(win, wrap="word", width=84, height=26, padx=12, pady=12)
        txt.grid(row=0, column=0)
        txt.insert("1.0", HONESTY_NOTE)
        txt.config(state="disabled")
        ttk.Button(win, text="Close", command=win.destroy).grid(row=1, column=0,
                                                                pady=(0, 10))

    def _update_advice(self, *_):
        """Re-check the current settings and show anything that would stop the
        beat from existing at all."""
        try:
            mode = self.mode.get()
            carrier = float(self.carrier.get())
            beat = float(self.beat.get())
            try:
                dur = float(self.duration.get())
            except Exception:
                dur = None
            msgs = check_parameters(mode, carrier, beat, dur)
        except Exception:
            self.advice.set("")
            return
        errs = [m for l, m in msgs if l == 'error']
        warns = [m for l, m in msgs if l == 'warn']
        infos = [m for l, m in msgs if l == 'info']
        if errs:
            self.advice_lbl.config(fg="#a00")
            self.advice.set("[!] " + "\n[!] ".join(errs))
        elif warns:
            self.advice_lbl.config(fg="#8a6d00")
            self.advice.set("[~] " + "\n[~] ".join(warns))
        elif infos:
            self.advice_lbl.config(fg="#555")
            self.advice.set(infos[0])
        else:
            self.advice.set("")

    def _build_controls(self, frm):
        pad = dict(padx=6, pady=3)
        r = 0
        ttk.Label(frm, text="Mode").grid(row=r, column=0, sticky="w", **pad)
        self.mode = tk.StringVar(value="binaural")
        mf = ttk.Frame(frm)
        mf.grid(row=r, column=1, columnspan=3, sticky="w", **pad)
        for i, (lbl, val) in enumerate([("Binaural", "binaural"),
                                        ("Isochronic", "isochronic"),
                                        ("Monaural", "monaural"), ("Bowl", "bowl")]):
            ttk.Radiobutton(mf, text=lbl, value=val, variable=self.mode,
                            command=self._on_mode).grid(row=0, column=i, padx=2)
        r += 1

        self.carrier_lbl = ttk.Label(frm, text="Carrier (Hz)")
        self.carrier_lbl.grid(row=r, column=0, sticky="w", **pad)
        self.carrier = tk.DoubleVar(value=200.0)
        ttk.Spinbox(frm, from_=50, to=1100, increment=10, width=8,
                    textvariable=self.carrier).grid(row=r, column=1, sticky="w", **pad)
        r += 1

        self.beat_lbl = ttk.Label(frm, text="Beat (Hz)")
        self.beat_lbl.grid(row=r, column=0, sticky="w", **pad)
        self.beat = tk.DoubleVar(value=6.0)
        ttk.Spinbox(frm, from_=0.5, to=40, increment=0.5, width=8,
                    textvariable=self.beat).grid(row=r, column=1, sticky="w", **pad)
        self.use_ramp = tk.BooleanVar(value=False)
        ttk.Checkbutton(frm, text="Ramp to", variable=self.use_ramp,
                        command=self._on_ramp).grid(row=r, column=2, sticky="e", **pad)
        self.beat_end = tk.DoubleVar(value=10.0)
        self.ramp_spin = ttk.Spinbox(frm, from_=0.5, to=40, increment=0.5, width=8,
                                     textvariable=self.beat_end, state="disabled")
        self.ramp_spin.grid(row=r, column=3, sticky="w", **pad)
        r += 1

        ttk.Label(frm, text="Bands (beat)").grid(row=r, column=0, sticky="w", **pad)
        bf = ttk.Frame(frm)
        bf.grid(row=r, column=1, columnspan=3, sticky="w", **pad)
        for i, (lbl, val) in enumerate(BANDS.items()):
            ttk.Button(bf, text=lbl, width=6,
                       command=lambda v=val: self.beat.set(v)).grid(row=0, column=i, padx=1)
        r += 1

        ttk.Label(frm, text="Tuning").grid(row=r, column=0, sticky="w", **pad)
        self.tuning = tk.StringVar(value="A=440 (notes)")
        ttk.Combobox(frm, textvariable=self.tuning, values=list(TUNINGS),
                     state="readonly", width=15).grid(row=r, column=1, columnspan=2,
                                                       sticky="w", **pad)
        r += 1

        ttk.Label(frm, text="Chakra (carrier)").grid(row=r, column=0, sticky="w", **pad)
        cf = ttk.Frame(frm)
        cf.grid(row=r, column=1, columnspan=3, sticky="w", **pad)
        for i, name in enumerate(CHAKRAS):
            ttk.Button(cf, text=name, width=7,
                       command=lambda idx=i: self._set_chakra(idx)).grid(row=0, column=i, padx=1)
        r += 1

        ttk.Label(frm, text="Duration (s)").grid(row=r, column=0, sticky="w", **pad)
        self.duration = tk.DoubleVar(value=300.0)
        ttk.Spinbox(frm, from_=1, to=7200, increment=30, width=8,
                    textvariable=self.duration).grid(row=r, column=1, sticky="w", **pad)
        r += 1

        self.duty_lbl = ttk.Label(frm, text="Isochronic duty")
        self.duty_lbl.grid(row=r, column=0, sticky="w", **pad)
        self.duty = tk.DoubleVar(value=0.5)
        self.duty_scale = ttk.Scale(frm, from_=0.1, to=0.9, variable=self.duty, length=150)
        self.duty_scale.grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        r += 1

        # Drone
        ttk.Label(frm, text="Drone (masking)").grid(row=r, column=0, sticky="w", **pad)
        self.drone_level = tk.DoubleVar(value=0.0)
        ttk.Scale(frm, from_=0.0, to=0.5, variable=self.drone_level, length=150).grid(
            row=r, column=1, columnspan=2, sticky="w", **pad)
        self.loop = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text="Loop (preview)", variable=self.loop).grid(
            row=r, column=3, sticky="w", **pad)
        r += 1

        # Noise + color
        ttk.Label(frm, text="Noise").grid(row=r, column=0, sticky="w", **pad)
        # Beat level relative to the music, in dB. 0 = beat in front (a test
        # tone); -20 = the commercial "meditation music" balance, where the beat
        # is present but never heard as such.
        self.tone_db = tk.DoubleVar(value=0.0)
        self.noise = tk.DoubleVar(value=0.0)
        ttk.Scale(frm, from_=0.0, to=0.4, variable=self.noise, length=150).grid(
            row=r, column=1, columnspan=2, sticky="w", **pad)
        self.noise_color = tk.StringVar(value="pink")
        ttk.Combobox(frm, textvariable=self.noise_color, values=NOISE_COLORS,
                     state="readonly", width=7).grid(row=r, column=3, sticky="w", **pad)
        r += 1

        # Beat level relative to the music. This is the control that separates a
        # test tone from a listenable track: commercial meditation audio sits the
        # beat 18-25 dB under the music.
        ttk.Label(frm, text="Beat level (dB)").grid(row=r, column=0, sticky="w", **pad)
        ttk.Scale(frm, from_=-30.0, to=0.0, variable=self.tone_db,
                  length=150).grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        self.lbl_tone_db = ttk.Label(frm, text="0 dB", width=8)
        self.lbl_tone_db.grid(row=r, column=3, sticky="w", **pad)

        def _show_tone_db(*_):
            v = self.tone_db.get()
            hint = ("beat in front" if v > -6 else
                    "blended" if v > -14 else
                    "under the music (commercial balance)")
            self.lbl_tone_db.config(text=f"{v:.0f} dB  {hint}")
        self.tone_db.trace_add("write", _show_tone_db)
        _show_tone_db()
        r += 1

        # Background music (loaded file, mixed under the tones, looped)
        ttk.Button(frm, text="Load music\u2026", command=self.load_music_file).grid(
            row=r, column=0, sticky="w", **pad)
        self.music = None
        self.music_path = None
        self.music_name = tk.StringVar(value="(none)")
        ttk.Label(frm, textvariable=self.music_name, foreground="#666",
                  width=22).grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        ttk.Button(frm, text="\u2715", width=2, command=self.clear_music).grid(
            row=r, column=3, sticky="w", **pad)
        r += 1
        # Music level in dB, to be comparable with Beat level: a mix is judged by
        # the DIFFERENCE between the two, and that difference is only readable
        # when both are on the same scale. music_level stays linear internally
        # because that is what the renderer multiplies by.
        ttk.Label(frm, text="Music level (dB)").grid(row=r, column=0, sticky="w", **pad)
        self.music_level = tk.DoubleVar(value=0.25)
        self.music_db = tk.DoubleVar(value=round(20.0 * math.log10(0.25), 1))
        ttk.Scale(frm, from_=-30.0, to=0.0, variable=self.music_db,
                  length=150).grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        self.lbl_music_db = ttk.Label(frm, text="", width=14)
        self.lbl_music_db.grid(row=r, column=3, sticky="w", **pad)

        def _sync_music_db(*_):
            db = self.music_db.get()
            self.music_level.set(10.0 ** (db / 20.0))
            gap = db - self.tone_db.get()
            self.lbl_music_db.config(
                text=f"{db:.0f} dB   ({gap:+.0f} dB vs beat)")
        self.music_db.trace_add("write", _sync_music_db)
        self.tone_db.trace_add("write", _sync_music_db)
        _sync_music_db()
        r += 1
        # Ducking: music dips when the tones are loud (0 = off)
        ttk.Label(frm, text="Ducking").grid(row=r, column=0, sticky="w", **pad)
        self.duck = tk.DoubleVar(value=0.0)
        ttk.Scale(frm, from_=0.0, to=0.9, variable=self.duck, length=150).grid(
            row=r, column=1, columnspan=2, sticky="w", **pad)
        r += 1

        btns = ttk.Frame(frm)
        btns.grid(row=r, column=0, columnspan=4, pady=(10, 2))
        self.play_btn = ttk.Button(btns, text="\u25b6  Play", command=self.play)
        self.play_btn.grid(row=0, column=0, padx=4)
        ttk.Button(btns, text="\u25a0  Stop", command=self.stop).grid(row=0, column=1, padx=4)
        self.save_btn = ttk.Button(btns, text="\U0001f4be  Export", command=self.save)
        self.save_btn.grid(row=0, column=2, padx=4)

    def _build_session(self, frm):
        # 32 chars truncated the label: with beat and music levels a line runs
        # to ~50 characters, so the dB values were computed but never seen.
        self.seg_list = tk.Listbox(frm, width=52, height=12, activestyle="none")
        self.seg_list.grid(row=0, column=0, columnspan=3, sticky="nsew")
        sb = ttk.Scrollbar(frm, orient="vertical", command=self.seg_list.yview)
        sb.grid(row=0, column=3, sticky="ns")
        # Horizontal scrollbar: a line carrying mode, carrier, beat, duration,
        # beat level, music name and music level is longer than any sensible
        # fixed width, and silently truncating it hides the very values the
        # user added it for.
        hsb = ttk.Scrollbar(frm, orient="horizontal", command=self.seg_list.xview)
        hsb.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.seg_list.config(yscrollcommand=sb.set, xscrollcommand=hsb.set)
        self.seg_list.bind("<Double-Button-1>", self._load_segment)

        self.total_lbl = tk.StringVar(value="Total: 0:00")
        ttk.Label(frm, textvariable=self.total_lbl).grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))

        b = ttk.Frame(frm)
        b.grid(row=3, column=0, columnspan=4, pady=6)
        ttk.Button(b, text="+ Add", width=6, command=self.add_segment).grid(row=0, column=0, padx=1)
        ttk.Button(b, text="Update", width=7, command=self.update_segment).grid(row=0, column=1, padx=1)
        ttk.Button(b, text="\u2212 Del", width=6, command=self.remove_segment).grid(row=0, column=2, padx=1)
        ttk.Button(b, text="\u25b2", width=3,
                   command=lambda: self.move_segment(-1)).grid(row=0, column=3, padx=1)
        ttk.Button(b, text="\u25bc", width=3,
                   command=lambda: self.move_segment(1)).grid(row=0, column=4, padx=1)
        ttk.Button(b, text="Clear", width=6, command=self.clear_segments).grid(row=0, column=5, padx=1)

        bm = ttk.Frame(frm)
        bm.grid(row=4, column=0, columnspan=4, pady=(2, 0))
        ttk.Button(bm, text="\u266a Set music", command=self.set_segment_music).grid(
            row=0, column=0, padx=2)
        ttk.Button(bm, text="\u266a Remove", command=self.clear_segment_music).grid(
            row=0, column=1, padx=2)

        b2 = ttk.Frame(frm)
        b2.grid(row=5, column=0, columnspan=4, pady=(2, 0))
        self.sess_play = ttk.Button(b2, text="\u25b6 Play session", command=self.play_session)
        self.sess_play.grid(row=0, column=0, padx=2)
        self.gen_btn = ttk.Button(b2, text="\U0001f3b5 Generate audio", command=self.save_session)
        self.gen_btn.grid(row=0, column=1, padx=2)

        b3 = ttk.Frame(frm)
        b3.grid(row=6, column=0, columnspan=4, pady=(2, 0))
        ttk.Button(b3, text="\U0001f4cb Paste (import)", command=self.import_session).grid(
            row=0, column=0, padx=2)
        ttk.Button(b3, text="\U0001f4c4 Copy (export)", command=self.export_session).grid(
            row=0, column=1, padx=2)

        ttk.Label(frm, text="Drone/noise/fade apply globally.\n"
                            "\u266a = per-segment music. Double-click a\n"
                            "segment to edit it, then press Update.",
                  foreground="#888", justify="left").grid(
            row=7, column=0, columnspan=4, sticky="w", pady=(6, 0))

    # ---------------- callbacks ----------------
    def _on_mode(self):
        mode = self.mode.get()
        iso, bowl = mode == "isochronic", mode == "bowl"
        self.duty_scale.state(["!disabled"] if iso else ["disabled"])
        self.duty_lbl.configure(foreground="black" if iso else "#aaa")
        self.carrier_lbl.configure(text="Fundamental (Hz)" if bowl else "Carrier (Hz)")
        self.beat_lbl.configure(text="Warble (Hz)" if bowl else "Beat (Hz)")

    def _on_ramp(self):
        self.ramp_spin.configure(state="normal" if self.use_ramp.get() else "disabled")

    def _set_chakra(self, idx):
        self.carrier.set(TUNINGS[self.tuning.get()][idx])

    # ---------------- session ----------------
    def _current_segment(self):
        # Beat level is captured WITH the segment, like carrier and beat: a
        # session that alternates a quiet bed and a foreground cue needs its own
        # balance in each part.
        seg = {"mode": self.mode.get(), "carrier": self.carrier.get(),
               "beat": self._beat_value(), "duration": self.duration.get(),
               "tone_level": 10.0 ** (self.tone_db.get() / 20.0)}
        if self.mode.get() == "isochronic":
            seg["duty"] = self.duty.get()
        return seg

    def _seg_label(self, seg):
        b = seg["beat"]
        bs = f"{b[0]:.1f}->{b[1]:.1f}" if isinstance(b, tuple) else f"{b:.1f}"
        # Both levels are shown in dB: what matters in a mix is the gap between
        # them, and a gap is only readable on one scale.
        tl = float(seg.get("tone_level", 1.0))
        beat_db = 20.0 * math.log10(max(tl, 1e-6))
        lvl = f" | beat {beat_db:+.0f}dB" if abs(beat_db) > 0.5 else ""
        if seg.get("music"):
            ml = float(seg.get("music_level", 0.25))
            # Name as well as level: with several segments carrying different
            # files, a bare dB value does not say WHICH music is on this part.
            name = os.path.basename(str(seg["music"]))
            if len(name) > 22:
                name = name[:19] + "..."
            mus = (f" | \u266a {name} "
                   f"{20.0 * math.log10(max(ml, 1e-6)):+.0f}dB")
        else:
            mus = ""
        return (f"{seg['mode'][:4]:4s} | {seg['carrier']:.0f}Hz | "
                f"{bs}Hz | {seg['duration']:.0f}s{lvl}{mus}")

    def set_segment_music(self):
        sel = self.seg_list.curselection()
        if not sel:
            self.status.set("Select a segment first.")
            return
        path = filedialog.askopenfilename(filetypes=[
            ("Audio", "*.wav *.mp3 *.flac *.ogg *.aiff *.aif *.m4a *.aac *.opus *.wma"),
            ("All files", "*.*")])
        if not path:
            return
        if not os.path.isfile(path):
            messagebox.showerror("Cannot load music", f"File not found: {path}")
            return
        i = sel[0]
        # Ask for this segment's own level instead of silently copying the
        # global one: the whole point of per-segment music is that each part
        # can sit at a different balance against its beat.
        seg = self.segments[i]
        cur_db = 20.0 * math.log10(max(float(seg.get("music_level",
                                                     self.music_level.get())), 1e-6))
        beat_db = 20.0 * math.log10(max(float(seg.get("tone_level", 1.0)), 1e-6))
        db = simpledialog.askfloat(
            "Music level for this segment",
            f"Level in dB for {os.path.basename(path)}\n\n"
            f"This segment's beat sits at {beat_db:+.0f} dB.\n"
            f"Music 15-25 dB above the beat gives the\n"
            f"'meditation track' balance, where the beat\n"
            f"is present but never heard as a tone.",
            initialvalue=round(cur_db, 1), minvalue=-40.0, maxvalue=0.0,
            parent=self.root)
        if db is None:
            return
        seg["music"] = path
        seg["music_level"] = 10.0 ** (db / 20.0)
        self._refresh_list(select=i)
        self.status.set(f"\u266a {os.path.basename(path)} on segment {i + 1} "
                        f"at {db:+.0f} dB ({db - beat_db:+.0f} dB vs its beat).")

    def clear_segment_music(self):
        sel = self.seg_list.curselection()
        if not sel:
            return
        i = sel[0]
        self.segments[i].pop("music", None)
        self.segments[i].pop("music_level", None)
        self._refresh_list(select=i)
        self.status.set(f"Music removed from segment {i + 1}.")

    def add_segment(self):
        seg = self._current_segment()
        self.segments.append(seg)
        self.seg_list.insert("end", self._seg_label(seg))
        self._update_total()
        self.status.set(f"Segment added ({len(self.segments)} total).")

    def remove_segment(self):
        sel = self.seg_list.curselection()
        if not sel:
            return
        i = sel[0]
        self.seg_list.delete(i)
        self.segments.pop(i)
        self._update_total()

    def clear_segments(self):
        self.seg_list.delete(0, "end")
        self.segments = []
        self._update_total()

    def _refresh_list(self, select=None):
        self.seg_list.delete(0, "end")
        for s in self.segments:
            self.seg_list.insert("end", self._seg_label(s))
        if select is not None:
            self.seg_list.selection_set(select)
            self.seg_list.see(select)
        self._update_total()

    def _update_total(self):
        if not self.segments:
            self.total_lbl.set("Total: 0:00")
            return
        tot = sum(s["duration"] for s in self.segments) - 2.0 * (len(self.segments) - 1)
        tot = max(0, int(round(tot)))
        h, rem = divmod(tot, 3600)
        m, s = divmod(rem, 60)
        self.total_lbl.set(f"Total: {h}:{m:02d}:{s:02d}" if h else f"Total: {m}:{s:02d}")

    def move_segment(self, delta):
        sel = self.seg_list.curselection()
        if not sel:
            return
        i = sel[0]
        j = i + delta
        if not 0 <= j < len(self.segments):
            return
        self.segments[i], self.segments[j] = self.segments[j], self.segments[i]
        self._refresh_list(select=j)

    def _load_segment(self, _event=None):
        """Double-click: load the selected segment back into the controls."""
        sel = self.seg_list.curselection()
        if not sel:
            return
        seg = self.segments[sel[0]]
        self.mode.set(seg["mode"])
        self.carrier.set(seg["carrier"])
        self.duration.set(seg["duration"])
        b = seg["beat"]
        if isinstance(b, tuple):
            self.use_ramp.set(True)
            self.beat.set(b[0])
            self.beat_end.set(b[1])
        else:
            self.use_ramp.set(False)
            self.beat.set(b)
        if seg["mode"] == "isochronic":
            self.duty.set(seg.get("duty", 0.5))
        self._on_mode()
        self._on_ramp()
        self.status.set(f"Segment {sel[0] + 1} loaded — edit, then press Update.")

    def update_segment(self):
        """Replace the selected segment with the current control values."""
        sel = self.seg_list.curselection()
        if not sel:
            self.status.set("Select a segment to update.")
            return
        i = sel[0]
        # Keep what the controls do not carry. _current_segment() only knows
        # mode/carrier/beat/duration/level, so replacing wholesale silently
        # dropped the music attached with the button: updating any segment
        # erased its file, and the session then played only whichever music
        # had survived.
        keep = {k: self.segments[i][k] for k in ("music", "music_level")
                if k in self.segments[i]}
        seg = self._current_segment()
        seg.update(keep)
        self.segments[i] = seg
        self._refresh_list(select=i)
        kept = " (music kept)" if keep else ""
        self.status.set(f"Segment {i + 1} updated.{kept}")

    def import_session(self):
        self._text_dialog(
            "Paste a session (Python 'segments' list)", "", self._apply_import)

    def _apply_import(self, text):
        try:
            raw = parse_segments(text)
            segs = [validate_segment(s) for s in raw]
            if not segs:
                raise ValueError("The list is empty.")
        except Exception as e:
            messagebox.showerror("Import failed", str(e))
            return False
        self.clear_segments()
        for s in segs:
            self.segments.append(s)
            self.seg_list.insert("end", self._seg_label(s))
        self._update_total()
        self.status.set(f"Session imported: {len(segs)} segments.")
        return True

    def export_session(self):
        if not self.segments:
            messagebox.showinfo("Empty session", "No segment to export.")
            return
        self._text_dialog("Copy the session (Ctrl+C)",
                          segments_to_text(self.segments), None)

    def _text_dialog(self, title, initial, on_ok):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.transient(self.root)
        txt = scrolledtext.ScrolledText(win, width=58, height=16, wrap="none")
        txt.pack(padx=8, pady=8)
        txt.insert("1.0", initial)
        bar = ttk.Frame(win)
        bar.pack(pady=(0, 8))
        if on_ok is not None:
            def ok():
                if on_ok(txt.get("1.0", "end")):
                    win.destroy()
            ttk.Button(bar, text="Import", command=ok).pack(side="left", padx=4)
            txt.focus_set()
        ttk.Button(bar, text="Close", command=win.destroy).pack(side="left", padx=4)

    # ---------------- background generation (no GUI freeze) ----------------
    AUDIO_TYPES = [("WAV", "*.wav"), ("FLAC", "*.flac"), ("OGG", "*.ogg"),
                   ("MP3", "*.mp3"), ("All files", "*.*")]

    def _set_busy(self, busy):
        for b in (self.play_btn, self.save_btn, self.sess_play, self.gen_btn):
            b.state(["disabled"] if busy else ["!disabled"])

    def _cancel_job(self):
        if getattr(self, "_cancel_ev", None) is not None:
            self._cancel_ev.set()

    def _start_stream_job(self, segments, path):
        """Stream `segments` to `path` in a worker thread, with progress + cancel.
        Non-WAV extensions are written to a temp WAV then converted (ffmpeg)."""
        opts = dict(noise=self.noise.get(), noise_color=self.noise_color.get(),
                    drone=self.drone_level.get(),
                    fade=3.0 if len(segments) > 1 else 2.0, xfade=2.0,
                    music=self.music, music_level=self.music_level.get(),
                    duck=self.duck.get())
        self._cancel_ev = threading.Event()
        self._resq = queue.Queue()
        self._prog = (0, 1)
        self.pbar.configure(mode="determinate", value=0)
        self.pframe.grid()
        self._set_busy(True)

        def work():
            tmp = None
            try:
                wav = path
                if not path.lower().endswith(".wav"):
                    tmp = tempfile.mktemp(suffix=".wav",
                                          dir=os.path.dirname(path) or None)
                    wav = tmp
                res = self.tb.stream_session(
                    segments, wav,
                    progress=lambda d, t: setattr(self, "_prog", (d, t)),
                    cancel=self._cancel_ev, **opts)
                if res is None:
                    self._resq.put(("cancelled", None))
                    return
                if tmp is not None:
                    convert_audio(tmp, path)
                self._resq.put(("saved", path))
            except Exception as e:
                self._resq.put(("error", str(e)))
            finally:
                if tmp is not None:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

        threading.Thread(target=work, daemon=True).start()
        self.root.after(120, self._poll_job)

    def _start_play_build(self):
        """Build the session preview audio in a worker thread (indeterminate bar)."""
        segments = [dict(s) for s in self.segments]
        opts = dict(noise=self.noise.get(), noise_color=self.noise_color.get(),
                    drone=self.drone_level.get(), fade=3.0, xfade=2.0,
                    music=self.music, music_level=self.music_level.get(),
                    duck=self.duck.get())
        self._cancel_ev = None
        self._resq = queue.Queue()
        self._prog = (0, 0)
        self.pbar.configure(mode="indeterminate")
        self.pbar.start(12)
        self.pframe.grid()
        self._set_busy(True)
        self.status.set("Building session preview…")

        def work():
            try:
                audio = self.tb.build_session(segments, **opts)
                self._resq.put(("play", audio))
            except Exception as e:
                self._resq.put(("error", str(e)))

        threading.Thread(target=work, daemon=True).start()
        self.root.after(120, self._poll_job)

    def _poll_job(self):
        d, t = self._prog
        if t:
            self.pbar.configure(value=100.0 * d / t)
            self.status.set(f"Generating… {d / self.SR:.0f} / {t / self.SR:.0f} s")
        try:
            kind, val = self._resq.get_nowait()
        except queue.Empty:
            self.root.after(120, self._poll_job)
            return
        self.pbar.stop()
        self.pframe.grid_remove()
        self._set_busy(False)
        if kind == "saved":
            self.status.set(f"Saved: {val}")
        elif kind == "cancelled":
            self.status.set("Generation cancelled.")
        elif kind == "play":
            audio = val
            # Same rule as the single preview and the export: attenuate only on
            # clipping, never rescale a mix the user has balanced.
            m = float(np.max(np.abs(audio)))
            scale = (0.9 / m) if m > 0.9 else 1.0
            data = np.ascontiguousarray((audio * scale * 32767).astype(np.int16))
            pygame.mixer.stop()
            self._snd = pygame.sndarray.make_sound(data)
            self._snd.play()
            self.status.set(f"Playing session: {len(self.segments)} segments, "
                            f"{len(audio) / self.SR:.0f}s")
        else:
            self.status.set("Error.")
            messagebox.showerror("Error", val)

    def play_session(self):
        if not self.segments or not self._need_audio():
            return
        self._start_play_build()

    def save_session(self):
        if not self.segments:
            messagebox.showinfo("Empty session", "Add some segments first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".wav", filetypes=self.AUDIO_TYPES,
            initialfile="session.wav")
        if not path:
            return
        self._start_stream_job([dict(s) for s in self.segments], path)

    # ---------------- single ----------------
    def _beat_value(self):
        if self.use_ramp.get():
            return (self.beat.get(), self.beat_end.get())
        return self.beat.get()

    def _render(self, duration):
        return self.tb.render(
            self.mode.get(), self.carrier.get(), self._beat_value(), duration,
            duty=self.duty.get(), noise=self.noise.get(),
            noise_color=self.noise_color.get(), drone=self.drone_level.get(), fade=2.0,
            music=self.music, music_level=self.music_level.get(), duck=self.duck.get(),
            tone_level=10.0 ** (self.tone_db.get() / 20.0))

    def load_music_file(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Audio", "*.wav *.mp3 *.flac *.ogg *.aiff *.aif *.m4a *.aac *.opus *.wma"),
            ("All files", "*.*")])
        if not path:
            return
        try:
            self.music = load_music(path, self.SR)
        except Exception as e:
            messagebox.showerror("Cannot load music", str(e))
            return
        self.music_path = path
        self.music_name.set(os.path.basename(path))
        self.status.set(f"Music loaded: {os.path.basename(path)} "
                        f"({len(self.music) / self.SR:.0f}s, looped under the tones).")

    def clear_music(self):
        self.music = None
        self.music_path = None
        self.music_name.set("(none)")
        self.status.set("Background music removed.")

    def play(self):
        if not self._need_audio():
            return
        try:
            dur = min(self.duration.get(), 12.0) if self.loop.get() else self.duration.get()
            audio = self._render(dur)
            # Do NOT normalise to peak here. Peak normalisation rescales the
            # whole mix, so lowering the beat by 20 dB and then dividing by the
            # new peak restores the original balance: the preview sounded
            # identical whatever Beat level was set, while the exported file was
            # correct. Only pull the level down if the mix actually clips.
            m = float(np.max(np.abs(audio)))
            scale = (0.9 / m) if m > 0.9 else 1.0
            if scale < 1.0:
                self.status.set(f"Peak {20 * math.log10(m):+.1f} dBFS -> "
                                f"attenuated to avoid clipping")
            data = np.ascontiguousarray((audio * scale * 32767).astype(np.int16))
            pygame.mixer.stop()
            self._snd = pygame.sndarray.make_sound(data)
            self._snd.play(loops=-1 if self.loop.get() else 0)
            self.status.set(f"Playing… {self.mode.get()} | {self.carrier.get():.0f} Hz")
        except Exception as e:
            self.status.set(f"Playback error: {e}")

    def stop(self):
        if self.audio_ok:
            pygame.mixer.stop()
        self.status.set("Stopped.")

    def save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".wav", filetypes=self.AUDIO_TYPES,
            initialfile=f"{self.mode.get()}_{int(self.carrier.get())}Hz.wav")
        if not path:
            return
        try:
            seg = validate_segment(self._current_segment())
        except Exception as e:
            messagebox.showerror("Invalid settings", str(e))
            return
        self._start_stream_job([seg], path)


def main():
    root = tk.Tk()
    BrainwaveStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()

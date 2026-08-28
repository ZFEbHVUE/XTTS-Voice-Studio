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

# A real bowl is a set of INDEPENDENT vibration modes, not a harmonic series.
# Each mode has its own frequency, its own loudness, its own decay time and --
# because the hammered bowl is never perfectly circular -- its own beat rate:
# the asymmetry splits each mode into two close frequencies whose interference
# gives that slow pulsing. Measuring a bowl with a phone analyser gives exactly
# this table, which is why it is editable rather than derived from ratios.
#
# freq_hz, amplitude (0-1), decay_s (time to fall ~63%), beat_hz (0 = no beat)
BOWL_PRESETS = {
    "Generic small":  [(432.0, 1.00, 18.0, 1.4),
                       (1188.0, 0.50, 6.0, 3.0),
                       (2238.0, 0.30, 2.5, 4.5)],
    "Generic medium": [(256.0, 1.00, 35.0, 0.9),
                       (704.0, 0.55, 12.0, 2.1),
                       (1326.0, 0.32, 5.0, 3.4),
                       (2089.0, 0.18, 2.5, 5.0)],
    "Generic large":  [(148.0, 1.00, 60.0, 0.6),
                       (407.0, 0.60, 22.0, 1.3),
                       (767.0, 0.35, 9.0, 2.2),
                       (1208.0, 0.20, 4.0, 3.1),
                       (1726.0, 0.12, 2.0, 4.0)],
}



def analyse_bowl_wav(path, sr_target=44100, max_modes=8, floor_db=-55.0):
    """Extract a bowl's vibration modes from a recording.

    Returns [(freq_hz, amp, decay_s, beat_hz), ...].

    Method, and why each step: a struck bowl rings its modes at once, so the
    spectrum of the whole take shows them as clear peaks -- that gives the
    frequencies. Each mode is then band-pass filtered on its own and its
    envelope followed: the slope of log(envelope) is the decay time, and the
    envelope's own ripple is the beat rate (the split pair inside that mode).
    Measuring the beat per mode is the point -- a single global beat is exactly
    the approximation that made the synthetic bowl sound wrong.
    """
    import numpy as np
    import soundfile as _sf
    y, sr = _sf.read(path, dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) > 1:
        y = y.mean(axis=1)
    if len(y) < sr // 2:
        return []

    # 1. Peaks of the long-term spectrum -> candidate mode frequencies
    n = int(2 ** np.ceil(np.log2(min(len(y), sr * 8))))
    seg = y[:n] * np.hanning(n)
    mag = np.abs(np.fft.rfft(seg))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    keep = (freqs > 60) & (freqs < min(6000, sr / 2 - 100))
    mag, freqs = mag[keep], freqs[keep]
    if not len(mag):
        return []
    peak = mag.max()
    # -55 dB, not -38: the upper modes of a bowl are genuinely quiet (the third
    # mode of the test bowl sits at -21 dB before smoothing and lower after) and
    # a tighter floor simply drops them. Weak candidates are filtered later by
    # amplitude and by the 3% spacing rule.
    thr = peak * (10 ** (floor_db / 20.0))
    # Smooth before peak-picking: each mode is a DOUBLET (f and f+beat), so raw
    # bins wobble and a strict local-maximum test finds nothing. The smoothing
    # width is a few bins -- wide enough to merge the pair, narrow enough to
    # keep two distinct modes apart.
    w = max(3, int(round(3.0 / max(freqs[1] - freqs[0], 1e-9))))
    if w % 2 == 0:
        w += 1
    sm = np.convolve(mag, np.ones(w) / w, mode="same")
    cand = []
    for i in range(2, len(sm) - 2):
        if sm[i] > thr and sm[i] >= sm[i-1] and sm[i] >= sm[i+1]:
            # parabolic interpolation for sub-bin accuracy
            # Parabolic interpolation for sub-bin accuracy. The denominator is
            # near zero on a smoothed peak and can be NEGATIVE, so clamping it
            # with max(x, 1e-12) turned it into a division by 1e-12 and threw
            # the frequency into the millions. Guard on magnitude, and keep the
            # correction inside one bin, which is all it can legitimately be.
            a, b, c = sm[i-1], sm[i], sm[i+1]
            den = a - 2.0 * b + c
            d = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
            d = float(np.clip(d, -0.5, 0.5))
            cand.append((float(freqs[i] + d * (freqs[1] - freqs[0])), float(b)))
    if not cand:
        return []
    cand.sort(key=lambda t: -t[1])
    # drop peaks within 3% of a stronger one (same mode, adjacent bins)
    picked = []
    for f, a in cand:
        if all(abs(f - pf) / pf > 0.03 for pf, _ in picked):
            picked.append((f, a))
        if len(picked) >= max_modes:
            break
    picked.sort(key=lambda t: t[0])
    amax = max(a for _, a in picked)

    # 2. Per-mode envelope -> decay time and beat rate
    from numpy.fft import rfft as _rfft, rfftfreq as _rfreq
    out = []
    N = len(y)
    Y = np.fft.rfft(y)
    ff = np.fft.rfftfreq(N, 1.0 / sr)
    for f, a in picked:
        bw = max(12.0, f * 0.04)
        band = np.zeros_like(Y)
        m = (ff > f - bw) & (ff < f + bw)
        band[m] = Y[m]
        comp = np.fft.irfft(band, n=N)
        env = np.abs(comp)
        w = max(1, int(0.02 * sr))
        env = np.convolve(env, np.ones(w) / w, mode="same")
        if env.max() <= 0:
            continue
        # decay: slope of log envelope over the part that is still above noise
        i0 = int(np.argmax(env))
        tail = env[i0:]
        good = tail > tail.max() * 0.08
        k = int(np.sum(good))
        if k > sr // 8:
            t = np.arange(k) / sr
            lg = np.log(np.maximum(tail[:k], 1e-12))
            slope = np.polyfit(t, lg, 1)[0]
            decay = float(-1.0 / slope) if slope < -1e-6 else 30.0
        else:
            decay = 30.0
        decay = float(min(max(decay, 0.3), 120.0))
        # beat: ripple frequency of that mode's own envelope
        e = tail[:k] if k > sr // 4 else tail
        e = e - e.mean()
        if len(e) > sr // 2:
            E = np.abs(_rfft(e * np.hanning(len(e))))
            EF = _rfreq(len(e), 1.0 / sr)
            band2 = (EF > 0.2) & (EF < 20.0)
            beat = float(EF[band2][np.argmax(E[band2])]) if band2.any() else 0.0
            if E[band2].max() < E.mean() * 3:
                beat = 0.0                       # ripple not convincing
        else:
            beat = 0.0
        out.append((round(f, 1), round(float(a / amax), 3),
                    round(decay, 1), round(beat, 2)))
    return out



def transpose_bowl_modes(modes, target_hz, scale_decay=True):
    """Move a measured bowl to another fundamental, keeping its identity.

    A bowl's character lives in the RATIOS between its modes, their decays and
    their beat rates -- not in its absolute pitch. Scaling all of it by one
    factor is the same as playing a bowl of a different size from the same
    workshop, which is exactly what a set of tuned bowls is.

    What scales with the factor k:
      - every mode frequency (by definition)
      - every beat rate: the detuning is a fraction of the mode, so it follows
      - the decays, if scale_decay: a smaller bowl rings shorter. Physically the
        radiated power rises with frequency, so decay goes roughly as 1/k. Pass
        False to keep the measured decays as they are.

    Amplitudes are left alone: they describe how the bowl was struck.
    """
    if not modes or not target_hz or target_hz <= 0:
        return list(modes or [])
    f0 = float(modes[0][0])
    if f0 <= 0:
        return list(modes)
    k = float(target_hz) / f0
    out = []
    for f, a, d, b in modes:
        nd = (float(d) / k) if (scale_decay and k > 0) else float(d)
        out.append((round(float(f) * k, 1), float(a),
                    round(max(nd, 0.05), 2), round(float(b) * k, 2)))
    return out


def parse_bowl_modes(text):
    """Read a bowl table: one mode per line, 'freq amp decay beat'.

    Amplitude, decay and beat are optional and fall back to sensible values, so
    a bare list of frequencies measured with a phone is already usable.
    """
    modes = []
    for raw in (text or "").splitlines():
        line = raw.split("#")[0].strip().replace(",", " ")
        if not line:
            continue
        parts = line.split()
        try:
            f = float(parts[0])
        except ValueError:
            continue
        if f <= 0:
            continue
        a = float(parts[1]) if len(parts) > 1 else 1.0
        # Higher modes radiate more and die sooner; a rough default when the
        # user only typed frequencies.
        d = float(parts[2]) if len(parts) > 2 else max(2.0, 40.0 * (modes and
              modes[0][0] or f) / f)
        b = float(parts[3]) if len(parts) > 3 else 0.0
        modes.append((f, max(0.0, a), max(0.05, d), max(0.0, b)))
    return modes


def bowl_modes_to_text(modes):
    head = "# freq_hz  amp  decay_s  beat_hz\n"
    return head + "\n".join(f"{f:g} {a:g} {d:g} {b:g}" for f, a, d, b in modes)



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

    def isochronic(self, carrier, beat, duration, duty=0.5, stereo_phase=0.0):
        """Isochronic pulses, optionally offset between the two channels.

        stereo_phase is a fraction of one beat period (0.5 = 180 degrees). It is
        a SPATIAL effect, not a stronger stimulus: with an offset the pulses
        alternate between the ears and the sound seems to move around or through
        the head. Nothing interferes inside the skull -- a 6 Hz modulation has a
        wavelength of about 57 m, far too long to focus anywhere in a 20 cm head.

        Beware on speakers: at 0.5 the channels are in opposition and a mono
        sum largely cancels them.
        """
        n = self._n(duration)
        car, bt = self._freq_array(carrier, n), self._freq_array(beat, n)
        tone = np.sin(self._phase(car))
        frac = np.mod(np.cumsum(bt) / self.sr, 1.0)

        def _gate(offset):
            f = np.mod(frac + offset, 1.0)
            return np.where(f < duty, np.sin(np.pi * f / duty) ** 2, 0.0)

        left = tone * _gate(0.0)
        if abs(stereo_phase) < 1e-6:
            right = left
        else:
            right = tone * _gate(float(stereo_phase))
        return self.amp * np.stack([left, right], axis=1)

    def monaural(self, carrier, beat, duration):
        n = self._n(duration)
        car, bt = self._freq_array(carrier, n), self._freq_array(beat, n)
        sig = 0.5 * (np.sin(self._phase(car)) + np.sin(self._phase(car + bt)))
        return self.amp * np.stack([sig, sig], axis=1)

    def bowl(self, carrier, beat, duration, strike_s=None, modes=None):
        """Struck singing bowl.

        modes: explicit list of (freq_hz, amp, decay_s, beat_hz) -- a measured
        bowl. Each mode is independent: its own frequency, loudness, decay and
        beat rate, which is what a phone analyser actually shows on a real bowl.

        Without modes, the historic behaviour: partials as RATIOS of `carrier`,
        detuned proportionally by `beat`. Proportional matters -- adding the
        same absolute detune to every partial made the five pairs beat at
        unrelated rates, so asking for 2 Hz produced 14.
        """
        n = self._n(duration)
        t = np.arange(n) / self.sr
        strike = float(strike_s) if strike_s else max(6.0, duration / 4.0)
        out = np.zeros(n)

        if modes:
            for f, amp, dec, bt in modes:
                env = np.exp(-t / max(float(dec), 0.05))
                ph = 2.0 * np.pi * float(f) * t
                out += amp * env * np.sin(ph)
                if bt and bt > 0:
                    # The mode's twin, offset by its OWN beat rate: this pair is
                    # what produces the pulsing, one rate per mode.
                    out += amp * env * np.sin(2.0 * np.pi * (float(f) + float(bt)) * t)
            longest = max((float(d) for _, _, d, _ in modes), default=strike)
        else:
            car = self._freq_array(carrier, n)
            rel = float(beat[0] if isinstance(beat, tuple) else beat) / max(float(
                carrier[0] if isinstance(carrier, tuple) else carrier), 1e-9)
            for ratio, amp in BOWL_PARTIALS:
                f = car * ratio
                tau = strike / (1.0 + 1.4 * (ratio - 1.0))
                env = np.exp(-t / max(tau, 0.05))
                out += amp * env * np.sin(self._phase(f))
                out += amp * env * np.sin(self._phase(f * (1.0 + rel)))
            longest = strike

        # Re-strike so a long segment keeps ringing instead of fading to nothing.
        period = int(max(longest, 1.0) * self.sr)
        if period > 0 and n > period:
            hits = np.zeros(n)
            for start in range(0, n, period):
                seg = out[:min(period, n - start)]
                hits[start:start + len(seg)] += seg
            out = hits
        at = min(int(0.004 * self.sr), n)          # audible strike
        if at > 1:
            out[:at] *= np.linspace(0.0, 1.0, at)
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
               duck=0.0, tone_level=1.0, stereo_phase=0.0,
               rot_rpm=0.0, rot_depth=1.0,
               level_drone=False, level_noise=False, bowl_modes=None):
        gen = getattr(self, mode)
        kw = ({"duty": duty, "stereo_phase": stereo_phase}
              if mode == "isochronic" else
              {"modes": bowl_modes} if mode == "bowl" and bowl_modes else {})
        audio = self.fade(gen(carrier, beat, duration, **kw), fade, fade)
        # The beat carries no loudness requirement. Commercial "theta meditation"
        # tracks sit it 18-25 dB UNDER the music, which is why you never hear it
        # as a test tone — the entrainment claim rests on its presence, not its
        # level. Rendering it at full scale is what makes a home-made file sound
        # like an audiometry session instead of music.
        if tone_level != 1.0:
            audio = audio * float(tone_level)
        tones = audio
        # level_drone / level_noise decide what "Beat level" governs. Both off
        # (historic): only the beat is attenuated while drone and noise keep
        # their own levels -- which is how you end up hearing nothing but the
        # drone, identical in every mode, since the mode changes the beat and
        # not the bed. Switched on, that source follows the beat, so the balance
        # set at 0 dB survives being pushed under the music.
        # Drone and noise follow Beat level independently: the drone is tuned to
        # the carrier and masks the beat directly, the noise is broadband and
        # masks the whole bed, so wanting one to follow and not the other is a
        # real case.
        _bd = float(tone_level) if level_drone else 1.0
        _bn = float(tone_level) if level_noise else 1.0
        if drone > 0:
            audio = self._mix(audio, self.drone(duration, carrier, drone) * _bd)
        if noise > 0:
            audio = self._mix(audio,
                              self.colored_noise(duration, noise, noise_color) * _bn)
        if music is not None and music_level > 0:
            mm = _MusicStream(music, music_level, self.sr).read(len(audio))
            if duck > 0:
                mm = mm * _Ducker(duck, self.sr).gains(tones)
            audio = self._mix(audio, mm)
        # Rotation last, on the finished mix: a global effect like drone and
        # noise, so the image keeps turning across segment boundaries instead
        # of snapping back at each one.
        audio = apply_rotation(audio, self.sr, rot_rpm, rot_depth)
        return audio

    def build_session(self, segments, noise=0.0, noise_color="pink",
                      drone=0.0, fade=3.0, xfade=2.0, music=None, music_level=0.25,
                      duck=0.0, tone_level=1.0, stereo_phase=0.0,
               rot_rpm=0.0, rot_depth=1.0,
               level_drone=False, level_noise=False):
        music_arrays = {}
        for s in segments:
            p = s.get("music")
            if p and p not in music_arrays:
                music_arrays[p] = load_music(p, self.sr)
        parts = []
        _rot_pos = 0                      # running sample count for rotation phase
        for seg in segments:
            gen = getattr(self, seg.get("mode", "binaural"))
            kw = {}
            if seg.get("mode") == "bowl" and seg.get("bowl_modes"):
                kw["modes"] = seg["bowl_modes"]
            if seg.get("mode") == "isochronic":
                if "duty" in seg:
                    kw["duty"] = seg["duty"]
                if seg.get("stereo_phase"):
                    kw["stereo_phase"] = float(seg["stereo_phase"])
            part = gen(seg["carrier"], seg["beat"], seg["duration"], **kw)
            _tl = float(seg.get("tone_level", tone_level))
            if _tl != 1.0:
                part = part * _tl
            # Drone and noise are the segment's own now, like everything else on
            # the left panel. The drone follows THIS segment's carrier instead
            # of the first segment's, which is what it should always have done.
            _pdur = len(part) / self.sr
            _bd = float(_tl) if seg.get("level_drone", level_drone) else 1.0
            _bn = float(_tl) if seg.get("level_noise", level_noise) else 1.0
            _dr = float(seg.get("drone", drone))
            if _dr > 0:
                part = self._mix(part, self.drone(_pdur, seg["carrier"], _dr) * _bd)
            _nz = float(seg.get("noise", noise))
            if _nz > 0:
                part = self._mix(part, self.colored_noise(
                    _pdur, _nz, seg.get("noise_color", noise_color)) * _bn)
            _dk = float(seg.get("duck", duck))
            arr = music_arrays.get(seg.get("music"))
            if arr is not None:                       # this segment's own music
                mm = _MusicStream(arr, float(seg.get("music_level", 0.25)),
                                  self.sr).read(len(part))
                if _dk > 0:
                    mm = mm * _Ducker(_dk, self.sr).gains(part)
                part = part + mm
            # Rotation belongs to the segment, like every other left-panel
            # control. The phase is carried across segments so two consecutive
            # parts at the same speed keep turning smoothly instead of snapping
            # back to centre at the join.
            _rr = float(seg.get("rot_rpm", rot_rpm))
            _rd = float(seg.get("rot_depth", rot_depth))
            if _rr > 0 and _rd > 0:
                part = apply_rotation(part, self.sr, _rr, _rd,
                                      phase0=2 * np.pi * (_rr / 60.0)
                                      * (_rot_pos / self.sr))
            _rot_pos += len(part)
            parts.append(part)
        full = self._crossfade_concat(parts, xfade)
        dur = len(full) / self.sr
        tones = full
        # Drone and noise were applied here, once, over the whole session; they
        # are per segment now (see the loop above). Only the global music bed
        # remains a session-wide layer.
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
        sph = float(seg.get("stereo_phase", 0.0))
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
            # This path had BOWL_PARTIALS hard-coded and no envelope, so an
            # export ignored both the measured modes and the strike decay that
            # the preview had -- the same "option added on one path only" trap
            # that already cost tone_level and the peak normalisation.
            bmodes = seg.get("bowl_modes")
            t0 = 0
            if bmodes:
                ph = np.zeros(2 * len(bmodes))
                norm = 1.0 / (2.0 * max(sum(a for _, a, _, _ in bmodes), 1e-9))
                longest = max(d for _, _, d, _ in bmodes)
                for i0 in range(0, n, block):
                    i1 = min(n, i0 + block)
                    tt = (t0 + np.arange(i1 - i0)) / self.sr
                    tt = np.mod(tt, max(longest, 1.0))      # re-strike
                    out = np.zeros(i1 - i0)
                    for j, (f, amp, dec, bt) in enumerate(bmodes):
                        env = np.exp(-tt / max(dec, 0.05))
                        p1 = ph[2 * j] + two_pi * f * np.arange(1, i1 - i0 + 1) / self.sr
                        out += amp * env * np.sin(p1)
                        ph[2 * j] = p1[-1] % two_pi
                        if bt and bt > 0:
                            p2 = ph[2 * j + 1] + two_pi * (f + bt) * np.arange(1, i1 - i0 + 1) / self.sr
                            out += amp * env * np.sin(p2)
                            ph[2 * j + 1] = p2[-1] % two_pi
                    t0 += (i1 - i0)
                    sig = self.amp * out * norm
                    yield np.stack([sig, sig], axis=1)
                return
            nppart = len(BOWL_PARTIALS)
            ph = np.zeros(2 * nppart)
            norm = 1.0 / (2.0 * sum(a for _, a in BOWL_PARTIALS))
            strike = max(6.0, seg["duration"] / 4.0)
            rel = float(beat[0] if isinstance(beat, tuple) else beat) / max(float(
                carrier[0] if isinstance(carrier, tuple) else carrier), 1e-9)
            for i0 in range(0, n, block):
                i1 = min(n, i0 + block)
                car = freqs(carrier, i0, i1)
                tt = np.mod((t0 + np.arange(i1 - i0)) / self.sr, strike)
                out = np.zeros(i1 - i0)
                for j, (ratio, amp) in enumerate(BOWL_PARTIALS):
                    tau = strike / (1.0 + 1.4 * (ratio - 1.0))
                    env = np.exp(-tt / max(tau, 0.05))
                    p1 = ph[2 * j] + two_pi * np.cumsum(car * ratio) / self.sr
                    p2 = ph[2 * j + 1] + two_pi * np.cumsum(car * ratio * (1.0 + rel)) / self.sr
                    ph[2 * j] = p1[-1] % two_pi
                    ph[2 * j + 1] = p2[-1] % two_pi
                    out += amp * env * (np.sin(p1) + np.sin(p2))
                t0 += (i1 - i0)
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
                tone = self.amp * np.sin(p1)
                if abs(sph) > 1e-6:
                    # Offset gate on the right channel: the pulses alternate
                    # between the ears and the sound seems to move. Spatial
                    # effect only -- see isochronic() for why nothing focuses
                    # inside the head.
                    f2 = np.mod(frac + sph, 1.0)
                    g2 = np.where(f2 < duty, np.sin(np.pi * f2 / duty) ** 2, 0.0)
                    yield np.stack([tone * gate, tone * g2], axis=1)
                    continue
                sig = tone * gate
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
        # Rotation phase advances with the samples already emitted FOR THIS
        # segment, so the image turns smoothly instead of restarting at each
        # block boundary.
        rr = float(seg.get("rot_rpm", 0.0))
        rd = float(seg.get("rot_depth", 1.0))
        rot_n = 0
        # This segment's own drone and noise, streamed alongside its tones. The
        # drone is tuned to THIS segment's carrier.
        sdr = (self._DroneStream(seg["carrier"], float(seg["drone"]), self.sr)
               if float(seg.get("drone", 0.0)) > 0 else None)
        snz = (self._NoiseStream(float(seg["noise"]),
                                 seg.get("noise_color", "pink"), self.sr)
               if float(seg.get("noise", 0.0)) > 0 else None)
        sdk = float(seg.get("duck", duck_depth))
        sbd = tl if seg.get("level_drone") else 1.0
        sbn = tl if seg.get("level_noise") else 1.0
        path = seg.get("music")
        arr = music_arrays.get(path) if path else None
        if arr is None:
            for tones in base:
                if tl != 1.0:
                    tones = tones * tl
                out = tones
                if sdr is not None:
                    out = out + sdr.read(len(out)) * sbd
                if snz is not None:
                    out = out + snz.read(len(out)) * sbn
                if rr > 0 and rd > 0:
                    out = apply_rotation(out, self.sr, rr, rd,
                                         phase0=2 * np.pi * (rr / 60.0)
                                         * (rot_n / self.sr))
                rot_n += len(out)
                yield out
            return
        mus = _MusicStream(arr, float(seg.get("music_level", 0.25)), self.sr)
        duck = _Ducker(sdk, self.sr) if sdk > 0 else None
        for tones in base:
            if tl != 1.0:
                tones = tones * tl
            if sdr is not None:
                tones = tones + sdr.read(len(tones)) * sbd
            if snz is not None:
                tones = tones + snz.read(len(tones)) * sbn
            m = mus.read(len(tones))
            if duck is not None:
                m = m * duck.gains(tones)
            out = tones + m
            if rr > 0 and rd > 0:
                out = apply_rotation(out, self.sr, rr, rd,
                                     phase0=2 * np.pi * (rr / 60.0)
                                     * (rot_n / self.sr))
            rot_n += len(out)
            yield out

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
                       duck=0.0, progress=None, cancel=None, block=None,
                       rot_rpm=0.0, rot_depth=1.0,
                       level_drone=False, level_noise=False):
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
        # Drone and noise are per segment now; nothing global left but the bed.
        dr = nz = None
        mus = _MusicStream(music, mlev, self.sr) if music is not None else None
        gduck = _Ducker(duck, self.sr) if (duck > 0 and mus is not None) else None
        # Always use the per-segment generator now: it carries drone, noise,
        # ducking and rotation, not just music, so it must run even when no
        # segment has a file attached.
        seg_gen = lambda seg: self._seg_stream_ex(seg, block, music_arrays, duck)
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
                    # Rotation with CONTINUOUS phase: the angle is derived from
                    # the absolute sample position, not from the start of the
                    # block, otherwise the image would jump back at every block
                    # boundary.
                    # Rotation is per segment now, applied inside _seg_stream_ex
                    # where the segment is known; nothing to do on the global
                    # mix here.
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



def apply_rotation(audio, sr, rpm=0.0, depth=1.0, phase0=0.0):
    """Slowly pan the stereo image around the head.

    Unlike the isochronic stereo offset (which shifts the PULSES between the
    ears and only exists in that mode), this is a plain pan and works on any
    material. Constant-power law (cos/sin): the two gains always satisfy
    gl^2 + gr^2 = 1, so the loudness never dips as the image travels -- a
    linear pan would lose 3 dB in the middle and pulse audibly at slow speeds.

    rpm   : turns per minute (0.5-6 is the useful range; 3 = one turn / 20 s)
    depth : 0..1, how far off-centre the image goes (1 = hard left/right)
    Returns audio unchanged when rpm or depth is 0.
    """
    if audio is None or rpm <= 0 or depth <= 0:
        return audio
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = np.stack([a, a], axis=1)
    n = len(a)
    t = np.arange(n, dtype=np.float64) / sr
    p = float(np.clip(depth, 0.0, 1.0)) * np.sin(2 * np.pi * (rpm / 60.0) * t + phase0)
    ang = (p + 1.0) * (np.pi / 4.0)
    gl = np.cos(ang).astype(np.float32)
    gr = np.sin(ang).astype(np.float32)
    out = np.empty_like(a)
    out[:, 0] = a[:, 0] * gl
    out[:, 1] = a[:, 1] * gr
    return out


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


    def edit_bowl(self):
        """Edit the bowl as a table of measured vibration modes.

        A real bowl is not a harmonic series: its modes sit at frequencies that
        are not integer multiples, each with its own decay and its own beat rate
        (the hammered shape is never perfectly circular, so every mode splits
        into a close pair). Measuring one with a phone analyser gives exactly
        this table, so the table is what the tool takes.
        """
        win = tk.Toplevel(self.root)
        win.title("Bowl modes")
        win.transient(self.root)

        tk.Label(win, justify="left", anchor="w", padx=10, pady=8,
                 text="One mode per line:   freq_hz  amp  decay_s  beat_hz\n"
                      "Only the frequency is required -- amp 1.0, a decay scaled\n"
                      "from the fundamental and no beat are assumed.\n"
                      "beat_hz is that mode's own pulsing (0 = steady)."
                 ).grid(row=0, column=0, columnspan=3, sticky="w")

        txt = scrolledtext.ScrolledText(win, width=46, height=12,
                                        font=("Courier", 10))
        txt.grid(row=1, column=0, columnspan=3, padx=10, pady=(0, 6))
        txt.insert("1.0", bowl_modes_to_text(self.bowl_modes) if self.bowl_modes
                   else "# freq_hz  amp  decay_s  beat_hz\n")

        pv = tk.StringVar(value="")
        ttk.Label(win, textvariable=pv, foreground="#555").grid(
            row=2, column=0, columnspan=3, sticky="w", padx=10)

        def _preset(name):
            txt.delete("1.0", "end")
            txt.insert("1.0", bowl_modes_to_text(BOWL_PRESETS[name]))

        pf = ttk.Frame(win)
        pf.grid(row=3, column=0, columnspan=3, sticky="w", padx=10, pady=(0, 6))
        ttk.Label(pf, text="Start from:").pack(side="left")
        for nm in BOWL_PRESETS:
            ttk.Button(pf, text=nm.replace("Generic ", ""), width=8,
                       command=lambda n=nm: _preset(n)).pack(side="left", padx=2)

        def _apply(close=True):
            modes = parse_bowl_modes(txt.get("1.0", "end"))
            if not modes:
                pv.set("No usable line -- the ratio-based bowl will be used.")
                self.bowl_modes = []
            else:
                self.bowl_modes = modes
                pv.set(f"{len(modes)} mode(s): "
                       + ", ".join(f"{f:g}Hz" for f, _, _, _ in modes[:6]))
            self._refresh_bowl_label()
            if close:
                win.destroy()

        bf = ttk.Frame(win)
        bf.grid(row=4, column=0, columnspan=3, pady=(0, 10))
        def _from_wav():
            path = filedialog.askopenfilename(
                title="Recording of a struck bowl",
                filetypes=[("Audio", "*.wav *.flac *.aiff *.aif *.ogg *.mp3"),
                           ("All files", "*.*")])
            if not path:
                return
            pv.set("Analysing\u2026")
            win.update_idletasks()
            try:
                modes = analyse_bowl_wav(path)
            except Exception as e:
                pv.set(f"Could not analyse: {e}")
                return
            if not modes:
                pv.set("No clear mode found. Record a single strike, let it ring, "
                       "and keep the file free of other sounds.")
                return
            txt.delete("1.0", "end")
            txt.insert("1.0", bowl_modes_to_text(modes))
            pv.set(f"{len(modes)} mode(s) measured from "
                   f"{os.path.basename(path)} \u2014 check and adjust by ear.")

        ttk.Button(bf, text="Analyse a WAV\u2026", command=_from_wav).pack(side="left", padx=4)
        ttk.Button(bf, text="Check", command=lambda: _apply(False)).pack(side="left", padx=4)
        ttk.Button(bf, text="Use these modes", command=_apply).pack(side="left", padx=4)
        ttk.Button(bf, text="Clear (use ratios)",
                   command=lambda: (txt.delete("1.0", "end"), _apply())).pack(side="left", padx=4)

    def _refresh_bowl_label(self):
        if self.bowl_modes:
            self.bowl_lbl.config(text=f"{len(self.bowl_modes)} modes "
                                      f"({self.bowl_modes[0][0]:g} Hz\u2026)")
        else:
            self.bowl_lbl.config(text="ratios")

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
        self.bowl_btn = ttk.Button(mf, text="Edit bowl\u2026", width=11,
                                   command=self.edit_bowl)
        self.bowl_btn.grid(row=0, column=4, padx=(10, 2))
        self.bowl_lbl = ttk.Label(mf, text="", foreground="#555")
        self.bowl_lbl.grid(row=0, column=5, padx=(4, 0), sticky="w")
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
        ttk.Button(cf, text="Add all 7", width=9,
                   command=self.add_chakra_sequence).grid(row=0, column=len(CHAKRAS),
                                                          padx=(8, 1))
        r += 1

        ttk.Label(frm, text="Duration (s)").grid(row=r, column=0, sticky="w", **pad)
        self.duration = tk.DoubleVar(value=300.0)
        ttk.Spinbox(frm, from_=1, to=7200, increment=30, width=8,
                    textvariable=self.duration).grid(row=r, column=1, sticky="w", **pad)
        r += 1

        self.duty_lbl = ttk.Label(frm, text="Isochronic duty")
        self.duty_lbl.grid(row=r, column=0, sticky="w", **pad)
        self.duty = tk.DoubleVar(value=0.5)
        # Stereo offset of the isochronic pulses, as a fraction of one beat
        # period. Purely spatial: the pulses alternate between the ears and the
        # sound seems to move. It does NOT focus anything inside the head.
        self.stereo_phase = tk.DoubleVar(value=0.0)
        # Measured-bowl model: a table of independent vibration modes. Empty
        # means the historic ratio-based bowl.
        self.bowl_modes = []
        self.duty_scale = ttk.Scale(frm, from_=0.1, to=0.9, variable=self.duty, length=150)
        self.duty_scale.grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        r += 1

        # Stereo offset of the pulses (isochronic only)
        self.sph_lbl = ttk.Label(frm, text="Stereo offset")
        self.sph_lbl.grid(row=r, column=0, sticky="w", **pad)
        self.sph_scale = ttk.Scale(frm, from_=0.0, to=0.5,
                                   variable=self.stereo_phase, length=150)
        self.sph_scale.grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        self.sph_val = ttk.Label(frm, text="", width=26)
        self.sph_val.grid(row=r, column=3, sticky="w", **pad)

        def _show_sph(*_):
            try:
                v = float(self.stereo_phase.get())
            except Exception:
                return
            deg = v * 360.0
            if v < 0.01:
                self.sph_val.config(text="0 deg (both ears together)")
            else:
                self.sph_val.config(
                    text=f"{deg:.0f} deg -- headphones only"
                         + ("; cancels on speakers" if v > 0.4 else ""))
        self.stereo_phase.trace_add("write", _show_sph)
        _show_sph()
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

        # Rotation: a slow constant-power pan of the whole mix. Global, like
        # drone and noise, so the image keeps turning across segment joins.
        ttk.Label(frm, text="Rotation (turns/min)").grid(row=r, column=0, sticky="w", **pad)
        self.rot_rpm = tk.DoubleVar(value=0.0)
        self.rot_depth = tk.DoubleVar(value=1.0)
        frm_rot = ttk.Frame(frm)
        frm_rot.grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        ttk.Scale(frm_rot, from_=0.0, to=6.0, variable=self.rot_rpm,
                  length=110).pack(side="left")
        ttk.Label(frm_rot, text="depth").pack(side="left", padx=(8, 2))
        ttk.Scale(frm_rot, from_=0.0, to=1.0, variable=self.rot_depth,
                  length=70).pack(side="left")
        self.lbl_rot = ttk.Label(frm, text="", width=30)
        self.lbl_rot.grid(row=r, column=3, sticky="w", **pad)

        def _show_rot(*_):
            try:
                v, d = float(self.rot_rpm.get()), float(self.rot_depth.get())
            except Exception:
                return
            if v < 0.05 or d < 0.02:
                self.lbl_rot.config(text="off (centred)")
            else:
                self.lbl_rot.config(
                    text=f"{v:.1f}/min = 1 turn / {60.0 / v:.0f}s, depth {d:.2f}")
        self.rot_rpm.trace_add("write", _show_rot)
        self.rot_depth.trace_add("write", _show_rot)
        _show_rot()
        r += 1

        # Beat level relative to the music. This is the control that separates a
        # test tone from a listenable track: commercial meditation audio sits the
        # beat 18-25 dB under the music.
        # Range goes to -80 dB. A 16-bit WAV has its noise floor at -96 dBFS, so
        # anything below about -90 no longer exists in the file at all -- -80 is
        # already inaudible while remaining representable. The slider alone
        # cannot resolve that span (the useful -30..0 would sit in the top
        # third), hence the entry box beside it for exact values.
        ttk.Label(frm, text="Beat level (dB)").grid(row=r, column=0, sticky="w", **pad)
        frm_tone = ttk.Frame(frm)
        frm_tone.grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        ttk.Scale(frm_tone, from_=-80.0, to=0.0, variable=self.tone_db,
                  length=150).pack(side="left")
        self.ent_tone_db = ttk.Entry(frm_tone, width=6)
        self.ent_tone_db.pack(side="left", padx=(6, 0))
        self.lbl_tone_db = ttk.Label(frm, text="0 dB", width=30)
        self.lbl_tone_db.grid(row=r, column=3, sticky="w", **pad)

        def _show_tone_db(*_):
            try:
                v = float(self.tone_db.get())
            except Exception:
                return
            hint = ("beat in front" if v > -6 else
                    "blended" if v > -14 else
                    "under the music" if v > -40 else
                    "inaudible in practice" if v > -75 else
                    "effectively silent")
            self.lbl_tone_db.config(text=f"{v:.1f} dB  {hint}")
            if self.ent_tone_db.focus_get() is not self.ent_tone_db:
                self.ent_tone_db.delete(0, "end")
                self.ent_tone_db.insert(0, f"{v:.1f}")

        def _tone_db_typed(_e=None):
            try:
                v = float(self.ent_tone_db.get().strip().replace(",", "."))
            except ValueError:
                _show_tone_db(); return
            self.tone_db.set(max(-80.0, min(0.0, v)))
        self.ent_tone_db.bind("<Return>", _tone_db_typed)
        self.ent_tone_db.bind("<FocusOut>", _tone_db_typed)
        self.tone_db.trace_add("write", _show_tone_db)
        _show_tone_db()
        # Two separate followers: the drone is tuned to the carrier and masks
        # the beat directly, the noise is broadband. Wanting one to follow Beat
        # level and not the other is a real case, so they get a box each.
        self.level_drone = tk.BooleanVar(value=False)
        self.level_noise = tk.BooleanVar(value=False)
        ttk.Label(frm_tone, text="also:").pack(side="left", padx=(8, 2))
        ttk.Checkbutton(frm_tone, text="drone",
                        variable=self.level_drone).pack(side="left")
        ttk.Checkbutton(frm_tone, text="noise",
                        variable=self.level_noise).pack(side="left", padx=(4, 0))
        r += 1

        # Background music (loaded file, mixed under the tones, looped)
        ttk.Button(frm, text="Load music\u2026", command=self.load_music_file).grid(
            row=r, column=0, sticky="w", **pad)
        self.music = None
        self.music_path = None
        # Global bed: its own layer, mixed on top of the assembled session and
        # of whatever music the individual segments carry.
        self.global_music = None
        self.global_music_path = None
        self.global_music_level = 0.25
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
        frm_mus = ttk.Frame(frm)
        frm_mus.grid(row=r, column=1, columnspan=2, sticky="w", **pad)
        ttk.Scale(frm_mus, from_=-80.0, to=0.0, variable=self.music_db,
                  length=150).pack(side="left")
        self.ent_music_db = ttk.Entry(frm_mus, width=6)
        self.ent_music_db.pack(side="left", padx=(6, 0))
        self.lbl_music_db = ttk.Label(frm, text="", width=30)
        self.lbl_music_db.grid(row=r, column=3, sticky="w", **pad)

        def _sync_music_db(*_):
            try:
                db = float(self.music_db.get())
            except Exception:
                return
            self.music_level.set(10.0 ** (db / 20.0))
            try:
                gap = db - float(self.tone_db.get())
            except Exception:
                gap = 0.0
            self.lbl_music_db.config(text=f"{db:.1f} dB   ({gap:+.0f} dB vs beat)")
            if self.ent_music_db.focus_get() is not self.ent_music_db:
                self.ent_music_db.delete(0, "end")
                self.ent_music_db.insert(0, f"{db:.1f}")

        def _music_db_typed(_e=None):
            try:
                v = float(self.ent_music_db.get().strip().replace(",", "."))
            except ValueError:
                _sync_music_db(); return
            self.music_db.set(max(-80.0, min(0.0, v)))
        self.ent_music_db.bind("<Return>", _music_db_typed)
        self.ent_music_db.bind("<FocusOut>", _music_db_typed)
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
        # Single click loads it: a double-click nobody discovers is why the
        # panel seemed to forget a segment's settings.
        self.seg_list.bind("<<ListboxSelect>>", self._load_segment)
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
        ttk.Button(bm, text="\u266a Set GLOBAL music", command=self.set_global_music).grid(
            row=0, column=0, padx=2)
        ttk.Button(bm, text="\u266a Remove global", command=self.clear_global_music).grid(
            row=0, column=1, padx=2)
        self.global_lbl = tk.StringVar(value="(none)")
        ttk.Label(bm, textvariable=self.global_lbl, foreground="#555").grid(
            row=0, column=2, padx=(8, 0), sticky="w")

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

        ttk.Label(frm, text="Set the tone and its music on the left, press Play\n"
                            "to check, then + Add -- the segment stores all of it.\n"
                            "Click a segment to load it back and edit it.\n"
                            "\u266a GLOBAL music plays under the whole session.",
                  foreground="#888", justify="left").grid(
            row=7, column=0, columnspan=4, sticky="w", pady=(6, 0))

    # ---------------- callbacks ----------------
    def _on_mode(self):
        mode = self.mode.get()
        iso, bowl = mode == "isochronic", mode == "bowl"
        self.duty_scale.state(["!disabled"] if iso else ["disabled"])
        self.duty_lbl.configure(foreground="black" if iso else "#aaa")
        # The stereo offset only exists for isochronic pulses: the other modes
        # already define what each channel carries.
        bowl = mode == "bowl"
        if hasattr(self, "bowl_btn"):
            self.bowl_btn.state(["!disabled"] if bowl else ["disabled"])
        self.sph_scale.state(["!disabled"] if iso else ["disabled"])
        for w in (self.sph_lbl, self.sph_val):
            w.configure(foreground="black" if iso else "#aaa")
        self.carrier_lbl.configure(text="Fundamental (Hz)" if bowl else "Carrier (Hz)")
        self.beat_lbl.configure(text="Warble (Hz)" if bowl else "Beat (Hz)")

    def _on_ramp(self):
        self.ramp_spin.configure(state="normal" if self.use_ramp.get() else "disabled")

    def add_chakra_sequence(self):
        """Append one segment per chakra, in order, keeping every other setting.

        A sequence rather than a sweep: a bowl does not change pitch while it
        rings -- its frequency comes from its geometry -- so a glissando would
        sound like a speeded-up tape, not like seven bowls. Each chakra is its
        own struck segment that decays before the next.

        The chakra frequencies are a tuning convention, not a measured
        physiological fact; see "About the evidence".
        """
        try:
            table = TUNINGS[self.tuning.get()]
        except Exception:
            self.status.set("Pick a tuning first.")
            return
        per = simpledialog.askfloat(
            "Chakra sequence",
            f"Duration of EACH of the {len(CHAKRAS)} segments, in seconds.\n\n"
            f"Tuning: {self.tuning.get()}\n"
            f"{', '.join(f'{n} {f:g}Hz' for n, f in zip(CHAKRAS, table))}\n\n"
            f"Every other setting on the left is kept as it is.",
            initialvalue=max(30.0, self._dget(self.duration, 300.0) / len(CHAKRAS)),
            minvalue=1.0, maxvalue=3600.0, parent=self.root)
        if per is None:
            return
        keep_carrier = self._dget(self.carrier, 200.0)
        keep_dur = self._dget(self.duration, 300.0)
        # With measured modes loaded, the carrier no longer sets the pitch --
        # the modes do. Transposing them onto each chakra keeps the bowl's
        # identity (its ratios, decays and beat rates) while moving its
        # fundamental, which is what a set of tuned bowls actually is.
        base_modes = list(self.bowl_modes) if self.bowl_modes else None
        stretch = None
        if base_modes:
            f0 = float(base_modes[0][0])
            ks = [table[i] / f0 for i in range(len(CHAKRAS))]
            stretch = max(max(ks), 1.0 / max(min(ks), 1e-9))
            if stretch > 2.0:
                if not messagebox.askyesno(
                        "Large transposition",
                        f"The measured bowl's fundamental is {f0:g} Hz and this "
                        f"tuning spans {table[0]:g}-{table[-1]:g} Hz, a factor of "
                        f"{stretch:.1f}.\n\n"
                        f"Past about 2x the measured decays and beats stop "
                        f"describing the real bowl. Continue anyway?"):
                    return
        added = 0
        for idx, name in enumerate(CHAKRAS):
            self.carrier.set(table[idx])
            self.duration.set(per)
            if base_modes:
                self.bowl_modes = transpose_bowl_modes(base_modes, table[idx])
            seg = self._current_segment()
            seg["chakra"] = name
            self.segments.append(seg)
            self.seg_list.insert("end", self._seg_label(seg))
            added += 1
        # Put the panel back where the user left it.
        self.carrier.set(keep_carrier)
        self.duration.set(keep_dur)
        if base_modes:
            self.bowl_modes = base_modes
            self._refresh_bowl_label()
        self._update_total()
        self.status.set(f"{added} chakra segments added "
                        f"({per:g}s each, {added * per / 60:.1f} min total).")

    def _set_chakra(self, idx):
        self.carrier.set(TUNINGS[self.tuning.get()][idx])

    # ---------------- session ----------------
    def _dget(self, var, default=0.0, name=None):
        """Read a numeric Tk variable, tolerating an empty field.

        A DoubleVar bound to an Entry raises TclError as soon as the user clears
        the box -- and every button that reads it dies with it, which is what
        made + Add throw instead of adding. Fall back to a sane value, put it
        back in the field so the user sees what was used, and say so.
        """
        try:
            return float(var.get())
        except Exception:
            try:
                var.set(default)
            except Exception:
                pass
            if name:
                self.status.set(f"{name} was empty -- using {default:g}.")
            return float(default)

    def _current_segment(self):
        # Beat level is captured WITH the segment, like carrier and beat: a
        # session that alternates a quiet bed and a foreground cue needs its own
        # balance in each part.
        seg = {"mode": self.mode.get(),
               "carrier": self._dget(self.carrier, 200.0, "Carrier"),
               "beat": self._beat_value(),
               "duration": self._dget(self.duration, 300.0, "Duration"),
               "tone_level": 10.0 ** (self._dget(self.tone_db, 0.0) / 20.0)}
        if self.mode.get() == "isochronic":
            seg["duty"] = self._dget(self.duty, 0.5)
            if abs(self._dget(self.stereo_phase, 0.0)) > 1e-6:
                seg["stereo_phase"] = self._dget(self.stereo_phase, 0.0)
        # The music loaded on the left goes in too. Play previews everything on
        # the left panel, so Add must store everything on the left panel --
        # music was the one exception, which made "tune, listen, add" impossible
        # to finish in one pass.
        if self._dget(self.rot_rpm, 0.0) > 0.05 and self._dget(self.rot_depth, 1.0) > 0.02:
            seg["rot_rpm"] = self._dget(self.rot_rpm, 0.0)
            seg["rot_depth"] = self._dget(self.rot_depth, 1.0)
        # Everything else the left panel offers. Stored only when non-default,
        # so a plain segment stays a short readable line.
        if self.mode.get() == "bowl" and self.bowl_modes:
            seg["bowl_modes"] = [tuple(mm) for mm in self.bowl_modes]
        if self.level_drone.get():
            seg["level_drone"] = True
        if self.level_noise.get():
            seg["level_noise"] = True
        if self._dget(self.drone_level, 0.0) > 0.001:
            seg["drone"] = self._dget(self.drone_level, 0.0)
        if self._dget(self.noise, 0.0) > 0.001:
            seg["noise"] = self._dget(self.noise, 0.0)
            seg["noise_color"] = self.noise_color.get()
        if self._dget(self.duck, 0.0) > 0.001:
            seg["duck"] = self._dget(self.duck, 0.0)
        # Tuning does not change the render (it only feeds the chakra buttons),
        # but storing it means clicking a segment restores the panel exactly as
        # it was left.
        seg["tuning"] = self.tuning.get()
        if getattr(self, "music_path", None):
            seg["music"] = self.music_path
            seg["music_level"] = self._dget(self.music_level, 0.25)
        return seg

    def _seg_label(self, seg):
        b = seg["beat"]
        bs = f"{b[0]:.1f}->{b[1]:.1f}" if isinstance(b, tuple) else f"{b:.1f}"
        # Both levels are shown in dB: what matters in a mix is the gap between
        # them, and a gap is only readable on one scale.
        tl = float(seg.get("tone_level", 1.0))
        beat_db = 20.0 * math.log10(max(tl, 1e-6))
        lvl = f" | beat {beat_db:+.0f}dB" if abs(beat_db) > 0.5 else ""
        chk = f" | {seg['chakra']}" if seg.get("chakra") else ""
        rr = float(seg.get("rot_rpm", 0.0))
        rot = f" | rot {rr:.1f}/min" if rr > 0.05 else ""
        # Compact flags for the rest: a segment that uses none of them stays a
        # short line, one that uses them says so.
        extra = ""
        if float(seg.get("drone", 0)) > 0.001:
            extra += f" | drone {float(seg['drone']):.2f}"
        if float(seg.get("noise", 0)) > 0.001:
            extra += f" | {seg.get('noise_color', 'pink')} {float(seg['noise']):.2f}"
        if float(seg.get("duck", 0)) > 0.001:
            extra += f" | duck {float(seg['duck']):.2f}"
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
                f"{bs}Hz | {seg['duration']:.0f}s{chk}{lvl}{rot}{extra}{mus}")

    def set_global_music(self):
        """The bed under the WHOLE session, on top of any per-segment music.
        Asked for once, at the end, which is when a session is finished."""
        path = filedialog.askopenfilename(
            title="Global music for the whole session",
            filetypes=[("Audio", "*.wav *.mp3 *.flac *.ogg *.aiff *.aif *.m4a "
                                 "*.aac *.opus *.wma"), ("All files", "*.*")])
        if not path:
            return
        try:
            arr = load_music(path, self.SR)
        except Exception as e:
            messagebox.showerror("Cannot load music", str(e))
            return
        db = simpledialog.askfloat(
            "Global music level",
            f"Level in dB for {os.path.basename(path)}\n\n"
            f"Plays under every segment, on top of any music\n"
            f"the segments already carry. Around -6 dB sits it\n"
            f"behind the tones.",
            initialvalue=-6.0, minvalue=-80.0, maxvalue=0.0, parent=self.root)
        if db is None:
            return
        self.global_music = arr
        self.global_music_path = path
        self.global_music_level = 10.0 ** (db / 20.0)
        self.global_lbl.set(f"{os.path.basename(path)}  {db:+.0f} dB")
        self.status.set(f"Global music: {os.path.basename(path)} at {db:+.0f} dB.")

    def clear_global_music(self):
        self.global_music = None
        self.global_music_path = None
        self.global_music_level = 0.25
        self.global_lbl.set("(none)")
        self.status.set("Global music removed.")

    # NOTE: no longer wired to a button. Music now travels with the segment,
    # captured by + Add from the left panel; these two remain in case a
    # "change the music of an existing segment without reloading it" button is
    # wanted later.
    def set_segment_music(self):
        sel = self.seg_list.curselection()
        if not sel:
            # This is the step everyone misses: the button acts on a SELECTED
            # segment, and nothing happens if none is highlighted.
            messagebox.showinfo(
                "Select a segment first",
                "Click a line in the session list (it turns blue), then press "
                "this button again.\n\n"
                "This attaches music to that ONE segment.\n"
                "For music across the whole session, use 'Global music...' on "
                "the left instead.")
            self.status.set("Select a segment in the list first.")
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
        cur_db = 20.0 * math.log10(max(float(seg.get(
            "music_level", self._dget(self.music_level, 0.25))), 1e-6))
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
        had = seg.get("music")
        if had:
            # The music now lives in the segment; clearing the panel makes that
            # visible and stops the next segment inheriting it silently.
            self._clear_music(announce=False)
            self.status.set(f"Segment {len(self.segments)} added with "
                            f"{os.path.basename(had)} — panel cleared for the next.")
        else:
            self.status.set(f"Segment {len(self.segments)} added (no music).")

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
            self.stereo_phase.set(seg.get("stereo_phase", 0.0))
        # Levels and music come back too: showing half a segment and letting
        # Update rewrite it from the visible half is how settings got lost.
        self.tone_db.set(20.0 * math.log10(max(float(seg.get("tone_level", 1.0)), 1e-6)))
        self.rot_rpm.set(float(seg.get("rot_rpm", 0.0)))
        self.rot_depth.set(float(seg.get("rot_depth", 1.0)))
        self.bowl_modes = [tuple(mm) for mm in seg.get("bowl_modes", [])]
        self._refresh_bowl_label()
        self.level_drone.set(bool(seg.get("level_drone", False)))
        self.level_noise.set(bool(seg.get("level_noise", False)))
        self.drone_level.set(float(seg.get("drone", 0.0)))
        self.noise.set(float(seg.get("noise", 0.0)))
        self.noise_color.set(seg.get("noise_color", "pink"))
        self.duck.set(float(seg.get("duck", 0.0)))
        if seg.get("tuning") in TUNINGS:
            self.tuning.set(seg["tuning"])
        if seg.get("music"):
            self._set_music_from_path(seg["music"], announce=False)
            self.music_db.set(20.0 * math.log10(
                max(float(seg.get("music_level", 0.25)), 1e-6)))
        else:
            self._clear_music(announce=False)
        self._on_mode()
        self._on_ramp()
        self.status.set(f"Segment {sel[0] + 1} loaded — edit then Update, "
                        f"or - Del to remove it.")

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
        opts = dict(noise=self._dget(self.noise, 0.0),
                    noise_color=self.noise_color.get(),
                    drone=self._dget(self.drone_level, 0.0),
                    fade=3.0 if len(segments) > 1 else 2.0, xfade=2.0,
                    # The session bed is the GLOBAL music, not the left panel:
                    # the left panel now belongs to whichever segment is being
                    # edited, and each segment carries its own music inside it.
                    music=self.global_music,
                    music_level=self.global_music_level,
                    duck=self._dget(self.duck, 0.0),
                    rot_rpm=self._dget(self.rot_rpm, 0.0),
                    rot_depth=self._dget(self.rot_depth, 1.0),
                    level_drone=self.level_drone.get(),
                    level_noise=self.level_noise.get())
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
        opts = dict(noise=self._dget(self.noise, 0.0),
                    noise_color=self.noise_color.get(),
                    drone=self._dget(self.drone_level, 0.0), fade=3.0, xfade=2.0,
                    # The session bed is the GLOBAL music, not the left panel:
                    # the left panel now belongs to whichever segment is being
                    # edited, and each segment carries its own music inside it.
                    music=self.global_music,
                    music_level=self.global_music_level,
                    duck=self._dget(self.duck, 0.0),
                    rot_rpm=self._dget(self.rot_rpm, 0.0),
                    rot_depth=self._dget(self.rot_depth, 1.0),
                    level_drone=self.level_drone.get(),
                    level_noise=self.level_noise.get())
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
            return (self._dget(self.beat, 6.0, "Beat"),
                    self._dget(self.beat_end, 10.0, "Ramp to"))
        return self._dget(self.beat, 6.0, "Beat")

    def _render(self, duration):
        # Same guards as _current_segment: Play must not die because a field is
        # momentarily empty while the user is typing in it.
        return self.tb.render(
            self.mode.get(), self._dget(self.carrier, 200.0, "Carrier"),
            self._beat_value(), duration,
            duty=self._dget(self.duty, 0.5), noise=self._dget(self.noise, 0.0),
            noise_color=self.noise_color.get(),
            drone=self._dget(self.drone_level, 0.0), fade=2.0,
            music=self.music, music_level=self._dget(self.music_level, 0.25),
            duck=self._dget(self.duck, 0.0),
            tone_level=10.0 ** (self._dget(self.tone_db, 0.0) / 20.0),
            stereo_phase=self._dget(self.stereo_phase, 0.0),
            rot_rpm=self._dget(self.rot_rpm, 0.0),
            rot_depth=self._dget(self.rot_depth, 1.0),
            level_drone=self.level_drone.get(),
            level_noise=self.level_noise.get(),
            bowl_modes=self.bowl_modes)

    def _set_music_from_path(self, path, announce=True):
        """Load a file into the left panel. Shared by the dialog and by segment
        recall, so selecting a segment restores its music without asking."""
        try:
            self.music = load_music(path, self.SR)
        except Exception as e:
            if announce:
                messagebox.showerror("Cannot load music", str(e))
            else:
                self.status.set(f"Music file missing: {os.path.basename(path)}")
            return False
        self.music_path = path
        self.music_name.set(os.path.basename(path))
        if announce:
            self.status.set(f"Music loaded: {os.path.basename(path)} "
                            f"({len(self.music) / self.SR:.0f}s). Set its level, "
                            f"Play to check, then + Add.")
        return True

    def _clear_music(self, announce=True):
        self.music = None
        self.music_path = None
        self.music_name.set("(none)")
        if announce:
            self.status.set("Music removed.")

    def load_music_file(self):
        path = filedialog.askopenfilename(filetypes=[
            ("Audio", "*.wav *.mp3 *.flac *.ogg *.aiff *.aif *.m4a *.aac *.opus *.wma"),
            ("All files", "*.*")])
        if path:
            self._set_music_from_path(path)

    def clear_music(self):
        self._clear_music()

    def play(self):
        if not self._need_audio():
            return
        try:
            _d = self._dget(self.duration, 300.0, "Duration")
            dur = min(_d, 12.0) if self.loop.get() else _d
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
            self.status.set(f"Playing… {self.mode.get()} | "
                            f"{self._dget(self.carrier, 200.0):.0f} Hz")
        except Exception as e:
            self.status.set(f"Playback error: {e}")

    def stop(self):
        if self.audio_ok:
            pygame.mixer.stop()
        self.status.set("Stopped.")

    def save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".wav", filetypes=self.AUDIO_TYPES,
            initialfile=f"{self.mode.get()}_"
                        f"{int(self._dget(self.carrier, 200.0))}Hz.wav")
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

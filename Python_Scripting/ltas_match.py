#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ltas_match.py — Long-Term Average Spectrum matching by least squares.

Fits the generator's 3-band peaking EQ (FFmpeg `equalizer`, RBJ biquad) plus a
broadband volume offset so that the clone's LTAS matches the reference's LTAS in
a least-squares sense over the speech band.

The peaking responses modelled here are the EXACT responses FFmpeg applies:
  low  : f0=120  Hz, BW=160  Hz  (Q = f0/BW = 0.75)
  mid  : f0=1500 Hz, BW=2000 Hz  (Q = 0.75)
  high : f0=5000 Hz, BW=3000 Hz  (Q = 1.667)
hp/lp are pydub's first-order RC recursions, set from the LTAS at the spectral
extremes, not by the peaking LS. At the 24 kHz of an XTTS render the low-pass is
a gentle treble tilt, not a cut above lp: lp 9000 takes 2.1 dB at 4 kHz and
4.5 dB at 8 kHz (measured through pydub). rc_lowpass_db / rc_highpass_db model
them exactly, so the EQ can be fitted on the curve a filter change will give.

Public API:
  compute_ltas(y, sr)                 -> (f_grid, ltas_db)
  fit_eq_ls(f_grid, ref_db, clone_db) -> dict(vol, eq_low, eq_mid, eq_high, hp, lp,
                                              residual_before, residual_after)
"""

import numpy as np
from scipy.signal import welch
from scipy.optimize import least_squares
from audio_checks import mono_audio

# Generator EQ band definitions — must mirror apply_eq() in the generator.
# eq_low sits at 120 Hz, not 200: a bell centred at 200 Hz delivers only ~19%
# of its gain at 80 Hz, so on a deep voice it could not reach the fundamental
# at all — correcting a 3 dB low-end deficit would have needed +16 dB. At
# 120 Hz it delivers 4.3 dB there. FFmpeg's `bass` shelf would reach lower
# still, but its width_type=q parameterisation does not match either standard
# RBJ formula (0.55-0.80 dB model error measured), and an unmodelled filter
# breaks the least-squares fit. This bell is exact to 0.006 dB.
EQ_BANDS = [
    ("eq_low",  120.0,  160.0),   # f0, BW(Hz)
    ("eq_mid",  1500.0, 2000.0),
    ("eq_high", 5000.0, 3000.0),
]

FIT_FMIN = 80.0      # speech band for the fit
FIT_FMAX = 8000.0
N_GRID   = 256       # log-spaced points

# Spectra are compared as mean power over bands max(250 Hz, 1/3 octave) wide,
# not point by point. Below ~1.1 kHz a voice's long-term spectrum is a comb:
# peaks at the harmonics of F0, hollows between them, almost nothing below F0,
# and where the teeth fall depends on the melody of the sentence, which a clone
# never repeats. Compared point by point, the teeth made most of the error
# (78 % of it from 180 Hz to 1 kHz on a synthetic 204 Hz voice), no EQ bell
# could reduce it, and the
# least squares in dB, pulled by the hollows, boosted eq_low on a clone that
# already had too much fundamental. 250 Hz spans one harmonic spacing up to
# that F0; above 1.1 kHz, where the harmonics merge, a third of an octave.
SMOOTH_FLOOR_HZ = 250.0
SMOOTH_OCTAVES  = 1.0 / 3.0
SMOOTH_LOWEST_HZ = 20.0  # bands never reach below it (DC, rumble)


# ── Exact filter magnitude responses (dB) ─────────────────────────────────────

def rbj_peaking_db(freqs, fs, f0, bw_hz, gain_db):
    """Exact magnitude response (dB) of FFmpeg `equalizer` (RBJ peaking biquad).

    width_type='h' in FFmpeg means bandwidth in Hz, giving Q = f0 / bw_hz.
    """
    gain_db = np.asarray(gain_db, dtype=float)
    if np.all(gain_db == 0.0):
        return np.zeros_like(freqs)
    A   = 10.0 ** (gain_db / 40.0)
    w0  = 2.0 * np.pi * f0 / fs
    Q   = f0 / bw_hz
    alpha = np.sin(w0) / (2.0 * Q)
    cosw0 = np.cos(w0)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cosw0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cosw0
    a2 = 1.0 - alpha / A

    w = 2.0 * np.pi * freqs / fs
    z1 = np.exp(-1j * w)
    z2 = np.exp(-2j * w)
    H = (b0 + b1 * z1 + b2 * z2) / (a0 + a1 * z1 + a2 * z2)
    return 20.0 * np.log10(np.abs(H) + 1e-12)


def rc_lowpass_db(freqs, fc, fs):
    """pydub's low_pass_filter, which the generator applies for lp (dB, 0 = off).

    A first-order RC recursion, y[n] = y[n-1] + a (x[n] - y[n-1]) with
    a = 1 / (1 + fs / (2 pi fc)). The analogue one-pole formula this module
    had (and nothing used) put lp 9000 at -1.6 dB at 6 kHz on a 24 kHz
    render, where pydub takes -3.4 dB, and at -2.5 dB at 8 kHz, where it
    takes -4.5 dB.
    """
    freqs = np.asarray(freqs, dtype=float)
    if not fc or fc <= 0:
        return np.zeros_like(freqs)
    a = 1.0 / (1.0 + fs / (2.0 * np.pi * fc))
    w = 2.0 * np.pi * freqs / fs
    mag2 = a * a / (1.0 - 2.0 * (1.0 - a) * np.cos(w) + (1.0 - a) ** 2)
    return 10.0 * np.log10(mag2 + 1e-24)


def rc_highpass_db(freqs, fc, fs):
    """pydub's high_pass_filter, which the generator applies for hp (dB, 0 = off).

    y[n] = a (y[n-1] + x[n] - x[n-1]) with a = 1 / (1 + 2 pi fc / fs).
    """
    freqs = np.asarray(freqs, dtype=float)
    if not fc or fc <= 0:
        return np.zeros_like(freqs)
    a = 1.0 / (1.0 + 2.0 * np.pi * fc / fs)
    w = 2.0 * np.pi * freqs / fs
    mag2 = a * a * (2.0 - 2.0 * np.cos(w)) / (1.0 - 2.0 * a * np.cos(w) + a * a)
    return 10.0 * np.log10(mag2 + 1e-24)


def filter_change_db(freqs, fs, old, new):
    """What changing hp/lp from `old` to `new` (dicts) does to a spectrum, dB."""
    return (rc_highpass_db(freqs, float(new.get('hp', 0) or 0), fs)
            - rc_highpass_db(freqs, float(old.get('hp', 0) or 0), fs)
            + rc_lowpass_db(freqs, float(new.get('lp', 0) or 0), fs)
            - rc_lowpass_db(freqs, float(old.get('lp', 0) or 0), fs))


def eq_peaking_db(freqs, fs, gains):
    """Total dB of the 3 peaking bands for gains = [g_low, g_mid, g_high]."""
    total = np.zeros_like(freqs)
    for (name, f0, bw), g in zip(EQ_BANDS, gains):
        total = total + rbj_peaking_db(freqs, fs, f0, bw, g)
    return total


# ── LTAS ──────────────────────────────────────────────────────────────────────

def ltas_psd(y, sr):
    """Welch power spectral density of the whole signal: (freqs, psd)."""
    y = mono_audio(y, sr)
    nper = min(4096, len(y))
    if nper < 256:
        nper = len(y)
    freqs, psd = welch(y, sr, nperseg=nper, noverlap=nper // 2)
    return freqs, np.maximum(psd, 1e-20)


def smoothing_widths(f_grid, floor_hz=SMOOTH_FLOOR_HZ):
    """Width (Hz) of the band read at each grid frequency."""
    f_grid = np.asarray(f_grid, float)
    third = f_grid * (2.0 ** (SMOOTH_OCTAVES / 2) - 2.0 ** (-SMOOTH_OCTAVES / 2))
    return np.maximum(floor_hz, third)


def grid_db(freqs, psd, f_grid, floor_hz=SMOOTH_FLOOR_HZ):
    """dB at each grid frequency: mean power over the band smoothing_widths
    gives (not normalised). floor_hz=0 reads the PSD point by point, as
    before 3.7.2."""
    freqs, psd, f_grid = (np.asarray(x, float) for x in (freqs, psd, f_grid))
    if not floor_hz:
        return np.interp(f_grid, freqs, 10.0 * np.log10(psd))
    df = freqs[1] - freqs[0]
    edges = np.concatenate([freqs - df / 2, [freqs[-1] + df / 2]])
    cum = np.concatenate([[0.0], np.cumsum(psd) * df])
    half = smoothing_widths(f_grid, floor_hz) / 2
    lo = np.maximum(f_grid - half, max(SMOOTH_LOWEST_HZ, edges[0]))
    hi = np.minimum(f_grid + half, edges[-1])
    power = (np.interp(hi, edges, cum) - np.interp(lo, edges, cum)) / np.maximum(hi - lo, 1e-9)
    return 10.0 * np.log10(np.maximum(power, 1e-20))


def compute_ltas(y, sr, fmin=FIT_FMIN, fmax=FIT_FMAX, n_grid=N_GRID,
                 floor_hz=SMOOTH_FLOOR_HZ):
    """Long-term average spectrum on a log-frequency grid, in dB.

    Returns (f_grid, ltas_db). The dB curve is *relative* (peak-normalised),
    so only its shape matters — absolute level is handled separately by `vol`.
    Each point is the mean power of a band around it (grid_db), unless
    floor_hz=0.
    """
    y = mono_audio(y, sr)
    if not (np.isfinite(fmin) and np.isfinite(fmax) and 0 < fmin < min(fmax, sr / 2 - 1)):
        raise ValueError("Invalid LTAS frequency interval for this sample rate")
    if not isinstance(n_grid, (int, np.integer)) or n_grid < 4:
        raise ValueError("LTAS requires at least four grid points")
    freqs, psd = ltas_psd(y, sr)
    f_grid = np.logspace(np.log10(fmin), np.log10(min(fmax, sr / 2 - 1)), n_grid)
    db = grid_db(freqs, psd, f_grid, floor_hz)
    return f_grid, db - db.max()               # peak-normalise (shape only)


# ── Roll-off based hp / lp ────────────────────────────────────────────────────

def derive_hp_lp(f_grid, ref_db, clone_db, cur_hp=0, cur_lp=0):
    """Set hp/lp from excess energy at the spectral extremes of the clone.

    The high-pass is a rumble guard, not a tone control — the EQ fits the
    tone. So hp is never allowed below 40 Hz: lowering it to compensate a
    low-end deficit is the EQ's job, and an unbounded loop walked it down 30 Hz
    per pass (95 -> 65 -> 35 -> 5 Hz = no filter at all, DC and rumble through).

    The low-pass is no guard at all on a 24 kHz render: pydub's first-order
    filter tilts the whole upper band (rc_lowpass_db), and even lp 12000 takes
    3.5 dB at 8 kHz. When the clone LACKS highs it is switched off. It was
    opened by 1000 Hz, and lp 10000 still takes 1.8 dB at 4 kHz and 4.1 dB at
    8 kHz: eq_high then spent its whole bound giving that back. A clone with
    excess highs still gets a lower one, the treble cut it then needs.
    """
    HP_MIN, HP_MAX = 40.0, 150.0
    hp, lp = cur_hp, cur_lp
    lo = f_grid < 120
    if lo.any():
        excess_low = float(np.mean(clone_db[lo] - ref_db[lo]))
        if excess_low > 3:
            hp = float(np.clip((cur_hp or HP_MIN) + 30, HP_MIN, HP_MAX))
        elif excess_low < -3 and cur_hp > HP_MIN:
            hp = float(np.clip(cur_hp - 30, HP_MIN, HP_MAX))
    hi = f_grid >= 6000  # measured upper band, inside the default 80–8000 Hz grid
    if hi.any():
        excess_high = float(np.mean(clone_db[hi] - ref_db[hi]))
        if excess_high > 3:
            lp = float(np.clip((cur_lp or 11000) - 1000, 6000, 12000))
        elif excess_high < -3 and cur_lp > 0:
            lp = 0.0
    return hp, lp


# ── Least-squares EQ fit ──────────────────────────────────────────────────────

def level_match_db(ref_y, clone_y, cur_vol=0, lo=-18, hi=18):
    """Volume correction (dB) from the RMS gap — the proper level match."""
    ref_y = mono_audio(ref_y)
    clone_y = mono_audio(clone_y)
    r = 20.0 * np.log10(np.sqrt(np.mean(ref_y ** 2)) + 1e-12)
    clone_y = mono_audio(clone_y)
    c = 20.0 * np.log10(np.sqrt(np.mean(clone_y ** 2)) + 1e-12)
    return int(np.clip(round(cur_vol + (r - c)), lo, hi))


def spectrum_weights(f_grid, ref_db, weighting='a'):
    if weighting not in ('a', 'voice', 'blend'):
        raise ValueError('weighting must be a, voice or blend')
    f_grid, ref_db = np.asarray(f_grid, float), np.asarray(ref_db, float)
    f2 = np.maximum(f_grid, 1.0) ** 2
    ra = (12194.0**2 * f2**2) / ((f2 + 20.6**2)
         * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2))
    a_db = 20.0 * np.log10(np.maximum(ra, 1e-12)) + 2.0
    w_a = np.maximum(10.0 ** (a_db / 20.0), 0.25)
    if weighting in ('voice', 'blend'):
        # Energy share of the reference, floored so quiet bands are not ignored
        # entirely (an EQ band with zero weight is unconstrained and can drift).
        e = 10.0 ** (np.asarray(ref_db, float) / 20.0)
        w_v = np.maximum(e / (e.max() + 1e-12), 0.10)
        w = np.sqrt(w_a * w_v) if weighting == 'blend' else w_v
    else:
        w = w_a
    w = w / w.mean()
    return w


def spectral_shape_error(f_grid, ref_db, clone_db, weighting='a'):
    """Weighted shape error measured on real audio, independent of level."""
    ref_db, clone_db = np.asarray(ref_db,float), np.asarray(clone_db,float)
    if ref_db.shape != clone_db.shape or not np.isfinite(ref_db-clone_db).all():
        raise ValueError('Incompatible or non-finite spectra')
    weights = spectrum_weights(f_grid, ref_db, weighting)
    delta = clone_db-ref_db
    delta -= np.average(delta,weights=weights)
    return float(np.sqrt(np.average(delta**2,weights=weights)))


def fit_eq_ls(f_grid, ref_db, clone_db, fs=24000, weighting='a',
              g_bounds=(-6.0, 6.0), cur_hp=0, cur_lp=0):
    """Fit [g_low, g_mid, g_high] minimising ‖(clone_shape + EQ) − ref_shape‖²
    over the speech band, using the EXACT generator EQ responses.

    A free broadband offset `c` absorbs any residual level mismatch from the
    peak-normalisation of the LTAS; it is a nuisance and discarded (level is
    handled by `level_match_db` from the raw RMS). Returns the rounded EQ
    gains, derived hp/lp, and the weighted RMS residual (dB) before/after.
    """
    f_grid, ref_db, clone_db = [np.asarray(x, float) for x in (f_grid, ref_db, clone_db)]
    if (f_grid.ndim != 1 or len(f_grid) < 4 or ref_db.shape != f_grid.shape
            or clone_db.shape != f_grid.shape or not all(np.isfinite(x).all() for x in (f_grid, ref_db, clone_db))
            or np.any(np.diff(f_grid) <= 0) or f_grid[0] <= 0 or f_grid[-1] >= fs / 2
            or not np.isfinite(fs) or fs <= 2 * max(b[1] for b in EQ_BANDS)):
        raise ValueError("Invalid spectrum or sample rate for EQ fitting")
    if weighting not in ('a', 'voice', 'blend'):
        raise ValueError("weighting must be a, voice or blend")
    if not np.isfinite(g_bounds).all() or g_bounds[0] >= g_bounds[1]:
        raise ValueError("Invalid EQ gain bounds")
    D = ref_db - clone_db                       # dB the clone must gain at each f
    # Weighting decides WHERE an error counts, and the right answer depends on
    # the goal:
    #   'a'     IEC A-weighting — equal-loudness. Correct for balance, but it
    #           discounts 80 Hz by ~22 dB, i.e. the fundamental of a bass voice
    #           weighs 4.6x less than the mid-range.
    #   'voice' the reference's OWN spectrum — errors count where this speaker
    #           actually has energy. Better when the goal is to resemble a
    #           particular person rather than to sound evenly loud.
    #   'blend' geometric mean of both: keeps the ear's emphasis without
    #           erasing the speaker's low end.
    w = spectrum_weights(f_grid, ref_db, weighting)
    sw = np.sqrt(w)

    def residual(p):
        c, gl, gm, gh = p
        model = eq_peaking_db(f_grid, fs, [gl, gm, gh]) + c
        return sw * (model - D)

    lo = [-24.0, g_bounds[0], g_bounds[0], g_bounds[0]]
    hi = [+24.0, g_bounds[1], g_bounds[1], g_bounds[1]]
    x0 = np.clip([float(np.average(D, weights=w)), 0., 0., 0.], lo, hi)

    sol = least_squares(residual, x0, bounds=(lo, hi), method='trf')
    _c, gl, gm, gh = sol.x

    # Bell filters do not sum to a constant gain: never recenter their gains.
    # Report the response of the rounded values actually exported to the renderer.
    gl, gm, gh = [float(np.clip(round(float(g), 1), *g_bounds)) for g in (gl, gm, gh)]
    remaining = eq_peaking_db(f_grid, fs, [gl, gm, gh]) - D
    remaining -= np.average(remaining, weights=w)
    res_before = float(np.sqrt(np.average((D - np.average(D, weights=w)) ** 2, weights=w)))
    res_after = float(np.sqrt(np.average(remaining ** 2, weights=w)))

    hp, lp = derive_hp_lp(f_grid, ref_db, clone_db, cur_hp, cur_lp)

    return dict(
        eq_low=round(float(gl), 1),
        eq_mid=round(float(gm), 1),
        eq_high=round(float(gh), 1),
        hp=hp, lp=lp,
        residual_before=round(res_before, 2),
        residual_after=round(res_after, 2),
    )


def fit_eq_psd(f_grid, ref_db, freqs, psd, fs, weighting='a', limit=4.0,
               start=(0.0, 0.0, 0.0), extra_db=None, floor_hz=SMOOTH_FLOOR_HZ):
    """EQ gains (absolute, within +/-limit) that bring a render to ref_db.

    (freqs, psd) is the render's PSD, taken with the gains `start`; extra_db,
    per PSD bin, is a filter change applied on top. The bells and the filters
    are linear, so their effect is exact bin by bin: the model takes `start`
    out, puts the candidate gains in, and smooths the result onto the grid
    like the measurement (grid_db). Fitting the cumulative gains under their
    own bounds keeps a band that sits at its bound there, instead of fitting
    a step it cannot take and clipping it afterwards, which left the other
    bands fitted for a move that never happened.

    Returns ({eq_low, eq_mid, eq_high}, the curve those rounded gains give).
    """
    f_grid, ref_db, freqs, psd = (np.asarray(x, float) for x in (f_grid, ref_db, freqs, psd))
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("Invalid EQ gain bound")
    inside = freqs < fs / 2           # the chain runs at fs: nothing above its Nyquist

    def bells(gains):
        out = np.zeros_like(freqs)
        out[inside] = eq_peaking_db(freqs[inside], fs, gains)
        return out
    start = np.clip(np.asarray(start, float), -limit, limit)
    shift = np.zeros_like(freqs) if extra_db is None else np.asarray(extra_db, float)
    base = psd * 10.0 ** ((shift - bells(start)) / 10.0)
    w = spectrum_weights(f_grid, ref_db, weighting)
    sw = np.sqrt(w)

    def model(gains):
        return grid_db(freqs, base * 10.0 ** (bells(gains) / 10.0), f_grid, floor_hz)

    def residual(p):
        return sw * (model(p[1:]) + p[0] - ref_db)
    c0 = float(np.average(ref_db - model(start), weights=w))
    sol = least_squares(residual, np.concatenate([[c0], start]), method='trf',
                        bounds=([-300.0] + [-limit] * 3, [300.0] + [limit] * 3))
    gains = [float(np.clip(round(float(g), 1), -limit, limit)) for g in sol.x[1:]]
    return dict(zip(('eq_low', 'eq_mid', 'eq_high'), gains)), model(gains)


__all__ = ['compute_ltas', 'fit_eq_ls', 'fit_eq_psd', 'level_match_db', 'eq_peaking_db',
           'rbj_peaking_db', 'ltas_psd', 'grid_db']


# ── Dynamics / sibilance measurements (closed-loop comp & de-ess fitting) ─────

def sibilance_db(y, sr):
    """Sibilance ratio in dB: energy(5-8 kHz) vs energy(300-3 kHz).
    Comparing this between reference and clone gives a MEASURED de-esser
    setting instead of an by-ear guess."""
    y = mono_audio(y, sr)
    n = min(len(y), int(sr * 60))
    Y = np.abs(np.fft.rfft(y[:n] * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(Y[m].mean()) if m.any() else 1e-12
    return 10.0 * np.log10(band(5000, 8000) / (band(300, 3000) + 1e-12) + 1e-12)


def crest_db(y):
    """Crest factor in dB (peak vs RMS) — a proxy for dynamic range.
    Clone crest >> reference crest means the clone is more 'peaky' and a
    matching compression amount can be derived by measurement."""
    y = mono_audio(y)
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    peak = float(np.max(np.abs(y)) + 1e-12)
    return 20.0 * np.log10(peak / rms)


def level_to_target_db(clone_y, target_dbfs, cur_vol=0, lo=-18, hi=18):
    """Volume correction (dB) to bring the clone to an absolute RMS target,
    instead of matching a (possibly very quiet) reference."""
    clone_y = mono_audio(clone_y)
    c = 20.0 * np.log10(np.sqrt(np.mean(clone_y ** 2)) + 1e-12)
    return int(np.clip(round(cur_vol + (target_dbfs - c)), lo, hi))

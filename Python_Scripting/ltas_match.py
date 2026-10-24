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
hp/lp are pydub one-pole (-6 dB/oct) filters, derived from the LTAS roll-off,
not folded into the peaking LS (different parameter nature).

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


def onepole_hp_db(freqs, fc):
    """One-pole high-pass magnitude (dB). 0 = bypass."""
    if fc <= 0:
        return np.zeros_like(freqs)
    r = fc / np.maximum(freqs, 1e-6)
    return 20.0 * np.log10(1.0 / np.sqrt(1.0 + r * r) + 1e-12)


def onepole_lp_db(freqs, fc):
    """One-pole low-pass magnitude (dB). 0 = bypass."""
    if fc <= 0:
        return np.zeros_like(freqs)
    r = freqs / fc
    return 20.0 * np.log10(1.0 / np.sqrt(1.0 + r * r) + 1e-12)


def eq_peaking_db(freqs, fs, gains):
    """Total dB of the 3 peaking bands for gains = [g_low, g_mid, g_high]."""
    total = np.zeros_like(freqs)
    for (name, f0, bw), g in zip(EQ_BANDS, gains):
        total = total + rbj_peaking_db(freqs, fs, f0, bw, g)
    return total


# ── LTAS ──────────────────────────────────────────────────────────────────────

def compute_ltas(y, sr, fmin=FIT_FMIN, fmax=FIT_FMAX, n_grid=N_GRID):
    """Long-term average spectrum on a log-frequency grid, in dB.

    Returns (f_grid, ltas_db). The dB curve is *relative* (peak-normalised),
    so only its shape matters — absolute level is handled separately by `vol`.
    """
    y = mono_audio(y, sr)
    if not (np.isfinite(fmin) and np.isfinite(fmax) and 0 < fmin < min(fmax, sr / 2 - 1)):
        raise ValueError("Invalid LTAS frequency interval for this sample rate")
    if not isinstance(n_grid, (int, np.integer)) or n_grid < 4:
        raise ValueError("LTAS requires at least four grid points")
    nper = min(4096, len(y))
    if nper < 256:
        nper = len(y)
    freqs, psd = welch(y, sr, nperseg=nper, noverlap=nper // 2)
    psd = np.maximum(psd, 1e-20)
    ltas_db = 10.0 * np.log10(psd)

    f_grid = np.logspace(np.log10(fmin), np.log10(min(fmax, sr / 2 - 1)), n_grid)
    grid_db = np.interp(f_grid, freqs, ltas_db)
    grid_db = grid_db - grid_db.max()          # peak-normalise (shape only)
    return f_grid, grid_db


# ── Roll-off based hp / lp ────────────────────────────────────────────────────

def derive_hp_lp(f_grid, ref_db, clone_db, cur_hp=0, cur_lp=0):
    """Set hp/lp from excess energy at the spectral extremes of the clone.

    hp/lp are SAFETY guards (rumble / hiss), not tone controls — the EQ fits the
    tone. So hp is never allowed below 40 Hz: lowering it to compensate a
    low-end deficit is the EQ's job, and an unbounded loop walked it down 30 Hz
    per pass (95 -> 65 -> 35 -> 5 Hz = no filter at all, DC and rumble through).
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
            lp = float(np.clip(cur_lp + 1000, 6000, 12000))
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


__all__ = ['compute_ltas', 'fit_eq_ls', 'level_match_db', 'eq_peaking_db', 'rbj_peaking_db']


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

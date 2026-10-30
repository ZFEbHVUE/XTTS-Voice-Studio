"""Does the acoustic score of the fidelity search (voice_fidelity, 3.10) tell voices apart,
and how far does a post-processing chain move it, next to the timbre score of 3.9?

 A. Excerpts of one speaker saying other words against excerpts of two speakers
    (area under the curve, spread).
 B. Perturbations that a [] chain can really make, in units of the within-speaker spread.

Run, from anywhere:   python -I fidelity_vs_timbre.py [VOICES_DIR]

VOICES_DIR defaults to ../Voices_Cloning, the four reference voices of the repository
(VivienneHD_FR, RemiHD_FR, HollieUK_EN, EthanUK_EN; 16 kHz). Three 12 s excerpts of each,
at 1, 15 and 29 s. Needs numpy, scipy, soundfile and librosa; a few minutes on a CPU.
It only prints. It measures the two measures, not the clones: a score that separates
speakers and reacts to an EQ is steadier to search on, and says nothing about how a clone sounds.
"""
import itertools
import os
import sys
import warnings

import numpy as np
import soundfile as sf
from scipy.signal import butter, fftconvolve, sosfilt

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
VOICES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'Voices_Cloning')
sys.path.insert(0, HERE)
import voice_fidelity as VF            # the acoustic score of 3.10
import xtts_optimize as XO             # the timbre score of 3.9

SR = 16000
NAMES = ['VivienneHD_FR', 'RemiHD_FR', 'HollieUK_EN', 'EthanUK_EN']
MISSING = [n for n in NAMES if not os.path.exists(os.path.join(VOICES, n + '.wav'))]
if MISSING:
    sys.exit(f'Missing in {VOICES}: ' + ', '.join(n + '.wav' for n in MISSING))
SEG_S, STARTS = 12.0, (1.0, 15.0, 29.0)


def load(name):
    y, sr = sf.read(os.path.join(VOICES, name + '.wav'), dtype='float32')
    assert sr == SR
    return y if y.ndim == 1 else y.mean(axis=1)


def segments(y):
    return [y[int(s * SR):int((s + SEG_S) * SR)] for s in STARTS]


def gpt_profile(y):
    return VF.acoustic_profile(y, SR)


def my_features(y, method=None):
    return XO.timbre_features_array(y, SR, sr_target=22050, hnr_method=method)


def gpt_score(ref, cand):
    r = VF.compare_profiles(ref, cand, 0.0)
    return r['acoustic_score'], r['gaps']


def my_score(ref, cand):
    return XO.timbre_score(ref, cand)


# ── A. same speaker, different words / different speakers ────────────────────────────
voices = {n: segments(load(n)) for n in NAMES}
G = {n: [gpt_profile(s) for s in segs] for n, segs in voices.items()}
M = {}
for n, segs in voices.items():
    first = my_features(segs[0])
    M[n] = [first] + [my_features(s, first['hnr_method']) for s in segs[1:]]

within_g, within_m, cross_g, cross_m = [], [], [], []
for n in NAMES:
    for i, j in itertools.combinations(range(3), 2):
        within_g.append(gpt_score(G[n][i], G[n][j])[0])
        within_m.append(my_score(M[n][i], M[n][j]))
for a, b in itertools.combinations(NAMES, 2):
    for i in range(3):
        for j in range(3):
            cross_g.append(gpt_score(G[a][i], G[b][j])[0])
            cross_m.append(my_score(M[a][i], M[b][j]))


def auc(within, cross):
    pairs = [(w > c) + .5 * (w == c) for w in within for c in cross]
    return float(np.mean(pairs))


print('A. WITHIN speaker (other words) vs ACROSS speakers: score, higher = more alike')
print(f"   fidelity (3.10)   within {np.mean(within_g):.3f} ± {np.std(within_g):.3f} "
      f"[{min(within_g):.3f}..{max(within_g):.3f}]   across {np.mean(cross_g):.3f} ± {np.std(cross_g):.3f} "
      f"[{min(cross_g):.3f}..{max(cross_g):.3f}]   AUC {auc(within_g, cross_g):.3f}")
print(f"   timbre (3.9)      within {np.mean(within_m):.3f} ± {np.std(within_m):.3f} "
      f"[{min(within_m):.3f}..{max(within_m):.3f}]   across {np.mean(cross_m):.3f} ± {np.std(cross_m):.3f} "
      f"[{min(cross_m):.3f}..{max(cross_m):.3f}]   AUC {auc(within_m, cross_m):.3f}")
print()
print('   per voice (segment 1 vs 2), gaps of the fidelity terms')
for n in NAMES:
    sc, gaps = gpt_score(G[n][0], G[n][1])
    t = XO.timbre_breakdown(M[n][0], M[n][1]) if hasattr(XO, 'timbre_breakdown') else None
    print(f"   {n:14} F0 {G[n][0]['f0_hz']:5.0f} Hz  per.dB {G[n][0]['periodicity_db']:5.1f}  "
          f"sib {G[n][0]['sibilance_db']:6.1f}  dyn {G[n][0]['dynamics_db']:5.1f}  floor {G[n][0]['floor_db']:6.1f}  "
          f"voiced {G[n][0]['voiced_frames']:4d} | gaps " +
          ' '.join(f"{k[:3]} {v:4.2f}" for k, v in gaps.items()))

# ── B. perturbations a post chain can make, reference = another segment of the same voice ─
rng = np.random.default_rng(7)


def hp(y, fc, order=2):
    return sosfilt(butter(order, fc, 'highpass', fs=SR, output='sos'), y)


def lp(y, fc, order=2):
    return sosfilt(butter(order, fc, 'lowpass', fs=SR, output='sos'), y)


def shelf_high(y, gain_db, fc=3500.):
    return y + (10 ** (gain_db / 20) - 1) * hp(y, fc)


def shelf_low(y, gain_db, fc=250.):
    return y + (10 ** (gain_db / 20) - 1) * lp(y, fc)


def band(y, gain_db, lo=1000., hi=3000.):
    b = sosfilt(butter(2, [lo, hi], 'bandpass', fs=SR, output='sos'), y)
    return y + (10 ** (gain_db / 20) - 1) * b


def noise(y, snr_db):
    n = rng.normal(0, 1, len(y))
    n *= np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(n ** 2)) * 10 ** (-snr_db / 20)
    return y + n


def reverb(y, rt60):
    t = np.arange(int(rt60 * SR)) / SR
    ir = rng.normal(0, 1, len(t)) * 10 ** (-3 * t / rt60)
    ir[0] = 1.
    ir /= np.sqrt(np.sum(ir ** 2))
    return fftconvolve(y, ir)[:len(y)]


def pitch(y, st):
    import librosa
    return librosa.effects.pitch_shift(y.astype(np.float32), sr=SR, n_steps=st)


def stretch(y, rate):
    import librosa
    return librosa.effects.time_stretch(y.astype(np.float32), rate=rate)


def compress(y, amount):
    """Soft saturation: the chain's compressor/limiter raises the floor and cuts the peaks."""
    return np.tanh(y * (1 + amount)) / np.tanh(1 + amount)


perturbations = [
    ('gain x0.5', lambda y: y * .5), ('gain x2', lambda y: y * 2.),
    ('eq_high +2 dB', lambda y: shelf_high(y, 2)), ('eq_high +4 dB', lambda y: shelf_high(y, 4)),
    ('eq_high +8 dB', lambda y: shelf_high(y, 8)), ('eq_high -4 dB', lambda y: shelf_high(y, -4)),
    ('eq_low +4 dB', lambda y: shelf_low(y, 4)), ('eq_low -4 dB', lambda y: shelf_low(y, -4)),
    ('eq_mid +3 dB', lambda y: band(y, 3)), ('eq_mid -3 dB', lambda y: band(y, -3)),
    ('lp 6 kHz', lambda y: lp(y, 6000)), ('lp 4 kHz', lambda y: lp(y, 4000)),
    ('hp 150 Hz', lambda y: hp(y, 150)), ('hp 300 Hz', lambda y: hp(y, 300)),
    ('noise SNR 30 dB', lambda y: noise(y, 30)), ('noise SNR 20 dB', lambda y: noise(y, 20)),
    ('noise SNR 10 dB', lambda y: noise(y, 10)),
    ('reverb RT60 0.2 s', lambda y: reverb(y, .2)), ('reverb RT60 0.5 s', lambda y: reverb(y, .5)),
    ('saturation 1', lambda y: compress(y, 1.)), ('saturation 3', lambda y: compress(y, 3.)),
    ('pitch +1 st', lambda y: pitch(y, 1)), ('pitch +2 st', lambda y: pitch(y, 2)),
    ('pitch -2 st', lambda y: pitch(y, -2)),
    ('speed x1.1', lambda y: stretch(y, 1.1)), ('speed x0.9', lambda y: stretch(y, .9)),
]

print()
print('B. perturbation of segment 3 vs reference = segment 1, same voice; mean over the 4 voices.')
print('   Δ = score(perturbed) - score(unperturbed), in units of the within-speaker sd (content noise).')
sd_g, sd_m = np.std(within_g), np.std(within_m)
print(f'   content noise sd: fidelity {sd_g:.4f}   timbre {sd_m:.4f}')
print(f"   {'perturbation':20} {'fidelity Δ':>10} {'(in sd)':>8} {'biggest gap':>13}   {'timbre Δ':>9} {'(in sd)':>8} {'biggest term':>13}")
base_g = {n: gpt_score(G[n][0], G[n][2])[0] for n in NAMES}
base_m = {n: my_score(M[n][0], M[n][2]) for n in NAMES}
for label, fn in perturbations:
    dg, dm, big_g, big_m = [], [], {}, {}
    for n in NAMES:
        y = fn(voices[n][2])
        try:
            pg = gpt_profile(y)
            sg, gaps = gpt_score(G[n][0], pg)
            dg.append(sg - base_g[n])
            k = max(gaps, key=lambda kk: gaps[kk] / VF.SCALES[kk])
            big_g[k] = big_g.get(k, 0) + 1
        except Exception as exc:
            dg.append(np.nan)
        feats = my_features(y, M[n][0]['hnr_method'])
        if feats is None:
            dm.append(np.nan)
        else:
            dm.append(my_score(M[n][0], feats) - base_m[n])
            bd = XO.timbre_breakdown(M[n][0], feats)
            k = bd['largest'] if isinstance(bd, dict) and 'largest' in bd else '?'
            big_m[k] = big_m.get(k, 0) + 1
    mg, mm = np.nanmean(dg), np.nanmean(dm)
    top_g = max(big_g, key=big_g.get) if big_g else '-'
    top_m = max(big_m, key=big_m.get) if big_m else '-'
    print(f"   {label:20} {mg:+10.4f} {mg / sd_g:+8.1f} {top_g:>13}   {mm:+9.4f} {mm / sd_m:+8.1f} {top_m:>13}")

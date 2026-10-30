"""Auditable acoustic fidelity proxies and paired validation, NumPy/SciPy only.

These distances are engineering criteria, not a calibrated perceptual score.
Speaker identity and ASR remain independent guards. The fitting reference should
contain the same words as the fitted clone; held-out comparisons are paired.
"""
import math
import numpy as np
from scipy.signal import resample_poly

METRIC_VERSION = 'fidelity_v1'
WEIGHTS = dict(envelope=.45, pitch=.15, harmonicity=.15,
               sibilance=.10, dynamics=.10, floor=.05)
SCALES = dict(envelope=3., pitch=2., harmonicity=3.,
              sibilance=3., dynamics=3., floor=6.)


def _weighted_median(values, weights):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, cumulative[-1] / 2)])


def acoustic_profile(audio, sr):
    """Gain-invariant voiced envelope, pitch, periodicity, dynamics and noise.

    Periodicity uses normalised frame autocorrelation, not Praat's HNR: the
    result is named separately and never mixed with a Praat/HPSS measurement.
    Empty, silent, clipped-only or unvoiced material cannot be a perfect match.
    """
    y = np.asarray(audio, dtype=np.float64)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.ndim != 1 or not np.isfinite(y).all() or not math.isfinite(float(sr)) or sr <= 0:
        raise ValueError('Fidelity requires finite mono audio and a positive sample rate')
    clipped = float(np.mean(np.abs(y) >= .999)) if y.size else 1.
    if sr != 16000:
        divisor = math.gcd(int(sr), 16000)
        y = resample_poly(y, 16000 // divisor, int(sr) // divisor)
    sr = 16000
    y = y[:sr * 45]
    size, hop = 640, 160
    if len(y) < size * 3 or float(np.sqrt(np.mean(y*y))) < 1e-6:
        raise ValueError('Fidelity needs at least 0.12 s of non-silent audio')
    frames = np.lib.stride_tricks.sliding_window_view(y, size)[::hop].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(frames*frames, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    active = db >= max(-75., float(np.percentile(db, 90)) - 30.)
    if active.sum() < 3:
        raise ValueError('No usable speech frames for fidelity')
    frames, rms = frames[active], rms[active]
    # FFT gives every lag at once; the denominator accounts for the actual
    # overlap, avoiding a window/lag bias in low-pitched voices.
    lags = np.arange(round(sr/450), round(sr/60) + 1)
    transform = np.fft.rfft(frames, n=2048, axis=1)
    acf = np.fft.irfft(transform * transform.conj(), n=2048, axis=1).real
    energy = np.pad(np.cumsum(frames*frames, axis=1), ((0, 0), (1, 0)))
    denominator = np.sqrt(np.maximum(energy[:, size-lags] *
        (energy[:, size, None] - energy[:, lags]), 1e-20))
    correlation = np.clip(acf[:, lags] / denominator, -1., .999)
    peaks = np.zeros_like(correlation, dtype=bool)
    peaks[:, 1:-1] = ((correlation[:, 1:-1] >= correlation[:, :-2]) &
                     (correlation[:, 1:-1] >= correlation[:, 2:]))
    candidates = np.where(peaks, correlation, -1.)
    best = candidates.max(axis=1)
    # Prefer the first strong peak to a numerically higher octave/subharmonic.
    near_best = peaks & (correlation >= np.maximum(.40, best[:, None] - .03))
    index = near_best.argmax(axis=1)
    periodicity = correlation[np.arange(len(frames)), index]
    voiced = near_best.any(axis=1) & (periodicity >= .40)
    if voiced.sum() < 3:
        raise ValueError('No reliable voiced frames for fidelity')
    f0 = sr / lags[index[voiced]]
    weights = rms[voiced] * np.maximum(periodicity[voiced], .01)
    semitones = 12 * np.log2(f0)
    centre = _weighted_median(semitones, weights)
    keep = np.abs(semitones-centre) <= 7.
    pitch = float(np.sum(semitones[keep]*weights[keep])/weights[keep].sum())
    window = np.hanning(size)
    power = np.abs(np.fft.rfft(frames*window, n=2048, axis=1))**2
    frequencies = np.fft.rfftfreq(2048, 1/sr)
    # Equal frame levels prevent the loudest vowel from defining the voice.
    normalised = power / np.maximum(power.sum(axis=1, keepdims=True), 1e-20)
    psd = np.average(normalised[voiced], axis=0, weights=periodicity[voiced])
    import ltas_match as L
    grid = np.geomspace(80., 7500., 48)
    envelope = L.grid_db(frequencies, psd, grid, floor_hz=250.)
    envelope -= envelope.mean()
    total = power.sum(axis=0)
    band = lambda lo, hi: float(total[(frequencies >= lo) & (frequencies < hi)].sum())
    sibilance = 10*np.log10((band(5000, 7900)+1e-20)/(band(300, 3000)+1e-20))
    voiced_r = np.clip(periodicity[voiced], .01, .995)
    harmonicity = float(np.median(10*np.log10(voiced_r/(1-voiced_r))))
    active_db = db[active]
    return dict(envelope=envelope.tolist(), pitch_st=pitch,
        f0_hz=float(2**(pitch/12)), periodicity_db=harmonicity,
        sibilance_db=float(sibilance),
        dynamics_db=float(np.percentile(active_db, 90)-np.percentile(active_db, 20)),
        floor_db=float(np.clip(np.percentile(db, 10)-np.percentile(db, 90), -60, 0)),
        clipped_fraction=clipped, voiced_frames=int(voiced.sum()), method=METRIC_VERSION)


def compare_profiles(reference, candidate, identity):
    identity = float(identity)
    if not math.isfinite(identity):
        raise ValueError('Non-finite identity')
    a = np.asarray(reference['envelope'], float)
    b = np.asarray(candidate['envelope'], float)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Incompatible acoustic envelopes')
    gaps = dict(envelope=float(np.sqrt(np.mean((a-b)**2))),
        pitch=abs(reference['pitch_st']-candidate['pitch_st']),
        harmonicity=abs(reference['periodicity_db']-candidate['periodicity_db']),
        sibilance=abs(reference['sibilance_db']-candidate['sibilance_db']),
        dynamics=abs(reference['dynamics_db']-candidate['dynamics_db']),
        floor=abs(reference['floor_db']-candidate['floor_db']))
    if not all(math.isfinite(v) for v in gaps.values()):
        raise ValueError('Non-finite acoustic distance')
    distance = sum(WEIGHTS[k]*(math.sqrt(1+(v/SCALES[k])**2)-1) for k, v in gaps.items())
    acoustic = 1/(1+distance)
    return dict(score=.5*identity+.5*acoustic, identity=identity,
                acoustic_score=acoustic, distance=distance, gaps=gaps,
                clipped_fraction=candidate['clipped_fraction'])


def paired_decision(rows, min_gain=.005, identity_tolerance=.015):
    """Select on paired differences over >=3 texts with ASR/clipping guards.

    The one-standard-error margin is descriptive, not a confidence interval.
    No split-half differences between different words are called model noise.
    """
    reasons = []
    if len(rows) < 3:
        return dict(adopted=False, reasons=['at least three paired texts required'])
    texts = [str(row.get('text', '')).strip().casefold() for row in rows]
    if not all(texts) or len(set(texts)) != len(texts):
        return dict(adopted=False, reasons=['distinct nonempty validation texts required'])
    gains, identity_gains = [], []
    for row in rows:
        base, candidate = row['base_metrics'], row['candidate_metrics']
        gain, delta = candidate['score']-base['score'], candidate['identity']-base['identity']
        if not all(math.isfinite(float(x)) for x in (gain, delta,
                base['clipped_fraction'], candidate['clipped_fraction'])):
            return dict(adopted=False, reasons=['non-finite validation metric'])
        gains.append(gain); identity_gains.append(delta)
        before, after = row['base_asr'], row['candidate_asr']
        asr_numbers = [d.get(k) for d in (before, after) for k in ('score', 'wer')]
        if any(x is None or not math.isfinite(float(x)) for x in asr_numbers):
            return dict(adopted=False, reasons=['missing or non-finite ASR validation'])
        if before.get('wer') is None or after.get('wer') is None:
            reasons.append('missing ASR validation')
        elif after['wer'] > before['wer'] + .02:
            reasons.append('word error rate increased')
        if after['score'] < before['score']-.03:
            reasons.append('ASR intelligibility decreased')
        if after.get('detected') != before.get('detected'):
            reasons.append('ASR language changed')
        if candidate['clipped_fraction'] > max(.0001, base['clipped_fraction']+.0001):
            reasons.append('clipping increased')
    mean = float(np.mean(gains))
    stderr = float(np.std(gains, ddof=1)/math.sqrt(len(gains)))
    if mean <= min_gain or mean <= stderr:
        reasons.append('paired fidelity gain too small or inconsistent')
    if min(gains) < -.02:
        reasons.append('large fidelity regression on one text')
    if min(identity_gains) < -identity_tolerance or np.mean(identity_gains) < -.005:
        reasons.append('speaker identity decreased')
    return dict(adopted=not reasons, reasons=sorted(set(reasons)), mean_gain=mean,
                paired_stderr=stderr, gains=gains, identity_gains=identity_gains,
                minimum_gain=min_gain, identity_tolerance=identity_tolerance)


def metric_definition():
    return dict(version=METRIC_VERSION, objective='0.5 * ECAPA + 0.5 * acoustic fidelity',
        weights=WEIGHTS.copy(), scales=SCALES.copy(),
        warning='Engineering proxy, not a perceptual rating or calibrated probability',
        validation='Paired multi-text differences, ASR and identity guards')

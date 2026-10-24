"""Small audio input checks shared by the analysis utilities."""
import numpy as np


def mono_audio(audio, sr=None, *, allow_silence=False):
    if sr is not None and (not np.isfinite(sr) or sr <= 0 or int(sr) != sr):
        raise ValueError("Sample rate must be a positive integer")
    y = np.asarray(audio, dtype=np.float64)
    if y.ndim == 2 and y.shape[1] in (1, 2):
        y = y.mean(axis=1)
    if y.ndim != 1 or not y.size or not np.isfinite(y).all():
        raise ValueError("Audio must be nonempty, finite mono or mono/stereo frames")
    if not allow_silence and np.max(np.abs(y)) < 1e-7:
        raise ValueError("Audio is silent; no meaningful voice measurement is possible")
    return y

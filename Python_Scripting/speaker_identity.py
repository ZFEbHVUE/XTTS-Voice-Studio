#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
speaker_identity.py — Text-independent voice identity scoring via ECAPA-TDNN.

Wraps SpeechBrain's `spkrec-ecapa-voxceleb` speaker-verification encoder. The
cosine similarity between two utterances' embeddings is the standard metric for
"is this the same speaker" — independent of what was said. Used by the
comparator to (a) pick the best XTTS seed and (b) report a meaningful identity
cosine score (not a calibrated probability or identity verdict).

The model loads once and is cached. First call downloads ~80 MB from HuggingFace.

API:
  enc = SpeakerEncoder(device='cuda')   # or 'cpu'
  emb = enc.embed(path_or_array, sr=None)
  sim = SpeakerEncoder.cosine(emb_a, emb_b)
  sim = enc.similarity(ref_path, clone_path)
"""

import numpy as np
import os
import tempfile
from audio_checks import mono_audio

_TARGET_SR = 16000   # ECAPA-TDNN was trained at 16 kHz


def _load_audio(path, target_sr=_TARGET_SR):
    import soundfile as sf
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return _resample(y.astype(np.float32), sr, target_sr), target_sr


def _resample(y, sr, target_sr):
    y = mono_audio(y, sr)
    if not isinstance(target_sr, int) or target_sr <= 0:
        raise ValueError("target_sr must be a positive integer")
    if sr == target_sr:
        return y.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(sr), int(target_sr))
    return resample_poly(y, target_sr // g, sr // g).astype(np.float32)


class SpeakerEncoder:
    _models = {}

    def __init__(self, device=None, savedir=None):
        import torch
        self.torch = torch
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # SpeechBrain expects an indexed CUDA device ('cuda:0'); a bare 'cuda'
        # triggers a parse warning and a fallback. Normalise it.
        if device == 'cuda':
            device = 'cuda:0'
        self.device = device
        self.savedir = os.path.abspath(os.path.expanduser(savedir or os.path.join(tempfile.gettempdir(), 'ecapa_spkrec')))
        key = (str(device), self.savedir)
        if key not in SpeakerEncoder._models:
            SpeakerEncoder._models[key] = self._build()
        self.model = SpeakerEncoder._models[key]

    def _build(self):
        # Import path moved across SpeechBrain versions — try newest first.
        EncoderClassifier = None
        for mod in ('speechbrain.inference.speaker',
                    'speechbrain.inference',
                    'speechbrain.pretrained'):
            try:
                EncoderClassifier = __import__(mod, fromlist=['EncoderClassifier']).EncoderClassifier
                break
            except Exception:
                continue
        if EncoderClassifier is None:
            raise ImportError(
                "speechbrain not available — pip install speechbrain")
        return EncoderClassifier.from_hparams(
            source='speechbrain/spkrec-ecapa-voxceleb',
            savedir=self.savedir,
            run_opts={'device': self.device},
        )

    def embed(self, audio, sr=None):
        """Return an L2-normalised ECAPA embedding for a path or float array."""
        if isinstance(audio, (str, os.PathLike)):
            y, _ = _load_audio(audio)
        else:
            y = np.asarray(audio, dtype=np.float32)
            y = mono_audio(y, sr).astype(np.float32)
            if sr is not None and sr != _TARGET_SR:
                y = _resample(y, sr, _TARGET_SR)
        sig = self.torch.from_numpy(y).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            emb = self.model.encode_batch(sig).squeeze().detach().cpu().numpy()
        emb = emb.astype(np.float64)
        if not np.isfinite(emb).all() or np.linalg.norm(emb) <= 1e-12:
            raise ValueError("Encoder produced an invalid embedding")
        n = np.linalg.norm(emb)
        return emb / n

    def similarity(self, ref, clone):
        return self.cosine(self.embed(ref), self.embed(clone))

    @staticmethod
    def cosine(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        if (a.ndim != 1 or a.shape != b.shape or not a.size
                or not np.isfinite(a).all() or not np.isfinite(b).all()
                or min(np.linalg.norm(a), np.linalg.norm(b)) <= 1e-12):
            raise ValueError("Cosine requires two finite nonzero vectors of the same size")
        return float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1., 1.))


    def self_ceiling(self, reference):
        """Descriptive split-half reference anchor, not an upper bound or accuracy.

        The historical method name is retained for compatibility. Session,
        phonetic content and noise can change this value in either direction.
        Arrays are assumed to be 16 kHz, as in embed().
        """
        y, _ = _load_audio(reference) if isinstance(reference, (str, os.PathLike)) else (reference, None)
        y = mono_audio(y, _TARGET_SR, allow_silence=True).astype(np.float32)
        h = len(y) // 2
        if h < _TARGET_SR or min(np.sqrt(np.mean(y[:h] ** 2)), np.sqrt(np.mean(y[h:] ** 2))) < 1e-5:            # under 1 s per half: meaningless
            return None
        return self.cosine(self.embed(y[:h], sr=_TARGET_SR),
                           self.embed(y[h:], sr=_TARGET_SR))


__all__ = ['SpeakerEncoder']


if __name__ == '__main__':
    # CLI: identity cosine between two audio files (for A/B measurements).
    #   python speaker_identity.py reference.wav candidate.wav [candidate2.wav ...]
    import sys as _sys
    if len(_sys.argv) < 3:
        _sys.exit("usage: python speaker_identity.py ref.wav candidate.wav [more.wav ...]")
    _enc = SpeakerEncoder()
    _ref = _enc.embed(_sys.argv[1])
    print(f"  reference: {_sys.argv[1]}")
    _ceil = _enc.self_ceiling(_sys.argv[1])
    if _ceil is not None:
        print(f"  reference split-half cosine: {_ceil:.4f} (descriptive anchor)")
    for _cand in _sys.argv[2:]:
        _c = _enc.cosine(_ref, _enc.embed(_cand))
        print(f"  identity cosine {_c:.4f}  {_cand}")

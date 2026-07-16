#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
speaker_identity.py — Text-independent voice identity scoring via ECAPA-TDNN.

Wraps SpeechBrain's `spkrec-ecapa-voxceleb` speaker-verification encoder. The
cosine similarity between two utterances' embeddings is the standard metric for
"is this the same speaker" — independent of what was said. Used by the
comparator to (a) pick the best XTTS seed and (b) report a meaningful identity
score (0 = unrelated, 1 = identical speaker).

The model loads once and is cached. First call downloads ~80 MB from HuggingFace.

API:
  enc = SpeakerEncoder(device='cuda')   # or 'cpu'
  emb = enc.embed(path_or_array, sr=None)
  sim = SpeakerEncoder.cosine(emb_a, emb_b)
  sim = enc.similarity(ref_path, clone_path)
"""

import numpy as np

_TARGET_SR = 16000   # ECAPA-TDNN was trained at 16 kHz


def _load_audio(path, target_sr=_TARGET_SR):
    import soundfile as sf
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return _resample(y.astype(np.float32), sr, target_sr), target_sr


def _resample(y, sr, target_sr):
    if sr == target_sr:
        return y.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(sr), int(target_sr))
    return resample_poly(y, target_sr // g, sr // g).astype(np.float32)


class SpeakerEncoder:
    _model = None

    def __init__(self, device=None, savedir=None):
        import torch
        self.torch = torch
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # SpeechBrain expects an indexed CUDA device ('cuda:0'); a bare 'cuda'
        # triggers a parse warning and a fallback. Normalise it.
        if device == 'cuda':
            device = 'cuda:0'
        self.device = device
        self.savedir = savedir or '/tmp/ecapa_spkrec'
        if SpeakerEncoder._model is None:
            SpeakerEncoder._model = self._build()
        self.model = SpeakerEncoder._model

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
        if isinstance(audio, str):
            y, _ = _load_audio(audio)
        else:
            y = np.asarray(audio, dtype=np.float32)
            if sr is not None and sr != _TARGET_SR:
                y = _resample(y, sr, _TARGET_SR)
        sig = self.torch.from_numpy(y).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            emb = self.model.encode_batch(sig).squeeze().detach().cpu().numpy()
        emb = emb.astype(np.float64)
        n = np.linalg.norm(emb) + 1e-12
        return emb / n

    def similarity(self, ref, clone):
        return self.cosine(self.embed(ref), self.embed(clone))

    @staticmethod
    def cosine(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        return float(np.dot(a, b) /
                     ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


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
    for _cand in _sys.argv[2:]:
        _c = _enc.cosine(_ref, _enc.embed(_cand))
        _verdict = ("same speaker" if _c > 0.80 else
                    "close" if _c > 0.60 else "different")
        print(f"  identity {_c:.4f}  ({_verdict})  {_cand}")

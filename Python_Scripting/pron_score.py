#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pron_score.py — Pronunciation / accent scoring via faster-whisper.

ECAPA scores speaker *identity* (timbre) and is blind to accent: a clone with a
strong foreign accent on French keeps the reference timbre, so ECAPA rates it
highly. To pick a seed that actually speaks clean French, we need a different,
text/phonetics-aware signal.

This scorer transcribes each candidate clone with Whisper and derives a
"native-ness" score from three observable signals:
  - detected language ≠ target  -> strong accent / wrong phonetics (heavy penalty)
  - low avg_logprob             -> Whisper is "unsure" — typically mispronunciation
  - high word error rate vs the known target text -> garbled pronunciation

None of these is a calibrated accent meter, but together they correlate well with
"sounds like a native speaker of <lang>" and — unlike ECAPA — they move when the
accent is wrong. Trust your ear for the final call; this just ranks the seeds.

API:
  sc = PronScorer(device='cuda', model='small')
  r  = sc.score(wav_path, lang='fr', target_text="...")  # -> dict
  #    r['score'] in [0,1] (higher = cleaner), plus r['detected'], r['lang_prob'],
  #    r['avg_logprob'], r['wer'], r['text']
"""

import re
import numpy as np


def _normalise(s):
    s = s.lower()
    s = re.sub(r"[^\w\sàâäçéèêëîïôöùûüÿœæ'-]", ' ', s, flags=re.UNICODE)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.split()


def word_error_rate(ref_text, hyp_text):
    """Levenshtein word error rate, clipped to [0,1]. 0 = perfect."""
    r = _normalise(ref_text); h = _normalise(hyp_text)
    if not r:
        return 0.0
    # word-level edit distance
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    return float(min(1.0, d[len(r), len(h)] / len(r)))


class PronScorer:
    _model = None
    _model_key = None

    def __init__(self, device=None, model='small', compute_type=None):
        from faster_whisper import WhisperModel
        import torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        if compute_type is None:
            compute_type = 'float16' if self.device == 'cuda' else 'int8'
        key = (model, self.device, compute_type)
        if PronScorer._model is None or PronScorer._model_key != key:
            PronScorer._model = WhisperModel(model, device=self.device,
                                             compute_type=compute_type)
            PronScorer._model_key = key
        self.model = PronScorer._model

    def score(self, wav_path, lang='fr', target_text=None):
        lang = lang.lower().split('-')[0]
        # Detection pass — language=None lets Whisper reveal a wrong-accent clone
        # that it hears as another language.
        segments, info = self.model.transcribe(
            wav_path, language=None, beam_size=5, vad_filter=True)
        segs = list(segments)
        detected  = (info.language or '').lower()
        lang_prob = float(getattr(info, 'language_probability', 0.0) or 0.0)
        text = ' '.join(s.text for s in segs).strip()

        # Duration-weighted avg_logprob
        if segs:
            durs = np.array([max(s.end - s.start, 1e-3) for s in segs])
            alps = np.array([s.avg_logprob for s in segs])
            avg_logprob = float(np.sum(alps * durs) / np.sum(durs))
        else:
            avg_logprob = -2.0

        wer = word_error_rate(target_text, text) if target_text else None

        # ── Combine into [0,1] (higher = cleaner / more native) ──────────────
        lp_norm = float(np.clip((avg_logprob + 1.0) / 0.9, 0.0, 1.0))  # -1.0→0, -0.1→1
        if detected != lang:
            # Heard as a different language → wrong phonetics. Cap hard.
            score = 0.10 * lp_norm
        else:
            if wer is not None:
                score = 0.55 * lp_norm + 0.45 * (1.0 - wer)
            else:
                score = lp_norm
            score *= (0.6 + 0.4 * lang_prob)   # mild confidence weighting

        return dict(score=float(np.clip(score, 0.0, 1.0)),
                    detected=detected, lang_prob=lang_prob,
                    avg_logprob=avg_logprob, wer=wer, text=text)


__all__ = ['PronScorer', 'word_error_rate']

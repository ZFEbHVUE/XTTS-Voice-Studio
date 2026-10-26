#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pron_score.py — Pronunciation / accent scoring via faster-whisper.

ECAPA scores speaker *identity* (timbre) and is blind to accent: a clone with a
strong foreign accent on French keeps the reference timbre, so ECAPA rates it
highly. To pick a seed that actually speaks clean French, we need a different,
text/phonetics-aware signal.

This scorer measures ASR intelligibility and language consistency using detected
language, ASR confidence and word error rate. It does not measure native accent
or speaker likeness. Language detection can be wrong on short clips; use these
heuristics for screening, then compare longer held-out sentences and listen.

API:
  sc = PronScorer(device='cuda', model='small')
  r  = sc.score(wav_path, lang='fr', target_text="...")  # -> dict
  #    r['score'] in [0,1] (higher = cleaner), plus r['detected'], r['lang_prob'],
  #    r['avg_logprob'], r['wer'], r['text']
"""

import re
import numpy as np


def _normalise(s):
    import unicodedata
    s = unicodedata.normalize('NFC', s).lower().replace('’', "'").replace('‐', '-')
    s = re.sub(r"[^\w\sàâäçéèêëîïôöùûüÿœæ'-]", ' ', s, flags=re.UNICODE)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.split()


def _french_cardinal(n):
    """Standard French cardinals, 0..999999, for ASR spelling equivalence.

    This is deliberately not a pronunciation or accent judgement. Dates,
    signed/decimal numbers, ordinals and digit-by-digit identifiers are left
    alone by _number_equivalents below.
    """
    small = ('zéro', 'un', 'deux', 'trois', 'quatre', 'cinq', 'six', 'sept',
             'huit', 'neuf', 'dix', 'onze', 'douze', 'treize', 'quatorze',
             'quinze', 'seize')
    if n < 17:
        return small[n]
    if n < 20:
        return 'dix-' + small[n - 10]
    if n < 70:
        tens = ('', '', 'vingt', 'trente', 'quarante', 'cinquante', 'soixante')[n // 10]
        unit = n % 10
        return tens + (' et un' if unit == 1 else '-' + small[unit] if unit else '')
    if n < 80:
        return 'soixante' + (' et ' if n == 71 else '-') + _french_cardinal(n - 60)
    if n == 80:
        return 'quatre-vingts'
    if n < 100:
        return 'quatre-vingt-' + _french_cardinal(n - 80)
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = 'cent' if hundreds == 1 else small[hundreds] + ' cent'
        return head + (' ' + _french_cardinal(rest) if rest else 's' if hundreds > 1 else '')
    thousands, rest = divmod(n, 1000)
    head = 'mille' if thousands == 1 else _french_cardinal(thousands) + ' mille'
    # "quatre-vingt mille" / "deux cent mille": no plural before mille.
    if thousands != 1:
        head = re.sub(r'(vingt|cent)s mille$', r'\1 mille', head)
    return head + (' ' + _french_cardinal(rest) if rest else '')


def _number_equivalents(ref_text, hyp_text, lang):
    """Match written-out French integers ONLY when the pair contains digits.

    The digit is an explicit anchor: ordinary words are never converted on
    their own. Keep unsupported numeric formats distinct instead of guessing
    how the speaker should have read a date, a decimal or a phone number.
    """
    if (lang or '').lower().split('-')[0] != 'fr':
        return ref_text, hyp_text
    import unicodedata
    texts = [unicodedata.normalize('NFC', t).lower() for t in (ref_text, hyp_text)]
    protected = re.compile(r'(?<!\w)(?:[+-]\d+(?:[.,]\d+)?|'
                           r'\d+(?:[.,:/-]\d+)+|0\d+|'
                           r'\d{1,3}(?:[ \u00a0\u202f]\d{3})+)(?!\w)')
    numbers = set()
    for text in texts:
        plain = protected.sub(' ', text)
        numbers.update(int(x) for x in re.findall(r"(?<![\w'])\d{1,6}(?!\w)", plain))
    spellings = sorted(((_french_cardinal(n), n) for n in numbers),
                       key=lambda pair: len(pair[0]), reverse=True)
    for spelling, n in spellings:
        words = re.split(r'[ -]+', spelling)
        pattern = r"(?<![\w'])" + r'[\s\u00a0\u202f‐‑–—-]+'.join(map(re.escape, words)) + r'(?!\w)'
        texts = [re.sub(pattern, str(n), t) for t in texts]
    return tuple(texts)


def word_error_rate(ref_text, hyp_text, lang=None):
    """Levenshtein WER, clipped to [0,1]; optional FR cardinal equivalence.

    With lang=None this preserves the historical spelling-sensitive metric.
    With lang='fr', 48 and quarante-huit denote the same word sequence. Extra
    words, missing words and different numerical values still count.
    """
    ref_text, hyp_text = _number_equivalents(ref_text, hyp_text, lang)
    r = _normalise(ref_text); h = _normalise(hyp_text)
    if not r:
        return 0.0 if not h else 1.0
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
            compute_type = 'float16' if self.device.startswith('cuda') else 'int8'
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
        lang_prob = float(np.clip(lang_prob, 0, 1)) if np.isfinite(lang_prob) else 0.0
        text = ' '.join(s.text for s in segs).strip()

        # Duration-weighted avg_logprob
        if segs:
            durs = np.array([max(s.end - s.start, 1e-3) for s in segs])
            alps = np.array([s.avg_logprob for s in segs])
            valid = np.isfinite(alps) & np.isfinite(durs) & (durs > 0)
            avg_logprob = float(np.sum(alps[valid] * durs[valid]) / np.sum(durs[valid])) if valid.any() else -2.0
        else:
            avg_logprob = -2.0

        wer_raw = word_error_rate(target_text, text) if target_text is not None else None
        wer = word_error_rate(target_text, text, lang=lang) if target_text is not None else None

        # ── Combine into [0,1] (higher = easier for this ASR model) ──────────────
        lp_norm = float(np.clip((avg_logprob + 1.0) / 0.9, 0.0, 1.0))  # -1.0→0, -0.1→1
        if not segs or not text:
            score = 0.0
        elif detected != lang:
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
                    avg_logprob=avg_logprob, wer=wer, wer_raw=wer_raw, text=text)


__all__ = ['PronScorer', 'word_error_rate']

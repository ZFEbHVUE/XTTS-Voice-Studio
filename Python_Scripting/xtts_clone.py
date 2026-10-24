#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xtts_clone.py — correct XTTS v2 generation that actually honours the cloning knobs.

Some Coqui versions overwrite cloning keyword arguments in the high-level
`TTS.tts_to_file()` path with their model configuration. The effective reference
durations then depend on that configuration rather than the requested settings.

This module uses the documented low-level path instead:
  latents = model.get_conditioning_latents(audio_path, gpt_cond_len,
                                           gpt_cond_chunk_len, max_ref_length,
                                           sound_norm_refs)
  out     = model.inference(text, lang, *latents, temperature=..., ...)

Benefits: every cloning knob takes effect, max_ref_len is exposed, and the
latents are computed once per voice (not once per line) — faster and consistent.

API:
  tts, model = load_xtts(device)
  lat = compute_latents(model, speaker_wav, gpt_cond_len=30, max_ref_len=30, ...)
  generate(model, "texte", "fr", lat, out_path, temperature=0.65, ...)
"""

import os
import numpy as np
from xtts_settings import validate

XTTS_SR = 24000   # XTTS v2 output sample rate


def load_xtts(device=None):
    import torch
    from TTS.api import TTS
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to(device)
    return tts, tts.synthesizer.tts_model


def compute_latents(model, speaker_wav, gpt_cond_len=30, gpt_cond_chunk_len=4,
                    max_ref_len=30, sound_norm_refs=False):
    """Precompute (gpt_cond_latent, speaker_embedding), honouring every knob.

    gpt_cond_chunk_len must be <= gpt_cond_len (XTTS asserts this).
    """
    paths = speaker_wav if isinstance(speaker_wav, (list, tuple)) else [speaker_wav]
    if not paths:
        raise ValueError('At least one voice reference is required')
    paths = [os.fspath(p) for p in paths]
    for path in paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Voice reference not found: {path}')
    p = validate(dict(gpt_cond_len=gpt_cond_len,
                      gpt_cond_chunk_len=gpt_cond_chunk_len,
                      max_ref_len=max_ref_len, sound_norm_refs=sound_norm_refs))
    return model.get_conditioning_latents(
        audio_path=list(paths),
        gpt_cond_len=p['gpt_cond_len'],
        gpt_cond_chunk_len=p['gpt_cond_chunk_len'],
        max_ref_length=p['max_ref_len'],
        sound_norm_refs=bool(p['sound_norm_refs']),
    )


def set_seed(seed):
    """Seed every RNG involved in generation. Convention: the CALLER passes
    None to leave generation unseeded; any integer — INCLUDING 0 — is a valid
    seed. The old `if seed > 0` guard silently skipped seed 0, so every 'seed 0'
    result was unseeded and unreproducible (it scored differently on every run
    while all other seeds were stable), and could still be picked as a winner.
    """
    if seed is None:
        return
    seed = validate({'seed': seed})['seed']
    if seed is None:
        return
    import torch, random
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate(model, text, lang, latents, out_path,
             temperature=0.65, length_penalty=1.0, repetition_penalty=5.0,
             top_k=50, top_p=0.85, speed=1.0, enable_text_splitting=True,
             seed=None, num_beams=1, do_sample=True):
    """Generate one clip from precomputed latents and write a 24 kHz WAV.
    num_beams>1 uses deterministic beam search. With one beam,
    do_sample=False selects greedy decoding. Text splitting is enabled by
    default to keep long texts within XTTS's token limit."""
    p = validate(dict(seed=seed, temp=temperature, len_pen=length_penalty,
                      rep_pen=repetition_penalty, top_k=top_k, top_p=top_p,
                      num_beams=num_beams, do_sample=do_sample))
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Text to generate must not be empty')
    if not np.isfinite(float(speed)) or float(speed) <= 0:
        raise ValueError('speed must be finite and greater than zero')
    import soundfile as sf
    if seed is not None:
        set_seed(seed)
    gpt_cond_latent, speaker_embedding = latents
    nb = p['num_beams']
    ds = bool(p['do_sample'])
    if nb > 1 and ds:
        # num_beams>1 together with do_sample=True routes HF to `beam_sample`
        # (sampled beam search), which produces inf/nan probabilities with the
        # XTTS GPT and triggers an unrecoverable CUDA device-side assert.
        # Beam search is meaningful DETERMINISTIC anyway — that is the only mode
        # where length_penalty does anything — so force it.
        ds = False
    out = model.inference(
        text, lang.lower().split('-')[0],
        gpt_cond_latent, speaker_embedding,
        temperature=float(temperature),
        length_penalty=float(length_penalty),
        repetition_penalty=float(repetition_penalty),
        top_k=int(top_k),
        top_p=float(top_p),
        do_sample=ds,
        num_beams=nb,
        speed=float(speed),
        enable_text_splitting=bool(enable_text_splitting),
    )
    wav = np.asarray(out['wav'], dtype=np.float32).squeeze()
    if wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
        raise ValueError('XTTS produced empty, non-mono or non-finite audio')
    audio_config = getattr(getattr(model, 'config', None), 'audio', None)
    sr = (audio_config.get('output_sample_rate', XTTS_SR)
          if isinstance(audio_config, dict)
          else getattr(audio_config, 'output_sample_rate', XTTS_SR))
    sf.write(out_path, wav, int(sr))
    return out_path


__all__ = ['load_xtts', 'compute_latents', 'generate', 'set_seed', 'XTTS_SR']

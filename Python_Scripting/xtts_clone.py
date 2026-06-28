#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xtts_clone.py — correct XTTS v2 generation that actually honours the cloning knobs.

The high-level `TTS.tts_to_file()` path (model.synthesize) OVERWRITES gpt_cond_len,
gpt_cond_chunk_len, max_ref_len and sound_norm_refs with the XttsConfig defaults
(12 / 4 / 10 / False) AFTER applying your kwargs — so passing them is a no-op, and
max_ref_len is never exposed at all. The speaker embedding therefore only ever
sees the first 10 s of reference, and the GPT conditioning only 12 s, regardless
of how much clean reference you provide.

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

import numpy as np

XTTS_SR = 24000   # XTTS v2 output sample rate


def load_xtts(device=None):
    import torch
    from TTS.api import TTS
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2').to(device)
    return tts, tts.synthesizer.tts_model


def compute_latents(model, speaker_wav, gpt_cond_len=30, gpt_cond_chunk_len=6,
                    max_ref_len=30, sound_norm_refs=False):
    """Precompute (gpt_cond_latent, speaker_embedding), honouring every knob.

    gpt_cond_chunk_len must be <= gpt_cond_len (XTTS asserts this).
    """
    paths = speaker_wav if isinstance(speaker_wav, (list, tuple)) else [speaker_wav]
    gpt_cond_chunk_len = min(int(gpt_cond_chunk_len), int(gpt_cond_len))
    return model.get_conditioning_latents(
        audio_path=list(paths),
        gpt_cond_len=int(gpt_cond_len),
        gpt_cond_chunk_len=int(gpt_cond_chunk_len),
        max_ref_length=int(max_ref_len),
        sound_norm_refs=bool(sound_norm_refs),
    )


def set_seed(seed):
    import torch, random
    seed = int(seed or 0)
    if seed > 0:
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def generate(model, text, lang, latents, out_path,
             temperature=0.65, length_penalty=1.0, repetition_penalty=5.0,
             top_k=50, top_p=0.85, speed=1.0, enable_text_splitting=False,
             seed=None):
    """Generate one clip from precomputed latents and write a 24 kHz WAV."""
    import soundfile as sf
    if seed is not None:
        set_seed(seed)
    gpt_cond_latent, speaker_embedding = latents
    out = model.inference(
        text, lang.lower().split('-')[0],
        gpt_cond_latent, speaker_embedding,
        temperature=float(temperature),
        length_penalty=float(length_penalty),
        repetition_penalty=float(repetition_penalty),
        top_k=int(top_k),
        top_p=float(top_p),
        speed=float(speed),
        enable_text_splitting=bool(enable_text_splitting),
    )
    wav = np.asarray(out['wav'], dtype=np.float32)
    sf.write(out_path, wav, XTTS_SR)
    return out_path


__all__ = ['load_xtts', 'compute_latents', 'generate', 'set_seed', 'XTTS_SR']

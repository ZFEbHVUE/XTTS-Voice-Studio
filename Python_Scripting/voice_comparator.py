#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_comparator.py — rendered tone fitting and all-field audio search.

The {} block is FROZEN input (seed, temp, rep_pen… are chosen upstream by the
Validator, which scores accent + identity). The comparator's only job is the
post-processing tone: it generates the clone once on the reference text, then
iteratively fits the generator's exact 3-band peaking EQ + volume by least
squares so the rendered clone's long-term average spectrum (LTAS) matches the
reference. Each pass renders the full chain (hp/lp/NR/comp/limiter), re-measures,
and adds the LS correction — closing the loop on the post-chain colouring.

With --optimise-audio, every free [] field gets explicit trials (including
off/on), followed by joint ECAPA refinement, pruning of the changes worth less
than the heuristic tolerance, and validation on two other texts. Speed, pan and
limiter are held unless --search-all-fields: ECAPA cannot judge them. The
search budget includes the trials and the pruning. A JSON audit and validation
WAVs are kept; a failed search leaves the tone fit. N and LANG are
routing/language choices. Generation knobs in {} stay
frozen here and are searched upstream. Generation goes through xtts_clone.py so the cloning knobs
(gpt_cond_len / gpt_cond_chunk_len / max_ref_len / sound_norm_refs) actually take
effect, consistent with the Validator and the final render.

Usage:
  python voice_comparator.py ref.wav [extra_refs...] FR \\
      --xtts-block "{1, 42, ...}" --audio-block "[1, FR, ...]" \\
      [--text-file script.txt | --text "Phrase de test."] \\
      [--iterations 3] [--conv-threshold 0.5] \\
      [--max-ref-len 30] [--output clone.wav] [--output-optimised clone_opt.wav]
"""


# Windows consoles default to cp1252 and raise UnicodeEncodeError on any
# non-ASCII character in output — a run that computed correctly then died on a
# print. Force UTF-8 where the interpreter allows it.
try:
    import sys as _sys
    if hasattr(_sys.stdout, 'reconfigure'):
        _sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        _sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import os
import sys
import re
import argparse
import json
from xtts_settings import parse_block, format_block, validate
import tempfile
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

LANGS = {'FR','EN','ES','DE','IT','PT','PL','TR','RU','NL','CS','AR','ZH-CN','HU','KO','JA','HI'}


# ── Block parsing / formatting ────────────────────────────────────────────────

def parse_xtts_block(block):
    return parse_block(block)


def parse_audio_block(block):
    raw = block.strip()
    if not raw.startswith('[') or not raw.endswith(']'):
        raise ValueError("Audio block must be enclosed in brackets")
    parts = [part.strip() for part in raw[1:-1].split(',')]
    if not parts or not parts[0].isdigit() or int(parts[0]) < 1:
        raise ValueError("Audio block needs a positive voice number")
    parts = parts[1:]
    lang = 'FR'
    if parts and parts[0].upper() in LANGS:
        lang = parts.pop(0).upper()
    keys = ['speed', 'vol', 'eq_low', 'eq_mid', 'eq_high', 'hp', 'lp',
            'NR', 'comp', 'de-ess', 'reverb', 'noise_gate', 'pan', 'limiter']
    if len(parts) > len(keys):
        raise ValueError("Too many audio parameters")
    out = {'lang': lang}
    for key, token in zip(keys, parts):
        if token:
            value = float(token)
            if not np.isfinite(value):
                raise ValueError(f"{key}: value must be finite")
            out[key] = value
    if out.get('speed', 1) <= 0 or not -1 <= out.get('pan', 0) <= 1:
        raise ValueError("speed must be positive and pan in [-1, 1]")
    if out.get('limiter', 1) not in (0, 1):
        raise ValueError("limiter must be 0 or 1")
    for key in ('hp', 'lp', 'NR', 'comp', 'de-ess', 'reverb'):
        if out.get(key, 0) < 0:
            raise ValueError(f"{key} must be nonnegative")
    return out


def block_language(block):
    """The language a [] block names (upper case), or None when it names none.

    parse_audio_block fills in FR for a block without one, so its 'lang' cannot
    tell "this block is French" from "this block says nothing"."""
    parts = [part.strip() for part in str(block).strip().strip('[]').split(',')]
    return parts[1].upper() if len(parts) > 1 and parts[1].upper() in LANGS else None


def resolve_language(asked, block):
    """(language, error): the command's or the block's, which must agree when
    both name one; FR when neither does. A block without a language used to
    impose FR even on a command that asked for another."""
    named = block_language(block) if block else None
    asked = asked.upper() if asked else None
    if asked and named and asked != named:
        return None, (f'Language mismatch: the command asks for {asked}, '
                      f'the [] block for {named}')
    return asked or named or 'FR', None


def fmt(v):
    f = float(v)
    if f == int(f):
        return str(int(f))
    return str(round(f, 3))


def format_blocks(N, xtts, audio, lang):
    aa = [audio.get('speed', 1.0), audio.get('vol', 0), audio.get('eq_low', 0),
          audio.get('eq_mid', 0), audio.get('eq_high', 0), audio.get('hp', 0),
          audio.get('lp', 0), audio.get('NR', 0), audio.get('comp', 0),
          audio.get('de-ess', 0), audio.get('reverb', 0), audio.get('noise_gate', 0),
          audio.get('pan', 0), audio.get('limiter', 1)]
    xtts_str = format_block(xtts, voice=N)
    audio_str = f"[{N}, {lang}, " + ', '.join(fmt(v) for v in aa) + ']'
    return xtts_str, audio_str


# ── Text ──────────────────────────────────────────────────────────────────────

def clean_text(raw, lang='en'):
    """Strip pause/parameter markup, keep prose for a short test clip.
    Falls back to a built-in sentence IN THE TARGET LANGUAGE when empty."""
    lines = []
    for ln in (raw or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if s.startswith('{') or s.startswith('['):
            continue
        s = re.sub(r'\[pause[^\]]*\]', ' ', s, flags=re.I)
        s = re.sub(r'\[/?parallel[^\]]*\]', ' ', s, flags=re.I)
        s = s.strip()
        if s:
            lines.append(s)
    text = ' '.join(lines).strip()
    from probe_texts import default_text as _dt
    return text or _dt(lang)


# ── Audio I/O ─────────────────────────────────────────────────────────────────

def load_mono(path, target_sr=None):
    import soundfile as sf
    y, sr = sf.read(path, always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if target_sr and sr != target_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), int(target_sr))
        y = resample_poly(y, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr
    return y, sr


# ── Post-processing ───────────────────────────────────────────────────────────


def _encoder(device):
    """SpeakerEncoder on `device`, falling back to CPU when VRAM is short.
    XTTS alone fills a 4 GB card, so asking for a second GPU model there
    aborted whole stages with an out-of-memory error."""
    from speaker_identity import SpeakerEncoder
    try:
        return SpeakerEncoder(device=device)
    except Exception as exc:
        if 'out of memory' not in str(exc).lower():
            raise
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        print("   [!] Not enough VRAM for ECAPA -- running it on CPU.")
        return SpeakerEncoder(device='cpu')


def _quiet_post(raw_path, out_path, audio, xtts, lang, gen_path):
    """apply_post with the generator's per-render logging suppressed.
    The screening and the optimiser call it hundreds of times; without this,
    one useful line drowns in thousands of '[*] Audio: EQ(...)' lines."""
    import io, contextlib
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = apply_post(raw_path, out_path, audio, xtts, lang, gen_path)
    # An unavailable effect must not masquerade as a successfully tested value.
    # 'eq error' and 'limiter error': the generator's EQ and limiter failures,
    # which return the audio unprocessed (the limiter's 'failed' wording only
    # appears when the unlimited gain also clips).
    messages = output.getvalue()
    if any(marker in messages.lower() for marker in
           ('effect skipped', 'eq error', 'reduction error', 'de-esser error',
            'reverb error', 'noise gate error', 'limiter skipped', 'limiter error',
            'limiter failed')):
        raise RuntimeError('Audio effect failed: ' + messages.strip()[-700:])
    return result


def _halves_tolerance(enc, ref_emb, path, floor=0.004):
    """Heuristic tolerance from the identity spread between two clip halves.
    Their words also differ: this is not an independent noise estimate or a
    statistical confidence interval.
    """
    if not np.isfinite(floor) or floor < 0:
        raise ValueError('Invalid identity tolerance floor')
    y, sr = load_mono(path, target_sr=16000)
    if len(y) < 2:
        raise ValueError('Audio is too short to measure split-half identity')
    half = len(y) // 2
    first = float(enc.cosine(ref_emb, enc.embed(y[:half], sr=sr)))
    second = float(enc.cosine(ref_emb, enc.embed(y[half:], sr=sr)))
    if not (np.isfinite(first) and np.isfinite(second)):
        raise ValueError('Non-finite split-half identity score')
    return max(floor, abs(first - second) / 2.)


_GEN_CACHE = {}


def _generator_chain(gen_path):
    """The generator's render_voice_chain, loaded once.
    The screening and optimiser call apply_post hundreds of times; re-executing
    the whole generator module on every call was pure overhead."""
    if gen_path in _GEN_CACHE:
        return _GEN_CACHE[gen_path]
    chain = None
    if gen_path and os.path.exists(gen_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location('gmg', gen_path)
        gmg = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(gmg)
            chain = getattr(gmg, 'render_voice_chain', None)
            if chain is None:
                print("   [!] The generator has no render_voice_chain: it is older "
                      "than this comparator. Install both files together.")
        except Exception as e:
            print(f"   [!] Could not load the generator's audio chain: {e}")
    _GEN_CACHE[gen_path] = chain
    return chain


def apply_post(raw_path, out_path, audio, xtts, lang, gen_path):
    """Apply the generator's own render chain to a raw clone."""
    from pydub import AudioSegment
    seg = AudioSegment.from_wav(raw_path)
    render_voice_chain = _generator_chain(gen_path)
    if render_voice_chain is None:
        raise RuntimeError('Generator audio chain could not be loaded; refusing to score an unprocessed clone')
    if render_voice_chain:
        config = {
            'speed': float(audio.get('speed', 1.0)), 'volume': float(audio.get('vol', 0)),
            'eq_low': float(audio.get('eq_low', 0)), 'eq_mid': float(audio.get('eq_mid', 0)),
            'eq_high': float(audio.get('eq_high', 0)), 'highpass': float(audio.get('hp', 0)),
            'lowpass': float(audio.get('lp', 0)), 'noise_reduction': float(audio.get('NR', 0)),
            'compression': float(audio.get('comp', 0)), 'deesser': float(audio.get('de-ess', 0)),
            'reverb': float(audio.get('reverb', 0)), 'noise_gate': float(audio.get('noise_gate', 0)),
            'pan': float(audio.get('pan', 0)),
            # The chain reads the limiter from CONFIG. It used to be put in
            # the xtts dict, where nothing reads it, so the comparator never
            # limited while the generator did.
            'limiter': int(audio.get('limiter', 1)),
            'language': lang.lower(),
        }
        xtts_fx = {
            'trim_start': int(xtts.get('trim_start', 0)), 'trim_end': int(xtts.get('trim_end', 0)),
            'fade_in': int(xtts.get('fade_in', 100)), 'fade_out': int(xtts.get('fade_out', 250)),
        }
        # The generator's own function: effects -> speed -> volume -> limiter.
        # Rebuilding that order here is how it drifted before (speed missing,
        # limiter before the volume).
        seg = render_voice_chain(seg, config, xtts_fx)
    seg.export(out_path, format='wav')
    return out_path


# ── Main ──────────────────────────────────────────────────────────────────────

def unseen_sentence(lang, fit_text, hold_text=None, explicit=None):
    """A final check must use neither the fitting text nor the selection text."""
    held = [hold_text] if isinstance(hold_text, str) else list(hold_text or [])
    excluded = {t.strip().casefold() for t in [fit_text, *held] if t}
    if explicit:
        candidate = clean_text(explicit, lang)
        if candidate.casefold() in excluded:
            raise ValueError("unseen-text must differ from fitting and hold-out sentences")
        return candidate
    from probe_texts import holdout_texts, probe_texts
    candidates = holdout_texts(lang, 2) + probe_texts(lang, 3)
    for candidate in candidates:
        if candidate.strip().casefold() not in excluded:
            return candidate
    raise ValueError("No independent final-check sentence; supply --unseen-text")


def transcript_excerpt(segments, limit=240):
    """Keep text paired with audio, tolerating zero-duration ASR word tokens.

    Such tokens are grouped with a timed neighbour, never silently dropped.
    Broken word alignment falls back to a COMPLETE, timed segment; it never
    pairs a truncated segment transcript with the whole segment's audio.
    """
    if limit <= 0:
        raise ValueError('Transcription text limit must be positive')

    def interval(entry):
        try:
            start, end = float(entry.start), float(entry.end)
        except (AttributeError, TypeError, ValueError):
            return None
        if not (np.isfinite(start) and np.isfinite(end) and 0 <= start <= end):
            return None
        return start, end

    diagnostics, diagnostic_count = [], 0

    def diagnose(reason, entry, segment_number):
        nonlocal diagnostic_count
        diagnostic_count += 1
        if len(diagnostics) < 8:
            token = str(getattr(entry, 'word', getattr(entry, 'text', '')) or '')[:64]
            start = repr(getattr(entry, 'start', None))[:80]
            end = repr(getattr(entry, 'end', None))[:80]
            diagnostics.append(f'[INFO] ASR segment {segment_number}: {reason}; '
                               f'text={token!r}, start={start}, end={end}')

    def report_diagnostics():
        for message in diagnostics:
            print(message)
        if diagnostic_count > len(diagnostics):
            print(f'[INFO] ASR: {diagnostic_count - len(diagnostics)} further timestamp issue(s).')

    parts, first, last, count = [], None, None, 0
    repaired, fallback, skipped = 0, 0, 0
    for segment_number, segment in enumerate(segments, 1):
        words = list(getattr(segment, 'words', None) or [])
        units, pending, previous_start, valid = [], [], None, True
        for word in words:
            text = str(getattr(word, 'word', '') or '').strip()
            if not text:
                continue
            bounds = interval(word)
            if bounds is None or (previous_start is not None and bounds[0] < previous_start):
                diagnose('invalid or out-of-order word timestamps; trying complete segment',
                         word, segment_number)
                valid = False
                break
            start, end = bounds
            previous_start = start
            if start == end:
                diagnose('zero-duration word; grouping with timed neighbour', word, segment_number)
                repaired += 1
                if units:
                    units[-1][0] += ' ' + text
                    units[-1][2] = max(units[-1][2], end)
                else:
                    pending.append((text, start))
                continue
            if pending:
                text = ' '.join([item[0] for item in pending] + [text])
                start = min(start, pending[0][1])
                pending = []
            units.append([text, start, end])

        if not words or not valid or not units:
            bounds = interval(segment)
            text = str(getattr(segment, 'text', '') or '').strip()
            if not text:
                text = ' '.join(str(getattr(word, 'word', '') or '').strip() for word in words).strip()
            if not text or bounds is None or bounds[0] == bounds[1]:
                diagnose('complete segment unavailable; ending or skipping excerpt', segment, segment_number)
                if parts:
                    break
                skipped += 1
                continue
            units = [[text, *bounds]]
            if words:
                fallback += 1

        stop = False
        for text, start, end in units:
            if count + len(text) + bool(parts) > limit:
                # A later segment may have usable word alignment. Before the
                # excerpt starts, skip an indivisible overlong segment safely.
                if not parts:
                    diagnose('complete text unit exceeds excerpt limit; skipping leading segment',
                             segment, segment_number)
                    skipped += 1
                    break
                stop = True
                break
            parts.append(text)
            first = float(start) if first is None else first
            last = float(end) if last is None else max(last, float(end))
            count += len(text) + (1 if len(parts)>1 else 0)
        if stop:
            break
    report_diagnostics()
    if not parts:
        raise ValueError('No usable timestamped excerpt within text limit; provide --text instead of --auto-text')
    if repaired or fallback or skipped:
        print(f'[INFO] ASR alignment: {repaired} zero-duration token(s) grouped; '
              f'{fallback} complete segment fallback(s); {skipped} leading segment(s) skipped.')
    return ' '.join(parts), first, last


def _filter_moves(held, filters):
    """'low-pass 9000 Hz -> off' for each filter that changes, '' if none."""
    def shown(value):
        return f'{float(value):.0f} Hz' if float(value) else 'off'
    return ', '.join(f'{name} {shown(held[key])} -> {shown(filters[key])}'
                     for key, name in (('hp', 'high-pass'), ('lp', 'low-pass'))
                     if abs(float(filters[key]) - float(held[key])) > 1e-9)


def fit_rendered_tone(reference, reference_sr, render_sr, initial, render,
                      iterations=3, eq_limit=4., weighting='a', target_dbfs=None,
                      convergence=.5):
    """Keep only corrections that improve the spectrum AFTER the actual chain.

    render(config) returns mono samples resampled to reference_sr. Digital EQ
    and filter responses must use render_sr, even when analysis runs at
    reference_sr.

    The first pass also settles hp/lp: what ltas_match.derive_hp_lp proposes
    and, while a low-pass is on, the low-pass off. pydub's first-order low-pass
    tilts the whole upper band of a 24 kHz render (lp 9000: -2.1 dB at 4 kHz,
    -4.5 dB at 8 kHz), and with it on, eq_high spent its bound giving back what
    the filter took. For each setting the EQ is fitted on the curve those
    filters will give (their exact responses), so it does not boost again what
    a filter change gives back, and the shape error it leaves is predicted.
    The low-pass off goes first when its prediction is better, the block's own
    filters last; the first render that improves the measured error is kept.

    Spectra are compared smoothed (ltas_match.grid_db: bands of max(250 Hz,
    1/3 octave)), so the harmonic comb of the voice, which follows the melody
    of the sentence, is not taken for a tone to correct. The EQ is fitted on
    the render's own PSD (ltas_match.fit_eq_psd): the bells and filters act on
    it bin by bin before the smoothing, as they do on the audio, so the
    prediction holds for every linear stage of the chain.
    """
    import ltas_match as L
    if not np.isfinite(eq_limit) or eq_limit <= 0:
        raise ValueError('eq-limit must be finite and positive')
    fmax=min(L.FIT_FMAX,render_sr/2-1,reference_sr/2-1)
    grid, ref_db=L.compute_ltas(reference,reference_sr,fmax=fmax)
    keys=('eq_low','eq_mid','eq_high')
    current=dict(initial)
    current.update({key:float(np.clip(current.get(key,0),-eq_limit,eq_limit)) for key in keys})
    for key in ('vol','hp','lp'):
        current.setdefault(key,0)
    def measure(config):
        audio=render(config)
        spectrum=L.ltas_psd(audio,reference_sr)
        db=L.grid_db(*spectrum,grid)
        db=db-db.max()
        return audio,spectrum,db,L.spectral_shape_error(grid,ref_db,db,weighting)
    def level(samples):
        return (L.level_to_target_db(samples,target_dbfs) if target_dbfs is not None
                else L.level_match_db(reference,samples))
    def propose(spectrum,held,filters):
        # The EQ fitted on the render's PSD with these filters in place of the
        # held ones, and the shape error the pair will leave.
        freqs,psd=spectrum
        shift=np.where(freqs<render_sr/2,L.filter_change_db(freqs,render_sr,held,filters),0.)
        gains,model=L.fit_eq_psd(grid,ref_db,freqs,psd,fs=render_sr,weighting=weighting,
                                 limit=eq_limit,start=[current[key] for key in keys],extra_db=shift)
        gains={key:round(float(np.clip(gains[key],-eq_limit,eq_limit)),1) for key in keys}
        return dict(filters=dict(filters),gains=gains,
                    predicted=L.spectral_shape_error(grid,ref_db,model,weighting))
    audio,spectrum,curve,error=measure(current)
    history=[]
    for index in range(max(1,iterations)):
        held={key:current[key] for key in ('hp','lp')}
        if index:
            options=[propose(spectrum,held,held)]      # filters frozen after pass 1
        else:
            hp,lp=L.derive_hp_lp(grid,ref_db,curve,held['hp'],held['lp'])
            derived=dict(hp=hp,lp=lp)
            options=[propose(spectrum,held,derived)]
            if _filter_moves(held,derived):
                options.append(propose(spectrum,held,held))
            if float(held['lp']) and float(lp):
                off=propose(spectrum,held,dict(derived,lp=0))
                if off['predicted']<options[0]['predicted']-.01:
                    options.insert(0,off)
        for attempt,option in enumerate(options):
            candidate=dict(current)
            candidate.update(option['gains'])
            candidate.update(option['filters'])
            candidate['vol']=float(np.clip(current['vol']+level(audio),-18,18))
            trial,trial_spectrum,trial_curve,trial_error=measure(candidate)
            # Re-match level after the EQ change; one more render, no XTTS generation.
            level_delta=level(trial)
            if level_delta:
                candidate['vol']=float(np.clip(candidate['vol']+level_delta,-18,18))
                trial,trial_spectrum,trial_curve,trial_error=measure(candidate)
            moves=_filter_moves(held,option['filters'])
            accepted=trial_error < error-.01
            same_shape=(abs(trial_error-error)<=.01 and not moves
                        and all(abs(candidate[k]-current[k])<1e-9 for k in keys))
            kept=accepted or same_shape
            history.append(dict(pass_number=index+1,before=error,after=trial_error,
                                predicted=option['predicted'],accepted=kept,config=dict(candidate)))
            note=moves or ('filters unchanged' if attempt else '')
            print(f'  pass {index+1}: rendered shape error {error:.2f} -> {trial_error:.2f} dB '
                  f"(predicted {option['predicted']:.2f}); "
                  f"EQ=({candidate['eq_low']:+.1f}/{candidate['eq_mid']:+.1f}/{candidate['eq_high']:+.1f}), "
                  f"volume={candidate['vol']:+.1f} dB; " + (f'{note}; ' if note else '')
                  + ('kept' if kept else 'rejected'))
            if kept:
                break
        if not kept:
            break
        step=max(abs(candidate[k]-current[k]) for k in (*keys,'vol'))
        current,audio,spectrum,curve,error=candidate,trial,trial_spectrum,trial_curve,trial_error
        # A filter change in pass 1 earns pass 2 even when the EQ barely moved.
        if (step<convergence and not moves) or same_shape:
            break
    return current,history


def audio_holdouts(lang, fit_text, explicit=None):
    """Two selection sentences; neither can be the fitting sentence."""
    from probe_texts import holdout_texts, probe_texts
    candidates = ([clean_text(explicit, lang)] if explicit else [])
    candidates += holdout_texts(lang, 2) + probe_texts(lang, 3)
    selected, seen = [], {fit_text.strip().casefold()}
    for candidate in candidates:
        key = candidate.strip().casefold()
        if key not in seen:
            selected.append(candidate)
            seen.add(key)
        if len(selected) == 2:
            return selected
    raise ValueError('Two independent audio-selection sentences are required')


def fit_audio_level(raw_path, out_path, block, xtts, lang, gen_path, target_dbfs=None):
    """Render the exported block, deriving its gain from a requested RMS target.

    Compression, NR, gate, pan and reverb may change the output level. Holding
    the old gain fixed did not hold the level fixed. Re-render after correction
    and reject unreachable targets instead of silently scoring another level.
    """
    import soundfile as sf
    block = dict(block)
    for _ in range(4 if target_dbfs is not None else 1):
        _quiet_post(raw_path, out_path, block, xtts, lang, gen_path)
        if target_dbfs is None:
            return block
        # Use channel power, not a mono downmix (panning changes that downmix's
        # level). Do not use level_to_target_db here: it rounds to whole dB.
        samples, _ = sf.read(out_path, always_2d=False)
        samples = np.asarray(samples, dtype=float)
        if not samples.size or float(np.max(np.abs(samples))) < 1e-10:
            raise ValueError('Cannot normalise an empty or silent render')
        rms_db = 20. * np.log10(np.sqrt(np.mean(samples ** 2)) + 1e-12)
        correction = float(target_dbfs - rms_db)
        if not np.isfinite(correction):
            raise ValueError('Cannot measure rendered RMS')
        if abs(correction) <= .35:
            return block
        block['vol'] = round(float(np.clip(block.get('vol', 0) + correction, -36, 24)), 3)
    raise ValueError('Requested RMS target was not reached within 0.35 dB')


def main():
    p = argparse.ArgumentParser(description='Identity-driven XTTS clone optimiser')
    p.add_argument('reference')
    p.add_argument('voice_refs', nargs='*')
    p.add_argument('--clone-refs', nargs='+', default=None,
                   help='Exact cloning reference subset; comparison target stays positional')
    p.add_argument('--xtts-block', required=True)
    p.add_argument('--audio-block', required=True)
    p.add_argument('--text', default=None,
                   help='Sentence to fit on (default: a built-in sentence in the '
                        'target language — see probe_texts.py)')
    p.add_argument('--text-file', default=None)
    p.add_argument('--unseen-text', help='Independent sentence for the final identity check')
    p.add_argument('--holdout-text', help='Unseen sentence in the target language for audio optimisation')
    p.add_argument('--auto-text', action='store_true',
                   help='Transcribe the reference (faster-whisper, CPU) and fit on '
                        'its own words — like-for-like phonetics for the LTAS match')
    p.add_argument('--whisper-model', default='small',
                   help='faster-whisper model for --auto-text (default: small)')
    p.add_argument('--output', default=None)
    p.add_argument('--output-optimised', default=None)
    p.add_argument('--output-raw', help='Save the untouched XTTS waveform for listening')
    p.add_argument('--raw-clone', metavar='WAV',
                   help='Fit on this untouched XTTS render instead of generating one: the '
                        '--output-raw of an earlier run with the same {} block, text and '
                        'clone references. Two fits then work on the very same clone.')
    p.add_argument('--reference-interval', nargs=2, type=float, metavar=('BEGIN', 'END'),
                   help='Seconds of the reference the clone is compared with, for --text: '
                        'the stretch that says the same words (printed by an --auto-text run)')
    p.add_argument('--eq-limit', type=float, default=4., help='Absolute cumulative EQ gain limit per band, default 4 dB')
    p.add_argument('--device', default=None)
    p.add_argument('--max-ref-len', type=int, default=None,
                   help='Seconds of reference used for the SPEAKER EMBEDDING '
                        '(default: value from block, or 30; longer is not always better).')
    p.add_argument('--gpt-cond-len', type=int, default=None,
                   help='Override seconds of reference for GPT conditioning '
                        '(default: value from the {} block, or 30)')
    p.add_argument('--iterations', type=int, default=3,
                   help='Max closed-loop EQ/vol refinement passes (default: 3)')
    p.add_argument('--conv-threshold', type=float, default=0.5,
                   help='Stop when the largest EQ/vol change in a pass falls '
                        'below this many dB (default: 0.5)')
    p.add_argument('--target-dbfs', type=float, default=None,
                   help='Production level mode: fit the volume toward this absolute '
                        'RMS target (e.g. -20) instead of matching the reference. '
                        'Use it when the reference is quiet — matching a -33 dBFS '
                        'reference yields an unusable meditation level.')
    p.add_argument('--no-fit-dynamics', dest='fit_dynamics', action='store_false',
                   help='Do not fit comp / de-ess by measurement (crest factor and '
                        '5-8 kHz sibilance vs the reference)')
    p.add_argument('--fit-identity', action='store_true',
                   help='After the tone fit, search the remaining post-processing '
                        'settings for one that measurably raises ECAPA identity '
                        '(the tone fit optimises spectrum, never identity). Keeps a '
                        'candidate only if it beats the heuristic tolerance.')
    p.add_argument('--screen-audio', action='store_true',
                   help='Sensitivity screening BEFORE optimising (the Optimetrics '
                        'reflex): sweep each audio parameter alone over its full range '
                        'and report how much ECAPA identity moves, so you know which '
                        'knobs matter for this voice and which are measurably inert.')
    p.add_argument('--optimise-audio', default='none', choices=['none', 'nelder', 'de'],
                   help="Explicit trials of every free [] field, including off/on, "
                        "then joint refinement against ECAPA, then pruning of the "
                        "changes worth less than the heuristic tolerance. 'nelder' = "
                        "simplex, 'de' = global search plus simplex. Two unseen "
                        "sentences must confirm the candidate before adoption.")
    p.add_argument('--optimise-budget', type=int, default=400,
                   help='Strict maximum fit evaluations including all-field trials and '
                        'pruning (default: 400; independent validation is reported separately)')
    p.add_argument('--search-all-fields', action='store_true',
                   help='Let --optimise-audio also change speed, pan and limiter. Held '
                        'by default: ECAPA does not measure the pace of a meditation, '
                        'hears pan as a mere gain (it averages both channels), and the '
                        'limiter protects against clipping.')
    p.add_argument('--eq-weighting', default='a', choices=['a', 'voice', 'blend'],
                   help="Where EQ errors count. 'a' = equal-loudness (default), "
                        "'voice' = the reference's own spectrum, 'blend' = both. "
                        "On a low voice 'a' discounts the fundamental ~4.6x, so "
                        "'voice' or 'blend' track a bass speaker more closely.")
    p.add_argument('--save-preset', action='store_true',
                   help='Store the resulting {} / [] pair in voice_presets.json, '
                        'reusable from the [Gen] editor')
    p.add_argument('--preset-name', default=None, metavar='NAME',
                   help='Name to store it under (default: from the reference file)')
    p.add_argument('--no-identity-check', action='store_true',
                   help='Skip the extra generation that measures identity on UNSEEN '
                        'text (the figure that matches real usage)')
    p.set_defaults(fit_dynamics=True)
    args = p.parse_args()

    # Separate trailing lang from refs
    refs = list(args.voice_refs)
    asked = None
    if refs and refs[-1].upper() in LANGS:
        asked = refs[-1].upper(); refs = refs[:-1]
    refs = args.clone_refs or list(dict.fromkeys([args.reference] + refs))
    for ref in [args.reference] + refs:
        if not os.path.isfile(ref):
            p.error(f'Reference not found: {ref}')
    speaker_wav = refs if len(refs) > 1 else refs[0]

    xtts = parse_xtts_block(args.xtts_block)
    if args.max_ref_len is not None:
        xtts['max_ref_len'] = args.max_ref_len
    if args.gpt_cond_len is not None:
        xtts['gpt_cond_len'] = args.gpt_cond_len
    xtts = validate(xtts)
    args.max_ref_len = xtts['max_ref_len']
    if not np.isfinite(args.eq_limit) or args.eq_limit<=0:
        p.error('eq-limit must be finite and positive')
    if args.target_dbfs is not None and (not np.isfinite(args.target_dbfs) or args.target_dbfs >= 0):
        p.error('target-dbfs must be finite and negative')
    if args.optimise_audio != 'none' and args.optimise_budget < 64:
        p.error('optimise-budget must be at least 64 for an all-field search')
    held_fields = {}                  # fields the audio search must not change
    if args.optimise_audio != 'none':
        from audio_parameter_search import EDITORIAL_FIELDS, worst_case_budget
        if not args.search_all_fields:
            held_fields = dict(EDITORIAL_FIELDS)
        need = worst_case_budget(args.eq_limit, args.target_dbfs is not None, held_fields)
        if args.optimise_budget < need:
            p.error(f'optimise-budget must be at least {need} for this search '
                    '(every free field tried, then pruned)')
    audio = parse_audio_block(args.audio_block)
    lang, mismatch = resolve_language(asked, args.audio_block)
    if mismatch:
        p.error(mismatch)
    audio['lang'] = lang
    if args.raw_clone and not os.path.isfile(args.raw_clone):
        p.error(f'Raw clone not found: {args.raw_clone}')
    if args.reference_interval is not None:
        if args.auto_text:
            p.error('--reference-interval comes with --text: --auto-text finds its own')
        begin, finish = args.reference_interval
        if not (np.isfinite(begin) and np.isfinite(finish) and 0 <= begin < finish):
            p.error('--reference-interval needs 0 <= BEGIN < END')

    reference_interval = None
    if args.auto_text:
        # Fit on the reference's own words: same phonemes on both sides of the
        # LTAS comparison, so the fit corrects the voice, not the text.
        print(f"[*] Transcribing reference with faster-whisper '{args.whisper_model}' (cpu)...")
        from faster_whisper import WhisperModel
        wm = WhisperModel(args.whisper_model, device='cpu', compute_type='int8')
        segs, _info = wm.transcribe(args.reference, language=lang.lower().split('-')[0],
                                    beam_size=5, vad_filter=True, word_timestamps=True)
        try:
            text, begin, finish = transcript_excerpt(segs)
            reference_interval=(begin,finish)
        except ValueError as exc:
            p.error(str(exc))
        print(f'[OK] Paired reference interval: {begin:.2f}-{finish:.2f}s; text: {text[:80]!r}')
    elif args.text_file and os.path.exists(args.text_file):
        with open(args.text_file, encoding='utf-8') as fh:
            text = clean_text(fh.read(), lang)
    else:
        text = clean_text(args.text, lang)
    if args.reference_interval is not None:
        reference_interval = tuple(args.reference_interval)
    # What the fit compares, in full: a second run can reuse it with --raw-clone,
    # --text and --reference-interval, and work on exactly the same material.
    print('[*] Fit input: ' + json.dumps(
        {'text': text, 'interval': list(reference_interval) if reference_interval else None},
        ensure_ascii=False))

    hold_text = None
    selection_texts = []
    if args.optimise_audio != 'none':
        if args.holdout_text and clean_text(args.holdout_text, lang).casefold() == text.casefold():
            p.error("holdout-text must differ from the fitting sentence")
        try:
            selection_texts = audio_holdouts(lang, text, args.holdout_text)
        except ValueError as exc:
            p.error(str(exc))
        hold_text = selection_texts
    seed = xtts.get('seed')

    print("=" * 64)
    print("  XTTS Voice Comparator -- closed-loop EQ/vol fit (LTAS, least squares)")
    print("=" * 64)
    print(f"  Reference : {os.path.basename(args.reference)}")
    print(f"  Language  : {lang}")
    print(f"  Text      : {text[:58]}{'...' if len(text) > 58 else ''}")
    print(f"  Seed      : {seed} (frozen -- set it with the Validator)")
    print("=" * 64)

    gen_path = os.path.join(SCRIPT_DIR, 'guided_meditation_generator_v23.py')
    with tempfile.TemporaryDirectory(prefix='xtts_work_') as tmpdir:
        import shutil
        # XTTS is needed to generate the clone, and later for the unseen-text
        # identity check and the audio optimisation's hold-out clone. A fit on a
        # reused clone without either needs no model at all.
        need_model = (not args.raw_clone or not args.no_identity_check
                      or args.optimise_audio != 'none')
        if need_model:
            # Load XTTS once, via the low-level path that actually honours the
            # cloning knobs (tts_to_file silently overrides gpt_cond_len/chunk/
            # max_ref_len/sound_norm_refs with config defaults 12/4/10/False).
            import torch
            import xtts_clone as XC
            device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"\n[*] Loading XTTS on {device}...")
            tts, model = XC.load_xtts(device)
            print("[OK] XTTS ready")

            # Precompute conditioning latents ONCE (independent of seed) with the
            # real cloning parameters — where max_ref_len/gpt_cond_len finally bite.
            gpt_cl  = int(args.gpt_cond_len if args.gpt_cond_len is not None
                          else xtts.get('gpt_cond_len', 30))
            gpt_ccl = int(xtts.get('gpt_cond_chunk_len', 4))   # same default as the generator
            snorm   = bool(xtts.get('sound_norm_refs', 0))
            print(f"[*] Conditioning: gpt_cond_len={gpt_cl}s  chunk={gpt_ccl}s  "
                  f"max_ref_len={args.max_ref_len}s  sound_norm_refs={snorm}")
            latents = XC.compute_latents(model, speaker_wav, gpt_cond_len=gpt_cl,
                                         gpt_cond_chunk_len=gpt_ccl,
                                         max_ref_len=args.max_ref_len, sound_norm_refs=snorm)

        import ltas_match as L

        raw = os.path.join(tmpdir, 'clone_raw.wav')
        if args.raw_clone:
            # The same clone as an earlier fit: only the fit differs between them.
            shutil.copyfile(args.raw_clone, raw)
            print(f"\n[*] Clone reused from {os.path.basename(args.raw_clone)}"
                  f"{'' if need_model else ' (XTTS not loaded)'}")
        else:
            # ── Generate the clone ONCE (the {} block is frozen, seed included) ───
            print(f"\n[*] Generating clone (seed {seed})...")
            XC.generate(model, text, lang, latents, raw,
                        temperature=xtts.get('temp', 0.65),
                        length_penalty=xtts.get('len_pen', 1.0),
                        repetition_penalty=xtts.get('rep_pen', 5.0),
                        top_k=int(xtts.get('top_k', 50)),
                        top_p=xtts.get('top_p', 0.85),
                        num_beams=int(xtts.get('num_beams', 1)),
                        do_sample=bool(xtts.get('do_sample', 1)),
                        speed=1.0, seed=seed)
            print("[OK] Clone generated")

        ref_y, ref_sr  = load_mono(args.reference)
        if reference_interval is not None:
            begin,finish=reference_interval
            ref_y=ref_y[round(begin*ref_sr):min(len(ref_y),round(finish*ref_sr))]
            if not len(ref_y):
                raise ValueError('Timestamped reference excerpt is empty')
        import soundfile as sf
        render_sr=int(sf.info(raw).samplerate)
        print(f'[*] EQ response sample rate: {render_sr} Hz; analysis sample rate: {ref_sr} Hz')
        f_grid, ref_db = L.compute_ltas(ref_y, ref_sr)
        ref_shape      = ref_db - ref_db.mean()

        # ── Closed-loop EQ/vol fit vs the reference (multi-pass) ──────────────────
        # Each pass renders the full chain (incl. hp/lp/NR/comp/limiter), measures the
        # rendered clone's LTAS against the reference, and adds the least-squares EQ
        # correction. Iterating closes the loop on the post-chain colouring, not just
        # the raw clone. Converges in 1–3 passes.
        print(f"\n{'-'*64}\n  Closed-loop EQ/vol fit (LTAS, least squares)\n{'-'*64}")
        cand = os.path.join(tmpdir, 'cand.wav')
        def render_tone(block):
            apply_post(raw,cand,block,xtts,lang,gen_path)
            return load_mono(cand,target_sr=ref_sr)[0]
        opt,tone_history=fit_rendered_tone(ref_y,ref_sr,render_sr,audio,render_tone,
                       iterations=args.iterations,eq_limit=args.eq_limit,weighting=args.eq_weighting,
                       target_dbfs=args.target_dbfs,convergence=args.conv_threshold)

        # ── Measured fit of de-ess and compression ────────────────────────────────
        # The tail of the [] block used to stay at zero because nothing measured it.
        # Sibilance (5-8 kHz vs 300-3 kHz) and crest factor ARE measurable against
        # the reference, so they get the same treatment as the EQ: propose a value,
        # render, re-measure, and KEEP IT ONLY IF THE GAP SHRINKS. The audio
        # search below (--optimise-audio) also tries these settings, NR, reverb
        # and the gate.
        left_at_zero = []     # read by the identity search, also without this fit
        if args.fit_dynamics:
            print(f"\n{'-'*64}\n  Measured de-ess / compression fit\n{'-'*64}")
            ref_sib, ref_crest = L.sibilance_db(ref_y, ref_sr), L.crest_db(ref_y)
            apply_post(raw, cand, opt, xtts, lang, gen_path)
            cy, _ = load_mono(cand, target_sr=ref_sr)
            sib_gap = L.sibilance_db(cy, ref_sr) - ref_sib
            crest_gap = L.crest_db(cy) - ref_crest
            print(f"  reference: sibilance {ref_sib:+.1f} dB   crest {ref_crest:.1f} dB")
            print(f"  clone gap: sibilance {sib_gap:+.1f} dB   crest {crest_gap:+.1f} dB")

            for name, gap, thr, scale, cap, what in (
                    ('de-ess', sib_gap, 1.5, 0.18, 0.5, 'sibilance'),
                    ('comp', crest_gap, 2.0, 0.12, 0.6, 'dynamics')):
                if gap <= thr:
                    if gap < 0:
                        # Expected for synthetic speech: XTTS output is smoother than
                        # a real recording. A de-esser/compressor can only REDUCE, so
                        # a negative gap is not correctable here — 0 is the right value.
                        print(f"  {name}: clone has LESS {what} than the reference "
                              f"({gap:+.1f} dB) — a {name} can only reduce it, nothing "
                              f"to increase; keeping {opt.get(name, 0):g}")
                    else:
                        print(f"  {name}: gap {gap:+.1f} dB within tolerance ({thr}); "
                              f"keeping {opt.get(name, 0):g}")
                    left_at_zero.append(name)
                    continue
                trial = dict(opt)
                trial[name] = round(min(cap, (gap - thr) * scale), 2)
                apply_post(raw, cand, trial, xtts, lang, gen_path)
                ty, _ = load_mono(cand, target_sr=ref_sr)
                new_gap = ((L.sibilance_db(ty, ref_sr) - ref_sib) if name == 'de-ess'
                           else (L.crest_db(ty) - ref_crest))
                if abs(new_gap) < abs(gap) - 0.2:
                    opt[name] = trial[name]
                    print(f"  {name} -> {trial[name]}  (gap {gap:+.1f} -> {new_gap:+.1f} dB)")
                else:
                    print(f"  {name} {trial[name]} did not help ({gap:+.1f} -> {new_gap:+.1f}) "
                          f"-> kept {opt.get(name, 0):g}")

        # ── Identity-driven post-processing search ────────────────────────────────
        # Everything above optimises TONE (LTAS residual). This stage asks a
        # different question, the one that actually matters: can post-processing
        # raise the ECAPA IDENTITY of the clone? Nobody had ever measured it — the
        # remaining [] fields stayed at 0 because no criterion ever asked them to
        # move. Each candidate is rendered and scored; only a gain above the
        # heuristic tolerance is kept.
        if args.fit_identity:
            print(f"\n{'-'*64}\n  Identity-driven post-processing search (ECAPA)\n{'-'*64}")
            try:
                from speaker_identity import SpeakerEncoder
                enc = _encoder(args.device)
                ref_emb = enc.embed(args.reference)

                def ident_of(block):
                    apply_post(raw, cand, block, xtts, lang, gen_path)
                    return float(enc.cosine(ref_emb, enc.embed(cand))), cand

                base_id, _ = ident_of(opt)
                # Noise floor: identity measured on each half of the same clone.
                # Two halves of one render are the same voice, so their spread is
                # the measurement variability for this clip.
                cy_all, csr = load_mono(cand, target_sr=16000)   # ECAPA rate
                h = len(cy_all) // 2
                halves = [float(enc.cosine(ref_emb, enc.embed(cy_all[:h], sr=csr))),
                          float(enc.cosine(ref_emb, enc.embed(cy_all[h:], sr=csr)))]
                noise = max(0.005, abs(halves[0] - halves[1]) / 2.0)
                print(f"  baseline (tone-fitted) identity {base_id:.4f}   "
                      f"heuristic tolerance ±{noise:.4f}")

                trials = [('no post-processing at all',
                           dict(opt, vol=0, eq_low=0, eq_mid=0, eq_high=0, hp=0, lp=0)),
                          ('EQ off (keep vol/filters)', dict(opt, eq_low=0, eq_mid=0, eq_high=0)),
                          ('brighter (eq_high +2)', dict(opt, eq_high=opt['eq_high'] + 2)),
                          ('darker (eq_high -2)',   dict(opt, eq_high=opt['eq_high'] - 2)),
                          *([('low-pass off', dict(opt, lp=0))] if opt.get('lp') else []),
                          ('narrow band (lp 8k)',   dict(opt, lp=8000)),
                          ('NR 0.2',                dict(opt, NR=0.2)),
                          ('comp 0.3',              dict(opt, comp=0.3)),
                          ('de-ess 0.3',            {**opt, 'de-ess': 0.3})]
                results = [('baseline', opt, base_id)]
                for label, blk in trials:
                    blk=dict(blk,**{key:float(np.clip(blk.get(key,0),-args.eq_limit,args.eq_limit)) for key in ('eq_low','eq_mid','eq_high')})
                    try:
                        v, _ = ident_of(blk)
                    except Exception as e:
                        print(f"  {label:26s} failed ({e})"); continue
                    results.append((label, blk, v))
                    d = v - base_id
                    flag = '  <-- above noise' if d > noise else ''
                    print(f"  {label:26s} identity {v:.4f}  ({d:+.4f}){flag}")

                lbl, blk, best_id = max(results, key=lambda r: r[2])
                if best_id > base_id + noise:
                    print(f"\n  ADOPTED '{lbl}': identity {base_id:.4f} -> {best_id:.4f} "
                          f"(+{best_id - base_id:.4f}, above the {noise:.4f} heuristic tolerance)")
                    opt = dict(blk)
                else:
                    print(f"\n  No post-processing setting raises identity beyond the noise "
                          f"floor (best {best_id:.4f} vs baseline {base_id:.4f}).")
                    print(f"  Conclusion for this voice: post-processing shapes TONE, not "
                          f"IDENTITY — the zero fields are zero because nothing they can do")
                    print(f"  makes the clone measurably more like the person.")
                apply_post(raw, cand, opt, xtts, lang, gen_path)   # restore chosen block
            except Exception as e:
                print(f"  [!] Identity search unavailable ({e})")

            if left_at_zero and not args.screen_audio:
                print(f"  ({', '.join(left_at_zero)} left at 0 because the measured gap "
                      f"points the wrong way for these tools.")
                print(f"   To find out whether ANY audio parameter can move identity for "
                      f"this voice, re-run with --screen-audio.)")

        _measured_noise = [None]      # filled by the screening, used by the gate
        _adopted = [False]            # True when the identity search replaced the block

        # ── Sensitivity screening (which knobs matter at all?) ────────────────────
        # The Optimetrics reflex: screen BEFORE optimising. One factor at a time
        # across its full musical range, measuring how much ECAPA identity actually
        # moves. This answers, per parameter and with numbers, the question the tool
        # could never answer before: does this knob do anything for THIS voice?
        if args.screen_audio:
            print(f"\n{'-'*64}\n  Sensitivity screening (one factor at a time, ECAPA identity)\n{'-'*64}")
            try:
                from speaker_identity import SpeakerEncoder
                enc_s = _encoder(args.device)
                ref_s = enc_s.embed(args.reference)

                def ident_blk(blk):
                    fit_audio_level(raw, cand, blk, xtts, lang, gen_path, args.target_dbfs)
                    return float(enc_s.cosine(ref_s, enc_s.embed(cand)))

                base_s = ident_blk(opt)
                # Heuristic sensitivity to the two different halves of the clip.
                noise_s = _halves_tolerance(enc_s, ref_s, cand)
                print(f"  baseline identity {base_s:.4f}   heuristic tolerance +/-{noise_s:.4f}\n")
                _measured_noise[0] = noise_s      # reused by the transfer gate below

                from audio_parameter_search import AudioSpace
                GRID = AudioSpace(opt, args.eq_limit,
                                  freeze_vol=args.target_dbfs is not None).grid().items()
                rows = []
                for axis, values in GRID:
                    ids = []
                    for v in values:
                        blk = dict(opt); blk[axis] = v
                        try:
                            ids.append((v, ident_blk(blk)))
                        except Exception as exc:
                            print(f'  [!] {axis}={v:g}: not scored ({exc})')
                    if not ids:
                        continue
                    swing = max(i for _, i in ids) - min(i for _, i in ids)
                    bestv, besti = max(ids, key=lambda t: t[1])
                    rows.append((axis, swing, bestv, besti, besti - base_s))
                rows.sort(key=lambda r: -r[1])

                print(f"  {'axis':>8}{'swing':>9}{'best value':>12}{'identity':>10}{'vs base':>9}")
                for axis, swing, bestv, besti, delta in rows:
                    verdict = ('MATTERS' if swing > 3 * noise_s else
                               'marginal' if swing > noise_s else 'below tolerance')
                    shown = 'off' if axis in ('hp', 'lp', 'noise_gate') and not bestv else str(bestv)
                    print(f"  {axis:>8}{swing:>9.4f}{shown:>12}{besti:>10.4f}"
                          f"{delta:>+9.4f}   {verdict}")
                live = [r[0] for r in rows if r[1] > 3 * noise_s]
                dead = [r[0] for r in rows if r[1] <= noise_s]
                print(f"\n  Knobs that move identity for this voice: "
                      f"{', '.join(live) if live else 'NONE'}")
                if dead:
                    print(f"  Effects below this ECAPA tolerance: {', '.join(dead)}")
                print("  (diagnostic only; --optimise-audio independently tries every "
                      "free field before the joint search)")
            except Exception as e:
                print(f"  [!] Screening unavailable ({e})")

        # All-field trials are part of the search, including effects initially
        # off. The result is pruned, then must transfer to TWO independent
        # sentences. A failure leaves the tone fit in place: the search is a
        # refinement, and the tone-fitted block above is a complete result.
        audio_search_report = None
        audio_validation_renders = []
        if args.optimise_audio != 'none':
            print(f"\n{'-'*64}\n  Audio-block search ({args.optimise_audio}) — "
                  f"target: ECAPA identity\n{'-'*64}")
            tone_block = dict(opt)
            try:
                from audio_parameter_search import search_audio, AUDIO_KEYS, DEFAULTS
                enc = _encoder(args.device)
                ref_emb = enc.embed(args.reference)

                def evaluate_audio(block):
                    rendered = fit_audio_level(raw, cand, block, xtts, lang, gen_path,
                                               args.target_dbfs)
                    return float(enc.cosine(ref_emb, enc.embed(cand))), rendered

                # Pruning tolerance: the screening's when it ran, otherwise
                # measured the same way on the tone-fitted render.
                tolerance = _measured_noise[0]
                if tolerance is None:
                    fit_audio_level(raw, cand, opt, xtts, lang, gen_path, args.target_dbfs)
                    tolerance = _halves_tolerance(enc, ref_emb, cand)
                print(f"  pruning tolerance {tolerance:.4f} (identity spread between "
                      f"the two halves of the tone-fitted render)")
                best_blk, audio_search_report = search_audio(
                    opt, evaluate_audio, method=args.optimise_audio,
                    budget=args.optimise_budget, eq_limit=args.eq_limit,
                    freeze_vol=args.target_dbfs is not None, held=held_fields,
                    tolerance=tolerance,
                    progress=lambda message: print('  ' + message, flush=True))
                base_blk = audio_search_report['baseline_block']
                base_id = audio_search_report['baseline_score']
                best_id = audio_search_report['best_score']
                pruning = audio_search_report['pruning']
                print(f"  identity on fitting text: {base_id:.4f} -> "
                      f"{audio_search_report['search_best_score']:.4f} found, "
                      f"{best_id:.4f} after pruning")
                if pruning['steps']:
                    print(f"  pruned back to their start: "
                          f"{', '.join(pruning['reverted']) or 'nothing'}; "
                          f"changes kept: {', '.join(pruning['kept']) or 'none'}")
                for key, values in audio_search_report['tested_values'].items():
                    one = audio_search_report['one_factor_best'][key]
                    print(f"  {key:>10}: tested {', '.join(fmt(v) for v in values)}; "
                          f"best alone {fmt(one['value'])}; candidate {fmt(best_blk[key])}")
                for key, reason in audio_search_report['held'].items():
                    if key == 'vol' and args.target_dbfs is not None:
                        print(f"  {key:>10}: {reason}; calculated gain {fmt(best_blk[key])}")
                    else:
                        print(f"  {key:>10}: held at {fmt(best_blk[key])} ({reason})")
                failed = audio_search_report['failures']
                if failed:
                    print(f"  [!] {len(failed)} candidate(s) failed; details in audio-search JSON.")
                unscored = audio_search_report['fields_without_scored_alternative']
                if unscored:
                    print('  [!] No working alternative for: ' + ', '.join(unscored)
                          + '; check the failed effects in the report.')
                # Candidate and baseline are already rounded exactly as exported.
                # Do not round them again AFTER their held-out validation.
                gain_fit = best_id - base_id
                validation = []
                if gain_fit > 1e-9:
                    for number, selection_text in enumerate(selection_texts, 1):
                        print(f"\n  Audio hold-out {number}: {selection_text}")
                        raw2 = os.path.join(tmpdir, f'clone_raw_holdout_{number}.wav')
                        base2 = os.path.join(tmpdir, f'base_holdout_{number}.wav')
                        cand2 = os.path.join(tmpdir, f'cand_holdout_{number}.wav')
                        XC.generate(model, selection_text, lang, latents, raw2,
                                    temperature=xtts.get('temp', .65),
                                    length_penalty=xtts.get('len_pen', 1.),
                                    repetition_penalty=xtts.get('rep_pen', 5.),
                                    top_k=int(xtts.get('top_k', 50)),
                                    top_p=xtts.get('top_p', .85),
                                    num_beams=int(xtts.get('num_beams', 1)),
                                    do_sample=bool(xtts.get('do_sample', 1)), speed=1., seed=seed)
                        _quiet_post(raw2, base2, base_blk, xtts, lang, gen_path)
                        h_base = float(enc.cosine(ref_emb, enc.embed(base2)))
                        _quiet_post(raw2, cand2, best_blk, xtts, lang, gen_path)
                        h_best = float(enc.cosine(ref_emb, enc.embed(cand2)))
                        if not (np.isfinite(h_base) and np.isfinite(h_best)):
                            raise ValueError('Non-finite independent identity score')
                        validation.append({'text': selection_text, 'base': h_base,
                                           'candidate': h_best, 'gain': h_best - h_base})
                        audio_validation_renders.append((raw2, base2, cand2))
                        print(f"  held-out clone {number}: {h_base:.4f} -> {h_best:.4f} "
                              f"({h_best - h_base:+.4f})")
                gains = [row['gain'] for row in validation]
                # Screen is diagnostic: enabling it must not change adoption.
                # Use the measured pruning tolerance in both cases, keeping
                # the existing 0.02 minimum for a held-out mean gain.
                need = max(.02, 2. * tolerance)
                adopted = (len(gains) == 2 and min(gains) > 0
                           and float(np.mean(gains)) > max(need, .4 * gain_fit))
                if adopted:
                    opt = dict(best_blk)
                    _adopted[0] = True
                    print(f"\n  ADOPTED: positive gains on both sentences; mean "
                          f"{np.mean(gains):+.4f} exceeds tolerance {need:.4f}.")
                else:
                    # Keep the target-normalised baseline, without changing its
                    # effects merely to make the tail of [] nonzero.
                    opt = dict(base_blk)
                    if gain_fit > 1e-9:
                        why = 'insufficient independent gains'
                    elif audio_search_report['search_best_score'] > base_id + 1e-9:
                        why = 'no gain beyond the pruning tolerance'
                    else:
                        why = 'no fitting gain'
                    print(f"\n  REJECTED: {why}; keeping the tone-fitted effects.")
                audio_search_report.update(
                    adopted=adopted, validation=validation, adoption_tolerance=need,
                    adoption_tolerance_rule='max(0.02, 2 * pruning_tolerance)',
                    validation_evaluations=2 * len(validation), final_block=dict(opt),
                    fixed_fields={'voice_number': 'routing identifier', 'lang': lang})
                print("  Final decision per field:")
                held = audio_search_report['held']
                for key in AUDIO_KEYS:
                    if key == 'vol' and key in held and args.target_dbfs is not None:
                        status = 'calculated from RMS target'
                    elif key in held:
                        status = 'held'
                    elif opt[key] != base_blk[key]:
                        status = 'adopted'
                    elif adopted and key in pruning['reverted']:
                        status = 'pruned: its change was worth less than the tolerance'
                    else:
                        status = 'tested; retained'
                    start = audio.get(key, DEFAULTS[key])
                    print(f"  {key:>10}: input {fmt(start)} -> final {fmt(opt[key])} ({status})")
                apply_post(raw, cand, opt, xtts, lang, gen_path)
            except Exception as exc:
                # Whatever happened half-way (an adoption included), the result
                # is the tone fit, exactly as if no search had been asked for.
                opt = tone_block
                _adopted[0] = False
                audio_search_report = dict(audio_search_report or {},
                    adopted=False, error=str(exc), final_block=dict(opt))
                print(f"  [!] Audio search failed: {exc}")
                print("      The tone-fitted block is kept; details in the audio-search report.")

        print(f"  Final []: vol={opt['vol']:+g}  eq_low={opt['eq_low']:+.1f}  "
              f"eq_mid={opt['eq_mid']:+.1f}  eq_high={opt['eq_high']:+.1f}  "
              f"hp={opt['hp']:.0f}  lp={opt['lp']:.0f}")

        xtts_str, opt_audio_str = format_blocks(1, xtts, opt, lang)

        # GUI hook: this exact prefix updates the Audio [] field in xtts_studio.py
        print(f"  Next []    : {opt_audio_str}")

        # ── Render outputs ────────────────────────────────────────────────────────
        def default_output(suffix):
            base = os.path.splitext(os.path.basename(args.reference))[0] + suffix
            path = os.path.abspath(base + '.wav')
            index = 1
            while os.path.exists(path):
                path = os.path.abspath(f'{base}_{index}.wav')
                index += 1
            return path
        out_clone = args.output or default_output('_comparison_base')
        out_opt = args.output_optimised or default_output('_comparison_optimised')
        out_raw = args.output_raw or default_output('_comparison_raw')
        if len({os.path.realpath(path) for path in (out_clone,out_opt,out_raw)})!=3:
            raise ValueError('Raw, base and optimised outputs must have distinct paths')
        for path in (out_clone, out_opt, out_raw):
            if os.path.realpath(path) in {os.path.realpath(r) for r in [args.reference] + refs}:
                raise ValueError('Comparison output must differ from voice references')
            if args.raw_clone and os.path.realpath(path) == os.path.realpath(args.raw_clone):
                raise ValueError('Comparison output must differ from --raw-clone')
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        shutil.copyfile(raw,out_raw)
        print(f"\n[*] Rendering base clone (original [])...")
        apply_post(raw, out_clone, audio, xtts, lang, gen_path)
        print(f"[*] Rendering optimised clone (fitted [])...")
        apply_post(raw, out_opt, opt, xtts, lang, gen_path)

        if audio_search_report is not None:
            report_path = os.path.splitext(out_opt)[0] + '_audio_search.json'
            saved = []
            for number, paths in enumerate(audio_validation_renders, 1):
                stem = os.path.splitext(out_opt)[0] + f'_audio_holdout{number}'
                outputs = [stem + suffix for suffix in ('_raw.wav', '_base.wav', '_candidate.wav')]
                for source, destination in zip(paths, outputs):
                    shutil.copyfile(source, destination)
                saved.append(dict(zip(('raw', 'base', 'candidate'), outputs)))
            audio_search_report['validation_renders'] = saved
            audio_search_report['fit_text'] = text
            audio_search_report['input_audio_block'] = args.audio_block
            audio_search_report['reference'] = os.path.abspath(args.reference)
            audio_search_report['xtts_block'] = xtts_str
            audio_search_report['final_audio_block'] = opt_audio_str
            with open(report_path, 'w', encoding='utf-8') as handle:
                json.dump(audio_search_report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            print(f'[*] Audio-search report: {report_path}')

        by, _ = load_mono(out_clone, target_sr=ref_sr)
        _, bd = L.compute_ltas(by, ref_sr)
        oy, _ = load_mono(out_opt, target_sr=ref_sr)
        _, od = L.compute_ltas(oy, ref_sr)
        res_base = float(np.sqrt(np.mean((ref_shape - (bd - bd.mean())) ** 2)))
        res_opt  = float(np.sqrt(np.mean((ref_shape - (od - od.mean())) ** 2)))
        print(f"  LTAS residual vs reference: base {res_base:.2f} dB -> optimised {res_opt:.2f} dB")

        # ── Side-by-side diagnostic ──────────────────────────────────────────────
        # A single residual says how far the clone is, not IN WHAT. This table
        # shows where the difference actually sits, so a surprising fitted value
        # can be traced to the measurement that produced it.
        try:
            from scipy.signal import welch

            def _hnr(y, sr):
                """Harmonics-to-noise ratio over voiced frames — the husky/veiled
                component of a voice. A neural vocoder reconstructs the harmonic
                structure well and smooths the aperiodic part, so a breathy speaker
                can lose exactly what makes her recognisable. Returns nan if Praat
                is unavailable."""
                try:
                    import parselmouth
                    snd = parselmouth.Sound(np.asarray(y[:sr * 60], dtype='float64'), sr)
                    v = snd.to_harmonicity().values
                    v = v[v > -200]
                    v = v[v > 0.0]
                    return float(np.median(v)) if len(v) else float('nan')
                except Exception:
                    return float('nan')

            def _probe(y, sr):
                f, psd = welch(y, sr, nperseg=1024)
                tot = float(np.sum(psd)) + 1e-12
                rms = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-12)
                pk = float(np.max(np.abs(y)) + 1e-12)
                band = lambda lo, hi: float(np.sum(psd[(f >= lo) & (f < hi)]) / tot * 100)
                return {
                    'RMS (dBFS)': rms,
                    'Centroid (Hz)': float(np.sum(f * psd) / tot),
                    'Crest (dB)': 20 * np.log10(pk) - rms,
                    'Low 0-300 Hz (%)': band(0, 300),
                    'Mid 0.3-3 kHz (%)': band(300, 3000),
                    'High 3-8 kHz (%)': band(3000, 8000),
                    'Sibilance (dB)': L.sibilance_db(y, sr),
                    'HNR (dB)': _hnr(y, sr),
                }

            ry, _ = load_mono(args.reference, target_sr=ref_sr)
            pr, pc = _probe(ry, ref_sr), _probe(oy, ref_sr)
            print(f"\n  {'':<20}{'reference':>11}{'clone':>10}{'delta':>10}")
            for k in pr:
                if pr[k] != pr[k] or pc[k] != pc[k]:        # nan: Praat missing
                    continue
                d = pc[k] - pr[k]
                flag = ''
                if k == 'HNR (dB)' and d > 3.0:
                    flag = '   <- clone is SMOOTHER: a husky voice loses its grain here'
                print(f"  {k:<20}{pr[k]:>11.1f}{pc[k]:>10.1f}{d:>+10.1f}{flag}")
            _which = ("identity-optimised" if _adopted[0] else "tone-fitted")
            print(f"  (clone measured after the {_which} [] -- delta is what remains)")
        except Exception as _e:
            print(f"  [!] Comparison table unavailable ({_e})")

        # ── Identity on UNSEEN text — the figure that matches real usage ──────────
        # With --auto-text the fit runs on the reference's OWN words, and identity
        # measured there is systematically optimistic: the clone repeats sentences
        # the speaker actually said, with matching prosody. Measured on new text —
        # i.e. on the meditation you will actually generate — it drops noticeably
        # (0.80 vs 0.69 on one measured voice). Report both so the number you read
        # is the one that applies.
        _fit_id = _unseen_id = None
        if not args.no_identity_check:
            try:
                from speaker_identity import SpeakerEncoder
                # ECAPA on CPU here: XTTS already fills a 4 GB card, and this is a
                # single embedding on a few seconds of audio — about a second of
                # CPU time, against an OOM that skipped the check entirely.
                _enc = _encoder('cpu')
                _ref = _enc.embed(args.reference)
                _fit_id = float(_enc.cosine(_ref, _enc.embed(out_opt)))
                try:
                    import torch as _t
                    _t.cuda.empty_cache()          # the 4 GB card OOMs here otherwise
                except Exception:
                    pass
                UNSEEN = unseen_sentence(lang, text, hold_text, args.unseen_text)
                _raw2 = os.path.join(tmpdir, 'clone_unseen_raw.wav')
                _out2 = os.path.join(tmpdir, 'clone_unseen.wav')
                XC.generate(model, UNSEEN, lang, latents, _raw2,
                            temperature=xtts.get('temp', 0.65),
                            length_penalty=xtts.get('len_pen', 1.0),
                            repetition_penalty=xtts.get('rep_pen', 5.0),
                            top_k=int(xtts.get('top_k', 50)),
                            top_p=xtts.get('top_p', 0.85),
                            num_beams=int(xtts.get('num_beams', 1)),
                            do_sample=bool(xtts.get('do_sample', 1)),
                            speed=1.0, seed=seed)
                apply_post(_raw2, _out2, opt, xtts, lang, gen_path)
                _unseen_id = float(_enc.cosine(_ref, _enc.embed(_out2)))
                print(f"  Identity: {_fit_id:.4f} on the fitted sentence, "
                      f"{_unseen_id:.4f} on UNSEEN text ({_unseen_id - _fit_id:+.4f})")
                if _fit_id - _unseen_id > 0.05:
                    print("  ^ Identity drops on this unseen sentence. Compare several unseen "
                          "renders by ear; one ECAPA score does not predict an entire meditation.")
            except Exception as _e:
                print(f"  [!] Unseen-text identity check skipped ({_e})")

        # Saved only now, once the identity figures exist. It used to run before
        # they were computed, so a preset never carried its scores.
        # A failed audio search leaves the tone fit, a complete block: it is
        # saved like the result of a run without the search.
        if getattr(args, 'save_preset', False) or getattr(args, 'preset_name', None):
            try:
                import voice_presets as VP
                _sc = {}
                for _k, _v in (('identity', _fit_id), ('identity_unseen', _unseen_id)):
                    if _v is not None:
                        _sc[_k] = _v
                nm = VP.save(args.preset_name or VP.name_from_reference(args.reference),
                             xtts_str, opt_audio_str,
                             reference=args.reference, source='comparator',
                             scores=_sc or None, references=refs)
                print(f"  Preset '{nm}' saved to {os.path.basename(VP.PRESET_DIR)}/")
            except Exception as e:
                print(f"  [!] Preset save failed: {e}")

        print(f"\n{'='*64}\n  RESULT\n{'='*64}")
        print(f"  {xtts_str}")
        print(f"  {opt_audio_str}")
        print(f"\n  Base clone      : {out_clone}")
        print(f"  Optimised clone : {out_opt}")
        if audio_search_report and audio_search_report.get('error'):
            print('  [!] The audio search failed: this [] is the tone fit, not a searched '
                  'block (see the audio-search report).')
        print("[OK] Done.")


if __name__ == '__main__':
    sys.exit(main() or 0)

#!/usr/bin/env python3
"""Compare reference caps while keeping GPT conditioning fixed.

max_ref_len truncates audio before BOTH XTTS conditioning paths. This experiment
combines one fixed GPT latent with each capped speaker embedding. Cosine scores
are descriptive; sentence dispersion is not a calibrated noise floor.
Use --keep for WAVs/JSON and --text-file for custom language-specific sentences.
"""
import argparse
import contextlib
import json
import math
import os
import tempfile
from xtts_settings import validate, parse_block, format_block
from test_truncation import load_texts, render_kwargs, conditioning_kwargs


def isolated_latents(model, reference, settings, values, reference_duration, clone):
    """Compute once per cap; returned tuples share the identical GPT latent."""
    options = conditioning_kwargs(settings)
    # This diagnostic explicitly replaces the speaker embedding per cap.
    options.pop('speaker_ref_len', None)
    full_cap = max(math.ceil(reference_duration), settings['gpt_cond_len'], max(values))
    full = clone.compute_latents(model, reference, **dict(options, max_ref_len=full_cap))
    result = {}
    for value in values:
        pair = full if value == full_cap else clone.compute_latents(
            model, reference, **dict(options, max_ref_len=value))
        result[value] = (full[0], pair[1])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('lang', nargs='?', default='FR')
    parser.add_argument('--values', type=int, nargs='+', default=[20, 30, 45, 60])
    parser.add_argument('--gpt-cond-len', type=int, default=45)
    parser.add_argument('--gpt-cond-chunk-len', type=int, default=4)
    parser.add_argument('--seed', type=int, default=180)
    parser.add_argument('--temp', type=float, default=.85)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=.85)
    parser.add_argument('--rep-pen', type=float, default=5.)
    parser.add_argument('--xtts-block')
    parser.add_argument('--text-file')
    parser.add_argument('--device')
    parser.add_argument('--keep')
    args = parser.parse_args()
    lang = args.lang.lower().replace('_', '-').split('-')[0]
    try:
        if args.text_file:
            texts = load_texts(args.text_file, lang, {})
        else:
            from probe_texts import probe_texts
            texts = probe_texts(lang)
        base = dict(seed=args.seed, temp=args.temp, top_k=args.top_k, top_p=args.top_p,
                    rep_pen=args.rep_pen, gpt_cond_len=args.gpt_cond_len,
                    gpt_cond_chunk_len=args.gpt_cond_chunk_len)
        if args.xtts_block:
            base.update(parse_block(args.xtts_block))
        settings = validate(base)
        if not args.values or min(args.values) <= 0:
            raise ValueError('Reference caps must be positive')
        if not os.path.isfile(args.reference):
            raise ValueError(f'Reference not found: {args.reference}')
    except ValueError as exc:
        parser.error(str(exc))
    import numpy as np
    import soundfile as sf
    import xtts_clone as clone
    from speaker_identity import SpeakerEncoder
    duration = sf.info(args.reference).duration
    if not math.isfinite(duration) or duration <= 0:
        parser.error('Reference audio is empty or invalid')
    values, seen = [], set()
    for value in dict.fromkeys(args.values):
        effective = min(value, duration)
        if effective not in seen:
            values.append(value)
            seen.add(effective)
        else:
            print(f'Skipping redundant cap {value}s (same {effective:.2f}s reference audio)')
    print(f'Fixed settings: {format_block(settings)}')
    print('GPT conditioning fixed from full reference; only speaker embeddings vary.')
    _, model = clone.load_xtts(args.device)
    encoder = SpeakerEncoder(device=args.device)
    reference = encoder.embed(args.reference)
    latents = isolated_latents(model, args.reference, settings, values, duration, clone)
    if args.keep:
        os.makedirs(args.keep, exist_ok=True)
    manager = contextlib.nullcontext(args.keep) if args.keep else tempfile.TemporaryDirectory(prefix='mrl_')
    with manager as directory:
        results = []
        for value in values:
            scores, files = [], []
            for i, text in enumerate(texts):
                path = os.path.join(directory, f'mrl{value}_{i}.wav')
                clone.generate(model, text, args.lang, latents[value], path, **render_kwargs(settings))
                scores.append(encoder.cosine(reference, encoder.embed(path)))
                files.append(path)
            row = dict(max_ref_len=value, effective_seconds=min(value, duration),
                       mean=float(np.mean(scores)), std=float(np.std(scores, ddof=1)) if len(scores)>1 else None,
                       scores=scores, files=files)
            results.append(row)
            print(f"cap {value}s (effective {row['effective_seconds']:.2f}s): cosine {row['mean']:.4f}; "
                  f"per-sentence scores {', '.join(f'{s:.4f}' for s in scores)}")
        baseline = next((r for r in results if r['max_ref_len']==30), results[0])
        best = max(results, key=lambda r:r['mean'])
        print(f"Highest observed mean: cap {best['max_ref_len']}s, delta {best['mean']-baseline['mean']:+.4f} "
              f"vs tested baseline {baseline['max_ref_len']}s.")
        print('Sentence dispersion is not a calibrated noise floor. Confirm on new text and by listening.')
        if len(results)<2:
            print('Only one effective cap: no comparison is possible with this reference.')
        if args.keep:
            with open(os.path.join(directory,'max_ref_len_report.json'),'w',encoding='utf-8') as report:
                json.dump(dict(settings=settings,gpt_fixed=True,texts=texts,results=results),report,
                          ensure_ascii=False,indent=2)
            print(f'WAVs and report: {directory}')


if __name__=='__main__':
    main()

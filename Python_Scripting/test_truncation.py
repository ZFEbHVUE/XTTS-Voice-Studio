#!/usr/bin/env python3
"""Probe possible missing sentence endings using ordered ASR word alignment.

ASR mismatches need listening confirmation. --xtts-block freezes all cloning
settings; only the selected temperature/seed axis changes. --audio-block also
runs the renderer's effects, speed and volume, including trims/fades from {}.
Use --keep to retain WAVs and a JSON report. Plain custom sentences: --text-file.
"""
import argparse
import contextlib
import json
import os
import tempfile
from xtts_settings import validate, parse_block, format_block
from text_alignment import ending_metrics

LONG_TEXTS = {
    'fr': ["Installe-toi confortablement, laisse tes épaules redescendre le long du dossier, et prends le temps de sentir l'air qui entre par le nez avant de ressortir lentement par la bouche.",
           "À chaque expiration, tu peux relâcher un peu plus la mâchoire, le front, la nuque, et laisser le poids de ton corps se déposer complètement dans le siège qui te porte."],
    'en': ["Settle in comfortably, let your shoulders come down along the back of the chair, and take the time to feel the air entering through your nose before it slowly leaves through your mouth.",
           "With every breath out you can release the jaw a little more, the forehead, the back of the neck, and let the weight of your body settle completely into the seat that carries you."]}


def diagnostic_settings(args):
    base = dict(seed=args.seed, temp=args.temp_fixed, top_k=args.top_k,
                top_p=args.top_p, rep_pen=args.rep_pen, gpt_cond_len=args.gpt_cond_len,
                gpt_cond_chunk_len=args.gpt_cond_chunk_len, max_ref_len=args.max_ref_len,
                trim_start=args.trim_start, trim_end=args.trim_end,
                fade_in=args.fade_in, fade_out=args.fade_out)
    if args.xtts_block:
        base.update(parse_block(args.xtts_block))
    return validate(base)


def render_kwargs(settings):
    return dict(temperature=settings['temp'], length_penalty=settings['len_pen'],
                repetition_penalty=settings['rep_pen'], top_k=settings['top_k'],
                top_p=settings['top_p'], seed=settings['seed'],
                num_beams=settings['num_beams'], do_sample=bool(settings['do_sample']),
                **({'typical_p': settings['typical_p']} if settings.get('typical_p', 1.) < 1 else {}))


def conditioning_kwargs(settings):
    out = {key: settings[key] for key in ('gpt_cond_len', 'gpt_cond_chunk_len',
                                        'max_ref_len', 'sound_norm_refs')}
    out.update({key: settings[key] for key in ('speaker_ref_len', 'ref_trim_db')
                if settings.get(key, 0)})
    return out


def load_texts(path, lang, defaults):
    if path:
        with open(path, encoding='utf-8-sig') as handle:
            texts = [line.strip() for line in handle if line.strip() and not line.lstrip().startswith('#')]
        if not texts:
            raise ValueError("Text file contains no sentences")
        return texts
    if lang not in defaults:
        raise ValueError(f"No built-in sentences for {lang}; supply --text-file in the selected language")
    return list(defaults[lang])


def apply_audio_block(src, dst, block, settings):
    from pydub import AudioSegment
    import guided_meditation_generator_v23 as renderer
    from xtts_settings import generator_params
    voice, cfg = renderer.parse_voice_config(block, use_last=False)
    if voice is None or cfg is None:
        raise ValueError("Invalid audio block")
    xp = generator_params(format_block(settings), renderer.DEFAULT_XTTS_PARAMS)
    # The generator's own chain: effects -> speed -> volume -> limiter.
    seg = renderer.render_voice_chain(AudioSegment.from_wav(src), cfg, xp)
    seg.export(dst, format='wav')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('lang', nargs='?', default='FR')
    parser.add_argument('--temps', type=float, nargs='+', default=[.45, .55, .65, .75])
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--seed', type=int, default=180)
    parser.add_argument('--top-k', type=int, default=50)
    parser.add_argument('--top-p', type=float, default=.85)
    parser.add_argument('--rep-pen', type=float, default=5.)
    parser.add_argument('--temp-fixed', type=float, default=.65)
    parser.add_argument('--gpt-cond-len', type=int, default=45)
    parser.add_argument('--gpt-cond-chunk-len', type=int, default=4)
    parser.add_argument('--max-ref-len', type=int, default=45)
    parser.add_argument('--trim-start', type=int, default=0)
    parser.add_argument('--trim-end', type=int, default=200)
    parser.add_argument('--fade-in', type=int, default=200)
    parser.add_argument('--fade-out', type=int, default=400)
    parser.add_argument('--xtts-block', help='Complete winning {} block (overrides fixed individual options)')
    parser.add_argument('--audio-block')
    parser.add_argument('--text-file')
    parser.add_argument('--whisper-model', default='small')
    parser.add_argument('--device')
    parser.add_argument('--keep')
    args = parser.parse_args()
    lang = args.lang.lower().replace('_', '-').split('-')[0]
    try:
        texts = load_texts(args.text_file, lang, LONG_TEXTS)
        base = diagnostic_settings(args)
        axis = 'seed' if args.seeds else 'temp'
        variants = [validate(dict(base, **{axis: value})) for value in (args.seeds or args.temps)]
        if not os.path.isfile(args.reference):
            raise ValueError(f"Reference not found: {args.reference}")
    except ValueError as exc:
        parser.error(str(exc))
    import soundfile as sf
    import xtts_clone as clone
    from faster_whisper import WhisperModel
    _, model = clone.load_xtts(args.device)
    latents = clone.compute_latents(model, args.reference, **conditioning_kwargs(base))
    asr = WhisperModel(args.whisper_model, device='cpu', compute_type='int8')
    if args.keep:
        os.makedirs(args.keep, exist_ok=True)
    manager = contextlib.nullcontext(args.keep) if args.keep else tempfile.TemporaryDirectory(prefix='trunc_')
    records = []
    print(f"Fixed block: {format_block(base)}; varying {axis}; text splitting enabled")
    print("ASR flags are suspected omissions; listen to the retained audio to confirm.")
    with manager as directory:
        for index, settings in enumerate(variants):
            value = settings[axis]
            for i, text in enumerate(texts):
                path = os.path.join(directory, f'{index}_{axis}{value}_{i}.wav')
                clone.generate(model, text, args.lang, latents, path, **render_kwargs(settings))
                if args.audio_block:
                    post = os.path.splitext(path)[0] + '_post.wav'
                    apply_audio_block(path, post, args.audio_block, settings)
                    path = post
                segments, _ = asr.transcribe(path, language=lang, beam_size=1)
                said = ' '.join(seg.text for seg in segments).strip()
                metrics = ending_metrics(said, text)
                info = sf.info(path)
                records.append(dict(axis=axis, value=value, sentence=i, expected=text,
                                    transcript=said, duration=info.duration, file=path, **metrics))
                print(f"{axis}={value}, sentence {i + 1}: ending matched {metrics['tail_kept']:.0%}, "
                      f"unmatched suffix {metrics['missing_suffix']} words" + (' [CHECK]' if metrics['suspect'] else ''))
        flagged = [r for r in records if r['suspect']]
        print(f"{len(flagged)}/{len(records)} renders have suspected omissions. "
              "No flag does not guarantee that every word was spoken.")
        if args.keep:
            with open(os.path.join(directory, 'truncation_report.json'), 'w', encoding='utf-8') as report:
                json.dump(dict(settings=base, audio_block=args.audio_block, samples=records), report,
                          ensure_ascii=False, indent=2)
            print(f"WAVs and report: {directory}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
engine_ab.py — the same voice rendered by two XTTS installs, then compared.

Moving from TTS 0.22.0 to the coqui-tts fork (0.28) changes the XTTS package,
transformers and where the model is downloaded from. Run this once in each
environment, then compare the two folders:

  conda activate xtts        # TTS 0.22.0
  python engine_ab.py render --preset elo_new --out ab_022
  conda activate xtts028     # coqui-tts 0.28
  python engine_ab.py render --preset elo_new --out ab_028
  python engine_ab.py compare ab_022 ab_028 [--identity]

render  : the preset's {} block and clone references (or --refs), on sentences
          of its language it was not tuned on (or --text-file), with the
          preset's seed and --seeds. Raw XTTS output, without the [] chain.
          manifest.json records the package versions and a fingerprint of
          every weight tensor of the loaded model.
compare : pairs the files by name and says whether the two installs load the
          same weights and render the same audio (to one step of 16-bit PCM).
          When they do not, --identity scores each side against the preset's
          reference (ECAPA), and the command for a blind listening test of the
          most different pair is printed.

The renders have no [] chain on purpose: what follows XTTS does not depend on
the package. Same weights and same audio mean the migration changes nothing
for this voice, seeds included.
"""
import argparse
import hashlib
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

MANIFEST = 'manifest.json'
LSB = 1.0 / 32768        # one step of 16-bit PCM, as the renders are written


def _slug(text, n=24):
    import unicodedata
    plain = ''.join(c for c in unicodedata.normalize('NFKD', text.lower())
                    if not unicodedata.combining(c))
    s = re.sub(r'[^a-z0-9]+', '_', plain).strip('_')
    return s[:n].strip('_') or 'text'


def preset_voice(name):
    """(block, references, reference, language) of a saved preset."""
    import voice_presets as VP
    entry = VP.get(name)
    if not entry:
        raise SystemExit(f"[ERR] No preset named '{name}' in {VP.PRESET_DIR}")
    refs = [r for r in entry.get('references') or [] if os.path.isfile(r)]
    if not refs and entry.get('reference') and os.path.isfile(entry['reference']):
        refs = [entry['reference']]
    m = re.match(r'\[\s*\d+\s*,\s*([A-Za-z]{2})', entry['audio'])
    return entry['xtts'], refs, entry.get('reference'), (m.group(1).upper() if m else None)


def weight_fingerprints(model):
    """md5 of each weight tensor, by name: equal values mean equal weights,
    whichever file or mirror they were loaded from."""
    out = {}
    for name, tensor in model.state_dict().items():
        data = tensor.detach().cpu().contiguous()
        out[name] = hashlib.md5(str(tuple(data.shape)).encode() + data.numpy().tobytes()).hexdigest()
    return out


def render(args, XC=None):
    import xtts_settings as XS
    import probe_texts as PT
    XC = XC or __import__('xtts_clone')
    block, refs, reference, lang = (preset_voice(args.preset) if args.preset
                                    else (args.xtts_block, [], None, None))
    if args.refs:
        refs = args.refs
    lang = (args.lang or lang or 'FR').upper()
    if not block:
        raise SystemExit('[ERR] Give --preset, or --xtts-block with --refs')
    if not refs:
        raise SystemExit('[ERR] No reference file found: give --refs')
    for r in refs:
        if not os.path.isfile(r):
            raise SystemExit(f'[ERR] Not found: {r}')
    p = XS.validate({**XS.DEFAULTS, **XS.parse_block(block)})
    if args.text_file:
        with open(args.text_file, encoding='utf-8') as fh:
            texts = [t.strip() for t in fh if t.strip()]
    else:
        texts = PT.final_check_texts(lang) + PT.holdout_texts(lang, 2)
    seeds = ([] if p['seed'] is None else [p['seed']]) + [s for s in args.seeds if s != p['seed']]
    os.makedirs(args.out, exist_ok=True)
    import torch
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tts, model = XC.load_xtts(device)
    latents = XC.compute_latents(model, refs, gpt_cond_len=p['gpt_cond_len'],
                                 gpt_cond_chunk_len=p['gpt_cond_chunk_len'],
                                 max_ref_len=p['max_ref_len'],
                                 sound_norm_refs=bool(p['sound_norm_refs']),
                                 speaker_ref_len=p['speaker_ref_len'],
                                 ref_trim_db=p['ref_trim_db'])
    files = []
    for i, text in enumerate(texts, 1):
        for seed in seeds:
            name = f'{i:02d}_{_slug(text)}_seed{seed}.wav'
            XC.generate(model, text, lang, latents, os.path.join(args.out, name),
                        temperature=p['temp'], length_penalty=p['len_pen'],
                        repetition_penalty=p['rep_pen'], top_k=int(p['top_k']),
                        top_p=p['top_p'], num_beams=int(p['num_beams']),
                        do_sample=bool(p['do_sample']), typical_p=p['typical_p'],
                        speed=1.0, seed=seed)
            files.append(dict(file=name, text=text, seed=seed))
            print(f'[OK] {name}')
    checkpoint = getattr(getattr(tts, 'synthesizer', None), 'tts_checkpoint', None)
    manifest = dict(engine=XC.engine_versions(), device=device,
                    checkpoint=str(checkpoint) if checkpoint else None,
                    preset=args.preset, xtts_block=block, references=list(refs),
                    reference=reference or refs[0], language=lang, files=files,
                    weights=weight_fingerprints(model))
    with open(os.path.join(args.out, MANIFEST), 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    print(f'[OK] {len(files)} renders and {MANIFEST} in {args.out} '
          f'({XC.engine_label(manifest["engine"])})')
    return manifest


def _read(path):
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(path, dtype='float64', always_2d=False)
    if getattr(y, 'ndim', 1) > 1:
        y = y.mean(axis=1)
    return np.asarray(y), sr


def compare_audio(a, b):
    """Sample-level comparison of two renders."""
    import numpy as np
    ya, sra = _read(a)
    yb, srb = _read(b)
    out = dict(seconds_a=len(ya) / sra, seconds_b=len(yb) / srb, same_rate=sra == srb)
    if sra == srb and len(ya) == len(yb):
        diff = float(np.max(np.abs(ya - yb))) if len(ya) else 0.0
        out.update(max_diff=diff, same=diff <= LSB * 1.5)
        if len(ya) and np.std(ya) > 0 and np.std(yb) > 0:
            out['correlation'] = float(np.corrcoef(ya, yb)[0, 1])
    else:
        out.update(max_diff=None, same=False)
    return out


def compare(args, encoder=None):
    with open(os.path.join(args.a, MANIFEST), encoding='utf-8') as fh:
        ma = json.load(fh)
    with open(os.path.join(args.b, MANIFEST), encoding='utf-8') as fh:
        mb = json.load(fh)
    import xtts_clone as XC
    print(f"A  {args.a}: {XC.engine_label(ma['engine'])} on {ma.get('device')}")
    print(f"B  {args.b}: {XC.engine_label(mb['engine'])} on {mb.get('device')}")
    for key in ('xtts_block', 'references', 'language'):
        if ma.get(key) != mb.get(key):
            print(f"[!] The two folders were rendered with different {key}: "
                  "the comparison mixes settings with installs.")
    wa, wb = ma.get('weights') or {}, mb.get('weights') or {}
    common = sorted(set(wa) & set(wb))
    differing = [k for k in common if wa[k] != wb[k]]
    only = sorted(set(wa) ^ set(wb))
    same_weights = bool(common) and not differing
    print(f"\nWeights: {len(common)} tensors in both, {len(common) - len(differing)} identical"
          + (f", {len(differing)} different (first: {differing[0]})" if differing else '')
          + (f", {len(only)} in one only (first: {only[0]})" if only else ''))
    names_b = {f['file'] for f in mb['files']}
    rows = []
    for f in ma['files']:
        if f['file'] not in names_b:
            print(f"[!] {f['file']} is missing from {args.b}")
            continue
        r = compare_audio(os.path.join(args.a, f['file']), os.path.join(args.b, f['file']))
        r['file'] = f['file']
        rows.append(r)
    print(f"\n{'file':44s} {'A s':>6s} {'B s':>6s}  {'max |A-B|':>9s}  same")
    for r in rows:
        diff = '-' if r['max_diff'] is None else f"{r['max_diff']:.2e}"
        print(f"{r['file'][:44]:44s} {r['seconds_a']:6.2f} {r['seconds_b']:6.2f}  {diff:>9s}  "
              f"{'yes' if r['same'] else 'NO'}")
    same_audio = bool(rows) and all(r['same'] for r in rows)
    verdict = ('same' if same_weights and same_audio else
               'weights' if not same_weights else 'audio')
    print()
    if verdict == 'same':
        print("Same weights, same audio: for this voice the migration changes nothing, "
              "seeds included. The presets can be used as they are.")
    elif verdict == 'weights':
        print("The two installs do not load the same XTTS weights: they download it from "
              "different places (TTS 0.22: coqui/XTTS-v2, coqui-tts 0.28: tts-hub/XTTS-v2). "
              "Compare the clones by ear before switching.")
    else:
        print("Same weights, different audio: generation differs between the installs. "
              "Compare the clones by ear before switching.")
    if verdict != 'same' and rows:
        worst = max(rows, key=lambda r: (r['max_diff'] is None, r['max_diff'] or 0))
        reference = ma.get('reference')
        print('\nBlind listening test of the most different pair:\n'
              f"  python {os.path.join(SCRIPT_DIR, 'listening_test.py')} "
              f"\"{os.path.join(args.a, worst['file'])}\" \"{os.path.join(args.b, worst['file'])}\""
              + (f" --reference \"{reference}\"" if reference else '') + ' --trials 12')
    if args.identity and verdict != 'same':
        identity(args, ma, rows, encoder)
    return dict(verdict=verdict, rows=rows, same_weights=same_weights)


def identity(args, manifest, rows, encoder=None):
    """ECAPA identity of each side against the preset reference."""
    import numpy as np
    reference = manifest.get('reference')
    if not reference or not os.path.isfile(reference):
        print('[!] --identity: the reference file is not available')
        return None
    if encoder is None:
        import speaker_identity as SI
        encoder = SI.SpeakerEncoder()
    ref = encoder.embed(*_read(reference))
    ref = ref / (np.linalg.norm(ref) + 1e-12)

    def score(path):
        e = encoder.embed(*_read(path))
        return float(np.dot(ref, e / (np.linalg.norm(e) + 1e-12)))
    a = np.array([score(os.path.join(args.a, r['file'])) for r in rows])
    b = np.array([score(os.path.join(args.b, r['file'])) for r in rows])
    d = b - a
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    print(f"\nIdentity vs {os.path.basename(reference)}: A {a.mean():.3f}, B {b.mean():.3f}, "
          f"B - A {d.mean():+.3f} (sd {sd:.3f} over {len(d)} pairs)")
    return dict(a=a.tolist(), b=b.tolist())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    r = sub.add_parser('render', help='render the A/B set with the XTTS of this environment')
    r.add_argument('--preset', help='voice preset name (its {} block and clone references)')
    r.add_argument('--xtts-block', help='a {} block, with --refs, instead of a preset')
    r.add_argument('--refs', nargs='+', help='reference file(s), instead of the preset\'s')
    r.add_argument('--lang', help="language (default: the preset's)")
    r.add_argument('--text-file', help='one sentence per line, instead of the built-in ones')
    r.add_argument('--seeds', nargs='*', type=int, default=[1, 2],
                   help="seeds besides the preset's own (default: 1 2)")
    r.add_argument('--out', required=True, help='folder for the renders and manifest.json')
    r.add_argument('--device', help='cuda or cpu (default: cuda when available)')
    c = sub.add_parser('compare', help='compare two folders made by render')
    c.add_argument('a')
    c.add_argument('b')
    c.add_argument('--identity', action='store_true',
                   help='when the audio differs, score both sides against the reference (ECAPA)')
    args = ap.parse_args(argv)
    if args.command == 'render':
        render(args)
    else:
        compare(args)


if __name__ == '__main__':
    main()

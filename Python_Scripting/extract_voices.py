#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Separation voix feminine / masculine avec gestion fine des silences
et de-reverberation optionnelle.

Regles :
  female_solo  -> F0 >= seuil, faible variance
  male_solo    -> F0 <  seuil, faible variance
  overlap      -> deux voix simultanees
  silence      -> pas de parole

Options --keep :
  female, male, overlap, all, female,male

Options --silence :
  auto  -> duree naturelle
  0     -> aucun silence
  N     -> exactement N secondes entre chaque segment garde

Options --dereverberate :
  none, noisereduce, wpe, deepfilter

Usage :
  python extract_voices.py input.mp3 output.wav --keep female --silence 0.5
"""

import numpy as np
import soundfile as sf
import librosa
import argparse
import os


def dereverberate(y, sr, method='none', device='cpu'):
    if method == 'none':
        return y
    elif method == 'noisereduce':
        try:
            import noisereduce as nr
        except ImportError:
            print("[!]  pip install noisereduce"); return y
        print("   [*]  noisereduce...")
        return nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.85).astype(np.float32)
    elif method == 'wpe':
        try:
            from nara_wpe.wpe import wpe
            from nara_wpe.utils import stft, istft
        except ImportError:
            print("[!]  pip install nara-wpe"); return y
        print("   [*]  WPE...")
        size, shift = 512, 128
        Y = stft(y, size=size, shift=shift).T[None, ...]
        Z = wpe(Y, taps=10, delay=3, iterations=3)[0]
        out = istft(Z.T, size=size, shift=shift)
        res = np.zeros_like(y); res[:min(len(out),len(y))] = out[:min(len(out),len(y))]
        return res.astype(np.float32)
    elif method == 'deepfilter':
        try:
            from df.enhance import enhance, init_df
        except ImportError:
            print("[!]  pip install deepfilternet"); return y
        use_gpu = device == 'cuda'
        print(f"   [*]  DeepFilterNet par chunks 30s ({'GPU' if use_gpu else 'CPU'})...")
        import torch
        if not use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        model, df_state, _ = init_df()
        t_sr = 48000
        y48 = librosa.resample(y, orig_sr=sr, target_sr=t_sr) if sr != t_sr else y.copy()
        cs = 30 * t_sr; ov = int(0.5 * t_sr)
        chunks = []; n_chunks = int(np.ceil(len(y48) / cs))
        for i in range(n_chunks):
            s = max(0, i*cs - ov); e = min(len(y48), (i+1)*cs + ov)
            c = y48[s:e]
            with torch.no_grad():
                # deepfilter manages device placement internally via get_device()
                # Always pass CPU tensor — deepfilter moves it to GPU itself
                t = torch.from_numpy(c[None]).float()
                enh = enhance(model, df_state, t).cpu().numpy()[0]
            ts = (i*cs - s) if i > 0 else 0
            te = len(enh) - (e - (i+1)*cs) if e > (i+1)*cs else len(enh)
            chunks.append(enh[ts:te])
            print(f"   [*]  {i+1}/{n_chunks}...", end='\r')
        print()
        out48 = np.concatenate(chunks)
        out = librosa.resample(out48, orig_sr=t_sr, target_sr=sr) if sr != t_sr else out48
        res = np.zeros_like(y); res[:min(len(out),len(y))] = out[:min(len(out),len(y))]
        return res.astype(np.float32)
    return y



def remove_music_demucs(input_file, demucs_model='htdemucs_ft', device='cpu'):
    """
    Use demucs to separate vocals from background music/noise.
    Returns (y_vocals, sr) — the clean vocal stem as a numpy array.
    Requires: pip install demucs
    """
    try:
        import demucs.separate
    except ImportError:
        print("[!] demucs not installed -> pip install demucs")
        return None, None

    import tempfile, shutil, glob, soundfile as sf

    tmp_dir = tempfile.mkdtemp(prefix='demucs_')
    try:
        print(f"   [*] demucs ({demucs_model}) separating sources...")
        demucs_args = [
            "--two-stems", "vocals",
            "-n", demucs_model,
            "--out", tmp_dir,
        ]
        if device == "cuda":
            demucs_args += ["--device", "cuda"]
        demucs_args.append(input_file)
        demucs.separate.main(demucs_args)

        # demucs outputs to: tmp_dir/<model>/<track_name>/vocals.wav
        pattern = os.path.join(tmp_dir, demucs_model, "**", "vocals.wav")
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            # fallback: search any vocals.wav
            matches = glob.glob(os.path.join(tmp_dir, "**", "vocals.wav"), recursive=True)
        if not matches:
            print("[!] demucs: vocals.wav not found in output")
            return None, None

        vocals_path = matches[0]
        y, sr = sf.read(vocals_path, dtype='float32', always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)   # stereo -> mono
        print(f"   [OK] demucs done -> {len(y)/sr:.1f}s vocals stem")
        return y, sr

    except Exception as e:
        print(f"   [!] demucs error: {e}")
        return None, None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def detect_segments(y, sr, min_silence=0.15, min_speech=0.2):
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop)
    rms_norm = rms / (np.max(rms) + 1e-10)
    threshold = max(0.02, np.median(rms_norm) * 0.4)
    is_speech = rms_norm > threshold
    segs = []; in_seg = False; t0 = 0.0
    for i, sp in enumerate(is_speech):
        if sp and not in_seg: t0 = float(times[i]); in_seg = True
        elif not sp and in_seg:
            if float(times[i]) - t0 >= min_speech: segs.append((t0, float(times[i])))
            in_seg = False
    if in_seg and float(times[-1]) - t0 >= min_speech: segs.append((t0, float(times[-1])))
    if not segs: return []
    merged = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s - pe < min_silence: merged[-1] = (ps, e)
        else: merged.append((s, e))
    return merged


def classify(y, sr, start, end, f0_thr=165, ov_range=80):
    seg = y[int(start*sr):int(end*sr)]
    if len(seg) < 2048: return 'silence', None
    f0, voiced, _ = librosa.pyin(seg, fmin=60, fmax=500, sr=sr, frame_length=2048, hop_length=512)
    f0v = f0[voiced & ~np.isnan(f0)]
    if len(f0v) < 3: return 'silence', None
    med = float(np.median(f0v))
    rng = float(np.percentile(f0v, 90) - np.percentile(f0v, 10))
    if rng > ov_range: return 'overlap', med
    return ('female_solo', med) if med >= f0_thr else ('male_solo', med)


def process(input_file, output_file, keep_set=None, silence_mode='auto',
            deverb_method='none', f0_thr=165, ov_range=80, min_dur=0.2,
            debug=False, remove_music=False, demucs_model='htdemucs_ft', device='cpu'):

    if keep_set is None: keep_set = {'female_solo'}

    print(f"[*]  {input_file}")
    print(f"   [*] Device: {device}")

    # ── Optional: demucs music removal ───────────────────────────────────
    if remove_music:
        y, sr = remove_music_demucs(input_file, demucs_model, device)
        if y is None:
            print("[!] Music removal failed - falling back to direct load")
            y, sr = librosa.load(input_file, sr=None, mono=True)
        else:
            # --remove-music alone: save clean vocals stem and exit.
            # No F0 filtering, no segment detection, no cuts.
            print(f"[OK] Music removed. Saving clean vocals directly (no F0 filtering).")
            sf.write(output_file, y, sr)
            print(f"[OK] {output_file}  ({len(y)/sr:.1f}s)")
            return
    else:
        y, sr = librosa.load(input_file, sr=None, mono=True)
    total_dur = len(y) / sr
    rm_str = "none"
    sil_str = 'duree naturelle' if silence_mode == 'auto' else \
              'aucun silence' if silence_mode == 0.0 else f'{silence_mode}s fixe'
    print(f"   Duree:{total_dur:.1f}s  Garder:{keep_set}  Silence:{sil_str}  Dereverb:{deverb_method}")

    if deverb_method != 'none':
        y = dereverberate(y, sr, deverb_method, device)
        # If keep is vocals_only, just save the cleaned file and exit — no F0 cuts.
        if 'vocals_only' in keep_set:
            print(f"[OK] Dereverberation done. Saving directly (no F0 filtering).")
            sf.write(output_file, y, sr)
            print(f"[OK] {output_file}  ({len(y)/sr:.1f}s)")
            return

    segs = detect_segments(y, sr)
    print(f"   {len(segs)} segments detectes")

    # Calculer le silence a inserer entre deux segments gardes
    if silence_mode == 'auto':
        sil_n = None  # variable selon les gaps
    elif silence_mode == 0.0:
        sil_n = 0
    else:
        sil_n = int(float(silence_mode) * sr)

    result = []
    last_kept_end = None

    for start, end in segs:
        dur = end - start
        kind, f0 = classify(y, sr, start, end, f0_thr, ov_range)

        if debug:
            print(f"   {start:6.1f}-{end:6.1f}s  {kind:12s}  {'[OK]' if kind in keep_set else '-> sil'}")

        if kind in keep_set and dur >= min_dur:
            # Inserer le silence AVANT ce segment
            if last_kept_end is not None:
                gap = start - last_kept_end
                if silence_mode == 'auto':
                    n = int(gap * sr)
                else:
                    n = sil_n
                if n > 0:
                    result.append(np.zeros(n, dtype=np.float32))

            # Garder le segment avec fade
            chunk = y[int(start*sr):int(end*sr)].copy()
            fade = min(int(0.02*sr), len(chunk)//4)
            if fade > 0:
                chunk[:fade] *= np.linspace(0, 1, fade)
                chunk[-fade:] *= np.linspace(1, 0, fade)
            result.append(chunk)
            last_kept_end = end

    if not result:
        print("[!]  Aucun segment garde !"); return

    # Silence final optionnel
    if last_kept_end is not None and last_kept_end < total_dur:
        if silence_mode == 'auto':
            result.append(np.zeros(int((total_dur - last_kept_end) * sr), dtype=np.float32))
        elif sil_n and sil_n > 0:
            result.append(np.zeros(sil_n, dtype=np.float32))

    final = np.concatenate(result)
    sf.write(output_file, final, sr)
    print(f"\n[OK]  {output_file}  ({len(final)/sr:.1f}s)")


KEEP_ALIASES = {
    'female': {'female_solo'}, 'male': {'male_solo'},
    'overlap': {'overlap'}, 'all': {'female_solo','male_solo','overlap'},
    'vocals only': {'vocals_only'}, 'vocals': {'vocals_only'},
}

def parse_keep(s):
    result = set()
    for p in s.split(','):
        p = p.strip().lower()
        if p in KEEP_ALIASES: result |= KEEP_ALIASES[p]
        elif p in ('female_solo','male_solo'): result.add(p)
        else: raise argparse.ArgumentTypeError(f"Inconnu: {p}")
    return result

def parse_silence(s):
    s = s.strip().lower()
    if s == 'auto': return 'auto'
    v = float(s)
    if v < 0: raise argparse.ArgumentTypeError("Valeur negative")
    return v

if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--keep", type=parse_keep, default={'female_solo'})
    p.add_argument("--silence", type=parse_silence, default='auto')
    p.add_argument("--dereverberate", choices=['none','noisereduce','wpe','deepfilter'], default='none')
    p.add_argument("--threshold", type=int, default=165)
    p.add_argument("--overlap-range", type=int, default=80)
    p.add_argument("--min-dur", type=float, default=0.2)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                   help="Device for demucs and deepfilter (default: cpu)")
    p.add_argument("--remove-music", action="store_true",
                   help="Use demucs to remove background music before voice separation")
    p.add_argument("--demucs-model", default="htdemucs_ft",
                   choices=["htdemucs", "htdemucs_ft", "mdx_extra"],
                   help="Demucs model to use for music removal (default: htdemucs_ft)")
    args = p.parse_args()
    process(args.input, args.output, args.keep, args.silence, args.dereverberate,
            args.threshold, args.overlap_range, args.min_dur, args.debug,
            args.remove_music, args.demucs_model, args.device)

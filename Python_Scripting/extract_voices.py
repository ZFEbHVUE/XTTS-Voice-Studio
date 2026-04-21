#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Female / male voice separation with fine silence handling
et de-reverberation optionnelle.

Regles :
  female_solo  -> F0 >= seuil, faible variance
  male_solo    -> F0 <  seuil, faible variance
  overlap      -> two simultaneous voices
  silence      -> pas de parole

Options --keep:
  female, male, overlap, all, female,male

Options --silence:
  auto  -> natural duration
  0     -> no silence
  N     -> exactement N secondes entre chaque segment garde

Options --dereverberate:
  none, noisereduce, wpe, deepfilter

Output format is auto-detected from the output file extension: .wav, .mp3, .flac, .ogg

Usage:
  python extract_voices.py input.mp3 output.wav --keep female --silence 0.5
  python extract_voices.py input.mp3 output.mp3 --keep female --mp3-bitrate 256 --mp3-mode cbr
  python extract_voices.py input.mp3 output.mp3 --remove-music --mp3-bitrate 192 --mp3-mode vbr
"""

import numpy as np
import soundfile as sf
import librosa
import argparse
import os
import tempfile
import subprocess


def dereverberate(y, sr, method='none', device='cpu'):
    if method == 'none':
        return y
    elif method == 'noisereduce':
        try:
            import noisereduce as nr
        except ImportError:
            print("[!] pip install noisereduce"); return y
        print("   [*] noisereduce...")
        result = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.85).astype(np.float32)
        # Sanitize: replace NaN/inf that noisereduce can produce on clean signals
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result
    elif method == 'wpe':
        try:
            from nara_wpe.wpe import wpe
            from nara_wpe.utils import stft, istft
        except ImportError:
            print("[!] pip install nara-wpe"); return y
        print("   [*] WPE...")
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
            print("[!] pip install deepfilternet"); return y
        use_gpu = device == 'cuda'
        print(f"   [*] DeepFilterNet by 30s chunks ({'GPU' if use_gpu else 'CPU'})...")
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



def save_audio(y, sr, output_file, mp3_bitrate=192, mp3_mode='cbr'):
    """
    Save audio array to file. Format is auto-detected from extension.
    Supported: .wav, .mp3, .flac, .ogg
    MP3 requires ffmpeg.
    """
    ext = os.path.splitext(output_file)[1].lower()

    if ext == '.wav':
        sf.write(output_file, y, sr)

    elif ext in ('.mp3', '.flac', '.ogg'):
        # Write temp WAV then convert via ffmpeg
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            sf.write(tmp.name, y, sr)
            codec_map = {'.flac': 'flac', '.ogg': 'libvorbis'}
            if ext == '.mp3':
                if mp3_mode == 'vbr':
                    # VBR: quality 0 (best) to 9 (worst) — map bitrate to quality
                    vbr_q = {128: 6, 160: 5, 192: 4, 256: 2, 320: 0}.get(mp3_bitrate, 4)
                    cmd = ['ffmpeg', '-i', tmp.name,
                           '-codec:a', 'libmp3lame', '-q:a', str(vbr_q),
                           '-y', output_file]
                else:  # CBR
                    cmd = ['ffmpeg', '-i', tmp.name,
                           '-codec:a', 'libmp3lame', '-b:a', f'{mp3_bitrate}k',
                           '-y', output_file]
            else:
                cmd = ['ffmpeg', '-i', tmp.name,
                       '-codec:a', codec_map[ext], '-y', output_file]
            subprocess.run(cmd, check=True, capture_output=True)
            mode_str = f"VBR q={vbr_q}" if ext == '.mp3' and mp3_mode == 'vbr' else f"CBR {mp3_bitrate}k"
            print(f"   [*] Encoded to {ext[1:].upper()} ({mode_str})")
        except subprocess.CalledProcessError as e:
            print(f"   [!] ffmpeg encoding error: {e}")
            print(f"   [!] Falling back to WAV: {output_file.replace(ext, '.wav')}")
            output_file = output_file.replace(ext, '.wav')
            sf.write(output_file, y, sr)
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
    else:
        print(f"   [!] Unknown extension '{ext}', saving as WAV")
        sf.write(output_file, y, sr)

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"[OK] {output_file}  ({len(y)/sr:.1f}s, {size_mb:.1f} MB)")


def detect_segments(y, sr, min_silence=0.15, min_speech=0.2):
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop)
    rms_norm = rms / (np.max(rms) + 1e-10)
    # Adaptive threshold: try median*0.4 first, but if it would yield
    # 0 segments (over-processed / already clean audio), fall back to
    # a percentile-based threshold so we always detect something.
    threshold = max(0.02, np.median(rms_norm) * 0.4)
    if np.sum(rms_norm > threshold) < 3:
        # fallback: use 10th percentile of non-zero values
        nonzero = rms_norm[rms_norm > 0.001]
        if len(nonzero) > 0:
            threshold = float(np.percentile(nonzero, 10))
        else:
            threshold = 0.01
        print(f"   [*] Adaptive RMS threshold: {threshold:.4f} (fallback mode)")
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
            debug=False, remove_music=False, demucs_model='htdemucs_ft', device='cpu',
            mp3_bitrate=192, mp3_mode='cbr', min_silence=0.15):

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
            save_audio(y, sr, output_file, mp3_bitrate, mp3_mode)
            return
    else:
        y, sr = librosa.load(input_file, sr=None, mono=True)
    total_dur = len(y) / sr
    rm_str = "none"
    sil_str = 'natural duration' if silence_mode == 'auto' else \
              'no silence' if silence_mode == 0.0 else f'{silence_mode}s fixed'
    print(f"   Duree:{total_dur:.1f}s  Keep:{keep_set}  Silence:{sil_str}  Dereverb:{deverb_method}")

    if deverb_method != 'none':
        y = dereverberate(y, sr, deverb_method, device)
        # If keep is vocals_only, just save the cleaned file and exit — no F0 cuts.
        if 'vocals_only' in keep_set:
            print(f"[OK] Dereverberation done. Saving directly (no F0 filtering).")
            save_audio(y, sr, output_file, mp3_bitrate, mp3_mode)
            return

    segs = detect_segments(y, sr, min_silence=min_silence)
    print(f"   {len(segs)} segments detected")

    # Compute silence to insert between kept segments
    if silence_mode == 'auto':
        sil_n = None  # variable — follows natural gaps
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

            # Keep the segment with fade in/out
            chunk = y[int(start*sr):int(end*sr)].copy()
            fade = min(int(0.02*sr), len(chunk)//4)
            if fade > 0:
                chunk[:fade] *= np.linspace(0, 1, fade)
                chunk[-fade:] *= np.linspace(1, 0, fade)
            result.append(chunk)
            last_kept_end = end

    if not result:
        print("[!]  No segment kept!"); return

    # Optional trailing silence
    if last_kept_end is not None and last_kept_end < total_dur:
        if silence_mode == 'auto':
            result.append(np.zeros(int((total_dur - last_kept_end) * sr), dtype=np.float32))
        elif sil_n and sil_n > 0:
            result.append(np.zeros(sil_n, dtype=np.float32))

    final = np.concatenate(result)
    sf.write(output_file, final, sr)
    print(f"\n[OK] {output_file}  ({len(final)/sr:.1f}s)")


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
        else: raise argparse.ArgumentTypeError(f"Unknown: {p}")
    return result

def parse_silence(s):
    s = s.strip().lower()
    if s == 'auto': return 'auto'
    v = float(s)
    if v < 0: raise argparse.ArgumentTypeError("Negative value")
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
    p.add_argument("--min-silence", type=float, default=0.15,
                   help="Minimum silence duration in seconds to split segments (default: 0.15). "
                        "Increase to 0.3-0.5 to detect sentence pauses in single-voice files.")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--mp3-bitrate", type=int, default=192,
                   choices=[128, 160, 192, 256, 320],
                   help="MP3 bitrate in kbps (default: 192) — only used when output is .mp3")
    p.add_argument("--mp3-mode", choices=["cbr", "vbr"], default="cbr",
                   help="MP3 encoding mode: cbr (constant) or vbr (variable, default: cbr)")
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
            args.remove_music, args.demucs_model, args.device,
            args.mp3_bitrate, args.mp3_mode, args.min_silence)

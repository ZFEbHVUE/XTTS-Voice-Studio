#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Female / male voice separation — v2 multi-feature approach.

Classification:
  female_solo  -> F0 >= threshold, faible variance, centroïde élevé
  male_solo    -> F0 <  threshold, faible variance, centroïde bas
  overlap      -> large plage F0 (2 voix simultanées)
  silence      -> trop peu de trames voisées

Options --keep:
  female, male, overlap, all, female,male, vocals only

Options --silence:
  auto  -> durée naturelle
  0     -> pas de silence
  N     -> N secondes entre chaque segment gardé

Options --dereverberate:
  none, noisereduce, wpe, deepfilter

Options --method:
  f0        -> classification par F0 + centroïde spectral (défaut)
  pyannote  -> diarisation pyannote.audio + genre par locuteur (meilleur, requiert --hf-token)

Options nouvelles:
  --analyze       Affiche un tableau par segment (F0, centroïde, label) — pas de sortie
  --split-output  Génère OUTPUT_female.ext + OUTPUT_male.ext en un seul passage

Format de sortie auto-détecté par extension: .wav, .mp3, .flac, .ogg

Usage:
  python extract_voices.py input.mp3 output.wav --keep female --silence 0.5
  python extract_voices.py input.mp3 --analyze
  python extract_voices.py input.mp3 output.wav --split-output
  python extract_voices.py input.mp3 output.wav --method pyannote --hf-token hf_xxx
  python extract_voices.py input.mp3 output.mp3 --keep female --mp3-bitrate 256 --mp3-mode vbr
  python extract_voices.py input.mp3 output.mp3 --remove-music --mp3-bitrate 192
"""

import numpy as np
import soundfile as sf
import librosa
import argparse
import os
import re
import tempfile
import subprocess


# ── Dereverberation ──────────────────────────────────────────────────────────

def _voice_health(y, sr):
    """Cheap voice-health proxies: HF energy ratio (4-8 kHz vs 0.3-3 kHz, dB) and
    spectral flatness. Denoisers that eat the voice show up as HF collapse."""
    n = min(len(y), sr * 60)
    Y = np.abs(np.fft.rfft(y[:n] * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(Y[m].mean()) if m.any() else 1e-12
    hf_db = 10 * np.log10(band(4000, 8000) / (band(300, 3000) + 1e-12) + 1e-12)
    flat = float(np.exp(np.mean(np.log(Y + 1e-12))) / (np.mean(Y) + 1e-12))
    return hf_db, flat


def dereverberate(y, sr, method='none', device='cpu'):
    """Denoise/dereverb with a measured voice-health guard: warns when the
    processing audibly degrades the voice (HF collapse) — the elo lesson:
    stacked denoisers destroy timbre and cloning identity."""
    if method == 'none':
        return y
    hf0, fl0 = _voice_health(y, sr)
    out = _dereverberate_impl(y, sr, method=method, device=device)
    hf1, fl1 = _voice_health(out, sr)
    d_hf = hf1 - hf0
    if d_hf < -3.0:
        print(f"   [!] WARNING: '{method}' removed {-d_hf:.1f} dB of the voice's "
              f"high band (4-8 kHz) — this dulls timbre and HURTS cloning identity.")
        print(f"   [!] Use ONE light denoising pass max; never stack denoisers "
              f"(deepfilter + noisereduce). If the source already sounds clean, use none.")
    else:
        print(f"   [*] Voice health after '{method}': HF {d_hf:+.1f} dB (ok)")
    return out


def _dereverberate_impl(y, sr, method='none', device='cpu'):
    if method == 'none':
        return y
    elif method == 'noisereduce':
        try:
            import noisereduce as nr
        except ImportError:
            print("[!] pip install noisereduce"); return y
        print("   [*] noisereduce...")
        result = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.85).astype(np.float32)
        return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
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
        res = np.zeros_like(y); res[:min(len(out), len(y))] = out[:min(len(out), len(y))]
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
        xf = int(0.02 * t_sr)                       # 20 ms seam crossfade
        chunks = []; n_chunks = int(np.ceil(len(y48) / cs))
        for i in range(n_chunks):
            s = max(0, i * cs - ov); e = min(len(y48), (i + 1) * cs + ov)
            c = y48[s:e]
            with torch.no_grad():
                t = torch.from_numpy(c[None]).float()
                enh = enhance(model, df_state, t).cpu().numpy()[0]
            # Keep xf extra samples before the nominal seam so consecutive
            # chunks overlap and can be crossfaded (hard cuts can click:
            # DeepFilter's output is not sample-consistent across boundaries).
            ts = max(0, (i * cs - s) - xf) if i > 0 else 0
            te = len(enh) - (e - (i + 1) * cs) if e > (i + 1) * cs else len(enh)
            chunks.append(enh[ts:te])
            print(f"   [*]  {i+1}/{n_chunks}...", end='\r')
        print()
        out48 = chunks[0]
        ramp = np.linspace(0, 1, xf, dtype=np.float32)
        for c in chunks[1:]:
            if len(out48) >= xf and len(c) >= xf:
                out48 = np.concatenate([out48[:-xf],
                                        out48[-xf:] * (1 - ramp) + c[:xf] * ramp,
                                        c[xf:]])
            else:
                out48 = np.concatenate([out48, c])
        out = librosa.resample(out48, orig_sr=t_sr, target_sr=sr) if sr != t_sr else out48
        res = np.zeros_like(y); res[:min(len(out), len(y))] = out[:min(len(out), len(y))]
        return res.astype(np.float32)
    return y


# ── Demucs music removal ─────────────────────────────────────────────────────

def remove_music_demucs(input_file, demucs_model='htdemucs_ft', device='cpu', shifts=2):
    """Use demucs to separate vocals. Returns (y_vocals, sr) or (None, None).
    shifts>1 = test-time augmentation (N shifted passes averaged) — demucs'
    main quality lever: cleaner vocal stem, fewer music residues, ~N x slower."""
    try:
        import demucs.separate
    except ImportError:
        print("[!] demucs not installed -> pip install demucs")
        return None, None

    import shutil, glob

    tmp_dir = tempfile.mkdtemp(prefix='demucs_')
    try:
        print(f"   [*] demucs ({demucs_model}, shifts={shifts}) separating sources...")
        demucs_args = ["--two-stems", "vocals", "-n", demucs_model,
                       "--shifts", str(int(shifts)), "--out", tmp_dir]
        if device == "cuda":
            demucs_args += ["--device", "cuda"]
        demucs_args.append(input_file)
        demucs.separate.main(demucs_args)

        pattern = os.path.join(tmp_dir, demucs_model, "**", "vocals.wav")
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            matches = glob.glob(os.path.join(tmp_dir, "**", "vocals.wav"), recursive=True)
        if not matches:
            print("[!] demucs: vocals.wav not found"); return None, None

        y, sr = sf.read(matches[0], dtype='float32', always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        print(f"   [OK] demucs done -> {len(y)/sr:.1f}s vocals stem")
        return y, sr
    except Exception as e:
        print(f"   [!] demucs error: {e}"); return None, None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Audio save ───────────────────────────────────────────────────────────────

_TEMPO = 1.0   # global time-stretch factor (pitch preserved); set from --tempo



# Age presets, in semitones of pitch shift.
#
# What actually reads as age is the FORMANT position (vocal-tract length), not
# the fundamental. The obvious approach would be to move the two independently
# — but the rubberband CLI has no --formantscale: it offers only --formant,
# which PRESERVES the formants during a pitch shift. Moving them deliberately
# is not available from the command line.
#
# So the presets do the opposite of --formant: they let the pitch shift carry
# the formants with it, which is exactly what makes a voice sound younger or
# older. The chipmunk effect people associate with this comes from EXCESSIVE
# shifts (+8, +12 st); at +2 or +3 the vocal tract simply reads as smaller and
# the result is plausible. That is why every value below stays under 4 st.
#
# Female and male differ: their starting F0 and tract length are not the same,
# so the same shift does not read the same way.
AGE_PRESETS = {
    # Rejuvenating: kept, but they only convince at small amplitudes. A young
    # voice is not an adult voice transposed — it also has a higher HNR, less
    # jitter and a different attack, none of which a pitch shift provides.
    'child':      {'F': +3.5, 'M': +4.5},
    'teen':       {'F': +2.0, 'M': +2.5},
    'younger':    {'F': +1.0, 'M': +1.5},
    # Ageing: this direction works, so it gets the finer steps. Going DOWN
    # lengthens the tract virtually, which stays physically plausible, and the
    # shifts needed are small.
    'older':      {'F': -1.0, 'M': -1.0},
    'mature':     {'F': -1.75, 'M': -1.75},
    'much_older': {'F': -2.0, 'M': -2.5},
    'elderly':    {'F': -3.0, 'M': -3.5},
}

# Rejuvenating a male voice does not work as well, and the reason is physical.
# An adult male sits near 110-130 Hz, a teenager near 160-180: covering that
# needs 5-7 semitones, well past the point where a pitch shift still sounds
# like a person. A female voice only has to travel 205 -> 230 Hz, so +2 does it
# and stays clean. On top of that, the male break is a change of LARYNX, not of
# pitch — no amount of shifting reproduces it. Warn instead of pretending.
def age_warning(semitones, f0_hint=None):
    """Return a warning string when the requested shift is not believable."""
    if semitones >= 3.0 and f0_hint and f0_hint < 155:
        target = f0_hint * (2 ** (semitones / 12.0))
        return (f"the source is a low voice ({f0_hint:.0f} Hz) and {semitones:+g} st "
                f"only reaches {target:.0f} Hz -- halfway to a young voice, so it "
                f"tends to read as 'shifted' rather than 'younger'. A real change "
                f"of age across the male break needs RVC, not a pitch shift.")
    if abs(semitones) > 4.0:
        return (f"{semitones:+g} st is past the range where the vocal tract stays "
                f"plausible; expect an audibly processed result.")
    return None


def shift_age(y, sr, semitones=0.0, preserve_formants=False):
    """Re-age a voice by shifting pitch, formants following along.

    preserve_formants=True adds --formant, which keeps the original timbre —
    useful to change the pitch WITHOUT changing the apparent speaker size, but
    it defeats the age effect. Left False by default for that reason.
    """
    if abs(semitones) < 1e-3:
        return y
    import shutil, subprocess, tempfile as _tf
    y = np.asarray(y, dtype=np.float32)
    rb = shutil.which('rubberband')
    if not rb:
        print("   [!] 'rubberband' NOT FOUND -> age shift SKIPPED (audio unchanged).")
        print("   [!] Install it: apt install rubberband-cli / brew install rubberband")
        return y
    try:
        vp = subprocess.run([rb, '--version'], capture_output=True, text=True)
        ver = (vp.stderr or '') + (vp.stdout or '')
        m = re.search(r'(\d+)\.', ver)
        major = int(m.group(1)) if m else 2
        fi = _tf.NamedTemporaryFile(suffix='.wav', delete=False).name
        fo = _tf.NamedTemporaryFile(suffix='.wav', delete=False).name
        sf.write(fi, y, sr)
        cmd = [rb, '--pitch', f'{semitones:g}']
        if preserve_formants:
            cmd.append('--formant')
        if major >= 3:
            cmd.append('--fine')            # R3 engine: much cleaner on speech
        cmd += [fi, fo]
        subprocess.run(cmd, check=True, capture_output=True)
        out, _ = sf.read(fo, dtype='float32')
        for f in (fi, fo):
            try:
                os.unlink(f)
            except OSError:
                pass
        print(f"   [*] Age shift: pitch {semitones:+g} st"
              f"{' (formants preserved)' if preserve_formants else ' (formants follow)'}"
              f" via rubberband {'R3' if major >= 3 else 'R2'}")
        return out
    except Exception as e:
        print(f"   [!] Age shift failed ({e}) -> audio unchanged")
        return y


def _time_stretch(y, sr, rate):
    """Time-stretch preserving pitch/timbre. rate>1 faster, rate<1 slower.
    Engine order: rubberband CLI (R3 '--fine' when available — best for voice),
    else librosa phase vocoder (audibly metallic on speech; loud warning)."""
    if rate is None or rate <= 0 or abs(rate - 1.0) < 1e-3:
        return y
    import shutil, subprocess, tempfile as _tf
    y = np.asarray(y, dtype=np.float32)
    rb = shutil.which('rubberband')
    if rb:
        try:
            vp = subprocess.run([rb, '--version'], capture_output=True, text=True)
            ver = (vp.stderr or '') + (vp.stdout or '')
            m = re.search(r'(\d+)\.', ver)
            major = int(m.group(1)) if m else 2
            fi = _tf.NamedTemporaryFile(suffix='.wav', delete=False).name
            fo = _tf.NamedTemporaryFile(suffix='.wav', delete=False).name
            sf.write(fi, y, sr)                 # stereo handled natively
            cmd = [rb, '--tempo', str(rate)]
            if major >= 3:
                cmd.append('--fine')            # R3 engine: much cleaner speech
            cmd += [fi, fo]
            subprocess.run(cmd, check=True, capture_output=True)
            out, _ = sf.read(fo, dtype='float32')
            os.unlink(fi); os.unlink(fo)
            print(f"   [*] time-stretch x{rate} via rubberband "
                  f"{'R3 (--fine)' if major >= 3 else 'R2 (install rubberband>=3 for cleaner voice)'}")
            return out
        except Exception as e:
            print(f"   [!] rubberband failed ({e}) -> librosa fallback")
    else:
        print("   [!] 'rubberband' binary NOT FOUND -> librosa phase-vocoder fallback,")
        print("   [!] which sounds METALLIC on voice. Fix: sudo apt install rubberband-cli")
    import librosa
    if y.ndim == 1:
        return librosa.effects.time_stretch(y, rate=rate)
    chans = [librosa.effects.time_stretch(np.ascontiguousarray(y[:, c]), rate=rate)
             for c in range(y.shape[1])]
    n = min(len(c) for c in chans)
    return np.stack([c[:n] for c in chans], axis=1)



def shorten_silences(y, sr, target_ms=500, min_gap_ms=600, floor_db=-42.0,
                     fade_ms=20):
    """Shorten the silent gaps WITHOUT cutting anything out of the speech.

    The segment pipeline solves a different problem: it splits the file, judges
    each piece and drops the ones that fail. On a source with long pauses that
    is the wrong tool — every cut is a chance to lose a syllable, which is why
    tightening the settings never worked: the loss came from the discarding, not
    from the thresholds.

    Here nothing is ever discarded. Every sample of speech is kept in order, and
    only the gaps between words are compressed to `target_ms`. A short crossfade
    at each junction avoids the click a hard splice would make.

    The threshold is relative to the material: it is placed `floor_db` below the
    signal's own loud level, so it adapts to a quiet recording instead of
    assuming a fixed dBFS.
    """
    import numpy as _np
    if y is None or len(y) == 0:
        return y
    hop = max(1, int(0.010 * sr))                      # 10 ms resolution
    frames = _np.array([_np.sqrt(_np.mean(y[i:i + hop] ** 2))
                        for i in range(0, len(y) - hop, hop)])
    if len(frames) < 3:
        return y
    db = 20.0 * _np.log10(frames + 1e-10)
    loud = _np.percentile(db, 90)                       # the speech level
    thresh = loud + floor_db
    speech = db > thresh

    target = max(0, int(target_ms * sr / 1000))
    min_gap = max(1, int(min_gap_ms / 10))              # in frames
    fade = max(1, int(fade_ms * sr / 1000))

    out, i, n_short, saved = [], 0, 0, 0
    while i < len(speech):
        if speech[i]:
            j = i
            while j < len(speech) and speech[j]:
                j += 1
            out.append(y[i * hop: min(len(y), j * hop)])
            i = j
        else:
            j = i
            while j < len(speech) and not speech[j]:
                j += 1
            gap_frames = j - i
            gap = y[i * hop: min(len(y), j * hop)]
            if gap_frames >= min_gap and len(gap) > target:
                # Keep the quietest `target` samples of the gap: the room tone
                # stays, the dead air goes.
                half = target // 2
                out.append(_np.concatenate([gap[:half], gap[-(target - half):]])
                           if target else gap[:0])
                n_short += 1
                saved += len(gap) - target
            else:
                out.append(gap)                          # short pause: untouched
            i = j

    if not out:
        return y
    merged = _np.concatenate(out)
    # Smooth the junctions so shortened gaps do not click.
    if fade > 1 and len(merged) > 2 * fade:
        ramp = _np.linspace(0.0, 1.0, fade, dtype=merged.dtype)
        merged[:fade] *= ramp
        merged[-fade:] *= ramp[::-1]
    print(f"   [*] Silences shortened: {n_short} gap(s) >= {min_gap_ms}ms "
          f"reduced to {target_ms}ms  ({saved / sr:.1f}s removed, "
          f"{len(y) / sr:.1f}s -> {len(merged) / sr:.1f}s)")
    print(f"   [*] No speech was cut: every voiced sample is kept in order.")
    return merged


def save_audio(y, sr, output_file, mp3_bitrate=192, mp3_mode='cbr'):
    """Save numpy array to file. Format auto-detected from extension."""
    if _TEMPO and abs(_TEMPO - 1.0) > 1e-3:
        y = _time_stretch(y, sr, _TEMPO)
    ext = os.path.splitext(output_file)[1].lower()

    if ext == '.wav':
        sf.write(output_file, y, sr)
        size_mb = os.path.getsize(output_file) / (1024 * 1024)
        print(f"[OK] {output_file}  ({len(y)/sr:.1f}s, {size_mb:.1f} MB)")

    elif ext in ('.mp3', '.flac', '.ogg'):
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        try:
            sf.write(tmp.name, y, sr)
            if ext == '.mp3':
                if mp3_mode == 'vbr':
                    vbr_q = {128: 6, 160: 5, 192: 4, 256: 2, 320: 0}.get(mp3_bitrate, 4)
                    cmd = ['ffmpeg', '-i', tmp.name,
                           '-codec:a', 'libmp3lame', '-q:a', str(vbr_q), '-y', output_file]
                    mode_str = f"VBR q={vbr_q}"
                else:
                    cmd = ['ffmpeg', '-i', tmp.name,
                           '-codec:a', 'libmp3lame', '-b:a', f'{mp3_bitrate}k', '-y', output_file]
                    mode_str = f"CBR {mp3_bitrate}k"
            else:
                codec_map = {'.flac': 'flac', '.ogg': 'libvorbis'}
                cmd = ['ffmpeg', '-i', tmp.name, '-codec:a', codec_map[ext], '-y', output_file]
                mode_str = ext[1:].upper()
            subprocess.run(cmd, check=True, capture_output=True)
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"[OK] {output_file}  ({len(y)/sr:.1f}s, {size_mb:.1f} MB, {mode_str})")
        except subprocess.CalledProcessError as e:
            print(f"   [!] ffmpeg error: {e}")
            fallback = output_file.replace(ext, '.wav')
            sf.write(fallback, y, sr)
            print(f"   [!] Saved as WAV fallback: {fallback}")
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
    else:
        print(f"   [!] Unknown extension '{ext}', saving as WAV")
        sf.write(output_file, y, sr)


# ── Segment detection (RMS-based VAD) ────────────────────────────────────────

def detect_segments(y, sr, min_silence=0.15, min_speech=0.2):
    hop = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop)
    rms_norm = rms / (np.max(rms) + 1e-10)

    threshold = max(0.02, np.median(rms_norm) * 0.4)
    if np.sum(rms_norm > threshold) < 3:
        nonzero = rms_norm[rms_norm > 0.001]
        threshold = float(np.percentile(nonzero, 10)) if len(nonzero) > 0 else 0.01
        print(f"   [*] Adaptive RMS threshold: {threshold:.4f} (fallback)")

    is_speech = rms_norm > threshold
    segs = []; in_seg = False; t0 = 0.0
    for i, sp in enumerate(is_speech):
        if sp and not in_seg:   t0 = float(times[i]); in_seg = True
        elif not sp and in_seg:
            if float(times[i]) - t0 >= min_speech: segs.append((t0, float(times[i])))
            in_seg = False
    if in_seg and float(times[-1]) - t0 >= min_speech:
        segs.append((t0, float(times[-1])))
    if not segs: return []
    merged = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s - pe < min_silence: merged[-1] = (ps, e)
        else: merged.append((s, e))
    return merged


# ── F0 estimation — pyin (CPU) ou torchcrepe (GPU) ──────────────────────────

def _f0_pyin(seg, sr):
    """Estimation F0 CPU via librosa.pyin. Retourne (f0, voiced, voiced_ratio)."""
    frame_len = min(2048, max(1024, len(seg) // 2))
    hop       = min(512,  frame_len // 4)
    f0, voiced, _ = librosa.pyin(seg, fmin=60, fmax=500, sr=sr,
                                   frame_length=frame_len, hop_length=hop)
    vr = float(np.sum(voiced)) / max(len(voiced), 1)
    return f0, voiced, vr, hop


def _f0_crepe(seg, sr, device='cuda'):
    """
    Estimation F0 GPU via torchcrepe.
    pip install torchcrepe
    Retourne (f0, voiced, voiced_ratio, hop).
    """
    try:
        import torchcrepe, torch
    except ImportError:
        print("[!] torchcrepe non disponible, fallback pyin")
        return _f0_pyin(seg, sr)

    # torchcrepe attend du 16 kHz
    hop_ms   = 10          # 10 ms
    hop_sr   = 16000
    hop_out  = int(hop_ms * hop_sr / 1000)   # samples à 16 kHz

    if sr != hop_sr:
        seg16 = librosa.resample(seg, orig_sr=sr, target_sr=hop_sr)
    else:
        seg16 = seg

    audio = torch.from_numpy(seg16[None]).float()
    with torch.no_grad():
        f0_t, conf_t = torchcrepe.predict(
            audio, hop_sr, hop_out,
            fmin=60, fmax=500,
            model='tiny',          # 'tiny' rapide, 'full' plus précis
            device=device,
            return_periodicity=True,
            batch_size=512,
        )
    f0     = f0_t[0].cpu().numpy()
    voiced = (conf_t[0].cpu().numpy() > 0.40)
    vr     = float(np.sum(voiced)) / max(len(voiced), 1)

    # hop en samples au sr original (approximatif, pour centroïde)
    hop_orig = int(hop_out * sr / hop_sr)
    return f0, voiced, vr, hop_orig


# ── Worker multiprocessing (top-level = picklable) ───────────────────────────

def _classify_worker(args):
    """
    Appelé par ProcessPoolExecutor — doit être une fonction top-level.
    args = (seg_bytes, sr, f0_thr, ov_range)
    Retourne (label, f0_med, cent_med, voiced_ratio_pct)
    """
    import numpy as np, librosa

    seg_bytes, sr, f0_thr, ov_range = args
    seg = np.frombuffer(seg_bytes, dtype=np.float32).copy()

    if len(seg) < 1024:
        return ('silence', None, None, 0)

    frame_len = min(2048, max(1024, len(seg) // 2))
    hop       = min(512,  frame_len // 4)

    f0, voiced, _ = librosa.pyin(seg, fmin=60, fmax=500, sr=sr,
                                   frame_length=frame_len, hop_length=hop)
    vr = float(np.sum(voiced)) / max(len(voiced), 1)

    if vr < 0.12:
        return ('silence', None, None, int(100 * vr))

    f0v = f0[voiced & ~np.isnan(f0)]
    if len(f0v) < 2:
        return ('silence', None, None, int(100 * vr))

    med = float(np.median(f0v))
    rng = float(np.percentile(f0v, 90) - np.percentile(f0v, 10)) if len(f0v) >= 4 else 0.0

    centroid = librosa.feature.spectral_centroid(y=seg, sr=sr, hop_length=hop)[0]
    med_cent = float(np.median(centroid))

    if rng > ov_range:
        return ('overlap', med, med_cent, int(100 * vr))

    diff = med - f0_thr
    if abs(diff) >= 15.0:
        label = 'female_solo' if diff >= 0 else 'male_solo'
    else:
        f0_score   = diff / 15.0
        cent_score = (med_cent - 1350.0) / 300.0
        score      = 0.65 * f0_score + 0.35 * cent_score
        label      = 'female_solo' if score >= 0 else 'male_solo'

    return (label, med, med_cent, int(100 * vr))


# ── Classification — dispatch CPU/GPU ────────────────────────────────────────

def classify(y, sr, start, end, f0_thr=165, ov_range=80, device='cpu'):
    """
    Classify un segment. device='cpu' -> pyin, device='cuda' -> torchcrepe.
    Retourne (label, f0_median, centroid_median).
    """
    seg = y[int(start * sr):int(end * sr)]
    if len(seg) < 1024:
        return 'silence', None, None

    if device == 'cuda':
        f0, voiced, vr, hop = _f0_crepe(seg, sr, device)
    else:
        f0, voiced, vr, hop = _f0_pyin(seg, sr)

    if vr < 0.12:
        return 'silence', None, None

    f0v = f0[voiced & ~np.isnan(f0)]
    if len(f0v) < 2:
        return 'silence', None, None

    med = float(np.median(f0v))
    rng = float(np.percentile(f0v, 90) - np.percentile(f0v, 10)) if len(f0v) >= 4 else 0.0

    centroid = librosa.feature.spectral_centroid(y=seg, sr=sr, hop_length=hop)[0]
    med_cent = float(np.median(centroid))

    if rng > ov_range:
        return 'overlap', med, med_cent

    diff = med - f0_thr
    if abs(diff) >= 15.0:
        label = 'female_solo' if diff >= 0 else 'male_solo'
    else:
        f0_score   = diff / 15.0
        cent_score = (med_cent - 1350.0) / 300.0
        score      = 0.65 * f0_score + 0.35 * cent_score
        label      = 'female_solo' if score >= 0 else 'male_solo'

    return label, med, med_cent


# ── Mode Analyse — multicore CPU ou GPU ──────────────────────────────────────

def analyze(input_file, f0_thr=165, ov_range=80, min_silence=0.15, device='cpu'):
    """
    Affiche un tableau par segment : label, F0, centroïde, voiced ratio.
    CPU  -> ProcessPoolExecutor  (tous les coeurs en parallèle)
    CUDA -> torchcrepe           (GPU batch)
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print(f"\n[ANALYZE] Chargement: {input_file}")
    y, sr = librosa.load(input_file, sr=None, mono=True)
    segs  = detect_segments(y, sr, min_silence=min_silence)
    total = len(y) / sr
    n_cpu = os.cpu_count() or 4

    print(f"[ANALYZE] Duree: {total:.1f}s | {len(segs)} segments | "
          f"Device: {device.upper()} | Workers CPU: {n_cpu}")
    print(f"          Seuil F0: {f0_thr} Hz | Plage overlap: {ov_range} Hz\n")
    print(f"  {'#':>3}  {'Debut':>7}  {'Fin':>7}  {'Dur':>5}  "
          f"{'Label':14}  {'F0 (Hz)':>8}  {'Centroide':>10}  {'VR%':>5}")
    print("  " + "-" * 76)

    results = [None] * len(segs)

    if device == 'cuda':
        # ── GPU : boucle torchcrepe, pas de multiprocessing
        for i, (start, end) in enumerate(segs):
            label, f0, cent = classify(y, sr, start, end, f0_thr, ov_range, device='cuda')
            # voiced ratio via crepe pour affichage
            seg = y[int(start * sr):int(end * sr)]
            if len(seg) >= 1024:
                _, _, vr, _ = _f0_crepe(seg, sr, 'cuda')
                vr_pct = int(100 * vr)
            else:
                vr_pct = 0
            results[i] = (label, f0, cent, vr_pct)
            print(f"  {i+1:>3}  {start:7.2f}  {end:7.2f}  {end-start:5.2f}  "
                  f"{label:14}  "
                  f"{f0:7.0f}" if f0 is not None else f"  {'':7}",
                  end='')
            # flush immediat pour voir dans la console GUI
            import sys; sys.stdout.flush()

    else:
        # ── CPU multicore : ProcessPoolExecutor
        # Serialise chaque segment en bytes (evite pickle de l'array complet)
        tasks = []
        for start, end in segs:
            seg = y[int(start * sr):int(end * sr)].astype(np.float32)
            tasks.append((seg.tobytes(), sr, f0_thr, ov_range))

        futures = {}
        with ProcessPoolExecutor(max_workers=n_cpu) as ex:
            for i, task in enumerate(tasks):
                fut = ex.submit(_classify_worker, task)
                futures[fut] = i

            done_count = 0
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()
                done_count += 1
                print(f"   ... {done_count}/{len(segs)} segments traites\r",
                      end='', flush=True)

        print()  # newline apres le \r

    # ── Affichage du tableau trié par index ──────────────────────────────
    counts = {}
    for i, (start, end) in enumerate(segs):
        label, f0, cent, vr_pct = results[i]
        counts[label] = counts.get(label, 0) + 1
        f0_s   = f"{f0:7.0f}"  if f0   is not None else "      -"
        cent_s = f"{cent:8.0f}" if cent is not None else "       -"
        print(f"  {i+1:>3}  {start:7.2f}  {end:7.2f}  {end-start:5.2f}  "
              f"{label:14}  {f0_s}  {cent_s}  {vr_pct:>4}%")

    print(f"\n  Resume: {dict(counts)}")
    print(f"  Conseil: si des labels H sont errones, augmentez --overlap-range (actuel: {ov_range})")
    print(f"           si seuil F/H mal place, ajustez --threshold (actuel: {f0_thr} Hz)")
    print(f"  F0 femme typique: 165-255 Hz | F0 homme typique: 85-180 Hz\n")


# ── Méthode SepFormer — séparation neuronale réelle ─────────────────────────

def _load_sepformer(device='cpu'):
    """
    Charge SepFormer. Compatible speechbrain >= 0.5 (nouvelle API)
    et ancienne API (pretrained).
    """
    savedir = os.path.expanduser("~/.cache/speechbrain/sepformer-wsj02mix")
    opts    = {"device": device}

    # Nouvelle API speechbrain >= 0.5
    try:
        from speechbrain.inference.separation import SepformerSeparation as Sep
        model = Sep.from_hparams(source="speechbrain/sepformer-wsj02mix",
                                 savedir=savedir, run_opts=opts)
        print("   [*] SpeechBrain >= 0.5 API")
        return model
    except ImportError:
        pass

    # Ancienne API speechbrain < 0.5
    from speechbrain.pretrained import SepformerSeparation as Sep
    model = Sep.from_hparams(source="speechbrain/sepformer-wsj02mix",
                             savedir=savedir, run_opts=opts)
    print("   [*] SpeechBrain < 0.5 API")
    return model


def _sepformer_separate(model, seg, sr, debug_dir=None, seg_idx=0):
    """
    Separe seg (mono float32) en 2 sources via separate_file().
    Retourne (src1, src2) numpy float32 au sr original.
    debug_dir: si fourni, sauvegarde src1/src2/mix pour les 3 premiers segments.
    """
    import torch

    SEP_SR = 8000

    seg8 = librosa.resample(seg.astype(np.float32), orig_sr=sr, target_sr=SEP_SR)
    peak = float(np.max(np.abs(seg8)))
    if peak < 1e-6:
        return seg.copy(), np.zeros_like(seg)
    seg8 = (seg8 / peak).astype(np.float32)

    # separate_file() est l'API stable — on ecrit un WAV temporaire
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    sf.write(tmp.name, seg8, SEP_SR)
    tmp.close()

    try:
        with torch.no_grad():
            est = model.separate_file(path=tmp.name)
            # est peut etre (time, n_src) ou (1, time, n_src) selon la version
            if est.dim() == 3:
                est = est[0]
            src1 = est[:, 0].cpu().numpy().astype(np.float32)
            src2 = est[:, 1].cpu().numpy().astype(np.float32)
    finally:
        os.unlink(tmp.name)

    # Resample → sr original
    src1 = librosa.resample(src1, orig_sr=SEP_SR, target_sr=sr)
    src2 = librosa.resample(src2, orig_sr=SEP_SR, target_sr=sr)

    # Ajuster la longueur
    n = len(seg)
    src1 = (src1[:n] if len(src1) >= n else np.pad(src1, (0, n - len(src1)))).astype(np.float32)
    src2 = (src2[:n] if len(src2) >= n else np.pad(src2, (0, n - len(src2)))).astype(np.float32)

    # Restaurer l'amplitude
    src1 *= peak
    src2 *= peak

    # Sauvegarde debug (3 premiers segments)
    if debug_dir and seg_idx < 3:
        os.makedirs(debug_dir, exist_ok=True)
        sf.write(os.path.join(debug_dir, f"seg{seg_idx:03d}_mix.wav"),  seg,  sr)
        sf.write(os.path.join(debug_dir, f"seg{seg_idx:03d}_src1.wav"), src1, sr)
        sf.write(os.path.join(debug_dir, f"seg{seg_idx:03d}_src2.wav"), src2, sr)
        print(f"   [DBG] Seg {seg_idx} => {debug_dir}")

    return src1, src2



def _pick_source(src1, src2, sr, target_gender, f0_thr):
    """
    Parmi les 2 sources SepFormer, retourne celle qui correspond à target_gender.
    Classifie chaque source par F0 médian.
    """
    def median_f0(seg):
        if len(seg) < 1024:
            return None
        frame_len = min(2048, max(1024, len(seg) // 2))
        hop = min(512, frame_len // 4)
        f0, voiced, _ = librosa.pyin(seg, fmin=60, fmax=500, sr=sr,
                                      frame_length=frame_len, hop_length=hop)
        f0v = f0[voiced & ~np.isnan(f0)] if voiced is not None else []
        return float(np.median(f0v)) if len(f0v) >= 2 else None

    f1 = median_f0(src1)
    f2 = median_f0(src2)

    # Fallback si F0 indétectable : retourne src avec le plus d'énergie
    if f1 is None and f2 is None:
        return src1 if np.mean(src1 ** 2) >= np.mean(src2 ** 2) else src2

    if f1 is None:
        return src2 if (target_gender == 'female_solo') == (f2 >= f0_thr) else src1
    if f2 is None:
        return src1 if (target_gender == 'female_solo') == (f1 >= f0_thr) else src2

    # Les deux F0 détectés : choisir celle dont le genre correspond
    src1_is_female = f1 >= f0_thr
    src2_is_female = f2 >= f0_thr
    want_female    = (target_gender == 'female_solo')

    if want_female:
        if src1_is_female and not src2_is_female:   return src1
        if src2_is_female and not src1_is_female:   return src2
        # Ambigüité : prendre la F0 la plus haute
        return src1 if f1 >= f2 else src2
    else:
        if not src1_is_female and src2_is_female:   return src1
        if not src2_is_female and src1_is_female:   return src2
        # Ambigüité : prendre la F0 la plus basse
        return src1 if f1 <= f2 else src2


def process_sepformer(input_file, output_file, keep_set=None, silence_mode='auto',
                      deverb_method='none', f0_thr=165, ov_range=80, min_dur=0.2,
                      debug=False, device='cpu', mp3_bitrate=192, mp3_mode='cbr',
                      min_silence=0.15, split_output=False):
    """
    Séparation réelle des voix via SepFormer (speechbrain).
    Pour chaque segment, SepFormer extrait 2 sources, on garde celle
    qui correspond au genre cible — la voix de fond disparaît réellement.

    pip install speechbrain  (déjà installé)
    Modèle : speechbrain/sepformer-wsj02mix (téléchargé automatiquement ~200 MB)
    """
    if keep_set is None:
        keep_set = {'female_solo'}

    print(f"[*] SepFormer — séparation réelle des voix")
    print(f"   [*] Chargement modèle SepFormer ({device.upper()})...")
    try:
        model = _load_sepformer(device)
    except Exception as e:
        print(f"[!] Erreur chargement SepFormer: {e}")
        print("    Fallback vers methode F0 classique...")
        process(input_file, output_file, keep_set, silence_mode, deverb_method,
                f0_thr, ov_range, min_dur, debug, False, 'htdemucs_ft', device,
                mp3_bitrate, mp3_mode, min_silence, split_output)
        return

    print(f"   [OK] SepFormer prêt")
    print(f"   [*] Chargement audio: {input_file}")
    y, sr = librosa.load(input_file, sr=None, mono=True)

    if deverb_method != 'none':
        y = dereverberate(y, sr, deverb_method, device)

    total_dur = len(y) / sr
    segs      = detect_segments(y, sr, min_silence=min_silence)
    print(f"   Durée: {total_dur:.1f}s | {len(segs)} segments détectés")

    sil_n = (None     if silence_mode == 'auto'
             else 0   if silence_mode == 0.0
             else int(float(silence_mode) * sr))

    def _assemble_one(target_kind):
        result      = []
        last_end    = None
        kept        = 0
        sep_idx     = 0   # index pour le debug SepFormer
        debug_dir   = (os.path.join(os.path.dirname(output_file),
                        "sepformer_debug") if debug else None)

        for idx, (start, end) in enumerate(segs):
            dur           = end - start
            kind, f0, _   = classify(y, sr, start, end, f0_thr, ov_range,
                                     device if device == 'cuda' else 'cpu')

            if dur < min_dur:
                continue

            # Inclure seulement les segments du genre cible (pas overlap —
            # sur les segments solo la voix adverse est en fond, SepFormer
            # peut la retirer; sur overlap les deux voix sont egales, moins fiable)
            if kind != target_kind:
                continue

            seg = y[int(start * sr):int(end * sr)].copy()

            try:
                src1, src2 = _sepformer_separate(model, seg, sr,
                                                  debug_dir=debug_dir,
                                                  seg_idx=sep_idx)
                sep_idx   += 1
                chosen     = _pick_source(src1, src2, sr, target_kind, f0_thr)
                if debug:
                    def _mf0(s):
                        fl = min(2048, max(1024, len(s)//2))
                        hp = min(512, fl//4)
                        f0r, vr, _ = librosa.pyin(s, fmin=60, fmax=500,
                                                   sr=sr, frame_length=fl, hop_length=hp)
                        fv = f0r[vr & ~np.isnan(f0r)] if vr is not None else []
                        return f"{np.median(fv):.0f}" if len(fv) >= 2 else "?"
                    print(f"   {start:6.1f}-{end:6.1f}s  {kind:12s}  "
                          f"F0 mix={f0:.0f if f0 else '?'}  "
                          f"src1={_mf0(src1)}Hz  src2={_mf0(src2)}Hz  "
                          f"-> {'src1' if np.array_equal(chosen, src1) else 'src2'}")
            except Exception as e:
                print(f"   [!] SepFormer seg {idx}: {e} -> chunk brut")
                chosen = seg

            # Silence avant
            if last_end is not None:
                gap = start - last_end
                n   = (int(gap * sr) if silence_mode == 'auto' else sil_n or 0)
                if n > 0:
                    result.append(np.zeros(n, dtype=np.float32))

            # Fade in/out
            fade = min(int(0.02 * sr), len(chosen) // 4)
            if fade > 0:
                chosen[:fade]  *= np.linspace(0, 1, fade)
                chosen[-fade:] *= np.linspace(1, 0, fade)

            result.append(chosen)
            last_end = end
            kept += 1

            print(f"   [{idx+1}/{len(segs)}] {start:.1f}-{end:.1f}s  "
                  f"{kind:14s}  -> garde  ({kept} gardes)", end='\r')

        print()
        return np.concatenate(result) if result else None

    if split_output:
        base, ext = os.path.splitext(output_file)
        for tag, kind in [('female', 'female_solo'), ('male', 'male_solo')]:
            print(f"\n[*] Extraction {tag.upper()}...")
            arr = _assemble_one(kind)
            if arr is not None:
                save_audio(arr, sr, base + f'_{tag}' + ext, mp3_bitrate, mp3_mode)
            else:
                print(f"[!] Aucun segment {tag} trouvé.")
    else:
        # keep_set peut contenir female_solo ou male_solo ou les deux
        # On prend le premier genre non-overlap de keep_set
        target = next((k for k in keep_set if k in ('female_solo', 'male_solo')), 'female_solo')
        print(f"\n[*] Extraction {target}...")
        arr = _assemble_one(target)
        if arr is not None:
            save_audio(arr, sr, output_file, mp3_bitrate, mp3_mode)
        else:
            print("[!] Aucun segment gardé.")


# ── Classification pyannote (methode diarisation) ────────────────────────────

def process_pyannote(input_file, output_file, keep_set, silence_mode,
                     deverb_method, f0_thr, hf_token, device='cpu',
                     mp3_bitrate=192, mp3_mode='cbr', split_output=False):
    """
    Diarise avec pyannote.audio, assigne le genre par locuteur (F0 agrégé
    sur l'ensemble de ses segments — bien plus robuste que segment par segment).

    Installs: pip install pyannote.audio
    Token:    https://hf.co/settings/tokens  (accept pyannote/speaker-diarization-3.1)
    """
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("[!] pyannote.audio non installé: pip install pyannote.audio")
        return

    import torch

    print(f"[*] Pyannote diarization: {input_file}")
    if not hf_token:
        print("[!] --hf-token requis pour pyannote.audio"); return

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    if device == 'cuda' and torch.cuda.is_available():
        pipeline = pipeline.to(torch.device('cuda'))

    diarization = pipeline(input_file)

    print("   [*] Chargement audio...")
    y, sr = librosa.load(input_file, sr=None, mono=True)
    if deverb_method != 'none':
        y = dereverberate(y, sr, deverb_method, device)

    # Collecter segments par locuteur
    speaker_segs = {}
    for turn, _, spk in diarization.itertracks(yield_label=True):
        speaker_segs.setdefault(spk, []).append((turn.start, turn.end))

    print(f"   {len(speaker_segs)} locuteur(s) détecté(s)")

    # F0 agrégé par locuteur → genre
    speaker_gender = {}
    for spk, segs in sorted(speaker_segs.items()):
        all_f0 = []
        for start, end in segs:
            seg = y[int(start * sr):min(int(end * sr), len(y))]
            if len(seg) < 1024: continue
            f0, voiced, _ = librosa.pyin(seg, fmin=60, fmax=500, sr=sr,
                                          frame_length=2048, hop_length=512)
            if voiced is not None:
                f0v = f0[voiced & ~np.isnan(f0)]
                all_f0.extend(f0v.tolist())

        if len(all_f0) >= 5:
            med    = float(np.median(all_f0))
            gender = 'female_solo' if med >= f0_thr else 'male_solo'
            total_dur = sum(e - s for s, e in segs)
            print(f"   {spk}: F0 médian={med:.0f} Hz | {len(segs)} seg(s) | "
                  f"{total_dur:.1f}s => {gender}")
        else:
            gender = 'unknown'
            print(f"   {spk}: trop peu de trames voisées => unknown")
        speaker_gender[spk] = gender

    # Construire liste labelée
    labeled = [(turn.start, turn.end, speaker_gender.get(spk, 'unknown'))
               for turn, _, spk in diarization.itertracks(yield_label=True)]

    def _assemble(target_kinds):
        result, last_end = [], None
        for start, end, kind in labeled:
            if kind not in target_kinds or (end - start) < 0.2: continue
            if last_end is not None:
                gap = start - last_end
                n = (int(gap * sr)          if silence_mode == 'auto'
                     else 0                 if silence_mode == 0.0
                     else int(float(silence_mode) * sr))
                if n > 0: result.append(np.zeros(n, dtype=np.float32))
            chunk = y[int(start * sr):int(end * sr)].copy()
            fade  = min(int(0.02 * sr), len(chunk) // 4)
            if fade > 0:
                chunk[:fade]  *= np.linspace(0, 1, fade)
                chunk[-fade:] *= np.linspace(1, 0, fade)
            result.append(chunk)
            last_end = end
        return np.concatenate(result) if result else None

    if split_output:
        base, ext = os.path.splitext(output_file)
        for tag, kinds in [('female', {'female_solo'}), ('male', {'male_solo'})]:
            arr = _assemble(kinds)
            if arr is not None:
                save_audio(arr, sr, base + f'_{tag}' + ext, mp3_bitrate, mp3_mode)
            else:
                print(f"[!] Aucun segment {tag} trouvé.")
    else:
        arr = _assemble(keep_set)
        if arr is not None: save_audio(arr, sr, output_file, mp3_bitrate, mp3_mode)
        else: print("[!] Aucun segment gardé.")


# ── Clustering global des locuteurs (k-means sur F0) ────────────────────────

def cluster_speakers(y, sr, segs, ov_range=80, device='cpu'):
    """
    Passe globale sur tous les segments pour collecter les F0 médians,
    puis k-means k=2 pour séparer les deux locuteurs.

    Retourne un dict {seg_idx: 'female'|'male'|'silence'} et un résumé
    (centroide_female_hz, centroide_male_hz).

    Avantage vs seuil fixe : s'adapte automatiquement aux deux voix
    présentes dans l'enregistrement, même en zone ambiguë.
    """
    from scipy.cluster.vq import kmeans2

    print("   [*] Clustering global F0 (k-means k=2)...")

    # -- Collecte F0 median par segment --------------------------------------
    seg_f0 = []   # (seg_idx, f0_median, voiced_ratio)
    for i, (start, end) in enumerate(segs):
        seg = y[int(start * sr):int(end * sr)]
        if len(seg) < 1024:
            seg_f0.append((i, None, 0.0))
            continue

        if device == 'cuda':
            f0, voiced, vr, _ = _f0_crepe(seg, sr, 'cuda')
        else:
            f0, voiced, vr, _ = _f0_pyin(seg, sr)

        if vr < 0.12:
            seg_f0.append((i, None, vr))
            continue

        f0v = f0[voiced & ~np.isnan(f0)]
        if len(f0v) < 2:
            seg_f0.append((i, None, vr))
            continue

        med = float(np.median(f0v))
        seg_f0.append((i, med, vr))

        print(f"   [*] Collecte F0: {i+1}/{len(segs)}\r", end="", flush=True)

    print()

    # -- Filtrer les segments avec F0 valide ---------------------------------
    valid = [(i, f0) for i, f0, vr in seg_f0 if f0 is not None]
    if len(valid) < 4:
        print("   [!] Pas assez de segments voises pour le clustering -> seuil fixe")
        return None, (None, None)

    idxs    = np.array([v[0] for v in valid])
    f0_vals = np.array([v[1] for v in valid], dtype=np.float64)

    # -- K-means k=2 sur F0 -------------------------------------------------
    # Normalisation pour kmeans2
    mu, sigma = f0_vals.mean(), f0_vals.std() + 1e-8
    f0_norm   = ((f0_vals - mu) / sigma).reshape(-1, 1)

    try:
        # minit='points' : initialise les centroïdes sur des points existants
        centroids, labels = kmeans2(f0_norm, 2, minit='points', iter=30)
    except Exception as e:
        print(f"   [!] kmeans2 error: {e} -> seuil fixe")
        return None, (None, None)

    # -- Identifier quel cluster est female (F0 plus élevé) ------------------
    c0_mean = float(np.mean(f0_vals[labels == 0]))
    c1_mean = float(np.mean(f0_vals[labels == 1]))
    female_cluster = 0 if c0_mean > c1_mean else 1
    male_cluster   = 1 - female_cluster

    f_centroid = c0_mean if female_cluster == 0 else c1_mean
    m_centroid = c0_mean if male_cluster   == 0 else c1_mean

    print(f"   [OK] Clustering: femme={f_centroid:.0f} Hz  homme={m_centroid:.0f} Hz")
    print(f"        (delta={f_centroid-m_centroid:.0f} Hz, "
          f"{sum(labels==female_cluster)} seg F / {sum(labels==male_cluster)} seg H)")

    # -- Construire le dict de labels ----------------------------------------
    result = {}
    for j, (seg_idx, _) in enumerate(valid):
        result[seg_idx] = 'female' if labels[j] == female_cluster else 'male'

    # Les segments silence/pas de F0
    for i, f0, _ in seg_f0:
        if f0 is None:
            result[i] = 'silence'

    return result, (f_centroid, m_centroid)


def cluster_speakers_ecapa(y, sr, segs, device='cpu'):
    """
    SPEAKER-based clustering: ECAPA-TDNN embedding per segment + k-means k=2.

    Unlike F0 clustering, this separates by WHO is speaking, not by pitch —
    so two women, or a high-pitched man vs a low-pitched woman, are separable
    (pitch-based separation fails there by construction). Clusters are then
    labelled female/male from their median F0 so --keep female/male keeps
    working. Same return contract as cluster_speakers:
      ({seg_idx: 'female'|'male'|'silence'}, (f0_female_hz, f0_male_hz))
    Falls back to F0 clustering if speechbrain/ECAPA is unavailable.
    """
    try:
        from speaker_identity import SpeakerEncoder
        enc = SpeakerEncoder(device=device)
    except Exception as e:
        print(f"   [!] ECAPA unavailable ({e}) -> F0 clustering fallback")
        return cluster_speakers(y, sr, segs, device=device)
    from scipy.signal import resample_poly
    from scipy.cluster.vq import kmeans2
    from math import gcd

    g16 = gcd(int(sr), 16000)
    embs, idxs = [], []
    print(f"   [*] ECAPA embeddings on {len(segs)} segments...")
    for i, (start, end) in enumerate(segs):
        if end - start < 0.5:                      # too short for a reliable embedding
            continue
        seg = y[int(start * sr):int(end * sr)]
        s16 = resample_poly(seg, 16000 // g16, sr // g16).astype(np.float32)
        try:
            e = np.asarray(enc.embed(s16, sr=16000), dtype=np.float64).ravel()
        except Exception:
            continue
        embs.append(e / (np.linalg.norm(e) + 1e-9))
        idxs.append(i)
        print(f"   [*] {len(embs)} embedded\r", end="", flush=True)
    print()
    if len(embs) < 4:
        print("   [!] Too few voiced segments for ECAPA clustering -> F0 fallback")
        return cluster_speakers(y, sr, segs, device=device)

    E = np.stack(embs)
    np.random.seed(0)                              # deterministic k-means init
    try:
        centroids, labels = kmeans2(E, 2, minit='++', iter=50)
    except Exception as e:
        print(f"   [!] kmeans2 error: {e} -> F0 fallback")
        return cluster_speakers(y, sr, segs, device=device)

    c0 = centroids[0] / (np.linalg.norm(centroids[0]) + 1e-9)
    c1 = centroids[1] / (np.linalg.norm(centroids[1]) + 1e-9)
    inter = float(np.dot(c0, c1))
    if inter > 0.85:
        print(f"   [!] Clusters very similar (cosine {inter:.2f}) — this recording "
              f"probably has a SINGLE speaker; the split will be arbitrary.")

    # Label clusters female/male from median F0 of a few representative segments
    def _cluster_f0(k, cen):
        members = [j for j in range(len(idxs)) if labels[j] == k]
        members.sort(key=lambda j: -float(np.dot(E[j], cen)))   # closest first
        vals = []
        for j in members[:6]:
            s, e_ = segs[idxs[j]]
            chunk = y[int(s * sr):int(e_ * sr)]
            try:
                f0r, vr, _ = librosa.pyin(chunk.astype(np.float32), fmin=60, fmax=500, sr=sr)
                fv = f0r[vr & ~np.isnan(f0r)] if vr is not None else []
                if len(fv) > 2:
                    vals.append(float(np.median(fv)))
            except Exception:
                pass
        return float(np.median(vals)) if vals else None

    f0_c0, f0_c1 = _cluster_f0(0, c0), _cluster_f0(1, c1)
    if f0_c0 is None and f0_c1 is None:
        print("   [!] Could not measure cluster F0 — labelling arbitrarily (0=female).")
        female_cluster, f_hz, m_hz = 0, 0.0, 0.0
    else:
        if f0_c0 is None: f0_c0 = (f0_c1 or 165) - 1
        if f0_c1 is None: f0_c1 = (f0_c0 or 165) - 1
        female_cluster = 0 if f0_c0 >= f0_c1 else 1
        f_hz = f0_c0 if female_cluster == 0 else f0_c1
        m_hz = f0_c1 if female_cluster == 0 else f0_c0

    dur = lambda k: sum(segs[idxs[j]][1] - segs[idxs[j]][0]
                        for j in range(len(idxs)) if labels[j] == k)
    print(f"   [OK] ECAPA clustering: female={f_hz:.0f} Hz ({dur(female_cluster):.0f}s)  "
          f"male={m_hz:.0f} Hz ({dur(1 - female_cluster):.0f}s)  "
          f"inter-cluster cosine {inter:.2f}")

    result = {i: 'silence' for i in range(len(segs))}
    for j, seg_idx in enumerate(idxs):
        result[seg_idx] = 'female' if labels[j] == female_cluster else 'male'
    return result, (f_hz, m_hz)


# ── Process principal (méthode F0) ───────────────────────────────────────────

def process(input_file, output_file, keep_set=None, silence_mode='auto',
            deverb_method='none', f0_thr=165, ov_range=80, min_dur=0.2,
            debug=False, remove_music=False, demucs_model='htdemucs_ft', device='cpu',
            mp3_bitrate=192, mp3_mode='cbr', min_silence=0.15, split_output=False,
            demucs_shifts=2, cluster_method='f0'):

    if keep_set is None:
        keep_set = {'female_solo'}

    print(f"[*]  {input_file}")
    print(f"   [*] Device: {device}")

    # -- Demucs music removal -------------------------------------------------
    if remove_music:
        y, sr = remove_music_demucs(input_file, demucs_model, device, shifts=demucs_shifts)
        if y is None:
            print("[!] Music removal failed — chargement direct")
            y, sr = librosa.load(input_file, sr=None, mono=True)
        else:
            print("[OK] Musique supprimée. Sauvegarde des voix (sans filtrage F0).")
            save_audio(y, sr, output_file, mp3_bitrate, mp3_mode)
            return
    else:
        y, sr = librosa.load(input_file, sr=None, mono=True)

    total_dur = len(y) / sr
    sil_str   = ('durée naturelle' if silence_mode == 'auto'
                 else 'sans silence' if silence_mode == 0.0
                 else f'{silence_mode}s fixe')
    print(f"   Durée:{total_dur:.1f}s  Keep:{keep_set}  Silence:{sil_str}  Dereverb:{deverb_method}")

    # -- Dereverbération -------------------------------------------------------
    if deverb_method != 'none':
        y = dereverberate(y, sr, deverb_method, device)
        if 'vocals_only' in keep_set:
            print("[OK] Dereverberation terminée. Sauvegarde directe.")
            save_audio(y, sr, output_file, mp3_bitrate, mp3_mode)
            return

    # -- Détection segments ---------------------------------------------------
    segs = detect_segments(y, sr, min_silence=min_silence)
    print(f"   {len(segs)} segments détectés")

    if not segs:
        print("[!] Aucun segment détecté."); return

    # Silence fixe en samples
    sil_n = None if silence_mode == 'auto' else (0 if silence_mode == 0.0 else int(float(silence_mode) * sr))

    # -- Traitement split : clustering global F0 (k-means k=2) ----------------
    # Plus robuste qu'un seuil fixe : s'adapte automatiquement aux deux voix.
    if split_output:
        # Passe 1 : clustering global
        if cluster_method == 'ecapa':
            seg_labels, (f_cent, m_cent) = cluster_speakers_ecapa(y, sr, segs, device)
        else:
            seg_labels, (f_cent, m_cent) = cluster_speakers(y, sr, segs, ov_range, device)

        if seg_labels is None:
            # Fallback : logique inverse seuil fixe
            print("   [!] Fallback seuil fixe pour le split")
            seg_labels = {}
            for i, (start, end) in enumerate(segs):
                kind, f0, _ = classify(y, sr, start, end, f0_thr, ov_range, device)
                seg_labels[i] = ('female' if kind == 'female_solo'
                                 else 'silence' if kind == 'silence'
                                 else 'male')

        results   = {'female': [], 'male': []}
        last_kept = {'female': None, 'male': None}

        for i, (start, end) in enumerate(segs):
            dur    = end - start
            target = seg_labels.get(i, 'silence')

            if debug:
                kind_raw, f0_raw, cent_raw = classify(y, sr, start, end, f0_thr, ov_range, device)
                f0_s   = f"{f0_raw:.0f}"   if f0_raw   is not None else "  -"
                cent_s = f"{cent_raw:.0f}" if cent_raw is not None else "  -"
                print(f"   {start:6.1f}-{end:6.1f}s  cluster={target:6s}  "
                      f"F0={f0_s:>6} Hz  Cent={cent_s:>6} Hz")

            if dur < min_dur or target == 'silence':
                continue

            lk = last_kept[target]
            if lk is not None:
                gap = start - lk
                n   = (int(gap * sr) if silence_mode == 'auto'
                       else 0 if silence_mode == 0.0
                       else sil_n)
                if n and n > 0:
                    results[target].append(np.zeros(n, dtype=np.float32))

            chunk = y[int(start * sr):int(end * sr)].copy()
            fade  = min(int(0.02 * sr), len(chunk) // 4)
            if fade > 0:
                chunk[:fade]  *= np.linspace(0, 1, fade)
                chunk[-fade:] *= np.linspace(1, 0, fade)
            results[target].append(chunk)
            last_kept[target] = end

        base, ext = os.path.splitext(output_file)
        for tag in ('female', 'male'):
            r = results[tag]
            if r:
                save_audio(np.concatenate(r), sr, base + f'_{tag}' + ext, mp3_bitrate, mp3_mode)
            else:
                print(f"[!] Aucun segment {tag} trouve.")

    # -- Traitement simple (keep_set) -----------------------------------------
    else:
        result      = []
        last_kept_end = None

        # ECAPA mode: one global speaker clustering, then map to keep labels.
        seg_labels_ec = None
        if cluster_method == 'ecapa' and keep_set and (keep_set & {'female_solo', 'male_solo'}):
            seg_labels_ec, _ = cluster_speakers_ecapa(y, sr, segs, device)

        for i, (start, end) in enumerate(segs):
            dur = end - start
            if seg_labels_ec is not None:
                lab  = seg_labels_ec.get(i, 'silence')
                kind = ('female_solo' if lab == 'female'
                        else 'male_solo' if lab == 'male' else 'silence')
                f0 = cent = None
            else:
                kind, f0, cent = classify(y, sr, start, end, f0_thr, ov_range, device)

            if debug:
                f0_s   = f"{f0:.0f}"   if f0   is not None else "  -"
                cent_s = f"{cent:.0f}" if cent is not None else "  -"
                mark   = '[OK]' if kind in keep_set else '-> skip'
                print(f"   {start:6.1f}-{end:6.1f}s  {kind:14s}  F0={f0_s:>6} Hz  Cent={cent_s:>6} Hz  {mark}")

            if kind in keep_set and dur >= min_dur:
                if last_kept_end is not None:
                    gap = start - last_kept_end
                    n   = (int(gap * sr) if silence_mode == 'auto'
                           else 0 if silence_mode == 0.0
                           else sil_n)
                    if n and n > 0:
                        result.append(np.zeros(n, dtype=np.float32))
                chunk = y[int(start * sr):int(end * sr)].copy()
                fade  = min(int(0.02 * sr), len(chunk) // 4)
                if fade > 0:
                    chunk[:fade]  *= np.linspace(0, 1, fade)
                    chunk[-fade:] *= np.linspace(1, 0, fade)
                result.append(chunk)
                last_kept_end = end

        if not result:
            print("[!] Aucun segment gardé !"); return

        if last_kept_end is not None and last_kept_end < total_dur:
            if silence_mode == 'auto':
                result.append(np.zeros(int((total_dur - last_kept_end) * sr), dtype=np.float32))
            elif sil_n and sil_n > 0:
                result.append(np.zeros(sil_n, dtype=np.float32))

        save_audio(np.concatenate(result), sr, output_file, mp3_bitrate, mp3_mode)


# ── Argument parsing ─────────────────────────────────────────────────────────

KEEP_ALIASES = {
    'female':      {'female_solo'},
    'male':        {'male_solo'},
    'overlap':     {'overlap'},
    'all':         {'female_solo', 'male_solo', 'overlap'},
    'vocals only': {'vocals_only'},
    'vocals':      {'vocals_only'},
}

def parse_keep(s):
    result = set()
    for p in s.split(','):
        p = p.strip().lower()
        if p in KEEP_ALIASES:    result |= KEEP_ALIASES[p]
        elif p in ('female_solo', 'male_solo'): result.add(p)
        else: raise argparse.ArgumentTypeError(f"Inconnu: {p}")
    return result

def parse_silence(s):
    s = s.strip().lower()
    if s == 'auto': return 'auto'
    v = float(s)
    if v < 0: raise argparse.ArgumentTypeError("Valeur négative")
    return v


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)

    p.add_argument("input",  help="Fichier audio/vidéo source")
    p.add_argument("output", nargs="?", default=None,
                   help="Fichier de sortie (non requis avec --analyze)")

    p.add_argument("--keep", type=parse_keep, default={'female_solo'},
                   help="Voix à conserver: female, male, overlap, all, female,male (défaut: female)")
    p.add_argument("--silence", type=parse_silence, default='auto',
                   help="Silence entre segments: auto | 0 | N secondes (défaut: auto)")
    p.add_argument("--threshold", type=int, default=165,
                   help="Seuil F0 Hz pour séparer F/H (défaut: 165)")
    p.add_argument("--overlap-range", type=int, default=80,
                   help="Plage F0 Hz au-delà de laquelle = overlap (défaut: 80)")
    p.add_argument("--min-dur", type=float, default=0.2,
                   help="Durée minimale d'un segment gardé en secondes (défaut: 0.2)")
    p.add_argument("--min-silence", type=float, default=0.15,
                   help="Durée minimale de silence pour séparer deux segments (défaut: 0.15)")
    p.add_argument("--dereverberate", choices=['none', 'noisereduce', 'wpe', 'deepfilter'],
                   default='none')
    p.add_argument("--method", choices=['f0', 'ecapa', 'sepformer', 'pyannote'], default='f0',
                   help="Methode: f0 (defaut) | sepformer (separation neuronale, recommande) | pyannote (requiert --hf-token)")
    p.add_argument("--hf-token", default="",
                   help="Token HuggingFace pour pyannote.audio")
    p.add_argument("--analyze", action="store_true",
                   help="Affiche stats par segment sans produire de sortie")
    p.add_argument("--split-output", action="store_true",
                   help="Génère OUTPUT_female.ext + OUTPUT_male.ext simultanément")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--mp3-bitrate", type=int, default=192,
                   choices=[128, 160, 192, 256, 320])
    p.add_argument("--mp3-mode", choices=["cbr", "vbr"], default="cbr")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--remove-music", action="store_true")
    p.add_argument("--demucs-model", default="htdemucs_ft",
                   choices=["htdemucs", "htdemucs_ft", "mdx_extra"])
    p.add_argument("--demucs-shifts", type=int, default=2,
                   help="demucs test-time augmentation passes (1=fast, 2=default, "
                        "5=best quality; separation time scales with it)")
    p.add_argument("--age", default=None,
                   choices=list(AGE_PRESETS.keys()),
                   help="Re-age the voice: child | teen | younger | older | "
                        "much_older. Moves the fundamental AND the formants "
                        "(vocal-tract length), which is what actually reads as "
                        "age — a pitch shift alone just sounds speeded up.")
    p.add_argument("--age-sex", default='F', choices=['F', 'M'],
                   help="Which preset table to use (default F). The same shift "
                        "does not read the same way on a female and a male voice.")
    p.add_argument("--pitch-shift", type=float, default=None, metavar='ST',
                   help="Manual fundamental shift in semitones, overrides --age")
    p.add_argument("--preserve-formants", action='store_true',
                   help="Keep the original timbre while shifting pitch. This "
                        "CANCELS the age effect (the formants carry it) — use it "
                        "only to retune a voice without changing who it sounds like.")
    p.add_argument("--shorten-silences", type=float, default=None, metavar='MS',
                   help="Shorten silent gaps to MS milliseconds instead of "
                        "splitting the file into segments. Nothing is discarded, "
                        "so no speech can be cut — use this on a source with long "
                        "pauses (e.g. --shorten-silences 500).")
    p.add_argument("--shorten-min-gap", type=float, default=600.0, metavar='MS',
                   help="Only gaps at least this long are shortened (default 600 ms), "
                        "so natural breaths are left alone.")
    p.add_argument("--tempo", type=float, default=1.0,
                   help="Time-stretch factor, pitch preserved (e.g. 0.85 slower, "
                        "1.25 faster; 1.0 = unchanged)")
    args = p.parse_args()

    _TEMPO = float(args.tempo)
    if abs(_TEMPO - 1.0) > 1e-3:
        print(f"[*] Tempo ×{_TEMPO} (pitch preserved) will be applied to outputs.")

    # Validate output requirement
    if not args.analyze and args.output is None:
        p.error("output est requis sauf avec --analyze")

    # Video input → extract audio
    VIDEO_EXT = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4a', '.m4v', '.ts', '.wmv', '.flv'}
    if os.path.splitext(args.input)[1].lower() in VIDEO_EXT:
        _tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        _tmp.close()
        print(f"[*] Vidéo détectée — extraction audio...")
        subprocess.run(['ffmpeg', '-y', '-i', args.input, '-vn', '-ac', '1', '-ar', '16000', _tmp.name],
                       check=True, capture_output=True)
        print(f"[*] Audio extrait: {_tmp.name}")
        args.input = _tmp.name

    # Mode analyse
    if args.analyze:
        analyze(args.input, args.threshold, args.overlap_range, args.min_silence, args.device)

    # Shorten silences: keep the whole take, compress only the gaps. Runs
    # instead of the segment pipeline, because the two answer different needs —
    # segmenting sorts speakers, shortening cleans up a talky source.
    elif args.shorten_silences is not None:
        import soundfile as _sf
        _y, _sr = librosa.load(args.input, sr=None, mono=True)
        print(f"[*]  {args.input}")
        print(f"   [*] Source: {len(_y) / _sr:.1f}s")
        _out = shorten_silences(_y, _sr,
                                target_ms=args.shorten_silences,
                                min_gap_ms=args.shorten_min_gap)
        _dest = args.output
        if _dest and not os.path.splitext(_dest)[1]:
            _dest += '.wav'
            print(f"   [!] No extension given -> saving as {os.path.basename(_dest)}")
        save_audio(_out, _sr, _dest, args.mp3_bitrate, args.mp3_mode)
        print("[OK] Done.")

    # Age shift: a straight transform of the whole take, no segmenting.
    elif args.age or args.pitch_shift is not None:
        _y, _sr = librosa.load(args.input, sr=None, mono=True)
        print(f"[*]  {args.input}")
        if args.age:
            _st = AGE_PRESETS[args.age][args.age_sex]
            print(f"   [*] Preset '{args.age}' ({args.age_sex}): pitch {_st:+g} st")
        else:
            _st = 0.0
        if args.pitch_shift is not None:
            _st = args.pitch_shift
        _f0h = None
        try:
            _f0, _vf, _ = librosa.pyin(_y[:int(30 * _sr)], fmin=60, fmax=400, sr=_sr)
            _fv = _f0[_vf & ~np.isnan(_f0)]
            if len(_fv):
                _f0h = float(np.median(_fv))
                print(f"   [*] Source F0: {_f0h:.0f} Hz")
        except Exception:
            pass
        _w = age_warning(_st, _f0h)
        if _w:
            print(f"   [!] {_w}")
        _out = shift_age(_y, _sr, semitones=_st,
                         preserve_formants=args.preserve_formants)
        _dest = args.output
        if _dest and not os.path.splitext(_dest)[1]:
            _dest += '.wav'
            print(f"   [!] No extension given -> saving as {os.path.basename(_dest)}")
        save_audio(_out, _sr, _dest, args.mp3_bitrate, args.mp3_mode)
        print("   [*] LISTEN before cloning: past about 3 semitones the vocal "
              "tract stops sounding plausible.")
        print("[OK] Done.")

    # Méthode SepFormer (séparation neuronale réelle)
    elif args.method == 'sepformer':
        process_sepformer(args.input, args.output, args.keep, args.silence,
                          args.dereverberate, args.threshold, args.overlap_range,
                          args.min_dur, args.debug, args.device,
                          args.mp3_bitrate, args.mp3_mode, args.min_silence, args.split_output)

    # Méthode pyannote
    elif args.method == 'pyannote':
        process_pyannote(args.input, args.output, args.keep, args.silence,
                         args.dereverberate, args.threshold, args.hf_token,
                         args.device, args.mp3_bitrate, args.mp3_mode, args.split_output)

    # Méthode F0 (défaut)
    else:
        process(args.input, args.output, args.keep, args.silence, args.dereverberate,
                args.threshold, args.overlap_range, args.min_dur, args.debug,
                args.remove_music, args.demucs_model, args.device,
                args.mp3_bitrate, args.mp3_mode, args.min_silence, args.split_output,
                demucs_shifts=args.demucs_shifts,
                cluster_method=('ecapa' if args.method == 'ecapa' else 'f0'))

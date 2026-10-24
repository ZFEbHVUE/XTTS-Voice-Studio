#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Voice routing and optional two-source separation.

f0: short pitch regions, with an unknown category. Pitch is not speaker identity.
ecapa: speaker embeddings, optionally anchored by --female-ref / --male-ref.
sepformer: separate all speech blocks first, then route the estimated sources.
pyannote: speaker turns; mixed/uncertain audio is retained in _review.wav.

--split-output writes _female / _male, plus _review.wav when needed and a JSON
report. With --silence auto outputs keep the original timeline. With --silence 0
selected regions are concatenated. Solo reference labels are chosen by the user;
they override pitch assumptions and should contain only that target speaker.
"""

import numpy as np
try:
    import soundfile as sf
except ImportError:
    sf = None
try:
    import librosa
except ImportError:
    librosa = None
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
        # nara_wpe wants (F, D, T): frequency bins first, then channels. The old
        # code passed (1, F, T), i.e. ONE bin with 257 "channels": a 2570x2570
        # cross-frequency predictor per iteration — ~15x slower, and it made the
        # signal WORSE (SI-SDR -5 dB vs input on a synthetic room, RT60 0.6 s).
        # Window/hop are fixed in TIME (32 ms / 8 ms), not samples, so taps=10
        # always models 80 ms of tail whatever the rate (nara's 512/128 @ 16 kHz
        # reference tuning), at the native rate: no band above 8 kHz is lost.
        # Blocks bound memory: the solver builds a (bins, taps, frames) tensor
        # that grows to several GB on a long 48 kHz file. 2 iterations: the 3rd
        # gains < 0.1 dB for +40 % time.
        size = int(round(0.032 * sr)) & ~1
        shift = size // 4
        bs, ov, xf = int(20 * sr), int(1 * sr), int(0.02 * sr)
        n = len(y); nb = max(1, int(np.ceil(n / bs)))
        print(f"   [*] WPE: {nb} block(s) of 20 s @ {sr} Hz (win {size}, hop {shift})...")
        res = np.zeros(n, dtype=np.float32)
        zprev = sprev = None
        for i in range(nb):
            s = max(0, i * bs - ov); e = min(n, (i + 1) * bs + ov)
            Y = stft(y[s:e].astype(np.float64), size=size, shift=shift).T[:, None, :]
            Z = wpe(Y, taps=10, delay=3, iterations=2)[:, 0, :]
            z = istft(Z.T, size=size, shift=shift).astype(np.float32)
            z = np.pad(z, (0, max(0, (e - s) - len(z))))[: e - s]
            a, b = i * bs, min(n, (i + 1) * bs)
            seg = z[a - s: b - s].copy()
            if zprev is not None and xf < len(seg):
                w = np.linspace(0.0, 1.0, xf, dtype=np.float32)
                seg[:xf] = zprev[a - sprev: a - sprev + xf] * (1 - w) + seg[:xf] * w
            res[a:b] = seg
            zprev, sprev = z, s
        return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0)
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

def detect_segments(y,sr,min_silence=.15,min_speech=.2):
    from audio_checks import mono_audio
    y=mono_audio(y,sr,allow_silence=True)
    if min_silence<0 or min_speech<0 or not np.isfinite(min_silence+min_speech):
        raise ValueError('Speech and silence durations must be finite and nonnegative')
    hop=max(1,round(.01*sr))
    starts=np.arange(0,len(y),hop)
    rms=np.array([np.sqrt(np.mean(y[i:min(len(y),i+hop)]**2)) for i in starts])
    if not len(rms) or rms.max()<1e-5:
        return []
    gate=max(1e-5,float(np.percentile(rms,95))*.04,float(np.median(rms))*.3)
    active=rms>gate
    raw=[];begin=None
    for index,on in enumerate(np.r_[active,False]):
        if on and begin is None:
            begin=int(starts[index])
        if not on and begin is not None:
            end=min(len(y),index*hop)
            raw.append([begin/sr,end/sr]);begin=None
    merged=[]
    for start,end in raw:
        if merged and start-merged[-1][1]<=min_silence:
            merged[-1][1]=end
        else:
            merged.append([start,end])
    return [(start,end) for start,end in merged if end-start>=min_speech]


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
    seg_bytes, sr, threshold, overlap_range = args
    chunk = np.frombuffer(seg_bytes, dtype=np.float32).copy()
    label, median, centroid = classify(chunk, sr, 0, len(chunk)/sr, threshold, overlap_range)
    return label, median, centroid, 0


def classify(y, sr, start, end, f0_thr=165, ov_range=80, device='cpu'):
    from voice_routing import pitch_label
    chunk = y[int(start*sr):int(end*sr)]
    if len(chunk) < 1024 or np.max(np.abs(chunk), initial=0) < 1e-5:
        return 'silence', None, None
    estimate = _f0_crepe(chunk, sr, device) if device == 'cuda' else _f0_pyin(chunk, sr)
    f0, voiced, ratio, _ = estimate
    values = np.asarray(f0, float)
    values = values[np.asarray(voiced, bool) & np.isfinite(values)]
    if ratio < .12 or len(values) < 3:
        return 'unknown', None, None
    label, median = pitch_label(values, f0_thr)
    return (label + '_solo' if label in ('female', 'male') else label), median, None


# ── Mode Analyse — multicore CPU ou GPU ──────────────────────────────────────

def analyze(input_file,f0_thr=165,ov_range=80,min_silence=.15,device='cpu'):
    return analyze_routing(input_file,'f0',f0_thr,min_silence,device)


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


def _fullband_from_estimates(mix, est1, est2, sr, sep_sr):
    """Turn two band-limited source estimates into full-band sources.

    Soft ratio mask |E1| / (|E1| + |E2|) on the ORIGINAL mixture's STFT. Below
    ~0.45 x sep_sr the mask comes straight from the estimates; above it, each
    frame reuses the mean mask of the top band the model actually covered.
    Returns (out1, out2) at the original rate, same length as mix.
    """
    n = len(mix)
    n_fft = 2048 if sr >= 32000 else 1024
    hop = n_fft // 4
    if n < n_fft:
        return est1, est2
    S = librosa.stft(mix.astype(np.float32), n_fft=n_fft, hop_length=hop)
    A = np.abs(librosa.stft(est1.astype(np.float32), n_fft=n_fft, hop_length=hop))
    B = np.abs(librosa.stft(est2.astype(np.float32), n_fft=n_fft, hop_length=hop))
    m = min(S.shape[1], A.shape[1], B.shape[1])
    S, A, B = S[:, :m], A[:, :m], B[:, :m]
    m1 = A / (A + B + 1e-9)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    cutoff = 0.45 * sep_sr
    ref = (freqs >= 0.30 * sep_sr) & (freqs < cutoff)
    hi = freqs >= cutoff
    if ref.any() and hi.any():
        m1[hi, :] = m1[ref, :].mean(axis=0, keepdims=True)
    out1 = librosa.istft(S * m1, hop_length=hop, length=n).astype(np.float32)
    out2 = librosa.istft(S * (1.0 - m1), hop_length=hop, length=n).astype(np.float32)
    return out1, out2


def _sepformer_separate(model, seg, sr, debug_dir=None, seg_idx=0, restore_high_band=False):
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

    # The model runs at 8 kHz, so its outputs stop at 4 kHz. Used as-is they
    # made a cloning reference with no upper band at all. Instead, the two
    # estimates only DECIDE how the original full-band signal is shared out:
    # a ratio mask applied to the original spectrum keeps its full bandwidth
    # and phase. Above the model's band the mask is carried over, frame by
    # frame, from the top of the band the model did see.
    if restore_high_band:
        src1, src2 = _fullband_from_estimates(seg, src1, src2, sr, SEP_SR)

    # Sauvegarde debug (3 premiers segments)
    if debug_dir and seg_idx < 3:
        os.makedirs(debug_dir, exist_ok=True)
        sf.write(os.path.join(debug_dir, f"seg{seg_idx:03d}_mix.wav"),  seg,  sr)
        sf.write(os.path.join(debug_dir, f"seg{seg_idx:03d}_src1.wav"), src1, sr)
        sf.write(os.path.join(debug_dir, f"seg{seg_idx:03d}_src2.wav"), src2, sr)
        print(f"   [DBG] Seg {seg_idx} => {debug_dir}")

    return src1, src2



def _estimate_pitch(chunk, sr, device='cpu'):
    return _f0_crepe(chunk, sr, device) if device == 'cuda' else _f0_pyin(chunk, sr)


def _load_processed_audio(input_file, deverb_method='none', device='cpu',
                          remove_music=False, demucs_model='htdemucs_ft', demucs_shifts=2):
    from audio_checks import mono_audio
    if remove_music:
        y, sr = remove_music_demucs(input_file, demucs_model, device, shifts=demucs_shifts)
        if y is None:
            raise RuntimeError('Music removal failed; separation aborted rather than using the original mix')
    else:
        y, sr = librosa.load(input_file, sr=None, mono=True)
    y = mono_audio(y, sr, allow_silence=True).astype(np.float32)
    if deverb_method != 'none':
        y = dereverberate(y, sr, deverb_method, device)
    return mono_audio(y, sr, allow_silence=True).astype(np.float32), sr


def _references(female_ref=None, male_ref=None):
    refs = {key:os.path.abspath(os.path.expanduser(os.fspath(path)))
            for key,path in (('female',female_ref),('male',male_ref)) if path}
    for path in refs.values():
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Solo reference not found: {path}')
    if len(set(os.path.realpath(p) for p in refs.values())) != len(refs):
        raise ValueError('Use different solo reference files for the two speakers')
    return refs


def _route(y, sr, segs, method, device, threshold, references=None,
           minimum=.35, margin=.08):
    if not segs:
        return []
    from voice_routing import embedding_regions, pitch_regions
    estimate = lambda audio,rate:_estimate_pitch(audio,rate,device)
    if method == 'ecapa':
        from speaker_identity import SpeakerEncoder
        encoder = SpeakerEncoder(device=device)
        return embedding_regions(y,sr,segs,encoder,estimate,references,minimum,margin,threshold)
    if references:
        raise ValueError('Reference matching requires method ecapa, sepformer or pyannote')
    return pitch_regions(y,sr,segs,estimate,threshold)


def _export_regions(y,sr,regions,output_file,keep_set,silence_mode,split_output,
                    mp3_bitrate=192,mp3_mode='cbr',min_dur=.2,method='f0',references=None):
    from voice_routing import assemble,write_report
    base,ext=os.path.splitext(output_file)
    ext=ext or '.wav'
    os.makedirs(os.path.dirname(os.path.abspath(output_file)),exist_ok=True)
    targets = ([('female',{'female'}),('male',{'male'})] if split_output else
               [('',{label.removesuffix('_solo') for label in keep_set})])
    if keep_set == {'female_solo','male_solo','overlap'}:
        targets=[('',{'female','male','overlap','unknown'})] if not split_output else targets
    for suffix,labels in targets:
        out=assemble(y,sr,regions,labels,silence_mode,minimum=min_dur)
        destination=base+'_'+suffix+ext if suffix else output_file
        if out is None:
            if os.path.exists(destination):
                raise FileExistsError(f'No confident new audio for {suffix or sorted(labels)}. '
                                      f'Old output still exists: {destination}. Choose a new output name.')
            print(f'[!] No confident audio for {suffix or sorted(labels)}; no file written.')
        else:
            save_audio(out,sr,destination,mp3_bitrate,mp3_mode)
    review=assemble(y,sr,regions,{'unknown','overlap'},silence_mode,minimum=0)
    if review is not None:
        save_audio(review,sr,base+'_review.wav',mp3_bitrate,mp3_mode)
        print('[*] Ambiguous / simultaneous speech retained in _review.wav.')
    report=write_report(base+'_separation.json',regions,method,references)
    print('[*] Labelled seconds: '+', '.join(f'{key}={seconds:.1f}s' for key,seconds in sorted(report['durations'].items())))
    print('[*] Report: '+base+'_separation.json')
    if method != 'sepformer':
        print('[*] Routing keeps original samples. It does not unmix simultaneous voices.')


def _source_labels(sources,sr,threshold,matcher=None):
    from voice_routing import pitch_label
    energy=[float(np.sqrt(np.mean(np.asarray(source,float)**2))) for source in sources]
    loudest=max(energy,default=0)
    labels=[]
    for source,rms in zip(sources,energy):
        if rms < max(1e-5,loudest*.03):
            labels.append(('unknown',None,'silent_source'))
            continue
        if matcher:
            labels.append(matcher.match(source,sr))
        else:
            f0,voiced,ratio,_=_f0_pyin(source,sr)
            values=np.asarray(f0,float)
            label,median=pitch_label(values[np.asarray(voiced,bool)],threshold)
            labels.append((label,median,'separated_source_pitch'))
    return labels


def _pick_source(src1,src2,sr,target_gender,f0_thr):
    # Compatibility helper: uncertainty never means "take the louder source".
    target=target_gender.removesuffix('_solo')
    labels=_source_labels((src1,src2),sr,f0_thr)
    matches=[source for source,result in zip((src1,src2),labels) if result[0]==target]
    return np.sum(matches,axis=0) if matches else None


def process_sepformer(input_file,output_file,keep_set=None,silence_mode='auto',
                      deverb_method='none',f0_thr=165,ov_range=80,min_dur=.2,
                      debug=False,device='cpu',mp3_bitrate=192,mp3_mode='cbr',
                      min_silence=.15,split_output=False,female_ref=None,male_ref=None,
                      match_minimum=.35,match_margin=.08,remove_music=False,
                      demucs_model='htdemucs_ft',demucs_shifts=2,restore_high_band=False):
    from voice_routing import Region,ReferenceMatcher,assemble,write_report
    keep_set=keep_set or {'female_solo'}
    if keep_set=={'overlap'} and not split_output:
        raise ValueError('SepFormer does not locate overlapping turns; choose female, male or split-output')
    if keep_set=={'vocals_only'} and split_output:
        raise ValueError('vocals only cannot be combined with split-output')
    refs=_references(female_ref,male_ref)
    matcher=None
    if refs:
        from speaker_identity import SpeakerEncoder
        matcher=ReferenceMatcher(SpeakerEncoder(device=device),refs,match_minimum,match_margin)
    y,sr=_load_processed_audio(input_file,deverb_method,device,remove_music,demucs_model,demucs_shifts)
    if keep_set=={'vocals_only'} and not split_output:
        save_audio(y,sr,output_file,mp3_bitrate,mp3_mode)
        return
    print('[*] SepFormer: processing every speech block, including overlapping voices.')
    print('[*] Native separator band is 8 kHz audio (upper band limited); listen before cloning.')
    if restore_high_band:
        print('[!] High-band masking is approximate and can reintroduce the other voice.')
    model=_load_sepformer(device)  # failure is explicit: never silently use F0 instead
    speech=detect_segments(y,sr,min_silence=min_silence)
    chunks=[]
    for start,end in speech:
        count=max(1,int(np.ceil((end-start)/8)))
        edges=np.linspace(start,end,count+1)
        chunks.extend(zip(edges[:-1],edges[1:]))
    stems={key:np.zeros_like(y) for key in ('female','male','unknown')}
    regions=[]
    for index,(start,end) in enumerate(chunks):
        first,last=round(start*sr),min(len(y),round(end*sr))
        # Context avoids sharp model restarts at adjacent block boundaries.
        left,right=max(0,first-round(.2*sr)),min(len(y),last+round(.2*sr))
        debug_dir=os.path.join(os.path.dirname(output_file),'sepformer_debug') if debug else None
        sources=_sepformer_separate(model,y[left:right],sr,debug_dir,index,restore_high_band)
        sources=tuple(np.asarray(src[first-left:last-left],dtype=np.float32) for src in sources)
        if any(len(src)!=last-first or not np.isfinite(src).all() for src in sources):
            raise ValueError(f'Invalid separated audio in block {index+1}')
        labels=_source_labels(sources,sr,f0_thr,matcher)
        # Both source assignments use the same separation call, never a second pass.
        for source,(label,score,reason) in zip(sources,labels):
            if reason=='silent_source':
                continue
            stems[label][first:last]+=source
            regions.append(Region(float(start),float(end),label,reason,score))
        print(f'[*] Separated block {index+1}/{len(chunks)}')
    peak=float(np.max(sum((np.abs(stem) for stem in stems.values()),np.zeros_like(y)),initial=0))
    gain=min(1.,.98/max(peak,1e-9))
    if gain<1:
        print(f'[*] Common output attenuation {20*np.log10(gain):.1f} dB to prevent clipping')
    base,ext=os.path.splitext(output_file);ext=ext or '.wav'
    os.makedirs(os.path.dirname(os.path.abspath(output_file)),exist_ok=True)
    targets=([('female',{'female'}),('male',{'male'})] if split_output else
             [('',{label.removesuffix('_solo') for label in keep_set})])
    if keep_set=={'female_solo','male_solo','overlap'} and not split_output:
        targets=[('',{'female','male','unknown'})]
    for suffix,wanted in targets:
        if 'overlap' in wanted:
            raise ValueError('SepFormer outputs sources; use --keep female,male or --split-output, not overlap/all')
        selected=[region for region in regions if region.label in wanted]
        combined=sum((stems[label] for label in wanted if label in stems),np.zeros_like(y))*gain
        out=assemble(combined,sr,selected,wanted,silence_mode,minimum=min_dur)
        if out is not None:
            save_audio(out,sr,base+'_'+suffix+ext if suffix else output_file,mp3_bitrate,mp3_mode)
        else:
            print(f'[!] No confident separated source for {suffix or sorted(wanted)}')
    review=assemble(stems['unknown']*gain,sr,regions,{'unknown'},silence_mode)
    if review is not None:
        save_audio(review,sr,base+'_review.wav',mp3_bitrate,mp3_mode)
    write_report(base+'_separation.json',regions,'sepformer',refs)


def process_pyannote(input_file,output_file,keep_set,silence_mode,deverb_method,
                     f0_thr,hf_token,device='cpu',mp3_bitrate=192,mp3_mode='cbr',
                     split_output=False,female_ref=None,male_ref=None,match_minimum=.35,
                     match_margin=.08,remove_music=False,demucs_model='htdemucs_ft',
                     demucs_shifts=2,min_dur=.2,diarization_model='pyannote/speaker-diarization-3.1'):
    from pyannote.audio import Pipeline
    from voice_routing import ReferenceMatcher,partition_turns,pitch_label
    import torch,inspect
    if not hf_token:
        raise ValueError('Pyannote requires HF_TOKEN in the environment or --hf-token, plus model access')
    refs=_references(female_ref,male_ref)
    matcher=None
    if refs:
        from speaker_identity import SpeakerEncoder
        matcher=ReferenceMatcher(SpeakerEncoder(device=device),refs,match_minimum,match_margin)
    y,sr=_load_processed_audio(input_file,deverb_method,device,remove_music,demucs_model,demucs_shifts)
    keyword='token' if 'token' in inspect.signature(Pipeline.from_pretrained).parameters else 'use_auth_token'
    pipeline=Pipeline.from_pretrained(diarization_model,**{keyword:hf_token})
    if pipeline is None:
        raise RuntimeError('Diarization model unavailable; check model access and installed pyannote version')
    pipeline.to(torch.device(device))
    result=pipeline(dict(waveform=torch.from_numpy(y[None]),sample_rate=sr))
    annotation=getattr(result,'speaker_diarization',result)
    turns=[(max(0,float(turn.start)),min(len(y)/sr,float(turn.end)),str(speaker))
           for turn,_,speaker in annotation.itertracks(yield_label=True) if turn.end>turn.start]
    # Obtain truly solo intervals before identifying each speaker. Do not learn
    # a woman's identity or pitch from a mixed segment assigned to both people.
    solo=partition_turns(turns,{speaker:speaker for _,_,speaker in turns})
    labels={}
    for speaker in sorted({speaker for _,_,speaker in turns}):
        candidates=sorted((r for r in solo if r.label==speaker),key=lambda r:-(r.end-r.start))[:8]
        if matcher:
            embeddings=[matcher.encoder.embed(y[round(r.start*sr):round(min(r.end,r.start+5)*sr)],sr=sr)
                        for r in candidates if r.end-r.start>=.5]
            labels[speaker]=matcher.match_embedding(np.mean(embeddings,axis=0))[0] if embeddings else 'unknown'
        else:
            medians=[]
            for r in candidates:
                if r.end-r.start<.2:
                    continue
                f0,voiced,_,_=_estimate_pitch(y[round(r.start*sr):round(min(r.end,r.start+5)*sr)],sr,device)
                values=np.asarray(f0,float)[np.asarray(voiced,bool)]
                values=values[np.isfinite(values)]
                if len(values)>=3:
                    medians.append(float(np.median(values)))
            labels[speaker]=pitch_label(np.repeat(medians,3),f0_thr)[0]
        print(f'[*] {speaker}: {labels[speaker]}')
    regions=partition_turns(turns,labels)
    _export_regions(y,sr,regions,output_file,keep_set,silence_mode,split_output,
                    mp3_bitrate,mp3_mode,min_dur,'pyannote',refs)


def cluster_speakers(y,sr,segs,ov_range=80,device='cpu'):
    # Compatibility interface: never force a high/low k=2 split.
    labels={}
    pitches={'female':[],'male':[]}
    for index,(start,end) in enumerate(segs):
        kind,median,_=classify(y,sr,start,end,device=device)
        label=kind.removesuffix('_solo')
        labels[index]=label
        if label in pitches and median is not None:
            pitches[label].append(median)
    return labels,tuple(float(np.median(pitches[key])) if pitches[key] else None for key in ('female','male'))


def cluster_speakers_ecapa(y,sr,segs,device='cpu'):
    from speaker_identity import SpeakerEncoder
    from voice_routing import embedding_regions
    # Existing callers receive one label per interval; new processing uses finer cells.
    regions=embedding_regions(y,sr,segs,SpeakerEncoder(device=device),lambda c,r:_estimate_pitch(c,r,device))
    labels={}
    for index,(start,end) in enumerate(segs):
        durations={}
        for region in regions:
            overlap=max(0,min(end,region.end)-max(start,region.start))
            durations[region.label]=durations.get(region.label,0)+overlap
        labels[index]=max(durations,key=durations.get) if durations else 'unknown'
    return labels,(None,None)


def process(input_file,output_file,keep_set=None,silence_mode='auto',
            deverb_method='none',f0_thr=165,ov_range=80,min_dur=.2,
            debug=False,remove_music=False,demucs_model='htdemucs_ft',device='cpu',
            mp3_bitrate=192,mp3_mode='cbr',min_silence=.15,split_output=False,
            demucs_shifts=2,cluster_method='f0',female_ref=None,male_ref=None,
            match_minimum=.35,match_margin=.08):
    keep_set=keep_set or {'female_solo'}
    refs=_references(female_ref,male_ref)
    y,sr=_load_processed_audio(input_file,deverb_method,device,remove_music,demucs_model,demucs_shifts)
    if keep_set=={'vocals_only'}:
        if split_output:
            raise ValueError('vocals only cannot be combined with split-output')
        save_audio(y,sr,output_file,mp3_bitrate,mp3_mode)
        return
    segments=detect_segments(y,sr,min_silence=min_silence)
    regions=_route(y,sr,segments,cluster_method,device,f0_thr,refs,match_minimum,match_margin)
    if debug:
        for region in regions:
            print(f'{region.start:.2f}-{region.end:.2f}s {region.label}: {region.reason}')
    _export_regions(y,sr,regions,output_file,keep_set,silence_mode,split_output,
                    mp3_bitrate,mp3_mode,min_dur,cluster_method,refs)


def analyze_routing(input_file,method='f0',threshold=165,min_silence=.15,device='cpu',
                    female_ref=None,male_ref=None,minimum=.35,margin=.08):
    if method not in ('f0','ecapa'):
        raise ValueError('Analyze previews routing with f0/ecapa. Use Separate for sepformer/pyannote.')
    y,sr=_load_processed_audio(input_file)
    regions=_route(y,sr,detect_segments(y,sr,min_silence=min_silence),method,device,
                   threshold,_references(female_ref,male_ref),minimum,margin)
    for region in regions:
        print(f'{region.start:8.2f}-{region.end:8.2f}s {region.label:8} {region.reason}')
    print('[*] Pitch labels are heuristic; use solo references for speaker identification.')


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
    if not np.isfinite(v) or v < 0: raise argparse.ArgumentTypeError("Silence must be finite and nonnegative")
    return v


def main():
    global _TEMPO
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
                   help="Legacy compatibility option; pitch range no longer claims overlap")
    p.add_argument("--min-dur", type=float, default=0.2,
                   help="Durée minimale d'un segment gardé en secondes (défaut: 0.2)")
    p.add_argument("--min-silence", type=float, default=0.15,
                   help="Durée minimale de silence pour séparer deux segments (défaut: 0.15)")
    p.add_argument("--dereverberate", choices=['none', 'noisereduce', 'wpe', 'deepfilter'],
                   default='none')
    p.add_argument("--method", choices=['f0', 'ecapa', 'sepformer', 'pyannote'], default='f0',
                   help="f0: pitch heuristic | ecapa: speaker matching | sepformer: unmix sources | pyannote: diarization")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_HUB_TOKEN", "")),
                   help="Token HuggingFace pour pyannote.audio")
    p.add_argument('--female-ref', help='Solo example for the female output speaker')
    p.add_argument('--male-ref', help='Solo example for the male output speaker')
    p.add_argument('--match-minimum', type=float, default=.35, help='Heuristic minimum reference cosine')
    p.add_argument('--match-margin', type=float, default=.08, help='Heuristic gap between reference matches')
    p.add_argument('--restore-high-band', action='store_true', help='Approximate SepFormer high band from the mixture; may reintroduce voices')
    p.add_argument('--diarization-model', default='pyannote/speaker-diarization-3.1',
                   choices=['pyannote/speaker-diarization-3.1','pyannote/speaker-diarization-community-1'])
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
    args.input=os.path.abspath(os.path.expanduser(args.input))
    if args.output: args.output=os.path.abspath(os.path.expanduser(args.output))

    import math
    if sf is None or librosa is None:
        p.error('Audio dependencies missing: install soundfile and librosa in your XTTS environment')
    for name in ('tempo','threshold','min_dur','min_silence','match_minimum','match_margin'):
        if not math.isfinite(getattr(args,name)):
            p.error(f'{name} must be finite')
    if args.tempo<=0 or args.threshold<=0 or args.min_dur<0 or args.min_silence<0:
        p.error('Positive tempo/threshold and nonnegative segment durations are required')
    if not 0<=args.match_minimum<=1 or not 0<=args.match_margin<=2:
        p.error('match-minimum in [0,1] and match-margin in [0,2] required')
    try:
        _references(args.female_ref,args.male_ref)
    except (ValueError,OSError) as exc:
        p.error(str(exc))
    if (args.female_ref or args.male_ref) and args.method=='f0':
        p.error('Choose --method ecapa, sepformer or pyannote to use speaker references')
    if args.output:
        if not os.path.splitext(args.output)[1]: args.output += '.wav'
        protected=[args.input,args.female_ref,args.male_ref]
        base,ext=os.path.splitext(args.output)
        destinations=[args.output,base+'_female'+ext,base+'_male'+ext,base+'_review.wav']
        if any(os.path.realpath(dest)==os.path.realpath(src) for dest in destinations for src in protected if src):
            p.error('Output files must differ from source and speaker references')
    if not os.path.isfile(args.input):
        p.error(f'Source not found: {args.input}')
    if args.keep=={'overlap'} and args.method!='pyannote' and not args.split_output:
        p.error('Only pyannote detects overlap intervals; SepFormer separates sources, F0/ECAPA route voices')
    _TEMPO = float(args.tempo)
    if abs(_TEMPO - 1.0) > 1e-3:
        print(f"[*] Tempo ×{_TEMPO} (pitch preserved) will be applied to outputs.")

    # Validate output requirement
    if not args.analyze and args.output is None:
        p.error("output est requis sauf avec --analyze")

    temporary_video_audio=None
    try:
        # Video input → extract audio
        VIDEO_EXT = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4a', '.m4v', '.ts', '.wmv', '.flv'}
        if os.path.splitext(args.input)[1].lower() in VIDEO_EXT:
            _tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            _tmp.close()
            temporary_video_audio=_tmp.name
            print(f"[*] Vidéo détectée — extraction audio...")
            # Keep the source sample rate. Forcing 16 kHz here threw away
            # everything above 8 kHz before any processing, so a reference
            # extracted from a video was band-limited for good -- and XTTS
            # conditions on exactly that upper band for timbre.
            subprocess.run(['ffmpeg', '-y', '-i', args.input, '-vn', '-ac', '1',
                            '-c:a', 'pcm_s16le', _tmp.name],
                           check=True, capture_output=True)
            print(f"[*] Audio extrait: {_tmp.name}")
            args.input = _tmp.name

        # Mode analyse
        if args.analyze:
            analyze_routing(args.input,args.method,args.threshold,args.min_silence,args.device,
                            args.female_ref,args.male_ref,args.match_minimum,args.match_margin)

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

        # Whole vocal track is a preprocessing operation, independent of routing backend.
        elif args.keep=={'vocals_only'}:
            process(args.input,args.output,args.keep,args.silence,args.dereverberate,
                    remove_music=args.remove_music,demucs_model=args.demucs_model,
                    device=args.device,mp3_bitrate=args.mp3_bitrate,mp3_mode=args.mp3_mode,
                    demucs_shifts=args.demucs_shifts,split_output=args.split_output)

        # Méthode SepFormer (séparation neuronale réelle)
        elif args.method == 'sepformer':
            process_sepformer(args.input, args.output, args.keep, args.silence,
                              args.dereverberate, args.threshold, args.overlap_range,
                              args.min_dur, args.debug, args.device,
                              args.mp3_bitrate, args.mp3_mode, args.min_silence, args.split_output,
                              female_ref=args.female_ref,male_ref=args.male_ref,
                              match_minimum=args.match_minimum,match_margin=args.match_margin,
                              remove_music=args.remove_music,demucs_model=args.demucs_model,
                              demucs_shifts=args.demucs_shifts,restore_high_band=args.restore_high_band)

        # Méthode pyannote
        elif args.method == 'pyannote':
            process_pyannote(args.input, args.output, args.keep, args.silence,
                             args.dereverberate, args.threshold, args.hf_token,
                             args.device, args.mp3_bitrate, args.mp3_mode, args.split_output,
                             female_ref=args.female_ref,male_ref=args.male_ref,
                             match_minimum=args.match_minimum,match_margin=args.match_margin,
                             remove_music=args.remove_music,demucs_model=args.demucs_model,
                             demucs_shifts=args.demucs_shifts,min_dur=args.min_dur,
                             diarization_model=args.diarization_model)

        # Méthode F0 (défaut)
        else:
            process(args.input, args.output, args.keep, args.silence, args.dereverberate,
                    args.threshold, args.overlap_range, args.min_dur, args.debug,
                    args.remove_music, args.demucs_model, args.device,
                    args.mp3_bitrate, args.mp3_mode, args.min_silence, args.split_output,
                    demucs_shifts=args.demucs_shifts,
                    cluster_method=('ecapa' if args.method == 'ecapa' else 'f0'),
                    female_ref=args.female_ref,male_ref=args.male_ref,
                    match_minimum=args.match_minimum,match_margin=args.match_margin)
    finally:
        if temporary_video_audio and os.path.exists(temporary_video_audio):
            os.unlink(temporary_video_audio)


if __name__ == '__main__':
    main()

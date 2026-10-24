#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vowel_pitch.py -- vowel-level pitch editing for XTTS output, formants kept.

Two independent edits, usable together:

  * VOWEL MARKS in the script, written inside the word, right after the vowel:
        Dé{-2}tends tes é{+1}pau{-1}les.      fixed shift, in semitones
        relâ{0>-3}che                          glide from 0 to -3 across the vowel
        Dé{x1.8}tends                          vowel 1.8 times longer, pitch unchanged
        Dé{-2 x1.8}tends                       both
        Dé{d0.40}tends                         vowel lasting 0.40 s (written by --extract)
    The vowels are located with a forced alignment (Montreal Forced Aligner,
    French model), then the pitch contour of each marked vowel is rewritten.

  * CONTOUR (--contour): a calm, "settling" intonation applied to every phrase:
    the pitch glides down by --declination semitones from start to end of each
    phrase, and its range is narrowed around the phrase median (--compress).
    Needs no alignment and no annotation.

Pitch is modified with PSOLA (Praat, through Parselmouth): the pitch moves,
the formants -- the timbre -- stay where they are, unlike a plain resampling
or rubberband without formant preservation.

Usage
  # strip the marks to get the text XTTS must read
  python vowel_pitch.py --strip annotated.txt clean.txt

  # calm contour only
  python vowel_pitch.py in.wav out.wav --contour --declination 2 --compress 0.8

  # vowel marks, aligning with MFA installed in its own conda env "mfa"
  python vowel_pitch.py in.wav out.wav --script annotated.txt --mfa "conda run -n mfa mfa"

  # imitate a singer: 1) extract the melody of every vowel of the song
  python vowel_pitch.py song.wav --script song.txt --mfa "conda run -n mfa mfa" --extract song_annotated.txt --glide
  #   2) generate song.txt (marks stripped) with XTTS -> clone.wav, then apply the melody as TARGETS
  python vowel_pitch.py clone.wav clone_sung.wav --script song_annotated.txt --mfa "conda run -n mfa mfa" --target

  # word tags [p:N] written by the [Txt] tab are understood too (all vowels of the word)

  # another language (the MFA model must be downloaded: english_mfa, german_mfa...)
  python vowel_pitch.py in.wav out.wav --script annotated_en.txt --mfa "conda run -n mfa mfa" --lang en

  # vowel marks with an existing alignment (MFA or hand-made in Praat)
  python vowel_pitch.py in.wav out.wav --script annotated.txt --textgrid in.TextGrid

  # all together, plus a global shift that keeps the timbre
  python vowel_pitch.py in.wav out.wav --script annotated.txt --mfa "conda run -n mfa mfa" \\
                        --contour --global-shift -1

Dependencies: pip install praat-parselmouth   (MFA only for vowel marks)
"""
import argparse, os, re, shutil, subprocess, sys, tempfile, unicodedata
from difflib import SequenceMatcher

try:
    import parselmouth
    from parselmouth.praat import call
except ImportError:
    sys.exit("[ERR] parselmouth missing -> pip install praat-parselmouth")

# {-2}  {0>-3}  {x1.8}  {d0.42}  {-2 x1.8}  {+1.5 d0.42}
#   pitch in semitones (fixed or glide a>b), then optionally a duration:
#   xF = the vowel lasts F times longer, dS = the vowel lasts S seconds.
#   For chants and mantras: a NOTE instead of semitones, a length in BEATS of
#   the [tempo=N] line above:   O{C3 b4}m  Na{D3 b1}ma{E3 b1}  Shi{G3 b2}va{E3}
#   (note = letter A-G, optional # or b, octave; C4 = middle C = 261.6 Hz)
MARK = re.compile(r"\{(?=[^}]*\d)\s*(?:([A-G](?:#|b)?\d)|([+-]?\d+(?:\.\d+)?)(?:>([+-]?\d+(?:\.\d+)?))?)?"
                  r"\s*,?\s*(?:([xdb])(\d+(?:\.\d+)?))?\s*\}")
NOTE_ST = {"C": -9, "D": -7, "E": -5, "F": -4, "G": -2, "A": 0, "B": 2}
TEMPO = re.compile(r"\[tempo=(\d+(?:\.\d+)?)\]", re.I)


def note_hz(name, transpose=0.0):
    """'A4' -> 440 Hz, 'C#3', 'Eb3'... optionally transposed in semitones."""
    st = NOTE_ST[name[0]] + (1 if "#" in name else -1 if name[1:2] == "b" else 0)
    octave = int(name[-1])
    return 440.0 * 2 ** ((st + 12 * (octave - 4) + transpose) / 12.0)
OLD_TAG = re.compile(r"\[p:([+-]?\d+(?:\.\d+)?|\?)\]")
VOWEL_LETTERS = ("aeiouyàâäéèêëîïôöùûüÿœæáíóúãõåøı"   # Latin: fr en de es it pt nl tr...
                 "ąęóěůýőű"                           # pl cs hu
                 "аеёиоуыэюяіїє")                     # Cyrillic: ru uk
LANG = "fr"          # set by --lang: decides the silent-final-e rule
# XTTS v2 languages -> MFA models known to exist. For the others (it, nl, hu, ar, hi)
# list what MFA offers with "mfa model download acoustic" and pass --mfa-model.
MODELS = {"fr": "french_mfa", "en": "english_mfa", "de": "german_mfa", "es": "spanish_mfa",
          "pt": "portuguese_mfa", "ru": "russian_mfa", "pl": "polish_mfa", "sv": "swedish_mfa",
          "tr": "turkish_mfa", "cs": "czech_mfa", "ko": "korean_mfa", "ja": "japanese_mfa",
          "zh": "mandarin_mfa", "uk": "ukrainian_mfa"}
# Scripts without vowel letters: vowel marks cannot be placed, contour and
# global shift still work.
NO_VOWEL_MARKS = {"ja", "zh", "ko", "ar", "hi"}
IPA_VOWELS = set("aeiouyøœəɛɔɑɐɪʊʏɨʉɯɤɵɘɞʌæɶ")
EDGE = 0.015          # s of ramp at the vowel edges, so a shift never clicks


# --------------------------------------------------------------------------
# Script parsing
# --------------------------------------------------------------------------
def script_lines(path):
    """Speech lines of a generator script, marks kept, control syntax removed."""
    out = []
    for raw in open(path, encoding="utf-8").read().replace("\r", "").split("\n"):
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("{"):
            continue
        if re.match(r"^\[\d+(\s*,.*)?\]$", line):          # voice/audio block
            continue
        line = re.sub(r"\[(pause|music|parallel|/parallel|tempo)[^\]]*\]", " ", line, flags=re.I)
        line = line.strip()
        if line:
            out.append(line)
    return out


def tempo_per_line(path, default=60.0):
    """Tempo (beats per minute) in force for each speech line, in order."""
    out, tempo = [], default
    for raw in open(path, encoding="utf-8").read().replace("\r", "").split("\n"):
        for m in TEMPO.finditer(raw):
            tempo = float(m.group(1))
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("{") or re.match(r"^\[\d+(\s*,.*)?\]$", line):
            continue
        rest = re.sub(r"\[(pause|music|parallel|/parallel|tempo)[^\]]*\]", " ", line, flags=re.I).strip()
        if rest:
            out.append(tempo)
    return out


def strip_marks(text):
    return TEMPO.sub("", MARK.sub("", OLD_TAG.sub("", text)))


def tokens(text):
    """Words, apostrophes and hyphens split, lowercase, without punctuation."""
    t = strip_marks(text).lower().replace("’", "'")
    return [w for w in re.split(r"[^\wàâäéèêëîïôöùûüÿœæç]+", t) if w]


def vowel_groups(word):
    """Orthographic vowel groups of a word; a mute final e / es / ent is dropped
    when the word has another vowel, since it is usually not pronounced."""
    groups = [(m.start(), m.end()) for m in re.finditer(f"[{VOWEL_LETTERS}]+", word)]
    mute = {"fr": r"(e|es|ent)$", "en": r"(e|es)$"}.get(LANG)     # silent final e
    if mute and len(groups) > 1 and re.search(mute, word):
        s, e = groups[-1]
        if word[s:e] == "e":
            groups.pop()
    return groups


def marks_by_word(line):
    """[(token, [(vowel_index, a, b), ...]), ...] for one script line."""
    # Marks contain '-' and '>', so they are swapped for placeholders before the
    # line is split into words on spaces, apostrophes and hyphens.
    found = []
    def keep(m):
        found.append(m)
        return f"\x00{len(found) - 1}\x00"
    line = OLD_TAG.sub(keep, MARK.sub(keep, line.replace("’", "'")))
    res = []
    for raw in re.split(r"[\s'\-]+", line):
        if not raw:
            continue
        clean, marks = "", []
        for part in re.split(r"(\x00\d+\x00)", raw):
            if part.startswith("\x00"):
                m = found[int(part.strip("\x00"))]
                if m.re is OLD_TAG:                      # word-level tag from the [Txt] tab
                    if m.group(1) != "?":
                        marks.append((-1, float(m.group(1)), float(m.group(1)), None))
                    continue
                note = m.group(1)
                a = float(m.group(2)) if m.group(2) else None
                b = float(m.group(3)) if m.group(3) else a
                dur = (m.group(4), float(m.group(5))) if m.group(4) else None
                if note:                                  # absolute note: kept as text, resolved later
                    a = b = note
                letters = re.sub(r"[^\w]", "", clean.lower())
                k = sum(1 for s_, e_ in vowel_groups(letters + "x") if e_ <= len(letters)) - 1
                marks.append((max(k, 0), a, b, dur))
            else:
                clean += part
        tok = re.sub(r"[^\w]", "", clean.lower())
        if tok:
            res.append((tok, marks))
    return res


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------
def read_textgrid(path):
    """{'words': [(t0, t1, label)], 'phones': [...]} from a TextGrid."""
    tg = call("Read from file", path)
    tiers = {}
    for i in range(1, call(tg, "Get number of tiers") + 1):
        name = call(tg, "Get tier name", i).lower()
        if not call(tg, "Is interval tier", i):
            continue
        items = []
        for j in range(1, call(tg, "Get number of intervals", i) + 1):
            lab = call(tg, "Get label of interval", i, j).strip()
            if lab and lab not in ("sil", "sp", "spn", "<eps>"):
                items.append((call(tg, "Get start time of interval", i, j),
                              call(tg, "Get end time of interval", i, j), lab))
        tiers[name] = items
    words = next((v for k, v in tiers.items() if "word" in k), None)
    phones = next((v for k, v in tiers.items() if "phone" in k), None)
    if words is None or phones is None:
        sys.exit(f"[ERR] {path}: tiers 'words' and 'phones' expected, found {list(tiers)}")
    return words, phones


def is_vowel(phone):
    if re.fullmatch(r"[AEIOU][A-Z]?[0-2]", phone):              # ARPAbet (english_us_arpa)
        return True
    base = unicodedata.normalize("NFD", phone)
    return any(c in IPA_VOWELS for c in base)


def speech_chunks(snd, n_lines, min_pause=0.35):
    """Sounding stretches of the audio, one per prompt line when possible.
    Extra stretches (a pause inside a sentence) are merged across the
    shortest gaps; fewer stretches than lines means the split is impossible."""
    tg = call(snd, "To TextGrid (silences)", 100, 0.0, -25.0, min_pause, 0.1, "silent", "sounding")
    iv = [(call(tg, "Get start time of interval", 1, j), call(tg, "Get end time of interval", 1, j))
          for j in range(1, call(tg, "Get number of intervals", 1) + 1)
          if call(tg, "Get label of interval", 1, j) == "sounding"]
    while len(iv) > n_lines:
        k = min(range(len(iv) - 1), key=lambda m: iv[m + 1][0] - iv[m][1])
        iv[k:k + 2] = [(iv[k][0], iv[k + 1][1])]
    return iv if len(iv) == n_lines else None


def run_mfa(mfa_cmd, wav, lines, workdir, model="french_mfa"):
    """Align the audio against the prompt, ONE SENTENCE AT A TIME.

    A whole meditation aligned in one piece goes wrong: across long pauses the
    aligner slides words into the wrong sentence (seen: "relâche" placed in
    the first sentence, "ta" in the middle of a pause). Each prompt line is
    therefore matched to its own stretch of speech, all stretches are aligned
    in a single MFA run, and the results are put back on the full timeline."""
    os.makedirs(workdir, exist_ok=True)
    snd = parselmouth.Sound(wav)
    if snd.n_channels > 1:
        snd = snd.convert_to_mono()
    dur = snd.get_total_duration()
    chunks = speech_chunks(snd, len(lines))
    if chunks is None:
        print(f"[!] {len(lines)} prompt lines but the pauses do not split the audio as many times: "
              "aligning it in one piece (less reliable)")
        chunks, texts = [(0.0, dur)], [" ".join(lines)]
    else:
        texts = lines
        print(f"[*] {len(chunks)} sentence(s) aligned separately")
    corpus = os.path.join(workdir, "corpus"); out = os.path.join(workdir, "aligned")
    os.makedirs(corpus, exist_ok=True); os.makedirs(out, exist_ok=True)
    offsets = []
    for n, ((t0, t1), text) in enumerate(zip(chunks, texts)):
        a, b = max(0.0, t0 - 0.15), min(dur, t1 + 0.15)
        snd.extract_part(a, b, parselmouth.WindowShape.RECTANGULAR, 1.0, False).save(
            os.path.join(corpus, f"s{n:03d}.wav"), "WAV")
        open(os.path.join(corpus, f"s{n:03d}.lab"), "w", encoding="utf-8").write(" ".join(tokens(text)))
        offsets.append(a)
    import shlex
    cmd = shlex.split(mfa_cmd) + ["align", "--clean", "--single_speaker", corpus, model, model, out]
    print("[*] " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    words, phones = [], []
    for n, off in enumerate(offsets):
        tgp = os.path.join(out, f"s{n:03d}.TextGrid")
        if not os.path.exists(tgp):
            sys.exit("[ERR] MFA failed:\n" + (r.stderr or r.stdout)[-1500:])
        w, p = read_textgrid(tgp)
        words += [(x + off, y + off, l) for x, y, l in w]
        phones += [(x + off, y + off, l) for x, y, l in p]
    # one TextGrid on the full timeline, readable in Praat and reusable
    merged = os.path.join(workdir, "merged.TextGrid")
    tg = call("Create TextGrid", 0.0, dur, "words phones", "")
    for tier, items in ((1, words), (2, phones)):
        last = 0.0
        for x, y, l in items:
            for t in (x, y):
                if 0 < t < dur and t > last - 1e-9:
                    try:
                        call(tg, "Insert boundary", tier, t)
                    except Exception:
                        pass
            call(tg, "Set interval text", tier, call(tg, "Get interval at time", tier, (x + y) / 2), l)
            last = y
    call(tg, "Save as text file", merged)
    return merged


def cached_alignment(wav, script, mfa_cmd, model):
    """Alignment kept next to the audio (<audio>.TextGrid) and reused as long as
    the audio and the WORDS of the prompt are unchanged: editing the marks
    ({-2} -> {-5}) does not realign, so only the first run waits for MFA."""
    import json
    cache = os.path.splitext(wav)[0] + ".TextGrid"
    words_now = " ".join(tokens(" ".join(script_lines(script))))
    stat = os.stat(wav)
    metadata = dict(words=words_now, model=model, command=mfa_cmd,
                    audio=os.path.realpath(wav), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    sidecar = cache + ".alignment.json"
    try:
        with open(sidecar, encoding="utf-8") as handle:
            valid = json.load(handle) == metadata
    except (OSError, ValueError):
        valid = False
    if os.path.exists(cache) and valid:
        print(f"[*] alignment reused: {cache}")
        return cache
    with tempfile.TemporaryDirectory(prefix="vp_mfa_") as work:
        tg = run_mfa(mfa_cmd, wav, script_lines(script), work, model)
        shutil.copy(tg, cache)
    with open(sidecar, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False)
    print(f"[*] alignment saved: {cache}")
    return cache


def nuclei(snd, t0, t1):
    """Vowel-like stretches of [t0, t1] found from the sound itself: voiced
    intensity peaks. Used when the aligner gives no phones for a word
    (an unknown word comes back as <unk> / spn)."""
    part = snd.extract_part(max(0.0, t0 - 0.05), t1 + 0.05, parselmouth.WindowShape.RECTANGULAR, 1.0, True)
    inten = part.to_intensity(minimum_pitch=75, time_step=0.005)
    pitch = part.to_pitch(time_step=0.005, pitch_floor=60, pitch_ceiling=600)
    ts = [t for t in inten.xs() if t0 <= t <= t1]
    if not ts:
        return []
    val = [inten.get_value(t) or 0.0 for t in ts]
    voiced = [(pitch.get_value_at_time(t) or 0) > 0 for t in ts]
    peaks = []
    for k in range(1, len(ts) - 1):
        if voiced[k] and val[k] >= val[k - 1] and val[k] >= val[k + 1]:
            lo = min(val[max(0, k - 12):k + 13])
            if val[k] - lo >= 2.0 and (not peaks or ts[k] - peaks[-1] >= 0.08):
                peaks.append(ts[k])
    res = []
    for k, pk in enumerate(peaks):
        left = (pk - peaks[k - 1]) / 2 if k else 0.06
        right = (peaks[k + 1] - pk) / 2 if k + 1 < len(peaks) else 0.06
        res.append((max(t0, pk - min(left, 0.06)), min(t1, pk + min(right, 0.06)), "~"))
    return res


def vowel_targets(script_path, words, phones, report=True, snd=None, span="syllable", transpose=0.0):
    """Time intervals and shifts of every marked vowel."""
    lines = script_lines(script_path)
    tempos = tempo_per_line(script_path)
    ann, ann_tempo = [], []
    for line, tempo in zip(lines, tempos):
        for wm in marks_by_word(line):
            ann.append(wm); ann_tempo.append(tempo)
    s_tok = [w for w, _ in ann]
    a_tok = [w.lower().replace("’", "'").strip("'") for _, _, w in words]
    sm = SequenceMatcher(a=s_tok, b=a_tok, autojunk=False)
    pairs = {}
    blocks = sm.get_matching_blocks()
    for blk in blocks:
        for k in range(blk.size):
            pairs[blk.a + k] = blk.b + k
    # An unknown word comes back as <unk>: between two matched stretches, a gap
    # of the same length on both sides is paired word for word.
    prev_a = prev_b = 0
    for blk in blocks:
        ga, gb = blk.a - prev_a, blk.b - prev_b
        if ga == gb and ga > 0:
            for k in range(ga):
                pairs[prev_a + k] = prev_b + k
        prev_a, prev_b = blk.a + blk.size, blk.b + blk.size
    targets, missed = [], 0
    for i, (tok, marks) in enumerate(ann):
        if not marks:
            continue
        if i not in pairs:
            print(f"[!] « {tok} » not found in the alignment -- marks ignored"); missed += 1; continue
        w0, w1, _ = words[pairs[i]]
        vowels = [(t0, t1, p) for t0, t1, p in phones if t0 >= w0 - 1e-3 and t1 <= w1 + 1e-3 and is_vowel(p)]
        if not vowels and snd is not None:
            vowels = nuclei(snd, w0, w1)
            if report:
                print(f"    {tok:<16} unknown to the dictionary: {len(vowels)} vowel(s) found from the sound")
        expanded = []
        for k, a, b, dur in marks:
            expanded += [(j, a, b, dur) for j in range(len(vowels))] if k == -1 else [(k, a, b, dur)]
        vowels.sort()
        # "syllable": a mark covers its vowel AND the consonants around it, up
        # to halfway to the neighbouring vowels (word edges for the first and
        # last). A vowel alone lasts 50-100 ms and a shift there is barely
        # heard; the whole syllable makes the mark audible.
        bounds = [w0] + [(vowels[k][1] + vowels[k + 1][0]) / 2 for k in range(len(vowels) - 1)] + [w1]
        for k, a, b, dur in expanded:
            if isinstance(a, str):                     # note name -> absolute frequency
                a = b = ("hz", note_hz(a, transpose))
            if dur and dur[0] == "b":                  # beats -> seconds at this line's tempo
                dur = ("d", dur[1] * 60.0 / ann_tempo[i])
            if k >= len(vowels):
                print(f"[!] « {tok} »: vowel #{k + 1} asked, {len(vowels)} aligned -- ignored"); missed += 1; continue
            v0, v1, p = vowels[k]
            t0, t1 = (bounds[k], bounds[k + 1]) if span == "syllable" else (v0, v1)
            # pitch over the syllable, duration over the vowel only: a held
            # note is a longer vowel, stretched consonants sound wrong
            targets.append((t0, t1, a, b, v0, v1, dur))
            if report:
                what = []
                if isinstance(a, tuple):
                    what.append(f"note {a[1]:.0f} Hz")
                elif a is not None:
                    what.append((f"{a:+g}" if a == b else f"{a:+g} > {b:+g}") + " st")
                if dur:
                    what.append(f"x{dur[1]:g} duration" if dur[0] == "x" else f"{dur[1]:g} s long")
                print(f"    {tok:<16} vowel {k + 1} [{p}]  {t0:7.2f}-{t1:6.2f} s   " + ", ".join(what))
    print(f"[*] {len(targets)} vowel(s) to modify" + (f", {missed} ignored" if missed else ""))
    return targets


# --------------------------------------------------------------------------
# Pitch editing (PSOLA)
# --------------------------------------------------------------------------
def phrases(snd, min_pause=0.3):
    """Sounding stretches separated by pauses of at least min_pause seconds."""
    tg = call(snd, "To TextGrid (silences)", 100, 0.0, -25.0, min_pause, 0.1, "silent", "sounding")
    res = []
    for j in range(1, call(tg, "Get number of intervals", 1) + 1):
        if call(tg, "Get label of interval", 1, j) == "sounding":
            res.append((call(tg, "Get start time of interval", 1, j), call(tg, "Get end time of interval", 1, j)))
    return res


def process(in_wav, out_wav, targets=(), contour=False, declination=2.0, compress=0.8,
            global_shift=0.0, fmin=60, fmax=500, target_mode=False):
    snd = parselmouth.Sound(in_wav)
    manip = call(snd, "To Manipulation", 0.01, fmin, fmax)
    pt = call(manip, "Extract pitch tier")
    n = call(pt, "Get number of points")
    pts = [(call(pt, "Get time from index", i), call(pt, "Get value at index", i)) for i in range(1, n + 1)]
    if not pts:
        sys.exit("[ERR] no voiced frame found -- is it speech?")
    st = [0.0] * len(pts)                                   # semitone offset per point
    vals = [v for _, v in pts]
    if contour:
        for p0, p1 in phrases(snd):
            idx = [i for i, (t, _) in enumerate(pts) if p0 <= t <= p1]
            if len(idx) < 3:
                continue
            seg = sorted(vals[i] for i in idx); med = seg[len(seg) // 2]
            for i in idx:
                t, v = pts[i]
                comp = 12 * (compress - 1.0) * (__import__("math").log2(v / med))   # range narrowed around the median
                st[i] += comp - declination * (t - p0) / max(p1 - p0, 1e-6)
        print(f"[*] contour: -{declination:g} st over each phrase, range x{compress:g}")
    import math
    med_all = sorted(vals)[len(vals) // 2]
    if target_mode:
        print(f"[*] target mode: marks = semitones from this voice's median ({med_all:.0f} Hz)")
    targets = [tuple(x) + (x[0], x[1], None) if len(x) == 4 else tuple(x) for x in targets]
    for t0, t1, a, b, _v0, _v1, _dur in targets:
        if a is None:
            continue
        for i, (t, v) in enumerate(pts):
            if t0 - EDGE <= t <= t1 + EDGE:
                x = min(max((t - t0) / max(t1 - t0, 1e-6), 0.0), 1.0)
                ramp = max(min(1.0, (t - (t0 - EDGE)) / EDGE, ((t1 + EDGE) - t) / EDGE), 0.0)
                if isinstance(a, tuple):              # a sung note: the syllable is SET to it
                    st[i] += 12 * math.log2(a[1] / v) * ramp
                    continue
                want = a + (b - a) * x
                if target_mode:
                    # the vowel is SET to median * 2^(want/12): the reference melody,
                    # transposed into this voice's register, replaces its own pitch
                    want = want - 12 * math.log2(v / med_all)
                st[i] += want * ramp
    xmin, xmax = call(pt, "Get start time"), call(pt, "Get end time")
    new = call("Create PitchTier", "edited", xmin, xmax)
    for (t, v), s in zip(pts, st):
        call(new, "Add point", t, v * 2 ** ((s + global_shift) / 12.0))
    call([new, manip], "Replace pitch tier")
    # Durations: the vowel is stretched or shortened, timbre and pitch kept.
    # A duration tier is a time-stretch factor along the original timeline;
    # it stays 1 everywhere else, with 10 ms slopes at the vowel edges.
    stretches = []
    for t0, t1, a, b, v0, v1, dur in targets:
        if not dur or v1 - v0 < 0.02:
            continue
        # the 10 ms slopes on each side add (f - 1) x 10 ms: an absolute length
        # (seconds, beats) is matched exactly, slopes included
        f = dur[1] if dur[0] == "x" else (dur[1] + 0.01) / (v1 - v0 + 0.01)
        # a held note can need a long stretch (a 0.2 s vowel held 4 beats);
        # beyond x4 the vowel starts to sound synthetic, it is reported
        f = min(max(f, 0.25), 25.0)
        if f > 4.0:
            print(f"[!] vowel at {v0:.2f} s stretched x{f:.1f}: may sound synthetic")
        stretches.append((v0, v1, f))
    if stretches:
        dt = call("Create DurationTier", "dur", xmin, xmax)
        for v0, v1, f in sorted(stretches):
            call(dt, "Add point", max(xmin, v0 - 0.01), 1.0)
            call(dt, "Add point", v0, f)
            call(dt, "Add point", v1, f)
            call(dt, "Add point", min(xmax, v1 + 0.01), 1.0)
        call([dt, manip], "Replace duration tier")
        print(f"[*] {len(stretches)} vowel duration(s) changed")
    out = call(manip, "Get resynthesis (overlap-add)")
    out.save(out_wav, "WAV")
    # Proof in the console: pitch measured in each modified stretch, before
    # and after, so a change can be checked without trusting one's ears.
    if targets:
        p_in = snd.to_pitch(time_step=0.005, pitch_floor=fmin, pitch_ceiling=fmax)
        p_out = out.to_pitch(time_step=0.005, pitch_floor=fmin, pitch_ceiling=fmax)
        # where an original time lands after stretching: integral of the
        # duration tier (piecewise linear, 10 ms slopes included)
        dpts = []
        for v0, v1, f in sorted(stretches):
            dpts += [(v0 - 0.01, 1.0), (v0, f), (v1, f), (v1 + 0.01, 1.0)]
        def shifted(t):
            extra = 0.0
            for (ta, fa), (tb, fb) in zip(dpts, dpts[1:]):
                if t <= ta:
                    break
                te = min(t, tb)
                fe = fa + (fb - fa) * (te - ta) / (tb - ta) if tb > ta else fb
                extra += ((fa + fe) / 2 - 1.0) * (te - ta)
            return t + extra
        print("[*] check (mean pitch in each modified stretch):")
        for t0, t1, a, b, v0, v1, dur in targets:
            if dur:
                ln = shifted(v1 + 0.01) - shifted(v0 - 0.01) - 0.02     # slopes included
                print(f"    {v0:6.2f}-{v1:5.2f} s   vowel {(v1 - v0) * 1000:4.0f} ms -> {ln * 1000:4.0f} ms")
            if a is None:
                continue
            f0 = call(p_in, "Get mean", t0, t1, "Hertz")
            f1 = call(p_out, "Get mean", shifted(t0), shifted(t1), "Hertz")
            if isinstance(a, tuple) and f1 == f1:
                print(f"    {t0:6.2f}-{t1:5.2f} s   note {a[1]:5.0f} Hz : sung at {f1:5.0f} Hz")
                continue
            if f0 == f0 and f1 == f1 and f0 > 0:
                print(f"    {t0:6.2f}-{t1:5.2f} s   {f0:5.0f} Hz -> {f1:5.0f} Hz   ({12 * math.log2(f1 / f0):+.1f} st)")
            else:
                print(f"    {t0:6.2f}-{t1:5.2f} s   unvoiced, nothing to measure")
    print(f"[OK] {out_wav}")


# --------------------------------------------------------------------------
# Extraction: reference recording -> annotated script
# --------------------------------------------------------------------------
def extract(ref_wav, script_path, words, phones, out_path, glide=False, step=0.5, fmin=60, fmax=700,
            durations=True):
    """Measure the pitch of every vowel of a reference recording (song, speech)
    and write the script back with {N} marks: N = semitones from the reference's
    median. With glide=True a vowel whose pitch moves writes {a>b}."""
    import math
    snd = parselmouth.Sound(ref_wav)
    pitch = snd.to_pitch(time_step=0.005, pitch_floor=fmin, pitch_ceiling=fmax)
    f = pitch.selected_array["frequency"]; tt = pitch.xs()
    voiced = sorted(x for x in f if x > 0)
    if not voiced:
        sys.exit("[ERR] no voiced frame in the reference")
    med = voiced[len(voiced) // 2]
    rnd = lambda x: round(x / step) * step
    def st_between(t0, t1):
        sel = [x for t, x in zip(tt, f) if t0 <= t <= t1 and x > 0]
        if len(sel) < 3:
            return None
        sel.sort()
        return 12 * math.log2(sel[len(sel) // 2] / med)

    # the length of each sung vowel, in seconds: applied as a target in --target mode
    dtag = (lambda t0, t1: f" d{t1 - t0:.2f}") if durations else (lambda t0, t1: "")
    lines = script_lines(script_path)
    ann = [wm for line in lines for wm in marks_by_word(line)]
    s_tok = [w for w, _ in ann]
    a_tok = [w.lower().replace("’", "'").strip("'") for _, _, w in words]
    pairs = {}
    for blk in SequenceMatcher(a=s_tok, b=a_tok, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            pairs[blk.a + k] = blk.b + k
    per_word = []                                 # list of per-vowel mark strings, one entry per script word
    for i, (tok, _) in enumerate(ann):
        if i not in pairs:
            per_word.append([]); continue
        w0, w1, _ = words[pairs[i]]
        vowels = [(t0, t1) for t0, t1, p in phones if t0 >= w0 - 1e-3 and t1 <= w1 + 1e-3 and is_vowel(p)]
        marks = []
        for t0, t1 in vowels:
            if glide:
                third = (t1 - t0) / 3
                a, b = st_between(t0, t0 + third), st_between(t1 - third, t1)
                if a is None or b is None:
                    m = st_between(t0, t1); marks.append(None if m is None else f"{rnd(m):+g}" + dtag(t0, t1))
                elif abs(rnd(a) - rnd(b)) >= 1:
                    marks.append(f"{rnd(a):+g}>{rnd(b):+g}" + dtag(t0, t1))
                else:
                    marks.append(f"{rnd((a + b) / 2):+g}" + dtag(t0, t1))
            else:
                m = st_between(t0, t1); marks.append(None if m is None else f"{rnd(m):+g}" + dtag(t0, t1))
        per_word.append(marks)

    # rewrite the script: marks inserted after the vowel groups of each word,
    # every other line (parameters, pauses, comments) kept as it is
    src = open(script_path, encoding="utf-8").read().replace("\r", "").split("\n")
    wi, out = 0, []
    ctrl = re.compile(r"(\[[^\]]*\]|\{[^}]*\})")
    for raw in src:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("{") or re.match(r"^\[\d+(\s*,.*)?\]$", line):
            out.append(raw); continue
        pieces = []
        for chunk in ctrl.split(raw):
            if ctrl.fullmatch(chunk or "") or not chunk:
                pieces.append("" if OLD_TAG.fullmatch(chunk or "") or MARK.fullmatch(chunk or "") else chunk); continue
            parts = re.split(r"([\s'’\-]+)", chunk)
            for j, p in enumerate(parts):
                if j % 2 == 1 or not re.search(r"\w", p):
                    pieces.append(p); continue
                marks = per_word[wi] if wi < len(per_word) else []
                wi += 1
                letters = p
                groups = vowel_groups(re.sub(r"[^\wàâäéèêëîïôöùûüÿœæçáíóúãõåøı]", "", letters.lower()))
                # map group ends (in the cleaned word) back to positions in the raw word
                idx_map = [k for k, c in enumerate(letters) if re.match(r"[\wàâäéèêëîïôöùûüÿœæçáíóúãõåøı]", c.lower())]
                inserts = {}
                for g, m in zip(groups, marks):
                    if m is not None and g[1] - 1 < len(idx_map):
                        inserts[idx_map[g[1] - 1] + 1] = "{" + m + "}"
                pieces.append("".join(c + inserts.get(k + 1, "") for k, c in enumerate(letters)))
        out.append("".join(pieces))
    open(out_path, "w", encoding="utf-8", newline="\n").write("\n".join(out))
    print(f"[*] reference median {med:.0f} Hz ; marks rounded to {step:g} st")
    print(f"[OK] {out_path}")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?"); ap.add_argument("output", nargs="?")
    ap.add_argument("--strip", nargs=2, metavar=("ANNOTATED", "CLEAN"), help="write the script without marks, then exit")
    ap.add_argument("--script", help="annotated script with {±N} or {a>b} vowel marks")
    ap.add_argument("--textgrid", help="existing alignment (tiers 'words' and 'phones')")
    ap.add_argument("--mfa", help='command that runs MFA, e.g. "conda run -n mfa mfa"')
    ap.add_argument("--contour", action="store_true", help="calm descending contour on every phrase")
    ap.add_argument("--declination", type=float, default=2.0, help="semitones lost over a phrase (default 2)")
    ap.add_argument("--compress", type=float, default=0.8, help="pitch range factor, 1 = unchanged (default 0.8)")
    ap.add_argument("--global-shift", type=float, default=0.0, help="semitones on the whole file, timbre kept")
    ap.add_argument("--extract", metavar="ANNOTATED_OUT",
                    help="read the pitch of every vowel of INPUT (a reference: song, speech) and write "
                         "--script back with {N} marks into ANNOTATED_OUT")
    ap.add_argument("--transpose", type=float, default=0.0,
                    help="semitones added to every NOTE mark (fit a melody to the voice's register)")
    ap.add_argument("--no-durations", action="store_true",
                    help="with --extract: write pitch only, not the length of each vowel")
    ap.add_argument("--vowel-only", action="store_true",
                    help="apply each mark to its vowel only (default: the whole syllable around it)")
    ap.add_argument("--glide", action="store_true", help="with --extract: write {a>b} when a vowel's pitch moves")
    ap.add_argument("--step", type=float, default=0.5, help="with --extract: rounding of the marks in semitones")
    ap.add_argument("--target", action="store_true",
                    help="marks are TARGETS (semitones from this voice's median, e.g. extracted from a song) "
                         "instead of shifts added to its own intonation")
    ap.add_argument("--lang", default="fr", help="fr, en, de, es, pt, ru, pl, sv (default fr)")
    ap.add_argument("--mfa-model", help="MFA acoustic model and dictionary name (default: from --lang)")
    ap.add_argument("--fmin", type=float, default=60); ap.add_argument("--fmax", type=float, default=500)
    a = ap.parse_args()
    global LANG
    LANG = a.lang.lower()[:2]
    if LANG in NO_VOWEL_MARKS and (a.script or a.extract):
        ap.error(f"vowel marks are not available for '{a.lang}' (no vowel letters); "
                 "use --contour and --global-shift")
    model = a.mfa_model or MODELS.get(LANG)
    if a.mfa and not model:
        ap.error(f"no default MFA model for '{a.lang}': give --mfa-model")
    if a.strip:
        src, dst = a.strip
        txt = open(src, encoding="utf-8").read().replace("\r", "")
        open(dst, "w", encoding="utf-8", newline="\n").write(strip_marks(txt))
        print(f"[OK] {dst} (marks removed)"); return
    if a.extract:
        if not (a.input and a.script):
            ap.error("--extract needs the reference WAV as INPUT and --script with its text")
        if a.textgrid:
            tg = a.textgrid
        elif a.mfa:
            tg = cached_alignment(a.input, a.script, a.mfa, model)
        else:
            ap.error("--extract needs --textgrid or --mfa")
        words, phones = read_textgrid(tg)
        extract(a.input, a.script, words, phones, a.extract, a.glide, a.step, a.fmin, max(a.fmax, 700),
                durations=not a.no_durations)
        return
    if not a.input or not a.output:
        ap.error("input and output WAV are required")
    targets = []
    if a.script:
        if a.textgrid:
            tg = a.textgrid
        elif a.mfa:
            tg = cached_alignment(a.input, a.script, a.mfa, model)
        else:
            ap.error("--script needs --textgrid or --mfa")
        words, phones = read_textgrid(tg)
        targets = vowel_targets(a.script, words, phones, snd=parselmouth.Sound(a.input),
                                span="vowel" if a.vowel_only else "syllable", transpose=a.transpose)
    if not (targets or a.contour or a.global_shift):
        ap.error("nothing to do: give --script, --contour or --global-shift")
    process(a.input, a.output, targets, a.contour, a.declination, a.compress, a.global_shift, a.fmin, a.fmax,
            target_mode=a.target)


if __name__ == "__main__":
    main()

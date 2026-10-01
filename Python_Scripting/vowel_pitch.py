#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vowel_pitch.py -- vowel-level pitch editing for XTTS output, formants kept.

Two independent edits, usable together:

  * VOWEL MARKS in the script, written inside the word, right after the vowel:
        Dé{-2}tends tes é{+1}pau{-1}les.      fixed shift, in semitones
        relâ{0>-3}che                          glide from 0 to -3 across the vowel
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

MARK = re.compile(r"\{([+-]?\d+(?:\.\d+)?)(?:>([+-]?\d+(?:\.\d+)?))?\}")
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
        line = re.sub(r"\[(pause|music|parallel|/parallel)[^\]]*\]", " ", line, flags=re.I)
        line = line.strip()
        if line:
            out.append(line)
    return out


def strip_marks(text):
    return MARK.sub("", OLD_TAG.sub("", text))


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
                        marks.append((-1, float(m.group(1)), float(m.group(1))))
                    continue
                a = float(m.group(1)); b = float(m.group(2)) if m.group(2) else a
                letters = re.sub(r"[^\w]", "", clean.lower())
                k = sum(1 for s_, e_ in vowel_groups(letters + "x") if e_ <= len(letters)) - 1
                marks.append((max(k, 0), a, b))
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


def run_mfa(mfa_cmd, wav, text, workdir, model="french_mfa"):
    corpus = os.path.join(workdir, "corpus"); out = os.path.join(workdir, "aligned")
    os.makedirs(corpus); os.makedirs(out)
    stem = "utt"
    # MFA wants a WAV: whatever the input (mp3, flac...), write a mono WAV copy
    snd = parselmouth.Sound(wav)
    if snd.n_channels > 1:
        snd = snd.convert_to_mono()
    snd.save(os.path.join(corpus, stem + ".wav"), "WAV")
    open(os.path.join(corpus, stem + ".lab"), "w", encoding="utf-8").write(" ".join(tokens(text)))
    cmd = mfa_cmd.split() + ["align", "--clean", "--single_speaker", corpus, model, model, out]
    print("[*] " + " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    tg = os.path.join(out, stem + ".TextGrid")
    if r.returncode != 0 or not os.path.exists(tg):
        sys.exit("[ERR] MFA failed:\n" + (r.stderr or r.stdout)[-1500:])
    return tg


def vowel_targets(script_path, words, phones, report=True):
    """Time intervals and shifts of every marked vowel."""
    lines = script_lines(script_path)
    ann = [wm for line in lines for wm in marks_by_word(line)]
    s_tok = [w for w, _ in ann]
    a_tok = [w.lower().replace("’", "'").strip("'") for _, _, w in words]
    sm = SequenceMatcher(a=s_tok, b=a_tok, autojunk=False)
    pairs = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            pairs[blk.a + k] = blk.b + k
    targets, missed = [], 0
    for i, (tok, marks) in enumerate(ann):
        if not marks:
            continue
        if i not in pairs:
            print(f"[!] « {tok} » not found in the alignment -- marks ignored"); missed += 1; continue
        w0, w1, _ = words[pairs[i]]
        vowels = [(t0, t1, p) for t0, t1, p in phones if t0 >= w0 - 1e-3 and t1 <= w1 + 1e-3 and is_vowel(p)]
        expanded = []
        for k, a, b in marks:
            expanded += [(j, a, b) for j in range(len(vowels))] if k == -1 else [(k, a, b)]
        for k, a, b in expanded:
            if k >= len(vowels):
                print(f"[!] « {tok} »: vowel #{k + 1} asked, {len(vowels)} aligned -- ignored"); missed += 1; continue
            t0, t1, p = vowels[k]
            targets.append((t0, t1, a, b))
            if report:
                glide = f"{a:+g}" if a == b else f"{a:+g} > {b:+g}"
                print(f"    {tok:<16} vowel {k + 1} [{p}]  {t0:7.2f}-{t1:6.2f} s   {glide} st")
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
    for t0, t1, a, b in targets:
        for i, (t, v) in enumerate(pts):
            if t0 - EDGE <= t <= t1 + EDGE:
                x = min(max((t - t0) / max(t1 - t0, 1e-6), 0.0), 1.0)
                want = a + (b - a) * x
                ramp = max(min(1.0, (t - (t0 - EDGE)) / EDGE, ((t1 + EDGE) - t) / EDGE), 0.0)
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
    out = call(manip, "Get resynthesis (overlap-add)")
    out.save(out_wav, "WAV")
    print(f"[OK] {out_wav}")


# --------------------------------------------------------------------------
# Extraction: reference recording -> annotated script
# --------------------------------------------------------------------------
def extract(ref_wav, script_path, words, phones, out_path, glide=False, step=0.5, fmin=60, fmax=700):
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
                    m = st_between(t0, t1); marks.append(None if m is None else f"{rnd(m):+g}")
                elif abs(rnd(a) - rnd(b)) >= 1:
                    marks.append(f"{rnd(a):+g}>{rnd(b):+g}")
                else:
                    marks.append(f"{rnd((a + b) / 2):+g}")
            else:
                m = st_between(t0, t1); marks.append(None if m is None else f"{rnd(m):+g}")
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
            tg = run_mfa(a.mfa, a.input, " ".join(script_lines(a.script)), tempfile.mkdtemp(prefix="vp_mfa_"), model)
        else:
            ap.error("--extract needs --textgrid or --mfa")
        words, phones = read_textgrid(tg)
        extract(a.input, a.script, words, phones, a.extract, a.glide, a.step, a.fmin, max(a.fmax, 700))
        return
    if not a.input or not a.output:
        ap.error("input and output WAV are required")
    targets = []
    if a.script:
        if a.textgrid:
            tg = a.textgrid
        elif a.mfa:
            work = tempfile.mkdtemp(prefix="vp_mfa_")
            tg = run_mfa(a.mfa, a.input, " ".join(script_lines(a.script)), work, model)
        else:
            ap.error("--script needs --textgrid or --mfa")
        words, phones = read_textgrid(tg)
        targets = vowel_targets(a.script, words, phones)
    if not (targets or a.contour or a.global_shift):
        ap.error("nothing to do: give --script, --contour or --global-shift")
    process(a.input, a.output, targets, a.contour, a.declination, a.compress, a.global_shift, a.fmin, a.fmax,
            target_mode=a.target)


if __name__ == "__main__":
    main()

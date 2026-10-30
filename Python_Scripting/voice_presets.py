#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_presets.py — the saved {} / [] library: one readable file per voice.

Storage: Voice_Presets/<name>.txt at the project root, e.g.

    # Voice preset: Fanny deep
    # Source    : pipeline  |  2026-07-29 23:16
    # Reference : Voices_Cloning/Fanny_Ardant_bis_deep_curated.wav
    # Scores    : held-out 0.812 | identity 0.725 | identity(unseen) 0.694
    # Paste these two lines into a script (the '#' lines are ignored).
    {1, 180, 0, 200, 100, 250, 0.85, 50, 0.85, 5, 1, 45, 4, 0}
    [1, FR, 1, -2, -9.7, 0.3, -1.5, 55, 8500, 0, 0, 0, 0, 0, 0, 1]
    #
    # ---- In words ----
    #   {} voice 1 | seed 180 | temp 0.85 | ...
    #   [] FR | speed 1 | volume +0 dB | EQ low -2 / mid -9.7 / high +0.3 dB | ...
    # ---- Run ----
    #   ...

The two blocks sit at the top, under the short header; everything after them is
commentary: the same two blocks in plain words, then one section per stage of
the run that produced them (curation, tone fit, audio search, unseen sentences,
files). Nothing below the blocks is read back as a setting.

Why a directory of text files rather than one JSON:
  * the generator strips '#' comment lines, so a preset file can be pasted
    whole into a script and just works — metadata included;
  * each voice is one file: copy it, mail it, keep it in git, delete it;
  * a plain file dialog can browse them, which a single JSON cannot;
  * a corrupt write costs one voice, not the whole library.

Metadata matters because the same voice is re-run often and the winner changes
between runs: two lines of blocks alone would not say which run produced them.
"""

import os
import re
import glob
import datetime
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
PRESET_DIR = os.path.join(os.path.dirname(_HERE), 'Voice_Presets')
_LEGACY_JSON = os.path.join(os.path.dirname(_HERE), 'voice_presets.json')

_SAFE = re.compile(r'[^A-Za-z0-9 ._+-]')


def _slug(name):
    s = _SAFE.sub('_', (name or 'voice').strip()).strip(' ._-')
    return s or 'voice'


def path_of(name):
    return os.path.join(PRESET_DIR, _slug(name) + '.txt')


def _fmt_scores(scores):
    if not scores:
        return None
    order = ['held_out', 'identity', 'identity_unseen', 'french', 'search']
    label = {'held_out': 'held-out', 'identity': 'identity',
             'identity_unseen': 'identity(unseen)', 'french': 'french',
             'search': 'search'}
    bits = [f"{label.get(k, k)} {float(scores[k]):.3f}"
            for k in order if k in scores and scores[k] is not None]
    bits += [f"{k} {float(v):.3f}" for k, v in scores.items()
             if k not in order and v is not None]
    return ' | '.join(bits) or None


_HEADING = re.compile(r'^\s*#\s*-{3,}\s*(.+?)\s*-{3,}\s*$')
_RULE = 66


def _one_line(text):
    """A comment must stay ONE line. A value holding a line break would leave the
    rest of it outside the '#' and the generator would read it as text, or a
    stray {...} as a block."""
    return re.sub(r'\s*[\r\n\u2028\u2029]+\s*', ' ', str(text)).strip()


def _num(v):
    """12 -> '12', 0.5 -> '0.5', 6.4710000001 -> '6.471'."""
    v = float(v)
    return str(int(v)) if v.is_integer() else format(v, '.10g')


def _signed(v):
    v = float(v)
    return ('+' if v >= 0 else '') + _num(v)


def explain_xtts(block):
    """The {} block in words. Returns a list of lines, or [] if unreadable.

    A field the block leaves out is 'not set' (the voice keeps its previous
    value); a 0 on a knob where 0 is invalid is the historical placeholder for
    'keep the inherited value'. Neither is shown as if it were a setting.
    """
    try:
        import xtts_settings as XS
        raw = str(block).strip()
        if not (raw.startswith('{') and raw.endswith('}')):
            return []
        parts = [x.strip() for x in raw[1:-1].split(',')]
        voice = int(float(parts[0]))
        vals = {}
        for key, token in zip(XS.FIELDS, parts[1:]):
            if not token:
                vals[key] = 'not set'
                continue
            v = XS._number(key, token)
            if key == 'seed' and v is None:
                vals[key] = 'none (unseeded)'
            elif key in XS.ZERO_PLACEHOLDERS and v == 0:
                vals[key] = 'inherited (0)'
            elif key in XS.BOOLEANS:
                vals[key] = 'on' if v else 'off'
            else:
                vals[key] = _num(v)
        for key in XS.FIELDS:
            vals.setdefault(key, 'not set')
    except Exception:
        return []

    def u(key, unit):
        v = vals[key]
        return f"{v} {unit}" if unit and v[0].isdigit() else v

    return _labelled([
        ('sampling', f"voice {voice} | seed {vals['seed']} | temp {vals['temp']} | "
                     f"top_k {vals['top_k']} | top_p {vals['top_p']} | "
                     f"typical_p {vals['typical_p']} (1=off) | "
                     f"rep_pen {vals['rep_pen']} | len_pen {vals['len_pen']}"),
        ('edges', f"trim_start {u('trim_start', 'ms')} | trim_end {u('trim_end', 'ms')} | "
                  f"fade_in {u('fade_in', 'ms')} | fade_out {u('fade_out', 'ms')}"),
        ('conditioning', f"gpt_cond_len {u('gpt_cond_len', 's')} | "
                         f"gpt_cond_chunk_len {u('gpt_cond_chunk_len', 's')} | "
                         f"max_ref_len {u('max_ref_len', 's')} | "
                         f"speaker_ref_len {u('speaker_ref_len', 's')} (0=shared cap) | "
                         f"ref_trim_db {vals['ref_trim_db']} (0=off) | "
                         f"sound_norm_refs {vals['sound_norm_refs']}"),
        ('decoding', f"num_beams {vals['num_beams']} | do_sample {vals['do_sample']}"),
    ])


def _labelled(rows):
    """('label', 'text') pairs -> aligned 'label : text' lines."""
    width = max(len(k) for k, _ in rows)
    return [f"{k.ljust(width)} : {v}" for k, v in rows]


_AUDIO_NAMES = ('speed', 'vol', 'eq_low', 'eq_mid', 'eq_high', 'hp', 'lp', 'NR',
                'comp', 'deess', 'reverb', 'gate', 'pan', 'limiter')


def explain_audio(block):
    """The [] block in words. Returns a list of lines, or [] if unreadable."""
    try:
        raw = str(block).strip()
        if not (raw.startswith('[') and raw.endswith(']')):
            return []
        parts = [x.strip() for x in raw[1:-1].split(',')]
        voice = int(float(parts[0]))
        rest = parts[1:]
        lang = None
        if rest and re.fullmatch(r'[A-Za-z][A-Za-z-]+', rest[0]):
            lang, rest = rest[0].upper(), rest[1:]
        vals = {}
        for key, token in zip(_AUDIO_NAMES, rest):
            if token:
                vals[key] = float(token)
    except Exception:
        return []

    def has(k):
        return k in vals

    def level(k, unit='', off='off'):
        if not has(k):
            return 'default'
        return off if vals[k] == 0 else (f"{_num(vals[k])} {unit}".strip())

    def db(k):
        return f"{_signed(vals[k])} dB" if has(k) else 'default'

    if not has('pan'):
        pan = 'default'
    elif vals['pan'] == 0:
        pan = 'centre'
    else:
        pan = f"{'right' if vals['pan'] > 0 else 'left'} {_num(abs(vals['pan']))}"
    limiter = 'default' if not has('limiter') else ('on' if vals['limiter'] else 'off')
    return _labelled([
        ('voice', f"{voice} | language {lang or 'default'}"),
        ('level, EQ', f"speed {_num(vals['speed']) if has('speed') else 'default'} | "
                      f"volume {db('vol')} | EQ low {db('eq_low')} / mid {db('eq_mid')} / "
                      f"high {db('eq_high')}"),
        ('filters', f"high-pass {level('hp', 'Hz')} | low-pass {level('lp', 'Hz')} | "
                    f"noise reduction {level('NR')} | compression {level('comp')} | "
                    f"de-esser {level('deess')}"),
        ('effects', f"reverb {level('reverb')} | noise gate {level('gate', 'dB')} | "
                    f"pan {pan} | limiter {limiter}"),
    ])


def explain_timbre(timbre):
    """Lines for the 'Timbre' section of a preset, from the comparator's
    `[*] Timbre` line: the three terms of the timbre distance of the final clone."""
    ref, fin = timbre['reference'], timbre['final']
    gaps, terms = timbre['gaps'], timbre['terms']
    names = {'pitch': 'pitch', 'brightness': 'brightness', 'hnr': 'HNR'}
    units = {'pitch': 'st', 'brightness': 'Hz', 'hnr': 'dB'}
    lines = [f"reference: F0 {ref['f0']:.0f} Hz | centroid {ref['centroid']:.0f} Hz | "
             f"HNR {ref['hnr']:.1f} dB",
             f"clone    : F0 {fin['f0']:.0f} Hz | centroid {fin['centroid']:.0f} Hz | "
             f"HNR {fin['hnr']:.1f} dB",
             'gap (cost): ' + ' | '.join(
                 f"{names[k]} {gaps[k]:+.1f} {units[k]} ({terms[k]:.2f})"
                 for k in ('pitch', 'brightness', 'hnr')),
             f"distance {timbre['distance']:.2f} -> timbre score {timbre['score']:.2f}; "
             f"largest cost: {names[timbre['largest']]}"]
    stages = timbre.get('stages') or {}
    if stages:
        lines.append('timbre score by stage: ' + ' | '.join(
            f"{k} {v:.2f}" if v is not None else f"{k} n/a" for k, v in stages.items()))
    return lines


def _heading(title):
    head = f"# ---- {_one_line(title)} "
    return head + '-' * max(4, _RULE - len(head))


def _section(title, lines):
    out = [_heading(title)]
    if isinstance(lines, str):
        lines = [lines]
    for line in lines:
        text = _one_line(line)
        if text:
            out.append('#   ' + text)
    return out


def read_file(path):
    """Parse any preset file. Returns None if it holds no usable block pair."""
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            raw = fh.read()
    except Exception:
        return None
    mx = re.search(r'^\s*(\{[^}\n]+\})\s*$', raw, re.M)
    ma = re.search(r'^\s*(\[\s*\d+\s*,\s*[A-Za-z]{2}[^\]\n]*\])\s*$', raw, re.M)
    if not mx or not ma:
        return None
    e = {'xtts': mx.group(1).strip(), 'audio': ma.group(1).strip(),
         'name': os.path.splitext(os.path.basename(path))[0], 'file': path}
    for key, field in (('Voice preset', 'name'), ('Source', 'source'),
                       ('Reference', 'reference'), ('Scores', 'scores_text'),
                       ('Voice', 'acoustics'), ('Note', 'note')):
        m = re.search(rf'^\s*#\s*{key}\s*:\s*(.+)$', raw, re.M)
        if m:
            e[field] = m.group(1).strip()
    refs = re.search(r'^\s*#\s*Clone references\s*:\s*(.+)$', raw, re.M)
    if refs:
        try:
            value = json.loads(refs.group(1))
            if isinstance(value, list) and all(isinstance(r, str) for r in value):
                e['references'] = value
        except ValueError:
            pass
    if 'source' in e and '|' in e['source']:
        src, _, date = e['source'].partition('|')
        e['source'], e['date'] = src.strip(), date.strip()
    info, current = {}, None
    for line in raw.splitlines():
        head = _HEADING.match(line)
        if head:
            current = head.group(1)
            info.setdefault(current, [])
        elif current is not None and line.lstrip().startswith('#'):
            text = line.lstrip()[1:].strip()
            if text:
                info[current].append(text)
        elif line.strip():
            current = None
    if info:
        e['info'] = info
    return e


def load():
    """All presets, keyed by name. Migrates a legacy JSON on first call."""
    _migrate_legacy()
    out = {}
    for p in sorted(glob.glob(os.path.join(PRESET_DIR, '*.txt'))):
        e = read_file(p)
        if e:
            out[e.get('name') or os.path.splitext(os.path.basename(p))[0]] = e
    return out


def get(name):
    e = read_file(path_of(name))
    if e:
        return e
    return load().get(name)


def save(name, xtts, audio, reference=None, source='tool', scores=None,
         note=None, acoustics=None, overwrite=True, references=None, info=None):
    """Write one preset file. Returns the name actually used.

    info: {section title: line or list of lines}, written below the two blocks
    as commentary (what the run did, what it measured, where its files are).
    The two blocks are also written out in words, whatever the caller gives.
    """
    if not name or not xtts or not audio:
        raise ValueError("name, xtts and audio are all required")
    os.makedirs(PRESET_DIR, exist_ok=True)
    final = name.strip()
    if not overwrite and os.path.exists(path_of(final)):
        i = 2
        while os.path.exists(path_of(f"{name} ({i})")):
            i += 1
        final = f"{name} ({i})"
    when = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    lines = [f"# Voice preset: {_one_line(final)}",
             f"# Source    : {_one_line(source)}  |  {when}"]
    if reference:
        lines.append(f"# Reference : {_one_line(reference)}")
    if references:
        lines.append('# Clone references: ' + json.dumps([os.path.abspath(r) for r in references], ensure_ascii=False))
    if acoustics:
        # One line, not the full diagnostic dump: what this VOICE is, so that
        # opening a preset months later identifies the speaker at a glance
        # without burying the two lines the file exists for.
        lines.append(f"# Voice     : {_one_line(acoustics)}")
    sc = _fmt_scores(scores)
    if sc:
        lines.append(f"# Scores    : {sc}")
    if note:
        lines.append(f"# Note      : {_one_line(note)}")
    lines += ["# Paste these two lines into a script (the '#' lines are ignored).",
              xtts.strip(), audio.strip()]
    sections = []
    xw, aw = explain_xtts(xtts), explain_audio(audio)
    if xw or aw:
        sections.append(('In words', [f"{{}} {x}" for x in xw] + [f"[] {x}" for x in aw]))
    for title, body in (info or {}).items():
        if body:
            sections.append((title, body))
    for title, body in sections:
        lines.append('#')
        lines += _section(title, body)
    lines.append('')
    tmp = path_of(final) + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    os.replace(tmp, path_of(final))          # atomic: never a half-written file
    return final


def delete(name):
    p = path_of(name)
    if os.path.exists(p):
        os.remove(p)
        return True
    return False


def describe(name):
    e = get(name)
    if not e:
        return ''
    bits = [e.get('acoustics', ''), e.get('source', ''), e.get('date', ''),
            e.get('scores_text', ''), e.get('note', '')]
    return '  |  '.join(b for b in bits if b)


def info_text(name):
    """The commentary of a preset as plain text: everything the file says about
    the run, header included, without the '#' marks."""
    p = path_of(name)
    try:
        with open(p, 'r', encoding='utf-8') as fh:
            raw = fh.read().splitlines()
    except Exception:
        return ''
    out = []
    for line in raw:
        t = line.lstrip()
        if not t.startswith('#') or t.startswith("# Paste these two lines"):
            continue
        t = t[1:]
        out.append(t[1:] if t.startswith(' ') else t)
    while out and not out[-1].strip():
        out.pop()
    return '\n'.join(out)


def insert_text(entry):
    """What the [Gen] editor receives: one comment line naming the voice (the
    generator ignores it), then the two blocks. A script with several voices
    stays readable without opening the preset files."""
    who = _one_line(entry.get('name') or 'voice')
    what = _one_line(entry.get('acoustics') or entry.get('source') or '')
    head = f"# {who}" + (f" - {what}" if what else '')
    return f"{head}\n{entry['xtts']}\n{entry['audio']}"


def name_from_reference(path):
    """Voice name from a reference filename, minus the pipeline's suffixes."""
    base = os.path.splitext(os.path.basename(path or 'voice'))[0]
    for suf in ('_curated', '_pipeline_clone', '_clone', '_optimised'):
        if base.endswith(suf):
            base = base[:-len(suf)]
    return base or 'voice'


def _migrate_legacy():
    """One-off import of the old single-JSON store, then rename it aside."""
    if not os.path.exists(_LEGACY_JSON):
        return
    try:
        import json
        with open(_LEGACY_JSON, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        for nm, e in (data or {}).items():
            if os.path.exists(path_of(nm)):
                continue
            save(nm, e.get('xtts', ''), e.get('audio', ''),
                 reference=e.get('reference'), source=e.get('source', 'imported'),
                 scores=e.get('scores'), note=e.get('note'))
        os.replace(_LEGACY_JSON, _LEGACY_JSON + '.migrated')
        print(f"   [*] Imported {len(data or {})} preset(s) into {PRESET_DIR}/")
    except Exception as exc:
        print(f"   [!] Legacy preset import failed: {exc}")


__all__ = ['save', 'load', 'get', 'delete', 'describe', 'name_from_reference',
           'path_of', 'read_file', 'PRESET_DIR', 'explain_xtts', 'explain_audio', 'explain_timbre',
           'info_text', 'insert_text']

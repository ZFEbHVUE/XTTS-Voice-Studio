#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_presets.py — the saved {} / [] library: one readable file per voice.

Storage: Voice_Presets/<name>.txt at the project root, e.g.

    # Voice preset: Fanny deep
    # Source    : pipeline  |  2026-07-29 23:16
    # Reference : Voices_Cloning/Fanny_Ardant_bis_deep_curated.wav
    # Scores    : held-out 0.812 | identity 0.725 | identity(unseen) 0.694
    {1, 180, 0, 200, 100, 250, 0.85, 50, 0.85, 5, 1, 45, 4, 0}
    [1, FR, 1, -2, -9.7, 0.3, -1.5, 55, 8500, 0, 0, 0, 0, 0, 0, 1]

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
    if 'source' in e and '|' in e['source']:
        src, _, date = e['source'].partition('|')
        e['source'], e['date'] = src.strip(), date.strip()
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
         note=None, acoustics=None, overwrite=True):
    """Write one preset file. Returns the name actually used."""
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
    lines = [f"# Voice preset: {final}",
             f"# Source    : {source}  |  {when}"]
    if reference:
        lines.append(f"# Reference : {reference}")
    if acoustics:
        # One line, not the full diagnostic dump: what this VOICE is, so that
        # opening a preset months later identifies the speaker at a glance
        # without burying the two lines the file exists for.
        lines.append(f"# Voice     : {acoustics}")
    sc = _fmt_scores(scores)
    if sc:
        lines.append(f"# Scores    : {sc}")
    if note:
        lines.append(f"# Note      : {note}")
    lines += ["# Paste these two lines into a script (the '#' lines are ignored).",
              xtts.strip(), audio.strip(), ""]
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
           'path_of', 'read_file', 'PRESET_DIR']

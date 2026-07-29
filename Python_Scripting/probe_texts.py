#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_texts.py — language-aware default sentences for the scoring tools.

Why this exists: the optimiser, validator and comparator all SCORE a clone by
transcribing it (faster-whisper) against the sentence they asked for. The
sentence must therefore be in the language passed to XTTS. Hard-coded French
defaults quietly broke every non-French run — XTTS read French text with the
selected language's phonetics and the accent score became noise.

Sentences are phonetically varied, neutral in content, and of comparable
length across languages so scores stay comparable.

Coverage is deliberately honest: languages below are written to be idiomatic.
For any other language the caller gets the English set plus an explicit warning
telling it to pass --text, rather than a silently wrong default.

API:
    probe_texts('fr')        -> list of search sentences (first is the default)
    holdout_texts('fr')      -> sentences reserved for validation (disjoint)
    default_text('fr')       -> a single sentence
"""

_PROBE = {
    'en': ["Hello, this is a short test sentence for tuning the voice.",
           "The evening light fades slowly behind the distant hills.",
           "Breathe deeply and let every tension go."],
    'fr': ["Bonjour, ceci est une phrase de test pour régler la voix avec soin.",
           "Le soleil se couche doucement derrière les collines lointaines.",
           "Respire profondément et laisse partir toutes les tensions."],
    'es': ["Hola, esta es una frase de prueba para ajustar la voz con cuidado.",
           "El sol se pone lentamente detrás de las colinas lejanas.",
           "Respira profundamente y deja marchar toda la tensión."],
    'de': ["Guten Tag, dies ist ein Testsatz, um die Stimme sorgfältig einzustellen.",
           "Die Sonne geht langsam hinter den fernen Hügeln unter.",
           "Atme tief ein und lass jede Anspannung los."],
    'it': ["Buongiorno, questa è una frase di prova per regolare la voce con cura.",
           "Il sole tramonta lentamente dietro le colline lontane.",
           "Respira profondamente e lascia andare ogni tensione."],
    'pt': ["Olá, esta é uma frase de teste para ajustar a voz com cuidado.",
           "O sol põe-se lentamente atrás das colinas distantes.",
           "Respira fundo e deixa ir toda a tensão."],
    'nl': ["Hallo, dit is een testzin om de stem zorgvuldig af te stellen.",
           "De zon zakt langzaam achter de verre heuvels.",
           "Adem diep in en laat elke spanning los."],
}

_HOLDOUT = {
    'en': ["Each breath out lets the shoulders and the jaw settle a little more.",
           "Forty-eight people were already waiting on platform number three."],
    'fr': ["Chaque expiration relâche un peu plus les épaules et la mâchoire.",
           "Quarante-huit personnes attendaient déjà sur le quai numéro trois."],
    'es': ["Cada exhalación relaja un poco más los hombros y la mandíbula.",
           "Cuarenta y ocho personas ya esperaban en el andén número tres."],
    'de': ["Jedes Ausatmen lockert die Schultern und den Kiefer ein wenig mehr.",
           "Achtundvierzig Personen warteten bereits auf Bahnsteig Nummer drei."],
    'it': ["Ogni espirazione rilassa un poco di più le spalle e la mascella.",
           "Quarantotto persone aspettavano già al binario numero tre."],
    'pt': ["Cada expiração relaxa um pouco mais os ombros e a mandíbula.",
           "Quarenta e oito pessoas já esperavam na plataforma número três."],
    'nl': ["Elke uitademing ontspant de schouders en de kaak een beetje meer.",
           "Achtenveertig mensen stonden al te wachten op perron nummer drie."],
}

_WARNED = set()


def _key(lang):
    return (lang or 'en').lower().split('-')[0].split('_')[0]


def _resolve(table, lang, what):
    k = _key(lang)
    if k in table:
        return table[k]
    if k not in _WARNED:
        _WARNED.add(k)
        print(f"   [!] No built-in {what} for language '{lang}'. Falling back to "
              f"English — the accent score will be meaningless.")
        print(f"   [!] Pass --text with a sentence in '{lang}' for a valid measurement.")
    return table['en']


def probe_texts(lang, n=3):
    """Sentences used to SEARCH. The first one is the single default."""
    return _resolve(_PROBE, lang, 'probe sentences')[:max(1, n)]


def holdout_texts(lang, n=2):
    """Sentences reserved for VALIDATION — never used during the search."""
    return _resolve(_HOLDOUT, lang, 'hold-out sentences')[:max(1, n)]


def default_text(lang):
    return probe_texts(lang, 1)[0]


def supported():
    return sorted(_PROBE.keys())


def default_lang(fallback='EN'):
    """Language the interface should start on.

    Neither 'always FR' nor 'always EN' is right: the first imposes one user's
    language on everyone, the second makes that user re-select it on every tab,
    every session. So: an explicit override wins, otherwise the system locale
    if we have sentences for it, otherwise English.

      XTTS_STUDIO_LANG=DE python xtts_studio.py     # explicit
      (LANG=fr_FR.UTF-8)                            # -> FR automatically
      (LANG=ja_JP.UTF-8)                            # -> EN, no Japanese set
    """
    import os
    cands = [os.environ.get('XTTS_STUDIO_LANG', '')]
    # Order matters: POSIX says LC_ALL overrides LC_MESSAGES, which overrides
    # LANG. 'C' and 'POSIX' are not languages and must not be treated as one —
    # locale.getdefaultlocale() returns 'C' in bare containers, and stopping
    # there silently defeated the whole detection.
    for var in ('LC_ALL', 'LC_MESSAGES', 'LANG', 'LANGUAGE'):
        cands.append(os.environ.get(var, ''))
    try:
        import locale
        cands.append((locale.getlocale()[0] or ''))
    except Exception:
        pass
    for c in cands:
        c = (c or '').strip()
        if not c or c.upper() in ('C', 'POSIX'):
            continue
        if _key(c) in _PROBE:
            return _key(c).upper()
    return fallback


__all__ = ['probe_texts', 'holdout_texts', 'default_text', 'supported',
           'default_lang']

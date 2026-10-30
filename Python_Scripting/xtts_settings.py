"""Shared, lossless XTTS parameter blocks. No audio dependencies.

Historical 9/11/14/15-field blocks remain readable. New blocks add
do_sample and max_ref_len after num_beams (17 fields including voice number).
Seed 0 is a real seed. Use 'none' to leave the random generators unseeded.
"""
import math

FIELDS = ('seed', 'trim_start', 'trim_end', 'fade_in', 'fade_out',
          'temp', 'top_k', 'top_p', 'rep_pen', 'len_pen', 'gpt_cond_len',
          'gpt_cond_chunk_len', 'sound_norm_refs', 'num_beams', 'do_sample',
          'max_ref_len', 'speaker_ref_len', 'ref_trim_db', 'typical_p')
DEFAULTS = dict(zip(FIELDS, (None, 0, 0, 100, 250, .65, 50, .85, 5., 1.,
                            30, 4, 0, 1, 1, 30, 0, 0, 1.)))
ALIASES = {'temperature': 'temp', 'repetition_penalty': 'rep_pen',
           'length_penalty': 'len_pen', 'max_ref_length': 'max_ref_len'}
LONG_NAMES = {v: k for k, v in ALIASES.items() if k != 'max_ref_length'}
INTEGERS = {'seed', 'top_k', 'gpt_cond_len', 'gpt_cond_chunk_len',
            'num_beams', 'max_ref_len', 'speaker_ref_len'}
BOOLEANS = {'sound_norm_refs', 'do_sample'}
# Historical zero placeholders retain the inherited value for positive knobs.
ZERO_PLACEHOLDERS = {'temp', 'top_p', 'rep_pen', 'len_pen', 'gpt_cond_len',
                     'gpt_cond_chunk_len', 'num_beams', 'max_ref_len'}


def _number(key, value):
    if key == 'seed' and (value is None or str(value).lower() == 'none'):
        return None
    if key in BOOLEANS and str(value).lower() in ('true', 'false'):
        return int(str(value).lower() == 'true')
    try:
        n = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{key}: expected a number, got {value!r}') from exc
    if not math.isfinite(n):
        raise ValueError(f'{key}: value must be finite')
    if key in INTEGERS | BOOLEANS:
        if n != int(n):
            raise ValueError(f'{key}: expected an integer')
        return int(n)
    return n


def validate(params):
    """Return canonical complete settings; reject bad values before model load."""
    out = DEFAULTS.copy()
    for key, value in params.items():
        key = ALIASES.get(key, key)
        if key in FIELDS:
            out[key] = _number(key, value)
    for key in BOOLEANS:
        if out[key] not in (0, 1):
            raise ValueError(f'{key}: expected 0 or 1')
    for key in ('temp', 'rep_pen', 'len_pen', 'gpt_cond_len',
                'gpt_cond_chunk_len', 'max_ref_len', 'num_beams'):
        if out[key] <= 0:
            raise ValueError(f'{key}: must be greater than zero')
    if not 0 < out['top_p'] <= 1:
        raise ValueError('top_p: expected a value in (0, 1]')
    if not 0 < out['typical_p'] <= 1:
        raise ValueError('typical_p: expected a value in (0, 1]')
    for key in ('trim_start', 'trim_end', 'fade_in', 'fade_out', 'top_k',
                'speaker_ref_len', 'ref_trim_db'):
        if out[key] < 0:
            raise ValueError(f'{key}: must be nonnegative')
    if out['seed'] is not None and not 0 <= out['seed'] < 2**32:
        raise ValueError('seed: expected none or an integer in [0, 4294967295]')
    if out['gpt_cond_chunk_len'] > out['gpt_cond_len']:
        raise ValueError('gpt_cond_chunk_len must be <= gpt_cond_len')
    # Sampled beam search is deliberately disabled in xtts_clone as well.
    if out['num_beams'] > 1:
        out['do_sample'] = 0
    return out


def parse_block(block):
    """Read supplied fields only, preserving missing/inherited settings."""
    raw = block.strip()
    if not (raw.startswith('{') and raw.endswith('}')):
        raise ValueError('XTTS block must be enclosed in braces')
    parts = [p.strip() for p in raw[1:-1].split(',')]
    if not 2 <= len(parts) <= len(FIELDS) + 1:
        raise ValueError('XTTS block: expected a voice number and up to 19 parameters')
    voice = _number('num_beams', parts[0])
    if voice < 1:
        raise ValueError('voice number must be positive')
    out = {}
    for key, token in zip(FIELDS, parts[1:]):
        if not token:
            continue
        value = _number(key, token)
        if key in ZERO_PLACEHOLDERS and value == 0:
            continue
        out[key] = value
    return out


def format_block(params, voice=1):
    p = validate(params)
    voice = _number('num_beams', voice)
    if voice < 1:
        raise ValueError('voice number must be positive')
    def fmt(v):
        if v is None:
            return 'none'
        return str(int(v)) if float(v).is_integer() else format(v, '.10g')
    # Keep the historical 17-field representation when both extensions are
    # off. New values are appended; no existing position changes meaning.
    last = len(FIELDS)
    while last > 16 and p[FIELDS[last - 1]] == DEFAULTS[FIELDS[last - 1]]:
        last -= 1
    return '{' + ', '.join([str(voice)] + [fmt(p[k]) for k in FIELDS[:last]]) + '}'


def generator_params(block, base):
    """Adapt common short names to the renderer's historical long names."""
    merged = {ALIASES.get(k, k): v for k, v in base.items()}
    merged.update(parse_block(block))
    checked = validate(merged)
    out = base.copy()
    out.update({LONG_NAMES.get(k, k): v for k, v in checked.items()})
    return out

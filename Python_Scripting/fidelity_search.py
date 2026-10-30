"""Search acoustic fidelity and select a small shortlist on paired held-out texts.

Actual rendering/model calls are supplied by the comparator. This module can be
tested without XTTS and does not infer quality from a parameter being nonzero.
"""
import math
from audio_parameter_search import search_audio, AUDIO_KEYS, DEFAULTS
from voice_fidelity import paired_decision, metric_definition, METRIC_VERSION


def block_key(block):
    return tuple(round(float(block.get(k, DEFAULTS[k])), 3) for k in AUDIO_KEYS)


def optimise_fidelity(initial, measure, validate, *, method='nelder', budget=400,
                      eq_limit=4., freeze_vol=False, held=None, min_gain=.005,
                      finalists=3, progress=None, anchors=None, anchor_labels=None):
    """measure(block) -> (metrics, rendered block); validate(block, rank)
    -> (paired metric/ASR rows, raw/base/candidate WAV triples).

    `initial` is the baseline: what is returned when no finalist is adopted. `anchors`
    are whole blocks (with `anchor_labels`, their names in the report) that compete
    against it as finalists, so a baseline that hurts cannot shelter itself.

    Compare the pruned joint winner with the strongest minimal one-field
    alternatives. A lower fit score may transfer better to new sentences.
    """
    if not math.isfinite(min_gain) or min_gain <= 0 or not 1 <= finalists <= 5:
        raise ValueError('Positive minimum gain and 1..5 finalists required')
    measurements, baseline = {}, []

    def evaluate(block):
        metrics, rendered = measure(block)
        metrics = dict(metrics)
        if not all(math.isfinite(float(metrics[k])) for k in
                   ('score', 'identity', 'acoustic_score', 'clipped_fraction')):
            raise ValueError('Non-finite fidelity metric')
        if not baseline:
            baseline.append(metrics)
        score = float(metrics['score'])
        reasons = []
        if metrics['identity'] < baseline[0]['identity']-.015:
            reasons.append('speaker identity guard')
        if metrics['clipped_fraction'] > max(.0001, baseline[0]['clipped_fraction']+.0001):
            reasons.append('clipping guard')
        if reasons:
            score = -1. - abs(score)
        measurements[block_key(rendered)] = dict(metrics, search_score=score, guards=reasons)
        return score, rendered

    candidate, report = search_audio(initial, evaluate, method=method, budget=budget,
        eq_limit=eq_limit, freeze_vol=freeze_vol, held=held,
        tolerance=min_gain/2, progress=progress, objective=METRIC_VERSION, anchors=anchors)
    base = report['baseline_block']
    # Anchors first: when the joint search ends on a block an anchor already was
    # (often the neutral one), the report names the anchor, the more telling name.
    labels = list(anchor_labels or [])
    options = [dict(block=r['block'], score=r['score'],
                    source='anchor: ' + (labels[i] if i < len(labels) else 'explicit block'))
               for i, r in enumerate(report['anchor_trials']) if r.get('score') is not None]
    options.append(dict(block=candidate, score=report['best_score'], source='pruned joint search'))
    options += [dict(block=r['block'], score=r['score'], source='one field: '+r['field'])
                for r in report['trial_candidates']]
    options.sort(key=lambda row: row['score'], reverse=True)      # stable: ties keep this order
    unique, selected = set(), []
    for option in options:
        key = block_key(option['block'])
        if key in unique or option['score'] <= report['baseline_score']+min_gain/2:
            continue
        unique.add(key)
        selected.append(option)
        if len(selected) == finalists:
            break
    checked, renders = [], []
    for rank, option in enumerate(selected, 1):
        entry = dict(rank=rank, block=option['block'], source=option['source'],
                     fit_metrics=measurements[block_key(option['block'])])
        if progress:
            progress(f'Fidelity finalist {rank}/{len(selected)}: {option["source"]}')
        try:
            rows, paths = validate(option['block'], rank)
            entry['render_indices'] = list(range(len(renders), len(renders)+len(paths)))
            renders.extend(paths)
            entry['validation'] = rows
            entry['decision'] = paired_decision(rows, min_gain=min_gain)
        except Exception as exc:
            entry['decision'] = dict(adopted=False, reasons=['validation failed: '+str(exc)])
            entry['validation'] = []
        checked.append(entry)
        if progress:
            decision = entry['decision']
            progress(('eligible' if decision['adopted'] else 'rejected') + ': ' +
                     ('paired gain %.4f' % decision['mean_gain'] if decision['adopted']
                      else '; '.join(decision['reasons'])))
    eligible = [row for row in checked if row['decision']['adopted']]
    winner = max(eligible, key=lambda r: r['decision']['mean_gain']) if eligible else None
    final = dict(winner['block'] if winner else base)
    report.update(metric_definition=metric_definition(), baseline_metrics=baseline[0],
        finalists=checked, selected_finalist=winner['rank'] if winner else None,
        adopted=bool(winner), final_block=final, adoption_tolerance=min_gain,
        adoption_tolerance_rule='paired mean > minimum and > one paired standard error; ASR/identity guards',
        validation=winner['validation'] if winner else [],
        validation_evaluations=sum(2*len(r['validation']) for r in checked),
        fitting_candidate_metrics=measurements.get(block_key(candidate)),
        final_fit_metrics=measurements[block_key(final)])
    return final, report, renders

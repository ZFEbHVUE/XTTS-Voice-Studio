"""Bounded Sobol exploration of coupled XTTS sampling settings, then paired seeds.

The incumbent and each finalist use the same three seeds and search sentences.
The final, never-seen text validation remains the responsibility of xtts_optimize.
"""
import math
import numpy as np
from scipy.stats import qmc


def sampling_candidates(incumbent, count=12):
    if not 1 <= count <= 64:
        raise ValueError('Joint sampling count must be in 1..64')
    keys = ('temp', 'rep_pen', 'top_p', 'top_k', 'len_pen', 'num_beams', 'do_sample', 'typical_p')
    base = {k: incumbent.get(k, 1.) for k in keys}
    # Include a lower repetition penalty explicitly: the previous 4..10 grid
    # could never discover the documented lower-penalty region.
    anchor = dict(base, temp=.65, rep_pen=2., top_p=.85, top_k=50,
                  num_beams=1, do_sample=True, len_pen=1., typical_p=1.)
    result = [anchor]
    points = qmc.Sobol(5, scramble=True, seed=173).random_base2(
        int(math.ceil(math.log2(max(2, count)))))
    ks = (0, 10, 20, 30, 50, 70, 100)
    for index, p in enumerate(points):
        trial = dict(base, temp=round(.40+.50*p[0], 3), rep_pen=round(1.5+6.5*p[1], 3),
            top_p=round(.70+.30*p[2], 3), top_k=ks[min(len(ks)-1, int(p[3]*len(ks)))],
            num_beams=1, do_sample=True, len_pen=1.,
            typical_p=round(.80+.20*p[4], 3) if index % 2 else 1.)
        if trial not in result:
            result.append(trial)
        if len(result) == count:
            break
    return result[:count]


def run_joint_search(incumbent, seeds, evaluate, *, count=12, max_draws=18,
                     min_gain=.005, progress=None):
    """evaluate(seed, parameters) returns the optimiser's record or None.
    max_draws counts complete multi-text evaluations, not individual sentences.
    Six draws are reserved for the two finalists and baseline on two extra seeds.
    """
    report = dict(method='sobol + paired seeds', requested=count, screened=0,
                  finalists=[], adopted=False, metric='existing XTTS objective',
                  minimum_gain=min_gain)
    screen_limit = min(count, max(0, max_draws-6))
    if not screen_limit:
        report['reason'] = 'budget leaves no room for screening and paired confirmation'
        return incumbent, report
    original_seed = incumbent['seed']
    paired_seeds = list(dict.fromkeys([original_seed, *seeds, 1, 2, 3]))[:3]
    report['seeds'] = paired_seeds
    trials = []
    for number, params in enumerate(sampling_candidates(incumbent, count)[:screen_limit], 1):
        if progress:
            progress(f'joint trial {number}/{screen_limit}: {params}')
        row = evaluate(original_seed, params)
        if row is None:
            break
        trials.append((params, row))
    report['screened'] = len(trials)
    report['candidates'] = [dict(parameters=params, score=row['score'], identity=row['identity'])
                            for params, row in trials]
    finalists = sorted((x for x in trials if x[1]['score'] > incumbent['score']),
                       key=lambda x: -x[1]['score'])[:2]
    if not finalists:
        report['reason'] = 'no joint candidate improves the search objective'
        return incumbent, report
    keys = ('temp', 'rep_pen', 'top_p', 'top_k', 'len_pen', 'num_beams', 'do_sample', 'typical_p')
    base_params = {k: incumbent.get(k, 1.) for k in keys}
    reference = [incumbent] + [evaluate(seed, base_params) for seed in paired_seeds[1:]]
    if any(r is None for r in reference):
        report['reason'] = 'incomplete paired baseline; incumbent retained'
        return incumbent, report
    eligible = []
    for params, first in finalists:
        rows = [first] + [evaluate(seed, params) for seed in paired_seeds[1:]]
        entry = dict(parameters=params, first_seed_score=first['score'])
        if any(r is None for r in rows):
            entry.update(adopted=False, reason='incomplete paired confirmation')
        else:
            gains = np.array([b['score']-a['score'] for a, b in zip(reference, rows)])
            ids = np.array([b['identity']-a['identity'] for a, b in zip(reference, rows)])
            speech = [db['accent']-da['accent'] for a, b in zip(reference, rows)
                      for da, db in zip(a.get('details', []), b.get('details', []))]
            stderr = float(np.std(gains, ddof=1)/math.sqrt(len(gains)))
            complete = all(len(a.get('details', [])) == len(b.get('details', [])) > 0
                           for a, b in zip(reference, rows))
            adopted = bool(complete and np.isfinite(gains).all() and np.isfinite(ids).all()
                and np.mean(gains) > max(min_gain, stderr) and min(gains) > -.02
                and min(ids) >= -.015 and np.mean(ids) >= -.005
                and speech and np.isfinite(speech).all() and min(speech) >= -.03)
            entry.update(adopted=adopted, gains=gains.tolist(), mean_gain=float(np.mean(gains)),
                         paired_stderr=stderr, identity_gains=ids.tolist(),
                         reason='paired gains and identity/intelligibility guards' if adopted
                         else 'insufficient paired gain or identity/intelligibility regression')
            if adopted:
                eligible.append((entry['mean_gain'], first))
        report['finalists'].append(entry)
        if progress:
            progress('joint finalist: '+entry['reason'])
    if eligible:
        winner = max(eligible, key=lambda x: x[0])[1]
        report.update(adopted=True, winner={k: winner[k] for k in keys},
                      reason='confirmed across the same three seeds')
        return winner, report
    report['reason'] = 'paired confirmation did not retain a joint candidate'
    return incumbent, report

"""Bounded search of the numeric fields of a voice [] block.

No XTTS/model dependencies: callers supply a function scoring an actual render.
Off is a separate state for filters and the gate, never an invalid cutoff. Every
free field receives explicit trials before any local/global joint refinement.
The result is then pruned: a change that does not earn its keep goes back to
its starting value. Scores are proxies; adoption on independent speech belongs
to the caller.

The search stops early, before its joint refinement, when that refinement
cannot pay: when nothing found so far gains more than the tolerance (pruning
would revert it all anyway), or when the caller's checkpoint finds that the best
change so far does not carry over to other sentences. The refinement works on
the same one sentence, so it cannot repair a change that fails to transfer.
"""
import math
import numpy as np


AUDIO_KEYS = ('speed', 'vol', 'eq_low', 'eq_mid', 'eq_high', 'hp', 'lp',
              'NR', 'comp', 'de-ess', 'reverb', 'noise_gate', 'pan', 'limiter')
DEFAULTS = dict(zip(AUDIO_KEYS, (1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                               0., 0., 0., 1)))
SWITCHED = {'hp': (40., 150.), 'lp': (6000., 16000.),
            'noise_gate': (-60., -25.)}
# Settings an identity score cannot judge. The comparator holds them unless
# asked otherwise: with them free, a gain elsewhere carried a pace of 1.23 or
# a pan of 0.96 into the adopted block (see test_a_gain_elsewhere_...).
EDITORIAL_FIELDS = {
    'speed': 'meditation pace, set by the Analyser or by ear; ECAPA does not measure pace',
    'pan': 'stereo placement; ECAPA averages both channels, so pan is only a gain to it',
    'limiter': 'clipping protection, not identity',
}
# The floor of the comparator's heuristic tolerance (two halves of one render).
PRUNE_TOLERANCE = .004
# What a search can score, by its name on the command line and in its report.
OBJECTIVES = {'identity': 'ecapa_identity', 'timbre': 'timbre', 'both': 'identity+timbre'}
# The criterion of --audio-objective fidelity (voice_fidelity.METRIC_VERSION).
FIDELITY = 'fidelity_v1'


def report_objective(audio_objective='identity', objective='identity'):
    """The name a search report carries for the command-line choices: --audio-objective
    fidelity is its own criterion; identity follows --objective."""
    if audio_objective == 'fidelity':
        return FIDELITY
    return OBJECTIVES.get(objective, 'ecapa_identity')


def objective_line(name):
    """The sentence a preset carries about what the [] search scored. `name` is
    a report name (OBJECTIVES' values); an unknown one reads as ECAPA only."""
    if name == 'timbre':
        return ('objective timbre (pitch, brightness and HNR of the voiced frames); '
                'the accent is not scored here')
    if name == 'identity+timbre':
        return ('objective identity+timbre, the second term of the {} search with '
                '--objective both; the accent is not scored here')
    if name == FIDELITY:
        return ('objective ECAPA identity + acoustic fidelity (voiced envelope, pitch, '
                'periodicity, sibilance, dynamics, floor); finalists checked on three other '
                'sentences with ASR guards')
    return 'objective ECAPA identity only; the {} objective applies to XTTS generation'


def score_label(name):
    """How the fit sentence's figures are called in a preset."""
    if name == FIDELITY:
        return 'fidelity'
    return 'ECAPA' if name in (None, 'ecapa_identity') else 'score'


def stop_line(rep):
    """One line when a search stopped before its joint refinement, else None."""
    stop = (rep or {}).get('stopped')
    if not stop:
        return None
    return (f"stopped after {stop.get('evaluations')} of {rep.get('budget')} evaluations, "
            f"before the joint refinement: {stop.get('reason')}")


class AudioSpace:
    """Physical values <-> comparable [0, 1] search coordinates.

    The interval [0, .2) means off for hp/lp/gate; [.2, 1] is an active
    cutoff/threshold. A simplex cannot accidentally turn lp=0 into lp=6000.
    Explicit activation trials avoid zero-sized SciPy steps at an off start.
    """
    def __init__(self, initial, eq_limit=4., freeze_vol=False, held=None):
        held = dict(held or {})
        unknown = sorted(set(held) - set(AUDIO_KEYS))
        if unknown:
            raise ValueError('Unknown held audio fields: ' + ', '.join(unknown))
        self.initial = dict(DEFAULTS, **initial)
        self.ranges = {'speed': (.75, 1.25), 'vol': (-18., 18.),
                       'NR': (0., .5), 'comp': (0., .6), 'de-ess': (0., .5),
                       'reverb': (0., .4), 'pan': (-1., 1.)}
        if not math.isfinite(eq_limit) or eq_limit <= 0:
            raise ValueError('eq-limit must be finite and positive')
        self.ranges.update({k: (-eq_limit, eq_limit)
                            for k in ('eq_low', 'eq_mid', 'eq_high')})
        # An existing speed/gain outside the exploration range stays a valid
        # incumbent, rather than being silently clipped before comparison.
        for key in ('speed', 'vol'):
            v = float(self.initial[key])
            lo, hi = self.ranges[key]
            self.ranges[key] = (min(lo, v), max(hi, v))
        self.held = {'vol': 'gain computed from --target-dbfs, not identity'} if freeze_vol else {}
        self.held.update(held)
        self.keys = tuple(k for k in AUDIO_KEYS if k not in self.held)
        if not self.keys:
            raise ValueError('Every audio field is held: nothing to search')
        self.initial = self.canonical(self.initial)

    def canonical(self, block):
        """Exactly the precision exported in [], so validation scores that block."""
        out = dict(self.initial, **block)
        for key in AUDIO_KEYS:
            value = float(out[key])
            if not math.isfinite(value):
                raise ValueError(f'{key}: non-finite value')
            if key in ('hp', 'lp', 'limiter'):
                out[key] = int(round(value))
            else:
                out[key] = round(value, 3)
        return out

    def encode(self, block):
        values = []
        for key in self.keys:
            v = float(block.get(key, DEFAULTS[key]))
            if key in SWITCHED:
                lo, hi = SWITCHED[key]
                value = .2 + .8 * (v - lo) / (hi - lo) if v else 0.
            elif key == 'limiter':
                value = v
            else:
                lo, hi = self.ranges[key]
                value = (v - lo) / (hi - lo)
            values.append(float(np.clip(value, 0., 1.)))
        return np.asarray(values)

    def decode(self, vector):
        if len(vector) != len(self.keys):
            raise ValueError('Wrong number of audio coordinates')
        block = dict(self.initial)
        for key, value in zip(self.keys, vector):
            value = float(np.clip(value, 0., 1.))
            if key in SWITCHED:
                lo, hi = SWITCHED[key]
                block[key] = 0. if value < .2 else lo + (hi - lo) * ((value - .2) / .8)
            elif key == 'limiter':
                block[key] = int(value >= .5)
            else:
                lo, hi = self.ranges[key]
                block[key] = lo + (hi - lo) * value
        return self.canonical(block)

    def grid(self):
        eq = self.ranges['eq_low'][1]
        proposed = {
            'speed': [.75, .85, 1., 1.15, 1.25],
            'vol': [self.initial['vol'] - 6, 0, self.initial['vol'] + 6],
            'eq_low': [-eq, 0, eq], 'eq_mid': [-eq, 0, eq],
            'eq_high': [-eq, 0, eq], 'hp': [0, 40, 95, 150],
            'lp': [0, 6000, 8000, 11000, 16000],
            'NR': [0, .15, .3, .5], 'comp': [0, .2, .4, .6],
            'de-ess': [0, .15, .3, .5], 'reverb': [0, .1, .25, .4],
            'noise_gate': [0, -60, -50, -40, -30],
            'pan': [-1, -.5, 0, .5, 1], 'limiter': [0, 1]}
        grid = {}
        for key in self.keys:
            values = list(proposed[key])
            if key in self.ranges:
                lo, hi = self.ranges[key]
                values = [float(np.clip(v, lo, hi)) for v in values]
            grid[key] = list(dict.fromkeys([self.initial[key], *values]))
        return grid


def worst_case_budget(eq_limit=4., freeze_vol=False, held=None):
    """The largest minimum budget a starting block can need: every one of its
    values off the trial grid. Lets a caller refuse a budget before rendering."""
    eq_probe = float(eq_limit) / 2.
    if round(eq_probe, 3) in (0., -eq_limit, eq_limit):
        # At very small bounds, export rounding may put the midpoint on the
        # grid. An outside incumbent is still a conservative off-grid probe.
        eq_probe = float(eq_limit) + 1.
    off_grid = {'speed': .95, 'vol': 1., 'eq_low': eq_probe, 'eq_mid': eq_probe, 'eq_high': eq_probe,
                'hp': 60, 'lp': 9000, 'NR': .05, 'comp': .05, 'de-ess': .05,
                'reverb': .05, 'noise_gate': -55, 'pan': .25, 'limiter': 1}
    space = AudioSpace(off_grid, eq_limit, freeze_vol, held)
    return 1 + sum(len(v) - 1 for v in space.grid().values()) + len(space.keys)


def _audit(count, budget, baseline_score, final_score, baseline, final_block,
           search_score, search_block, tolerance, reserve, steps, tried, grid_best,
           trial_results, space, failures, objective, method, stopped=None, early=None,
           trial_candidates=None, anchor_trials=None):
    """The report of a search, whether it ran to the end or stopped early."""
    unscored = [axis for axis, rows in trial_results.items()
                if not any(row['status'] == 'scored' and row['value'] != space.initial[axis]
                           for row in rows)]
    return {'schema_version': 1, 'objective': objective, 'method': method,
            'evaluations': count, 'budget': budget,
            'baseline_score': baseline_score, 'best_score': final_score,
            'baseline_block': baseline, 'candidate_block': final_block,
            'search_best_score': search_score, 'search_best_block': search_block,
            'pruning': {'tolerance': tolerance, 'reserved_evaluations': reserve,
                        'steps': steps,
                        'reverted': [s['field'] for s in steps if s['decision'] == 'reverted'],
                        'kept': [s['field'] for s in steps if s['decision'] == 'kept']},
            'stopped': stopped, 'early_validation': early,
            'tested_values': tried, 'one_factor_best': grid_best,
            'trial_candidates': list(trial_candidates or []),
            'anchor_trials': list(anchor_trials or []),
            'field_trials': trial_results, 'fields_without_scored_alternative': unscored,
            'held': space.held, 'failures': failures,
            'all_free_fields_tried': True, 'all_free_fields_scored': not unscored}


def search_audio(initial, evaluate, *, method='nelder', budget=400,
                 eq_limit=4., freeze_vol=False, held=None,
                 tolerance=PRUNE_TOLERANCE, progress=None,
                 objective='ecapa_identity', checkpoint=None, anchors=None):
    """Return (best block, audit) without exceeding `budget` scoring calls.

    evaluate(block) returns (score, rendered_block). The latter may include a
    gain calculated to meet a production RMS target. A frozen volume is thus
    a constrained quantity, not a value ignored by the fitting process.
    `held` maps fields to the reason they keep their starting value. After the
    search, each changed field goes back to its starting value unless that
    costs more than `tolerance`; the pruned block stays within one tolerance
    of the best one found. One evaluation per free field is kept for this.

    `objective` only names what evaluate() scores, for the report.
    `anchors` are whole blocks the caller wants scored as they stand, after the
    one-factor trials (the report keeps each one's score and rendered block, in
    order, next to the one-factor candidates, for a caller that validates a
    shortlist). They count against the budget and may become the best block;
    their held fields keep the starting value, as everywhere in the search.
    After the one-factor and combined trials, and before the joint refinement,
    the search may stop (audit['stopped'] says why, and the baseline is
    returned): when the best block found gains less than `tolerance` over the
    baseline, or when checkpoint(best_block, baseline_block) answers
    {'proceed': False, 'reason': ..., 'validation': ...} (the refused block is
    then returned as the candidate, for the caller to report). A checkpoint
    that proceeds leaves the search as it was.
    """
    if method not in ('nelder', 'de'):
        raise ValueError('Audio search method must be nelder or de')
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Pruning tolerance must be finite and non-negative')
    space = AudioSpace(initial, eq_limit, freeze_vol, held)
    grid = space.grid()
    reserve = len(space.keys)
    minimum = 1 + sum(len(v) - 1 for v in grid.values()) + reserve
    budget = int(budget)
    if budget < minimum:
        raise ValueError(f'audio budget {budget} is too small: at least {minimum} '
                         'evaluations are required to try every free field '
                         'and prune the result')
    cache, tried, failures = {}, {k: [] for k in space.keys}, []
    count = 0
    limit = budget - reserve          # the search proper; pruning gets the rest
    best_score, best_block = -math.inf, None

    def score(block):
        nonlocal count, best_score, best_block
        block = space.canonical(block)
        key = tuple(block[k] for k in AUDIO_KEYS)
        if key in cache:
            return cache[key]
        if count >= limit:
            raise StopIteration
        count += 1
        try:
            value, rendered = evaluate(dict(block))
            value = float(value)
            if not math.isfinite(value):
                raise ValueError('Non-finite audio score')
            rendered = space.canonical(rendered)
        except Exception as exc:
            if best_block is None:
                raise
            failures.append({'block': block, 'error': str(exc)})
            cache[key] = (-math.inf, block)
            return cache[key]
        cache[key] = (value, rendered)
        if value > best_score + 1e-9:
            best_score, best_block = value, dict(rendered)
        if progress and (count == 1 or count % 25 == 0):
            progress(f'audio search {count}/{budget}; best score {best_score:.4f}')
        return value, rendered

    baseline_score, baseline = score(space.initial)
    best_block = dict(baseline)
    grid_best, trial_results, trial_candidates = {}, {}, []
    # Real one-factor trials, including both states of every switch. These are
    # part of the optimisation budget and feed the joint search, not a report
    # disconnected from the optimiser.
    for axis, values in grid.items():
        rows = []
        trial_results[axis] = []
        for value in values:
            trial = dict(space.initial, **{axis: value})
            result, rendered = score(trial)
            tried[axis].append(value)
            rows.append((result, value))
            trial_results[axis].append({'value': value,
                'score': result if math.isfinite(result) else None,
                'status': 'scored' if math.isfinite(result) else 'failed'})
            if math.isfinite(result):
                trial_candidates.append(dict(field=axis, value=value, score=result,
                                             block=dict(rendered)))
        value, setting = max(rows, key=lambda pair: pair[0])
        grid_best[axis] = {'value': setting, 'score': value if math.isfinite(value) else None}

    anchor_trials = []
    for anchor in anchors or []:
        if count >= limit:
            anchor_trials.append(dict(status='not tried (fit budget)'))
            continue
        # A held field keeps its starting value here too, whatever the block says.
        value, rendered = score(dict(anchor, **{key: space.initial[key] for key in space.held}))
        anchor_trials.append(dict(status='scored' if math.isfinite(value) else 'failed',
            score=value if math.isfinite(value) else None, block=dict(rendered)))

    # Combine promising changes before joint refinement. In particular a zero
    # in NR/comp/reverb must not start with SciPy's tiny default step .00025.
    coarse_limit = max(count, limit // 2)
    # Try the combined one-factor winners even with a short budget; otherwise
    # early fields could consume the whole coarse allowance before NR/reverb.
    joint = dict(space.initial)
    for key, row in grid_best.items():
        if row['score'] is not None and row['score'] > baseline_score + 1e-9:
            joint[key] = row['value']
    if count < limit:
        score(joint)
    for key, row in grid_best.items():
        if count >= coarse_limit:
            break
        score(dict(best_block, **{key: row['value']}))
    for axis, values in grid.items():
        anchor = dict(best_block)
        for value in values:
            if count >= coarse_limit:
                break
            score(dict(anchor, **{axis: value}))

    # Is the joint refinement worth its evaluations? It is the expensive part
    # (most of the budget) and it works on the one sentence the trials above
    # already used: it can polish a change that transfers, never make one
    # transfer.
    stopped, early = None, None
    coarse_gain = best_score - baseline_score
    if coarse_gain < tolerance:
        stopped = {'rule': 'no gain above the tolerance',
                   'reason': f'the best change gained {coarse_gain:+.4f}, less than the '
                             f'tolerance {tolerance:.4f}: the pruning would revert it'}
    elif checkpoint is not None:
        verdict = checkpoint(dict(best_block), dict(baseline)) or {}
        early = verdict.get('validation')
        if not verdict.get('proceed', True):
            stopped = {'rule': 'does not carry over to other sentences',
                       'reason': str(verdict.get('reason') or 'rejected by the checkpoint')}
    if stopped is not None:
        # Nothing worth proposing: the baseline. A candidate that failed its
        # checkpoint is returned as the (rejected) candidate it was, so the
        # caller can report what was refused.
        refused = stopped['rule'] != 'no gain above the tolerance'
        stopped.update(stage='trials', evaluations=count, gain=coarse_gain)
        if progress:
            progress(f'audio search {count}/{budget}; stopped before the joint '
                     f'refinement: {stopped["reason"]}')
        final_block, final_score = ((dict(best_block), best_score) if refused
                                    else (dict(baseline), baseline_score))
        return final_block, _audit(
            count, budget, baseline_score, final_score, baseline, final_block,
            best_score, best_block, tolerance, reserve, [], tried, grid_best,
            trial_results, space, failures, objective, method,
            stopped=stopped, early=early, trial_candidates=trial_candidates,
            anchor_trials=anchor_trials)

    from scipy.optimize import differential_evolution, minimize
    bounds = [(0., 1.)] * len(space.keys)

    def negative_score(vector):
        value, _ = score(space.decode(vector))
        return -value if math.isfinite(value) else 1e6

    try:
        remaining = limit - count
        if method == 'de' and remaining >= 10:
            # An explicit population avoids Sobol rounding above a small budget.
            population_size = min(64, max(5, remaining // 3))
            population = np.random.default_rng(0).random((population_size, len(space.keys)))
            population[0] = space.encode(best_block)
            generations = max(0, (remaining * 2 // 3) // population_size - 1)
            differential_evolution(negative_score, bounds, init=population,
                                   maxiter=generations, polish=False, seed=0, tol=1e-4)
        remaining = limit - count
        if remaining > len(space.keys):
            start = space.encode(best_block)
            simplex = np.tile(start, (len(start) + 1, 1))
            for i, value in enumerate(start):
                simplex[i + 1, i] = value + .1 if value <= .9 else value - .1
            minimize(negative_score, start, method='Nelder-Mead', bounds=bounds,
                     options={'maxiter': limit, 'maxfev': remaining,
                              'initial_simplex': simplex, 'xatol': 1e-3, 'fatol': 1e-5})
    except StopIteration:
        pass

    # Pruning. The block is adopted or rejected as a whole, so a field that
    # does nothing for the score keeps whatever value the optimiser left it at
    # and rides along with the fields that do. Each changed field therefore
    # goes back to its starting value when the block without its change still
    # scores within `tolerance` of the best: the floor is fixed, so the total
    # loss is one tolerance, not one per field. The fields whose one-factor
    # trials moved the score least are tried first.
    search_score, search_block = best_score, dict(best_block)
    limit = budget
    swing = {}
    for axis, rows in trial_results.items():
        scores = [row['score'] for row in rows if row['score'] is not None]
        swing[axis] = max(scores) - min(scores) if len(scores) > 1 else 0.
    changed = sorted((k for k in space.keys if search_block[k] != baseline[k]),
                     key=lambda k: (swing.get(k, 0.), AUDIO_KEYS.index(k)))
    floor = search_score - tolerance
    pruned, pruned_score, steps = dict(search_block), search_score, []
    for key in changed:
        step = {'field': key, 'value': pruned[key], 'start': baseline[key],
                'one_factor_swing': swing.get(key, 0.)}
        try:
            value, rendered = score(dict(pruned, **{key: baseline[key]}))
        except StopIteration:            # cannot happen with the reserve; never guess
            steps.append(dict(step, decision='not tried (budget)'))
            break
        step['score_without'] = value if math.isfinite(value) else None
        if math.isfinite(value) and value >= floor:
            pruned, pruned_score = dict(rendered), value
            steps.append(dict(step, decision='reverted'))
        else:
            steps.append(dict(step, decision='kept'))
    report = _audit(count, budget, baseline_score, pruned_score, baseline, pruned,
                    search_score, search_block, tolerance, reserve, steps, tried,
                    grid_best, trial_results, space, failures, objective, method,
                    stopped=None, early=early, trial_candidates=trial_candidates,
                    anchor_trials=anchor_trials)
    if progress:
        progress(f'audio search {count}/{budget}; all {len(space.keys)} free fields tried, '
                 f'{len(report["pruning"]["reverted"])} change(s) pruned')
    return pruned, report

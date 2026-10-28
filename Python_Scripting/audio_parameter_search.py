"""Bounded search of the numeric fields of a voice [] block.

No XTTS/model dependencies: callers supply a function scoring an actual render.
Off is a separate state for filters and the gate, never an invalid cutoff. Every
free field receives explicit trials before any local/global joint refinement.
The result is then pruned: a change that does not earn its keep goes back to
its starting value. Scores are proxies; adoption on independent speech belongs
to the caller.
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


def search_audio(initial, evaluate, *, method='nelder', budget=400,
                 eq_limit=4., freeze_vol=False, held=None,
                 tolerance=PRUNE_TOLERANCE, progress=None):
    """Return (best block, audit) without exceeding `budget` scoring calls.

    evaluate(block) returns (score, rendered_block). The latter may include a
    gain calculated to meet a production RMS target. A frozen volume is thus
    a constrained quantity, not a value ignored by the fitting process.
    `held` maps fields to the reason they keep their starting value. After the
    search, each changed field goes back to its starting value unless that
    costs more than `tolerance`; the pruned block stays within one tolerance
    of the best one found. One evaluation per free field is kept for this.
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
            progress(f'audio search {count}/{budget}; best identity {best_score:.4f}')
        return value, rendered

    baseline_score, baseline = score(space.initial)
    best_block = dict(baseline)
    grid_best, trial_results = {}, {}
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
        value, setting = max(rows, key=lambda pair: pair[0])
        grid_best[axis] = {'value': setting, 'score': value if math.isfinite(value) else None}

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

    from scipy.optimize import differential_evolution, minimize
    bounds = [(0., 1.)] * len(space.keys)

    def objective(vector):
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
            differential_evolution(objective, bounds, init=population,
                                   maxiter=generations, polish=False, seed=0, tol=1e-4)
        remaining = limit - count
        if remaining > len(space.keys):
            start = space.encode(best_block)
            simplex = np.tile(start, (len(start) + 1, 1))
            for i, value in enumerate(start):
                simplex[i + 1, i] = value + .1 if value <= .9 else value - .1
            minimize(objective, start, method='Nelder-Mead', bounds=bounds,
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
    unscored = [axis for axis, rows in trial_results.items()
                if not any(row['status'] == 'scored' and row['value'] != space.initial[axis]
                           for row in rows)]
    report = {'schema_version': 1, 'objective': 'ecapa_identity', 'method': method,
              'evaluations': count, 'budget': budget,
              'baseline_score': baseline_score, 'best_score': pruned_score,
              'baseline_block': baseline, 'candidate_block': pruned,
              'search_best_score': search_score, 'search_best_block': search_block,
              'pruning': {'tolerance': tolerance, 'reserved_evaluations': reserve,
                          'steps': steps,
                          'reverted': [s['field'] for s in steps if s['decision'] == 'reverted'],
                          'kept': [s['field'] for s in steps if s['decision'] == 'kept']},
              'tested_values': tried, 'one_factor_best': grid_best,
              'field_trials': trial_results, 'fields_without_scored_alternative': unscored,
              'held': space.held, 'failures': failures,
              'all_free_fields_tried': True, 'all_free_fields_scored': not unscored}
    if progress:
        progress(f'audio search {count}/{budget}; all {len(space.keys)} free fields tried, '
                 f'{len(report["pruning"]["reverted"])} change(s) pruned')
    return pruned, report

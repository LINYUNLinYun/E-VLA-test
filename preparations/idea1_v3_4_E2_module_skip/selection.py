"""Pure, measured-cost FFN selection for the module-skip experiment.

Layer numbers and masks are one-based. Costs are net native savings in ms,
including the compensation add; no theoretical cost fallback is provided.
"""
from functools import lru_cache
import hashlib
import json
import math
import random

RANDOM_SEED = 341101
_EPS = 1e-9


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _finite(value, name, minimum=None):
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f'{name} must be finite' + (f' and >= {minimum}' if minimum is not None else ''))
    return value


@lru_cache(maxsize=None)
def enumerate_ffn_masks(n_layers=17):
    """All independent layer sets, including the empty set, in stable order."""
    if not isinstance(n_layers, int) or n_layers < 1:
        raise ValueError('n_layers must be a positive integer')
    masks = []

    def visit(layer, selected):
        if layer > n_layers:
            masks.append(tuple(selected))
            return
        visit(layer + 1, selected)
        selected.append(layer)
        visit(layer + 2, selected)
        selected.pop()

    visit(1, [])
    return tuple(sorted(masks))


def _measured_costs(cost_table):
    if cost_table.get('measured') is not True or cost_table.get('units') != 'ms':
        raise ValueError('a measured native cost table in ms is required')
    n_layers = int(cost_table.get('n_layers', 17))
    costs, eligible, observations = {}, set(), {}
    for row in cost_table['layer_costs']:
        layer = int(row['layer'])
        if layer in costs or not 1 <= layer <= n_layers:
            raise ValueError(f'duplicate or out-of-range layer {layer}')
        saving = _finite(row['ffn_saved_ms'], f'L{layer} FFN savings')
        noise = _finite(row.get('measurement_noise_ms', 0), f'L{layer} measurement noise', 0)
        costs[layer] = saving
        observations[layer] = dict(row)
        if saving > noise and saving > 0:
            eligible.add(layer)
    if set(costs) != set(range(1, n_layers + 1)):
        raise ValueError('cost table must contain every FFN layer exactly once')
    return n_layers, costs, eligible, observations


def freeze_budget(cost_table, budget_id, masks=None):
    """Freeze the same cost-only legal set for raw/error/fixed/no-comp FFN.

    If the prescribed window has no legal mask, freeze the closest achievable
    *cost tier*. An exact distance tie chooses the larger savings. This choice
    never reads scores or closed-loop outcomes. Original readings stay present.
    """
    if budget_id not in ('B4', 'B6'):
        raise ValueError('budget_id must be B4 or B6')
    n_layers, costs, eligible, observations = _measured_costs(cost_table)
    anchor = cost_table['anchors'][budget_id]
    target = _finite(anchor['saved_ms'], 'anchor savings', 0)
    fast_ms = _finite(anchor['fast_ms'], 'anchor Fast latency', 0)
    tau = max(0.5, 0.01 * fast_ms)
    candidates = []
    supplied = enumerate_ffn_masks(n_layers) if masks is None else masks
    seen = set()
    for raw_mask in supplied:
        mask = tuple(int(layer) for layer in raw_mask)
        if mask != tuple(sorted(set(mask))) or any(b - a == 1 for a, b in zip(mask, mask[1:])):
            raise ValueError('masks must contain sorted, unique, non-adjacent layer numbers')
        if any(layer < 1 or layer > n_layers for layer in mask):
            raise ValueError('mask layer outside measured range')
        if mask in seen:
            continue
        seen.add(mask)
        if not set(mask) <= eligible:
            continue
        candidates.append(dict(mask=list(mask), saved_ms=math.fsum(costs[layer] for layer in mask),
                               skip_count=len(mask)))
    if not candidates:
        raise ValueError('no legal masks using measured positive FFN savings')
    matched = [row for row in candidates if target - tau - _EPS <= row['saved_ms'] <= target + tau + _EPS]
    nearest = not matched
    if nearest:
        actual = min((row['saved_ms'] for row in candidates), key=lambda x: (abs(x - target), -x))
        matched = [row for row in candidates if abs(row['saved_ms'] - actual) <= _EPS]
        effective_window = [actual, actual]
    else:
        actual = None
        effective_window = [target - tau, target + tau]
    matched.sort(key=lambda row: tuple(row['mask']))
    result = dict(schema_version=1, budget_id=budget_id, n_layers=n_layers,
                  anchor_saved_ms=target, anchor_fast_ms=fast_ms, tau_ms=tau,
                  prescribed_window_ms=[target - tau, target + tau],
                  effective_window_ms=effective_window, nearest_tier=nearest,
                  nearest_saved_ms=actual, cost_table_sha256=_digest(cost_table),
                  candidates=matched, eligible_layers=sorted(eligible),
                  measured_layer_costs=[observations[layer] for layer in sorted(observations)],
                  excluded_layers=sorted(set(costs) - eligible),
                  cost_model='sum of measured per-layer net FFN savings',
                  nearest_tie_rule='larger savings at equal distance', numerical_epsilon_ms=_EPS)
    result['budget_hash'] = _digest(result)
    return result


def _scores(scores, n_layers):
    if isinstance(scores, dict):
        values = {int(layer): _finite(score, f'L{layer} score', 0) for layer, score in scores.items()}
        if set(values) != set(range(1, n_layers + 1)):
            raise ValueError('scores must cover every measured layer')
        return values
    if len(scores) != n_layers:
        raise ValueError(f'expected {n_layers} scores')
    return {index + 1: _finite(score, f'L{index + 1} score', 0) for index, score in enumerate(scores)}


def select_ffn(scores, frozen_budget):
    """Minimize summed sensitivity within an already frozen measured window."""
    values = _scores(scores, frozen_budget['n_layers'])
    candidates = frozen_budget['candidates']
    if not candidates:
        raise ValueError('frozen budget contains no candidates')
    # Cost is only a stable tie break after sensitivity; no window is widened.
    best = min(candidates, key=lambda row: (math.fsum(values[layer] for layer in row['mask']),
                                           abs(row['saved_ms'] - frozen_budget['anchor_saved_ms']),
                                           tuple(row['mask'])))
    return list(best['mask'])


def mask_record(mask, frozen_budget):
    """Return actual skip count, support, and frozen estimated savings."""
    wanted = list(mask)
    for row in frozen_budget['candidates']:
        if row['mask'] == wanted:
            return dict(row, support_layers=list(wanted), budget_id=frozen_budget['budget_id'],
                        budget_hash=frozen_budget['budget_hash'])
    raise ValueError('mask is outside the frozen candidate set')


def fixed_episode_mask(frozen_budget, case, root_seed=RANDOM_SEED):
    """Uniform one-mask-per-episode draw, independent of scheduling/requests."""
    identity = {key: case[key] for key in ('suite', 'task', 'init')}
    seed_key = ['E2/module_skip/ffn_fixed', int(root_seed), frozen_budget['budget_id'], identity]
    seed = int.from_bytes(hashlib.sha256(json.dumps(seed_key, sort_keys=True, separators=(',', ':')).encode()).digest()[:16], 'big')
    rows = frozen_budget['candidates']
    if not rows:
        raise ValueError('frozen budget contains no candidates')
    return list(rows[random.Random(seed).randrange(len(rows))]['mask'])

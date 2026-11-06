"""Exact mixed A/F selection inside one frozen, measured cost window.

U minimizes the sum of nonnegative module sensitivities. W minimizes the sum
of sensitivity / standalone net saved ms, an additional density heuristic.
Both search exactly the same masks; AF uses its directly measured joint cost
for the budget, while its objective remains the sum of the A and F terms.
No costs are quantized and an empty window is an error, never widened.
"""
from functools import lru_cache
import hashlib
import json
import math
import operator
import os
from pathlib import Path
import tempfile
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_DIR = ROOT / 'results/idea1_v3_4_E2_mixed_cost/candidates'
STATE_ORDER = ('keep', 'attn', 'ffn', 'af')
_BYTE_BITS = ((np.arange(256, dtype=np.uint16)[:, None] >> np.arange(8)) & 1).astype(np.float64)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(value, name, minimum=None):
    value = float(value)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f'{name} must be finite' + (f' and >= {minimum}' if minimum is not None else ''))
    return value


def _table(cost_table):
    if cost_table.get('measured') is not True or cost_table.get('units') != 'ms':
        raise ValueError('Measured native net savings in ms are required')
    n = operator.index(cost_table.get('n_layers', 17))
    if not 1 <= n <= 17:
        raise ValueError('Expected one to seventeen candidate layers')
    rows = {}
    for row in cost_table['layer_costs']:
        layer = operator.index(row['layer'])
        if layer in rows or not 1 <= layer <= n:
            raise ValueError('Duplicate or out-of-range layer in measured costs')
        rows[layer] = row
    if set(rows) != set(range(1, n + 1)):
        raise ValueError('Every candidate layer requires A/F/AF measured costs')
    costs, normalizers, eligible, readings = [], [], [], []
    for layer in range(1, n + 1):
        row = rows[layer]
        c = [_finite(row[f'{name}_saved_ms'], f'L{layer} {name} net saved ms')
             for name in ('attn', 'ffn', 'af')]
        noise = [_finite(row.get(f'{name}_noise_ms', row.get('measurement_noise_ms', 0)),
                         f'L{layer} {name} measurement noise', 0)
                 for name in ('attn', 'ffn', 'af')]
        valid = [value > 0 and value > error for value, error in zip(c, noise)]
        # AF can remain useful even if standalone A is below its noise floor.
        # Both standalone denominators must nevertheless be strictly positive,
        # so the shared U/W candidate set never admits an undefined W score.
        valid[2] = valid[2] and c[0] > 0 and c[1] > 0
        costs.append([0., *c])
        normalizers.append(c[:2])
        eligible.append([True, *valid])
        readings.append(dict(layer=layer, costs_ms=dict(zip(STATE_ORDER[1:], c)),
                             noise_ms=dict(zip(STATE_ORDER[1:], noise)),
                             eligible=dict(zip(STATE_ORDER, [True, *valid]))))
    return n, costs, normalizers, eligible, readings


def legal_sequence_count(n_layers=17, eligible_states=None):
    """Count colored independent sets on a path, including the empty mask."""
    n = operator.index(n_layers)
    if n < 0:
        raise ValueError('Negative layer count')
    if eligible_states is None:
        numbers = [3] * n
    else:
        if len(eligible_states) != n:
            raise ValueError('Wrong eligibility shape')
        numbers = [sum(bool(value) for value in row[1:]) for row in eligible_states]
    counts = [0] * (n + 2)
    counts[n] = counts[n + 1] = 1
    for layer in range(n - 1, -1, -1):
        counts[layer] = counts[layer + 1] + numbers[layer] * counts[layer + 2]
    return counts[0]


def _enumerate_window(costs, eligible, lower, upper):
    n = len(costs)
    capacity = legal_sequence_count(n, eligible)
    attn = np.empty(capacity, dtype=np.uint32)
    ffn = np.empty(capacity, dtype=np.uint32)
    saved = np.empty(capacity, dtype=np.float64)
    count = 0

    def visit(layer, a, f, value):
        nonlocal count
        if value > upper:
            return  # Every eligible added state has strictly positive cost.
        if layer >= n:
            if lower <= value <= upper:
                attn[count], ffn[count], saved[count] = a, f, value
                count += 1
            return
        visit(layer + 1, a, f, value)
        bit = 1 << layer
        for state in (1, 2, 3):
            if eligible[layer][state]:
                visit(layer + 2, a | (bit if state & 1 else 0),
                      f | (bit if state & 2 else 0), value + costs[layer][state])

    visit(0, 0, 0, 0.)
    if not count:
        raise ValueError('No exact legal mixed masks in the frozen cost window')
    attn, ffn, saved = attn[:count], ffn[:count], saved[:count]
    keys = attn.astype(np.uint64) | (ffn.astype(np.uint64) << n)
    order = np.argsort(keys, kind='stable')
    return attn[order], ffn[order], saved[order]


def freeze_mixed_budget(cost_table, budget_id, candidate_dir=None):
    """Create/reuse a compact NPZ, return a JSON-friendly scientific identity.

    The only costs used for eligibility, filtering, and AF savings are measured
    net savings. The prescribed window is applied once to each full mask.
    """
    if budget_id not in ('B4', 'B6'):
        raise ValueError('budget_id must be B4 or B6')
    n, costs, normalizers, eligible, readings = _table(cost_table)
    anchor = cost_table['anchors'][budget_id]
    target = _finite(anchor['saved_ms'], 'anchor net saved ms', 0)
    fast_ms = _finite(anchor['fast_ms'], 'anchor Fast ms', 0)
    tau = max(.5, .01 * fast_ms)
    lower, upper = target - tau, target + tau
    cost_hash = _digest(cost_table)
    folder = Path(DEFAULT_CANDIDATE_DIR if candidate_dir is None else candidate_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{budget_id}_{cost_hash[:20]}.npz'
    metadata = dict(schema_version=1, cost_table_hash=cost_hash, budget_id=budget_id,
                    n_layers=n, cost_window_ms=[lower, upper], state_order=list(STATE_ORDER))
    if not path.exists():
        attn, ffn, saved = _enumerate_window(costs, eligible, lower, upper)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=folder, prefix=path.stem + '.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, attn_bits=attn, ffn_bits=ffn, costs=saved,
                                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    with np.load(path, allow_pickle=False) as archive:
        if json.loads(str(archive['metadata'])) != metadata:
            raise ValueError('Candidate archive does not match measured cost identity')
        count = len(archive['costs'])
        if not count or len(archive['attn_bits']) != count or len(archive['ffn_bits']) != count:
            raise ValueError('Invalid mixed candidate archive')
    budget = dict(schema_version=1, budget_id=budget_id, n_layers=n, units='ms',
        anchor_saved_ms=target, anchor_fast_ms=fast_ms, tau_ms=tau,
        prescribed_window_ms=[lower, upper], effective_window_ms=[lower, upper],
        nearest_tier=False, cost_quantization_ms=0., numerical_window_expansion_ms=0.,
        candidate_path=str(path), candidate_sha256=_sha(path), candidate_count=count,
        all_legal_sequences=legal_sequence_count(n), eligible_legal_sequences=legal_sequence_count(n, eligible),
        cost_table_hash=cost_hash, state_order=list(STATE_ORDER), state_costs=costs,
        normalizer_costs=normalizers, eligible_states=eligible, measured_layer_costs=readings,
        cost_model='sum of measured per-layer state cost; AF uses directly measured joint net savings',
        unweighted_objective='sum of selected module sensitivities',
        weighted_objective='sum of selected module sensitivity / standalone net saved ms',
        AF_objective='sum of separate A and F terms; a local additive approximation',
        AF_eligibility='joint saving above joint noise; both standalone normalizers positive',
        tie_break='minimum objective; closest total saving to anchor; smallest uint64(A_bits | F_bits << n)',
        optimality_scope='exact finite candidate search at float64 objective precision, no cost quantization')
    budget['budget_hash'] = _digest(budget)
    return budget


@lru_cache(maxsize=8)
def _load_candidates(path, expected_sha):
    if _sha(path) != expected_sha:
        raise ValueError('Frozen mixed candidate archive hash mismatch')
    with np.load(path, allow_pickle=False) as archive:
        attn, ffn, saved = (archive[name].copy() for name in ('attn_bits', 'ffn_bits', 'costs'))
        metadata = json.loads(str(archive['metadata']))
    if attn.dtype != np.uint32 or ffn.dtype != np.uint32 or saved.dtype != np.float64:
        raise ValueError('Wrong candidate array dtypes')
    if not (attn.ndim == ffn.ndim == saved.ndim == 1 and len(attn) == len(ffn) == len(saved)):
        raise ValueError('Wrong candidate array shapes')
    keys = attn.astype(np.uint64) | (ffn.astype(np.uint64) << metadata['n_layers'])
    if not len(keys) or np.any(keys[1:] <= keys[:-1]) or not np.isfinite(saved).all():
        raise ValueError('Candidate keys must be sorted and unique with finite costs')
    lower, upper = metadata['cost_window_ms']
    if np.any(saved < lower) or np.any(saved > upper):
        raise ValueError('Candidate costs exceed the exact frozen window')
    for value in (attn, ffn, saved, keys):
        value.setflags(write=False)
    return attn, ffn, saved, keys, metadata


def _candidates(budget):
    values = _load_candidates(str(budget['candidate_path']), budget['candidate_sha256'])
    metadata = values[-1]
    if (len(values[0]) != budget['candidate_count'] or metadata['n_layers'] != budget['n_layers']
            or metadata['cost_table_hash'] != budget['cost_table_hash']
            or metadata['budget_id'] != budget['budget_id']
            or metadata['cost_window_ms'] != budget['effective_window_ms']):
        raise ValueError('Candidate archive/budget shape mismatch')
    return values


def objective_scores(scores, budget, cost_weighted=False):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (budget['n_layers'], 2) or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('Expected finite nonnegative (layers,2) A/F sensitivities')
    if not cost_weighted:
        return scores.copy()
    denominator = np.asarray(budget['normalizer_costs'], dtype=np.float64)
    if denominator.shape != scores.shape:
        raise ValueError('Wrong standalone cost normalizer shape')
    result = np.full_like(scores, np.inf)
    np.divide(scores, denominator, out=result, where=denominator > 0)
    if not np.isfinite(result[denominator > 0]).all():
        raise ValueError('Density objective overflowed a positive cost normalizer')
    return result


@lru_cache(maxsize=8)
def _score_chunks(path, expected_sha):
    attn, ffn, _, _, metadata = _load_candidates(path, expected_sha)
    chunks = []
    for start in range(0, metadata['n_layers'], 8):
        width = min(8, metadata['n_layers'] - start)
        mask = (1 << width) - 1
        a = ((attn >> start) & mask).astype(np.uint8)
        f = ((ffn >> start) & mask).astype(np.uint8)
        a.setflags(write=False)
        f.setflags(write=False)
        chunks.append((start, width, a, f))
    return tuple(chunks)


def select_mixed(scores, budget, cost_weighted=False):
    """Return sorted 1-based A/F masks; the union obeys layer spacing."""
    attn, ffn, saved, keys, _ = _candidates(budget)
    weights = objective_scores(scores, budget, cost_weighted)
    # Undefined/nonpositive standalone normalizers forbid all corresponding
    # selected states during freezing. Replace their unused infinities for LUT
    # arithmetic so 0*inf cannot contaminate legal candidate objectives.
    finite = np.where(np.isfinite(weights), weights, 0.)
    objective = np.zeros(len(saved), dtype=np.float64)
    for start, width, a, f in _score_chunks(str(budget['candidate_path']), budget['candidate_sha256']):
        basis = _BYTE_BITS[:1 << width, :width]
        objective += (basis @ finite[start:start + width, 0])[a]
        objective += (basis @ finite[start:start + width, 1])[f]
    best = np.flatnonzero(objective == objective.min())
    if len(best) > 1:
        distance = np.abs(saved[best] - budget['anchor_saved_ms'])
        best = best[distance == distance.min()]
    index = int(best[0])  # NPZ is sorted by the declared stable mask key.
    n = budget['n_layers']
    return ([i + 1 for i in range(n) if int(attn[index]) & (1 << i)],
            [i + 1 for i in range(n) if int(ffn[index]) & (1 << i)])


def _bits(layers, n):
    values = list(layers)
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError('Boolean layer numbers are invalid')
    try:
        converted = [operator.index(value) for value in values]
    except TypeError as error:
        raise ValueError('Integer one-based layer numbers required') from error
    if converted != sorted(set(converted)) or any(not 1 <= value <= n for value in converted):
        raise ValueError('Layer masks must be sorted, unique, and in range')
    return sum(1 << (value - 1) for value in converted)


def validate_budget_mask(skip_attn, skip_ffn, budget):
    """O(log N) membership using one cached archive; return candidate index."""
    n = budget['n_layers']
    a, f = _bits(skip_attn, n), _bits(skip_ffn, n)
    support = a | f
    if support & (support << 1):
        raise ValueError('Skipped support layers must be non-adjacent')
    _, _, _, keys, _ = _candidates(budget)
    wanted = np.uint64(a | (f << n))
    index = int(np.searchsorted(keys, wanted))
    if index == len(keys) or keys[index] != wanted:
        raise ValueError('Mixed mask is outside the exact frozen cost-window candidates')
    return index


def mask_record(skip_attn, skip_ffn, budget, scores=None, cost_weighted=False):
    attn_layers, ffn_layers = list(skip_attn), list(skip_ffn)
    index = validate_budget_mask(attn_layers, ffn_layers, budget)
    _, _, saved, keys, _ = _candidates(budget)
    support = sorted(set(attn_layers) | set(ffn_layers))
    af = sorted(set(attn_layers) & set(ffn_layers))
    row = dict(budget_id=budget['budget_id'], budget_hash=budget['budget_hash'],
        selected_attn=attn_layers, selected_ffn=ffn_layers, selected_support=support,
        support_layers=support, selected_af=af, selected_module_count=len(attn_layers) + len(ffn_layers),
        support_count=len(support), attn_count=len(attn_layers), ffn_count=len(ffn_layers), af_count=len(af),
        saved_ms=float(saved[index]), estimated_saved_ms=float(saved[index]),
        candidate_index=index, candidate_key=int(keys[index]),
        saving_basis='sum of measured A/F/AF state savings; joint AF cost is not A+F')
    if scores is not None:
        weights = objective_scores(scores, budget, cost_weighted)
        row['selection_objective_value'] = math.fsum(
            [float(weights[layer - 1, 0]) for layer in attn_layers] +
            [float(weights[layer - 1, 1]) for layer in ffn_layers])
    return row


def benchmark_selection(scores, budget, repeats=20):
    """CPU-only timing; real policy timing must include mask selection too."""
    if repeats < 1:
        raise ValueError('A positive repeat count is required')
    _load_candidates.cache_clear()
    _score_chunks.cache_clear()
    tick = time.perf_counter()
    select_mixed(scores, budget, False)
    first_ms = (time.perf_counter() - tick) * 1000
    results = {}
    for weighted, name in ((False, 'U'), (True, 'W')):
        elapsed = []
        selected = None
        for _ in range(repeats):
            tick = time.perf_counter()
            selected = select_mixed(scores, budget, weighted)
            elapsed.append((time.perf_counter() - tick) * 1000)
        results[name] = dict(repeats=repeats, mean_ms=float(np.mean(elapsed)),
            median_ms=float(np.median(elapsed)), p95_ms=float(np.percentile(elapsed, 95)),
            selected_attn=selected[0], selected_ffn=selected[1])
    return dict(device='cpu', candidate_count=budget['candidate_count'], first_load_and_U_selection_ms=first_ms,
                cached_selection=results, scope='CPU selection only; add measured Full/VJP and native request costs separately')

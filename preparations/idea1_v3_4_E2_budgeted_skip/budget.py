"""Build arbitrary measured VLM-saving budgets for the frozen mixed selector.

The ratio denominator is an uninstrumented complete native VLM elapsed time,
never the complete request time, summed module costs, or instrumented profiles.
Positive targets require a nonempty mask; zero has exactly the empty candidate.
All mask filtering uses the original float64 costs and the requested window.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from preparations.idea1_v3_4_E2_mixed_cost.selection import (
    STATE_ORDER, _candidates, _digest, _enumerate_window, _table,
    legal_sequence_count,
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _number(value, name, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a number, not a boolean')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be a finite number') from error
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if minimum is not None and value < minimum:
        raise ValueError(f'{name} must be >= {minimum}')
    if maximum is not None and value > maximum:
        raise ValueError(f'{name} must be <= {maximum}')
    return value


def _reference(cost_path, explicit):
    if explicit is not None:
        value = _number(explicit, 'reference_vlm_ms', minimum=0)
        if value == 0:
            raise ValueError('reference_vlm_ms must be strictly positive')
        return value, dict(kind='explicit', value_ms=value, units='ms',
            scope='caller-supplied complete native VLM elapsed time', overrides_automatic=True)
    path = next((cost_path.parent / name for name in ('request_timings.jsonl', 'timings.jsonl')
                 if (cost_path.parent / name).exists()), None)
    if path is None:
        return None, dict(kind='unavailable', value_ms=None, units='ms',
            reason='No sibling request_timings.jsonl or timings.jsonl; a ratio requires an explicit reference')
    values, samples = [], {}
    timing_bytes = path.read_bytes()
    for line_number, line in enumerate(timing_bytes.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, TypeError) as error:
            raise ValueError(f'Invalid timing JSON at {path}:{line_number}') from error
        if row.get('phase') != 'hot' or row.get('route') != 'keep':
            continue
        if any(row.get(name, False) for name in ('profile', 'profiled', 'instrumented')):
            continue
        if 'vlm_ms' not in row:
            raise ValueError(f'Hot keep row lacks vlm_ms at {path}:{line_number}')
        value = _number(row['vlm_ms'], 'hot keep vlm_ms', minimum=0)
        if value == 0:
            raise ValueError('Hot keep complete VLM timing must be positive')
        values.append(value)
        samples.setdefault(str(row.get('sample', 'unknown')), []).append(value)
    if not values:
        raise ValueError('Timing file contains no uninstrumented phase=hot, route=keep VLM measurements')
    value = float(np.median(values))
    return value, dict(kind='measured_hot_keep', value_ms=value, units='ms',
        scope='complete native VLM elapsed time, excluding vision/AE/request wall',
        path=str(path.resolve()), sha256=hashlib.sha256(timing_bytes).hexdigest(), count=len(values),
        filter=dict(phase='hot', route='keep', field='vlm_ms', instrumented=False),
        aggregation='median over all matching hot keep requests',
        per_input=[dict(sample=sample, count=len(items), median_ms=float(np.median(items)))
                   for sample, items in sorted(samples.items())])


def max_reachable_saved_ms(state_costs, eligible_states):
    """Exact path DP over keep/A/F/AF eligibility and non-adjacent support."""
    n = len(state_costs)
    if len(eligible_states) != n:
        raise ValueError('Mismatched costs and eligibility')
    maximum = [0.] * (n + 2)
    for layer in range(n - 1, -1, -1):
        choices = [maximum[layer + 1]]
        choices.extend(float(state_costs[layer][state]) + maximum[layer + 2]
                       for state in (1, 2, 3) if eligible_states[layer][state])
        maximum[layer] = max(choices)
    if not math.isfinite(maximum[0]):
        raise ValueError('Reachable savings overflow finite measured milliseconds')
    return maximum[0]


def _write_json(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp',
                                         mode='w', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def prepare_budget(cost_table_path, output_dir, *, saved_ms=None, saved_fraction=None,
                   tolerance_ms=.5, reference_vlm_ms=None):
    """Return a JSON-friendly budget, also saved under a specification hash.

    Exactly one of saved_ms/saved_fraction is required. Fractions are in [0,1]
    and refer only to measured complete VLM time. Absolute-ms budgets can be
    built without a timing file; when present its reference is still recorded.
    The output is directly consumable by the old select_mixed and MixedPolicy.
    """
    if (saved_ms is None) == (saved_fraction is None):
        raise ValueError('Provide exactly one of saved_ms or saved_fraction')
    tolerance = _number(tolerance_ms, 'tolerance_ms', minimum=0)
    cost_path = Path(cost_table_path).resolve()
    cost_bytes = cost_path.read_bytes()
    table = json.loads(cost_bytes)
    n, costs, normalizers, eligible, readings = _table(table)
    reference, reference_source = _reference(cost_path, reference_vlm_ms)
    requested_fraction = None
    if saved_fraction is not None:
        requested_fraction = _number(saved_fraction, 'saved_fraction', minimum=0, maximum=1)
        if reference is None:
            raise ValueError('saved_fraction requires measured hot keep VLM timing or explicit reference_vlm_ms')
        target = requested_fraction * reference
        mode = 'saved_fraction'
    else:
        target = _number(saved_ms, 'saved_ms', minimum=0)
        mode = 'saved_ms'
    maximum = max_reachable_saved_ms(costs, eligible)
    if target > maximum:
        raise ValueError(f'Requested saving {target:.12g} ms exceeds max_reachable_saved_ms={maximum:.12g} '
                         'under the unchanged eligibility and spacing constraints')
    if reference is not None and target > reference:
        raise ValueError(f'Requested saving {target:.12g} ms exceeds complete VLM reference {reference:.12g} ms')
    zero = target == 0
    prescribed = [target - tolerance, target + tolerance]
    effective = [0., 0.] if zero else prescribed
    cost_hash, cost_file_hash = _digest(table), hashlib.sha256(cost_bytes).hexdigest()
    specification = dict(schema_version=1, cost_table_hash=cost_hash,
        cost_table_file_sha256=cost_file_hash, requested_mode=mode,
        requested_saved_ms=None if mode == 'saved_fraction' else target,
        requested_saved_fraction=requested_fraction, target_saved_ms=target, tolerance_ms=tolerance,
        reference_vlm_ms=reference, reference_source=reference_source,
        zero_budget=zero, positive_budget_requires_nonempty_mask=True,
        effective_window_ms=effective)
    spec_hash = _digest(specification)
    budget_id = 'vlm-skip-' + spec_hash[:24]
    folder = Path(output_dir).resolve() / budget_id
    candidate_path, artifact_path = folder / 'candidates.npz', folder / 'budget.json'
    if artifact_path.exists():
        existing = load_budget(artifact_path)
        if existing['specification_hash'] != spec_hash or existing['specification'] != specification:
            raise ValueError('Existing budget artifact has a different specification')
        return existing
    metadata = dict(schema_version=1, cost_table_hash=cost_hash, budget_id=budget_id,
        n_layers=n, cost_window_ms=effective, state_order=list(STATE_ORDER),
        budget_specification_hash=spec_hash)
    if zero:
        attn = np.asarray([0], dtype=np.uint32)
        ffn = np.asarray([0], dtype=np.uint32)
        saved = np.asarray([0.], dtype=np.float64)
    else:
        try:
            attn, ffn, saved = _enumerate_window(costs, eligible, *effective)
        except ValueError as error:
            raise ValueError(f'No legal nonempty mixed mask in requested window {effective}; '
                             f'max_reachable_saved_ms={maximum:.12g}') from error
        nonempty = (attn | ffn) != 0
        attn, ffn, saved = attn[nonempty], ffn[nonempty], saved[nonempty]
        if not len(saved):
            raise ValueError(f'No legal nonempty mixed mask in requested window {effective}; '
                             f'max_reachable_saved_ms={maximum:.12g}')
    folder.mkdir(parents=True, exist_ok=True)
    if candidate_path.exists():
        with np.load(candidate_path, allow_pickle=False) as archive:
            same = (json.loads(str(archive['metadata'])) == metadata
                    and np.array_equal(archive['attn_bits'], attn)
                    and np.array_equal(archive['ffn_bits'], ffn)
                    and np.array_equal(archive['costs'], saved))
        if not same:
            raise ValueError('Existing candidate file differs from the requested exact budget')
    else:
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=folder, prefix='candidates.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, attn_bits=attn, ffn_bits=ffn, costs=saved,
                                    metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, candidate_path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    budget = dict(schema_version=2, budget_id=budget_id, budget_kind='arbitrary_vlm_saved_time',
        n_layers=n, units='ms', artifact_path=str(artifact_path),
        specification_hash=spec_hash, specification=specification,
        requested_mode=mode, requested_saved_ms=specification['requested_saved_ms'],
        requested_saved_fraction=requested_fraction, target_saved_ms=target,
        anchor_saved_ms=target, anchor_fast_ms=None,
        anchor_fast_scope='No complete-request Fast anchor; this budget concerns VLM time only',
        tau_ms=tolerance, tolerance_ms=tolerance, zero_budget=zero,
        positive_budget_requires_nonempty_mask=True,
        prescribed_window_ms=prescribed, effective_window_ms=effective,
        max_reachable_saved_ms=maximum, reference_vlm_ms=reference,
        reference_source=reference_source, ratio_scope='complete native VLM elapsed time, not complete request',
        candidate_path=str(candidate_path), candidate_sha256=_sha(candidate_path), candidate_count=len(saved),
        cost_table_path=str(cost_path), cost_table_hash=cost_hash, cost_table_file_sha256=cost_file_hash,
        state_order=list(STATE_ORDER), state_costs=costs, normalizer_costs=normalizers,
        eligible_states=eligible, measured_layer_costs=readings,
        all_legal_sequences=legal_sequence_count(n), eligible_legal_sequences=legal_sequence_count(n, eligible),
        nearest_tier=False, cost_quantization_ms=0., numerical_window_expansion_ms=0.,
        cost_model='sum of measured net state savings; joint AF uses its directly measured cost',
        unweighted_objective='sum of selected module sensitivities',
        weighted_objective='sum of selected module sensitivity / standalone net saved ms',
        AF_objective='sum of separate A and F terms; local additive approximation',
        AF_eligibility='joint saving above joint noise; both standalone normalizers positive',
        tie_break='minimum objective; closest total saving to target; smallest uint64(A_bits | F_bits << n)',
        optimality_scope='exact finite candidate search at float64 precision; no quantization or relaxed window')
    budget['budget_hash'] = _digest(budget)
    _candidates(budget)  # Validate exact compatibility before publishing JSON.
    _write_json(artifact_path, budget)
    return budget


def load_budget(path):
    """Read-only load with budget, specification, archive, and metadata checks."""
    path = Path(path).resolve()
    if path.is_dir():
        path = path / 'budget.json'
    budget = json.loads(path.read_text())
    scientific = dict(budget)
    expected = scientific.pop('budget_hash', None)
    if not expected or _digest(scientific) != expected:
        raise ValueError('Budget artifact hash mismatch')
    if _digest(budget['specification']) != budget['specification_hash']:
        raise ValueError('Budget specification hash mismatch')
    for source_path, source_hash, label in [
        (budget['cost_table_path'], budget['cost_table_file_sha256'], 'Calibration cost table'),
        *([(budget['reference_source']['path'], budget['reference_source']['sha256'],
            'Measured VLM reference timing')]
          if budget['reference_source']['kind'] == 'measured_hot_keep' else []),
    ]:
        try:
            actual_hash = _sha(source_path)
        except OSError as error:
            raise ValueError(f'{label} source missing or unreadable: {source_path}; prepare a new budget') from error
        if actual_hash != source_hash:
            raise ValueError(f'{label} source changed (SHA256 mismatch): {source_path}; prepare a new budget')
    if _sha(budget['candidate_path']) != budget['candidate_sha256']:
        raise ValueError('Budget candidate archive hash mismatch')
    values = _candidates(budget)
    if values[-1].get('budget_specification_hash') != budget['specification_hash']:
        raise ValueError('Candidate archive specification mismatch')
    if budget['zero_budget']:
        if (budget['effective_window_ms'] != [0., 0.] or len(values[0]) != 1
                or values[0][0] != 0 or values[1][0] != 0 or values[2][0] != 0):
            raise ValueError('A zero budget must contain exactly the empty mask')
    elif np.any((values[0] | values[1]) == 0):
        raise ValueError('A positive saving budget cannot contain the empty mask')
    return budget

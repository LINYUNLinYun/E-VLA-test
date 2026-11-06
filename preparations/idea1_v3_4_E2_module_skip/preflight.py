"""Bounded checks of new gates, five-request routing, and client isolation.

The old eight-sample observation bank supplies real inputs. Its timestamps are
not treated as period-five prediction-quality evidence. Numerical mismatches
fail this gate; only a two-client allocation OOM permits one-client fallback.
"""
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_E2_cycle3.common import coefficients
from .common import CONDITIONS, ROOT
from .model import Selector, forecast_error, verify_eager_graph
from .policy import Policy


def clear(policies):
    for policy in policies:
        policy.selector.clear()
        policy.reset()
    gc.collect()
    torch.cuda.empty_cache()


def _condition(name, budget='B4'):
    return dict(next(row for row in CONDITIONS
                     if row['condition'] == name and row['budget'] == budget))


def _history_digest(policy):
    """Exact tensor-byte identities, bounded to the client's two Full banks."""
    def update(digest, value):
        if isinstance(value, dict):
            for name in sorted(value, key=str):
                digest.update(str(name).encode())
                update(digest, value[name])
        else:
            array = value.detach().cpu().numpy()
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    rows = []
    for history in policy.history:
        digest = hashlib.sha256()
        update(digest, history['residuals'])
        rows.append(dict(step=int(history['step']), sha256=digest.hexdigest()))
    return rows


def _snapshot(result, policy, hash_history=True):
    route = result['route']
    fields = ('path', 'D', 'selected_skip', 'mask_refresh', 'mask_age_env_steps',
              'full_age_rounds', 'full_age_env_steps', 'history_full_steps',
              'has_second_history', 'extrapolation_ratio', 'scoring_mode',
              'score_source_full_steps', 'steady_vjp_count', 'patch_layers',
              'current_observation_kv')
    return dict(raw=result['raw'].copy(), actions=result['actions'].copy(),
                route={name: route[name] for name in fields},
                arrays={name: value.copy() for name, value in result['arrays'].items()},
                history=_history_digest(policy) if hash_history else None)


def _compare(expected, actual, warm=False):
    differences = []
    for name in ('raw', 'actions'):
        if not np.allclose(expected[name], actual[name], atol=1e-5, rtol=1e-5):
            differences.append(name)
    fields = set(expected['route']) - ({'scoring_mode'} if warm else set())
    for name in fields:
        if expected['route'][name] != actual['route'][name]:
            differences.append('route.' + name)
    for name in ('gate_gradient', 'module_score', 'prediction_error_rms'):
        if name in expected['arrays']:
            if name not in actual['arrays'] or not np.allclose(
                    expected['arrays'][name], actual['arrays'][name], atol=1e-4, rtol=1e-3):
                differences.append('arrays.' + name)
    if expected['history'] != actual['history']:
        differences.append('history')
    if differences:
        raise AssertionError('Preflight mismatch: ' + ', '.join(differences))


def _gate_replays(backend, items, out):
    """Six A/F graph/eager checks, then one old block warm-score check."""
    histories = []
    current = None
    for observation, prompt in items:
        current, _ = backend.prepare_modules(observation, prompt, capture=True)
        histories.append(current.pop('residuals'))
    error = forecast_error(histories[2], histories[1], histories[0], 1., structure='mixed')
    del histories
    rows = []
    for structure in ('ffn', 'mixed'):
        selector = Selector(backend, structure=structure)
        try:
            nonzero = error[:, 1] if structure == 'ffn' else error
            # Contiguous native error coordinates are copied into graph buffers.
            nonzero = nonzero.contiguous()
            zero = torch.zeros_like(nonzero)
            for mode, direction in (('R', None), ('E-zero', zero), ('E-observed', nonzero)):
                proof = verify_eager_graph(selector, current, 341201, coefficients(), error=direction)
                if mode == 'E-zero' and not np.array_equal(selector.grad.float().cpu().numpy(),
                                                          np.zeros_like(selector.grad.float().cpu().numpy())):
                    raise AssertionError(f'{structure}: zero error gave nonzero graph gradient')
                rows.append(dict(structure=structure, mode=mode, passed=True, **proof))
                print(json.dumps(dict(preflight='gate_replay', structure=structure,
                                      mode=mode, passed=True)), flush=True)
        finally:
            selector.clear()
        del selector, nonzero, zero
    del error
    gc.collect()
    torch.cuda.empty_cache()

    from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import Selector as OldSelector
    selector = Selector(backend, structure='block')
    try:
        grad, score, _ = selector.run(current, 341202, coefficients())
        raw = selector.last_raw.float().cpu().numpy().copy()
    finally:
        selector.clear()
    del selector
    old = OldSelector(backend)
    try:
        _, old_grad, old_score, _ = old.run(current, 341202, coefficients(), k=4)
        old_raw = old.raw.float().cpu().numpy().copy()
        checks = dict(raw_equal=bool(np.allclose(raw, old_raw, atol=1e-5, rtol=1e-5)),
                      gradient_equal=bool(np.allclose(grad, old_grad, atol=1e-4, rtol=1e-3)),
                      score_equal=bool(np.allclose(score, old_score, atol=1e-4, rtol=1e-3)))
        if not all(checks.values()):
            raise AssertionError(('block R old Selector compatibility', checks))
        rows.append(dict(structure='block', mode='old-R-warm', passed=True, **checks))
    finally:
        old.clear()
    del old, current
    atomic(out / 'selector_gate.json', dict(passed=True, checks=rows,
        native_error_source='three saved real observations; not period-five prediction-quality evidence',
        tolerance=dict(raw_actions=dict(atol=1e-5, rtol=1e-5),
                       gradient_score=dict(atol=1e-4, rtol=1e-3))))
    return rows


def _constant_route_and_warm(backend, budgets, item, suite, out):
    observation, prompt = item
    expected = []
    policy = Policy(backend, budgets)
    try:
        policy.configure(_condition('F-R+'))
        for replan in range(10):
            result = policy.predict(observation, prompt, suite, 341300 + replan)
            expected.append(_snapshot(result, policy, hash_history=False))
            policy.acknowledge(result['actions'][:5], None)
    finally:
        clear([policy])
    del policy
    policy = Policy(backend, budgets)
    routes = []
    try:
        policy.configure(_condition('F-E+'))
        for replan in range(15):
            previous = list(policy.history)
            result = policy.predict(observation, prompt, suite, 341300 + replan)
            route = result['route']
            full, step = replan % 5 == 0, replan * 5
            if result['step'] != step or route['path'] != ('full' if full else 'fast'):
                raise AssertionError(('period-five route', replan, route))
            if route['D'] != (10 if full else 2) or route['steady_vjp_count'] != int(full):
                raise AssertionError(('Full-only one VJP', replan, route))
            expected_history = [s for s in (0, 25, 50) if s <= step][-2:]
            if route['history_full_steps'] != expected_history:
                raise AssertionError(('Full history commit order', replan, route))
            if not route['current_observation_kv']:
                raise AssertionError(('current K/V', replan))
            if not full:
                expected_ratio = 0. if replan < 5 else (replan % 5) / 5
                if not np.isclose(route['extrapolation_ratio'], expected_ratio):
                    raise AssertionError(('hold/extrapolation ratio', replan, route))
                if route['mask_refresh'] or route['patch_layers'] != route['selected_skip']:
                    raise AssertionError(('Fast reused Full mask/patch', replan, route))
                if route['has_second_history'] != (replan >= 5):
                    raise AssertionError(('history depth', replan, route))
            if replan < 10:
                _compare(expected[replan], _snapshot(result, policy, hash_history=False), warm=True)
                if route['scoring_mode'] != 'warm_original':
                    raise AssertionError(('first two Full use raw scores', replan, route))
            else:
                if route['scoring_mode'] != 'comp_error' or route['score_source_full_steps'] != [0, 25, 50]:
                    raise AssertionError(('third Full error source', replan, route))
                if replan == 10:
                    if [row['step'] for row in previous] != [0, 25]:
                        raise AssertionError('Third Full did not read two prior histories')
                    actual_error = forecast_error(policy.history[-1]['residuals'],
                        previous[-1]['residuals'], previous[-2]['residuals'], 1., structure='ffn')
                    if not torch.equal(actual_error, policy.error_buffer):
                        raise AssertionError('E score read histories after committing current Full')
                if replan == 11 and (step != 55 or route['mask_age_env_steps'] != 5):
                    raise AssertionError('First Fast using E mask was not step55')
            routes.append(dict(replan=replan, step=step, **{
                key: route[key] for key in ('path', 'D', 'selected_skip', 'extrapolation_ratio',
                    'history_full_steps', 'score_source_full_steps', 'scoring_mode', 'steady_vjp_count')}))
            policy.acknowledge(result['actions'][:5], None)
            if full:
                print(json.dumps(dict(preflight='constant_route', replan=replan,
                    step=step, scoring_mode=route['scoring_mode'], passed=True)), flush=True)
    finally:
        clear([policy])
    del policy
    proof = dict(passed=True, warm_R_E_matching_requests=10, constant_observation_requests=15,
                 first_error_full_step=50, first_fast_using_error_mask_step=55,
                 history_read_before_write=True, fast_vjp_count=0, routes=routes,
                 scope='route/algebra check on one fixed real input; not extrapolation quality')
    atomic(out / 'route_gate.json', proof)
    return proof


def _isolation(backend, budgets, items, suite, out, force_single):
    if force_single:
        atomic(out / 'isolation_gate.json', dict(passed=True, actors=1,
            reason='Explicit one-client mode; single-client route gate passed',
            two_client_isolation_tested=False))
        return 1
    cfgs = [_condition('F-E+', 'B4'), _condition('F-R+', 'B6')]
    # Independent runs establish both clients' actual outputs before any
    # interleaving. These remain the evidence for a bounded OOM-only fallback.
    expected = []
    for actor, config in enumerate(cfgs):
        policy = Policy(backend, budgets)
        records = []
        try:
            policy.configure(config)
            for replan in range(15):
                observation, prompt = items[(replan // 5 + actor) % len(items)]
                result = policy.predict(observation, prompt, suite, 341500 + actor * 100 + replan)
                records.append(_snapshot(result, policy))
                policy.acknowledge(result['actions'][:5], None)
        finally:
            clear([policy])
        del policy
        expected.append(records)
        print(json.dumps(dict(preflight='independent_client', actor=actor,
                              requests=15, passed=True)), flush=True)
    policies = []
    checks = []
    try:
        for config in cfgs:
            policy = Policy(backend, budgets)
            policies.append(policy)
            policy.configure(config)
        for replan in range(15):
            for actor, policy in enumerate(policies):
                observation, prompt = items[(replan // 5 + actor) % len(items)]
                result = policy.predict(observation, prompt, suite, 341500 + actor * 100 + replan)
                _compare(expected[actor][replan], _snapshot(result, policy))
                checks.append(dict(actor=actor, replan=replan, passed=True))
                policy.acknowledge(result['actions'][:5], None)
        atomic(out / 'isolation_gate.json', dict(passed=True, actors=2, checks=checks,
            same_task_different_real_observations=True, history_comparison='exact SHA256 of FP32 tensor bytes',
            max_allocated_mib=torch.cuda.max_memory_allocated() / 2**20))
        print(json.dumps(dict(preflight='interleaved_clients', actors=2,
                              requests_per_client=15, passed=True)), flush=True)
        return 2
    except torch.cuda.OutOfMemoryError as error:
        atomic(out / 'single_actor_required.json', dict(
            reason='Two isolated module-skip client allocations OOM; independent clients passed',
            error=str(error), bounded_fallback=True))
        atomic(out / 'isolation_gate.json', dict(passed=True, actors=1,
            two_client_isolation_tested=False, independent_clients_passed=True,
            reason='OOM-only bounded fallback; no numerical mismatch was accepted'))
        return 1
    except AssertionError as error:
        atomic(out / 'isolation_gate.json', dict(passed=False, actors=0, checks=checks,
            reason=str(error), numerical_tolerances_unchanged=True))
        raise
    finally:
        clear(policies)


def preflight(backend, out, budgets, force_single=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    from preparations.kv_reuse_probe.probe import load_observation
    manifest = ROOT / 'results/idea1_v3_4_E2_comp_error_cost/manifest.json'
    sample = json.loads(manifest.read_text())['samples'][0]
    items = [load_observation(sample[name]) for name in ('older', 'last', 'current')]
    if len({prompt for _, prompt in items}) != 1:
        raise ValueError('Preflight bank must contain same-task observations with one prompt')
    suite = sample.get('suite', 'libero_spatial')
    gates = _gate_replays(backend, items, out)
    route = _constant_route_and_warm(backend, budgets, items[0], suite, out)
    actors = _isolation(backend, budgets, items, suite, out, force_single)
    atomic(out / 'compatibility_gate.json', dict(passed=True, actors=actors,
        selector_checks=len(gates), period=5, full_D=10, fast_D=2,
        warm_R_E_requests=route['warm_R_E_matching_requests'],
        constant_observation_requests=route['constant_observation_requests'],
        bank=str(manifest), bank_sample=sample.get('id'),
        observation_bank_scope='Real inputs only; old timestamps are not cycle-five quality evidence'))
    return actors

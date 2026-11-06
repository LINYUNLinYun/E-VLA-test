"""Bounded same-runtime checks of mixed scoring, routes and client isolation."""
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_E2_cycle3.common import coefficients
from preparations.idea1_v3_4_E2_module_skip.model import Selector, forecast_error, verify_eager_graph
from .policy import Policy
from .selection import select_mixed, mask_record

ROOT = Path(__file__).resolve().parents[2]


def clear(policies):
    for policy in policies:
        policy.selector.clear()
        policy.reset()
    gc.collect()
    torch.cuda.empty_cache()


def cfg(budget, weighted):
    from .common import CONDITIONS
    return dict(next(row for row in CONDITIONS
                     if row['budget'] == budget and bool(row['cost_weighted']) == weighted))


def snapshot(result, policy):
    def update(h, value):
        if isinstance(value, dict):
            for key in sorted(value, key=str):
                h.update(str(key).encode())
                update(h, value[key])
        else:
            x = value.detach().cpu().numpy()
            h.update(str(x.shape).encode())
            h.update(x.tobytes())
    histories = []
    for row in policy.history:
        h = hashlib.sha256()
        update(h, row['residuals'])
        histories.append((row['step'], h.hexdigest()))
    return dict(raw=result['raw'].copy(), actions=result['actions'].copy(),
                route=result['route'], history=histories,
                score=result['arrays'].get('module_score'))


def compare(a, b):
    for name in ('raw', 'actions'):
        if not np.allclose(a[name], b[name], atol=1e-5, rtol=1e-5):
            raise AssertionError('Client isolation numerical mismatch: ' + name)
    if a['route'] != b['route'] or a['history'] != b['history']:
        raise AssertionError('Client route/history isolation mismatch')
    if a['score'] is not None and not np.allclose(a['score'], b['score'], atol=1e-4, rtol=1e-3):
        raise AssertionError('Client scoring isolation mismatch')


def check_route(result, policy, previous, index):
    route = result['route']
    full = index % 5 == 0
    if result['step'] != index * 5 or route['D'] != (10 if full else 2):
        raise AssertionError('Period5 D10/D2/step routing mismatch')
    if route['steady_vjp_count'] != int(full) or route['mask_refresh'] != full:
        raise AssertionError('Scoring must occur once per Full and never on Fast')
    expected_history = [step for step in (0, 25, 50) if step <= index * 5][-2:]
    if route['history_full_steps'] != expected_history:
        raise AssertionError('Full history commit order mismatch')
    receipt = route['native_module_receipt']
    for kind in ('attn', 'ffn'):
        selected = route['selected_' + kind]
        actual = [] if full else selected
        if receipt['skip_' + kind] != actual or receipt['patched'][kind] != actual:
            raise AssertionError('Native module bypass/patch mismatch: ' + kind)
    if not receipt['current_observation_kv'] or not route['current_observation_kv']:
        raise AssertionError('Current K/V must be retained')
    if not full:
        ratio = 0. if index < 5 else (index % 5) / 5
        if not np.isclose(route['extrapolation_ratio'], ratio):
            raise AssertionError('Actual-step extrapolation ratio mismatch')
    if route['scoring_mode'] != ('warm_original' if index < 10 else 'comp_error'):
        raise AssertionError('First error-score Full must be step50')
    if index == 10:
        if [row['step'] for row in previous] != [0, 25]:
            raise AssertionError('Error score did not read two previous Full states')
        expected = forecast_error(policy.history[-1]['residuals'], previous[-1]['residuals'],
                                  previous[-2]['residuals'], 1., structure='mixed')
        if not torch.equal(expected, policy.error_buffer):
            raise AssertionError('Mixed error buffer differs from read-before-write forecast')


def preflight(backend, out, budgets, force_single=False):
    out = Path(out)
    from preparations.kv_reuse_probe.probe import load_observation
    bank = ROOT / 'results/idea1_v3_4_E2_comp_error_cost/manifest.json'
    sample = json.loads(bank.read_text())['samples'][0]
    items = [load_observation(sample[name]) for name in ('older', 'last', 'current')]
    suite = sample.get('suite', 'libero_spatial')
    if len({prompt for _, prompt in items}) != 1:
        raise ValueError('Preflight inputs must have the same prompt')
    histories = []
    for obs, prompt in items:
        current, _ = backend.prepare_modules(obs, prompt, capture=True)
        histories.append(current.pop('residuals'))
    error = forecast_error(histories[2], histories[1], histories[0], structure='mixed')
    selector = Selector(backend, structure='mixed')
    gates, choices = [], []
    try:
        for mode, direction in [('R', None), ('E-zero', torch.zeros_like(error)), ('E', error)]:
            proof = verify_eager_graph(selector, current, 341201, coefficients(),
                                       error=direction, structure='mixed')
            if mode == 'E-zero' and np.any(selector.grad.float().cpu().numpy()):
                raise AssertionError('Zero compensation error produced nonzero scores')
            gates.append(dict(mode=mode, passed=True, **proof))
            score = np.abs(selector.grad.float().cpu().numpy())
            for name, budget in budgets.items():
                for weighted in (False, True):
                    a, f = select_mixed(score, budget, cost_weighted=weighted)
                    choices.append(dict(mode=mode, budget=name, cost_weighted=weighted,
                                        attn=a, ffn=f, **mask_record(a, f, budget)))
            print(json.dumps(dict(preflight='mixed_score', mode=mode, passed=True)), flush=True)
    finally:
        selector.clear()
    from .cost_validation import run as validate_cost
    table = json.loads((ROOT / 'results/idea1_v3_4_E2_mixed_cost/hardware_probe/cost_table.json').read_text())
    validate_cost(backend, out.parent.parent, budgets, table, items[-1][0], items[-1][1],
                  int(sample['current']['seed']), score, histories)
    del selector, current, error, histories
    atomic(out / 'selector_gate.json', dict(passed=True, checks=gates, same_input_choices=choices))
    gc.collect()
    torch.cuda.empty_cache()
    configurations = [cfg('B4', False), cfg('B6', True)]
    independent = []
    for actor, config in enumerate(configurations):
        policy = Policy(backend, budgets)
        records = []
        try:
            policy.configure(config)
            for index in range(15):
                obs, prompt = items[(index // 5 + actor) % len(items)]
                previous = list(policy.history)
                result = policy.predict(obs, prompt, suite, 341500 + actor * 100 + index)
                check_route(result, policy, previous, index)
                records.append(snapshot(result, policy))
                policy.acknowledge(result['actions'][:5], None)
        finally:
            clear([policy])
        independent.append(records)
        print(json.dumps(dict(preflight='independent_mixed_client', actor=actor, passed=True)), flush=True)
    actors = 1
    policies = []
    checks = []
    if not force_single:
        try:
            for config in configurations:
                policy = Policy(backend, budgets)
                policy.configure(config)
                policies.append(policy)
            for index in range(15):
                for actor, policy in enumerate(policies):
                    obs, prompt = items[(index // 5 + actor) % len(items)]
                    result = policy.predict(obs, prompt, suite, 341500 + actor * 100 + index)
                    compare(independent[actor][index], snapshot(result, policy))
                    checks.append(dict(actor=actor, replan=index, passed=True))
                    policy.acknowledge(result['actions'][:5], None)
            actors = 2
        except torch.cuda.OutOfMemoryError as error:
            atomic(out / 'single_actor_required.json', dict(reason=str(error),
                   independent_clients_passed=True, bounded_oom_fallback=True))
        finally:
            clear(policies)
    atomic(out / 'route_gate.json', dict(passed=True, independent_requests=30,
           first_error_full_step=50, first_error_fast_step=55,
           current_kv_and_native_operator_receipts=True, history_read_before_write=True))
    atomic(out / 'isolation_gate.json', dict(passed=True, actors=actors,
           independent_clients_passed=True, interleaved_checks=checks,
           two_client_isolation_tested=actors == 2))
    atomic(out / 'compatibility_gate.json', dict(passed=True, actors=actors,
           bank=str(bank), selector_checks=len(gates), raw_action_tolerance=dict(atol=1e-5, rtol=1e-5),
           score_tolerance=dict(atol=1e-4, rtol=1e-3),
           bank_scope='Saved real same-task observations; timestamps are not closed-loop forecast quality'))
    return actors

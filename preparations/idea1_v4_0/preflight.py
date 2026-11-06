"""Bounded mechanism checks on one real input before closed-loop execution."""
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_selection_async.runtime import samples
from preparations.kv_reuse_probe.probe import load_observation
from .sampling import sample_initial
from .selection import Graph


def run(backend, policy, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sample = samples()[0]
    obs, prompt = load_observation(sample['current'])
    seed = int(sample['current']['seed'])
    context, _ = backend.prepare(obs, prompt, ())
    checks, saved = [], {}
    initial = np.random.default_rng(seed).standard_normal((50,32), dtype=np.float32)
    for steps in (1, 2):
        actions, raw, _ = backend.sample(context, seed, steps)
        a, r, cost = sample_initial(backend, context, initial, steps)
        delta = float(np.max(np.abs(raw.float().cpu().numpy()-r.float().cpu().numpy())))
        passed = bool(np.allclose(actions, a, atol=1e-5, rtol=1e-5) and delta <= 1e-5 and cost['nodes']['actual_NFE']==steps)
        checks.append(dict(kind='same_gaussian_native_sampler', steps=steps, passed=passed, raw_max_abs=delta))
        saved[f'native_d{steps}'] = raw.float().cpu().numpy()
    original = Graph(backend, context, seed, 'original')
    original_result = original.run(context, seed)
    original_raw = original.raw.float().cpu().numpy().copy()
    checks.append(dict(kind='original_four_directions', passed=original_result['gradient'].shape==(4,17)
        and np.isfinite(original_result['gradient']).all().item()))
    original.close()
    import torch
    context['selection_error'] = torch.zeros((17, context['seq'], 2048), device='cuda', dtype=torch.float32)
    error = Graph(backend, context, seed, 'error')
    error_result = error.run(context, seed)
    error_raw = error.raw.float().cpu().numpy().copy()
    checks.append(dict(kind='zero_forecast_error_has_zero_gate_gradient',
        passed=bool(np.max(np.abs(error_result['gradient']))==0 and
                    np.allclose(original_raw, error_raw, atol=1e-5, rtol=1e-5)),
        gradient_max_abs=float(np.max(np.abs(error_result['gradient']))),
        raw_max_abs=float(np.max(np.abs(original_raw-error_raw)))))
    error.close()
    from preparations.idea1_v3_4_selection_async.core import choose
    for count in (5,6,7,8):
        mask, _, _ = choose(original_result['gradient'], count)
        checks.append(dict(kind='all_dynamic_budgets_legal', skip=count, mask=mask,
            passed=len(mask)==count and all(b-a>1 for a,b in zip(mask,mask[1:]))))
    from .common import conditions
    for config in conditions()[1:]:
        policy.configure(config)
        routes = []
        count = 7 if config['scoring']=='compensation_error' else 2
        for index in range(count):
            result = policy.predict(obs,prompt,sample['suite'],seed+index)
            routes.append(result['route'])
            length = result['execute_length']
            policy.acknowledge(result['actions'][:length],np.tile([0.,0.,0.,1.],(length,1)))
        policy.finish()
        attempted = [r for r in routes if r['draft_attempted']]
        passed = bool(attempted and all(r['actual_fast_nfe']==config['fast_steps'] for r in attempted))
        if config['initial']=='kalman':
            passed = passed and all(r['kalman_used'] for r in attempted)
        if config['scoring']=='compensation_error':
            passed = passed and any(r['scoring_mode']=='comp_error_extrapolate' for r in routes)
        checks.append(dict(kind='policy_intervention_smoke',condition=config['id'],passed=bool(passed),
                           routes=[r['request_category'] for r in routes]))
    policy.clear()
    np.savez(output/'reference.npz', **saved)
    result = dict(passed=all(c['passed'] for c in checks), checks=checks, time=time.time())
    atomic(output/'complete.json', result)
    if not result['passed']:
        raise RuntimeError('New sampler or forecast-error gate mechanism failed')
    return result


if __name__ == '__main__':
    import argparse
    from preparations.idea1_v2_1.runtime_client import rpc
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    args = parser.parse_args()
    result = rpc(args.port, dict(op='preflight'))
    print(result, flush=True)

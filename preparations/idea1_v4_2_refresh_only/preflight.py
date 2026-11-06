import json
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_v12_compensation.server import selection
from preparations.kv_reuse_probe.probe import load_observation
from preparations.idea1_v4_1_warmstart.sampling import sample_gaussian
from .common import MASK, conditions
from .policy import Policy


def run(backend, output):
    output = Path(output)
    arrays, checks = {}, []
    original = backend.sample
    def forbidden(*args, **kwargs):
        raise AssertionError('Full/legacy action expert must never be called')
    backend.sample = forbidden
    try:
        for item in selection()[::10]:
            obs, prompt = load_observation(item['current'])
            suite, seed = item['case']['suite'], int(item['current']['seed'])
            native, _ = backend.prepare(obs, prompt, MASK)
            expected, raw, _ = sample_gaussian(backend, native, seed, 2)
            reference = raw.float().cpu().numpy()
            for cfg in conditions():
                policy = Policy(backend)
                policy.configure(cfg)
                for i in range(6):
                    result = policy.predict(obs, prompt, suite, seed+i)
                    route = result['route']
                    assert not route['full_ae_executed'] and route['D'] == 2
                    assert route['residual_refresh'] == (cfg['compensation'] is not None and i % 3 == 0)
                    if i == 0 and cfg['compensation'] is None:
                        passed = bool(np.allclose(reference, result['raw'], atol=1e-5, rtol=1e-5)
                            and np.allclose(expected, result['actions'], atol=1e-5, rtol=1e-5))
                        checks.append(dict(name=suite+'_native_v12_match', passed=passed,
                            max_abs=float(np.abs(reference-result['raw']).max())))
                    if cfg['compensation'] is not None:
                        history = policy.history
                        ctx, _ = backend.prepare_residual(obs, prompt, MASK,
                            history=history[-1]['residuals'],
                            older=history[-2]['residuals'] if len(history)==2 else None,
                            ratio=route['extrapolation_ratio'], reference=True)
                        a, r, _ = sample_gaussian(backend, ctx, seed+i, 2)
                        checks.append(dict(name=f'{suite}_patch_reference_{i}',
                            passed=bool(np.allclose(r.float().cpu().numpy(),result['raw'],atol=1e-5,rtol=1e-5)
                                and np.allclose(a,result['actions'],atol=1e-5,rtol=1e-5))))
                    if i == 0:
                        arrays[suite+'_'+cfg['id']] = result['raw']
                    policy.acknowledge(result['actions'][:5])
                assert policy.counts['full'] == 0 and policy.counts['draft'] == 6
                checks.append(dict(name=suite+'_'+cfg['id']+'_schedule', passed=True))
    finally:
        backend.sample = original
    result = dict(passed=all(c['passed'] for c in checks), checks=checks,
        full_ae_calls=0, native_encoder_graph=False, native_ae_graph=False)
    np.savez(output/'gate_actions.npz', **arrays)
    atomic(output/'preflight.json', result)
    if not result['passed']:
        raise RuntimeError('Residual-only refresh mechanism check failed')
    return result

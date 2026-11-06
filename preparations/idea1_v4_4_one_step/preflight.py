from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3.history import DecoderCoordinates
from preparations.idea1_v3_4_v12_compensation.server import selection
from preparations.kv_reuse_probe.probe import load_observation
from preparations.idea1_v4_1_warmstart.sampling import sample_gaussian, sample_initial
from preparations.idea1_v4_1_warmstart.schedule import mix
from .common import MASK, conditions
from .policy import Policy


def run(backend, output):
    output = Path(output)
    arrays, checks = {}, []
    original_sample, original_verify = backend.sample, backend.verify
    def forbidden(*args, **kwargs):
        raise AssertionError('Full/legacy action expert and verifier must never be called')
    backend.sample = backend.verify = forbidden
    try:
        for item in selection()[::10]:
            obs, prompt = load_observation(item['current'])
            suite, seed = item['case']['suite'], int(item['current']['seed'])
            firsts = []
            for cfg in conditions():
                policy = Policy(backend)
                policy.configure(cfg)
                for i in range(6):
                    result = policy.predict(obs, prompt, suite, seed+i)
                    route = result['route']
                    assert route['D'] == 1 and not route['full_ae_executed'] and not route['full_action_executed']
                    assert route['residual_refresh'] == (i % 3 == 0)
                    assert result['timing']['nodes']['actual_NFE'] == 1
                    history = policy.history
                    ctx, _ = backend.prepare_residual(obs, prompt, MASK,
                        history=history[-1]['residuals'],
                        older=history[-2]['residuals'] if len(history)==2 else None,
                        ratio=route['extrapolation_ratio'], reference=True)
                    if route['initial_state'] == 'gaussian':
                        a, r, _ = sample_gaussian(backend, ctx, seed+i, 1)
                    else:
                        coordinates = DecoderCoordinates(backend.adapter.backend.output_transform, ctx['transformed_state'])
                        prior = policy.kalman.last_forecast
                        encoded = coordinates.encode_commands(prior, np.zeros((50,32), np.float32))
                        assert np.allclose(coordinates.decode(encoded), prior, atol=2e-5, rtol=2e-5)
                        initial = mix(encoded, seed+i, .75, 1.)
                        a, r, _ = sample_initial(backend, ctx, initial, 1, .75)
                        assert policy.kalman.filter.count == policy.kalman.rotation_filter.count == result['step']
                    passed = bool(np.allclose(r.float().cpu().numpy(),result['raw'],atol=1e-5,rtol=1e-5)
                        and np.allclose(a,result['actions'],atol=1e-5,rtol=1e-5))
                    checks.append(dict(name=f'{suite}_{cfg["initial"]}_reference_{i}', passed=passed,
                        raw_max_abs=float(np.max(np.abs(r.float().cpu().numpy()-result['raw'])))))
                    if i == 0:
                        firsts.append(result['raw'])
                    arrays[f'{suite}_{cfg["initial"]}_round{i}'] = result['raw']
                    policy.acknowledge(result['actions'][:5])
                assert policy.counts['full'] == policy.counts['verify'] == 0 and policy.counts['draft'] == 6
                assert policy.counts['kalman'] == (5 if cfg['initial']=='kalman' else 0)
            checks.append(dict(name=suite+'_identical_gaussian_bootstrap',
                passed=bool(np.allclose(firsts[0],firsts[1],atol=1e-5,rtol=1e-5))))
    finally:
        backend.sample, backend.verify = original_sample, original_verify
    result = dict(passed=all(c['passed'] for c in checks), checks=checks,
        full_ae_calls=0, verifier_calls=0, actual_NFE=1,
        native_encoder_graph=False, native_ae_graph=False)
    np.savez(output/'gate_actions.npz', **arrays)
    atomic(output/'preflight.json', result)
    if not result['passed']:
        raise RuntimeError('One-step residual-only mechanism check failed')
    return result

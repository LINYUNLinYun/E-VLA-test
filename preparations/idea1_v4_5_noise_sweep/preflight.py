from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3.history import DecoderCoordinates
from preparations.idea1_v3_4_v12_compensation.server import selection
from preparations.kv_reuse_probe.probe import load_observation
from preparations.idea1_v4_1_warmstart.sampling import sample_gaussian, sample_initial
from preparations.idea1_v4_1_warmstart.schedule import mix
from .common import MASK, OLD, allowed_conditions
from .policy import Policy


def run(backend, output):
    output = Path(output)
    arrays, checks, historical_checks = {}, [], []
    configs=allowed_conditions()
    configs=[configs[0], configs[-1]]+configs[1:-1]
    historical=np.load(OLD/'cross_service_reference.npz')
    original_sample, original_verify = backend.sample, backend.verify
    def forbidden(*args, **kwargs):
        raise AssertionError('Full/legacy action expert and verifier must never be called')
    backend.sample = backend.verify = forbidden
    try:
        for item in selection()[::10]:
            obs, prompt = load_observation(item['current'])
            suite, seed = item['case']['suite'], int(item['current']['seed'])
            firsts = []
            for cfg in configs:
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
                        initial = mix(encoded, seed+i, cfg['start_time'], cfg['noise_scale'])
                        a, r, _ = sample_initial(backend, ctx, initial, 1, cfg['start_time'])
                        assert policy.kalman.filter.count == policy.kalman.rotation_filter.count == result['step']
                    passed = bool(np.allclose(r.float().cpu().numpy(),result['raw'],atol=1e-5,rtol=1e-5)
                        and np.allclose(a,result['actions'],atol=1e-5,rtol=1e-5))
                    checks.append(dict(name=f'{suite}_{cfg["id"]}_reference_{i}', passed=passed,
                        raw_max_abs=float(np.max(np.abs(r.float().cpu().numpy()-result['raw'])))))
                    if i == 0:
                        firsts.append(result['raw'])
                    arrays[f'{suite}_{cfg["id"]}_round{i}'] = result['raw']
                    expected_start=1. if route['initial_state']=='gaussian' else cfg['start_time']
                    assert result['timing']['nodes']['nodes']==[expected_start,0.]
                    assert route['actual_noise_std']==expected_start
                    if route['initial_state']=='kalman6':
                        eps=np.random.default_rng(seed+i).standard_normal((50,32),dtype=np.float32)
                        expected=((1-cfg['start_time'])*encoded+cfg['noise_scale']*cfg['start_time']*eps).astype(np.float32)
                        assert np.array_equal(result['arrays']['kalman_initial_prefix'], expected[:5])
                        assert np.array_equal(result['arrays']['kalman_encoded_prefix'][:,7:], np.zeros((5,25),np.float32))
                    if cfg['initial']=='gaussian' or cfg['start_time']==.75:
                        old_key=f'{suite}_{cfg["initial"]}_round{i}'
                        old=historical[old_key]
                        historical_checks.append(dict(name=old_key,passed=bool(np.allclose(old,result['raw'],atol=1e-5,rtol=1e-5)),max_abs=float(np.abs(old-result['raw']).max())))
                    policy.acknowledge(result['actions'][:5])
                assert policy.counts['full'] == policy.counts['verify'] == 0 and policy.counts['draft'] == 6
                assert policy.counts['kalman'] == (5 if cfg['initial']=='kalman' else 0)
            checks.append(dict(name=suite+'_identical_gaussian_bootstrap',
                passed=all(np.allclose(firsts[0],value,atol=1e-5,rtol=1e-5) for value in firsts[1:])))
    finally:
        backend.sample, backend.verify = original_sample, original_verify
        historical.close()
    result = dict(passed=all(c['passed'] for c in checks), checks=checks,
        full_ae_calls=0, verifier_calls=0, actual_NFE=1,
        native_encoder_graph=False, native_ae_graph=False,
        historical_gate_passed=all(c['passed'] for c in historical_checks), historical_checks=historical_checks)
    np.savez(output/'gate_actions.npz', **arrays)
    atomic(output/'preflight.json', result)
    if not result['passed']:
        raise RuntimeError('One-step residual-only mechanism check failed')
    return result

from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_selection_async.runtime import samples
from preparations.kv_reuse_probe.probe import load_observation
from .sampling import sample_initial, sample_gaussian
from .schedule import mix
from .common import config


def run(backend, policy, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sample = samples()[0]
    obs, prompt = load_observation(sample['current'])
    seed = int(sample['current']['seed'])
    context, _ = backend.prepare(obs,prompt,())
    checks, arrays = [], {}
    for steps in (1,2):
        prior = np.full((50,32),.25,np.float32)
        initial = mix(prior,seed,1.,1.)
        native, raw, _ = backend.sample(context,seed,steps)
        got, warm, costs = sample_initial(backend,context,initial,steps,1.,time_encoding='legacy')
        delta = float(np.abs(raw.float().cpu().numpy()-warm.float().cpu().numpy()).max())
        checks.append(dict(kind='noise_endpoint_recovers_native',steps=steps,raw_max_abs=delta,
            passed=bool(delta<=1e-5 and np.allclose(native,got,atol=1e-5,rtol=1e-5))))
        corrected, fixed, costs = sample_gaussian(backend,context,seed,steps)
        supplied, same, _ = sample_initial(backend,context,initial,steps,1.)
        checks.append(dict(kind='corrected_noise_endpoint',steps=steps,
            legacy_delta=float(np.abs(raw.float().cpu().numpy()-fixed.float().cpu().numpy()).max()),
            passed=bool(np.isfinite(corrected).all() and costs['time_embedding']=='openpi_linspace' and
                np.array_equal(same.float().cpu().numpy(),fixed.float().cpu().numpy()))))
        _, warm, costs = sample_initial(backend,context,mix(prior,seed,.2,1.),steps,.2)
        checks.append(dict(kind='partial_time_nodes',steps=steps,nodes=costs['nodes'],
            passed=bool(np.isfinite(warm.float().cpu().numpy()).all() and
                np.isclose(sum(costs['nodes']['dt']),-.2) and costs['nodes']['actual_NFE']==steps)))
        arrays[f'legacy_native_d{steps}'] = raw.float().cpu().numpy()
        arrays[f'time_fixed_d{steps}'] = fixed.float().cpu().numpy()
    # One real policy smoke reaches both Fast requests with the new schedule.
    cfg = config(dict(id='smoke',fast_steps=1,start_time=.2,noise_scale=1.))
    policy.configure(cfg)
    routes = []
    for i in range(3):
        result = policy.predict(obs,prompt,sample['suite'],seed+i)
        routes.append(result['route'])
        n = result['execute_length']
        policy.acknowledge(result['actions'][:n],np.tile([0.,0.,0.,1.],(n,1)))
    policy.finish()
    attempted = [r for r in routes if r['draft_attempted']]
    checks.append(dict(kind='warmstart_policy_smoke',passed=bool(attempted and all(
        r['initial_time_nodes']==[.2] and r['actual_fast_nfe']==1 and
        r['kalman_observations']>0 and not r['native_ae_graph'] for r in attempted)),routes=routes))
    policy.clear()
    np.savez(output/'arrays.npz',**arrays)
    result = dict(passed=all(c['passed'] for c in checks),checks=checks)
    atomic(output/'complete.json',result)
    if not result['passed']:
        raise RuntimeError('New warm-start mechanism check failed')
    return result


if __name__ == '__main__':
    import argparse
    from preparations.idea1_v2_1.runtime_client import rpc
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,required=True)
    args = parser.parse_args()
    print(rpc(args.port,dict(op='preflight')))

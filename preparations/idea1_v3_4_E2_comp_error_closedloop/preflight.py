import json
import numpy as np
from .common import ROOT,COST,CONDITIONS,sha
from preparations.idea1_v3.io import atomic

def preflight(b,pol,out):
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_4_E2_cycle3.selection import prepare_full
    proof=json.loads((COST/'summary.json').read_text());manifest=json.loads((COST/'manifest.json').read_text())
    assert proof['passed'] and proof['actual_gpu_measurement'] and len(proof['checks'])==8
    assert all(c['neutral_full_bitwise_equal'] and c['graph_eager_gradient_max_abs']==0 for c in proof['checks'])
    name='preparations/idea1_v3_4_E2_comp_error_cost/model.py';assert sha(ROOT/name)==manifest['source_hashes'][name]
    pairs=json.loads((ROOT/'results/idea1_v3_4_E2_E4_single/selection.json').read_text())
    pair=next(p for p in pairs if p['e2'] and p['case']['suite']=='libero_spatial' and p['case']['task']==0)
    obs,prompt=load_observation(pair['current']);seed=pair['current']['seed'];full,_=prepare_full(b,obs,prompt,trace=False);checks=[]
    for cfg in CONDITIONS:
        pol.configure(cfg);routes=[]
        for i in range(10):
            r=pol.predict(obs,prompt,'libero_spatial',seed+i);route=r['route'];routes.append(route)
            full_round=i%3==0;expected_D=10 if full_round else 2
            expected_actions,expected_raw,_=b.sample(full,seed+i,expected_D)
            assert np.allclose(r['raw'],expected_raw.float().cpu().numpy(),atol=1e-5,rtol=1e-5),(cfg['id'],i,'raw')
            assert np.allclose(r['actions'],expected_actions,atol=1e-5,rtol=1e-5),(cfg['id'],i,'actions')
            assert route['D']==expected_D and route['scoring_mode']==('warm_original' if i<6 else 'comp_error')
            if full_round:
                assert route['skip']==[] and route['patch_layers']==[]
                if i>=6:
                    assert route['score_source_full_steps']==[(i-6)*5,(i-3)*5,i*5]
                    assert route['score_error_ratio']==1.
                    assert np.array_equal(r['arrays']['prediction_error_rms'],np.zeros(17))
                    assert np.array_equal(r['arrays']['gate_gradient'],np.zeros(17))
            else:
                assert r['timing']['selector_total_ms']==0 and r['timing']['error_build_ms']==0
                assert route['skip']==routes[i-i%3]['selected_skip']
            pol.acknowledge(r['actions'][:5],None)
        assert pol.counts['warm_selector']==2 and pol.counts['error_selector']==2
        checks.append(dict(condition=cfg['id'],passed=True,constant_observation_native_full_oracle=True,warm_full_count=2,full_error_steps=[30,45],zero_forecast_error_zero_gradient=True,fast_no_probe=True));print(json.dumps(checks[-1]),flush=True)
    pol.configure(CONDITIONS[0])
    atomic(out/'gate.json',dict(passed=True,checks=checks,reused_nonzero_error_gradient_checks=8,reused_gpu_proof_sha256=sha(COST/'summary.json'),error_model_sha256=sha(ROOT/name),causal_history=True,native_full_execution=True))

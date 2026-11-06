import json
import numpy as np
from .common import ROOT, OUT, CONDITIONS, sha
from preparations.idea1_v3.io import atomic

def preflight(b,pol,out):
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_4_E2_cycle3.selection import prepare_full
    proof=json.loads((OUT/'cost/summary.json').read_text())
    assert proof['passed'] and proof['samples']==8
    pairs=json.loads((ROOT/'results/idea1_v3_4_E2_E4_single/selection.json').read_text())
    pair=next(p for p in pairs if p['e2'] and p['case']['suite']=='libero_spatial' and p['case']['task']==0)
    obs,prompt=load_observation(pair['current']);seed=pair['current']['seed']
    full,_=prepare_full(b,obs,prompt,trace=False);checks=[]
    for cfg in CONDITIONS:
        pol.configure(cfg);routes=[]
        for i in range(10):
            r=pol.predict(obs,prompt,'libero_spatial',seed+i);z=r['route'];routes.append(z)
            is_full=i%3==0;D=10 if is_full else 2
            expected_actions,expected_raw,_=b.sample(full,seed+i,D)
            assert np.allclose(r['raw'],expected_raw.float().cpu().numpy(),atol=1e-5,rtol=1e-5),(cfg['id'],i,'raw')
            assert np.allclose(r['actions'],expected_actions,atol=1e-5,rtol=1e-5),(cfg['id'],i,'actions')
            assert z['D']==D and z['scoring_mode']==('warm_original' if i<6 else 'comp_error')
            assert z['steady_vjp_count']==((1 if i<6 else cfg['projections']) if is_full else 0)
            assert z['selector_forward_count']==int(is_full)
            if is_full:
                assert z['skip']==[] and z['patch_layers']==[]
                if i>=6:
                    assert z['score_source_full_steps']==[(i-6)*5,(i-3)*5,i*5]
                    assert z['score_error_ratio']==1.
                    assert np.array_equal(r['arrays']['prediction_error_rms'],np.zeros(17))
                    assert np.array_equal(r['arrays']['gate_gradient'],np.zeros((cfg['projections'],17)))
            else:
                assert r['timing']['selector_total_ms']==r['timing']['error_build_ms']==0
                assert z['skip']==routes[i-i%3]['selected_skip']
            pol.acknowledge(r['actions'][:5],None)
        assert pol.counts['warm_selector']==2 and pol.counts['error_selector']==2
        checks.append(dict(condition=cfg['id'],passed=True,native_full_oracle=True,shared_forward_count=1,hot_vjp_count=cfg['projections'],zero_error_zero_gradient=True,fast_no_probe=True))
        print(json.dumps(checks[-1]),flush=True)
    pol.configure(CONDITIONS[0])
    atomic(out/'gate.json',dict(passed=True,checks=checks,reused_gpu_proof_sha256=sha(OUT/'cost/summary.json'),causal_history=True))

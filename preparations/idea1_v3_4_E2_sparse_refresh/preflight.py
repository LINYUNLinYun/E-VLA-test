import json
import numpy as np
from .common import ROOT,COST,CONDITIONS,sha
from preparations.idea1_v3.io import atomic

def preflight(b,pol,out):
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_4_E2_cycle3.selection import prepare_full
    proof=json.loads((COST/'summary.json').read_text());manifest=json.loads((COST/'manifest.json').read_text())
    assert proof['passed'] and len(proof['checks'])==8
    name='preparations/idea1_v3_4_E2_comp_error_cost/model.py'
    assert sha(ROOT/name)==manifest['source_hashes'][name]
    pairs=json.loads((ROOT/'results/idea1_v3_4_E2_E4_single/selection.json').read_text())
    pair=next(p for p in pairs if p['e2'] and p['case']['suite']=='libero_spatial' and p['case']['task']==0)
    obs,prompt=load_observation(pair['current']);seed=pair['current']['seed']
    full,_=prepare_full(b,obs,prompt,trace=False);checks=[]
    configs=[dict(CONDITIONS[0],id='preflight_stride1',refresh_stride=1),*CONDITIONS]
    for cfg in configs:
        pol.configure(cfg);mask=None;mask_step=None;source=[];cost=[];routes=[]
        for i in range(36):
            r=pol.predict(obs,prompt,'libero_spatial',seed+i);z=r['route'];routes.append(z)
            cycle=i//3;is_full=i%3==0;refresh=is_full and cycle%cfg['refresh_stride']==0;D=10 if is_full else 2
            expected_actions,expected_raw,_=b.sample(full,seed+i,D)
            assert np.allclose(r['raw'],expected_raw.float().cpu().numpy(),atol=1e-5,rtol=1e-5),(cfg['id'],i,'raw')
            assert np.allclose(r['actions'],expected_actions,atol=1e-5,rtol=1e-5),(cfg['id'],i,'actions')
            assert z['D']==D and z['steady_vjp_count']==z['selector_forward_count']==int(refresh)
            assert z['mask_refresh']==refresh and z['full_refresh']==is_full
            if refresh:
                mask=z['selected_skip'];mask_step=i*5
                if cycle>=2:
                    source=[(cycle-2)*15,(cycle-1)*15,cycle*15]
                    assert np.array_equal(r['arrays']['prediction_error_rms'],np.zeros(17))
                    assert np.array_equal(r['arrays']['gate_gradient'],np.zeros(17))
            else:assert r['timing']['selector_total_ms']==r['timing']['error_build_ms']==0
            assert z['selected_skip']==mask and z['mask_age_env_steps']==i*5-mask_step
            assert z['history_full_steps']==([0] if cycle==0 else [(cycle-1)*15,cycle*15])
            assert z['score_source_full_steps']==source
            if not is_full:
                assert z['skip']==mask and z['full_age_env_steps']==i%3*5
                assert z['extrapolation_ratio']==(0. if cycle==0 else i%3/3)
            if i>=18:cost.append(dict(replan=i,path=z['path'],refresh=refresh,policy_ms=r['timing']['policy_wall_ms'],selector_ms=r['timing']['selector_total_ms']))
            pol.acknowledge(r['actions'][:5],None)
        assert sum(z['steady_vjp_count'] for z in routes)==len(range(0,12,cfg['refresh_stride']))
        check=dict(condition=cfg['id'],stride=cfg['refresh_stride'],passed=True,full_history_refreshes=12,vjps=sum(z['steady_vjp_count'] for z in routes),warm_vjps=pol.counts['warm_selector'],measured_requests=18,measured_vjps=sum(r['refresh'] for r in cost),mean_request_ms=float(np.mean([r['policy_ms'] for r in cost])),mean_selector_ms=float(np.mean([r['selector_ms'] for r in cost])),cost=cost)
        checks.append(check);print(json.dumps({k:v for k,v in check.items() if k!='cost'}),flush=True)
    pol.configure(CONDITIONS[0])
    atomic(out/'gate.json',dict(passed=True,checks=checks,reused_nonzero_error_gradient_checks=8,reused_gpu_proof_sha256=sha(COST/'summary.json'),causal_history=True,scope='Constant real Spatial observation; 18 warm +18 measured requests per stride, cold excluded here and included in closed-loop wall'))

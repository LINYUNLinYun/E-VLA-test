import json
import numpy as np
from .common import ROOT,BASE,CONDITIONS
from .policy import Policy
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_E2_adjacency_200.preflight import clear,check_optimum,same_full
from preparations.idea1_v3_4_E2_selection_200.preflight import snapshot,compare

def preflight(b,out,force_single=False):
    import torch
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_4_E2_selection_200.policy import Policy as Old
    from preparations.idea1_v3_4_E2_cycle3_extrapolate.backend import prepare_patch
    bank=json.loads((ROOT/'results/idea1_v3_4_E2_comp_error_cost/manifest.json').read_text())['samples'][0]
    items=[load_observation(bank[key]) for key in ('older','last','current')]
    assert len({prompt for obs,prompt in items})==1
    compatibility=[];reference={}
    for budget in (14,12):
        cfg=dict(next(c for c in CONDITIONS if c['budget']==budget),adjacency_rule='max2')
        old=Old(b);old.configure(cfg);expected=[]
        for i in range(9):
            obs,prompt=items[i//3];r=old.predict(obs,prompt,'libero_spatial',22000+i);expected.append(snapshot(r));old.acknowledge(r['actions'][:5],None)
        clear([old]);del old
        pol=Policy(b);pol.configure(cfg)
        for i in range(9):
            obs,prompt=items[i//3];r=pol.predict(obs,prompt,'libero_spatial',22000+i)
            assert compare(expected[i],snapshot(r)),('max2 original extrapolation compatibility',budget,i)
            pol.acknowledge(r['actions'][:5],None)
        clear([pol]);del pol;reference[budget]=expected
        compatibility.append(dict(budget=budget,requests=9,passed=True))
    rules=[];long_oracles=[];obs,prompt=items[0]
    for cfg in CONDITIONS:
        pol=Policy(b);pol.configure(cfg)
        for i in range(9):
            r=pol.predict(obs,prompt,'libero_spatial',22000+i)
            if i%3==0:
                check_optimum(r,cfg)
                if i==0:assert same_full(reference[cfg['budget']][0],snapshot(r))
            else:
                ratio=0. if i<3 else (i%3)/3
                assert np.isclose(r['route']['extrapolation_ratio'],ratio)
                assert r['route']['steady_vjp_count']==0 and r['route']['captured_layers']==list(range(1,18))
            full,_=b.prepare(obs,prompt,());actions,raw,_=b.sample(full,22000+i,10 if i%3==0 else 2)
            assert np.allclose(r['raw'],raw.float().cpu().numpy(),atol=1e-5,rtol=1e-5) and np.allclose(r['actions'],actions,atol=1e-5,rtol=1e-5),('constant observation oracle',cfg['id'],i)
            pol.acknowledge(r['actions'][:5],None)
        if cfg['adjacency_rule']=='unrestricted':
            skip=list(range(2,2+cfg['skip_count']));last=pol.history[-1];older=pol.history[-2]
            patched,_=prepare_patch(b,obs,prompt,skip,last['residuals'],older['residuals'],1/3);aa,rr,_=b.sample(patched,24000,2)
            full,_=b.prepare(obs,prompt,());oracle,raw,_=b.sample(full,24000,2)
            assert np.allclose(rr.float().cpu().numpy(),raw.float().cpu().numpy(),atol=1e-5,rtol=1e-5) and np.allclose(aa,oracle,atol=1e-5,rtol=1e-5),('forced long-run residual oracle',skip)
            long_oracles.append(dict(skip=skip,passed=True))
        clear([pol]);del pol;rules.append(dict(condition=cfg['id'],requests=9,passed=True))
    atomic(out/'compatibility_gate.json',dict(passed=True,max2_original=compatibility,constant_observation_oracle=rules,forced_long_run_oracles=long_oracles,reused_numerical_gate=str(BASE/'audit.json')))
    if force_single:
        atomic(out/'isolation_gate.json',dict(passed=True,actors=1,reason='Explicit fallback after two-client resource failure'))
        return 1
    cfgs=[CONDITIONS[0],CONDITIONS[3]];expected=[]
    for actor,cfg in enumerate(cfgs):
        pol=Policy(b);pol.configure(cfg);records=[]
        for i in range(9):
            obs,prompt=items[(i//3+actor)%3];r=pol.predict(obs,prompt,'libero_spatial',26000+actor*100+i);records.append(snapshot(r));pol.acknowledge(r['actions'][:5],None)
        expected.append(records);clear([pol]);del pol
    policies=[Policy(b),Policy(b)];checks=[]
    try:
        for pol,cfg in zip(policies,cfgs):pol.configure(cfg)
        for i in range(9):
            for actor,pol in enumerate(policies):
                obs,prompt=items[(i//3+actor)%3];r=pol.predict(obs,prompt,'libero_spatial',26000+actor*100+i)
                checks.append(dict(actor=actor,replan=i,passed=compare(expected[actor][i],snapshot(r))));pol.acknowledge(r['actions'][:5],None)
        passed=all(c['passed'] for c in checks)
        atomic(out/'isolation_gate.json',dict(passed=passed,actors=2 if passed else 1,checks=checks,same_task_different_real_observations=True,max_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        return 2 if passed else 1
    except torch.cuda.OutOfMemoryError:
        atomic(out/'single_actor_required.json',dict(reason='Two new-rule residual/gradient clients OOM; bounded restart using one actor'))
        raise
    finally:clear(policies)

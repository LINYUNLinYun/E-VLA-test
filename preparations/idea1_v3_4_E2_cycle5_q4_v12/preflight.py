import json
import numpy as np
from .common import ROOT,CONDITIONS,choose,legal
from .model import Selector,calculate,projection_weights
from .policy import Policy
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_E2_cycle3.common import coefficients
from preparations.idea1_v3_4_E2_adjacency_200.preflight import clear,check_optimum
from preparations.idea1_v3_4_E2_selection_200.preflight import snapshot,compare

def equal(a,b,gradient=False):
    return bool(np.allclose(a,b,atol=1e-4 if gradient else 1e-5,rtol=1e-3 if gradient else 1e-5))

def preflight(b,out,force_single=False):
    import torch
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_4_E2_cycle5_spaced.policy import Policy as Old
    from preparations.idea1_v3_4_E2_online_time_weights.selector import calculate as single_calculate
    samples=json.loads((ROOT/'results/idea1_v3_4_E2_comp_error_cost/manifest.json').read_text())['samples']
    items=[load_observation(samples[0][key]) for key in ('older','last','current')]
    compatibility=[]
    for cfg4 in CONDITIONS:
        cfg=dict(cfg4,projections=1);old=Old(b);old.configure(cfg);expected=[]
        for i in range(10):
            obs,prompt=items[i//5];r=old.predict(obs,prompt,'libero_spatial',22000+i);expected.append(snapshot(r));old.acknowledge(r['actions'][:5],None)
        clear([old]);del old
        pol=Policy(b);pol.configure(cfg)
        for i in range(10):
            obs,prompt=items[i//5];r=pol.predict(obs,prompt,'libero_spatial',22000+i);actual=snapshot(r)
            if i%5==0:actual['arrays']['gate_gradient']=actual['arrays']['gate_gradient'][0]
            assert compare(expected[i],actual),('q1 unchanged compatibility',cfg['id'],i)
            pol.acknowledge(r['actions'][:5],None)
        clear([pol]);del pol;compatibility.append(dict(condition=cfg['id'],requests=10,passed=True))
    gradient_checks=[]
    for sample in samples:
        obs,prompt=load_observation(sample['current']);seed=sample['current']['seed']
        c,_=b.prepare(obs,prompt,());b.sample(c,seed,10)
        selector=Selector(b);selector.q=4;skip,grads,score,t=selector.run(c,seed,coefficients(),6)
        raw=selector.raw.float().cpu().numpy().copy();eager_raw,eager_grads=calculate(selector.model,selector.item,selector.W,selector.g)
        assert equal(raw,eager_raw.float().cpu().numpy()) and equal(grads,eager_grads.float().cpu().numpy(),True)
        separate=[]
        for direction in range(4):
            independent,grad=single_calculate(selector.model,selector.item,selector.W[direction],selector.g)
            assert equal(raw,independent.float().cpu().numpy()) and equal(grads[direction],grad.float().cpu().numpy(),True)
            separate.append(grad.float().cpu().numpy().copy())
        independent_score=np.sqrt(np.mean(np.square(np.stack(separate)),axis=0))
        assert equal(score,independent_score,True) and skip==choose(independent_score,6,'spaced')
        gradient_checks.append(dict(suite=sample['suite'],task=sample['task'],passed=True,graph_eager_and_four_independent=True,shared_forward=1,vjp_count=4))
        selector.clear();del selector
    oracles=[];obs,prompt=items[0]
    for cfg in CONDITIONS:
        pol=Policy(b);pol.configure(cfg)
        for i in range(10):
            r=pol.predict(obs,prompt,'libero_spatial',23000+i);rt=r['route'];full=i%5==0
            assert rt['steady_vjp_count']==(4 if full else 0) and rt['selector_forward_count']==int(full)
            assert rt['scoring_mode']=='original_residual' and legal(rt['selected_skip'],6,'spaced')
            if full:
                assert r['arrays']['gate_gradient'].shape==(4,17);check_optimum(r,cfg)
            if not full and cfg['compensation']=='extrapolate':assert np.isclose(rt['extrapolation_ratio'],0. if i<5 else (i%5)/5)
            skip=() if full or cfg['compensation']=='extrapolate' else rt['selected_skip']
            c,_=b.prepare(obs,prompt,skip);actions,raw,_=b.sample(c,23000+i,10 if full else 2)
            assert equal(r['raw'],raw.float().cpu().numpy()) and equal(r['actions'],actions),('native cycle5 q4 oracle',cfg['id'],i)
            pol.acknowledge(r['actions'][:5],None)
        clear([pol]);del pol;oracles.append(dict(condition=cfg['id'],requests=10,passed=True))
    atomic(out/'compatibility_gate.json',dict(passed=True,q1_compatibility=compatibility,gradient_checks=gradient_checks,cycle5_oracles=oracles,first_full_q4=True))
    if force_single:
        atomic(out/'isolation_gate.json',dict(passed=True,actors=1,reason='Bounded fallback after two-client resource failure'));return 1
    expected=[]
    for actor,cfg in enumerate(CONDITIONS):
        pol=Policy(b);pol.configure(cfg);records=[]
        for i in range(15):
            obs,prompt=items[(i//5+actor)%3];r=pol.predict(obs,prompt,'libero_spatial',26000+actor*100+i);records.append(snapshot(r));pol.acknowledge(r['actions'][:5],None)
        expected.append(records);clear([pol]);del pol
    policies=[Policy(b),Policy(b)];checks=[]
    try:
        for pol,cfg in zip(policies,CONDITIONS):pol.configure(cfg)
        for i in range(15):
            for actor,pol in enumerate(policies):
                obs,prompt=items[(i//5+actor)%3];r=pol.predict(obs,prompt,'libero_spatial',26000+actor*100+i)
                checks.append(dict(actor=actor,replan=i,passed=compare(expected[actor][i],snapshot(r))));pol.acknowledge(r['actions'][:5],None)
        passed=all(c['passed'] for c in checks)
        atomic(out/'isolation_gate.json',dict(passed=passed,actors=2 if passed else 1,checks=checks,max_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        return 2 if passed else 1
    except torch.cuda.OutOfMemoryError:
        atomic(out/'single_actor_required.json',dict(reason='Two q4 Full-D10 clients OOM; bounded restart with one actor'));raise
    finally:clear(policies)

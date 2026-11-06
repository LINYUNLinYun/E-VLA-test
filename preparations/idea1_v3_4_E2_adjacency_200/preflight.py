import gc,json
import numpy as np
from .common import ROOT,BASE,CONDITIONS,choose,legal
from .policy import Policy
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_E2_selection_200.preflight import snapshot,compare

def clear(policies):
    for pol in policies:pol.selector.clear();pol.reset()
    gc.collect()

def optimum(score,k,rule):
    score=np.asarray(score,dtype=np.float64)
    if rule=='unrestricted':return float(np.sort(score)[:k].sum())
    assert rule=='spaced'
    dp=np.full((len(score)+1,k+1),np.inf);dp[:,0]=0.
    for i,value in enumerate(score,1):
        for j in range(1,k+1):
            previous=0. if i==1 and j==1 else np.inf if i==1 else dp[i-2,j-1]
            dp[i,j]=min(dp[i-1,j],previous+value)
    return float(dp[-1,k])

def check_optimum(r,cfg):
    score=r['arrays']['layer_score'];skip=r['route']['selected_skip']
    assert legal(skip,cfg['skip_count'],cfg['adjacency_rule'])
    actual=float(np.asarray(score,dtype=np.float64)[np.array(skip)-1].sum())
    target=optimum(score,cfg['skip_count'],cfg['adjacency_rule'])
    assert np.isclose(actual,target,atol=1e-6,rtol=1e-6),(actual,target,cfg)

def same_full(a,b):
    return bool(np.allclose(a['raw'],b['raw'],atol=1e-5,rtol=1e-5) and np.allclose(a['actions'],b['actions'],atol=1e-5,rtol=1e-5) and all(np.allclose(a['arrays'][key],b['arrays'][key],atol=1e-4,rtol=1e-3) for key in ('gate_gradient','layer_score')))

def preflight(b,out,force_single=False):
    import torch
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_4_E2_selection_200.policy import Policy as Old
    bank=json.loads((ROOT/'results/idea1_v3_4_E2_comp_error_cost/manifest.json').read_text())['samples'][0]
    items=[load_observation(bank[key]) for key in ('older','last','current')]
    assert len({p for o,p in items})==1
    compatibility=[];reference={}
    for budget in (14,12):
        cfg=dict(next(c for c in CONDITIONS if c['budget']==budget),adjacency_rule='max2')
        old=Old(b);old.configure(cfg);expected=[]
        for i in range(6):
            obs,prompt=items[i//3];r=old.predict(obs,prompt,'libero_spatial',22000+i);expected.append(snapshot(r));old.acknowledge(r['actions'][:5],None)
        clear([old]);del old
        pol=Policy(b);pol.configure(cfg)
        for i in range(6):
            obs,prompt=items[i//3];r=pol.predict(obs,prompt,'libero_spatial',22000+i)
            assert compare(expected[i],snapshot(r)),('max2 compatibility',budget,i)
            pol.acknowledge(r['actions'][:5],None)
        clear([pol]);del pol;reference[budget]=expected
        compatibility.append(dict(budget=budget,requests=6,passed=True))
    rules=[]
    for cfg in CONDITIONS:
        pol=Policy(b);pol.configure(cfg)
        for i in range(6):
            obs,prompt=items[i//3];r=pol.predict(obs,prompt,'libero_spatial',22000+i)
            if i%3==0:
                assert same_full(reference[cfg['budget']][i],snapshot(r)),('Full action/gradient unchanged',cfg['id'],i)
                check_optimum(r,cfg)
            else:
                c,_=b.prepare(obs,prompt,r['route']['selected_skip']);actions,raw,_=b.sample(c,22000+i,2)
                assert np.allclose(r['raw'],raw.float().cpu().numpy(),atol=1e-5,rtol=1e-5) and np.allclose(r['actions'],actions,atol=1e-5,rtol=1e-5),('native Fast oracle',cfg['id'],i)
                assert r['route']['steady_vjp_count']==0
            pol.acknowledge(r['actions'][:5],None)
        clear([pol]);del pol;rules.append(dict(condition=cfg['id'],requests=6,passed=True))
    atomic(out/'compatibility_gate.json',dict(passed=True,max2_original=compatibility,new_rules=rules,independent_optimizer='unrestricted sorted-k, spaced dynamic programming',reused_numerical_gate=str(BASE/'audit.json')))
    if force_single:
        atomic(out/'isolation_gate.json',dict(passed=True,actors=1,reason='Explicit fallback after two-client resource failure'))
        return 1
    cfgs=[CONDITIONS[0],CONDITIONS[3]];expected=[]
    for actor,cfg in enumerate(cfgs):
        pol=Policy(b);pol.configure(cfg);rows=[]
        for i in range(6):
            obs,prompt=items[(i//3+actor)%3];r=pol.predict(obs,prompt,'libero_spatial',26000+actor*100+i);rows.append(snapshot(r));pol.acknowledge(r['actions'][:5],None)
        expected.append(rows);clear([pol]);del pol
    policies=[Policy(b),Policy(b)];checks=[]
    try:
        for pol,cfg in zip(policies,cfgs):pol.configure(cfg)
        for i in range(6):
            for actor,pol in enumerate(policies):
                obs,prompt=items[(i//3+actor)%3];r=pol.predict(obs,prompt,'libero_spatial',26000+actor*100+i)
                checks.append(dict(actor=actor,replan=i,passed=compare(expected[actor][i],snapshot(r))));pol.acknowledge(r['actions'][:5],None)
        passed=all(c['passed'] for c in checks)
        atomic(out/'isolation_gate.json',dict(passed=passed,actors=2 if passed else 1,checks=checks,same_task_different_real_observations=True,max_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        return 2 if passed else 1
    except torch.cuda.OutOfMemoryError:
        atomic(out/'single_actor_required.json',dict(reason='Two new-rule clients OOM; bounded restart using one actor'))
        raise
    finally:clear(policies)

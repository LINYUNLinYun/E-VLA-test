"""Check the changed route, extrapolation horizon and policy isolation."""
import json
import numpy as np
from .common import ROOT,CONDITIONS,legal
from .policy import Policy
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_E2_adjacency_200.preflight import clear,check_optimum
from preparations.idea1_v3_4_E2_selection_200.preflight import snapshot,compare

def preflight(b,out,force_single=False):
    import torch
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_4_E2_adjacency_200.policy import Policy as OldNone
    from preparations.idea1_v3_4_E2_adjacency_extrapolate_200.policy import Policy as OldExtrap
    bank=json.loads((ROOT/'results/idea1_v3_4_E2_comp_error_cost/manifest.json').read_text())['samples'][0]
    items=[load_observation(bank[key]) for key in ('older','last','current')]
    assert len({prompt for obs,prompt in items})==1
    compatibility=[]
    for budget,comp,Old in ((14,'none',OldNone),(12,'extrapolate',OldExtrap)):
        cfg=dict(next(c for c in CONDITIONS if c['budget']==budget and c['method']=='jacobian' and c['compensation']==comp),period=3)
        old=Old(b);old.configure(cfg);expected=[]
        for i in range(9):
            obs,prompt=items[i//3];r=old.predict(obs,prompt,'libero_spatial',22000+i);expected.append(snapshot(r));old.acknowledge(r['actions'][:5],None)
        clear([old]);del old
        pol=Policy(b);pol.configure(cfg)
        for i in range(9):
            obs,prompt=items[i//3];r=pol.predict(obs,prompt,'libero_spatial',22000+i)
            assert compare(expected[i],snapshot(r)),('cycle3 spaced compatibility',budget,comp,i)
            pol.acknowledge(r['actions'][:5],None)
        clear([pol]);del pol;compatibility.append(dict(budget=budget,compensation=comp,requests=9,passed=True))
    # Every condition: two full cycles cover the first hold and all four ratios.
    oracles=[];obs,prompt=items[0]
    for cfg in CONDITIONS:
        pol=Policy(b);pol.configure(cfg)
        for i in range(10):
            r=pol.predict(obs,prompt,'libero_spatial',23000+i);rt=r['route'];full=i%5==0
            assert rt['path']==('full' if full else 'fast') and rt['D']==(10 if full else 2)
            assert legal(rt['selected_skip'],cfg['skip_count'],'spaced')
            assert rt['steady_vjp_count']==int(full and cfg['method']=='jacobian')
            if full and cfg['method'] in ('jacobian','cosine'):check_optimum(r,cfg)
            if not full and cfg['compensation']=='extrapolate':
                assert np.isclose(rt['extrapolation_ratio'],0. if i<5 else (i%5)/5)
                assert rt['captured_layers']==list(range(1,18))
            skip=() if full or cfg['compensation']=='extrapolate' else rt['selected_skip']
            c,_=b.prepare(obs,prompt,skip);actions,raw,_=b.sample(c,23000+i,10 if full else 2)
            assert np.allclose(r['raw'],raw.float().cpu().numpy(),atol=1e-5,rtol=1e-5) and np.allclose(r['actions'],actions,atol=1e-5,rtol=1e-5),('cycle5 native oracle',cfg['id'],i)
            pol.acknowledge(r['actions'][:5],None)
        clear([pol]);del pol;oracles.append(dict(condition=cfg['id'],requests=10,passed=True))
    atomic(out/'compatibility_gate.json',dict(passed=True,cycle3_compatibility=compatibility,cycle5_oracles=oracles,
           optimizer='independent spaced dynamic programming',schedule=['Full-D10']+['Fast-D2']*4))
    if force_single:
        atomic(out/'isolation_gate.json',dict(passed=True,actors=1,reason='Bounded fallback after two-client allocation failure'))
        return 1
    cfgs=[next(c for c in CONDITIONS if c['budget']==bgt and c['method']=='jacobian' and c['compensation']==comp) for bgt,comp in ((14,'none'),(12,'extrapolate'))]
    expected=[]
    for actor,cfg in enumerate(cfgs):
        pol=Policy(b);pol.configure(cfg);records=[]
        for i in range(15):
            obs,prompt=items[(i//5+actor)%3];r=pol.predict(obs,prompt,'libero_spatial',26000+actor*100+i);records.append(snapshot(r));pol.acknowledge(r['actions'][:5],None)
        expected.append(records);clear([pol]);del pol
    policies=[Policy(b),Policy(b)];checks=[]
    try:
        for pol,cfg in zip(policies,cfgs):pol.configure(cfg)
        for i in range(15):
            for actor,pol in enumerate(policies):
                obs,prompt=items[(i//5+actor)%3];r=pol.predict(obs,prompt,'libero_spatial',26000+actor*100+i)
                checks.append(dict(actor=actor,replan=i,passed=compare(expected[actor][i],snapshot(r))));pol.acknowledge(r['actions'][:5],None)
        passed=all(c['passed'] for c in checks)
        atomic(out/'isolation_gate.json',dict(passed=passed,actors=2 if passed else 1,checks=checks,same_task_different_real_observations=True,max_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        return 2 if passed else 1
    except torch.cuda.OutOfMemoryError:
        atomic(out/'single_actor_required.json',dict(reason='Two isolated cycle5 clients OOM; bounded restart with one actor'))
        raise
    finally:clear(policies)

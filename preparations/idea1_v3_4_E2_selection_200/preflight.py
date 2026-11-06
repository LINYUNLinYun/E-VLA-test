import gc,json
import numpy as np
from .common import ROOT,OUT,JAC,CONDITIONS
from .policy import Policy
from preparations.idea1_v3.io import atomic

def clear(policies):
    for pol in policies:pol.selector.clear();pol.reset()
    gc.collect()

def snapshot(r):
    return dict(raw=r['raw'].copy(),actions=r['actions'].copy(),mask=r['route']['selected_skip'],arrays={k:v.copy() for k,v in r['arrays'].items()})

def compare(a,b):
    ok=np.allclose(a['raw'],b['raw'],atol=1e-5,rtol=1e-5) and np.allclose(a['actions'],b['actions'],atol=1e-5,rtol=1e-5) and a['mask']==b['mask']
    for key in ('gate_gradient','layer_score'):
        if key in a['arrays']:ok=ok and key in b['arrays'] and np.allclose(a['arrays'][key],b['arrays'][key],atol=1e-4,rtol=1e-3)
    return bool(ok)

def preflight(b,out,force_single=False):
    import torch
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.policy import Policy as OldNone
    from preparations.idea1_v3_4_E2_cycle3_extrapolate.policy import Policy as OldExtrap
    from preparations.idea1_v3_4_E2_cycle3.selection import prepare_full
    bank=json.loads((ROOT/'results/idea1_v3_4_E2_comp_error_cost/manifest.json').read_text())['samples'][0]
    items=[load_observation(bank[key]) for key in ('older','last','current')]
    assert len({p for o,p in items})==1
    old_gate=next(p for p in (JAC/'workers').glob('gpu*/gate.json') if json.loads(p.read_text())['passed'])
    compatibility=[]
    for cfg in [c for c in CONDITIONS if c['budget']==14 and c['phase']=='expand200']:
        expected=[];old=(OldNone if cfg['compensation']=='none' else OldExtrap)(b);old.configure(cfg)
        for i in range(6):
            obs,prompt=items[i//3];r=old.predict(obs,prompt,'libero_spatial',11000+i);expected.append(snapshot(r));old.acknowledge(r['actions'][:5],None)
        clear([old]);del old
        pol=Policy(b);pol.configure(cfg)
        for i in range(6):
            obs,prompt=items[i//3];r=pol.predict(obs,prompt,'libero_spatial',11000+i)
            assert compare(expected[i],snapshot(r)),('original compatibility',cfg['id'],i)
            pol.acknowledge(r['actions'][:5],None)
        clear([pol]);del pol
        compatibility.append(dict(condition=cfg['id'],requests=6,passed=True));print(json.dumps(compatibility[-1]),flush=True)
    oracle=[];obs,prompt=items[0];full,_=prepare_full(b,obs,prompt,trace=False)
    for cfg in [c for c in CONDITIONS if c['phase']=='random40']:
        pol=Policy(b);pol.configure(cfg)
        for i in range(6):
            r=pol.predict(obs,prompt,'libero_spatial',13000+i);D=10 if i%3==0 else 2;aa,rr,_=b.sample(full,13000+i,D)
            assert np.allclose(r['raw'],rr.float().cpu().numpy(),atol=1e-5,rtol=1e-5) and np.allclose(r['actions'],aa,atol=1e-5,rtol=1e-5),(cfg['id'],i,'random residual oracle')
            if i%3:assert r['route']['extrapolation_ratio']==(0. if i<3 else (i-3)/3)
            assert r['route']['steady_vjp_count']==0
            pol.acknowledge(r['actions'][:5],None)
        clear([pol]);del pol
        oracle.append(dict(condition=cfg['id'],requests=6,passed=True));print(json.dumps(oracle[-1]),flush=True)
    atomic(out/'compatibility_gate.json',dict(passed=True,checks=compatibility,random_constant_observation_oracle=oracle,reused_full10_proof=str(old_gate)))
    if force_single:return 1
    cfgs=[next(c for c in CONDITIONS if c['budget']==budget and c['method']=='jacobian' and c['compensation']==comp) for budget,comp in [(14,'none'),(12,'extrapolate')]]
    expected=[]
    for actor,cfg in enumerate(cfgs):
        pol=Policy(b);pol.configure(cfg);rows=[]
        for i in range(9):
            obs,prompt=items[(i//3+actor)%3];r=pol.predict(obs,prompt,'libero_spatial',15000+actor*100+i);rows.append(snapshot(r));pol.acknowledge(r['actions'][:5],None)
        expected.append(rows);clear([pol]);del pol
    policies=[Policy(b),Policy(b)];checks=[]
    try:
        for p,cfg in zip(policies,cfgs):p.configure(cfg)
        for i in range(9):
            for actor,pol in enumerate(policies):
                obs,prompt=items[(i//3+actor)%3];r=pol.predict(obs,prompt,'libero_spatial',15000+actor*100+i)
                checks.append(dict(actor=actor,replan=i,passed=compare(expected[actor][i],snapshot(r))))
                pol.acknowledge(r['actions'][:5],None)
        passed=all(r['passed'] for r in checks)
        atomic(out/'isolation_gate.json',dict(passed=passed,checks=checks,same_task_different_real_observations=True,max_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        print(json.dumps(dict(isolation_passed=passed,environment_workers=2 if passed else 1)),flush=True)
        return 2 if passed else 1
    except torch.cuda.OutOfMemoryError:
        atomic(out/'single_actor_required.json',dict(reason='Two-selector CUDA graph allocation OOM; restart service with one actor'))
        raise
    finally:clear(policies)

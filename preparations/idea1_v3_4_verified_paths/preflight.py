import json
import numpy as np
from .common import ROOT,OUT
from .policy import Policy
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_E2_adjacency_200.preflight import clear,check_optimum

def snapshot(r):
    return dict(raw=r['raw'].copy(),actions=r['actions'].copy(),execute_length=r['execute_length'],counts={k:r['counts'][k] for k in ('full','draft','verify')},
                accepted_length=r['route'].get('accepted_length'),arrays={k:v.copy() for k,v in r['arrays'].items()},path=r['route']['path'])

def compare(a,b):
    ok=np.allclose(a['raw'],b['raw'],atol=1e-5,rtol=1e-5) and np.allclose(a['actions'],b['actions'],atol=1e-5,rtol=1e-5)
    ok=ok and all(a[k]==b[k] for k in ('execute_length','counts','accepted_length'))
    for k in ('gate_gradient','layer_score'):
        if k in a['arrays']:ok=ok and k in b['arrays'] and np.allclose(a['arrays'][k],b['arrays'][k],atol=1e-4,rtol=1e-3)
    return bool(ok)

def preflight(b,out,force_single=False):
    import torch
    from preparations.kv_reuse_probe.probe import load_observation
    from preparations.idea1_v3_3.policy import Policy as Old
    m=json.loads((OUT/'manifest.json').read_text());cfgs=[c for c in m['conditions'] if c['mode']=='verified']
    bank=json.loads((ROOT/'results/idea1_v3_4_E2_comp_error_cost/manifest.json').read_text())['samples']
    compatibility=[]
    for row in bank[::2]:
        items=[load_observation(row[k]) for k in ('older','last','current')]
        cfg=dict(cfgs[0],budget=14,period=3,selection='fixed',projections=0,skip=[3,4,9,11],compensation='none',experiment='E1',draft_steps=2)
        old=Old(b);old.configure(cfg);expected=[]
        for i in range(9):
            obs,prompt=items[i//3];r=old.predict(obs,prompt,row['suite'],22000+i);expected.append(snapshot(r));n=r['execute_length'];old.acknowledge(r['actions'][:n],np.zeros((n,4)))
        del old
        pol=Policy(b);pol.configure(cfg)
        for i in range(9):
            obs,prompt=items[i//3];r=pol.predict(obs,prompt,row['suite'],22000+i)
            assert compare(expected[i],snapshot(r)),('original strict verifier compatibility',row['suite'],i)
            n=r['execute_length'];pol.acknowledge(r['actions'][:n],np.zeros((n,4)))
        clear([pol]);del pol;compatibility.append(dict(suite=row['suite'],requests=9,passed=True))
    items=[load_observation(bank[0][k]) for k in ('older','last','current')];routes=[]
    for cfg in cfgs:
        pol=Policy(b);pol.configure(cfg)
        for i in range(8):
            obs,prompt=items[(i//3)%3];r=pol.predict(obs,prompt,'libero_spatial',23000+i);rt=r['route']
            assert r['counts']['kalman']==0 and not hasattr(pol,'kalman') and 1<=r['execute_length']<=5
            assert rt['verify_performed']==rt['draft_attempted']
            if rt['path']=='full':
                c,_=b.prepare(obs,prompt,());actions,raw,_=b.sample(c,23000+i,10)
                assert np.allclose(r['raw'],raw.float().cpu().numpy(),atol=1e-5,rtol=1e-5) and np.allclose(r['actions'],actions,atol=1e-5,rtol=1e-5)
                if cfg['selection']=='jacobian':check_optimum(r,cfg)
            n=r['execute_length'];pol.acknowledge(r['actions'][:n],np.zeros((n,4)))
        clear([pol]);del pol;routes.append(dict(condition=cfg['id'],requests=8,passed=True))
    atomic(out/'compatibility_gate.json',dict(passed=True,strict_verifier_compatibility=compatibility,verified_paths=routes,kalman_instances=0))
    if force_single:
        atomic(out/'isolation_gate.json',dict(passed=True,actors=1,reason='Bounded one-client resource fallback'));return 1
    pair=[cfgs[1],cfgs[2]];expected=[]
    for actor,cfg in enumerate(pair):
        pol=Policy(b);pol.configure(cfg);records=[]
        for i in range(10):
            obs,prompt=items[(i//3+actor)%3];r=pol.predict(obs,prompt,'libero_spatial',26000+actor*100+i);records.append(snapshot(r));n=r['execute_length'];pol.acknowledge(r['actions'][:n],np.zeros((n,4)))
        expected.append(records);clear([pol]);del pol
    policies=[Policy(b),Policy(b)];checks=[]
    try:
        for pol,cfg in zip(policies,pair):pol.configure(cfg)
        for i in range(10):
            for actor,pol in enumerate(policies):
                obs,prompt=items[(i//3+actor)%3];r=pol.predict(obs,prompt,'libero_spatial',26000+actor*100+i)
                checks.append(dict(actor=actor,replan=i,passed=compare(expected[actor][i],snapshot(r))));n=r['execute_length'];pol.acknowledge(r['actions'][:n],np.zeros((n,4)))
        passed=all(r['passed'] for r in checks);atomic(out/'isolation_gate.json',dict(passed=passed,actors=2 if passed else 1,checks=checks,max_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
        return 2 if passed else 1
    except torch.cuda.OutOfMemoryError:
        atomic(out/'single_actor_required.json',dict(reason='Two verified clients OOM; bounded one-client retry'));raise
    finally:clear(policies)

import json
import numpy as np
import torch
from .common import ROOT,OUT
from .policy import Policy
from .model import calculate
from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import borrow
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_verified_paths.policy import Policy as Original
from preparations.idea1_v3_4_verified_paths.preflight import snapshot,compare
from preparations.idea1_v3_4_E2_adjacency_200.preflight import check_optimum

def release(pol):pol.selector.clear();pol.reset()
def ack(pol,r):pol.acknowledge(r['actions'][:r['execute_length']],np.zeros((r['execute_length'],4)))
def preflight(b,out):
    from preparations.kv_reuse_probe.probe import load_observation
    manifest=json.loads((OUT/'manifest.json').read_text());cfgs=manifest['conditions']
    sample=json.loads((ROOT/'results/idea1_v3_4_E2_comp_error_cost/manifest.json').read_text())['samples'][0]
    items=[load_observation(sample[k]) for k in ('older','last','current')];compat=[]
    for cfg in cfgs:
        old=Original(b);old.configure(cfg);expected=[]
        for i in range(2):
            obs,prompt=items[0];r=old.predict(obs,prompt,sample['suite'],23000+i);expected.append(snapshot(r));ack(old,r)
        release(old);del old
        pol=Policy(b);pol.configure(cfg)
        for i in range(2):
            obs,prompt=items[0];r=pol.predict(obs,prompt,sample['suite'],23000+i)
            if i==1 and r['route']['path']=='full':
                actual=snapshot(r)
                assert all(actual[k]==expected[i][k] for k in ('execute_length','counts','accepted_length'))
                assert np.allclose(actual['raw'],expected[i]['raw'],atol=1e-5,rtol=1e-5) and np.allclose(actual['actions'],expected[i]['actions'],atol=1e-5,rtol=1e-5)
            else:assert compare(expected[i],snapshot(r)),('warm compatibility',cfg['id'],i)
            check_optimum(r,cfg) if r['route']['path']=='full' else None
            assert r['route']['scoring_mode']==('comp_error_hold' if i==1 and r['route']['path']=='full' else 'warm_original');ack(pol,r)
        release(pol);del pol;compat.append(dict(condition=cfg['id'],requests=2,passed=True))
    gradient_checks=[]
    for budget in (12,14):
        cfg=dict(next(c for c in cfgs if c['budget']==budget),mode='full',period=1)
        pol=Policy(b);pol.configure(cfg)
        for i in range(3):
            obs,prompt=items[i];r=pol.predict(obs,prompt,sample['suite'],24000+i);check_optimum(r,cfg)
            c,_=b.prepare(obs,prompt,());actions,raw,_=b.sample(c,24000+i,10)
            assert np.allclose(r['raw'],raw.float().cpu().numpy(),atol=1e-5,rtol=1e-5) and np.allclose(r['actions'],actions,atol=1e-5,rtol=1e-5)
            if i:
                e=pol.selector.error;g=torch.ones(17,device='cuda',requires_grad=True)
                eager_raw,eager_grad=calculate(e.model,e.item,e.W,g)
                assert np.allclose(e.raw.float().cpu().numpy(),eager_raw.float().cpu().numpy(),atol=1e-5,rtol=1e-5)
                with torch.no_grad():neutral=borrow(b,c).forward(e.item,gates=torch.ones(17,device='cuda'))
                assert np.allclose(neutral.float().cpu().numpy(),eager_raw.float().cpu().numpy(),atol=1e-5,rtol=1e-5)
                assert np.allclose(e.grad.float().cpu().numpy(),eager_grad.float().cpu().numpy(),atol=1e-4,rtol=1e-3)
                for j in range(4):
                    single_raw,single_grad=calculate(e.model,e.item,e.W[j:j+1],g)
                    assert np.allclose(eager_grad[j].float().cpu().numpy(),single_grad[0].float().cpu().numpy(),atol=1e-4,rtol=1e-3)
                gradient_checks.append(dict(budget=budget,full_index=i,mode=r['route']['scoring_mode'],passed=True))
            ack(pol,r)
        release(pol);del pol
    # Run one actual verified sequence past both history transitions.
    cfg=cfgs[1];pol=Policy(b);pol.configure(cfg);modes=[]
    for i in range(12):
        obs,prompt=items[(i//3)%3];r=pol.predict(obs,prompt,sample['suite'],25000+i);modes.append(r['route']['scoring_mode'])
        assert r['counts']['kalman']==0 and r['route']['verify_performed']==r['route']['draft_attempted'];ack(pol,r)
    release(pol)
    atomic(out/'compatibility_gate.json',dict(passed=True,warm_original=compat,error_gradient_graph_eager_independent=gradient_checks,verified_sequence=12,modes=modes,kalman_instances=0,max_allocated_mib=torch.cuda.max_memory_allocated()/2**20))
    atomic(out/'isolation_gate.json',dict(passed=True,actors=1,reason='Single policy client per GPU after measured two-client memory failures in previous run'))
    return 1

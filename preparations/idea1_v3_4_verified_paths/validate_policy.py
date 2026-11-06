"""CPU tests of rejection, partial execution, refresh and per-client state."""
import json,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from . import policy as p
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_E2_adjacency_200.common import choose

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/idea1_v3_4_verified_paths'
MASKS={12:[2,3,5,14,16,17],14:[3,4,9,11]}

class Selector:
    q=1;rule='spaced'
    def __init__(self,b):pass
    def clear(self):pass
    def run(self,c,seed,coeff,k):
        g=np.sin(np.arange(self.q*17).reshape(self.q,17)+seed).astype(np.float32)
        score=np.sqrt(np.mean(g*g,axis=0));return choose(score,k,'spaced'),g,score,dict(selector_total_ms=0.,selector_capture_ms=0.,selector_hot_ms=0.)

class Backend:
    adapter=SimpleNamespace(measure=lambda fn:(fn(),0.))
    def __init__(self,events=()):self.events=list(events);self.index=0;self.patches=[]
    def prepare(self,obs,instruction,skip=()):
        value=float(obs['value']);return dict(K=torch.full((2,2),value),V=torch.full((2,2),value),value=value),dict(vision_ms=0.,vlm_ms=0.)
    def sample(self,c,seed,steps):
        raw=torch.zeros(50,32);raw[:,6]=-1.
        event=self.events[self.index] if self.index<len(self.events) else 'accept'
        if steps==2 and event=='post_gripper':raw[2:,6]=1.
        return raw[:,:7].numpy().copy(),raw,dict(nodes=list(range(steps+1)),ae_ms=0.)
    def verify(self,full,current,raw,seed,suite):
        event=self.events[self.index] if self.index<len(self.events) else 'accept';self.index+=1
        hats=np.stack([raw.numpy().copy()]*2)
        if event=='reject':hats[:,0,:6]+=.2
        elif event=='partial2':hats[:,2,:6]+=.2
        elif event=='verify_gripper':hats[:,1,6]=1.
        elif event=='post_gripper':hats[:,:,6]=-1.
        current['K'].fill_(-999.)
        assert float(full['K'][0,0])==full['value'],'Draft overwrote Full KV'
        return hats,dict(verify_ae_ms=0.)

def config(budget,period,selection='fixed',q=0):
    return dict(mode='verified',period=period,budget=budget,selection=selection,projections=q,skip=MASKS[budget],compensation='extrapolate',tau=.10,protect_gripper=True,kalman=False)

def run(cfg,events=(),n=16):
    b=Backend(events);pol=p.Policy(b);pol.configure(cfg);records=[]
    for i in range(n):
        r=pol.predict(dict(value=float(i)),'test','libero_spatial',100+i);records.append(r)
        pol.acknowledge(r['actions'][:r['execute_length']],np.zeros((r['execute_length'],4)))
        assert r['counts']['kalman']==0 and not hasattr(pol,'kalman')
    return pol,records,b

def main():
    p.Selector=Selector
    def prepare(b,obs,instruction,trace=False):
        c,t=b.prepare(obs,instruction)
        if trace:c['hidden_trace']=[torch.full((2,2),float(obs['value']))]*19
        return c,t
    def patch(b,obs,instruction,skip,last,older,ratio):
        b.patches.append(dict(last=float(last[1][0,0]),older=None if older is None else float(older[1][0,0]),ratio=ratio))
        return b.prepare(obs,instruction,skip)
    p.prepare_full=prepare;p.capture=lambda hidden:{i:hidden[0].clone() for i in range(1,18)};p.prepare_patch=patch
    checks=[]
    for budget in (12,14):
        for period in (3,5):
            for selection,q in (('fixed',0),('jacobian',1),('jacobian',4)):
                cfg=config(budget,period,selection,q);pol,rr,b=run(cfg)
                assert [r['route']['path'] for r in rr]==['full' if i%period==0 else 'fast_verified' for i in range(len(rr))]
                assert [r['route']['request_category'] for r in rr[:period]]==['full_startup']+['fast_accept_5']*(period-1)
                for i,r in enumerate(rr):
                    rt=r['route'];assert r['step']==i*5 and rt['steady_vjp_count']==(q if i%period==0 else 0)
                    if rt['draft_attempted']:assert np.isclose(rt['draft_extrapolation_ratio'],0. if i<period else (i%period)/period)
                expected_full=sum(i%period==0 for i in range(len(rr)))
                assert pol.counts==dict(full=expected_full,draft=len(rr)-expected_full,verify=len(rr)-expected_full,selector=expected_full if selection=='jacobian' else 0,vjp=expected_full*q,kalman=0)
                checks.append(dict(budget=budget,period=period,selection=selection,q=q,passed=True))
    cfg=config(12,5)
    pol,rr,b=run(cfg,['reject']+['accept']*8,n=7)
    assert rr[1]['route']['request_category']=='full_reject_spatial' and rr[1]['route']['accepted_length']==0 and rr[1]['execute_length']==5
    assert rr[1]['counts']['full']==2 and rr[1]['counts']['draft']==rr[1]['counts']['verify']==1
    assert [r['route'].get('draft_extrapolation_ratio') for r in rr[2:6]]==[1.,2.,3.,4.]
    assert rr[6]['route']['request_category']=='full_periodic'
    pol,rr,b=run(config(12,3),['partial2','partial2','accept'],n=5)
    assert [r['step'] for r in rr]==[0,5,7,9,14] and [r['execute_length'] for r in rr[:3]]==[5,2,2]
    assert np.isclose(rr[4]['route']['draft_extrapolation_ratio'],5/9)
    for event in ('verify_gripper','post_gripper'):
        pol,rr,b=run(cfg,[event],n=4)
        if event=='verify_gripper':assert rr[1]['route']['request_category']=='full_reject_gripper'
        else:
            assert rr[1]['route']['request_category']=='fast_accept_partial' and rr[1]['execute_length']==2
            assert rr[2]['route']['request_category']=='full_pending_gripper'
    baseline=dict(mode='full',period=1,budget=18,selection='none',projections=0,skip=[],compensation='none',tau=.10,protect_gripper=True,kalman=False)
    pol,rr,b=run(baseline,n=6);assert pol.counts==dict(full=6,draft=0,verify=0,selector=0,vjp=0,kalman=0)
    pol.configure(cfg);assert pol.full is None and pol.history==[] and pol.round==pol.step==0 and pol.last_gripper is None
    from .finish import audit
    audit_checks=[]
    scenarios=[(config(b,pd,sel,q),ev) for b in (12,14) for pd in (3,5) for sel,q in (('fixed',0),('jacobian',1),('jacobian',4)) for ev in ([],['reject'],['partial2','partial2'],['post_gripper'],['verify_gripper'])]+[(baseline,[])]
    with tempfile.TemporaryDirectory(prefix='verified-audit-') as tmp:
        for index,(cfg,events) in enumerate(scenarios):
            cfg=dict(cfg,id=f'synthetic_{index}');pol,records,b=run(cfg,events,n=12);d=Path(tmp)/str(index);d.mkdir();qs=[];arrays={}
            for i,record in enumerate(records):
                q=dict(record);arrays[f'raw_{i}']=q.pop('raw');arrays[f'actions_{i}']=q.pop('actions')
                for name,value in q.pop('arrays').items():arrays[f'{name}_{i}']=value
                qs.append(dict(q,replan=i,input_hash='synthetic',executed_length=q['execute_length']))
            (d/'infer.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in qs));np.savez(d/'arrays.npz',**arrays)
            (d/'identity.json').write_text(json.dumps(dict(init_vector_sha256='synthetic',bddl_sha256='synthetic',controller={},prompt='test',env_seed=7,settling=10)))
            r=dict(case=dict(suite='libero_spatial',task=0,init=19),predictions=len(qs),actions=sum(q['executed_length'] for q in qs),counts=qs[-1]['counts'])
            _,audited=audit(cfg,d,r,False);assert len(audited)==12
            audit_checks.append(dict(selection=cfg['selection'],budget=cfg['budget'],period=cfg['period'],events=events,passed=True))
    result=dict(passed=True,checks=checks,audit_scenarios=len(audit_checks),partial_execution_actual_step=True,reject_cost_counts=True,gripper_rules=True,unclamped_actual_age_gap_ratios=True,independent_full_kv=True,kalman_instances=0)
    atomic(OUT/'cpu_gate.json',result);print(json.dumps(result))

if __name__=='__main__':main()

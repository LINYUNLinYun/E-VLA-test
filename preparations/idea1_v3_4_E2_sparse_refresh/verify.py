"""CPU schedule isolation: Full history stays fresh while mask/VJP is held."""
import json
from types import SimpleNamespace
import numpy as np
import torch
from . import policy as p
from .common import OUT,CONDITIONS,legal
from preparations.idea1_v3.io import atomic

class Select:
    def __init__(self,error=False):self.calls=0;self.error=error
    def clear(self):pass
    def run(self,c,seed,coeff,*args):
        self.calls+=1
        if self.error:
            expected=torch.stack([torch.full((2,2),float(2*i)) for i in range(1,18)])
            assert torch.equal(args[0],expected),'History must include unscored Fulls'
        skip=[2,3,5,14,16,17] if self.calls%2 else [1,3,5,12,14,16]
        return skip,np.ones(17),np.ones(17),dict(selector_total_ms=0.,selector_hot_ms=0.,selector_capture_ms=0.)

class Backend:
    def __init__(self):self.full_calls=0;self.fast_calls=0;self.adapter=SimpleNamespace(measure=lambda f:(f(),0.))
    def sample(self,c,seed,D):
        return np.zeros((50,7),np.float32),torch.zeros(50,32),dict(nodes=[None]*(D+1))

def fake_full(b,obs,instruction,trace):
    cycle=b.full_calls;b.full_calls+=1
    bank={i:torch.full((2,2),float(i*cycle**2)) for i in range(1,18)}
    return dict(hidden_trace=bank),{}

def fake_patch(b,obs,instruction,skip,last,older,ratio):
    b.fast_calls+=1;cycle=b.full_calls-1
    assert last[1][0,0]==cycle**2
    assert (older is not None)==(cycle>0)
    if older is not None:assert older[1][0,0]==(cycle-1)**2
    assert ratio==(0. if cycle==0 else (1/3 if b.fast_calls%2 else 2/3))
    return {},{}

def main():
    original=(p.prepare_full,p.capture,p.prepare_patch)
    p.prepare_full=fake_full;p.capture=lambda trace:trace;p.prepare_patch=fake_patch
    checks=[]
    try:
        for stride in (1,2,3):
            b=Backend();pol=p.Policy.__new__(p.Policy);pol.b=b;pol.selector=Select();pol.error_selector=Select(error=True);pol.config={};pol.reset()
            cfg=dict(CONDITIONS[0],refresh_stride=stride);pol.configure(cfg);mask=None;mask_step=None;last_source=[]
            for i in range(54):
                r=pol.predict({},'test','libero_spatial',i);z=r['route'];cycle=i//3;full=i%3==0;refresh=full and cycle%stride==0
                assert z['steady_vjp_count']==z['selector_forward_count']==int(refresh)
                if refresh:
                    mask=z['selected_skip'];mask_step=i*5
                    if cycle>=2:last_source=[(cycle-2)*15,(cycle-1)*15,cycle*15]
                assert z['mask_refresh']==refresh and z['selected_skip']==mask and legal(mask,6)
                assert z['mask_age_env_steps']==i*5-mask_step
                assert z['history_full_steps']==([0] if cycle==0 else [(cycle-1)*15,cycle*15])
                assert z['score_source_full_steps']==last_source
                if not refresh:assert r['timing']['selector_total_ms']==r['timing']['error_build_ms']==0
                pol.acknowledge(r['actions'][:5],None)
            expected=len(range(0,18,stride))
            assert b.full_calls==18 and b.fast_calls==36 and pol.counts['selector']==expected
            assert pol.counts['full_without_selector']==18-expected
            assert pol.counts['warm_selector']==(2 if stride==1 else 1)
            checks.append(dict(stride=stride,passed=True,fulls=18,vjps=expected,history_includes_unscored_fulls=True))
    finally:p.prepare_full,p.capture,p.prepare_patch=original
    OUT.mkdir(parents=True,exist_ok=True);result=dict(passed=True,checks=checks,scope='CPU mock schedule and real forecast_error arithmetic; GPU proof separate')
    atomic(OUT/'cpu_verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()

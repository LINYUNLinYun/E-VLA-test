"""CPU-only test of every condition's route and independent residual state."""
import json
from types import SimpleNamespace
import numpy as np
import torch
from . import policy as p
from .common import CONDITIONS,OUT,choose,legal,combinations,global_mask
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_E2_adjacency_200.preflight import optimum

class Backend:
    adapter=SimpleNamespace(measure=lambda fn:(fn(),0.))
    def prepare(self,obs,instruction,skip):return dict(),dict(nodes=[])
    def sample(self,c,seed,steps):return np.zeros((50,7),np.float32),torch.zeros(50,32),dict()

class Selector:
    rule='spaced'
    def __init__(self,b):pass
    def clear(self):pass
    def run(self,c,seed,coeff,k):
        grad=np.sin(np.arange(17)+seed).astype(np.float32);score=abs(grad)
        return choose(score,k,'spaced'),grad,score,dict(selector_total_ms=0.,selector_hot_ms=0.,selector_capture_ms=0.)

def main():
    assert len(CONDITIONS)==18 and sum(c['n'] for c in CONDITIONS)==2320
    assert (len(combinations(4,'spaced')),len(combinations(6,'spaced')))==(1001,924)
    assert legal(global_mask(6),6,'spaced')
    for k in (4,6):
        for seed in range(16):
            score=np.random.default_rng(seed).normal(size=17).astype(np.float32);skip=choose(score,k,'spaced')
            assert np.isclose(score[np.array(skip)-1].astype(np.float64).sum(),optimum(score,k,'spaced'),atol=1e-6,rtol=1e-6)
    p.Selector=Selector
    p.prepare_full=lambda b,obs,instruction,trace:(dict(hidden_trace=[torch.zeros(2,2)]*19) if trace else {},dict(nodes=[]))
    p.cosine_scores=lambda hidden:torch.arange(17,dtype=torch.float32)
    p.capture=lambda hidden:{i:torch.zeros(2,2) for i in range(1,18)}
    p.prepare_patch=lambda b,obs,instruction,skip,last,previous,ratio:({},dict(nodes=[]))
    checks=[]
    for cfg in CONDITIONS:
        pol=p.Policy(Backend());pol.configure(cfg);fulls=selectors=0;first=None;previous=None
        for i in range(15):
            r=pol.predict({},'test','libero_spatial',100+i);rt=r['route'];full=i%5==0;fulls+=int(full)
            assert legal(rt['selected_skip'],cfg['skip_count'],'spaced')
            assert rt['path']==('full' if full else 'fast') and rt['D']==(10 if full else 2)
            assert rt['full_age_rounds']==i%5 and rt['full_age_env_steps']==i%5*5
            assert rt['steady_vjp_count']==int(full and cfg['method']=='jacobian')
            if first is None:first=list(rt['selected_skip'])
            if cfg['method'] in ('random_fixed','global_fixed'):assert rt['selected_skip']==first
            if cfg['method'] in ('jacobian','cosine') and not full:assert rt['selected_skip']==previous
            selectors+=int(rt['mask_refresh']);assert r['counts']['selector']==selectors and r['counts']['full']==fulls
            if cfg['compensation']=='extrapolate':
                assert rt['history_full_steps']==list(range(max(0,(i//5-1)*25),i//5*25+1,25))
                assert rt['patch_layers']==([] if full else rt['selected_skip'])
                assert np.isclose(rt['extrapolation_ratio'],0. if full or i<5 else (i%5)/5)
                assert rt['has_second_history']==(not full and i>=5)
            previous=list(rt['selected_skip']);pol.acknowledge(r['actions'][:5],None)
        pol.configure(cfg);assert pol.round==0 and pol.history==[] and pol.skip is None
        checks.append(dict(condition=cfg['id'],requests=15,passed=True))
    atomic(OUT/'cpu_gate.json',dict(passed=True,checks=checks,global_fixed=global_mask(6),new_episodes=2320))
    print(json.dumps(dict(passed=True,conditions=len(checks),requests=15*len(checks),global_fixed=global_mask(6))))

if __name__=='__main__':main()

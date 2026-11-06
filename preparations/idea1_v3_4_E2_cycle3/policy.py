import time
import numpy as np
from .common import choose,random_mask,coefficients,legal
from .selection import Selector,prepare_full,cosine_scores

class Policy:
    def __init__(self,b):self.b=b;self.selector=Selector(b);self.config={};self.reset()
    def reset(self):
        self.pending=None;self.round=0;self.step=0;self.fast_count=0;self.skip=None;self.mask_step=None;self.last_full_step=None;self.last_full_round=None
        self.counts=dict(full=0,draft=0,verify=0,selector=0)
    def configure(self,config):self.selector.clear();self.config=config;self.reset();return {}
    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None:raise RuntimeError('Unacknowledged action')
        tick=time.perf_counter();cfg=self.config;method=cfg['method'];k=cfg['skip_count'];full=self.round%3==0;arrays={}
        selection=dict(selector_total_ms=0.,selector_hot_ms=0.,selector_capture_ms=0.);refreshed=False
        if full:
            c,t=prepare_full(self.b,obs,instruction,trace=method=='cosine')
            actions,raw,cost=self.b.sample(c,seed,10)
            if method=='cosine':
                start=time.perf_counter();score=cosine_scores(c.pop('hidden_trace')).cpu().numpy();self.skip=choose(score,k)
                ms=(time.perf_counter()-start)*1000;selection.update(selector_total_ms=ms,selector_hot_ms=ms);arrays['layer_score']=score;refreshed=True
            elif method=='jacobian':
                self.skip,grad,score,selection=self.selector.run(c,seed,coefficients(),k);arrays.update(gate_gradient=grad,layer_score=score);refreshed=True
            elif self.skip is None:
                start=time.perf_counter();self.skip=random_mask(cfg['random_seed'],seed,k);ms=(time.perf_counter()-start)*1000
                selection.update(selector_total_ms=ms,selector_hot_ms=ms);refreshed=True
            if refreshed:self.mask_step=self.step;self.counts['selector']+=1
            self.last_full_step=self.step;self.last_full_round=self.round;self.counts['full']+=1
        else:
            if method=='random_online' and self.fast_count>0:
                start=time.perf_counter();self.skip=random_mask(cfg['random_seed'],seed,k);ms=(time.perf_counter()-start)*1000;selection.update(selector_total_ms=ms,selector_hot_ms=ms)
                self.mask_step=self.step;self.counts['selector']+=1;refreshed=True
            assert legal(self.skip,k)
            c,t=self.b.prepare(obs,instruction,self.skip);actions,raw,cost=self.b.sample(c,seed,2)
            self.fast_count+=1;self.counts['draft']+=1
        t.update(cost,**selection);t['policy_wall_ms']=(time.perf_counter()-tick)*1000;t['policy_hot_ms']=t['policy_wall_ms']-t['selector_capture_ms']
        result=dict(actions=np.asarray(actions,np.float32),raw=raw.float().cpu().numpy(),arrays=arrays,execute_length=5,route=dict(path='full' if full else 'fast',D=10 if full else 2,skip=[] if full else list(self.skip),selected_skip=list(self.skip),mask_refresh=refreshed,mask_age_env_steps=self.step-self.mask_step,full_age_rounds=self.round-self.last_full_round,full_age_env_steps=self.step-self.last_full_step,method=method,budget=cfg['budget']),timing=t,seed=seed,round=self.round,step=self.step,counts=dict(self.counts),full_cache_age=None if full else dict(rounds=self.round-self.last_full_round,env_steps=self.step-self.last_full_step))
        self.pending=result;self.round+=1;return result
    def acknowledge(self,executed,quaternions):
        if self.pending is None:raise ValueError('No proposal')
        executed=np.asarray(executed,np.float32);assert 0<len(executed)<=5 and np.array_equal(executed,self.pending['actions'][:len(executed)])
        self.step+=len(executed);self.pending=None;return {}

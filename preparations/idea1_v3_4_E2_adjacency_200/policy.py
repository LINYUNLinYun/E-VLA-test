import time
import numpy as np
import torch
from .common import choose,legal,longest_run
from preparations.idea1_v3_4_E2_cycle3.common import coefficients
from preparations.idea1_v3_4_E2_cycle3.policy import Policy as Parent
from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import Selector as BaseSelector,current_item

class Selector(BaseSelector):
    rule='max2'
    def run(self,c,seed,coeff,k=4):
        tick=time.perf_counter();item=current_item(c,seed);W=torch.as_tensor(coeff,device='cuda');cold=0.
        if self.graph is None or self.runtime_id!=id(c['runtime']):
            if self.graph is not None:self.clear()
            cold=self.capture(c,item,W)
        for name,value in item.items():self.item[name].copy_(value)
        self.W.copy_(W);self.graph.replay();grad=self.grad.float().cpu().numpy().copy();assert np.isfinite(grad).all()
        score=abs(grad);skip=choose(score,k,self.rule);torch.cuda.synchronize();total=(time.perf_counter()-tick)*1000
        return skip,grad,score,dict(selector_total_ms=total,selector_hot_ms=total-cold,selector_capture_ms=cold)

class Policy(Parent):
    def __init__(self,b):self.b=b;self.selector=Selector(b);self.config={};self.reset()
    def configure(self,config):
        assert config['method']=='jacobian' and config['compensation']=='none'
        assert config['adjacency_rule'] in ('unrestricted','spaced','max2')
        self.selector.rule=config['adjacency_rule']
        return super().configure(config)
    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None:raise RuntimeError('Unacknowledged action')
        tick=time.perf_counter();cfg=self.config;k=cfg['skip_count'];rule=cfg['adjacency_rule'];full=self.round%3==0;arrays={}
        selection=dict(selector_total_ms=0.,selector_hot_ms=0.,selector_capture_ms=0.)
        if full:
            c,t=self.b.prepare(obs,instruction,());actions,raw,cost=self.b.sample(c,seed,10)
            self.skip,grad,score,selection=self.selector.run(c,seed,coefficients(),k);arrays.update(gate_gradient=grad,layer_score=score)
            self.mask_step=self.step;self.counts['selector']+=1;self.last_full_step=self.step;self.last_full_round=self.round;self.counts['full']+=1
        else:
            assert legal(self.skip,k,rule)
            c,t=self.b.prepare(obs,instruction,self.skip);actions,raw,cost=self.b.sample(c,seed,2);self.fast_count+=1;self.counts['draft']+=1
        assert legal(self.skip,k,rule)
        t.update(cost,**selection);t['policy_wall_ms']=(time.perf_counter()-tick)*1000;t['policy_hot_ms']=t['policy_wall_ms']-t['selector_capture_ms']
        route=dict(path='full' if full else 'fast',D=10 if full else 2,skip=[] if full else list(self.skip),selected_skip=list(self.skip),mask_refresh=full,mask_age_env_steps=self.step-self.mask_step,full_age_rounds=self.round-self.last_full_round,full_age_env_steps=self.step-self.last_full_step,method='jacobian',budget=cfg['budget'],compensation='none',steady_vjp_count=int(full),adjacency_rule=rule,selected_longest_run=longest_run(self.skip),jacobian_model_steps=10,jacobian_output_steps=[1,2,3,4,5],jacobian_full_layers=18)
        result=dict(actions=np.asarray(actions,np.float32),raw=raw.float().cpu().numpy(),arrays=arrays,execute_length=5,route=route,timing=t,seed=seed,round=self.round,step=self.step,counts=dict(self.counts),full_cache_age=None if full else dict(rounds=self.round-self.last_full_round,env_steps=self.step-self.last_full_step))
        self.pending=result;self.round+=1;return result

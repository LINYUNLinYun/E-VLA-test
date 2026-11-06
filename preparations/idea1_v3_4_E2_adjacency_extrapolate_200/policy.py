import time
import numpy as np
from preparations.idea1_v3_4_E2_cycle3.policy import Policy as Parent
from preparations.idea1_v3_4_E2_adjacency_200.policy import Selector
from preparations.idea1_v3_4_E2_adjacency_200.common import legal,longest_run
from preparations.idea1_v3_4_E2_cycle3.common import coefficients
from preparations.idea1_v3_4_E2_cycle3.selection import prepare_full
from preparations.idea1_v3_4_E2_cycle3_extrapolate.backend import capture,prepare_patch

class Policy(Parent):
    def __init__(self,b):self.b=b;self.selector=Selector(b);self.config={};self.reset()
    def reset(self):super().reset();self.history=[]
    def configure(self,config):
        assert config['method']=='jacobian' and config['compensation']=='extrapolate'
        assert config['adjacency_rule'] in ('unrestricted','spaced','max2')
        self.selector.rule=config['adjacency_rule']
        return super().configure(config)
    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None:raise RuntimeError('Unacknowledged action')
        tick=time.perf_counter();cfg=self.config;k=cfg['skip_count'];rule=cfg['adjacency_rule'];full=self.round%3==0;arrays={};ratio=0.;has_older=False
        selection=dict(selector_total_ms=0.,selector_hot_ms=0.,selector_capture_ms=0.)
        if full:
            c,t=prepare_full(self.b,obs,instruction,trace=True);actions,raw,cost=self.b.sample(c,seed,10)
            hidden=c.pop('hidden_trace');residuals,ms=self.b.adapter.measure(lambda:capture(hidden));t['residual_capture_ms']=ms
            self.skip,grad,score,selection=self.selector.run(c,seed,coefficients(),k);arrays.update(gate_gradient=grad,layer_score=score)
            self.history=(self.history+[dict(step=self.step,residuals=residuals)])[-2:]
            self.mask_step=self.step;self.counts['selector']+=1;self.last_full_step=self.step;self.last_full_round=self.round;self.counts['full']+=1
        else:
            assert legal(self.skip,k,rule)
            last=self.history[-1];previous=self.history[-2] if len(self.history)==2 else None;has_older=previous is not None
            age=self.step-last['step'];gap=None if previous is None else last['step']-previous['step'];ratio=0. if gap is None else age/gap
            c,t=prepare_patch(self.b,obs,instruction,self.skip,last['residuals'],None if previous is None else previous['residuals'],ratio);t['residual_capture_ms']=0.
            actions,raw,cost=self.b.sample(c,seed,2);self.fast_count+=1;self.counts['draft']+=1
        assert legal(self.skip,k,rule)
        t.update(cost,**selection);t['policy_wall_ms']=(time.perf_counter()-tick)*1000;t['policy_hot_ms']=t['policy_wall_ms']-t['selector_capture_ms']
        route=dict(path='full' if full else 'fast',D=10 if full else 2,skip=[] if full else list(self.skip),selected_skip=list(self.skip),mask_refresh=full,mask_age_env_steps=self.step-self.mask_step,full_age_rounds=self.round-self.last_full_round,full_age_env_steps=self.step-self.last_full_step,method='jacobian',budget=cfg['budget'],compensation='extrapolate',steady_vjp_count=int(full),adjacency_rule=rule,selected_longest_run=longest_run(self.skip),jacobian_model_steps=10,jacobian_output_steps=[1,2,3,4,5],jacobian_full_layers=18,history_full_steps=[h['step'] for h in self.history],has_second_history=has_older,extrapolation_ratio=ratio,patch_layers=[] if full else list(self.skip),captured_layers=list(range(1,18)),history_bytes=sum(v.numel()*v.element_size() for h in self.history for v in h['residuals'].values()))
        result=dict(actions=np.asarray(actions,np.float32),raw=raw.float().cpu().numpy(),arrays=arrays,execute_length=5,route=route,timing=t,seed=seed,round=self.round,step=self.step,counts=dict(self.counts),full_cache_age=None if full else dict(rounds=self.round-self.last_full_round,env_steps=self.step-self.last_full_step))
        self.pending=result;self.round+=1;return result

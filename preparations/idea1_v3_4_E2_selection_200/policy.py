"""Original four rules, native cycle3, optional own-history extrapolation."""
import time
import numpy as np
from preparations.idea1_v3_4_E2_cycle3.policy import Policy as Parent
from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import Selector
from preparations.idea1_v3_4_E2_cycle3.selection import prepare_full,cosine_scores
from preparations.idea1_v3_4_E2_cycle3.common import choose,random_mask,coefficients,legal
from preparations.idea1_v3_4_E2_cycle3_extrapolate.backend import capture,prepare_patch

class Policy(Parent):
    def __init__(self,b):self.b=b;self.selector=Selector(b);self.config={};self.reset()
    def reset(self):super().reset();self.history=[]
    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None:raise RuntimeError('Unacknowledged action')
        tick=time.perf_counter();cfg=self.config;method=cfg['method'];k=cfg['skip_count'];comp=cfg['compensation']=='extrapolate'
        full=self.round%3==0;arrays={};refreshed=False;ratio=0.;has_older=False
        selection=dict(selector_total_ms=0.,selector_hot_ms=0.,selector_capture_ms=0.)
        if full:
            c,t=prepare_full(self.b,obs,instruction,trace=comp or method=='cosine');actions,raw,cost=self.b.sample(c,seed,10)
            hidden=c.pop('hidden_trace',None)
            if comp:
                residuals,ms=self.b.adapter.measure(lambda:capture(hidden));t['residual_capture_ms']=ms
            if method=='cosine':
                start=time.perf_counter();score=cosine_scores(hidden).cpu().numpy();self.skip=choose(score,k);ms=(time.perf_counter()-start)*1000
                selection.update(selector_total_ms=ms,selector_hot_ms=ms);arrays['layer_score']=score;refreshed=True
            elif method=='jacobian':
                self.skip,grad,score,selection=self.selector.run(c,seed,coefficients(),k);arrays.update(gate_gradient=grad,layer_score=score);refreshed=True
            elif self.skip is None:
                start=time.perf_counter();self.skip=random_mask(cfg['random_seed'],seed,k);ms=(time.perf_counter()-start)*1000
                selection.update(selector_total_ms=ms,selector_hot_ms=ms);refreshed=True
            if comp:self.history.append(dict(step=self.step,residuals=residuals));self.history=self.history[-2:]
            if refreshed:self.mask_step=self.step;self.counts['selector']+=1
            self.last_full_step=self.step;self.last_full_round=self.round;self.counts['full']+=1
        else:
            if method=='random_online' and self.fast_count>0:
                start=time.perf_counter();self.skip=random_mask(cfg['random_seed'],seed,k);ms=(time.perf_counter()-start)*1000
                selection.update(selector_total_ms=ms,selector_hot_ms=ms);self.mask_step=self.step;self.counts['selector']+=1;refreshed=True
            assert legal(self.skip,k)
            if comp:
                last=self.history[-1];previous=self.history[-2] if len(self.history)==2 else None;has_older=previous is not None
                age=self.step-last['step'];gap=None if previous is None else last['step']-previous['step'];ratio=0. if gap is None else age/gap
                c,t=prepare_patch(self.b,obs,instruction,self.skip,last['residuals'],None if previous is None else previous['residuals'],ratio);t['residual_capture_ms']=0.
            else:c,t=self.b.prepare(obs,instruction,self.skip)
            actions,raw,cost=self.b.sample(c,seed,2);self.fast_count+=1;self.counts['draft']+=1
        t.update(cost,**selection);t['policy_wall_ms']=(time.perf_counter()-tick)*1000;t['policy_hot_ms']=t['policy_wall_ms']-t['selector_capture_ms']
        route=dict(path='full' if full else 'fast',D=10 if full else 2,skip=[] if full else list(self.skip),selected_skip=list(self.skip),mask_refresh=refreshed,mask_age_env_steps=self.step-self.mask_step,full_age_rounds=self.round-self.last_full_round,full_age_env_steps=self.step-self.last_full_step,method=method,budget=cfg['budget'],compensation=cfg['compensation'],steady_vjp_count=int(full and method=='jacobian'))
        if comp:route.update(history_full_steps=[h['step'] for h in self.history],has_second_history=has_older,extrapolation_ratio=ratio,patch_layers=[] if full else list(self.skip),captured_layers=list(range(1,18)),history_bytes=sum(v.numel()*v.element_size() for h in self.history for v in h['residuals'].values()))
        if method=='jacobian':route.update(jacobian_model_steps=10,jacobian_output_steps=[1,2,3,4,5],jacobian_full_layers=18)
        result=dict(actions=np.asarray(actions,np.float32),raw=raw.float().cpu().numpy(),arrays=arrays,execute_length=5,route=route,timing=t,seed=seed,round=self.round,step=self.step,counts=dict(self.counts),full_cache_age=None if full else dict(rounds=self.round-self.last_full_round,env_steps=self.step-self.last_full_step))
        self.pending=result;self.round+=1;return result

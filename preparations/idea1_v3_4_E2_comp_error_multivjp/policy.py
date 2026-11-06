import time
import numpy as np
from preparations.idea1_v3_4_E2_cycle3.policy import Policy as Parent
from preparations.idea1_v3_4_E2_cycle3_full10_jacobian.model import Selector as WarmSelector
from .model import Selector as ErrorSelector,forecast_error,projection_weights
from preparations.idea1_v3_4_E2_cycle3.selection import prepare_full
from preparations.idea1_v3_4_E2_cycle3_extrapolate.backend import capture,prepare_patch
from .common import legal,coefficients

class Policy(Parent):
    def __init__(self,b):
        self.b=b;self.selector=WarmSelector(b);self.error_selector=ErrorSelector(b);self.config={};self.reset()
    def reset(self):
        super().reset();self.history=[];self.error_buffer=None;self.warm_released=False
        self.score_mode=None;self.score_source_steps=[];self.score_error_ratio=None
        self.counts.update(warm_selector=0,error_selector=0)
    def configure(self,config):
        self.selector.clear();self.error_selector.clear();self.config=config;self.reset();return {}
    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None:raise RuntimeError('Unacknowledged action')
        tick=time.perf_counter();cfg=self.config;k=cfg['skip_count'];full=self.round%3==0;arrays={};ratio=0.;has_older=False
        selection=dict(selector_total_ms=0.,selector_hot_ms=0.,selector_capture_ms=0.,error_build_ms=0.,selector_transition_ms=0.)
        if full:
            c,t=prepare_full(self.b,obs,instruction,trace=True);actions,raw,cost=self.b.sample(c,seed,10)
            residuals,capture_ms=self.b.adapter.measure(lambda:capture(c.pop('hidden_trace')));t['residual_capture_ms']=capture_ms
            if len(self.history)<2:
                self.skip,grad,score,warm=self.selector.run(c,seed,coefficients(),k)
                selection.update(warm);self.score_mode='warm_original';self.score_source_steps=[];self.score_error_ratio=None
                self.counts['warm_selector']+=1
            else:
                start=time.perf_counter();transition_ms=0.
                if not self.warm_released:
                    release_tick=time.perf_counter();self.selector.clear();transition_ms=(time.perf_counter()-release_tick)*1000;self.warm_released=True
                last,older=self.history[-1],self.history[-2]
                self.score_source_steps=[older['step'],last['step'],self.step]
                self.score_error_ratio=(self.step-last['step'])/(last['step']-older['step'])
                self.error_buffer,build_ms=self.b.adapter.measure(lambda:forecast_error(residuals,last['residuals'],older['residuals'],self.score_error_ratio,out=self.error_buffer))
                self.skip,grad,score,error_cost=self.error_selector.run(c,seed,projection_weights(cfg['projections']),self.error_buffer,k)
                total_ms=(time.perf_counter()-start)*1000
                selection.update(error_cost,selector_total_ms=total_ms,selector_hot_ms=total_ms-error_cost['selector_capture_ms']-transition_ms,selector_capture_ms=error_cost['selector_capture_ms']+transition_ms,error_build_ms=build_ms,selector_transition_ms=transition_ms)
                arrays['prediction_error_rms']=self.error_buffer.float().square().mean((1,2)).sqrt().cpu().numpy()
                self.score_mode='comp_error';self.counts['error_selector']+=1
            arrays.update(gate_gradient=grad,layer_score=score)
            self.history.append(dict(step=self.step,residuals=residuals));self.history=self.history[-2:]
            self.mask_step=self.step;self.last_full_step=self.step;self.last_full_round=self.round
            self.counts['full']+=1;self.counts['selector']+=1
        else:
            assert legal(self.skip,k) and self.history
            last=self.history[-1];previous=self.history[-2] if len(self.history)==2 else None;has_older=previous is not None
            age=self.step-last['step'];gap=None if previous is None else last['step']-previous['step'];ratio=0. if gap is None else age/gap
            c,t=prepare_patch(self.b,obs,instruction,self.skip,last['residuals'],None if previous is None else previous['residuals'],ratio)
            t['residual_capture_ms']=0.;actions,raw,cost=self.b.sample(c,seed,2);self.fast_count+=1;self.counts['draft']+=1
        t.update(cost,**selection);t['policy_wall_ms']=(time.perf_counter()-tick)*1000;t['policy_hot_ms']=t['policy_wall_ms']-t['selector_capture_ms']
        route=dict(path='full' if full else 'fast',D=10 if full else 2,skip=[] if full else list(self.skip),selected_skip=list(self.skip),mask_refresh=full,mask_age_env_steps=self.step-self.mask_step,full_age_rounds=self.round-self.last_full_round,full_age_env_steps=self.step-self.last_full_step,method='comp_error',budget=cfg['budget'],compensation='extrapolate',history_full_steps=[h['step'] for h in self.history],has_second_history=has_older,extrapolation_ratio=ratio,patch_layers=[] if full else list(self.skip),captured_layers=list(range(1,18)),history_bytes=sum(v.numel()*v.element_size() for h in self.history for v in h['residuals'].values()),scoring_mode=self.score_mode,score_source_full_steps=list(self.score_source_steps),score_error_ratio=self.score_error_ratio,error_buffer_bytes=0 if self.error_buffer is None else self.error_buffer.numel()*self.error_buffer.element_size(),jacobian_model_steps=10,jacobian_output_steps=[1,2,3,4,5],jacobian_full_layers=18,configured_projections=cfg['projections'],selector_forward_count=int(full),steady_vjp_count=(1 if self.score_mode=='warm_original' else cfg['projections']) if full else 0)
        result=dict(actions=np.asarray(actions,np.float32),raw=raw.float().cpu().numpy(),arrays=arrays,execute_length=5,route=route,timing=t,seed=seed,round=self.round,step=self.step,counts=dict(self.counts),full_cache_age=None if full else dict(rounds=self.round-self.last_full_round,env_steps=self.step-self.last_full_step))
        self.pending=result;self.round+=1;return result

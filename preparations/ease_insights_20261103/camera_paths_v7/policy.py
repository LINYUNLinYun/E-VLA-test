"""Full+2Fast cadence with separately cached attention or FFN residual banks."""
import copy
import hashlib
import time
import numpy as np
from .common import validate_condition,FIXED_LAYERS


class Policy:
    def __init__(self,backend):self.b=backend;self.config=None;self.reset()
    def reset(self):
        self.pending=None;self.round=self.step=self.draft_rounds=0;self.history=[]
        self.history_steps=[];self.last_full_step=None
        self.counts=dict(full=0,draft=0,verify=0);self.b.reset_vision()
    def configure(self,config):
        validate_condition(config);self.config=copy.deepcopy(config);self.reset();return {}
    def predict(self,obs,instruction,suite,seed):
        if self.config is None:raise ValueError('Configure before prediction')
        if self.pending is not None:raise RuntimeError('Unacknowledged execution')
        tick=time.perf_counter();cfg=self.config;kind=cfg['component']
        full=cfg['mode']=='full' or self.last_full_step is None or self.draft_rounds>=2
        ratio=0.;kwargs={}
        if kind is not None:
            if full:kwargs['capture']=kind
            else:
                if not self.history:raise RuntimeError('Missing Full residual history')
                if len(self.history)==2:
                    gap=self.history_steps[-1]-self.history_steps[-2]
                    if gap<=0:raise ValueError('Invalid Full history interval')
                    ratio=(self.step-self.history_steps[-1])/gap
                kwargs.update(**{'skip_'+kind:FIXED_LAYERS},history=self.history[-1],
                    older=self.history[-2] if len(self.history)==2 else None,ratio=ratio)
        self.b.request_vision(dict(kind='native'))
        context,timing=self.b.prepare_components(obs,instruction,**kwargs)
        actions,raw,cost=self.b.sample(context,seed,10 if full else 2)
        actions=np.asarray(actions,np.float32);raw=raw.float().cpu().numpy()
        if actions.shape!=(50,7) or raw.shape!=(50,32) or not np.isfinite(actions).all() or not np.isfinite(raw).all():
            raise ValueError('Invalid component-path outputs')
        if full:
            if kind is not None:
                bank=dict(identity=context['module_identity'],residuals={kind:{l:context['residuals'][kind][l] for l in FIXED_LAYERS}})
                self.history=[*self.history,bank][-2:]
            self.last_full_step=self.step;self.history_steps=[*self.history_steps,self.step][-2:]
            self.draft_rounds=0;self.counts['full']+=1
        else:self.draft_rounds+=1;self.counts['draft']+=1
        timing.update(cost);timing['policy_wall_ms']=(time.perf_counter()-tick)*1000
        noise=np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32)
        route=dict(path='full' if full else 'vlm_component',refresh=full,
            component=kind,skip_layers=[],capture_layers=[],compensation=cfg['compensation'],
            component_layers=list(FIXED_LAYERS) if kind is not None else [],
            extrapolation_ratio=ratio,component_execution_audit=context['module_meta'],component_reference_only=False,
            ae_bypassed_layers=[],ae_execution_audit=cost['ae_execution_audit'],
            camera_execution_audit=cost['camera_execution_audit'],camera_cache_used=False,
            history_steps=list(self.history_steps),selector_calls=0)
        result=dict(actions=actions,raw=raw,arrays={},execute_length=5,timing=timing,seed=seed,
            round=self.round,step=self.step,counts=dict(self.counts),full_cache_age=self.step-self.last_full_step,
            route=route,model_noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest())
        self.pending=result;self.round+=1;return result
    def acknowledge(self,executed,quaternions=None):
        if self.pending is None:raise ValueError('No proposal')
        executed=np.asarray(executed,np.float32)
        if not 0<len(executed)<=5 or not np.array_equal(executed,self.pending['actions'][:len(executed)]):raise ValueError('Execution mismatch')
        self.step+=len(executed);self.pending=None;return {}

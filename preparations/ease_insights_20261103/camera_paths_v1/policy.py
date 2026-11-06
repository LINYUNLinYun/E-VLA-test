"""Explicit camera history, reset per episode, with one Full plus four Fast rounds."""
import hashlib
import time
import numpy as np
from .common import validate_condition


class Policy:
    def __init__(self,backend):
        self.b=backend;self.config=None;self.reset()

    def reset(self):
        self.pending=None;self.round=self.step=self.draft_rounds=0
        self.cache=None;self.cache_step=None;self.history_steps=[]
        self.counts=dict(full=0,draft=0,verify=0);self.b.reset_vision()

    def configure(self,config):
        validate_condition(config);self.config=dict(config);self.reset();return {}

    def predict(self,obs,instruction,suite,seed):
        if self.config is None:raise ValueError('Configure before prediction')
        if self.pending is not None:raise RuntimeError('Unacknowledged execution')
        tick=time.perf_counter();cfg=self.config
        full=cfg['mode']=='full' or self.cache is None or self.draft_rounds>=cfg['period']
        if full:request=dict(kind='full' if cfg['mode']=='full' else 'refresh')
        elif cfg['fresh_camera'] is None:request=dict(kind='both')
        else:request=dict(kind='single',camera=cfg['fresh_camera'],cache=self.cache,
            source_step=self.cache_step,age_steps=self.step-self.cache_step)
        self.b.request_vision(request)
        context,timing=self.b.prepare_residual(obs,instruction)
        actions,raw,cost=self.b.sample(context,seed,10 if full else 5,ae_skip=())
        if full:
            if cfg['mode']!='full' and self.b.last_full_vision is None:raise RuntimeError('Full did not capture both raw camera maps')
            self.cache=self.b.last_full_vision if cfg['mode']!='full' else None;self.cache_step=self.step
            self.history_steps=[*self.history_steps,self.step][-2:]
            self.draft_rounds=0;self.counts['full']+=1
        else:self.draft_rounds+=1;self.counts['draft']+=1
        actions=np.asarray(actions,np.float32);raw=raw.float().cpu().numpy()
        if not np.isfinite(actions).all() or not np.isfinite(raw).all():raise ValueError('Nonfinite camera policy actions')
        timing.update(cost);timing['policy_wall_ms']=(time.perf_counter()-tick)*1000
        noise=np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32)
        result=dict(actions=actions,raw=raw,arrays={},execute_length=5,timing=timing,seed=seed,
            round=self.round,step=self.step,counts=dict(self.counts),full_cache_age=self.step-self.cache_step,
            route=dict(path='full' if full else ('both_fresh' if cfg['fresh_camera'] is None else 'camera_hold'),refresh=full,
                skip_layers=[],capture_layers=[],compensation=None,ae_bypassed_layers=[],
                ae_execution_audit=cost['ae_execution_audit'],camera_execution_audit=cost['camera_execution_audit'],
                history_steps=list(self.history_steps),cache_source_step=self.cache_step,
                selector_calls=0,execution_audit=context['execution_audit']),
            model_noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest())
        self.pending=result;self.round+=1;return result

    def acknowledge(self,executed,quaternions=None):
        if self.pending is None:raise ValueError('No proposal')
        executed=np.asarray(executed,np.float32)
        if not 0<len(executed)<=5 or not np.array_equal(executed,self.pending['actions'][:len(executed)]):raise ValueError('Execution mismatch')
        self.step+=len(executed);self.pending=None;return {}

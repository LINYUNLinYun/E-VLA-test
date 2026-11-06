import hashlib,time
import numpy as np
class Policy:
    def __init__(self,b):self.b=b;self.config={};self.reset()
    def reset(self):
        self.pending=None;self.full=None;self.round=self.step=self.draft_rounds=0;self.counts=dict(full=0,draft=0,verify=0)
    def configure(self,config):self.config=config;self.reset();return {}
    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None:raise RuntimeError('Unacknowledged execution')
        start=time.perf_counter();mask=tuple(self.config['skip']);b=self.b
        age=None if self.full is None else self.step-self.full['step']
        refresh=self.full is None or self.draft_rounds>=2
        if refresh:
            c,t=b.prepare_replay(obs,instruction,capture_layers=mask)
            a,r,cost=b.sample(c,seed,10)
            self.full=dict(step=self.step,residuals=c['residuals']);self.draft_rounds=0;self.counts['full']+=1
        else:
            c,t=b.prepare_replay(obs,instruction,skip=mask,history=self.full['residuals'])
            a,r,cost=b.sample(c,seed,2);self.draft_rounds+=1;self.counts['draft']+=1
        raw=r.float().cpu().numpy();actions=np.asarray(a,np.float32)
        t.update(cost);t['policy_wall_ms']=(time.perf_counter()-start)*1000
        if not np.isfinite(raw).all() or not np.isfinite(actions).all():raise ValueError('Nonfinite model output')
        result=dict(actions=actions,raw=raw,execute_length=5,arrays={},timing=t,route=dict(refresh=refresh,path='full' if refresh else self.config['id'],
            skip=list(mask),history_step=self.full['step'],replayed_layers=[] if refresh else list(mask)),
            full_cache_age=age,round=self.round,step=self.step,seed=seed,counts=dict(self.counts),
            model_noise_sha256=hashlib.sha256(np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32).tobytes()).hexdigest())
        self.pending=result;self.round+=1;return result
    def acknowledge(self,executed,quaternions):
        if self.pending is None:raise RuntimeError('No proposal')
        a=np.asarray(executed,np.float32)
        if not 0<len(a)<=5 or not np.array_equal(a,self.pending['actions'][:len(a)]):raise ValueError('Execution mismatch')
        self.step+=len(a);self.pending=None;return {}

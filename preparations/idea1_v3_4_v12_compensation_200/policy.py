import hashlib,time
import numpy as np
from preparations.idea1_v3_4_v12_compensation.common import MASK,STATIC

class Policy:
    def __init__(self,b):self.b=b;self.config={};self.reset()
    def reset(self):
        self.pending=None;self.round=self.step=self.draft_rounds=0
        self.history=[];self.counts=dict(full=0,draft=0,verify=0)
    def configure(self,config):self.config=config;self.reset();return {}
    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None:raise RuntimeError('Unacknowledged execution')
        tick=time.perf_counter();cfg=self.config;b=self.b;mode=cfg['mode'];patch=cfg['compensation']
        isfull=mode=='full' or (mode=='periodic' and (not self.history or self.draft_rounds>=cfg['period']))
        age=None if not self.history else self.step-self.history[-1]['step']
        gap=None if len(self.history)<2 else self.history[-1]['step']-self.history[-2]['step']
        ratio=0. if gap is None else age/gap
        if isfull:
            c,t=b.prepare_residual(obs,instruction,capture=patch in ('reuse','extrapolate'))
            a,r,cost=b.sample(c,seed,10)
            self.history.append(dict(step=self.step,residuals=c.get('residuals')));self.history=self.history[-2:]
            self.draft_rounds=0;self.counts['full']+=1;path='full'
        else:
            hist=self.history[-1]['residuals'] if patch in ('reuse','extrapolate') else None
            older=self.history[-2]['residuals'] if patch=='extrapolate' and len(self.history)==2 else None
            c,t=b.prepare_residual(obs,instruction,MASK,history=hist,older=older,ratio=ratio,kind=patch if patch in STATIC else None)
            a,r,cost=b.sample(c,seed,2);self.draft_rounds+=1;self.counts['draft']+=1
            path=patch or 'v12'
        raw=r.float().cpu().numpy();actions=np.asarray(a,np.float32)
        t.update(cost);t['policy_wall_ms']=(time.perf_counter()-tick)*1000
        result=dict(actions=actions,raw=raw,arrays={},execute_length=5,timing=t,seed=seed,
            route=dict(path=path,compensation=patch,has_second_history=gap is not None,extrapolation_ratio=ratio,
                       refresh=isfull,history_steps=[h['step'] for h in self.history]),
            round=self.round,step=self.step,counts=dict(self.counts),full_cache_age=age,
            model_noise_sha256=hashlib.sha256(np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32).tobytes()).hexdigest())
        if not np.isfinite(actions).all() or not np.isfinite(raw).all():raise ValueError('Nonfinite actions')
        self.pending=result;self.round+=1;return result
    def acknowledge(self,executed,quaternions):
        if self.pending is None:raise ValueError('No proposal')
        executed=np.asarray(executed,np.float32)
        if not 0<len(executed)<=5 or not np.array_equal(executed,self.pending['actions'][:len(executed)]):raise ValueError('Execution mismatch')
        self.step+=len(executed);self.pending=None;return {}

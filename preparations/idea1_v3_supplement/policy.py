"""Supplement E1 routing; frozen v3 policy remains untouched."""
import time
import numpy as np
from preparations.idea1_v3.policy import Policy as OriginalPolicy,TritonBackend,V14,acceptance as strict_acceptance,verification_seed

REVISION='supplement-e1-router-v1'


def validate(config):
    if config.get('experiment')!='E1' or config.get('draft_steps',2)!=2: raise ValueError('V14 D2 required')
    if config.get('tau') not in (.10,.15,.25,.40): raise ValueError('Unfrozen tau')
    if config.get('period') not in (2,4,8,None): raise ValueError('Unfrozen period')
    if config.get('gripper_rule','strict') not in ('strict','confirmed_disagreement','off'): raise ValueError('Unknown gripper rule')
    if config.get('reject_steps',10) not in (2,10): raise ValueError('Reject D must be 2 or 10')


def acceptance(draft,hats,tau,last_gripper,rule):
    if rule=='strict': return strict_acceptance(draft,hats,tau,last_gripper)
    d=np.asarray(draft)[:5]; h=np.asarray(hats)[:,:len(d)]
    error=h[...,:6]-d[None,:,:6]; distance=np.linalg.norm(error,axis=-1)/np.sqrt(6.)
    length=int(np.cumprod(np.all(distance<=tau,axis=0)).sum())
    disagreement=np.all((h[...,6]<0)!=(d[None,:,6]<0),axis=0)
    veto=False
    if rule=='confirmed_disagreement':
        indices=np.flatnonzero(disagreement[:length])
        if len(indices): length=int(indices[0]); veto=True
    return length,dict(distance=distance.tolist(),signed_error=error.tolist(),
        verify_gripper=veto,post_verify_gripper=False,confirmed_disagreement=disagreement.tolist())


class Policy(OriginalPolicy):
    def __init__(self,config,backend=None):
        validate(config); self.config=dict(config); self.backend=backend or TritonBackend()
        self.protocol=dict(self.backend.protocol,config=config,revision=REVISION,
            full_D_by_reason=dict(startup=10,periodic=10,zero_accept=config.get('reject_steps',10),gripper=config.get('reject_steps',10),gripper_window=config.get('reject_steps',10)))
        self.reset()

    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None: raise RuntimeError('Previous execution must be acknowledged before replanning')
        tick=time.perf_counter(); offset=len(getattr(self.backend,'startup_events',[]))
        cfg=self.config; rule=cfg.get('gripper_rule','strict'); timing={}; route={}; arrays={}
        age=None if self.full is None else dict(rounds=self.round-self.full['round'],env_steps=self.step-self.full['step'],wall_s=time.perf_counter()-self.full['time'])
        def full(reason):
            steps=10 if reason in ('startup','periodic') else cfg.get('reject_steps',10)
            context,t=self.backend.prepare(obs,instruction,())
            actions,raw,costs=self.backend.sample(context,seed,steps)
            timing.update({f'full_{k}':v for k,v in dict(t,**costs).items()})
            self.full=dict(context,round=self.round,step=self.step,time=time.perf_counter())
            self.draft_rounds=0; self.pending_gripper=False; self.counts['full']+=1
            route.update(path='full',full_reason=reason,full_steps=steps)
            return actions,raw,5
        # Reason priority is startup > strict pending > periodic > draft rejection.
        if self.full is None: actions,raw,length=full('startup')
        elif rule=='strict' and self.pending_gripper: actions,raw,length=full('gripper_window')
        elif cfg.get('period') is not None and self.draft_rounds>=cfg['period']: actions,raw,length=full('periodic')
        else:
            context,t=self.backend.prepare(obs,instruction,V14)
            actions,raw,costs=self.backend.sample(context,seed,2)
            timing.update({f'draft_{k}':v for k,v in dict(t,**costs).items()}); self.counts['draft']+=1
            hats,costs=self.backend.verify(self.full,context,raw,seed,suite)
            timing.update(costs); self.counts['verify']+=1
            raw_np=raw.float().cpu().numpy() if hasattr(raw,'cpu') else np.asarray(raw)
            if hats.shape!=(2,50,32) or not np.isfinite(hats).all(): raise ValueError('Invalid verifier reconstruction')
            length,details=acceptance(raw_np,hats,cfg['tau'],self.last_gripper,rule)
            arrays.update(draft_future=raw_np[:5],reconstructed_future=hats[:,:5])
            route.update(path='v14_verified',accepted_length=length,**details)
            self.draft_rounds+=1
            switch=details['verify_gripper'] or details['post_verify_gripper']
            if length==0: actions,raw,length=full('gripper' if switch else 'zero_accept')
            elif rule=='strict' and switch: self.pending_gripper=True
        raw_np=raw.float().cpu().numpy() if hasattr(raw,'cpu') else np.asarray(raw)
        actions=np.asarray(actions,np.float32)
        if actions.shape!=(50,7) or not np.isfinite(actions).all(): raise ValueError('Invalid action chunk')
        route.update(gripper_rule=rule,reject_steps=cfg.get('reject_steps',10),router_revision=REVISION)
        result=dict(actions=actions,execute_length=min(length,5),raw=raw_np,route=route,timing=timing,
            counts=dict(self.counts),full_cache_age=age,seed=seed,verification_seed=verification_seed(seed),round=self.round,step=self.step,arrays=arrays,
            startup_events=getattr(self.backend,'startup_events',[])[offset:])
        timing['policy_wall_ms']=(time.perf_counter()-tick)*1000
        self.round+=1; self.pending=result
        return result

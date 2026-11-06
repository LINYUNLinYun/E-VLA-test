"""Execution-acknowledged, per-episode state for all four v3.3 experiments."""
from pathlib import Path
import json
import sys
import time
import numpy as np
from preparations.idea1_v3.policy import V14, verification_seed
from .rules import gate, acceptance

XMASK=(4,5,6,13,14,15,22,23,24,31,32,33)


class Policy:
    def __init__(self, backend):
        self.backend=backend
        self.protocol=backend.protocol
        self.config={}
        self.reset()

    def configure(self,config):
        self.config=dict(config); self.reset()
        return {'ok':True}

    def reset(self):
        self.full=None; self.pending=None; self.pending_gripper=False
        self.round=self.step=self.draft_rounds=0
        self.last_gripper=None; self.last_kalman=False
        self.commands=[]; self.quaternions=[]
        self.counts=dict(full=0,verify=0,draft=0,kalman=0,v14=0)
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'policies/realtime-vla-flash'))
        from scripts.idea1_v2_1.filter import Parameters,CommandKalman
        frozen=json.loads(Path(__file__).with_name('kalman_frozen.json').read_text())
        self.kalman=CommandKalman(Parameters(**frozen['parameters']),frozen['shape_q'],frozen['shape_r'])
        return {'ok':True}

    def acknowledge(self,executed,quaternions):
        a=np.asarray(executed,np.float32); q=np.asarray(quaternions,float)
        if self.pending is None or not 0<len(a)<=self.pending['execute_length'] or q.shape!=(len(a),4): raise ValueError('Invalid execution acknowledgement')
        if not np.array_equal(a,self.pending['actions'][:len(a)]): raise ValueError('Executed actions differ from proposal')
        for action,quat in zip(a,q):
            self.kalman.observe(action); self.commands.append(action.copy()); self.quaternions.append(quat.copy())
        self.last_gripper=float(self.pending['raw'][len(a)-1,6])
        self.last_kalman=self.pending['route']['path']=='kalman_verified'
        self.step+=len(a); self.pending=None
        return {'ok':True}

    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None: raise RuntimeError('Unacknowledged proposal')
        cfg=self.config; b=self.backend; x=cfg['model']=='xiaomi'
        horizon,execute=(10,10) if x else (50,5)
        mask=XMASK if x else V14
        tick=time.perf_counter(); timing={}; route={}; arrays={}
        diagnostic_ms=0.
        age=None if self.full is None else dict(rounds=self.round-self.full['round'],env_steps=self.step-self.full['step'])
        def full(reason):
            c,t=b.prepare(obs,instruction,()); a,r,cost=b.sample(c,seed,2 if x else cfg.get('full_steps',10))
            timing.update({f'full_{k}':v for k,v in dict(t,**cost).items()})
            self.full=dict(c,round=self.round,step=self.step)
            self.full_raw=r.detach().clone() if hasattr(r,'detach') else np.asarray(r).copy()
            self.counts['full']+=1; self.draft_rounds=0; self.pending_gripper=False
            route.update(path='full',full_reason=reason)
            return a,r,execute
        mode=cfg.get('mode','verified')
        if mode=='full':
            actions,raw,length=full('baseline')
        elif mode=='pure':
            c,t=b.prepare(obs,instruction,mask); actions,raw,cost=b.sample(c,seed,cfg.get('draft_steps',2))
            timing.update(t,**cost); self.counts['draft']+=1; route.update(path='pure',full_reason=None); length=execute
        elif self.full is None: actions,raw,length=full('startup')
        elif self.pending_gripper: actions,raw,length=full('pending_gripper')
        elif self.draft_rounds>=2: actions,raw,length=full('periodic')
        else:
            use_kalman=False
            if cfg['experiment']=='E2':
                allowed,blocked=gate(self.commands,self.quaternions,cfg.get('angle',20),cfg.get('change',.05),cfg.get('pose',5))
                if cfg.get('alternate'): allowed=len(self.commands)>=5; blocked=[] if allowed else ['history']
                route.update(gate_blocked=blocked,consecutive_kalman_block=self.last_kalman)
                use_kalman=allowed and not self.last_kalman
            if use_kalman:
                begin=time.perf_counter(); c=b.prepare_metadata(obs,instruction)
                xyz,_=self.kalman.forecast(50)
                actions=np.tile(self.commands[-1],(50,1)); actions[:,:3]=np.clip(xyz,-1,1)
                raw=b.encode_commands(c,actions)
                timing['kalman_metadata_ms']=(time.perf_counter()-begin)*1000
                self.counts['kalman']+=1
                route.update(path='kalman_verified',padding='zeros25',vision_calls=0,vlm_calls=0)
            else:
                c,t=b.prepare(obs,instruction,mask); actions,raw,cost=b.sample(c,seed,cfg.get('draft_steps',2))
                timing.update({f'draft_{k}':v for k,v in dict(t,**cost).items()}); self.counts['v14']+=1
                route.update(path='skip_verified' if x else 'v14_verified')
            self.counts['draft']+=1; self.draft_rounds+=1
            if mode=='periodic': length=execute; route.update(path='periodic_draft')
            else:
                hats,cost=b.verify(self.full,c,raw,seed,suite); timing.update(cost); self.counts['verify']+=1
                raw_np=raw.float().cpu().numpy() if hasattr(raw,'cpu') else np.asarray(raw)
                length,details=acceptance(raw_np,hats,cfg.get('tau',.10),execute,cfg.get('alpha'),self.last_gripper,protect=not x)
                arrays.update(draft_future=raw_np[:execute],reconstructed_future=hats[:,:execute])
                route.update(accepted_length=length,**details)
                if use_kalman and not getattr(self,'padding_diagnosed',False):
                    diagnosis_start=time.perf_counter()
                    alt=raw.clone(); alt[:,7:]=self.full_raw[:,7:]
                    other,cost=b.verify(self.full,c,alt,seed,suite)
                    route['padding_diagnostic']=dict(zero_errors=details['distance'],full_tail_errors=acceptance(alt.float().cpu().numpy(),other,cfg.get('tau',.1),5,last_gripper=self.last_gripper)[1]['distance'])
                    self.padding_diagnosed=True
                    diagnostic_ms=(time.perf_counter()-diagnosis_start)*1000
                switch=details['verify_gripper'] or details['post_verify_gripper']
                if length==0:
                    route['rejected_path']=route['path']; actions,raw,length=full('gripper' if switch else 'zero_accept')
                elif switch: self.pending_gripper=True
        raw_np=raw.float().cpu().numpy() if hasattr(raw,'cpu') else np.asarray(raw)
        actions=np.asarray(actions,np.float32)
        if actions.shape!=(horizon,7) or not np.isfinite(actions).all(): raise ValueError('Invalid decoded model output')
        timing['diagnostic_excluded_ms']=diagnostic_ms
        timing['policy_wall_ms']=(time.perf_counter()-tick)*1000-diagnostic_ms
        result=dict(actions=actions,raw=raw_np,execute_length=length,route=route,timing=timing,arrays=arrays,counts=dict(self.counts),full_cache_age=age,round=self.round,step=self.step,seed=seed,verification_seed=verification_seed(seed),noise_hash=getattr(b,'last_noise_hash',None))
        self.round+=1; self.pending=result
        return result

import hashlib
import time
import numpy as np
from preparations.idea1_v3.policy import V14,verification_seed
from preparations.idea1_v3_3.policy import Policy as ParentPolicy
from preparations.idea1_v3_3.rules import acceptance
from .backend import shifted_tail


class Policy(ParentPolicy):
    def reset(self):
        result=super().reset()
        self.previous_raw=None; self.previous_executed=None
        self.e3_selected={'zero':0,'nonzero':0}; self.shadow_pending=None
        return result

    def acknowledge(self,executed,quaternions):
        proposal=self.pending
        result=super().acknowledge(executed,quaternions)
        self.previous_raw=np.asarray(proposal['raw']).copy()
        self.previous_executed=len(executed)
        return result

    def predict(self,obs,instruction,suite,seed):
        if self.config['id']=='E3_parent':
            return self.parent_predict(obs,instruction,suite,seed)
        if self.pending is not None: raise RuntimeError('Unacknowledged execution')
        tick=time.perf_counter(); b=self.backend; cfg=self.config; timing={}; route={}
        age=None if self.full is None else dict(rounds=self.round-self.full['round'],env_steps=self.step-self.full['step'])
        mode=cfg['mode']
        isfull=mode=='full' or (mode=='periodic' and (self.full is None or self.draft_rounds>=2))
        if isfull:
            c,t=b.prepare(obs,instruction,()); actions,raw,cost=b.sample(c,seed,cfg.get('full_steps',10))
            self.full=dict(c,round=self.round,step=self.step)
            self.counts['full']+=1; self.draft_rounds=0
            route=dict(path='full',full_reason='baseline' if mode=='full' else ('startup' if age is None else 'periodic'))
        else:
            if 'cutoff' in cfg:
                c,t=b.prepare_hybrid(obs,instruction,self.full,cfg['cutoff'])
            else: c,t=b.prepare(obs,instruction,V14)
            actions,raw,cost=b.sample(c,seed,2)
            self.counts['draft']+=1; self.draft_rounds+=1
            route=dict(path='hybrid' if 'cutoff' in cfg else 'v14',cutoff=cfg.get('cutoff'))
        timing.update(t,**cost)
        timing['policy_wall_ms']=(time.perf_counter()-tick)*1000
        raw_np=raw.float().cpu().numpy()
        result=dict(actions=np.asarray(actions,np.float32),raw=raw_np,execute_length=5,route=route,timing=timing,
            arrays={},counts=dict(self.counts),full_cache_age=age,round=self.round,step=self.step,seed=seed,
            model_noise_sha256=hashlib.sha256(np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32).tobytes()).hexdigest())
        if result['actions'].shape!=(50,7) or not np.isfinite(result['actions']).all(): raise ValueError('Invalid decoded output')
        self.round+=1; self.pending=result
        return result

    def parent_predict(self,obs,instruction,suite,seed):
        b=self.backend; original=b.verify; captured={}
        def capture(full,current,raw,seed,suite):
            hats,cost=original(full,current,raw,seed,suite)
            captured.update(full=full,current=current,raw=raw.detach().clone(),hats=hats.copy(),last_gripper=self.last_gripper)
            return hats,cost
        b.verify=capture
        try: result=super().predict(obs,instruction,suite,seed)
        finally: b.verify=original
        # Full route time is preserved; all later mechanism work is a separate RPC.
        if captured:
            length=result['route']['accepted_length']
            group='zero' if length==0 else 'nonzero'
            limit=4 if group=='zero' else 2
            selected=self.e3_selected[group]<limit
            result['shadow_selection']=dict(group=group,selected=selected,ordinal=self.e3_selected[group])
            if selected:
                self.e3_selected[group]+=1
                self.shadow_pending=dict(captured,obs=obs,instruction=instruction,suite=suite,seed=seed,
                    step=result['step'],round=result['round'],group=group,
                    route=dict(result['route']),parent_timing=dict(result['timing']),
                    previous=None if self.previous_raw is None else shifted_tail(self.previous_raw,self.previous_executed))
        return result

    def shadow(self):
        p=self.shadow_pending
        if p is None: raise ValueError('No selected actual parent request')
        self.shadow_pending=None
        b=self.backend; start=time.perf_counter(); seed=p['seed']; suite=p['suite']
        obs,instruction=p['obs'],p['instruction']; draft=p['current']; rawdraft=p['raw']
        previous=p.pop('previous'); cfull,prep=b.prepare(obs,instruction,())
        baseline,rawfull,cost=b.sample(cfull,seed,10)
        outputs={'full_d10':(baseline,rawfull,cost),
                 'v14_d2':(b.decode(draft,rawdraft),rawdraft,{})}
        outputs['full_d2']=b.sample(cfull,seed,2)
        outputs['v14_d10']=b.sample(draft,seed,10)
        noise_seed=int.from_bytes(hashlib.sha256(f'idea1-v34/refine/{seed}'.encode()).digest()[:4],'little')
        for name,context,initial,q in (
            ('draft_full_q01',cfull,rawdraft,.1),('draft_full_q03',cfull,rawdraft,.3),
            ('draft_v14_q03',draft,rawdraft,.3),
            ('remaining_full_q03',cfull,previous,.3),('remaining_v14_q03',draft,previous,.3)):
            if initial is not None:
                outputs[name]=b.warm_sample(context,initial,noise_seed,q,2)
        metrics=[]; arrays={}
        from preparations.kv_reuse_probe.probe import action_metrics
        for name,(actions,raw,timing) in outputs.items():
            raw_np=raw.float().cpu().numpy()
            hats,verify_cost=b.verify(p['full'],draft,raw,seed,suite)
            length,detail=acceptance(raw_np,hats,.1,5,last_gripper=p['last_gripper'])
            arrays[name]=actions; arrays[name+'_raw']=raw_np
            metrics.append(dict(candidate=name,cost=timing,accepted_length=length,acceptance=detail,
                raw_first5_rmse=float(np.sqrt(np.mean((raw_np[:5,:7]-rawfull.float().cpu().numpy()[:5,:7])**2))),
                action_metrics=action_metrics(baseline,actions)))
        return dict(arrays=arrays,metrics=metrics,reference_full_prepare=prep,reference_full_sample=cost,
            shadow_wall_s=time.perf_counter()-start,group=p['group'],round=p['round'],step=p['step'],seed=seed,
            refine_noise_seed=noise_seed,route=p['route'],parent_timing=p['parent_timing'],previous_available=previous is not None)

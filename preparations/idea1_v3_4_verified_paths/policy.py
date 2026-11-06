"""Periodic Full / extrapolated D2 proposals with the existing verifier.

Only accepted action prefixes update state. Rejections pay for draft, verify,
and Full. No Kalman object or forecast is constructed by this policy.
"""
import time
import numpy as np
from preparations.idea1_v3.policy import verification_seed
from preparations.idea1_v3_3.rules import acceptance
from preparations.idea1_v3_4_E2_cycle3.selection import prepare_full
from preparations.idea1_v3_4_E2_cycle3.common import coefficients
from preparations.idea1_v3_4_E2_cycle3_extrapolate.backend import capture,prepare_patch
from preparations.idea1_v3_4_E2_adjacency_200.common import legal
from preparations.idea1_v3_4_E2_cycle5_q4_v12.model import Selector

class Policy:
    def __init__(self,b):
        self.b=b;self.selector=Selector(b);self.config={};self.reset()
    def reset(self):
        self.full=None;self.history=[];self.skip=None;self.mask_step=None
        self.round=self.step=self.draft_rounds=0;self.pending=None;self.pending_gripper=False;self.last_gripper=None
        self.counts=dict(full=0,draft=0,verify=0,selector=0,vjp=0,kalman=0)
    def configure(self,config):
        assert config['mode'] in ('full','verified') and config['period'] in (1,3,5)
        assert config['compensation'] in ('none','extrapolate') and config['selection'] in ('none','fixed','jacobian')
        if config['mode']=='verified':
            assert config['budget'] in (12,14) and config['period'] in (3,5)
            assert config['tau']==.10 and config['protect_gripper'] and not config.get('kalman',False)
            if config['selection']=='fixed':assert legal(config['skip'],18-config['budget'],'max2')
            else:assert config['selection']=='jacobian' and config['projections'] in (1,4)
        self.selector.clear();self.config=dict(config);self.reset()
        if config['selection']=='fixed':self.skip=list(config['skip']);self.mask_step=0
        if config['selection']=='jacobian':self.selector.q=config['projections'];self.selector.rule='spaced'
        return {}
    def acknowledge(self,executed,quaternions):
        executed=np.asarray(executed,np.float32);quaternions=np.asarray(quaternions)
        assert self.pending is not None and 0<len(executed)<=self.pending['execute_length']
        assert quaternions.shape==(len(executed),4) and np.isfinite(quaternions).all()
        assert np.array_equal(executed,self.pending['actions'][:len(executed)])
        self.last_gripper=float(self.pending['raw'][len(executed)-1,6]);self.step+=len(executed);self.pending=None
        return {}
    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None:raise RuntimeError('Unacknowledged execution')
        cfg=self.config;b=self.b;start=time.perf_counter();timing={};arrays={}
        age=None if self.full is None else dict(rounds=self.round-self.full['round'],env_steps=self.step-self.full['step'])
        route=dict(draft_attempted=False,verify_performed=False,accepted_length=None,full_reason=None,fallback=False,kalman=False,
                   period=cfg['period'],budget=cfg['budget'],selection=cfg['selection'],compensation=cfg['compensation'],
                   full_age_at_entry=age,draft_rounds_at_entry=self.draft_rounds,draft_skip=[],steady_vjp_count=0,selector_forward_count=0,mask_refresh=False)

        def full(reason):
            tick=time.perf_counter();patch=cfg['compensation']=='extrapolate'
            c,t=prepare_full(b,obs,instruction,trace=patch);actions,raw,cost=b.sample(c,seed,10)
            if patch:
                residuals,ms=b.adapter.measure(lambda:capture(c.pop('hidden_trace')));t['residual_capture_ms']=ms
                self.history=(self.history+[dict(step=self.step,residuals=residuals)])[-2:]
            if cfg['selection']=='jacobian':
                self.skip,grad,score,s=self.selector.run(c,seed,coefficients(),18-cfg['budget'])
                arrays.update(gate_gradient=grad,layer_score=score);timing.update(s)
                self.mask_step=self.step;self.counts['selector']+=1;self.counts['vjp']+=cfg['projections']
                route.update(mask_refresh=True,steady_vjp_count=cfg['projections'],selector_forward_count=1)
            self.full=dict(c,round=self.round,step=self.step);self.draft_rounds=0;self.pending_gripper=False;self.counts['full']+=1
            timing.update({f'full_{k}':v for k,v in dict(t,**cost).items()});timing['full_path_wall_ms']=(time.perf_counter()-tick)*1000
            route.update(path='full',full_reason=reason,full_steps=10)
            return actions,raw,5

        if cfg['mode']=='full':actions,raw,length=full('baseline')
        elif self.full is None:actions,raw,length=full('startup')
        elif self.pending_gripper:actions,raw,length=full('pending_gripper')
        elif self.draft_rounds>=cfg['period']-1:actions,raw,length=full('periodic')
        else:
            tick=time.perf_counter();assert self.skip is not None
            route.update(draft_attempted=True,draft_skip=list(self.skip),verify_full_round=self.full['round'],verify_full_step=self.full['step'],draft_steps=2)
            if cfg['compensation']=='extrapolate':
                last=self.history[-1];older=self.history[-2] if len(self.history)>1 else None
                gap=None if older is None else last['step']-older['step'];draft_age=self.step-last['step'];ratio=0. if gap is None else draft_age/gap
                c,t=prepare_patch(b,obs,instruction,self.skip,last['residuals'],None if older is None else older['residuals'],ratio)
                route.update(draft_history_full_steps=[h['step'] for h in self.history],draft_residual_age=draft_age,draft_full_gap=gap,draft_extrapolation_ratio=ratio,captured_layers=list(range(1,18)))
            else:c,t=b.prepare(obs,instruction,self.skip)
            actions,raw,cost=b.sample(c,seed,2);self.counts['draft']+=1;self.draft_rounds+=1
            timing.update({f'draft_{k}':v for k,v in dict(t,**cost).items()});timing['draft_path_wall_ms']=(time.perf_counter()-tick)*1000
            tick=time.perf_counter();hats,cost=b.verify(self.full,c,raw,seed,suite)
            assert hats.shape==(2,50,32) and np.isfinite(hats).all()
            raw_np=raw.float().cpu().numpy();length,detail=acceptance(raw_np,hats,cfg['tau'],5,last_gripper=self.last_gripper,protect=cfg['protect_gripper'])
            timing.update(cost);timing['verify_path_wall_ms']=(time.perf_counter()-tick)*1000
            self.counts['verify']+=1;route.update(verify_performed=True,accepted_length=length,**detail)
            arrays.update(draft_future=raw_np[:5].copy(),reconstructed_future=hats[:,:5].copy())
            switch=detail['verify_gripper'] or detail['post_verify_gripper']
            if length==0:
                route['fallback']=True;actions,raw,length=full('reject_gripper' if switch else 'reject_spatial')
            else:
                route['path']='fast_verified'
                if switch:self.pending_gripper=True
        raw_np=raw.float().cpu().numpy();actions=np.asarray(actions,np.float32)
        assert raw_np.shape==(50,32) and actions.shape==(50,7) and np.isfinite(raw_np).all() and np.isfinite(actions).all()
        if route['path']=='full':category='full_'+route['full_reason']
        else:category='fast_accept_5' if length==5 else 'fast_accept_partial'
        route.update(request_category=category,selected_skip=[] if self.skip is None else list(self.skip),mask_age_env_steps=None if self.mask_step is None else self.step-self.mask_step,
                     history_full_steps_after=[h['step'] for h in self.history],full_round_after=self.full['round'],full_step_after=self.full['step'],draft_rounds_after=self.draft_rounds,pending_gripper_after=self.pending_gripper)
        timing.setdefault('selector_total_ms',0.);timing.setdefault('selector_capture_ms',0.);timing.setdefault('selector_hot_ms',0.)
        timing['policy_wall_ms']=(time.perf_counter()-start)*1000
        result=dict(actions=actions,raw=raw_np,execute_length=length,route=route,timing=timing,arrays=arrays,counts=dict(self.counts),full_cache_age=age,
                    round=self.round,step=self.step,seed=seed,verification_seed=verification_seed(seed))
        self.round+=1;self.pending=result;return result

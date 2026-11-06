"""Selection jobs overlap Full's native AE and the acknowledged action prefix.

The first following Fast waits for the selection made at its preceding Full.
Sparse selection never skips the per-Full residual history update.
"""
import gc
import time
from collections import OrderedDict
import numpy as np
import torch
from preparations.idea1_v3.policy import verification_seed
from preparations.idea1_v3_3.rules import acceptance
from preparations.idea1_v3_4_E2_cycle3.selection import prepare_full
from preparations.idea1_v3_4_E2_cycle3_extrapolate.backend import capture,prepare_patch
from .core import Graph


class Policy:
    def __init__(self,backend):
        self.b=backend;self.graphs=OrderedDict()
        self.low=torch.cuda.Stream(priority=0);self.high=torch.cuda.Stream(priority=-1)
        self.config={};self.job=None;self.reset()

    def reset(self):
        assert self.job is None
        self.full=None;self.history=[];self.skip=None;self.mask_step=None
        self.round=self.step=self.draft_rounds=0;self.pending=None
        self.pending_gripper=False;self.last_gripper=None;self.job_sequence=0
        self.counts=dict(full=0,draft=0,verify=0,selector=0,completed_selector=0,vjp=0,kalman=0)

    def configure(self,config):
        assert self.pending is None and self.job is None
        assert config['mode'] in ('full','verified') and config['period'] in (1,3,5)
        assert config['budget']==12 and config['selection_stride'] in (1,2,3)
        assert config['projections'] in (1,2,4) and config['attention'] in ('original','split_flash')
        assert config['aggregation'] in ('layer_rms_sum','signed_sum_rms')
        if config['mode']=='verified':
            assert config['tau']==.10 and config['protect_gripper']
            assert config['compensation']=='extrapolate' and config['selection']=='jacobian'
        self.config=dict(config);self.reset();return {}

    def clear(self):
        assert self.job is None and self.pending is None
        for graph in self.graphs.values():graph.close()
        self.graphs.clear();gc.collect();torch.cuda.empty_cache();self.reset()

    def acknowledge(self,executed,quaternions):
        executed=np.asarray(executed,np.float32);quaternions=np.asarray(quaternions)
        assert self.pending is not None and 0<len(executed)<=self.pending['execute_length']
        assert quaternions.shape==(len(executed),4) and np.isfinite(quaternions).all()
        assert np.array_equal(executed,self.pending['actions'][:len(executed)])
        self.last_gripper=float(self.pending['raw'][len(executed)-1,6])
        self.step+=len(executed);self.pending=None;return {}

    def _graph(self,c,seed):
        cfg=self.config;key=(id(c['runtime']),cfg['projections'],cfg['attention'],cfg['batched'])
        cold=0.
        if key not in self.graphs:
            # A single large graph fits alongside native inference and verifier.
            # Free the previous shape before constructing a new private pool.
            for old in self.graphs.values():old.close()
            self.graphs.clear();gc.collect();torch.cuda.empty_cache()
            tick=time.perf_counter()
            graph=Graph(self.b,c,seed,cfg['projections'],cfg['attention'],cfg['batched'])
            for _ in range(3):graph.run(c,seed)
            cold=(time.perf_counter()-tick)*1000;self.graphs[key]=graph
        return self.graphs[key],cold

    def _collect(self,arrays,events,timing,reason):
        if self.job is None:return
        job=self.job;graph=job.pop('graph');tick=time.perf_counter()
        ready=graph.end.query();r=graph.collect(6,self.config['aggregation'])
        self.skip=r['mask'];self.mask_step=job['origin_step'];self.counts['completed_selector']+=1
        for name,key in (('gate_gradient','gradient'),('layer_score','layer_score'),('combination_score','combination_scores')):
            arrays[f"{name}_job{job['id']}"]=r[key]
        event=dict(job,completion_round=self.round,completion_step=self.step,reason=reason,
            ready_before_wait=ready,wait_ms=r['wait_ms'],gpu_ms=r['gpu_ms'],score_ms=r['score_ms'],
            selected_skip=list(self.skip),collect_wall_ms=(time.perf_counter()-tick)*1000)
        events.append(event);timing['selector_wait_ms']=timing.get('selector_wait_ms',0.)+r['wait_ms']
        timing['selector_collect_ms']=timing.get('selector_collect_ms',0.)+event['collect_wall_ms']
        self.job=None

    def finish(self):
        assert self.pending is None
        arrays={};events=[];timing={};self._collect(arrays,events,timing,'episode_end')
        return dict(arrays=arrays,selector_events=events,timing=timing,counts=dict(self.counts))

    def predict(self,obs,instruction,suite,seed):
        if self.pending is not None:raise RuntimeError('Unacknowledged execution')
        cfg=self.config;b=self.b;start=time.perf_counter();timing={};arrays={};events=[]
        age=None if self.full is None else dict(rounds=self.round-self.full['round'],env_steps=self.step-self.full['step'])
        route=dict(draft_attempted=False,verify_performed=False,accepted_length=None,full_reason=None,
            fallback=False,kalman=False,period=cfg['period'],budget=cfg['budget'],
            selection=cfg['selection'],compensation=cfg['compensation'],full_age_at_entry=age,
            draft_rounds_at_entry=self.draft_rounds,draft_skip=[],steady_vjp_count=0,
            selector_forward_count=0,mask_refresh=False)

        def full(reason):
            # A gripper/periodic Full can precede Fast. Complete the old job before
            # reusing its static input buffers, including after sparse refresh.
            self._collect(arrays,events,timing,'before_full')
            tick=time.perf_counter();patch=cfg['mode']=='verified'
            c,t=prepare_full(b,obs,instruction,trace=patch)
            if patch:
                residuals,ms=b.adapter.measure(lambda:capture(c.pop('hidden_trace')))
                t['residual_capture_ms']=ms
                self.history=(self.history+[dict(step=self.step,residuals=residuals)])[-2:]
            refresh=cfg['mode']=='verified' and self.counts['full']%cfg['selection_stride']==0
            if refresh:
                graph,cold=self._graph(c,seed);timing['selector_capture_ms']=cold
                launch=time.perf_counter();item=graph.stage(c,seed)
                if cfg['asynchronous']:graph.launch(item,self.low)
                self.job=dict(graph=graph,id=self.job_sequence,origin_round=self.round,
                    origin_step=self.step,actual_full_index=self.counts['full'],seed=seed)
                self.job_sequence+=1;self.counts['selector']+=1;self.counts['vjp']+=cfg['projections']
                timing['selector_launch_ms']=(time.perf_counter()-launch)*1000
                route.update(mask_refresh=True,steady_vjp_count=cfg['projections'],selector_forward_count=1)
            if cfg['asynchronous'] and refresh:
                self.high.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.high):actions,raw,cost=b.sample(c,seed,10)
            else:actions,raw,cost=b.sample(c,seed,10)
            if refresh and not cfg['asynchronous']:
                graph.launch(item);self._collect(arrays,events,timing,'synchronous_full')
            self.full=dict(c,round=self.round,step=self.step)
            self.draft_rounds=0;self.pending_gripper=False;self.counts['full']+=1
            route['actual_full_index']=self.counts['full']-1
            timing.update({f'full_{k}':v for k,v in dict(t,**cost).items()})
            timing['full_path_wall_ms']=(time.perf_counter()-tick)*1000
            route.update(path='full',full_reason=reason,full_steps=10)
            return actions,raw,5

        if cfg['mode']=='full':actions,raw,length=full('baseline')
        elif self.full is None:actions,raw,length=full('startup')
        elif self.pending_gripper:actions,raw,length=full('pending_gripper')
        elif self.draft_rounds>=cfg['period']-1:actions,raw,length=full('periodic')
        else:
            self._collect(arrays,events,timing,'before_fast')
            tick=time.perf_counter();assert self.skip is not None
            route.update(draft_attempted=True,draft_skip=list(self.skip),verify_full_round=self.full['round'],
                verify_full_step=self.full['step'],draft_steps=2,mask_origin_step=self.mask_step)
            last=self.history[-1];older=self.history[-2] if len(self.history)>1 else None
            gap=None if older is None else last['step']-older['step'];draft_age=self.step-last['step']
            ratio=0. if gap is None else draft_age/gap
            c,t=prepare_patch(b,obs,instruction,self.skip,last['residuals'],None if older is None else older['residuals'],ratio)
            route.update(draft_history_full_steps=[h['step'] for h in self.history],draft_residual_age=draft_age,
                draft_full_gap=gap,draft_extrapolation_ratio=ratio,captured_layers=list(range(1,18)))
            actions,raw,cost=b.sample(c,seed,2);self.counts['draft']+=1;self.draft_rounds+=1
            timing.update({f'draft_{k}':v for k,v in dict(t,**cost).items()})
            timing['draft_path_wall_ms']=(time.perf_counter()-tick)*1000;tick=time.perf_counter()
            hats,cost=b.verify(self.full,c,raw,seed,suite)
            assert hats.shape==(2,50,32) and np.isfinite(hats).all()
            raw_np=raw.float().cpu().numpy()
            length,detail=acceptance(raw_np,hats,cfg['tau'],5,last_gripper=self.last_gripper,protect=cfg['protect_gripper'])
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
        category='full_'+route['full_reason'] if route['path']=='full' else ('fast_accept_5' if length==5 else 'fast_accept_partial')
        route.update(request_category=category,selected_skip=[] if self.skip is None else list(self.skip),
            mask_age_env_steps=None if self.mask_step is None else self.step-self.mask_step,
            history_full_steps_after=[h['step'] for h in self.history],full_round_after=self.full['round'],
            full_step_after=self.full['step'],draft_rounds_after=self.draft_rounds,pending_gripper_after=self.pending_gripper,
            actual_fulls_after=self.counts['full'],selection_stride=cfg['selection_stride'],aggregation=cfg['aggregation'],
            projections=cfg['projections'],batched=cfg['batched'],asynchronous=cfg['asynchronous'],cuda_graph=True,
            pending_selector=None if self.job is None else {k:v for k,v in self.job.items() if k!='graph'})
        timing.setdefault('selector_capture_ms',0.);timing.setdefault('selector_wait_ms',0.)
        timing['policy_wall_ms']=(time.perf_counter()-start)*1000
        result=dict(actions=actions,raw=raw_np,execute_length=length,route=route,timing=timing,arrays=arrays,
            selector_events=events,counts=dict(self.counts),full_cache_age=age,round=self.round,step=self.step,
            seed=seed,verification_seed=verification_seed(seed))
        self.round+=1;self.pending=result;return result

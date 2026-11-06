"""Metadata successor parity on one real fixed observation, including Kalman.

This is a request/state-machine replay with acknowledged proposed prefixes,
not an environment rollout or evidence of task success.
"""
import numpy as np
from .policy import Policy as Original
from .metadata_r2 import Policy


def replay(backend,obs,instruction,suite,seed):
    old,new=Original(backend),Policy(backend)
    cfg=dict(id='metadata_parity',model='pi0',experiment='E2',alternate=True,tau=.1)
    for p in (old,new): p.configure(cfg)
    rows=[]
    for index in range(5):
        a=old.predict(obs,instruction,suite,seed+index)
        b=new.predict(obs,instruction,suite,seed+index)
        comparisons=dict(actions_equal=bool(np.array_equal(a['actions'],b['actions'])),raw_equal=bool(np.array_equal(a['raw'],b['raw'])),
            route_equal=a['route']==b['route'],length_equal=a['execute_length']==b['execute_length'],counts_equal=a['counts']==b['counts'],
            arrays_equal=all(np.array_equal(a['arrays'][k],b['arrays'][k]) for k in a['arrays']))
        for p,r in ((old,a),(new,b)):
            commands=r['actions'][:r['execute_length']]
            p.acknowledge(commands,np.tile(np.asarray(obs['robot0_eef_quat']),(len(commands),1)))
        comparisons['ack_state_equal']=old.step==new.step and old.last_kalman==new.last_kalman and np.array_equal(old.commands,new.commands)
        rows.append(dict(index=index,path=a['route']['path'],rejected_path=a['route'].get('rejected_path'),passed=all(comparisons.values()),**comparisons))
    return dict(passed=all(r['passed'] for r in rows),checks=rows,
        scope='Five fixed-real-observation requests; startup, Kalman candidate, V14 and periodic Full; proposed prefixes acknowledged identically. No new scientific episode.')


def check(backend,obs,instruction,suite,seed):
    from scripts.spec.triton import pi0_infer as kernels
    vision,vlm=kernels.vision_encoder,kernels.transformer_encoder
    original=backend.prepare_metadata
    counts=dict(vision=0,vlm=0); prepares=[]
    def visual(*args,**kw): counts['vision']+=1; return vision(*args,**kw)
    def language(*args,**kw): counts['vlm']+=1; return vlm(*args,**kw)
    def metadata(*args,**kw):
        before=dict(counts); result=original(*args,**kw)
        prepares.append({k:counts[k]-before[k] for k in counts}); return result
    kernels.vision_encoder,kernels.transformer_encoder=visual,language
    backend.prepare_metadata=metadata
    try:
        result=replay(backend,obs,instruction,suite,seed)
        result['metadata_only_prepare_calls']=prepares
        result['passed']=result['passed'] and bool(prepares) and all(not any(r.values()) for r in prepares)
        return result
    finally:
        kernels.vision_encoder,kernels.transformer_encoder=vision,vlm
        backend.prepare_metadata=original

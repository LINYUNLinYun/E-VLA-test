"""Command differences with explicit controller units and gripper direction."""
import numpy as np

METRICS = ('translation_command_l2','rotation_command_l2','continuous6_command_rmse',
           'translation_target_difference_mm','rotation_increment_vector_difference_deg',
           'full_open_variant_close','full_close_variant_open','zero_command_in_pair',
           'nonzero_gripper_sign_disagreement','translation_signed_x_mm',
           'translation_signed_y_mm','translation_signed_z_mm')


def differences(full,variant):
    a,b=np.asarray(full,dtype=np.float64),np.asarray(variant,dtype=np.float64)
    if a.shape!=(50,7) or b.shape!=(50,7) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected finite paired 50x7 commands')
    raw=b-a
    clipped=np.clip(b[:,:6],-1,1)-np.clip(a[:,:6],-1,1)
    early_close=(a[:,6]<0)&(b[:,6]>0)
    early_open=(a[:,6]>0)&(b[:,6]<0)
    return np.stack([np.linalg.norm(raw[:,:3],axis=1),np.linalg.norm(raw[:,3:6],axis=1),
       np.sqrt(np.mean(raw[:,:6]**2,axis=1)),np.linalg.norm(clipped[:,:3],axis=1)*50,
       np.linalg.norm(clipped[:,3:6],axis=1)*.5*180/np.pi,
       early_close,early_open,(a[:,6]==0)|(b[:,6]==0),early_close|early_open,
       clipped[:,0]*50,clipped[:,1]*50,clipped[:,2]*50],axis=-1)


def longest_run(flags):
    result,current=0,0
    for flag in flags:
        current=current+1 if bool(flag) else 0
        result=max(result,current)
    return result


def task_cluster_summary(values,tasks,*,strata=None,seed=20261011):
    values=np.asarray(values,dtype=np.float64)
    tasks=np.asarray(tasks)
    keys=sorted(set(tasks.tolist()))
    means=np.stack([values[tasks==key].mean(axis=0) for key in keys])
    rng=np.random.default_rng(seed)
    groups=[np.arange(len(keys))]
    if strata is not None:
        lookup={key:strata[np.flatnonzero(tasks==key)[0]] for key in keys}
        groups=[np.array([i for i,key in enumerate(keys) if lookup[key]==suite]) for suite in sorted(set(lookup.values()))]
    weights=np.zeros((2000,len(keys)),dtype=np.float64)
    for group in groups:
        draw=rng.choice(group,size=(2000,len(group)),replace=True)
        for i,row in enumerate(draw):
            weights[i]+=np.bincount(row,minlength=len(keys))
    boot=(weights/len(keys)) @ means.reshape(len(keys),-1)
    ci=np.quantile(boot,[.025,.975],axis=0).reshape((2,)+means.shape[1:])
    return means.mean(axis=0),ci,len(keys)

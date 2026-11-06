"""Pure causal routing and prefix acceptance for the frozen v3.3 protocol."""
import numpy as np


def gate(commands, quaternions, angle=20., change=.05, pose=5.):
    failures = []
    if len(commands) < 5 or len(quaternions) < 5:
        return False, ['history']
    a = np.asarray(commands[-5:], float)
    q = np.asarray(quaternions[-5:], float)
    if a.shape != (5,7) or not np.isfinite(a).all():
        return False, ['commands_invalid']
    norms = np.linalg.norm(a[:,:3], axis=1)
    turns = []
    for i in range(4):
        if norms[i] < 1e-4 and norms[i+1] < 1e-4: turns.append(0.)
        elif min(norms[i:i+2]) < 1e-4: turns.append(180.)
        else: turns.append(float(np.degrees(np.arccos(np.clip(np.dot(a[i,:3],a[i+1,:3])/(norms[i]*norms[i+1]),-1,1)))))
    if max(turns)>angle: failures.append('a_turn')
    if np.linalg.norm(np.diff(a[:,:3],axis=0),axis=1).max()>change: failures.append('b_change')
    if np.any(np.diff(a[:,6]<0)): failures.append('gripper')
    if np.linalg.norm(np.diff(a[:,3:6],axis=0),axis=1).max()>.05: failures.append('rotation_command')
    if pose is not None:
        if q.shape != (5,4) or not np.isfinite(q).all() or np.any(np.linalg.norm(q,axis=1)<1e-8): failures.append('c_invalid')
        else:
            q=q/np.linalg.norm(q,axis=1)[:,None]
            degrees=np.degrees(2*np.arccos(np.clip(np.abs(q@q.T),0,1)))
            if degrees.max()>pose: failures.append('c_pose')
    return not failures, failures


def acceptance(raw, hats, tau, horizon=5, alpha=None, last_gripper=None, protect=True):
    d=np.asarray(raw)[:horizon]; h=np.asarray(hats)[:,:len(d)]
    errors=np.linalg.norm(h[...,:6]-d[None,:,:6],axis=-1)/np.sqrt(6.)
    e=errors.max(axis=0)
    if alpha is None:
        length=int(np.cumprod(e<=tau).sum())
    else:
        length=0
        for k in range(1,len(e)+1):
            if np.all(e[:k]<=alpha*tau) and np.sum(e[:k]>tau)<=1 and np.sum(e[:k]**2)<=k*tau*tau:
                length=k
    spatial_length=length
    verify_switch=post_switch=False
    if protect and last_gripper is not None:
        previous=np.concatenate([np.full((len(h),1),last_gripper),h[:,:-1,6]],axis=1)
        verify_switch=bool(np.any((previous<0)!=(h[...,6]<0)))
        switches=np.flatnonzero(((np.r_[last_gripper,d[:-1,6]]<0)!=(d[:,6]<0))[:length])
        if len(switches): post_switch=True; length=int(switches[0])
        if verify_switch: length=0
    flags=e>tau; longest=current=0
    for flag in flags:
        current=current+1 if flag else 0; longest=max(longest,current)
    return length,dict(distance=errors.tolist(),spatial_length=spatial_length,first_exceed=int(np.flatnonzero(flags)[0]) if flags.any() else None,
        exceed_count=int(flags.sum()),longest_exceed=longest,verify_gripper=verify_switch,post_verify_gripper=post_switch)

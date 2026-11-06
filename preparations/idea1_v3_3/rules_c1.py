"""E2 c revision: world xyzw five-step pairwise orientation must be strict <."""
import numpy as np
from .rules import gate as original_gate


def pose_angle(quaternions):
    q=np.asarray(quaternions,float)
    if q.shape!=(5,4) or not np.isfinite(q).all() or np.any(np.linalg.norm(q,axis=1)<1e-8): return None
    q=q/np.linalg.norm(q,axis=1)[:,None]
    return float(np.degrees(2*np.arccos(np.clip(np.abs(q@q.T),0,1))).max())


def gate(commands,quaternions,angle=20.,change=.05,pose=5.):
    allowed,blocked=original_gate(commands,quaternions,angle,change,pose=None)
    if pose is not None and len(commands)>=5 and len(quaternions)>=5:
        observed=pose_angle(quaternions[-5:])
        if observed is None: blocked.append('c_invalid')
        elif observed>=pose: blocked.append('c_pose')
    return not blocked,blocked

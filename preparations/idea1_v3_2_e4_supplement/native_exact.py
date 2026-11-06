"""Versioned correction: native CALVIN never clips the six decoded channels.

The older pilot adapter remains immutable evidence, not an exact-action claim.
"""
import numpy as np
from .native_envs import NativeCalvinEnv


def exact_calvin_action(action):
    value=np.asarray(action,dtype=np.float64)
    if value.shape!=(7,) or not np.isfinite(value).all():
        raise ValueError('Finite native CALVIN action7 required')
    result=value.copy();result[-1]=1. if value[-1]>0 else -1.
    return result,dict(native_action=result.tolist(),raw_decoded_action=value.tolist(),
        clipping=False,outside_unit_components=np.flatnonzero(np.abs(value[:6])>1).tolist(),
        action_contract='official native relative7; decoded first6 unchanged; only gripper threshold',
        native_position_scale_m=.02,native_euler_scale_rad=.05)


class ExactNativeCalvinEnv(NativeCalvinEnv):
    def step_native(self,action,kind='relative'):
        if kind!='relative': raise ValueError('Exact evaluation accepts native relative7 only')
        native,audit=exact_calvin_action(action)
        self.obs,_,_,info=self.env.step(native)
        success=self.task in self.oracle.get_task_info_for_set(self.start_info,info,{self.task}) if hasattr(self,'task') else None
        return self.obs,success,audit

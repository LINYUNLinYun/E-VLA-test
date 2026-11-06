"""Versioned native reset correction; a NEW instance is required per condition.

Restores the real controller's stored command, not just the policy observation.
Native reset retains its original finger width/motor targets and physics step.
Never call control_gripper here: that would replace the target width's motors.
"""
import numpy as np
from .native_exact import ExactNativeCalvinEnv


class FreshResetCalvinEnv(ExactNativeCalvinEnv):
    def reset_state(self,robot_obs,scene_obs):
        target=np.asarray(robot_obs,dtype=np.float64)
        if target.shape!=(15,) or not np.isfinite(target).all() or target[-1] not in (-1,1):
            raise ValueError('Official finite robot15 target with stored gripper sign required')
        # Native robot.reset restores joints, motor width and Cartesian targets,
        # but omits this internal state. Set it before that reset/physics step.
        self.env.robot.gripper_action=int(target[-1])
        super().reset_state(target,scene_obs)
        self.obs=self.env.get_obs()
        if float(self.obs['robot_obs'][-1])!=target[-1]:
            raise RuntimeError('Actual native controller sign failed reset')
        return self.obs


def compare_observations(first,second):
    """Explicit tolerances: float state1e-12 abs, RGB bitwise, no relative tol."""
    rows={}
    for key in ('robot_obs','scene_obs','base_rgb','wrist_rgb','noise'):
        a,b=np.asarray(first[key]),np.asarray(second[key])
        if a.shape!=b.shape:raise ValueError('Reset gate shape mismatch: '+key)
        delta=float(np.max(np.abs(a.astype(float)-b.astype(float))))
        atol=0. if key in ('base_rgb','wrist_rgb','noise') else 1e-12
        rows[key]=dict(shape=list(a.shape),bitwise_equal=bool(np.array_equal(a,b)),max_abs=delta,
            atol=atol,rtol=0.,passed=bool(np.isfinite(a).all() and np.isfinite(b).all() and delta<=atol))
    return dict(passed=all(row['passed'] for row in rows.values()),comparisons=rows)

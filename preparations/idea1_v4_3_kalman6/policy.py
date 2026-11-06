"""Reuse the complete v4.1 policy, changing only the Kalman prior dimensions."""
import time
from preparations.idea1_v4_1_warmstart.policy import Policy as PreviousPolicy
from .common import read
from .rules import KalmanPrior


class Policy(PreviousPolicy):
    def __init__(self, backend):
        super().__init__(backend)
        self.thresholds=read()['phase_rule']['thresholds']

    def reset(self):
        super().reset()
        if self.config.get('initial')=='kalman':
            assert self.config.get('kalman_dimensions',6)==6
            self.kalman=KalmanPrior()

    def predict(self, *args, **kwargs):
        tick=time.perf_counter()
        result=super().predict(*args,**kwargs)
        route=result['route']
        route.update(kalman_dimensions=6 if self.kalman is not None else 0,
            kalman_rotation_observations=0 if self.kalman is None else self.kalman.rotation_filter.count,
            kalman_rotation_space='incremental_controller_command',kalman_gripper_prediction=False)
        if route['draft_attempted'] and self.kalman is not None:
            assert self.kalman.rotation_filter.count==self.step
            result['arrays']['kalman_prior_command_prefix']=self.kalman.last_forecast[:5].copy()
            result['timing']['kalman_rotation_forecast_ms']=self.kalman.rotation_forecast_ms
        result['timing']['policy_wall_ms']=(time.perf_counter()-tick)*1000
        return result

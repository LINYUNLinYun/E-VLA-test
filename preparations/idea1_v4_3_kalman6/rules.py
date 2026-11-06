"""Predict incremental controller commands; gripper remains a held value."""
import json
import time
import numpy as np
from preparations.idea1_v4_0.rules import KalmanPrior as PositionPrior
from .common import ROOT


class KalmanPrior(PositionPrior):
    def __init__(self):
        super().__init__()
        from scripts.idea1_v2_1.filter import Parameters, CommandKalman
        frozen=json.loads((ROOT/'preparations/idea1_v3_3/kalman_frozen.json').read_text())
        self.rotation_filter=CommandKalman(Parameters(**frozen['parameters']),np.eye(3),np.eye(3))
        self.last_forecast=None
        self.rotation_forecast_ms=0.

    def observe(self, command):
        action=np.asarray(command,dtype=np.float32)
        if action.shape!=(7,) or not np.isfinite(action).all():
            raise ValueError('Finite seven-dimensional executed controller command required')
        super().observe(action)
        self.rotation_filter.observe(np.clip(action[3:6],-1,1))
        assert self.rotation_filter.count==self.filter.count

    def forecast(self):
        result=super().forecast()
        tick=time.perf_counter()
        rotation,_=self.rotation_filter.forecast(50)
        self.rotation_forecast_ms=(time.perf_counter()-tick)*1000
        result[:,3:6]=np.clip(rotation,-1,1)
        self.last_forecast=result.copy()
        return result

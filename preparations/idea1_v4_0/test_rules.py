import unittest
import numpy as np
from .rules import KalmanPrior, motion_features, skip_budget


class MechanismTests(unittest.TestCase):
    def test_kalman_forecast_does_not_assimilate_future_proposals(self):
        a, b = KalmanPrior(), KalmanPrior()
        for i in range(5):
            command = np.array([.15+i*.005, .02, 0, 0, 0, 0, -1], np.float32)
            a.observe(command)
            b.observe(command)
        state = a.filter.x.copy()
        first = a.forecast()
        second = a.forecast()
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(state, a.filter.x)
        np.testing.assert_array_equal(first, b.forecast())
        self.assertEqual(a.filter.count, 5)
        self.assertEqual(first.shape, (50,7))
        np.testing.assert_array_equal(first[:,6], -np.ones(50))

    def test_phase_uses_executed_five_action_window_and_all_budgets(self):
        commands = np.tile([.2, 0, 0, 0, 0, 0, -1.], (5,1))
        poses = np.tile([0.,0.,0.,1.], (5,1))
        original = commands.copy()
        features = motion_features(commands, poses)
        self.assertEqual(features['risk'], 0.)
        np.testing.assert_array_equal(commands, original)
        thresholds = dict(q25=.25, q50=.5, q85=.85)
        for risk, count in ((.1,8),(.4,7),(.7,6),(.9,5)):
            self.assertEqual(skip_budget(dict(ready=True,gripper=False,risk=risk),thresholds)[0],count)
        self.assertEqual(skip_budget(dict(ready=True,gripper=True,risk=0.),thresholds)[0],5)
        self.assertEqual(skip_budget(dict(ready=False,risk=0.,gripper=False),thresholds)[0],6)

    def test_forecast_error_uses_actual_observed_full_gap(self):
        import torch
        from preparations.idea1_v3_4_E2_comp_error_cost.model import forecast_error
        current = {i:torch.full((2,3), 8.) for i in range(1,18)}
        last = {i:torch.full((2,3), 4.) for i in range(1,18)}
        older = {i:torch.full((2,3), 2.) for i in range(1,18)}
        error = forecast_error(current,last,older,ratio=1.5)
        torch.testing.assert_close(error,torch.ones((17,2,3)))
        torch.testing.assert_close(forecast_error(current,last,last,ratio=0.),torch.full((17,2,3),4.))


if __name__ == '__main__':
    unittest.main()

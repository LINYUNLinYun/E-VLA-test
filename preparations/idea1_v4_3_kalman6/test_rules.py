import unittest
import numpy as np
from preparations.idea1_v4_0.rules import KalmanPrior as PositionPrior
from .rules import KalmanPrior


class SixCommandTests(unittest.TestCase):
    def commands(self):
        t=np.arange(18,dtype=np.float32)
        return np.stack((.1+.01*t,-.2+.005*t,.3-.003*t,.01*t,-.02*t,.005*t,np.where(t<10,-1.,1.)),axis=1).astype(np.float32)

    def test_translation_identical_rotation_predicted_gripper_held(self):
        a,b=PositionPrior(),KalmanPrior()
        for action in self.commands():
            a.observe(action);b.observe(action)
        old,new=a.forecast(),b.forecast()
        self.assertTrue(np.array_equal(old[:,:3],new[:,:3]))
        self.assertTrue(np.array_equal(old[:,6],new[:,6]))
        self.assertGreater(float(np.max(np.abs(old[:,3:6]-new[:,3:6]))),1e-3)
        self.assertEqual(b.rotation_filter.count,18)

    def test_forecast_is_causal_and_nonmutating(self):
        prior=KalmanPrior()
        with self.assertRaises(RuntimeError):prior.forecast()
        for action in self.commands()[:5]:prior.observe(action)
        x,p=prior.rotation_filter.x.copy(),prior.rotation_filter.P.copy()
        a,b=prior.forecast(),prior.forecast()
        self.assertTrue(np.array_equal(a,b))
        self.assertTrue(np.array_equal(x,prior.rotation_filter.x))
        self.assertTrue(np.array_equal(p,prior.rotation_filter.P))
        self.assertEqual(prior.filter.count,5)

    def test_rotation_tracks_same_filter_on_rotation_channels(self):
        six=KalmanPrior()
        for command in self.commands():
            six.observe(command)
        # The rotation covariance intentionally uses identity shapes rather
        # than transferring the translation-calibrated anisotropic shape.
        from scripts.idea1_v2_1.filter import Parameters, CommandKalman
        direct=CommandKalman(six.rotation_filter.parameters,np.eye(3),np.eye(3))
        for command in self.commands():direct.observe(command[3:6])
        expected,_=direct.forecast(50)
        self.assertTrue(np.array_equal(six.forecast()[:,3:6],np.clip(expected,-1,1).astype(np.float32)))
        with self.assertRaises(ValueError):six.observe(np.zeros(6))


if __name__=='__main__':unittest.main()

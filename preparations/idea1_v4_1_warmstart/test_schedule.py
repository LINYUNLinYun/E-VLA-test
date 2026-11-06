import unittest
import numpy as np
from .schedule import mix, nodes


class ScheduleTest(unittest.TestCase):
    def test_noise_endpoint_is_native_epsilon(self):
        prior = np.full((50,32), 7., np.float32)
        expected = np.random.default_rng(123).standard_normal((50,32), dtype=np.float32)
        np.testing.assert_array_equal(mix(prior, 123, 1., 1.), expected)

    def test_initial_state_uses_matching_signal_and_noise(self):
        prior = np.full((50,32), 2., np.float32)
        epsilon = np.random.default_rng(123).standard_normal((50,32), dtype=np.float32)
        np.testing.assert_allclose(mix(prior, 123, .2, .5), .8*prior+.1*epsilon, atol=1e-6)
        np.testing.assert_array_equal(mix(prior, 123, .2, 0.), .8*prior)

    def test_intervals_integrate_only_remaining_time(self):
        for tau in (.05,.1,.2,.35,.5,.75,1.):
            for n in (1,2):
                s = nodes(tau,n)
                self.assertEqual(len(s['timesteps']),n)
                self.assertAlmostEqual(sum(s['dt']),-tau)
                self.assertEqual(s['nodes'][-1],0.)
                # Constant velocity must move tau*v, independent of NFE.
                self.assertAlmostEqual(3.+sum(s['dt'])*2.,3.-tau*2.)


if __name__ == '__main__':
    unittest.main()

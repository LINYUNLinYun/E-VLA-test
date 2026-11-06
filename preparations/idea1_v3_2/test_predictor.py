import unittest
import numpy as np
try:
    from .predictor import Predictor, Config, predict_history, candidates
except ImportError:
    from predictor import Predictor, Config, predict_history, candidates


class TestPredictors(unittest.TestCase):
    def test_future_cannot_change_prediction(self):
        rng = np.random.default_rng(13)
        a = rng.normal(size=(80, 7))*.2
        b = a.copy(); b[30:] = 1e6
        for c in candidates():
            np.testing.assert_array_equal(predict_history(a, 30, c), predict_history(b, 30, c))

    def test_reset_and_forecast_nonmutation(self):
        for family in ('WLP', 'ConstantTurn', 'VAR'):
            m = Predictor(Config(family))
            for x in np.arange(20):
                m.observe(np.array([x/30, .2, -.1]))
            before = np.asarray(m.history).copy(); p = m.forecast()
            np.testing.assert_array_equal(p, m.forecast())
            np.testing.assert_array_equal(before, m.history)
            m.reset(); np.testing.assert_array_equal(m.forecast(), np.zeros((50, 3)))

    def test_constant_and_rank_deficient_history_stable(self):
        rng = np.random.default_rng(42)
        histories = [np.zeros((40, 3)), np.ones((40, 3))*.2,
                     np.tile(np.linspace(-.8, .8, 40)[:, None], (1, 3)), rng.normal(size=(50, 3))]
        for c in candidates():
            for a in histories:
                p = predict_history(a, len(a), c)
                self.assertEqual(p.shape, (50, 3)); self.assertTrue(np.isfinite(p).all())
                self.assertTrue((np.abs(p) <= 1).all())

    def test_turn_direction(self):
        theta = np.arange(20)*.05
        a = .5*np.stack([np.cos(theta), np.sin(theta), np.zeros(20)], axis=1)
        p = predict_history(a, 20, Config('ConstantTurn', window=10, damping=1.), 3)
        expected = .5*np.stack([np.cos(np.arange(20, 23)*.05), np.sin(np.arange(20, 23)*.05), np.zeros(3)], axis=1)
        np.testing.assert_allclose(p, expected, atol=1e-10)

    def test_invalid_history(self):
        for family in ('WLP', 'ConstantTurn', 'VAR'):
            m = Predictor(Config(family))
            with self.assertRaises(ValueError):
                m.observe([np.nan, 0, 0])
            with self.assertRaises(ValueError):
                m.forecast(0)
            with self.assertRaises(ValueError):
                predict_history(np.zeros((4, 3)), 5, Config(family))

    def test_no_Rfuture_interface(self):
        # Forecast API has only executed-history, cutoff, config, horizon.
        import inspect
        self.assertEqual(list(inspect.signature(predict_history).parameters), ['actions', 'cutoff', 'config', 'horizon'])
        self.assertEqual(len(candidates()), 51)


if __name__ == '__main__':
    unittest.main()

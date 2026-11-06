import unittest
import numpy as np
try:
    from .predictor_experiment import metrics, decompose, macro, bootstrap, prefix_metrics, SUITES
except ImportError:
    from predictor_experiment import metrics, decompose, macro, bootstrap, prefix_metrics, SUITES


class TestExperimentMetrics(unittest.TestCase):
    def test_separate_vector_spaces_have_coherent_components(self):
        try:
            from .predictor_revision2 import coherent_metrics
        except ImportError:
            from predictor_revision2 import coherent_metrics
        rng = np.random.default_rng(330)
        p, f = rng.normal(size=(3, 50, 3)), rng.normal(size=(3, 50, 3))
        d = coherent_metrics(p, f)
        for prefix in ('increment', 'cumulative'):
            mask = np.isfinite(d[prefix+'_lateral_mm'])
            np.testing.assert_allclose(d[prefix+'_l2_mm'][mask]**2,
                                       d[prefix+'_along_signed_mm'][mask]**2+d[prefix+'_lateral_mm'][mask]**2,
                                       atol=1e-8)

    def test_decomposition_matches_frozen_v21(self):
        rng = np.random.default_rng(100)
        a, b = rng.normal(size=(3, 50, 3)), rng.normal(size=(3, 50, 3))
        actual = metrics(a, b)
        for i in range(3):
            expected = decompose(a[i], b[i])
            for field in ('l2_mm', 'along_mm', 'lateral_mm'):
                np.testing.assert_allclose(actual[field][i], expected[field], equal_nan=True)

    def test_real_tail_prefix_excluded(self):
        a = np.ones((2, 5, 3)); b = np.zeros_like(a)
        result = prefix_metrics(a, b, [dict(_prefix=1), dict(_prefix=4)])
        self.assertTrue(np.isnan(result['l2_mm'][0, 1:]).all())
        self.assertTrue(np.isnan(result['l2_mm'][1, 4:]).all())
        self.assertTrue(np.isfinite(result['l2_mm'][1, :4]).all())

    def test_macro_ignores_replan_imbalance(self):
        m = [dict(suite='a', task=1, init=0)]*10+[dict(suite='a', task=2, init=0)]
        self.assertEqual(macro(np.array([0.]*10+[10.]), m), 5.)

    def test_paired_bootstrap_zero_for_identity(self):
        m = [dict(suite=s, task=t, init=0) for s in SUITES for t in range(1, 10)]
        a = np.arange(36, dtype=float)
        self.assertEqual(bootstrap(a, a, m, 100), [0., 0.])


if __name__ == '__main__':
    unittest.main()

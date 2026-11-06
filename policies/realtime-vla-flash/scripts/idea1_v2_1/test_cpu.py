"""Small synthetic CPU checks. Never open source results or start experiments."""
from dataclasses import replace
import unittest
import numpy as np
from ..idea1_v2.core import Kalman as OldKalman
from .filter import (CommandKalman, Parameters, candidate_manifest, check_covariance,
                     covariance_shape, fit_shapes, predict_episode)
from .diagnostics import decompose, geometry, phase_shift_oracle, scene_annotations


class FilterTests(unittest.TestCase):
    def test_forecast_copy_and_gain_effect(self):
        p = Parameters(.01, .01)
        a, b = CommandKalman(p), CommandKalman(replace(p, g=.5))
        for value in np.linspace(0, .7, 12):
            a.observe([value, 0, 0]); b.observe([value, 0, 0])
        np.testing.assert_array_equal(a.x, b.x)
        np.testing.assert_array_equal(a.P, b.P)
        x, covariance = a.x.copy(), a.P.copy()
        mean, cov = a.forecast()
        self.assertGreater(np.max(np.abs(mean-b.forecast()[0])), .01)
        np.testing.assert_array_equal(a.x, x)
        np.testing.assert_array_equal(a.P, covariance)
        for c in cov:
            check_covariance(c)

    def test_no_future_leakage(self):
        history = np.zeros((10, 7)); history[:4, 0] = np.arange(4)*.1
        records = [dict(environment_action_step=4, _actions=history,
                        full_actions=np.ones((50, 7)), semantic_phase='transport')]
        before = list(predict_episode(records, Parameters(.1, .1), np.eye(3), np.eye(3)))[0][1]
        history[4:] = 999
        records[0]['full_actions'] *= -100
        records[0]['semantic_phase'] = 'future'
        after = list(predict_episode(records, Parameters(.1, .1), np.eye(3), np.eye(3)))[0][1]
        np.testing.assert_array_equal(before, after)

    def test_old_dt_unit_invariance(self):
        actions = np.random.default_rng(7).normal(size=(12, 7))
        results = []
        for dt in (.01, .05, .25):
            k = OldKalman(.01, .0001, dt)
            for a in actions:
                k.observe(a)
            results.append(k.forecast()[0])
        for other in results[1:]:
            np.testing.assert_allclose(results[0], other, rtol=1e-10, atol=1e-10)

    def test_shape_covariance_and_invalid_rejection(self):
        differences = np.tile([1., 2., 3.], (10, 1))
        for shrink in (0., .5, 1.):
            shape, meta = covariance_shape(differences, shrink)
            check_covariance(shape)
            self.assertAlmostEqual(np.trace(shape), 3)
            self.assertGreater(meta['eigenvalues'][0], 0)
        for matrix in (np.diag([-1., 1., 1.]), np.diag([1e-20, 1., 1.]), np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                check_covariance(matrix)

    def test_fold_shapes_ignore_held_out_data(self):
        rng = np.random.default_rng(3)
        groups = {('suite', 0, i): [{'_actions': rng.normal(size=(20, 7))}] for i in range(5)}
        train = {k: v for k, v in groups.items() if k[2] != 2}
        a = fit_shapes(train, 'qr_half')
        groups[('suite', 0, 2)][0]['_actions'][:] = 1e9
        b = fit_shapes(train, 'qr_half')
        np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
        self.assertNotIn(['suite', 0, 2], a[2]['training_episodes'])
        with self.assertRaises(ValueError):
            fit_shapes({('suite', 1, 0): groups[('suite', 0, 0)]}, 'qr_half')

    def test_candidate_bound(self):
        manifest = candidate_manifest()
        self.assertEqual(len(manifest['stage1']), 49)
        self.assertEqual(49+3*len(manifest['stage2']['g'])*len(manifest['stage2']['beta'])*len(manifest['stage2']['shapes']), 211)


class DiagnosticTests(unittest.TestCase):
    def test_known_geometry(self):
        full = np.tile([.1, 0, 0], (50, 1))
        predicted = full+np.array([.1, .2, 0])
        result = decompose(predicted, full)
        np.testing.assert_allclose(result['along_mm'], np.arange(1, 51)*5)
        np.testing.assert_allclose(result['lateral_mm'], np.arange(1, 51)*10)
        stopped = decompose(full, np.zeros_like(full))
        self.assertTrue(np.isnan(stopped['along_mm']).all())
        self.assertTrue(np.isnan(stopped['cosine']).all())
        reversal = full.copy(); reversal[5] *= -1
        self.assertFalse(decompose(full, reversal)['tangent_valid'][5])

    def test_time_shift_equal_overlap_and_ambiguity(self):
        full = np.tile([.2, 0, 0], (50, 1))
        constant = phase_shift_oracle(full, full)
        self.assertTrue(constant['unidentifiable'])
        self.assertEqual(constant['reference_offsets'], [4, 5, 6, 7, 8, 9])
        ramp = np.zeros((50, 3)); ramp[:, 0] = np.arange(50)*.01
        shifted = ramp.copy(); shifted[1:] = ramp[:-1]
        result = phase_shift_oracle(shifted, ramp)
        self.assertEqual(result['best_shift'], 1)
        self.assertAlmostEqual(result['residual_mm'], 0)
        self.assertFalse(result['unidentifiable'])

    def test_eef_history_and_posthoc_transition(self):
        telemetry = [dict(eef_pos=[i*.005, 0, 0], semantic_phase='approach' if i < 8 else 'transport') for i in range(15)]
        self.assertEqual(geometry(telemetry, 4)['group'], 'cold_start')
        self.assertEqual(geometry(telemetry, 5)['group'], 'near_straight')
        self.assertEqual(geometry([dict(eef_pos=[0, 0, 0])]*6, 5)['group'], 'low_motion')
        before = geometry(telemetry, 5)
        telemetry[6]['eef_pos'] = [999, 999, 999]
        self.assertEqual(geometry(telemetry, 5), before)
        record = dict(_telemetry=telemetry, environment_action_step=5, semantic_phase='approach')
        annotations = scene_annotations(record)
        self.assertTrue(annotations['transition'])
        self.assertTrue(annotations['transition_is_posthoc'])


class ReportingIntegrationTests(unittest.TestCase):
    def test_short_calibration_forecast_matches_full_prefix_score(self):
        from .pipeline import candidate_score
        actions = np.tile([.2, .1, .05, 0, 0, 0, 0], (20, 1))
        def records():
            return [dict(environment_action_step=t, _actions=actions, _prefix=5,
                         full_actions=np.tile([.2, .1, .05, 0, 0, 0, 0], (50, 1))) for t in (0, 5, 10)]
        training = {('suite', 0, i): records() for i in range(4)}
        score, shapes = candidate_score(training, {('suite', 0, 4): records()}, Parameters(.01, .1))
        self.assertEqual(score['points'], 3)
        self.assertTrue(np.isfinite(score['prefix_equal_mean_mm']))
        from .reporting import cluster_interval
        metadata = [dict(suite='suite', task=t, init=i) for t in range(2) for i in range(2)]
        self.assertEqual(cluster_interval(np.ones(4), metadata), [1., 1.])


if __name__ == '__main__':
    unittest.main()

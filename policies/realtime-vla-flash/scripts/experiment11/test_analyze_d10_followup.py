from __future__ import annotations

import unittest

from scripts.experiment11.analyze_d10_followup import _metric_rows, stratified_task_ci
from scripts.experiment11.protocol import SUITES


class D10AnalysisTest(unittest.TestCase):
    @staticmethod
    def _episode_map(successes_per_task: int):
        return {
            (task, init): {'success': init < successes_per_task}
            for task in range(10) for init in range(5)
        }

    def test_denoising_interaction_is_variant_delta_minus_full_delta(self):
        variant_d10 = {suite: self._episode_map(3) for suite in SUITES}
        variant_d3 = {suite: self._episode_map(2) for suite in SUITES}
        full_d10 = {suite: self._episode_map(5) for suite in SUITES}
        full_d3 = {suite: self._episode_map(0) for suite in SUITES}
        rows = _metric_rows('V16_A02', 'V16_A02_D10', variant_d10,
                            'V16_A02_D3', variant_d3, full_d10, full_d3)
        overall = next(row for row in rows if row['suite'] == 'ALL_EQUAL_SUITE')
        self.assertEqual(overall['d10_successes'], 120)
        self.assertEqual(overall['d3_successes'], 80)
        self.assertAlmostEqual(overall['d10_minus_d3_success_rate'], 0.2)
        self.assertAlmostEqual(overall['interaction_vs_full'], -0.8)
        self.assertEqual(overall['task_clusters'], 40)

    def test_bootstrap_is_stratified_across_suites(self):
        low, high = stratified_task_ci({'object': [0.0, 0.0], 'spatial': [1.0, 1.0]}, seed=7)
        self.assertAlmostEqual(low, 0.5)
        self.assertAlmostEqual(high, 0.5)


if __name__ == '__main__':
    unittest.main()

"""CPU-only unit checks for Experiment 11 post-hoc analysis helpers."""
from __future__ import annotations

import unittest

from scripts.experiment11.analyze_closed_loop import (
    condition_rows,
    outcome_label,
    paired_summary,
    telemetry_metrics,
)


class Experiment11AnalysisTest(unittest.TestCase):
    def test_frozen_matrix_mapping(self):
        conditions, candidates = condition_rows()
        self.assertEqual(len(conditions), 31)
        self.assertEqual(len(candidates), 29)

    def test_outcome_cohorts_are_paired(self):
        self.assertEqual(outcome_label(True, False), 'baseline_only_success')
        self.assertEqual(outcome_label(False, True), 'variant_only_success')
        self.assertEqual(outcome_label(True, True), 'both_success')

    def test_task_clustered_pair_difference(self):
        baseline = {'suite': {
            (0, 0): {'success': True}, (0, 1): {'success': False},
            (1, 0): {'success': False}, (1, 1): {'success': False},
        }}
        variant = {'suite': {
            (0, 0): {'success': True}, (0, 1): {'success': True},
            (1, 0): {'success': False}, (1, 1): {'success': True},
        }}
        result = paired_summary(baseline, variant, ('suite',), 0)
        self.assertEqual(result['episodes'], 4)
        self.assertAlmostEqual(result['variant_minus_baseline_success_rate'], 0.5)
        self.assertEqual(result['baseline_only_success'], 0)
        self.assertEqual(result['variant_only_success'], 2)

    def test_persistent_deviation_is_consecutive_and_contact_relative(self):
        base_rows = []
        variant_rows = []
        for step in range(6):
            base_rows.append({
                'step': step, 'eef_pos': [0.0, 0.0, 0.0],
                'semantic_phase': 'approach' if step < 4 else 'contact',
                'robot_contacts': {'object': step >= 4}, 'primary_objects': ['object'],
            })
            delta = 0.03 if step in (1, 2, 3) else 0.0
            variant_rows.append({'step': step, 'eef_pos': [delta, 0.0, 0.0]})
        result = telemetry_metrics(base_rows, variant_rows)
        self.assertEqual(result['first_persistent_2cm_deviation_step'], 1)
        self.assertEqual(result['baseline_first_primary_object_contact_step'], 4)
        self.assertTrue(result['divergence_before_baseline_contact'])
        self.assertEqual(result['phase_at_first_persistent_deviation'], 'approach')

    def test_two_out_of_three_above_threshold_do_not_count_as_persistent(self):
        base_rows = [{'step': step, 'eef_pos': [0.0, 0.0, 0.0],
                      'robot_contacts': {}, 'primary_objects': []} for step in range(4)]
        variant_rows = [{'step': 0, 'eef_pos': [0.03, 0.0, 0.0]},
                        {'step': 1, 'eef_pos': [0.03, 0.0, 0.0]},
                        {'step': 2, 'eef_pos': [0.0, 0.0, 0.0]},
                        {'step': 3, 'eef_pos': [0.0, 0.0, 0.0]}]
        result = telemetry_metrics(base_rows, variant_rows)
        self.assertIsNone(result['first_persistent_2cm_deviation_step'])
        self.assertIsNone(result['divergence_before_baseline_contact'])


if __name__ == '__main__':
    unittest.main()

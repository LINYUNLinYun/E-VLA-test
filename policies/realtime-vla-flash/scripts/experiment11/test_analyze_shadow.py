from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from scripts.experiment11.analyze_shadow import (
    SUITES, K_THRESHOLDS, action_differences, add_prefix_metrics, consistency_k,
    paired_phase_contrasts, prefix_metrics,
    render_figures, render_k_figure,
)
from collections import defaultdict


class ShadowAnalysisTest(unittest.TestCase):
    def test_phase_contrasts_pair_within_task_and_keep_cluster_unit(self):
        samples = {}
        for task, approach, fine in (('task0', 1.0, 3.0), ('task1', 2.0, 5.0),
                                     ('task2', 0.5, 1.5)):
            approach_key = ('V16_A02', 'ALL', 'approach', 1)
            fine_key = ('V16_A02', 'ALL', 'alignment_release', 1)
            samples.setdefault(approach_key, {'translation_command_l2': {}})
            samples.setdefault(fine_key, {'translation_command_l2': {}})
            samples[approach_key]['translation_command_l2'][task] = [approach]
            samples[fine_key]['translation_command_l2'][task] = [fine]
        rows = paired_phase_contrasts(samples)
        result = next(row for row in rows if row['variant_id'] == 'V16_A02' and
                      row['suite'] == 'ALL' and row['phase_a'] == 'approach' and
                      row['phase_b'] == 'alignment_release' and
                      row['metric'] == 'translation_command_l2')
        self.assertEqual(result['n_paired_tasks'], 3)
        self.assertAlmostEqual(result['task_equal_delta_mean'], 2.0)
        self.assertEqual(result['bootstrap_unit'], 'suite::task; all episodes/inits for a task kept together')

    def test_prefix_summary_keeps_overall_and_replan_phase_cells(self):
        samples = defaultdict(lambda: defaultdict(list))
        task_samples = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        values = {'translation_command_l2_mean': 0.2, 'gripper_disagreement_rate': 0.1}
        add_prefix_metrics(samples, task_samples, 'V16_A02', 'libero_object',
                           'approach', 5, 'libero_object::task00', values)
        self.assertIn(('V16_A02', 'ALL', 'all', 5), samples)
        self.assertIn(('V16_A02', 'ALL', 'approach', 5), samples)
        self.assertEqual(samples[('V16_A02', 'ALL', 'all', 5)]['translation_command_l2_mean'], [0.2])
        self.assertEqual(task_samples[('V16_A02', 'ALL', 'all', 5)]
                         ['translation_command_l2_mean']['libero_object::task00'], [0.2])

    def test_identical_predictions_have_zero_error_and_full_prefix(self):
        actions = np.zeros((50, 7), dtype=np.float32)
        diff = action_differences(actions, actions.copy())
        self.assertEqual(float(np.max(diff['translation_command_l2'])), 0.0)
        self.assertEqual(float(np.max(diff['rotation_command_l2'])), 0.0)
        self.assertFalse(bool(np.any(diff['gripper_disagreement'])))
        self.assertEqual(consistency_k(diff, 0.01, 0.01), 50)

    def test_contiguous_prefix_stops_at_first_bad_action(self):
        full = np.zeros((50, 7), dtype=np.float32)
        variant = full.copy()
        variant[2, 0] = 1.0
        diff = action_differences(full, variant)
        self.assertEqual(consistency_k(diff, 0.1, 0.1), 2)

    def test_prefix_rmse_uses_six_continuous_action_components(self):
        full = np.zeros((50, 7), dtype=np.float32)
        variant = full.copy()
        variant[0, 0] = 1.0
        diff = action_differences(full, variant)
        result = prefix_metrics(diff, 1)
        self.assertAlmostEqual(result['continuous6_component_rmse'], np.sqrt(1.0 / 6.0), places=6)

    def test_gripper_disagreement_uses_zero_threshold_sign(self):
        full = np.zeros((50, 7), dtype=np.float32)
        variant = full.copy()
        variant[0, 6] = -0.1
        diff = action_differences(full, variant)
        self.assertTrue(bool(diff['gripper_disagreement'][0]))
        self.assertEqual(consistency_k(diff, 1.0, 1.0), 0)

    def test_shadow_visualizers_render_without_a_variant_xyz_trace(self):
        variant = 'V13_Custom'
        offset_rows = []
        stage_rows = []
        k_rows = []
        for suite in SUITES:
            for metric in ('translation_command_l2', 'rotation_command_l2'):
                offset_rows.append({'variant_id': variant, 'suite': suite, 'replan_phase': 'all',
                                    'offset': 1, 'metric': metric,
                                    'sample_weighted_mean': 0.01, 'sample_p95': 0.02})
            for metric in ('translation_command_l2', 'rotation_command_l2'):
                stage_rows.append({'variant_id': variant, 'suite': suite,
                                   'actual_full_phase': 'approach', 'offset': 1,
                                   'metric': metric, 'sample_weighted_mean': 0.01})
            for horizon in K_THRESHOLDS:
                k_rows.append({'variant_id': variant, 'suite': suite, 'replan_phase': 'all',
                               'translation_scale_factor': 0.05, 'rotation_scale_factor': 0.05,
                               'minimum_prefix_k': horizon, 'task_equal_probability': 0.5})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            render_figures(offset_rows, stage_rows, [variant], output)
            render_k_figure(k_rows, [variant], output)
            self.assertEqual(len(list(output.glob('offset50_*.png'))), 4)
            self.assertTrue((output / f'prefix5_actual_phase_{variant}.png').is_file())
            self.assertTrue((output / 'consistent_prefix_k_5pct.png').is_file())


if __name__ == '__main__':
    unittest.main()

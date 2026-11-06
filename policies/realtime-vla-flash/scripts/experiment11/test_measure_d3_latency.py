from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from scripts.experiment11 import measure_d3_latency as latency


class D3LatencyHelpersTest(unittest.TestCase):
    def test_attempt_id_is_safe_and_bounded(self):
        self.assertEqual(latency.validate_attempt_id('retry1_process_isolated'), 'retry1_process_isolated')
        for value in ('', '.', '..', '../overwrite', 'bad/name', 'x' * 65):
            with self.subTest(value=value), self.assertRaises(ValueError):
                latency.validate_attempt_id(value)

    def test_condition_sample_validation_requires_200_unique_keys(self):
        rows = [{'condition': 'Full18_D3_R5_FORMAL', 'suite': latency.SUITES[index // 50],
                 'task_id': (index % 50) // 5, 'init_index': index % 5}
                for index in range(latency.SAMPLE_PER_CONDITION)]
        latency.validate_condition_samples('Full18_D3_R5_FORMAL', rows)
        with self.assertRaisesRegex(RuntimeError, 'expected 200'):
            latency.validate_condition_samples('Full18_D3_R5_FORMAL', rows[:-1])
        duplicate = rows.copy()
        duplicate[-1] = dict(duplicate[0])
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            latency.validate_condition_samples('Full18_D3_R5_FORMAL', duplicate)

    def test_suite_shard_requires_exact_task_and_init_coverage(self):
        rows = [{'condition': 'Full18_D3_R5_FORMAL', 'suite': 'libero_spatial',
                 'task_id': index // 5, 'init_index': index % 5}
                for index in range(50)]
        latency.validate_condition_samples('Full18_D3_R5_FORMAL', rows,
                                           expected_count=50, expected_suite='libero_spatial')
        rows[-1] = dict(rows[0])
        with self.assertRaisesRegex(RuntimeError, 'duplicate'):
            latency.validate_condition_samples('Full18_D3_R5_FORMAL', rows,
                                               expected_count=50, expected_suite='libero_spatial')

    def test_aggregate_writer_uses_condition_rows_and_emits_complete_summary(self):
        conditions = [
            {'name': latency.BASELINE, 'skip': [], 'steps': 3, 'replan_steps': 5},
            {'name': 'V16_A02_D3_R5_FORMAL', 'skip': [4, 10], 'steps': 3, 'replan_steps': 5},
        ]
        rows = []
        for condition, variant_id, duration in ((conditions[0], 'Full18', 4.0),
                                                 (conditions[1], 'V16_A02', 3.0)):
            for index in range(latency.SAMPLE_PER_CONDITION):
                rows.append({
                    'condition': condition['name'], 'suite': latency.SUITES[index // 50],
                    'task_id': (index % 50) // 5, 'init_index': index % 5,
                    'variant_id': variant_id, 'cuda_policy_ms': duration,
                    'backend_infer_wall_ms': duration + 1.0,
                })
        with tempfile.TemporaryDirectory() as tmp:
            sample_path, aggregate_path = latency.write_aggregates(
                rows, conditions, 1, Path(tmp), 'unit_test',
                [condition['name'] for condition in conditions], 1.5, 'checkpoint-hash')
            with aggregate_path.open(newline='', encoding='utf-8') as stream:
                aggregate_rows = list(csv.DictReader(stream))
            self.assertEqual(sample_path.name, 'isolated_d3_samples.csv')
            self.assertEqual([row['variant_id'] for row in aggregate_rows], ['Full18', 'V16_A02'])
            self.assertEqual([int(row['samples']) for row in aggregate_rows], [200, 200])
            self.assertAlmostEqual(float(aggregate_rows[1]['speedup_vs_full_p50']), 4.0 / 3.0)


if __name__ == '__main__':
    unittest.main()

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.experiment10.common import dump_json, sha256
from scripts.experiment10 import common
from scripts.experiment11 import launch_shadow


class ShadowSchedulerTest(unittest.TestCase):
    def test_managed_gpus_tolerates_null_status_worker_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dump_json(root / 'experiment11_shadow_status.json', {
                'active': None,
                'workers': None,
            })
            with patch.object(common, 'RESULTS', root):
                self.assertEqual(common.managed_gpus(), set())

    def test_managed_gpus_still_counts_live_records_with_null_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dump_json(root / 'experiment11_shadow_status.json', {
                'active': [{'pid': os.getpid(), 'gpu': 3}],
                'workers': None,
            })
            with patch.object(common, 'RESULTS', root):
                self.assertEqual(common.managed_gpus(), {3})

    def test_managed_gpus_accepts_single_active_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dump_json(root / 'experiment11_latency_status.json', {
                'active': {'pid': os.getpid(), 'gpu': 1},
            })
            with patch.object(common, 'RESULTS', root):
                self.assertEqual(common.managed_gpus(), {1})

    def test_live_pid_uses_proc_state(self):
        self.assertTrue(launch_shadow.live_pid(os.getpid()))
        self.assertFalse(launch_shadow.live_pid(2**30))

    def test_validated_job_requires_full_only_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selection = root / 'protocol/selected_shadow_manifest.json'
            candidate = root / 'protocol/candidate_manifest.json'
            full_log = root / 'closed_loop/Full18_D3_R5_FORMAL/libero_object/run/episode_log.json'
            for path in (selection, candidate):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}\n')
            full_log.parent.mkdir(parents=True, exist_ok=True)
            dump_json(full_log, [{'infer_record_count': 1} for _ in range(50)])
            predictions = root / 'predictions.jsonl'
            marker_path = root / 'validated.json'
            partial = root / 'predictions.jsonl.partial'
            predictions.write_text('{"probe": true}\n')
            job = {'variant_id': 'V13_Custom', 'suite': 'libero_object',
                   'skip_layers': [3, 4, 7, 12, 15],
                   'condition_name': 'V13_Custom_D3_R5_FORMAL'}
            marker = {
                'complete': True, 'variant_id': job['variant_id'], 'suite': job['suite'],
                'skip_layers': job['skip_layers'], 'condition_name': job['condition_name'],
                'episode_count': 50, 'replan_prediction_count': 50,
                'action_horizon': 50, 'denoise_steps': 3, 'replan_execute_prefix': 5,
                'environment_steps_executed_by_variant': 0,
                'full_recomputed_parity_max_abs_error': 1e-7,
                'source_hashes': {
                    'full_episode_log_sha256': sha256(full_log),
                    'selected_shadow_manifest_sha256': sha256(selection),
                    'candidate_manifest_sha256': sha256(candidate),
                },
                'predictions_sha256': sha256(predictions),
            }
            dump_json(marker_path, marker)
            with patch.object(launch_shadow, 'RESULTS', root), \
                    patch.object(launch_shadow, 'output_paths', return_value=(predictions, marker_path, partial)):
                self.assertTrue(launch_shadow.validate_job(job))
                marker['environment_steps_executed_by_variant'] = 1
                dump_json(marker_path, marker)
                self.assertFalse(launch_shadow.validate_job(job))


if __name__ == '__main__':
    unittest.main()

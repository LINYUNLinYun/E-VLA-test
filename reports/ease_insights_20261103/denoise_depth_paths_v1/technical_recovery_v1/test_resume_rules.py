"""Exercise technical exclusions and preservation using temporary records only."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from reports.ease_insights_20261103.denoise_depth_paths_v1.technical_recovery_v1 import resume_rules as rules


class ResumeRulesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.failed = self.root / 'failed'
        self.failed.mkdir()
        self.manifest = dict(manifest_hash='science')
        for name in ('identity.json', 'protocol.json', 'fixed_input_cost.json'):
            self.write(self.failed / name, self.manifest)
        self.write(self.failed / 'gate.json', dict(self.manifest, passed=True, checks=[dict(passed=True)]))
        self.write(self.failed / 'cross_service.json', dict(passed=True))
        self.diagnostic = self.root / 'diagnostic.json'
        self.write(self.diagnostic, dict(status='pass', actual_input_hash='input', expected_input_hash='input', VLA_inference_performed=False))
        self.episode = self.root / 'episode'
        self.episode.mkdir()
        (self.episode / 'raw.bin').write_bytes(b'original scientific record')
        self.receipt = dict(valid=True, condition='ae12_d4', manifest_hash='science', case=dict(suite='suite', task=3, init=45),
                            success=False, worker_pid=11, artifacts={'raw.bin': rules.sha(self.episode / 'raw.bin')})
        self.write(self.episode / 'episode_receipt.json', self.receipt)
        (self.failed / 'worker.log').write_text(json.dumps(self.receipt) + '\n')
        self.value = dict(scientific_manifest_hash='science', valid_outcomes_to_repeat=0,
                          failure=dict(VLA_case_prediction_performed=False),
                          fresh_environment_diagnostic=dict(path=str(self.diagnostic), sha256=rules.sha(self.diagnostic)),
                          failed_service=dict(directory=str(self.failed), complete_service_receipt_present=False,
                                              artifacts={p.name: rules.sha(p) for p in self.failed.iterdir()}),
                          accepted_cases_before_resume=[dict(condition='ae12_d4', case=self.receipt['case'],
                              artifacts=self.receipt['artifacts'], receipt=dict(path=str(self.episode / 'episode_receipt.json'),
                              sha256=rules.sha(self.episode / 'episode_receipt.json')))])
        patcher = patch.object(rules, 'plan', return_value=self.value)
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value) + '\n')

    def test_only_recorded_incomplete_service_is_recognized(self):
        self.assertTrue(rules.incomplete_service(self.failed, self.manifest))
        self.assertFalse(rules.incomplete_service(self.root / 'another', self.manifest))
        with self.assertRaises(ValueError):
            rules.incomplete_service(self.failed, dict(manifest_hash='other science'))

    def test_completed_or_changed_service_cannot_be_excluded(self):
        self.write(self.failed / 'result.json', dict(complete=True))
        with self.assertRaises(ValueError):
            rules.incomplete_service(self.failed, self.manifest)
        (self.failed / 'result.json').unlink()
        self.write(self.failed / 'gate.json', dict(passed=False))
        with self.assertRaises(ValueError):
            rules.incomplete_service(self.failed, self.manifest)

    def test_exclusion_set_and_complete_service_count_are_exact(self):
        costs = dict(valid_services=8, excluded_or_pending=[dict(path=str(self.failed / 'fixed_input_cost.json'))])
        self.assertEqual(len(rules.allowed_cost_exclusions(costs, self.manifest)), 1)
        for altered in (dict(costs, valid_services=7), dict(costs, excluded_or_pending=[]),
                        dict(costs, excluded_or_pending=costs['excluded_or_pending'] + [dict(path='unexpected')])):
            with self.assertRaises(ValueError):
                rules.allowed_cost_exclusions(altered, self.manifest)

    def test_administrative_writer_change_preserves_outcomes(self):
        self.write(self.episode / 'episode_receipt.json', dict(self.receipt, worker_pid=22))
        self.write(self.episode / 'recovery_receipt.json', dict(recovered=True))
        value = rules.verify_accepted_science_preserved()
        self.assertEqual(len(value['administrative_receipt_changes']), 1)
        self.write(self.episode / 'episode_receipt.json', dict(self.receipt, worker_pid=22, success=True))
        with self.assertRaises(ValueError):
            rules.verify_accepted_science_preserved()

    def test_raw_scientific_changes_are_rejected(self):
        self.assertTrue(rules.verify_accepted_science_preserved()['all_raw_scientific_artifacts_unchanged'])
        (self.episode / 'raw.bin').write_bytes(b'altered outcome')
        with self.assertRaises(ValueError):
            rules.verify_accepted_science_preserved()

    def test_changed_diagnostic_is_rejected(self):
        self.write(self.diagnostic, dict(status='pass', actual_input_hash='input', expected_input_hash='input', VLA_inference_performed=True))
        with self.assertRaises(ValueError):
            rules.incomplete_service(self.failed, self.manifest)


if __name__ == '__main__':
    unittest.main()

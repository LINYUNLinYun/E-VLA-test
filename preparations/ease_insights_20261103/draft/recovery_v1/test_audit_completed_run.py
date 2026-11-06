"""CPU regression checks for preserving scientific evidence during reconciliation."""
import json
from pathlib import Path
import tempfile
import unittest

from audit_completed_run import audit


class RecoveryAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'original'
        self.case = dict(suite='libero_object', task_id=0, init_id=20)
        self.directory = self.run / 'episodes/libero_object_t00_i20'
        self.native = dict(task_suite_name='libero_object', task_id=0, episode_idx=20,
            failure_reason=None, success=False, executed_action_count=280,
            trajectory_length=280, max_steps=280, infer_calls=1)
        self.receipt = dict(self.case, mode='draft', source_suite='libero_object',
            status='complete', success=False, rpc_calls=1, episode=self.native)
        self.write(self.root / 'frozen.json', dict(source_sha256={}))
        self.write(self.run / 'manifest.json', dict(cases=[self.case], mode='draft', source_suite='libero_object'))
        self.write(self.run / 'cases.json', [self.case])
        self.write(self.run / 'result.json', dict(status='complete', cases=1, total_cases=1, successes=0))
        self.write(self.run / 'client_complete.json', dict(status='complete', cases=1, successes=0))
        self.write(self.directory / 'result.json', self.receipt)
        self.write(self.directory / 'native/episode_log.json', [self.native])
        self.request = dict(self.case, replan_id=0, mode='draft', source_suite='libero_object',
            route_calls=dict(full=0, draft=1, verify=0))
        (self.run / 'requests.jsonl').write_text(json.dumps(self.request) + '\n')

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def check(self):
        return audit(self.run, self.root / 'frozen.json')

    def test_complete_scientific_failure_is_authoritative(self):
        result = self.check()
        self.assertEqual(result['status'], 'complete_original_verified')
        self.assertEqual(result['successes'], 0)
        self.assertEqual(result['next_action'], 'reuse_original_run_receipt_without_gpu')

    def test_native_completion_without_outer_receipt_must_not_rerun(self):
        (self.directory / 'result.json').unlink()
        result = self.check()
        self.assertEqual(result['cases'][0]['classification'], 'native_complete_missing_receipt_do_not_rerun')
        self.assertTrue(result['cases'][0]['earliest_native_scientific_result_preserved'])
        self.assertNotEqual(result['status'], 'complete_original_verified')

    def test_partial_request_tail_blocks_completion(self):
        with (self.run / 'requests.jsonl').open('a') as stream:
            stream.write('{"suite":')
        result = self.check()
        self.assertIn('malformed_request_records', result['issues'])
        self.assertIn('unterminated_request_tail', result['issues'])
        self.assertEqual(result['requests']['malformed'][0]['line'], 2)

    def test_extra_request_cannot_be_merged_into_complete_case(self):
        with (self.run / 'requests.jsonl').open('a') as stream:
            stream.write(json.dumps(self.request) + '\n')
        result = self.check()
        self.assertIn('request_ids_not_contiguous', result['cases'][0]['issues'])
        self.assertIn('native_request_count_mismatch', result['cases'][0]['issues'])
        self.assertNotEqual(result['status'], 'complete_original_verified')

    def test_incomplete_native_is_technical_even_if_receipt_claims_complete(self):
        native = dict(self.native, executed_action_count=12, trajectory_length=12)
        self.write(self.directory / 'native/episode_log.json', [native])
        self.write(self.directory / 'result.json', dict(self.receipt, episode=native))
        result = self.check()
        self.assertFalse(result['cases'][0]['earliest_native_scientific_result_preserved'])
        self.assertNotEqual(result['status'], 'complete_original_verified')


if __name__ == '__main__':
    unittest.main()

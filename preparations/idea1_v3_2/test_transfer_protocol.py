import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from .transfer_protocol import MODES, cases, key, digest, load_manifest, complete_result, freeze
from .transfer_report import summarize
from .transfer_run import namespace_ownership


class TestTransferProtocol(unittest.TestCase):
    def test_nvml_host_pid_difference_is_not_foreign_by_itself(self):
        import os
        from .transfer_run import process_identity
        class Current:
            pid = os.getpid()
            @staticmethod
            def poll(): return None
        before = dict(uuid='GPU-test', compute_pids=[])
        after = dict(uuid='GPU-test', compute_pids=[999999999])
        proof = namespace_ownership(Current(), process_identity(os.getpid()), before, after, 'GPU-test')
        self.assertIn('sole_new_NVML', proof['method'])
        with self.assertRaises(RuntimeError):
            namespace_ownership(Current(), process_identity(os.getpid()), before,
                                dict(uuid='GPU-test', compute_pids=[99999, 88888]), 'GPU-test')

    def test_exact_preregistered_cases(self):
        rows = cases()
        self.assertEqual(len(rows), 18)
        self.assertEqual(len({key(r['environment'], r['case'], m) for r in rows for m in MODES}), 108)
        self.assertEqual(len(MODES), 6)
        with self.assertRaises(ValueError):
            key('calvin', 8, 'full_d10')
        with self.assertRaises(ValueError):
            key('pusht', 0, 'best_draft')

    def test_manifest_tamper_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'manifest.json'
            m = dict(modes=list(MODES), source_files={}, receipt_files={})
            m['manifest_hash'] = digest(m); p.write_text(json.dumps(m))
            self.assertEqual(load_manifest(p)['modes'], list(MODES))
            m['modes'] = ['best_draft']; p.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                load_manifest(p)

    def test_only_matching_complete_is_resumable(self):
        m = dict(manifest_hash='a'); c = dict(environment='calvin', case=0)
        result = dict(status='complete', success=False, environment='calvin', case=0, mode='full_d10', manifest_hash='a', gpu_uuid='GPU-1')
        self.assertTrue(complete_result(result, m, c, 'full_d10', 'GPU-1'))
        for field, value in [('status', 'technical_failure'), ('success', None), ('manifest_hash', 'b'), ('gpu_uuid', 'GPU-2')]:
            changed = {**result, field: value}
            self.assertFalse(complete_result(changed, m, c, 'full_d10', 'GPU-1'))

    def test_technical_failures_not_policy_zero(self):
        manifest = dict(environment_receipt=dict(environments={n: dict(status='supported') for n in ('calvin', 'simpler', 'pusht')}))
        records = [dict(environment='calvin', case=0, mode='full_d10', status='complete', success=False),
                   dict(environment='calvin', case=1, mode='full_d10', status='technical_failure', success=None)]
        summary, pairs = summarize(manifest, records)
        r = next(r for r in summary if r['environment'] == 'calvin' and r['mode'] == 'full_d10')
        self.assertEqual(r['complete'], 1); self.assertEqual(r['technical'], 1); self.assertEqual(r['pending'], 6)
        self.assertEqual(r['success_rate'], 0.)
        empty = next(r for r in summary if r['environment'] == 'pusht' and r['mode'] == 'draft_goal')
        self.assertIsNone(empty['success_rate'])
        self.assertTrue(all(p['full_zero_inconclusive'] for p in pairs))

    def test_freeze_requires_native_success_and_prompt_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            env = dict(status='complete', environments={n: dict(status='supported', success_oracle='native checker') for n in ('calvin', 'simpler', 'pusht')})
            env['case_lists'] = {n: [dict(r, task='testtask', instruction='unit-test instruction', native_hz=10) for r in cases() if r['environment'] == n]
                                 for n in ('calvin', 'simpler', 'pusht')}
            policy = dict(status='complete', modes=list(MODES), checkpoint_metadata=dict(model='frozen'), prompt_proof=dict(passed=True))
            for name, value in [('env', env), ('policy', policy), ('reference', dict(real_libero=True))]:
                (base/(name+'.json')).write_text(json.dumps(value))
            with patch('preparations.idea1_v3_2.transfer_protocol.root_path', side_effect=lambda p: Path(p)):
                manifest = freeze(base/'out', base/'env.json', base/'policy.json', base/'reference.json', [2, 3])
                self.assertEqual(len(manifest['cases'])*len(manifest['modes']), 108)
                self.assertEqual(manifest['cases'][0]['gpu'], 2)
                self.assertEqual(manifest['cases'][1]['gpu'], 3)
                single = freeze(base/'single', base/'env.json', base/'policy.json', base/'reference.json', [3])
                self.assertTrue(all(c['gpu'] == 3 and c['shard'] == 0 for c in single['cases']))
                with self.assertRaises(FileExistsError):
                    freeze(base/'out', base/'env.json', base/'policy.json', base/'reference.json', [2, 3])
                del env['environments']['calvin']['success_oracle']
                (base/'env.json').write_text(json.dumps(env))
                with self.assertRaises(ValueError):
                    freeze(base/'out2', base/'env.json', base/'policy.json', base/'reference.json', [2, 3])

    def test_dry_run_has_no_gpu_or_child_launch(self):
        command = [sys.executable, '-m', 'preparations.idea1_v3_2.transfer_run']
        r = subprocess.run(command, text=True, capture_output=True, check=True)
        payload = json.loads(r.stdout)
        self.assertEqual(payload['gpu_calls'], 0)
        self.assertEqual(payload['subprocess_launches'], 0)


if __name__ == '__main__':
    unittest.main()

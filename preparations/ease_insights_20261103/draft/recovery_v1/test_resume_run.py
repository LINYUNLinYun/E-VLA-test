"""Exercise prepare and full resume control flow with a CPU-only fake runner."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from audit_completed_run import audit, digest
from resume_run import prepare, run_prepared
import test_audit_completed_run as audit_tests


FAKE_RUNNER = '''import argparse, hashlib, json
from pathlib import Path
def atomic_json(p, value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
def main():
 p=argparse.ArgumentParser()
 for flag in ('output','cases','mode','source-suite','gpu','gpu-uuid','ready-timeout'):
  p.add_argument('--'+flag)
 a=p.parse_args();run=Path(a.output);cases=json.loads(Path(a.cases).read_text())
 manifest=dict(cases=cases,protocol={},mode=a.mode,source_suite=a.source_suite,gpu=a.gpu_uuid,
  code_sha256={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
 atomic_json(run/'manifest.json',manifest)
 attempted=[]
 if (run/'force_failure').exists():
  c=cases[-1];key='%s_t%02d_i%02d'%(c['suite'],c['task_id'],c['init_id']);d=run/'episodes'/key
  native=dict(task_suite_name=c['suite'],task_id=c['task_id'],episode_idx=c['init_id'],failure_reason='CUDA OOM',
   success=False,executed_action_count=0,trajectory_length=0,max_steps=280,infer_calls=0)
  atomic_json(d/'native/episode_log.json',[native])
  atomic_json(d/'result.json',dict(c,status='technical_failure',wall_seconds=4))
  atomic_json(run/'result.json',dict(status='technical_failure',started_unix=200,completed_unix=204,total_cases=2,
   preparation_seconds=1,closed_loop_seconds=3,wall_seconds=5,gpu=a.gpu_uuid))
  raise RuntimeError('CPU test injected OOM')
 for c in cases:
  key='%s_t%02d_i%02d'%(c['suite'],c['task_id'],c['init_id']);d=run/'episodes'/key
  if (d/'result.json').exists():continue
  attempted.append(key)
  native=dict(task_suite_name=c['suite'],task_id=c['task_id'],episode_idx=c['init_id'],failure_reason=None,
   success=True,executed_action_count=12,trajectory_length=12,max_steps=280,infer_calls=1)
  receipt=dict(c,mode=a.mode,source_suite=a.source_suite,status='complete',success=True,rpc_calls=1,episode=native)
  atomic_json(d/'native/episode_log.json',[native]);atomic_json(d/'result.json',receipt)
  request=dict(c,replan_id=0,mode=a.mode,source_suite=a.source_suite,route_calls=dict(full=0,draft=1,verify=0))
  with (run/'requests.jsonl').open('a') as f:f.write(json.dumps(request)+'\\n')
 atomic_json(run/'client_complete.json',dict(status='complete',cases=2,successes=1))
 atomic_json(run/'test_attempted.json',attempted)
 atomic_json(run/'result.json',dict(status='complete',started_unix=200,completed_unix=204,total_cases=2,
  cases=2,successes=1,preparation_seconds=1,closed_loop_seconds=3,wall_seconds=5,gpu=a.gpu_uuid))
 assert json.loads((run/'result.json').read_text())['status']=='recovery_pending_validation'
'''


class ResumeTests(unittest.TestCase):
    write = audit_tests.RecoveryAuditTests.write

    def setUp(self):
        audit_tests.RecoveryAuditTests.setUp(self)
        self.pending = dict(suite='libero_object', task_id=1, init_id=20)
        self.failed_dir = self.run / 'episodes/libero_object_t01_i20'
        native = dict(self.native, task_id=1, failure_reason='CUDA OOM', executed_action_count=0, trajectory_length=0, infer_calls=0)
        self.write(self.failed_dir / 'native/episode_log.json', [native])
        self.write(self.failed_dir / 'result.json', dict(self.pending, status='technical_failure', wall_seconds=3))
        cases = [self.case, self.pending]
        self.runner = self.root / 'runtime/run.py'
        self.runner.parent.mkdir()
        self.runner.write_text(FAKE_RUNNER)
        hashes = {str(self.runner): digest(self.runner)}
        self.write(self.root / 'frozen.json', dict(source_sha256=hashes))
        self.write(self.run / 'manifest.json', dict(cases=cases, protocol={}, mode='draft', source_suite='libero_object', gpu='old_gpu', code_sha256=hashes))
        self.write(self.run / 'cases.json', cases)
        self.write(self.run / 'result.json', dict(status='technical_failure', started_unix=100, completed_unix=111,
            wall_seconds=12, preparation_seconds=2, total_cases=2, mode='draft', source_suite='libero_object'))
        (self.run / 'client_complete.json').unlink()
        self.prefix = (self.run / 'requests.jsonl').read_bytes()
        with (self.run / 'requests.jsonl').open('ab') as f:
            f.write(b'{"suite":')
        self.ledger = self.root / 'recovery/resume_ledger.json'

    def test_prepare_preserves_evidence_and_is_idempotent(self):
        original = {str(p): digest(p) for p in self.directory.rglob('*') if p.is_file()}
        plan = prepare(self.run, self.root / 'frozen.json', self.ledger)
        self.assertEqual(plan['status'], 'prepared')
        self.assertEqual((self.run / 'requests.jsonl').read_bytes(), self.prefix)
        self.assertFalse((self.run / 'result.json').exists())
        self.assertFalse(self.failed_dir.exists())
        archive = Path(plan['archive'])
        self.assertTrue((archive / 'technical_episodes/libero_object_t01_i20/result.json').exists())
        self.assertTrue((archive / 'attempt0/requests.jsonl.saved').read_bytes().endswith(b'{"suite":'))
        self.assertEqual({p: digest(Path(p)) for p in original}, original)
        self.assertEqual(prepare(self.run, self.root / 'frozen.json', self.ledger)['status'], 'prepared')
        self.assertEqual(list(archive.rglob('cases.json')), [])

    def test_resume_skips_complete_cases_validates_and_accumulates(self):
        prepare(self.run, self.root / 'frozen.json', self.ledger)
        result = run_prepared(self.ledger, SimpleNamespace(gpu='0', gpu_uuid='new_gpu', ready_timeout=30))
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(json.loads((self.run / 'test_attempted.json').read_text()), ['libero_object_t01_i20'])
        receipt = json.loads((self.run / 'result.json').read_text())
        self.assertEqual(receipt['cases'], 2)
        self.assertEqual(receipt['wall_seconds'], 17)
        self.assertEqual(receipt['preparation_seconds'], 3)
        self.assertEqual(receipt['closed_loop_seconds'], 12)
        self.assertEqual(receipt['recovery']['prior_technical_case_wall_seconds'], 3)
        self.assertTrue((self.run / 'requests.jsonl').read_bytes().startswith(self.prefix))
        provenance = json.loads((self.run / 'manifest.json').read_text())['recovery_provenance']
        self.assertEqual(provenance['case_service_attempt'], {'libero_object_t00_i20': 0, 'libero_object_t01_i20': 1})
        self.assertEqual(audit(self.run, self.root / 'frozen.json')['issues'], [])

    def test_legacy_then_nested_recovery_preserves_history_and_costs(self):
        allocation = SimpleNamespace(gpu='0', gpu_uuid='new_gpu', ready_timeout=30)
        prepare(self.run, self.root / 'frozen.json', self.ledger)
        (self.run / 'force_failure').write_text('CPU only')
        with self.assertRaises(RuntimeError):
            run_prepared(self.ledger, allocation)
        receipt = json.loads((self.run / 'result.json').read_text())
        self.assertEqual(receipt['wall_seconds'], 17)
        # Emulate the pre-fix v1 receipt encountered in production.
        del receipt['recovery']['attempt_history']
        self.write(self.run / 'result.json', receipt)
        second_ledger = self.ledger.with_name('second_resume.json')
        second = prepare(self.run, self.root / 'frozen.json', second_ledger)
        self.assertEqual(second['next_service_attempt'], 2)
        self.assertEqual(second['prior_total_technical_case_wall_seconds'], 7)
        self.assertEqual(second['retained_case_service_attempt'], {'libero_object_t00_i20': 0})
        with self.assertRaises(RuntimeError):
            run_prepared(second_ledger, allocation)
        receipt = json.loads((self.run / 'result.json').read_text())
        self.assertEqual(receipt['wall_seconds'], 22)
        self.assertEqual(receipt['preparation_seconds'], 4)
        self.assertEqual(receipt['recovery']['total_technical_case_wall_seconds'], 11)
        third_ledger = self.ledger.with_name('third_resume.json')
        third = prepare(self.run, self.root / 'frozen.json', third_ledger)
        self.assertEqual(third['next_service_attempt'], 3)
        self.assertEqual(third['prior_total_technical_case_wall_seconds'], 11)
        (self.run / 'force_failure').unlink()
        run_prepared(third_ledger, allocation)
        receipt = json.loads((self.run / 'result.json').read_text())
        self.assertEqual(receipt['wall_seconds'], 27)
        self.assertEqual(receipt['preparation_seconds'], 5)
        self.assertEqual(receipt['recovery']['total_technical_case_wall_seconds'], 11)
        history = receipt['recovery']['attempt_history']
        self.assertEqual([row['attempt'] for row in history], [0, 1, 2, 3])
        self.assertEqual([row['wall_seconds'] for row in history], [12, 5, 5, 5])
        provenance = json.loads((self.run / 'manifest.json').read_text())['recovery_provenance']
        self.assertEqual(provenance['case_service_attempt'], {'libero_object_t00_i20': 0, 'libero_object_t01_i20': 3})
        self.assertEqual(audit(self.run, self.root / 'frozen.json')['issues'], [])


if __name__ == '__main__':
    unittest.main()

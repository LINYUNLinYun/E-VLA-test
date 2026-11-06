import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from . import resume_half_deadline as guard


def identity(pid=10):
    return dict(pid=pid,start_ticks='100',argv='owned command',cwd=str(guard.ROOT),state='S',ppid=1)


class DeadlineGuardTests(unittest.TestCase):
    def test_pid_reuse_or_command_change_never_matches(self):
        old=identity();self.assertTrue(guard.exact_process(old,dict(old)))
        for key in ('pid','start_ticks','argv','cwd'):
            changed=dict(old);changed[key]='different';self.assertFalse(guard.exact_process(old,changed))
        self.assertFalse(guard.exact_process(old,dict(old,state='Z')))

    def test_endpoint_must_match_all_ownership_fields(self):
        valid=dict(server_pid=20,gpu_uuid=guard.UUID,manifest_hash='parent',family='libero_transfer',legacy_gate_passed=True)
        self.assertEqual(guard.endpoint(valid,20,guard.UUID,'parent'),valid)
        for key in valid:
            changed=dict(valid);changed[key]='different'
            with self.assertRaises(RuntimeError): guard.endpoint(changed,20,guard.UUID,'parent')

    def test_changed_worker_has_zero_signals_or_gpu_queries(self):
        plan=dict(worker=identity(),model=identity(20))
        with patch.object(guard,'sources'),patch.object(guard,'process_identity',return_value=dict(identity(),start_ticks='new')),patch.object(guard.os,'kill') as kill,patch.object(guard,'rpc') as rpc:
            with self.assertRaises(RuntimeError): guard.verify_old(plan)
            kill.assert_not_called();rpc.assert_not_called()

    def test_drain_signals_only_parent_after_natural_complete(self):
        with tempfile.TemporaryDirectory() as root:
            directory=Path(root);episode=directory/'episode';episode.mkdir()
            (episode/'result.json').write_text(json.dumps(dict(status='complete',manifest_hash='parent',success=False)))
            worker=identity();child=dict(identity(30),ppid=10,argv='python -m preparations.idea1_v3_2_e4_supplement.run_plus --child-index 20 --output '+str(episode))
            plan=dict(worker=worker,parent_manifest_hash='parent');receipt={}
            with patch.object(guard,'verify_old'),patch.object(guard,'children_of',return_value=[child]),patch.object(guard,'alive',return_value=False),patch.object(guard,'process_identity',return_value=worker),patch.object(guard.time,'sleep'),patch.object(guard.os,'kill') as kill:
                guard.drain_old_worker(plan,receipt,directory,lambda:None)
            self.assertEqual([(call.args[0],call.args[1]) for call in kill.call_args_list],[(10,guard.signal.SIGSTOP),(10,guard.signal.SIGTERM),(10,guard.signal.SIGCONT)])
            self.assertFalse(receipt['renderer_signals_sent']);self.assertFalse(receipt['drained_results'][0]['success'])

    def test_180s_timeout_resumes_only_parent_never_kills_child(self):
        with tempfile.TemporaryDirectory() as root:
            worker=identity();child=dict(identity(30),ppid=10,argv='python -m preparations.idea1_v3_2_e4_supplement.run_plus --child-index 20')
            with patch.object(guard,'verify_old'),patch.object(guard,'children_of',return_value=[child]),patch.object(guard,'alive',return_value=True),patch.object(guard,'process_identity',return_value=worker),patch.object(guard.time,'monotonic',side_effect=[0,181]),patch.object(guard.time,'sleep'),patch.object(guard.os,'kill') as kill:
                with self.assertRaises(TimeoutError): guard.drain_old_worker(dict(worker=worker),{},Path(root),lambda:None)
            self.assertEqual([(call.args[0],call.args[1]) for call in kill.call_args_list],[(10,guard.signal.SIGSTOP),(10,guard.signal.SIGCONT)])

    def test_claim_archive_only_exact_dead_owner_incomplete_and_recoverable(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);half=root/'half';claims=half/'claims';claims.mkdir(parents=True);run=root/'run';run.mkdir()
            job=dict(group='perturbed350',source_index=50,case={},case_hash='case',modes=['full_d10','v14_d10','draft_goal'])
            plan=dict(worker=identity(),half_root=str(half),half_manifest_hash='half',parent_manifest_hash='parent')
            owner=dict(pid=10,start_ticks='100',gpu_uuid=guard.UUID,half_manifest_hash='half',case_hash='case')
            for name,values in [('perturbed350_0050',owner),('perturbed350_0060',dict(owner,pid=99)),('complete',owner)]:
                path=claims/name;path.mkdir();(path/'owner.json').write_text(json.dumps(values))
            (claims/'complete/complete.json').write_text('{}')
            with patch.object(guard,'alive',return_value=False),patch.object(guard,'source_results',return_value=([],[])):
                rows=guard.archive_claims(plan,run,dict(jobs=[job],source_results_root='old'))
            self.assertEqual(len(rows),1);self.assertTrue(rows[0]['recoverable'])
            self.assertTrue((run/'archived_claims/perturbed350_0050/owner.json').exists())
            self.assertTrue((claims/'perturbed350_0060/owner.json').exists());self.assertTrue((claims/'complete/complete.json').exists())

    def test_idle_wait_uses_at_least30s_and_double_final_uuid_sample(self):
        with tempfile.TemporaryDirectory() as root:
            busy=dict(idle=False,uuid=guard.UUID);idle=dict(idle=True,uuid=guard.UUID)
            with patch.object(guard,'gpu3_sample',side_effect=[busy,busy,idle,idle]) as sample,patch.object(guard.time,'time',return_value=0),patch.object(guard.time,'sleep') as sleep:
                result=guard.wait_idle({},Path(root),lambda:None)
            self.assertTrue(result['idle']);self.assertEqual(sample.call_count,4);self.assertEqual([c.args[0] for c in sleep.call_args_list],[30,30])

    def test_585_complete_is_confirmation_only_no_signal_no_gpu_no_popen(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);half=root/'half';half.mkdir();(half/'manifest.json').write_text('{}')
            plan=dict(half_root=str(half),source_sha256={});plan['plan_hash']=guard.digest(plan)
            with patch.object(guard,'approval'),patch.object(guard,'sources'),patch.object(guard,'verify_old'),patch.object(guard,'audit',return_value=dict(completed=585)),patch.object(guard.time,'time',return_value=guard.TRIGGER+1),patch.object(guard.signal,'signal'),patch.object(guard.os,'kill') as kill,patch.object(guard,'gpu3_sample') as gpu,patch.object(guard.subprocess,'Popen') as popen:
                guard.execute(plan,SimpleNamespace(plan=str(root/'plan.json'),approval='approved'))
            kill.assert_not_called();gpu.assert_not_called();popen.assert_not_called()
            receipt=json.loads(next(root.glob('execution_*/execution.json')).read_text());self.assertEqual(receipt['status'],'already_complete_no_restart')

    def test_cleanup_replacement_is_never_signalled_or_shutdown(self):
        process=Mock();process.pid=20;process.poll.return_value=None
        with patch.object(guard,'process_identity',return_value=dict(identity(20),start_ticks='new')),patch.object(guard,'rpc') as rpc:
            with self.assertRaises(RuntimeError): guard.cleanup_new_model(process,identity(20),{})
            process.terminate.assert_not_called();rpc.assert_not_called()

    def test_cleanup_wrong_endpoint_pid_never_sends_shutdown(self):
        process=Mock();process.pid=20;process.poll.return_value=None
        wrong=dict(server_pid=99,gpu_uuid=guard.UUID,manifest_hash='parent',family='libero_transfer',legacy_gate_passed=True)
        with patch.object(guard,'process_identity',return_value=identity(20)),patch.object(guard,'rpc',return_value=dict(protocol=wrong)) as rpc:
            with self.assertRaises(RuntimeError): guard.cleanup_new_model(process,identity(20),dict(port=19143,parent_manifest_hash='parent'))
            self.assertEqual([call.args[1]['op'] for call in rpc.call_args_list],['health']);process.terminate.assert_not_called()

    def test_waiting_complete_audited_before_missing_old_pid(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);half=root/'half';half.mkdir();(half/'manifest.json').write_text('{}')
            plan=dict(half_root=str(half),source_sha256={});plan['plan_hash']=guard.digest(plan)
            with patch.object(guard,'approval'),patch.object(guard,'sources'),patch.object(guard,'verify_old',side_effect=RuntimeError('Oldworker exited naturally')) as verify,patch.object(guard,'audit',return_value=dict(completed=585)),patch.object(guard.time,'time',return_value=guard.TRIGGER-60),patch.object(guard.signal,'signal'),patch.object(guard,'gpu3_sample') as gpu:
                guard.execute(plan,SimpleNamespace(plan=str(root/'plan.json'),approval='approved'))
            verify.assert_not_called();gpu.assert_not_called()

    def test_waiting_heavy_identity_checks_only_every5min(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);half=root/'half';half.mkdir();(half/'manifest.json').write_text('{}')
            plan=dict(half_root=str(half),source_sha256={});plan['plan_hash']=guard.digest(plan);now=[guard.TRIGGER-700];checks=[]
            def sleep(seconds): now[0]+=seconds
            def audit(*args): return dict(completed=585 if now[0]>=guard.TRIGGER-50 else 1)
            with patch.object(guard,'approval'),patch.object(guard,'sources'),patch.object(guard,'verify_old',side_effect=lambda plan:checks.append(now[0])),patch.object(guard,'audit',side_effect=audit),patch.object(guard.time,'time',side_effect=lambda:now[0]),patch.object(guard.time,'sleep',side_effect=sleep),patch.object(guard.signal,'signal'):
                guard.execute(plan,SimpleNamespace(plan=str(root/'plan.json'),approval='approved'))
            self.assertTrue(all(b-a>=300 for a,b in zip(checks,checks[1:])))

    def test_revoked_or_cancelled_approval_cannot_launch(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'approval.json';value=dict(approved=True,revoked=False,priority='plus_half585_only',plan_hash='p',new_deadline_unix=guard.DEADLINE)
            path.write_text(json.dumps(value));self.assertTrue(guard.approval(dict(plan_hash='p'),path))
            Path(str(path)+'.cancel').touch()
            with self.assertRaises(RuntimeError): guard.approval(dict(plan_hash='p'),path)


if __name__=='__main__': unittest.main()

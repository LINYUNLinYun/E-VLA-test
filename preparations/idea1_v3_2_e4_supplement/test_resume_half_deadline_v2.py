import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from . import resume_half_deadline_v2 as guard


def plan():
    return dict(trigger_unix=guard.TRIGGER,old_deadline_unix=guard.OLD_DEADLINE,new_deadline_unix=guard.NEW_DEADLINE,
        child_drain_timeout_s=1500,shutdown_safety_margin_s=180,heavy_check_interval_s=300,approval_poll_interval_s=60,idle_poll_interval_s=30,
        worker=dict(pid=10,start_ticks='100',argv='ownedworker',cwd=str(guard.ROOT),state='S'),parent_manifest_hash='parent')


class FakeClock:
    def __init__(self,wall): self.initial=wall;self.elapsed=0
    def time(self): return self.initial+self.elapsed
    def monotonic(self): return self.elapsed
    def sleep(self,seconds): self.elapsed+=seconds


class GuardV2Tests(unittest.TestCase):
    def test_timing_validation(self):
        self.assertEqual(guard.validate_timing(plan())['timeout_s'],1500)
        for key,value in [('trigger_unix',guard.TRIGGER+60),('child_drain_timeout_s',180),('child_drain_timeout_s',1501),
            ('shutdown_safety_margin_s',0),('shutdown_safety_margin_s',301),('approval_poll_interval_s',61),('heavy_check_interval_s',60),('idle_poll_interval_s',1),('new_deadline_unix',guard.NEW_DEADLINE+1)]:
            invalid=plan();invalid[key]=value
            with self.assertRaises(ValueError,msg=(key,value)): guard.validate_timing(invalid)

    def run_drain(self,duration,timeout=1500,start=None):
        directory=self.directory;episode=directory/'episode';episode.mkdir()
        (episode/'result.json').write_text(json.dumps(dict(status='complete',manifest_hash='parent',success=False)))
        current=plan();current['child_drain_timeout_s']=timeout;child=dict(pid=30,start_ticks='300',ppid=10,state='S',argv='python -m preparations.idea1_v3_2_e4_supplement.run_plus --child-index 1 --output '+str(episode))
        clock=FakeClock(start if start is not None else guard.TRIGGER);receipt={}
        with patch.object(guard,'verify_old'),patch.object(guard,'children_of',return_value=[child]),patch.object(guard,'alive',side_effect=lambda identity:clock.elapsed<duration if identity['pid']==30 else False),patch.object(guard,'process_identity',return_value=current['worker']),patch.object(guard.time,'time',side_effect=clock.time),patch.object(guard.time,'monotonic',side_effect=clock.monotonic),patch.object(guard.time,'sleep',side_effect=clock.sleep),patch.object(guard.os,'kill') as kill:
            try: guard.drain_old_worker(current,receipt,directory,lambda:None);error=None
            except Exception as exc:error=exc
        return clock,receipt,error,[(call.args[0],call.args[1]) for call in kill.call_args_list]

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.directory=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()

    def test_child_finishes_after600s_not_old180cap_no_renderer_signals(self):
        clock,receipt,error,signals=self.run_drain(600)
        self.assertIsNone(error);self.assertGreater(clock.elapsed,180);self.assertLess(clock.elapsed,1500)
        self.assertEqual(signals,[(10,guard.signal.SIGSTOP),(10,guard.signal.SIGTERM),(10,guard.signal.SIGCONT)])
        self.assertFalse(receipt['renderer_signals_sent'])

    def test_frozen_plan_timeout_is_actually_used_and_cont_only(self):
        clock,receipt,error,signals=self.run_drain(600,timeout=300)
        self.assertIsInstance(error,TimeoutError);self.assertGreaterEqual(clock.elapsed,300);self.assertLess(clock.elapsed,301)
        self.assertEqual(signals,[(10,guard.signal.SIGSTOP),(10,guard.signal.SIGCONT)])

    def test_late_drain_is_additionally_bounded_by_olddeadline_margin(self):
        start=guard.OLD_DEADLINE-180-10
        clock,receipt,error,signals=self.run_drain(600,start=start)
        self.assertIsInstance(error,TimeoutError);self.assertLess(clock.elapsed,11)
        self.assertEqual(receipt['drain_timing']['actual_cap_s'],10)
        self.assertEqual(signals,[(10,guard.signal.SIGSTOP),(10,guard.signal.SIGCONT)])

    def test_child_exit_just_after_cap_still_cannot_bypass_timeout(self):
        clock,receipt,error,signals=self.run_drain(300.05,timeout=300)
        self.assertIsInstance(error,TimeoutError)
        self.assertEqual(signals,[(10,guard.signal.SIGSTOP),(10,guard.signal.SIGCONT)])

    def test_too_late_never_stops_original_parent(self):
        current=plan()
        with patch.object(guard,'verify_old'),patch.object(guard.time,'time',return_value=guard.OLD_DEADLINE-180),patch.object(guard.os,'kill') as kill:
            with self.assertRaises(TimeoutError):guard.drain_old_worker(current,{},self.directory,lambda:None)
            kill.assert_not_called()

    def execution(self,completed,now,verify=None,drain=None):
        half=self.directory/'half';half.mkdir();(half/'manifest.json').write_text('{}')
        current=plan();current.update(half_root=str(half),source_sha256={});current['plan_hash']=guard.digest(current)
        args=SimpleNamespace(plan=str(self.directory/'plan.json'),approval='root_approval')
        with patch.object(guard.base,'approval'),patch.object(guard,'sources'),patch.object(guard,'audit',side_effect=completed),patch.object(guard,'verify_old',side_effect=verify) as verify_mock,patch.object(guard,'drain_old_worker',side_effect=drain) as drain_mock,patch.object(guard.base,'stop_old_model') as stop,patch.object(guard.time,'time',return_value=now),patch.object(guard.signal,'signal'),patch.object(guard.os,'kill') as kill,patch.object(guard.base,'wait_idle') as gpu,patch.object(guard.subprocess,'Popen') as popen:
            try:guard.execute(current,args);error=None
            except Exception as exc:error=exc
        return error,verify_mock,drain_mock,stop,kill,gpu,popen

    def test_already_complete_no_process_model_or_gpu_mutation_even_oldworker_missing(self):
        error,verify,drain,stop,kill,gpu,popen=self.execution([dict(completed=585)],guard.TRIGGER-60,verify=RuntimeError('oldpidgone'))
        self.assertIsNone(error);verify.assert_not_called();drain.assert_not_called();stop.assert_not_called();kill.assert_not_called();gpu.assert_not_called();popen.assert_not_called()

    def test_after_natural_drain_reaudit_complete_keeps_original_model_untouched(self):
        error,verify,drain,stop,kill,gpu,popen=self.execution([dict(completed=584),dict(completed=585)],guard.TRIGGER+1)
        self.assertIsNone(error);drain.assert_called_once();stop.assert_not_called();gpu.assert_not_called();popen.assert_not_called()
        receipt=json.loads(next(self.directory.glob('execution_*/execution.json')).read_text())
        self.assertEqual(receipt['status'],'complete_after_drain_original_model_untouched')

    def test_pid_reuse_or_foreign_endpoint_aborts_before_any_mutation(self):
        for fields in ('start_ticks','argv','cwd'):
            expected=plan()['worker'];current=dict(expected);current[fields]='replacement';self.assertFalse(guard.exact_process(expected,current))
        foreign=dict(server_pid=99,gpu_uuid=guard.UUID,manifest_hash='parent',family='libero_transfer',legacy_gate_passed=True)
        with self.assertRaises(RuntimeError):guard.endpoint(foreign,20,guard.UUID,'parent')
        error,verify,drain,stop,kill,gpu,popen=self.execution([dict(completed=1)],guard.TRIGGER+1,verify=RuntimeError('foreign/reusedidentity'))
        self.assertIsInstance(error,RuntimeError);drain.assert_not_called();stop.assert_not_called();kill.assert_not_called();gpu.assert_not_called();popen.assert_not_called()

    def test_revoked_and_cancelled_new_approval(self):
        path=self.directory/'approval.json';current=plan();current['plan_hash']='p'
        value=dict(approved=True,revoked=False,priority='plus_half585_only',plan_hash='p',new_deadline_unix=guard.NEW_DEADLINE)
        path.write_text(json.dumps(value));guard.base.approval(current,path)
        value['revoked']=True;path.write_text(json.dumps(value))
        with self.assertRaises(RuntimeError):guard.base.approval(current,path)
        value['revoked']=False;path.write_text(json.dumps(value));Path(str(path)+'.cancel').touch()
        with self.assertRaises(RuntimeError):guard.base.approval(current,path)

    def test_waiting_uses_plan5min_checks_without_early_mutations(self):
        half=self.directory/'half';half.mkdir();(half/'manifest.json').write_text('{}')
        current=plan();current.update(half_root=str(half),source_sha256={});current['plan_hash']=guard.digest(current)
        clock=FakeClock(guard.TRIGGER-700);checks=[];sleeps=[]
        def sleep(seconds):sleeps.append(seconds);clock.sleep(seconds)
        with patch.object(guard.base,'approval'),patch.object(guard,'sources'),patch.object(guard,'audit',side_effect=lambda *args:dict(completed=585 if clock.elapsed>=650 else 1)),patch.object(guard,'verify_old',side_effect=lambda plan:checks.append(clock.elapsed)),patch.object(guard.time,'time',side_effect=clock.time),patch.object(guard.time,'sleep',side_effect=sleep),patch.object(guard.signal,'signal'),patch.object(guard.os,'kill') as kill,patch.object(guard.base,'wait_idle') as gpu,patch.object(guard.subprocess,'Popen') as popen:
            guard.execute(current,SimpleNamespace(plan=str(self.directory/'plan.json'),approval='approved'))
        self.assertTrue(all(b-a>=300 for a,b in zip(checks,checks[1:])));self.assertTrue(all(s<=60 for s in sleeps))
        kill.assert_not_called();gpu.assert_not_called();popen.assert_not_called()

    def test_prepare_new_closure_does_not_pin_old_revocable_approval(self):
        captured=dict(plan_hash='baseidentitycapture',source_sha256={str(Path(guard.base.__file__).resolve()):guard.sha(guard.base.__file__)})
        output=self.directory/'prepared'
        with patch.object(guard.base,'prepare',return_value=captured):
            prepared=guard.prepare(SimpleNamespace(output=str(output)))
        self.assertEqual(prepared['trigger_unix'],guard.TRIGGER);self.assertEqual(prepared['child_drain_timeout_s'],1500)
        self.assertIn(str(Path(guard.__file__).resolve()),prepared['source_sha256'])
        self.assertFalse(any('root_approval' in path for path in prepared['source_sha256']))
        self.assertFalse(json.loads((output/'approval_template.json').read_text())['approved'])


if __name__=='__main__':unittest.main()

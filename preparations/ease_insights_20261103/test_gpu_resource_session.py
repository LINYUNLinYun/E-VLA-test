"""CPU fakes only: no driver, GPU, real scheduler, or scientific rollout."""
import argparse
from contextlib import contextmanager, nullcontext
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from preparations.ease_insights_20261103 import gpu_resource_session as session

UUID = 'GPU-12345678-1234-1234-1234-123456789abc'


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value))


class Child:
    def __init__(self, pid, worker=False):
        self.pid, self.worker, self.returncode = pid, worker, None

    def poll(self):
        if self.worker:
            self.returncode = 0
        return self.returncode


class FakeRuntime:
    def __init__(self, root):
        self.root = root; self.clock = 100.; self.starts = 0; self.closes = 0
        self.live = True; self.sleeps = 0; self.release_after = 2; self.events = []; self.children = []

    def now(self): return self.clock
    def monotonic(self): return self.clock
    def write(self, path, value): write(path, value)
    def bind_scheduler(self, gpu):
        binding = dict(job_id='slot_one', lease_device=1, lease_inode=77,
            lease_path='/tmp/fake-scheduler-lease', lease_fds=[101],
            session=dict(pid=99, start_ticks='10', state='S', argv=['/usr/bin/python3', '-m', session.MODULE]),
            scheduler=dict(pid=88, start_ticks='5', state='S', argv=['/usr/bin/python3', '-m', session.identity.SCHEDULER]))
        return copy.deepcopy(getattr(self, 'resume_binding', binding))
    def write_resume_receipt(self, path, value): session.Runtime.write_resume_receipt(self, path, value)
    def validate_resume(self, request, active): return session.Runtime.validate_resume(self, request, active)
    def start_keeper(self, gpu, output):
        self.starts += 1; self.events.append('keeper_start')
        return dict(ready=dict(pid=100, gpu_uuid=gpu, nonce='nonce'), child=Child(100))
    def keeper_live(self, keeper): return self.live
    def close_keeper(self, keeper):
        self.closes += 1; self.events.append('keeper_close')
        return dict(process_exited=True, exit_code=0)
    def children_reaped(self): return all(child.poll() is not None for child, _, _ in self.children)
    def reap_scientific_children(self):
        for child, _, _ in self.children:
            if child.poll() is None: self.cleanup_child(child)
    def sleep(self, seconds):
        self.clock += seconds; self.sleeps += 1
        if self.sleeps >= self.release_after:
            (self.root/'release').touch()
    def scan_owned(self, gpu, keeper, maximum):
        self.events.append('scan_sole_keeper'); return dict(consistent=True)
    def reserve_port(self, port): self.events.append('port')
    def popen(self, argv, env, handle):
        worker = 'worker' in argv
        child = Child(202 if worker else 201, worker)
        self.children.append((child, copy.deepcopy(argv), copy.deepcopy(env)))
        return child
    def cleanup_child(self, child):
        self.events.append('cleanup_'+str(child.pid)); child.returncode = 0


class FakeScience:
    def __init__(self, fail=False): self.calls = []; self.fail = fail
    def prepare(self, task, plan, gpu, port): return task
    def execute(self, task, active):
        self.calls.append((task['id'], id(active.keeper), active.binding['lease_inode']))
        active.runtime.events.append('task_'+task['id'])
        if self.fail: raise RuntimeError('technical failure')
        return dict(complete=True)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root/'tasks').mkdir()
        self.args = argparse.Namespace(output=str(self.root/'sessions/one'), gpu_uuid=UUID, port=12345)
        self.plan = dict(campaign_root=str(self.root), _sha256='plan')
        self.rt = FakeRuntime(self.root); self.science = FakeScience()
        self.verify = patch.object(session, 'verify_plan'); self.verify.start()

    def tearDown(self): self.verify.stop(); self.temp.cleanup()

    def task(self, name, activate=True):
        path = self.root/'tasks'/(name+'.json'); activation = self.root/'transactions/tx/committed.json'
        task = dict(id=name, logical_job_id='original_'+name, activation_file=str(activation))
        write(path, task)
        if activate:
            values = session.read(activation).get('task_sha256', {}) if activation.exists() else {}
            values[name] = session.sha(path.read_bytes())
            write(activation, dict(schema='ease-gpu-resource-activation-v1', task_sha256=values))
        return task

    def active(self):
        active = session.Session(self.args, self.plan, self.rt, self.science)
        active.binding = self.rt.bind_scheduler(UUID); active.keeper = self.rt.start_keeper(UUID, self.args.output)
        return active

    def resume(self, active, name='request1', **changes):
        own = active.binding['session']
        request = dict(schema=session.RESUME_SCHEMA, action='resume_after_attention', plan_sha256=self.plan['_sha256'],
            session_pid=own['pid'], proc_start_ticks=own['start_ticks'], gpu_uuid=UUID,
            attention_sha256=session.sha(json.dumps(active.attention, sort_keys=True).encode()), reason='Root reviewed technical failure')
        request.update(changes)
        path = self.root/'resumes'/active.output.name/(name+'.json')
        write(path, request)
        return path

    def resume_receipt(self, active, name='request1'):
        return session.read(active.output/'resume_receipts'/(name+'.json'))

    def test_two_tasks_keep_same_keeper_and_lease_then_empty_hold_until_release(self):
        self.task('a'); self.task('b')
        result = session.Session(self.args, self.plan, self.rt, self.science).run()
        self.assertEqual(result['phase'], 'released')
        self.assertEqual([x[0] for x in self.science.calls], ['a', 'b'])
        self.assertEqual(len({x[1:] for x in self.science.calls}), 1)
        self.assertEqual((self.rt.starts, self.rt.closes), (1, 1))
        self.assertEqual(self.rt.events, ['keeper_start', 'task_a', 'task_b', 'keeper_close'])
        self.assertGreaterEqual(self.rt.sleeps, 2)
        marker = session.read(Path(self.args.output)/'complete.json')
        self.assertTrue(marker['complete']); self.assertFalse(marker['scientific_success'])
        self.assertEqual(marker['schema'], 'ease-gpu-resource-slot-release-v1')
        self.assertEqual(marker['proc_start_ticks'], '10')

    def test_empty_inbox_remains_holding_and_never_closes_keeper_on_step(self):
        active = self.active()
        self.assertEqual(active.step()['phase'], 'idle_holding')
        self.assertEqual(active.step()['phase'], 'idle_holding')
        self.assertEqual(self.rt.closes, 0)

    def test_operator_signal_waits_for_task_boundary_and_keeps_lease_until_release(self):
        self.task('a'); self.task('b'); active = self.active()
        original = self.science.execute
        def execute(task, current):
            current.request_signal(signal.SIGTERM)
            return original(task, current)
        self.science.execute = execute
        self.assertEqual(active.step()['phase'], 'task_complete')
        self.assertEqual(active.step()['phase'], 'attention_holding')
        self.assertEqual([x[0] for x in self.science.calls], ['a'])
        self.assertEqual(self.rt.closes, 0)
        (self.root/'release').touch()
        self.assertEqual(active.step()['phase'], 'release_requested')

    def test_resource_complete_marker_waits_for_confirmed_keeper_cleanup(self):
        self.rt.release_after = 1
        attempts = []
        def close(keeper):
            attempts.append(1)
            self.assertFalse((Path(self.args.output)/'complete.json').exists())
            return dict(process_exited=len(attempts) > 1, exit_code=0)
        self.rt.close_keeper = close
        result = session.Session(self.args, self.plan, self.rt, self.science).run()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(result['phase'], 'released_after_fatal')
        self.assertTrue(session.read(Path(self.args.output)/'complete.json')['owned_scientific_children_reaped'])

    def test_unactivated_task_is_not_consumed(self):
        self.task('a', False); active = self.active()
        self.assertEqual(active.step()['phase'], 'idle_holding')
        self.assertEqual(self.science.calls, [])

    def test_wrong_activation_hash_is_rejected(self):
        self.task('a'); activation = self.root/'transactions/tx/committed.json'
        write(activation, dict(schema='ease-gpu-resource-activation-v1', task_sha256={'a': 'wrong'}))
        with self.assertRaisesRegex(RuntimeError, 'activated'):
            self.active().step()
        self.assertEqual(self.science.calls, [])

    def test_technical_failure_holds_and_cannot_be_retried_by_other_session(self):
        self.task('a'); self.science.fail = True
        active = self.active()
        self.assertEqual(active.step()['phase'], 'attention_holding')
        self.assertEqual(active.step()['phase'], 'attention_holding')
        other = session.Session(self.args, self.plan, self.rt, FakeScience())
        other.keeper, other.binding = active.keeper, active.binding
        self.assertEqual(other.step()['phase'], 'idle_holding')
        self.assertEqual(len(self.science.calls), 1)
        self.assertTrue(self.rt.live); self.assertEqual(self.rt.closes, 0)

    def test_fatal_keeper_failure_holds_global_lease_until_root_release(self):
        active = self.active(); self.rt.live = False
        self.rt.release_after = 2
        result = active.run()
        self.assertEqual(result['phase'], 'released_after_fatal')
        self.assertIsNotNone(active.fatal)
        self.assertEqual(self.rt.closes, 1)
        self.assertEqual(self.science.calls, [])

    def test_keeper_startup_failure_holds_global_lease_until_root_release(self):
        active = session.Session(self.args, self.plan, self.rt, self.science)
        self.rt.start_keeper = lambda *args: (_ for _ in ()).throw(RuntimeError('driver unavailable'))
        result = active.run()
        self.assertEqual(result['phase'], 'released_after_fatal')
        self.assertEqual(active.fatal['scope'], 'keeper_startup')
        self.assertEqual(self.rt.closes, 0)

    def test_claim_mutex_skips_a_claimed_task(self):
        self.task('a'); active = self.active()
        key = session.sha(b'a')
        with session.claim(self.root/'claims'/(key+'.lock')):
            self.assertEqual(active.step()['phase'], 'idle_holding')
        self.assertEqual(active.step()['phase'], 'task_complete')

    def test_task_mutation_after_outcome_is_refused(self):
        self.task('a'); active = self.active(); active.step()
        task = session.read(self.root/'tasks/a.json'); task['logical_job_id'] = 'changed'; write(self.root/'tasks/a.json', task)
        with self.assertRaisesRegex(RuntimeError, 'Append-only'):
            active.step()

    def test_explicit_resume_retains_keeper_lease_and_old_attention_then_new_task_runs(self):
        self.task('a'); self.science.fail = True; active = self.active()
        self.assertEqual(active.step()['phase'], 'attention_holding')
        keeper, binding = active.keeper, copy.deepcopy(active.binding)
        outcome = self.root/'outcomes'/(session.sha(b'a')+'.json'); previous = outcome.read_bytes()
        request_path = self.resume(active)
        self.science.fail = False
        self.assertEqual(active.step()['phase'], 'idle_holding')
        self.assertIsNone(active.attention)
        self.assertIs(active.keeper, keeper); self.assertEqual(active.binding, binding)
        self.assertEqual((self.rt.starts, self.rt.closes), (1, 0))
        self.assertEqual([call[0] for call in self.science.calls], ['a'])
        self.assertEqual(outcome.read_bytes(), previous)
        receipt = self.resume_receipt(active)
        self.assertEqual(receipt['phase'], 'resumed')
        self.assertEqual(receipt['request_sha256'], session.sha(request_path.read_bytes()))
        self.assertTrue(receipt['source_proof']['verified'])
        self.assertEqual(receipt['source_proof']['inherited_scheduler_lease']['lease_inode'], 77)
        self.task('b')
        self.assertEqual(active.step()['phase'], 'task_complete')
        self.assertEqual([call[0] for call in self.science.calls], ['a', 'b'])
        self.assertEqual(outcome.read_bytes(), previous)

    def test_stale_wrong_and_foreign_resume_requests_are_rejected_without_fatal_or_release(self):
        cases = [('schema', 'unknown'), ('action', 'release'), ('plan_sha256', 'stale'),
                 ('session_pid', 199), ('session_pid', True), ('proc_start_ticks', 'stale'),
                 ('gpu_uuid', 'GPU-87654321-1234-1234-1234-123456789abc'),
                 ('attention_sha256', 'stale'), ('reason', ''), ('reason', '  ')]
        for number, (field, value) in enumerate(cases):
            with self.subTest(field=field, value=value):
                active = self.active(); active.attention = dict(task_id='a', error='technical failure')
                original = copy.deepcopy(active.attention)
                name = 'invalid'+str(number); self.resume(active, name, **{field: value})
                self.assertEqual(active.step()['phase'], 'attention_holding')
                self.assertEqual(self.resume_receipt(active, name)['phase'], 'rejected')
                self.assertEqual(active.attention, original); self.assertIsNone(active.fatal)
                self.assertEqual(self.rt.closes, 0); self.assertEqual(self.science.calls, [])

    def test_changed_sources_binding_children_or_scan_cannot_clear_attention(self):
        cases = ['source', 'scheduler', 'session', 'lease', 'children', 'scan', 'unverified']
        for case in cases:
            with self.subTest(case=case):
                self.rt = FakeRuntime(self.root)
                active = self.active(); active.attention = dict(error='technical failure')
                original = copy.deepcopy(active.attention); self.resume(active, 'failure_'+case)
                target = None
                if case == 'source':
                    target = patch.object(session, 'verify_plan', side_effect=RuntimeError('Trusted source changed'))
                elif case in ('scheduler', 'session', 'lease'):
                    changed = copy.deepcopy(active.binding)
                    if case == 'lease': changed['lease_inode'] += 1
                    else: changed[case]['start_ticks'] = '999'
                    self.rt.resume_binding = changed
                elif case == 'children':
                    self.rt.children.append((Child(999), [], {}))
                elif case == 'scan':
                    target = patch.object(self.rt, 'scan_owned', side_effect=session.KeeperFailure('foreign compute appeared'))
                elif case == 'unverified':
                    active.binding = dict(verified=False)
                with target if target is not None else nullcontext():
                    self.assertEqual(active.step()['phase'], 'attention_holding')
                self.assertEqual(self.resume_receipt(active, 'failure_'+case)['phase'], 'rejected')
                self.assertEqual(active.attention, original); self.assertIsNone(active.fatal)
                self.assertEqual(self.rt.closes, 0); self.assertEqual(self.science.calls, [])

    def test_fatal_or_missing_keeper_never_resumes_and_writes_rejection(self):
        active = self.active(); active.attention = dict(error='technical failure')
        self.resume(active, 'fatal'); active.fatal = dict(scope='driver_failure')
        self.assertEqual(active.step()['phase'], 'fatal_attention')
        self.assertEqual(self.resume_receipt(active, 'fatal')['phase'], 'rejected')
        active.fatal = None; self.resume(active, 'gone'); self.rt.live = False
        with self.assertRaises(session.KeeperFailure):
            active.step()
        self.assertEqual(self.resume_receipt(active, 'gone')['phase'], 'rejected')
        self.assertIsNotNone(active.attention); self.assertEqual(self.rt.closes, 0)

    def test_consumed_request_is_once_only_and_changed_bytes_cannot_be_reused(self):
        active = self.active(); active.attention = dict(error='technical failure')
        request = self.resume(active)
        self.assertEqual(active.step()['phase'], 'idle_holding')
        receipt_path = active.output/'resume_receipts/request1.json'; original_receipt = receipt_path.read_bytes()
        scans = self.rt.events.count('scan_sole_keeper')
        active.attention = dict(error='next technical failure')
        self.assertEqual(active.step()['phase'], 'attention_holding')
        self.assertEqual(self.rt.events.count('scan_sole_keeper'), scans)
        value = session.read(request); value['attention_sha256'] = session.sha(json.dumps(active.attention, sort_keys=True).encode())
        write(request, value)
        self.assertEqual(active.step()['phase'], 'attention_holding')
        self.assertEqual(receipt_path.read_bytes(), original_receipt)
        audit = list((active.output/'resume_receipts').glob('changed_*.json'))
        self.assertEqual(len(audit), 1); self.assertEqual(session.read(audit[0])['phase'], 'rejected')
        self.assertEqual(active.step()['phase'], 'attention_holding')
        self.assertEqual(len(list((active.output/'resume_receipts').glob('*.json'))), 2)
        self.assertEqual(self.rt.events.count('scan_sole_keeper'), scans)

    def test_long_consumed_request_id_still_gets_bounded_changed_bytes_audit(self):
        active = self.active(); active.attention = dict(error='technical failure')
        name = 'r'*200; request = self.resume(active, name)
        self.assertEqual(active.step()['phase'], 'idle_holding')
        receipt_path = active.output/'resume_receipts'/(name+'.json')
        original = receipt_path.read_bytes()
        active.attention = dict(error='next technical failure')
        changed = session.read(request); changed['reason'] = 'changed consumed request bytes'
        write(request, changed)
        self.assertEqual(active.step()['phase'], 'attention_holding')
        self.assertEqual(receipt_path.read_bytes(), original)
        audits = list((active.output/'resume_receipts').glob('changed_*.json'))
        self.assertEqual(len(audits), 1)
        self.assertLessEqual(len(audits[0].name.encode()), 255)
        audit = session.read(audits[0])
        self.assertEqual(audit['phase'], 'rejected'); self.assertEqual(audit['request_id'], name)
        self.assertEqual(audit['request_sha256'], session.sha(request.read_bytes()))
        self.assertEqual(active.step()['phase'], 'attention_holding')
        self.assertEqual(len(list((active.output/'resume_receipts').glob('*.json'))), 2)
        self.assertIsNone(active.fatal); self.assertEqual(self.rt.closes, 0)

    def test_request_mutation_during_scan_and_receipt_io_failure_preserve_hold(self):
        active = self.active(); active.attention = dict(error='technical failure')
        request = self.resume(active, 'mutated')
        def scan(*args):
            value = session.read(request); value['reason'] = 'changed during verification'; write(request, value)
            return dict(consistent=True)
        with patch.object(self.rt, 'scan_owned', side_effect=scan):
            self.assertEqual(active.step()['phase'], 'attention_holding')
        self.assertEqual(self.resume_receipt(active, 'mutated')['phase'], 'rejected')
        self.resume(active, 'io_failure')
        with patch.object(self.rt, 'write_resume_receipt', side_effect=OSError('disk full')):
            self.assertEqual(active.step()['phase'], 'attention_holding')
        self.assertFalse((active.output/'resume_receipts/io_failure.json').exists())
        self.assertIsNone(active.fatal); self.assertEqual(self.rt.closes, 0)

    def test_malformed_or_symlink_resume_is_rejected_and_receipt_is_append_only(self):
        active = self.active(); active.attention = dict(error='technical failure')
        malformed = self.resume(active, 'malformed'); malformed.write_text('{not JSON')
        outside = self.root/'foreign.json'; write(outside, {})
        linked = malformed.parent/'foreign.json'; linked.symlink_to(outside)
        self.assertEqual(active.step()['phase'], 'attention_holding')
        self.assertEqual(self.resume_receipt(active, 'malformed')['phase'], 'rejected')
        self.assertEqual(self.resume_receipt(active, 'foreign')['phase'], 'rejected')
        receipt_path = active.output/'resume_receipts/malformed.json'; before = receipt_path.read_bytes()
        with self.assertRaises(FileExistsError):
            self.rt.write_resume_receipt(receipt_path, dict(changed=True))
        self.assertEqual(receipt_path.read_bytes(), before)

    def test_missing_session_source_still_records_exact_rejected_request_hash(self):
        active = self.active(); active.attention = dict(error='technical failure')
        request = self.resume(active); request_hash = session.sha(request.read_bytes())
        original_read = Path.read_bytes
        def missing_source(path):
            if path == Path(session.__file__):
                raise FileNotFoundError('trusted session source disappeared')
            return original_read(path)
        with patch.object(Path, 'read_bytes', missing_source):
            self.assertEqual(active.step()['phase'], 'attention_holding')
        receipt = self.resume_receipt(active)
        self.assertEqual(receipt['phase'], 'rejected')
        self.assertEqual(receipt['request_sha256'], request_hash)
        self.assertFalse(receipt['source_proof']['verified'])
        self.assertIsNone(active.fatal); self.assertEqual(self.rt.closes, 0)

    def test_keeper_control_writer_is_not_inherited_by_other_children_and_eof_reaches_keeper(self):
        read_fd, write_fd = os.pipe()
        child = subprocess.Popen([sys.executable, '-c', 'import os,sys;sys.exit(0 if os.read(int(sys.argv[1]),1)==b"" else 7)', str(read_fd)],
                                 pass_fds=(read_fd,), close_fds=True)
        os.close(read_fd)
        # A server-style child receives neither keeper writer nor scheduler FD.
        probe = subprocess.run([sys.executable, '-c', 'import os,sys\ntry:os.fstat(int(sys.argv[1]))\nexcept OSError:sys.exit(0)\nsys.exit(9)', str(write_fd)],
                               close_fds=True, pass_fds=(), timeout=5)
        self.assertEqual(probe.returncode, 0)
        os.close(write_fd); self.assertEqual(child.wait(timeout=5), 0)


class RunnerFake:
    MODEL_PYTHON = '/absolute/python'
    def __init__(self, root):
        self.root = root; self.calls = []; self.complete = False; self.bad_health = False; self.bad_cross = False
        self.manifest = dict(manifest_hash='manifest', conditions=[dict(id='full_d10')])
    def read_manifest(self, path, verify_sources=False): self.calls.append(('manifest', verify_sources)); return self.manifest
    def validate_output(self, output, manifest): self.calls.append('validate_output')
    def planned_config(self, manifest, condition): self.calls.append('planned_config')
    def stage_for(self, manifest): return 'discovery'
    def cases(self, manifest, stage, shard, shards): self.calls.append(('cases', stage, shard, shards)); return [dict(case=0)]
    def run(self, args):
        assert args.launch is False
        self.calls.append('pure_command_builder')
        return dict(server=['server', '--output', str(self.root/'full_d10/runs'/('fake_'+str(len(self.calls))))], worker=['worker'])
    def recover_case(self, *args): self.calls.append('recover_case')
    def completed(self, *args): return self.complete
    def finalize(self, *args): pass
    def resolve_binding(self, uuid): self.calls.append('resolve_binding'); return dict(egl_index=3)
    @contextmanager
    def lock(self, path, blocking=True): self.calls.append(('shard_lock', blocking)); yield
    def environment(self): return dict(PYTHONPATH='frozen', SCIENTIFIC_ENV='preserved')
    def runtime_revision(self, manifest): return dict(frozen=True)
    def atomic_json(self, path, value): write(path, value)
    def rpc(self, port, request):
        self.calls.append('health')
        return dict(protocol=dict(server_pid=201, manifest_hash='manifest', condition_id='full_d10', gpu_uuid=UUID,
                                  real_input_parity_passed=not self.bad_health))
    def cross_service(self, *args):
        self.calls.append('cross_service')
        if self.bad_cross: raise RuntimeError('cross-service gate failed')
        self.complete = True


class ScienceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.r = RunnerFake(self.root); self.science = session.Science(self.r, self.r); self.rt = FakeRuntime(self.root)
        args = argparse.Namespace(output=str(self.root/'session'), gpu_uuid=UUID, port=12345)
        self.active = session.Session(args, dict(campaign_root=str(self.root), _sha256='plan'), self.rt, self.science)
        self.active.keeper = self.rt.start_keeper(UUID, self.root); self.active.binding = self.rt.bind_scheduler(UUID)
        self.task = dict(schema=session.TASK_SCHEMA, id='t1', logical_job_id='original', env={},
            argv=['/absolute/python', '-m', session.identity.LOCAL_RUNNER, '--output', str(self.root), '--condition', 'full_d10',
                  '--gpu-uuid', '{gpu_uuid}', '--port', '{port}', '--launch'], output=str(self.root), condition='full_d10', stage='discovery',
            shard_index=0, shards=1, manifest_hash='manifest', complete_file=str(self.root/'full_d10/discovery_shard00of01_complete.json'),
            activation_file=str(self.root/'transactions/tx/committed.json'))
        self.plan = dict(allowed_outputs=[str(self.root)])

    def tearDown(self): self.temp.cleanup()

    def prepare(self): return self.science.prepare(self.task, self.plan, UUID, 12345)

    def test_frozen_checks_commands_env_health_cross_and_complete_are_preserved(self):
        prepared = self.prepare(); result = self.science.execute(prepared, self.active)
        self.assertTrue(result['complete'])
        for item in [('manifest', True), 'validate_output', 'planned_config', ('cases', 'discovery', 0, 1),
                     'pure_command_builder', 'recover_case', ('shard_lock', False), 'resolve_binding', 'health', 'cross_service']:
            self.assertIn(item, self.r.calls)
        self.assertEqual([argv for _, argv, _ in self.rt.children], [prepared['server'], prepared['worker']])
        for child, argv, env in self.rt.children:
            self.assertEqual(env['SCIENTIFIC_ENV'], 'preserved'); self.assertEqual(env['CUDA_VISIBLE_DEVICES'], UUID)
        self.assertEqual(self.rt.events[-2:], ['cleanup_202', 'cleanup_201'])
        identity = session.read(prepared['logs']/'identity.json')
        self.assertTrue(identity['technical_runtime_revision']['frozen'])
        self.assertIn('resource_execution_revision', identity)

    def test_already_valid_cases_skip_gpu_server_and_worker(self):
        self.r.complete = True
        result = self.science.execute(self.prepare(), self.active)
        self.assertEqual(result['already_completed'], 1); self.assertEqual(self.rt.children, [])
        self.assertNotIn('resolve_binding', self.r.calls)

    def test_bad_scientific_metadata_and_argv_are_rejected(self):
        for key, value in [('manifest_hash', 'changed'), ('condition', 'unknown'), ('stage', 'confirm'), ('shards', 2), ('env', {'X': 'changed'})]:
            with self.subTest(key=key):
                saved = self.task[key]; self.task[key] = value
                with self.assertRaises(RuntimeError): self.prepare()
                self.task[key] = saved
        self.assertEqual(self.rt.children, [])

    def test_health_or_cross_service_failure_never_launches_worker(self):
        for field in ['bad_health', 'bad_cross']:
            with self.subTest(field=field):
                setattr(self.r, field, True); self.rt.children.clear()
                with self.assertRaises(RuntimeError): self.science.execute(self.prepare(), self.active)
                self.assertEqual(len(self.rt.children), 1)
                self.assertEqual(self.rt.events[-1], 'cleanup_201')
                setattr(self.r, field, False)

    def test_worker_incomplete_is_not_marked_complete(self):
        self.r.cross_service = lambda *args: None
        with self.assertRaisesRegex(RuntimeError, 'all assigned cases'):
            self.science.execute(self.prepare(), self.active)
        self.assertFalse(Path(self.task['complete_file']).exists())

    def test_frozen_manifest_source_failure_prevents_all_children(self):
        self.r.read_manifest = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('Frozen implementation source changed'))
        with self.assertRaisesRegex(ValueError, 'source changed'):
            self.prepare()
        self.assertEqual(self.rt.children, [])


class ResourceBoundaryTests(unittest.TestCase):
    def test_assigned_scan_is_bounded_and_never_equates_host_pid_to_local_pid(self):
        runtime = session.Runtime()
        with patch.object(runtime, 'keeper_live', return_value=True), patch.object(session.subprocess, 'check_output',
                side_effect=[f'{UUID}, 999999, 80\n', f'{UUID}, 100\n']) as query:
            result = runtime.scan_owned(UUID, {}, 512)
        self.assertEqual(result['compute_pid_namespace'], 'unresolved')
        for item in query.call_args_list:
            self.assertIn('--id='+UUID, item.args[0])
        with patch.object(runtime, 'keeper_live', return_value=True), patch.object(session.subprocess, 'check_output',
                return_value=f'{UUID}, 1, 80\n{UUID}, 2, 90\n'):
            with self.assertRaises(session.KeeperFailure):
                runtime.scan_owned(UUID, {}, 512)

    def test_keeper_handshake_checks_nonce_identity_and_passes_only_two_pipe_fds(self):
        with tempfile.TemporaryDirectory() as root:
            runtime = session.Runtime(); child = Child(100); own = dict(pid=99, start_ticks='10')
            child_identity = {}
            def spawn(argv, **kwargs):
                ready_fd = int(argv[argv.index('--ready-fd')+1]); control_fd = int(argv[argv.index('--control-fd')+1])
                self.assertEqual(kwargs['pass_fds'], (control_fd, ready_fd)); self.assertTrue(kwargs['close_fds'])
                child_identity.update(pid=100, start_ticks='20', ppid=99, uid=os.getuid(), state='S', argv=argv)
                receipt = dict(schema='ease-gpu-owned-keeper-v1', phase='ready', gpu_uuid=UUID, pid=100,
                    proc_start_ticks='20', nonce=argv[argv.index('--nonce')+1], context_created=True,
                    allocated_bytes=session.ALLOCATION_BYTES, total_memory_bytes=16*1024**3, driver_version=12040)
                os.write(ready_fd, (json.dumps(receipt)+'\n').encode()); return child
            with patch.object(runtime, 'self_identity', return_value=own), patch.object(runtime, 'identity', side_effect=lambda pid: copy.deepcopy(child_identity)), \
                 patch.object(session.subprocess, 'Popen', side_effect=spawn), patch.object(runtime, 'cleanup_child'):
                keeper = runtime.start_keeper(UUID, root)
                self.assertEqual(keeper['ready']['pid'], 100)
                child.wait = lambda timeout=None: setattr(child, 'returncode', 0) or 0
                runtime.close_keeper(keeper)


if __name__ == '__main__':
    unittest.main()

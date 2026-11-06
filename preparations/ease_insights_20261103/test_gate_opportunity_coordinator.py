"""CPU only: temporary artifacts, fake controllers/GPUs/pidfds/helper boundary."""
import copy
import json
import os
import signal
from pathlib import Path
import unittest
from unittest.mock import patch

from preparations.ease_insights_20261103 import gate_opportunity_coordinator as c
from preparations.ease_insights_20261103 import reassign_waiting_gate_v2 as h
from preparations.ease_insights_20261103 import test_reassign_waiting_gate_v2 as fixtures


class FakeBackend(c.Backend):
    def __init__(self, fixture):
        super().__init__(fixture.root, fixture.runtime)
        self.calls = []
        self.runtime.executable = lambda pid: copy.deepcopy(c.SYSTEM_EXECUTABLE)
        self.lease_calls = []
        self.hash_valid = True
        self.gpu_override = None
        self.before_invoke = None
        self.after_invoke = None

    def verify_helper(self):
        c.require(self.hash_valid, 'Audited helper hash changed')

    def gpus(self):
        if self.gpu_override is not None:
            return copy.deepcopy(self.gpu_override)
        return {uuid: self.runtime.gpu(uuid) for uuid in (fixtures.SOURCE, fixtures.CANDIDATE)}

    def invoke(self, pending, apply):
        self.verify_helper()
        self.calls.append((pending['audit'], apply))
        if self.before_invoke:
            self.before_invoke(pending, apply)
        result = super().invoke(pending, apply)
        if self.after_invoke:
            self.after_invoke(pending, apply, result)
        return result


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.MigrationTests()
        self.fixture.setUp()
        self.fixture.runtime.processes[self.fixture.scheduler_pid]['argv'].extend(['--idle-seconds', '0'])
        self.addCleanup(self.fixture.tearDown)
        self.b = FakeBackend(self.fixture)
        self.coordinator = c.Coordinator(self.b)

    def read_state(self):
        return self.b.read(self.b.output/'state.json')

    def restart(self):
        self.coordinator = c.Coordinator(self.b, self.read_state())
        return self.coordinator

    def pause_after_dryrun(self):
        def crash(pending, apply, audit):
            if not apply:
                raise SystemExit('simulated process loss')
        self.b.after_invoke = crash
        with self.assertRaises(SystemExit):
            self.coordinator.step()
        self.b.after_invoke = None
        return self.read_state()['pending']['audit']

    def test_default_inspect_does_not_write_or_invoke_helper_or_probe_lease(self):
        with (patch.object(c, 'Backend', return_value=self.b), patch.object(self.b, 'write', side_effect=AssertionError('write')),
              patch.object(self.b.runtime, 'lease_available', side_effect=AssertionError('lease'))):
            self.assertEqual(c.main([]), 0)
        self.assertEqual(self.b.calls, [])
        self.assertFalse(self.b.output.exists())

    def test_success_uses_single_audit_and_original_scheduler(self):
        queue_before = copy.deepcopy(self.fixture.queued)
        self.assertEqual(self.coordinator.step(), 'complete')
        self.assertEqual([apply for _, apply in self.b.calls], [False, True])
        self.assertEqual(len({audit for audit, _ in self.b.calls}), 1)
        state = self.read_state()
        self.assertIsNone(state['pending'])
        self.assertEqual(state['controllers']['scheduler']['pid'], self.fixture.scheduler_pid)
        proposal = self.b.read(self.b.queue)
        self.assertEqual(proposal['jobs'][0]['complete_file'], queue_before['jobs'][0]['complete_file'])
        self.assertEqual(proposal['jobs'][0]['launch_gate'], queue_before['jobs'][0]['launch_gate'])

    def test_actual_runner_and_unrelated_jobs_are_ignored(self):
        self.fixture.runtime.processes[self.fixture.gate_pid]['argv'] = self.fixture.original_argv
        self.assertEqual(self.coordinator.step(), 'waiting')
        self.assertEqual(self.b.calls, [])
        self.assertEqual(self.fixture.runtime.signals, [])

    def test_unrelated_gated_module_is_ignored(self):
        queue = self.b.read(self.b.queue)
        queue['jobs'][0]['argv'][-1] = 'irrelevant'
        index = queue['jobs'][0]['argv'].index(h.LOCAL_RUNNER)
        queue['jobs'][0]['argv'][index] = h.LEGACY_DISCOVERY_RUNNER
        fixtures.write(self.b.queue, queue)
        self.assertEqual(self.coordinator.step(), 'waiting')
        self.assertEqual(self.b.calls, [])

    def test_busy_and_leased_candidate_no_attempt(self):
        for attr in ('target_idle', 'lease_free'):
            with self.subTest(attr=attr):
                setattr(self.fixture.runtime, attr, False)
                self.assertEqual(self.coordinator.step(), 'waiting')
                setattr(self.fixture.runtime, attr, True)
        self.assertEqual(self.b.calls, [])

    def test_assigned_candidate_is_ignored(self):
        states = self.b.read(self.b.states)
        states['unrelated_active'] = dict(status='running', gpu_uuid=fixtures.CANDIDATE)
        fixtures.write(self.b.states, states)
        self.assertEqual(self.coordinator.step(), 'waiting')
        self.assertEqual(self.b.calls, [])

    def test_stale_inventory_and_stale_receipt_never_apply(self):
        self.b.gpu_override = self.b.gpus()
        self.b.gpu_override[fixtures.CANDIDATE]['checked_at'] -= 11
        self.assertEqual(self.coordinator.step(), 'needs_attention')
        self.assertEqual(self.b.calls, [])
        self.b.gpu_override = None
        self.coordinator = c.Coordinator(self.b)
        os.utime(self.fixture.receipt, (self.b.now()-30, self.b.now()-30))
        self.assertEqual(self.coordinator.step(), 'waiting')
        self.assertEqual(self.b.calls, [])

    def test_source_already_idle_is_ignored(self):
        self.fixture.runtime.source_idle = True
        self.assertEqual(self.coordinator.step(), 'waiting')
        self.assertEqual(self.b.calls, [])

    def test_oldest_eligible_gate_selected(self):
        queue, states = self.b.read(self.b.queue), self.b.read(self.b.states)
        newer = copy.deepcopy(queue['jobs'][0]); newer['id'] = 'newer'
        newer['argv'][6] = str(self.b.scheduler/'launch_gates/newer.json')
        newer_state = copy.deepcopy(states['selected']); newer_state.update(pid=700004, started=self.fixture.started+10)
        newer_state['argv'] = h.concrete(newer['argv'], newer_state)
        queue['jobs'].insert(0, newer); states['newer'] = newer_state
        self.fixture.runtime.processes[700004] = dict(self.fixture.runtime.processes[self.fixture.gate_pid],
            pid=700004, argv=newer_state['argv'])
        fixtures.write(self.b.queue, queue); fixtures.write(self.b.states, states)
        self.fixture.runtime.matches_started = lambda process, started: started == self.fixture.started or (process['pid'] == 700004 and started == self.fixture.started+10)
        self.assertEqual(self.coordinator.step(), 'complete')
        self.assertEqual(self.b.read(Path(self.b.calls[0][0])/'audit.json')['old_job_id'], 'selected')

    def test_plan_integrity_refuses_modified_science_before_apply(self):
        for target in ('complete_file', 'ready_files', 'depends_on', 'scientific_seed', 'argv', 'profile', 'other_job'):
            with self.subTest(target=target):
                self.coordinator = c.Coordinator(self.b)
                def mutate(pending, apply, audit):
                    if apply:
                        return
                    if target == 'profile':
                        audit['profile']['output'] = '/wrong'
                    elif target == 'other_job':
                        path = Path(pending['audit'])/'queue.proposed.json'
                        value = self.b.read(path); value['jobs'][1]['complete_file'] = '/wrong'
                        fixtures.write(path, value); audit['queue_after_sha256'] = c.sha(path.read_bytes())
                    else:
                        audit['new_job'][target] = ['/wrong'] if target in ('ready_files', 'depends_on', 'argv') else '/wrong'
                self.b.after_invoke = mutate
                self.assertEqual(self.coordinator.step(), 'needs_attention')
                self.assertFalse(any(apply for _, apply in self.b.calls))
        self.assertEqual(self.fixture.runtime.signals, [])

    def test_crash_after_dryrun_resumes_same_audit(self):
        audit = self.pause_after_dryrun()
        self.assertEqual(self.restart().step(), 'complete')
        self.assertEqual([apply for _, apply in self.b.calls], [False, False, True])
        self.assertEqual({path for path, _ in self.b.calls}, {audit})

    def test_crash_before_apply_revalidates_same_audit(self):
        def crash(pending, apply):
            if apply:
                raise SystemExit('simulated crash before helper apply')
        self.b.before_invoke = crash
        with self.assertRaises(SystemExit):
            self.coordinator.step()
        audit = self.read_state()['pending']['audit']
        self.assertEqual(self.read_state()['pending']['phase'], 'applying')
        self.b.before_invoke = None
        self.assertEqual(self.restart().step(), 'complete')
        self.assertEqual({path for path, _ in self.b.calls}, {audit})

    def test_stop_before_apply_cancels_without_helper_apply(self):
        audit = self.pause_after_dryrun()
        self.b.output.joinpath('stop').touch()
        self.assertEqual(self.restart().step(), 'cancelled_before_apply')
        self.assertEqual(self.coordinator.step(), 'stopped')
        self.assertIsNone(self.read_state()['pending'])
        self.assertEqual(self.b.calls, [(audit, False)])

    def test_interruption_after_dryrun_retries_same_audit_once(self):
        def interrupt(pending, apply, audit):
            if not apply:
                raise InterruptedError('interrupted dry-run')
        self.b.after_invoke = interrupt
        self.assertEqual(self.coordinator.step(), 'recovery_pending')
        audit = self.read_state()['pending']['audit']
        self.assertEqual(self.restart().step(), 'needs_attention')
        self.assertEqual({path for path, _ in self.b.calls}, {audit})
        self.assertFalse(any(apply for _, apply in self.b.calls))

    def test_once_finishes_pending_recovery_before_returning(self):
        original_sleep = self.fixture.runtime.sleep
        seen = [False]
        def interrupt(seconds):
            if not seen[0] and any('__wait_reassign_' in j['id'] for j in self.b.read(self.b.queue)['jobs']):
                seen[0] = True
                raise InterruptedError('interrupted after commit')
            original_sleep(seconds)
        self.fixture.runtime.sleep = interrupt
        original_identity = self.b.runtime.identity
        self.b.runtime.identity = lambda pid: dict(pid=pid, start_ticks='123') if pid == os.getpid() else original_identity(pid)
        with patch.object(c, 'Backend', return_value=self.b):
            self.assertEqual(c.main(['--launch', '--once']), 0)
        self.assertIsNone(self.read_state()['pending'])
        self.assertEqual(self.read_state()['phase'], 'complete')

    def test_postcommit_interruption_resumes_same_audit(self):
        original_sleep = self.fixture.runtime.sleep
        interrupted = [False]
        def crash(seconds):
            if not interrupted[0] and any('__wait_reassign_' in j['id'] for j in self.b.read(self.b.queue)['jobs']):
                interrupted[0] = True
                raise InterruptedError('simulated interruption')
            original_sleep(seconds)
        self.fixture.runtime.sleep = crash
        self.assertEqual(self.coordinator.step(), 'recovery_pending')
        pending = self.read_state()['pending']
        self.assertEqual(self.fixture.runtime.processes[self.fixture.gate_pid]['state'], 'T')
        self.assertEqual(self.restart().step(), 'complete')
        self.assertEqual({path for path, _ in self.b.calls}, {pending['audit']})
        self.assertEqual([apply for _, apply in self.b.calls], [False, True, True])

    def test_precommit_stop_interruption_reuses_helper_restore(self):
        def stop_crash():
            raise SystemExit('killed fake process')
        self.fixture.runtime.on_stop = stop_crash
        # Helper restores even BaseException; same audit remains durable.
        with self.assertRaises(SystemExit):
            self.coordinator.step()
        self.fixture.runtime.on_stop = None
        self.assertEqual(self.restart().step(), 'aborted_restored')
        self.assertIsNone(self.read_state()['pending'])

    def test_resource_refusal_safe_cooldown_no_audit_spam(self):
        def busy(pending, apply):
            if apply:
                self.fixture.runtime.target_idle = False
        self.b.before_invoke = busy
        self.assertEqual(self.coordinator.step(), 'resource_lost')
        calls = list(self.b.calls)
        self.assertEqual(self.coordinator.step(), 'cooldown')
        self.assertEqual(self.b.calls, calls)
        self.assertIsNone(self.read_state()['pending'])
        self.assertEqual(self.fixture.runtime.signals, [])

    def test_unknown_and_identity_errors_preserve_attention_without_retry(self):
        for message in ('Recovery PID identity changed; no signals sent', 'arbitrary Refused error'):
            self.coordinator = c.Coordinator(self.b)
            def reject(pending, apply):
                if apply:
                    raise h.Refused(message)
            self.b.before_invoke = reject
            self.assertEqual(self.coordinator.step(), 'needs_attention')
            self.assertIsNotNone(self.read_state()['pending'])
            before = len(self.b.calls)
            self.assertEqual(self.restart().step(), 'needs_attention')
            self.assertEqual(len(self.b.calls), before)

    def test_unknown_commit_never_retries_or_starts_second_transaction(self):
        self.fixture.runtime.replace_mode = 'ambiguous'
        self.assertEqual(self.coordinator.step(), 'needs_attention')
        pending = self.read_state()['pending']
        self.assertEqual(self.b.read(Path(pending['audit'])/'audit.json')['commit_outcome'], 'unknown')
        before = len(self.b.calls)
        self.assertEqual(self.restart().step(), 'needs_attention')
        self.assertEqual(len(self.b.calls), before)
        self.assertEqual(len(list((self.b.root/'gate_reassignments').iterdir())), 1)

    def test_helper_hash_change_refuses_even_recovery(self):
        self.pause_after_dryrun()
        self.b.hash_valid = False
        self.assertEqual(self.restart().step(), 'needs_attention')
        self.assertEqual([apply for _, apply in self.b.calls], [False])

    def test_stop_marker_and_terminal_do_not_abandon_postcommit_cleanup(self):
        self.fixture.runtime.auto_ack = False
        self.assertEqual(self.coordinator.step(), 'recovery_pending')
        pending = self.read_state()['pending']
        self.b.output.joinpath('stop').touch()
        self.fixture.runtime.auto_ack = True
        self.assertEqual(self.restart().step(), 'complete')
        self.assertEqual(self.coordinator.step(), 'stopped')
        self.assertEqual({path for path, _ in self.b.calls}, {pending['audit']})

    def test_terminal_pending_attempts_cleanup_and_surfaces_controller_refusal(self):
        self.fixture.runtime.auto_ack = False
        self.assertEqual(self.coordinator.step(), 'recovery_pending')
        fixtures.write(self.b.supervisor/'status.json', dict(terminal=True))
        self.assertEqual(self.restart().step(), 'needs_attention')
        self.assertEqual([apply for _, apply in self.b.calls], [False, True, True])
        self.assertIsNotNone(self.read_state()['pending'])

    def test_natural_terminal_or_stop_without_pending_does_not_apply(self):
        fixtures.write(self.b.supervisor/'status.json', dict(terminal=True))
        self.assertEqual(self.coordinator.step(), 'supervisor_terminal')
        self.assertEqual(self.b.calls, [])

    def test_singleton_rejects_second_owner_and_receipt_has_start_identity(self):
        original_identity = self.b.runtime.identity
        self.b.runtime.identity = lambda pid: dict(pid=pid, start_ticks='123') if pid == os.getpid() else original_identity(pid)
        with c.singleton(self.b):
            with self.assertRaises(BlockingIOError):
                with c.singleton(self.b):
                    self.fail('second owner')
            self.assertEqual(self.b.read(self.b.output/'process.json')['start_ticks'], '123')

    def test_changed_scheduler_identity_requires_attention(self):
        self.fixture.runtime.target_idle = False
        self.assertEqual(self.coordinator.step(), 'waiting')
        self.fixture.runtime.processes[self.fixture.scheduler_pid]['start_ticks'] = 'changed'
        self.assertEqual(self.coordinator.step(), 'needs_attention')
        self.assertEqual(self.b.calls, [])

    def test_toctou_scientific_and_unrelated_queue_changes_never_commit_or_term(self):
        original_queue = copy.deepcopy(self.fixture.queued)
        for index, target in enumerate(('scientific_seed', 'env', 'ready_files', 'depends_on', 'unrelated_job')):
            with self.subTest(target=target):
                fixtures.write(self.b.queue, original_queue)
                self.fixture.runtime.signals.clear()
                self.fixture.runtime.clock += 1
                self.coordinator = c.Coordinator(self.b)
                def mutate(pending, apply):
                    if not apply:
                        return
                    queue = self.b.read(self.b.queue)
                    if target == 'scientific_seed':
                        queue['jobs'][0]['scientific_seed'] = 999
                    elif target == 'env':
                        queue['jobs'][0]['env']['SCIENCE_CHANGED'] = '1'
                    elif target == 'ready_files':
                        queue['jobs'][0]['ready_files'].append('/unexpected_ready')
                    elif target == 'depends_on':
                        queue['jobs'][0]['depends_on'].append('unexpected_dependency')
                    else:
                        queue['jobs'][1]['complete_file'] = '/changed_unrelated_marker'
                    fixtures.write(self.b.queue, queue)
                self.b.before_invoke = mutate
                self.assertEqual(self.coordinator.step(), 'needs_attention')
                self.assertEqual([job['id'] for job in self.b.read(self.b.queue)['jobs']], ['selected', 'dependent'])
                self.assertEqual([sig for _, sig in self.fixture.runtime.signals], [signal.SIGSTOP, signal.SIGCONT])
                self.assertEqual(self.fixture.runtime.processes[self.fixture.gate_pid]['state'], 'S')
                pending = self.read_state()['pending']
                self.assertEqual(json.loads(pending['reviewed']['before']), original_queue)
                self.assertEqual(self.b.read(Path(pending['audit'])/'audit.json')['stage'], 'aborted_restored')

    def test_reviewed_snapshot_survives_restart_and_helper_proposal_overwrite(self):
        def crash(pending, apply):
            if apply:
                raise SystemExit('crash before apply')
        self.b.before_invoke = crash
        with self.assertRaises(SystemExit):
            self.coordinator.step()
        original_reviewed = self.read_state()['pending']['reviewed']
        queue = self.b.read(self.b.queue)
        queue['jobs'][1]['complete_file'] = '/changed_after_crash'
        fixtures.write(self.b.queue, queue)
        self.b.before_invoke = None
        self.assertEqual(self.restart().step(), 'needs_attention')
        self.assertEqual(self.read_state()['pending']['reviewed'], original_reviewed)
        self.assertEqual(self.fixture.runtime.signals, [])

    def test_postcommit_recovery_refuses_changed_helper_proposal_snapshot(self):
        self.fixture.runtime.auto_ack = False
        self.assertEqual(self.coordinator.step(), 'recovery_pending')
        pending = self.read_state()['pending']
        path = Path(pending['audit'])/'queue.proposed.json'
        proposal = self.b.read(path); proposal['jobs'][0]['scientific_seed'] = 999
        fixtures.write(path, proposal)
        original_signals = list(self.fixture.runtime.signals)
        self.assertEqual(self.restart().step(), 'needs_attention')
        self.assertEqual(self.fixture.runtime.signals, original_signals)
        self.assertEqual(self.read_state()['pending']['reviewed'], pending['reviewed'])

    def scheduler_exit(self, keep_live_job=False):
        self.fixture.runtime.processes[self.fixture.scheduler_pid]['state'] = 'absent'
        if not keep_live_job:
            self.fixture.runtime.processes[self.fixture.gate_pid]['state'] = 'absent'
            fixtures.write(self.b.states, {job['id']: dict(status='complete') for job in self.b.read(self.b.queue)['jobs']})

    def scheduler_restart(self, adopted=True, relative=False):
        old = self.fixture.runtime.processes[self.fixture.scheduler_pid]
        new = dict(old, pid=700005, state='S', start_ticks='new_scheduler_ticks', argv=list(old['argv']))
        if relative:
            new['argv'][0] = 'python3'
        self.fixture.runtime.processes[700005] = new
        fixtures.write(self.b.scheduler/'controller_process.json', dict(pid=700005, started=self.fixture.started))
        if adopted:
            report = {key: new[key] for key in ('pid', 'start_ticks', 'argv')}
            report.update(status='live', state='S')
            observation = dict(status='live', process=report, processes=[report])
            fixtures.write(self.b.supervisor/'state.json', dict(phase='waiting_confirmation', updated_unix=self.b.now(),
                scheduler=dict(report, last_verified_unix=self.b.now()), scheduler_observation=observation))
            fixtures.write(self.b.supervisor/'status.json', dict(phase='waiting_confirmation', terminal=False,
                time=self.b.now(), scheduler=observation))
        return new

    def test_exited_to_missing_receipt_to_unadopted_to_adopted_scheduler(self):
        supervisor = self.fixture.runtime.processes[self.fixture.pipeline_pid]
        supervisor['argv'].extend(['--scheduler-pid', str(self.fixture.scheduler_pid)])
        self.fixture.runtime.target_idle = False
        self.assertEqual(self.coordinator.step(), 'waiting')
        original = self.read_state()['controllers']['scheduler']
        self.scheduler_exit()
        self.assertEqual(self.coordinator.step(), 'waiting_scheduler_adoption')
        (self.b.scheduler/'controller_process.json').unlink()
        self.assertEqual(self.restart().step(), 'waiting_scheduler_adoption')
        self.scheduler_restart(adopted=False)
        self.assertEqual(self.coordinator.step(), 'waiting_scheduler_adoption')
        self.assertEqual(self.read_state()['controllers']['scheduler'], original)
        new = self.scheduler_restart(adopted=True)
        self.assertEqual(self.coordinator.step(), 'waiting')
        self.assertEqual(self.read_state()['controllers']['scheduler']['pid'], new['pid'])
        self.assertEqual(self.coordinator.step(), 'waiting')
        self.assertEqual(self.b.calls, [])

    def test_old_live_unknown_or_live_job_cannot_be_treated_as_idle_gap(self):
        self.fixture.runtime.target_idle = False
        self.assertEqual(self.coordinator.step(), 'waiting')
        self.scheduler_restart()
        self.assertEqual(self.coordinator.step(), 'needs_attention')
        self.assertEqual(self.b.calls, [])

    def test_unknown_former_scheduler_blocks_adoption(self):
        self.fixture.runtime.target_idle = False
        self.assertEqual(self.coordinator.step(), 'waiting')
        self.fixture.runtime.processes[self.fixture.scheduler_pid]['state'] = '?'
        self.scheduler_restart()
        self.assertEqual(self.coordinator.step(), 'needs_attention')

    def test_exited_scheduler_with_live_gpu_job_is_attention(self):
        self.fixture.runtime.target_idle = False
        self.assertEqual(self.coordinator.step(), 'waiting')
        self.scheduler_exit(keep_live_job=True)
        self.assertEqual(self.coordinator.step(), 'needs_attention')

    def test_stale_or_incomplete_adoption_waits_without_rebinding(self):
        self.fixture.runtime.target_idle = False
        self.assertEqual(self.coordinator.step(), 'waiting')
        self.scheduler_exit()
        self.scheduler_restart()
        status = self.b.read(self.b.supervisor/'status.json'); status['time'] -= c.ADOPTION_MAX_AGE+1
        fixtures.write(self.b.supervisor/'status.json', status)
        self.assertEqual(self.coordinator.step(), 'waiting_scheduler_adoption')
        self.assertEqual(self.read_state()['controllers']['scheduler']['pid'], self.fixture.scheduler_pid)
        self.scheduler_restart()
        self.assertEqual(self.coordinator.step(), 'waiting')

    def test_replaced_supervisor_is_not_automatically_adopted(self):
        self.fixture.runtime.target_idle = False
        self.assertEqual(self.coordinator.step(), 'waiting')
        self.fixture.runtime.processes[self.fixture.pipeline_pid]['start_ticks'] = 'replacement'
        self.assertEqual(self.coordinator.step(), 'needs_attention')

    def test_pending_transaction_keeps_original_scheduler_identity(self):
        self.fixture.runtime.auto_ack = False
        self.assertEqual(self.coordinator.step(), 'recovery_pending')
        original = self.read_state()['controllers']['scheduler']
        self.scheduler_exit(keep_live_job=True)
        self.scheduler_restart()
        self.assertEqual(self.restart().step(), 'needs_attention')
        self.assertEqual(self.read_state()['controllers']['scheduler'], original)
        self.assertEqual([sig for _, sig in self.fixture.runtime.signals], [signal.SIGSTOP])

    def add_eligible_pending(self):
        queue = self.b.read(self.b.queue)
        queue['jobs'].append(dict(id='eligible_pending', argv=['/unused'], complete_file=str(self.b.root/'pending_complete')))
        fixtures.write(self.b.queue, queue)

    def test_one_gate_with_other_eligible_pending_defers_to_scheduler(self):
        self.add_eligible_pending()
        self.assertEqual(self.coordinator.step(), 'waiting_scheduler_capacity')
        self.assertEqual(self.b.calls, [])
        self.assertEqual(self.fixture.runtime.signals, [])

    def test_one_gate_without_other_eligible_pending_can_migrate(self):
        self.assertEqual(self.coordinator.step(), 'complete')
        self.assertEqual([apply for _, apply in self.b.calls], [False, True])

    def test_two_live_slots_allow_migration_despite_other_eligible_pending(self):
        self.add_eligible_pending()
        queue, states = self.b.read(self.b.queue), self.b.read(self.b.states)
        queue['jobs'].append(dict(id='other_running', argv=['/unused'], complete_file=str(self.b.root/'running_complete')))
        states['other_running'] = dict(status='running', pid=700006, started=self.fixture.started, start_ticks='other')
        self.fixture.runtime.processes[700006] = dict(self.fixture.runtime.processes[self.fixture.gate_pid],
            pid=700006, start_ticks='other', argv=['/unused'])
        fixtures.write(self.b.queue, queue); fixtures.write(self.b.states, states)
        self.assertEqual(self.coordinator.step(), 'complete')
        self.assertEqual([apply for _, apply in self.b.calls], [False, True])

    def test_pending_with_missing_ready_file_does_not_prevent_lone_gate_migration(self):
        self.add_eligible_pending()
        queue = self.b.read(self.b.queue)
        queue['jobs'][-1]['ready_files'] = [str(self.b.root/'missing')]
        fixtures.write(self.b.queue, queue)
        self.assertEqual(self.coordinator.step(), 'complete')

    def test_fixed_allocator_rejects_nonzero_idle_delay_and_restricted_gpus(self):
        original = list(self.fixture.runtime.processes[self.fixture.scheduler_pid]['argv'])
        for flag, value in (('--idle-seconds', '25'), ('--gpus', '0'), ('--max-gpus', '1')):
            self.coordinator = c.Coordinator(self.b)
            argv = list(original)
            if flag in argv:
                argv[argv.index(flag)+1] = value
            else:
                argv.extend([flag, value])
            self.fixture.runtime.processes[self.fixture.scheduler_pid]['argv'] = argv
            self.assertEqual(self.coordinator.step(), 'needs_attention')
            self.assertEqual(self.b.calls, [])

    def adopt_relative_scheduler(self, gate=False):
        self.fixture.runtime.target_idle = False
        self.assertEqual(self.coordinator.step(), 'waiting')
        self.scheduler_exit()
        new = self.scheduler_restart(relative=True)
        self.assertEqual(self.coordinator.step(), 'waiting')
        if gate:
            self.fixture.runtime.processes[self.fixture.gate_pid].update(state='S', ppid=new['pid'])
            fixtures.write(self.b.states, dict(selected=self.fixture.state))
            for name in ('state.json', 'status.json'):
                value = self.b.read(self.b.supervisor/name); value['phase'] = 'waiting_discovery'
                fixtures.write(self.b.supervisor/name, value)
            self.fixture.runtime.target_idle = True
        return new

    def test_actual_relative_python3_scheduler_adoption_preserves_raw_identity(self):
        self.fixture.runtime.processes[self.fixture.pipeline_pid]['argv'].extend(['--scheduler-pid', str(self.fixture.scheduler_pid)])
        new = self.adopt_relative_scheduler()
        identity = self.read_state()['controllers']['scheduler']
        self.assertEqual(new['argv'][0], 'python3')
        self.assertEqual(identity['argv'][0], str(c.SYSTEM_PYTHON))
        self.assertEqual(identity['_scheduler_proof']['raw_argv'], new['argv'])
        self.assertEqual(identity['_scheduler_proof']['executable'], c.SYSTEM_EXECUTABLE)
        self.assertEqual(self.restart().step(), 'waiting')
        self.assertEqual(self.read_state()['controllers']['scheduler']['_scheduler_proof'], identity['_scheduler_proof'])

    def test_relative_adoption_requires_raw_reported_argv(self):
        self.fixture.runtime.target_idle = False
        self.assertEqual(self.coordinator.step(), 'waiting')
        self.scheduler_exit()
        self.scheduler_restart(relative=True)
        for name in ('state.json', 'status.json'):
            value = self.b.read(self.b.supervisor/name)
            if name == 'state.json':
                value['scheduler']['argv'][0] = str(c.SYSTEM_PYTHON)
                value['scheduler_observation']['process']['argv'][0] = str(c.SYSTEM_PYTHON)
            else:
                value['scheduler']['process']['argv'][0] = str(c.SYSTEM_PYTHON)
            fixtures.write(self.b.supervisor/name, value)
        self.assertEqual(self.coordinator.step(), 'waiting_scheduler_adoption')
        self.assertEqual(self.read_state()['controllers']['scheduler']['pid'], self.fixture.scheduler_pid)

    def test_relative_scheduler_can_migrate_and_resume_same_raw_proof(self):
        new = self.adopt_relative_scheduler(gate=True)
        def crash(pending, apply):
            if apply:
                raise SystemExit('before apply with reviewed raw proof')
        self.b.before_invoke = crash
        with self.assertRaises(SystemExit):
            self.coordinator.step()
        pending = self.read_state()['pending']
        self.assertEqual(pending['scheduler']['_scheduler_proof']['raw_argv'], new['argv'])
        self.b.before_invoke = None
        self.assertEqual(self.restart().step(), 'complete')
        audit = self.b.read(Path(pending['audit'])/'audit.json')
        self.assertEqual(audit['scheduler']['_scheduler_proof'], pending['scheduler']['_scheduler_proof'])

    def test_relative_scheduler_wrong_executable_is_refused(self):
        self.fixture.runtime.target_idle = False
        self.assertEqual(self.coordinator.step(), 'waiting')
        self.scheduler_exit(); self.scheduler_restart(relative=True)
        self.fixture.runtime.executable = lambda pid: dict(c.SYSTEM_EXECUTABLE, path='/tmp/python3')
        self.assertEqual(self.coordinator.step(), 'needs_attention')
        self.assertEqual(self.b.calls, [])

    def test_arbitrary_relative_python_and_wrong_module_are_refused(self):
        original = copy.deepcopy(self.fixture.runtime.processes[self.fixture.scheduler_pid])
        for executable, module in (('./python3', h.SCHEDULER), ('python', h.SCHEDULER),
                                   ('python3.10', h.SCHEDULER), ('python3', h.PIPELINE)):
            process = copy.deepcopy(original)
            process['argv'][0] = executable
            process['argv'][process['argv'].index('-m')+1] = module
            self.fixture.runtime.processes[self.fixture.scheduler_pid] = process
            self.coordinator = c.Coordinator(self.b)
            self.assertEqual(self.coordinator.step(), 'needs_attention')
            self.assertEqual(self.b.calls, [])

    def test_raw_relative_to_absolute_change_is_not_hidden_by_normalization(self):
        new = self.adopt_relative_scheduler()
        proof = self.read_state()['controllers']['scheduler']['_scheduler_proof']
        self.fixture.runtime.processes[new['pid']]['argv'][0] = str(c.SYSTEM_PYTHON)
        self.assertEqual(self.restart().step(), 'needs_attention')
        self.assertEqual(self.read_state()['controllers']['scheduler']['_scheduler_proof'], proof)
        self.assertEqual(self.b.calls, [])

    def test_executable_inode_change_after_adoption_is_refused(self):
        self.adopt_relative_scheduler()
        self.fixture.runtime.executable = lambda pid: dict(c.SYSTEM_EXECUTABLE, inode=c.SYSTEM_EXECUTABLE['inode']+1)
        self.assertEqual(self.restart().step(), 'needs_attention')
        self.assertEqual(self.b.calls, [])

    def test_scheduler_before_after_raw_argv_start_and_exe_are_reverified(self):
        original = copy.deepcopy(self.fixture.runtime.processes[self.fixture.scheduler_pid])
        original['argv'][0] = 'python3'
        for changed in ('argv', 'start_ticks', 'executable'):
            self.fixture.runtime.processes[self.fixture.scheduler_pid] = copy.deepcopy(original)
            calls = [0]
            def executable(pid):
                calls[0] += 1
                if calls[0] == 1:
                    if changed == 'argv':
                        self.fixture.runtime.processes[pid]['argv'].insert(1, '-B')
                    elif changed == 'start_ticks':
                        self.fixture.runtime.processes[pid]['start_ticks'] = 'changed'
                return dict(c.SYSTEM_EXECUTABLE, inode=c.SYSTEM_EXECUTABLE['inode']+1) if changed == 'executable' and calls[0] > 1 else copy.deepcopy(c.SYSTEM_EXECUTABLE)
            self.fixture.runtime.executable = executable
            self.coordinator = c.Coordinator(self.b)
            self.assertEqual(self.coordinator.step(), 'needs_attention')
            self.assertEqual(self.b.calls, [])

    def test_pending_relative_recovery_keeps_raw_proof_and_refuses_exec_change(self):
        new = self.adopt_relative_scheduler(gate=True)
        self.fixture.runtime.auto_ack = False
        self.assertEqual(self.coordinator.step(), 'recovery_pending')
        pending = self.read_state()['pending']
        self.fixture.runtime.processes[new['pid']]['argv'][0] = str(c.SYSTEM_PYTHON)
        self.assertEqual(self.restart().step(), 'needs_attention')
        self.assertEqual(self.read_state()['pending']['scheduler']['_scheduler_proof'], pending['scheduler']['_scheduler_proof'])
        self.assertEqual([sig for _, sig in self.fixture.runtime.signals], [signal.SIGSTOP])

    def mixed_legacy_queue(self, running_kind):
        queue, states = self.b.read(self.b.queue), self.b.read(self.b.states)
        scripts = (['python3', '/project/preparations/ease_insights_20261103/draft/run.py', '--gpu', '{gpu}'],
                   ['/usr/bin/python3', '/project/preparations/ease_insights_20261103/draft/recovery_v1/resume_run.py', 'run', '--ledger', '/unused'])
        extras = []
        for index, inner in enumerate(scripts):
            job = dict(id='historical_script_'+str(index), argv=['/usr/bin/python3', '-m', h.GATE,
                '--gpu-uuid', '{gpu_uuid}', '--status-file', '/unused/history_'+str(index)+'.json', '--', *inner],
                complete_file='/unused/completed_'+str(index), depends_on=[])
            extras.append(job)
            states[job['id']] = dict(status='complete', pid=899990+index)
        # Even an invalid historical gate must be skipped before classification.
        extras.append(dict(id='historical_invalid', argv=['/usr/bin/python3', '-m', h.GATE], complete_file='/unused/historical'))
        states['historical_invalid'] = dict(status='complete', pid=899992)
        inner = scripts[0] if running_kind == 'script' else ['/usr/bin/python3', '-m', h.LEGACY_DISCOVERY_RUNNER, '--launch']
        job = dict(id='unrelated_running', argv=['/usr/bin/python3', '-m', h.GATE,
            '--gpu-uuid', '{gpu_uuid}', '--status-file', '/unused/unrelated.json', '--', *inner],
            complete_file='/unused/unrelated_complete', depends_on=[])
        extras.append(job)
        state = dict(status='running', pid=700007, started=self.fixture.started, gpu_uuid=fixtures.SOURCE, start_ticks='unrelated')
        state['argv'] = h.concrete(job['argv'], state)
        states[job['id']] = state
        self.fixture.runtime.processes[state['pid']] = dict(self.fixture.runtime.processes[self.fixture.gate_pid],
            pid=state['pid'], start_ticks='unrelated', argv=state['argv'])
        queue['jobs'] = [*extras, *queue['jobs']]
        fixtures.write(self.b.queue, queue); fixtures.write(self.b.states, states)
        return extras

    def test_default_inspect_mixed_historical_and_running_legacy_scripts(self):
        self.mixed_legacy_queue('script')
        _, _, _, gates = c.inventory(self.b)
        self.assertEqual([job for _, job in gates], ['selected'])
        with (patch.object(c, 'Backend', return_value=self.b), patch.object(self.b, 'write', side_effect=AssertionError('write')),
              patch.object(self.b.runtime, 'lease_available', side_effect=AssertionError('lease'))):
            self.assertEqual(c.main([]), 0)
        self.assertEqual(self.b.calls, [])
        self.assertFalse(self.b.output.exists())

    def test_mixed_script_queue_migrates_only_local_gate_and_preserves_other_jobs(self):
        extras = self.mixed_legacy_queue('script')
        self.assertEqual(self.coordinator.step(), 'complete')
        after = {job['id']: job for job in self.b.read(self.b.queue)['jobs']}
        self.assertTrue(all(after[job['id']] == job for job in extras))
        self.assertTrue(all(pid == self.fixture.gate_pid for pid, _ in self.fixture.runtime.signals))

    def test_mixed_module_queue_migrates_only_local_gate_and_preserves_other_jobs(self):
        extras = self.mixed_legacy_queue('module')
        self.assertEqual(self.coordinator.step(), 'complete')
        after = {job['id']: job for job in self.b.read(self.b.queue)['jobs']}
        self.assertTrue(all(after[job['id']] == job for job in extras))
        self.assertTrue(all(pid == self.fixture.gate_pid for pid, _ in self.fixture.runtime.signals))

    def test_genuinely_local_malformed_gate_is_not_silently_ignored(self):
        original = copy.deepcopy(self.fixture.queued)
        for malformed in ('wrapper', 'relative_inner', 'missing_separator'):
            queue = copy.deepcopy(original)
            argv = queue['jobs'][0]['argv']
            if malformed == 'wrapper':
                argv[3] = '--incorrect-gpu-option'
            elif malformed == 'relative_inner':
                argv[8] = 'python3'
            else:
                argv.remove('--')
            fixtures.write(self.b.queue, queue)
            self.coordinator = c.Coordinator(self.b)
            self.assertEqual(self.coordinator.step(), 'needs_attention')
            self.assertEqual(self.b.calls, [])
            self.assertEqual(self.fixture.runtime.signals, [])

    def test_canonical_source_path_and_hash_pinned(self):
        c.Backend().verify_helper()
        with patch.object(c, 'HELPER_SHA256', 'wrong'):
            with self.assertRaisesRegex(ValueError, 'hash changed'):
                c.Backend().verify_helper()
        with self.assertRaisesRegex(ValueError, 'Noncanonical'):
            c.Backend(self.fixture.root).verify_helper()


if __name__ == '__main__':
    unittest.main()

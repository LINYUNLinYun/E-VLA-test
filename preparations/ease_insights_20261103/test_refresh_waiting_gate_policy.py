"""CPU fake only: never inspect a real PID, signal, GPU, lease, or queue."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import unittest
from unittest.mock import patch

from preparations.ease_insights_20261103 import refresh_waiting_gate_policy as refresh
from preparations.ease_insights_20261103 import test_reassign_waiting_gate_v2 as legacy


class FakeRuntime(legacy.FakeRuntime):
    def executable(self, pid):
        assert pid == self.fixture.scheduler_pid
        return copy.deepcopy(refresh.coordinator.SYSTEM_EXECUTABLE)

    def gpu(self, uuid):
        raise AssertionError('No GPU queries are permitted')

    def lease_available(self, uuid):
        raise AssertionError('No lease probes are permitted')

    def sleep(self, seconds):
        self.clock += seconds
        if not self.auto_ack:
            return
        new = [j for j in refresh.h.read_json(self.fixture.queue)['jobs'] if '__gate_policy_5s2s_' in j['id']]
        if new:
            states = refresh.h.read_json(self.fixture.states)
            states[new[0]['id']] = dict(status='pending')
            legacy.write(self.fixture.states, states)
            if self.pipeline_ack:
                legacy.write(self.fixture.pipeline/'status.json', dict(time=self.clock, phase='waiting_discovery', terminal=False))


class RefreshTests(unittest.TestCase):
    def setUp(self):
        legacy.MigrationTests.setUp(self)
        self.runtime = FakeRuntime(self)
        # Reuse only the existing isolated fixture's process values.
        common = dict(uid=os.getuid(), state='S', children=[], sigterm_default=True)
        self.runtime.processes = {
            self.scheduler_pid: dict(common, pid=self.scheduler_pid, ppid=1, start_ticks='100', argv=[
                'python3', '-u', '-m', refresh.h.SCHEDULER, '--queue', str(self.queue), '--output', str(self.scheduler),
                '--max-gpus', '2', '--scan-seconds', '30', '--idle-seconds', '0']),
            self.pipeline_pid: dict(common, pid=self.pipeline_pid, ppid=1, start_ticks='101', argv=[
                '/absolute/python', '-u', '-m', refresh.h.LOCAL_SUPERVISOR, '--root', str(self.discovery),
                '--scheduler-pid', str(self.scheduler_pid), '--launch']),
            self.gate_pid: dict(common, pid=self.gate_pid, ppid=self.scheduler_pid, start_ticks='111', argv=self.state['argv']),
            700004: dict(pid=700004, state='absent'),
        }
        self.job['expected_episodes'] = 120
        self.job['launch_gate']['assigned_gpu_only'] = True
        legacy.write(self.queue, self.queued)
        self.audit = self.root/'gate_policy_refreshes/fixture'
        self.args = argparse.Namespace(job_id='selected', output_audit=str(self.audit), apply=False, review_sha256=None)
        scheduler_args = self.runtime.processes[self.scheduler_pid]['argv'][4:]
        supervisor_args = self.runtime.processes[self.pipeline_pid]['argv'][4:]
        legacy.write(self.scheduler/'controller_process.json', dict(pid=self.scheduler_pid, started=self.started,
            argv=[str(refresh.PROJECT/'preparations/ease_insights_20261103/scheduler.py'), *scheduler_args]))
        legacy.write(self.pipeline/'supervisor_process.json', dict(pid=self.pipeline_pid, started_unix=self.started,
            argv=[str(refresh.PROJECT/'preparations/ease_insights_20261103/vlm_local_search_v1/supervisor.py'), *supervisor_args]))
        self.coord = self.root/'gate_opportunity_coordinator'
        legacy.write(self.coord/'process.json', dict(pid=700004, start_ticks='102', helper_sha256=refresh.HELPER_SHA256,
                                                   coordinator_sha256=refresh.COORDINATOR_SHA256))
        legacy.write(self.coord/'state.json', dict(schema='ease-gate-opportunity-coordinator-v1', pending=None, phase='stopped'))
        (self.coord/'stop').touch()
        self.patches = [patch.object(refresh, 'ROOT', self.root), patch.object(refresh, 'SCHEDULER_PID', self.scheduler_pid),
            patch.object(refresh, 'SCHEDULER_TICKS', '100'), patch.object(refresh, 'SUPERVISOR_PID', self.pipeline_pid),
            patch.object(refresh, 'SUPERVISOR_TICKS', '101'), patch.object(refresh, 'COORDINATOR_PID', 700004),
            patch.object(refresh, 'COORDINATOR_TICKS', '102'), patch.object(refresh, 'ACK_TIMEOUT', 1),
            patch.object(refresh, 'JOBS', {'selected': (self.gate_pid, '111', 'full_d10')}),
            patch.object(refresh, 'verify_sources', side_effect=self.sources)]
        for item in self.patches:
            item.start()
        os.utime(self.receipt, (self.runtime.clock, self.runtime.clock))

    def sources(self, paths, audit=None):
        return {str(self.manifest_path): refresh.file_hash(self.manifest_path)}

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        legacy.MigrationTests.tearDown(self)

    def run_tool(self):
        return refresh.operate(self.args, self.runtime)

    def review(self):
        audit = self.run_tool()
        raw = (self.audit/'review.template.json').read_bytes()
        (self.audit/'reviewed.json').write_bytes(raw)
        self.args.apply = True
        self.args.review_sha256 = refresh.h.digest(raw)
        return audit

    def signals(self):
        return [sig for _, sig in self.runtime.signals]

    def test_dryrun_no_signal_gpu_lease_or_queue_state_mutation(self):
        before, states = self.queue.read_bytes(), self.states.read_bytes()
        result = self.run_tool()
        self.assertEqual(result['stage'], 'dry_run')
        self.assertEqual(self.queue.read_bytes(), before)
        self.assertEqual(self.states.read_bytes(), states)
        self.assertEqual(self.signals(), [])
        self.assertEqual(result['new_job']['env'], dict(self.job['env'], **refresh.POLICY_ENV))
        self.assertIn('_scheduler_proof', result['scheduler'])

    def test_apply_without_review_or_dryrun_is_refused(self):
        self.args.apply = True
        with self.assertRaisesRegex(refresh.Refused, 'dry-run'):
            self.run_tool()
        self.assertEqual(self.signals(), [])

    def test_success_preserves_science_and_dependencies_and_original_state(self):
        old = copy.deepcopy(self.job)
        self.review()
        result = self.run_tool()
        expected = copy.deepcopy(old); new = result['new_job']
        expected['id'] = new['id']; expected['argv'][6] = new['argv'][6]
        expected['env'].update(refresh.POLICY_ENV)
        expected['launch_gate'].update(continuous_idle_seconds=5, poll_seconds=2)
        self.assertEqual(new, expected)
        self.assertEqual(refresh.h.read_json(self.queue)['jobs'][1]['depends_on'], [new['id']])
        self.assertEqual(refresh.h.read_json(self.states)['selected'], self.state)
        self.assertEqual(result['stage'], 'complete')
        self.assertEqual(self.signals(), [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT])
        self.run_tool()
        self.assertEqual(len(self.signals()), 3)

    def test_exec_race_only_restores_own_stop(self):
        self.review(); before = self.queue.read_bytes()
        self.runtime.on_stop = lambda: self.runtime.processes[self.gate_pid].update(argv=refresh.h.concrete(self.original_argv, self.state), children=['900001'])
        with self.assertRaisesRegex(refresh.Refused, 'exec identity changed'):
            self.run_tool()
        self.assertEqual(self.signals(), [signal.SIGSTOP, signal.SIGCONT])
        self.assertEqual(self.queue.read_bytes(), before)
        self.assertEqual(refresh.h.read_json(self.audit/'audit.json')['stage'], 'aborted_restored')

    def test_queue_change_after_stop_restores_own_stop(self):
        self.review()
        def mutate():
            queue = refresh.h.read_json(self.queue); queue['external_change'] = True; legacy.write(self.queue, queue)
        self.runtime.on_stop = mutate
        with self.assertRaisesRegex(refresh.Refused, 'Final queue CAS'):
            self.run_tool()
        self.assertEqual(self.signals(), [signal.SIGSTOP, signal.SIGCONT])

    def test_review_tamper_during_stop_is_refused_and_restored(self):
        self.review()
        self.runtime.on_stop = lambda: (self.audit/'reviewed.json').write_bytes(b'{}')
        with self.assertRaisesRegex(refresh.Refused, 'review file hash'):
            self.run_tool()
        self.assertEqual(self.signals(), [signal.SIGSTOP, signal.SIGCONT])

    def test_scientific_proposal_tamper_even_with_new_review_hash_refused(self):
        self.review()
        audit = refresh.h.read_json(self.audit/'audit.json')
        proposed = refresh.h.read_json(self.audit/'queue.proposed.json')
        proposed['jobs'][0]['expected_episodes'] = 1
        raw = refresh.h.json_bytes(proposed); (self.audit/'queue.proposed.json').write_bytes(raw)
        audit['new_job'] = proposed['jobs'][0]; audit['queue_after_sha256'] = refresh.h.digest(raw)
        legacy.write(self.audit/'audit.json', audit)
        reviewed = refresh.review_value(audit, (self.audit/'queue.before.json').read_bytes(), raw)
        raw = refresh.h.json_bytes(reviewed); (self.audit/'reviewed.json').write_bytes(raw)
        self.args.review_sha256 = refresh.h.digest(raw)
        with self.assertRaisesRegex(refresh.Refused, 'scientific job semantics'):
            self.run_tool()
        self.assertEqual(self.signals(), [])

    def test_both_acknowledgements_required_and_same_audit_retry(self):
        self.review(); self.runtime.pipeline_ack = False
        with self.assertRaisesRegex(refresh.Refused, 'acknowledgements'):
            self.run_tool()
        self.assertEqual(self.signals(), [signal.SIGSTOP])
        committed = self.queue.read_bytes()
        self.runtime.pipeline_ack = True
        self.assertEqual(self.run_tool()['stage'], 'complete')
        self.assertEqual(self.queue.read_bytes(), committed)
        self.assertEqual(self.signals(), [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT])

    def test_scheduler_ack_required(self):
        self.review(); self.runtime.auto_ack = False
        with self.assertRaisesRegex(refresh.Refused, 'acknowledgements'):
            self.run_tool()
        self.assertEqual(self.signals(), [signal.SIGSTOP])

    def test_dead_old_gate_cleanup_allows_legitimate_replacement_artifacts(self):
        self.review(); self.runtime.pipeline_ack = False
        with self.assertRaisesRegex(refresh.Refused, 'acknowledgements'):
            self.run_tool()
        self.runtime.processes[self.gate_pid]['state'] = 'Z'
        directory = self.discovery/'full_d10'; directory.mkdir()
        (directory/'episode.json').write_text('{"valid": true}')
        self.runtime.pipeline_ack = True
        self.assertEqual(self.run_tool()['stage'], 'complete')
        self.assertEqual(self.signals(), [signal.SIGSTOP])

    def test_live_stopped_old_gate_cleanup_allows_replacement_artifacts(self):
        self.review(); self.runtime.pipeline_ack = False
        with self.assertRaisesRegex(refresh.Refused, 'acknowledgements'):
            self.run_tool()
        directory = self.discovery/'full_d10'; directory.mkdir()
        (directory/'episode.json').write_text('{"valid": true}')
        self.runtime.pipeline_ack = True
        self.assertEqual(self.run_tool()['stage'], 'complete')
        self.assertEqual(self.signals(), [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT])

    def test_replace_exception_boundaries(self):
        for mode, expected in [('fail_before', [signal.SIGSTOP, signal.SIGCONT]),
                               ('fail_after', [signal.SIGSTOP]), ('ambiguous', [signal.SIGSTOP]), ('unreadable', [signal.SIGSTOP])]:
            with self.subTest(mode=mode):
                if mode != 'fail_before':
                    # Each case must own a fresh fixture and audit.
                    self.tearDown(); self.setUp()
                self.review(); self.runtime.replace_mode = mode
                with self.assertRaises(OSError):
                    self.run_tool()
                self.assertEqual(self.signals(), expected)
                if mode in ('ambiguous', 'unreadable'):
                    with self.assertRaisesRegex(refresh.Refused, 'ambiguous'):
                        self.run_tool()
                    self.assertEqual(self.signals(), expected)

    def test_precommit_crash_exact_runner_can_only_resume(self):
        audit = self.review()
        audit['stop_authorization'] = refresh.h.stop_authorization(audit)
        audit['stage'] = 'stop_intent'; audit['events'].append(dict(stage='stop_intent', at=self.runtime.now()))
        legacy.write(self.audit/'audit.json', audit)
        self.runtime.processes[self.gate_pid].update(state='T', argv=refresh.h.concrete(self.original_argv, self.state))
        self.assertEqual(self.run_tool()['stage'], 'aborted_restored')
        self.assertEqual(self.signals(), [signal.SIGCONT])

    def test_coordinator_alive_pending_or_no_stop_refused(self):
        for kind in ('alive', 'pending', 'no_stop'):
            with self.subTest(kind=kind):
                if kind == 'alive':
                    self.runtime.processes[700004]['state'] = 'S'
                elif kind == 'pending':
                    self.runtime.processes[700004]['state'] = 'absent'
                    legacy.write(self.coord/'state.json', dict(schema='ease-gate-opportunity-coordinator-v1', pending={'anything': True}, phase='stopped'))
                else:
                    (self.coord/'stop').unlink()
                with self.assertRaises(refresh.Refused):
                    self.run_tool()
        self.assertEqual(self.signals(), [])

    def test_zero_case_and_already_exec_model_refused(self):
        directory = self.discovery/'full_d10'; directory.mkdir()
        (directory/'episode.json').write_text('{}')
        with self.assertRaisesRegex(refresh.Refused, 'zero-case'):
            self.run_tool()
        (directory/'episode.json').unlink()
        self.runtime.processes[self.gate_pid]['argv'] = refresh.h.concrete(self.original_argv, self.state)
        with self.assertRaisesRegex(refresh.Refused, 'CPU gate'):
            self.run_tool()
        self.assertEqual(self.signals(), [])

    def test_scheduler_executable_proof_and_receipt_are_bound(self):
        self.review()
        self.runtime.processes[self.scheduler_pid]['argv'][0] = '/usr/bin/python3'
        with self.assertRaisesRegex(ValueError, 'raw argv/start/executable'):
            self.run_tool()
        self.assertEqual(self.signals(), [])

    def test_source_change_after_stop_restores_before_commit(self):
        self.review()
        self.runtime.on_stop = lambda: os.utime(self.manifest_path, None)
        with self.assertRaisesRegex(refresh.Refused, 'source metadata changed'):
            self.run_tool()
        self.assertEqual(self.signals(), [signal.SIGSTOP, signal.SIGCONT])

    def test_completed_marker_and_unknown_job_refused(self):
        self.args.job_id = 'unknown'
        with self.assertRaisesRegex(refresh.Refused, 'two pinned'):
            self.run_tool()
        self.args.job_id = 'selected'
        marker = Path(self.job['complete_file']); marker.parent.mkdir(); marker.touch()
        with self.assertRaisesRegex(refresh.Refused, 'Completion marker'):
            self.run_tool()
        self.assertEqual(self.signals(), [])


if __name__ == '__main__':
    unittest.main()

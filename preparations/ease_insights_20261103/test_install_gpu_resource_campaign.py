"""Isolated CPU fake installer transactions; no GPU or real PID is touched."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import unittest
from unittest.mock import patch

from preparations.ease_insights_20261103 import install_gpu_resource_campaign as install
from preparations.ease_insights_20261103 import test_reassign_waiting_gate_v2 as legacy


class FakeRuntime(legacy.FakeRuntime):
    def __init__(self, fixture):
        super().__init__(fixture)
        self.terms = set(); self.spawns = 0; self.spawned_at = None
        self.on_stop_pid = None; self.adopt = True; self.register = True; self.lock_proven = False

    def gpu(self, uuid): raise AssertionError('No GPU query')
    def lease_available(self, uuid): raise AssertionError('No GPU lease probe')
    def identity(self, pid): return copy.deepcopy(self.processes.get(pid, dict(pid=pid, state='absent')))
    def gate_pids(self): return getattr(self.fixture, 'gate_pids', (self.fixture.gate_pid,))
    def open_pidfd(self, pid):
        assert pid in (self.fixture.scheduler_pid, *self.gate_pids())
        return pid
    def close_pidfd(self, fd): pass
    def matches_started(self, process, started): return started in (self.fixture.started, self.spawned_at)
    def send(self, fd, sig):
        assert fd in (self.fixture.scheduler_pid, *self.gate_pids())
        self.signals.append((fd, sig)); process = self.processes[fd]
        if sig == signal.SIGSTOP:
            process['state'] = 'T'
            if self.on_stop_pid: self.on_stop_pid(fd)
        elif sig == signal.SIGTERM:
            assert process['state'] == 'T'
            assert install.h.exact_module(process['argv'], install.h.GATE if fd in self.gate_pids() else install.h.SCHEDULER)
            self.terms.add(fd)
        elif sig == signal.SIGCONT:
            process['state'] = 'Z' if fd in self.terms else 'S'
            if fd == self.fixture.scheduler_pid and fd in self.terms:
                # OS reaps the dead scheduler and its already-dead gate.
                process['state'] = 'absent'
                for pid in self.gate_pids(): self.processes[pid]['state'] = 'absent'
        else: raise AssertionError('Unexpected signal')

    def register_receipt(self):
        argv = self.processes[800001]['argv']
        legacy.write(self.fixture.scheduler/'controller_process.json', dict(pid=800001, started=self.spawned_at,
            argv=[str(install.PROJECT/'preparations/ease_insights_20261103/scheduler.py'), *install.h.module_invocation(argv)['args']]))

    def start_scheduler(self, argv, log):
        self.spawns += 1; self.spawned_at = self.clock
        assert (self.fixture.campaign/'transactions/initial/committed.json').exists()
        assert (self.fixture.campaign/'deployed.json').exists()
        assert self.identity(self.fixture.scheduler_pid)['state'] == 'absent'
        try:
            with install.h.lock(self.fixture.root/'pipeline/scheduler_restart.lock'): pass
        except BlockingIOError:
            self.lock_proven = True
        self.processes[800001] = dict(pid=800001, start_ticks='200', argv=argv, uid=os.getuid(), ppid=900001,
                                      state='S', children=[], sigterm_default=True)
        if self.register: self.register_receipt()
        return self.identity(800001)

    def sleep(self, seconds):
        self.clock += seconds
        if self.spawns and self.adopt:
            legacy.write(self.fixture.pipeline/'state.json', dict(phase='waiting_discovery', scheduler=dict(pid=800001)))
            legacy.write(self.fixture.pipeline/'status.json', dict(phase='waiting_discovery', terminal=False, time=self.clock))


class InstallerTests(unittest.TestCase):
    def setUp(self):
        legacy.MigrationTests.setUp(self)
        self.campaign = self.root/'campaign'; self.campaign.mkdir()
        self.audit = self.campaign/'installation_audit'
        self.job['expected_episodes'] = 120
        selected = [self.job]
        for index in range(4):
            job = copy.deepcopy(self.job); job['id'] = 'pending_'+str(index)
            job['complete_file'] = str(self.discovery/('pending_'+str(index))/'discovery_shard00of01_complete.json')
            selected.append(job)
        complete = dict(id='complete_anchor', argv=['not_running'], complete_file=str(self.root/'valid_anchor.json'))
        self.before = dict(jobs=selected+[complete], metadata='unchanged')
        self.after = dict(jobs=[dict(id='slot0', argv=['session0']), dict(id='slot1', argv=['session1']), complete], metadata='unchanged')
        legacy.write(self.queue, self.before)
        states = dict(selected=self.state, complete_anchor=dict(status='complete'))
        states.update({j['id']: dict(status='pending') for j in selected[1:]}); legacy.write(self.states, states)
        self.installation = dict(schema='ease-gpu-campaign-install-v1', plan_sha256='plan', logical_jobs=selected,
            logical_states_before={j['id']: states[j['id']] for j in selected}, task_sha256={'task'+str(i): 'sha'+str(i) for i in range(5)},
            slot_job_ids=['slot0', 'slot1'], queue_proposed_sha256=install.h.digest(install.h.json_bytes(self.after)))
        self.runtime = FakeRuntime(self)
        common = dict(uid=os.getuid(), state='S', sigterm_default=True)
        self.runtime.processes = {
            self.scheduler_pid: dict(common, pid=self.scheduler_pid, start_ticks='100', ppid=1,
                argv=['/usr/bin/python3', '-u', '-m', install.h.SCHEDULER, '--queue', str(self.queue), '--output', str(self.scheduler),
                      '--max-gpus', '2', '--scan-seconds', '30', '--idle-seconds', '0'], children=[str(self.gate_pid)]),
            self.pipeline_pid: dict(common, pid=self.pipeline_pid, start_ticks='101', ppid=1, children=[],
                argv=['/absolute/python', '-u', '-m', install.h.LOCAL_SUPERVISOR, '--root', str(self.discovery), '--launch']),
            self.gate_pid: dict(common, pid=self.gate_pid, start_ticks='111', ppid=self.scheduler_pid, children=[], argv=self.state['argv']),
            700004: dict(pid=700004, state='absent'),
        }
        legacy.write(self.scheduler/'controller_process.json', dict(pid=self.scheduler_pid, started=self.started,
            argv=[str(install.PROJECT/'preparations/ease_insights_20261103/scheduler.py'),
                  *install.h.module_invocation(self.runtime.processes[self.scheduler_pid]['argv'])['args']]))
        legacy.write(self.pipeline/'supervisor_process.json', dict(pid=self.pipeline_pid, started_unix=self.started,
            argv=[str(install.PROJECT/'preparations/ease_insights_20261103/vlm_local_search_v1/supervisor.py'),
                  *install.h.module_invocation(self.runtime.processes[self.pipeline_pid]['argv'])['args']]))
        self.coord = self.root/'gate_opportunity_coordinator'
        legacy.write(self.coord/'process.json', dict(pid=700004, start_ticks='102'))
        legacy.write(self.coord/'state.json', dict(phase='stopped', pending=None)); (self.coord/'stop').touch()
        (self.root/'pipeline').mkdir(); (self.root/'pipeline/scheduler_restart.lock').touch()
        self.args = argparse.Namespace(campaign_root=str(self.campaign), output_audit=str(self.audit), apply=False, review_sha256=None)
        self.patches = [patch.object(install, 'ROOT', self.root), patch.object(install, 'SCHEDULER_PID', self.scheduler_pid),
            patch.object(install, 'SCHEDULER_TICKS', '100'), patch.object(install, 'SUPERVISOR_PID', self.pipeline_pid),
            patch.object(install, 'SUPERVISOR_TICKS', '101'), patch.object(install, 'COORDINATOR_PID', 700004),
            patch.object(install, 'COORDINATOR_TICKS', '102'), patch.object(install, 'GATE_PINS', {'selected': (self.gate_pid, '111')}),
            patch.object(install, 'ACK_TIMEOUT', 1), patch.object(install, 'verify_bundle', side_effect=self.bundle)]
        for item in self.patches: item.start()

    def tearDown(self):
        for item in reversed(self.patches): item.stop()
        legacy.MigrationTests.tearDown(self)

    def bundle(self, paths):
        hashes = {str(self.manifest_path): install.hashed(self.manifest_path)}
        return dict(installation=self.installation, plan={}, before=install.h.json_bytes(self.before),
                    proposed=install.h.json_bytes(self.after), hashes=hashes, stats=install.stats(hashes))

    def run_tool(self): return install.operate(self.args, self.runtime)
    def prepare(self):
        # The live before fixture must match exact serialized prepared bytes.
        self.queue.write_bytes(install.h.json_bytes(self.before))
        audit = self.run_tool(); raw = (self.audit/'review.template.json').read_bytes()
        (self.audit/'reviewed.json').write_bytes(raw)
        self.args.apply = True; self.args.review_sha256 = install.h.digest(raw)
        return audit
    def signals(self): return self.runtime.signals

    def test_dryrun_writes_only_review_artifacts(self):
        self.queue.write_bytes(install.h.json_bytes(self.before)); before, states = self.queue.read_bytes(), self.states.read_bytes()
        self.assertEqual(self.run_tool()['stage'], 'dry_run')
        self.assertEqual(self.queue.read_bytes(), before); self.assertEqual(self.states.read_bytes(), states)
        self.assertEqual(self.signals(), []); self.assertEqual(self.runtime.spawns, 0)

    def test_install_order_activation_scan2_identity_and_both_acks(self):
        self.prepare(); states = self.states.read_bytes()
        result = self.run_tool()
        self.assertEqual(result['stage'], 'complete'); self.assertEqual(self.runtime.spawns, 1)
        self.assertEqual(self.signals(), [(self.scheduler_pid, signal.SIGSTOP), (self.gate_pid, signal.SIGSTOP),
            (self.gate_pid, signal.SIGTERM), (self.gate_pid, signal.SIGCONT),
            (self.scheduler_pid, signal.SIGTERM), (self.scheduler_pid, signal.SIGCONT)])
        self.assertTrue(self.runtime.lock_proven)
        self.assertEqual(self.states.read_bytes(), states)
        self.assertEqual(install.h.read_json(self.queue), self.after)
        self.assertIn('2', result['new_scheduler_argv'])
        self.run_tool(); self.assertEqual(self.runtime.spawns, 1)

    def test_gate_exec_race_only_undoes_own_stops(self):
        self.prepare(); before = self.queue.read_bytes()
        def race(pid):
            if pid == self.gate_pid:
                self.runtime.processes[pid].update(argv=install.h.concrete(self.original_argv, self.state), children=['900099'])
        self.runtime.on_stop_pid = race
        with self.assertRaises(install.Refused): self.run_tool()
        self.assertEqual(self.queue.read_bytes(), before)
        self.assertEqual(self.signals(), [(self.scheduler_pid, signal.SIGSTOP), (self.gate_pid, signal.SIGSTOP),
            (self.gate_pid, signal.SIGCONT), (self.scheduler_pid, signal.SIGCONT)])
        self.assertEqual(install.h.read_json(self.audit/'audit.json')['stage'], 'aborted_restored')

    def test_unknown_children_before_stop_refused_without_signals(self):
        self.runtime.processes[self.scheduler_pid]['children'].append('999')
        self.queue.write_bytes(install.h.json_bytes(self.before))
        with self.assertRaisesRegex(install.Refused, 'unknown/additional'): self.run_tool()
        self.assertEqual(self.signals(), [])

    def test_child_spawn_race_after_scheduler_stop_restores_only_scheduler(self):
        self.prepare()
        self.runtime.on_stop_pid = lambda pid: self.runtime.processes[self.scheduler_pid]['children'].append('999')
        with self.assertRaisesRegex(install.Refused, 'spawned/changed'): self.run_tool()
        self.assertEqual(self.signals(), [(self.scheduler_pid, signal.SIGSTOP), (self.scheduler_pid, signal.SIGCONT)])

    def test_queue_cas_change_restores_both_own_stops(self):
        self.prepare()
        def change(pid):
            if pid == self.gate_pid:
                value = install.h.read_json(self.queue); value['external'] = True; legacy.write(self.queue, value)
        self.runtime.on_stop_pid = change
        with self.assertRaisesRegex(install.Refused, 'CAS/review'): self.run_tool()
        self.assertEqual([sig for _, sig in self.signals()], [signal.SIGSTOP, signal.SIGSTOP, signal.SIGCONT, signal.SIGCONT])

    def test_source_change_after_stop_restores_own_stops(self):
        self.prepare()
        self.runtime.on_stop_pid = lambda pid: os.utime(self.manifest_path, None) if pid == self.gate_pid else None
        with self.assertRaisesRegex(install.Refused, 'evidence changed'): self.run_tool()
        self.assertEqual([sig for _, sig in self.signals()], [signal.SIGSTOP, signal.SIGSTOP, signal.SIGCONT, signal.SIGCONT])

    def test_commit_success_then_exception_never_rolls_back_or_resumes(self):
        self.prepare(); self.runtime.replace_mode = 'fail_after'
        with self.assertRaises(OSError): self.run_tool()
        self.assertEqual([sig for _, sig in self.signals()], [signal.SIGSTOP, signal.SIGSTOP])
        self.assertEqual(install.h.read_json(self.queue), self.after)
        self.runtime.replace_mode = 'success'
        self.assertEqual(self.run_tool()['stage'], 'complete'); self.assertEqual(self.runtime.spawns, 1)

    def test_unknown_commit_never_resumes_or_spawns(self):
        self.prepare(); self.runtime.replace_mode = 'ambiguous'
        with self.assertRaises(OSError): self.run_tool()
        with self.assertRaisesRegex(install.Refused, 'Unknown queue'): self.run_tool()
        self.assertEqual([sig for _, sig in self.signals()], [signal.SIGSTOP, signal.SIGSTOP])
        self.assertEqual(self.runtime.spawns, 0)

    def test_activation_failure_keeps_allocator_stopped_and_recovers(self):
        self.prepare(); original = install.campaign.append_identical
        with patch.object(install.campaign, 'append_identical', side_effect=OSError('activation disk error')):
            with self.assertRaises(OSError): self.run_tool()
        self.assertEqual(self.runtime.processes[self.scheduler_pid]['state'], 'T')
        self.assertNotIn((self.scheduler_pid, signal.SIGTERM), self.signals())
        self.assertEqual(self.run_tool()['stage'], 'complete')

    def test_supervisor_ack_timeout_recovers_same_new_pid_after_reparenting(self):
        self.prepare(); self.runtime.adopt = False
        with self.assertRaisesRegex(install.Refused, 'adoption ack'): self.run_tool()
        self.assertEqual(self.runtime.spawns, 1)
        self.runtime.processes[800001]['ppid'] = 1
        self.runtime.adopt = True
        self.assertEqual(self.run_tool()['stage'], 'complete'); self.assertEqual(self.runtime.spawns, 1)

    def test_post_start_receipt_timeout_never_spawns_again(self):
        self.prepare(); self.runtime.register = False
        with self.assertRaisesRegex(install.Refused, 'identity ack timed out'): self.run_tool()
        self.assertEqual(self.runtime.spawns, 1)
        self.runtime.register_receipt()
        self.assertEqual(self.run_tool()['stage'], 'complete'); self.assertEqual(self.runtime.spawns, 1)

    def test_review_tamper_after_stop_restores_without_commit(self):
        self.prepare()
        self.runtime.on_stop_pid = lambda pid: (self.audit/'reviewed.json').write_text('{}') if pid == self.gate_pid else None
        with self.assertRaisesRegex(install.Refused, 'review identity'): self.run_tool()
        self.assertEqual([sig for _, sig in self.signals()], [signal.SIGSTOP, signal.SIGSTOP, signal.SIGCONT, signal.SIGCONT])

    def interrupted_precommit(self, inner=False):
        audit = self.prepare()
        audit['stop_authorizations'] = {role: install.authorization(process) for role, process in
            [('scheduler', audit['scheduler']), *audit['gates'].items()]}
        audit['stage'] = 'stop_intent'; audit['events'].append(dict(stage='stop_intent', at=self.runtime.now()))
        legacy.write(self.audit/'audit.json', audit)
        self.runtime.processes[self.scheduler_pid]['state'] = 'T'
        self.runtime.processes[self.gate_pid]['state'] = 'T'
        if inner:
            self.runtime.processes[self.gate_pid].update(argv=install.h.concrete(self.original_argv, self.state), children=['900099'])
        return audit

    def test_interrupted_precommit_resumes_exact_execed_runner_and_allocator(self):
        self.interrupted_precommit(inner=True)
        self.assertEqual(self.run_tool()['stage'], 'aborted_restored')
        self.assertEqual(self.signals(), [(self.gate_pid, signal.SIGCONT), (self.scheduler_pid, signal.SIGCONT)])
        self.assertEqual(self.runtime.spawns, 0)

    def test_changed_source_recovery_only_restores_own_precommit_stops(self):
        self.interrupted_precommit(inner=True); os.utime(self.manifest_path, None)
        with self.assertRaisesRegex(install.Refused, 'bundle/source identity changed'): self.run_tool()
        self.assertEqual(self.signals(), [(self.gate_pid, signal.SIGCONT), (self.scheduler_pid, signal.SIGCONT)])

    def test_spawn_intent_without_bound_receipt_never_repeats_spawn(self):
        self.prepare()
        self.runtime.start_scheduler = lambda *args: (_ for _ in ()).throw(OSError('spawn interrupted before PID persisted'))
        with self.assertRaises(OSError): self.run_tool()
        with self.assertRaisesRegex(install.Refused, 'no duplicate allocator spawn'): self.run_tool()
        self.assertEqual(self.runtime.spawns, 0)

    def test_spawn_effect_then_exception_recovers_exact_receipt_without_second_spawn(self):
        self.prepare(); spawn = self.runtime.start_scheduler
        def interrupted(*args):
            spawn(*args); raise OSError('child started before parent persisted PID')
        self.runtime.start_scheduler = interrupted
        with self.assertRaises(OSError): self.run_tool()
        self.assertEqual(self.runtime.spawns, 1)
        self.assertEqual(self.run_tool()['stage'], 'complete')
        self.assertEqual(self.runtime.spawns, 1)

    def test_committed_queue_only_allows_append_without_reordering_slots_or_metadata(self):
        audit = self.prepare()
        appended = copy.deepcopy(self.after); appended['jobs'].append(dict(id='future_pending', argv=['future']))
        self.assertEqual(install.outcome(appended, audit), 'committed')
        inserted = copy.deepcopy(appended); inserted['jobs'].insert(0, inserted['jobs'].pop())
        self.assertEqual(install.outcome(inserted, audit), 'unknown')
        reordered = copy.deepcopy(self.after); reordered['jobs'][0], reordered['jobs'][1] = reordered['jobs'][1], reordered['jobs'][0]
        self.assertEqual(install.outcome(reordered, audit), 'unknown')
        metadata = copy.deepcopy(self.after); metadata['metadata'] = 'changed'
        self.assertEqual(install.outcome(metadata, audit), 'unknown')

    def test_delayed_stop_in_D_state_is_not_declared_restored_until_T_then_cont(self):
        self.prepare(); send = self.runtime.send
        def delayed(fd, sig):
            if fd == self.scheduler_pid and sig == signal.SIGSTOP:
                self.runtime.signals.append((fd, sig)); self.runtime.processes[fd]['state'] = 'D'
            else:
                send(fd, sig)
        self.runtime.send = delayed
        with self.assertRaisesRegex(install.Refused, 'STOP did not settle'): self.run_tool()
        self.assertEqual(install.h.read_json(self.audit/'audit.json')['stage'], 'needs_cleanup')
        self.assertEqual(self.signals(), [(self.scheduler_pid, signal.SIGSTOP)])
        self.runtime.processes[self.scheduler_pid]['state'] = 'T'
        self.assertEqual(self.run_tool()['stage'], 'aborted_restored')
        self.assertEqual(self.signals(), [(self.scheduler_pid, signal.SIGSTOP), (self.scheduler_pid, signal.SIGCONT)])

    def test_stop_send_error_with_delayed_effect_requires_same_audit_cleanup(self):
        self.prepare(); send = self.runtime.send
        def delayed_error(fd, sig):
            if fd == self.scheduler_pid and sig == signal.SIGSTOP:
                self.runtime.signals.append((fd, sig)); self.runtime.processes[fd]['state'] = 'D'
                raise OSError('STOP dispatch wrapper failed after pending effect')
            send(fd, sig)
        self.runtime.send = delayed_error
        with self.assertRaises(OSError): self.run_tool()
        self.assertEqual(install.h.read_json(self.audit/'audit.json')['stage'], 'needs_cleanup')
        self.assertEqual(self.signals(), [(self.scheduler_pid, signal.SIGSTOP)])
        self.runtime.processes[self.scheduler_pid]['state'] = 'T'
        self.assertEqual(self.run_tool()['stage'], 'aborted_restored')

    def test_post_cont_crash_S_state_requires_exact_per_role_restore_intent(self):
        audit = self.interrupted_precommit(inner=True)
        actual = self.runtime.identity(self.gate_pid)
        target = {key: actual[key] for key in ('pid', 'start_ticks', 'argv', 'uid', 'ppid')}
        audit['restore_intents'] = {'selected': dict(target, from_state='T', signal='SIGCONT')}
        audit['events'].append(dict(stage='precommit_restore_intent', at=self.runtime.now()))
        legacy.write(self.audit/'audit.json', audit)
        self.runtime.processes[self.gate_pid]['state'] = 'S'
        self.assertEqual(self.run_tool()['stage'], 'aborted_restored')
        self.assertEqual(self.signals(), [(self.scheduler_pid, signal.SIGCONT)])


class DualGateInstallerTests(unittest.TestCase):
    bundle = InstallerTests.bundle
    run_tool = InstallerTests.run_tool
    prepare = InstallerTests.prepare
    signals = InstallerTests.signals

    def setUp(self):
        InstallerTests.setUp(self)
        self.second_pid = 700005; self.gate_pids = (self.gate_pid, self.second_pid)
        self.second_job = self.installation['logical_jobs'][1]
        self.second_job['argv'][6] = str(self.scheduler/'launch_gates'/(self.second_job['id']+'.json'))
        self.second_state = dict(status='running', pid=self.second_pid, gpu=1, gpu_uuid=legacy.CANDIDATE,
            port=54322, started=self.started, start_ticks='222')
        self.second_state['argv'] = install.h.concrete(self.second_job['argv'], self.second_state)
        self.installation['logical_states_before'][self.second_job['id']] = copy.deepcopy(self.second_state)
        states = install.h.read_json(self.states); states[self.second_job['id']] = self.second_state; legacy.write(self.states, states)
        self.runtime.processes[self.second_pid] = dict(pid=self.second_pid, start_ticks='222',
            state='S', uid=os.getuid(), ppid=self.scheduler_pid, argv=self.second_state['argv'], children=[], sigterm_default=True)
        self.runtime.processes[self.scheduler_pid]['children'].append(str(self.second_pid))
        install.GATE_PINS[self.second_job['id']] = (self.second_pid, '222')
        legacy.write(self.queue, self.before)

    def tearDown(self): InstallerTests.tearDown(self)

    def test_two_gates_are_stopped_then_individually_exited_before_allocator(self):
        self.prepare()
        self.assertEqual(self.run_tool()['stage'], 'complete')
        self.assertEqual(self.signals(), [(self.scheduler_pid, signal.SIGSTOP),
            (self.gate_pid, signal.SIGSTOP), (self.second_pid, signal.SIGSTOP),
            (self.gate_pid, signal.SIGTERM), (self.gate_pid, signal.SIGCONT),
            (self.second_pid, signal.SIGTERM), (self.second_pid, signal.SIGCONT),
            (self.scheduler_pid, signal.SIGTERM), (self.scheduler_pid, signal.SIGCONT)])
        audit = install.h.read_json(self.audit/'audit.json')
        self.assertEqual(set(audit['exited_gate_proofs']), {str(self.gate_pid), str(self.second_pid)})
        self.assertEqual(self.runtime.spawns, 1)

    def test_second_gate_exec_race_restores_both_and_never_terms_runner(self):
        self.prepare(); before = self.queue.read_bytes()
        def race(pid):
            if pid == self.second_pid:
                self.runtime.processes[pid].update(argv=install.h.gate_parts(self.second_state['argv'])[2], children=['900099'])
        self.runtime.on_stop_pid = race
        with self.assertRaises(install.Refused): self.run_tool()
        self.assertEqual(self.queue.read_bytes(), before)
        self.assertEqual(self.signals(), [(self.scheduler_pid, signal.SIGSTOP),
            (self.gate_pid, signal.SIGSTOP), (self.second_pid, signal.SIGSTOP),
            (self.gate_pid, signal.SIGCONT), (self.second_pid, signal.SIGCONT), (self.scheduler_pid, signal.SIGCONT)])
        self.assertEqual(self.runtime.spawns, 0)

    def test_second_gate_delayed_stop_resumes_first_once_then_reenters_cleanup(self):
        self.prepare(); send = self.runtime.send
        def delayed(fd, sig):
            if fd == self.second_pid and sig == signal.SIGSTOP:
                self.runtime.signals.append((fd, sig)); self.runtime.processes[fd]['state'] = 'D'
            else:
                send(fd, sig)
        self.runtime.send = delayed
        with self.assertRaisesRegex(install.Refused, 'STOP did not settle'): self.run_tool()
        self.assertEqual(install.h.read_json(self.audit/'audit.json')['stage'], 'needs_cleanup')
        self.assertEqual(self.runtime.processes[self.gate_pid]['state'], 'S')
        self.assertEqual(self.runtime.processes[self.scheduler_pid]['state'], 'T')
        self.runtime.processes[self.second_pid]['state'] = 'T'
        self.assertEqual(self.run_tool()['stage'], 'aborted_restored')
        self.assertEqual(self.signals().count((self.gate_pid, signal.SIGCONT)), 1)
        self.assertEqual(self.signals()[-2:], [(self.second_pid, signal.SIGCONT), (self.scheduler_pid, signal.SIGCONT)])
        self.assertFalse(any(sig == signal.SIGTERM for _, sig in self.signals()))

    def test_second_gate_cleanup_failure_recovers_without_rollback_or_new_allocator(self):
        self.prepare(); send = self.runtime.send; failed = []
        def interrupt(fd, sig):
            send(fd, sig)
            if fd == self.second_pid and sig == signal.SIGTERM and not failed:
                failed.append(True); raise OSError('second gate TERM effect preceded wrapper failure')
        self.runtime.send = interrupt
        with self.assertRaises(OSError): self.run_tool()
        committed = self.queue.read_bytes()
        self.assertEqual(self.runtime.processes[self.gate_pid]['state'], 'Z')
        self.assertEqual(self.runtime.processes[self.second_pid]['state'], 'T')
        self.assertEqual(self.runtime.spawns, 0)
        self.assertEqual(self.run_tool()['stage'], 'complete')
        self.assertEqual(self.queue.read_bytes(), committed); self.assertEqual(self.runtime.spawns, 1)
        self.assertEqual(self.signals().count((self.gate_pid, signal.SIGTERM)), 1)

    def test_second_gate_start_identity_or_extra_child_refuses_without_stop(self):
        self.runtime.processes[self.second_pid]['start_ticks'] = 'changed'
        self.queue.write_bytes(install.h.json_bytes(self.before))
        with self.assertRaisesRegex(install.Refused, 'pinned childless'): self.run_tool()
        self.assertEqual(self.signals(), [])


if __name__ == '__main__': unittest.main()

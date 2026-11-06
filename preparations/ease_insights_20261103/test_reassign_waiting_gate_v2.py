"""CPU fixtures only: every GPU query, pidfd, signal and controller is a fake."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import patch

from preparations.ease_insights_20261103 import reassign_waiting_gate_v2 as migration


SOURCE = "GPU-12345678-1234-1234-1234-123456789abc"
CANDIDATE = "GPU-87654321-1234-1234-1234-123456789abc"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class FakeRuntime:
    def __init__(self, fixture):
        self.fixture = fixture
        self.clock = time.time()
        self.processes = {}
        self.signals = []
        self.queries = []
        self.lease_free = True
        self.source_idle = False
        self.target_idle = True
        self.auto_ack = True
        self.pipeline_ack = True
        self.replace_mode = "success"
        self.on_stop = None
        self.term_pending = False

    def now(self):
        return self.clock

    def monotonic(self):
        return self.clock

    def identity(self, pid):
        return copy.deepcopy(self.processes[pid])

    def matches_started(self, identity, started):
        return started == self.fixture.started

    def gpu(self, uuid):
        self.queries.append(uuid)
        idle = self.source_idle if uuid == SOURCE else self.target_idle
        return dict(idle=idle, compute_process_count=0 if idle else 1,
                    memory_mib=0 if idle else 1000, gpu_uuid=uuid, checked_at=self.clock)

    def lease_available(self, uuid):
        assert uuid == CANDIDATE
        return self.lease_free

    def open_pidfd(self, pid):
        assert pid == self.fixture.gate_pid
        return pid

    def close_pidfd(self, fd):
        assert fd == self.fixture.gate_pid

    def send(self, fd, sig):
        assert fd == self.fixture.gate_pid
        self.signals.append((fd, sig))
        process = self.processes[fd]
        if sig == signal.SIGSTOP:
            process["state"] = "T"
            if self.on_stop:
                self.on_stop()
        elif sig == signal.SIGTERM:
            assert process["state"] == "T"
            assert migration.exact_module(process["argv"], migration.GATE)
            self.term_pending = True
        elif sig == signal.SIGCONT:
            process["state"] = "Z" if self.term_pending else "S"
        else:
            raise AssertionError("Unexpected fake signal")

    def exited(self, fd):
        return self.processes[fd]["state"] in ("Z", "absent")

    def replace_queue(self, temporary, queue):
        if self.replace_mode == "fail_before":
            raise OSError("fixture write failed before replace")
        if self.replace_mode == "ambiguous":
            write(queue, dict(jobs=[]))
            raise OSError("fixture ambiguous replace failure")
        os.replace(temporary, queue)
        if self.replace_mode == "fail_after":
            raise OSError("fixture replace succeeded but reported failure")
        if self.replace_mode == "unreadable":
            queue.write_text("unreadable fixture JSON")
            raise OSError("fixture cannot determine replace outcome")

    def sleep(self, seconds):
        self.clock += seconds
        if not self.auto_ack:
            return
        queue = migration.read_json(self.fixture.queue)
        new = [job for job in queue["jobs"] if "__wait_reassign_" in job["id"]]
        if new:
            # Simulated independent scheduler persistence, not a tool mutation.
            states = migration.read_json(self.fixture.states)
            states[new[0]["id"]] = dict(status="pending")
            write(self.fixture.states, states)
            if self.pipeline_ack:
                write(self.fixture.pipeline / "status.json", dict(time=self.clock,phase="waiting_discovery",terminal=False))


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "experiment"
        self.queue = self.root / "queue.json"
        self.scheduler = self.root / "scheduler"
        self.discovery = self.root / "vlm_local_search_v1/discovery"
        self.pipeline = self.root / "vlm_local_search_v1/supervisor"
        self.states = self.scheduler / "jobs_state.json"
        self.audit = self.root / "gate_reassignments" / "fixture"
        self.started = 123456.0
        self.scheduler_pid, self.pipeline_pid, self.gate_pid = 700001, 700002, 700003
        self.receipt = self.scheduler / "launch_gates" / "selected.json"
        manifest = dict(phase="local_search_discovery_v1",output_root=str(self.discovery),
            planned_conditions=["full_d10"],conditions=[dict(id="full_d10")],discovery_inits=[36,37,38])
        manifest["manifest_hash"] = migration.digest(json.dumps(manifest,sort_keys=True,separators=(",", ":")).encode())
        self.manifest_path = self.discovery/"manifest.json";write(self.manifest_path,manifest)
        self.original_argv = ["/absolute/python", "-m", migration.LOCAL_RUNNER,
            "--output", str(self.discovery), "--condition", "full_d10", "--gpu-uuid", "{gpu_uuid}", "--port", "{port}", "--launch"]
        self.job = dict(id="selected", argv=["/absolute/python", "-m", migration.GATE,
            "--gpu-uuid", "{gpu_uuid}", "--status-file", str(self.receipt), "--", *self.original_argv],
            complete_file=str(self.discovery/"full_d10/discovery_shard00of01_complete.json"), depends_on=[], ready_files=[str(self.manifest_path)],
            scientific_seed=71, env=dict(CUDA_VISIBLE_DEVICES="{gpu_uuid}"),
            manifest_hash=manifest["manifest_hash"],scientific_phase=manifest["phase"],
            launch_gate=dict(original_argv=self.original_argv, continuous_idle_seconds=60, poll_seconds=10))
        self.queued = dict(phase="frozen", jobs=[self.job, dict(id="dependent", argv=["not_launched"],
                          complete_file=str(self.root / "dependent.json"), depends_on=["selected"])])
        self.state = dict(status="running", pid=self.gate_pid, gpu=0, gpu_uuid=SOURCE, port=54321,
                          started=self.started, start_ticks="111")
        self.state["argv"] = migration.concrete(self.job["argv"], self.state)
        write(self.queue, self.queued)
        write(self.states, dict(selected=self.state))
        write(self.scheduler / "controller_process.json", dict(pid=self.scheduler_pid, started=self.started))
        write(self.pipeline / "supervisor_process.json", dict(pid=self.pipeline_pid, started_unix=self.started))
        write(self.pipeline / "state.json", dict(phase="waiting_discovery"))
        write(self.pipeline / "status.json", dict(time=0,phase="waiting_discovery",terminal=False))
        self.runtime = FakeRuntime(self)
        common = dict(uid=os.getuid(), state="S", children=[], sigterm_default=True)
        self.runtime.processes = {
            self.scheduler_pid: dict(common, pid=self.scheduler_pid, ppid=1, start_ticks="100",
                                     argv=["/absolute/python", "-u", "-m", migration.SCHEDULER, "--queue", str(self.queue),
                                           "--output", str(self.scheduler)]),
            self.pipeline_pid: dict(common, pid=self.pipeline_pid, ppid=1, start_ticks="101",
                                    argv=["/absolute/python", "-u", "-m", migration.LOCAL_SUPERVISOR, "--root", str(self.discovery), "--launch"]),
            self.gate_pid: dict(common, pid=self.gate_pid, ppid=self.scheduler_pid, start_ticks="111", argv=self.state["argv"]),
        }
        write(self.receipt, dict(status="busy", pid=self.gate_pid, gpu_uuid=SOURCE, idle_seconds=0, elapsed_seconds=4000))
        os.utime(self.receipt, (self.runtime.clock, self.runtime.clock))
        self.args = argparse.Namespace(queue=str(self.queue), jobs_state=str(self.states), scheduler_output=str(self.scheduler),
            job_id="selected", candidate_gpu_uuid=CANDIDATE, output_audit=str(self.audit), apply=True)
        self.root_patch = patch.object(migration, "EXPERIMENT_ROOT", self.root)
        self.root_patch.start()
        self.timeout_patch = patch.object(migration, "ACK_TIMEOUT", 1.0)
        self.timeout_patch.start()

    def tearDown(self):
        self.timeout_patch.stop()
        self.root_patch.stop()
        self.temp.cleanup()

    def run_tool(self):
        return migration.operate(self.args, self.runtime)

    def signal_names(self):
        return [sig for pid, sig in self.runtime.signals]

    def test_default_dryrun_only_writes_audit_and_plan(self):
        self.args.apply = False
        original = self.queue.read_bytes()
        state = self.states.read_bytes()
        result = self.run_tool()
        self.assertEqual(result["stage"], "dry_run")
        self.assertEqual(self.runtime.signals, [])
        self.assertEqual(self.queue.read_bytes(), original)
        self.assertEqual(self.states.read_bytes(), state)
        self.assertEqual(self.runtime.queries, [SOURCE, CANDIDATE])
        self.assertTrue((self.audit / "queue.before.json").exists())

    def test_rejects_model_pid_and_identity_mismatch(self):
        for change in [dict(argv=self.original_argv), dict(start_ticks="112"), dict(ppid=42), dict(state="T")]:
            with self.subTest(change=change):
                saved = copy.deepcopy(self.runtime.processes[self.gate_pid])
                self.runtime.processes[self.gate_pid].update(change)
                with self.assertRaises(migration.Refused):
                    self.run_tool()
                self.runtime.processes[self.gate_pid] = saved
        self.assertEqual(self.runtime.signals, [])
        self.assertEqual(self.runtime.queries, [])

    def test_rejects_busy_candidate_busy_lease_and_idle_source(self):
        for field, value in [("target_idle", False), ("lease_free", False), ("source_idle", True)]:
            saved = getattr(self.runtime, field)
            setattr(self.runtime, field, value)
            with self.assertRaises(migration.Refused):
                self.run_tool()
            setattr(self.runtime, field, saved)
        self.assertEqual(self.runtime.signals, [])

    def test_rejects_completed_job_stale_receipt_or_enqueue_phase(self):
        Path(self.job["complete_file"]).parent.mkdir(parents=True,exist_ok=True)
        Path(self.job["complete_file"]).touch()
        with self.assertRaises(migration.Refused):
            self.run_tool()
        Path(self.job["complete_file"]).unlink()
        os.utime(self.receipt, (self.runtime.clock - 30, self.runtime.clock - 30))
        with self.assertRaises(migration.Refused):
            self.run_tool()
        os.utime(self.receipt, (self.runtime.clock, self.runtime.clock))
        write(self.pipeline / "state.json", dict(phase="enqueue_confirmation"))
        with self.assertRaises(migration.Refused):
            self.run_tool()
        self.assertEqual(self.runtime.signals, [])

    def test_rejects_changed_queue_and_restores_wait_before_commit(self):
        def change_queue():
            queue = migration.read_json(self.queue)
            queue["external_change"] = True
            write(self.queue, queue)
        self.runtime.on_stop = change_queue
        with self.assertRaisesRegex(migration.Refused, "Queue changed"):
            self.run_tool()
        self.assertEqual(self.signal_names(), [signal.SIGSTOP, signal.SIGCONT])
        self.assertEqual(self.runtime.processes[self.gate_pid]["state"], "S")
        self.assertEqual(migration.read_json(self.audit / "audit.json")["stage"], "aborted_restored")

    def test_queue_write_failure_restores_old_cpu_wait(self):
        self.runtime.replace_mode = "fail_before"
        original = self.queue.read_bytes()
        with self.assertRaises(OSError):
            self.run_tool()
        self.assertEqual(self.queue.read_bytes(), original)
        self.assertEqual(self.signal_names(), [signal.SIGSTOP, signal.SIGCONT])

    def test_exec_race_resumes_model_without_term_or_queue_commit(self):
        self.runtime.on_stop = lambda: self.runtime.processes[self.gate_pid].update(argv=migration.concrete(self.original_argv,self.state))
        original = self.queue.read_bytes()
        with self.assertRaisesRegex(migration.Refused, "exec identity changed"):
            self.run_tool()
        self.assertEqual(self.queue.read_bytes(), original)
        self.assertEqual(self.signal_names(), [signal.SIGSTOP, signal.SIGCONT])

    def test_commit_preserves_science_updates_dependencies_and_is_idempotent(self):
        result = self.run_tool()
        queue = migration.read_json(self.queue)
        new = result["new_job"]
        expected = copy.deepcopy(self.job)
        expected["id"] = new["id"]
        expected["argv"][6] = new["argv"][6]
        self.assertEqual(new, expected)
        self.assertEqual(new["argv"][8:], self.original_argv)
        self.assertEqual(queue["jobs"][1]["depends_on"], [new["id"]])
        self.assertEqual(migration.read_json(self.states)["selected"], self.state)
        self.assertEqual(result["stage"], "complete")
        self.assertEqual(self.signal_names(), [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT])
        self.assertTrue(self.runtime.exited(self.gate_pid))
        queries, signals, queue_bytes = list(self.runtime.queries), list(self.runtime.signals), self.queue.read_bytes()
        self.assertEqual(self.run_tool()["stage"], "complete")
        self.assertEqual(self.runtime.queries, queries)
        self.assertEqual(self.runtime.signals, signals)
        self.assertEqual(self.queue.read_bytes(), queue_bytes)

    def test_waits_for_both_acks_and_repeat_apply_only_finishes_cleanup(self):
        self.runtime.pipeline_ack = False
        with self.assertRaisesRegex(migration.Refused, "acknowledgements"):
            self.run_tool()
        self.assertEqual(self.signal_names(), [signal.SIGSTOP])
        self.assertEqual(self.runtime.processes[self.gate_pid]["state"], "T")
        self.assertEqual(migration.read_json(self.audit / "audit.json")["stage"], "needs_cleanup")
        committed_queue = self.queue.read_bytes()
        query_count = len(self.runtime.queries)
        self.runtime.pipeline_ack = True
        self.assertEqual(self.run_tool()["stage"], "complete")
        self.assertEqual(self.queue.read_bytes(), committed_queue)
        self.assertEqual(len(self.runtime.queries), query_count)
        self.assertEqual(self.signal_names(), [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT])

    def test_replace_success_then_exception_never_resumes_old_wait(self):
        self.runtime.replace_mode = "fail_after"
        with self.assertRaises(OSError):
            self.run_tool()
        self.assertEqual(self.signal_names(), [signal.SIGSTOP])
        self.assertEqual(migration.read_json(self.audit / "audit.json")["commit_outcome"], "committed")
        self.runtime.replace_mode = "success"
        self.assertEqual(self.run_tool()["stage"], "complete")
        self.assertEqual(self.signal_names(), [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT])

    def test_identity_is_rechecked_after_slow_cleanup_audit_write(self):
        real_save = migration.save_audit

        def save(paths, audit, stage, runtime, **details):
            real_save(paths, audit, stage, runtime, **details)
            if stage == "terminating_cpu_gate":
                self.runtime.processes[self.gate_pid]["argv"] = self.original_argv

        with patch.object(migration, "save_audit", side_effect=save):
            with self.assertRaisesRegex(migration.Refused, "no longer the same stopped"):
                self.run_tool()
        self.assertEqual(self.signal_names(), [signal.SIGSTOP])
        self.assertEqual(migration.read_json(self.audit / "audit.json")["stage"], "needs_cleanup")

    def test_ambiguous_or_unreadable_commit_never_resumes_old_wait(self):
        self.runtime.replace_mode = "ambiguous"
        with self.assertRaises(OSError):
            self.run_tool()
        self.assertEqual(self.signal_names(), [signal.SIGSTOP])
        receipt = migration.read_json(self.audit / "audit.json")
        self.assertEqual(receipt["stage"], "needs_cleanup")
        self.assertEqual(receipt["commit_outcome"], "unknown")
        with self.assertRaisesRegex(migration.Refused, "ambiguous"):
            self.run_tool()
        self.assertEqual(self.signal_names(), [signal.SIGSTOP])

    def test_unreadable_commit_keeps_stopped_and_can_recover_verified_proposal(self):
        self.runtime.replace_mode = "unreadable"
        with self.assertRaises(OSError):
            self.run_tool()
        self.assertEqual(self.signal_names(), [signal.SIGSTOP])
        self.queue.write_bytes((self.audit / "queue.proposed.json").read_bytes())
        self.assertEqual(self.run_tool()["stage"], "complete")
        self.assertEqual(self.signal_names(), [signal.SIGSTOP, signal.SIGTERM, signal.SIGCONT])

    def test_only_fixed_experiment_paths_are_allowed(self):
        self.args.queue = str(self.root / "other_queue.json")
        with self.assertRaisesRegex(migration.Refused, "standard queue"):
            self.run_tool()
        self.assertEqual(self.runtime.signals, [])
        self.assertEqual(self.runtime.queries, [])

    def stopped_crash_fixture(self, inner=True):
        self.args.apply=False
        self.run_tool()
        audit=migration.read_json(self.audit/"audit.json")
        audit["stop_authorization"]=migration.stop_authorization(audit)
        audit["stage"]="stop_intent"
        audit["events"].append(dict(stage="stop_intent",at=self.runtime.now()))
        write(self.audit/"audit.json",audit)
        self.runtime.processes[self.gate_pid].update(state="T")
        if inner:
            self.runtime.processes[self.gate_pid]["argv"]=migration.concrete(self.original_argv,self.state)
        self.args.apply=True
        return audit

    def test_python_unbuffered_and_no_bytecode_flags_are_parsed_exactly(self):
        for flags in ([],["-u"],["-B"],["-u","-B"],["-Bu"]):
            argv=["/absolute/python3.10",*flags,"-m",migration.LOCAL_SUPERVISOR,"--launch"]
            self.assertTrue(migration.exact_module(argv,migration.LOCAL_SUPERVISOR))
        for argv in (["python","-m",migration.GATE], ["/bin/sh","-m",migration.GATE],
            ["/absolute/python","-c","pass","-m",migration.GATE],
            ["/absolute/python","-X","presite=evil","-m",migration.GATE],
            ["/absolute/python","-Wignore","-m",migration.GATE],
            ["/absolute/python","-u","-u","-m",migration.GATE],
            ["/absolute/python","-m",migration.GATE+".other"]):
            self.assertFalse(migration.exact_module(argv,migration.GATE))

    def test_gate_python_uB_preserves_inner_command_and_receipt_slot(self):
        self.job["argv"][1:1]=["-uB"]
        self.state["argv"]=migration.concrete(self.job["argv"],self.state)
        self.runtime.processes[self.gate_pid]["argv"]=self.state["argv"]
        write(self.queue,self.queued);write(self.states,dict(selected=self.state))
        result=self.run_tool()
        self.assertEqual(migration.gate_parts(result["new_job"]["argv"])[2],self.original_argv)
        self.assertEqual(result["new_job"]["argv"][1],"-uB")

    def test_local_job_cannot_use_legacy_supervisor_receipt(self):
        self.runtime.processes[self.pipeline_pid]["argv"]=["/absolute/python","-u","-m",migration.PIPELINE,"--root",str(self.root),"--launch"]
        with self.assertRaisesRegex(migration.Refused,"verified live owned controller"):
            self.run_tool()
        self.assertEqual(self.runtime.signals,[])

    def test_legacy_status_cannot_substitute_for_local_ack(self):
        self.runtime.pipeline_ack=False
        original_sleep=self.runtime.sleep
        def sleep(seconds):
            original_sleep(seconds)
            write(self.root/"pipeline/status.json",dict(time=self.runtime.clock,branches=dict(vlm="waiting_confirmation"),terminal=False))
        self.runtime.sleep=sleep
        with self.assertRaisesRegex(migration.Refused,"acknowledgements"):
            self.run_tool()
        self.assertEqual(self.signal_names(),[signal.SIGSTOP])
        self.assertEqual(migration.read_json(self.audit/"audit.json")["profile"]["module"],migration.LOCAL_SUPERVISOR)

    def test_terminal_local_status_is_not_a_valid_ack(self):
        original_sleep=self.runtime.sleep
        def sleep(seconds):
            original_sleep(seconds)
            write(self.pipeline/"status.json",dict(time=self.runtime.clock,phase="needs_attention",terminal=True))
        self.runtime.sleep=sleep
        with self.assertRaisesRegex(migration.Refused,"nonterminal waiting"):
            self.run_tool()
        self.assertEqual(self.signal_names(),[signal.SIGSTOP])

    def test_supervisor_pid_reuse_after_commit_blocks_cleanup(self):
        self.runtime.pipeline_ack=False
        with self.assertRaises(migration.Refused):self.run_tool()
        self.runtime.processes[self.pipeline_pid]["start_ticks"]="reused"
        self.runtime.pipeline_ack=True
        with self.assertRaisesRegex(migration.Refused,"supervisor identity"):
            self.run_tool()
        self.assertEqual(self.signal_names(),[signal.SIGSTOP])

    def test_crash_after_stop_exact_inner_runner_is_only_continued(self):
        self.stopped_crash_fixture()
        self.runtime.processes[self.gate_pid]["children"]=["800001"]
        original=self.queue.read_bytes()
        result=self.run_tool()
        self.assertEqual(result["stage"],"aborted_restored")
        self.assertEqual(self.signal_names(),[signal.SIGCONT])
        self.assertEqual(self.queue.read_bytes(),original)
        self.assertEqual(self.runtime.processes[self.gate_pid]["state"],"S")
        self.assertEqual(self.run_tool()["stage"],"aborted_restored")
        self.assertEqual(self.signal_names(),[signal.SIGCONT])

    def test_crash_after_stop_cpu_gate_is_only_continued(self):
        self.stopped_crash_fixture(inner=False)
        self.assertEqual(self.run_tool()["stage"],"aborted_restored")
        self.assertEqual(self.signal_names(),[signal.SIGCONT])

    def test_arbitrary_or_merely_same_module_inner_argv_is_not_restored(self):
        self.stopped_crash_fixture()
        self.runtime.processes[self.gate_pid]["argv"][-1]="--preflight-only"
        with self.assertRaisesRegex(migration.Refused,"exact owned gate"):
            self.run_tool()
        self.assertEqual(self.runtime.signals,[])
        self.assertEqual(self.runtime.processes[self.gate_pid]["state"],"T")

    def test_crash_restore_requires_same_start_owner_and_parent(self):
        self.stopped_crash_fixture()
        for change in (dict(start_ticks="reused"),dict(uid=os.getuid()+1),dict(ppid=700099)):
            before=copy.deepcopy(self.runtime.processes[self.gate_pid])
            self.runtime.processes[self.gate_pid].update(change)
            with self.assertRaises(migration.Refused):self.run_tool()
            self.runtime.processes[self.gate_pid]=before
        self.assertEqual(self.runtime.signals,[])

    def test_crash_restore_rechecks_identity_after_pidfd_open(self):
        self.stopped_crash_fixture()
        original_open=self.runtime.open_pidfd
        def open_pidfd(pid):
            fd=original_open(pid)
            self.runtime.processes[pid]["argv"]=["/absolute/python","-m","different.module"]
            return fd
        self.runtime.open_pidfd=open_pidfd
        with self.assertRaises(migration.Refused):self.run_tool()
        self.assertEqual(self.runtime.signals,[])

    def test_crash_restore_rechecks_after_slow_restore_audit_write(self):
        self.stopped_crash_fixture()
        original_save=migration.save_audit
        def save(paths,audit,stage,runtime,**details):
            original_save(paths,audit,stage,runtime,**details)
            if stage=="precommit_restore_intent":self.runtime.processes[self.gate_pid]["ppid"]=700099
        with patch.object(migration,"save_audit",side_effect=save):
            with self.assertRaises(migration.Refused):self.run_tool()
        self.assertEqual(self.runtime.signals,[])

    def test_crash_restore_requires_durable_own_stop_intent(self):
        audit=self.stopped_crash_fixture()
        audit.pop("stop_authorization");write(self.audit/"audit.json",audit)
        with self.assertRaisesRegex(migration.Refused,"STOP intent"):
            self.run_tool()
        self.assertEqual(self.runtime.signals,[])

    def test_unknown_commit_cannot_continue_crash_stopped_inner_runner(self):
        self.stopped_crash_fixture()
        write(self.queue,dict(jobs=[]))
        with self.assertRaisesRegex(migration.Refused,"ambiguous"):
            self.run_tool()
        self.assertEqual(self.runtime.signals,[])

    def test_committed_queue_cannot_continue_or_terminate_inner_runner(self):
        self.stopped_crash_fixture()
        self.queue.write_bytes((self.audit/"queue.proposed.json").read_bytes())
        with self.assertRaisesRegex(migration.Refused,"Recovery PID identity changed"):
            self.run_tool()
        self.assertEqual(self.runtime.signals,[])

    def test_v1_audit_is_not_silently_upgraded(self):
        audit=self.stopped_crash_fixture(inner=False)
        audit.pop("schema");write(self.audit/"audit.json",audit)
        with self.assertRaisesRegex(migration.Refused,"exact v2"):
            self.run_tool()
        self.assertEqual(self.runtime.signals,[])

    def test_wrong_local_output_or_complete_marker_is_rejected(self):
        original_marker=self.job["complete_file"]
        self.job["complete_file"]=str(self.root/"foreign_complete.json");write(self.queue,self.queued)
        with self.assertRaisesRegex(migration.Refused,"Completion marker"):
            self.run_tool()
        self.job["complete_file"]=original_marker
        index=self.original_argv.index("--output")+1
        self.original_argv[index]=str(self.root/"vlm_confirmation")
        self.job["argv"]=self.job["argv"][:8]+self.original_argv
        self.state["argv"]=migration.concrete(self.job["argv"],self.state)
        self.runtime.processes[self.gate_pid]["argv"]=self.state["argv"]
        write(self.queue,self.queued);write(self.states,dict(selected=self.state))
        with self.assertRaisesRegex(migration.Refused,"fixed local-search stage"):
            self.run_tool()
        self.assertEqual(self.runtime.signals,[])

    def test_legacy_vlm_profile_uses_only_bound_legacy_pipeline(self):
        legacy=self.root/"vlm_confirmation";path=legacy/"manifest.json"
        manifest=dict(phase="independent_confirmation",planned_conditions=["full_d10"],
            conditions=[dict(id="full_d10")],confirmation_inits=list(range(31,36)))
        manifest["manifest_hash"]=migration.digest(json.dumps(manifest,sort_keys=True,separators=(",", ":")).encode())
        write(path,manifest)
        self.original_argv[2]=migration.LEGACY_CONFIRM_RUNNER
        self.original_argv[self.original_argv.index("--output")+1]=str(legacy)
        self.job["argv"]=self.job["argv"][:8]+self.original_argv
        self.job["ready_files"]=[str(path)]
        self.job["complete_file"]=str(legacy/"full_d10/confirm_shard00of01_complete.json")
        self.state["argv"]=migration.concrete(self.job["argv"],self.state)
        self.runtime.processes[self.gate_pid]["argv"]=self.state["argv"]
        self.pipeline=self.root/"pipeline"
        write(self.pipeline/"supervisor_process.json",dict(pid=self.pipeline_pid,started_unix=self.started))
        write(self.pipeline/"state.json",dict(branches=dict(vlm=dict(phase="waiting_confirmation"),draft=dict(phase="report_ready"))))
        write(self.pipeline/"status.json",dict(time=0,branches=dict(vlm="waiting_confirmation",draft="report_ready"),terminal=False))
        self.runtime.processes[self.pipeline_pid]["argv"]=["/absolute/python","-u","-m",migration.PIPELINE,"--root",str(self.root),"--launch"]
        write(self.queue,self.queued);write(self.states,dict(selected=self.state))
        original_sleep=self.runtime.sleep
        def sleep(seconds):
            original_sleep(seconds)
            write(self.pipeline/"status.json",dict(time=self.runtime.clock,branches=dict(vlm="waiting_confirmation",draft="report_ready"),terminal=False))
        self.runtime.sleep=sleep
        result=self.run_tool()
        self.assertEqual(result["profile"]["module"],migration.PIPELINE)
        self.assertEqual(result["stage"],"complete")

    def test_pending_or_ptraced_stop_is_not_claimed_restored(self):
        self.stopped_crash_fixture()
        for state in ("D","S","R","t"):
            self.runtime.processes[self.gate_pid]["state"]=state
            with self.assertRaises(migration.Refused):self.run_tool()
        self.assertEqual(self.runtime.signals,[])
        self.assertEqual(migration.read_json(self.audit/"audit.json")["stage"],"stop_intent")

    def test_precommit_restore_cont_then_crash_is_idempotent_without_more_signals(self):
        audit=self.stopped_crash_fixture()
        audit.update(stage="precommit_restore_intent",restore_from_state="T")
        write(self.audit/"audit.json",audit)
        self.runtime.processes[self.gate_pid]["state"]="S"
        self.assertEqual(self.run_tool()["stage"],"aborted_restored")
        self.assertEqual(self.runtime.signals,[])

    def test_persisted_new_id_prevents_false_not_committed_recovery(self):
        audit=self.stopped_crash_fixture()
        states=migration.read_json(self.states);states[audit["new_job"]["id"]]=dict(status="pending");write(self.states,states)
        with self.assertRaisesRegex(migration.Refused,"persisted replacement ID"):
            self.run_tool()
        self.assertEqual(self.runtime.signals,[])


if __name__ == "__main__":
    unittest.main()

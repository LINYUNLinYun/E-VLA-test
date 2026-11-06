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

from preparations.ease_insights_20261103 import reassign_waiting_gate as migration


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
            assert process["argv"][2] == migration.GATE
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
                write(self.fixture.pipeline / "status.json", dict(time=self.clock))


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "experiment"
        self.queue = self.root / "queue.json"
        self.scheduler = self.root / "scheduler"
        self.pipeline = self.root / "pipeline"
        self.states = self.scheduler / "jobs_state.json"
        self.audit = self.root / "gate_reassignments" / "fixture"
        self.started = 123456.0
        self.scheduler_pid, self.pipeline_pid, self.gate_pid = 700001, 700002, 700003
        self.receipt = self.scheduler / "launch_gates" / "selected.json"
        self.original_argv = ["/absolute/python", "-m", "preparations.ease_insights_20261103.vlm.confirm_runner",
                              "--condition", "frozen", "--gpu", "{gpu}", "--port", "{port}"]
        self.job = dict(id="selected", argv=["/absolute/python", "-m", migration.GATE,
                        "--gpu-uuid", "{gpu_uuid}", "--status-file", str(self.receipt), "--", *self.original_argv],
                        complete_file=str(self.root / "frozen_completion.json"), depends_on=[], ready_files=["frozen_manifest.json"],
                        scientific_seed=71, env=dict(CUDA_VISIBLE_DEVICES="{gpu_uuid}"),
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
        write(self.pipeline / "state.json", dict(branches={name: dict(phase="waiting_confirmation") for name in ("draft", "vlm")}))
        write(self.pipeline / "status.json", dict(time=0))
        self.runtime = FakeRuntime(self)
        common = dict(uid=os.getuid(), state="S", children=[], sigterm_default=True)
        self.runtime.processes = {
            self.scheduler_pid: dict(common, pid=self.scheduler_pid, ppid=1, start_ticks="100",
                                     argv=["/absolute/python", "-m", migration.SCHEDULER, "--queue", str(self.queue),
                                           "--output", str(self.scheduler)]),
            self.pipeline_pid: dict(common, pid=self.pipeline_pid, ppid=1, start_ticks="101",
                                    argv=["/absolute/python", "-m", migration.PIPELINE, "--root", str(self.root), "--launch"]),
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
        Path(self.job["complete_file"]).touch()
        with self.assertRaises(migration.Refused):
            self.run_tool()
        Path(self.job["complete_file"]).unlink()
        os.utime(self.receipt, (self.runtime.clock - 30, self.runtime.clock - 30))
        with self.assertRaises(migration.Refused):
            self.run_tool()
        os.utime(self.receipt, (self.runtime.clock, self.runtime.clock))
        write(self.pipeline / "state.json", dict(branches=dict(vlm=dict(phase="enqueue_confirmation"))))
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
        self.runtime.on_stop = lambda: self.runtime.processes[self.gate_pid].update(argv=self.original_argv)
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


if __name__ == "__main__":
    unittest.main()

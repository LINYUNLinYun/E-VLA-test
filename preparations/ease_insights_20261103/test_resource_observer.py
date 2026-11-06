"""CPU-only observer fixtures; all GPU queries are mocked."""
import argparse
import contextlib
import fcntl
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from preparations.ease_insights_20261103 import resource_observer as observer


UUID = "GPU-12345678-1234-1234-1234-123456789abc"
OTHER = "GPU-87654321-1234-1234-1234-123456789abc"
ARGV = ["/absolute/python", "-m", "experiment", "--gpu", UUID]


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.proc = self.root / "proc"
        self.queue = dict(jobs=[dict(id="job", argv=ARGV)])
        self.states = dict(job=dict(status="running", pid=42, gpu_uuid=UUID, argv=ARGV, proc_start_ticks="123"))
        self.process(42)

    def tearDown(self):
        self.temporary.cleanup()

    def process(self, pid, argv=ARGV, ticks="123", state="S"):
        path = self.proc / str(pid)
        path.mkdir(parents=True, exist_ok=True)
        fields = [state] + ["0"] * 18 + [ticks]
        (path / "stat").write_text(f"{pid} (python (worker)) " + " ".join(fields))
        (path / "cmdline").write_bytes(b"\0".join(x.encode() for x in argv) + b"\0")

    def test_no_running_current_job_never_queries_gpu(self):
        for queue, state in [(dict(jobs=[]), self.states), (self.queue, {}),
                             (self.queue, dict(job=dict(status="pending")))]:
            with patch.object(observer.subprocess, "check_output") as query:
                record, status = observer.sample(queue, state, self.proc)
            query.assert_not_called()
            self.assertIsNone(record)
            self.assertEqual(status["status"], "waiting_experiment")

    def test_gate_wait_does_not_query_and_exec_is_recognized(self):
        wrapped = ["/absolute/python", "-m", observer.GATE_MODULE, "--gpu-uuid", UUID,
                   "--status-file", "gate.json", "--", *ARGV]
        self.states["job"]["argv"] = wrapped
        self.process(42, wrapped)
        with patch.object(observer.subprocess, "check_output") as query:
            record, status = observer.sample(self.queue, self.states, self.proc)
        query.assert_not_called()
        self.assertIsNone(record)
        self.assertEqual(status["skipped_jobs"][0]["reason"], "waiting_launch_gate")
        self.process(42)
        jobs, skipped = observer.running_jobs(self.queue, self.states, self.proc)
        self.assertEqual(len(jobs), 1)
        self.assertEqual(skipped, [])

    def test_gate_python_options_and_script_separator_are_recognized(self):
        targets = [["-W", "default", "-m", observer.GATE_MODULE],
                   ["-X", "dev", "-m", observer.GATE_MODULE],
                   ["--check-hash-based-pycs", "always", "-m", observer.GATE_MODULE],
                   ["--", "/absolute/await_assigned_gpu.py"]]
        for target in targets:
            wrapped = ["/absolute/python", *target, "--gpu-uuid", UUID, "--status-file", "gate.json", "--", *ARGV]
            self.states["job"]["argv"] = wrapped
            self.process(42, wrapped)
            with patch.object(observer.subprocess, "check_output") as query:
                record, status = observer.sample(self.queue, self.states, self.proc)
            query.assert_not_called()
            self.assertIsNone(record)
            self.assertEqual(status["skipped_jobs"][0]["reason"], "waiting_launch_gate")
            self.process(42)
            self.assertEqual(len(observer.running_jobs(self.queue, self.states, self.proc)[0]), 1)

    def test_pid_identity_and_dead_unknown_processes_are_preserved(self):
        jobs, skipped = observer.running_jobs(self.queue, self.states, self.proc)
        self.assertEqual(jobs[0]["local_pid"], 42)
        self.assertEqual(jobs[0]["proc_start_ticks"], "123")
        self.states["job"]["proc_start_ticks"] = "122"
        self.assertEqual(observer.running_jobs(self.queue, self.states, self.proc)[1][0]["reason"],
                         "process_start_ticks_mismatch")
        self.states["job"]["proc_start_ticks"] = "123"
        for argv, state, reason in [(["unrelated"], "S", "process_argv_mismatch"),
                                    (ARGV, "Z", "process_dead"), ([], "S", "process_unknown")]:
            self.process(42, argv=argv, state=state)
            with patch.object(observer.subprocess, "check_output") as query:
                record, status = observer.sample(self.queue, self.states, self.proc)
            query.assert_not_called()
            self.assertIsNone(record)
            self.assertEqual(status["skipped_jobs"][0]["reason"], reason)

    def test_pid_reuse_during_proc_read_is_unknown(self):
        before = (self.proc / "42" / "stat").read_text()
        after = before.rsplit(" ", 1)[0] + " 124"
        with patch.object(Path, "read_text", side_effect=[before, after]):
            process = observer.proc_identity(42, self.proc)
        self.assertEqual(process["status"], "unknown")
        self.assertEqual(process["reason"], "process_identity_changed")

    def test_scheduler_epoch_start_fallback_rejects_reused_pid(self):
        del self.states["job"]["proc_start_ticks"]
        self.states["job"]["started"] = 1000002
        (self.proc / "stat").write_text("btime 1000000\n")
        with patch.object(observer.os, "sysconf", return_value=100):
            jobs, skipped = observer.running_jobs(self.queue, self.states, self.proc)
            self.assertEqual(jobs[0]["proc_started_at"], 1000001.23)
            self.assertEqual(skipped, [])
            self.states["job"]["started"] = 999900
            self.assertEqual(observer.running_jobs(self.queue, self.states, self.proc)[1][0]["reason"],
                             "process_start_time_mismatch")
            (self.proc / "stat").unlink()
            self.assertEqual(observer.running_jobs(self.queue, self.states, self.proc)[1][0]["reason"],
                             "unknown_recorded_process_start")

    def test_same_queue_id_with_changed_command_does_not_use_old_state(self):
        self.queue["jobs"][0]["argv"] = ["/absolute/python", "replacement.py"]
        with patch.object(observer.subprocess, "check_output") as query:
            record, status = observer.sample(self.queue, self.states, self.proc)
        query.assert_not_called()
        self.assertIsNone(record)
        self.assertEqual(status["skipped_jobs"][0]["reason"], "unknown_command_identity")

    def test_only_assigned_uuid_and_host_pid_is_not_local(self):
        replies = [f"{UUID}, 1234, 80\n", f"{UUID}, 99999, 1000\n{UUID}, 88888, 234\n"]
        with patch.object(observer.subprocess, "check_output", side_effect=replies) as query:
            record, status = observer.sample(self.queue, self.states, self.proc)
        self.assertEqual(query.call_count, 2)
        for call in query.call_args_list:
            self.assertIn("--id=" + UUID, call.args[0])
            self.assertEqual(call.kwargs["timeout"], 5)
        gpu = record["gpus"][0]
        self.assertEqual(gpu["contention_indication"], "multiple_compute_processes")
        self.assertEqual(gpu["compute_processes"][0]["host_pid"], 99999)
        self.assertEqual(record["jobs"][0]["local_pid"], 42)
        self.assertEqual(gpu["raw"]["compute_apps"], replies[1])
        self.assertEqual(status["status"], "observing")
        self.assertLessEqual(record["query_started_at"], record["query_finished_at"])

    def test_errors_unknown_readings_and_raw_output_are_retained(self):
        replies = [subprocess.CalledProcessError(1, "nvidia-smi", output="partial", stderr="failure"),
                   f"{UUID}, N/A, [Not Supported]\n"]
        with patch.object(observer.subprocess, "check_output", side_effect=replies):
            record, status = observer.sample(self.queue, self.states, self.proc)
        gpu = record["gpus"][0]
        self.assertIsNone(gpu["memory_used_mib"])
        self.assertIsNone(gpu["compute_processes"][0]["host_pid"])
        self.assertIsNone(gpu["compute_processes"][0]["used_memory_mib"])
        self.assertEqual(record["query_errors"][0]["stderr"], "failure")
        self.assertEqual(status["status"], "query_error")
        self.assertEqual(record["multi_gpu_processes"], {})
        with patch.object(observer.subprocess, "check_output", side_effect=[f"{UUID}, N/A, N/A\n",
                                                                                 subprocess.TimeoutExpired("smi", 5)]):
            result = observer.query_gpu(UUID)
        self.assertIsNone(result["compute_processes"])
        self.assertEqual(result["contention_indication"], "unknown")
        self.assertIsNone(result["utilization_gpu_percent"])

    def test_multiple_gpu_processes_and_duplicate_assignment_aggregation(self):
        other_argv = ["/absolute/python", "other.py"]
        self.queue["jobs"].extend([dict(id="second", argv=other_argv), dict(id="third", argv=ARGV)])
        self.states.update(second=dict(status="running", pid=43, gpu_uuid=OTHER, argv=other_argv, proc_start_ticks="123"),
                           third=dict(status="running", pid=44, gpu_uuid=UUID, argv=ARGV, proc_start_ticks="123"))
        self.process(43, other_argv)
        self.process(44)

        def query(argv, **kwargs):
            gpu = next(arg[5:] for arg in argv if arg.startswith("--id="))
            return f"{gpu}, 100, 10\n" if any(arg.startswith("--query-gpu=") for arg in argv) else f"{gpu}, 9090, 100\n"

        with patch.object(observer.subprocess, "check_output", side_effect=query) as check:
            record, status = observer.sample(self.queue, self.states, self.proc)
        self.assertEqual(check.call_count, 4)
        self.assertEqual(len(record["jobs"]), 3)
        self.assertEqual(record["multi_gpu_processes"], {"9090": sorted([UUID, OTHER])})
        self.assertEqual(set(status["queried_gpu_uuids"]), {UUID, OTHER})

    def test_wrong_uuid_never_used_and_invalid_selectors_never_queried(self):
        with patch.object(observer.subprocess, "check_output", side_effect=[f"{OTHER}, 0, 0\n", f"{OTHER}, 2, 3\n"]):
            result = observer.query_gpu(UUID)
        self.assertEqual(len(result["query_errors"]), 2)
        self.assertIsNone(result["compute_processes"])
        for invalid in ["0", UUID + "," + OTHER]:
            with patch.object(observer.subprocess, "check_output") as query:
                with self.assertRaises(ValueError):
                    observer.query_gpu(invalid)
            query.assert_not_called()

    def test_once_appends_only_running_samples_and_prevents_duplicate_observer(self):
        queue, states, output = self.root / "queue.json", self.root / "state.json", self.root / "observer"
        queue.write_text(json.dumps(dict(jobs=[])))
        states.write_text("{}")
        args = argparse.Namespace(queue=str(queue), jobs_state=str(states), output=str(output), once=True)
        with patch.object(observer.subprocess, "check_output") as query, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(observer.run(args), 0)
        query.assert_not_called()
        self.assertFalse((output / "samples.jsonl").exists())
        heartbeat = json.loads((output / "status.json").read_text())
        self.assertEqual(heartbeat["status"], "waiting_experiment")
        self.assertNotIn("observed_at", heartbeat)
        record = dict(observed_at=1, jobs=[dict(job_id="job")])
        with patch.object(observer, "sample", return_value=(record, dict(status="observing"))), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(observer.run(args), 0)
            self.assertEqual(observer.run(args), 0)
        self.assertEqual(len((output / "samples.jsonl").read_text().splitlines()), 2)
        with (output / "observer.lock").open("a") as lock, patch.object(observer, "sample") as sample, \
             contextlib.redirect_stdout(io.StringIO()):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(observer.run(args), 1)
        sample.assert_not_called()
        self.assertEqual(list(output.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()

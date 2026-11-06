"""CPU-only continuation tests. All queues/process probes are private fixtures."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from preparations.ease_insights_20261103.pipeline import MODULE as SCHEDULER_MODULE, write, sha
from .common import digest
from .supervisor import Paths, SharedSchedulerWatch, Supervisor, jobs_for


def selection_document(status="selected"):
    obj = dict(revision="ease-vlm-local-search-selection-v1", complete_discovery_episodes=720,
        expected_discovery_episodes=720, independent_validation_launched=False,
        selection=dict(status=status, selected=None if status == "no_candidate" else dict(paths=["jac_anchor", "local_goal"])))
    obj["selection_hash"] = digest(obj)
    return obj


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ease-supervisor-cpu-")
        self.base = Path(self.temporary.name)
        self.paths = Paths.defaults(self.base / "local/discovery", self.base / "local/confirmation",
            self.base / "queue.json", self.base / "scheduler", self.base / "local/supervisor")
        write(self.paths.queue, dict(jobs=[]))
        self.watch = Mock()
        self.watch.ensure.return_value = dict(status="live")
        self.watch.observe.return_value = dict(status="exited")

    def tearDown(self):
        self.temporary.cleanup()

    def supervisor(self, command_runner=None):
        return Supervisor(self.paths, command_runner=command_runner, scheduler_watch=self.watch)

    def selected_state(self, supervisor, phase):
        selection = self.paths.discovery / "analysis/selection.json"
        write(selection, selection_document())
        supervisor.state.update(phase=phase, selection=str(selection), selection_sha256=sha(selection))

    def test_waiting_does_not_run_analysis_or_write_queue(self):
        command = Mock(side_effect=AssertionError("Must not launch a stage before completion"))
        supervisor = self.supervisor(command)
        supervisor.completion_gate = Mock(return_value=dict(complete=False, completed=0, expected=6))
        before = self.paths.queue.read_bytes()
        result = supervisor.tick()
        self.assertEqual(result["phase"], "waiting_discovery")
        self.assertFalse(result["terminal"])
        self.assertEqual(self.paths.queue.read_bytes(), before)
        command.assert_not_called()
        self.watch.ensure.assert_called_once()

    def test_no_candidate_is_terminal_without_confirmation_or_expansion(self):
        def command(argv, log):
            write(self.paths.discovery / "analysis/selection.json", selection_document("no_candidate"))
            return 0
        supervisor = self.supervisor(command)
        supervisor.state["phase"] = "freeze_selection"
        result = supervisor.tick()
        self.assertEqual(result["phase"], "no_candidate")
        self.assertTrue(result["terminal"])
        self.assertEqual(json.loads(self.paths.queue.read_text())["jobs"], [])
        self.watch.ensure.assert_not_called()
        self.assertFalse(self.paths.confirmation.exists())

    def test_confirmation_queue_append_is_idempotent_and_has_idle_gate(self):
        supervisor = self.supervisor()
        self.selected_state(supervisor, "enqueue_confirmation")
        manifest = dict(phase="local_search_confirmation_v1", cases_per_condition=200,
            manifest_hash="fixture", planned_conditions=["full_d10", "jac_anchor", "local_goal"])
        jobs = jobs_for(self.paths.confirmation, manifest)
        supervisor.state["jobs"] = jobs
        first = supervisor.tick()
        self.assertEqual(first["phase"], "waiting_confirmation")
        queue = json.loads(self.paths.queue.read_text())
        self.assertEqual(len(queue["jobs"]), 3)
        for original, wrapped in zip(jobs, queue["jobs"]):
            self.assertEqual(wrapped["launch_gate"]["original_argv"], original["argv"])
            self.assertEqual(wrapped["launch_gate"]["continuous_idle_seconds"], 60)
        original_bytes = self.paths.queue.read_bytes()
        supervisor.state["phase"] = "enqueue_confirmation"
        supervisor.tick()
        self.assertEqual(self.paths.queue.read_bytes(), original_bytes)

    def test_selection_tampering_stops_before_queue_append(self):
        supervisor = self.supervisor()
        self.selected_state(supervisor, "enqueue_confirmation")
        Path(supervisor.state["selection"]).write_text("{}")
        result = supervisor.tick()
        self.assertEqual(result["phase"], "needs_attention")
        self.assertEqual(json.loads(self.paths.queue.read_text())["jobs"], [])

    def test_failed_cpu_action_is_not_automatically_retried(self):
        command = Mock(return_value=1)
        supervisor = self.supervisor(command)
        supervisor.state["phase"] = "freeze_selection"
        self.assertEqual(supervisor.tick()["phase"], "needs_attention")
        supervisor.tick()
        self.assertEqual(command.call_count, 1)

    def test_current_failed_job_needs_attention_but_historical_failure_does_not(self):
        manifest = dict(phase="local_search_discovery_v1", cases_per_condition=120,
            manifest_hash="fixture", planned_conditions=["full_d10"])
        job = jobs_for(self.paths.discovery, manifest)[0]
        supervisor = self.supervisor()
        with patch("preparations.ease_insights_20261103.vlm_local_search_v1.supervisor.read_manifest", return_value=manifest), patch(
                "preparations.ease_insights_20261103.vlm_local_search_v1.supervisor.check_scope"):
            gate = supervisor.completion_gate(self.paths.discovery, "discovery", dict(jobs=[job]), {"superseded_id": dict(status="failed")})
            self.assertFalse(gate["complete"])
            with self.assertRaises(RuntimeError):
                supervisor.completion_gate(self.paths.discovery, "discovery", dict(jobs=[job]), {job["id"]: dict(status="failed")})

    def test_shared_restart_receipt_is_adopted_without_second_scheduler(self):
        supervisor = Supervisor(self.paths)
        shared_paths = supervisor.watch.paths
        self.assertEqual(shared_paths.output, self.paths.queue.parent / "pipeline")
        argv = ["python3", "-m", SCHEDULER_MODULE, "--queue", str(self.paths.queue),
            "--output", str(self.paths.scheduler_output), "--max-gpus", "2", "--idle-seconds", "0"]
        write(shared_paths.output / "scheduler_restart.json", dict(pid=123))
        probe = Mock(return_value=dict(pid=123, status="live", argv=argv, start_ticks="1"))
        spawn = Mock(side_effect=AssertionError("Must adopt the original pipeline's scheduler"))
        watch = SharedSchedulerWatch(shared_paths, probe=probe, spawn=spawn)
        observation = watch.ensure({}, dict(jobs=[]), {})
        self.assertEqual(observation["status"], "live")
        spawn.assert_not_called()

    def test_unknown_process_identity_prevents_restart(self):
        supervisor = Supervisor(self.paths)
        write(supervisor.watch.paths.output / "scheduler_restart.json", dict(pid=456))
        probe = Mock(return_value=dict(pid=456, status="unknown"))
        spawn = Mock(side_effect=AssertionError("Unknown identity is not confirmed exit"))
        watch = SharedSchedulerWatch(supervisor.watch.paths, probe=probe, spawn=spawn)
        self.assertEqual(watch.ensure({}, dict(jobs=[]), {})["status"], "needs_attention")
        spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()

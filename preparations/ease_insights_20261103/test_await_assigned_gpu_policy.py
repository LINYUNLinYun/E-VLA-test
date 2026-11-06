"""CPU-only checks for resource policy and the final assigned-GPU recheck."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from preparations.ease_insights_20261103 import await_assigned_gpu as gate


UUID = "GPU-12345678-1234-1234-1234-123456789abc"
IDLE = dict(idle=True, memory_mib=0, compute_process_count=0)
BUSY = dict(idle=False, memory_mib=700, compute_process_count=1)
FAST = {"EASE_GPU_GATE_IDLE_SECONDS": "5", "EASE_GPU_GATE_POLL_SECONDS": "2"}


class Executed(Exception):
    pass


class PolicyTests(unittest.TestCase):
    def simulate(self, observations, environment=None):
        now = [0.0]
        records = []
        sleeps = []
        executed_environment = {}
        command = [sys.executable, "original.py", "--gpu-uuid", "literal arg", "--", "$(literal)"]

        def sleep(seconds):
            sleeps.append(seconds)
            if len(sleeps) > 30:
                self.fail("Gate failed to exec within the finite fake observation stream")
            now[0] += seconds

        def execute(program, argv):
            self.assertIs(argv, command)
            executed_environment.update(os.environ)
            raise Executed

        with patch.dict(os.environ, environment or {}, clear=True), \
             patch.object(gate.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(gate.time, "sleep", side_effect=sleep), \
             patch.object(gate, "scan", side_effect=observations) as scan, \
             patch.object(gate, "write_status", side_effect=lambda path, record: records.append(record)), \
             patch.object(gate.os, "execvp", side_effect=execute) as executed, \
             contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(Executed):
                gate.wait_and_exec(UUID, "unused.json", command)
        executed.assert_called_once_with(command[0], command)
        self.assertTrue(all(call.args == (UUID,) for call in scan.call_args_list))
        return records, sleeps, scan.call_count, executed_environment

    def test_default_policy_uses_twenty_five_seconds_and_two_second_poll(self):
        records, sleeps, calls, _ = self.simulate([IDLE] * 15)
        self.assertEqual([r["elapsed_seconds"] for r in records], list(range(0, 27, 2)))
        self.assertEqual(sleeps, [2] * 13)
        self.assertEqual(calls, 15)
        for record in records:
            self.assertEqual(record["idle_threshold_seconds"], 25)
            self.assertEqual(record["poll_seconds"], 2)
            self.assertEqual(record["resource_revision"], "assigned-gpu-idle-v2")
        self.assertEqual(records[-1]["final_idle_check"],
                         dict(gpu_uuid=UUID, status="idle", elapsed_seconds=26, **IDLE))

    def test_explicit_fast_policy_preserves_argv_and_only_strips_resource_env(self):
        environment = dict(FAST, CUDA_VISIBLE_DEVICES=UUID, SCIENTIFIC_SETTING="unchanged")
        records, sleeps, calls, executed_environment = self.simulate([IDLE] * 5, environment)
        self.assertEqual([r["elapsed_seconds"] for r in records], [0, 2, 4, 6])
        self.assertEqual(sleeps, [2] * 3)
        self.assertEqual(calls, 5)
        self.assertEqual(records[-1]["idle_threshold_seconds"], 5)
        self.assertEqual(records[-1]["idle_seconds"], 6)
        self.assertEqual(records[-1]["poll_seconds"], 2)
        self.assertEqual(executed_environment,
                         dict(CUDA_VISIBLE_DEVICES=UUID, SCIENTIFIC_SETTING="unchanged"))

    def test_overrides_are_independent_and_documented_bounds_are_valid(self):
        cases = [({gate.IDLE_ENV: "5"}, (5, 2)),
                 ({gate.POLL_ENV: "2"}, (25, 2)),
                 ({gate.IDLE_ENV: "1", gate.POLL_ENV: "0.1"}, (1, 0.1)),
                 ({gate.IDLE_ENV: "3600", gate.POLL_ENV: "60"}, (3600, 60))]
        for environment, expected in cases:
            with self.subTest(environment=environment), patch.dict(os.environ, environment, clear=True):
                self.assertEqual(gate.resource_policy(), expected)

    def test_invalid_configuration_refuses_to_query_or_exec(self):
        for name, invalid in [(gate.IDLE_ENV, "0.999"), (gate.IDLE_ENV, "3600.001"),
                              (gate.POLL_ENV, "0.099"), (gate.POLL_ENV, "60.001")]:
            self.assert_configuration_rejected(name, invalid)
        for name in (gate.IDLE_ENV, gate.POLL_ENV):
            for invalid in ("", " ", "oops", "0", "-1", "nan", "inf", "-inf", "1e999"):
                self.assert_configuration_rejected(name, invalid)

    def assert_configuration_rejected(self, name, value):
        with self.subTest(name=name, value=value), \
             patch.dict(os.environ, {name: value}, clear=True), \
             patch.object(gate, "scan") as scan, \
             patch.object(gate, "write_status") as status, \
             patch.object(gate.os, "execvp") as execute:
            with self.assertRaisesRegex(ValueError, name):
                gate.wait_and_exec(UUID, "unused.json", ["original"])
            scan.assert_not_called()
            status.assert_not_called()
            execute.assert_not_called()

    def test_cli_rejects_invalid_configuration_without_launching(self):
        with patch.dict(os.environ, {gate.IDLE_ENV: "nan"}, clear=True), \
             patch.object(gate, "scan") as scan, \
             patch.object(gate.os, "execvp") as execute, \
             contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(SystemExit) as error:
                gate.main(["--gpu-uuid", UUID, "--status-file", "unused.json", "--", "original"])
        self.assertEqual(error.exception.code, 2)
        self.assertIn(gate.IDLE_ENV, stderr.getvalue())
        scan.assert_not_called()
        execute.assert_not_called()

    def test_busy_and_query_error_each_reset_the_fast_idle_window(self):
        failure = subprocess.TimeoutExpired("fake-nvidia-smi", 15)
        records, sleeps, _, _ = self.simulate([IDLE, BUSY, IDLE, failure] + [IDLE] * 5, FAST)
        self.assertEqual([r["status"] for r in records],
                         ["waiting", "busy", "waiting", "check_error", "waiting",
                          "waiting", "waiting", "launching"])
        self.assertEqual(records[1]["idle_seconds"], 0)
        self.assertEqual(records[3]["idle_seconds"], 0)
        self.assertEqual(records[-1]["elapsed_seconds"], 14)
        self.assertEqual(records[-1]["idle_seconds"], 6)
        self.assertEqual(sleeps, [2] * 7)

    def test_fresh_busy_or_failed_query_blocks_exec_and_restarts_full_window(self):
        for final_result, expected_status in [(BUSY, "busy"),
                                             (RuntimeError("fresh query failed"), "check_error")]:
            with self.subTest(expected_status=expected_status):
                observations = [IDLE] * 4 + [final_result] + [IDLE] * 5
                records, sleeps, calls, _ = self.simulate(observations, FAST)
                blocked = records[3]
                self.assertEqual(blocked["elapsed_seconds"], 6)
                self.assertEqual(blocked["status"], expected_status)
                self.assertEqual(blocked["idle_seconds"], 0)
                self.assertEqual(blocked["final_idle_check"]["status"], expected_status)
                self.assertEqual(blocked["final_idle_check"]["gpu_uuid"], UUID)
                self.assertEqual(blocked["final_idle_check"]["elapsed_seconds"], 6)
                self.assertEqual(records[4]["final_idle_check"], blocked["final_idle_check"])
                self.assertEqual(records[-1]["status"], "launching")
                self.assertEqual(records[-1]["elapsed_seconds"], 14)
                self.assertEqual(records[-1]["idle_seconds"], 6)
                self.assertEqual(records[-1]["final_idle_check"]["status"], "idle")
                self.assertEqual(sleeps, [2] * 7)
                self.assertEqual(calls, 10)

    def test_real_cpu_exec_preserves_lease_and_argv_and_removes_only_policy_env(self):
        with tempfile.TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "status.json"
            output_path = Path(temporary) / "executed.json"
            with open(Path(temporary) / "lease", "w") as lease:
                target = (
                    "import json,os,sys; from pathlib import Path; "
                    "os.fstat(int(sys.argv[1])); "
                    "Path(sys.argv[2]).write_text(json.dumps(dict(argv=sys.argv[3:], "
                    "idle=os.environ.get('EASE_GPU_GATE_IDLE_SECONDS'), "
                    "poll=os.environ.get('EASE_GPU_GATE_POLL_SECONDS'), "
                    "scientific=os.environ.get('SCIENTIFIC_SETTING'))))"
                )
                literal_args = ["space preserved", "--", "$(literal)"]
                command = [sys.executable, "-c", target, str(lease.fileno()), str(output_path), *literal_args]
                script = (
                    "from unittest.mock import patch\n"
                    "from preparations.ease_insights_20261103 import await_assigned_gpu as gate\n"
                    f"with patch.object(gate, 'scan', return_value={IDLE!r}), "
                    "patch.object(gate.time, 'monotonic', side_effect=[0, 0, 0, 6, 6]), "
                    "patch.object(gate.time, 'sleep'):\n"
                    f"    gate.wait_and_exec({UUID!r}, {str(status_path)!r}, {command!r})\n"
                )
                environment = dict(os.environ, **FAST, SCIENTIFIC_SETTING="unchanged")
                result = subprocess.run([sys.executable, "-c", script], env=environment,
                                        pass_fds=(lease.fileno(),), capture_output=True,
                                        text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output_path.read_text()),
                             dict(argv=literal_args, idle=None, poll=None, scientific="unchanged"))
            status = json.loads(status_path.read_text())
            self.assertEqual(status["status"], "launching")
            self.assertEqual(status["final_idle_check"]["gpu_uuid"], UUID)
            self.assertEqual(status["final_idle_check"]["status"], "idle")


if __name__ == "__main__":
    unittest.main()

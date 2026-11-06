"""CPU-only launch-gate tests; fake clocks and GPU queries never wait or use CUDA."""
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
OTHER_UUID = "GPU-87654321-1234-1234-1234-123456789abc"
IDLE = dict(idle=True, memory_mib=0, compute_process_count=0)
BUSY = dict(idle=False, memory_mib=0, compute_process_count=1)


class Executed(Exception):
    pass


class GateTests(unittest.TestCase):
    def simulate(self, observations):
        now = [0.0]
        records = []
        output = io.StringIO()
        command = [sys.executable, "original.py", "--option", "value with spaces", "--"]

        def sleep(seconds):
            self.assertEqual(seconds, 2)
            now[0] += seconds

        with patch.object(gate.time, "monotonic", side_effect=lambda: now[0]), \
             patch.object(gate.time, "sleep", side_effect=sleep), \
             patch.object(gate, "scan", side_effect=observations) as scan, \
             patch.object(gate, "write_status", side_effect=lambda path, record: records.append(record)), \
             patch.object(gate.os, "execvp", side_effect=Executed) as execute, \
             contextlib.redirect_stdout(output):
            with self.assertRaises(Executed):
                gate.wait_and_exec(UUID, "unused.json", command)
        execute.assert_called_once_with(command[0], command)
        self.assertTrue(all(call.args == (UUID,) for call in scan.call_args_list))
        return records, [json.loads(line)["status"] for line in output.getvalue().splitlines()]

    def test_twenty_five_continuous_idle_seconds_and_quiet_polling(self):
        records, transitions = self.simulate([IDLE] * 15)
        self.assertEqual([r["elapsed_seconds"] for r in records], list(range(0, 27, 2)))
        self.assertEqual(records[-1]["status"], "launching")
        self.assertEqual(records[-1]["idle_seconds"], 26)
        self.assertEqual(transitions, ["waiting", "launching"])

    def test_busy_resets_the_full_idle_window(self):
        records, transitions = self.simulate([IDLE] * 3 + [BUSY] * 2 + [IDLE] * 15)
        self.assertEqual(records[-1]["elapsed_seconds"], 36)
        self.assertEqual(records[3]["idle_seconds"], 0)
        self.assertEqual(transitions, ["waiting", "busy", "waiting", "launching"])

    def test_query_errors_reset_and_continue_waiting(self):
        errors = [FileNotFoundError("nvidia-smi missing"),
                  subprocess.TimeoutExpired("nvidia-smi", 15), ValueError("bad query")]
        records, transitions = self.simulate([IDLE] * 3 + errors + [IDLE] * 15)
        self.assertEqual(records[-1]["elapsed_seconds"], 38)
        self.assertTrue(all(r["idle_seconds"] == 0 for r in records[3:6]))
        self.assertEqual(transitions, ["waiting", "check_error", "waiting", "launching"])

    def test_queries_are_restricted_to_exact_assigned_uuid(self):
        with patch.object(gate.subprocess, "check_output", side_effect=["", f"{UUID}, 499\n"]) as query:
            self.assertTrue(gate.scan(UUID)["idle"])
        self.assertEqual(query.call_count, 2)
        for call in query.call_args_list:
            self.assertIn("--id=" + UUID, call.args[0])
            self.assertEqual(call.kwargs["timeout"], 15)
        for apps, memory in [(f"{UUID}, 22\n", "0"), ("", "500")]:
            with patch.object(gate.subprocess, "check_output", side_effect=[apps, f"{UUID}, {memory}\n"]):
                self.assertFalse(gate.scan(UUID)["idle"])

    def test_multiple_devices_and_untrusted_query_output_are_rejected(self):
        for selector in ["0", UUID + "," + OTHER_UUID, "GPU-1234"]:
            with patch.object(gate.subprocess, "check_output") as query:
                with self.assertRaises(ValueError):
                    gate.scan(selector)
                query.assert_not_called()
        for outputs in [[f"{OTHER_UUID}, 2\n"], ["", f"{OTHER_UUID}, 0\n"],
                        ["", ""], ["", f"{UUID}, N/A\n"], ["", f"{UUID}, nan\n"],
                        ["", f"{UUID}, 0\n{OTHER_UUID}, 0\n"]]:
            with patch.object(gate.subprocess, "check_output", side_effect=outputs):
                with self.assertRaises(ValueError):
                    gate.scan(UUID)

    def test_cli_preserves_original_argv(self):
        command = [sys.executable, "task.py", "--gpu-uuid", "literal arg", "--"]
        with patch.object(gate, "wait_and_exec") as wait:
            gate.main(["--gpu-uuid", UUID, "--status-file", "status.json", "--", *command])
        wait.assert_called_once_with(UUID, "status.json", command)

    def test_atomic_status_and_inherited_lease_survive_real_cpu_exec(self):
        with tempfile.TemporaryDirectory() as temporary:
            status = Path(temporary) / "nested" / "status.json"
            with open(Path(temporary) / "lease", "w") as lease:
                target = "import os,sys; os.fstat(int(sys.argv[1]))"
                script = (
                    "from unittest.mock import patch\n"
                    "from preparations.ease_insights_20261103 import await_assigned_gpu as gate\n"
                    f"with patch.object(gate, 'scan', return_value={IDLE!r}), "
                    "patch.object(gate.time, 'monotonic', side_effect=[0, 0, 0, 26, 26]), "
                    "patch.object(gate.time, 'sleep'):\n"
                    f"    gate.wait_and_exec({UUID!r}, {str(status)!r}, "
                    f"{[sys.executable, '-c', target, str(lease.fileno())]!r})\n"
                )
                result = subprocess.run([sys.executable, "-c", script], pass_fds=(lease.fileno(),),
                                        capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(status.read_text())
            self.assertEqual(record["status"], "launching")
            self.assertEqual(record["gpu_uuid"], UUID)
            self.assertEqual(record["idle_seconds"], 26)
            self.assertEqual(list(status.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()

"""CPU-only keeper tests. Every CUDA call goes to FakeCUDA, never a driver."""
import contextlib
import ctypes
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import uuid

from preparations.ease_insights_20261103 import gpu_owned_keeper as keeper


UUID = "GPU-12345678-1234-1234-1234-123456789abc"
OTHER_UUID = "GPU-87654321-1234-1234-1234-123456789abc"


class FakeFunction:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args):
        return self.function(*args)


class FakeCUDA:
    """Imitate the official output pointer ABI, including 64-bit device pointers."""
    def __init__(self, gpu_uuid=UUID, capacity=24 * 1024 * keeper.MIB,
                 count=1, failures=None, log_path=None, forbidden_fd=None):
        self.gpu_uuid, self.capacity, self.count = gpu_uuid, capacity, count
        self.failures = failures or {}
        self.calls = []
        self.log_path, self.forbidden_fd = log_path, forbidden_fd
        for name in ("cuInit", "cuDriverGetVersion", "cuDeviceGetCount", "cuDeviceGet",
                     "cuDeviceGetUuid_v2", "cuDeviceTotalMem_v2", "cuCtxCreate_v2",
                     "cuCtxDestroy_v2", "cuMemAlloc_v2", "cuMemFree_v2"):
            setattr(self, name, FakeFunction(self._function(name)))

    def _function(self, name):
        def invoke(*args):
            self.calls.append(name)
            if self.log_path:
                with open(self.log_path, "a") as log:
                    log.write(name + "\n")
            if name in self.failures:
                return self.failures[name]
            if name == "cuInit" and self.forbidden_fd is not None:
                try:
                    os.fstat(self.forbidden_fd)
                except OSError:
                    pass
                else:
                    raise AssertionError("keeper retained an unrelated inherited FD")
            outputs = {
                "cuDriverGetVersion": (ctypes.c_int, 12040),
                "cuDeviceGetCount": (ctypes.c_int, self.count),
                "cuDeviceGet": (ctypes.c_int, 7),
                "cuDeviceTotalMem_v2": (ctypes.c_size_t, self.capacity),
                "cuCtxCreate_v2": (ctypes.c_void_p, 0xCAFE),
                "cuMemAlloc_v2": (keeper.DEVICE_POINTER, 0x123456789 if ctypes.sizeof(ctypes.c_void_p) == 8 else 0x123456),
            }
            if name in outputs:
                kind, value = outputs[name]
                ctypes.cast(args[0], ctypes.POINTER(kind)).contents.value = value
            elif name == "cuDeviceGetUuid_v2":
                ctypes.cast(args[0], ctypes.POINTER(keeper.CUuuid)).contents.bytes[:] = uuid.UUID(self.gpu_uuid[4:]).bytes
            return 0
        return invoke


class KeeperTests(unittest.TestCase):
    def setUp(self):
        self.no_real_driver = patch.object(keeper.ctypes, "CDLL", side_effect=AssertionError("Real CUDA is forbidden"))
        self.no_real_driver.start()
        self.addCleanup(self.no_real_driver.stop)

    @staticmethod
    def close_fd(fd):
        try:
            os.close(fd)
        except OSError:
            pass

    def config_with_pipes(self):
        control_read, control_write = os.pipe()
        ready_read, ready_write = os.pipe()
        for fd in (control_read, control_write, ready_read, ready_write):
            self.addCleanup(self.close_fd, fd)
        return keeper.Config(UUID, control_read, ready_write, "cpu-test-nonce",
                             os.getppid(), keeper.proc_start_ticks(os.getppid())), control_write, ready_read

    def error_run(self, config, fake):
        def factory(gpu_uuid, allocated_bytes):
            return keeper.DriverAllocation(gpu_uuid, allocated_bytes, loader=lambda name: fake)
        with patch.dict(os.environ), contextlib.redirect_stderr(io.StringIO()):
            result = keeper.run_keeper(config, allocation_factory=factory)
        return result

    def test_import_has_no_driver_or_torch_side_effects(self):
        code = (
            "import ctypes,sys; from unittest.mock import patch\n"
            "with patch.object(ctypes,'CDLL',side_effect=AssertionError('driver load')):\n"
            " import preparations.ease_insights_20261103.gpu_owned_keeper\n"
            " assert 'torch' not in sys.modules\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_driver_binds_exact_abi_and_creates_one_small_owned_allocation(self):
        fake = FakeCUDA()
        allocation = keeper.DriverAllocation(UUID, 64 * keeper.MIB, loader=lambda name: fake)
        allocation.acquire()
        self.assertEqual(allocation.driver_version, 12040)
        self.assertEqual(allocation.total_memory_bytes, fake.capacity)
        self.assertEqual(fake.calls.count("cuCtxCreate_v2"), 1)
        self.assertEqual(fake.calls.count("cuMemAlloc_v2"), 1)
        self.assertEqual(fake.cuMemAlloc_v2.argtypes, [ctypes.POINTER(keeper.DEVICE_POINTER), ctypes.c_size_t])
        self.assertEqual(fake.cuCtxCreate_v2.argtypes, [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int])
        self.assertEqual(fake.cuDeviceGetUuid_v2.argtypes, [ctypes.POINTER(keeper.CUuuid), ctypes.c_int])
        self.assertEqual(fake.cuMemFree_v2.argtypes, [keeper.DEVICE_POINTER])
        self.assertEqual(fake.cuMemAlloc_v2.restype, ctypes.c_int)
        self.assertEqual(allocation.close(), [])
        self.assertEqual(fake.calls[-2:], ["cuMemFree_v2", "cuCtxDestroy_v2"])
        self.assertEqual(allocation.close(), [])
        self.assertEqual(fake.calls.count("cuMemFree_v2"), 1)

    def test_all_invalid_config_is_rejected_before_loading_driver(self):
        invalid = dict(gpu_uuid=["0", "GPU-1234", UUID + "," + OTHER_UUID],
                       allocation_mib=[0, -1, 1025, "", "1.5", "nan", "inf", True],
                       parent_pid=[0, -1, "nan"], parent_start_ticks=[0, -1, "inf"],
                       nonce=["", "line\nbreak", "x" * 257], control_fd=[0, -1, "nan"])
        for field, values in invalid.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    config, _, ready_read = self.config_with_pipes()
                    fake = FakeCUDA()
                    self.assertEqual(self.error_run(replace(config, **{field: value}), fake), 1)
                    record = json.loads(os.read(ready_read, 8192))
                    self.assertEqual(record["phase"], "error")
                    self.assertEqual(fake.calls, [])

    def test_pipe_direction_nonpipe_and_parent_identity_are_validated(self):
        config, control_write, _ = self.config_with_pipes()
        for changed in (replace(config, control_fd=control_write),
                        replace(config, control_fd=config.ready_fd),
                        replace(config, parent_pid=os.getpid()),
                        replace(config, parent_start_ticks=config.parent_start_ticks + 1)):
            with self.subTest(config=changed), self.assertRaises((ValueError, RuntimeError)):
                keeper.validate_config(changed)
        with tempfile.TemporaryFile() as regular:
            with self.assertRaisesRegex(ValueError, "pipe"):
                keeper.validate_config(replace(config, control_fd=regular.fileno()))

    def test_uuid_device_count_and_capacity_checks_precede_context_creation(self):
        for fake in (FakeCUDA(gpu_uuid=OTHER_UUID), FakeCUDA(count=0), FakeCUDA(count=2),
                     FakeCUDA(capacity=64 * keeper.MIB * 20 - 1)):
            with self.subTest(fake=fake):
                config, _, ready_read = self.config_with_pipes()
                self.assertEqual(self.error_run(config, fake), 1)
                record = json.loads(os.read(ready_read, 8192))
                self.assertEqual(record["phase"], "error")
                self.assertNotIn("cuCtxCreate_v2", fake.calls)
        exact = FakeCUDA(capacity=64 * keeper.MIB * 20)
        allocation = keeper.DriverAllocation(UUID, 64 * keeper.MIB, loader=lambda name: exact)
        allocation.acquire()
        allocation.close()

    def test_cuda_failure_reports_error_after_cleanup(self):
        stages = ("cuInit", "cuDriverGetVersion", "cuDeviceGetCount", "cuDeviceGet",
                  "cuDeviceGetUuid_v2", "cuDeviceTotalMem_v2", "cuCtxCreate_v2", "cuMemAlloc_v2")
        for stage in stages:
            with self.subTest(stage=stage):
                config, _, ready_read = self.config_with_pipes()
                fake = FakeCUDA(failures={stage: 2})
                self.assertEqual(self.error_run(config, fake), 1)
                record = json.loads(os.read(ready_read, 8192))
                self.assertEqual(record["phase"], "error")
                self.assertIn(stage, record["error"])
                self.assertFalse(record["context_created"])
                self.assertEqual("cuCtxDestroy_v2" in fake.calls, stage == "cuMemAlloc_v2")

    def test_failed_free_still_destroys_only_own_context(self):
        fake = FakeCUDA(failures={"cuMemFree_v2": 999})
        allocation = keeper.DriverAllocation(UUID, keeper.MIB, loader=lambda name: fake)
        allocation.acquire()
        self.assertEqual(len(allocation.close()), 1)
        self.assertEqual(fake.calls[-2:], ["cuMemFree_v2", "cuCtxDestroy_v2"])
        self.assertIsNone(allocation.context)
        self.assertIsNone(allocation.pointer)

    def test_closed_or_nonempty_control_pipe_never_initializes_cuda(self):
        for data in (None, b"unexpected"):
            with self.subTest(data=data):
                config, control_write, ready_read = self.config_with_pipes()
                if data is None:
                    os.close(control_write)
                else:
                    os.write(control_write, data)
                fake = FakeCUDA()
                self.assertEqual(self.error_run(config, fake), 1)
                self.assertEqual(json.loads(os.read(ready_read, 8192))["phase"], "error")
                self.assertEqual(fake.calls, [])

    def test_ready_pipe_failure_cleans_up_all_owned_cuda(self):
        config, _, ready_read = self.config_with_pipes()
        os.close(ready_read)
        fake = FakeCUDA()
        self.assertEqual(self.error_run(config, fake), 1)
        self.assertEqual(fake.calls[-2:], ["cuMemFree_v2", "cuCtxDestroy_v2"])

    def test_parent_identity_loss_after_ready_releases_even_with_open_writer(self):
        config, _, ready_read = self.config_with_pipes()
        fake = FakeCUDA()
        with patch.object(keeper, "_check_parent", side_effect=[None, None, RuntimeError("parent identity changed")]):
            self.assertEqual(self.error_run(config, fake), 1)
        record = json.loads(os.read(ready_read, 8192))
        self.assertEqual(record["phase"], "ready")
        self.assertEqual(os.read(ready_read, 1), b"")
        self.assertEqual(fake.calls[-2:], ["cuMemFree_v2", "cuCtxDestroy_v2"])

    def test_cpu_helper_holds_until_eof_or_owned_signal_and_closes_unrelated_fd(self):
        for ending in ("eof", signal.SIGTERM, signal.SIGINT):
            with self.subTest(ending=ending), tempfile.TemporaryDirectory() as temporary:
                control_read, control_write = os.pipe()
                ready_read, ready_write = os.pipe()
                unrelated = os.open(Path(temporary) / "unrelated", os.O_CREAT | os.O_RDWR, 0o600)
                log_path = Path(temporary) / "mock_cuda_calls"
                script = (
                    "import sys; from unittest.mock import patch\n"
                    "from preparations.ease_insights_20261103 import gpu_owned_keeper as k\n"
                    "from preparations.ease_insights_20261103.test_gpu_owned_keeper import FakeCUDA\n"
                    f"fake=FakeCUDA(log_path={str(log_path)!r},forbidden_fd={unrelated})\n"
                    "with patch.object(k.ctypes,'CDLL',return_value=fake):\n"
                    " raise SystemExit(k.main(sys.argv[1:]))\n"
                )
                argv = [sys.executable, "-c", script, "--gpu-uuid", UUID,
                        "--control-fd", str(control_read), "--ready-fd", str(ready_write),
                        "--nonce", "subprocess-proof", "--parent-pid", str(os.getpid()),
                        "--parent-start-ticks", str(keeper.proc_start_ticks(os.getpid()))]
                process = subprocess.Popen(argv, pass_fds=(control_read, ready_write, unrelated),
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                os.close(control_read); os.close(ready_write); os.close(unrelated)
                try:
                    readable, _, _ = select.select([ready_read], [], [], 5)
                    self.assertTrue(readable, "mock helper failed to report ready")
                    record = json.loads(os.read(ready_read, 8192))
                    self.assertEqual(record["phase"], "ready", record)
                    self.assertEqual(record["schema"], keeper.SCHEMA)
                    self.assertEqual(record["pid"], process.pid)
                    self.assertEqual(record["proc_start_ticks"], keeper.proc_start_ticks(process.pid))
                    self.assertEqual(record["nonce"], "subprocess-proof")
                    self.assertEqual(record["gpu_uuid"], UUID)
                    self.assertEqual(record["allocated_bytes"], 64 * keeper.MIB)
                    self.assertGreaterEqual(record["total_memory_bytes"], 20 * record["allocated_bytes"])
                    self.assertTrue(record["context_created"])
                    self.assertEqual(record["driver_version"], 12040)
                    self.assertIn("nspid", record)
                    self.assertIn("pid_namespace_inode", record)
                    self.assertEqual(os.read(ready_read, 1), b"", "ready must be one record then EOF")
                    self.assertIsNone(process.poll(), "empty queue/ready EOF must not release control lease")
                    self.assertNotIn("cuMemFree_v2", log_path.read_text())
                    if ending == "eof":
                        os.close(control_write); control_write = None
                    else:
                        process.send_signal(ending)  # Exact owned CPU dummy child only.
                    stdout, stderr = process.communicate(timeout=5)
                    self.assertEqual(process.returncode, 0, stderr)
                    self.assertEqual(stdout, "")
                    self.assertEqual(log_path.read_text().splitlines()[-2:], ["cuMemFree_v2", "cuCtxDestroy_v2"])
                    self.assertFalse(Path(f"/proc/{process.pid}").exists(), "owned child must be reaped")
                finally:
                    if control_write is not None:
                        os.close(control_write)
                    os.close(ready_read)
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()

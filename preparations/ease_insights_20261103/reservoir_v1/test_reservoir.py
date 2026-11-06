"""CPU contracts; CUDA calls use an independent allocation accounting fake."""
import ctypes
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from preparations.ease_insights_20261103.reservoir_v1 import keeper as k, guardian as g
from preparations.ease_insights_20261103.test_gpu_owned_keeper import FakeCUDA, FakeFunction, UUID


class MemoryCUDA(FakeCUDA):
    def __init__(self, **kwargs):
        self.live = {}
        self.next_pointer = 0x100000000
        self.fail_above = None
        super().__init__(gpu_uuid=UUID, **kwargs)

    def _function(self, name):
        original = super()._function(name)
        if name == "cuMemAlloc_v2":
            def allocate(output, size):
                self.calls.append(name)
                if self.fail_above is not None and sum(self.live.values()) + size > self.fail_above:
                    return 2
                self.next_pointer += 0x100000
                self.live[self.next_pointer] = size
                ctypes.cast(output, ctypes.POINTER(k.base.DEVICE_POINTER)).contents.value = self.next_pointer
                return 0
            return allocate
        if name == "cuMemFree_v2":
            def free(pointer):
                self.calls.append(name)
                del self.live[pointer.value]
                return 0
            return free
        return original


class DriverContracts(unittest.TestCase):
    def setUp(self):
        self.patcher = patch.object(ctypes, "CDLL", side_effect=AssertionError("Real CUDA forbidden"))
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_import_does_not_load_driver_torch_or_start_processes(self):
        code = (
            "import ctypes,subprocess,sys\nfrom unittest.mock import patch\n"
            "with patch.object(ctypes,'CDLL',side_effect=AssertionError('driver')), "
            "patch.object(subprocess,'Popen',side_effect=AssertionError('child')):\n"
            " import preparations.ease_insights_20261103.reservoir_v1.guardian\n"
            " assert 'torch' not in sys.modules\n"
        )
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_resize_keeps_context_and_eight_GiB_floor_then_restores_sixteen(self):
        fake = MemoryCUDA()
        allocation = k.Reservoir(UUID, loader=lambda name: fake)
        allocation.acquire()
        context, anchor = allocation.context.value, allocation.pointer.value
        self.assertEqual(sum(fake.live.values()), 16 * 1024**3)
        for target in (8192, 16384, 8192, 16384):
            allocation.resize(target)
            self.assertEqual(sum(fake.live.values()), target * 1024**2)
            self.assertEqual(allocation.context.value, context)
            self.assertEqual(allocation.pointer.value, anchor)
            self.assertIn(anchor, fake.live)
        self.assertEqual(fake.calls.count("cuCtxCreate_v2"), 1)
        self.assertEqual(allocation.close(), [])
        self.assertEqual(fake.live, {})
        self.assertEqual(fake.calls.count("cuCtxDestroy_v2"), 1)

    def test_failed_growth_keeps_all_preexisting_allocations_and_can_be_retried(self):
        fake = MemoryCUDA()
        allocation = k.Reservoir(UUID, loader=lambda name: fake)
        allocation.acquire()
        allocation.resize(8192)
        before = dict(fake.live)
        fake.fail_above = 10 * 1024**3
        with self.assertRaisesRegex(RuntimeError, "CUDA error 2"):
            allocation.resize(16384)
        self.assertEqual(sum(fake.live.values()), 10 * 1024**3)
        self.assertTrue(set(before).issubset(fake.live))
        self.assertEqual(allocation.allocated_bytes, sum(fake.live.values()))
        fake.fail_above = None
        allocation.resize(16384)
        self.assertEqual(sum(fake.live.values()), 16 * 1024**3)
        allocation.close()
        self.assertEqual(fake.live, {})

    def test_invalid_targets_never_change_memory(self):
        fake = MemoryCUDA()
        allocation = k.Reservoir(UUID, loader=lambda name: fake)
        allocation.acquire()
        before = dict(fake.live)
        for value in (0, 8193, 32768, True, "16384", 8192.0):
            with self.assertRaises(ValueError):
                allocation.resize(value)
            self.assertEqual(before, fake.live)
        allocation.close()

    def test_rejects_capacity_below_reservation_bound_and_can_cleanup_anchor(self):
        fake = MemoryCUDA(capacity=20 * 1024**3)
        allocation = k.Reservoir(UUID, loader=lambda name: fake)
        with self.assertRaisesRegex(ValueError, "75%"):
            allocation.acquire()
        self.assertEqual(sum(fake.live.values()), 64 * 1024**2)
        allocation.close()
        self.assertEqual(fake.live, {})

    def test_commands_require_correct_nonce_sequence_and_bounded_targets(self):
        value = dict(schema=k.SCHEMA, op="resize", nonce="same-parent", sequence=1, target_mib=8192)
        self.assertEqual(k.command(value, "same-parent", 0), (1, 8192))
        for changes in ({"nonce": "other"}, {"sequence": 0}, {"sequence": True},
                        {"target_mib": 24576}, {"op": "release"}):
            with self.assertRaises(ValueError):
                k.command(dict(value, **changes), "same-parent", 0)

    def test_real_parent_child_pipes_keep_same_child_across_resize_and_EOF(self):
        # Execute the real keeper loop in a separate CPU process, substituting
        # only its CUDA ABI with MemoryCUDA. This exercises nonce/sequence,
        # inherited directional pipes, ready receipts and parent identity.
        real_popen = subprocess.Popen
        code = (
            "import argparse\n"
            "from preparations.ease_insights_20261103.reservoir_v1 import keeper as k\n"
            "from preparations.ease_insights_20261103.reservoir_v1.test_reservoir import MemoryCUDA\n"
            "p=argparse.ArgumentParser()\n"
            "for name in ('gpu-uuid','control-fd','ready-fd','nonce','parent-pid','parent-start-ticks'):\n"
            " p.add_argument('--'+name,required=True)\n"
            "config=k.base.Config(**vars(p.parse_args()))\n"
            "raise SystemExit(k.run(config, allocation_factory=lambda u: "
            "k.Reservoir(u,loader=lambda name:MemoryCUDA())))\n"
        )
        def fake_driver_child(argv, **kwargs):
            return real_popen([sys.executable, "-B", "-u", "-c", code, *argv[5:]], **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            client = g.Client(UUID, Path(tmp), g.bc.legacy.Runtime())
            with patch.object(g.subprocess, "Popen", side_effect=fake_driver_child):
                try:
                    ready = client.start()
                    pid = client.child.pid
                    self.assertEqual(ready["allocated_bytes"], 16 * 1024**3)
                    self.assertEqual(client.resize(8192)["allocated_bytes"], 8 * 1024**3)
                    self.assertEqual(client.resize(16384)["allocated_bytes"], 16 * 1024**3)
                    self.assertEqual(client.child.pid, pid)
                    self.assertTrue(client.live())
                finally:
                    client.close()
            self.assertEqual(client.child.returncode, 0)


class LifecycleContracts(unittest.TestCase):
    GPU = next(iter(g.HOST_KEEPERS))

    def row(self):
        return dict(host_compute_pids=[g.HOST_KEEPERS[self.GPU]], compute_process_count=1, memory_mib=460)

    def test_gate_requires_distinct_fresh_samples_over_at_least_fifteen_seconds(self):
        window = g.IdleWindow()
        for query in range(100, 116, 2):
            ready, elapsed = window.update(query, self.row(), self.GPU)
            self.assertFalse(ready)
        self.assertEqual(window.update(114, self.row(), self.GPU), (False, 14))
        self.assertEqual(window.update(116, self.row(), self.GPU), (True, 16))

    def test_busy_gap_and_time_reversal_reset_availability_history(self):
        for bad_query, bad_row in ((120, self.row()), (99, self.row()),
                (104, dict(self.row(), host_compute_pids=[1234], memory_mib=16000))):
            window = g.IdleWindow()
            window.update(100, self.row(), self.GPU)
            window.update(102, self.row(), self.GPU)
            ready, elapsed = window.update(bad_query, bad_row, self.GPU)
            self.assertFalse(ready)
            self.assertEqual(elapsed, 0)

    def test_stale_or_different_observer_cannot_supply_idle_proof(self):
        status = dict(pid=385842, proc_start_ticks="2618445988",
            run_id="060b0172-ba52-4c9b-9c39-03c7fb604d69", source_sha256=g.WATCH_SHA,
            poll_seconds=2, error=None, query_finished_at=100,
            devices=[dict(self.row(), gpu_uuid=self.GPU)])
        self.assertEqual(g.cached_device(status, self.GPU, 102)[0], 100)
        for changed, now in ((status, 104), (dict(status, source_sha256="wrong"), 101),
                             (dict(status, devices=[]), 101)):
            with self.assertRaises(RuntimeError):
                g.cached_device(changed, self.GPU, now)

    def test_preflight_requires_original_context_plus_exact_sized_new_context(self):
        original, bulk = g.HOST_KEEPERS[self.GPU], 777
        good = [(original, 450), (bulk, 8580)]
        self.assertEqual(g.owned_observation(self.GPU, good, bulk, 8192)["reservoir_host_pid"], bulk)
        for bad in (good + [(888, 50)], [(original, 450)], [(original, 600), (bulk, 8580)],
                    [(original, 450), (bulk, 7000)], [(original, 450), (bulk, float("nan"))]):
            with self.assertRaises(RuntimeError):
                g.owned_observation(self.GPU, bad, bulk, 8192)

    def test_success_and_failure_restore_idle_memory_after_child_cleanup(self):
        for fail in (False, True):
            events = []
            class Client:
                def resize(self, mib):
                    events.append(mib)
            class Runtime:
                def reap_scientific_children(self):
                    events.append("children_reaped")
            class Session:
                runtime = Runtime()
            class Science:
                def execute(self, prepared, session):
                    events.append("science")
                    if fail:
                        raise RuntimeError("preserve_scientific_failure")
                    return "done"
            if fail:
                with self.assertRaisesRegex(RuntimeError, "preserve_scientific_failure"):
                    g.execute_condition({}, Session(), Science(), Client())
            else:
                self.assertEqual(g.execute_condition({}, Session(), Science(), Client()), "done")
            self.assertEqual(events, [8192, "science", "children_reaped", 16384])


if __name__ == "__main__":
    unittest.main()

"""CPU proofs for stream ordering, lease ownership and worker failure cleanup."""
from contextlib import nullcontext
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from .policy import PipelinePolicy
from .scheduler import Pool
from .submission import Submission
from .scheduling_trace import duration, intersection


class Event:
    def __init__(self, done=False):
        self.complete = done

    def record(self, stream=None):
        self.complete = True

    def query(self):
        return self.complete

    def synchronize(self):
        self.complete = True

    def elapsed_time(self, other):
        if not other.query():
            raise AssertionError("Unfinished event must not be timed")
        return 1.0


class Stream:
    def __init__(self, **unused):
        self.cuda_stream = id(self)
        self.waits, self.drains = [], 0

    def wait_event(self, event):
        self.waits.append(event)

    def synchronize(self):
        self.drains += 1


class OwnerPool(Pool):
    def __init__(self):
        super().__init__()
        self.owner = threading.get_ident()

    def _check(self):
        assert self.owner == threading.get_ident(), "Worker touched the pool"

    def reserve(self):
        self._check()
        return super().reserve()

    def submitted(self, *args):
        self._check()
        return super().submitted(*args)

    def release(self, *args):
        self._check()
        return super().release(*args)

    def poll(self):
        self._check()
        return super().poll()


class Backend:
    device = "cuda:0"
    identity = {}
    model = SimpleNamespace(config=SimpleNamespace(action_horizon=50))

    def __init__(self):
        self.calls, self.selected = [], []
        self.barrier = None
        self.fail = False

    def prepare(self, value):
        return value

    def noise(self, seed):
        return seed

    def build_prefix(self, prepared):
        self.calls.append("prefill")
        if self.barrier:
            self.barrier.wait(timeout=3)
        return dict(source=prepared)

    def generate(self, prefix, prepared, noise):
        self.calls.append("ae")
        self.selected.append(prefix["source"])
        if self.barrier:
            self.barrier.wait(timeout=3)
        if self.fail:
            raise RuntimeError("worker failed after dispatch")
        return np.zeros((50, 7))

    def output(self, prepared, raw):
        return raw


class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.foreground = Stream()
        self.patches = [
            patch("preparations.idea2.policy.event", Event),
            patch("preparations.idea2.submission.event", Event),
            patch("preparations.idea2.policy.native_raw", lambda r:r["obs"]),
            patch("preparations.idea2.policy.torch.cuda.Stream", Stream),
            patch("preparations.idea2.policy.torch.cuda.current_stream", lambda *a:self.foreground),
            patch("preparations.idea2.policy.torch.cuda.stream", lambda stream:nullcontext()),
            patch("preparations.idea2.policy.torch.cuda.max_memory_allocated", lambda *a:0),
            patch("preparations.idea2.policy.torch.cuda.max_memory_reserved", lambda *a:0),
            patch("preparations.idea2.submission.torch.cuda.nvtx.range_push", lambda *a:None),
            patch("preparations.idea2.submission.torch.cuda.nvtx.range_pop", lambda:None),
            patch("preparations.idea1_v2_1.runtime_client.input_digest", lambda *a:"hash"),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def make(self, order, workers=False):
        backend = Backend()
        policy = PipelinePolicy(backend, submission_order=order, diagnostics=True, host_workers=workers)
        policy.pool = OwnerPool()
        self.addCleanup(policy.close)
        policy.predict(0, "task", "suite", 1)
        backend.calls.clear()
        return backend, policy

    def test_both_single_thread_orders_pin_old_kv(self):
        for order, expected in (("ae_first", ["ae", "prefill"]), ("prefill_first", ["prefill", "ae"])):
            backend, policy = self.make(order)
            row = policy.predict(1, "task", "suite", 2)
            self.assertEqual(backend.calls, expected)
            self.assertEqual(backend.selected, [0, 0])
            self.assertEqual(row["prefix_observation_id"], 0)
            lease = policy.pool.acquire()
            self.assertEqual(lease.observation.observation_id, 1)
            policy.pool.release(lease, Event(True))
            self.assertEqual(row["pool"]["live_snapshots"], 1)

    def test_workers_concurrently_submit_without_mutating_pool(self):
        for order in ("ae_first", "prefill_first"):
            backend, policy = self.make(order, True)
            waits_before = len(policy.ae_stream.waits)
            backend.barrier = threading.Barrier(2)
            row = policy.predict(1, "task", "suite", 2)
            self.assertEqual(row["prefix_observation_id"], 0)
            self.assertEqual(backend.selected, [0, 0])
            self.assertEqual(policy.pool.live_snapshots, 1)
            self.assertEqual(len(policy.ae_stream.waits), waits_before + 1)
            self.assertTrue(row["diagnostics"]["ae"]["done_now"])
            policy.reset()
            self.assertEqual(policy.pool.live_snapshots, 0)

    def test_worker_error_drains_sibling_and_releases_lease(self):
        backend, policy = self.make("ae_first", True)
        backend.barrier = threading.Barrier(2)
        backend.fail = True
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            policy.predict(1, "task", "suite", 2)
        self.assertGreater(policy.ae_stream.drains, 0)
        self.assertGreater(policy.background.drains, 0)
        self.assertEqual(policy.pool.live_snapshots, 0)
        self.assertTrue(all(s.leases == 0 for s in policy.pool.slots))
        backend.fail, backend.barrier = False, None
        self.assertEqual(policy.predict(2, "task", "suite", 3)["request"], 0)

    def test_single_submit_error_releases_old_lease_and_reservation(self):
        for order in ("ae_first", "prefill_first"):
            backend, policy = self.make(order)
            backend.fail = True
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                policy.predict(1, "task", "suite", 2)
            self.assertEqual(policy.pool.live_snapshots, 0)
            self.assertTrue(all(s.leases == 0 for s in policy.pool.slots))

    def test_incomplete_event_is_queried_without_synchronizing(self):
        job = Submission(Event(True), Event(), 1, 2, False, 7)
        self.assertIsNone(job.metrics()["gpu_event_span_ms"])
        self.assertFalse(job.done.query())

    def test_output_failure_drains_pending_prefix_and_releases_lease(self):
        for order, workers in (("ae_first", False), ("prefill_first", False), ("ae_first", True)):
            backend, policy = self.make(order, workers)
            observed = []
            original_output = backend.output
            background_drains = policy.background.drains
            def fail_output(prepared, raw):
                pending = [s for s in policy.pool.slots if s.state == "BUILDING"]
                self.assertEqual(len(pending), 1)
                ready = pending[0].ready
                ready.complete = False
                def finish_prefix():
                    observed.append("prefill drained")
                    self.assertGreater(policy.background.drains, background_drains)
                    self.assertTrue(all(s.leases == 0 for s in policy.pool.slots))
                    ready.complete = True
                ready.synchronize = finish_prefix
                raise ValueError("CPU output transform failed")
            backend.output = fail_output
            with self.assertRaisesRegex(ValueError, "CPU output transform failed"):
                policy.predict(1, "task", "suite", 2)
            self.assertEqual(observed, ["prefill drained"])
            self.assertEqual(policy.pool.live_snapshots, 0)
            self.assertTrue(all(s.leases == 0 for s in policy.pool.slots))
            backend.output = original_output
            policy.reset()
            result = policy.predict(2, "task", "suite", 3)
            self.assertEqual(result["request"], 0)
            self.assertEqual(result["prefix_observation_id"], 0)

    def test_completion_wait_failure_releases_lease(self):
        from .submission import submit as original_submit
        backend, policy = self.make("ae_first")
        def submit_with_failed_wait(stream, label, function, *args, **kwargs):
            value, job = original_submit(stream, label, function, *args, **kwargs)
            if label == "ae":
                def fail_wait():
                    raise RuntimeError("injected recoverable completion wait failure")
                job.done.synchronize = fail_wait
            return value, job
        with patch("preparations.idea2.policy.submit", submit_with_failed_wait):
            with self.assertRaisesRegex(RuntimeError, "completion wait failure"):
                policy.predict(1, "task", "suite", 2)
        self.assertEqual(policy.pool.live_snapshots, 0)
        self.assertTrue(all(s.leases == 0 for s in policy.pool.slots))
        policy.reset()
        self.assertEqual(policy.predict(2, "task", "suite", 3)["request"], 0)

    def test_bootstrap_failure_can_recover_through_reset(self):
        backend, policy = self.make("ae_first")
        policy.reset()
        original_build = backend.build_prefix
        def fail_build(prepared):
            raise ValueError("bootstrap prefix failed")
        backend.build_prefix = fail_build
        with self.assertRaisesRegex(ValueError, "bootstrap prefix failed"):
            policy.predict(1, "task", "suite", 2)
        self.assertTrue(all(s.leases == 0 for s in policy.pool.slots))
        self.assertTrue(any(s.state == "BUILDING" and s.ready is None for s in policy.pool.slots))
        policy.reset()
        backend.build_prefix = original_build
        self.assertEqual(policy.predict(2, "task", "suite", 3)["request"], 0)

    def test_graph_layout_reset_restores_task_before_bootstrap(self):
        backend, policy = self.make("ae_first")
        old = SimpleNamespace(compatible=lambda *a:False, close=lambda:None)
        policy.cuda_graphs, policy.graphs = True, old
        created = []
        class Graphs:
            setup_ms = 1.0
            def __init__(self,*args):
                created.append(self)
            def build_prefix(self,pool,identity,prepared):
                assert pool.task=="task"
                return backend.build_prefix(prepared)
            def generate(self,pool,lease,prepared,noise):
                return backend.generate(lease.value,prepared,noise)
            def close(self):
                pass
        with patch("preparations.idea2.runtime_graph.RuntimeGraphs",Graphs):
            response = policy.predict(1,"task","suite",2)
        self.assertEqual(len(created),1)
        self.assertEqual(response["request"],0)
        self.assertEqual(response["prefix_observation_id"],0)
        self.assertEqual(policy.pool.task,"task")
        self.assertEqual(response["graph_setup_ms"],1.0)

    def test_kernel_interval_math_excludes_envelope_gaps(self):
        self.assertEqual(duration([(0, 2), (8, 10)]), 4)
        self.assertEqual(intersection([(0, 2), (8, 10)], [(3, 7)]), 0)
        self.assertEqual(intersection([(0, 2), (8, 10)], [(1, 9)]), 2)


if __name__ == "__main__":
    unittest.main()

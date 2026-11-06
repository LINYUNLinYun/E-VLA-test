"""CPU-only checks for the results-level early-return scheduling control."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parents[2]))
control = importlib.import_module("results.idea2.d10_dynamic_sm_20261024.async_serial_control")


class EventOrderTests(unittest.TestCase):
    def run_prefix(self, gate, pending):
        calls = []
        graph = control.ControlGraphs.__new__(control.ControlGraphs)
        graph.backend = SimpleNamespace(prefix_dependency_gate=gate)
        graph.kind, graph.ae_pending, graph.vision_pending = "vision_only", pending, False
        graph.ae_done = object()
        graph.prefix_stream = SimpleNamespace(wait_event=lambda event: calls.append(("wait", event)))
        def original(self, pool, identity, prepared):
            calls.extend(["stage", "vision"])
            self.vision_pending = True
            if self.kind == "vision_only" and self.ae_pending:
                self.prefix_stream.wait_event(self.ae_done)
                self.ae_pending = False
            calls.append("vlm")
            return "existing_prefix"
        with patch.object(control.BASE_GRAPHS, "build_prefix", original):
            self.assertEqual(graph.build_prefix(None, None, None), "existing_prefix")
        return graph, calls

    def test_control_moves_the_same_single_wait_before_all_prefix_work(self):
        graph, calls = self.run_prefix("ae_before_vision", True)
        self.assertEqual(calls, [("wait", graph.ae_done), "stage", "vision", "vlm"])
        self.assertFalse(graph.ae_pending)
        self.assertEqual(graph.last_prefix_dependency_gate, "ae_before_vision")

    def test_o1_retains_vision_overlap_and_the_original_single_wait(self):
        graph, calls = self.run_prefix("ae_before_vlm", True)
        self.assertEqual(calls, ["stage", "vision", ("wait", graph.ae_done), "vlm"])
        self.assertFalse(graph.ae_pending)

    def test_bootstrap_does_not_wait_for_an_unrecorded_ae_event(self):
        for gate in control.GATES.values():
            graph, calls = self.run_prefix(gate, False)
            self.assertEqual(calls, ["stage", "vision", "vlm"])
            self.assertEqual(graph.last_prefix_dependency_gate, "bootstrap_no_ae")


class LifecycleTests(unittest.TestCase):
    def fixture(self):
        calls = []
        policy = control.ControlPolicy.__new__(control.ControlPolicy)
        pool = control.bench.policy_module.Pool()
        class Backend:
            identity = {}
            model = SimpleNamespace(config=SimpleNamespace(action_horizon=10))
            selective_overlap = "vision_only"
            graph_engine_class = Mock(side_effect=AssertionError("Unexpected capture"))
            @property
            def prefix_dependency_gate(self):
                return getattr(self, "_gate", "ae_before_vlm")
            @prefix_dependency_gate.setter
            def prefix_dependency_gate(self, value):
                if any(slot.ready is not None or slot.readers or slot.state != "FREE" for slot in pool.slots):
                    raise AssertionError("Gate switched before producer/reader drain")
                calls.append(("gate", value))
                self._gate = value
        policy.backend, policy.pool = Backend(), pool
        def reset_events():
            calls.append("reset_events")
            policy.graphs.ae_pending = policy.graphs.vision_pending = False
        policy.graphs = SimpleNamespace(kind="vision_only", slots=[object(), object()],
            ae_pending=True, vision_pending=True, reset_events=reset_events)
        policy.request, policy.submission_order = 1, "ae_first"
        policy.cuda_graphs, policy.ae_priority, policy.ae_stream = True, 0, object()
        policy.workers, policy.diagnostics, policy._prefix_timings = None, False, []
        policy.prefix_dependency_gate = "ae_before_vlm"
        pool.slots[0].state = "ACTIVE"
        pool.slots[0].ready = SimpleNamespace(synchronize=lambda: calls.append("drain_producer"))
        pool.slots[0].readers = [SimpleNamespace(synchronize=lambda: calls.append("drain_reader"))]
        return policy, calls

    def test_real_reset_drains_before_switch_without_replacing_owner_or_slots(self):
        policy, calls = self.fixture()
        owner, slots, stream = policy.graphs, tuple(policy.graphs.slots), policy.ae_stream
        kwargs = dict(mode="async_reuse", interval=1, submission_order="ae_first", cuda_graphs=True, ae_priority=0)
        policy.configure(**kwargs, prefix_dependency_gate="ae_before_vision")
        self.assertEqual(calls, ["drain_producer", "drain_reader", "reset_events", ("gate", "ae_before_vision")])
        policy.configure(**kwargs, prefix_dependency_gate="ae_before_vlm")
        self.assertEqual(calls[-2:], ["reset_events", ("gate", "ae_before_vlm")])
        self.assertIs(policy.graphs, owner)
        self.assertEqual(tuple(policy.graphs.slots), slots)
        self.assertIs(policy.ae_stream, stream)
        self.assertEqual(policy.graphs.kind, "vision_only")
        policy.backend.graph_engine_class.assert_not_called()

    def test_unreleased_lease_prevents_switch(self):
        policy, calls = self.fixture()
        policy.pool.slots[0].leases = 1
        with self.assertRaisesRegex(RuntimeError, "Finish and release"):
            policy.configure(mode="async_reuse", interval=1, prefix_dependency_gate="ae_before_vision")
        self.assertEqual(calls, [])
        self.assertEqual(policy.prefix_dependency_gate, "ae_before_vlm")

    def test_initial_gate_is_ready_before_first_lazy_capture(self):
        backend = SimpleNamespace(device="cpu", identity={}, num_steps=10, await_previous_prefix=True,
            selective_overlap="vision_only", model=SimpleNamespace(config=SimpleNamespace(action_horizon=10)),
            graph_core_class=type("PreparedModulationCore", (), {}), graph_engine_class=Mock())
        with patch.object(control.bench.policy_module.torch.cuda, "Stream", side_effect=[object(), object()]):
            policy = control.ControlPolicy(backend, cuda_graphs=True, ae_priority=0, submission_order="ae_first",
                                           prefix_dependency_gate="ae_before_vision")
        self.assertIsNone(policy.graphs)
        self.assertEqual(backend.prefix_dependency_gate, "ae_before_vision")
        backend.graph_engine_class.assert_not_called()


class EvidenceTests(unittest.TestCase):
    def test_recorder_keeps_actual_gate_and_rejects_new_engines(self):
        policy = SimpleNamespace(prefix_dependency_gate="ae_before_vision",
                                 graphs=SimpleNamespace(last_prefix_dependency_gate="ae_before_vision"))
        def record(self, *args, **kwargs):
            self.rows.append(dict(steady_recapture=0, selective_kind="vision_only"))
            return "response"
        recorder = control.ControlRecorder(None)
        recorder.capture_engines = 1
        with patch.object(control.BASE_RECORDER, "predict", record):
            self.assertEqual(recorder.predict(policy), "response")
            self.assertEqual(recorder.rows[-1]["prefix_dependency_gate"], "ae_before_vision")
            recorder.capture_engines = 2
            with self.assertRaisesRegex(RuntimeError, "invariant failed"):
                recorder.predict(policy)

    def test_wrapper_source_snapshot_and_actual_entrypoint_are_saved(self):
        with tempfile.TemporaryDirectory() as temporary:
            dest = Path(temporary)
            (dest / "source").mkdir()
            result = dict(rows=[], groups=[])
            control.write_results(dest, result)
            saved = json.loads((dest / "result.json").read_text())
            snapshot = Path(saved["wrapper"]["source_snapshot"])
            self.assertEqual(hashlib.sha256(snapshot.read_bytes()).hexdigest(), saved["wrapper"]["sha256"])
            self.assertTrue(saved["wrapper"]["source_unchanged"])
            self.assertIn(control.MODULE, saved["command"])

    def test_import_never_initializes_cuda(self):
        self.assertFalse(control.bench.torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()

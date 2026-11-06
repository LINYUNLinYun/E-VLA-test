"""CPU-only batch runtime contracts; no device probe or graph capture."""
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .batch_backend import FixedLaneBatch
from .batch_runtime import BatchProgressiveRuntime, LabeledGraph, _batch_comparison
from .openpi_backend import Prepared
from .progressive_kv import ProgressiveRuntime
from .runtime_graph import input_tensors, layout


class FakeBackend:
    def __init__(self, batch=2, decode_scale=1):
        self.batch_size = batch
        self.contract = FixedLaneBatch(tuple(f"robot{i}" for i in range(batch)))
        self.model = SimpleNamespace(config=SimpleNamespace(action_horizon=2, action_dim=3))
        self.decode_scale = decode_scale
        self.device = torch.device("cpu")

    def prepare_batch(self, raws, *, lane_ids=None):
        self.contract.validate(lane_ids)
        if len(raws) != self.batch_size:
            raise ValueError("wrong lane count")
        value = Prepared(None, [torch.full((self.batch_size, 2), float(raws[0]))],
            [torch.ones(self.batch_size, dtype=torch.bool)], torch.zeros(self.batch_size, 4, dtype=torch.long),
            torch.ones(self.batch_size, 4, dtype=torch.bool), torch.zeros(self.batch_size, 3))
        value.batch_identity = self.contract.identity
        return value

    def validate_prepared(self, prepared):
        self.contract.validate_identity(prepared.batch_identity)
        if any(value.shape[0] != self.batch_size for value in input_tensors(prepared)):
            raise ValueError("batch dimension changed")

    def validate_noise(self, noise):
        if tuple(noise.shape) != (self.batch_size, 2, 3):
            raise ValueError("wrong noise shape")

    def output_batch(self, prepared, actions):
        self.validate_prepared(prepared)
        return np.asarray(actions) * self.decode_scale


def cpu_runtime(batch=2, mode="pipeline"):
    backend = FakeBackend(batch)
    prepared = backend.prepare_batch([1] * batch)
    noise = torch.zeros(batch, 2, 3)
    engine = BatchProgressiveRuntime.__new__(BatchProgressiveRuntime)
    engine.backend = backend
    engine.batch_backend = backend
    engine.mode = mode
    engine.batch_size = batch
    engine.lane_ids = backend.contract.lane_ids
    engine.signature = layout(prepared, noise)
    engine._noise_template = noise
    engine._runtime_batch_identity = backend.contract.identity
    engine.graphs = SimpleNamespace(num_layers=18)
    engine.resource_layout = dict(policy="full_gpu")
    engine.pool = SimpleNamespace(slots=[SimpleNamespace(generation=4,
        observation=SimpleNamespace(observation_id=1)), SimpleNamespace(generation=3,
        observation=SimpleNamespace(observation_id=0))])
    engine.green_pair = None
    engine._closed = False
    return engine, prepared, noise


class BatchRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.assertFalse(torch.cuda.is_initialized())

    def tearDown(self):
        self.assertFalse(torch.cuda.is_initialized())

    def test_batch_download_preserves_second_lane_and_b1_dimension(self):
        engine, _, _ = cpu_runtime()
        tensor = torch.arange(12).reshape(2, 2, 3)
        output = engine.download_actions(tensor)
        self.assertEqual(output.shape, (2, 2, 3))
        np.testing.assert_array_equal(output[1], tensor[1].numpy())
        tensor.zero_()
        self.assertEqual(output[1, 0, 0], 6)
        engine, _, _ = cpu_runtime(1)
        self.assertEqual(engine.download_actions(torch.zeros(1, 2, 3)).shape, (1, 2, 3))

    def test_original_single_lane_download_and_summary_unchanged(self):
        engine = ProgressiveRuntime.__new__(ProgressiveRuntime)
        output = engine.download_actions(torch.ones(1, 2, 3))
        self.assertEqual(output.shape, (2, 3))
        summary = engine.verification_summary(None, dict(matched_schedule=output), output, 10)
        self.assertEqual(summary, dict(matched_schedule=dict(mae=0., max_abs=0., finite=True), actual_steps=10))

    def test_batch_download_rejects_lost_lane(self):
        engine, _, _ = cpu_runtime()
        with self.assertRaisesRegex(ValueError, "full batch shape"):
            engine.download_actions(torch.zeros(1, 2, 3))

    def test_each_lane_gate_rejects_error_hidden_by_batch_average(self):
        backend = FakeBackend(4)
        prepared = backend.prepare_batch([1] * 4)
        reference = np.zeros((4, 2, 3), dtype=np.float64)
        actual = reference.copy()
        actual[3] = 3e-6
        result = _batch_comparison(backend, prepared, reference, actual)
        self.assertLess(result["mae"], 1e-6)
        self.assertFalse(result["passed"])
        self.assertEqual([lane["passed"] for lane in result["per_lane"]], [True, True, True, False])
        self.assertEqual(result["per_lane"][3]["lane_id"], "robot3")

    def test_decoded_gate_is_checked_independently(self):
        backend = FakeBackend(2, decode_scale=100)
        prepared = backend.prepare_batch([1, 1])
        reference = np.zeros((2, 2, 3))
        actual = reference + 1e-7
        result = _batch_comparison(backend, prepared, reference, actual)
        self.assertLess(result["normalized"]["mae"], 1e-6)
        self.assertGreater(result["decoded"]["mae"], 1e-6)
        self.assertFalse(result["passed"])

    def test_same_schedule_gate_does_not_require_fresh_equivalence(self):
        engine, prepared, _ = cpu_runtime()
        output = np.ones((2, 2, 3))
        result = engine.verification_summary(prepared,
            dict(matched_schedule=output, versus_fresh=output + .1, versus_old=output - .1), output, 10)
        self.assertTrue(result["passed"])
        self.assertFalse(result["versus_fresh"]["passed"])
        self.assertEqual(result["actual_steps"], 10)

    def test_base_verify_replays_same_frontier_for_all_lanes_without_building_kv(self):
        engine, prepared, noise = cpu_runtime()
        noise[1].fill_(100)
        engine.backend.velocity = lambda prefix, p, x, t: x + prefix.cache[0][0]
        def generate(prefix, p, x):
            for _ in range(2):
                x = engine.backend.velocity(prefix, p, x, None)
            return x
        engine.backend.generate = generate
        slot = SimpleNamespace(prefix=SimpleNamespace(cache=[(torch.tensor(10.), torch.tensor(10.))],
            pad_masks=prepared.token_masks, position_ids=prepared.tokens))
        old = SimpleNamespace(prefix=SimpleNamespace(cache=[(torch.tensor(-1.), torch.tensor(-1.))]))
        expected = (noise + 9).numpy()
        result = engine.verify(slot, old, prepared, noise, expected, [0, 1])
        self.assertTrue(result["matched_schedule"]["passed"])
        self.assertEqual(result["actual_steps"], 2)
        self.assertEqual(len(result["matched_schedule"]["per_lane"]), 2)
        self.assertFalse(result["versus_fresh"]["passed"])
        self.assertFalse(result["versus_old"]["passed"])

    def test_serial_uses_same_captures_but_waits_for_full_prefix(self):
        engine, prepared, noise = cpu_runtime()
        next_input = engine.backend.prepare_batch([2, 2])
        def request(self, p, n, **kwargs):
            self_test.assertEqual(kwargs["lead"], 18)
            self_test.assertFalse(kwargs["next_encoder"])
            return np.zeros((2, 2, 3)), dict(request=3, old_slot=0, old_generation=4), kwargs["lookahead"]()
        self_test = self
        with patch.object(ProgressiveRuntime, "request", request):
            output, row, future = engine.request(prepared, noise, lead=3, schedule="serial",
                                                 next_encoder=True, lookahead=lambda: next_input)
        self.assertIs(future, next_input)
        self.assertEqual(row["schedule"], "serial")
        self.assertEqual(row["requested_lead"], 3)
        self.assertEqual(row["full_kv_slots"], 2)
        self.assertEqual(output.shape, (2, 2, 3))

    def test_pipeline_keeps_requested_lead_and_encoder_overlap(self):
        engine, prepared, noise = cpu_runtime()
        record = dict(request=3, old_slot=0, old_generation=4)
        with patch.object(ProgressiveRuntime, "request", return_value=(np.zeros((2, 2, 3)), record, None)) as invoke:
            _, row, _ = engine.request(prepared, noise, schedule="pipeline", lead=6, next_encoder=True)
        self.assertEqual(invoke.call_args.kwargs["lead"], 6)
        self.assertTrue(invoke.call_args.kwargs["next_encoder"])
        self.assertEqual(row["frontier_scope"], "whole_batch_per_round")

    def test_old_kv_age_uses_consumed_generation_before_release_not_ahead_ticket(self):
        engine, prepared, noise = cpu_runtime()
        engine.pool.vision_ahead = SimpleNamespace(observation=SimpleNamespace(observation_id=999))
        def request(self, p, n, **kwargs):
            # Base release/poll reclaims metadata, but does not re-reserve.
            self.pool.slots[0].observation = None
            return np.zeros((2, 2, 3)), dict(request=8, old_slot=0, old_generation=4), None
        with patch.object(ProgressiveRuntime, "request", request):
            _, row, _ = engine.request(prepared, noise)
        self.assertEqual(row["old_kv_observation_id"], 1)
        self.assertEqual(row["old_kv_age_requests"], 7)
        self.assertEqual(row["old_kv_age_scope"], "request_generations_before_lease_release")

    def test_old_kv_age_rejects_reused_slot_generation(self):
        engine, prepared, noise = cpu_runtime()
        def request(self, p, n, **kwargs):
            self.pool.slots[0].generation = 5
            return np.zeros((2, 2, 3)), dict(request=8, old_slot=0, old_generation=4), None
        with patch.object(ProgressiveRuntime, "request", request):
            with self.assertRaisesRegex(RuntimeError, "consumed slot generation"):
                engine.request(prepared, noise)

    def test_fresh_serial_rejects_force_old(self):
        engine, prepared, noise = cpu_runtime()
        with patch.object(ProgressiveRuntime, "request") as invoke:
            with self.assertRaisesRegex(ValueError, "Fresh serial"):
                engine.request(prepared, noise, schedule="serial", force_old=True)
            invoke.assert_not_called()

    def test_old_epoch_and_layout_fail_before_submission(self):
        engine, prepared, noise = cpu_runtime()
        engine.backend.contract.reset()
        with patch.object(ProgressiveRuntime, "request") as invoke:
            with self.assertRaisesRegex(ValueError, "epoch"):
                engine.request(prepared, noise)
            invoke.assert_not_called()
        fresh = engine.backend.prepare_batch([1, 1])
        with self.assertRaisesRegex(ValueError, "explicitly reset"):
            engine._validate_input(fresh)
        engine._runtime_batch_identity = engine.backend.contract.identity
        fresh.tokens = fresh.tokens[:, :3]
        with self.assertRaisesRegex(ValueError, "shape/dtype/device"):
            engine._validate_input(fresh)

    def test_lookahead_lane_epoch_is_checked_before_use(self):
        engine, prepared, noise = cpu_runtime()
        invalid = engine.backend.prepare_batch([2, 2])
        invalid.batch_identity = (("robot1", "robot0"), 0)
        def request(self, p, n, **kwargs):
            kwargs["lookahead"]()
            raise AssertionError("invalid lookahead accepted")
        with patch.object(ProgressiveRuntime, "request", request):
            with self.assertRaisesRegex(ValueError, "identity/epoch"):
                engine.request(prepared, noise, lookahead=lambda: invalid)

    def test_whole_batch_reset_drains_and_invalidates_old_prepared(self):
        engine, old, _ = cpu_runtime()
        calls = []
        engine.drain = lambda: calls.append("drain")
        def reset(self, prepared):
            calls.append("reset")
            self._validate_input(prepared)
        with patch.object(ProgressiveRuntime, "reset", reset):
            fresh = engine.reset_batch([3, 4])
        self.assertEqual(calls, ["drain", "reset"])
        self.assertEqual(fresh.batch_identity, (("robot0", "robot1"), 1))
        self.assertEqual(engine._runtime_batch_identity, fresh.batch_identity)
        with self.assertRaisesRegex(ValueError, "epoch"):
            engine._validate_input(old)

    def test_inplace_update_keeps_addresses_and_uses_new_contents(self):
        engine, target, _ = cpu_runtime()
        source = engine.backend.prepare_batch([7, 7])
        calls = []
        engine.drain = lambda: calls.append("drain")
        addresses = [value.data_ptr() for value in input_tensors(target)]
        self.assertIs(engine.update_inputs_inplace(target, source), target)
        self.assertEqual(calls, ["drain"])
        self.assertEqual(addresses, [value.data_ptr() for value in input_tensors(target)])
        torch.testing.assert_close(target.images[0], source.images[0])

    def test_graph_label_wraps_real_replay_and_forwards_pool(self):
        events = []
        @contextmanager
        def scope(label):
            events.append(("enter", label))
            yield
            events.append(("exit", label))
        graph = SimpleNamespace(replay=lambda: events.append("replay"), pool=lambda: 42)
        wrapped = LabeledGraph(graph, "vlm", range_factory=scope)
        wrapped.replay()
        self.assertEqual(events, [("enter", "vlm"), "replay", ("exit", "vlm")])
        self.assertEqual(wrapped.pool(), 42)

    def test_completion_metrics_use_encoder_origin_and_include_both_branches(self):
        engine, _, _ = cpu_runtime()
        class Event:
            def __init__(self, timestamp):
                self.timestamp = timestamp
            def elapsed_time(self, other):
                return other.timestamp - self.timestamp
            def synchronize(self):
                pass
        row = dict(_events=dict(vision_begin=Event(0), vision_done=Event(2), reset=Event(3),
            prefix_done=Event(18), ae_begin=Event(5), ae_done=Event(20),
            next_vision_begin=Event(18), next_vision_done=Event(25)))
        self.assertIs(engine.finish_record(row), row)
        self.assertEqual(row["prefill_completion_event_ms"], 18)
        self.assertEqual(row["ae_completion_event_ms"], 20)
        self.assertEqual(row["both_completion_event_ms"], 20)
        self.assertEqual(row["all_submitted_work_completion_event_ms"], 25)
        self.assertNotIn("_events", row)
        self.assertIs(engine.finish_record(row), row)

    def test_invalid_constructor_rejects_before_any_graph_or_cuda_work(self):
        engine, prepared, noise = cpu_runtime()
        with patch.object(ProgressiveRuntime, "__init__") as construct:
            with self.assertRaisesRegex(ValueError, "mode"):
                BatchProgressiveRuntime(engine.backend, prepared, noise, mode="unknown")
            with self.assertRaisesRegex(ValueError, "green_library"):
                BatchProgressiveRuntime(engine.backend, prepared, noise, mode="green")
            construct.assert_not_called()

    def test_close_hands_off_streams_until_caller_tensors_are_released(self):
        engine, _, _ = cpu_runtime()
        events = []
        engine.green_pair = SimpleNamespace(close=lambda: events.append("contexts"))
        with patch.object(ProgressiveRuntime, "close", lambda self: events.append("graphs")):
            resources = engine.close()
            self.assertIsNone(engine.close())
        self.assertEqual(events, ["graphs"])
        events.append("caller_tensors_released")
        resources.close()
        self.assertEqual(events, ["graphs", "caller_tensors_released", "contexts"])

    def test_failed_drain_keeps_green_contexts_alive(self):
        engine, _, _ = cpu_runtime()
        events = []
        engine.green_pair = SimpleNamespace(close=lambda: events.append("contexts"))
        with patch.object(ProgressiveRuntime, "close", side_effect=RuntimeError("drain failed")):
            with self.assertRaisesRegex(RuntimeError, "drain failed"):
                engine.close()
        self.assertEqual(events, [])
        self.assertIsNotNone(engine.green_pair)


if __name__ == "__main__":
    unittest.main()

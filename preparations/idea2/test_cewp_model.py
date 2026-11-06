"""Matched PI0/PI05 suffix and Euler checks against original OpenPI methods."""
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace, MethodType

import torch

from . import graph_sampler
from .runtime_graph import DynamicCore
from .cewp_segmented import SelectiveGraphs, embeddings, language_prefix
from .scheduler import Observation, Pool
from .test_submission import Event


class ModelStrataTests(unittest.TestCase):
    def test_two_steps_use_full_schedule_not_truncated_ten_steps(self):
        result = graph_sampler.cpu_self_test(pi05=True, num_steps=2)
        self.assertTrue(result["passed"])
        self.assertEqual(result["original_time_nodes"], [1.0, 0.5])
        self.assertEqual(result["safe_time_nodes"], [1.0, 0.5])
        self.assertEqual(result["original_nfe"], 2)

    def test_prepared_schedule_preserves_each_velocity_and_fp32_time(self):
        from .cewp_preparation import PreparedScheduleCore
        for steps in (1, 2, 5, 10):
            with self.subTest(steps=steps), patch.object(graph_sampler, "GraphSafeBackend",
                    lambda backend, current, nxt: PreparedScheduleCore(backend, current)):
                result = graph_sampler.cpu_self_test(pi05=True, num_steps=steps)
                self.assertTrue(result["passed"])
                self.assertEqual(result["velocity_error"]["max_abs"], 0)

    def test_original_math_for_both_models_and_each_step_count(self):
        for pi05 in (False, True):
            for steps in (1, 2, 5, 10):
                with self.subTest(pi05=pi05, steps=steps):
                    result = graph_sampler.cpu_self_test(pi05=pi05, num_steps=steps)
                    self.assertTrue(result["passed"])
                    self.assertEqual(result["original_nfe"], steps)
                    self.assertEqual(result["graph_safe_nfe"], steps)
                    self.assertEqual(result["normalized_error"]["max_abs"], 0)

    def test_dynamic_suffix_matches_original_embedding_and_attention(self):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        def factory(backend, current, next_observation):
            current.token_masks[0, 0] = False
            core = DynamicCore(backend, current)
            actions = torch.randn(1, backend.model.config.action_horizon, 2)
            timestep = torch.tensor([0.3], dtype=torch.float32)
            expected, pads, pattern, cond = backend.model.embed_suffix(current.state, actions, timestep)
            actual = core._suffix(current, actions, timestep)
            if backend.model.pi05:
                actual, actual_cond = actual
                self.assertTrue(torch.equal(actual_cond, cond))
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(core.suffix_masks.shape[1], expected.shape[1])
            self.assertTrue(torch.equal(core.suffix_masks, pads))
            self.assertTrue(torch.equal(core.suffix_pattern, pattern))
            prefix = core.build_prefix(current)
            core.generate(prefix, current, actions)
            prefix_part = prefix.pad_masks[:, None, :].expand(1, pads.shape[1], -1)
            attention = backend.model._prepare_attention_masks_4d(
                torch.cat([prefix_part, make_att_2d_masks(pads, pattern)], dim=2))
            self.assertTrue(torch.equal(core.suffix_attention, attention))
            expected_positions = prefix.pad_masks.sum(dim=-1)[:, None]+pads.cumsum(dim=1)-1
            self.assertTrue(torch.equal(core.suffix_positions, expected_positions))
            return core
        for pi05 in (False, True):
            for steps in (1, 2, 5, 10):
                with self.subTest(pi05=pi05, steps=steps), patch.object(graph_sampler, "GraphSafeBackend", factory):
                    self.assertTrue(graph_sampler.cpu_self_test(pi05=pi05, num_steps=steps)["passed"])

    def test_split_prefix_preserves_masks_cache_and_output(self):
        def factory(backend, current, next_observation):
            current.token_masks[0, 0] = False
            next_observation.image_masks[0].fill_(False)
            core = DynamicCore(backend, current)
            core.build_prefix = MethodType(
                lambda self, prepared: language_prefix(self, embeddings(self, prepared)), core)
            return core
        for pi05 in (False, True):
            with self.subTest(pi05=pi05), patch.object(graph_sampler, "GraphSafeBackend", factory):
                self.assertTrue(graph_sampler.cpu_self_test(pi05=pi05, num_steps=1)["passed"])


class SelectiveOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.pool = Pool(task="task")
        self.pool.offer(Observation(0, 0, "task", None))
        self.identity, _ = self.pool.reserve()
        self.prefix = object()
        self.graphs = SelectiveGraphs.__new__(SelectiveGraphs)
        self.graphs.slots = [SimpleNamespace(prefix=self.prefix), SimpleNamespace(prefix=object())]
        self.graphs.kind = "vlm_only"
        self.graphs.vision_pending = True
        self.graphs.vision_done = object()
        self.graphs.ae_stream = Mock()

    def test_published_or_unfinished_reader_slot_cannot_be_overwritten(self):
        self.pool.submitted(self.identity, self.prefix, Event(True))
        self.pool.poll()
        lease = self.pool.acquire()
        with self.assertRaisesRegex(RuntimeError, "exclusive Pool reservation"):
            self.graphs.build_prefix(self.pool, self.identity, None)
        self.pool.release(lease, Event(False))
        with self.assertRaisesRegex(RuntimeError, "exclusive Pool reservation"):
            self.graphs.build_prefix(self.pool, self.identity, None)

    def test_released_lease_is_rejected_before_gpu_scheduling(self):
        self.pool.submitted(self.identity, self.prefix, Event(True))
        lease = self.pool.acquire()
        self.pool.release(lease, Event(True))
        with self.assertRaisesRegex(RuntimeError, "live lease"):
            self.graphs.generate(self.pool, lease, None, None)
        self.graphs.ae_stream.wait_event.assert_not_called()
        self.assertTrue(self.graphs.vision_pending)

    def test_old_epoch_lease_is_rejected_before_gpu_scheduling(self):
        self.pool.submitted(self.identity, self.prefix, Event(True))
        lease = self.pool.acquire()
        self.pool.release(lease, Event(True))
        self.pool.drain_reset(1, "next task")
        with self.assertRaisesRegex(RuntimeError, "Stale slot"):
            self.graphs.generate(self.pool, lease, None, None)
        self.graphs.ae_stream.wait_event.assert_not_called()
        self.assertTrue(self.graphs.vision_pending)


class PreviousPrefixWaitTests(unittest.TestCase):
    def setUp(self):
        from .test_submission import SubmissionTests
        SubmissionTests.setUp(self)

    def make(self):
        from .test_submission import SubmissionTests
        backend, policy = SubmissionTests.make(self, "ae_first")
        backend.await_previous_prefix = True
        return backend, policy

    def reserve_unfinished(self, policy, observation_id):
        observation = Observation(observation_id, policy.pool.epoch, "task", observation_id,
                                  host_receipt_time=0.0, host_receipt_unix=0.0)
        policy.pool.offer(observation)
        identity, _ = policy.pool.reserve()
        test = self
        class Pending(Event):
            waits = 0
            def synchronize(self):
                self.waits += 1
                # An unfinished producer may not become the selected snapshot.
                test.assertFalse(self.complete)
                test.assertEqual(policy.pool._slot(identity).state, "BUILDING")
                test.assertEqual(policy.pool._slot(policy.pool.active).observation.observation_id, 0)
                self.complete = True
        event = Pending(False)
        policy.pool.submitted(identity, dict(source=observation_id), event)
        policy.pool.poll()
        self.assertEqual(policy.pool._slot(policy.pool.active).observation.observation_id, 0)
        return event

    def test_previous_completion_precedes_publication_and_old_kv_selection(self):
        backend, policy = self.make()
        pending = self.reserve_unfinished(policy, 1)
        policy.request = 2
        response = policy.predict(2, "task", "suite", 3)
        self.assertEqual(pending.waits, 1)
        self.assertEqual(response["prefix_observation_id"], 1)
        self.assertEqual(response["observation_request_age"], 1)
        self.assertEqual(backend.selected[-1], 1)

    def test_current_observation_is_neither_waited_for_nor_consumed(self):
        backend, policy = self.make()
        pending = self.reserve_unfinished(policy, 1)
        response = policy.predict(1, "task", "suite", 2)
        self.assertEqual(pending.waits, 0)
        self.assertFalse(pending.query())
        self.assertEqual(response["prefix_observation_id"], 0)
        self.assertEqual(backend.selected[-1], 0)


class ModulationRestorationTests(unittest.TestCase):
    def test_partial_capture_failure_restores_all_dense_forwards_and_parameters(self):
        from .cewp_preparation import PreparedModulationCore, PreparedScheduleCore
        core = PreparedModulationCore.__new__(PreparedModulationCore)
        core.modulation_modules = [torch.nn.Linear(4, 12), torch.nn.Linear(4, 12)]
        core.conditions = [torch.randn(1, 4)]
        core.condition_index = 1
        core.modulations = [[m(core.conditions[0])] for m in core.modulation_modules]
        forwards = [m.forward for m in core.modulation_modules]
        parameters = [[p.detach().clone() for p in m.parameters()] for m in core.modulation_modules]
        observed = []
        def fail_during_capture(self, *args, **kwargs):
            for i, module in enumerate(self.modulation_modules):
                observed.append(module.forward != forwards[i])
                self_test.assertIs(module(self.conditions[0]), self.modulations[i][0])
            raise RuntimeError("capture failed after dense dispatch")
        self_test = self
        with patch.object(PreparedScheduleCore, "generate", fail_during_capture):
            with self.assertRaisesRegex(RuntimeError, "capture failed"):
                core.generate(None, None, None)
        self.assertEqual(observed, [True, True])
        for module, forward, saved in zip(core.modulation_modules, forwards, parameters):
            self.assertEqual(module.forward, forward)
            for parameter, original in zip(module.parameters(), saved):
                self.assertTrue(torch.equal(parameter, original))
            self.assertTrue(torch.equal(module(core.conditions[0]), forward(core.conditions[0])))


if __name__ == "__main__":
    unittest.main()

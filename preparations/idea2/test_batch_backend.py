"""CPU-only checks for the independent-request fixed-batch adapter."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from .batch_backend import BatchOpenPIBackend, FixedLaneBatch
from .openpi_backend import OpenPIBackend


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(pytorch_compile_mode=None, action_horizon=3, action_dim=4)
        language = SimpleNamespace(config=SimpleNamespace())
        expert = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace()))
        self.paligemma_with_expert = SimpleNamespace(
            paligemma=SimpleNamespace(language_model=language), gemma_expert=expert)

    def eval(self):
        pass

    def _preprocess_observation(self, observation, *, train):
        assert not train
        return ([observation["images"]["camera"]], [observation["image_masks"]["camera"]],
                observation["tokens"], observation["token_masks"], observation["state"])


class FakePolicy:
    _is_pytorch_model = True
    _pytorch_device = "cpu"

    def __init__(self):
        self._model = FakeModel()
        self.input_calls = []
        self.output_calls = []

    def _input_transform(self, raw):
        self.input_calls.append(deepcopy(raw))
        raw["state"][0] += 0.5  # Must not mutate the caller's observation.
        return raw

    def _output_transform(self, data):
        self.output_calls.append(deepcopy(data))
        return dict(actions=data["actions"][..., :2] + data["state"][0])


def raw_input(value):
    return dict(images=dict(camera=np.full((2, 2, 3), value, dtype=np.uint8)),
                image_masks=dict(camera=np.array(True)),
                tokens=np.array([value, value + 1, value + 2], dtype=np.int32),
                token_masks=np.array([True, True, False]),
                state=np.array([value, value + 0.25], dtype=np.float32))


class BatchBackendTests(unittest.TestCase):
    def setUp(self):
        self.policy = FakePolicy()
        self.backend = BatchOpenPIBackend(self.policy, lane_ids=("robot-a", "robot-b"))
        self.observation_patch = patch(
            "preparations.idea2.batch_backend._observation_from_dict", side_effect=lambda data: data)
        self.observation_patch.start()
        self.addCleanup(self.observation_patch.stop)

    def prepare(self, values=(1, 7)):
        return self.backend.prepare_batch([raw_input(value) for value in values])

    def test_prepare_keeps_distinct_images_tokens_states_and_raw_ownership(self):
        raws = [raw_input(1), raw_input(7)]
        originals = deepcopy(raws)
        prepared = self.backend.prepare_batch(raws, lane_ids=("robot-a", "robot-b"))
        self.assertEqual(prepared.batch_identity, (("robot-a", "robot-b"), 0))
        self.assertEqual(tuple(prepared.images[0].shape), (2, 2, 2, 3))
        self.assertEqual(prepared.images[0].dtype, torch.uint8)
        self.assertEqual(prepared.tokens.dtype, torch.int32)
        self.assertEqual(prepared.image_masks[0].dtype, torch.bool)
        self.assertTrue(torch.equal(prepared.images[0][0], torch.from_numpy(raws[0]["images"]["camera"])))
        self.assertTrue(torch.equal(prepared.images[0][1], torch.from_numpy(raws[1]["images"]["camera"])))
        np.testing.assert_array_equal(prepared.state.numpy(), [[1.5, 1.25], [7.5, 7.25]])
        for original, raw in zip(originals, raws, strict=True):
            np.testing.assert_array_equal(original["state"], raw["state"])
        # Writing one lane's batch storage cannot change another lane or raw.
        prepared.images[0][0].zero_()
        self.assertTrue(torch.all(prepared.images[0][1] == 7))
        np.testing.assert_array_equal(raws[0]["images"]["camera"], originals[0]["images"]["camera"])

    def test_batch_one_has_original_leading_dimension_and_transform_values(self):
        backend = BatchOpenPIBackend(FakePolicy(), lane_ids=("robot-a",))
        raw = raw_input(3)
        prepared = backend.prepare_batch([raw])
        reference = backend.policy._input_transform(deepcopy(raw))
        self.assertTrue(torch.equal(prepared.state, torch.from_numpy(reference["state"])[None]))
        self.assertTrue(torch.equal(prepared.tokens, torch.from_numpy(reference["tokens"])[None]))
        self.assertEqual(tuple(prepared.image_masks[0].shape), (1,))

    def test_structure_shape_and_dtype_fail_before_device_transfer(self):
        cases = []
        wrong_shape = raw_input(2)
        wrong_shape["tokens"] = wrong_shape["tokens"][:2]
        cases.append((wrong_shape, "shape/dtype"))
        wrong_dtype = raw_input(2)
        wrong_dtype["tokens"] = wrong_dtype["tokens"].astype(np.int64)
        cases.append((wrong_dtype, "shape/dtype"))
        wrong_structure = raw_input(2)
        del wrong_structure["image_masks"]
        cases.append((wrong_structure, "structure"))
        for wrong, message in cases:
            with self.subTest(message=message), patch(
                    "preparations.idea2.batch_backend._to_device") as transfer:
                with self.assertRaisesRegex(ValueError, message):
                    self.backend.prepare_batch([raw_input(1), wrong])
                transfer.assert_not_called()

    def test_wrong_observation_count_and_lane_order_rejected(self):
        with self.assertRaisesRegex(ValueError, "Observation count"):
            self.backend.prepare_batch([raw_input(1)])
        with self.assertRaisesRegex(ValueError, "identities/order"):
            self.backend.prepare_batch([raw_input(1), raw_input(7)], lane_ids=("robot-b", "robot-a"))

    def test_noise_matches_original_per_lane_and_survives_regrouping(self):
        seeds = (137, 239)
        actual = self.backend.noise_batch(seeds)
        for lane, seed in enumerate(seeds):
            original = OpenPIBackend.noise(self.backend, seed)
            self.assertTrue(torch.equal(actual[lane:lane + 1], original))
        single = BatchOpenPIBackend(FakePolicy(), lane_ids=("robot-a",))
        self.assertTrue(torch.equal(actual[:1], single.noise_batch([seeds[0]])))
        reordered = BatchOpenPIBackend(FakePolicy(), lane_ids=("robot-b", "robot-a"))
        self.assertTrue(torch.equal(actual.flip(0), reordered.noise_batch(seeds[::-1])))
        self.assertFalse(torch.equal(actual[0], actual[1]))
        self.assertTrue(torch.equal(actual, self.backend.noise_batch(seeds)))

    def test_noise_count_order_shape_and_dtype_rejected(self):
        with self.assertRaisesRegex(ValueError, "seed count"):
            self.backend.noise_batch([1])
        with self.assertRaisesRegex(ValueError, "integer"):
            self.backend.noise_batch([1, 1.5])
        with self.assertRaisesRegex(ValueError, "identities/order"):
            self.backend.noise_batch([1, 2], lane_ids=("robot-b", "robot-a"))
        with self.assertRaisesRegex(ValueError, "Noise shape"):
            self.backend.validate_noise(torch.zeros(1, 3, 4))
        with self.assertRaisesRegex(ValueError, "float32"):
            self.backend.validate_noise(torch.zeros(2, 3, 4, dtype=torch.float64))

    def test_output_uses_each_lanes_state_and_retains_the_full_batch(self):
        prepared = self.prepare()
        actions = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        result = self.backend.output_batch(prepared, actions)
        expected = np.stack([actions[0, :, :2].numpy() + 1.5,
                             actions[1, :, :2].numpy() + 7.5])
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.shape, (2, 3, 2))
        self.assertEqual(len(self.policy.output_calls), 2)
        np.testing.assert_array_equal(self.policy.output_calls[0]["state"], prepared.state[0].numpy())
        np.testing.assert_array_equal(self.policy.output_calls[1]["state"], prepared.state[1].numpy())
        np.testing.assert_array_equal(self.backend.output_batch(prepared, actions.numpy()), expected)

    def test_output_uses_given_prepared_even_after_lookahead_preparation(self):
        prepared = self.prepare((1, 7))
        self.prepare((11, 17))
        result = self.backend.output_batch(prepared, torch.zeros(2, 3, 4))
        np.testing.assert_array_equal(result[:, 0, 0], [1.5, 7.5])

    def test_output_transform_cannot_mutate_prepared_or_actions(self):
        prepared = self.prepare()
        original_state = prepared.state.clone()
        actions = torch.ones(2, 3, 4)
        def mutating_transform(data):
            data["state"].fill(-100)
            data["actions"].fill(-200)
            return dict(actions=data["actions"])
        self.policy._output_transform = mutating_transform
        self.backend.output_batch(prepared, actions)
        self.assertTrue(torch.equal(prepared.state, original_state))
        self.assertTrue(torch.equal(actions, torch.ones_like(actions)))

    def test_output_shape_and_corrupted_prepared_batch_rejected(self):
        prepared = self.prepare()
        with self.assertRaisesRegex(ValueError, "Action shape"):
            self.backend.output_batch(prepared, torch.zeros(1, 3, 4))
        prepared.tokens = prepared.tokens[:1]
        with self.assertRaisesRegex(ValueError, "fixed batch dimension"):
            self.backend.validate_prepared(prepared)

    def test_whole_batch_reset_rejects_stale_prepared_and_keeps_lane_order(self):
        prepared = self.prepare()
        with self.assertRaises(AttributeError):
            prepared.batch_identity = (("robot-a", "robot-b"), 1)
        with self.assertRaisesRegex(ValueError, "identities/order"):
            self.backend.contract.reset(("robot-b", "robot-a"))
        self.assertEqual(self.backend.contract.epoch, 0)
        self.assertEqual(self.backend.contract.reset(), (("robot-a", "robot-b"), 1))
        self.assertEqual(prepared.batch_identity, (("robot-a", "robot-b"), 0))
        with self.assertRaisesRegex(ValueError, "identity/epoch"):
            self.backend.validate_prepared(prepared)
        with self.assertRaisesRegex(ValueError, "identity/epoch"):
            self.backend.output_batch(prepared, torch.zeros(2, 3, 4))
        fresh = self.prepare()
        self.assertEqual(fresh.batch_identity, self.backend.contract.identity)
        self.assertIs(self.backend.validate_prepared(fresh), fresh)

    def test_contract_rejects_ambiguous_lane_ids_and_is_cpu_only(self):
        for lane_ids in ((), ("a", "a"), "robot-a", ("",), (1,)):
            with self.subTest(lane_ids=lane_ids), self.assertRaises(ValueError):
                FixedLaneBatch(lane_ids)
        contract = FixedLaneBatch(("a", "b"), epoch=4)
        self.assertEqual(contract.identity, (("a", "b"), 4))
        self.assertEqual(contract.reset(), (("a", "b"), 5))
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main()

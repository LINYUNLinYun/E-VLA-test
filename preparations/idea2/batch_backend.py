"""Opt-in, fixed-lane batching for independent OpenPI observations.

This adapter does not collect future observations from one robot. Each lane is
an independent request, and its identity/order stays fixed for the whole batch.
The scheduler must drain outstanding work before a whole-batch reset; changing
this CPU contract does not drain CUDA work or reset a KV pool automatically.
Importing this module does not initialize CUDA.
"""
import copy
import operator
from collections.abc import Mapping

import numpy as np
import torch

from .openpi_backend import OpenPIBackend, Prepared


class FixedLaneBatch:
    """CPU-only lane/episode identity; independent lane resets are unsupported."""

    def __init__(self, lane_ids, *, epoch=0):
        if isinstance(lane_ids, (str, bytes)):
            raise ValueError("lane_ids must be a nonempty sequence of unique strings")
        lane_ids = tuple(lane_ids)
        if (not lane_ids or any(not isinstance(lane, str) or not lane for lane in lane_ids)
                or len(set(lane_ids)) != len(lane_ids)):
            raise ValueError("lane_ids must be a nonempty sequence of unique strings")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self._lane_ids = lane_ids
        self._epoch = epoch

    @property
    def lane_ids(self):
        return self._lane_ids

    @property
    def batch_size(self):
        return len(self._lane_ids)

    @property
    def epoch(self):
        return self._epoch

    @property
    def identity(self):
        return self.lane_ids, self.epoch

    def validate(self, lane_ids=None):
        if lane_ids is not None and (isinstance(lane_ids, (str, bytes))
                                     or tuple(lane_ids) != self.lane_ids):
            raise ValueError("Fixed batch lane identities/order changed; lane replacement is unsupported")
        return self.identity

    def validate_identity(self, identity):
        if identity != self.identity:
            raise ValueError("Prepared batch identity/epoch differs from the active whole-batch contract")
        return self.identity

    def reset(self, lane_ids=None):
        """Advance every lane's epoch after the caller has drained its runtime."""
        self.validate(lane_ids)
        self._epoch += 1
        return self.identity


class BatchPrepared(Prepared):
    """A normal Prepared with a read-only, immutable whole-batch identity."""

    def __init__(self, observation, images, image_masks, tokens, token_masks, state, *, batch_identity):
        super().__init__(observation, images, image_masks, tokens, token_masks, state)
        self._batch_identity = batch_identity

    @property
    def batch_identity(self):
        return self._batch_identity


def _stack_transformed(values, path="input"):
    """Validate every transformed CPU leaf before any device transfer begins."""
    first = values[0]
    if isinstance(first, Mapping):
        keys = tuple(first)
        if any(not isinstance(value, Mapping) or set(value) != set(keys) for value in values):
            raise ValueError(f"Batch input structure differs at {path}")
        return {key: _stack_transformed([value[key] for value in values], f"{path}.{key}")
                for key in keys}
    if isinstance(first, (list, tuple)):
        if any(type(value) is not type(first) or len(value) != len(first) for value in values):
            raise ValueError(f"Batch input structure differs at {path}")
        items = [_stack_transformed([value[index] for value in values], f"{path}[{index}]")
                 for index in range(len(first))]
        return tuple(items) if isinstance(first, tuple) else items
    if any(isinstance(value, (Mapping, list, tuple)) for value in values):
        raise ValueError(f"Batch input structure differs at {path}")
    arrays = [np.array(value, copy=True) for value in values]
    if any(array.shape != arrays[0].shape or array.dtype != arrays[0].dtype for array in arrays):
        raise ValueError(f"Batch input shape/dtype differs at {path}")
    if arrays[0].dtype.kind in "OUSV":
        raise ValueError(f"Batch input leaf must have a numeric dtype at {path}")
    return torch.from_numpy(np.stack(arrays, axis=0))


def _to_device(tree, device):
    if isinstance(tree, dict):
        return {key: _to_device(value, device) for key, value in tree.items()}
    if isinstance(tree, tuple):
        return tuple(_to_device(value, device) for value in tree)
    if isinstance(tree, list):
        return [_to_device(value, device) for value in tree]
    return tree.to(device)


def _observation_from_dict(data):
    from openpi.models.model import Observation
    return Observation.from_dict(data)


class BatchOpenPIBackend(OpenPIBackend):
    """Separate batch API; the original single-observation methods are unchanged.

    Omitted ``lane_ids`` means the constructor's established order. Callers must
    provide observations/seeds in that order, or pass their actual lane IDs for
    explicit checking. A batch-size change requires another adapter and graphs.
    """

    def __init__(self, policy, num_steps=10, *, lane_ids):
        self.contract = FixedLaneBatch(lane_ids)
        super().__init__(policy, num_steps=num_steps)

    @property
    def batch_size(self):
        return self.contract.batch_size

    @torch.no_grad()
    def prepare_batch(self, raws, *, lane_ids=None):
        identity = self.contract.validate(lane_ids)
        raws = tuple(raws)
        if len(raws) != self.batch_size:
            raise ValueError("Observation count must match the fixed batch size")
        transformed = [self.policy._input_transform(copy.deepcopy(raw)) for raw in raws]
        cpu_data = _stack_transformed(transformed)
        data = _to_device(cpu_data, self.device)
        observation = _observation_from_dict(data)
        values = self.model._preprocess_observation(observation, train=False)
        prepared = BatchPrepared(observation, *values, batch_identity=identity)
        self.validate_prepared(prepared)
        return prepared

    def validate_prepared(self, prepared, *, lane_ids=None):
        self.contract.validate(lane_ids)
        self.contract.validate_identity(getattr(prepared, "batch_identity", None))
        tensors = [*prepared.images, *prepared.image_masks, prepared.tokens,
                   prepared.token_masks, prepared.state]
        if not tensors or any(not isinstance(value, torch.Tensor) or value.ndim < 1
                              or value.shape[0] != self.batch_size for value in tensors):
            raise ValueError("Every Prepared tensor must retain the fixed batch dimension")
        return prepared

    def validate_noise(self, noise):
        expected = (self.batch_size, self.model.config.action_horizon, self.model.config.action_dim)
        if not isinstance(noise, torch.Tensor) or tuple(noise.shape) != expected:
            raise ValueError(f"Noise shape must be {expected}")
        if (noise.device.type != self.device.type
                or self.device.index is not None and noise.device.index != self.device.index
                or noise.dtype != torch.float32):
            raise ValueError("Noise must preserve the original device and float32 dtype")
        return noise

    def noise_batch(self, seeds, *, lane_ids=None):
        self.contract.validate(lane_ids)
        seeds = tuple(seeds)
        if len(seeds) != self.batch_size:
            raise ValueError("Noise seed count must match the fixed batch size")
        try:
            seeds = tuple(operator.index(seed) for seed in seeds)
        except TypeError as exc:
            raise ValueError("Each noise seed must be an integer") from exc
        # Independent original B=1 generators preserve each robot's noise when
        # the same lanes are regrouped, unlike one generator for a larger draw.
        noise = torch.cat([super(BatchOpenPIBackend, self).noise(seed) for seed in seeds], dim=0)
        return self.validate_noise(noise)

    def output_batch(self, prepared, actions, *, lane_ids=None):
        self.validate_prepared(prepared, lane_ids=lane_ids)
        expected = (self.batch_size, self.model.config.action_horizon, self.model.config.action_dim)
        if not isinstance(actions, (torch.Tensor, np.ndarray)) or tuple(actions.shape) != expected:
            raise ValueError(f"Action shape must be {expected}")
        states = prepared.state.detach().cpu().numpy()
        normalized = actions.detach().cpu().numpy() if isinstance(actions, torch.Tensor) else actions
        outputs = []
        for lane in range(self.batch_size):
            output = self.policy._output_transform(dict(
                state=np.array(states[lane], copy=True),
                actions=np.array(normalized[lane], copy=True)))
            outputs.append(np.array(output["actions"], copy=True))
        if any(value.shape != outputs[0].shape or value.dtype != outputs[0].dtype for value in outputs):
            raise ValueError("Decoded action shape/dtype differs between lanes")
        return np.stack(outputs, axis=0)

"""Explicit retained-memory floor for a future resident CUDA process.

No torch import, GPU query or allocation occurs on import or while planning.
Allocation is deliberately separate, for a numerically accepted future runtime.
The active frozen experiment does not import this module.
"""
from dataclasses import dataclass


GIB = 1024 ** 3
GRANULARITY = 256 * 1024 ** 2


@dataclass(frozen=True)
class RetentionPlan:
    measured_working_bytes: int
    invariant_storage_bytes: int
    minimum_retained_bytes: int
    extra_allocation_bytes: int
    combined_peak_estimate_bytes: int
    available_bytes: int

    def __post_init__(self):
        values = tuple(self.__dict__.values())
        if any(type(n) is not int or n <= 0 for n in values):
            raise ValueError('Retention plan sizes must be positive integer bytes')
        if self.invariant_storage_bytes > self.measured_working_bytes:
            raise ValueError('Profile does not account for invariant model storage')
        if self.minimum_retained_bytes != self.invariant_storage_bytes + self.extra_allocation_bytes:
            raise ValueError('Retained floor does not equal invariant plus extra storage')
        if self.minimum_retained_bytes <= self.measured_working_bytes:
            raise ValueError('Measured working memory needs explicit retained headroom')
        if self.combined_peak_estimate_bytes != self.measured_working_bytes + self.extra_allocation_bytes:
            raise ValueError('Active peak must include the extra retained allocation')
        if self.combined_peak_estimate_bytes > self.available_bytes:
            raise ValueError('Model plus retained allocation exceeds reviewed capacity')


def plan_retention(*, measured_working_bytes, invariant_storage_bytes, available_bytes):
    for number in (measured_working_bytes, invariant_storage_bytes, available_bytes):
        if type(number) is not int or number <= 0:
            raise ValueError('Memory measurements must be positive integer bytes')
    headroom = max(2 * GIB, (measured_working_bytes + 3) // 4)
    target = measured_working_bytes + headroom
    target = ((target + GRANULARITY - 1) // GRANULARITY) * GRANULARITY
    extra = target - invariant_storage_bytes
    return RetentionPlan(measured_working_bytes, invariant_storage_bytes, target,
                         extra, measured_working_bytes + extra, available_bytes)


def _storage_snapshot(tensors):
    """Count unique CUDA storage, including aliases only once."""
    allocations = {}
    devices = set()
    for tensor in tensors:
        device = tensor.device
        if device.type != 'cuda':
            raise ValueError('Only live CUDA invariant storage can count toward the floor')
        device_key = (device.type, device.index)
        devices.add(device_key)
        storage = tensor.untyped_storage()
        pointer, size = int(storage.data_ptr()), int(storage.nbytes())
        if pointer <= 0 or size <= 0:
            raise ValueError('Invariant CUDA storage must have an allocated address and size')
        key = (device_key, pointer)
        if key in allocations and allocations[key] != size:
            raise ValueError('Aliased storage has inconsistent size')
        allocations[key] = size
    if len(devices) != 1:
        raise ValueError('One resident memory floor must belong to exactly one CUDA device')
    return allocations


def invariant_storage_bytes(tensors):
    return sum(_storage_snapshot(tuple(tensors)).values())


def _allocate_cuda_bytes(number, device):
    import torch
    return torch.empty(number, dtype=torch.uint8, device=device)


class MemoryRetainer:
    """Hold a fixed floor while prompt scratch and CUDA graphs are replaced.

    The profile supplied to plan_retention must include the maximum reviewed
    whole pipeline footprint, including model, graph, scratch and renderer or
    environment allocations. available_bytes excludes the original
    keeper, other reviewed allocations and required device safety margin.
    Actual GPU behavior remains unvalidated until the future memory gate.
    """

    def __init__(self, tensors, plan, *, allocate=None):
        self.anchors = tuple(tensors)
        self.plan = plan
        self._snapshot = _storage_snapshot(self.anchors)
        if sum(self._snapshot.values()) != plan.invariant_storage_bytes:
            raise ValueError('Actual invariant storage differs from the reviewed memory plan')
        self._allocate = allocate if allocate is not None else _allocate_cuda_bytes
        self._buffer = None
        self._buffer_snapshot = None
        self._held = False

    @property
    def held(self):
        return self._held

    def retain(self):
        if self._held:
            self.assert_held()
            return
        if self._buffer is not None:
            raise RuntimeError('A failed allocation needs explicit shutdown reconciliation')
        self._buffer = self._allocate(self.plan.extra_allocation_bytes, self.anchors[0].device)
        try:
            snapshot = _storage_snapshot((self._buffer,))
            if set(snapshot) & set(self._snapshot):
                raise RuntimeError('Retained buffer aliases invariant model storage')
            if sum(snapshot.values()) < self.plan.extra_allocation_bytes:
                raise RuntimeError('Retained buffer is smaller than the reviewed requirement')
            if next(iter(snapshot))[0] != next(iter(self._snapshot))[0]:
                raise RuntimeError('Retained buffer is on another CUDA device')
            self._buffer_snapshot = snapshot
            self._held = True
            self.assert_held()
        except Exception:
            self._held = False
            raise

    def assert_held(self):
        try:
            if not self._held or self._buffer is None:
                raise RuntimeError('Retained memory floor is not established')
            if _storage_snapshot(self.anchors) != self._snapshot:
                raise RuntimeError('Invariant CUDA storage changed')
            if _storage_snapshot((self._buffer,)) != self._buffer_snapshot:
                raise RuntimeError('Retained buffer storage changed')
            total = sum(self._snapshot.values()) + sum(self._buffer_snapshot.values())
            if total < self.plan.minimum_retained_bytes:
                raise RuntimeError('Live storage is below the reviewed retention floor')
        except Exception:
            self._held = False
            raise

    def release_for_shutdown(self):
        """Explicit final shutdown only, never condition switches or analysis."""
        self._buffer = None
        self._buffer_snapshot = None
        self._held = False

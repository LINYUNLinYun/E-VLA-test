"""CPU fake-storage contracts; these allocate no real device memory."""
from types import SimpleNamespace
import unittest

from .memory_floor import GIB, MemoryRetainer, invariant_storage_bytes, plan_retention


class Storage:
    def __init__(self, address, size):
        self.address, self.size = address, size

    def data_ptr(self):
        return self.address

    def nbytes(self):
        return self.size


class Tensor:
    def __init__(self, address, size, *, device=None, storage=None):
        self.device = device if device is not None else SimpleNamespace(type='cuda', index=0)
        self.storage = storage if storage is not None else Storage(address, size)

    def untyped_storage(self):
        return self.storage


class MemoryFloorContracts(unittest.TestCase):
    def setup_floor(self, allocate=None):
        anchors = [Tensor(1, 4 * GIB), Tensor(2, 2 * GIB)]
        plan = plan_retention(measured_working_bytes=8 * GIB,
            invariant_storage_bytes=invariant_storage_bytes(anchors), available_bytes=24 * GIB)
        allocations = []
        def default_allocate(number, device):
            allocations.append(number)
            return Tensor(3, number, device=device)
        retainer = MemoryRetainer(anchors, plan, allocate=allocate or default_allocate)
        return anchors, plan, allocations, retainer

    def test_eight_gib_requirement_retains_ten_without_second_model_allocation(self):
        _, plan, allocations, retainer = self.setup_floor()
        self.assertEqual(plan.minimum_retained_bytes, 10 * GIB)
        self.assertEqual(plan.extra_allocation_bytes, 4 * GIB)
        self.assertEqual(plan.combined_peak_estimate_bytes, 12 * GIB)
        retainer.retain()
        retainer.retain()
        self.assertEqual(allocations, [4 * GIB])
        self.assertTrue(retainer.held)

    def test_insufficient_capacity_rejected_before_allocating(self):
        with self.assertRaises(ValueError):
            plan_retention(measured_working_bytes=8 * GIB,
                invariant_storage_bytes=6 * GIB, available_bytes=11 * GIB)

    def test_shared_weight_views_count_the_same_storage_once(self):
        original = Tensor(1, 6 * GIB)
        alias = Tensor(1, 6 * GIB, storage=original.storage)
        self.assertEqual(invariant_storage_bytes((original, alias)), 6 * GIB)

    def test_other_device_or_cpu_storage_cannot_count_as_reservation(self):
        with self.assertRaises(ValueError):
            invariant_storage_bytes((Tensor(1, GIB), Tensor(2, GIB,
                device=SimpleNamespace(type='cuda', index=1))))
        with self.assertRaises(ValueError):
            invariant_storage_bytes((Tensor(1, GIB, device=SimpleNamespace(type='cpu', index=None)),))

    def test_storage_replacement_invalidates_floor_even_if_tensor_object_survives(self):
        anchors, _, _, retainer = self.setup_floor()
        retainer.retain()
        anchors[0].storage = Storage(9, 4 * GIB)
        with self.assertRaises(RuntimeError):
            retainer.assert_held()
        self.assertFalse(retainer.held)

    def test_undersized_or_aliased_buffer_is_not_credited(self):
        for allocate in (lambda number, device: Tensor(3, number - 1, device=device),
                         lambda number, device: Tensor(1, number, device=device)):
            _, _, _, retainer = self.setup_floor(allocate)
            with self.assertRaises(RuntimeError):
                retainer.retain()
            self.assertFalse(retainer.held)

    def test_failed_allocation_does_not_claim_held_memory(self):
        def fail(number, device):
            raise MemoryError('Synthetic allocation failure')
        _, _, _, retainer = self.setup_floor(fail)
        with self.assertRaises(MemoryError):
            retainer.retain()
        self.assertFalse(retainer.held)

    def test_explicit_shutdown_removes_floor_credit(self):
        _, _, _, retainer = self.setup_floor()
        retainer.retain()
        retainer.release_for_shutdown()
        self.assertFalse(retainer.held)
        with self.assertRaises(RuntimeError):
            retainer.assert_held()


if __name__ == '__main__':
    unittest.main(verbosity=2)

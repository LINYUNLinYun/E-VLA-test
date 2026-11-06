"""CPU-only interface rejection tests; do not initialize a CUDA context."""
import unittest
from unittest.mock import patch

import torch

from . import progressive_kv_kernels as kernels


class ProgressiveKVInterfaceTests(unittest.TestCase):
    def test_cpu_control_rejected_before_launch(self):
        with patch.object(kernels, "_publish") as launch:
            with self.assertRaisesRegex(ValueError, "CUDA Tensor"):
                kernels.publish(torch.zeros((), dtype=torch.int32), 1)
            launch.assert_not_called()
        with self.assertRaisesRegex(ValueError, "CUDA Tensor"):
            kernels.snapshot(torch.zeros((), dtype=torch.int32), torch.zeros(10, dtype=torch.int32), 0)

    def test_nonintegral_and_negative_indices_rejected(self):
        for value in (-1, 1.5, True, 2**31):
            with self.assertRaises(ValueError):
                kernels._integer(value, "layer")

    def test_overlapping_views_rejected(self):
        backing = torch.empty(32)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            kernels._disjoint((backing[:16], backing[8:24]))
        kernels._disjoint((backing[:16], backing[16:]))

    def test_select_rejects_bad_pairs_before_launch(self):
        with patch.object(kernels, "_step", return_value=0), patch.object(kernels, "_select") as launch:
            with self.assertRaisesRegex(ValueError, "exactly two"):
                kernels.select_layer((None,), (None, None), (None, None), None, 0, 0)
            launch.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""CPU validation of numerical gate; real-model gate must run separately."""
import unittest
import numpy as np
from .residual_experiment import compare


class NumericalGate(unittest.TestCase):
    def test_reject_nonfinite(self):
        self.assertFalse(compare([np.nan],[np.nan])["passed"])
        self.assertFalse(compare([np.inf],[np.inf])["passed"])

    def test_reject_broadcasting_shape_mismatch(self):
        self.assertFalse(compare([[1,2]],[1,2])["passed"])

    def test_gate_tolerance(self):
        self.assertTrue(compare([1.0],[1.000001])["passed"])
        self.assertFalse(compare([1.0],[1.001])["passed"])


if __name__ == "__main__":
    unittest.main()

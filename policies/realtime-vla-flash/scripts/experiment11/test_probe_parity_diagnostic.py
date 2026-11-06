"""CPU checks for the diagnostic's authority and accounting boundaries."""
import ast
from pathlib import Path
import unittest
import numpy as np
from scripts.experiment11.replay_state import client_state_from_telemetry


class DiagnosticBoundaryTests(unittest.TestCase):
    def test_state_precision_restored_without_mutating_source(self):
        original = np.array([.123456789, .2, .3, 0., 0., 0., .0381234567, -.0381234567])
        row = {'eef_pos': original[:3].tolist(), 'eef_quat': [0., 0., 0., 1.],
               'gripper_qpos': original[6:].tolist()}
        cached = original.astype(np.float32)
        restored = client_state_from_telemetry(row, cached)
        np.testing.assert_array_equal(restored, original)
        np.testing.assert_array_equal(cached, original.astype(np.float32))
        self.assertEqual(restored.dtype, np.float64)
        self.assertGreater(np.max(np.abs(restored-cached)), 0.)

    def test_mismatching_telemetry_is_rejected(self):
        row = {'eef_pos': [0., 0., 1.], 'eef_quat': [0., 0., 0., 1.], 'gripper_qpos': [0., 0.]}
        with self.assertRaisesRegex(RuntimeError, 'round-trip'):
            client_state_from_telemetry(row, np.zeros(8, dtype=np.float32))

    def test_no_environment_creation_or_advancement(self):
        source = Path(__file__).with_name('diagnose_probe_parity.py').read_text()
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        attributes = {node.func.attr for node in calls if isinstance(node.func, ast.Attribute)}
        self.assertNotIn('step', attributes)
        self.assertNotIn('reset', attributes)
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        self.assertFalse(any(name and (name.startswith('libero') or name.startswith('robosuite'))
                             for name in imports))

    def test_original_ledger_and_strict_capture_gate_are_reused(self):
        source = Path(__file__).with_name('diagnose_probe_parity.py').read_text()
        self.assertIn("directory / 'call_budget.sqlite', 14400", source)
        self.assertIn('capture_full(backend, sample, budget)', source)
        self.assertIn('accounted_infer(', source)
        self.assertIn("raise RuntimeError('Diagnostic output already exists')", source)
        tree = ast.parse(source)
        self.assertFalse(any(isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Attribute) and t.attr == 'parity_lengths' for t in n.targets)
            for n in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()

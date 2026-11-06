import ast
from pathlib import Path
import unittest
import tempfile
import json
from scripts.experiment11 import run_task_type_layer_probe as probe


class ProbeProtocolTest(unittest.TestCase):
    def test_first_actual_replan_per_phase_without_inventing_missing(self):
        inference = [{'infer_id': 2, 'frame_idx': 10}, {'infer_id': 0, 'frame_idx': 0}, {'infer_id': 1, 'frame_idx': 5}]
        telemetry = [{'step': 0, 'when': 'initial', 'semantic_phase': 'initial'},
                     {'step': 5, 'when': 'post_action', 'semantic_phase': 'approach'},
                     {'step': 10, 'when': 'post_action', 'semantic_phase': 'contact'}]
        selected = probe.choose_replans(inference, telemetry)
        self.assertEqual(selected['approach']['infer_id'], 0)
        self.assertEqual(selected['contact_grasp']['infer_id'], 2)
        self.assertNotIn('transport', selected)

    def test_l18_excluded_and_attempt_path_validated(self):
        self.assertEqual(probe.LAYERS, tuple(range(1,18)))
        with self.assertRaises(ValueError):
            probe.paths('../bad')

    def test_runner_never_creates_or_steps_an_environment(self):
        tree = ast.parse(Path(probe.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, {'OffScreenRenderEnv', 'make_env'})
                if isinstance(node.func, ast.Attribute):
                    self.assertNotEqual(node.func.attr, 'step')

    def test_retry_paths_are_isolated_without_changing_budget_root(self):
        original = probe.job_directory('round2_20261011', 'libero_object', 4)
        retry = probe.job_directory('round2_20261011', 'libero_object', 4, 'warmup_retry1')
        self.assertNotEqual(original, retry)
        self.assertIn('retries/warmup_retry1', str(retry))
        self.assertEqual(probe.paths('round2_20261011')[0], original.parents[1])
        with self.assertRaises(ValueError):
            probe.job_directory('round2_20261011', 'libero_object', 4, '../bad')

    def test_warmup_retry_requires_both_strict_diagnostic_proofs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                probe.verify_warmup_diagnostics(root)
            for task in (4,5):
                record = {'sample_id': f'libero_object__task{task:02d}__init00__approach',
                          'history_scope': 'original task0 warmup plus last replan of init4 per preceding task; not complete replay',
                          'paths': {'target_after_history': {'max_abs': 1e-7}}}
                (root/f'parity_diagnostic_task{task:02d}_history_v1.json').write_text(json.dumps(record))
            probe.verify_warmup_diagnostics(root)
            record['paths']['target_after_history']['max_abs'] = .0078
            (root/'parity_diagnostic_task05_history_v1.json').write_text(json.dumps(record))
            with self.assertRaises(RuntimeError):
                probe.verify_warmup_diagnostics(root)


if __name__ == '__main__':
    unittest.main()

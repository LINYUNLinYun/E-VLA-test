"""CPU CLI workflows, with native-model imports explicitly forbidden."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from .budget import load_budget


ROOT = Path(__file__).resolve().parents[2]
# A separate interpreter exercises the public entry point and its persisted
# artifacts. An accidental eager model import fails before any device access.
CPU_ENTRY = """
import builtins
import runpy
import sys
original_import = builtins.__import__
def cpu_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split('.')[0] in ('torch', 'triton'):
        raise AssertionError('CPU budget CLI imported native model dependency: ' + name)
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = cpu_import
sys.argv = ['budgeted_skip'] + sys.argv[1:]
runpy.run_module('preparations.idea1_v3_4_E2_budgeted_skip', run_name='__main__')
"""


class BudgetCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='budget_cli_', dir='/tmp')
        self.folder = Path(self.temp.name)
        self.costs = self.folder / 'cost_table.json'
        self.costs.write_text(json.dumps(dict(measured=True, units='ms', n_layers=3,
            layer_costs=[dict(layer=i, attn_saved_ms=10., ffn_saved_ms=2. if i == 2 else 1.,
                             af_saved_ms=12. if i == 2 else 11.) for i in range(1, 4)])))
        self.artifacts = self.folder / 'artifacts'

    def tearDown(self):
        self.temp.cleanup()

    def cli(self, *args, success=True):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        result = subprocess.run([sys.executable, '-c', CPU_ENTRY, *map(str, args)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        return result

    def prepare(self, *target):
        return self.cli('prepare', '--cost-table', self.costs, '--output', self.artifacts,
                        '--tolerance-ms', '0', *target)

    def test_prepare_and_select_persist_one_budget_for_both_objectives(self):
        prepared = self.prepare('--save-ms', '2')
        budget = load_budget(prepared['artifact_path'])
        self.assertEqual(prepared['budget_hash'], budget['budget_hash'])
        self.assertEqual(prepared['anchor_saved_ms'], 2.)
        self.assertEqual(prepared['effective_window_ms'], [2., 2.])
        self.assertEqual(prepared['candidate_count'], 2)
        self.assertIsNone(prepared['reference_vlm_ms'])
        scores = np.asarray([[100., .1], [100., .35], [100., .2]])
        json_path, archive_path = self.folder / 'scores.json', self.folder / 'scores.npz'
        json_path.write_text(json.dumps(scores.tolist()))
        np.savez(archive_path, error=scores, original=scores * 3)
        plain = self.cli('select', '--budget', prepared['artifact_path'], '--scores', json_path)
        self.assertEqual(plain['selected_attn'], [])
        self.assertEqual(plain['selected_ffn'], [1, 3])
        self.assertFalse(plain['cost_weighted'])
        output = self.folder / 'selected' / 'weighted.json'
        weighted = self.cli('select', '--budget', prepared['artifact_path'], '--scores', archive_path,
                            '--score-key', 'error', '--weighted', '--output', output)
        self.assertEqual(weighted['selected_attn'], [])
        self.assertEqual(weighted['selected_ffn'], [2])
        self.assertTrue(weighted['cost_weighted'])
        self.assertEqual(json.loads(output.read_text()), weighted)
        self.assertEqual(plain['budget_hash'], weighted['budget_hash'])
        self.assertEqual(plain['saved_ms'], weighted['saved_ms'])

    def test_fraction_syntax_uses_measured_complete_VLM_reference(self):
        timings = self.folder / 'request_timings.jsonl'
        rows = [dict(phase='capture', route='keep', sample=0, vlm_ms=2000.),
                dict(phase='hot', route='F+', sample=0, vlm_ms=1000.)]
        rows += [dict(phase='hot', route='keep', sample=i, vlm_ms=ms)
                 for i, ms in enumerate((18., 20., 22.))]
        timings.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        prepared = self.prepare('--save-fraction', '1/10')
        budget = load_budget(prepared['artifact_path'])
        self.assertEqual(prepared['anchor_saved_ms'], 2.)
        self.assertEqual(prepared['reference_vlm_ms'], 20.)
        self.assertEqual(budget['requested_saved_fraction'], .1)
        self.assertEqual(budget['reference_source']['kind'], 'measured_hot_keep')
        self.assertEqual(budget['reference_source']['path'], str(timings.resolve()))
        self.assertEqual(budget['reference_source']['count'], 3)

    def test_ambiguous_score_archive_fails_without_writing_a_mask(self):
        prepared = self.prepare('--save-ms', '2')
        archive, output = self.folder / 'scores.npz', self.folder / 'mask.json'
        np.savez(archive, error=np.ones((3, 2)), original=np.zeros((3, 2)))
        failed = self.cli('select', '--budget', prepared['artifact_path'], '--scores', archive,
                          '--output', output, success=False)
        self.assertIn('--score-key is required', failed.stderr)
        self.assertFalse(output.exists())

    def test_conflicting_target_units_fail_before_artifact_creation(self):
        failed = self.cli('prepare', '--cost-table', self.costs, '--output', self.artifacts,
                          '--save-ms', '2', '--save-fraction', '1/10', success=False)
        self.assertEqual(failed.returncode, 2)
        self.assertIn('not allowed with argument', failed.stderr)
        self.assertFalse(self.artifacts.exists())


if __name__ == '__main__':
    unittest.main()

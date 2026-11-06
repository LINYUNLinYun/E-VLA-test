"""Ensure the controller cannot analyze partial or attention-marked confirmation."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('finish_confirmation.py')
SPEC = importlib.util.spec_from_file_location('AE_pair_confirmation_controller_test', SOURCE)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


class ControllerTest(unittest.TestCase):
    def test_pending_does_not_invoke_scientific_analysis(self):
        with tempfile.TemporaryDirectory() as folder:
            value = dict(new_cases=359, completed_shards=5, attention=False, all_complete=False)
            with patch.object(controller, 'REPORT', Path(folder)), patch.object(controller, 'verify', return_value={}), \
                 patch.object(controller, 'progress', return_value=value), patch.object(controller.subprocess, 'run') as invoke:
                result = controller.run('manifest', 'source', False)
            self.assertEqual(result, dict(status='pending', new_cases=359))
            invoke.assert_not_called()

    def test_attention_prevents_analysis_even_at_full_count(self):
        with tempfile.TemporaryDirectory() as folder:
            value = dict(new_cases=360, completed_shards=6, attention=True, all_complete=True)
            with patch.object(controller, 'REPORT', Path(folder)), patch.object(controller, 'verify', return_value={}), \
                 patch.object(controller, 'progress', return_value=value), patch.object(controller.subprocess, 'run') as invoke:
                with self.assertRaises(RuntimeError): controller.run('manifest', 'source', False)
            invoke.assert_not_called()

    def test_controller_source_pin_rejects_changed_code_before_manifest_read(self):
        with self.assertRaises(ValueError): controller.verify('absent', 'incorrect_source_hash')

    def test_completion_requires_each_condition_and_all_six_shards(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            study = root / 'study'
            report = root / 'report'
            report.mkdir()
            slot = root / 'results/ease_insights_20261103/resource_pool_v4/sessions/slot'
            slot.mkdir(parents=True)
            (slot / 'status.json').write_text(json.dumps(dict(phase='running_condition')))
            ids = ['full_d10', 'path_a', 'path_b']
            manifest = dict(manifest_hash='science', planned_conditions=ids)
            for cid in ids:
                for i in range(120):
                    p = study / cid / 'episodes' / f'task{i:03d}' / 'init47'
                    p.mkdir(parents=True)
                    (p / 'episode_receipt.json').write_text('{}')
                for shard in range(2): (study / cid / f'confirm_shard{shard:02d}of02_complete.json').write_text('{}')
            with patch.object(controller, 'ROOT', root), patch.object(controller, 'REPORT', report), patch.object(controller, 'STUDY', study):
                self.assertTrue(controller.progress({}, manifest)['all_complete'])
                (study / 'path_b/confirm_shard01of02_complete.json').unlink()
                self.assertFalse(controller.progress({}, manifest)['all_complete'])
                (study / 'path_b/confirm_shard01of02_complete.json').write_text('{}')
                (study / 'path_b/episodes/task119/init47').rename(study / 'path_b/episodes/task119/init47.invalid.technical')
                value = controller.progress({}, manifest)
                self.assertFalse(value['all_complete'])
                self.assertEqual(value['new_cases'], 359)


if __name__ == '__main__': unittest.main()

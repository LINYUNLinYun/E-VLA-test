"""CPU-only cache provenance dispatch and actual event-binding boundaries."""
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from common import atomic_json
from pi0_prompt_cache_audit_helpers import cache_events, verify_episode_cache
import throughput_provenance_audit_v2 as auditor


class CacheAuditTests(unittest.TestCase):
    def event(self):
        return dict(call=1, capacity=2, resident_lengths=[10, 14], hit=True, startup_forward_calls=0,
            startup_action_NFE=0, cuda_peak_allocated_mib=12000., cuda_peak_reserved_mib=13000.,
            input_hash='input', noise_hash='noise', seed=7, skip=[3, 4, 9, 11],
            schedule={'kind': 'flow', 'm': .5, 'num_steps': 2}, prediction_path='/record/raw.npz')

    def write_event(self, path, event):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w') as stream: stream.write(json.dumps(event)+'\n')

    def test_explicit_manifest_kind_dispatch_preserves_legacy_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root/'revision_manifest.json', {'kind': 'pi0_prompt_length_runtime_cache2'})
            with patch('pi0_prompt_cache_revision.verify_revision', return_value=('cache', {})) as cached, \
                 patch.object(auditor, 'verify_legacy_revision', return_value=('legacy', {})) as legacy:
                self.assertEqual(auditor.verify_revision(root)[0], 'cache')
                legacy.assert_not_called()
                atomic_json(root/'revision_manifest.json', {'kind': 'two_env_sequential_server_execution_revision'})
                self.assertEqual(auditor.verify_revision(root)[0], 'legacy')
                cached.assert_called_once()

    def test_hit_startup_and_allocator_cap_are_fail_closed(self):
        for change in ({'startup_forward_calls': 4}, {'cuda_peak_reserved_mib': 23*1024}, {'resident_lengths': [10, 14, 21]}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'prompt_cache_events.jsonl'
                self.write_event(path, dict(self.event(), **change))
                with self.assertRaises(ValueError): cache_events(path.parent)

    def test_formal_inference_must_match_actual_session_cache_event(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            self.write_event(session/'server/prompt_cache_events.jsonl', self.event())
            inference = dict(call=1, input_hash='input', noise_hash='noise', seed=7,
                server_prediction_path='/record/raw.npz', audit={'skip': [3, 4, 9, 11],
                    'schedule': {'kind': 'flow', 'm': .5, 'num_steps': 2}})
            path = session/'infer.jsonl'
            self.write_event(path, inference)
            index, record = types.SimpleNamespace(), {}
            with patch('pi0_prompt_cache_audit_helpers.verify_patch'):
                verify_episode_cache(index, {'throughput_session': str(session)}, record, path)
                self.assertIn('prompt_cache_ledger_prefix', record)
                inference['seed'] = 8
                self.write_event(path, inference)
                with self.assertRaises(ValueError): verify_episode_cache(index, {'throughput_session': str(session)}, record, path)


if __name__ == '__main__':
    unittest.main()

"""Independent, CPU-only acceptance of the benchmark execution boundary."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from benchmark_gpu_identity import CausalIdentity
import benchmark_revision
from common import sha


class IndependentBenchmarkRevision(unittest.TestCase):
    def test_analysis_route_preserves_verified_file_and_original_audit(self):
        from benchmark_revision_analysis import install_analysis_route
        with tempfile.TemporaryDirectory(prefix='v22-independent-view-') as temporary:
            parent, view = Path(temporary)/'source', Path(temporary)/'view'
            episode = parent/'episodes'/'condition'/'episode'
            episode.mkdir(parents=True)
            video = episode/'video.mp4'
            video.write_bytes(b'fixture: test path identity, not video validity')
            view.mkdir()
            (view/'episodes').symlink_to(parent/'episodes', target_is_directory=True)
            original = mock.Mock(return_value=({'complete': True}, {'key': {'directory': str(episode)}}))
            module = SimpleNamespace(audit=original)
            install_analysis_route(module, view, parent)
            _, valid = module.audit(view, False, False)
            original.assert_called_once_with(view, False, False)
            alias = Path(valid['key']['directory'])/'video.mp4'
            self.assertEqual(str(alias.relative_to(view)), 'episodes/condition/episode/video.mp4')
            self.assertEqual(alias.resolve(), video.resolve())
            self.assertEqual(sha(alias), sha(video))
            original.return_value = ({}, {'key': {'directory': str(Path(temporary)/'outside')}})
            with self.assertRaises(ValueError):
                module.audit(view)
            with self.assertRaises(ValueError):
                module.audit(parent)

    def binding(self, ending=None):
        memory = [{}, {9001: 32}, {9001: 32}, {9001: 160}, {9001: 32}, {9001: 32}]
        memory += [ending if ending is not None else {9001: 32}]
        snapshots = iter(dict(gpu='0', uuid='GPU-abc', processes=pids, timestamp=i)
                         for i, pids in enumerate(memory))
        events = []
        local = dict(pid=77, start_ticks='101', pid_namespace='pid:[202]')
        identity = CausalIdentity('0', events.append, scanner=lambda gpu: next(snapshots),
                                  identity=lambda: dict(local))
        torch = mock.Mock()
        torch.cuda.is_initialized.return_value = False
        torch.cuda.get_device_properties.return_value.uuid = 'GPU-abc'
        return identity, torch, local, events

    def test_namespace_mismatch_requires_real_causal_challenge(self):
        binding, torch, local, events = self.binding()
        binding.bind(torch)
        self.assertEqual(binding.pid, 9001)
        self.assertNotEqual(binding.pid, local['pid'])
        binding.assert_exclusive()
        self.assertEqual([e['label'] for e in events], ['empty_baseline', 'context_created',
            'context_stable', 'challenge_allocated', 'challenge_released',
            'causal_identity_bound', 'point_boundary'])
        self.assertEqual(torch.empty.call_count, 2)
        self.assertGreaterEqual(torch.cuda.synchronize.call_count, 3)

    def test_arbitrary_single_pid_or_empty_or_competitor_is_rejected(self):
        for pids in ({9002: 32}, {}, {9001: 32, 9002: 32}):
            with self.subTest(pids=pids):
                binding, torch, _, _ = self.binding(pids)
                binding.bind(torch)
                with self.assertRaises(RuntimeError):
                    binding.assert_exclusive()

    def test_namespace_reuse_or_preinitialized_cuda_is_rejected(self):
        binding, torch, local, _ = self.binding()
        binding.bind(torch)
        local['pid_namespace'] = 'pid:[different]'
        with self.assertRaises(RuntimeError):
            binding.assert_exclusive()
        binding, torch, _, _ = self.binding()
        torch.cuda.is_initialized.return_value = True
        with self.assertRaises(RuntimeError):
            binding.bind(torch)
        torch.cuda.init.assert_not_called()

    def test_copied_metadata_and_medium_signature_are_bound(self):
        with tempfile.TemporaryDirectory(prefix='v22-independent-benchmark-') as temporary:
            parent, revised = Path(temporary)/'parent', Path(temporary)/'revision'
            parent.mkdir()
            manifest = dict(manifest_hash='scientific-manifest', code_hashes={'frozen.py': 'frozen'},
                            protocol_hashes={'pi0': 'protocol'})
            for name, value in [('condition_manifest.json', manifest), ('runtime_acceptance.json', {})]:
                (parent/name).write_text(json.dumps(value))
            for name in ('protocols', 'parity'):
                (parent/name).mkdir()
                (parent/name/'pi0.json').write_text('{}')
            with mock.patch.object(benchmark_revision, 'verify_frozen', return_value=manifest):
                benchmark_revision.prepare(parent, revised)
                benchmark_revision.verify_revision(revised, require_acceptance=False)
                acceptance = dict(schema=1, accepted=True, reviewer_model='gpt-6.1-sol',
                    reasoning_effort='medium', revision_manifest_sha256=sha(revised/'revision_manifest.json'),
                    evidence=['real-causal-gate.json'])
                (revised/'benchmark_revision_acceptance.json').write_text(json.dumps(acceptance))
                benchmark_revision.verify_revision(revised)
                for field, value in [('reasoning_effort', 'high'), ('revision_manifest_sha256', 'stale')]:
                    (revised/'benchmark_revision_acceptance.json').write_text(json.dumps(dict(acceptance, **{field: value})))
                    with self.assertRaises(ValueError):
                        benchmark_revision.verify_revision(revised)
                (revised/'benchmark_revision_acceptance.json').write_text(json.dumps(acceptance))
                (revised/'parity/pi0.json').write_text('{"changed": true}')
                with self.assertRaises(ValueError):
                    benchmark_revision.verify_revision(revised)


if __name__ == '__main__':
    unittest.main()

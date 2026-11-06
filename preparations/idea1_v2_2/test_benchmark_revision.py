"""CPU-only failure-closed tests; never initialize CUDA or invoke nvidia-smi."""
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_gpu_identity import CausalIdentity
import benchmark_revision as revision
from benchmark_revision_analysis import install_analysis_route
from common import atomic_json, sha


class FakeTensor:
    def fill_(self, value):
        return self


class FakeCuda:
    def is_initialized(self): return False
    def init(self): pass
    def synchronize(self): pass
    def empty_cache(self): pass
    def get_device_properties(self, device):
        return type('Properties', (), {'uuid': 'GPU-abc'})()


class FakeTorch:
    cuda = FakeCuda()
    uint8 = object()
    def empty(self, *args, **kwargs): return FakeTensor()


class IdentityTests(unittest.TestCase):
    def make(self, process_sets=None, identity=None):
        self.events = []
        sets = process_sets if process_sets is not None else [{}, {41: 2}, {41: 2}, {41: 130}, {41: 2}, {41: 2}]
        rows = iter([dict(gpu='0', uuid='GPU-abc', processes=p, timestamp=n) for n, p in enumerate(sets)])
        return CausalIdentity('0', self.events.append, scanner=lambda gpu: next(rows),
                              identity=identity or (lambda: dict(pid=1, start_ticks='2', pid_namespace='pid:[3]')))

    def test_allocation_release_binds_unique_pid(self):
        binding = self.make()
        binding.bind(FakeTorch())
        self.assertEqual(binding.pid, 41)
        self.assertEqual(len(self.events), 6)
        with self.assertRaises(RuntimeError): binding.bind(FakeTorch())

    def test_nonempty_baseline_rejected(self):
        binding = self.make([{41: 2}])
        with self.assertRaises(RuntimeError): binding.bind(FakeTorch())

    def test_empty_or_multiple_context_rejected(self):
        for processes in ({}, {41: 2, 42: 2}):
            with self.subTest(processes=processes):
                with self.assertRaises(RuntimeError): self.make([{}, processes]).bind(FakeTorch())

    def test_challenge_must_increase_and_decrease(self):
        for sets in ([{}, {41: 2}, {41: 2}, {41: 3}],
                     [{}, {41: 2}, {41: 2}, {41: 130}, {41: 130}]):
            with self.assertRaises(RuntimeError): self.make(sets).bind(FakeTorch())

    def test_empty_extra_or_replaced_pid_fails_after_binding(self):
        for processes in ({}, {41: 2, 42: 2}, {42: 2}):
            binding = self.make([{}, {41: 2}, {41: 2}, {41: 130}, {41: 2}, {41: 2}, processes])
            binding.bind(FakeTorch())
            with self.assertRaises(RuntimeError): binding.assert_exclusive()

    def test_local_identity_and_uuid_change_rejected(self):
        binding = self.make()
        binding.bind(FakeTorch())
        binding.identity = lambda: dict(pid=1, start_ticks='different', pid_namespace='pid:[3]')
        with self.assertRaises(RuntimeError): binding.assert_exclusive()
        binding = self.make()
        binding.bind(FakeTorch())
        binding.scanner = lambda gpu: dict(uuid='GPU-other', processes={41: 2})
        with self.assertRaises(RuntimeError): binding.assert_exclusive()

    def test_cuda_uuid_or_initialization_error_never_binds(self):
        torch = copy.deepcopy(FakeTorch())
        with patch.object(torch.cuda, 'init', side_effect=RuntimeError('CUDA init failed')):
            binding = self.make()
            with self.assertRaises(RuntimeError): binding.bind(torch)
            self.assertIsNone(binding.pid)
        torch = copy.deepcopy(FakeTorch())
        with patch.object(torch.cuda, 'get_device_properties', return_value=type('P', (), {'uuid': 'GPU-other'})()):
            with self.assertRaises(RuntimeError): self.make().bind(torch)

    def test_unbound_checker_fails(self):
        with self.assertRaises(RuntimeError): self.make().assert_exclusive()


class ProvenanceTests(unittest.TestCase):
    def test_formal_stage_has_no_cli_route(self):
        with patch('sys.argv', ['benchmark_revision.py', 'formal', '--revision', '/unused']):
            with self.assertRaises(SystemExit): revision.main()

    def test_prepare_requires_new_directory(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(revision, 'verify_frozen', return_value={}):
            with self.assertRaises(ValueError): revision.prepare(directory, directory)

    def test_nonempty_formal_artifact_view_preserves_paths_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            parent, view = Path(directory)/'parent', Path(directory)/'view'
            episode = parent/'episodes'/'condition'/'episode'
            episode.mkdir(parents=True)
            view.mkdir()
            (view/'episodes').symlink_to(parent/'episodes', target_is_directory=True)
            atomic_json(episode/'telemetry.jsonl', dict(eef_pos=[0, 0, 0]))
            atomic_json(episode/'video.mp4', dict(fixture='nonempty video path'))
            atomic_json(episode/'validated.json', dict(valid=True, artifacts={
                name: sha(episode/name) for name in ('telemetry.jsonl', 'video.mp4')}))
            def verified_audit(root, strict, check_outputs):
                validation = json.loads((episode/'validated.json').read_text())
                for name, expected in validation['artifacts'].items():
                    self.assertEqual(sha(episode/name), expected)
                return {'complete': True}, {'episode': {'directory': str(episode.resolve())}}
            module = types.SimpleNamespace(audit=verified_audit)
            install_analysis_route(module, view, parent)
            _, valid = module.audit(view)
            alias = Path(valid['episode']['directory'])
            for name in ('validated.json', 'telemetry.jsonl', 'video.mp4'):
                routed = alias/name
                self.assertTrue(str(routed.relative_to(view)).startswith('episodes/'))
                self.assertEqual(routed.resolve(), (episode/name).resolve())
                self.assertEqual(sha(routed), sha(episode/name))


if __name__ == '__main__':
    unittest.main()

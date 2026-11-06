"""CPU-only LRU state, embedding refresh and eviction cleanup tests."""
from collections import OrderedDict
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
from common import atomic_json, sha
from pi0_prompt_cache_revision import compare_gate, reserve_gate

from pi0_prompt_cache import PromptLengthCache


class Backend:
    def __init__(self):
        self.embeddings = {('suite', 'A'): [1]*10, ('suite', 'B'): [2]*14,
                           ('suite', 'same_length_other_text'): [3]*10, ('suite', 'C'): [4]*21}
        self.checkpoint = {'immutable': object()}
        self.runtime, self.length = None, None
        self.graphs, self.schedules, self.parity_lengths = OrderedDict(), {}, set()
        self.constructed = []
    def prepare(self, observation, meta, noise):
        embedding = self.embeddings['suite', meta['task_description']]
        if self.length != len(embedding):
            self.runtime = types.SimpleNamespace(embedding=None, mask=None)
            self.length = len(embedding)
            self.constructed.append(self.length)
        self.runtime.embedding = list(embedding)
        self.transformed = dict(observation)
        self.inputs = (observation['value'], noise)


class CacheTests(unittest.TestCase):
    def request(self, cache, name, value=1, noise=7):
        cache.prepare({'value': value}, {'suite': 'suite', 'task_description': name}, noise)
        return cache.last_event

    def test_abab_two_initial_misses_then_no_reconstruction(self):
        backend, schedules = Backend(), {}
        cache = PromptLengthCache(backend, schedules)
        events = [self.request(cache, name) for name in ('A', 'B', 'A', 'B')]
        self.assertEqual([event['hit'] for event in events], [False, False, True, True])
        self.assertEqual(backend.constructed, [10, 14])
        self.assertEqual(len(cache.slots), 2)
        self.assertIs(cache.slots[10].checkpoint, cache.slots[14].checkpoint)
        self.assertIsNot(cache.slots[10].runtime, cache.slots[14].runtime)

    def test_same_length_different_text_and_observation_are_refreshed(self):
        backend = Backend()
        cache = PromptLengthCache(backend, {})
        self.request(cache, 'A', value=1)
        runtime = backend.runtime
        event = self.request(cache, 'same_length_other_text', value=99, noise=42)
        self.assertTrue(event['hit'])
        self.assertIs(backend.runtime, runtime)
        self.assertEqual(backend.runtime.embedding, [3]*10)
        self.assertEqual(backend.inputs, (99, 42))
        self.assertEqual(backend.transformed, {'value': 99})

    def test_mask_graph_schedule_state_is_separate_per_runtime(self):
        backend = Backend()
        cache = PromptLengthCache(backend, {})
        self.request(cache, 'A')
        backend.graphs[('full', 2)] = 'Afull'
        backend.schedules[2] = 'Aschedule'
        backend.parity_lengths.add(10)
        self.request(cache, 'B')
        self.assertFalse(backend.graphs)
        self.assertFalse(backend.schedules)
        backend.graphs[('V14', 2)] = 'Bvariant'
        self.request(cache, 'A')
        self.assertEqual(backend.graphs, {('full', 2): 'Afull'})
        self.assertEqual(backend.schedules, {2: 'Aschedule'})
        self.assertEqual(backend.parity_lengths, {10})

    def test_eviction_removes_adapter_schedules_and_all_old_references_first(self):
        backend, schedules = Backend(), {}
        cache = PromptLengthCache(backend, schedules)
        self.request(cache, 'A')
        old = cache.slots[10]
        runtime_id = id(old.runtime)
        schedules[runtime_id, ('nodes',)] = object()
        self.request(cache, 'B')
        self.request(cache, 'C')
        self.assertEqual(list(cache.slots), [14, 21])
        self.assertNotIn((runtime_id, ('nodes',)), schedules)
        self.assertIsNone(old.runtime)
        self.assertFalse(old.graphs)
        self.assertFalse(old.schedules)
        # A recycled runtime id cannot find the prior runtime's adapter entry.
        self.assertFalse(any(key[0] == runtime_id for key in schedules))

    def test_installed_after_allocation_or_capacity_other_than_two_rejected(self):
        with self.assertRaises(ValueError): PromptLengthCache(Backend(), {}, capacity=3)
        backend = Backend()
        backend.runtime = object()
        with self.assertRaises(ValueError): PromptLengthCache(backend, {})


class GateTests(unittest.TestCase):
    def test_sixteen_call_budget_is_cumulative_and_never_modifies_signed_base(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            budget = attempt/'readiness/pi0/diagnostic_budget.json'
            atomic_json(budget, dict(smoke_reserved=7))
            atomic_json(budget.with_name('throughput_diagnostic_budget.json'), dict(entries=[dict(reserved_predictions=8)]))
            atomic_json(attempt/'parity/pi0.json', dict(reserved_calls=100))
            original = sha(budget)
            manifest = dict(prior_diagnostics={'pi0': {'conservative_predictions': 100}})
            revisions = []
            for index in range(3):
                revision = attempt/f'revision{index}'
                atomic_json(revision/'revision_manifest.json', dict(source_attempt=str(attempt), index=index))
                revisions.append(revision)
            self.assertEqual(reserve_gate(revisions[0], manifest, {'reserved_calls': 100}), 231)
            with self.assertRaises(RuntimeError): reserve_gate(revisions[0], manifest, {'reserved_calls': 100})
            self.assertEqual(reserve_gate(revisions[1], manifest, {'reserved_calls': 100}), 247)
            with self.assertRaises(RuntimeError): reserve_gate(revisions[2], manifest, {'reserved_calls': 100})
            self.assertEqual(original, sha(budget))

    def test_abab_exact_gate_rejects_hit_startup_or_raw_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, cached = root/'reference', root/'cached'
            requests = [dict(index=index, prompt_length=10 if index % 2 == 0 else 14) for index in range(8)]
            atomic_json(root/'revision_manifest.json', {'requests': requests})
            for index in range(8):
                for base in (reference, cached):
                    (base/'outputs').mkdir(parents=True, exist_ok=True)
                    noise, prediction = base/f'{index}_noise.npz', base/f'{index}_prediction.npz'
                    np.savez(noise, initial_noise=np.ones((2, 7), dtype=np.float32))
                    np.savez(prediction, raw_prediction=np.ones((2, 7)), actions=np.ones((2, 7)))
                    row = dict(actions=[[1.]*7]*2, input_hash='input', noise_hash='noise', raw_prediction_hash='raw',
                        seed=index//4, loaded_code_hashes={}, noise_path=str(noise), server_prediction_path=str(prediction),
                        audit=dict(NFE=2, actual_NFE=2, skip=[], schedule={'kind': 'flow'}, cache_key='same'),
                        execution_prompt_cache=dict(prompt_length=requests[index]['prompt_length'], resident_lengths=[10, 14],
                            hit=index>=2, startup_forward_calls=0 if index>=2 else 4, startup_action_NFE=0 if index>=2 else 40, evictions=[]))
                    atomic_json(base/'outputs'/f'{index}.json', row)
            self.assertEqual(len(compare_gate(root, cached, reference)), 8)
            import json
            path = cached/'outputs/4.json'
            row = json.loads(path.read_text())
            row['execution_prompt_cache']['startup_forward_calls'] = 4
            atomic_json(path, row)
            with self.assertRaises(ValueError): compare_gate(root, cached, reference)


if __name__ == '__main__':
    unittest.main()

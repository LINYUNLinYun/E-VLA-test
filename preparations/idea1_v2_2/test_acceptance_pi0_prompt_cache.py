"""Independent CPU checks of bounded runtime lifetime, not numerical GPU parity."""
from collections import OrderedDict
import unittest
import weakref

from pi0_prompt_cache import PromptLengthCache


class Runtime:
    pass


class Backend:
    def __init__(self):
        self.embeddings = {('s', text): list(range(length)) for text, length in
                           [('A', 10), ('B', 14), ('C', 21), ('A-other', 10)]}
        self.runtime = self.length = None
        self.graphs, self.schedules, self.parity_lengths = OrderedDict(), {}, set()
        self.alive, self.allocations, self.prepares = weakref.WeakSet(), [], []

    def prepare(self, obs, meta, noise):
        text = meta['task_description']
        length = len(self.embeddings['s', text])
        self.prepares.append((text, obs, noise))
        if self.runtime is None:
            if len(self.alive) >= 2:
                raise AssertionError('Third runtime allocated before eviction')
            self.runtime = Runtime()
            self.alive.add(self.runtime)
            self.allocations.append(length)
            self.length = length
        self.runtime.prompt = text
        self.inputs, self.transformed = noise, dict(obs)


class IndependentPromptCacheTests(unittest.TestCase):
    def prepare(self, cache, text, value):
        cache.prepare({'value': value}, {'suite': 's', 'task_description': text}, value+10)

    def test_eviction_precedes_allocation_and_lru_order_is_exact(self):
        backend = Backend()
        cache = PromptLengthCache(backend, {})
        observed = []
        for index, text in enumerate(('A', 'B', 'C', 'B', 'A')):
            self.prepare(cache, text, index)
            observed.append((cache.last_event['hit'], list(cache.slots)))
            self.assertLessEqual(len(backend.alive), 2)
        self.assertEqual(observed, [(False, [10]), (False, [10, 14]),
            (False, [14, 21]), (True, [21, 14]), (False, [14, 10])])
        self.assertEqual(backend.allocations, [10, 14, 21, 10])

    def test_hit_always_runs_original_prepare_with_fresh_text_input_and_noise(self):
        backend = Backend()
        cache = PromptLengthCache(backend, {})
        self.prepare(cache, 'A', 1)
        runtime = backend.runtime
        self.prepare(cache, 'A-other', 2)
        self.assertIs(backend.runtime, runtime)
        self.assertEqual(backend.runtime.prompt, 'A-other')
        self.assertEqual(backend.transformed, {'value': 2})
        self.assertEqual(backend.inputs, 12)
        self.assertEqual(len(backend.prepares), 2)
        self.assertEqual(backend.allocations, [10])

    def test_schedule_cleanup_removes_only_evicted_runtime(self):
        backend, schedules = Backend(), {}
        cache = PromptLengthCache(backend, schedules)
        self.prepare(cache, 'A', 1)
        evicted = id(backend.runtime)
        schedules[evicted, ('native10',)] = 'old'
        self.prepare(cache, 'B', 2)
        retained = id(backend.runtime)
        schedules[retained, ('u50',)] = 'keep'
        self.prepare(cache, 'C', 3)
        self.assertEqual(schedules, {(retained, ('u50',)): 'keep'})


if __name__ == '__main__':
    unittest.main()

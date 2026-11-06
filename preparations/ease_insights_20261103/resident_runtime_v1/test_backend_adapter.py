"""CPU integration fixtures for resident ownership and episode handoff."""
import unittest

from .backend_adapter import ResidentBackendAdapter
from .prompt_runtime import PromptRuntimeOwner
from .test_prompt_runtime import Operations, Tensor, runtime


class Embedding(Tensor):
    def __len__(self):
        return self.shape[0]


class Backend:
    """Match the pinned backend's destructive length-change branch, on CPU."""
    def __init__(self):
        self.runtime = self.length = None
        self.graphs, self.schedules, self.parity_lengths = {}, {}, set()
        self.embeddings = {('suite', text): Embedding((length, 2048), token=text)
                           for text, length in (('short', 5), ('long', 9), ('same_length', 5))}
        self.model_constructions = self.destructive_replacements = self.prepares = 0

    def prepare(self, observation, meta, noise):
        embedding = self.embeddings[meta['suite'], meta['task_description']]
        if self.length != len(embedding):
            if self.runtime is not None:
                self.destructive_replacements += 1
            self.runtime = runtime(len(embedding))
            self.length = len(embedding)
            self.model_constructions += 1
            self.graphs.clear()
            self.schedules.clear()
        self.runtime.weights['language_embeds'].token = embedding.token
        self.last_observation, self.last_noise = observation, noise
        self.prepares += 1


class Retainer:
    def __init__(self):
        self.held, self.calls = False, 0

    def retain(self):
        self.held = True
        self.calls += 1

    def assert_held(self):
        if not self.held:
            raise RuntimeError('Synthetic floor no longer held')


class AdapterContracts(unittest.TestCase):
    def setup_adapter(self):
        wrapped, ops, retainer = Backend(), Operations(), Retainer()
        adapter = ResidentBackendAdapter(wrapped, build_retainer=lambda owner: retainer,
            build_owner=lambda target: PromptRuntimeOwner(target, operations=ops))
        return wrapped, ops, retainer, adapter

    def prepare_case(self, adapter, prompt):
        adapter.begin_episode(pending_actions=0)
        observation, noise = object(), object()
        adapter.prepare(observation, {'suite': 'suite', 'task_description': prompt}, noise)
        adapter.finish_episode(pending_actions=0)
        return observation, noise

    def test_short_long_short_keeps_one_model_and_floor_through_episode_handoffs(self):
        wrapped, ops, retainer, adapter = self.setup_adapter()
        self.prepare_case(adapter, 'short')
        first_runtime = wrapped.runtime
        schedule = object()
        wrapped.schedules['D2'] = schedule
        for prompt in ('long', 'short'):
            observation, noise = self.prepare_case(adapter, prompt)
            self.assertIs(wrapped.runtime, first_runtime)
            self.assertIs(wrapped.schedules['D2'], schedule)
            self.assertIs(wrapped.last_observation, observation)
            self.assertIs(wrapped.last_noise, noise)
            self.assertTrue(retainer.held)
        self.assertEqual(wrapped.model_constructions, 1)
        self.assertEqual(wrapped.destructive_replacements, 0)
        self.assertEqual(adapter.model_initializations, 1)
        self.assertEqual(retainer.calls, 1)
        self.assertEqual(ops.captures, 2)

    def test_same_length_prompt_uses_original_embedding_update(self):
        wrapped, ops, retainer, adapter = self.setup_adapter()
        self.prepare_case(adapter, 'short')
        language = wrapped.runtime.weights['language_embeds']
        self.prepare_case(adapter, 'same_length')
        self.assertIs(wrapped.runtime.weights['language_embeds'], language)
        self.assertEqual(language.token, 'same_length')
        self.assertEqual(ops.captures, 0)
        self.assertTrue(retainer.held)

    def test_pending_proposal_blocks_both_episode_boundaries(self):
        _, _, _, adapter = self.setup_adapter()
        with self.assertRaises(ValueError):
            adapter.begin_episode(pending_actions=1)
        adapter.begin_episode(pending_actions=0)
        with self.assertRaises(ValueError):
            adapter.finish_episode(pending_actions=1)

    def test_mid_episode_prompt_change_rejected_without_allocating_second_model(self):
        wrapped, _, _, adapter = self.setup_adapter()
        adapter.begin_episode(pending_actions=0)
        adapter.prepare(None, {'suite': 'suite', 'task_description': 'short'}, None)
        with self.assertRaises(ValueError):
            adapter.prepare(None, {'suite': 'suite', 'task_description': 'long'}, None)
        self.assertEqual(wrapped.model_constructions, 1)

    def test_external_runtime_replacement_is_a_terminal_adapter_fault(self):
        wrapped, _, _, adapter = self.setup_adapter()
        self.prepare_case(adapter, 'short')
        wrapped.runtime = runtime(5)
        with self.assertRaises(RuntimeError):
            adapter.begin_episode(pending_actions=0)
        with self.assertRaises(RuntimeError):
            adapter.begin_episode(pending_actions=0)

    def test_lost_floor_blocks_the_next_episode(self):
        _, _, retainer, adapter = self.setup_adapter()
        self.prepare_case(adapter, 'short')
        retainer.held = False
        with self.assertRaises(RuntimeError):
            adapter.begin_episode(pending_actions=0)

    def test_installation_after_model_initialization_is_rejected(self):
        wrapped = Backend()
        wrapped.prepare(None, {'suite': 'suite', 'task_description': 'short'}, None)
        with self.assertRaises(ValueError):
            ResidentBackendAdapter(wrapped, build_retainer=lambda owner: Retainer())


if __name__ == '__main__':
    unittest.main(verbosity=2)

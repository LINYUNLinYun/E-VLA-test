"""CPU-only behavioral tests for budget changes across real policy boundaries.

The fake policy owns proposals, Full histories, and a held Fast mask. Budget
changes must respect those independent states without reconstructing the policy
or clearing its selector. Candidate artifacts are built by the real CPU builder.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .budget import prepare_budget
from .runtime import BudgetedPolicy


def measured_table():
    return dict(measured=True, units='ms', n_layers=3,
        layer_costs=[dict(layer=i, attn_saved_ms=10., ffn_saved_ms=2. if i == 2 else 1.,
                         af_saved_ms=12. if i == 2 else 11.,
                         attn_noise_ms=0., ffn_noise_ms=0., af_noise_ms=0.) for i in range(1, 4)])


class FakeBackend:
    def __init__(self):
        self.calls = []

    def record(self, **call):
        self.calls.append(call)


class FakeSelector:
    def __init__(self):
        self.clear_calls = 0

    def clear(self):
        self.clear_calls += 1


class FakePolicy:
    """Stateful execution contract; deliberately independent of the wrapper."""
    def __init__(self, backend, budgets):
        self.b = backend
        self.budgets = budgets
        self.selector = FakeSelector()
        self.config = {}
        self.configure_calls = []
        self.acknowledgements = []
        self.predict_arguments = []
        self.fail_next_configure = False
        self.fail_next_predict = False
        self.failed_predict_budget_ids = []
        self.reset()

    def reset(self):
        self.pending = None
        self.round = self.step = 0
        self.history = []
        self.selected_ffn = None

    def configure(self, config):
        if self.fail_next_configure:
            self.fail_next_configure = False
            raise RuntimeError('fake native configuration refused before state mutation')
        self.selector.clear()
        self.config = dict(config)
        self.reset()
        self.configure_calls.append(deepcopy(config))
        return dict(configured=True)

    def predict(self, obs, instruction, suite, seed):
        if self.pending is not None:
            raise RuntimeError('Fake policy requires execution acknowledgement')
        if self.fail_next_predict:
            self.fail_next_predict = False
            self.failed_predict_budget_ids.append(self.config['budget'])
            raise RuntimeError('fake native prediction failed before progress')
        self.predict_arguments.append((obs, instruction, suite, seed))
        full = self.round % 5 == 0
        budget = self.budgets[self.config['budget']]
        target = budget['anchor_saved_ms']
        before = list(self.history)
        if full:
            # Both masks are legal for the real three-layer candidate fixture.
            self.selected_ffn = [] if target == 0 else ([1] if target < 1.5 else [1, 3])
            scoring = 'comp_error' if len(before) == 2 else 'warm_original'
            self.history = (self.history + [dict(step=self.step, residuals=object())])[-2:]
        else:
            if self.selected_ffn is None:
                raise RuntimeError('Fast execution needs a preceding Full mask')
            scoring = 'held'
        self.b.record(round=self.round, step=self.step, full=full,
            target_saved_ms=target, budget_id=budget['budget_id'], budget_hash=budget['budget_hash'],
            history_before=before, selector=self.selector, scoring=scoring,
            actual_ffn=[] if full else list(self.selected_ffn))
        actions = np.full((50, 7), (self.round + target) / 100., dtype=np.float32)
        result = dict(actions=actions, raw=np.zeros((50, 32), np.float32), execute_length=5,
            round=self.round, step=self.step,
            route=dict(path='full' if full else 'fast', budget=self.config['budget'],
                       selected_attn=[], selected_ffn=list(self.selected_ffn),
                       selected_skip=list(self.selected_ffn), scoring_mode=scoring))
        self.pending = result
        self.round += 1
        return result

    def acknowledge(self, executed, quaternions=None):
        if self.pending is None:
            raise RuntimeError('No proposal to acknowledge')
        if not 0 < len(executed) <= 5 or not np.array_equal(executed, self.pending['actions'][:len(executed)]):
            raise ValueError('Execution differs from the proposal')
        self.acknowledgements.append((executed, quaternions))
        self.step += len(executed)
        self.pending = None
        return dict(acknowledged=len(executed))


class RuntimeBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='budget_runtime_', dir='/tmp')
        self.folder = Path(self.temp.name)
        self.cost_path = self.folder / 'costs.json'
        self.cost_path.write_text(json.dumps(measured_table()))
        self.low = prepare_budget(self.cost_path, self.folder / 'artifacts', saved_ms=1.,
                                  tolerance_ms=0., reference_vlm_ms=20.)
        self.high = prepare_budget(self.cost_path, self.folder / 'artifacts', saved_fraction=.1,
                                   tolerance_ms=0., reference_vlm_ms=20.)
        self.backend = FakeBackend()
        self.instances = []

    def tearDown(self):
        self.temp.cleanup()

    def factory(self, backend, budgets):
        instance = FakePolicy(backend, budgets)
        self.instances.append(instance)
        return instance

    def runtime(self, **kwargs):
        return BudgetedPolicy(self.backend, self.low, policy_factory=self.factory, **kwargs)

    def predict(self, policy, seed=7):
        return policy.predict({'frame': seed}, 'fixed instruction', 'libero_spatial', seed)

    def run_acknowledged(self, policy, seed=7):
        result = self.predict(policy, seed)
        policy.acknowledge(result['actions'][:5], None)
        return result

    def test_fast_cycle_keeps_old_mask_and_budget_then_next_full_switches(self):
        policy = self.runtime(cost_weighted=True)
        inner = self.instances[0]
        self.assertEqual(len(inner.configure_calls), 1)
        selector = inner.selector
        clear_calls = selector.clear_calls
        self.run_acknowledged(policy)  # Full0 creates the first real history bank.
        old_history = inner.history[0]
        status = policy.set_budget(budget_artifact=self.high['artifact_path'])
        self.assertEqual(status['applied_at'], 'next_full')
        self.assertTrue(status['queued'])
        for round_index in range(1, 5):
            result = self.run_acknowledged(policy, seed=round_index)
            self.assertEqual(result['route']['path'], 'fast')
            self.assertEqual(result['route']['selected_ffn'], [1])
            self.assertEqual(result['route']['budget_target_saved_ms'], 1.)
            self.assertEqual(result['route']['budget_hash'], self.low['budget_hash'])
            self.assertFalse(result['route']['budget_change_applied'])
            self.assertEqual(self.backend.calls[-1]['target_saved_ms'], 1.)
        result = self.predict(policy, seed=5)
        self.assertEqual(result['round'], 5)
        self.assertEqual(result['route']['path'], 'full')
        self.assertEqual(result['route']['selected_ffn'], [1, 3])
        self.assertEqual(result['route']['budget_target_saved_ms'], 2.)
        self.assertEqual(result['route']['budget_target_saved_fraction'], .1)
        self.assertEqual(result['route']['budget_reference_vlm_ms'], 20.)
        self.assertTrue(result['route']['budget_change_applied'])
        self.assertEqual(result['route']['budget_hash'], self.high['budget_hash'])
        self.assertIsNone(policy.queued_budget)
        self.assertIs(self.backend.calls[-1]['history_before'][0], old_history)
        self.assertIs(inner.selector, selector)
        self.assertEqual(selector.clear_calls, clear_calls)
        self.assertEqual(len(inner.configure_calls), 1)
        self.assertEqual(len(self.instances), 1)
        self.assertTrue(inner.config['cost_weighted'])

    def test_unacknowledged_last_fast_blocks_switch_before_next_full(self):
        policy = self.runtime()
        inner = self.instances[0]
        for index in range(4):
            self.run_acknowledged(policy, index)  # Full0 plus Fast1..3.
        last_fast = self.predict(policy, 4)  # round is now 5, but this proposal is pending.
        self.assertEqual(inner.round, 5)
        policy.set_budget(budget_artifact=self.high)
        calls = len(self.backend.calls)
        with self.assertRaisesRegex(RuntimeError, 'Unacknowledged'):
            self.predict(policy, 5)
        self.assertEqual(len(self.backend.calls), calls)
        self.assertEqual(policy.current_budget['budget_hash'], self.low['budget_hash'])
        self.assertEqual(policy.queued_budget['budget_hash'], self.high['budget_hash'])
        self.assertEqual(inner.config['budget'], self.low['budget_id'])
        policy.acknowledge(last_fast['actions'][:5], None)
        switched = self.predict(policy, 5)
        self.assertTrue(switched['route']['budget_change_applied'])
        self.assertEqual(switched['route']['budget_target_saved_ms'], 2.)

    def test_third_full_switch_preserves_two_bank_error_history_and_selector(self):
        policy = self.runtime()
        inner = self.instances[0]
        for index in range(10):
            self.run_acknowledged(policy, index)
        before = list(inner.history)
        self.assertEqual([bank['step'] for bank in before], [0, 25])
        selector, clears = inner.selector, inner.selector.clear_calls
        policy.set_budget(budget_artifact=self.high)
        result = self.predict(policy, 10)
        observed = self.backend.calls[-1]
        self.assertEqual(result['step'], 50)
        self.assertEqual(observed['scoring'], 'comp_error')
        self.assertEqual([bank['step'] for bank in observed['history_before']], [0, 25])
        self.assertTrue(all(a is b for a, b in zip(observed['history_before'], before)))
        self.assertEqual([bank['step'] for bank in inner.history], [25, 50])
        self.assertIs(inner.history[0]['residuals'], before[-1]['residuals'])
        self.assertIs(inner.selector, selector)
        self.assertEqual(selector.clear_calls, clears)

    def assert_failed_full_preserves_budget_queue(self, fail_call, error_type):
        policy = self.runtime()
        inner = self.instances[0]
        for index in range(10):
            self.run_acknowledged(policy, index)
        before = list(inner.history)
        selector, clears = inner.selector, inner.selector.clear_calls
        configure_count, calls = len(inner.configure_calls), len(self.backend.calls)
        old_config = deepcopy(inner.config)
        old_mask = list(inner.selected_ffn)
        policy.set_budget(budget_artifact=self.high)

        with self.assertRaises(error_type):
            fail_call(policy, inner)

        self.assertEqual((inner.round, inner.step), (10, 50))
        self.assertIsNone(inner.pending)
        self.assertEqual(len(self.backend.calls), calls)
        self.assertEqual(policy.current_budget['budget_hash'], self.low['budget_hash'])
        self.assertEqual(policy.queued_budget['budget_hash'], self.high['budget_hash'])
        self.assertEqual(inner.config, old_config)
        self.assertEqual(list(inner.budgets), [self.low['budget_id']])
        self.assertEqual(inner.selected_ffn, old_mask)
        self.assertEqual(len(inner.history), 2)
        self.assertTrue(all(a is b for a, b in zip(inner.history, before)))
        self.assertIs(inner.selector, selector)
        self.assertEqual(selector.clear_calls, clears)
        self.assertEqual(len(inner.configure_calls), configure_count)

        success = self.predict(policy, 10)
        self.assertEqual(success['route']['path'], 'full')
        self.assertTrue(success['route']['budget_change_applied'])
        self.assertEqual(success['route']['budget_hash'], self.high['budget_hash'])
        self.assertEqual(success['route']['budget_target_saved_ms'], 2.)
        self.assertIsNone(policy.queued_budget)
        observed = self.backend.calls[-1]
        self.assertEqual(observed['scoring'], 'comp_error')
        self.assertTrue(all(a is b for a, b in zip(observed['history_before'], before)))
        self.assertEqual([bank['step'] for bank in inner.history], [25, 50])
        self.assertIs(inner.history[0], before[-1])
        self.assertIs(inner.selector, selector)
        self.assertEqual(selector.clear_calls, clears)
        self.assertEqual(len(inner.configure_calls), configure_count)
        policy.acknowledge(success['actions'][:5], None)
        self.assertFalse(self.predict(policy, 11)['route']['budget_change_applied'])
        return inner

    def test_missing_full_predict_argument_preserves_queue_for_successful_retry(self):
        inner = self.assert_failed_full_preserves_budget_queue(
            lambda policy, inner: policy.predict(obs={}, instruction='prompt', suite='suite'),
            TypeError)
        self.assertEqual(inner.failed_predict_budget_ids, [])

    def test_native_full_prediction_failure_preserves_queue_for_successful_retry(self):
        def fail(policy, inner):
            inner.fail_next_predict = True
            return self.predict(policy, 10)
        inner = self.assert_failed_full_preserves_budget_queue(fail, RuntimeError)
        # The failure happens after the newly queued budget reaches the policy,
        # so passing requires rollback rather than merely postponing all work.
        self.assertEqual(inner.failed_predict_budget_ids, [self.high['budget_id']])

    def test_configure_new_episode_adopts_queue_and_resets_histories(self):
        policy = self.runtime()
        inner = self.instances[0]
        for index in range(3):
            self.run_acknowledged(policy, index)
        self.assertTrue(inner.history)
        policy.set_budget(budget_artifact=self.high)
        configured = policy.configure(dict(id='deployment', episode_case={'suite': 's', 'task': 1, 'init': 2}, cost_weighted=True))
        self.assertEqual(configured, dict(configured=True))
        self.assertEqual(policy.current_budget['budget_hash'], self.high['budget_hash'])
        self.assertIsNone(policy.queued_budget)
        self.assertEqual(inner.history, [])
        self.assertEqual((inner.round, inner.step), (0, 0))
        self.assertIsNone(inner.pending)
        self.assertEqual(inner.config['id'], 'deployment')
        self.assertTrue(inner.config['cost_weighted'])
        self.assertEqual(inner.config['budget'], self.high['budget_id'])
        result = self.predict(policy)
        self.assertEqual(result['route']['budget_target_saved_ms'], 2.)
        self.assertFalse(result['route']['budget_change_applied'])  # Applied by explicit configure.
        self.assertEqual(self.backend.calls[-1]['scoring'], 'warm_original')

    def test_rejected_configuration_preserves_active_queue_and_execution_state(self):
        policy = self.runtime()
        self.run_acknowledged(policy)
        inner = self.instances[0]
        before = list(inner.history)
        config = deepcopy(inner.config)
        clears = inner.selector.clear_calls
        policy.set_budget(budget_artifact=self.high)
        for invalid in (dict(period=4), dict(family='block'), dict(model='other'), dict(mode='adaptive'),
                        dict(weights=[.2] * 5), dict(budget='unverified-budget'), dict(cost_weighted='false')):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                policy.configure(invalid)
            self.assertEqual(policy.current_budget['budget_hash'], self.low['budget_hash'])
            self.assertEqual(policy.queued_budget['budget_hash'], self.high['budget_hash'])
            self.assertEqual(inner.config, config)
            self.assertEqual(inner.round, 1)
            self.assertEqual(inner.selector.clear_calls, clears)
            self.assertTrue(all(a is b for a, b in zip(inner.history, before)))

    def test_inner_configuration_failure_rolls_back_budget_mapping(self):
        policy = self.runtime()
        self.run_acknowledged(policy)
        inner = self.instances[0]
        before_budgets, before_config = inner.budgets, deepcopy(inner.config)
        before = list(inner.history)
        policy.set_budget(budget_artifact=self.high)
        inner.fail_next_configure = True
        with self.assertRaisesRegex(RuntimeError, 'configuration refused'):
            policy.configure()
        self.assertIs(inner.budgets, before_budgets)
        self.assertEqual(inner.config, before_config)
        self.assertTrue(all(a is b for a, b in zip(inner.history, before)))
        self.assertEqual(policy.current_budget['budget_hash'], self.low['budget_hash'])
        self.assertEqual(policy.queued_budget['budget_hash'], self.high['budget_hash'])

    def test_predict_and_acknowledge_forward_the_execution_contract(self):
        policy = self.runtime()
        inner = self.instances[0]
        observation = {'frame': object()}
        result = policy.predict(obs=observation, instruction='prompt', suite='suite', seed=42)
        received = inner.predict_arguments[-1]
        self.assertIs(received[0], observation)
        self.assertEqual(received[1:], ('prompt', 'suite', 42))
        executed, orientations = result['actions'][:5], np.zeros((5, 4), np.float32)
        reply = policy.acknowledge(executed=executed, quaternions=orientations)
        self.assertEqual(reply, dict(acknowledged=5))
        self.assertIs(inner.acknowledgements[-1][0], executed)
        self.assertIs(inner.acknowledgements[-1][1], orientations)
        self.assertEqual(inner.step, 5)
        self.assertIsNone(inner.pending)

    def test_constructor_rejects_ambiguous_boolean_before_policy_creation(self):
        with self.assertRaisesRegex(ValueError, 'cost_weighted.*boolean'):
            self.runtime(cost_weighted='false')
        self.assertEqual(self.instances, [])
        self.assertEqual(self.backend.calls, [])

    def test_CPU_budget_specification_is_prepared_off_prediction_path(self):
        artifact_root = self.folder / 'queued_preparations'
        policy = self.runtime(artifact_root=artifact_root)
        inner = self.instances[0]
        status = policy.set_budget(saved_fraction=.1, reference_vlm_ms=20., tolerance_ms=0.)
        queued = policy.queued_budget
        self.assertEqual(status['applied_at'], 'next_full')
        self.assertEqual(queued['anchor_saved_ms'], 2.)
        self.assertEqual(queued['requested_saved_fraction'], .1)
        self.assertTrue(Path(queued['artifact_path']).is_relative_to(artifact_root))
        self.assertTrue(Path(queued['artifact_path']).exists())
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(inner.round, 0)
        self.assertEqual(len(inner.configure_calls), 1)
        result = self.predict(policy)
        self.assertTrue(result['route']['budget_change_applied'])
        self.assertEqual(result['route']['budget_target_saved_ms'], 2.)

    def test_verified_budget_dictionaries_are_not_mutable_control_channels(self):
        policy = self.runtime()
        policy.set_budget(budget_artifact=self.high)
        damaged = deepcopy(self.high)
        damaged['anchor_saved_ms'] = 99.
        with self.assertRaises(ValueError):
            policy.set_budget(budget_artifact=damaged)
        current_view, queued_view = policy.current_budget, policy.queued_budget
        current_view['anchor_saved_ms'] = 99.
        queued_view['anchor_saved_ms'] = 99.
        self.assertEqual(policy.current_budget['anchor_saved_ms'], 1.)
        self.assertEqual(policy.queued_budget['anchor_saved_ms'], 2.)
        with self.assertRaises(ValueError):
            policy.set_budget(budget_artifact=self.high, saved_ms=2.)
        self.assertEqual(policy.queued_budget['budget_hash'], self.high['budget_hash'])

    def test_different_hardware_calibration_is_rejected_without_losing_queue(self):
        policy = self.runtime()
        policy.set_budget(budget_artifact=self.high)
        alternate = measured_table()
        alternate['calibration_identity'] = 'a different hardware measurement'
        path = self.folder / 'other_costs.json'
        path.write_text(json.dumps(alternate))
        budget = prepare_budget(path, self.folder / 'other_budgets', saved_ms=1., tolerance_ms=0., reference_vlm_ms=20.)
        with self.assertRaises(ValueError):
            policy.set_budget(budget_artifact=budget)
        self.assertEqual(policy.current_budget['budget_hash'], self.low['budget_hash'])
        self.assertEqual(policy.queued_budget['budget_hash'], self.high['budget_hash'])


if __name__ == '__main__':
    unittest.main()

"""CPU regressions for transactional RPC client and prompt registration.

The policy factory is independent of budget artifacts and model execution.
These tests exercise ClientSessions.handle, the dispatcher used by serve().
"""
import io
import json
import unittest

from .server import ClientSessions


class FakePolicy:
    def __init__(self, backend, budget, *, cost_weighted=False, artifact_root=None):
        self.backend = backend
        self.budget = budget
        self.cost_weighted = cost_weighted
        self.artifact_root = artifact_root
        self.current_budget = dict(budget_id='initial')
        self.queued_budget = None
        self.pending = None
        self.closed = False
        self.configure_calls = []
        self.predict_calls = []
        self.ack_calls = []

    def configure(self, config=None):
        if config and config.get('invalid'):
            raise ValueError('invalid configuration')
        self.configure_calls.append(config)
        self.pending = None
        return dict(configured=True)

    def set_budget(self, **specification):
        self.queued_budget = dict(budget_id='queued')
        return dict(queued=True, applied_at='next_full', specification=specification)

    def predict(self, obs, instruction, suite, seed):
        if self.pending is not None:
            raise RuntimeError('Unacknowledged action')
        self.predict_calls.append((obs, instruction, suite, seed))
        result = dict(actions=[[float(seed)] * 7 for _ in range(5)], execute_length=5,
                      route=dict(path='full'))
        self.pending = result
        return result

    def acknowledge(self, executed, quaternions=None):
        if self.pending is None:
            raise ValueError('No proposal')
        if not 0 < len(executed) <= 5 or executed != self.pending['actions'][:len(executed)]:
            raise ValueError('Execution differs from proposal')
        self.ack_calls.append((executed, quaternions))
        self.pending = None
        return dict(acknowledged=len(executed))

    def close(self):
        self.closed = True
        self.pending = None


class PolicyFactory:
    def __init__(self):
        self.created = []

    def __call__(self, *args, **kwargs):
        policy = FakePolicy(*args, **kwargs)
        self.created.append(policy)
        return policy


class ClientSessionsTests(unittest.TestCase):
    def setUp(self):
        self.factory = PolicyFactory()
        self.backend, self.budget = object(), object()
        self.protocol = dict(service_mode='CPU fixture')
        self.events = io.StringIO()
        self.sessions = ClientSessions(self.backend, self.budget, self.protocol,
            actors=1, policy_factory=self.factory, events=self.events)

    def predict(self, client='client', instruction='task', seed=1, **overrides):
        request = dict(op='predict', client=client, obs=object(),
                       instruction=instruction, suite='suite', seed=seed)
        request.update(overrides)
        return self.sessions.handle(request)

    def test_unknown_operation_does_not_allocate_or_use_the_only_slot(self):
        with self.assertRaisesRegex(ValueError, 'Unknown RPC operation'):
            self.sessions.handle(dict(op='unknown', client='bad'))
        self.assertEqual(self.factory.created, [])
        self.assertEqual(self.sessions.policies, {})
        self.assertIsNone(self.sessions.group_prompt)
        response = self.sessions.handle(dict(op='configure', client='good', config={}))
        self.assertEqual(response, dict(configured=True))
        self.assertEqual(set(self.sessions.policies), {'good'})

    def test_invalid_new_configuration_closes_policy_without_consuming_slot(self):
        with self.assertRaisesRegex(ValueError, 'invalid configuration'):
            self.sessions.handle(dict(op='configure', client='bad', config=dict(invalid=True)))
        self.assertEqual(self.sessions.policies, {})
        self.assertTrue(self.factory.created[0].closed)
        self.assertIsNone(self.sessions.group_prompt)
        self.sessions.handle(dict(op='configure', client='good', config={}))
        self.assertEqual(set(self.sessions.policies), {'good'})
        self.assertFalse(self.sessions.policies['good'].closed)

    def test_missing_observation_does_not_commit_client_or_first_prompt(self):
        with self.assertRaises(TypeError):
            self.sessions.handle(dict(op='predict', client='bad', instruction='typo', suite='suite', seed=1))
        self.assertEqual(self.sessions.policies, {})
        self.assertIsNone(self.sessions.group_prompt)
        self.assertTrue(self.factory.created[0].closed)
        self.predict(client='good', instruction='correct')
        self.assertEqual(self.sessions.group_prompt, 'correct')
        self.assertEqual(set(self.sessions.policies), {'good'})

    def test_failed_existing_client_call_preserves_the_registered_policy(self):
        self.sessions.handle(dict(op='configure', client='existing', config={}))
        policy = self.sessions.policies['existing']
        with self.assertRaises(TypeError):
            self.sessions.handle(dict(op='predict', client='existing', instruction='typo', suite='suite', seed=1))
        self.assertIs(self.sessions.policies['existing'], policy)
        self.assertFalse(policy.closed)
        self.assertIsNone(self.sessions.group_prompt)
        self.predict(client='existing', instruction='correct')
        self.assertEqual(self.sessions.group_prompt, 'correct')

    def test_valid_predict_commits_prompt_and_preserves_request_arguments(self):
        observation = object()
        request = dict(op='predict', client='client', obs=observation,
                       instruction='task', suite='suite', seed=17)
        reply = self.sessions.handle(request)
        self.assertNotIn('ok', reply)  # Transport, not the dispatcher, adds ok.
        self.assertEqual(self.sessions.group_prompt, 'task')
        policy = self.sessions.policies['client']
        self.assertIs(policy.predict_calls[0][0], observation)
        self.assertEqual(policy.predict_calls[0][1:], ('task', 'suite', 17))
        self.assertEqual((request['op'], request['client']), ('predict', 'client'))
        executed, orientations = reply['actions'][:3], [[0, 0, 0, 1]] * 3
        result = self.sessions.handle(dict(op='ack', client='client', executed=executed, quaternions=orientations))
        self.assertEqual(result, dict(acknowledged=3))
        self.assertIs(policy.ack_calls[0][0], executed)
        self.assertIs(policy.ack_calls[0][1], orientations)
        self.assertIsNone(policy.pending)

    def test_different_prompt_is_rejected_without_registering_second_client(self):
        self.sessions.actors = 2
        self.predict(client='first', instruction='task')
        first = self.sessions.policies['first']
        with self.assertRaisesRegex(ValueError, 'share their task/prompt'):
            self.predict(client='different', instruction='another task')
        self.assertEqual(set(self.sessions.policies), {'first'})
        self.assertEqual(self.sessions.group_prompt, 'task')
        self.assertIsNotNone(first.pending)
        # If the dispatcher allocated the rejected policy before validation,
        # the failed object must still be closed.
        for policy in self.factory.created:
            if policy is not first:
                self.assertTrue(policy.closed)

    def test_two_same_task_clients_have_independent_pending_actions(self):
        self.sessions.actors = 2
        a = self.predict(client='a', seed=1)
        b = self.predict(client='b', seed=2)
        pa, pb = self.sessions.policies['a'], self.sessions.policies['b']
        self.assertIsNot(pa, pb)
        self.assertNotEqual(a['actions'], b['actions'])
        self.sessions.handle(dict(op='ack', client='a', executed=a['actions']))
        self.assertIsNone(pa.pending)
        self.assertIsNotNone(pb.pending)
        self.assertEqual(self.sessions.group_prompt, 'task')

    def test_reset_all_is_atomic_while_any_client_has_pending_actions(self):
        self.sessions.actors = 2
        reply = self.predict(client='pending')
        self.sessions.handle(dict(op='configure', client='idle', config={}))
        before = dict(self.sessions.policies)
        with self.assertRaisesRegex(RuntimeError, 'Unacknowledged action'):
            self.sessions.handle(dict(op='reset_all'))
        self.assertEqual(self.sessions.policies, before)
        self.assertEqual(self.sessions.group_prompt, 'task')
        self.assertTrue(all(not policy.closed for policy in before.values()))
        self.sessions.handle(dict(op='ack', client='pending', executed=reply['actions']))
        self.assertEqual(self.sessions.handle(dict(op='reset_all')), {})
        self.assertEqual(self.sessions.policies, {})
        self.assertIsNone(self.sessions.group_prompt)
        self.assertTrue(all(policy.closed for policy in before.values()))

    def test_capacity_rejection_preserves_the_existing_client(self):
        self.sessions.handle(dict(op='configure', client='a', config={}))
        existing = self.sessions.policies['a']
        with self.assertRaisesRegex(RuntimeError, 'Client count exceeds'):
            self.sessions.handle(dict(op='configure', client='b', config={}))
        self.assertEqual(set(self.sessions.policies), {'a'})
        self.assertIs(self.sessions.policies['a'], existing)
        self.assertEqual(len(self.factory.created), 1)
        self.assertFalse(existing.closed)

    def test_health_and_budget_event_forwarding_need_no_model_or_artifact(self):
        initial = self.sessions.handle(dict(op='health'))
        self.assertEqual(initial, dict(protocol=self.protocol, clients={}))
        self.assertEqual(self.factory.created, [])
        response = self.sessions.handle(dict(op='set_budget', client='client', saved_ms=2., tolerance_ms=.1))
        self.assertTrue(response['queued'])
        event = json.loads(self.events.getvalue())
        self.assertEqual(event['client'], 'client')
        self.assertEqual(event['specification'], dict(saved_ms=2., tolerance_ms=.1))
        self.assertIsInstance(event['time'], float)
        health = self.sessions.handle(dict(op='health'))
        self.assertEqual(health['clients'], dict(client=dict(current_budget='initial', queued_budget='queued')))
        self.assertIsNone(self.sessions.group_prompt)
        self.assertIsNone(self.sessions.policies['client'].pending)


if __name__ == '__main__':
    unittest.main()

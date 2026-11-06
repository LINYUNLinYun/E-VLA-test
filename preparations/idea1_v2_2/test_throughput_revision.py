"""CPU-only tests for identity, budgets, gate equivalence and transaction boundaries."""
from contextlib import contextmanager
import fcntl
import multiprocessing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from common import Ledger, atomic_json, sha
import throughput_revision as revision
import throughput_revision_handoff as handoff
from throughput_revision_client import pilot_authorization


def reserve_rows(root, queue, barrier):
    ledger = Ledger(root)
    barrier.wait()
    completed = []
    for index in range(3):
        row = dict(key=f'key{index}', condition='condition')
        token = ledger.reserve(row, require_artifacts=True)
        if token is None:
            continue
        directory = Path(root)/'episodes'/token
        atomic_json(directory/'artifact.json', dict(index=index))
        atomic_json(directory/'validated.json', dict(valid=True, episode_key=row['key'], success=True,
            artifacts={'artifact.json': sha(directory/'artifact.json')}))
        ledger.update(token, directory=str(directory), status='valid', success=True,
                      validated_sha256=sha(directory/'validated.json'))
        completed.append(row['key'])
    queue.put(completed)


class BudgetTests(unittest.TestCase):
    def test_independent_budget_accumulates_and_original_is_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            path = attempt/'readiness'/'pi0'/'diagnostic_budget.json'
            atomic_json(path, dict(smoke_reserved=2, entries=[]))
            atomic_json(attempt/'parity'/'pi0.json', dict(reserved_calls=10))
            before = sha(path)
            one, two = attempt/'revision1', attempt/'revision2'
            atomic_json(one/'revision_manifest.json', {'revision': 1})
            atomic_json(two/'revision_manifest.json', {'revision': 2})
            manifest = dict(prior_diagnostics={'pi0': {'conservative_predictions': 3}})
            self.assertEqual(revision.reserve_gate(attempt, one, 'pi0', manifest, {'reserved_calls': 10}), 23)
            with self.assertRaises(RuntimeError): revision.reserve_gate(attempt, one, 'pi0', manifest, {'reserved_calls': 10})
            self.assertEqual(revision.reserve_gate(attempt, two, 'pi0', manifest, {'reserved_calls': 10}), 31)
            self.assertEqual(before, sha(path))
            third = attempt/'revision3'
            atomic_json(third/'revision_manifest.json', {'revision': 3})
            with self.assertRaises(RuntimeError): revision.reserve_gate(attempt, third, 'pi0', manifest, {'reserved_calls': 240})

    def test_medium_acceptance_binds_phase_and_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root/'revision_manifest.json', dict(revision=1))
            atomic_json(root/'pi0.gate_ready.json', dict(exact=True))
            signature = dict(schema=1, accepted=True, reviewer_model='gpt-6.1-sol', reasoning_effort='medium',
                model='pi0', phase='gate', evidence=['independent review'],
                revision_manifest_sha256=sha(root/'revision_manifest.json'),
                phase_receipt_sha256=sha(root/'pi0.gate_ready.json'))
            atomic_json(root/'acceptance'/'pi0.gate.json', signature)
            revision.acceptance(root, 'pi0', 'gate', root/'pi0.gate_ready.json')
            atomic_json(root/'pi0.gate_ready.json', dict(exact=False))
            with self.assertRaises(ValueError): revision.acceptance(root, 'pi0', 'gate', root/'pi0.gate_ready.json')


class GateTests(unittest.TestCase):
    def test_raw_exact_gate_and_drift_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for phase in ('reference', 'interleaved'):
                for index in range(4):
                    base = root/phase
                    base.mkdir(exist_ok=True)
                    noise, raw = base/f'{index}_noise.npz', base/f'{index}_raw.npz'
                    np.savez(noise, initial_noise=np.ones((2, 7), dtype=np.float32))
                    np.savez(raw, raw_prediction=np.ones((2, 7), dtype=np.float32), actions=np.ones((2, 7), dtype=np.float32))
                    atomic_json(base/f'{index}.json', dict(input_hash='input', noise_hash='noise', raw_prediction_hash='raw',
                        seed=index, loaded_code_hashes={'scientific': 'hash'}, actions=[[1.]*7]*2,
                        audit=dict(NFE=2, actual_NFE=2, schedule={'kind': 'flow'}, skip=[],
                                   instrumentation_mode='instrumented_audit', cache_key='same'),
                        noise_path=str(noise), server_prediction_path=str(raw)))
            self.assertTrue(all(row['exact'] for row in revision.compare_gate(root)))
            np.savez(root/'interleaved'/'2_raw.npz', raw_prediction=np.zeros((2, 7)), actions=np.ones((2, 7)))
            with self.assertRaises(ValueError): revision.compare_gate(root)


class LedgerTests(unittest.TestCase):
    def test_pilot_session_and_exact_candidate_are_required_for_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root/'revision_manifest.json', dict(revision=1))
            session = root/'session1'
            revision.claim_pilot(root, 'pi0', session)
            with self.assertRaises(RuntimeError): revision.claim_pilot(root, 'pi0', root/'session2')
            candidate = root/'episodes'/'candidate'
            ready = dict(session=str(session), results=[
                dict(slot=0, token='token0', key='key0', candidate_directory=str(candidate), validated_sha256='hash'),
                dict(slot=1, token='token1', key='key1', candidate_directory='other', validated_sha256='other')])
            atomic_json(root/'pi0.pilot_ready.json', ready)
            with patch('throughput_revision_client.acceptance', return_value='medium'):
                self.assertEqual(pilot_authorization(root, session, 'pi0', 0, 'token0', {'key': 'key0'}, candidate, 'hash'), 'medium')
                with self.assertRaises(ValueError): pilot_authorization(root, session, 'pi0', 0, 'wrongtoken', {'key': 'key0'}, candidate, 'hash')
                with self.assertRaises(ValueError): pilot_authorization(root, root/'session2', 'pi0', 0, 'token0', {'key': 'key0'}, candidate, 'hash')

    def test_two_processes_commit_each_key_once(self):
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context('fork')
            queue, barrier = context.Queue(), context.Barrier(2)
            processes = [context.Process(target=reserve_rows, args=(directory, queue, barrier)) for _ in range(2)]
            for process in processes: process.start()
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            actual = queue.get(timeout=1)+queue.get(timeout=1)
            self.assertEqual(sorted(actual), ['key0', 'key1', 'key2'])
            entries = Ledger(directory).read()['entries']
            self.assertEqual(len(entries), 3)
            self.assertTrue(all(entry['status'] == 'valid' for entry in entries))

    def test_unapproved_pilot_cannot_recover_as_valid_after_owner_dies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ledger = Ledger(root)
            row = {'key': 'pilot'}
            token = ledger.reserve(row)
            candidate = root/'episodes'/'actual'
            atomic_json(candidate/'validated.json', dict(valid=True, episode_key='pilot', success=True))
            ledger.update(token, pid=99999999, start_ticks='dead', directory=str(root/'escrow'/token),
                          pilot_candidate_directory=str(candidate))
            with self.assertRaises(RuntimeError): ledger.reserve(row)
            self.assertEqual(ledger.read()['entries'][0]['status'], 'invalid')
            self.assertTrue((candidate/'validated.json').exists())


class HandoffTests(unittest.TestCase):
    def test_active_episode_never_signaled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root/'revision_manifest.json', dict(revision=1))
            atomic_json(root/'episode_ledger.json', dict(entries=[dict(status='reserved', pid=12, start_ticks='34')]))
            with patch.object(handoff, 'verify_revision', return_value=({'source_attempt': str(root)}, {})), \
                 patch.object(handoff, 'acceptance', return_value='accepted'), \
                 patch.object(handoff, 'process_identity', return_value={}), \
                 patch.object(handoff, 'validate_targets', return_value={}), \
                 patch.object(handoff.os, 'pidfd_open', side_effect=[100, 101]), \
                 patch.object(handoff.os, 'close'), \
                 patch.object(handoff.signal, 'pidfd_send_signal') as send:
                result = handoff.handoff(root, 'db-cogact', 2, 12, 13, '34', '35', execute=True)
                self.assertEqual(result['status'], 'busy_active_episode_no_signal')
                send.assert_not_called()

    def test_worker_signal_is_inside_exact_ledger_transaction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root/'revision_manifest.json', dict(revision=1))
            atomic_json(root/'episode_ledger.json', dict(entries=[dict(status='valid', pid=12, start_ticks='34')]))
            signals = []
            def send(fd, signal):
                with (root/'episode_ledger.lock').open('a') as stream:
                    with self.assertRaises(BlockingIOError):
                        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                signals.append(fd)
            @contextmanager
            def available_lock(path):
                yield
            with patch.object(handoff, 'verify_revision', return_value=({'source_attempt': str(root)}, {})), \
                 patch.object(handoff, 'acceptance', return_value='accepted'), \
                 patch.object(handoff, 'process_identity', return_value={}), \
                 patch.object(handoff, 'validate_targets', return_value={}), \
                 patch.object(handoff.os, 'pidfd_open', side_effect=[100, 101]), \
                 patch.object(handoff.os, 'close'), patch.object(handoff, 'wait_pidfd'), \
                 patch.object(handoff, 'start_ticks', return_value=None), \
                 patch.object(handoff, 'exclusive', side_effect=available_lock), \
                 patch.object(handoff.signal, 'pidfd_send_signal', side_effect=send):
                result = handoff.handoff(root, 'db-cogact', 2, 12, 13, '34', '35', execute=True)
            self.assertEqual(signals, [100])
            self.assertTrue(result['locks_released'])


if __name__ == '__main__':
    unittest.main()

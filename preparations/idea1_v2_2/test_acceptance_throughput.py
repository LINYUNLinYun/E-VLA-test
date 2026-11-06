"""Independent CPU checks for the two-environment execution revision."""
import copy
import json
import multiprocessing
from pathlib import Path
import queue
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import Ledger, atomic_json, conditions, digest, lock, planned, sha
from throughput_revision import compare_gate


def reserve_after_boundary(root, row, started, completed):
    started.set()
    completed.put(Ledger(root).reserve(row))


class IndependentThroughputAcceptance(unittest.TestCase):
    def test_provenance_snapshot_allows_only_append_after_live_prefix(self):
        import hashlib
        import throughput_provenance_audit as provenance
        with tempfile.TemporaryDirectory(prefix='v22-independent-prefix-') as temporary:
            path = Path(temporary)/'server.jsonl'
            prefix = b'{"call": 1}\n'
            path.write_bytes(prefix)
            index = dict(partial=True, auditor_source_sha256=sha(provenance.__file__),
                revisions={}, sessions={'session': dict(server_prediction_ledger_prefix=dict(
                    path=str(path), prefix_bytes=len(prefix), prefix_sha256=hashlib.sha256(prefix).hexdigest()))})
            provenance.verify_index_snapshot(index)
            path.write_bytes(prefix+b'{"call": 2}\n')
            provenance.verify_index_snapshot(index)
            with self.assertRaises(ValueError):
                provenance.verify_index_snapshot(dict(index, partial=False))
            path.write_bytes(b'{"call": 9}\n'+b'{"call": 2}\n')
            with self.assertRaises(ValueError):
                provenance.verify_index_snapshot(index)

    def test_provenance_snapshot_rejects_replaced_receipt_bytes(self):
        import throughput_provenance_audit as provenance
        with tempfile.TemporaryDirectory(prefix='v22-independent-receipt-') as temporary:
            path = Path(temporary)/'receipt.json'
            atomic_json(path, dict(accepted=True))
            index = dict(partial=True, auditor_source_sha256=sha(provenance.__file__),
                revisions={'revision': dict(verified_file_hashes={str(path): sha(path)})}, sessions={})
            provenance.verify_index_snapshot(index)
            atomic_json(path, dict(accepted=False))
            with self.assertRaises(ValueError):
                provenance.verify_index_snapshot(index)

    def gate_fixture(self, root):
        for phase in ('reference', 'interleaved'):
            for index in range(4):
                directory = root/phase
                directory.mkdir(exist_ok=True)
                noise = np.full((2, 3), index, dtype=np.float32)
                actions = np.full((2, 7), .25+index, dtype=np.float32)
                np.savez(directory/f'{index}_noise.npz', initial_noise=noise)
                np.savez(directory/f'{index}_prediction.npz', raw_prediction=actions, actions=actions)
                atomic_json(directory/f'{index}.json', dict(input_hash='frozen-real-point', noise_hash=str(index),
                    raw_prediction_hash=str(index), seed=100+index, loaded_code_hashes={'frozen.py': 'source'},
                    audit=dict(NFE=2, actual_NFE=2, schedule=dict(kind='flow', m=.5, num_steps=2),
                               skip=[] if index % 2 == 0 else [3, 4, 9, 11],
                               instrumentation_mode='instrumented_audit', cache_key=str(index % 2)),
                    noise_path=str(directory/f'{index}_noise.npz'),
                    server_prediction_path=str(directory/f'{index}_prediction.npz'), actions=actions.tolist()))

    def test_gate_requires_exact_raw_arrays_not_relaxed_allclose(self):
        with tempfile.TemporaryDirectory(prefix='v22-independent-interleave-') as temporary:
            root = Path(temporary)
            self.gate_fixture(root)
            self.assertEqual(len(compare_gate(root)), 4)
            path = root/'interleaved/0_prediction.npz'
            with np.load(path) as data:
                actions, raw = data['actions'].copy(), data['raw_prediction'].copy()
            raw[0, 0] += 1e-7
            self.assertTrue(np.allclose(actions, raw, rtol=1e-4, atol=1e-5))
            np.savez(path, raw_prediction=raw, actions=actions)
            with self.assertRaises(ValueError):
                compare_gate(root)

    def test_gate_rejects_seed_mask_or_loaded_code_drift(self):
        with tempfile.TemporaryDirectory(prefix='v22-independent-gate-identity-') as temporary:
            root = Path(temporary)
            self.gate_fixture(root)
            path = root/'interleaved/0.json'
            original = json.loads(path.read_text())
            for field, value in [('seed', -1), ('loaded_code_hashes', {'frozen.py': 'other'})]:
                atomic_json(path, dict(original, **{field: value}))
                with self.assertRaises(ValueError):
                    compare_gate(root)
            changed = copy.deepcopy(original)
            changed['audit']['skip'] = [1]
            atomic_json(path, changed)
            with self.assertRaises(ValueError):
                compare_gate(root)

    def test_pilot_escrow_prevents_dead_owner_autocommit(self):
        with tempfile.TemporaryDirectory(prefix='v22-independent-pilot-escrow-') as temporary:
            root = Path(temporary)
            manifest = dict(conditions=conditions(), protocol_hashes={m: digest(m)
                            for m in ('pi0', 'xiaomi-robotic0', 'db-cogact')})
            row = next(planned(manifest))
            ledger = Ledger(root)
            token = ledger.reserve(row)
            candidate = root/'episodes'/'candidate'
            atomic_json(candidate/'validated.json', dict(valid=True, episode_key=row['key'], success=True))
            old_hash = sha(candidate/'validated.json')
            ledger.update(token, directory=str(root/'escrow'/token), pid=999999999,
                          start_ticks='nonexistent', pilot_candidate_directory=str(candidate))
            with self.assertRaises(RuntimeError):
                ledger.reserve(row, require_artifacts=True)
            entries = ledger.read()['entries']
            self.assertEqual(entries[0]['status'], 'invalid')
            self.assertFalse(any(e['status'] == 'valid' for e in entries))
            self.assertEqual(sha(candidate/'validated.json'), old_hash)

    def test_ledger_boundary_lock_blocks_next_reservation(self):
        # spawn must not inherit the parent's already-held flock descriptor.
        context = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory(prefix='v22-independent-handoff-lock-') as temporary:
            root = Path(temporary)
            manifest = dict(conditions=conditions(), protocol_hashes={m: digest(m)
                            for m in ('pi0', 'xiaomi-robotic0', 'db-cogact')})
            first, following = list(planned(manifest))[:2]
            ledger = Ledger(root)
            token = ledger.reserve(first)
            ledger.update(token, status='valid', success=False)
            started, completed = context.Event(), context.Queue()
            with lock(root/'episode_ledger.lock'):
                self.assertFalse(any(e['status'] == 'reserved' for e in ledger.read()['entries']))
                process = context.Process(target=reserve_after_boundary,
                    args=(root, following, started, completed))
                process.start()
                self.assertTrue(started.wait(2))
                with self.assertRaises(queue.Empty):
                    completed.get(timeout=.2)
                self.assertEqual(len(ledger.read()['entries']), 1)
            self.assertTrue(completed.get(timeout=3))
            process.join(3)
            self.assertEqual(process.exitcode, 0)


if __name__ == '__main__':
    unittest.main()

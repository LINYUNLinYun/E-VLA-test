import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from runtime_runner import Ledger, planned, bootstrap_plan, generate_masks
from runtime_client import model_seed
from runtime_analysis import paired_bootstrap


class RuntimeTests(unittest.TestCase):
    def test_matrix_and_order(self):
        candidates = [dict(id=str(i), mask=[i+4]) for i in range(12)]
        rows = planned(candidates)
        self.assertEqual(len(rows), 2600)
        self.assertEqual(len({tuple((r[k] for k in ('kind', 'condition', 'suite', 'task', 'init'))) for r in rows}), 2600)
        self.assertTrue(all(r['init'] == 0 for r in rows[:520]))
        self.assertEqual(len(bootstrap_plan()), 16)

    def test_valid_failure_never_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp))
            token = ledger.reserve('key', 'formal')
            ledger.update(token, status='valid', success=False)
            with self.assertRaises(ValueError):
                ledger.reserve('key', 'formal', True, 'no')

    def test_invalid_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Ledger(Path(tmp))
            token = ledger.reserve('key', 'formal')
            ledger.update(token, status='invalid')
            with self.assertRaises(ValueError):
                ledger.reserve('key', 'formal', True)
            token = ledger.reserve('key', 'formal', True, 'fixed import')
            ledger.update(token, status='invalid')
            with self.assertRaises(ValueError):
                ledger.reserve('key', 'formal', True, 'again')

    def test_seed_shared_across_conditions(self):
        self.assertEqual(model_seed('libero_object', 0, 0, 0), model_seed('libero_object', 0, 0, 0))
        self.assertNotEqual(model_seed('libero_object', 0, 0, 0), model_seed('libero_object', 0, 0, 1))

    def test_bootstrap_known_constant_delta(self):
        from runtime_client import SUITES
        pairs = [(s, t, False, True) for s in SUITES for t in range(10)]
        self.assertEqual(paired_bootstrap(pairs, 100), (100., 100.))

    def test_finite_masks(self):
        for model, depth in [('xiaomi-robotic0', 36), ('db-cogact', 28)]:
            with tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / 'probes.json'
                source.write_text(json.dumps(dict(points=[dict(impacts=list(range(depth)), features=[dict(layer=i+1, kind='hidden', cosine=i/depth) for i in range(depth)])])))
                masks = generate_masks(model, source)
                self.assertEqual(len(masks), 12)
                self.assertEqual(len({tuple(r['mask']) for r in masks}), 12)
                self.assertTrue(all(depth not in r['mask'] for r in masks))

    def test_short_runs_do_not_merge_or_overflow_during_fill(self):
        for model, depth in [('xiaomi-robotic0', 36), ('db-cogact', 28)]:
            # Monotone ranks made the old adjacent triples merge into one long run.
            # Reversed and shuffled ranks also exercise boundary groups and fill.
            orders = [list(range(depth)), list(reversed(range(depth)))]
            orders += [np.random.default_rng(seed).permutation(depth).tolist() for seed in range(10)]
            for impacts in orders:
                with self.subTest(model=model, impacts=impacts), tempfile.TemporaryDirectory() as tmp:
                    source = Path(tmp) / 'probes.json'
                    source.write_text(json.dumps(dict(points=[dict(impacts=impacts, features=[
                        dict(layer=i+1, kind='hidden', cosine=i/depth) for i in range(depth)])])))
                    candidates = generate_masks(model, source)
                    candidate = next(c for c in candidates if c['algorithm'] == 'short_runs')
                    mask = candidate['mask']
                    self.assertEqual(len(mask), 12 if depth == 36 else 9)
                    self.assertFalse(any(mask[i+3]-mask[i] == 3 for i in range(len(mask)-3)))
                    self.assertEqual(len({tuple(c['mask']) for c in candidates}), 12)

    def test_short_runs_dedup_rejects_long_random_run(self):
        # Force all same-budget initial masks to collide, then feed an illegal
        # contiguous proposal before a legal unique proposal to the dedup path.
        model, depth = 'db-cogact', 28
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'probes.json'
            source.write_text(json.dumps(dict(points=[dict(impacts=list(range(depth)), features=[
                dict(layer=i+1, kind='hidden', cosine=i/depth) for i in range(depth)])])))
            baseline = generate_masks(model, source)
            short = next(c['mask'] for c in baseline if c['algorithm'] == 'short_runs')
            # Make seeded_random equal the later short_runs candidate, forcing
            # its dedup branch. Other random slots use distinct legal masks.
            original_rng = np.random.default_rng
            class ForcedRng:
                def __init__(self):
                    self.rng = original_rng(1701)
                    self.high_calls = 0
                def choice(self, values, size, replace=False):
                    if size == 9:
                        self.high_calls += 1
                        if self.high_calls == 1:
                            return np.asarray(short)
                        if self.high_calls == 2:
                            return np.arange(3, 12)  # Illegal four-or-more run.
                        return np.asarray([3, 4, 5, 7, 8, 9, 12, 13, 14])
                    return self.rng.choice(values, size, replace=replace)
            forced = ForcedRng()
            with patch('runtime_runner.np.random.default_rng', return_value=forced):
                candidates = generate_masks(model, source)
            mask = next(c['mask'] for c in candidates if c['algorithm'] == 'short_runs')
            self.assertGreaterEqual(forced.high_calls, 3)
            self.assertEqual(mask, [3, 4, 5, 7, 8, 9, 12, 13, 14])
            self.assertEqual(len({tuple(c['mask']) for c in candidates}), 12)


if __name__ == '__main__':
    unittest.main()

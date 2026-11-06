"""CPU tests of arbitrary saving budgets and frozen-selector compatibility."""
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from preparations.idea1_v3_4_E2_mixed_cost.selection import (
    mask_record, objective_scores, select_mixed, validate_budget_mask,
)
from .budget import load_budget, prepare_budget


def measured_table(costs):
    rows = []
    for layer, values in enumerate(costs, 1):
        row = dict(layer=layer)
        for name, value in zip(('attn', 'ffn', 'af'), values):
            row[name + '_saved_ms'] = value
            row[name + '_noise_ms'] = 0.
        rows.append(row)
    return dict(measured=True, units='ms', n_layers=len(rows), layer_costs=rows)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='arbitrary_vlm_budget_')
        self.root = Path(self.temp.name)
        self.table_path = self.root / 'cost_table.json'
        self.output = self.root / 'artifacts'
        self.write_table([(10., 1., 11.), (10., 2., 12.), (10., 1., 11.)])

    def tearDown(self):
        self.temp.cleanup()

    def write_table(self, costs):
        self.table_path.write_text(json.dumps(measured_table(costs)))

    def write_timings(self, values=(18., 20., 22.)):
        rows = [dict(phase='capture', route='keep', sample=0, vlm_ms=900.),
                dict(phase='common_warmup', route='keep', sample=0, vlm_ms=800.),
                dict(phase='profile_and_gate', route='keep', sample=0, vlm_ms=700.),
                dict(phase='hot', route='F+', sample=0, vlm_ms=600.),
                dict(phase='hot', route='keep', sample=0, profile=True, vlm_ms=500.)]
        rows += [dict(phase='hot', route='keep', sample=i, vlm_ms=value, request_wall_ms=9999.)
                 for i, value in enumerate(values)]
        path = self.root / 'request_timings.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return path

    def build(self, **kwargs):
        return prepare_budget(self.table_path, self.output, **kwargs)

    def test_absolute_budget_without_old_B4_B6_anchors(self):
        budget = self.build(saved_ms=2.)
        self.assertNotIn(budget['budget_id'], ('B4', 'B6'))
        self.assertIsNone(budget['reference_vlm_ms'])
        self.assertEqual(budget, load_budget(budget['artifact_path']))
        self.assertEqual(budget['candidate_count'], 2)
        scores = np.asarray([[100., 1.], [100., 3.], [100., 1.]])
        self.assertEqual(select_mixed(scores, budget, False), ([], [1, 3]))
        self.assertEqual(select_mixed(scores, budget, True), ([], [2]))
        self.assertEqual(mask_record([], [2], budget)['saved_ms'], 2.)

    def test_ratio_uses_only_uninstrumented_hot_keep_complete_VLM(self):
        path = self.write_timings()
        budget = self.build(saved_fraction=.1)
        self.assertEqual(budget['reference_vlm_ms'], 20.)
        self.assertEqual(budget['target_saved_ms'], 2.)
        self.assertEqual(budget['reference_source']['count'], 3)
        self.assertEqual(budget['reference_source']['path'], str(path.resolve()))
        self.assertEqual(budget['reference_source']['kind'], 'measured_hot_keep')
        self.assertIsNone(budget['anchor_fast_ms'])
        explicit = self.build(saved_fraction=.2, reference_vlm_ms=10.)
        self.assertEqual(explicit['target_saved_ms'], 2.)
        self.assertEqual(explicit['reference_source']['kind'], 'explicit')
        self.assertNotEqual(explicit['budget_id'], budget['budget_id'])

    def test_missing_or_profile_only_reference_is_not_fabricated(self):
        with self.assertRaisesRegex(ValueError, 'requires measured hot keep'):
            self.build(saved_fraction=.1)
        (self.root / 'timings.jsonl').write_text(json.dumps(
            dict(phase='profile_and_gate', route='keep', vlm_ms=100.)) + '\n')
        with self.assertRaisesRegex(ValueError, 'no uninstrumented'):
            self.build(saved_fraction=.1)
        explicit = self.build(saved_fraction=.1, reference_vlm_ms=20.)
        self.assertEqual(explicit['target_saved_ms'], 2.)

    def test_zero_has_only_empty_mask_and_positive_excludes_empty(self):
        self.write_table([(.2, 1., 1.1)])
        zero = self.build(saved_ms=0., tolerance_ms=100.)
        self.assertTrue(zero['zero_budget'])
        self.assertEqual(zero['effective_window_ms'], [0., 0.])
        self.assertEqual(zero['candidate_count'], 1)
        self.assertEqual(select_mixed([[1., 1.]], zero), ([], []))
        self.assertEqual(mask_record([], [], zero)['saved_ms'], 0.)
        with self.assertRaises(ValueError):
            validate_budget_mask([1], [], zero)
        positive = self.build(saved_ms=.1, tolerance_ms=.5)
        self.assertFalse(positive['zero_budget'])
        self.assertEqual(select_mixed([[100., 100.]], positive), ([1], []))
        with self.assertRaises(ValueError):
            validate_budget_mask([], [], positive)

    def test_invalid_numbers_and_mutually_exclusive_units(self):
        for kwargs in ({}, dict(saved_ms=1., saved_fraction=.1), dict(saved_ms=-1.),
                       dict(saved_ms=float('nan')), dict(saved_ms=float('inf')),
                       dict(saved_ms=True), dict(saved_ms=1., tolerance_ms=-.1),
                       dict(saved_ms=1., tolerance_ms=float('nan')),
                       dict(saved_fraction=-.1, reference_vlm_ms=20.),
                       dict(saved_fraction=1.01, reference_vlm_ms=20.),
                       dict(saved_fraction=float('nan'), reference_vlm_ms=20.),
                       dict(saved_fraction=.1, reference_vlm_ms=0.),
                       dict(saved_fraction=.1, reference_vlm_ms=float('inf'))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.build(**kwargs)

    def test_spacing_maximum_and_infeasible_window_are_explicit(self):
        self.write_table([(.2, 1., 1.1)] * 3)
        budget = self.build(saved_ms=2.2, tolerance_ms=0.)
        self.assertEqual(budget['max_reachable_saved_ms'], 2.2)
        self.assertEqual(budget['candidate_count'], 1)
        self.assertEqual(select_mixed(np.ones((3, 2)), budget), ([1, 3], [1, 3]))
        with self.assertRaisesRegex(ValueError, 'max_reachable_saved_ms=2.2'):
            self.build(saved_ms=2.3, tolerance_ms=100.)
        self.write_table([(1., 2., 3.)])
        with self.assertRaisesRegex(ValueError, 'No legal nonempty.*max_reachable_saved_ms=3'):
            self.build(saved_ms=1.5, tolerance_ms=.1)

    def test_eligibility_and_positive_normalizer_limit_maximum(self):
        table = measured_table([(0., 1., 100.), (.2, .3, .4), (0., 1., 100.)])
        table['layer_costs'][1]['ffn_noise_ms'] = .5
        table['layer_costs'][1]['af_noise_ms'] = .5
        self.table_path.write_text(json.dumps(table))
        budget = self.build(saved_ms=2., tolerance_ms=0.)
        self.assertEqual(budget['max_reachable_saved_ms'], 2.)
        self.assertEqual(select_mixed(np.ones((3, 2)), budget), ([], [1, 3]))

    def test_target_tolerance_cost_and_reference_cache_identity(self):
        base = self.build(saved_ms=2.)
        same = self.build(saved_ms=2.)
        target = self.build(saved_ms=1.8)
        tolerance = self.build(saved_ms=2., tolerance_ms=.25)
        reference = self.build(saved_ms=2., reference_vlm_ms=20.)
        self.assertEqual(base, same)
        self.assertEqual(len({r['candidate_path'] for r in (base, target, tolerance, reference)}), 4)
        self.assertEqual(len({r['candidate_sha256'] for r in (base, target, tolerance, reference)}), 4)
        self.write_timings()
        auto = self.build(saved_ms=2.)
        self.write_timings((18., 20., 22., 20.))
        changed_source = self.build(saved_ms=2.)
        self.assertEqual(auto['reference_vlm_ms'], changed_source['reference_vlm_ms'])
        self.assertNotEqual(auto['candidate_path'], changed_source['candidate_path'])
        self.write_table([(10., 1., 11.), (10., 2.1, 12.), (10., 1., 11.)])
        changed_cost = self.build(saved_ms=2.)
        self.assertNotEqual(changed_cost['candidate_path'], changed_source['candidate_path'])

    def test_loaded_artifact_and_candidate_hashes_are_checked(self):
        budget = self.build(saved_ms=2.)
        path = Path(budget['artifact_path'])
        modified = dict(budget, target_saved_ms=1.9)
        path.write_text(json.dumps(modified))
        with self.assertRaisesRegex(ValueError, 'artifact hash mismatch'):
            load_budget(path)
        path.write_text(json.dumps(budget))
        archive = Path(budget['candidate_path'])
        archive.write_bytes(archive.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'archive hash mismatch'):
            load_budget(path)

    def test_load_rejects_changed_or_missing_calibration_sources(self):
        timings = self.write_timings()
        budget = self.build(saved_fraction=.1)
        cost_bytes, timing_bytes = self.table_path.read_bytes(), timings.read_bytes()
        self.table_path.write_bytes(cost_bytes + b' ')
        with self.assertRaisesRegex(ValueError, 'Calibration cost table source changed'):
            load_budget(budget['artifact_path'])
        self.table_path.write_bytes(cost_bytes)
        timings.write_bytes(timing_bytes + b'\n')
        with self.assertRaisesRegex(ValueError, 'Measured VLM reference timing source changed'):
            load_budget(budget['artifact_path'])
        timings.write_bytes(timing_bytes)
        self.assertEqual(budget, load_budget(budget['artifact_path']))
        timings.unlink()
        with self.assertRaisesRegex(ValueError, 'Measured VLM reference timing source missing'):
            load_budget(budget['artifact_path'])
        explicit = self.build(saved_fraction=.1, reference_vlm_ms=20.)
        self.assertEqual(explicit, load_budget(explicit['artifact_path']))
        self.table_path.unlink()
        with self.assertRaisesRegex(ValueError, 'Calibration cost table source missing'):
            load_budget(explicit['artifact_path'])

    def test_arbitrary_window_optimality_matches_brute_force(self):
        costs = [(.2, .7, .95), (.3, .8, 1.15), (.4, .9, 1.1), (.25, .6, .9)]
        self.write_table(costs)
        budget = self.build(saved_ms=1.37, tolerance_ms=.23)
        candidates = []
        for states in itertools.product(range(4), repeat=4):
            if not any(states) or any(states[i] and states[i + 1] for i in range(3)):
                continue
            saved = sum(budget['state_costs'][i][state] for i, state in enumerate(states))
            if not budget['effective_window_ms'][0] <= saved <= budget['effective_window_ms'][1]:
                continue
            a = [i + 1 for i, state in enumerate(states) if state & 1]
            f = [i + 1 for i, state in enumerate(states) if state & 2]
            candidates.append((a, f))
        self.assertEqual(budget['candidate_count'], len(candidates))
        scores = np.asarray([[.31, .79], [.17, .47], [.9, .24], [.22, .4]])
        for weighted in (False, True):
            weights = objective_scores(scores, budget, weighted)
            def risk(mask):
                a, f = mask
                return math.fsum([weights[layer - 1, 0] for layer in a] +
                                 [weights[layer - 1, 1] for layer in f])
            actual = select_mixed(scores, budget, weighted)
            self.assertAlmostEqual(risk(actual), min(map(risk, candidates)), places=14)


if __name__ == '__main__':
    unittest.main()

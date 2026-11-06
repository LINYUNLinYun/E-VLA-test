"""CPU evidence for exact constraints, distinct objectives, and AF semantics."""
import itertools
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .selection import (
    _load_candidates, freeze_mixed_budget, legal_sequence_count, mask_record,
    objective_scores, select_mixed, validate_budget_mask,
)


def table(costs, target=2., noises=None):
    rows = []
    for layer, values in enumerate(costs, 1):
        noise = (0., 0., 0.) if noises is None else noises[layer - 1]
        row = dict(layer=layer)
        for name, value, error in zip(('attn', 'ffn', 'af'), values, noise):
            row[f'{name}_saved_ms'] = value
            row[f'{name}_noise_ms'] = error
        rows.append(row)
    return dict(measured=True, units='ms', n_layers=len(rows), layer_costs=rows,
                anchors={name: dict(saved_ms=target, fast_ms=40.) for name in ('B4', 'B6')})


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='mixed_selection_test_')

    def tearDown(self):
        _load_candidates.cache_clear()
        self.temp.cleanup()

    def freeze(self, value):
        return freeze_mixed_budget(value, 'B4', candidate_dir=self.temp.name)

    def test_colored_independent_set_count(self):
        self.assertEqual(legal_sequence_count(0), 1)
        self.assertEqual(legal_sequence_count(1), 4)
        self.assertEqual(legal_sequence_count(17), 2117473)

    def test_unweighted_and_density_really_differ_same_budget(self):
        budget = self.freeze(table([(10., 1., 11.), (10., 2., 12.), (10., 1., 11.)]))
        scores = np.asarray([[100., 1.], [100., 3.], [100., 1.]])
        raw = select_mixed(scores, budget, False)
        weighted = select_mixed(scores, budget, True)
        self.assertEqual(raw, ([], [1, 3]))
        self.assertEqual(weighted, ([], [2]))
        self.assertEqual(mask_record(*raw, budget)['saved_ms'], 2.)
        self.assertEqual(mask_record(*weighted, budget)['saved_ms'], 2.)

    def test_exact_solver_matches_independent_exhaustive_search(self):
        measured = table([(.2, .7, .95), (.3, .8, 1.15), (.4, .9, 1.1),
                          (.25, .6, .9), (.35, .75, 1.25)], target=1.8)
        budget = self.freeze(measured)
        expected = []
        for states in itertools.product(range(4), repeat=5):
            if any(states[i] and states[i + 1] for i in range(4)):
                continue
            cost = sum(budget['state_costs'][i][state] for i, state in enumerate(states))
            if not budget['effective_window_ms'][0] <= cost <= budget['effective_window_ms'][1]:
                continue
            a = [i + 1 for i, state in enumerate(states) if state & 1]
            f = [i + 1 for i, state in enumerate(states) if state & 2]
            expected.append((a, f, cost))
        self.assertEqual(budget['candidate_count'], len(expected))
        rng = np.random.default_rng(341101)
        for _ in range(12):
            scores = rng.uniform(.01, 2., (5, 2))
            for weighted in (False, True):
                weights = objective_scores(scores, budget, weighted)
                def value(record):
                    a, f, cost = record
                    score = math.fsum([weights[layer - 1, 0] for layer in a] +
                                      [weights[layer - 1, 1] for layer in f])
                    key = sum(1 << (layer - 1) for layer in a) | (sum(1 << (layer - 1) for layer in f) << 5)
                    return score, abs(cost - budget['anchor_saved_ms']), key
                optimum = min(expected, key=value)
                actual = select_mixed(scores, budget, weighted)
                self.assertEqual(actual, optimum[:2])
                record = mask_record(*actual, budget, scores=scores, cost_weighted=weighted)
                self.assertAlmostEqual(record['selection_objective_value'], value(optimum)[0], places=13)

    def test_joint_AF_cost_is_not_standalone_sum(self):
        budget = self.freeze(table([(.4, .8, .9)], target=1.))
        record = mask_record([1], [1], budget)
        self.assertEqual(record['saved_ms'], .9)
        self.assertNotEqual(record['saved_ms'], .4 + .8)
        self.assertEqual(record['support_count'], 1)
        self.assertEqual(record['selected_module_count'], 2)
        weights = objective_scores([[2., 3.]], budget, True)
        self.assertTrue(np.array_equal(weights, np.asarray([[2. / .4, 3. / .8]])))

    def test_standalone_noise_guard_and_joint_eligibility(self):
        budget = self.freeze(table([(.1, 1., 1.05)], target=.6, noises=[(.2, .01, .02)]))
        with self.assertRaises(ValueError):
            validate_budget_mask([1], [], budget)
        self.assertIsInstance(validate_budget_mask([1], [1], budget), int)
        self.assertFalse(budget['eligible_states'][0][1])
        self.assertTrue(budget['eligible_states'][0][3])
        invalid = self.freeze(table([(0., 1., 1.05)], target=1.))
        with self.assertRaises(ValueError):
            validate_budget_mask([1], [1], invalid)
        self.assertEqual(select_mixed([[.1, .2]], invalid, True), ([], [1]))

    def test_exact_window_boundary_no_rounding_or_expansion(self):
        budget = self.freeze(table([(1.5, 10., 20.),
                                    (np.nextafter(1.5, np.inf), 10., 20.)], target=1.))
        self.assertEqual(budget['candidate_count'], 1)
        self.assertIsInstance(validate_budget_mask([1], [], budget), int)
        with self.assertRaises(ValueError):
            validate_budget_mask([2], [], budget)
        with self.assertRaises(ValueError):
            self.freeze(table([(10., 20., 30.)], target=2.))

    def test_mask_identity_validation_and_archive_integrity(self):
        budget = self.freeze(table([(.5, .5, .9)] * 3, target=1.))
        for a, f in (([1, 1], []), ([True], []), ([2, 1], []), ([1], [2]), ([0], [])):
            with self.assertRaises(ValueError):
                validate_budget_mask(a, f, budget)
        self.assertEqual(budget, self.freeze(table([(.5, .5, .9)] * 3, target=1.)))
        archive = Path(budget['candidate_path'])
        archive.write_bytes(archive.read_bytes() + b'changed')
        _load_candidates.cache_clear()
        with self.assertRaises(ValueError):
            select_mixed(np.ones((3, 2)), budget)

    def test_invalid_measured_costs_or_scores_fail(self):
        with self.assertRaises(ValueError):
            self.freeze(dict(table([(.5, 1., 1.2)]), measured=False))
        budget = self.freeze(table([(.5, 1., 1.2)], target=1.))
        for scores in (np.ones(2), [[-.1, .1]], [[float('nan'), .1]], [[float('inf'), .1]]):
            with self.assertRaises(ValueError):
                select_mixed(scores, budget)


if __name__ == '__main__':
    unittest.main()

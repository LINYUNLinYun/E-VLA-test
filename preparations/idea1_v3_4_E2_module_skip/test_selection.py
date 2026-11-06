"""CPU tests of measured-budget selection and queue persistence; no model load."""
import copy
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import common
from .selection import (enumerate_ffn_masks, fixed_episode_mask, freeze_budget,
                        mask_record, select_ffn)


def measured(costs, target=2.1, fast_ms=40.0, noise=0.0):
    return dict(schema_version=1, measured=True, units='ms', n_layers=len(costs),
                layer_costs=[dict(layer=i + 1, ffn_saved_ms=cost, measurement_noise_ms=noise)
                             for i, cost in enumerate(costs)],
                anchors={key: dict(saved_ms=target, fast_ms=fast_ms) for key in ('B4', 'B6')},
                provenance=dict(input_manifest_sha256='test fixture', gpu_uuid='CPU fixture'))


class BudgetSelectionTest(unittest.TestCase):
    def test_complete_spaced_enumeration(self):
        masks = enumerate_ffn_masks()
        self.assertEqual(len(masks), 4181)
        self.assertEqual(len(set(masks)), len(masks))
        self.assertTrue(all(all(b - a > 1 for a, b in zip(mask, mask[1:])) for mask in masks))
        self.assertIn((1, 3, 5, 7, 9, 11, 13, 15, 17), masks)

    def test_minimum_sensitivity_obeys_one_total_window(self):
        table = measured([1.0, 1.2, 1.1, 0.8, 1.0])
        frozen = freeze_budget(table, 'B4')
        scores = [1, 8, 3, 9, 6]
        selected = select_ffn(scores, frozen)
        self.assertEqual(selected, [1, 3])
        self.assertFalse(frozen['nearest_tier'])
        for candidate in frozen['candidates']:
            self.assertGreaterEqual(candidate['saved_ms'], 1.6 - 1e-9)
            self.assertLessEqual(candidate['saved_ms'], 2.6 + 1e-9)
        optimum = min(sum(scores[layer - 1] for layer in row['mask']) for row in frozen['candidates'])
        self.assertEqual(sum(scores[layer - 1] for layer in selected), optimum)
        self.assertEqual(mask_record(selected, frozen)['skip_count'], 2)

    def test_tolerance_is_anchor_fast_latency_and_frozen(self):
        table = measured([1.0] * 5, target=2.0, fast_ms=123.0)
        frozen = freeze_budget(table, 'B4')
        self.assertAlmostEqual(frozen['tau_ms'], 1.23)
        before = copy.deepcopy(frozen)
        select_ffn([1, 2, 3, 4, 5], frozen)
        select_ffn([5, 4, 3, 2, 1], frozen)
        self.assertEqual(frozen, before)

    def test_discrete_infeasible_budget_freezes_nearest_tier_by_cost(self):
        frozen = freeze_budget(measured([2.0], target=1.0), 'B4')
        self.assertTrue(frozen['nearest_tier'])
        self.assertEqual(frozen['nearest_saved_ms'], 2.0)
        self.assertEqual(frozen['effective_window_ms'], [2.0, 2.0])
        self.assertEqual(select_ffn([100], frozen), [1])
        self.assertEqual(fixed_episode_mask(frozen, dict(suite='s', task=1, init=19)), [1])

    def test_nonpositive_or_noise_limited_costs_stay_in_readings(self):
        table = measured([1.0, -0.1, 0.01, 1.0], target=1.0, noise=0.02)
        frozen = freeze_budget(table, 'B4')
        self.assertEqual(frozen['excluded_layers'], [2, 3])
        self.assertEqual(frozen['measured_layer_costs'][1]['ffn_saved_ms'], -0.1)
        self.assertTrue(all(not ({2, 3} & set(row['mask'])) for row in frozen['candidates']))

    def test_missing_or_unmeasured_costs_are_rejected(self):
        table = measured([1] * 5)
        table['measured'] = False
        with self.assertRaises(ValueError):
            freeze_budget(table, 'B4')
        table['measured'] = True
        table['layer_costs'].pop()
        with self.assertRaises(ValueError):
            freeze_budget(table, 'B4')

    def test_scores_and_candidate_masks_are_validated(self):
        table = measured([1] * 5)
        frozen = freeze_budget(table, 'B4')
        for scores in ([1, 2], [1, -2, 3, 4, 5], [1, math.nan, 3, 4, 5]):
            with self.assertRaises(ValueError):
                select_ffn(scores, frozen)
        with self.assertRaises(ValueError):
            freeze_budget(table, 'B4', [[1, 2]])
        with self.assertRaises(ValueError):
            mask_record([1, 2], frozen)

    def test_fixed_mask_is_episode_identity_based_and_legal(self):
        frozen = freeze_budget(measured([1] * 17, target=4), 'B4')
        case = dict(suite='libero_spatial', task=1, init=19)
        first = fixed_episode_mask(frozen, case)
        self.assertEqual(first, fixed_episode_mask(frozen, dict(init=19, task=1, suite='libero_spatial', request=99)))
        self.assertIn(first, [row['mask'] for row in frozen['candidates']])
        different = {tuple(fixed_episode_mask(frozen, dict(case, init=init))) for init in range(19, 30)}
        self.assertGreater(len(different), 1)


class QueueTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.out = Path(self.temporary.name)
        self.override = patch.object(common, 'OUT', self.out)
        self.override.start()
        self.cases = [dict(suite='libero_spatial', task=1, init=init) for init in (19, 20)]
        self.group = 'test_task'
        common.atomic(self.out / 'queue.json', dict(groups=[dict(key=self.group, phase='small40',
                      suite='libero_spatial', task=1, state='pending',
                      cases=[dict(case=case, state='pending') for case in self.cases])]))

    def tearDown(self):
        self.override.stop()
        self.temporary.cleanup()

    def test_claim_release_and_two_client_limit(self):
        self.assertEqual(common.claim_group('gpu-uuid')['key'], self.group)
        self.assertIsNone(common.claim_group('other'))
        self.assertEqual(common.claim_case(self.group, 'client0'), self.cases[0])
        with self.assertRaises(ValueError):
            common.claim_case(self.group, 'client0')
        self.assertEqual(common.claim_case(self.group, 'client1'), self.cases[1])
        self.assertIsNone(common.claim_case(self.group, 'client2'))
        common.finish_case(self.group, self.cases[0])
        common.release('gpu-uuid')
        common.claim_group('replacement')
        self.assertEqual(common.claim_case(self.group, 'client2'), self.cases[1])
        common.finish_case(self.group, self.cases[1])
        common.finish_group(self.group)
        self.assertFalse(common.pending())

    def test_technical_failures_are_bounded_and_fatal_failures_stop(self):
        cfg = common.CONDITIONS[1]
        self.assertEqual(common.begin_attempt(cfg, self.cases[0]), 1)
        self.assertTrue(common.record_technical_failure(cfg, self.cases[0], RuntimeError('RPC reset')))
        self.assertEqual(common.begin_attempt(cfg, self.cases[0]), 2)
        self.assertFalse(common.record_technical_failure(cfg, self.cases[0], RuntimeError('RPC reset')))
        with self.assertRaises(RuntimeError):
            common.begin_attempt(cfg, self.cases[0])
        common.begin_attempt(cfg, self.cases[1])
        self.assertFalse(common.record_technical_failure(cfg, self.cases[1], ValueError('history mismatch'), fatal=True))
        with self.assertRaises(RuntimeError):
            common.begin_attempt(cfg, self.cases[1])

    def test_pilot_and_mechanism_barriers_hold_all_main_cases(self):
        from . import controller, evaluate
        cases = [dict(suite=suite, task=task, init=init) for task in (1, 3, 5, 7, 9)
                 for suite in common.SUITES for init in (19, 20)]
        groups = common._make_groups(cases)
        common.atomic(self.out / 'queue.json', dict(groups=groups))
        with patch.object(controller, 'OUT', self.out), patch.object(evaluate, 'OUT', self.out):
            self.assertEqual(controller.ready_groups(), 8)
            claimed = []
            for index in range(8):
                group = common.claim_group(f'gpu{index}')
                self.assertEqual(group['phase'], 'pilot')
                claimed.append(group)
            self.assertIsNone(common.claim_group('no-main'))
            for group in claimed:
                case = common.claim_case(group['key'], 'client')
                common.finish_case(group['key'], case)
                common.finish_group(group['key'])
            self.assertTrue((self.out / 'pilot_complete.json').exists())
            self.assertEqual(controller.ready_groups(), 0)
            self.assertIsNone(common.claim_group('still-no-main'))
            manifest = dict(cases=cases, conditions=common.CONDITIONS, pilot_cases=[cases[0]])
            pilot = evaluate.ordered_conditions(manifest, cases[0], 'pilot')
            self.assertEqual([cfg['id'] for cfg in pilot], ['B4_F-R+', 'B6_F-R+'])
            with self.assertRaises(ValueError):
                evaluate.ordered_conditions(manifest, cases[0], 'small40')
            common.atomic(self.out / 'pilot_mechanism_complete.json', dict(passed=True))
            self.assertEqual(controller.ready_groups(), 20)
            self.assertEqual(common.claim_group('main-now')['phase'], 'small40')
            self.assertEqual(len(evaluate.ordered_conditions(manifest, cases[0], 'small40')), 12)

    def test_offline_mechanism_runs_once_in_the_existing_service(self):
        from . import controller
        calls = []
        def rpc(port, request):
            calls.append((port, request['op']))
            return dict(passed=True, summary=dict(requests=17))
        common.atomic(self.out / 'pilot_complete.json', dict(passed=True))
        fake = SimpleNamespace(rpc=rpc)
        with patch.object(controller, 'OUT', self.out), patch.dict('sys.modules',
                {'preparations.idea1_v2_1.runtime_client': fake}):
            controller.offline_if_ready(12345, self.out / 'worker')
            controller.offline_if_ready(12345, self.out / 'other_worker')
        self.assertEqual(calls, [(12345, 'reset_all'), (12345, 'offline')])
        self.assertTrue((self.out / 'pilot_mechanism_complete.json').exists())

    def test_offline_failure_never_unlocks_main(self):
        from . import controller
        common.atomic(self.out / 'pilot_complete.json', dict(passed=True))
        def rpc(port, request):
            if request['op'] == 'offline':
                raise RuntimeError('technical numerical failure')
            return {}
        with patch.object(controller, 'OUT', self.out), patch.dict('sys.modules',
                {'preparations.idea1_v2_1.runtime_client': SimpleNamespace(rpc=rpc)}):
            with self.assertRaises(RuntimeError):
                controller.offline_if_ready(12345, self.out / 'worker')
        self.assertFalse((self.out / 'pilot_mechanism_complete.json').exists())
        self.assertTrue((self.out / 'offline_failure.json').exists())

    def test_matrix_has_exactly_eighty_reused_and_four_hundred_new(self):
        self.assertEqual(len(common.CONDITIONS), 12)
        self.assertEqual(sum(cfg['n'] for cfg in common.CONDITIONS), 480)
        self.assertEqual(sum(cfg['n'] for cfg in common.CONDITIONS if cfg['reused']), 80)
        self.assertEqual(sum(cfg['n'] for cfg in common.CONDITIONS if not cfg['reused']), 400)
        self.assertEqual({cfg['budget'] for cfg in common.CONDITIONS}, {'B4', 'B6'})


if __name__ == '__main__':
    unittest.main()

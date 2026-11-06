import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from scripts.experiment10.common import select_steps, split
from scripts.experiment10.numerics import activity, representation, action_metrics
from scripts.experiment10.search_rules import Rules, pareto_select, heuristic_search


class ProtocolTests(unittest.TestCase):
    def test_activity_association_constant_resamples_are_undefined(self):
        from scripts.experiment10.activity_association_stability import correlations
        x = np.asarray([[1., 2., 3.], [0., 0., 0.], [1., 1., 2.]])
        y = np.asarray([[3., 2., 1.], [1., 2., 3.], [2., 2., 3.]])
        result = correlations(x, y)
        self.assertAlmostEqual(result[0], -1.)
        self.assertTrue(np.isnan(result[1]))
        self.assertAlmostEqual(result[2], 1.)

    def test_focused_update_details_preserve_locations(self):
        from scripts.experiment10.update_details import update_details
        x = torch.zeros((10,10)); x[3,7] = 2
        stats, tokens, channels = update_details(x)
        self.assertIsNone(stats['token_norm_p90_over_p50'])
        self.assertEqual(stats['top10pct_channel_energy_fraction'], 1)
        self.assertEqual(stats['top10pct_token_energy_fraction'], 1)
        self.assertEqual(np.argmax(tokens), 3)
        self.assertEqual(np.argmax(channels), 7)
        stats, _, _ = update_details(torch.zeros((4,8)))
        self.assertTrue(stats['all_zero'])
        self.assertIsNone(stats['top10pct_channel_energy_fraction'])
        self.assertIsNone(stats['top10pct_token_energy_fraction'])

    def test_async_full_output_cannot_follow_executed_draft(self):
        from scripts.experiment10.request_ledger import RequestLedger
        ledger = RequestLedger(100)
        ledger.reset('episode0'); ledger.observe(0, 0, 10)
        old = ledger.submit()
        ledger.executed(4)
        with self.assertRaises(ValueError):
            ledger.submit()
        self.assertFalse(ledger.resolve(old, 20)[0])
        ledger.observe(1, 4, 21)
        fresh = ledger.submit()
        self.assertTrue(ledger.resolve(fresh, 50)[0])
        self.assertFalse(ledger.resolve(fresh, 51)[0])
        delayed = ledger.submit()
        self.assertFalse(ledger.resolve(delayed, 122)[0])
        obsolete = ledger.submit()
        ledger.reset('episode1'); ledger.observe(0, 0, 130)
        self.assertFalse(ledger.resolve(obsolete, 140)[0])

    def test_command_gate_uses_executed_history_and_contiguous_prefix(self):
        from scripts.experiment10.command_predictor import CommandPredictor, accepted_prefix
        predictor = CommandPredictor(np.ones(6)*.01, np.ones(6)*.01)
        candidate = np.zeros((12, 7)); candidate[:, 6] = -1
        self.assertEqual(accepted_prefix(predictor, candidate, 3, 3, 10, 100)[0], 0)
        predictor.observe_executed(candidate[:4])
        before = predictor.state.copy()
        self.assertEqual(accepted_prefix(predictor, candidate, 3, 3, 10, 100)[0], 12)
        candidate[3, 0] = 100
        self.assertEqual(accepted_prefix(predictor, candidate, 3, 3, 10, 100)[0], 3)
        np.testing.assert_array_equal(before, predictor.state)
        candidate[3, 0] = 0; candidate[2, 6] = 1
        self.assertEqual(accepted_prefix(predictor, candidate, 3, 3, 10, 100)[0], 2)
        self.assertEqual(accepted_prefix(predictor, candidate, 3, 3, 101, 100)[0], 0)
        self.assertTrue(np.all(np.linalg.eigvalsh(predictor.covariance) >= -1e-12))

    def test_same_observation_verification_stops_at_first_component_or_gripper_difference(self):
        from scripts.experiment10.command_predictor import verified_prefix
        candidate = np.zeros((12, 7)); reference = candidate.copy()
        self.assertEqual(verified_prefix(candidate, reference, 8, .01, .01)[0], 8)
        candidate[3, 0] = .02
        self.assertEqual(verified_prefix(candidate, reference, 8, .01, .01), (3, 'translation_disagreement'))
        candidate[3, 0] = 0; candidate[2, 4] = .02
        self.assertEqual(verified_prefix(candidate, reference, 8, .01, .01), (2, 'rotation_disagreement'))
        candidate[2, 4] = 0; candidate[1, 6] = 1
        self.assertEqual(verified_prefix(candidate, reference, 8, .01, .01), (1, 'gripper_disagreement'))
        self.assertEqual(verified_prefix(candidate, reference, 0, .01, .01)[0], 0)
        with self.assertRaises(ValueError):
            verified_prefix(candidate, reference, 8, 0, .01)

    def test_first_deviation_requires_persistence(self):
        from scripts.experiment10.first_deviation import first_persistent, pose_error_degrees
        self.assertIsNone(first_persistent([0, .03, .03, 0], .02))
        self.assertEqual(first_persistent([0, .03, .03, .03, 0], .02), 1)
        np.testing.assert_allclose(pose_error_degrees([[1, 0, 0, 0]], [[-1, 0, 0, 0]]), [0])
        np.testing.assert_allclose(pose_error_degrees([[1, 0, 0, 0]], [[0, 1, 0, 0]]), [180])
        with self.assertRaises(ValueError):
            pose_error_degrees([[0, 0, 0, 0]], [[1, 0, 0, 0]])

    def test_phase_contrast_requires_same_init_pairs(self):
        import pandas as pd
        from scripts.experiment10.analyze_phase import contrasts
        rows = []
        for task in range(4):
            for layer in (3, 4):
                for phase, init, value in [('approach', 0, 1.), ('contact_grasp', 1, 2.)]:
                    rows.append({'suite': 'libero_goal', 'split': 'search', 'task_id': task,
                                 'episode_idx': init, 'layer': layer, 'semantic_phase': phase,
                                 'normalized_continuous_rmse': value})
        episodes = pd.DataFrame(rows)
        tasks = episodes.groupby(['suite', 'split', 'task_id', 'layer', 'semantic_phase'],
                                 as_index=False).normalized_continuous_rmse.mean()
        self.assertTrue(contrasts(tasks, episodes).empty)
        episodes['episode_idx'] = 0
        result = contrasts(tasks, episodes)
        self.assertFalse(result.empty)
        self.assertTrue((result.paired_episode_count == 4).all())
        self.assertTrue((result.relative_layer_difference == 0).all())

    def test_residual_cancellation_identity(self):
        from scripts.experiment10.analyze_updates import cancellation
        cosine, ratio = cancellation([1, 1, 1], [1, 1, 1], [2, 0, np.sqrt(2)])
        np.testing.assert_allclose(cosine, [1, -1, 0], atol=1e-14)
        np.testing.assert_allclose(ratio, [1, 0, 1/np.sqrt(2)])

    def test_residual_cancellation_zero_is_undefined(self):
        from scripts.experiment10.analyze_updates import cancellation
        cosine, ratio = cancellation([0, 0], [0, 1], [0, 1])
        self.assertTrue(np.isnan(cosine).all())
        self.assertTrue(np.isnan(ratio[0]))
        self.assertEqual(ratio[1], 1)

    def test_residual_cancellation_invalid_capture(self):
        from scripts.experiment10.analyze_updates import cancellation
        with self.assertRaises(RuntimeError):
            cancellation([1], [1], [3])
        with self.assertRaises(ValueError):
            cancellation([np.nan], [1], [1])

    def test_replanning_only_first_middle_last(self):
        telemetry = [{'step': i, 'semantic_phase': 'approach'} for i in range(25)]
        chosen, missing = select_steps(telemetry, {0, 4, 8, 12, 16, 20, 24})
        self.assertEqual(set(chosen), {0, 12, 24})
        self.assertEqual(len(missing), 3)

    def test_no_terminal_sampling(self):
        chosen, _ = select_steps([{'step': 0, 'semantic_phase': 'initial'},
                                  {'step': 1, 'semantic_phase': 'approach', 'done_after_step': True}], {0, 1})
        self.assertEqual(set(chosen), {0})

    def test_split(self):
        self.assertEqual([split(i) for i in (0, 5, 6, 7, 8, 9)],
                         ['search', 'search', 'validation', 'validation', 'evaluation', 'evaluation'])

    def test_all_zero_is_not_cosine_one(self):
        zero = torch.zeros(4, 8)
        self.assertEqual(activity(zero)['zero_fraction'], 1)
        self.assertIsNone(representation(zero, zero)['cosine'])
        self.assertIsNone(representation(zero, zero)['centered_linear_cka'])

    def test_constant_cka_undefined(self):
        x = torch.ones(4, 8)
        result = representation(x, x)
        self.assertAlmostEqual(result['cosine'], 1, places=6)
        self.assertIsNone(result['centered_linear_cka'])

    def test_scale_invariance_near_zero(self):
        x = torch.tensor([[0., .001, 1.], [-1., .1, .01]])
        a, b = activity(x), activity(x * 1000)
        for alpha in (.001, .01, .1):
            self.assertEqual(a['near_zero_' + str(alpha)], b['near_zero_' + str(alpha)])

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            activity(torch.tensor([float('nan')]))

    def test_same_actions_zero_error(self):
        a = np.ones((50, 32))
        rows = action_metrics(a, a, a[:, :7], a[:, :7])
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r['normalized_continuous_rmse'] == 0 for r in rows))

    def test_gripper_and_horizon(self):
        a, b = np.ones((50, 32)), np.ones((50, 32))
        a[4:, :6] = 100
        a[0, 6] = -1
        rows = action_metrics(a, b, a[:, :7], b[:, :7])
        self.assertEqual(rows[2]['normalized_continuous_rmse'], 0)
        self.assertEqual(rows[2]['gripper_flip_rate'], .25)
        self.assertGreater(rows[3]['normalized_continuous_rmse'], 0)

    def test_near_zero_energy_not_fake_nmse(self):
        a = np.zeros((50, 32))
        row = action_metrics(a, a, a[:, :7], a[:, :7])[0]
        self.assertIsNone(row['translation_normalized_nmse'])
        self.assertTrue(row['translation_near_zero_baseline'])

    def test_mask_constraints_and_challenges(self):
        rules = Rules(frozenset((12, 13, 14, 15, 16)))
        self.assertTrue(rules.allowed((3, 5, 7)))
        self.assertFalse(rules.allowed((3, 4)))
        self.assertTrue(rules.allowed((3, 4), challenge=True))
        self.assertFalse(rules.allowed((3, 4, 12), challenge=True))
        self.assertFalse(rules.allowed((1, 3), challenge=True))
        self.assertFalse(rules.allowed((3, 4, 5), challenge=True))

    def test_infeasible_count_not_silently_relaxed(self):
        rules = Rules(frozenset(range(8, 17)))
        self.assertEqual(list(rules.legal_masks(4)), [])
        self.assertEqual(rules.nearest_v13(), [])

    def test_pareto_minimization(self):
        selected = pareto_select({(3,): (1, 2), (5,): (2, 1), (7,): (3, 3)}, width=2)
        self.assertEqual(set(selected), {(3,), (5,)})

    def test_search_budget_and_nonadditive_evaluations(self):
        rules = Rules(frozenset(range(8, 17)))
        actual = []
        def oracle(mask):
            actual.append(mask)
            return (sum(mask), len(mask))
        result = heuristic_search(rules, oracle, {}, budget=20)
        self.assertLessEqual(len(actual), 20)
        self.assertEqual(len(actual), len(set(actual)))
        self.assertIn(4, result['infeasible_main_counts'])

    def test_idle_gpu_does_not_take_small_external_process(self):
        from scripts.experiment10.run_pipeline import free_gpus
        results = [SimpleNamespace(stdout='0, GPU-a, 1, 0\n1, GPU-b, 100, 0\n'),
                   SimpleNamespace(stdout='GPU-b, 123\n')]
        with patch('scripts.experiment10.run_pipeline.subprocess.run', side_effect=results):
            self.assertEqual(free_gpus(), [0])

    def test_experiment_gpu_lease_is_exclusive(self):
        from scripts.experiment10.common import claim_gpu
        with tempfile.TemporaryDirectory() as directory:
            with patch('scripts.experiment10.common.RESULTS', Path(directory)):
                a = claim_gpu(2)
                self.assertIsNotNone(a)
                self.assertIsNone(claim_gpu(2))
                a.close()
                b = claim_gpu(2)
                self.assertIsNotNone(b)
                b.close()

    def test_parallel_beam_uses_measured_combination_scores(self):
        rules = Rules(frozenset(range(8, 17)))
        batches = []
        def batch(masks):
            batches.append(list(masks))
            return {m: (sum(m), len(m)) for m in masks}
        def prohibited(_):
            raise AssertionError('Should use parallel actual oracle')
        result = heuristic_search(rules, prohibited, {}, budget=20, evaluate_batch=batch)
        self.assertGreater(max(map(len, batches)), 1)
        self.assertLessEqual(sum(map(len, batches)), 20)

    def test_phase_main_effect_not_mistaken_for_interaction(self):
        import pandas as pd
        from scripts.experiment10.analyze_phase import contrasts
        rows = [{'suite': 'libero_goal', 'split': 'search', 'task_id': t, 'layer': l,
                 'semantic_phase': p, 'normalized_continuous_rmse': float(l+delta)}
                for t in range(4) for l in (3,5) for p, delta in [('approach',0),('contact_grasp',1)]]
        result = contrasts(pd.DataFrame(rows))
        self.assertTrue(np.allclose(result.relative_layer_difference, 0))
        self.assertTrue(np.allclose(result.p_signflip, 1))

    def test_phase_layer_reversal_is_visible(self):
        import pandas as pd
        from scripts.experiment10.analyze_phase import contrasts
        rows = [{'suite': 'libero_goal', 'split': 'search', 'task_id': t, 'layer': l,
                 'semantic_phase': p, 'normalized_continuous_rmse': value}
                for t in range(4) for l,p,value in [(3,'approach',.2),(3,'contact_grasp',.8),
                                                    (5,'approach',.8),(5,'contact_grasp',.2)]]
        result = contrasts(pd.DataFrame(rows))
        self.assertAlmostEqual(float(result[result.layer==3].relative_layer_difference.iloc[0]), .6)
        self.assertAlmostEqual(float(result[result.layer==5].relative_layer_difference.iloc[0]), -.6)

    def test_prefix_recovery_and_seed_rule(self):
        from scripts.experiment10.serve_policy import Policy
        class FakeBackend:
            weight_hash = 'checkpoint'
            calls = []
            def prepare(self, observation, meta, noise):
                self.noise = noise
            def infer(self, skip=(), steps=10):
                self.calls.append((tuple(skip), steps))
                return np.zeros((50,32)), np.zeros((50,7)), {'cuda_policy_ms': 1.}
        backend = FakeBackend()
        policy = Policy(backend, 'libero_goal', (3,5), 3, 'prefix_once', 4, 'test')
        obs = {'__trace_episode_idx__': 2, '__trace_infer_id__': 0, '__reset_policy_state__': True,
               'observation/image': 0, 'observation/wrist_image': 0, 'observation/state': 0, 'prompt': 'test'}
        a = policy.infer(obs)
        expected = np.random.default_rng(np.random.SeedSequence([12345,2,0])).standard_normal((50,32)).astype(np.float32)
        np.testing.assert_array_equal(backend.noise, expected)
        obs.update({'__trace_infer_id__': 1, '__reset_policy_state__': False})
        b = policy.infer(obs)
        self.assertEqual(a['accepted_prefix_len'], 4)
        self.assertEqual(a['policy_timing']['is_full_pipeline_round'], 0)
        self.assertEqual(a['policy_timing']['include_in_draft_accept_metrics'], 1)
        self.assertEqual(b['accepted_prefix_len'], 12)
        self.assertEqual(b['policy_timing']['is_full_pipeline_round'], 1)
        self.assertEqual(backend.calls, [((3,5),3),((),10)])

    def test_verify_then_execute_uses_full_fallback_when_first_action_disagrees(self):
        from scripts.experiment10.serve_policy import Policy
        class FakeBackend:
            weight_hash = 'checkpoint'
            def __init__(self, first_delta):
                self.first_delta = first_delta
                self.calls = []
            def prepare(self, observation, meta, noise):
                self.noise = noise
            def infer(self, skip=(), steps=10):
                self.calls.append((tuple(skip), steps))
                actions = np.zeros((50, 7), dtype=np.float32)
                if steps == 3:
                    actions[0, 0] = self.first_delta
                return np.zeros((50, 32)), actions, {'cuda_policy_ms': float(steps)}
        observation = {'__trace_episode_idx__': 0, '__trace_infer_id__': 0, '__reset_policy_state__': True,
                       'observation/image': 0, 'observation/wrist_image': 0, 'observation/state': 0, 'prompt': 'test'}
        fallback_backend = FakeBackend(.02)
        fallback = Policy(fallback_backend, 'libero_goal', (), 3, 'verify_then_execute', 8,
                          'VerifyD3_H8', .01, .01).infer(observation)
        self.assertEqual(fallback['accepted_prefix_len'], 12)
        self.assertEqual(fallback['policy_timing']['used_full_fallback'], 1)
        self.assertEqual(fallback['policy_timing']['denoise_steps'], 10)
        self.assertEqual(fallback_backend.calls, [((), 3), ((), 10)])
        accepted_backend = FakeBackend(.005)
        accepted = Policy(accepted_backend, 'libero_goal', (), 3, 'verify_then_execute', 8,
                          'VerifyD3_H8', .01, .01).infer(observation)
        self.assertEqual(accepted['accepted_prefix_len'], 8)
        self.assertEqual(accepted['policy_timing']['used_full_fallback'], 0)
        self.assertEqual(accepted['policy_timing']['denoise_steps'], 3)

    def test_parallel_verifier_selects_candidate_prefix_or_reference_fallback(self):
        from scripts.experiment10.parallel_verify import combine_parallel_outputs
        condition = {'name': 'parallel', 'skip': [], 'steps': 3, 'horizon': 8,
                     'translation_threshold_m': .01, 'rotation_threshold_rad': .01}
        def output(actions, steps):
            return {'actions': actions, 'accepted_prefix_len': 12,
                    'policy_timing': {'checkpoint_sha256': 'checkpoint', 'noise_sha256': 'noise',
                                      'cuda_policy_ms': float(steps), 'sample_actions_ms': float(steps),
                                      'denoise_steps': steps, 'vlm_kv_shell_indices_1based': []}}
        candidate_actions = np.zeros((50, 7), dtype=np.float32)
        reference_actions = np.zeros((50, 7), dtype=np.float32)
        candidate_actions[:8, 0] = .005
        accepted = combine_parallel_outputs(output(candidate_actions, 3),
                                            output(reference_actions, 10), condition, 30.)
        self.assertEqual(accepted['accepted_prefix_len'], 8)
        self.assertEqual(accepted['policy_timing']['used_full_fallback'], 0)
        self.assertEqual(accepted['policy_timing']['denoise_steps'], 3)
        self.assertEqual(accepted['policy_timing']['parallel_inference_wall_ms'], 30.)

        candidate_actions[0, 0] = .02
        fallback = combine_parallel_outputs(output(candidate_actions, 3),
                                            output(reference_actions, 10), condition, 40.)
        self.assertEqual(fallback['accepted_prefix_len'], 12)
        self.assertEqual(fallback['policy_timing']['used_full_fallback'], 1)
        self.assertEqual(fallback['policy_timing']['denoise_steps'], 10)
        np.testing.assert_array_equal(fallback['actions'], reference_actions)

        bad_reference = output(reference_actions, 10)
        bad_reference['policy_timing']['noise_sha256'] = 'different'
        with self.assertRaises(RuntimeError):
            combine_parallel_outputs(output(candidate_actions, 3), bad_reference, condition, 40.)

    def test_kalman_gate_falls_back_until_history_then_accepts_contiguous_draft(self):
        import time
        from scripts.experiment10.serve_policy import Policy
        class FakeBackend:
            weight_hash = 'checkpoint'
            def __init__(self):
                self.calls = []
            def prepare(self, observation, meta, noise):
                pass
            def infer(self, skip=(), steps=10):
                self.calls.append((tuple(skip), steps))
                return np.zeros((50, 32)), np.zeros((50, 7)), {'cuda_policy_ms': float(steps)}
        backend = FakeBackend()
        policy = Policy(backend, 'libero_goal', (), 3, 'kalman_gate', 4,
                        'KalmanD3_Q50', 3., 3., np.ones(6)*.01,
                        np.ones(6)*.01, max_observation_age_ms=100., minimum_executed_history=4)
        observation = {'__trace_episode_idx__': 0, '__trace_infer_id__': 0,
                       '__reset_policy_state__': True, '__executed_actions_since_last_infer__': [],
                       '__observation_time_monotonic_ms__': time.monotonic()*1000.,
                       'observation/image': 0, 'observation/wrist_image': 0,
                       'observation/state': 0, 'prompt': 'test'}
        first = policy.infer(observation)
        self.assertEqual(first['accepted_prefix_len'], 12)
        self.assertEqual(first['policy_timing']['execution_gate_reason'], 'insufficient_executed_history')
        self.assertEqual(first['policy_timing']['is_full_pipeline_round'], 1)
        self.assertEqual(backend.calls, [((), 10)])

        observation.update({'__trace_infer_id__': 1, '__reset_policy_state__': False,
                            '__executed_actions_since_last_infer__': np.zeros((4, 7)).tolist(),
                            '__observation_time_monotonic_ms__': time.monotonic()*1000.})
        second = policy.infer(observation)
        self.assertEqual(second['accepted_prefix_len'], 4)
        self.assertEqual(second['policy_timing']['used_full_fallback'], 0)
        self.assertEqual(second['policy_timing']['predictor_history_count'], 4)
        self.assertEqual(second['policy_timing']['is_full_pipeline_round'], 0)
        self.assertEqual(second['policy_timing']['include_in_draft_accept_metrics'], 1)
        self.assertEqual(backend.calls, [((), 10), ((), 3)])

    def test_kalman_gate_rejects_stale_observation(self):
        import time
        from scripts.experiment10.serve_policy import Policy
        class FakeBackend:
            weight_hash = 'checkpoint'
            def __init__(self):
                self.calls = []
            def prepare(self, observation, meta, noise):
                pass
            def infer(self, skip=(), steps=10):
                self.calls.append((tuple(skip), steps))
                return np.zeros((50, 32)), np.zeros((50, 7)), {'cuda_policy_ms': float(steps)}
        backend = FakeBackend()
        policy = Policy(backend, 'libero_goal', (), 3, 'kalman_gate', 4,
                        'KalmanD3_Q50', 3., 3., np.ones(6)*.01,
                        np.ones(6)*.01, max_observation_age_ms=10., minimum_executed_history=4)
        observation = {'__trace_episode_idx__': 0, '__trace_infer_id__': 0,
                       '__reset_policy_state__': True, '__executed_actions_since_last_infer__': np.zeros((4, 7)).tolist(),
                       '__observation_time_monotonic_ms__': time.monotonic()*1000.-100.,
                       'observation/image': 0, 'observation/wrist_image': 0,
                       'observation/state': 0, 'prompt': 'test'}
        result = policy.infer(observation)
        self.assertEqual(result['accepted_prefix_len'], 12)
        self.assertEqual(result['policy_timing']['execution_gate_reason'], 'observation_too_old_or_invalid')
        self.assertEqual(backend.calls, [((), 3), ((), 10)])


if __name__ == '__main__':
    unittest.main()

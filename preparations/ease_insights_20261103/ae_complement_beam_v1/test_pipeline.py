"""CPU checks for replay identity, error weighting and prospective proxy gates."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from .common import MASKS, SUITES, cases, load_observation, sha
from .metrics import errors, aggregate
from .analyze import choose


RULE = dict(overall_task_mean_NMSE_max=.05, max_suite_mean_NMSE=.1,
            minimum_opposite_suite_mean_error_gap=.003, minimum_positive_tasks=6)


class PipelineTest(unittest.TestCase):
    def rows(self):
        return [dict(sample_id=f'{s}/{t}', suite=s, task=t, nmse=np.ones((3, 153))) for s in SUITES for t in range(10)]

    def arrays(self):
        return np.ones((3, 154, 50, 7), dtype=np.float32)

    def profile(self):
        tasks = {s: np.ones((10, 153)) for s in SUITES}
        for s in SUITES:
            tasks[s][:, :2] = .005
        tasks[SUITES[0]][:, 1] = .015
        tasks[SUITES[1]][:, 0] = .015
        means = {s: x.mean(axis=0) for s, x in tasks.items()}
        overall = np.stack(list(means.values())).mean(axis=0)
        return tasks, means, overall

    def test_all153_distinct_skip_two_masks(self):
        self.assertEqual(len(MASKS), 153)
        self.assertEqual(len({tuple(m) for m in MASKS}), 153)
        self.assertTrue(all(len(m) == 2 and 1 <= m[0] < m[1] <= 18 for m in MASKS))

    def test_task_shards_do_not_split_phases(self):
        obs = [dict(sample_id=f'{s}/{t}/{p}', suite=s, task=t, phase=p) for s in SUITES for t in range(10) for p in ('initial', 'contact')]
        m = dict(observations=obs)
        shards = [cases(m, 'screen', i, 2) for i in range(2)]
        self.assertEqual(sum(map(len, shards)), 80)
        keys = [{(x['suite'], x['task']) for x in shard} for shard in shards]
        self.assertFalse(keys[0] & keys[1])

    def test_nmse_uses_first5_and_all7_dimensions(self):
        a = self.arrays()
        a[:, 1:, :5] = 2
        np.testing.assert_array_equal(errors(a)['nmse'], np.ones((3, 153)))
        a[:, 1:, :5] = 1
        a[:, 1:, 5:] = 100
        self.assertEqual(float(errors(a)['nmse'].max()), 0)

    def test_zero_reference_uses_frozen_scale_floor(self):
        a = np.zeros_like(self.arrays())
        a[:, 1:, :5] = .01
        np.testing.assert_allclose(errors(a)['nmse'], 1, rtol=1e-6)

    def test_nonfinite_or_missing_mask_arrays_are_rejected(self):
        with self.assertRaises(ValueError):
            errors(self.arrays()[:, :-1])
        a = self.arrays()
        a[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            errors(a)

    def test_snapshot_counts_do_not_reweight_tasks(self):
        rows = self.rows()
        rows[0]['nmse'] = np.zeros((3, 153))
        rows.append(dict(rows[0], sample_id='additional_phase', nmse=np.full((3, 153), 2.)))
        tasks, means, overall = aggregate(rows, SUITES)
        np.testing.assert_array_equal(overall, np.ones(153))
        np.testing.assert_array_equal(tasks[SUITES[0]][0], np.ones(153))

    def test_duplicate_or_missing_task_rows_are_rejected(self):
        rows = self.rows()
        for changed in (rows + [rows[0]], rows[1:]):
            with self.assertRaises(ValueError):
                aggregate(changed, SUITES)

    def test_only_opposite_stable_suite_profiles_qualify(self):
        tasks, means, overall = self.profile()
        selected = choose(tasks, means, overall, SUITES, RULE)
        self.assertEqual(selected['status'], 'proxy_candidate')
        self.assertEqual(selected['selected']['mask_indices'], [0, 1])
        self.assertFalse(selected['quality_evidence'])

    def test_uniformly_better_mask_is_not_complementarity(self):
        tasks, _, _ = self.profile()
        for s in SUITES:
            tasks[s][:, 0] = .005
            tasks[s][:, 1] = .015
        means = {s: x.mean(axis=0) for s, x in tasks.items()}
        overall = np.stack(list(means.values())).mean(axis=0)
        self.assertEqual(choose(tasks, means, overall, SUITES, RULE)['status'], 'no_proxy_candidate')

    def test_few_positive_tasks_cannot_pass_suite_mean_gate_alone(self):
        tasks, _, _ = self.profile()
        tasks[SUITES[0]][:, 1] = .005
        tasks[SUITES[0]][:5, 1] = .025
        means = {s: x.mean(axis=0) for s, x in tasks.items()}
        overall = np.stack(list(means.values())).mean(axis=0)
        self.assertEqual(choose(tasks, means, overall, SUITES, RULE)['status'], 'no_proxy_candidate')

    def test_preprocessed_snapshot_contract_rejects_wrong_shapes_or_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'input.npz'
            data = dict(agent_image=np.zeros((224, 224, 3), dtype=np.uint8), wrist_image=np.zeros((224, 224, 3), dtype=np.uint8), robot_state=np.zeros(8, dtype=np.float32))
            np.savez(p, **data)
            row = dict(path=str(p), sha256=sha(p))
            self.assertEqual(load_observation(row)['robot_state'].shape, (8,))
            with self.assertRaises(ValueError):
                load_observation(dict(row, sha256='wrong'))
            np.savez(p, **dict(data, robot_state=np.zeros(7)))
            with self.assertRaises(ValueError):
                load_observation(dict(path=str(p), sha256=sha(p)))


if __name__ == '__main__':
    unittest.main()

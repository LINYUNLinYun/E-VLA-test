"""CPU exact-inference tests, including null/cluster failure cases."""
import copy
import unittest

from .analyze import apply_verdict, task_block_signflip


def direction(p=1/1024, positive=True, lower=True, loo=True):
    return dict(inference=dict(primary_task_block=dict(p_one_sided=p), sensitivity_case_mcnemar_p=.001,
        positive_direction=positive, task_cluster_lower_bound_positive=lower, every_leave_one_task_out_positive=loo))


class AnalysisTests(unittest.TestCase):
    def test_all_positive_blocks_exact_enumeration(self):
        result = task_block_signflip([1] * 10)
        self.assertEqual(result["extreme_assignments"], 1)
        self.assertEqual(result["p_one_sided"], 1/1024)
        self.assertEqual(result["observed_mean_difference"], .2)

    def test_zero_blocks_ties_are_included(self):
        self.assertEqual(task_block_signflip([0] * 10)["p_one_sided"], 1.)
        self.assertEqual(task_block_signflip([1] + [0] * 9)["p_one_sided"], .5)

    def test_negative_direction_is_not_significant(self):
        self.assertEqual(task_block_signflip([-1] * 10)["p_one_sided"], 1.)

    def test_requires_ten_five_case_task_blocks(self):
        for values in ([1] * 9, [6] * 10, [.2] * 10):
            with self.assertRaises(ValueError):
                task_block_signflip(values)

    def test_both_directions_and_usable_paths_required(self):
        directions = [direction(), direction()]
        self.assertEqual(apply_verdict(directions, {"A": dict(usable=True), "B": dict(usable=True)}), "complementarity_reproduced")
        for field in ("positive_direction", "task_cluster_lower_bound_positive", "every_leave_one_task_out_positive"):
            values = [direction(), direction()]
            values[1]["inference"][field] = False
            self.assertEqual(apply_verdict(values, {"A": dict(usable=True), "B": dict(usable=True)}), "not_reproduced")
        self.assertEqual(apply_verdict([direction(), direction()], {"A": dict(usable=False), "B": dict(usable=True)}), "not_reproduced")
        with self.assertRaises(ValueError):
            apply_verdict([direction(), direction()], {})

    def test_holm_uses_exactly_two_primary_tests(self):
        values = [direction(.03), direction(.001)]
        self.assertEqual(apply_verdict(values, {"A": dict(usable=True), "B": dict(usable=True)}), "complementarity_reproduced")
        self.assertEqual([d["inference"]["primary_holm2_p"] for d in values], [.03, .002])
        values = [direction(.025), direction(.025)]
        self.assertEqual(apply_verdict(values, {"A": dict(usable=True), "B": dict(usable=True)}), "not_reproduced")
        self.assertEqual([d["inference"]["primary_holm2_p"] for d in values], [.05, .05])


if __name__ == "__main__":
    unittest.main()

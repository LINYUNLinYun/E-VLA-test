"""Known exact nulls and cluster boundaries for prospective800-case confirmation."""
import unittest
from reports.ease_insights_20261103.camera_paths_v4.confirmation_math import bootstrap,task_signflip


class ConfirmationMathTest(unittest.TestCase):
    def test_all_zero_ties_preserve_all_assignments(self):
        r=task_signflip([0]*10);self.assertEqual(r['p_one_sided'],1);self.assertEqual(r['extreme_assignments'],1024)
    def test_four_positive_task_signs_do_not_reach_point_zero_five(self):
        self.assertEqual(task_signflip([5]*4+[0]*6)['p_one_sided'],1/16)
        self.assertEqual(task_signflip([1]*10)['p_one_sided'],1/1024)
    def test_integer_task_test_is_invariant_to_positive_scaling(self):
        x=[1,-1,0,1,1,0,1,0,-1,1]
        self.assertEqual(task_signflip(x)['p_one_sided'],task_signflip([5*v for v in x])['p_one_sided'])
    def test_task_cluster_zero_is_exact_with_heterogeneous_states(self):
        r=bootstrap([1,-1,1,-1,0]*10,[t for t in range(10) for _ in range(5)])
        self.assertEqual((r['task_cluster']['low'],r['task_cluster']['high']),(0.,0.))
        self.assertLess(r['case']['low'],0);self.assertGreater(r['case']['high'],0)
    def test_unanimous_task_interval_has_known_exact_endpoints(self):
        ids=[t for t in range(10) for _ in range(5)]
        for sign in (-1,1):
            r=bootstrap([sign]*50,ids)
            self.assertEqual((r['task_cluster']['low'],r['task_cluster']['high']),(sign,sign))
    def test_wrong_unit_counts_and_outcome_values_are_rejected(self):
        for counts in ([1]*9,[6]*10,[1.]*10):
            with self.assertRaises(ValueError):task_signflip(counts)
        with self.assertRaises(ValueError):bootstrap([0]*50,list(range(50)))
        with self.assertRaises(ValueError):bootstrap([2]*50,[t for t in range(10) for _ in range(5)])


if __name__=='__main__':unittest.main()

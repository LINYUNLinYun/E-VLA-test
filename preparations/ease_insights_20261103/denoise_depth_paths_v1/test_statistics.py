"""Prospective inference checks use integer task sums and the frozen two-test family."""
import unittest
from .statistics import bootstrap
from .analyze import task_block_signflip
from reports.ease_insights_20261103.denoise_depth_paths_v1.confirmation_statistics_audit import direction


class ConfirmationStatisticsContracts(unittest.TestCase):
    def compare(self,values):
        result=bootstrap(values,[t for t in range(10) for _ in range(3)],20261108,20261108)
        other=direction(values)
        for tag,reference in [('case','case_interval'),('task_cluster','task_interval')]:
            for bound in ('low','high'):self.assertEqual(result[tag][bound],other[reference][bound])
        p=task_block_signflip([sum(values[3*t:3*t+3]) for t in range(10)])['p_one_sided']
        self.assertEqual(p,other['p_one_sided']);return result,p

    def test_ties_preserve_zero_boundary_and_all1024_assignments(self):
        r,p=self.compare([0]*30);self.assertEqual(p,1.)
        self.assertEqual((r['task_cluster']['low'],r['task_cluster']['high']),(0.,0.))

    def test_strong_task_consistent_effect(self):
        r,p=self.compare([1,0,0]*10);self.assertEqual(p,1/1024);self.assertGreater(r['task_cluster']['low'],0)

    def test_mixed_cluster_counts_match_independent_integer_bootstrap(self):
        r,p=self.compare([1,0,0]*3+[0,0,0]*7);self.assertEqual(r['task_cluster']['low'],0.)

    def test_wrong_cluster_design_rejected(self):
        with self.assertRaises(ValueError):bootstrap([0]*30,list(range(30)),1,1)


if __name__=='__main__':unittest.main()

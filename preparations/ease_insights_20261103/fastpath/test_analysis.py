import unittest
from .analyze import exact_mcnemar_greater, paired_statistics

class Analysis(unittest.TestCase):
    def test_directional_exact_test(self):
        self.assertEqual(exact_mcnemar_greater(0,0),1.)
        self.assertEqual(exact_mcnemar_greater(4,0),1/16)
        self.assertEqual(exact_mcnemar_greater(0,4),1.)
    def test_identical_paths_have_zero_paired_and_cluster_effect(self):
        rows=[dict(task=t,init=i,A=bool(i%2),B=bool(i%2)) for t in range(3) for i in range(5)]
        result=paired_statistics(rows,'A','B',1000)
        self.assertEqual(result['difference'],0.)
        self.assertEqual(result['paired_episode_bootstrap_ci95'],[0.,0.])
        self.assertEqual(result['task_cluster_bootstrap_ci95'],[0.,0.])
        self.assertEqual(result['tasks_tied'],3)
    def test_cluster_sensitivity_retains_task_replication_unit(self):
        rows=[dict(task=t,init=i,A=t==0,B=t==1) for t in range(2) for i in range(5)]
        result=paired_statistics(rows,'A','B',1000)
        self.assertEqual((result['wins'],result['losses']),(5,5))
        self.assertEqual((result['tasks_better'],result['tasks_worse']),(1,1))
        self.assertEqual(result['task_cluster_bootstrap_ci95'],[-1.,1.])

if __name__=='__main__': unittest.main()

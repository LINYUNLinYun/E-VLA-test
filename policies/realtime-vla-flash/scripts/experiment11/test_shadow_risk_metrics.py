import unittest
import numpy as np
from scripts.experiment11.shadow_risk_metrics import differences,longest_run,task_cluster_summary


class RiskMetricsTest(unittest.TestCase):
    def test_identical_nonzero_commands_have_zero_continuous_error(self):
        a=np.ones((50,7))*.7
        self.assertTrue(np.all(differences(a,a)[:,:5]==0))

    def test_clips_each_command_before_scaling(self):
        a=np.zeros((50,7)); b=a.copy();a[:,0]=2;b[:,0]=1
        d=differences(a,b)
        self.assertTrue(np.all(d[:,0]==1))
        self.assertTrue(np.all(d[:,3]==0))
        b[:,0]=.5
        self.assertTrue(np.all(differences(a,b)[:,3]==25))

    def test_gripper_direction_and_zero_are_not_conflated(self):
        a=np.zeros((50,7));b=a.copy()
        a[0,6],b[0,6]=-1,1
        a[1,6],b[1,6]=1,-1
        a[2,6],b[2,6]=0,-1
        d=differences(a,b)
        self.assertEqual(d[0,5],1);self.assertEqual(d[1,6],1)
        self.assertEqual(d[2,7],1);self.assertEqual(d[2,8],0)
        self.assertEqual(longest_run([True,True,False,True]),2)

    def test_equal_tasks_not_equal_frames_and_stratified_cluster_interval(self):
        means,ci,n=task_cluster_summary(np.array([[1.],[1.],[9.]]),np.array(['a','a','b']),strata=np.array(['s','s','s']))
        self.assertEqual(n,2);self.assertEqual(means[0],5)
        self.assertLessEqual(ci[0,0],5);self.assertGreaterEqual(ci[1,0],5)


if __name__=='__main__':
    unittest.main()

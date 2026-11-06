import unittest

import numpy as np

from scripts.experiment9.probe_action_expert_layers import action_metrics, summarize_candidates


class ProbeMetricsTest(unittest.TestCase):
    def test_same_direction_wrong_amplitude_is_not_zero_error(self):
        baseline=np.ones((50,7))
        metrics=action_metrics(2*baseline,baseline)
        self.assertAlmostEqual(metrics['first12_nmse'],1)
        self.assertAlmostEqual(metrics['first50_nmse'],1)
        self.assertEqual(metrics['first12_gripper_flip'],0)

    def test_replan_window_ignores_unexecuted_tail(self):
        baseline=np.ones((50,7));variant=baseline.copy();variant[12:]*=-1
        metrics=action_metrics(variant,baseline)
        self.assertEqual(metrics['first12_nmse'],0)
        self.assertGreater(metrics['first50_nmse'],0)
        self.assertEqual(metrics['first12_gripper_flip'],0)
        self.assertAlmostEqual(metrics['first50_gripper_flip'],38/50)

    def test_task_cluster_weights_do_not_treat_noise_repeats_as_tasks(self):
        rows=[]
        for task,count,error in ((0,100,0.0),(1,1,1.0)):
            for _ in range(count):
                rows.append({'suite':'libero_object','task_id':task,'semantic_phase':'initial','layer':1,'first12_nmse':error,'first12_gripper_flip':0})
        result=summarize_candidates(rows)[0]
        self.assertEqual(result['task_count'],2)
        self.assertAlmostEqual(result['task_mean_nmse'],.5)
        self.assertFalse(result['eligible'])

    def test_missing_phases_cannot_pass_even_with_identical_actions(self):
        rows=[{'suite':'libero_object','task_id':t,'semantic_phase':'initial','layer':1,'first12_nmse':0,'first12_gripper_flip':0} for t in range(10)]
        self.assertFalse(summarize_candidates(rows)[0]['eligible'])


if __name__=='__main__':
    unittest.main()

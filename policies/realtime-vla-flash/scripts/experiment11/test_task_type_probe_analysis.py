import unittest
import numpy as np
from scripts.experiment11.analyze_task_type_probe import episode_task_means, ranks, spearman, centered_interaction
from scripts.experiment11.account_probe_internal_work import count_length_transitions


class ProbeAnalysisTests(unittest.TestCase):
    def test_runtime_reconstruction_counts_initial_and_length_changes(self):
        self.assertEqual(count_length_transitions([]),0)
        self.assertEqual(count_length_transitions([13,13,12,12,13]),3)
        self.assertEqual(count_length_transitions([13]*20),1)

    def test_episode_then_task_weighting(self):
        rows = [{'suite':'s','task_id':0,'episode_idx':0,'x':0.}]*10
        rows += [{'suite':'s','task_id':0,'episode_idx':1,'x':10.},
                 {'suite':'s','task_id':1,'episode_idx':0,'x':20.}]
        means, count = episode_task_means(rows,('x',))
        self.assertEqual(count,3)
        self.assertEqual(means['s',0][0],5.)
        self.assertEqual(means['s',1][0],20.)
        self.assertEqual(np.asarray(list(means.values())).mean(),12.5)

    def test_missing_denominator_is_not_zero_filled(self):
        rows = [{'suite':'s','task_id':0,'episode_idx':0,'x':None},
                {'suite':'s','task_id':0,'episode_idx':1,'x':2.}]
        means,_ = episode_task_means(rows,('x',))
        self.assertEqual(means['s',0][0],2.)

    def test_ranks_handle_ties_and_association_requires_variation(self):
        np.testing.assert_array_equal(ranks([1,1,2]),[1.5,1.5,3])
        self.assertEqual(spearman([1,1,1],[1,2,3]),(None,3))
        self.assertAlmostEqual(spearman([1,2,3],[3,2,1])[0],-1.)

    def test_interaction_removes_additive_main_effects(self):
        additive=np.arange(17)[:,None]+np.arange(4)[None,:]
        np.testing.assert_allclose(centered_interaction(additive),0.)
        changed=additive.astype(float);changed[0,0]+=1.
        result=centered_interaction(changed)
        self.assertGreater(result[0,0],0.)
        np.testing.assert_allclose(result.mean(axis=0),0.,atol=1e-15)
        np.testing.assert_allclose(result.mean(axis=1),0.,atol=1e-15)
        np.testing.assert_allclose(centered_interaction(np.stack([changed,changed])),np.stack([result,result]))


if __name__=='__main__':
    unittest.main()

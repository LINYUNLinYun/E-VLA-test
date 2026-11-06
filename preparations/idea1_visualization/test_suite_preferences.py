"""Scientific invariants for suite-profile comparisons."""
import unittest
import numpy as np

from .suite_preferences import (FEATURES,baseline_gain,decompose,depth_projection,
                                suite_tree,task_bootstrap)


class SuitePreferenceTests(unittest.TestCase):
    def setUp(self):
        self.keys=[(feature,layer) for feature in FEATURES for layer in (3,18)]
        self.projection=depth_projection(self.keys)
        self.modules=np.repeat(np.arange(4),2)

    def test_suite_difficulty_and_general_layer_strength_do_not_fake_preference(self):
        score=.4+np.arange(8)[:,None]*.02+np.array([-.04,.03,.02,-.01])[None,:]
        values=decompose(score,self.projection,self.modules)
        np.testing.assert_allclose(values['module'],0,atol=1e-14)
        np.testing.assert_allclose(values['depth'],0,atol=1e-14)

    def test_module_and_depth_interactions_are_separately_recovered(self):
        module=np.array([[.05,-.05,0,0],[-.05,.05,0,0],[0,0,.04,-.04],[0,0,-.04,.04]])
        depth=np.tile(np.array([[.03,-.02,-.01,0],[-.03,.02,.01,0]]),(4,1))
        score=.5+module[self.modules]+depth
        values=decompose(score,self.projection,self.modules)
        np.testing.assert_allclose(values['module'],module,atol=1e-14)
        np.testing.assert_allclose(values['depth'],depth,atol=1e-14)
        np.testing.assert_allclose(values['overall'],module[self.modules]+depth,atol=1e-14)

    def test_task_resampling_is_paired_across_layers(self):
        base=np.tile(np.arange(10)/20,(4,1))
        cases=np.stack([base,base+.1])
        draws=task_bootstrap(cases,draws=100,seed=1)
        np.testing.assert_allclose(draws[:,1]-draws[:,0],.1,atol=1e-14)

    def test_nonuniform_depth_sampling_preserves_linear_curve_area(self):
        keys=[(feature,layer) for feature in FEATURES for layer in (3,4,8,18)]
        projection=depth_projection(keys)
        values=np.tile((np.array([3,4,8,18])-3)/15,4)
        np.testing.assert_allclose(projection@values,.5,atol=1e-14)

    def test_baseline_normalization_uses_remaining_score_headroom(self):
        baseline=np.array([.2,.5])
        np.testing.assert_allclose(baseline_gain(baseline+.6*(1-baseline),baseline),.6)
        with self.assertRaises(ValueError):baseline_gain([1],[1])

    def test_nearly_equal_suite_fingerprints_merge_first(self):
        rng=np.random.default_rng(3)
        score=.5+rng.normal(0,.1,(8,4))
        score[:,1]=score[:,0]+rng.normal(0,1e-7,8)
        tree,distance=suite_tree(score,self.projection)
        self.assertEqual(set(tree[0,:2].astype(int)),{0,1})
        self.assertLess(distance[0,1],1e-8)


if __name__=='__main__':unittest.main()

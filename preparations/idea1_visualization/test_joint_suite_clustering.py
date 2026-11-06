"""Check shared-space fitting and held-out isolation for joint suite plots."""
import unittest
import numpy as np

from .joint_suite_clustering import (composition,fit_joint_projection,fit_partition,
                                    prepare_vectors)


class JointSuiteClusteringTests(unittest.TestCase):
    def test_heldout_vectors_do_not_change_pca_fit_or_training_coordinates(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(18,7));train=np.arange(18)<12
        first=fit_joint_projection(x,train)
        changed=x.copy();changed[~train]*=1000
        second=fit_joint_projection(changed,train)
        np.testing.assert_allclose(first[0].mean_,second[0].mean_,atol=1e-12)
        np.testing.assert_allclose(first[0].components_,second[0].components_,atol=1e-12)
        np.testing.assert_allclose(first[3][train],second[3][train],atol=1e-12)
        self.assertGreater(np.linalg.norm(first[3][~train]-second[3][~train]),1)

    def test_kmeans_centers_and_training_labels_ignore_heldout_points(self):
        z=np.array([[-2.,0],[-2.1,.1],[-1.9,-.1],[2.,0],[2.1,.1],[1.9,-.1],[0.,0]])
        train=np.arange(7)<6
        labels,centers,_=fit_partition(z,train,2,0)
        changed=z.copy();changed[~train]=1e6
        other,other_centers,_=fit_partition(changed,train,2,0)
        np.testing.assert_array_equal(centers,other_centers)
        np.testing.assert_array_equal(labels[train],other[train])

    def test_l2_is_per_observation_and_preserves_source(self):
        original=np.array([[3.,4.],[0.,0.],[-5.,12.]],np.float32)
        before=original.copy();value=prepare_vectors(original,'l2')
        np.testing.assert_array_equal(original,before)
        np.testing.assert_allclose(np.linalg.norm(value[[0,2]],axis=1),1)
        np.testing.assert_array_equal(value[1],0)
        self.assertTrue(np.isfinite(value).all())

    def test_composition_distinguishes_suite_and_cluster_denominators(self):
        labels=np.array([0,0,1,1,1]);suites=np.array([0,1,0,2,2])
        counts,by_suite,by_cluster=composition(labels,suites,3)
        np.testing.assert_array_equal(counts,[[1,1,0,0],[1,0,2,0],[0,0,0,0]])
        np.testing.assert_allclose(by_suite.sum(0),[1,1,1,0])
        np.testing.assert_allclose(by_cluster.sum(1),[1,1,0])
        self.assertAlmostEqual(by_suite[1,2],1.)
        self.assertAlmostEqual(by_cluster[1,2],2/3)


if __name__=='__main__':unittest.main()

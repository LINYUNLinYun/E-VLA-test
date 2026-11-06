"""Prevent held-out task labels leaking into cluster naming or classifier fits."""
import unittest
import numpy as np
from .task_pair_separability import align_binary_clusters,linear_predictions,phase_controls


class TaskPairTests(unittest.TestCase):
    def test_cluster_color_mapping_uses_training_labels_only(self):
        clusters=np.array([1,1,0,0,1,0]);labels=np.array([0,0,1,1,0,1]);train=np.array([1,1,1,1,0,0],bool)
        mapped,flip=align_binary_clusters(clusters,labels,train)
        changed=labels.copy();changed[~train]=1-changed[~train]
        other,other_flip=align_binary_clusters(clusters,changed,train)
        self.assertTrue(flip);self.assertEqual(flip,other_flip)
        np.testing.assert_array_equal(mapped,other)
        np.testing.assert_array_equal(mapped[train],labels[train])

    def test_linear_fit_does_not_use_heldout_labels(self):
        x=np.array([[-2.],[-1.9],[2.],[1.9],[-2.1],[2.1]])
        labels=np.array([0,0,1,1,0,1]);train=np.array([1,1,1,1,0,0],bool)
        before=linear_predictions(x,labels,train)
        labels[~train]=1-labels[~train]
        np.testing.assert_array_equal(before,linear_predictions(x,labels,train))

    def test_same_phase_control_does_not_report_single_class_test_as_separable(self):
        x=np.arange(6,dtype=float)[:,None];labels=np.array([0,0,1,1,0,1])
        train=np.array([1,1,1,1,0,0],bool)
        phases=np.array(['initial','initial','initial','initial','initial','approach'])
        result=phase_controls(x,labels,train,phases)
        self.assertIsNone(result[0]['balanced_accuracy'])
        self.assertEqual(result[0]['test_classes'],1)


if __name__=='__main__':unittest.main()

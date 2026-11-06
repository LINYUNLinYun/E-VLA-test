"""Check that representation diagnostics preserve held-out and conditioning rules."""
import unittest

import numpy as np

from .analyze import classifier,conditional_probes,fit_projection,project


class AnalysisTests(unittest.TestCase):
    def test_rank_one_repeated_inputs_do_not_scatter_on_null_pca_axes(self):
        rng=np.random.default_rng(5)
        values=rng.normal(size=(2,100)).astype(np.float32)
        x=np.repeat(values,12,axis=0)
        pca,active,_=fit_projection(x,maximum_dimensions=10)
        z=project(pca,active,x)
        self.assertEqual(int(active.sum()),1)
        self.assertEqual(int(np.count_nonzero(z[:,~active])),0)
        np.testing.assert_allclose(z[:12],np.repeat(z[:1],12,axis=0),atol=1e-12)

    def test_heldout_labels_are_not_used_to_fit_classifier(self):
        x=np.array([[0.,0.],[.1,0.],[1.,1.],[1.1,1.],[.05,0.],[1.05,1.]])
        labels=np.array([0,0,1,1,1,0])
        train=np.array([1,1,1,1,0,0],bool)
        score=classifier(x,labels,train,~train)
        self.assertEqual(score['accuracy'],0.)
        self.assertEqual(score['n_train'],4)
        self.assertEqual(score['n_test'],2)

    def test_unseen_phase_in_heldout_is_visible_in_coverage(self):
        x=np.array([[0.],[.1],[1.],[1.1],[.05],[2.]])
        labels=np.array(['approach','approach','transport','transport','approach','release'])
        train=np.array([1,1,1,1,0,0],bool)
        score=classifier(x,labels,train,~train)
        self.assertEqual(score['test_unseen_class_fraction'],.5)
        self.assertEqual(score['test_classes'],2)

    def test_single_training_class_is_not_reported_as_estimated_probe(self):
        score=classifier(np.array([[0.],[1.]]),np.array(['approach','approach']),
                         np.array([True,False]),np.array([False,True]))
        self.assertIsNone(score['accuracy'])
        self.assertEqual(score['train_classes'],1)

    def test_conditioning_does_not_mix_task_or_phase_groups(self):
        task=np.array([0,0,0,1,1,1]);phase=np.array(['initial','approach','approach']*2)
        train=np.array([1,1,0,1,1,0],bool)
        result=conditional_probes(np.arange(6.)[:,None],task,phase,train,~train,
                                   feature='test',layer=0,suite='test')
        first=next(r for r in result if r['condition']=='within_task' and r['condition_value']=='0')
        self.assertEqual((first['n_train'],first['n_test']),(2,1))
        second=next(r for r in result if r['condition']=='within_phase' and r['condition_value']=='approach')
        self.assertEqual((second['n_train'],second['n_test']),(2,2))


if __name__=='__main__':
    unittest.main()

import unittest
from unittest.mock import patch
import numpy as np
import torch
from .model import calculate
from .scoring import score_and_choose, refresh_due
from preparations.idea1_v3_4_E2_adjacency_200.common import combinations, legal


class Mechanisms(unittest.TestCase):
    def test_signed_combination_keeps_cancellation(self):
        g=np.full((4,17),100.,np.float32)
        g[:,[0,2]]=[2.,-2.]
        g[:,[4,6]]=[.2,.2]
        a,_,_=score_and_choose(g,2,'layer_rms_sum')
        b,_,scores=score_and_choose(g,2,'signed_sum_rms')
        self.assertEqual(a,[5,7])
        self.assertEqual(b,[1,3])
        self.assertEqual(float(scores.min()),0.)

    def test_combo_matches_independent_enumeration(self):
        g=np.random.default_rng(33).normal(size=(4,17)).astype(np.float32)
        for k in (4,6):
            chosen,_,scores=score_and_choose(g,k,'signed_sum_rms')
            expected=np.array([np.sqrt(np.mean(np.square(g[:,m].sum(axis=1))))
                for m in combinations(k,'spaced')])
            np.testing.assert_allclose(scores,expected,rtol=1e-6,atol=1e-6)
            self.assertTrue(legal(chosen,k,'spaced'))
            self.assertEqual(chosen,(combinations(k,'spaced')[np.argmin(expected)]+1).tolist())

    def test_refresh_counts_actual_fulls(self):
        for stride in (1,2,3):
            fulls=0;observed=[]
            for route in ['full','fast','full','full','fast','fast','full','full','full']:
                if route=='full':
                    if refresh_due(fulls,stride):observed.append(fulls)
                    fulls+=1
            self.assertEqual(observed,list(range(0,6,stride)))

    def test_batched_reverse_uses_one_forward_and_all_four_signs(self):
        class Tiny:
            def __init__(self):self.calls=0
            def forward(self,item,gates):
                self.calls+=1
                return torch.sin(item['a']@gates).reshape(5,7)
        rng=np.random.default_rng(5)
        a=torch.as_tensor(rng.normal(size=(35,17)),dtype=torch.float64)
        w=torch.as_tensor(rng.choice([-1.,1.],size=(4,5,7)),dtype=torch.float64)
        results=[]
        for batch_size in (1,2,4):
            model=Tiny();g=torch.ones(17,dtype=torch.float64,requires_grad=True)
            raw,grad,_=calculate(model,{'a':a},w,g,batch_size)
            self.assertEqual(model.calls,1)
            results.append((raw,grad))
        for raw,grad in results[1:]:
            torch.testing.assert_close(raw,results[0][0],rtol=0.,atol=0.)
            torch.testing.assert_close(grad,results[0][1],rtol=1e-12,atol=1e-12)

    def test_real_policy_refreshes_history_on_unscored_and_fallback_fulls(self):
        from . import policy as module
        from types import SimpleNamespace
        class FakeSelector:
            q=4;rule='spaced'
            def clear(self):pass
            def run(self,c,seed,coeff,k):
                g=np.zeros((4,17),np.float32)
                chosen,score,self.last_combination_scores=score_and_choose(g,k)
                return chosen,g,score,dict(selector_total_ms=1.,selector_hot_ms=1.,selector_capture_ms=0.)
        def context(*args,**kwargs):return dict(hidden_trace=[]),{}
        class FakeBackend:
            adapter=SimpleNamespace(measure=lambda fn:(fn(),0.))
            def sample(self,c,seed,steps):
                return np.zeros((50,7),np.float32),torch.zeros((50,32)),{}
            def verify(self,*args):return np.zeros((2,50,32)),{}
        for stride in (2,3):
            with patch.object(module,'prepare_full',context),patch.object(module,'prepare_patch',context),\
                 patch.object(module,'capture',lambda hidden:{i:torch.tensor(0.) for i in range(1,18)}):
                pol=module.Policy(FakeBackend());pol.selector=FakeSelector()
                cfg=dict(mode='verified',period=3,compensation='extrapolate',selection='jacobian',budget=14,
                    tau=.10,protect_gripper=True,projections=4,selection_stride=stride,vjp_batch_size=1,
                    aggregation='layer_rms_sum')
                pol.configure(cfg);seen=[];full_steps=[]
                for request in range(14):
                    # Force some early Fulls, independent of the nominal cadence.
                    length=0 if request in (1,5,9) else 5
                    detail=dict(verify_gripper=False,post_verify_gripper=False)
                    with patch.object(module,'acceptance',return_value=(length,detail)):
                        result=pol.predict({},'test','libero_spatial',request)
                    rt=result['route']
                    if rt['path']=='full':
                        index=len(full_steps);full_steps.append(result['step'])
                        self.assertEqual(rt['mask_refresh'],index%stride==0)
                        self.assertEqual(rt['history_full_steps_after'],full_steps[-2:])
                        if rt['mask_refresh']:seen.append(index)
                    pol.acknowledge(result['actions'][:5],np.zeros((5,4)))
                self.assertEqual(seen,list(range(0,len(full_steps),stride)))
                self.assertEqual(pol.counts['vjp'],4*len(seen))


if __name__=='__main__':unittest.main()

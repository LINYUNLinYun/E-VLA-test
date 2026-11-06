"""CPU-only cadence/configuration isolation;no model construction or GPU."""
import unittest
import numpy as np
import torch
from .common import conditions,FULL
from .policy import Policy


class FakeBackend:
    def __init__(self,nan=False):self.requests=[];self.nfes=[];self.nan=nan;self.pending=None
    def reset_vision(self):self.pending=None
    def request_vision(self,r):
        if self.pending is not None:raise RuntimeError('Unconsumed fake request')
        self.pending=r;self.requests.append(r)
    def prepare_residual(self,*args):
        r=self.pending;self.pending=None
        audit=dict(kind=r['kind'],fresh_images_consumed=True,numerical_reference_only=False,
            actual_vision_layers_by_camera=[list(FULL),list(FULL)] if r['kind']=='native' else r['layers'],vision_cache='none')
        return dict(execution_audit=dict(qkv_layers=list(range(1,19))),camera_execution_audit=audit),dict(vision_ms=1.)
    def sample(self,context,seed,steps,ae_skip=()):
        self.nfes.append(steps);actions=np.arange(350,dtype=np.float32).reshape(50,7)
        if self.nan:actions[0,0]=np.nan
        return actions,torch.zeros(50,32),dict(actual_NFE=steps,ae_execution_audit=dict(actual_NFE=steps,actual_layer_calls=18*steps),camera_execution_audit=context['camera_execution_audit'])


class FreshPolicy(unittest.TestCase):
    def test_all_four_configs_use_full_then_two_fast_except_full_control(self):
        for cfg in conditions():
            b=FakeBackend();p=Policy(b);p.configure(cfg);replies=[]
            for i in range(6):
                r=p.predict({},'test','libero_spatial',17+i);replies.append(r);p.acknowledge(r['actions'][:5])
            expected=[10]*6 if cfg['id']=='full_d10' else [10,2,2,10,2,2]
            self.assertEqual(b.nfes,expected);self.assertEqual([r['step'] for r in replies],list(range(0,30,5)))
            for r in replies:self.assertFalse(r['route']['camera_cache_used']);self.assertIsNone(r['route']['compensation'])
            if cfg['id']!='full_d10':self.assertEqual(replies[4]['route']['history_steps'],[0,15])
    def test_candidate_fast_rounds_swap_only_camera_depth(self):
        for cfg in conditions()[2:]:
            b=FakeBackend();p=Policy(b);p.configure(cfg)
            for i in range(3):r=p.predict({},'test','libero_object',1);p.acknowledge(r['actions'][:5])
            self.assertEqual(b.requests[0],dict(kind='native'))
            self.assertEqual(b.requests[1:],[dict(kind='allocated',layers=cfg['vision_kept'])]*2)
            self.assertEqual(sum(map(len,cfg['vision_kept'])),45)
    def test_dense_d2_control_keeps_original_native_batch2(self):
        b=FakeBackend();p=Policy(b);p.configure(conditions()[1])
        for i in range(4):r=p.predict({},'test','libero_goal',1);p.acknowledge(r['actions'][:5])
        self.assertEqual(b.requests,[dict(kind='native')]*4)
    def test_no_new_prediction_before_exact_execution_ack(self):
        p=Policy(FakeBackend());p.configure(conditions()[2]);r=p.predict({},'test','libero_spatial',1)
        with self.assertRaises(RuntimeError):p.predict({},'test','libero_spatial',2)
        with self.assertRaises(ValueError):p.acknowledge(r['actions'][:5]+1)
        p.acknowledge(r['actions'][:3]);self.assertEqual(p.step,3)
    def test_configuration_reset_starts_fresh_full(self):
        b=FakeBackend();p=Policy(b);p.configure(conditions()[2]);r=p.predict({},'test','libero_10',1);p.acknowledge(r['actions'][:5]);p.configure(conditions()[3]);r=p.predict({},'test','libero_10',1)
        self.assertEqual((r['round'],r['step']),(0,0));self.assertTrue(r['route']['refresh']);self.assertEqual(r['counts'],dict(full=1,draft=0,verify=0))
    def test_invalid_mask_or_nonfinite_output_rejected(self):
        cfg=conditions()[2];cfg['vision_kept'][1]=list(range(1,19));p=Policy(FakeBackend())
        with self.assertRaises(ValueError):p.configure(cfg)
        p=Policy(FakeBackend(nan=True));p.configure(conditions()[2])
        with self.assertRaises(ValueError):p.predict({},'test','libero_spatial',1)
        self.assertIsNone(p.pending);self.assertEqual(p.round,0)


    def test_configured_layer_lists_are_not_mutated_by_the_caller(self):
        cfg=conditions()[2];p=Policy(FakeBackend());p.configure(cfg);cfg['vision_kept'][1].clear()
        self.assertEqual(len(p.config['vision_kept'][1]),18)


if __name__=='__main__':unittest.main()

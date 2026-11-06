"""CPU-only orchestration checks;real model/GPU gates remain untested."""
from contextlib import contextmanager
import unittest
from unittest.mock import patch
import numpy as np
import torch
from .common import conditions
from .policy import Policy
from .backend import Backend,FreshBackend


class FakeBackend:
    def __init__(self):self.calls=[];self.requests=[];self.fresh=0;self.invalid=False
    def reset_vision(self):pass
    def request_vision(self,r):self.requests.append(r)
    def prepare_residual(self,obs,instruction):
        self.fresh+=1;return dict(execution_audit={'fresh_counter':self.fresh}),{}
    def sample(self,context,seed,steps,ae_skip=(),prefix_pattern='all'):
        self.calls.append((steps,prefix_pattern));raw=torch.zeros((50,32));raw[0,0]=float('nan') if self.invalid else seed
        return np.zeros((50,7),np.float32),raw,dict(ae_execution_audit={'steps':steps},camera_execution_audit={'fresh_counter':self.fresh},prefix_execution_audit={'pattern':prefix_pattern})


class IntegrationTest(unittest.TestCase):
    def configured(self,cid):
        b=FakeBackend();p=Policy(b);cfg=next(c for c in conditions() if c['id']==cid);p.configure(cfg);return b,p,cfg
    def test_candidates_only_pack_two_fast_rounds_and_all_images_are_fresh(self):
        for cid,pattern in (('center128_d2','center_columns'),('checker128_d2','checkerboard')):
            b,p,_=self.configured(cid)
            for i in range(6):
                r=p.predict({},'p','s',i);self.assertEqual(r['step'],5*i);p.acknowledge(r['actions'][:5])
            self.assertEqual(b.calls,[(10,'all'),(2,pattern),(2,pattern)]*2)
            self.assertEqual(b.requests,[dict(kind='native')]*6);self.assertEqual(b.fresh,6)
    def test_controls_keep_full_prefix_with_original_d10_d2_cadence(self):
        for cid,wanted in (('full_d10',[(10,'all')]*3),('dense_d2',[(10,'all'),(2,'all'),(2,'all')])):
            b,p,_=self.configured(cid)
            for i in range(3):r=p.predict({},'p','s',i);p.acknowledge(r['actions'][:5])
            self.assertEqual(b.calls,wanted)
    def test_configuration_is_deepcopied_and_reset_restores_full(self):
        b,p,cfg=self.configured('center128_d2');cfg['prefix_pattern']='checkerboard'
        for i in range(2):r=p.predict({},'p','s',i);p.acknowledge(r['actions'][:5])
        self.assertEqual(b.calls[-1],(2,'center_columns'));p.configure(next(c for c in conditions() if c['id']=='center128_d2'))
        r=p.predict({},'p','s',3);self.assertEqual(r['round'],0);self.assertEqual(r['step'],0);self.assertEqual(b.calls[-1],(10,'all'))
    def test_pending_proposal_and_acknowledged_action_prefix_are_enforced(self):
        _,p,_=self.configured('center128_d2');r=p.predict({},'p','s',0)
        with self.assertRaises(RuntimeError):p.predict({},'p','s',1)
        with self.assertRaises(ValueError):p.acknowledge(np.ones((5,7),np.float32))
        p.acknowledge(r['actions'][:2]);self.assertEqual(p.step,2)
    def test_nonfinite_outputs_do_not_advance_round(self):
        b,p,_=self.configured('center128_d2');b.invalid=True
        with self.assertRaises(ValueError):p.predict({},'p','s',0)
        self.assertEqual(p.round,0);self.assertIsNone(p.pending)
    def test_backend_exports_real_sampler_fields_and_full_vision_identity(self):
        b=object.__new__(Backend);b.adapter=b.prefix_visits=b.prefix_packer=object()
        prefix=dict(packing_wall_ms=2.,packing_cuda_ms=1.,restore_wall_ms=.1)
        result=(np.zeros((50,7),np.float32),torch.zeros((50,32)),None,dict(nodes=[0,5],actual_NFE=2),3.,.2)
        camera=dict(kind='native');ctx=dict(camera_execution_audit=camera)
        with patch('reports.ease_insights_20261103.prefix_kv_preparation_v5.backend.sample_with_prefix',return_value=(result,dict(actual_NFE=2),prefix)) as call:
            _,_,cost=b.sample(ctx,7,2,prefix_pattern='checkerboard')
        self.assertEqual(call.call_args.args[-1],'checkerboard');self.assertIs(cost['camera_execution_audit'],camera)
        self.assertIs(cost['prefix_execution_audit'],prefix);self.assertEqual(cost['prefix_pack_wall_ms'],2.)
    def test_ae_skipping_or_missing_fresh_identity_rejects_before_sampling(self):
        b=object.__new__(Backend)
        for ctx,skip in (({},()),({'camera_execution_audit':{}},(1,))):
            with self.assertRaises(ValueError):b.sample(ctx,7,2,ae_skip=skip)
    def test_masked_reference_requires_all_thirty_six_calls(self):
        b=object.__new__(Backend)
        class Oracle:
            def __init__(self,n):self.n=n
            @contextmanager
            def use(self,seq,pattern):
                d=dict(calls=[{}]*self.n,original_helper_restored=False);yield d;d['original_helper_restored']=True
        with patch.object(FreshBackend,'sample_native_reference',return_value=('actions','raw','cost')):
            b.masked_prefix_oracle=Oracle(36);r=b.sample_masked_reference({'seq':519},7,'checkerboard');self.assertEqual(len(r),4)
            b.masked_prefix_oracle=Oracle(35)
            with self.assertRaises(ValueError):b.sample_masked_reference({'seq':519},7,'checkerboard')


if __name__=='__main__':unittest.main()

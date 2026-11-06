"""CPU-only wiring tests; no real model,environment,native kernel or CUDA context."""
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from reports.ease_insights_20261103.prefix_kv_preparation_v2.packing import PrefixPacker
from reports.ease_insights_20261103.prefix_kv_preparation_v2.test_packing import fixture
from .sampling import PrefixVisits,sample_with_prefix


class FakeAdapter:
    def __init__(self,namespace,fail=False,wrong_shape=False):
        self.namespace=namespace;self.fail=fail;self.wrong_shape=wrong_shape;self.calls=[]
    def sample(self,context,seed,schedule,instrumented):
        self.calls.append((seed,schedule,instrumented));buffers=context['runtime'].buffers;seq=context['seq']
        buffers['encoder_K'][:,:seq].copy_(context['K']);buffers['encoder_V'][:,:seq].copy_(context['V'])
        if self.wrong_shape:buffers['decoder_attn_buf']=torch.empty(1)
        for step in range(schedule['num_steps']):
            for layer in range(18):
                self.namespace._decoder_layer({},buffers,seq,layer)
                if self.fail:raise RuntimeError('sample failed')
        value=float(context['K'][0,:,0].mean())
        return (np.full((50,7),value,np.float32),torch.full((50,32),value),np.zeros((50,32),np.float32),
            dict(actual_NFE=schedule['num_steps'],nodes='fake nodes'),1.,1.)


class Sampling(unittest.TestCase):
    def setup_context(self,**kwargs):
        context=fixture(True);called=[]
        def native(weights,buffers,seq,index):called.append((seq,index))
        ns=SimpleNamespace(_decoder_layer=native);observer=PrefixVisits(ns);adapter=FakeAdapter(ns,**kwargs)
        return context,ns,native,called,observer,adapter,PrefixPacker()

    def test_all_keeps_original_storage_and_exact_adapter_call(self):
        c,ns,fn,calls,observer,adapter,packer=self.setup_context();original=dict(c['runtime'].buffers)
        result,ae,audit=sample_with_prefix(adapter,observer,packer,c,123,10)
        self.assertEqual(adapter.calls,[(123,dict(kind='native',m=None,num_steps=10),False)])
        self.assertEqual(calls,[(519,i) for _ in range(10) for i in range(18)])
        self.assertEqual(ae['actual_layer_calls'],180);self.assertEqual(audit['actual_total_key_tokens'],570)
        self.assertEqual(audit['packing_wall_ms'],0.);self.assertFalse(audit['packed'])
        self.assertFalse(packer.cache);self.assertIs(ns._decoder_layer,fn)
        self.assertTrue(all(c['runtime'].buffers[k] is v for k,v in original.items()))

    def test_identity_pack_matches_fake_unmodified_d2_d10(self):
        c,ns,fn,calls,observer,adapter,packer=self.setup_context()
        for nfe in (2,10):
            before=adapter.sample(c,1,dict(kind='native',m=None,num_steps=nfe),False)
            after,ae,audit=sample_with_prefix(adapter,observer,packer,c,1,nfe,'identity')
            self.assertTrue(np.array_equal(before[0],after[0]));self.assertTrue(torch.equal(before[1],after[1]))
            self.assertEqual(audit['actual_prefix_length'],519);self.assertEqual(ae['actual_NFE'],nfe)

    def test_center_and_checker_native_calls_consume_compact_shape(self):
        c,ns,fn,calls,observer,adapter,packer=self.setup_context()
        for pattern in ('center_columns','checkerboard'):
            calls.clear();result,ae,audit=sample_with_prefix(adapter,observer,packer,c,3,2,pattern)
            self.assertEqual(calls,[(263,i) for _ in range(2) for i in range(18)])
            self.assertEqual(audit['actual_total_key_tokens'],314);self.assertEqual(audit['actual_helper_calls'],36)
            self.assertEqual(audit['decoder_absolute_positions'],list(range(519,570)))
            self.assertEqual(audit['visual_tokens_per_camera'],[128,128]);self.assertTrue(audit['original_buffers_restored'])
            self.assertIsNone(audit['packing_cuda_ms']);self.assertIs(ns._decoder_layer,fn)

    def test_sampler_exception_restores_helper_and_original_buffers(self):
        c,ns,fn,calls,observer,adapter,packer=self.setup_context(fail=True);original=dict(c['runtime'].buffers)
        with self.assertRaisesRegex(RuntimeError,'sample failed'):sample_with_prefix(adapter,observer,packer,c,1,2,'center_columns')
        self.assertIs(ns._decoder_layer,fn);self.assertFalse(packer.active)
        self.assertTrue(all(c['runtime'].buffers[k] is v for k,v in original.items()));self.assertFalse(observer.lock.locked())

    def test_wrong_actual_shape_is_rejected_and_restored(self):
        c,ns,fn,calls,observer,adapter,packer=self.setup_context(wrong_shape=True);original=dict(c['runtime'].buffers)
        with self.assertRaises(Exception):sample_with_prefix(adapter,observer,packer,c,1,2,'checkerboard')
        self.assertIs(ns._decoder_layer,fn);self.assertTrue(all(c['runtime'].buffers[k] is v for k,v in original.items()))
        self.assertFalse(observer.lock.locked());self.assertFalse(packer.active)

    def test_full_cannot_silently_drop_prefix(self):
        c,ns,fn,calls,observer,adapter,packer=self.setup_context()
        with self.assertRaises(ValueError):sample_with_prefix(adapter,observer,packer,c,1,10,'checkerboard')
        self.assertEqual(adapter.calls,[]);self.assertFalse(packer.cache)

    def test_full_after_packed_request_sees_original_shapes(self):
        c,ns,fn,calls,observer,adapter,packer=self.setup_context()
        sample_with_prefix(adapter,observer,packer,c,1,2,'center_columns');calls.clear()
        result,ae,audit=sample_with_prefix(adapter,observer,packer,c,2,10,'all')
        self.assertEqual(calls,[(519,i) for _ in range(10) for i in range(18)])
        self.assertEqual(audit['actual_total_key_tokens'],570)

    def test_prepatched_namespace_is_not_overwritten(self):
        c,ns,fn,calls,observer,adapter,packer=self.setup_context();external=lambda *args:None;ns._decoder_layer=external
        with self.assertRaises(ValueError):sample_with_prefix(adapter,observer,packer,c,1,2,'center_columns')
        self.assertIs(ns._decoder_layer,external);self.assertFalse(observer.lock.locked());self.assertFalse(packer.active)


    def test_invalid_context_does_not_leak_observer_lock(self):
        c,ns,fn,calls,observer,adapter,packer=self.setup_context()
        del c['runtime'].buffers['decoder_rope_weights']
        from reports.ease_insights_20261103.prefix_kv_preparation_v1.layout_contract import plan
        with self.assertRaises(KeyError):
            with observer.use(c,plan('identity',519)):pass
        self.assertFalse(observer.lock.locked());self.assertIs(ns._decoder_layer,fn)


if __name__=='__main__':unittest.main()

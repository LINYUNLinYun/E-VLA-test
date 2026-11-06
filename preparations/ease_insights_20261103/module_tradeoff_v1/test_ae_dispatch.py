"""CPU-only dispatch/cleanup contracts; no numerical or scientific evidence."""
from types import SimpleNamespace
import unittest
from .ae_dispatch import ScopedLayers,checked_skip,sample_with_dispatch


class DispatchContracts(unittest.TestCase):
    def make(self):
        actual=[]
        def native(w,b,seq,index):actual.append(index+1);return index+100
        namespace=SimpleNamespace(_decoder_layer=native)
        return namespace,ScopedLayers(namespace),actual,native

    def test_full_dispatch_preserves_order_and_returns(self):
        ns,scope,actual,native=self.make()
        with scope.use(()) as visits:
            for _ in range(10):
                for i in range(18):self.assertEqual(ns._decoder_layer(None,None,512,i),i+100)
        receipt=visits.receipt(10)
        self.assertEqual(actual,list(range(1,19))*10)
        self.assertEqual(receipt['actual_layer_calls'],180)
        self.assertIs(ns._decoder_layer,native)

    def test_d2_bypass_is_actual_and_one_based(self):
        ns,scope,actual,native=self.make();skip=(2,6,7,8,9,16)
        with scope.use(skip) as visits:
            for _ in range(2):
                for i in range(18):ns._decoder_layer(None,None,512,i)
        receipt=visits.receipt(2)
        self.assertEqual(receipt['actual_layer_calls'],24)
        self.assertEqual(receipt['bypassed_layer_calls'],12)
        self.assertFalse(set(actual)&set(skip))
        self.assertIs(ns._decoder_layer,native)

    def test_native_exception_restores_owned_binding(self):
        ns,scope,actual,native=self.make()
        with self.assertRaisesRegex(RuntimeError,'injected'):
            with scope.use((2,)):
                ns._decoder_layer(None,None,512,0)
                raise RuntimeError('injected')
        self.assertIs(ns._decoder_layer,native)
        with scope.use(()) as retry:
            for _ in range(2):
                for i in range(18):ns._decoder_layer(None,None,512,i)
        self.assertEqual(retry.receipt(2)['actual_layer_calls'],36)

    def test_reentrant_use_is_rejected_without_uninstalling_outer(self):
        ns,scope,actual,native=self.make()
        with scope.use(()) as visits:
            wrapper=ns._decoder_layer
            with self.assertRaises(RuntimeError):
                with scope.use((2,)):pass
            self.assertIs(ns._decoder_layer,wrapper)
        self.assertIs(ns._decoder_layer,native)

    def test_unknown_replacement_not_adopted(self):
        ns,scope,actual,native=self.make();other=lambda *a:None;ns._decoder_layer=other
        with self.assertRaises(RuntimeError):
            with scope.use(()):pass
        self.assertIs(ns._decoder_layer,other)

    def test_order_or_nfe_mismatch_cannot_pass_receipt(self):
        ns,scope,actual,native=self.make()
        with scope.use(()) as visits:
            for _ in range(2):
                for i in reversed(range(18)):ns._decoder_layer(None,None,512,i)
        with self.assertRaises(RuntimeError):visits.receipt(2)
        with self.assertRaises(ValueError):visits.receipt(3)

    def test_mask_validation_precedes_any_binding_change(self):
        ns,scope,actual,native=self.make()
        for bad in ((0,),(19,),(2,2),(3,2),(True,),tuple(range(1,8))):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                with scope.use(bad):pass
            self.assertIs(ns._decoder_layer,native)
        self.assertEqual(checked_skip((2,8)),(2,8))

    def test_original_sampler_called_once_with_unchanged_schedule(self):
        ns,scope,actual,native=self.make();calls=[];sentinel=object()
        class Adapter:
            def sample(self,context,seed,schedule,instrumented):
                calls.append((context,seed,schedule,instrumented))
                for _ in range(schedule['num_steps']):
                    for layer in range(18):ns._decoder_layer(None,None,512,layer)
                return sentinel,None,None,dict(actual_NFE=schedule['num_steps']),1.,2.
        result,receipt=sample_with_dispatch(Adapter(),scope,'context',7,2,(2,8))
        self.assertIs(result[0],sentinel)
        self.assertEqual(calls,[('context',7,dict(kind='native',m=None,num_steps=2),False)])
        self.assertEqual(receipt['actual_layer_calls'],32)

    def test_full_mask_rejected_before_sampler_or_binding(self):
        ns,scope,actual,native=self.make()
        class Adapter:
            def sample(self,*a):raise AssertionError('Must not call')
        with self.assertRaises(ValueError):sample_with_dispatch(Adapter(),scope,None,7,10,(2,))
        self.assertIs(ns._decoder_layer,native)

    def test_sampler_cannot_report_unobserved_nfe(self):
        ns,scope,actual,native=self.make()
        class Adapter:
            def sample(self,*a):return None,None,None,dict(actual_NFE=2),1.,2.
        with self.assertRaises(RuntimeError):sample_with_dispatch(Adapter(),scope,None,7,2,())
        self.assertIs(ns._decoder_layer,native)


if __name__=='__main__':unittest.main()

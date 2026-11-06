"""CPU-only ordered-call and cleanup tests; no GPU or quality claim."""
from types import SimpleNamespace
import unittest
from preparations.ease_insights_20261103.module_tradeoff_v1.ae_dispatch import ScopedLayers
from .dispatch import sample_with_dispatch
from .nodes import scheduled_nfe


class FutureDispatchContracts(unittest.TestCase):
    def setup_adapter(self,reverse=False,missing=False,fail=False):
        actual=[];ns=SimpleNamespace(_decoder_layer=lambda w,b,s,i:actual.append(i+1));scope=ScopedLayers(ns);original=ns._decoder_layer
        class Adapter:
            def sample(self,context,seed,schedule,instrumented):
                if fail:raise RuntimeError('injected')
                for _ in range(schedule['num_steps']):
                    for i in (reversed(range(18)) if reverse else range(18)):
                        if not missing or i!=17:ns._decoder_layer(None,None,512,i)
                return None,None,None,dict(actual_NFE=schedule['num_steps']),1.,2.
        return Adapter(),scope,ns,original,actual

    def test_all_proposed_schedules_keep_original_call_order(self):
        for steps,skip in ((2,()),(3,(2,6,7,8)),(4,(2,4,6,7,8,9)),(10,())):
            a,s,ns,original,actual=self.setup_adapter();_,r=sample_with_dispatch(a,s,None,7,steps,skip)
            self.assertEqual(r['actual_layer_calls'],(18-len(skip))*steps);self.assertIs(ns._decoder_layer,original)
            self.assertEqual(actual,[l for l in range(1,19) if l not in skip]*steps)

    def test_missing_call_is_rejected(self):
        a,s,ns,original,_=self.setup_adapter(missing=True)
        with self.assertRaises(RuntimeError):sample_with_dispatch(a,s,None,7,3,())
        self.assertIs(ns._decoder_layer,original)

    def test_reversed_layer_order_is_rejected(self):
        a,s,ns,original,_=self.setup_adapter(reverse=True)
        with self.assertRaises(RuntimeError):sample_with_dispatch(a,s,None,7,4,())
        self.assertIs(ns._decoder_layer,original)

    def test_exception_restores_native_binding(self):
        a,s,ns,original,_=self.setup_adapter(fail=True)
        with self.assertRaises(RuntimeError):sample_with_dispatch(a,s,None,7,3,(2,8))
        self.assertIs(ns._decoder_layer,original)

    def test_Full_must_keep_all_layers(self):
        a,s,ns,original,_=self.setup_adapter()
        with self.assertRaises(ValueError):sample_with_dispatch(a,s,None,7,10,(2,))
        self.assertIs(ns._decoder_layer,original)

    def test_exact_nodes_accept_3_and_4_and_reject_truncation(self):
        for n in (2,3,4,10):self.assertEqual(scheduled_nfe([1-i/n for i in range(n+1)]),n)
        with self.assertRaises(ValueError):scheduled_nfe([1.,.7,.3,0.])


if __name__=='__main__':unittest.main()

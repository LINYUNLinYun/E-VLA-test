import unittest
from .policy import Policy
from preparations.idea1_v3_4_v12_compensation.test_policy import Fake

class Policies(unittest.TestCase):
    def test_periodic_static_skips_residual_capture(self):
        for kind in ('channel','hadamard','linear'):
            b=Fake();p=Policy(b);p.configure(dict(mode='periodic',compensation=kind,period=2))
            for i in range(7):
                r=p.predict(i,'prompt','suite',i);p.acknowledge(r['actions'][:5],None)
            self.assertEqual([c['capture'] for c in b.calls if 'capture' in c],[False,False,False])
            self.assertTrue(all(c['history'] is None and c['older'] is None and c['kind']==kind for c in b.calls if 'kind' in c))
            self.assertEqual(p.counts['full'],3);self.assertEqual(p.counts['draft'],4)
    def test_interleaved_histories_and_pending(self):
        b=Fake();a=Policy(b);c=Policy(b);cfg=dict(mode='periodic',compensation='reuse',period=2)
        a.configure(cfg);c.configure(cfg)
        ra=a.predict(11,'a','suite',1);rc=c.predict(22,'b','suite',2)
        a.acknowledge(ra['actions'][:5],None)
        with self.assertRaises(RuntimeError):c.predict(23,'b','suite',3)
        a.predict(12,'a','suite',4);self.assertEqual(b.calls[-1]['history'],11)
        c.acknowledge(rc['actions'][:5],None);c.predict(23,'b','suite',5);self.assertEqual(b.calls[-1]['history'],22)
        a.configure(cfg);self.assertEqual(a.history,[]);self.assertEqual(c.history[-1]['residuals'],22)

if __name__=='__main__':unittest.main()

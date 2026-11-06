import unittest
import numpy as np
from .policy import Policy

class Raw:
    def float(self):return self
    def cpu(self):return self
    def numpy(self):return np.zeros((50,32),np.float32)
class Fake:
    def __init__(self):self.calls=[]
    def prepare_residual(self,obs,instruction,skip=(),**kwargs):
        self.calls.append(dict(obs=obs,skip=skip,**kwargs));return dict(residuals=obs),{}
    def sample(self,*args):return np.zeros((50,7),np.float32),Raw(),{}

class CacheTests(unittest.TestCase):
    def test_static_never_refreshes(self):
        for name in ('channel','hadamard','linear'):
            b=Fake();p=Policy(b);p.configure(dict(mode='pure',compensation=name,period=2))
            for i in range(8):
                r=p.predict(i,'prompt','suite',i);p.acknowledge(r['actions'][:5],None)
                self.assertFalse(r['route']['refresh'])
            self.assertEqual(p.history,[])
            self.assertEqual([c['kind'] for c in b.calls],[name]*8)
            self.assertTrue(all(tuple(c['skip'])==(2,3,5,14,16,17) for c in b.calls))
    def test_own_history_ratio_and_reset(self):
        b=Fake();p=Policy(b);cfg=dict(mode='periodic',compensation='extrapolate',period=2)
        p.configure(cfg)
        for i in range(6):
            r=p.predict(i,'prompt','suite',i)
            p.acknowledge(r['actions'][:5],None)
        self.assertEqual([c['capture'] for c in b.calls if 'capture' in c],[True,True])
        self.assertIsNone(b.calls[1]['older'])
        self.assertEqual(b.calls[4]['history'],3)
        self.assertEqual(b.calls[4]['older'],0)
        self.assertAlmostEqual(b.calls[4]['ratio'],1/3)
        self.assertAlmostEqual(b.calls[5]['ratio'],2/3)
        p.configure(cfg);self.assertEqual(p.history,[]);self.assertEqual(p.step,0)
        p.predict(99,'new','suite',0);self.assertTrue(b.calls[-1]['capture'])
    def test_ack_and_pure(self):
        b=Fake();p=Policy(b);p.configure(dict(mode='pure',compensation=None,period=2))
        r=p.predict(0,'prompt','suite',0)
        with self.assertRaises(RuntimeError):p.predict(1,'prompt','suite',1)
        p.acknowledge(r['actions'][:2],None);self.assertEqual(p.step,2);self.assertEqual(p.history,[])

if __name__=='__main__':unittest.main()

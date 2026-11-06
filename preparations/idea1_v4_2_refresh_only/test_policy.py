import unittest
import numpy as np
from .common import conditions, MASK
from .policy import Policy


class Raw:
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return np.zeros((50,32),np.float32)


class Backend:
    def __init__(self): self.preparations=[]; self.samples=[]
    def prepare_residual(self, obs, prompt, skip=(), **kwargs):
        self.preparations.append(dict(obs=obs, skip=skip, **kwargs))
        return dict(residuals=obs), {}
    def sample(self, *args): raise AssertionError('Full action expert forbidden')


def sample(backend, context, seed, steps):
    backend.samples.append(steps)
    return np.zeros((50,7),np.float32), Raw(), {}


class ScheduleTests(unittest.TestCase):
    def test_no_compensation_never_refreshes(self):
        b=Backend(); p=Policy(b,sample); p.configure(conditions()[0])
        for i in range(8):
            r=p.predict(i,'p','s',i); p.acknowledge(r['actions'][:5])
        self.assertEqual(len(b.preparations),8)
        self.assertTrue(all(c['skip']==MASK and c['history'] is None for c in b.preparations))
        self.assertEqual(b.samples,[2]*8)
        self.assertEqual(p.counts['full'],0)

    def test_refresh_uses_own_history_and_always_draft(self):
        b=Backend(); p=Policy(b,sample); p.configure(conditions()[1]); routes=[]
        for i in range(7):
            r=p.predict(i,'p','s',i); routes.append(r['route']); p.acknowledge(r['actions'][:5])
        self.assertEqual([r['residual_refresh'] for r in routes],[True,False,False,True,False,False,True])
        self.assertEqual([r['history_steps'] for r in routes],[[0],[0],[0],[0,15],[0,15],[0,15],[15,30]])
        self.assertTrue(np.allclose([r['extrapolation_ratio'] for r in routes],[0,0,0,0,1/3,2/3,0]))
        self.assertEqual(b.samples,[2]*7)
        self.assertEqual(len([c for c in b.preparations if c.get('capture')]),3)
        self.assertTrue(all(r['action_source']=='v12_d2' and not r['full_ae_executed'] for r in routes))

    def test_ack_and_reset(self):
        p=Policy(Backend(),sample); p.configure(conditions()[1])
        r=p.predict(0,'p','s',0)
        with self.assertRaises(RuntimeError): p.predict(1,'p','s',1)
        with self.assertRaises(ValueError): p.acknowledge(np.ones((5,7),np.float32))
        p.acknowledge(r['actions'][:2]); self.assertEqual(p.step,2)
        p.configure(conditions()[1]); self.assertEqual(p.history,[]); self.assertEqual(p.step,0)


if __name__ == '__main__':
    unittest.main()

import unittest
from .common import MASKS
from .policy import Policy
from preparations.idea1_v3_4_v12_compensation.test_policy import Raw
import numpy as np

class Fake:
    def __init__(self):self.calls=[]
    def prepare_replay(self,obs,instruction,**kwargs):self.calls.append(dict(obs=obs,**kwargs));return dict(residuals=obs),{}
    def sample(self,*args):return np.zeros((50,7),np.float32),Raw(),{}

class Tests(unittest.TestCase):
    def test_masks_and_cache_schedule(self):
        self.assertEqual(MASKS['all_reuse'],list(range(1,18)));self.assertEqual(MASKS['prefix3_reuse'],list(range(4,18)))
        for name in ('all_reuse','prefix3_reuse'):
            b=Fake();p=Policy(b);p.configure(dict(id=name,skip=MASKS[name]))
            for i in range(6):
                r=p.predict(i,'prompt','suite',i);p.acknowledge(r['actions'][:5],None)
                self.assertEqual(r['route']['refresh'],i%3==0)
            self.assertEqual(b.calls[0]['capture_layers'],tuple(MASKS[name]))
            self.assertEqual(b.calls[1]['history'],0);self.assertEqual(b.calls[2]['history'],0);self.assertEqual(b.calls[4]['history'],3)
            p.configure(dict(id=name,skip=MASKS[name]));self.assertIsNone(p.full);self.assertEqual(p.step,0)
    def test_actual_execution_age(self):
        b=Fake();p=Policy(b);p.configure(dict(id='all_reuse',skip=MASKS['all_reuse']))
        r=p.predict(0,'p','s',1)
        with self.assertRaises(RuntimeError):p.predict(0,'p','s',1)
        p.acknowledge(r['actions'][:2],None);r=p.predict(1,'p','s',2);self.assertEqual(r['full_cache_age'],2)
if __name__=='__main__':unittest.main()

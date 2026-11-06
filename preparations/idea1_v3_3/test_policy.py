import unittest
import numpy as np
from .policy import Policy


class Backend:
    protocol={}
    def prepare(self,*args): return {},{}
    def sample(self,c,seed,steps): return np.zeros((50,7),np.float32),np.zeros((50,32),np.float32),dict(NFE=steps)
    def verify(self,*args): return np.zeros((2,50,32)),{}


class PolicyTests(unittest.TestCase):
    def test_period_and_ack_reset(self):
        p=Policy(Backend()); p.configure(dict(model='pi0',experiment='E1',draft_steps=1,tau=.1))
        paths=[]
        for _ in range(5):
            r=p.predict({},'x','libero_goal',7); paths.append((r['route']['path'],r['route'].get('full_reason')))
            p.acknowledge(r['actions'][:r['execute_length']],np.tile([0,0,0,1],(r['execute_length'],1)))
        self.assertEqual(paths[0],('full','startup')); self.assertEqual(paths[3],('full','periodic'))
        self.assertEqual(p.step,25); p.reset(); self.assertIsNone(p.full); self.assertEqual(p.commands,[])

    def test_pending_ack_blocks_replan(self):
        p=Policy(Backend()); p.configure(dict(model='pi0',experiment='E1'))
        r=p.predict({},'x','libero_goal',7)
        with self.assertRaises(RuntimeError): p.predict({},'x','libero_goal',7)
        with self.assertRaises(ValueError): p.acknowledge(np.ones((1,7)),np.array([[0,0,0,1]]))

    def test_partial_acceptance_and_rejection(self):
        class Partial(Backend):
            def verify(self,*args):
                h=np.zeros((2,50,32)); h[:,2:,:6]=1; return h,{}
        p=Policy(Partial()); p.configure(dict(model='pi0',experiment='E3',tau=.1))
        r=p.predict({},'x','libero_goal',7); p.acknowledge(r['actions'][:5],np.tile([0,0,0,1],(5,1)))
        r=p.predict({},'x','libero_goal',7); self.assertEqual(r['execute_length'],2)
        p.acknowledge(r['actions'][:2],np.tile([0,0,0,1],(2,1))); self.assertEqual(p.step,7)


if __name__=='__main__': unittest.main()

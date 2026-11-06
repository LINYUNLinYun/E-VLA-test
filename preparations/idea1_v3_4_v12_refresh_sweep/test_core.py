import unittest
from .policy import Policy
from .common import MASK
from preparations.idea1_v3_4_residual_replay_quick.test_core import Fake

class Tests(unittest.TestCase):
    def test_periods_and_ages(self):
        for period in (2,4,6):
            b=Fake();p=Policy(b);p.configure(dict(id=f'p{period}',skip=MASK,period=period));ages=[];full=[]
            for i in range(16):
                r=p.predict(i,'p','s',i)
                self.assertEqual(r['route']['refresh'],i%(period+1)==0)
                if r['route']['refresh']:full.append(i)
                else:ages.append(r['full_cache_age']);self.assertEqual(r['route']['history_step'],full[-1]*5)
                p.acknowledge(r['actions'][:5],None)
            self.assertEqual(max(ages),period*5)
            p.configure(dict(id=f'p{period}',skip=MASK,period=period));self.assertIsNone(p.full)
    def test_pending_and_executed_age(self):
        p=Policy(Fake());p.configure(dict(id='p6',skip=MASK,period=6));r=p.predict(0,'p','s',0)
        with self.assertRaises(RuntimeError):p.predict(1,'p','s',1)
        p.acknowledge(r['actions'][:2],None);r=p.predict(1,'p','s',1);self.assertEqual(r['full_cache_age'],2)
if __name__=='__main__':unittest.main()

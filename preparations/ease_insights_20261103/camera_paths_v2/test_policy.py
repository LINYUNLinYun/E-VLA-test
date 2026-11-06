"""CPU scheduling and ownership checks; no model or GPU is loaded."""
import unittest
import numpy as np
from .common import conditions
from .policy import Policy


class Raw:
    def float(self):return self
    def cpu(self):return self
    def numpy(self):return np.zeros((50,32),np.float32)


class FakeBackend:
    def __init__(self):self.requests=[];self.samples=[];self.resets=0
    def reset_vision(self):self.resets+=1;self.last_full_vision=None;self.request=None
    def request_vision(self,request):
        if self.request is not None:raise RuntimeError('unconsumed')
        self.request=request;self.requests.append(request)
    def prepare_residual(self,obs,instruction):
        request=self.request;self.request=None
        if request['kind']=='refresh':self.last_full_vision=object()
        return dict(execution_audit={},camera_execution_audit=request),{}
    def sample(self,context,seed,steps,ae_skip):
        self.samples.append(steps)
        return np.zeros((50,7),np.float32),Raw(),dict(ae_execution_audit={},camera_execution_audit=context['camera_execution_audit'])


class CameraPolicyContracts(unittest.TestCase):
    def exercise(self,cid):
        b=FakeBackend();p=Policy(b);p.configure(next(c for c in conditions() if c['id']==cid));rows=[]
        for i in range(10):
            r=p.predict({},'prompt','suite',i);rows.append(r);p.acknowledge(r['actions'][:5])
        return b,p,rows

    def test_equal_budget_camera_pair_and_full_schedule(self):
        for cid,camera in [('agent_fresh_d5',0),('wrist_fresh_d5',1)]:
            b,p,rows=self.exercise(cid)
            self.assertEqual([r['route']['refresh'] for r in rows],[True,False,False,False,False]*2)
            self.assertEqual(b.samples,[10,5,5,5,5]*2)
            self.assertEqual(rows[-1]['counts'],dict(full=2,draft=8,verify=0))
            for i in (1,2,3,4,6,7,8,9):
                q=b.requests[i];self.assertEqual(q['camera'],camera)
                self.assertEqual((q['source_step'],q['age_steps']),(25*(i//5),5*(i%5)))
                self.assertIs(q['cache'],b.requests[(i//5)*5+1]['cache'])
            self.assertIsNot(b.requests[1]['cache'],b.requests[6]['cache'])

    def test_both_fresh_control_has_same_NFE_and_refresh_but_no_held_view(self):
        b,p,rows=self.exercise('both_fresh_d5')
        self.assertEqual(b.samples,[10,5,5,5,5]*2)
        self.assertEqual([r['kind'] for r in b.requests],['refresh','both','both','both','both']*2)
        self.assertTrue(all('cache' not in r for r in b.requests))

    def test_full_never_uses_D5_or_held_view(self):
        b,p,rows=self.exercise('full_d10')
        self.assertEqual(b.samples,[10]*10);self.assertTrue(all(r['kind']=='full' for r in b.requests))
        self.assertIsNone(p.cache)

    def test_reconfigure_cannot_reuse_other_episode_history(self):
        b,p,_=self.exercise('agent_fresh_d5');old=p.cache
        p.configure(next(c for c in conditions() if c['id']=='wrist_fresh_d5'))
        r=p.predict({},'new prompt','suite',1)
        self.assertEqual((r['round'],r['step'],r['route']['history_steps']),(0,0,[0]))
        self.assertIsNot(p.cache,old);self.assertEqual(b.requests[-1]['kind'],'refresh')

    def test_no_prediction_until_execution_acknowledged(self):
        b,p,_=self.exercise('agent_fresh_d5');r=p.predict({},'prompt','suite',11)
        with self.assertRaises(RuntimeError):p.predict({},'prompt','suite',12)
        with self.assertRaises(ValueError):p.acknowledge(np.ones((5,7),np.float32))
        p.acknowledge(r['actions'][:3]);self.assertEqual(p.step,53)


if __name__=='__main__':unittest.main()

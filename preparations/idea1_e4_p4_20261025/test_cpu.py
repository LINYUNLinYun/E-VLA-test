import copy
from pathlib import Path
import unittest
import numpy as np
from preparations.idea1_v3_3.policy import Policy as Original
from .period_policy import Policy as Revised
from .policy import Policy as Timed


class Backend:
    def __init__(self):self.protocol={};self.error=None;self.verify_calls=0
    def prepare(self,obs,instruction,skip):return dict(skip=tuple(skip)),dict(prepare_ms=0.)
    def sample(self,c,seed,steps):
        raw=np.random.default_rng(seed+len(c['skip'])).normal(0,.01,(10,7)).astype(np.float32)
        raw[:,6]=-1.;return raw.copy(),raw,dict(ae_ms=0.,NFE=steps)
    def verify(self,full,current,raw,seed,suite):
        self.verify_calls+=1;h=np.stack([raw.copy(),raw.copy()])
        if self.error=='reject':h[:,:,:6]+=2.
        if self.error=='partial':h[:,4:,:6]+=2.
        return h,dict(verify_ae_ms=0.)


def acknowledge(p,r,n=None):
    n=r['execute_length'] if n is None else n
    return p.acknowledge(r['actions'][:n],np.tile([0.,0.,0.,1.],(n,1)))


class Tests(unittest.TestCase):
    def test_source_scope_and_p2_regression(self):
        root=Path(__file__).resolve().parents[2]
        old=(root/'preparations/idea1_v3_3/policy.py').read_text()
        new=Path(__file__).with_name('period_policy.py').read_text()
        self.assertEqual(new,old.replace('self.draft_rounds>=2',"self.draft_rounds>=cfg.get('refresh_period',2)"))
        for mode in ['periodic','verified']:
            left,right=Original(Backend()),Revised(Backend())
            cfg=dict(model='xiaomi',experiment='E4',mode=mode,tau=.8,refresh_period=2)
            left.configure(cfg);right.configure(cfg)
            for i in range(18):
                for p in [left,right]:p.backend.error=['partial','reject',None,None][i%4]
                a=left.predict({},'realshape','libero_spatial',i);b=right.predict({},'realshape','libero_spatial',i)
                for k in ['actions','raw']:np.testing.assert_array_equal(a[k],b[k])
                for k in ['execute_length','route','counts','full_cache_age','round','step']:self.assertEqual(a[k],b[k])
                acknowledge(left,a);acknowledge(right,b)

    def test_p4_counter_and_no_gripper_guards(self):
        for mode in ['periodic','verified']:
            p=Revised(Backend());p.configure(dict(model='xiaomi',experiment='E4',mode=mode,tau=.8,refresh_period=4))
            paths=[]
            for i in range(11):
                r=p.predict({},'x','libero_spatial',i);paths.append(r['route']['path']);acknowledge(p,r)
                if 'verify_gripper' in r['route']:self.assertFalse(r['route']['verify_gripper']);self.assertFalse(r['route']['post_verify_gripper'])
            self.assertEqual([i for i,v in enumerate(paths) if v=='full'],[0,5,10])
            if mode=='periodic':self.assertEqual(p.backend.verify_calls,0)
        p.backend.error='reject';r=p.predict({},'x','libero_spatial',20)
        self.assertEqual(r['route']['full_reason'],'zero_accept');self.assertEqual(p.draft_rounds,0);acknowledge(p,r)
        p.backend.error='partial';r=p.predict({},'x','libero_spatial',21)
        self.assertEqual(r['execute_length'],4);self.assertEqual(p.draft_rounds,1)

    def test_ack_full_server_wall(self):
        p=Timed(Backend());p.configure(dict(model='xiaomi',experiment='E4',mode='periodic',tau=.8,refresh_period=4))
        r=p.predict({},'x','libero_spatial',1);ack=acknowledge(p,r)
        self.assertGreaterEqual(ack['ack_policy_wall_ms'],0)
        self.assertEqual(p.step,10);self.assertEqual(len(p.commands),10)


if __name__=='__main__':unittest.main()

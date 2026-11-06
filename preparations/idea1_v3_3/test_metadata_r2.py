import unittest
from unittest.mock import patch
import numpy as np
from .policy import Policy as Original
from .metadata_r2 import Policy
from .test_policy import Backend


class Metadata(unittest.TestCase):
    def test_original_actions_routes_and_ack_identical(self):
        a,b=Original(Backend()),Policy(Backend())
        for p in (a,b): p.configure(dict(model='pi0',experiment='E1',draft_steps=1,tau=.1))
        for index in range(5):
            old,new=a.predict({},'x','libero_goal',index),b.predict({},'x','libero_goal',index)
            self.assertTrue(np.array_equal(old['actions'],new['actions'])); self.assertTrue(np.array_equal(old['raw'],new['raw']))
            self.assertEqual(old['route'],new['route']); self.assertEqual(old['execute_length'],new['execute_length'])
            for p,r in ((a,old),(b,new)): p.acknowledge(r['actions'][:r['execute_length']],np.tile([0,0,0,1],(r['execute_length'],1)))

    def test_kalman_has_no_draft_noise(self):
        p=Policy(Backend()); actions=np.zeros((50,7),np.float32)
        for rejected in (False,True):
            route=dict(path='full' if rejected else 'kalman_verified',distance=[[0.]])
            if rejected: route['rejected_path']='kalman_verified'
            row=dict(route=route,actions=actions,noise_hash='last',timing=dict(policy_wall_ms=1.))
            with patch.object(Original,'predict',return_value=row): result=p.predict({},'x','libero_goal',7)
            self.assertIsNone(result['draft_noise_hash']); self.assertIsNotNone(result['verify_noise_hash'])
            self.assertEqual(result['full_noise_hash'] is not None,rejected)
            self.assertIs(result['actions'],actions)


if __name__=='__main__': unittest.main()

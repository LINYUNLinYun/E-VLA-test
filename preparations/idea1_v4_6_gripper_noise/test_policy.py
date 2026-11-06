import unittest
import numpy as np
from preparations.idea1_v4_4_one_step.test_policy import Backend, Coordinates, gaussian, initial
from preparations.idea1_v4_1_warmstart.schedule import mix
from .common import conditions
from .policy import Policy,make_initial


class GripperNoiseTests(unittest.TestCase):
    def test_only_gripper_input_changes_and_old_value_is_removed(self):
        prior=np.random.default_rng(9).standard_normal((50,32),dtype=np.float32)
        for tau in (.05,.10):
            expected=mix(prior,42,tau,1.)
            old=make_initial(prior,42,tau,1.,'hold')
            self.assertTrue(np.array_equal(old,expected))
            new=make_initial(prior,42,tau,1.,'standard_gaussian')
            noise=np.random.default_rng(42).standard_normal((50,32),dtype=np.float32)
            self.assertTrue(np.array_equal(new[:,:6],old[:,:6]))
            self.assertTrue(np.array_equal(new[:,7:],old[:,7:]))
            self.assertTrue(np.array_equal(new[:,6],noise[:,6]))
            changed=prior.copy();changed[:,6]+=17.
            self.assertTrue(np.array_equal(new,make_initial(changed,42,tau,1.,'standard_gaussian')))

    def test_policy_uses_one_ae_original_tau_and_causal_ack(self):
        for cfg in conditions():
            b=Backend();p=Policy(b,gaussian,initial,Coordinates);p.configure(cfg)
            routes=[]
            for i in range(4):
                r=p.predict(i,'p','s',i);routes.append(r['route'])
                self.assertEqual(r['route']['actual_gripper_noise_std'],1.)
                self.assertEqual(r['route']['gripper_prior_weight'],0.)
                self.assertEqual(r['route']['actual_start_time'],1. if i==0 else cfg['start_time'])
                self.assertFalse(r['route']['full_action_executed'])
                self.assertEqual(p.kalman.filter.count,p.step)
                if i:
                    expected=np.random.default_rng(i).standard_normal((50,32),dtype=np.float32)
                    self.assertTrue(np.array_equal(r['arrays']['kalman_initial_prefix'][:,6],expected[:5,6]))
                with self.assertRaises(RuntimeError):p.predict(i,'p','s',i)
                p.acknowledge(r['actions'][:5])
            self.assertEqual(b.samples,[('gaussian',1,1.)]+[('kalman',1,cfg['start_time'])]*3)
            self.assertEqual([v['residual_refresh'] for v in routes],[True,False,False,True])
            self.assertEqual(p.counts['full'],0);self.assertEqual(p.counts['verify'],0)
            p.configure(cfg);self.assertEqual(p.kalman.filter.count,0);self.assertEqual(p.history,[])


if __name__=='__main__':unittest.main()

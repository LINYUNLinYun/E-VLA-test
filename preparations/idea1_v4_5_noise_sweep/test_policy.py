import unittest
import numpy as np
from preparations.idea1_v4_4_one_step.test_policy import Backend, Coordinates, gaussian, initial
from .common import conditions, allowed_conditions, all_conditions
from .policy import Policy


class SweepTests(unittest.TestCase):
    def policy(self, cfg):
        b=Backend(); p=Policy(b,gaussian,initial,Coordinates); p.configure(cfg)
        return b,p

    def test_every_tau_reaches_the_mixer_and_one_step_sampler(self):
        for cfg in all_conditions():
            with self.subTest(tau=cfg['start_time']):
                b,p=self.policy(cfg)
                r=p.predict(0,'p','s',11)
                self.assertEqual(b.samples,[('gaussian',1,1.)])
                p.acknowledge(r['actions'][:5])
                r=p.predict(1,'p','s',42)
                tau=cfg['start_time']
                self.assertEqual(b.samples[-1],('kalman',1,tau))
                prior=p.kalman.last_forecast
                encoded=np.zeros((50,32),np.float32); encoded[:,:7]=prior
                noise=np.random.default_rng(42).standard_normal((50,32),dtype=np.float32)
                expected=((1-tau)*encoded+tau*noise).astype(np.float32)
                self.assertTrue(np.array_equal(r['arrays']['kalman_initial_prefix'],expected[:5]))
                self.assertEqual(r['route']['actual_noise_std'],tau)
                self.assertEqual(r['route']['kalman_prior_weight'],1-tau)
                self.assertEqual(r['route']['actual_start_time'],tau)
                self.assertEqual(r['route']['kalman_source_last_step'],4)

    def test_every_request_is_one_ae_and_residual_only_full(self):
        for cfg in conditions():
            b,p=self.policy(cfg); routes=[]
            for i in range(7):
                r=p.predict(i,'p','s',i); routes.append(r['route']); p.acknowledge(r['actions'][:5])
            self.assertEqual([r['residual_refresh'] for r in routes],[True,False,False,True,False,False,True])
            self.assertTrue(np.allclose([r['extrapolation_ratio'] for r in routes],[0,0,0,0,1/3,2/3,0]))
            self.assertTrue(all(v[1]==1 for v in b.samples))
            self.assertTrue(all(not r['full_ae_executed'] and not r['full_action_executed'] and not r['verifier'] for r in routes))
            self.assertEqual(p.counts['full'],0); self.assertEqual(p.counts['verify'],0)

    def test_ack_only_and_reset_prevent_history_leak_between_candidates(self):
        b,p=self.policy(conditions()[0]); r=p.predict(0,'p','s',0)
        with self.assertRaises(RuntimeError): p.predict(1,'p','s',1)
        with self.assertRaises(ValueError): p.acknowledge(np.ones((5,7),np.float32))
        p.acknowledge(r['actions'][:2]); r=p.predict(1,'p','s',1)
        self.assertEqual(p.kalman.filter.count,2); self.assertEqual(p.kalman.rotation_filter.count,2)
        self.assertEqual(r['route']['kalman_source_last_step'],1)
        p.acknowledge(r['actions'][:5]); p.configure(conditions()[1])
        self.assertEqual(p.step,0); self.assertEqual(p.history,[])
        self.assertEqual(p.kalman.filter.count,0)
        r=p.predict(0,'p','s',0)
        self.assertEqual(r['route']['initial_state'],'gaussian')
        self.assertEqual(p.kalman.filter.count,0)


if __name__=='__main__': unittest.main()

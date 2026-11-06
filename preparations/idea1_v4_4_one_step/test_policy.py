import types
import unittest
import numpy as np
from .common import conditions, MASK
from .policy import Policy


class Raw:
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return np.zeros((50,32), np.float32)


class Backend:
    def __init__(self):
        self.preparations, self.samples = [], []
        self.adapter = types.SimpleNamespace(backend=types.SimpleNamespace(output_transform=None))
    def prepare_residual(self, obs, prompt, skip=(), **kwargs):
        self.preparations.append(dict(obs=obs, skip=skip, **kwargs))
        return dict(residuals=obs, transformed_state=np.zeros(7)), {}
    def sample(self, *args): raise AssertionError('Full AE forbidden')


class Coordinates:
    def __init__(self, *args): pass
    def encode_commands(self, commands, raw):
        raw[:,:7] = commands
        return raw


def output():
    action = np.zeros((50,7), np.float32)
    t = np.arange(50, dtype=np.float32)
    action[:,:6] = t[:,None]*.1
    action[:,6] = -1
    return action, Raw(), {}


def gaussian(b, context, seed, steps):
    b.samples.append(('gaussian',steps,1.))
    return output()


def initial(b, context, value, steps, start):
    b.samples.append(('kalman',steps,start))
    return output()


class OneStepTests(unittest.TestCase):
    def policy(self, cfg):
        b=Backend(); p=Policy(b,gaussian,initial,Coordinates); p.configure(cfg)
        return b,p

    def test_every_round_is_one_step_including_full_refresh(self):
        for cfg in conditions():
            b,p=self.policy(cfg); routes=[]
            for i in range(7):
                r=p.predict(i,'p','s',i); routes.append(r['route']); p.acknowledge(r['actions'][:5])
            self.assertEqual([r['residual_refresh'] for r in routes],[True,False,False,True,False,False,True])
            self.assertEqual([r['history_steps'] for r in routes],[[0],[0],[0],[0,15],[0,15],[0,15],[15,30]])
            self.assertTrue(np.allclose([r['extrapolation_ratio'] for r in routes],[0,0,0,0,1/3,2/3,0]))
            self.assertTrue(all(s[1]==1 for s in b.samples))
            self.assertTrue(all(not r['full_action_executed'] for r in routes))
            self.assertEqual(p.counts['full'],0)

    def test_kalman_only_sees_acknowledged_prefix_and_resets(self):
        b,p=self.policy(conditions()[1]); r=p.predict(0,'p','s',0)
        self.assertEqual(b.samples,[('gaussian',1,1.)])
        self.assertEqual(p.kalman.filter.count,0)
        with self.assertRaises(RuntimeError): p.predict(1,'p','s',1)
        with self.assertRaises(ValueError): p.acknowledge(np.ones((5,7),np.float32))
        p.acknowledge(r['actions'][:2])
        self.assertEqual(p.kalman.filter.count,2)
        self.assertEqual(p.kalman.rotation_filter.count,2)
        r=p.predict(1,'p','s',1)
        self.assertEqual(b.samples[-1],('kalman',1,.75))
        self.assertEqual(r['route']['kalman_source_last_step'],1)
        expected=p.kalman.forecast()
        self.assertTrue(np.array_equal(r['arrays']['kalman_prior_command_prefix'],expected[:5]))
        p.acknowledge(r['actions'][:5]); p.configure(conditions()[1])
        self.assertEqual(p.kalman.filter.count,0)
        self.assertEqual(p.history,[])

    def test_filter_uses_controller_clipping_and_held_gripper(self):
        b,p=self.policy(conditions()[1]); r=p.predict(0,'p','s',0)
        p.acknowledge(r['actions'][:5]); r=p.predict(1,'p','s',1)
        prior=r['arrays']['kalman_prior_command_prefix']
        self.assertTrue(np.all(np.abs(prior[:,:6])<=1))
        self.assertTrue(np.all(prior[:,6]==-1))
        self.assertEqual(p.kalman.filter.count,5)


if __name__=='__main__': unittest.main()

"""Routing/history tests that distinguish Full scoring from Fast execution."""
import unittest
from unittest.mock import patch
import numpy as np
import torch
from . import policy as module
from .common import config
from .selection import freeze_budget


class Adapter:
    def measure(self, fn):
        return fn(), 0.


class Backend:
    adapter = Adapter()
    def __init__(self):
        self.calls = []
    def prepare_modules(self, obs, instruction, **kwargs):
        self.calls.append(kwargs)
        c = dict(seq=2, residuals={'ffn':{i:torch.full((2,2),float(obs)) for i in range(1,18)}})
        return c, dict(vlm_ms=0.)
    def sample(self, c, seed, steps):
        return np.zeros((50,7),np.float32), torch.zeros(50,32), dict(ae_ms=0.)


class Selector:
    def __init__(self, *a):
        self.errors = []
    def clear(self):
        self.errors = []
    def run(self, c, seed, coeff, error=None, **kwargs):
        self.errors.append(None if error is None else error.clone())
        score = np.arange(17,dtype=np.float32)
        return score, score, dict(selector_total_ms=0.,selector_hot_ms=0.,selector_capture_ms=0.)


class Routing(unittest.TestCase):
    def setUp(self):
        self.mock = patch.object(module,'Selector',Selector)
        self.mock.start()
        table = dict(measured=True, units='ms', layer_costs=[dict(layer=i,ffn_saved_ms=1.,measurement_noise_ms=0.) for i in range(1,18)],
                     anchors={key:dict(saved_ms=n,fast_ms=10) for key,n in [('B4',4),('B6',6)]})
        self.budgets = {key:freeze_budget(table,key) for key in ('B4','B6')}
    def tearDown(self):
        self.mock.stop()
    def run_requests(self, scoring, count=15):
        b=Backend();p=module.Policy(b,self.budgets)
        cfg=config('B4',scoring);cfg.update(episode_case=dict(suite='libero_spatial',task=1,init=19))
        p.configure(cfg);rows=[]
        for i in range(count):
            r=p.predict(i//5+1,'prompt','libero_spatial',100+i)
            rows.append(r)
            p.acknowledge(r['actions'][:5],None)
        return b,p,rows
    def test_error_uses_two_prior_fulls_and_activates_at_step55(self):
        b,p,rows=self.run_requests('F-E+')
        self.assertEqual([r['step'] for r in rows if r['route']['steady_vjp_count']],[0,25,50])
        self.assertEqual(rows[10]['route']['score_source_full_steps'],[0,25,50])
        self.assertEqual(rows[10]['route']['scoring_mode'],'comp_error')
        self.assertEqual(rows[11]['step'],55)
        self.assertTrue(all(v is None for v in p.selector.errors[:2]))
        self.assertTrue(torch.equal(p.selector.errors[2],torch.zeros_like(p.selector.errors[2])))
        for i,r in enumerate(rows):
            self.assertEqual(r['route']['D'],10 if i%5==0 else 2)
            if i%5:
                self.assertAlmostEqual(r['route']['extrapolation_ratio'],0. if i<5 else (i%5)/5)
    def test_fixed_has_no_vjp_and_one_mask_for_episode(self):
        b,p,rows=self.run_requests('F-Fixed+')
        self.assertEqual(len(p.selector.errors),0)
        self.assertTrue(all(r['route']['steady_vjp_count']==0 for r in rows))
        self.assertTrue(all(r['route']['selected_skip']==rows[0]['route']['selected_skip'] for r in rows))
    def test_no_compensation_keeps_same_mask_budget_and_omits_patches(self):
        b,p,rows=self.run_requests('F-R0',6)
        self.assertTrue(all(c.get('history') is None for c in b.calls if 'skip_ffn' in c))
        self.assertEqual(rows[0]['route']['selected_skip'],rows[1]['route']['selected_skip'])
    def test_execution_ack_controls_history_age(self):
        b=Backend();p=module.Policy(b,self.budgets);p.configure(config('B4','F-R+'))
        r=p.predict(1,'prompt','libero_spatial',1)
        with self.assertRaises(RuntimeError):p.predict(1,'prompt','libero_spatial',2)
        p.acknowledge(r['actions'][:3],None)
        r=p.predict(1,'prompt','libero_spatial',2)
        self.assertEqual(r['step'],3)
        self.assertEqual(r['route']['full_age_env_steps'],3)


if __name__ == '__main__':
    unittest.main()

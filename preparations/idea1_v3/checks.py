"""CPU checks of v3 routing/state contracts; no GPU/checkpoint loading."""
import argparse
import json
import tempfile
from pathlib import Path
import types
import unittest

import numpy as np

from preparations.idea1_v3.policy import Policy, acceptance, verification_seed
from preparations.idea1_v3.cli import make_manifest


class Backend:
    protocol=dict(H=50,R=5)
    def __init__(self): self.visited=[]; self.fulls=[]; self.reject=False
    def prepare(self,obs,instruction,skip=(),encode=True):
        self.visited.append((obs,skip,encode))
        return dict(K=np.full((2,3),obs).copy(),state=obs,seq=3),{}
    def sample(self,c,seed,steps):
        raw=np.full((50,32),.01,np.float32); raw[:,6]=-1
        return raw[:,:7].copy(),raw,dict(steps=steps)
    def verify(self,full,current,raw,seed,suite):
        self.fulls.append((full['K'].copy(),current['state']))
        hats=np.stack([raw.copy(),raw.copy()])
        if self.reject: hats[:,:,:6]+=1
        return hats,{}
    def learned(self,c,suite): return self.sample(c,123,0)


class Checks(unittest.TestCase):
    def policy(self,**kw): return Policy(dict(experiment='E1',draft_steps=2,tau=.15,period=2,**kw),Backend())
    def predict(self,p,obs=1): return p.predict(obs,'instruction','libero_goal',123)
    def ack(self,p,r,n=None): p.acknowledge(r['actions'][:n or r['execute_length']])
    def test_latest_full_and_period(self):
        p=self.policy(); first=self.predict(p,1); self.ack(p,first)
        for obs in (2,3):
            r=self.predict(p,obs); self.assertEqual(r['route']['path'],'v14_verified'); self.ack(p,r)
            np.testing.assert_array_equal(p.backend.fulls[-1][0],np.ones((2,3)))
            self.assertEqual(p.backend.fulls[-1][1],obs)
        r=self.predict(p,4); self.assertEqual(r['route']['full_reason'],'periodic')
        self.assertEqual(r['counts'],dict(full=2,verify=2,draft=2)); self.assertEqual(r['execute_length'],5)
    def test_rejection_executes_only_full(self):
        p=self.policy(); r=self.predict(p); self.ack(p,r); p.backend.reject=True
        r=self.predict(p,2)
        self.assertEqual(r['route']['full_reason'],'zero_accept'); self.assertEqual(r['route']['accepted_length'],0)
        self.assertEqual(p.step,5); self.assertEqual(r['execute_length'],5); self.ack(p,r)
        self.assertEqual(p.step,10); self.assertEqual(p.full['state'],2)
    def test_ack_reset_and_interleaved_envs(self):
        p=self.policy(); q=self.policy(); r=self.predict(p,1); other=self.predict(q,9)
        with self.assertRaises(RuntimeError): self.predict(p,2)
        self.ack(p,r,2); self.assertEqual(p.step,2); self.assertEqual(q.step,0)
        p.reset(); self.assertIsNone(p.full); self.assertIsNone(p.last_gripper)
        self.assertEqual(self.predict(p,5)['route']['full_reason'],'startup')
        self.ack(q,other); self.assertEqual(q.step,5)
    def test_e4_pure_draft(self):
        p=Policy(dict(experiment='E4'),Backend())
        for obs in (1,2,3):
            r=self.predict(p,obs); self.assertEqual(r['counts']['full'],0); self.assertEqual(r['counts']['verify'],0)
            self.assertFalse(p.backend.visited[-1][2]); self.ack(p,r)
        self.assertIsNone(p.full)
    def test_acceptance_both_branches_and_gripper(self):
        draft=np.zeros((50,32)); draft[:,6]=-1
        hats=np.stack([draft.copy(),draft.copy()]); hats[1,3,:6]=.2
        length,details=acceptance(draft,hats,.15,-1); self.assertEqual(length,3)
        hats[0,1,6]=1
        length,details=acceptance(draft,hats,.15,-1)
        self.assertEqual(length,0); self.assertTrue(details['verify_gripper'])
    def test_seed_domain(self):
        self.assertNotEqual(verification_seed(123),123); self.assertEqual(verification_seed(123),verification_seed(123))
    def test_manifest_budget_owner_and_suite_weight(self):
        args=types.SimpleNamespace(stage='P0',config=None,execution='A',expand=False,shards=1)
        manifest=make_manifest(args); self.assertEqual(len(manifest['tasks']),8)
        self.assertEqual(len({x['key'] for x in manifest['tasks']}),8)
        args.stage='calibration'; args.shards=4
        tasks=make_manifest(args)['tasks']; self.assertEqual(len(tasks),96)
        self.assertEqual({x['owner'] for x in tasks},{0,1,2,3})
        args.stage='E4'
        tasks=make_manifest(args)['tasks']; self.assertEqual(len(tasks),40)
        self.assertEqual(len({x['checkpoint']['draft_path'] for x in tasks}),4)
        self.assertTrue(all('base_sha256' in x['checkpoint'] for x in tasks))
    def test_npz_execution_equivalence(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); a=np.arange(1600,dtype=np.float32).reshape(50,32)
            np.savez_compressed(d/'a.npz',raw=a); np.savez(d/'b.npz',raw=a)
            with np.load(d/'a.npz') as x,np.load(d/'b.npz') as y: np.testing.assert_array_equal(x['raw'],y['raw'])

if __name__=='__main__': unittest.main()

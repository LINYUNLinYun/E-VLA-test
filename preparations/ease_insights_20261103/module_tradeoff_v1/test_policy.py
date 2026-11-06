"""CPU-only scheduling checks for the previously unsupported empty-VLM mask."""
import unittest
import numpy as np
from .policy import Policy


class Raw:
    def __init__(self,value):self.value=value
    def float(self):return self
    def cpu(self):return self
    def numpy(self):return self.value


class FakeBackend:
    def __init__(self):self.prepared=[];self.samples=[]
    def prepare_residual(self,obs,instruction,**kwargs):
        self.prepared.append(kwargs)
        return dict(residuals={l:float(len(self.prepared)) for l in kwargs.get('capture_layers',())},execution_audit={}),{}
    def sample(self,context,seed,steps,ae_skip):
        self.samples.append(dict(seed=seed,steps=steps,ae_skip=tuple(ae_skip)))
        return np.zeros((50,7),np.float32),Raw(np.zeros((50,32),np.float32)),dict(ae_execution_audit=dict(actual_NFE=steps,actual_layer_calls=steps*(18-len(ae_skip))))


def config(cid,vlm=(),ae=(),full=False):
    return dict(id=cid,mode='full' if full else 'periodic',period=2,H=50,R=5,D=10 if full else 2,
                refresh_D=10,skip=list(vlm),ae_skip=list(ae),compensation=None if full else 'extrapolate')


def validator(cfg):return tuple(cfg['skip']),tuple(cfg['ae_skip'])


class PolicyContracts(unittest.TestCase):
    def run_rounds(self,cfg):
        backend=FakeBackend();policy=Policy(backend,validator);policy.configure(cfg);responses=[]
        for i in range(6):
            r=policy.predict({},'prompt','suite',7+i);responses.append(r)
            policy.acknowledge(r['actions'][:5])
        return policy,backend,responses

    def test_empty_vlm_mask_still_runs_fast_ae_path(self):
        _,backend,rows=self.run_rounds(config('ae_only',ae=(2,8)))
        self.assertEqual([r['route']['refresh'] for r in rows],[True,False,False,True,False,False])
        self.assertEqual([s['steps'] for s in backend.samples],[10,2,2,10,2,2])
        self.assertEqual([s['ae_skip'] for s in backend.samples],[(),(2,8),(2,8),(),(2,8),(2,8)])
        self.assertEqual(rows[-1]['route']['history_steps'],[0,15])
        self.assertAlmostEqual(rows[4]['route']['extrapolation_ratio'],1/3)
        self.assertAlmostEqual(rows[5]['route']['extrapolation_ratio'],2/3)
        self.assertEqual(rows[-1]['counts'],dict(full=2,draft=4,verify=0))

    def test_vlm_path_uses_only_own_full_histories(self):
        _,backend,rows=self.run_rounds(config('vlm_only',vlm=(2,3,5,14,16,17)))
        self.assertEqual(set(backend.prepared[4]['history']),{2,3,5,14,16,17})
        self.assertEqual(set(backend.prepared[4]['older']),{2,3,5,14,16,17})
        self.assertTrue(all(s['ae_skip']==() for s in backend.samples))
        self.assertEqual([r['route']['refresh'] for r in rows],[True,False,False,True,False,False])

    def test_full_never_bypasses_or_uses_d2(self):
        _,backend,rows=self.run_rounds(config('full_d10',full=True))
        self.assertTrue(all(r['route']['refresh'] and r['route']['ae_bypassed_layers']==[] for r in rows))
        self.assertTrue(all(s['steps']==10 and s['ae_skip']==() for s in backend.samples))

    def test_reconfigure_resets_other_path_history(self):
        policy,backend,rows=self.run_rounds(config('a',vlm=(2,3)))
        policy.configure(config('b',ae=(2,8)))
        r=policy.predict({},'another prompt','suite',7)
        self.assertEqual((r['round'],r['step'],r['route']['history_steps']),(0,0,[0]))
        self.assertFalse(r['route']['has_second_history'])
        self.assertEqual(r['counts'],dict(full=1,draft=0,verify=0))


if __name__=='__main__':unittest.main()

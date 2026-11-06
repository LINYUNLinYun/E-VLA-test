"""CPU-only D2 semantics, immutable resumption, and initial-input gates."""
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from . import policy_plus_v14_d2 as policy
from .plus_v14_d2 import completed,VIDEOS
from .run_plus_v14_d2 import initial_comparison,KEYS
from . import run_plus_v14_d2 as runner
from .native_envs import digest
from preparations.idea1_v3_2.transfer_policy import MODES


class Backend:
    def prepare(self,raw,instruction,skip,encode):
        self.prepared=(raw,instruction,skip,encode)
        return {},dict(vision_ms=1.,vlm_ms=2.)
    def sample(self,context,seed,steps):
        self.steps=steps
        return np.zeros((50,7),np.float32),np.zeros((50,32),np.float32),dict(actual_NFE=2,nodes=[1.,.5,0.],ae_ms=3.,decode_ms=1.)


@contextmanager
def observed():
    yield Counter(vision_encoder_calls=1,vlm_prefix_calls=1,ae_layer_calls=36)


class Contracts(unittest.TestCase):
    def test_actual_d2_route_no_frozen_modes_mutation(self):
        original=tuple(MODES);backend=Backend();p=policy.V14D2Policy(dict(manifest_hash='test'),backend)
        raw=dict(base_rgb=np.zeros((4,5,3),np.uint8),wrist_rgb=np.zeros((4,5,3),np.uint8),state8=np.zeros(8,np.float32))
        with self.assertRaisesRegex(RuntimeError,'GATE'): p.predict('v14_d2',raw,'actual BDDL',7)
        with patch.object(policy,'route_observations',observed):
            result=p.predict('v14_d2',raw,'actual BDDL',7,category='preflight')
        self.assertEqual(backend.steps,2);self.assertEqual(tuple(backend.prepared[2]),(3,4,9,11));self.assertTrue(backend.prepared[3])
        self.assertEqual(result['observed_calls']['ae_layer_calls'],36);self.assertFalse(result['route']['fallback'])
        self.assertEqual(tuple(MODES),original);self.assertNotIn('v14_d2',MODES)
        with self.assertRaises(ValueError): p.predict('v14_d10',raw,'actual BDDL',7)

    def test_d10_calls_cannot_be_mislabeled_d2(self):
        @contextmanager
        def wrong(): yield Counter(vision_encoder_calls=1,vlm_prefix_calls=1,ae_layer_calls=180)
        raw=dict(base_rgb=np.zeros((4,5,3),np.uint8),wrist_rgb=np.zeros((4,5,3),np.uint8),state8=np.zeros(8,np.float32))
        p=policy.V14D2Policy(dict(manifest_hash='test'),Backend())
        with patch.object(policy,'route_observations',wrong),self.assertRaisesRegex(RuntimeError,'contract'):
            p.predict('v14_d2',raw,'actual',7,category='preflight')

    def test_initial_physical_exact_image_budget(self):
        obs={k:np.zeros((4,5,3),np.uint8) if k.endswith('image') else np.zeros(4,np.float64) for k in KEYS}
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'initial.npz';np.savez_compressed(path,**obs,instruction='actual BDDL')
            self.assertTrue(initial_comparison(obs,'actual BDDL',path)['passed'])
            obs['agentview_image'].flat[:4]=1
            self.assertTrue(initial_comparison(obs,'actual BDDL',path)['passed'])
            obs['agentview_image'].flat[4]=1
            self.assertFalse(initial_comparison(obs,'actual BDDL',path)['passed'])
            obs['agentview_image'][:]=0;obs['robot0_eef_pos'][0]=1e-15
            self.assertFalse(initial_comparison(obs,'actual BDDL',path)['passed'])

    def test_resumption_ignores_technical_and_rejects_duplicate(self):
        job=dict(group='perturbed350',source_index=0,case_hash='case')
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)/'perturbed350/case0000/v14_d2';base.mkdir(parents=True)
            for index,status in enumerate(('technical_failure','complete')):
                attempt=base/('attempt%03d'%index);attempt.mkdir()
                (attempt/'result.json').write_text(json.dumps(dict(status=status,manifest_hash='newD2',case_hash='case',mode='v14_d2',success=True if status=='complete' else None,steps=2)))
            good,bad=completed(temporary,job,'newD2');self.assertEqual((len(good),len(bad)),(1,1))
            (base/'attempt002').mkdir();(base/'attempt002/result.json').write_text((base/'attempt001/result.json').read_text())
            with self.assertRaisesRegex(ValueError,'Duplicate'): completed(temporary,job,'newD2')

    def test_cross_gpu_budget_allclose_diagnostic(self):
        c=policy.compare([0.],[.015625],.02)
        self.assertTrue(c['passed']);self.assertFalse(c['allclose_002'])
        self.assertFalse(policy.compare([0.],[.021],.02)['passed'])
        self.assertEqual(VIDEOS,[0,50,100,150,200,250,300])

    def test_real_episode_native_route_oracle_and_xyz(self):
        obs={k:np.zeros((4,5,3),np.uint8) if k.endswith('image') else np.zeros(4,np.float64) for k in KEYS}
        class Env:
            def __init__(self,*args,**kwargs):
                self.obs=obs;self.info=dict(instruction='BDDL actual',native_hz=20,settle_steps=5,initial_vector_sha256='init')
                self.env=SimpleNamespace(sim=SimpleNamespace(get_state=lambda:np.zeros(9)))
            def reset(self): pass
            def step_native(self,action): return obs,dict(success=True)
            def actual(self): return dict(eef_xyz_world=[0.,0.,0.])
            def close(self): pass
        calls=[]
        def rpc(port,request):
            calls.append(request)
            return {} if request['op']=='reset' else dict(actions=np.zeros((50,7)).tolist(),manifest_hash='D2',mode='v14_d2',execute_length=5,
                observed_calls=dict(ae_layer_calls=36,vision_encoder_calls=1,vlm_prefix_calls=1,verify_calls=0,constructor_full_record_calls=0))
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);np.savez_compressed(root/'initial.npz',**obs,instruction='BDDL actual')
            (root/'result.json').write_text(json.dumps(dict(native_info=dict(initial_vector_sha256='init'))))
            seed=int(digest(['plus-evaluation-v1','case',0])[:8],16)
            (root/'infer.jsonl').write_text(json.dumps(dict(seed=seed,input_hashes={k:runner.value_hash(v) for k,v in obs.items()}))+'\n')
            job=dict(case=dict(environment='libero_plus',mother_suite='libero_goal',native_max_steps=600),case_hash='case',group='perturbed350',source_index=0,
                paired_results=[dict(mode='full_d10',path=str(root/'result.json'))],initial_observation=dict(path=str(root/'initial.npz')),first_input=dict(path=str(root/'infer.jsonl')),video=False)
            a=SimpleNamespace(output=str(root/'attempt'),gpu=0,gpu_uuid='uuid',port=1,deadline_unix=0)
            with patch.object(runner,'NativeLiberoPlusEnv',Env),patch.object(runner,'verify_service'),patch.object(runner,'rpc',rpc):
                result=runner.episode(a,job,dict(manifest_hash='D2'))
            self.assertEqual(result['status'],'complete');self.assertTrue(result['success']);self.assertEqual(result['steps'],1)
            request=next(r for r in calls if r['op']=='predict');self.assertEqual(request['mode'],'v14_d2');self.assertEqual(request['seed'],seed)
            telemetry=[json.loads(line) for line in (root/'attempt/telemetry.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in telemetry],[0,1]);self.assertIn('eef_xyz_world',telemetry[1]['actual'])


if __name__=='__main__': unittest.main()

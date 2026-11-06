import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from . import run_plus


class PlusDriverTests(unittest.TestCase):
    def test_matching_head_only(self):
        self.assertEqual(run_plus.modes({'matching_draft':'draft_goal'}),['full_d10','v14_d10','draft_goal'])

    def test_native_rpc_and_oracle_interrupt(self):
        case=dict(environment='libero_original',mother_suite='libero_goal',native_max_steps=600)
        class Env:
            def __init__(self,*args,**kwargs):
                self.obs={key:np.zeros(shape,dtype=np.uint8 if len(shape)==3 else float) for key,shape in
                    [('agentview_image',(2,2,3)),('robot0_eye_in_hand_image',(2,2,3)),('robot0_eef_pos',(3,)),('robot0_eef_quat',(4,)),('robot0_gripper_qpos',(2,))]}
                self.info=dict(instruction='actual bddl',native_hz=20)
            def reset(self): return self.obs
            def step_native(self,action): return self.obs,dict(success=True,native_action=action.tolist())
            def actual(self): return dict(eef_xyz_world=[0,0,0])
            def close(self): pass
        calls=[]
        def rpc(port,request):
            calls.append(request)
            return {} if request['op']=='reset' else dict(actions=np.zeros((50,7)).tolist(),manifest_hash='m',execute_length=5)
        original=(run_plus.NativeLiberoPlusEnv,run_plus.verify_service,run_plus.rpc)
        run_plus.NativeLiberoPlusEnv=Env;run_plus.verify_service=lambda *args:None;run_plus.rpc=rpc
        try:
            with tempfile.TemporaryDirectory() as root:
                result=run_plus.episode(SimpleNamespace(output=str(Path(root)/'attempt'),gpu_uuid='uuid',gpu=3,port=1,deadline_unix=0,video=False),case,'full_d10',dict(manifest_hash='m'))
                self.assertEqual(result['status'],'complete');self.assertTrue(result['success']);self.assertEqual(result['steps'],1)
                predict=next(c for c in calls if c['op']=='predict')
                self.assertEqual(predict['environment'],'libero');self.assertEqual(predict['instruction'],'actual bddl')
                self.assertIn('observation',predict);self.assertNotIn('raw',predict)
        finally: run_plus.NativeLiberoPlusEnv,run_plus.verify_service,run_plus.rpc=original


if __name__=='__main__': unittest.main()

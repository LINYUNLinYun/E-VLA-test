import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from .native_exact import exact_calvin_action
from . import evaluation_v2 as evaluation


class ExactEvaluationTests(unittest.TestCase):
    def test_no_first6_clipping(self):
        raw=np.array([2.,-3.,.2,4.,-5.,.5,0.]);before=raw.copy()
        out,audit=exact_calvin_action(raw)
        np.testing.assert_array_equal(out[:6],raw[:6]);np.testing.assert_array_equal(raw,before)
        self.assertEqual(out[-1],-1);self.assertEqual(audit['outside_unit_components'],[0,1,3,4])
        self.assertFalse(audit['clipping'])

    def test_resume_requires_identity(self):
        seq={'index':0};plan={'plan_hash':'p','policy_manifest_hash':'m'}
        r=dict(status='complete',plan_hash='p',manifest_hash='m',mode='full_d10',sequence_index=0,sequence_hash=evaluation.digest(seq))
        self.assertTrue(evaluation.result_matches(r,plan,seq,'full_d10'))
        for key in ('status','plan_hash','manifest_hash','sequence_hash','mode'):
            wrong=dict(r);wrong[key]='wrong'
            self.assertFalse(evaluation.result_matches(wrong,plan,seq,'full_d10'))

    def test_oracle_interrupt_and_checkpoints(self):
        class Env:
            def reset_sequence(self,*args):
                self.obs=dict(robot_obs=np.zeros(15),scene_obs=np.zeros(24),rgb_obs=dict(rgb_static=np.zeros((2,2,3),dtype=np.uint8),rgb_gripper=np.zeros((2,2,3),dtype=np.uint8)))
            def start_subtask(self,task): return task
            def policy_raw(self,*args): return dict(base_rgb=self.obs['rgb_obs']['rgb_static'],wrist_rgb=self.obs['rgb_obs']['rgb_gripper'],state7=np.zeros(7))
            def step_native(self,action): return self.obs,True,dict(native_action=action.tolist())
            def actual(self): return dict(eef_xyz_world=[0,0,0])
        original=evaluation.rpc
        evaluation.rpc=lambda port,request: {} if request['op']=='reset' else dict(actions=np.zeros((5,7)).tolist(),manifest_hash='m',execute_length=5,action_contract='native_CALVIN_relative7_Euler_no_extraDelta')
        try:
            with tempfile.TemporaryDirectory() as root:
                directory=Path(root)/'seq0000'/'full_d10'/'attempt000'
                result=evaluation.run_case(SimpleNamespace(port=1,deadline_unix=0,video=False),Env(),dict(index=0,initial_condition={},tasks=['one','two']),
                    'full_d10',dict(plan_hash='p',policy_manifest_hash='m'),directory)
                self.assertEqual(result['status'],'complete');self.assertEqual(result['native_steps'],2)
                self.assertEqual(result['completed_subtasks'],2)
                self.assertTrue((directory/'task_1.json').exists())
                self.assertEqual(len((directory/'telemetry.jsonl').read_text().splitlines()),2)
                self.assertTrue((directory/'initial_native_observation.npz').exists())
        finally: evaluation.rpc=original


if __name__=='__main__': unittest.main()

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from . import evaluation_v3 as driver


class FakeEnv:
    instances=[]
    drift=False
    def __init__(self):
        self.instances.append(self);self.reset_calls=0;self.closed=False;self.task_starts=[]
    def reset_sequence(self,initial):
        self.reset_calls+=1
        self.obs=dict(robot_obs=np.zeros(15),scene_obs=np.zeros(24),rgb_obs=dict(rgb_static=np.zeros((2,2,3),dtype=np.uint8),rgb_gripper=np.zeros((2,2,3),dtype=np.uint8)))
        self.obs['robot_obs'][-1]=-1 if self.drift else 1
    def start_subtask(self,task): self.task_starts.append(task);return task
    def policy_raw(self,*args):
        return dict(base_rgb=self.obs['rgb_obs']['rgb_static'],wrist_rgb=self.obs['rgb_obs']['rgb_gripper'],state7=np.r_[self.obs['robot_obs'][:6],self.obs['robot_obs'][-1]])
    def step_native(self,action):
        self.obs['scene_obs'][0]+=1
        return self.obs,True,dict(native_action=action.tolist())
    def actual(self): return dict(eef_xyz_world=[0,0,0])
    def close(self): self.closed=True


class DriverV3Tests(unittest.TestCase):
    def setUp(self): FakeEnv.instances=[];FakeEnv.drift=False
    def test_prepared_once_no_second_reset(self):
        env=FakeEnv();env.reset_sequence({});proxy=driver.PreparedSequence(env,{})
        proxy.reset_sequence({});self.assertEqual(env.reset_calls,1)
        with self.assertRaises(ValueError): proxy.reset_sequence({})

    def test_fresh_condition_and_gate_before_every_quality_request(self):
        sequence=dict(index=0,initial_condition={},tasks=['a','b','c','d','e'])
        plan=dict(plan_hash='p',policy_manifest_hash='m');args=SimpleNamespace(port=1,deadline_unix=0,video=False)
        original=driver.rollout.rpc;calls=[]
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);reference,receipt=driver.reference_for_sequence(root,sequence,plan,FakeEnv)
            def rpc(port,request):
                if request['op']=='predict':
                    mode=request['mode'];gate=root/'seq0000'/mode/'prequality_attempt000/gate.json'
                    self.assertTrue(json.loads(gate.read_text())['passed'])
                    self.assertTrue(json.loads(gate.read_text())['before_any_quality_request']);calls.append(request)
                    return dict(actions=np.zeros((5,7)).tolist(),manifest_hash='m',execute_length=5,action_contract='native_CALVIN_relative7_Euler_no_extraDelta')
                return {}
            driver.rollout.rpc=rpc
            try:
                for mode in ('full_d10','v14_d10'):
                    value=driver.run_fresh_case(args,sequence,mode,plan,root/'seq0000'/mode/'attempt000',reference,receipt,FakeEnv)
                    self.assertEqual(value['status'],'complete');self.assertEqual(value['completed_subtasks'],5)
                    self.assertTrue(driver.result_matches(value,plan,sequence,mode))
                self.assertEqual(len(FakeEnv.instances),3)
                self.assertTrue(all(env.closed and env.reset_calls==1 for env in FakeEnv.instances))
                self.assertEqual([len(env.task_starts) for env in FakeEnv.instances],[0,5,5])
                self.assertEqual(len(calls),10)
                self.assertTrue((root/'seq0000/paired_reference/full_actual_reference.npz').exists())
            finally: driver.rollout.rpc=original

    def test_bad_actual_initial_is_invalid_not_policy_zero(self):
        sequence=dict(index=0,initial_condition={},tasks=['a']);plan=dict(plan_hash='p',policy_manifest_hash='m')
        with tempfile.TemporaryDirectory() as root:
            reference,receipt=driver.reference_for_sequence(root,sequence,plan,FakeEnv);FakeEnv.drift=True
            old=driver.rollout.rpc;driver.rollout.rpc=lambda *args: self.fail('No RPC may be sent on failed initial gate')
            try:
                value=driver.run_fresh_case(SimpleNamespace(deadline_unix=0),sequence,'full_d10',plan,
                    Path(root)/'seq0000/full_d10/attempt000',reference,receipt,FakeEnv)
                self.assertEqual(value['status'],'invalid_initial_condition');self.assertIsNone(value['success'])
                self.assertEqual(value['requests'],0);self.assertFalse(value['runtime_initial_gate']['passed'])
                self.assertTrue(FakeEnv.instances[-1].closed)
            finally: driver.rollout.rpc=old

    def test_old_v2_result_never_reused(self):
        sequence=dict(index=0);plan=dict(plan_hash='p',policy_manifest_hash='m')
        old=dict(status='complete',plan_hash='p',manifest_hash='m',mode='full_d10',sequence_index=0,sequence_hash=driver.digest(sequence))
        self.assertFalse(driver.result_matches(old,plan,sequence,'full_d10'))


if __name__=='__main__': unittest.main()

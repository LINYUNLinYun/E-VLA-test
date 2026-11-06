"""Pure CPU bridge contracts; fake simulator edges, no model or renderer."""
import math
from types import SimpleNamespace
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from .corrected_transfer_envs import (CorrectedCalvinTransfer,CorrectedGoogleTransfer,
    CorrectedPushTProjection,LiberoStateBridge,pose,rigid_matrix,source_native_reference)
from .policy_native import native_libero_state


def reference():
    q=Rotation.from_rotvec([3.2,0,0]).as_quat()
    return dict(robot_base_world_matrix=np.eye(4).tolist(),actual_eef_world_xyz=[.2,.3,.4],
        actual_eef_world_quaternion_xyzw=q.tolist(),actual_gripper_qpos=[.02,-.02],
        reference_state8=[.2,.3,.4,3.2-2*math.pi,0,0,.02,-.02],
        controller={'output_max':[.05,.05,.05,.5,.5,.5]})


class FakeCalvin:
    def __init__(self):
        self.info={'native_hz':30};self.resets=0
        self.obs={'robot_obs':np.r_[.2,.3,.4,Rotation.from_rotvec([3.2,0,0]).as_euler('xyz'),.04,np.zeros(7),1.],
            'rgb_obs':{'rgb_static':np.arange(12,dtype=np.uint8).reshape(2,2,3),
                       'rgb_gripper':np.ones((2,2,3),dtype=np.uint8)}}
        self.env=SimpleNamespace(robot=SimpleNamespace(robot_uid=3),cid=1,p=SimpleNamespace(
            getBasePositionAndOrientation=lambda *a,**k:([0,0,.12],[0,0,0,1]),
            getDynamicsInfo=lambda *a,**k:(None,None,None,[0,0,.12],[0,0,0,1])),step=self.native_step)
        self.oracle=SimpleNamespace(get_task_info_for_set=lambda before,after,tasks:tasks if after.get('success') else set())
    def native_step(self,command):
        self.command=command;return self.obs,0,False,{'success':True}
    def reset_sequence(self,condition):self.resets+=1;self.condition=condition;return self.obs
    def start_subtask(self,task):self.task=task;self.start_info={};return 'native instruction'
    def actual(self):return {'eef_xyz_world':self.obs['robot_obs'][:3].tolist()}
    def close(self):self.closed=True


class FakeGoogle:
    def __init__(self,frame='ee_align'):
        from .native_envs import simpler_native_contract
        contract=simpler_native_contract('google_robot_static');contract['frame']=frame
        self.info={'controller_contract':contract}
    def reset(self,seed,options=None):self.seed=seed;return {}
    def actual(self):
        q=reference()['actual_eef_world_quaternion_xyzw']
        return dict(eef_xyz_world=[.2,.3,.4],eef_quaternion_wxyz=[q[3]]+q[:3],
            robot_base_xyz_world=[0,0,0],robot_base_quaternion_wxyz=[1,0,0,0],
            actual_fingertip_world_positions={'left':[0,.02,0],'right':[0,-.02,0]})
    def rgb(self):return np.arange(12,dtype=np.uint8).reshape(2,2,3)
    def step_native(self,command):
        self.command=command;return {},dict(success=True,terminated=False,truncated=False,native_action=command.tolist())
    def close(self):pass


class FakePushT:
    def seed(self,seed):self.seed_value=seed
    def reset(self):return np.array([256.,256.,100.,200.,.1])
    def render(self,*args):return np.zeros((4,4,3),dtype=np.uint8)
    def step(self,command):self.command=command;return np.r_[command,100.,200.,.1],.5,False,{'coverage':.5}
    def close(self):pass


class CorrectedTransferTests(unittest.TestCase):
    def test_source_native_branch_not_old_principal_vector(self):
        r=reference();_,state,q=source_native_reference(r)
        self.assertGreater(state[3],math.pi)
        self.assertAlmostEqual(state[3],3.2)
        self.assertFalse(np.allclose(state,r['reference_state8']))
        self.assertTrue(np.allclose(state,native_libero_state(r['actual_eef_world_xyz'],q,r['actual_gripper_qpos'])))

    def test_continuity_and_episode_reset(self):
        r=reference();bridge=LiberoStateBridge(r)
        for angle in [3.1,3.14,3.2,3.3]:
            state=bridge.encode(pose([.2,.3,.4],Rotation.from_rotvec([angle,0,0]).as_matrix()),np.eye(4),.04)
            self.assertAlmostEqual(float(state[3]),angle,places=5)
        bridge.reset();self.assertTrue(np.array_equal(bridge.lift.previous,bridge.source_q))
        self.assertTrue(np.allclose(bridge.projection_state([256,256]),bridge.source_state))

    def test_base_frame_alignment_and_physical_delta(self):
        r=reference();source=np.eye(4);source[:3,3]=[1,2,3];r['robot_base_world_matrix']=source.tolist()
        bridge=LiberoStateBridge(r);target=pose([4,5,6],Rotation.from_euler('z',math.pi/2).as_matrix())
        local=pose([.1,.2,.3],Rotation.from_rotvec([3.2,0,0]).as_matrix())
        state=bridge.encode(target.dot(local),target,.04)
        self.assertTrue(np.allclose(state[:3],[1.1,2.2,3.3]))
        with self.assertRaises(ValueError):rigid_matrix(np.zeros((4,4)))
        reflected=np.eye(4);reflected[0,0]=-1
        with self.assertRaises(ValueError):rigid_matrix(reflected)

    def test_formal_calvin_reset_carry_and_absolute_source_osc(self):
        native=FakeCalvin();env=CorrectedCalvinTransfer(reference(),native=native)
        raw=env.reset_sequence({'slider':'left'});env.start_subtask('taskA')
        self.assertEqual(native.resets,1);self.assertTrue(np.allclose(env.base,np.eye(4)))
        self.assertGreater(raw['state8'][3],math.pi)
        self.assertTrue(np.array_equal(raw['base_rgb'],native.obs['rgb_obs']['rgb_static']))
        success,terminated,audit=env.step([2,-2,0,0,0,2,2])
        self.assertTrue(success);self.assertFalse(terminated)
        self.assertTrue(np.allclose(native.command[0],[.25,.25,.4]))
        actual=Rotation.from_quat(native.command[1]).as_matrix()
        expected=Rotation.from_rotvec([0,0,.5]).as_matrix().dot(env.eef()[:3,:3])
        self.assertTrue(np.allclose(actual,expected));self.assertEqual(native.command[2],-1)
        self.assertEqual(audit['clipped_components'],[0,1,5,6])
        self.assertEqual(audit['native_action_kind'],'absolute_pose')
        env.start_subtask('taskB');self.assertEqual(native.resets,1)
        self.assertFalse(env.info['target_trained_native_policy'])

    def test_google_not_widowx_mapping(self):
        native=FakeGoogle();env=CorrectedGoogleTransfer('google_robot_move_near',reference(),native=native)
        raw=env.reset(49);self.assertGreater(raw['state8'][3],math.pi)
        self.assertEqual(native.seed,49);self.assertFalse(np.any(raw['wrist_rgb']))
        self.assertTrue(np.array_equal(raw['base_rgb'],native.rgb()))
        self.assertTrue(np.allclose(raw['state8'][-2:],[.02,-.02]))
        success,done,audit=env.step([1,0,0,0,1,0,1])
        self.assertTrue(success);self.assertFalse(done)
        self.assertTrue(np.allclose(native.command,[.05,0,0,0,.5,0,1]))
        self.assertIn('Google',audit['gripper'])
        with self.assertRaises(ValueError):CorrectedGoogleTransfer('widowx_stack_cube',reference(),native=native)
        with self.assertRaises(ValueError):CorrectedGoogleTransfer('google_robot_pick_coke_can',reference(),native=FakeGoogle('ee_align2'))

    def test_rotated_source_and_target_action_axes(self):
        native=FakeCalvin();target_q=Rotation.from_euler('z',math.pi/2).as_quat().tolist()
        native.env.p.getBasePositionAndOrientation=lambda *a,**k:([0,0,.12],target_q)
        env=CorrectedCalvinTransfer(reference(),native=native)
        env.reset_sequence({});env.start_subtask('task')
        env.step([1,0,0,0,0,0,0])
        self.assertTrue(np.allclose(native.command[0],[.2,.35,.4]))

    def test_pusht_is_explicit_projection_not_native_pi0(self):
        native=FakePushT();env=CorrectedPushTProjection(reference(),native)
        raw=env.reset(99);self.assertGreater(raw['state8'][3],math.pi)
        self.assertEqual(native.seed_value,99);self.assertFalse(env.info['comparable_to_arm_success'])
        _,_,audit=env.step([20,1,1,1,1,1,1])
        self.assertTrue(np.allclose(native.command,[256+.05*512/.6]*2))
        self.assertEqual(audit['ignored_channels'],[2,3,4,5,6]);self.assertFalse(audit['native_2d_pi0'])
        native.reset=lambda:np.array([510.,510.,0.,0.,0.]);env.reset(100)
        env.step([1,1,0,0,0,0,0]);self.assertTrue(np.allclose(native.command,[512,512]))

    def test_invalid_reference_or_physical_inputs_rejected(self):
        r=reference();r['actual_eef_world_quaternion_xyzw']=[0,0,0,2]
        with self.assertRaises(ValueError):LiberoStateBridge(r)
        bridge=LiberoStateBridge(reference())
        with self.assertRaises(ValueError):bridge.encode(np.eye(4),np.eye(4),-.01)
        with self.assertRaises(ValueError):bridge.projection_state([float('nan'),0])


if __name__=='__main__':unittest.main()

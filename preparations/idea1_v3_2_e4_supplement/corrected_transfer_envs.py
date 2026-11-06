"""Corrected B-family LIBERO cross-embodiment bridge, never native fine-tunes.

Import and pure conversion functions are CPU-only. Environment constructors
are explicit; Google rendering still requires the existing exact GPU grant.
No case enumeration lives here: the frozen larger-sample driver owns it.
"""
import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation

from preparations.idea1_v3_2.transfer_envs import (
    align_pose_to_source, calvin_absolute_action, pose, pusht_native_action,
    simpler_native_action,
)
from .policy_native import ContinuousQuaternionLift, native_libero_state, unit_quaternion_xyzw

GOOGLE_TASKS = ('google_robot_pick_coke_can', 'google_robot_move_near')


def rigid_matrix(value):
    matrix = np.asarray(value,dtype=np.float64)
    if matrix.shape != (4,4) or not np.isfinite(matrix).all():
        raise ValueError('Finite rigid 4x4 transform required')
    rotation = matrix[:3,:3]
    if (not np.allclose(matrix[3],[0,0,0,1],atol=1e-8) or
            not np.allclose(rotation.T.dot(rotation),np.eye(3),atol=1e-6) or
            not np.isclose(np.linalg.det(rotation),1.,atol=1e-6)):
        raise ValueError('Proper rigid robot-base transform required')
    return matrix


def source_native_reference(reference):
    """Reconstruct the actual native branch, not the old principal-rotvec state."""
    base = rigid_matrix(reference['robot_base_world_matrix'])
    quaternion = unit_quaternion_xyzw(reference['actual_eef_world_quaternion_xyzw'])
    state = native_libero_state(reference['actual_eef_world_xyz'],quaternion,reference['actual_gripper_qpos'])
    if 'controller' in reference:
        controller = reference['controller']
        if not np.allclose(controller['output_max'],[.05,.05,.05,.5,.5,.5],atol=1e-9):
            raise ValueError('Factual source OSC scale differs from the pinned bridge')
    return base,state,quaternion


def _legacy_reference(reference):
    """Old action math needs this field only for validation, not state encoding."""
    _,state,_ = source_native_reference(reference)
    return dict(reference,reference_state8=state.tolist())


class LiberoStateBridge:
    def __init__(self,reference):
        self.source_base,self.source_state,self.source_q = source_native_reference(reference)
        self.lift = ContinuousQuaternionLift(self.source_q)

    def reset(self):
        self.lift.reset()

    def encode(self,eef_world,target_base_world,width_m):
        width = float(width_m)
        if not np.isfinite(width) or width < 0:
            raise ValueError('Actual nonnegative gripper opening width required')
        aligned = align_pose_to_source(rigid_matrix(eef_world),rigid_matrix(target_base_world),self.source_base)
        self.lift.encode(Rotation.from_matrix(aligned[:3,:3]).as_quat())
        return native_libero_state(aligned[:3,3],self.lift.previous,[width/2.,-width/2.]).astype(np.float32)

    def projection_state(self,agent_xy):
        xy = np.asarray(agent_xy,dtype=np.float64)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError('Actual finite native PushT agent XY required')
        state = self.source_state.copy()
        state[:2] += (xy-256.)*(.6/512.)
        return state.astype(np.float32)


def bridge_metadata(reference,environment):
    _,native_state,q = source_native_reference(reference)
    return dict(family='libero_transfer_corrected_B',environment=environment,
        source_reference_sha256=hashlib.sha256(json.dumps(reference,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        state_rotation='native LIBERO noncanonical axis-angle; continuous quaternion lift anchored actual source q',
        source_quaternion_xyzw=q.tolist(),source_native_state8=native_state.tolist(),
        source_action='native decoded LIBERO OSC7 clipped[-1,1]; .05m translation/.5rad rotvec; positive gripper closes',
        tcp_alignment='identity TCP offset assumption, not estimated from target outcomes',
        sampling='driver owns preregistered canonical100 sequence/50+init/100seed schedules; no debug8/5/5 fallback',
        target_trained_native_policy=False)


def calvin_base_world(native):
    simulator = native.env
    position,q = simulator.p.getBasePositionAndOrientation(simulator.robot.robot_uid,physicsClientId=simulator.cid)
    dynamics = simulator.p.getDynamicsInfo(simulator.robot.robot_uid,-1,physicsClientId=simulator.cid)
    com = pose(position,Rotation.from_quat(q).as_matrix())
    inertial = pose(dynamics[3],Rotation.from_quat(dynamics[4]).as_matrix())
    return rigid_matrix(com.dot(np.linalg.inv(inertial))),dict(
        pybullet_base_com_world_matrix=com.tolist(),base_local_inertial_matrix=inertial.tolist())


class CorrectedCalvinTransfer:
    """Formal D scene/reset/oracle, but source LIBERO physical delta→absolute pose."""
    def __init__(self,reference,native=None):
        self.reference=_legacy_reference(reference);self.bridge=LiberoStateBridge(reference)
        if native is None:
            from .native_envs import NativeCalvinEnv
            native = NativeCalvinEnv()
        self.native=native
        self.base,base_audit=calvin_base_world(native)
        self.info=dict(native.info,**bridge_metadata(reference,'calvin'),**base_audit,
            target_robot_base_world_matrix=self.base.tolist(),target_controller='native absolute worldXYZ + xyzw quaternion + gripper sign',
            policy_input_contract='virtual source-aligned XYZ meters + native-branch axis-angle radians + virtual signed Panda fingers meters',
            distinction='not target-trained CALVIN native relative H5/.02m/.05Euler branch',
            absent_wrist=False,virtual_fingers='actual opening width maps to signed Panda width/2,-width/2',
            cadence='sourceR5 retained at native30Hz; not time-matched to source20Hz')

    def reset_sequence(self,initial_condition):
        self.bridge.reset();self.native.reset_sequence(initial_condition)
        return self.raw()

    def start_subtask(self,task):
        # Formal evaluator carries scene and quaternion lift across subtasks.
        return self.native.start_subtask(task)

    def eef(self):
        robot=np.asarray(self.native.obs['robot_obs'])
        return pose(robot[:3],Rotation.from_euler('xyz',robot[3:6]).as_matrix())

    def raw(self):
        obs=self.native.obs
        return dict(base_rgb=np.asarray(obs['rgb_obs']['rgb_static'],dtype=np.uint8),
            wrist_rgb=np.asarray(obs['rgb_obs']['rgb_gripper'],dtype=np.uint8),
            state8=self.bridge.encode(self.eef(),self.base,float(obs['robot_obs'][6])))

    def step(self,action):
        command,audit=calvin_absolute_action(action,self.eef(),self.base,self.reference)
        self.native.obs,_,_,info=self.native.env.step(command)
        task=self.native.task
        success=task in self.native.oracle.get_task_info_for_set(self.native.start_info,info,{task})
        audit.update(family='libero_transfer_corrected_B',native_action_kind='absolute_pose',
                     target_trained_native_relative=False)
        return bool(success),False,audit

    def telemetry(self):return self.native.actual()
    def close(self):self.native.close()


class CorrectedGoogleTransfer:
    """Google-only ee_align physical controller; never apply WidowX conventions."""
    def __init__(self,task,reference,native=None,**renderer_kwargs):
        if task not in GOOGLE_TASKS:
            raise ValueError('Initial corrected Google bridge supports only frozen coke and move_near tasks')
        self.reference=_legacy_reference(reference);self.bridge=LiberoStateBridge(reference)
        if native is None:
            from .native_envs import NativeSimplerEnv
            native=NativeSimplerEnv(task,**renderer_kwargs)
        self.native=native
        contract=native.info['controller_contract']
        from .native_envs import simpler_native_contract
        expected=simpler_native_contract('google_robot_static')
        if any(contract.get(key)!=expected[key] for key in ('control_mode','frame','normalize_arm','use_target_arm','gripper','native_hz')):
            raise ValueError('Google actual-origin physical ee_align controller required, not WidowX ee_align2')
        self.info=dict(native.info,**bridge_metadata(reference,'simpler'),task=task,absent_wrist=True,
            missing_wrist='explicit black placeholder, not a real camera observation',
            virtual_fingers='measured fingertip distance maps to signed Panda width/2,-width/2',
            cadence='sourceR5 retained at nativeGoogle3Hz; not time-matched to source20Hz')

    def reset(self,seed,options=None):
        self.bridge.reset();self.native.reset(seed=seed,options=options)
        if hasattr(self.native,'native'):
            self.info['instruction']=str(self.native.native.get_language_instruction())
        return self.raw()

    def _poses(self):
        actual=self.native.actual()
        eef=pose(actual['eef_xyz_world'],Rotation.from_quat(np.asarray(actual['eef_quaternion_wxyz'])[[1,2,3,0]]).as_matrix())
        base=pose(actual['robot_base_xyz_world'],Rotation.from_quat(np.asarray(actual['robot_base_quaternion_wxyz'])[[1,2,3,0]]).as_matrix())
        tips=list(actual['actual_fingertip_world_positions'].values())
        if len(tips)!=2:raise ValueError('Two actual Google fingertip positions required')
        width=float(np.linalg.norm(np.asarray(tips[0])-np.asarray(tips[1])))
        return eef,base,width

    def raw(self):
        eef,base,width=self._poses();image=self.native.rgb()
        self.info['target_robot_base_world_matrix']=base.tolist()
        return dict(base_rgb=image,wrist_rgb=np.zeros_like(image),state8=self.bridge.encode(eef,base,width))

    def step(self,action):
        command,audit=simpler_native_action(action,self.reference)
        _,native_audit=self.native.step_native(command)
        audit.update(native_controller_audit=native_audit,absent_wrist=True,
            gripper='Google positive normalized target delta closes; no WidowX absolute-open mapping')
        return bool(native_audit['success']),bool(native_audit['terminated'] or native_audit['truncated']),audit

    def telemetry(self):return self.native.actual()
    def close(self):self.native.close()


class CorrectedPushTProjection:
    """Explicit LIBERO7→XY stress projection, separate from native2D PI0."""
    def __init__(self,reference,native):
        self.native=native;self.bridge=LiberoStateBridge(reference)
        self.info=dict(bridge_metadata(reference,'pusht'),projection='512px↔0.6m; XY only, absolute action clipped to pixel workspace',
            fixed_synthetic_channels='source factual native-branch Z/axis-angle/fingers, not actual arm state',
            ignored_source_action_channels=[2,3,4,5,6],absent_wrist=True,
            missing_wrist='explicit black placeholder',comparable_to_arm_success=False,
            cadence='sourceR5 retained at native10Hz, not nativePI0H50/R50')

    def reset(self,seed):
        self.bridge.reset();self.native.seed(int(seed));self.obs=self.native.reset()
        return self.raw()

    def raw(self):
        image=np.asarray(self.native.render('rgb_array'),dtype=np.uint8)
        return dict(base_rgb=image,wrist_rgb=np.zeros_like(image),state8=self.bridge.projection_state(self.obs[:2]))

    def step(self,action):
        command,audit=pusht_native_action(action,self.obs[:2])
        self.obs,reward,done,info=self.native.step(command)
        audit.update(native_reward=float(reward),native_info={k:np.asarray(v).tolist() for k,v in info.items()},
                     stress_projection=True,native_2d_pi0=False)
        return bool(done),False,audit

    def telemetry(self):
        return dict(agent_xy_pixels=np.asarray(self.obs[:2]).tolist(),block_xy_angle=np.asarray(self.obs[2:]).tolist())
    def close(self):self.native.close()

"""Native environment adapters for an explicitly zero-shot action-interface probe.

Compatible with the existing Python 3.8/3.9/3.10 environment interpreters.
No model imports or renderer construction occur at module import time.
"""
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
CALVIN_DATA = ROOT/'calvin/dataset/calvin_debug_dataset/validation'
MODES = ('full_d10','v14_d10','draft_goal','draft_object','draft_spatial','draft_10')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def case_list(environment):
    if environment == 'calvin':
        annotations = np.load(str(CALVIN_DATA/'lang_annotations/auto_lang_ann.npy'),allow_pickle=True).item()
        return [dict(case=i,seed=i,task=str(task),instruction=str(instruction),
                     start_frame=int(interval[0]),end_frame=int(interval[1]),max_steps=360,native_hz=30)
                for i,(task,instruction,interval) in enumerate(zip(annotations['language']['task'],
                    annotations['language']['ann'],annotations['info']['indx']))]
    if environment == 'simpler':
        return [dict(case=i,seed=i,task='google_robot_pick_coke_can',instruction='pick coke can',
                     max_steps=80,native_hz=3) for i in range(5)]
    if environment == 'pusht':
        return [dict(case=i,seed=i,task='pusht_projection_stress_probe',
                     instruction='push the T-shaped block onto the matching target',max_steps=300,native_hz=10)
                for i in range(5)]
    raise ValueError('Unknown environment: '+environment)


def pose(position,rotation):
    result=np.eye(4,dtype=np.float64)
    result[:3,:3]=np.asarray(rotation,dtype=np.float64)
    result[:3,3]=np.asarray(position,dtype=np.float64)
    return result


def checked_source(reference):
    source=np.asarray(reference['robot_base_world_matrix'],dtype=np.float64)
    state=np.asarray(reference['reference_state8'],dtype=np.float64)
    if source.shape!=(4,4) or state.shape!=(8,) or not np.isfinite(source).all() or not np.isfinite(state).all():
        raise ValueError('Invalid factual LIBERO source reference')
    if not np.allclose(source[:3,:3].T.dot(source[:3,:3]),np.eye(3),atol=1e-6):
        raise ValueError('Source rotation is not orthonormal')
    return source,state


def align_pose_to_source(eef_world,target_base_world,source_base_world):
    """Preserve robot-base-relative pose; identity TCP convention is explicit."""
    return source_base_world.dot(np.linalg.inv(target_base_world)).dot(eef_world)


def state8_from_pose(eef_world,target_base_world,reference,width):
    source,_=checked_source(reference)
    aligned=align_pose_to_source(eef_world,target_base_world,source)
    return np.r_[aligned[:3,3],Rotation.from_matrix(aligned[:3,:3]).as_rotvec(),width/2.,-width/2.].astype(np.float32)


def clip_libero_action(action):
    action=np.asarray(action,dtype=np.float64)
    if action.shape!=(7,) or not np.isfinite(action).all():
        raise ValueError('Expected finite native decoded 7D action')
    clipped=np.clip(action,-1.,1.)
    return clipped,dict(raw=action.tolist(),clipped=clipped.tolist(),
        clipped_components=np.flatnonzero(action!=clipped).tolist())


def physical_delta(action,source_base_world,target_base_world):
    clipped,audit=clip_libero_action(action)
    rotation=target_base_world[:3,:3].dot(source_base_world[:3,:3].T)
    translation=rotation.dot(clipped[:3]*.05)
    rotation_vector=rotation.dot(clipped[3:6]*.5)
    audit.update(physical_translation_target_world_m=translation.tolist(),
                 physical_rotation_vector_target_world_rad=rotation_vector.tolist())
    return translation,rotation_vector,clipped[-1],audit


def calvin_absolute_action(action,eef_world,target_base_world,reference):
    source,_=checked_source(reference)
    translation,rotation_vector,gripper,audit=physical_delta(action,source,target_base_world)
    target_rotation=Rotation.from_rotvec(rotation_vector).as_matrix().dot(eef_world[:3,:3])
    command=[(eef_world[:3,3]+translation).tolist(),Rotation.from_matrix(target_rotation).as_quat().tolist(),
             -1 if gripper>0 else 1]
    audit.update(native_action=command,native_rotation='xyzw quaternion; left composition matching LIBERO OSC',
                 gripper='LIBERO positive closes; CALVIN negative closes')
    return command,audit


def simpler_native_action(action,reference):
    source,_=checked_source(reference)
    clipped,audit=clip_libero_action(action)
    # ee_align means translation/rotvec in robot-base axes, not EE axes.
    command=np.r_[source[:3,:3].T.dot(clipped[:3]*.05),
                  source[:3,:3].T.dot(clipped[3:6]*.5),clipped[-1]]
    audit.update(native_action=command.tolist(),
        convention='native ee_align physical meters/radians; gripper closedness delta positive closes; no arbitrary scale')
    return command,audit


def pusht_state8(agent_xy,reference):
    _,state=checked_source(reference)
    state=state.copy()
    state[:2]+= (np.asarray(agent_xy)-256.)*(.6/512.)
    return state.astype(np.float32)


def pusht_native_action(action,agent_xy):
    clipped,audit=clip_libero_action(action)
    target=np.clip(np.asarray(agent_xy,dtype=np.float64)+clipped[:2]*.05*(512./.6),0.,512.)
    audit.update(native_action=target.tolist(),ignored_channels=[2,3,4,5,6],
                 convention='a-priori 512px equals0.6m, absolute pixel target from actual XY plus physical source XY delta')
    return target,audit


def gpu_renderer_binding(gpu,gpu_uuid,require_idle=False):
    """Resolve UUID to exact physical PCI bus before any SAPIEN construction."""
    if gpu is None or not gpu_uuid:
        raise RuntimeError('RENDER_BINDING_REQUIRED: assigned physical GPU and UUID are required')
    output=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=uuid,pci.bus_id',
                                    '--format=csv,noheader'],universal_newlines=True).strip()
    actual_uuid,pci=[value.strip() for value in output.split(',')]
    if actual_uuid!=gpu_uuid:
        raise RuntimeError('RENDER_BINDING_MISMATCH: physical GPU UUID differs')
    compute=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid',
                                    '--format=csv,noheader,nounits'],universal_newlines=True).strip()
    if require_idle and compute:
        raise RuntimeError('RENDER_DEVICE_BUSY: assigned GPU has compute PIDs before renderer construction: '+compute)
    bus=pci.split(':')[1]
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (gpu_uuid,str(gpu)):
        raise RuntimeError('RENDER_BINDING_MISMATCH: CUDA_VISIBLE_DEVICES must be assigned UUID or physical index')
    return dict(gpu=int(gpu),gpu_uuid=gpu_uuid,pci_bus_id=pci,
        preconstructor_compute_pids=compute.splitlines(),idle_required=require_idle,
        sapien_device='pci:'+bus,verification='SAPIEN documented exact PCI device selector resolved from assigned NVIDIA UUID; no default selector')


class CalvinAdapter:
    def __init__(self,case,reference,**unused):
        import hydra
        from omegaconf import OmegaConf
        config=OmegaConf.load(str(CALVIN_DATA/'.hydra/merged_config.yaml'))
        for name in list(config.cameras):
            if name not in ('static','gripper'): del config.cameras[name]
        # Native PyBullet TinyRenderer avoids an unbound EGL/GPU context.
        self.env=hydra.utils.instantiate(config.env,show_gui=False,use_vr=False,use_scene_info=True,use_egl=False)
        self.oracle=hydra.utils.instantiate(config.tasks)
        self.case,self.reference=case,reference
        base_position,base_quat=self.env.p.getBasePositionAndOrientation(self.env.robot.robot_uid,physicsClientId=self.env.cid)
        # PyBullet reports the inertial COM pose, not the URDF robot-base frame.
        dynamics=self.env.p.getDynamicsInfo(self.env.robot.robot_uid,-1,physicsClientId=self.env.cid)
        world_com=pose(base_position,Rotation.from_quat(base_quat).as_matrix())
        local_inertial=pose(dynamics[3],Rotation.from_quat(dynamics[4]).as_matrix())
        self.base=world_com.dot(np.linalg.inv(local_inertial))
        self.hz=float(self.env.control_freq)
        if self.hz!=30: raise ValueError('CALVIN native30Hz required')
        self.info=dict(environment='calvin',renderer='native PyBullet TinyRenderer CPU',native_hz=self.hz,
            source_semantics='robot-base-aligned pose with identity TCP offset assumption',
            target_robot_base_world_matrix=self.base.tolist(),oracle='Tasks.get_task_info_for_set(start_info,current_info,{annotated_task})',
            pybullet_base_com_world_matrix=world_com.tolist(),base_local_inertial_matrix=local_inertial.tolist(),
            confounds=['untrained cross-embodiment state/action mapping','identity TCP offset assumption','native30Hz versus source20Hz; R5 retained'])
        provenance=[CALVIN_DATA/'.hydra/merged_config.yaml',CALVIN_DATA/'lang_annotations/auto_lang_ann.npy',
            ROOT/'calvin/calvin_env/data/franka_panda/panda_longer_finger.urdf',
            Path(inspect.getfile(type(self.env.robot))),Path(inspect.getfile(type(self.oracle)))]
        self.info['native_asset_controller_sha256']={str(p):sha(p) for p in provenance}

    def reset(self):
        self.env.seed(self.case['seed'])
        frame=CALVIN_DATA/('episode_%07d.npz'%self.case['start_frame'])
        with np.load(str(frame),allow_pickle=False) as data:
            self.obs=self.env.reset(robot_obs=data['robot_obs'],scene_obs=data['scene_obs'])
        self.start_info=self.env.get_info()
        self.current_info=self.start_info
        self.info.update(frame_path=str(frame),frame_sha256=sha(frame),instruction=self.case['instruction'],task=self.case['task'])
        return self.raw()

    def eef(self):
        robot=self.obs['robot_obs']
        return pose(robot[:3],Rotation.from_euler('xyz',robot[3:6]).as_matrix())

    def raw(self):
        return dict(base_rgb=np.asarray(self.obs['rgb_obs']['rgb_static'],dtype=np.uint8),
                    wrist_rgb=np.asarray(self.obs['rgb_obs']['rgb_gripper'],dtype=np.uint8),
                    state8=state8_from_pose(self.eef(),self.base,self.reference,float(self.obs['robot_obs'][6])))

    def telemetry(self):
        return dict(eef_xyz_world=self.eef()[:3,3].tolist(),eef_rotation_world=self.eef()[:3,:3].tolist(),
                    actual_gripper_width_m=float(self.obs['robot_obs'][6]))

    def step(self,action):
        command,audit=calvin_absolute_action(action,self.eef(),self.base,self.reference)
        self.obs,_,_,self.current_info=self.env.step(command)
        success=self.case['task'] in self.oracle.get_task_info_for_set(self.start_info,self.current_info,{self.case['task']})
        return bool(success),False,audit

    def close(self):
        self.env.close()
        self.env.cid=-1  # native destructor otherwise disconnects the same client twice


class PushTAdapter:
    def __init__(self,case,reference,**unused):
        os.environ.setdefault('SDL_VIDEODRIVER','dummy')
        from diffusion_policy.env.pusht.pusht_env import PushTEnv
        self.env=PushTEnv(render_size=256,render_action=False)
        self.case,self.reference=case,reference
        self.hz=float(self.env.control_hz)
        self.info=dict(environment='pusht',native_hz=self.hz,renderer='native pygame CPU',
            oracle='native coverage > success_threshold (0.95)',projection='512 pixels correspond to0.6m; center anchored to factual source referenceXY',
            confounds=['2D projection stress probe, no native 7D robot policy meaning','ignore Z/rotation/gripper actions','Z/orientation/gripper state fixed from real source reference','missing wrist is black padding','native10Hz versus source20Hz; R5 retained'])
        self.info['native_asset_controller_sha256']={inspect.getfile(PushTEnv):sha(inspect.getfile(PushTEnv))}

    def reset(self):
        self.env.seed(self.case['seed'])
        self.obs=self.env.reset()
        self.info['success_threshold']=float(self.env.success_threshold)
        return self.raw()

    def raw(self):
        image=np.asarray(self.env.render('rgb_array'),dtype=np.uint8)
        return dict(base_rgb=image,wrist_rgb=np.zeros_like(image),state8=pusht_state8(self.obs[:2],self.reference))

    def telemetry(self):
        return dict(agent_xy_pixels=np.asarray(self.obs[:2]).tolist(),block_xy_angle=np.asarray(self.obs[2:]).tolist())

    def step(self,action):
        command,audit=pusht_native_action(action,self.obs[:2])
        self.obs,reward,done,info=self.env.step(command)
        audit.update(native_reward=float(reward),native_info={k:np.asarray(v).tolist() for k,v in info.items()})
        return bool(done),False,audit

    def close(self): self.env.close()


class SimplerAdapter:
    def __init__(self,case,reference,gpu=None,gpu_uuid=None,**unused):
        binding=gpu_renderer_binding(gpu,gpu_uuid,require_idle=bool(unused.get('require_idle',False)))
        import simpler_env
        self.env=simpler_env.make(case['task'],renderer_kwargs=dict(device=binding['sapien_device'],offscreen_only=True))
        self.native=self.env.unwrapped
        self.case,self.reference=case,reference
        self.hz=float(self.native.control_freq)
        config=self.native.agent.controller.configs
        self.info=dict(environment='simpler',native_hz=self.hz,renderer_binding=binding,
            controller_mode=self.native.control_mode,controller_configs=repr(config),
            oracle='native task evaluate/info success, not termination or model confidence',
            confounds=['untrained robot-base alignment; identity TCP offset assumption','virtual signed Panda finger coordinates from actual fingertip distance','black missing wrist padding','native3Hz versus source20Hz; R5 retained'])
        arm=self.native.agent.controller.controllers['arm']
        if arm.config.frame!='ee_align' or arm.config.normalize_action or not arm.config.use_delta:
            raise RuntimeError('Unsupported native Google arm controller semantics')
        grip=self.native.agent.controller.controllers['gripper']
        if not grip.config.use_delta or not grip.config.normalize_action:
            raise RuntimeError('Unsupported native Google gripper semantics')
        from mani_skill2_real2sim import format_path
        provenance=[Path(format_path(self.native.agent.urdf_path)),
            Path(inspect.getfile(type(arm))),Path(inspect.getfile(type(grip))),Path(inspect.getfile(type(self.native)))]
        self.info['native_asset_controller_sha256']={str(p):sha(p) for p in provenance}
        self.info['model_database_sha256']=hashlib.sha256(json.dumps(self.native.model_db,sort_keys=True).encode()).hexdigest()

    def reset(self):
        self.obs,self.reset_info=self.env.reset(seed=self.case['seed'])
        actual=str(self.native.get_language_instruction())
        if actual!=self.case['instruction']:
            raise ValueError('Native task instruction differs from frozen case: '+actual)
        self.base=pose(self.native.agent.robot.pose.p,Rotation.from_quat(self.native.agent.robot.pose.q[[1,2,3,0]]).as_matrix())
        self.info.update(target_robot_base_world_matrix=self.base.tolist(),instruction=actual)
        return self.raw()

    def eef(self):
        p=self.native.tcp.pose
        return pose(p.p,Rotation.from_quat(p.q[[1,2,3,0]]).as_matrix())

    def raw(self):
        image=np.asarray(self.obs['image']['overhead_camera']['rgb'],dtype=np.uint8)
        # This checkout's convenience get_fingers_info calls removed
        # get_global_pose; current SAPIEN exposes the same actual world pose.
        right=np.asarray(self.native.agent.finger_right_tip_link.pose.p)
        left=np.asarray(self.native.agent.finger_left_tip_link.pose.p)
        width=float(np.linalg.norm(right-left))
        return dict(base_rgb=image,wrist_rgb=np.zeros_like(image),state8=state8_from_pose(self.eef(),self.base,self.reference,width))

    def telemetry(self): return dict(eef_xyz_world=self.eef()[:3,3].tolist(),eef_rotation_world=self.eef()[:3,:3].tolist())

    def step(self,action):
        command,audit=simpler_native_action(action,self.reference)
        space=self.env.action_space
        clipped=np.clip(command,space.low,space.high)
        audit.update(native_space_low=space.low.tolist(),native_space_high=space.high.tolist(),native_space_clipped=bool(np.any(command!=clipped)))
        self.obs,reward,terminated,truncated,info=self.env.step(clipped)
        if 'success' not in info: raise RuntimeError('Native Simpler oracle success missing')
        audit.update(native_reward=float(reward),native_terminated=bool(terminated),native_truncated=bool(truncated),native_success=bool(info['success']))
        return bool(info['success']),bool(terminated or truncated),audit

    def close(self): self.env.close()


def make_environment(environment,case,reference,**kwargs):
    checked_source(reference)
    classes=dict(calvin=CalvinAdapter,simpler=SimplerAdapter,pusht=PushTAdapter)
    return classes[environment](case,reference,**kwargs)

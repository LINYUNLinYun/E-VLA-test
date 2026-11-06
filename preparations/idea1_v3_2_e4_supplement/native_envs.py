"""Native task/control contracts and positive gates; no LIBERO action projection.

The formal CALVIN evaluation set is shared by ABC->D and ABCD->D. Those names
identify different training distributions, not two different scene-D test sets.
Module import is CPU-only and compatible with the existing Python3.8 env.
"""
import ast
import contextlib
import hashlib
import inspect
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2]
CALVIN=ROOT/'calvin'
DEBUG_D=CALVIN/'dataset/calvin_debug_dataset/validation'
CALVIN_ENV=CALVIN/'calvin_env'
CALVIN_MODELS=CALVIN/'calvin_models'
OFFICIAL_EVALUATOR=CALVIN_MODELS/'calvin_agent/evaluation/evaluate_policy.py'
OFFICIAL_SEQUENCES=CALVIN_MODELS/'calvin_agent/evaluation/multistep_sequences.py'
OFFICIAL_INITIAL=CALVIN_MODELS/'calvin_agent/evaluation/utils.py'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.temporary')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False))
    temporary.replace(path)


def formal_calvin_contract():
    tree=ast.parse(OFFICIAL_EVALUATOR.read_text())
    constants={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Num):
            for target in node.targets:
                if isinstance(target,ast.Name): constants[target.id]=node.value.n
    return dict(name='CALVIN official local checkout long-horizon evaluation',
        sequence_count=int(constants['NUM_SEQUENCES']),subtasks_per_sequence=5,
        max_native_steps_per_subtask=int(constants['EP_LEN']),native_hz=30,
        sequence_generator='get_sequences(1000); deterministic native temp_seed(0)',
        initial_state='native get_env_state_for_initial_condition, robot_obs15 + scene_obs24',
        environment_reset='once per sequence; carry actual state through successive subtasks',
        model_reset='at the beginning of each subtask',
        success='Tasks.get_task_info_for_set(subtask_start_info,current_info,{subtask}) each step',
        metrics=['success_at_least_1','success_at_least_2','success_at_least_3','success_at_least_4','success_at_least_5','mean_completed_chain_length'],
        instruction='new_playtable_validation.yaml[subtask][0]',
        training_split_labels=['ABC->D','ABCD->D'],test_set_shared_between_training_splits=True,
        checkpoint_training_split_must_be_verified=True,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (OFFICIAL_EVALUATOR,OFFICIAL_SEQUENCES,OFFICIAL_INITIAL)})


def native_calvin_config_audit():
    from omegaconf import OmegaConf
    configuration=OmegaConf.load(str(DEBUG_D/'.hydra/merged_config.yaml'))
    expected_scene=OmegaConf.load(str(CALVIN_ENV/'conf/scene/calvin_scene_D_eval.yaml'))
    cameras={name:OmegaConf.load(str(CALVIN_ENV/'conf/cameras/cameras'/('%s.yaml'%name))) for name in ('static','gripper')}
    camera_equal={name:OmegaConf.to_container(configuration.cameras[name],resolve=True)==OmegaConf.to_container(value,resolve=True)
                  for name,value in cameras.items()}
    # Random initial object positions in a dataset config do not define the
    # formal evaluator's states: native robot_obs/scene_obs overwrite them.
    cfg_scene=OmegaConf.to_container(configuration.scene,resolve=True)
    expected=OmegaConf.to_container(OmegaConf.merge(configuration,{'scene':expected_scene}).scene,resolve=True)
    checks={key:cfg_scene[key]==expected[key] for key in ('global_scaling','robot_base_position','robot_base_orientation','surfaces')}
    required=[CALVIN_ENV/'data'/expected['objects']['fixed_objects']['table']['file']]
    required += [CALVIN_ENV/'data'/row['file'] for row in expected['objects']['movable_objects'].values()]
    required += [CALVIN_ENV/'data'/configuration.robot.filename]
    files=[DEBUG_D/'.hydra/merged_config.yaml',CALVIN_ENV/'conf/scene/calvin_scene_D_eval.yaml']
    files += [CALVIN_ENV/'conf/cameras/cameras'/('%s.yaml'%name) for name in cameras]
    return dict(status='supported' if all(camera_equal.values()) and all(checks.values()) and all(p.exists() for p in required) else 'requires_assets',
        cameras_equal_local_official_configs=camera_equal,scene_geometry_checks=checks,
        actual_scene_table_file=cfg_scene['objects']['fixed_objects']['table']['file'],
        required_native_assets={str(p):dict(exists=p.exists(),sha256=sha(p) if p.exists() else None) for p in required},
        config_sha256={str(p):sha(p) for p in files},
        full_training_archives_required=False,
        scope='Local official checkout scene-D geometry/cameras and formal sequence-state reset; not a claim that a downloaded full-validation archive was checked.')


def official_calvin_sequences(count=1000,workers=1):
    from calvin_agent.evaluation.multistep_sequences import get_sequences
    if not 1<=int(count)<=1000: raise ValueError('Formal sequence count must be1..1000')
    # A smaller request to get_sequences changes state allocation, so always
    # generate the full canonical1000 then take a preregistered prefix.
    rows=get_sequences(1000,num_workers=int(workers))
    return [dict(index=i,initial_condition=dict(initial),tasks=list(tasks))
            for i,(initial,tasks) in enumerate(rows[:int(count)])]


def official_calvin_initial_state(initial_condition):
    """Execute the verbatim native state builder without importing MCIL/models.

    The official utils module imports training/model classes at module scope;
    those imports are unrelated to its deterministic scene-state construction.
    """
    import pyhash
    tree=ast.parse(OFFICIAL_INITIAL.read_text())
    selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
              and node.name in ('temp_seed','get_env_state_for_initial_condition')]
    namespace=dict(np=np,pi=np.pi,contextlib=contextlib,hasher=pyhash.fnv1_32())
    module=ast.Module(body=selected,type_ignores=[])
    exec(compile(module,str(OFFICIAL_INITIAL),'exec'),namespace)
    return namespace['get_env_state_for_initial_condition'](initial_condition)


def chain_metrics(completed):
    values=np.asarray(completed,dtype=np.int64)
    if values.ndim!=1 or not len(values) or np.any(values<0) or np.any(values>5): raise ValueError('Expected completed chain lengths0..5')
    return dict(sequences=len(values),mean_completed_chain_length=float(values.mean()),
                success_at_least={str(i):float(np.mean(values>=i)) for i in range(1,6)})


def calvin_native_action(action,kind='relative'):
    action=np.asarray(action,dtype=np.float64)
    if action.shape!=(7,) or not np.isfinite(action).all(): raise ValueError('CALVIN native action must be finite7D')
    sign=1 if action[-1]>0 else -1
    if kind=='relative':
        result=np.r_[np.clip(action[:6],-1.,1.),sign]
        return result,dict(kind='native relative worldXYZ + relative Euler; target accumulated by original controller',
            clipping=bool(np.any(result[:6]!=action[:6])),native_action=result.tolist(),
            native_position_scale_m=.02,native_euler_scale_rad=.05,gripper='-1 closes,+1 opens')
    if kind=='absolute':
        result=[action[:3].tolist(),action[3:6].tolist(),sign]
        return result,dict(kind='native absolute worldXYZ+Euler three-item tuple',native_action=result,
            gripper='-1 closes,+1 opens',no_libero_normalization=True)
    raise ValueError('Unknown native CALVIN action kind')


def calvin_policy_state(robot_obs,gripper_semantics):
    """Checkpoint training semantics must be explicit, never inferred from shape."""
    state=np.asarray(robot_obs,dtype=np.float64)
    if state.shape!=(15,) or not np.isfinite(state).all(): raise ValueError('Actual native robot_obs15 required')
    if gripper_semantics=='action_sign':
        if state[-1] not in (-1,1): raise ValueError('Stored actual native gripper sign must be±1')
        gripper=state[-1]
    elif gripper_semantics=='opening_width_m': gripper=state[6]
    else: raise ValueError('Explicit checkpoint gripper semantics required')
    return np.r_[state[:6],gripper].astype(np.float32)


class NativeCalvinEnv:
    def __init__(self):
        import hydra
        from omegaconf import OmegaConf
        cfg=OmegaConf.load(str(DEBUG_D/'.hydra/merged_config.yaml'))
        cfg.scene=OmegaConf.load(str(CALVIN_ENV/'conf/scene/calvin_scene_D_eval.yaml'))
        for name in list(cfg.cameras):
            if name not in ('static','gripper'): del cfg.cameras[name]
        for name in ('static','gripper'):
            cfg.cameras[name]=OmegaConf.load(str(CALVIN_ENV/'conf/cameras/cameras'/('%s.yaml'%name)))
        self.env=hydra.utils.instantiate(cfg.env,show_gui=False,use_vr=False,use_scene_info=True,use_egl=False)
        task_path=CALVIN_MODELS/'conf/callbacks/rollout/tasks/new_playtable_tasks.yaml'
        self.oracle=hydra.utils.instantiate(OmegaConf.load(str(task_path)))
        self.annotations=OmegaConf.to_container(OmegaConf.load(str(CALVIN_MODELS/'conf/annotations/new_playtable_validation.yaml')))
        self.info=dict(native_hz=float(self.env.control_freq),renderer='native PyBullet TinyRenderer CPU',
            state_contract='unaltered native robot_obs15 and scene_obs24',
            cameras='unaltered rgb_static200x200 and rgb_gripper84x84; no flip',
            tcp_link_id=int(self.env.robot.tcp_link_id),tcp_definition='native getLinkState(tcp_link_id)COM pose, virtual center between finger tips',
            relative_position_scale_m=float(self.env.robot.max_rel_pos),relative_euler_scale_rad=float(self.env.robot.max_rel_orn),
            use_target_pose=bool(self.env.robot.use_target_pose),task_oracle_sha256=sha(task_path))
        if self.info['native_hz']!=30 or self.info['relative_position_scale_m']!=.02 or self.info['relative_euler_scale_rad']!=.05:
            raise RuntimeError('Unexpected native CALVIN controller contract')

    def reset_state(self,robot_obs,scene_obs):
        self.obs=self.env.reset(robot_obs=np.asarray(robot_obs),scene_obs=np.asarray(scene_obs))
        return self.obs

    def reset_sequence(self,initial_condition):
        robot_obs,scene_obs=official_calvin_initial_state(initial_condition)
        return self.reset_state(robot_obs,scene_obs)

    def start_subtask(self,task):
        if task not in self.annotations: raise ValueError('Unknown official CALVIN subtask')
        self.task=task;self.start_info=self.env.get_info()
        return self.annotations[task][0]

    def step_native(self,action,kind='relative'):
        native,audit=calvin_native_action(action,kind)
        self.obs,_,_,info=self.env.step(native)
        success=self.task in self.oracle.get_task_info_for_set(self.start_info,info,{self.task}) if hasattr(self,'task') else None
        return self.obs,success,audit

    def actual(self):
        state=np.asarray(self.obs['robot_obs'])
        return dict(eef_xyz_world=state[:3].tolist(),eef_euler_world=state[3:6].tolist(),gripper_width_m=float(state[6]))

    def policy_raw(self,gripper_semantics):
        return dict(base_rgb=np.asarray(self.obs['rgb_obs']['rgb_static'],dtype=np.uint8),
            wrist_rgb=np.asarray(self.obs['rgb_obs']['rgb_gripper'],dtype=np.uint8),
            state7=calvin_policy_state(self.obs['robot_obs'],gripper_semantics))

    def close(self): self.env.close();self.env.cid=-1


def capture_calvin_native_input(sequence_file,output,index=0):
    sequence=json.loads(Path(sequence_file).read_text())['sequences'][int(index)]
    env=NativeCalvinEnv();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    try:
        obs=env.reset_sequence(sequence['initial_condition'])
        task=sequence['tasks'][0];instruction=env.start_subtask(task)
        path=output/'native_input.npz'
        np.savez_compressed(str(path),base_rgb=obs['rgb_obs']['rgb_static'],wrist_rgb=obs['rgb_obs']['rgb_gripper'],
            state7=np.asarray(obs['robot_obs'][:7],dtype=np.float32),robot_obs=obs['robot_obs'],scene_obs=obs['scene_obs'],
            instruction=np.asarray(instruction))
        receipt=dict(status='complete',sequence=sequence,first_subtask=task,instruction=instruction,
            input_path=str(path),input_sha256=sha(path),native_info=env.info,gpu_used=False,
            state_builder='Verbatim official temp_seed/get_env_state_for_initial_condition AST, native pyhash FNV1_32; no training/model import',
            initial_builder_sha256=sha(OFFICIAL_INITIAL))
        write_json(output/'receipt.json',receipt)
        return receipt
    finally: env.close()


def export_calvin_sign_capture(source):
    """Preserve earlier width capture and add explicit training-sign variant."""
    source=Path(source)
    with np.load(str(source),allow_pickle=False) as saved:
        data={name:saved[name].copy() for name in saved.files}
    previous=np.asarray(data['state7']).copy()
    data['state7']=calvin_policy_state(data['robot_obs'],'action_sign')
    output=source.with_name('native_input_gripper_sign.npz')
    if output.exists(): raise FileExistsError(output)
    np.savez_compressed(str(output),**data)
    receipt=dict(status='complete',source_width_capture=str(source),source_sha256=sha(source),
        sign_capture=str(output),sign_capture_sha256=sha(output),actual_robot_obs=data['robot_obs'].tolist(),
        previous_width_state7=previous.tolist(),training_sign_state7=data['state7'].tolist(),
        state_source='actual captured robot_obs[:6] + actual stored robot_obs[-1]±1; no predicted action used',
        preserved_inputs='same captured RGB, scene_obs, complete robot_obs and instruction; only explicitly declared state7gripper field differs')
    write_json(source.with_name('gripper_sign_capture_receipt.json'),receipt)
    return receipt


def replay_calvin_demonstrations(output):
    """Positive native-controller/oracle gate using real recorded absolute actions."""
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    annotations=np.load(str(DEBUG_D/'lang_annotations/auto_lang_ann.npy'),allow_pickle=True).item()
    env=NativeCalvinEnv();results=[]
    try:
        for index,(task,interval) in enumerate(zip(annotations['language']['task'],annotations['info']['indx'])):
            start,end=map(int,interval)
            with np.load(str(DEBUG_D/('episode_%07d.npz'%start)),allow_pickle=False) as initial:
                env.reset_state(initial['robot_obs'],initial['scene_obs'])
            env.start_subtask(str(task));rows=[];success=False
            for frame in range(start,end+1):
                path=DEBUG_D/('episode_%07d.npz'%frame)
                if not path.exists(): raise FileNotFoundError('Recorded demo frame missing: '+str(path))
                with np.load(str(path),allow_pickle=False) as data:
                    action=np.asarray(data['actions']);recorded=np.asarray(data['robot_obs'])
                _,achieved,audit=env.step_native(action,'absolute')
                success=success or bool(achieved)
                rows.append(dict(frame=frame,actual=env.actual(),native_action=audit['native_action'],
                    recorded_eef_xyz=recorded[:3].tolist(),task_oracle_success=bool(achieved)))
            (output/('demo_%02d.jsonl'%index)).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            results.append(dict(index=index,task=str(task),start_frame=start,end_frame=end,
                native_steps=len(rows),success=success,first_action_file_sha256=sha(DEBUG_D/('episode_%07d.npz'%start))))
        receipt=dict(status='complete',gate='real recorded native CALVIN absolute-action replay',
            model_used=False,gpu_used=False,native_info=env.info,demonstrations=results,
            successes=sum(r['success'] for r in results),total=len(results),
            passed=all(r['success'] for r in results),
            caveat='Debug D real-demo native action/oracle positive gate; not model evaluation and not1000-sequence long-horizon success.')
        write_json(output/'receipt.json',receipt)
        return receipt
    finally: env.close()


def calibrate_calvin_native(output,hold_steps=15):
    """Measure native axis/rotation/gripper response, separate from any pi0 SO3 encoding."""
    annotations=np.load(str(DEBUG_D/'lang_annotations/auto_lang_ann.npy'),allow_pickle=True).item()
    frame=int(annotations['info']['indx'][0][0])
    with np.load(str(DEBUG_D/('episode_%07d.npz'%frame)),allow_pickle=False) as data:
        robot=np.asarray(data['robot_obs']).copy();scene=np.asarray(data['scene_obs']).copy()
    env=NativeCalvinEnv();probes=[]
    try:
        for kind in ('translation','rotation'):
            for axis in range(3):
                for sign in (-1,1):
                    env.reset_state(robot,scene)
                    initial=np.asarray(env.obs['robot_obs']).copy()
                    initial_rotation=Rotation.from_euler('xyz',initial[3:6])
                    target=initial[:6].copy()
                    if kind=='translation': target[axis]+=.01*sign
                    else:
                        vector=np.zeros(3);vector[axis]=.08*sign
                        target[3:6]=(Rotation.from_rotvec(vector)*initial_rotation).as_euler('xyz')
                    for _ in range(int(hold_steps)): env.step_native(np.r_[target,1.],'absolute')
                    actual=np.asarray(env.obs['robot_obs'])
                    delta=actual[:3]-initial[:3] if kind=='translation' else (
                        Rotation.from_euler('xyz',actual[3:6])*initial_rotation.inv()).as_rotvec()
                    threshold=.001 if kind=='translation' else .008
                    probes.append(dict(kind=kind,axis=axis,sign=sign,requested=.01*sign if kind=='translation' else .08*sign,
                        actual_world_delta=delta.tolist(),direction_passed=bool(delta[axis]*sign>threshold),
                        scope='Physical SO3 delta only, not a chosen model-input logarithm branch'))
        widths=[]
        env.reset_state(robot,scene)
        target=np.asarray(env.obs['robot_obs'])[:6].copy()
        for sign in (1,-1,1):
            for _ in range(25): env.step_native(np.r_[target,sign],'absolute')
            widths.append(float(env.obs['robot_obs'][6]))
        gripper=dict(command_sequence=[1,-1,1],actual_widths_m=widths,
            passed=bool(widths[0]-widths[1]>.005 and widths[2]-widths[1]>.005))
        link=env.env.p.getLinkState(env.env.robot.robot_uid,env.env.robot.tcp_link_id,physicsClientId=env.env.cid)
        tcp=dict(native_tcp_link_id=int(env.env.robot.tcp_link_id),native_observation_uses='getLinkState indices0/1 (COM)',
            current_com_xyz=list(link[0]),current_link_frame_xyz=list(link[4]),
            com_minus_link_xyz=(np.asarray(link[0])-np.asarray(link[4])).tolist(),
            note='TCP pose is nativevirtual fingertip frame; no Google/WidowX/Panda cross-TCP identity calibration inferred.')
        receipt=dict(status='complete',gate='native CALVIN measured physical axis/SO3/gripper interface',
            model_used=False,gpu_used=False,hold_steps=int(hold_steps),native_info=env.info,
            probes=probes,gripper=gripper,tcp=tcp,
            passed=all(p['direction_passed'] for p in probes) and gripper['passed'])
        write_json(Path(output)/'receipt.json',receipt)
        return receipt
    finally: env.close()


def diagnose_calvin_demo(output,index=6):
    """Compare native oracle on replay and reconstructed recorded states, preserving failures."""
    annotations=np.load(str(DEBUG_D/'lang_annotations/auto_lang_ann.npy'),allow_pickle=True).item()
    start,end=map(int,annotations['info']['indx'][int(index)]);task=str(annotations['language']['task'][int(index)])
    with np.load(str(DEBUG_D/('episode_%07d.npz'%start)),allow_pickle=False) as data:
        robot=np.asarray(data['robot_obs']).copy();scene=np.asarray(data['scene_obs']).copy()
    env=NativeCalvinEnv()
    def info_summary(info):
        uid=info['robot_info']['uid']
        obj=info['scene_info']['movable_objects']['block_pink']
        return dict(pink_xyz=list(obj['current_pos']),
            nonrobot_surface_contacts=[list(pair) for pair in sorted(set((int(c[2]),int(c[4])) for c in obj['contacts'] if c[2]!=uid))])
    try:
        env.reset_state(robot,scene);env.start_subtask(task)
        actual_start=info_summary(env.start_info);replay=[]
        for frame in range(start,end+1):
            with np.load(str(DEBUG_D/('episode_%07d.npz'%frame)),allow_pickle=False) as data:
                action=np.asarray(data['actions'])
            _,success,_=env.step_native(action,'absolute')
            replay.append(dict(frame=frame,oracle_success=bool(success),state=info_summary(env.env.get_info())))
        env.reset_state(robot,scene);env.start_subtask(task)
        recorded_start=info_summary(env.start_info);recorded=[]
        for frame in range(start,end+1):
            with np.load(str(DEBUG_D/('episode_%07d.npz'%frame)),allow_pickle=False) as data:
                env.reset_state(data['robot_obs'],data['scene_obs'])
            info=env.env.get_info()
            success=task in env.oracle.get_task_info_for_set(env.start_info,info,{task})
            recorded.append(dict(frame=frame,oracle_success=bool(success),state=info_summary(info)))
        result=dict(status='complete',index=int(index),task=task,
            replay_start=actual_start,recorded_start=recorded_start,
            replay_any_success=any(r['oracle_success'] for r in replay),
            reconstructed_recorded_any_success=any(r['oracle_success'] for r in recorded),
            replay=replay,reconstructed_recorded=recorded,
            scope='Recorded states reconstructed with native reset and its one physics step; contact reconstruction is not an exact replay of original hidden dynamics.',
            policy_used=False,gpu_used=False)
        write_json(Path(output)/'receipt.json',result)
        return result
    finally: env.close()


SIMPLER_NATIVE_TASKS={
    'google_robot_pick_coke_can':dict(robot='google_robot_static',native_hz=3,max_steps=80,gripper='+1 close relative delta'),
    'google_robot_move_near':dict(robot='google_robot_static',native_hz=3,max_steps=80,gripper='+1 close relative delta'),
    'google_robot_open_top_drawer':dict(robot='google_robot_static',native_hz=3,max_steps=113,gripper='+1 close relative delta'),
    'google_robot_close_top_drawer':dict(robot='google_robot_static',native_hz=3,max_steps=113,gripper='+1 close relative delta'),
    'widowx_spoon_on_towel':dict(robot='widowx',native_hz=5,max_steps=60,gripper='-1 close,+1 open absolute'),
    'widowx_carrot_on_plate':dict(robot='widowx',native_hz=5,max_steps=60,gripper='-1 close,+1 open absolute'),
    'widowx_stack_cube':dict(robot='widowx',native_hz=5,max_steps=60,gripper='-1 close,+1 open absolute'),
    'widowx_put_eggplant_in_basket':dict(robot='widowx',native_hz=5,max_steps=120,gripper='-1 close,+1 open absolute'),
}


def simpler_native_contract(robot):
    if robot=='google_robot_static':
        return dict(control_mode='arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner',
            arm_translation='meters in robot-base axes at actualEEF origin',arm_rotation='rotation vector radians, left-composed in base axes',
            normalize_arm=False,use_target_arm=False,frame='ee_align',gripper='normalized joint target delta; positive closes',native_hz=3)
    if robot=='widowx':
        return dict(control_mode='arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos',
            arm_translation='meters, native ee_align2 delta accumulated on previous target',
            arm_rotation='rotation vector radians around actualEEF, native ee_align2 composition',
            normalize_arm=False,use_target_arm=True,frame='ee_align2',gripper='normalized absolute joint position; -1 closes,+1 opens',native_hz=5)
    raise ValueError('Unsupported native robot')


def simpler_native_asset_audit():
    """Import registration/config only; no environment, renderer, CUDA or scene."""
    import gymnasium as gym
    import simpler_env
    from mani_skill2_real2sim import ASSET_DIR,format_path
    from mani_skill2_real2sim.utils.registration import REGISTERED_ENVS
    from mani_skill2_real2sim.agents.configs.google_robot.defaults import GoogleRobotDefaultConfig
    from mani_skill2_real2sim.agents.configs.widowx.defaults import WidowXDefaultConfig
    robots={'google_robot_static':GoogleRobotDefaultConfig(),'widowx':WidowXDefaultConfig()}
    rows={}
    for task,expected in SIMPLER_NATIVE_TASKS.items():
        identifier,_=simpler_env.ENVIRONMENT_MAP[task]
        spec=gym.spec(identifier);cls=REGISTERED_ENVS[identifier].cls
        if isinstance(cls,str):
            module,name=cls.split(':');cls=getattr(importlib.import_module(module),name)
        # Some native config methods use zero-argument super(). A metadata-only
        # subclass supplies correct MRO without invoking any environment init;
        # its destructor must not close an unconstructed engine.
        audit_class=type('ConfigAudit_'+cls.__name__,(cls,),{'__del__':lambda self:None})
        holder=object.__new__(audit_class)
        config=cls._setup_prepackaged_env_init_config(holder)
        paths=[Path(format_path(robots[expected['robot']].urdf_path)),Path(config['rgb_overlay_path'])]
        scene=config['scene_name']
        if 'dummy' not in scene: paths.append(Path(ASSET_DIR)/'hab2_bench_assets/stages'/(scene+'.glb'))
        elif scene=='dummy_drawer': paths.append(Path(ASSET_DIR)/'custom'/(holder.station_name+'.urdf'))
        model_json=getattr(cls,'DEFAULT_MODEL_JSON',None)
        asset_root=getattr(cls,'DEFAULT_ASSET_ROOT',None)
        if model_json and asset_root: paths.append(Path(format_path(asset_root))/model_json)
        rows[task]=dict(environment_id=identifier,native_spec_max_steps=spec.max_episode_steps,
            source_class=cls.__module__+'.'+cls.__name__,source_class_sha256=sha(inspect.getfile(cls)),
            prepackaged_config={k:v for k,v in config.items() if isinstance(v,(str,int,float,bool,list,dict))},
            native_controller_contract=simpler_native_contract(expected['robot']),
            expected_native_hz=expected['native_hz'],frequency_matches=config['control_freq']==expected['native_hz'],
            declared_asset_files={str(p):dict(exists=p.exists(),sha256=sha(p) if p.exists() else None) for p in paths},
            declared_assets_present=all(p.exists() for p in paths),
            object_meshes_runtime_checked=False,scope='CPU source/config and declared robot/scene/overlay/database assets; actual object mesh construction and measured control gate pending GPU grant')
    return dict(status='cpu_source_audit_complete',gpu_used=False,renderer_constructed=False,tasks=rows,
        passed=all(row['frequency_matches'] and row['declared_assets_present'] for row in rows.values()))


class NativeSimplerEnv:
    """Actual native RGB/proprioception and native7D controller, never virtual Panda state."""
    def __init__(self,task,gpu,gpu_uuid,require_idle=True):
        from preparations.idea1_v3_2.transfer_envs import gpu_renderer_binding
        binding=gpu_renderer_binding(gpu,gpu_uuid,require_idle=require_idle)
        import simpler_env
        self.env=simpler_env.make(task,renderer_kwargs=dict(device=binding['sapien_device'],offscreen_only=True))
        self.native=self.env.unwrapped;self.task=task
        expected=SIMPLER_NATIVE_TASKS[task];contract=simpler_native_contract(expected['robot'])
        arm=self.native.agent.controller.controllers['arm'];gripper=self.native.agent.controller.controllers['gripper']
        if self.native.control_mode!=contract['control_mode'] or arm.config.normalize_action or arm.config.frame!=contract['frame']:
            raise RuntimeError('Actual native Simpler controller differs from audited contract')
        self.camera='overhead_camera' if expected['robot']=='google_robot_static' else '3rd_view_camera'
        self.info=dict(task=task,binding=binding,controller_contract=contract,
            actual_control_hz=float(self.native.control_freq),actual_arm_config=repr(arm.config),actual_gripper_config=repr(gripper.config),
            tcp_link_name=self.native.agent.config.ee_link_name,
            observation_contract='unaltered native RGB and native agent/extra fields; no virtualPanda8D state',
            success_oracle='native info success; native termination/truncation recorded separately')

    def reset(self,seed=0,options=None):
        self.obs,self.reset_info=self.env.reset(seed=int(seed),options=options)
        return self.obs

    def actual(self):
        tcp=self.native.tcp.pose
        links=self.native.agent.robot.get_links()
        names={link.name:link for link in links}
        tips=[link for name,link in names.items() if 'finger' in name and 'tip' in name]
        return dict(eef_xyz_world=np.asarray(tcp.p).tolist(),eef_quaternion_wxyz=np.asarray(tcp.q).tolist(),
            robot_base_xyz_world=np.asarray(self.native.agent.robot.pose.p).tolist(),
            robot_base_quaternion_wxyz=np.asarray(self.native.agent.robot.pose.q).tolist(),
            native_qpos=np.asarray(self.native.agent.robot.get_qpos()).tolist(),
            actual_fingertip_world_positions={link.name:np.asarray(link.pose.p).tolist() for link in tips})

    def step_native(self,action):
        action=np.asarray(action,dtype=np.float64)
        if action.shape!=(7,) or not np.isfinite(action).all(): raise ValueError('Native Simpler controller requires finite7D physical action')
        command=np.clip(action,self.env.action_space.low,self.env.action_space.high)
        self.obs,reward,terminated,truncated,info=self.env.step(command)
        if 'success' not in info: raise RuntimeError('Native task success oracle missing')
        return self.obs,dict(success=bool(info['success']),terminated=bool(terminated),truncated=bool(truncated),
            native_reward=float(reward),native_action=command.tolist(),space_clipped=bool(np.any(action!=command)))

    def rgb(self): return np.asarray(self.obs['image'][self.camera]['rgb'],dtype=np.uint8)
    def close(self): self.env.close()


def calibrate_simpler_native(task,gpu,gpu_uuid,output,hold_steps=5):
    """GPU-gated measured native delta axes, physical rotations and gripper signs."""
    env=NativeSimplerEnv(task,gpu,gpu_uuid,require_idle=True)
    robot=SIMPLER_NATIVE_TASKS[task]['robot'];probes=[]
    try:
        for kind in ('translation','rotation'):
            for axis in range(3):
                for sign in (-1,1):
                    env.reset(seed=0)
                    before=env.actual();tcp_rotation=Rotation.from_quat(np.asarray(before['eef_quaternion_wxyz'])[[1,2,3,0]])
                    base_rotation=Rotation.from_quat(np.asarray(before['robot_base_quaternion_wxyz'])[[1,2,3,0]])
                    pulse=np.zeros(7);pulse[(0 if kind=='translation' else 3)+axis]=sign*(.01 if kind=='translation' else .08)
                    pulse[-1]=1. if robot=='widowx' else 0.
                    env.step_native(pulse)
                    hold=np.zeros(7);hold[-1]=pulse[-1]
                    for _ in range(int(hold_steps)): env.step_native(hold)
                    after=env.actual()
                    if kind=='translation':
                        measured=base_rotation.inv().apply(np.asarray(after['eef_xyz_world'])-before['eef_xyz_world'])
                    else:
                        observed=Rotation.from_quat(np.asarray(after['eef_quaternion_wxyz'])[[1,2,3,0]])
                        measured=base_rotation.inv().apply((observed*tcp_rotation.inv()).as_rotvec())
                    threshold=.001 if kind=='translation' else .008
                    probes.append(dict(kind=kind,axis=axis,sign=sign,native_pulse=pulse.tolist(),
                        actual_base_axes_delta=measured.tolist(),direction_passed=bool(measured[axis]*sign>threshold)))
        env.reset(seed=0);grips=[]
        sequence=(-1,1,-1) if robot=='google_robot_static' else (1,-1,1)
        for command in sequence:
            action=np.zeros(7);action[-1]=command
            for _ in range(5): env.step_native(action)
            grips.append(float(env.native.agent.get_gripper_closedness()))
        gripper=dict(native_command_sequence=list(sequence),measured_native_closedness=grips,
            passed=bool(grips[1]-grips[0]>.2 and grips[1]-grips[2]>.2))
        result=dict(status='complete',gate='native Simpler measured physical controller/TCP/axis/gripper interface',
            task=task,native_info=env.info,probes=probes,gripper=gripper,model_used=False,
            passed=all(p['direction_passed'] for p in probes) and gripper['passed'],
            scope='Native controller calibration only, physical SO3 changes; no LIBERO-to-target TCP identity or model rotvec-branch calibration inferred.')
        write_json(Path(output)/'receipt.json',result)
        return result
    finally: env.close()


def pusht_native_action(action):
    action=np.asarray(action,dtype=np.float64)
    if action.shape!=(2,) or not np.isfinite(action).all(): raise ValueError('Native PushT action is a finite2D absolute pixel target')
    return np.clip(action,0.,512.)


def replay_pusht_expert(dataset,output,episodes=3):
    """Replay real expert states/actions as native2D; no seven-dimensional projection."""
    import zarr
    from diffusion_policy.env.pusht.pusht_env import PushTEnv
    dataset=Path(dataset)
    store=zarr.ZipStore(str(dataset),mode='r') if dataset.is_file() else zarr.DirectoryStore(str(dataset))
    group=zarr.open_group(store=store,mode='r')
    states=group['data']['state'];actions=group['data']['action'];ends=np.asarray(group['meta']['episode_ends'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    results=[]
    try:
        begin=0
        for index,end in enumerate(ends[:int(episodes)]):
            end=int(end)
            env=PushTEnv(render_size=96,render_action=False,reset_to_state=np.asarray(states[begin]))
            try:
                obs=env.reset();rows=[];success=False;max_reward=0.
                for offset in range(begin,end):
                    command=pusht_native_action(actions[offset])
                    obs,reward,done,info=env.step(command)
                    success=success or bool(done);max_reward=max(max_reward,float(reward))
                    rows.append(dict(dataset_index=offset,native_action_xy=command.tolist(),actual_agent_xy=obs[:2].tolist(),
                        actual_block_xy_angle=obs[2:].tolist(),reward=float(reward),native_success=bool(done)))
                (output/('episode_%03d.jsonl'%index)).write_text(''.join(json.dumps(r)+'\n' for r in rows))
                results.append(dict(episode=index,steps=end-begin,success=success,max_reward=max_reward,success_threshold=float(env.success_threshold)))
            finally: env.close()
            begin=end
        receipt=dict(status='complete',gate='native2D expert replay using recorded initial physics state and actions',
            model_used=False,gpu_used=False,dataset=str(dataset),dataset_sha256=sha(dataset) if dataset.is_file() else None,
            episodes=results,passed=bool(results and all(r['success'] for r in results)),
            caveat='Positive environment/action/oracle gate only; never a claim that a7D pi0 policy natively supports PushT.')
        write_json(output/'receipt.json',receipt)
        return receipt
    finally:
        if hasattr(store,'close'): store.close()

"""Twelve metadata-selected physical input interventions; no closed-loop or video claims."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from .draft_diagnostic_audit import ROOT,DEFAULT_OUT,read,dump
from .draft_diagnostic_protocol import OBS_KEYS,hashes
from .draft_diagnostic_language import action_difference
from .draft_diagnostic_state import state_difference
from .libero_plus_env import NativeLiberoPlusEnv,ORIGINAL,PLUS,isolated_child_environment
from preparations.idea1_v3_2.transfer_client import rpc

MODEL_FIELDS=('body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat',
              'geom_pos','geom_quat','geom_size','geom_type','geom_bodyid','geom_friction','geom_contype','geom_conaffinity',
              'geom_solref','geom_solimp','geom_margin','geom_gap',
              'jnt_pos','jnt_axis','jnt_range','jnt_type','dof_damping','dof_frictionloss',
              'cam_pos','cam_quat','cam_fovy','actuator_gainprm','actuator_biasprm')


def physical(env):
    sim=env.env.sim
    return dict(qpos=np.asarray(sim.data.qpos).copy(),qvel=np.asarray(sim.data.qvel).copy(),time=np.asarray(sim.data.time),
        act=np.asarray(sim.data.act).copy(),ctrl=np.asarray(sim.data.ctrl).copy(),
        contacts=np.asarray([[c.dist,*c.pos,*c.frame] for c in sim.data.contact[:sim.data.ncon]],dtype=np.float64),
        **{'model_'+key:np.asarray(getattr(sim.model,key)).copy() for key in MODEL_FIELDS})


def observations(env):
    env.obs=env.env.env._get_observations(force_update=True)
    return {key:np.asarray(env.obs[key]).copy() for key in OBS_KEYS}


def base_robot_qpos():
    path=PLUS/'libero/libero/envs/robots/mounted_panda.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='MountedPanda')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='init_qpos')
    ret=next(n for n in fn.body if isinstance(n,ast.Return))
    return np.asarray(eval(compile(ast.Expression(ret.value),str(path),'eval'),{'np':np}),np.float64)


def background_geom_allowed(name):
    """Only explicit static arena surfaces; object instances cannot pass by substring."""
    if not name:return False
    return name in ('floor','floor_visual','table_visual','table_collision','ground','groundplane') or name.startswith(
        ('wall_','walls_','table_leg','floor_','living_room_table_','coffee_table_','study_table_','kitchen_table_'))


def texture_ownership(model,texture_id):
    import mujoco
    mat_texids=np.asarray(model.mat_texid).reshape(model.nmat,-1)
    materials=np.flatnonzero(np.any(mat_texids==texture_id,axis=1));result=[]
    for matid in materials:
        geoms=np.flatnonzero(np.asarray(model.geom_matid)==matid)
        geometry=[]
        for g in geoms:
            body=int(model.geom_bodyid[g]);cursor=body;chain=[]
            while cursor:
                chain.append(cursor);cursor=int(model.body_parentid[cursor])
            static=all(int(model.body_jntnum[b])==0 and int(model.body_mocapid[b])<0 for b in chain)
            geometry.append(dict(geom_id=int(g),geom_name=model.geom_id2name(int(g)),body_id=body,
                body_name=model.body_id2name(body),static_body_chain=chain,static_body=static))
        result.append(dict(material_id=int(matid),material_name=mujoco.mj_id2name(model._model,mujoco.mjtObj.mjOBJ_MATERIAL,int(matid)),geoms=geometry))
    return result


def robot_intervention(env):
    robot=env.env.robots[0];sim=env.env.sim
    old_goal=dict(position=np.asarray(robot.controller.goal_pos).tolist(),orientation=np.asarray(robot.controller.goal_ori).tolist())
    pos=np.asarray(robot._ref_joint_pos_indexes,dtype=int);vel=np.asarray(robot._ref_joint_vel_indexes,dtype=int)
    target=base_robot_qpos()
    if pos.shape!=(7,) or target.shape!=(7,):raise RuntimeError('Unexpected robot joint indexing')
    old_qpos=np.asarray(sim.data.qpos).copy();old_qvel=np.asarray(sim.data.qvel).copy()
    sim.data.qpos[pos]=target;sim.data.qvel[vel]=0.;sim.forward()
    robot.controller.update(force=True);robot.controller.reset_goal()
    obs=observations(env)
    untouched_pos=np.delete(np.arange(len(old_qpos)),pos);untouched_vel=np.delete(np.arange(len(old_qvel)),vel)
    gate=bool(np.array_equal(old_qpos[untouched_pos],sim.data.qpos[untouched_pos]) and np.array_equal(old_qvel[untouched_vel],sim.data.qvel[untouched_vel]))
    return obs,dict(kind='robot_arm_qpos_to_inherited_MountedPanda_default_and_arm_qvel_zero',
        pos_indexes=pos.tolist(),vel_indexes=vel.tolist(),target=target.tolist(),
        nonrobot_qpos_qvel_bitwise_equal=gate,controller_goal_reset=True,
        controller_goal_before=old_goal,controller_goal_after=dict(position=np.asarray(robot.controller.goal_pos).tolist(),orientation=np.asarray(robot.controller.goal_ori).tolist()),
        limitation='Robot visual pose and proprioception change jointly. No physical stepping; no claim of exact mother-init trajectory state.')


def background_intervention(env,case):
    """Only copy matching named background texture pixels from mother BDDL model."""
    from libero.libero.envs.env_wrapper import ControlEnv
    import mujoco
    bddl=ORIGINAL/'libero/libero/bddl_files'/case['mother_suite']/(case['mother_task']+'.bddl')
    mother=ControlEnv(str(bddl),use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False,hard_reset=True)
    try:
        model=env.env.sim.model;source=mother.sim.model
        def names(m):
            return {mujoco.mj_id2name(m._model,mujoco.mjtObj.mjOBJ_TEXTURE,i):i for i in range(m.ntex)}
        originals=names(source);changed=[];skipped=[]
        for name,index in names(model).items():
            if name not in originals or not name or not (name.startswith(('texplane','tex-wall','tex-table','tex-living_room_table','tex-coffee','tex-study','tex-kitchen')) or name=='textable'):
                continue
            ownership=texture_ownership(model,index)
            uses=[g for material in ownership for g in material['geoms']]
            if not uses or not all(background_geom_allowed(g['geom_name']) and g['static_body'] for g in uses):
                skipped.append(dict(name=name,reason='texture is unused or shared with non-arena geom',ownership=ownership));continue
            other=originals[name];shape=(int(model.tex_height[index]),int(model.tex_width[index]),int(model.tex_nchannel[index]))
            other_shape=(int(source.tex_height[other]),int(source.tex_width[other]),int(source.tex_nchannel[other]))
            if shape!=other_shape or int(model.tex_type[index])!=int(source.tex_type[other]):
                skipped.append(dict(name=name,reason='texture shape/type mismatch',target_shape=shape,source_shape=other_shape));continue
            size=int(np.prod(shape));start=int(model.tex_adr[index]);source_start=int(source.tex_adr[other])
            target=model.tex_data.reshape(-1)[start:start+size];pixels=source.tex_data.reshape(-1)[source_start:source_start+size]
            if not np.array_equal(target,pixels):
                pixel_count=int(np.sum(target!=pixels));before_hash=hashes(dict(texture=target))['texture']
                target[:]=pixels;env.env.sim._render_context_offscreen.upload_texture(index)
                changed.append(dict(texture_name=name,texture_id=index,source_texture_id=other,shape=shape,
                    changed_channels=pixel_count,before_sha256=before_hash,after_sha256=hashes(dict(texture=target))['texture'],ownership=ownership))
        return observations(env),dict(kind='matching_named_background_texture_bytes_from_mother_BDDL_model',source_bddl=str(bddl),
            changed_textures=changed,skipped_textures=skipped,material_parameters_unchanged=True,
            intervention_nonempty=bool(changed),limitation='Texture-only removal; other material coefficients remain actual Plus values. No whole-scene reload.')
    finally:mother.close()


def run(a,row):
    out=a.directory/'physical_input_pairs'/('case%04d'%row['source_index']);out.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),Path(__file__).with_name('draft_diagnostic_language.py'),Path(__file__).with_name('draft_diagnostic_state.py')]
    env=None;receipt=dict(status='running',case=row,video=False,closed_loop=False,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    try:
        health=rpc(a.port,dict(op='health'))['protocol']
        if health['gpu_uuid']!=a.gpu_uuid or not health['recorded_gpu3_gate']['passed']:raise RuntimeError('Strict gated same-device service required')
        env=NativeLiberoPlusEnv(row['case'],render=True,gpu=a.gpu,gpu_uuid=a.gpu_uuid);env.reset()
        actual={key:np.asarray(env.obs[key]).copy() for key in OBS_KEYS};before=physical(env)
        if row['category']=='Sensor Noise':
            counter=observations(env);details=dict(kind='bypass_official_wrapper_image_corruption_without_sim_step',
                expected_only_agentview_changed=True,wrist_bitwise_equal=bool(np.array_equal(actual['robot0_eye_in_hand_image'],counter['robot0_eye_in_hand_image'])))
        elif row['category']=='Robot Initial States':counter,details=robot_intervention(env)
        elif row['category']=='Background Textures':counter,details=background_intervention(env,row['case'])
        else:raise ValueError('Language uses saved-observation client')
        after=physical(env);equal={key:np.array_equal(value,after[key]) for key,value in before.items()}
        if row['category']=='Robot Initial States':
            valid=details['nonrobot_qpos_qvel_bitwise_equal'] and all(v for k,v in equal.items() if k not in ('qpos','qvel','contacts'))
        else:valid=all(equal.values())
        if row['category']=='Sensor Noise':valid=valid and details['wrist_bitwise_equal']
        if row['category']=='Background Textures':valid=valid and details['intervention_nonempty']
        np.savez_compressed(out/'actual_observation.npz',**actual);np.savez_compressed(out/'counterfactual_observation.npz',**counter)
        np.savez_compressed(out/'physical_before.npz',**before);np.savez_compressed(out/'physical_after.npz',**after)
        receipt.update(physical_equal=equal,intervention=details,valid_single_factor_input_pair=bool(valid),
                       state_geometry=state_difference(actual,counter),
                       actual_observation_hashes=hashes(actual),counterfactual_observation_hashes=hashes(counter),gpu_uuid=a.gpu_uuid,service_pid=health['server_pid'])
        if not valid:
            receipt.update(status='intervention_gate_failed',reason='Physical identity, ownership or nonempty intervention gate failed; no model comparison')
            return
        responses={}
        with (out/'responses.jsonl').open('x') as stream:
            for arm in ('actual','counterfactual','counterfactual','actual'):
                response=rpc(a.port,dict(op='predict',mode=row['head'],mother_suite=row['case']['mother_suite'],
                    observation=actual if arm=='actual' else counter,instruction=row['actual_instruction'],seed=7,category='preflight'))
                if response['manifest_hash']!=health['manifest_hash']:raise RuntimeError('Service identity changed')
                if arm in responses:
                    for key in ('actions','raw_actions'):
                        if not np.array_equal(np.asarray(response[key],np.float32),np.asarray(responses[arm][key],np.float32)):
                            raise RuntimeError('Same-input same-device repeat differs')
                responses[arm]=response;stream.write(json.dumps(dict(arm=arm,response=response))+'\n')
        receipt.update(status='complete',actions=action_difference(responses['actual']['actions'],responses['counterfactual']['actions']),
            limitation='Input sensitivity at one true state; no closed-loop success-cause claim.')
    except BaseException:
        receipt.update(status='technical_failure',error=traceback.format_exc());raise
    finally:
        if env is not None:env.close()
        dump(out/'receipt.json',receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=DEFAULT_OUT)
    p.add_argument('--port',type=int,default=19148);p.add_argument('--gpu',type=int,default=0);p.add_argument('--gpu-uuid')
    p.add_argument('--deadline-unix',type=float,default=0);p.add_argument('--child-index',type=int);p.add_argument('--launch',action='store_true');a=p.parse_args()
    cases=[r for r in read(a.directory/'paired_plan.json')['cases'] if r['category']!='Language Instructions']
    if not a.launch:print(json.dumps(dict(dry_run=True,cases=len(cases),calls=4*len(cases),video=False)));return
    if not a.gpu_uuid or time.time()>=a.deadline_unix:raise ValueError('Device/deadline required')
    if a.child_index is not None:run(a,cases[a.child_index]);return
    for index,row in enumerate(cases):
        if time.time()>=a.deadline_unix:raise TimeoutError('Input diagnostic deadline')
        path=a.directory/'physical_input_pairs'/('case%04d'%row['source_index'])/'receipt.json'
        if path.exists():
            if read(path).get('status') in ('complete','intervention_gate_failed'):continue
            raise RuntimeError('Existing incomplete pair requires inspection: '+str(path))
        env=isolated_child_environment(row['case'],a.directory/'physical_input_pairs/config_plus',gpu=a.gpu,gpu_uuid=a.gpu_uuid)
        command=[str(PLUS/'.venv/bin/python'),'-m',__package__+'.draft_diagnostic_inputs',
            '--directory',str(a.directory),'--port',str(a.port),'--gpu',str(a.gpu),'--gpu-uuid',a.gpu_uuid,'--deadline-unix',str(a.deadline_unix),'--child-index',str(index),'--launch']
        logpath=a.directory/('physical_input_case%04d.log'%row['source_index'])
        with logpath.open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            code=child.wait()
            if code:raise subprocess.CalledProcessError(code,command)


if __name__=='__main__':main()

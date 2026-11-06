"""Fresh one-direction rotation and gripper holds, no model or quality RPC."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from scipy.spatial.transform import Rotation
from .evaluate_transfer import actual_record,make_adapter,verify_registered_initialization
from .transfer_actual_gate import save_raw,write
from .transfer_registration import load_registration


def rot(actual):
    if 'eef_euler_world' in actual:return Rotation.from_euler('xyz',actual['eef_euler_world'])
    return Rotation.from_quat(np.asarray(actual['eef_quaternion_wxyz'])[[1,2,3,0]])


def controller(env,environment):
    if environment=='calvin':
        robot=env.native.env.robot
        return dict(gripper_action=int(robot.gripper_action),target_position=np.asarray(robot.target_pos).tolist(),
                    target_orientation=np.asarray(robot.target_orn).tolist(),
                    gripper_joint_states=[env.native.env.p.getJointState(robot.robot_uid,j,physicsClientId=env.native.env.cid)[:2] for j in robot.gripper_joint_ids])
    controls=env.native.native.agent.controller.controllers;result={}
    for name,c in controls.items():
        row={}
        for attr in ('_target_qpos','_last_drive_qpos_targets','_start_qpos'):
            v=getattr(c,attr,None)
            if v is not None:row[attr]=np.asarray(v).tolist()
        pose=getattr(c,'_target_pose',None)
        if pose is not None:row['target_pose_at_robot_base']=dict(xyz=np.asarray(pose.p).tolist(),quaternion_wxyz=np.asarray(pose.q).tolist())
        row['actual_joint_qpos']=np.asarray(c.qpos).tolist();result[name]=row
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--environment',choices=['calvin','simpler'],required=True)
    p.add_argument('--registration',required=True);p.add_argument('--authorization',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    registration=load_registration(a.registration,verify_sources=True);auth=json.loads(Path(a.authorization).read_text())
    if auth['gpu']!=3 or auth['gpu_uuid']!='GPU-555cde36-5e41-bc44-0c69-ea47358fd703' or auth['registration_hash']!=registration['registration_hash']:raise PermissionError('Exact GPU3/Bv2 identity required')
    reference=json.loads(Path(registration['input_files']['source_reference']['path']).read_text())
    case=next(c for c in registration['cases'] if c['environment']==a.environment)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    receipt=dict(status='running',registration_hash=registration['registration_hash'],environment=a.environment,quality_requests=0,model_calls=0,rows=[],source_sha256=registration['source_files'],gate_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    plans=[(axis,sign) for axis in range(3) for sign in (1,-1)] if a.environment=='simpler' else []
    plans.append(('gripper',None));start=time.monotonic()
    try:
        for index,(axis,sign) in enumerate(plans):
            if time.monotonic()-start>270:raise TimeoutError('4.5minute gate bound')
            env=None;row=dict(axis=axis,sign=sign,cycles=[]);receipt['rows'].append(row)
            try:
                env=make_adapter(case,reference,auth)
                if a.environment=='calvin':env.reset_sequence(case['sequence']['initial_condition']);env.start_subtask(case['sequence']['tasks'][0])
                else:env.reset(case['seed'],case['reset_options'])
                verify_registered_initialization(case,env)
                initial=actual_record(env);initial_rotation=rot(initial['actual'])
                row.update(initial=initial,info=env.info,initial_raw=save_raw(out/('initial_%02d.npz'%index),env.raw()))
                if axis!='gripper':
                    target_base=np.asarray(env.info['target_robot_base_world_matrix'])[:3,:3]
                    source_base=np.asarray(reference['robot_base_world_matrix'])[:3,:3]
                    expected=target_base.dot(source_base.T).dot(np.eye(3)[axis]*sign)
                    row['expected_world_axis']=expected.tolist()
                for cycle in range(12 if axis=='gripper' else 5):
                    command=np.zeros(7);command[6]=1 if cycle<6 else -1
                    if axis!='gripper':command[:]=0;command[3+axis]=sign*.1
                    before=actual_record(env);_,_,audit=env.step(command);after=actual_record(env)
                    record=dict(cycle=cycle,command=command.tolist(),before=before,after=after,controller=controller(env,a.environment),audit=audit,
                                raw=save_raw(out/('raw_%02d_%02d.npz'%(index,cycle)),env.raw()))
                    if axis!='gripper':
                        delta=(rot(after['actual'])*initial_rotation.inv()).as_rotvec()
                        record.update(cumulative_world_rotation_rad=delta.tolist(),expected_axis_projection_rad=float(delta.dot(expected)))
                    row['cycles'].append(record);write(out/'receipt.json',receipt)
                if axis!='gripper' and row['cycles'][-1]['expected_axis_projection_rad']<=0:
                    raise RuntimeError('Measured held rotation contradicts expected world direction; stop without tuning')
            finally:
                if env is not None:env.close()
        receipt['status']='complete_measured_requires_parent_review'
    except Exception:receipt.update(status='technical_or_direction_failure_stop',error=traceback.format_exc())
    receipt['wall_s']=time.monotonic()-start;write(out/'receipt.json',receipt)
    print(json.dumps(dict(status=receipt['status'],rows=len(receipt['rows']),wall_s=receipt['wall_s'],quality_requests=0)))
    if receipt['status']!='complete_measured_requires_parent_review':raise SystemExit(1)


if __name__=='__main__':main()

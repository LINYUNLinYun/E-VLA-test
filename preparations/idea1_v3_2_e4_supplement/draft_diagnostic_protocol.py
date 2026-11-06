"""Forty original LIBERO diagnostic episodes: init x settle, then eight prompt arms. No video."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
from .draft_diagnostic_audit import ROOT,HEADS,legacy_language,read
from .libero_plus_env import NativeLiberoPlusEnv,isolated_child_environment
from preparations.idea1_v3_2.transfer_client import rpc
from preparations.idea1_v3_1.common import atomic_json

OBS_KEYS=('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')


def make_plan():
    manifest=read(ROOT/'results/idea1_v3_2_e4_supplement/libero_plus_half_20261022T1216Z/manifest.json')
    episodes=[]
    for suite,head in HEADS.items():
        candidates=sorted((j for j in manifest['jobs'] if j['case']['environment']=='libero_original' and j['case']['mother_suite']==suite),key=lambda j:j['case']['task_index'])[:2]
        assert len(candidates)==2
        for job in candidates:
            case=job['case']
            for init_index in (0,6):
                for settle in ('new','old'):
                    episodes.append(dict(case=case,head=head,init_index=init_index,settle=settle,prompt='legacy_filename'))
            episodes.append(dict(case=case,head=head,init_index=0,settle='new',prompt='effective_bddl'))
    for index,episode in enumerate(episodes):episode['episode_index']=index
    sources=[Path(__file__),Path(__file__).with_name('draft_diagnostic_service.py')]
    return dict(kind='draft_original_protocol_diagnostic',episodes=episodes,count=len(episodes),seed=7,cap=600,H=50,R=5,video=False,
        source_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        selection='First two original mother tasks by native task_index for each of four distinct heads; success ignored.',
        interpretation='32 fixed legacy-prompt init0/6 x new/old settle episodes; 8 effective-BDDL prompt additions at init0/new settle. Same GPU/model within experiment. Two tasks per head is mechanism screening, not a suite score.',
        settle=dict(new=dict(steps=5,action=[0.]*7),old=dict(steps=10,action=[0.]*6+[-1.])))


def state_snapshot(env):
    sim=env.env.sim
    return dict(time=float(sim.data.time),qpos=np.asarray(sim.data.qpos).copy(),qvel=np.asarray(sim.data.qvel).copy(),
                body_xpos=np.asarray(sim.data.body_xpos).copy(),body_xquat=np.asarray(sim.data.body_xquat).copy(),
                model_body_pos=np.asarray(sim.model.body_pos).copy(),model_geom_pos=np.asarray(sim.model.geom_pos).copy(),
                model_geom_size=np.asarray(sim.model.geom_size).copy())


def hashes(values):
    return {key:hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for key,value in values.items()}


def run_episode(a,plan,row):
    output=a.output/('episode%03d'%row['episode_index']);output.mkdir(parents=True,exist_ok=False)
    result=dict(status='running',design=row,seed=7,max_steps=600,gpu_uuid=a.gpu_uuid,video=False,steps=0,requests=0)
    atomic_json(output/'result.json',result);env=None
    try:
        health=rpc(a.port,dict(op='health'))['protocol']
        if health.get('gpu_uuid')!=a.gpu_uuid or not health.get('draft_only') or not health.get('recorded_gpu3_gate',{}).get('passed'):
            raise RuntimeError('Required same-device draft-only strict-gated service absent')
        result['service_identity']=dict(gpu_uuid=health['gpu_uuid'],server_pid=health['server_pid'],manifest_hash=health['manifest_hash'])
        env=NativeLiberoPlusEnv(row['case'],render=True,gpu=a.gpu,gpu_uuid=a.gpu_uuid)
        env.env.seed(7);env.env.reset()
        vector=np.asarray(env.selected.get_task_init_states(env.index)[row['init_index']])
        env.obs=env.env.set_init_state(vector.copy());reset_state=state_snapshot(env)
        # Physical state before settle is saved for pair identity checks.
        np.savez_compressed(output/'reset_physical.npz',init_vector=vector,**reset_state)
        settle=plan['settle'][row['settle']]
        for _ in range(settle['steps']):env.obs,_,_,_=env.env.step(settle['action'])
        physical=state_snapshot(env);np.savez_compressed(output/'initial_physical.npz',**physical)
        observation={key:np.asarray(env.obs[key]).copy() for key in OBS_KEYS}
        np.savez_compressed(output/'initial_observation.npz',**observation)
        instruction=legacy_language(row['case']['mother_task']) if row['prompt']=='legacy_filename' else env.info['instruction']
        result.update(instruction=instruction,benchmark_instruction=legacy_language(row['case']['mother_task']),effective_bddl_instruction=env.info['instruction'],
                      reset_physical_hashes=hashes(reset_state),initial_physical_hashes=hashes(physical),initial_observation_hashes=hashes(observation),
                      initial_vector_sha256=hashes(dict(vector=vector))['vector'],native_info=env.info)
        # env.info describes the historical wrapper defaults; explicit actual fields avoid ambiguity.
        result['actual_protocol']=dict(seed=7,init_index=row['init_index'],settle=settle,max_steps=600,H=50,R=5,prompt=row['prompt'])
        rpc(a.port,dict(op='reset'));success=False
        with (output/'infer.jsonl').open('x') as inference,(output/'telemetry.jsonl').open('x') as telemetry:
            while result['steps']<600 and not success:
                if time.time()>=a.deadline_unix:raise TimeoutError('Diagnostic deadline')
                observation={key:np.asarray(env.obs[key]) for key in OBS_KEYS}
                response=rpc(a.port,dict(op='predict',mode=row['head'],mother_suite=row['case']['mother_suite'],
                    observation=observation,instruction=instruction,seed=7,category='quality'))
                if response['manifest_hash']!=health['manifest_hash']:raise RuntimeError('Service identity changed')
                actions=np.asarray(response['actions'],np.float32)
                if actions.shape!=(50,7) or not np.isfinite(actions).all():raise ValueError('Invalid action chunk')
                inference.write(json.dumps(dict(request=result['requests'],step=result['steps'],input_hashes=hashes(observation),**response))+'\n');inference.flush()
                result['requests']+=1
                for action in actions[:min(5,600-result['steps'])]:
                    _,audit=env.step_native(action);result['steps']+=1;success=audit['success']
                    telemetry.write(json.dumps(dict(step=result['steps'],actual=env.actual(),executed_command=audit))+'\n')
                    if success:break
                atomic_json(output/'result.json',result)
        result.update(status='complete',success=bool(success),termination='success' if success else 'step_cap')
    except BaseException:
        result.update(status='technical_failure',success=None,error=traceback.format_exc());raise
    finally:
        if env is not None:env.close()
        atomic_json(output/'result.json',result)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=19148);p.add_argument('--gpu',type=int,default=0);p.add_argument('--gpu-uuid')
    p.add_argument('--deadline-unix',type=float,default=0);p.add_argument('--child-index',type=int)
    p.add_argument('--indices',type=int,nargs='+');p.add_argument('--launch',action='store_true');a=p.parse_args()
    plan=make_plan()
    if not a.launch:
        a.output.mkdir(parents=True,exist_ok=True)
        if any(a.output.glob('episode*/result.json')):raise RuntimeError('Cannot refresh a plan after episodes exist')
        atomic_json(a.output/'protocol_plan.json',plan)
        print(json.dumps(dict(dry_run=True,count=plan['count'],plan=str(a.output/'protocol_plan.json'))));return
    if not a.gpu_uuid or time.time()>=a.deadline_unix:raise ValueError('Explicit device and future deadline required')
    a.output.mkdir(parents=True,exist_ok=True)
    planpath=a.output/'protocol_plan.json'
    if planpath.exists():
        if read(planpath)!=plan:raise RuntimeError('Existing plan mismatch')
    else:atomic_json(planpath,plan)
    if a.child_index is not None:run_episode(a,plan,plan['episodes'][a.child_index]);return
    for row in plan['episodes']:
        index=row['episode_index']
        if a.indices is not None and index not in a.indices:continue
        resultpath=a.output/('episode%03d'%index)/'result.json'
        if resultpath.exists():
            if read(resultpath).get('status')=='complete':continue
            raise RuntimeError('Existing incomplete episode requires explicit inspection: '+str(resultpath))
        childenv=isolated_child_environment(row['case'],a.output/'config_original',gpu=a.gpu,gpu_uuid=a.gpu_uuid)
        command=[str(ROOT/'policies/libero-eval-v21/.venv/bin/python'),'-m',__package__+'.draft_diagnostic_protocol',
                 '--output',str(a.output),'--port',str(a.port),'--gpu',str(a.gpu),'--gpu-uuid',a.gpu_uuid,
                 '--deadline-unix',str(a.deadline_unix),'--child-index',str(index),'--launch']
        with (a.output/('episode%03d.log'%index)).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=childenv,stdout=log,stderr=subprocess.STDOUT)
            code=child.wait()
            if code:raise subprocess.CalledProcessError(code,command)


if __name__=='__main__':main()

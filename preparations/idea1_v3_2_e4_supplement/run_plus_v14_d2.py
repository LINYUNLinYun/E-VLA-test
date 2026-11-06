"""One immutable native D2 attempt with historical initial-input comparison."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from .native_envs import digest,write_json
from .evaluation import check_deadline,verify_service
from .evaluation_v2 import append
from .libero_plus_env import NativeLiberoPlusEnv
from .plus_v14_d2 import verify_manifest
from preparations.idea1_v3_2.transfer_client import rpc

KEYS=('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')


def value_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def initial_comparison(observation,instruction,path):
    checks={};image_changes=0;passed=True
    with np.load(path,allow_pickle=False) as old:
        for key in KEYS:
            a,b=np.asarray(observation[key]),old[key]
            if a.shape!=b.shape or a.dtype!=b.dtype: raise ValueError('Historical initial shape/dtype differs: '+key)
            delta=np.abs(a.astype(np.float64)-b.astype(np.float64));changed=int(np.count_nonzero(delta));maximum=float(delta.max(initial=0))
            image=key.endswith('image');image_changes+=changed if image else 0
            okay=(maximum<=1) if image else np.array_equal(a,b)
            checks[key]=dict(current_sha256=value_hash(a),historical_sha256=value_hash(b),max_abs=maximum,
                             changed_channels=changed,exact_equal=bool(np.array_equal(a,b)),passed=bool(okay))
            passed=passed and okay
        same_language=instruction==str(old['instruction']);passed=passed and same_language and image_changes<=4
    return dict(passed=bool(passed),fields=checks,instruction_equal=same_language,
                image_changed_channels_total=image_changes,image_budget=dict(max_abs=1,max_changed_channels=4))


def episode(a,job,m):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);env=None;writer=None;start=time.perf_counter()
    case=job['case'];identity=m['manifest_hash'];result=dict(status='running',success=None,mode='v14_d2',
        manifest_hash=identity,case_hash=job['case_hash'],case=case,group=job['group'],source_index=job['source_index'],
        gpu_uuid=a.gpu_uuid,pid=os.getpid(),steps=0,requests=0,paired_results=job['paired_results'],D=2,H=50,R=5)
    write_json(output/'result.json',result)
    try:
        verify_service(a.port,identity,'libero_transfer');check_deadline(a.deadline_unix)
        env=NativeLiberoPlusEnv(case,render=True,gpu=a.gpu,gpu_uuid=a.gpu_uuid);env.reset();result['native_info']=env.info
        if env.info['native_hz']!=20 or case['native_max_steps']!=600 or env.info['settle_steps']!=5: raise ValueError('Native cadence/settling/cap changed')
        instruction=env.info['instruction'];obs={key:np.asarray(env.obs[key]) for key in KEYS}
        np.savez_compressed(str(output/'initial_native_observation.npz'),**obs,instruction=instruction)
        comparison=initial_comparison(obs,instruction,job['initial_observation']['path'])
        full=next(r for r in job['paired_results'] if r['mode']=='full_d10');old_result=json.loads(Path(full['path']).read_text())
        comparison['initial_vector_sha256_equal']=env.info['initial_vector_sha256']==old_result['native_info']['initial_vector_sha256']
        old_first=json.loads(Path(job['first_input']['path']).open().readline())
        comparison['first_input_hashes']={k:dict(current=value_hash(obs[k]),historical=old_first['input_hashes'][k],equal=value_hash(obs[k])==old_first['input_hashes'][k]) for k in KEYS}
        # Hashes disclose renderer differences; the numerical budget above decides.
        comparison['passed']=comparison['passed'] and comparison['initial_vector_sha256_equal']
        result['initial_comparison']=comparison;write_json(output/'initial_comparison.json',comparison)
        if not comparison['passed']: raise RuntimeError('Initial physical/input mismatch exceeds preregistered budget')
        rpc(a.port,dict(op='reset'))
        physical=np.asarray(env.env.sim.get_state().flatten());np.savez_compressed(str(output/'initial_physical_state.npz'),state=physical)
        result['initial_physical_state_sha256']=value_hash(physical)
        result['historical_physical_state_availability']='Historical run saved init-vector hash and robot physical observation only; full simulator state unavailable for direct comparison'
        append(output/'telemetry.jsonl',dict(step=0,actual=env.actual(),executed_command=None))
        if job['video']:
            import imageio
            writer=imageio.get_writer(str(output/'video.mp4'),fps=20,codec='libx264',macro_block_size=2);writer.append_data(env.rgb())
        success=False
        while result['steps']<600 and not success:
            check_deadline(a.deadline_unix);seed=int(digest(['plus-evaluation-v1',job['case_hash'],result['requests']])[:8],16)
            if result['requests']==0 and seed!=old_first['seed']: raise ValueError('Historical request seed mismatch')
            observation={key:np.asarray(env.obs[key]) for key in KEYS};tick=time.perf_counter()
            response=rpc(a.port,dict(op='predict',family='libero_transfer',environment='libero_plus' if case['environment']=='libero_plus' else 'libero',
                mother_suite=case['mother_suite'],mode='v14_d2',observation=observation,instruction=instruction,seed=seed,category='quality'))
            actions=np.asarray(response['actions'],dtype=float);calls=response['observed_calls']
            if response.get('manifest_hash')!=identity or response.get('mode')!='v14_d2' or response.get('execute_length')!=5 or actions.shape!=(50,7) or not np.isfinite(actions).all(): raise ValueError('D2 output identity/H50R5 violated')
            if calls['ae_layer_calls']!=36 or calls['vision_encoder_calls']!=1 or calls['vlm_prefix_calls']!=1 or calls['verify_calls'] or calls['constructor_full_record_calls']: raise ValueError('D2 actual execution contract violated')
            append(output/'infer.jsonl',dict(request=result['requests'],step=result['steps'],seed=seed,completed_s=time.perf_counter()-tick,
                actions=actions.tolist(),response={k:v for k,v in response.items() if k!='actions'},input_hashes={k:value_hash(v) for k,v in observation.items()}))
            result['requests']+=1
            for action in actions[:min(5,600-result['steps'])]:
                _,audit=env.step_native(action);result['steps']+=1;success=audit['success']
                append(output/'telemetry.jsonl',dict(step=result['steps'],actual=env.actual(),executed_command=audit))
                if writer is not None: writer.append_data(env.rgb())
                if success: break
            write_json(output/'result.json',result)
        result.update(status='complete',success=bool(success),termination='native_success_oracle' if success else 'step_cap')
    except Exception: result.update(status='technical_failure',success=None,error=traceback.format_exc())
    finally:
        try:
            if writer is not None:
                writer.close();decode=subprocess.run(['ffmpeg','-v','error','-i',str(output/'video.mp4'),'-f','null','-'],capture_output=True)
                result['video']=dict(path=str(output/'video.mp4'),fps=20,full_stream_decode_passed=decode.returncode==0)
                if decode.returncode: raise RuntimeError('Full video decode failed: '+decode.stderr.decode(errors='replace'))
        except Exception: result.update(status='technical_failure',success=None,video_error=traceback.format_exc())
        try:
            if env is not None: env.close()
        except Exception: result.update(status='technical_failure',success=None,close_error=traceback.format_exc())
        result['wall_s']=time.perf_counter()-start;write_json(output/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);p.add_argument('--group',required=True)
    p.add_argument('--source-index',type=int,required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--launch',action='store_true');a=p.parse_args()
    m=verify_manifest(a.protocol,(a.group,a.source_index));job=next(j for j in m['jobs'] if j['group']==a.group and j['source_index']==a.source_index)
    if not a.launch: print(json.dumps(dict(dry_run=True,mode='v14_d2',source_index=job['source_index'])));return
    if a.deadline_unix>m['deadline_unix']: raise ValueError('Deadline exceeds frozen manifest')
    result=episode(a,job,m);raise SystemExit(0 if result['status']=='complete' else 1)


if __name__=='__main__': main()

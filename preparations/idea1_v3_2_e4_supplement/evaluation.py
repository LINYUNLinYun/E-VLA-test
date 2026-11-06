"""Native long-horizon evaluation client; technical failures never become policy zeros."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from preparations.idea1_v3_2.transfer_client import rpc,raw_hash
from .native_envs import ROOT,NativeCalvinEnv,chain_metrics,digest,sha,write_json


def json_default(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def request_seed(family,case,subtask,replan):
    return int(digest(['supplement-evaluation-v1',family,str(case),int(subtask),int(replan)])[:8],16)


def check_deadline(deadline):
    if deadline and time.time()>=deadline: raise TimeoutError('Frozen evaluation deadline reached')


def protocol_identity(path):
    protocol=json.loads(Path(path).read_text())
    expected=digest({k:v for k,v in protocol.items() if k!='manifest_hash'})
    if protocol.get('manifest_hash')!=expected: raise ValueError('Policy protocol hash mismatch')
    return protocol


def verify_service(port,identity,family):
    health=rpc(port,dict(op='health'))
    protocol=health['protocol']
    if protocol.get('manifest_hash')!=identity or protocol.get('family')!=family:
        raise ValueError('RPC service identity/family does not match fixed policy protocol')
    return protocol


def native_preflight(args,identity):
    with np.load(str(args.preflight_input),allow_pickle=False) as data:
        raw=dict(base_rgb=data['base_rgb'],wrist_rgb=data['wrist_rgb'],state7=data['state7'],
                 gripper_semantics=args.gripper_semantics)
        instruction=str(data['instruction'])
    if args.gripper_semantics=='action_sign' and float(raw['state7'][-1]) not in (-1,1):
        raise ValueError('Preflight capture does not contain training-sign semantics')
    gate=rpc(args.port,dict(op='preflight',raw=raw,instruction=instruction,seed=7))
    if gate.get('manifest_hash')!=identity or not gate.get('passed'):
        raise RuntimeError('Native same-input strict Full preflight failed')
    return gate


def run_calvin_sequence(args,env,sequence,mode,identity):
    directory=Path(args.output)/('seq%04d'%sequence['index'])/mode
    directory.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();writer=None
    telemetry=[];infers=[];tasks=[]
    result=dict(status='running',environment='calvin',family=args.family,mode=mode,
        sequence_index=sequence['index'],manifest_hash=identity,success=None,completed_subtasks=0,
        initial_condition=sequence['initial_condition'],tasks=sequence['tasks'],
        gripper_semantics=args.gripper_semantics,seed_rule='family+sequence+subtask+replan; mode excluded',
        formal_sequence_unit=True,pilot=bool(args.pilot),model_used=True,gpu_renderer_used=False)
    try:
        check_deadline(args.deadline_unix)
        env.reset_sequence(sequence['initial_condition'])
        actual_initial=np.r_[env.obs['robot_obs'],env.obs['scene_obs']]
        result['initial_native_state_sha256']=hashlib.sha256(actual_initial.tobytes()).hexdigest()
        result['native_info']=env.info
        if args.video and int(sequence['index'])==0:
            import imageio
            writer=imageio.get_writer(str(directory/'video.mp4'),fps=30,codec='libx264',pixelformat='yuv420p',macro_block_size=2)
            result['video']=dict(path=str(directory/'video.mp4'),fps=30,kind='native control Hz')
        total_steps=0
        def capture(subtask,step,audit=None):
            telemetry.append(dict(global_step=total_steps,subtask_index=subtask,subtask_step=step,
                sim_time_s=total_steps/30.,actual=env.actual(),robot_obs=np.asarray(env.obs['robot_obs']).tolist(),executed_command=audit))
            if writer is not None: writer.append_data(np.asarray(env.obs['rgb_obs']['rgb_static'],dtype=np.uint8))
        capture(0,0)
        for task_index,task in enumerate(sequence['tasks']):
            rpc(args.port,dict(op='reset'))
            instruction=env.start_subtask(task)
            task_row=dict(index=task_index,task=task,instruction=instruction,success=False,steps=0,predictions=0)
            tasks.append(task_row)
            reached=False
            while task_row['steps']<360 and not reached:
                check_deadline(args.deadline_unix)
                raw=env.policy_raw(args.gripper_semantics);raw['gripper_semantics']=args.gripper_semantics
                state=np.asarray(raw['state7']).copy()
                seed=request_seed(args.family,sequence['index'],task_index,task_row['predictions'])
                tick=time.perf_counter()
                response=rpc(args.port,dict(op='predict',family=args.family,environment='calvin',mode=mode,
                    raw=raw,instruction=instruction,seed=seed,category='quality'))
                elapsed=time.perf_counter()-tick
                if response.get('manifest_hash')!=identity: raise ValueError('Response policy identity changed')
                actions=np.asarray(response['actions'],dtype=np.float64)
                if actions.shape!=(5,7) or not np.isfinite(actions).all() or response.get('execute_length')!=5:
                    raise ValueError('Native CALVIN requires completed finiteH5/R5 decoded native actions')
                if response.get('action_contract')!='native_CALVIN_relative7_Euler_no_extraDelta':
                    raise ValueError('Native CALVIN action contract changed or a LIBERO route was substituted')
                noise=np.random.default_rng(seed).standard_normal((5,32),dtype=np.float32)
                row=dict(index=len(infers),subtask_index=task_index,subtask_step=task_row['steps'],seed=seed,
                    state7=state.tolist(),instruction=instruction,noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
                    observation_image_sha256={k:hashlib.sha256(np.ascontiguousarray(raw[k]).tobytes()).hexdigest() for k in ('base_rgb','wrist_rgb')},
                    client_completed_s=elapsed,actions=actions.tolist(),response={k:v for k,v in response.items() if k!='actions'})
                infers.append(row);task_row['predictions']+=1
                if args.save_observations:
                    np.savez_compressed(str(directory/('request_%05d.npz'%row['index'])),base_rgb=raw['base_rgb'],wrist_rgb=raw['wrist_rgb'],state7=state)
                elif task_row['predictions']==1:
                    np.savez_compressed(str(directory/('subtask_%d_initial.npz'%task_index)),base_rgb=raw['base_rgb'],wrist_rgb=raw['wrist_rgb'],state7=state)
                for action in actions[:min(5,360-task_row['steps'])]:
                    tick=time.perf_counter();_,success,audit=env.step_native(action,'relative')
                    audit['environment_step_s']=time.perf_counter()-tick
                    total_steps+=1;task_row['steps']+=1;capture(task_index,task_row['steps'],audit)
                    if success: reached=True;break
            task_row['success']=reached
            if not reached:
                result['failed_task']=task;result['failure_subtask_index']=task_index;break
            result['completed_subtasks']+=1
        result.update(status='complete',success=result['completed_subtasks']==5,native_steps=total_steps,
            task_results=tasks,predictions=len(infers),wall_s=time.perf_counter()-start)
    except Exception:
        result.update(status='technical_failure',success=None,task_results=tasks,error=traceback.format_exc(),wall_s=time.perf_counter()-start)
    finally:
        for filename,rows in (('telemetry.jsonl',telemetry),('infer.jsonl',infers)):
            (directory/filename).write_text(''.join(json.dumps(r,default=json_default,allow_nan=False)+'\n' for r in rows))
        if writer is not None:
            writer.close()
            decode=subprocess.run(['ffmpeg','-v','error','-i',str(directory/'video.mp4'),'-f','null','-'],capture_output=True)
            result['video']['full_stream_decode_passed']=decode.returncode==0
            if decode.returncode:
                result.update(status='technical_failure',success=None,video_decode_error=decode.stderr.decode(errors='replace'))
        write_json(directory/'result.json',result)
    return result


def run_calvin(args):
    protocol=protocol_identity(args.protocol);identity=protocol['manifest_hash']
    sequences_payload=json.loads(Path(args.sequence_file).read_text())
    canonical=sequences_payload['sequences']
    if len(canonical)!=1000: raise ValueError('Use generated fullcanonical1000, then fixedprefix')
    if not 1<=args.sequence_count<=1000: raise ValueError('Sequence count out of range')
    selected=canonical[:args.sequence_count]
    if args.family!='calvin_abc_d_native': raise ValueError('This driver native path never substitutes LIBERO model input/actions')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    service=verify_service(args.port,identity,args.family)
    receipt=dict(status='running',environment='calvin',family=args.family,manifest_hash=identity,
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),controller_pid=os.getpid(),
        scientific_unit='official5-subtask sequence',sequences=selected,modes=args.modes,pilot=args.pilot,
        sequence_file_sha256=sha(args.sequence_file),canonical_sequence_sha256=sequences_payload['sequence_sha256'],
        source_files={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'preparations/idea1_v3_2_e4_supplement/native_envs.py']},
        state_gripper_semantics=args.gripper_semantics,service=service,results=[])
    write_json(output/'run.json',receipt)
    receipt['native_input_preflight']=native_preflight(args,identity)
    env=NativeCalvinEnv()
    try:
        for sequence in selected:
            for mode in args.modes:
                result=run_calvin_sequence(args,env,sequence,mode,identity)
                receipt['results'].append(dict(sequence=result['sequence_index'],mode=mode,status=result['status'],
                    completed_subtasks=result['completed_subtasks'],failed_task=result.get('failed_task'),wall_s=result['wall_s']))
                write_json(output/'run.json',receipt)
                if result['status']!='complete': raise RuntimeError('Native attempt technical failure; preserved, not counted as policy zero')
        receipt['metrics']={mode:chain_metrics([r['completed_subtasks'] for r in receipt['results'] if r['mode']==mode and r['status']=='complete']) for mode in args.modes}
        receipt['failed_task_counts']={mode:dict(Counter(r['failed_task'] for r in receipt['results'] if r['mode']==mode and r.get('failed_task'))) for mode in args.modes}
        receipt['status']='complete'
    except Exception:
        receipt.update(status='technical_failure',error=traceback.format_exc());raise
    finally:
        env.close();write_json(output/'run.json',receipt)
    print(json.dumps(dict(status=receipt['status'],metrics=receipt['metrics'])))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--environment',choices=('calvin',),default='calvin')
    parser.add_argument('--family',default='calvin_abc_d_native')
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--sequence-file',required=True)
    parser.add_argument('--sequence-count',type=int,default=100)
    parser.add_argument('--modes',nargs='+',choices=('full_d10','v14_d10'),default=['full_d10','v14_d10'])
    parser.add_argument('--gripper-semantics',choices=('action_sign','opening_width_m'),required=True)
    parser.add_argument('--preflight-input',required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--deadline-unix',type=float,required=True)
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--save-observations',action='store_true')
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    if not args.launch: print(json.dumps(dict(dry_run=True,**vars(args))));return
    run_calvin(args)


if __name__=='__main__': main()

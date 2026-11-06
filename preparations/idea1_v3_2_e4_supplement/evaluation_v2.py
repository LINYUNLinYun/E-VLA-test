"""Resumable exact-action native CALVIN fixed-prefix evaluator.

Completed cases are reused only under identical plan/source/policy hashes.
Interrupted attempts remain immutable; a new attempt resets the whole sequence.
Every native step/request and completed subtask is persisted immediately.
"""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
from .native_envs import ROOT,digest,sha,write_json,chain_metrics
from .native_exact import ExactNativeCalvinEnv
from .evaluation import protocol_identity,verify_service,native_preflight,request_seed,check_deadline,json_default
from preparations.idea1_v3_2.transfer_client import rpc


def append(path,row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row,default=json_default,allow_nan=False)+'\n');f.flush()


def result_matches(result,plan,sequence,mode):
    return (result.get('status')=='complete' and result.get('plan_hash')==plan['plan_hash']
        and result.get('manifest_hash')==plan['policy_manifest_hash'] and result.get('mode')==mode
        and result.get('sequence_index')==sequence['index'] and result.get('sequence_hash')==digest(sequence))


def run_case(args,env,sequence,mode,plan,directory):
    directory.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    result=dict(status='running',plan_hash=plan['plan_hash'],manifest_hash=plan['policy_manifest_hash'],
        sequence_index=sequence['index'],sequence_hash=digest(sequence),sequence=sequence,mode=mode,
        completed_subtasks=0,success=None,task_results=[],native_steps=0,requests=0,
        gripper_semantics='action_sign',action_contract='native_exact_no_first6_clipping',
        environment='calvin',family='calvin_abc_d_native',outside_unit_components=0,total_decoded_components=0)
    write_json(directory/'result.json',result)
    writer=None
    try:
        env.reset_sequence(sequence['initial_condition'])
        result['initial_native_state_sha256']=__import__('hashlib').sha256(np.r_[env.obs['robot_obs'],env.obs['scene_obs']].tobytes()).hexdigest()
        np.savez_compressed(str(directory/'initial_native_observation.npz'),robot_obs=env.obs['robot_obs'],
            scene_obs=env.obs['scene_obs'],base_rgb=env.obs['rgb_obs']['rgb_static'],wrist_rgb=env.obs['rgb_obs']['rgb_gripper'])
        result['initial_observation_path']=str(directory/'initial_native_observation.npz')
        result['initial_match_policy']='Record actual robot/scene floats and RGB; hash inequality alone is not physical initial-state mismatch'
        for previous in sorted(directory.parents[1].glob('*/attempt*/initial_native_observation.npz')):
            if previous.parent==directory: continue
            with np.load(str(previous)) as other:
                result.setdefault('initial_comparisons',[]).append(dict(reference=str(previous),
                    robot_max_abs=float(np.max(np.abs(np.asarray(env.obs['robot_obs'])-other['robot_obs']))),
                    scene_max_abs=float(np.max(np.abs(np.asarray(env.obs['scene_obs'])-other['scene_obs']))),
                    base_rgb_mean_abs=float(np.mean(np.abs(np.asarray(env.obs['rgb_obs']['rgb_static'],dtype=float)-other['base_rgb']))),
                    wrist_rgb_mean_abs=float(np.mean(np.abs(np.asarray(env.obs['rgb_obs']['rgb_gripper'],dtype=float)-other['wrist_rgb'])))))
        if args.video and sequence['index']%10==0:
            import imageio
            writer=imageio.get_writer(str(directory/'video.mp4'),fps=30,codec='libx264',macro_block_size=2)
        for ti,task in enumerate(sequence['tasks']):
            rpc(args.port,dict(op='reset'));instruction=env.start_subtask(task)
            row=dict(index=ti,task=task,instruction=instruction,steps=0,requests=0,success=False)
            result['task_results'].append(row);write_json(directory/'result.json',result)
            while row['steps']<360 and not row['success']:
                check_deadline(args.deadline_unix)
                raw=env.policy_raw('action_sign');raw['gripper_semantics']='action_sign'
                seed=request_seed('calvin_abc_d_native',sequence['index'],ti,row['requests'])
                tick=time.perf_counter();response=rpc(args.port,dict(op='predict',family='calvin_abc_d_native',environment='calvin',
                    mode=mode,raw=raw,instruction=instruction,seed=seed,category='quality'))
                actions=np.asarray(response['actions'],dtype=np.float64)
                if response.get('manifest_hash')!=plan['policy_manifest_hash'] or actions.shape!=(5,7) or not np.isfinite(actions).all():
                    raise ValueError('Wrong service identity/nonfinite/native shape')
                if response.get('execute_length')!=5 or response.get('action_contract')!='native_CALVIN_relative7_Euler_no_extraDelta':
                    raise ValueError('Native H5/R5 action contract mismatch')
                append(directory/'infer.jsonl',dict(task=ti,step=row['steps'],seed=seed,state7=np.asarray(raw['state7']).tolist(),
                    client_completed_s=time.perf_counter()-tick,actions=actions.tolist(),response={k:v for k,v in response.items() if k!='actions'},
                    image_hashes={k:__import__('hashlib').sha256(np.asarray(raw[k]).tobytes()).hexdigest() for k in ('base_rgb','wrist_rgb')}))
                result['outside_unit_components']+=int(np.sum(np.abs(actions[:,:6])>1));result['total_decoded_components']+=30
                result['requests']+=1;row['requests']+=1
                for action in actions[:min(5,360-row['steps'])]:
                    _,success,audit=env.step_native(action)
                    row['steps']+=1;result['native_steps']+=1;row['success']=bool(success)
                    append(directory/'telemetry.jsonl',dict(task_index=ti,subtask_step=row['steps'],global_step=result['native_steps'],
                        actual=env.actual(),robot_obs=np.asarray(env.obs['robot_obs']).tolist(),executed_command=audit,oracle_success=bool(success)))
                    if writer is not None: writer.append_data(env.obs['rgb_obs']['rgb_static'])
                    if success: break  # discard remainder immediately at actual native oracle success
                write_json(directory/'result.json',result)
            append(directory/'tasks.jsonl',row);write_json(directory/('task_%d.json'%ti),row)
            if not row['success']:
                result['failed_task']=task;break
            result['completed_subtasks']+=1;write_json(directory/'result.json',result)
        result.update(status='complete',success=result['completed_subtasks']==5)
    except Exception:
        result.update(status='technical_failure',success=None,error=traceback.format_exc())
    finally:
        if writer is not None:
            writer.close()
            import subprocess
            check=subprocess.run(['ffmpeg','-v','error','-i',str(directory/'video.mp4'),'-f','null','-'],capture_output=True)
            result['video']=dict(path=str(directory/'video.mp4'),fps=30,full_stream_decode_passed=check.returncode==0)
            if check.returncode: result.update(status='technical_failure',success=None,video_error=check.stderr.decode(errors='replace'))
        result['wall_s']=time.perf_counter()-started;write_json(directory/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--sequence-file',required=True)
    p.add_argument('--sequence-count',type=int,default=100);p.add_argument('--modes',nargs='+',default=['full_d10','v14_d10'],choices=['full_d10','v14_d10'])
    p.add_argument('--max-new-sequences',type=int,default=10);p.add_argument('--preflight-input',required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--output',required=True);p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--video',action='store_true');p.add_argument('--launch',action='store_true');a=p.parse_args()
    a.gripper_semantics='action_sign'
    payload=json.loads(Path(a.sequence_file).read_text());canonical=payload['sequences']
    if len(canonical)!=1000 or not 1<=a.sequence_count<=1000 or a.max_new_sequences<1: raise ValueError('Canonical prefix/shard invalid')
    protocol=protocol_identity(a.protocol)
    plan=dict(schema=2,scientific_unit='native official five-task sequence',policy_manifest_hash=protocol['manifest_hash'],
        canonical_sequence_file_sha256=sha(a.sequence_file),sequences=canonical[:a.sequence_count],modes=a.modes,
        gripper_semantics='training_stored_action_sign',action_semantics='official first6 passthrough; gripper threshold only',
        source_files={str(path.relative_to(ROOT)):sha(path) for path in (Path(__file__),Path(__file__).with_name('native_exact.py'),Path(__file__).with_name('native_envs.py'))})
    plan['plan_hash']=digest(plan)
    if not a.launch: print(json.dumps(dict(dry_run=True,plan_hash=plan['plan_hash'],target_sequences=a.sequence_count,per_launch_new_sequences=a.max_new_sequences)));return
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'plan.json').exists():
        if json.loads((output/'plan.json').read_text())!=plan: raise ValueError('Existing immutable plan mismatch')
    else: write_json(output/'plan.json',plan)
    verify_service(a.port,protocol['manifest_hash'],'calvin_abc_d_native');gate=native_preflight(a,protocol['manifest_hash'])
    write_json(output/('gate_%d.json'%int(time.time())),gate)
    env=ExactNativeCalvinEnv();results=[];new_sequences=0
    try:
        for sequence in plan['sequences']:
            missing=[]
            for mode in a.modes:
                base=output/('seq%04d'%sequence['index'])/mode
                completed=[]
                for path in sorted(base.glob('attempt*/result.json')):
                    value=json.loads(path.read_text())
                    if result_matches(value,plan,sequence,mode): completed.append(value)
                if len(completed)>1: raise ValueError('Duplicate complete scientific case')
                if completed: results.append(completed[0])
                else: missing.append((mode,base))
            if not missing: continue
            if new_sequences>=a.max_new_sequences: break
            check_deadline(a.deadline_unix);new_sequences+=1
            for mode,base in missing:
                attempts=list(base.glob('attempt*'));directory=base/('attempt%03d'%len(attempts))
                result=run_case(a,env,sequence,mode,plan,directory);results.append(result)
                write_json(output/'progress.json',dict(status='running',plan_hash=plan['plan_hash'],results=results))
                if result['status']!='complete': raise RuntimeError('Technical failure preserved; resume creates new attempt, not policy zero')
        write_json(output/'progress.json',dict(status='complete' if len(results)==a.sequence_count*len(a.modes) else 'shard_complete',
            plan_hash=plan['plan_hash'],target_sequences=a.sequence_count,completed_cases=len(results),new_sequences=new_sequences,
            metrics={m:chain_metrics([r['completed_subtasks'] for r in results if r['mode']==m and r['status']=='complete']) for m in a.modes},results=results))
    finally: env.close()


if __name__=='__main__': main()

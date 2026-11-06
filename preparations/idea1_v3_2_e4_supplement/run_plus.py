"""Isolated native LIBERO-Plus/original controls, immutable resumable attempts."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from .native_envs import ROOT,digest,sha,write_json
from .libero_plus_env import PLUS,NativeLiberoPlusEnv,isolated_child_environment
from .evaluation import protocol_identity,verify_service,check_deadline
from .evaluation_v2 import append
from preparations.idea1_v3_2.transfer_client import rpc


def modes(case): return ['full_d10','v14_d10',case['matching_draft']]


def episode(a,case,mode,protocol):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);env=None;writer=None;start=time.perf_counter()
    identity=protocol['manifest_hash'];case_hash=digest(case)
    result=dict(status='running',success=None,manifest_hash=identity,case_hash=case_hash,case=case,mode=mode,
        environment=case['environment'],gpu_uuid=a.gpu_uuid,pid=os.getpid(),steps=0,requests=0,
        input_contract='exact original unflipped native observation; server native_observation once',
        source_files={str(path.relative_to(ROOT)):sha(path) for path in (Path(__file__),Path(__file__).with_name('libero_plus_env.py'))})
    write_json(output/'result.json',result)
    try:
        verify_service(a.port,identity,'libero_transfer');check_deadline(a.deadline_unix)
        env=NativeLiberoPlusEnv(case,render=True,gpu=a.gpu,gpu_uuid=a.gpu_uuid);env.reset()
        result['native_info']=env.info
        instruction=env.info['instruction'];rpc(a.port,dict(op='reset'))
        np.savez_compressed(str(output/'initial_native_observation.npz'),**{key:env.obs[key] for key in
            ('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')},instruction=instruction)
        if a.video:
            import imageio
            writer=imageio.get_writer(str(output/'video.mp4'),fps=env.info['native_hz'],codec='libx264',macro_block_size=2)
            writer.append_data(env.rgb())
        success=False
        while result['steps']<case['native_max_steps'] and not success:
            check_deadline(a.deadline_unix)
            seed=int(digest(['plus-evaluation-v1',case_hash,result['requests']])[:8],16)
            observation={key:np.asarray(env.obs[key]) for key in ('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')}
            tick=time.perf_counter();response=rpc(a.port,dict(op='predict',family='libero_transfer',
                environment='libero_plus' if case['environment']=='libero_plus' else 'libero',mother_suite=case['mother_suite'],
                mode=mode,observation=observation,instruction=instruction,seed=seed,category='quality'))
            actions=np.asarray(response['actions'],dtype=float)
            if response.get('manifest_hash')!=identity or actions.shape!=(50,7) or not np.isfinite(actions).all():
                raise ValueError('Native LIBERO response identity/finite H50x7 mismatch')
            if response.get('execute_length',5)!=5: raise ValueError('R5 contract mismatch')
            append(output/'infer.jsonl',dict(request=result['requests'],step=result['steps'],seed=seed,
                completed_s=time.perf_counter()-tick,actions=actions.tolist(),response={key:value for key,value in response.items() if key!='actions'},
                input_hashes={key:__import__('hashlib').sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for key,value in observation.items()}))
            result['requests']+=1
            for action in actions[:min(5,case['native_max_steps']-result['steps'])]:
                _,audit=env.step_native(action);result['steps']+=1;success=audit['success']
                append(output/'telemetry.jsonl',dict(step=result['steps'],actual=env.actual(),executed_command=audit))
                if writer is not None: writer.append_data(env.rgb())
                if success: break
            write_json(output/'result.json',result)
        result.update(status='complete',success=bool(success),termination='native_success_oracle' if success else 'step_cap')
    except Exception:
        result.update(status='technical_failure',success=None,error=traceback.format_exc())
    finally:
        if writer is not None:
            writer.close();decode=subprocess.run(['ffmpeg','-v','error','-i',str(output/'video.mp4'),'-f','null','-'],capture_output=True)
            result['video']=dict(path=str(output/'video.mp4'),fps=env.info['native_hz'],full_stream_decode_passed=decode.returncode==0)
            if decode.returncode: result.update(status='technical_failure',success=None,video_error=decode.stderr.decode(errors='replace'))
        if env is not None: env.close()
        result['wall_s']=time.perf_counter()-start;write_json(output/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);p.add_argument('--output',required=True)
    p.add_argument('--indices',nargs='+',type=int);p.add_argument('--baseline',action='store_true')
    p.add_argument('--max-new-cases',type=int,default=10);p.add_argument('--port',type=int,required=True)
    p.add_argument('--gpu',type=int,required=True);p.add_argument('--gpu-uuid',required=True);p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--child-index',type=int);p.add_argument('--child-mode');p.add_argument('--video',action='store_true');p.add_argument('--launch',action='store_true');a=p.parse_args()
    protocol=protocol_identity(a.protocol);selection=protocol['case_selection'];cases=selection['original_baselines' if a.baseline else 'selected']
    indices=a.indices if a.indices is not None else list(range(len(cases)))
    if len(set(indices))!=len(indices) or any(i<0 or i>=len(cases) for i in indices): raise ValueError('Invalid frozen case indices')
    if a.child_index is not None:
        if not a.launch: raise ValueError('Child requires explicit launch')
        case=cases[a.child_index]
        if a.child_mode not in modes(case): raise ValueError('Nonmatching mother-suite draft')
        value=episode(a,case,a.child_mode,protocol);raise SystemExit(0 if value['status']=='complete' else 1)
    if not a.launch: print(json.dumps(dict(dry_run=True,manifest_hash=protocol['manifest_hash'],cases=len(indices),conditions=3*len(indices),max_new_cases=a.max_new_cases)));return
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    plan=dict(manifest_hash=protocol['manifest_hash'],indices=indices,baseline=a.baseline,case_hashes=[digest(cases[i]) for i in indices],
        conditions=[modes(cases[i]) for i in indices],source_files={str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('libero_plus_env.py')):sha(Path(__file__).with_name('libero_plus_env.py'))})
    plan['plan_hash']=digest(plan)
    if (output/'plan.json').exists() and json.loads((output/'plan.json').read_text())!=plan: raise ValueError('Existing scientific plan/source mismatch')
    write_json(output/'plan.json',plan);new_cases=0;rows=[]
    for index in indices:
        case=cases[index];pending=[]
        for mode in modes(case):
            base=output/('case%04d'%index)/mode
            complete=[path for path in base.glob('attempt*/result.json') if (lambda value:value.get('status')=='complete' and value.get('manifest_hash')==protocol['manifest_hash'] and value.get('case_hash')==digest(case) and value.get('mode')==mode)(json.loads(path.read_text()))]
            if len(complete)>1: raise ValueError('Duplicate completed scientific case')
            if complete: rows.append(dict(index=index,mode=mode,reused=str(complete[0])))
            else: pending.append((mode,base))
        if not pending: continue
        if new_cases>=a.max_new_cases: break
        new_cases+=1
        for mode,base in pending:
            check_deadline(a.deadline_unix);directory=base/('attempt%03d'%len(list(base.glob('attempt*'))))
            env=isolated_child_environment(case,output/('config_original' if a.baseline else 'config_plus'),a.gpu,a.gpu_uuid)
            interpreter=ROOT/'policies/libero-eval-v21/.venv/bin/python' if a.baseline else PLUS/'.venv/bin/python'
            command=[str(interpreter),'-m',__spec__.name,'--protocol',a.protocol,'--output',str(directory),'--port',str(a.port),
                '--gpu',str(a.gpu),'--gpu-uuid',a.gpu_uuid,'--deadline-unix',str(a.deadline_unix),'--child-index',str(index),'--child-mode',mode,'--launch']
            if a.baseline: command.append('--baseline')
            if a.video and index==indices[0]: command.append('--video')
            base.mkdir(parents=True,exist_ok=True)
            with (base/(directory.name+'.log')).open('w') as log:
                process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                try: code=process.wait(timeout=max(1,a.deadline_unix-time.time()))
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try: process.wait(timeout=15)
                    except subprocess.TimeoutExpired: process.kill();process.wait()
                    code=124
            rows.append(dict(index=index,mode=mode,pid=process.pid,exit_code=code,result=str(directory/'result.json')))
            write_json(output/'progress.json',dict(plan_hash=plan['plan_hash'],status='running',rows=rows))
            if code: raise RuntimeError('Child technical failure preserved; no policy-zero substitution')
    write_json(output/'progress.json',dict(plan_hash=plan['plan_hash'],status='shard_complete',new_cases=new_cases,rows=rows))


if __name__=='__main__': main()

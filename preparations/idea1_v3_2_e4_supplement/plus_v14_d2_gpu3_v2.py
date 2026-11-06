"""GPU3-only strict D2 continuation, no videos, immutable v2 artifacts."""
import argparse
import json
import os
from pathlib import Path
import pickle
import socket
import struct
import subprocess
import sys
import time
import traceback
import numpy as np
from .native_envs import ROOT,digest,sha,write_json
from .evaluation import protocol_identity
from .plus_v14_d2 import bound_file,completed
from .policy_plus_v14_d2 import V14D2Policy,array
from .policy_plus_mirror import audit_cpu_backend
from .parallel_continue import process_identity,same_process
from .policy_server import native_libero_raw
from .libero_plus_env import PLUS,isolated_child_environment
from .run_plus_v14_d2 import episode
from preparations.idea1_v3.policy import TritonBackend,V14
from preparations.idea1_v3_2.transfer_policy import route_observations,observed_counts,json_safe
from preparations.idea1_v3_1.server import receive
from preparations.idea1_v3_2_e4_supplement.policy_native_watch_v3 import rpc

GPU=3
UUID='GPU-555cde36-5e41-bc44-0c69-ea47358fd703'
KIND='native_LIBERO_Plus_half_V14_D2_GPU3_strict_novideo_v2'


def register(parent_path,output,deadline):
    parent=protocol_identity(parent_path)
    if parent['manifest_hash']!='242190d617125f75f960a70a12cbeb3b53b7fd4227e6a93dbe1c6ffbdeca7479':raise ValueError('Exact preserved failed cross-device D2 parent required')
    if not time.time()<deadline<=time.time()+12*3600:raise ValueError('Fresh deadline at most12h required')
    m={k:v for k,v in parent.items() if k not in ('manifest_hash','created_unix')}
    m.update(kind=KIND,previous_d2_manifest=bound_file(parent_path),previous_d2_manifest_hash=parent['manifest_hash'],
        deadline_unix=deadline,created_unix=time.time(),required_gpu_index=GPU,required_gpu_uuid=UUID,
        video_source_indices=[],video_policy='All new195 episodes video=false; existing585 reused without baseline rerun',
        startup_gate=dict(historical_D10_raw_max_abs=1e-5,historical_D10_decoded_max_abs=1e-5,D2_native_max_abs=1e-5,
            rtol=0,device='Only original physical GPU3 UUID; no cross-GPU tolerance, fallback, or device trial'),
        cross_gpu_limitation='No cross-GPU quality accepted: this v2 requires the exact historical physical GPU3 UUID.',
        idle_policy=dict(sample_interval_seconds=30,max_memory_mib=16,required_compute_processes=0,required_utilization_percent=0),
        staged_execution='Run one new D2 case then await explicit continue.json bound to manifest and first-result sha; deadline closes own service automatically')
    m['jobs']=[dict(job,video=False) for job in parent['jobs']]
    m['source_files']=dict(parent['source_files']);m['source_files'][str(Path(__file__).resolve())]=sha(__file__)
    m['manifest_hash']=digest(m);output=Path(output);output.mkdir(parents=True,exist_ok=False);write_json(output/'manifest.json',m)
    return m


def verify(path,job_identity=None):
    m=protocol_identity(path)
    if m.get('kind')!=KIND or m.get('required_gpu_uuid')!=UUID or m.get('required_gpu_index')!=3 or any(j['video'] for j in m['jobs']):raise ValueError('GPU3-only no-video v2 identity required')
    for name,expected in m['source_files'].items():
        path=Path(name);path=path if path.is_absolute() else ROOT/path
        if sha(path)!=expected:raise ValueError('Frozen source changed: '+str(path))
    evidence=[m['previous_d2_manifest'],m['half_manifest'],m['parent_manifest'],m['audit_ledger']]+m['bound_files']
    if job_identity is not None:
        job=next(j for j in m['jobs'] if (j['group'],j['source_index'])==job_identity)
        evidence=[job['initial_observation'],job['first_input']]+[dict(path=r['path'],sha256=r['sha256']) for r in job['paired_results']]
    for entry in evidence:
        if sha(entry['path'])!=entry['sha256']:raise ValueError('Frozen evidence changed: '+entry['path'])
    return m


def sample_gpu3():
    base=['nvidia-smi','-i',UUID]
    processes=subprocess.check_output(base+['--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True).strip()
    device=subprocess.check_output(base+['--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    fields=[v.strip() for v in device.split(',')]
    if len(fields)!=4 or fields[0]!='3' or fields[1]!=UUID:raise RuntimeError('Physical GPU3 identity changed')
    memory,utilization=int(fields[2]),int(fields[3])
    return dict(index=3,uuid=UUID,compute_processes=processes,memory_mib=memory,utilization_percent=utilization,
                idle=not processes and memory<=16 and utilization==0,time_unix=time.time())


def detailed_comparison(actual,expected):
    actual,expected=array(actual),array(expected)
    if actual.shape!=expected.shape or actual.ndim!=2 or not np.isfinite(actual).all() or not np.isfinite(expected).all():raise ValueError('Comparison arrays invalid')
    delta=np.abs(actual-expected);index=np.unravel_index(np.argmax(delta),delta.shape)
    return dict(passed=bool(np.allclose(actual,expected,atol=1e-5,rtol=0)),max_abs=float(delta.max()),
        bitwise_equal=bool(np.array_equal(actual,expected)),argmax=list(map(int,index)),
        argmax_actual=float(actual[index]),argmax_expected=float(expected[index]),per_channel_max_abs=delta.max(axis=0).tolist(),
        active_first7_max_abs=float(delta[:,:7].max()),padding_after7_max_abs=float(delta[:,7:].max()) if delta.shape[1]>7 else None,
        atol=1e-5,rtol=0)


def strict_preflight(policy,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);checks=[]
    for saved in policy.manifest['startup_inputs']:
        if time.time()>=policy.manifest['deadline_unix']:raise TimeoutError('Strict preflight deadline reached')
        with np.load(saved['observation_path'],allow_pickle=False) as source:
            obs={k:source[k] for k in source.files if k!='instruction'};instruction=str(source['instruction'])
        with Path(saved['infer_path']).open() as stream:old=json.loads(stream.readline())
        seed=old['seed'];policy.backend.ensure_prompt(instruction);skip=V14 if saved['mode']=='v14_d10' else ()
        with route_observations() as observed:
            context,_=TritonBackend.prepare(policy.backend,obs,instruction,skip,encode=True)
            # Current real prefix/state retained for diagnosis. Historical
            # intermediate K/V was never saved; do not imply it can be compared.
            current_context={name:array(context[name]) for name in ('K','V','state','transformed_state')}
            actions,raw,_=TritonBackend.sample(policy.backend,context,seed,10)
        raw,actions=array(raw),array(actions);historical_calls=observed_counts(observed)
        old_raw,old_actions=array(old['response']['raw_actions']),array(old['actions'])
        historical=dict(raw=detailed_comparison(raw,old_raw),decoded=detailed_comparison(actions,old_actions),observed_calls=historical_calls,
            first5_gripper_sign_equal=bool(np.array_equal(np.sign(actions[:5,6]),np.sign(old_actions[:5,6]))))
        with route_observations() as observed:
            context,_=TritonBackend.prepare(policy.backend,obs,instruction,V14,encode=True)
            native,native_raw,_=TritonBackend.sample(policy.backend,context,seed,2)
        calls=observed_counts(observed)
        wrapped=policy.predict('v14_d2',native_libero_raw(obs),instruction,seed,category='preflight')
        parity=dict(raw=detailed_comparison(native_raw,wrapped['raw_actions']),decoded=detailed_comparison(native,wrapped['actions']),reference_calls=calls)
        parity['passed']=parity['raw']['passed'] and parity['decoded']['passed'] and calls['ae_layer_calls']==36 and calls['vision_encoder_calls']==1 and calls['vlm_prefix_calls']==1 and calls['verify_calls']==0 and calls['constructor_full_record_calls']==0
        path=directory/(saved['mode']+'_arrays.npz')
        np.savez_compressed(path,historical_raw=old_raw,current_D10_raw=raw,historical_decoded=old_actions,current_D10_decoded=actions,
            native_D2_raw=array(native_raw),wrapped_D2_raw=array(wrapped['raw_actions']),native_D2_decoded=array(native),wrapped_D2_decoded=array(wrapped['actions']),
            noise=np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32),instruction=instruction,seed=seed,**current_context)
        row=dict(saved=saved,historical=historical,d2_native_parity=parity,arrays=bound_file(path),
            evidence_scope='Full current output arrays, prefix K/V, normalized state, transformed state, seed/noise; historical output arrays only, historical K/V unavailable')
        checks.append(row);write_json(directory/'gate_progress.json',dict(checks=checks))
    passed=bool(checks) and all(c['historical']['raw']['passed'] and c['historical']['decoded']['passed'] and c['historical']['first5_gripper_sign_equal'] and c['historical']['observed_calls']['ae_layer_calls']==180 and c['d2_native_parity']['passed'] for c in checks)
    policy.gate_passed=passed;policy.protocol['legacy_gate_passed']=passed
    result=dict(passed=passed,checks=checks,quality_requests=0,required_gpu_index=3,required_gpu_uuid=UUID,no_video=True,
                strict_atol=1e-5,strict_rtol=0,historical585_reused_without_rerun=True)
    write_json(directory/'gate.json',result);return result


def serve(a,m):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=UUID:raise RuntimeError('Only exact original GPU3 UUID allowed')
    if Path(a.service_receipt).exists():raise FileExistsError(a.service_receipt)
    receipt=dict(status='constructing',manifest_hash=m['manifest_hash'],server_pid=os.getpid(),required_gpu_uuid=UUID,no_video=True)
    write_json(a.service_receipt,receipt)
    try:
        initial=sample_gpu3()
        if not initial['idle']:raise RuntimeError('GPU3 no longer idle before constructor')
        policy=V14D2Policy(m);cpu=audit_cpu_backend(policy.backend)
        if time.time()>=a.deadline_unix:raise TimeoutError('Deadline reached before CUDA allocation')
        if cpu['cuda_initialized_before_guard']:raise RuntimeError('CUDA initialized before final guard')
        final=sample_gpu3()
        if not final['idle']:raise RuntimeError('GPU3 no longer idle before first CUDA allocation')
        receipt['final_preCUDA_guard']=dict(sample=final,cpu=cpu,passed=True)
        gate=strict_preflight(policy,Path(a.output)/'strict_gate');receipt['gate']=gate;write_json(a.service_receipt,receipt)
        if not gate['passed']:raise RuntimeError('SameGPU3 strict D10/D2 parity failed; quality prohibited')
        policy.protocol.update(server_pid=os.getpid(),gpu_uuid=UUID,required_gpu_index=3,no_video=True,strict_same_gpu_gate=True)
        receipt.update(status='ready',protocol=policy.protocol);write_json(a.service_receipt,receipt)
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);server.bind(('127.0.0.1',a.port));server.listen(4);server.settimeout(2)
            while time.time()<a.deadline_unix:
                try:connection,_=server.accept()
                except socket.timeout:continue
                stop=False
                with connection:
                    try:
                        size=struct.unpack('!I',receive(connection,4))[0]
                        if size>128*1024*1024:raise ValueError('Oversized RPC')
                        request=pickle.loads(receive(connection,size));op=request.pop('op')
                        if op=='health':result=dict(protocol=policy.protocol)
                        elif op=='shutdown':result={};stop=True
                        elif op=='reset':result=policy.reset()
                        elif op=='predict':
                            if request.pop('family',None)!='libero_transfer' or request.pop('environment',None) not in ('libero','libero_plus'):raise ValueError('Native Plus family required')
                            request.pop('mother_suite',None);request['raw']=native_libero_raw(request.pop('observation'));result=policy.predict(**request)
                        else:raise ValueError('Unknown RPC')
                        result=json_safe(dict(result,ok=True,manifest_hash=m['manifest_hash'],protocol_hash=m['manifest_hash']))
                    except Exception:result=dict(ok=False,error=traceback.format_exc(),failure_code='TECHNICAL')
                    payload=pickle.dumps(result,protocol=4);connection.sendall(struct.pack('!I',len(payload))+payload)
                receipt['counters']=json_safe(policy.counters);write_json(a.service_receipt,receipt)
                if stop:break
        receipt['status']='complete'
    except Exception:receipt.update(status='technical_failure',error=traceback.format_exc());raise
    finally:receipt['completed_unix']=time.time();write_json(a.service_receipt,receipt)


def health(a,m,pid):
    result=rpc(a.port,dict(op='health'),timeout=3)['protocol']
    if result.get('manifest_hash')!=m['manifest_hash'] or result.get('server_pid')!=pid or result.get('gpu_uuid')!=UUID or not result.get('strict_same_gpu_gate') or not result.get('legacy_gate_passed'):raise RuntimeError('Owned strict GPU3 service identity mismatch')
    return result


def run_queue(a,m,receipt,receipt_path,limit):
    new=0
    for job in m['jobs']:
        if completed(a.output,job,m['manifest_hash'])[0]:continue
        if time.time()>=a.deadline_unix or new>=limit:break
        base=Path(a.output)/job['group']/('case%04d'%job['source_index'])/'v14_d2';base.mkdir(parents=True,exist_ok=True)
        indices=[int(p.name[7:]) for p in base.glob('attempt*') if p.is_dir() and p.name[7:].isdigit()]
        destination=base/('attempt%03d'%(max(indices,default=-1)+1))
        env=isolated_child_environment(job['case'],Path(a.output)/('config_'+job['group']),3,UUID)
        interpreter=PLUS/'.venv/bin/python' if job['group']=='perturbed350' else ROOT/'policies/libero-eval-v21/.venv/bin/python'
        command=[str(interpreter),'-m',__spec__.name,'--stage','episode','--output',str(Path(a.output).resolve()),'--attempt-output',str(destination.resolve()),'--group',job['group'],'--source-index',str(job['source_index']),'--port',str(a.port),'--deadline-unix',str(a.deadline_unix),'--launch']
        with (base/(destination.name+'.log')).open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            row=dict(pid=process.pid,process_identity=process_identity(process.pid),group=job['group'],source_index=job['source_index'],output=str(destination),command=command)
            receipt['active_episode']=row;write_json(receipt_path,receipt)
            # Child checks the frozen deadline and closes MuJoCo naturally.
            code=process.wait()
        receipt.pop('active_episode',None);row.update(exit_code=code,reaped=True);receipt.setdefault('episodes',[]).append(row);write_json(receipt_path,receipt);new+=1
        if code:raise RuntimeError('Technical episode preserved; queue stopped, no policy zero')
    return new


def watch(a,m):
    output=Path(a.output);receipt_path=output/'watcher.json'
    if receipt_path.exists():raise FileExistsError('Use a new watcher receipt/output registration')
    if not time.time()<a.deadline_unix<=m['deadline_unix']:raise ValueError('Fresh frozen deadline required')
    receipt=dict(status='waiting_original_gpu3_idle',owner=process_identity(os.getpid()),manifest_hash=m['manifest_hash'],
        watcher_source_sha256=sha(__file__),required_gpu_index=3,required_gpu_uuid=UUID,no_video=True,
        deadline_unix=a.deadline_unix,check_interval_s=30,service_pid=None,episodes=[])
    write_json(receipt_path,receipt);service=None
    try:
        while time.time()<a.deadline_unix:
            sample=sample_gpu3()
            with (output/'gpu3_idle_samples.jsonl').open('a') as stream:stream.write(json.dumps(sample)+'\n')
            if sample['idle']:
                final=sample_gpu3()
                if final['idle']:break
            time.sleep(min(30,max(0,a.deadline_unix-time.time())))
        else:receipt['status']='deadline_waiting_gpu3';return
        receipt['selected_idle_sample']=final
        command=[sys.executable,'-m',__spec__.name,'--stage','serve','--output',str(output.resolve()),'--service-receipt',str((output/'service.json').resolve()),'--port',str(a.port),'--deadline-unix',str(a.deadline_unix),'--launch']
        with (output/'service.log').open('x') as log:
            service=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID,JAX_PLATFORMS='cpu'),stdout=log,stderr=subprocess.STDOUT)
            receipt.update(status='loading_owned_gpu3_service',service_pid=service.pid,service_command=command);write_json(receipt_path,receipt)
            while time.time()<a.deadline_unix:
                if service.poll() is not None:raise RuntimeError('Service startup failed; inspect service.json/strict_gate; quality not dispatched')
                try:receipt['service_health']=health(a,m,service.pid);break
                except (OSError,ConnectionError,TimeoutError):time.sleep(2)
            else:raise TimeoutError('Service startup deadline')
            receipt['status']='executing_one_case_for_review';write_json(receipt_path,receipt);run_queue(a,m,receipt,receipt_path,1)
            done=[r for job in m['jobs'] for r in completed(output,job,m['manifest_hash'])[0]]
            if not done:raise RuntimeError('First quality did not complete before deadline')
            first=done[0];receipt.update(status='awaiting_first_case_review',first_completed=first,
                continuation_required=dict(manifest_hash=m['manifest_hash'],first_result_sha256=first['sha256'],continue_remaining=True))
            write_json(receipt_path,receipt)
            marker=output/'continue.json'
            while time.time()<a.deadline_unix:
                if service.poll() is not None:raise RuntimeError('Owned service exited while waiting review')
                if marker.exists():
                    authorization=json.loads(marker.read_text())
                    if authorization!=receipt['continuation_required']:raise ValueError('Continue receipt identity mismatch')
                    receipt['continuation']=bound_file(marker);break
                time.sleep(min(5,max(0,a.deadline_unix-time.time())))
            else:receipt['status']='deadline_waiting_review';return
            receipt['status']='executing_remaining_D2_only';write_json(receipt_path,receipt);run_queue(a,m,receipt,receipt_path,195)
            receipt['status']='queue_scanned'
    except Exception:receipt.update(status='technical_failure',error=traceback.format_exc());raise
    finally:
        if service is not None and service.poll() is None:
            try:health(a,m,service.pid);receipt['shutdown_response']=rpc(a.port,dict(op='shutdown'),timeout=5)
            except Exception:receipt['shutdown_note']='RPC unavailable; service self-exits at frozen deadline, no kill sent'
            # With an established listener shutdown is immediate. During model
            # construction the child retains its own deadline and is reaped.
            service.wait();receipt['service_exit_code']=service.returncode;receipt['service_reaped']=True
        elif service is not None:service.wait();receipt['service_reaped']=True
        receipt['finished_unix']=time.time();write_json(receipt_path,receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,choices=('register','watch','serve','episode'))
    p.add_argument('--output',required=True);p.add_argument('--previous-manifest');p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--port',type=int,default=19163);p.add_argument('--service-receipt');p.add_argument('--attempt-output');p.add_argument('--group');p.add_argument('--source-index',type=int);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.stage=='register':
        if not a.previous_manifest:p.error('--previous-manifest required')
        m=register(a.previous_manifest,a.output,a.deadline_unix);print(json.dumps(dict(manifest_hash=m['manifest_hash'],target=195,no_video=True)));return
    identity=(a.group,a.source_index) if a.stage=='episode' else None;m=verify(Path(a.output)/'manifest.json',identity)
    if not a.launch:print(json.dumps(dict(dry_run=True,stage=a.stage,manifest_hash=m['manifest_hash'],gpu_uuid=UUID,no_video=True)));return
    if a.deadline_unix>m['deadline_unix']:raise ValueError('Deadline exceeds manifest')
    if a.stage=='watch':watch(a,m)
    elif a.stage=='serve':serve(a,m)
    else:
        job=next(j for j in m['jobs'] if (j['group'],j['source_index'])==identity)
        args=argparse.Namespace(output=a.attempt_output,port=a.port,gpu=3,gpu_uuid=UUID,deadline_unix=a.deadline_unix)
        result=episode(args,job,m);raise SystemExit(0 if result['status']=='complete' else 1)


if __name__=='__main__':main()

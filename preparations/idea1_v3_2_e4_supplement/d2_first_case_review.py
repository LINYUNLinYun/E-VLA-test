"""Read-only first-case audit followed by exact, exclusive continuation marker.

Observes one preregistered watcher process. Never samples a GPU, constructs a
model/environment, changes results, or signals another process.
"""
import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path
import time
import traceback
import numpy as np
from .native_envs import digest,sha,write_json
from .parallel_continue import process_identity
from .evaluation import protocol_identity
from .run_plus_v14_d2 import KEYS,initial_comparison,value_hash
from .policy_server import native_libero_raw
from preparations.idea1_v2_1.runtime_client import input_digest,value_digest
from preparations.idea1_v3_2.transfer_client import rpc

IDENTITY='f35a09c66577443fcc19d8c7fea0127003ce405267afcd74898dd3c0ad3273c4'
UUID='GPU-555cde36-5e41-bc44-0c69-ea47358fd703'


def require(condition,message):
    if not condition:raise ValueError(message)


def finite(value,shape,name):
    result=np.asarray(value,float)
    require(result.shape==shape and np.isfinite(result).all(),'Invalid '+name)
    return result


def jsonl(path):
    with Path(path).open() as stream:return [json.loads(line) for line in stream if line.strip()]


def bound(path):return dict(path=str(Path(path).resolve()),sha256=sha(path))


def audit_episode(path,job,manifest):
    path=Path(path).resolve();directory=path.parent;result=json.loads(path.read_text())
    require(result.get('status')=='complete' and type(result.get('success')) is bool,'First case must be scientifically complete')
    for key,expected in dict(manifest_hash=manifest['manifest_hash'],case_hash=job['case_hash'],mode='v14_d2',gpu_uuid=UUID,
                             group=job['group'],source_index=job['source_index'],H=50,R=5,D=2).items():
        require(result.get(key)==expected,'First-case identity mismatch: '+key)
    require(result['case']==job['case'] and digest(result['case'])==job['case_hash'],'Case data mismatch')
    require(result['paired_results']==job['paired_results'],'Result paired results differ from manifest')
    evidence=[job['initial_observation'],job['first_input']]+[dict(path=r['path'],sha256=r['sha256']) for r in job['paired_results']]
    for entry in evidence:require(sha(entry['path'])==entry['sha256'],'First-job bound evidence changed: '+entry['path'])
    require(job['video'] is False and not list(directory.rglob('*.mp4')) and 'video' not in result,'New case must contain no MP4/video')
    info=result['native_info'];require(info['native_hz']==20 and info['settle_steps']==5 and info['max_steps']==600,'Native cadence changed')
    with np.load(directory/'initial_native_observation.npz',allow_pickle=False) as source:
        observation={k:source[k] for k in KEYS};instruction=str(source['instruction'])
    require(instruction==info['instruction'],'Initial effective BDDL instruction mismatch')
    actual_initial=initial_comparison(observation,instruction,job['initial_observation']['path'])
    require(actual_initial['passed'],'Initial numerical comparison failed')
    full=next(r for r in job['paired_results'] if r['mode']=='full_d10')
    old_result=json.loads(Path(full['path']).read_text())
    require(info['initial_vector_sha256']==old_result['native_info']['initial_vector_sha256'],'Initial native vector hash mismatch')
    stored=json.loads((directory/'initial_comparison.json').read_text())
    require(stored==result['initial_comparison'] and stored.get('passed') and stored.get('initial_vector_sha256_equal'),'Stored initial gate inconsistent')
    for key in KEYS:
        require(stored['fields'][key]==actual_initial['fields'][key],'Stored initial field audit differs: '+key)
    with np.load(directory/'initial_physical_state.npz',allow_pickle=False) as source:
        state=source['state'];require(np.isfinite(state).all() and state.ndim==1 and state.size>0,'Invalid saved physical state')
        require(value_hash(state)==result['initial_physical_state_sha256'],'Physical state hash mismatch')
    infer=jsonl(directory/'infer.jsonl');telemetry=jsonl(directory/'telemetry.jsonl');steps=result['steps'];requests=result['requests']
    require(type(steps) is int and 1<=steps<=600 and type(requests) is int and requests==(steps+4)//5,'Step/request prefix counts differ')
    require(len(infer)==requests and len(telemetry)==steps+1,'Incomplete infer/XYZ telemetry')
    require([r['step'] for r in telemetry]==list(range(steps+1)),'Telemetry steps not contiguous')
    for item in telemetry:
        finite(item['actual']['eef_xyz_world'],(3,),'actual XYZ')
        finite(item['actual']['eef_quaternion_xyzw'],(4,),'actual quaternion')
        finite(item['actual']['gripper_qpos'],(2,),'actual gripper')
    require(telemetry[0]['executed_command'] is None,'Step0 must have no command')
    for name,key in [('eef_xyz_world','robot0_eef_pos'),('eef_quaternion_xyzw','robot0_eef_quat'),('gripper_qpos','robot0_gripper_qpos')]:
        require(np.array_equal(np.asarray(telemetry[0]['actual'][name]),observation[key]),'Initial XYZ/physical telemetry differs')
    decoded=[]
    for index,row in enumerate(infer):
        response=row['response'];seed=int(digest(['plus-evaluation-v1',job['case_hash'],index])[:8],16)
        require(row['request']==index and row['step']==index*5 and row['seed']==seed and response['seed']==seed,'Request/step/seed rule mismatch')
        noise=np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32)
        require(response['noise_hash']==value_digest(noise) and response['noise_used_in_inference'] is True,'Noise hash/inference mismatch')
        require(response['manifest_hash']==manifest['manifest_hash'] and response['protocol_hash']==manifest['manifest_hash'] and response['mode']=='v14_d2' and response['category']=='quality' and response['execute_length']==5,'RPC route identity differs')
        actions=finite(row['actions'],(50,7),'decoded50x7');finite(response['raw_actions'],(50,32),'raw50x32');decoded.append(actions)
        calls=response['observed_calls']
        for key,value in dict(vision_encoder_calls=1,vlm_prefix_calls=1,ae_layer_calls=36,learned_head_calls=0,verify_calls=0,constructor_full_record_calls=0,ae_euler_updates=2).items():
            require(calls.get(key)==value,'Actual route count differs: '+key)
        timing=response['timing'];require(timing['actual_NFE']==2 and np.array_equal(timing['nodes'],[1.,.5,0.]),'Uniform D2 integration differs')
        route=response['route']
        for key,value in dict(mode='v14_d2',fresh_prefix=True,full_vision=True,skip_vlm=[3,4,9,11],kv_cross_request=False,fallback=False,verify=False).items():require(route.get(key)==value,'Route contract differs: '+key)
        require(set(row['input_hashes'])==set(KEYS) and all(isinstance(v,str) and len(v)==64 for v in row['input_hashes'].values()),'Input hash schema differs')
        if index==0:
            require(row['input_hashes']=={k:value_hash(v) for k,v in observation.items()},'First saved observation hash differs')
            require(response['input_hash']==input_digest(native_libero_raw(observation),instruction),'First server processed input hash differs')
    successes=[]
    for index,row in enumerate(telemetry[1:]):
        audit=row['executed_command'];action=decoded[index//5][index%5];clipped=np.clip(action,-1,1)
        require(np.array_equal(action,finite(audit['raw_action'],(7,),'executed raw command')),'Executed action is not exact R5 prefix')
        require(np.array_equal(clipped,finite(audit['native_action'],(7,),'native clipped command')),'Native clipping differs')
        require(audit['clipped_components']==np.flatnonzero(action!=clipped).tolist(),'Clipped component indices differ')
        require(type(audit['success']) is bool,'Native oracle value must be bool');successes.append(audit['success'])
    require(not any(successes[:-1]) and successes[-1]==result['success'],'Native success oracle/early stop differs')
    require(result['termination']==('native_success_oracle' if result['success'] else 'step_cap'),'Termination/oracle mismatch')
    require(result['success'] or steps==600,'Policy failure must reach native600 step cap')
    inputs=[path,directory/'infer.jsonl',directory/'telemetry.jsonl',directory/'initial_native_observation.npz',directory/'initial_physical_state.npz',directory/'initial_comparison.json']
    return dict(passed=True,success=result['success'],steps=steps,requests=requests,actual_XYZ_rows=len(telemetry),inputs=[bound(p) for p in inputs]+evidence,
        limits='Later full RGB observations were not saved by frozen runner; their declared hashes are schema-checked. First raw native input and all request seed/noise, outputs, route counters, and executed telemetry are checked.')


def audit_service(root,watcher,manifest,port):
    service_path=Path(root)/'service.json';service=json.loads(service_path.read_text());pid=watcher['service_pid']
    require(service['server_pid']==pid and service['manifest_hash']==manifest['manifest_hash'] and service['status']=='ready','Service receipt identity/status mismatch')
    protocol=rpc(port,dict(op='health'),timeout=10)['protocol']
    for key,value in dict(server_pid=pid,manifest_hash=manifest['manifest_hash'],gpu_uuid=UUID,required_gpu_index=3,no_video=True,
                          strict_same_gpu_gate=True,legacy_gate_passed=True,D=2,H=50,R=5,modes=['v14_d2']).items():require(protocol.get(key)==value,'Live service differs: '+key)
    require(service['protocol']==protocol,'Live service protocol differs from saved receipt')
    guard=service['final_preCUDA_guard'];require(guard['passed'] and guard['sample']['idle'] and guard['sample']['uuid']==UUID and guard['sample']['index']==3,'Final preCUDA GPU3 guard missing')
    gate=service['gate'];require(gate['passed'] and gate['strict_atol']==1e-5 and gate['strict_rtol']==0 and gate['quality_requests']==0 and gate['required_gpu_uuid']==UUID and gate['no_video'],'Strict gate receipt differs')
    require({c['saved']['mode'] for c in gate['checks']}=={'full_d10','v14_d10'},'Both strict historical routes required')
    inputs=[bound(service_path)]
    for check in gate['checks']:
        require(sha(check['arrays']['path'])==check['arrays']['sha256'],'Strict gate arrays changed');inputs.append(check['arrays'])
        with np.load(check['arrays']['path'],allow_pickle=False) as data:
            for actual,expected in [('current_D10_raw','historical_raw'),('current_D10_decoded','historical_decoded'),('native_D2_raw','wrapped_D2_raw'),('native_D2_decoded','wrapped_D2_decoded')]:
                a,b=data[actual],data[expected];shape=(50,32) if actual.endswith('raw') else (50,7)
                require(a.shape==b.shape==shape and np.isfinite(a).all() and np.isfinite(b).all() and np.allclose(a,b,atol=1e-5,rtol=0),'Saved strict numerical gate failed: '+actual)
        require(check['historical']['raw']['passed'] and check['historical']['decoded']['passed'] and check['d2_native_parity']['passed'],'Stored strict gate checks failed')
    return dict(passed=True,server_pid=pid,protocol=protocol,inputs=inputs)


def review(root,watcher,expected_manifest,port):
    root=Path(root).resolve();manifest=protocol_identity(root/'manifest.json')
    require(manifest['manifest_hash']==expected_manifest==IDENTITY,'Fixed preregistered manifest differs')
    require(watcher['manifest_hash']==expected_manifest and watcher['status']=='awaiting_first_case_review','Watcher not at fixed first-case review')
    for name,expected in manifest['source_files'].items():
        path=Path(name);path=path if path.is_absolute() else Path(__file__).resolve().parents[2]/path
        require(sha(path)==expected,'Frozen source changed: '+str(path))
    first=watcher['first_completed'];job=manifest['jobs'][0]
    require(first['group']==job['group'] and first['source_index']==job['source_index'],'First case selection differs')
    path=Path(first['path']).resolve();require(root in path.parents and path.name=='result.json','First result escaped experiment directory')
    require(sha(path)==first['sha256'],'First result SHA mismatch')
    expected=dict(manifest_hash=expected_manifest,first_result_sha256=first['sha256'],continue_remaining=True)
    require(watcher['continuation_required']==expected,'Continuation object differs')
    service=audit_service(root,watcher,manifest,port);episode=audit_episode(path,job,manifest)
    return dict(passed=True,continuation_required=expected,episode=episode,service=service,
                inputs=[bound(root/'manifest.json'),bound(root/'watcher.json')],source_sha256=sha(__file__))


def write_marker(path,expected):
    path=Path(path)
    temporary=path.with_name(path.name+'.review-%d.temporary'%os.getpid())
    try:
        with temporary.open('x') as stream:json.dump(expected,stream,indent=2);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
        return 'created_exact_continuation'
    except FileExistsError:
        require(json.loads(path.read_text())==expected,'Existing continuation marker differs; never overwritten')
        return 'existing_exact_continuation_verified'
    finally:
        if temporary.exists():temporary.unlink()


def verify_watcher_command(current,root):
    argv=shlex.split(current['argv'])
    def option(name):
        require(argv.count(name)==1 and argv.index(name)+1<len(argv),'Watcher missing/duplicate option '+name)
        return argv[argv.index(name)+1]
    require(option('-m')=='preparations.idea1_v3_2_e4_supplement.plus_v14_d2_gpu3_v2','Watcher module differs')
    require(option('--stage')=='watch','Watcher stage differs')
    output=Path(option('--output'));output=output if output.is_absolute() else Path(current['cwd'])/output
    require(output.resolve()==Path(root).resolve(),'Watcher output directory differs')


def watch(a):
    root=Path(a.root).resolve();receipt_path=Path(a.receipt)
    require(not receipt_path.exists(),'New review receipt required')
    receipt=dict(status='watching_fixed_watcher',watcher_pid=a.watcher_pid,watcher_start_ticks=str(a.watcher_start_ticks),
        manifest_hash=a.manifest_hash,deadline_unix=a.deadline_unix,source_sha256=sha(__file__),root=str(root),poll_interval_s=30)
    write_json(receipt_path,receipt)
    try:
        while time.time()<a.deadline_unix:
            current=process_identity(a.watcher_pid)
            if current is None or current['state']=='Z' or current['start_ticks']!=str(a.watcher_start_ticks):receipt['status']='watcher_terminated';return
            verify_watcher_command(current,root)
            watcher=json.loads((root/'watcher.json').read_text());require(watcher['owner']['pid']==a.watcher_pid and watcher['owner']['start_ticks']==str(a.watcher_start_ticks),'Watcher owner differs')
            if watcher['status']=='awaiting_first_case_review':
                receipt['review']=review(root,watcher,a.manifest_hash,a.port)
                # Recheck immutable result and owner immediately before marker.
                now=process_identity(a.watcher_pid);require(now is not None and now['start_ticks']==str(a.watcher_start_ticks),'Watcher ended during review')
                require(time.time()<a.deadline_unix,'Deadline reached during review')
                require(sha(watcher['first_completed']['path'])==watcher['first_completed']['sha256'],'First result changed during review')
                receipt['marker']=write_marker(root/'continue.json',watcher['continuation_required']);receipt['status']='approved_exact_continuation';return
            if watcher['status'] in ('technical_failure','queue_scanned','deadline_waiting_gpu3','deadline_waiting_review'):receipt['status']='watcher_terminal_'+watcher['status'];return
            time.sleep(min(30,max(0,a.deadline_unix-time.time())))
        receipt['status']='deadline_reached'
    except Exception:receipt.update(status='technical_review_failure_no_marker',error=traceback.format_exc())
    finally:receipt['finished_unix']=time.time();write_json(receipt_path,receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--receipt',required=True)
    p.add_argument('--watcher-pid',type=int,required=True);p.add_argument('--watcher-start-ticks',required=True)
    p.add_argument('--manifest-hash',default=IDENTITY);p.add_argument('--port',type=int,default=19163);p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--launch',action='store_true');a=p.parse_args()
    if not a.launch:print(json.dumps(dict(dry_run=True,**vars(a))));return
    watch(a)


if __name__=='__main__':main()

"""Two exclusive B1 shards: legacy initialization and bitwise16 gate each."""
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
from .policy_server import native_libero_raw
from .policy_plus_mirror import audit_cpu_backend
from .policy_plus_v14_d2 import V14D2Policy,array
from .plus_v14_d2 import completed,bound_file
from .parallel_continue import process_identity
from .libero_plus_env import PLUS,isolated_child_environment
from .run_plus_v14_d2 import episode
from . import d2_cross_device_probe as diagnostic
from .d2_legacy_history_probe import BANK_SHA,ORDER
from preparations.idea1_v3_2.transfer_policy import TransferPolicy,RawTritonBackend,route_observations,observed_counts,json_safe
from preparations.idea1_v3_1.server import receive
from preparations.idea1_v3_2.transfer_client import rpc

KIND='native_LIBERO_Plus_half_V14_D2_quality_v3_legacy_bitwise16'
UUID=diagnostic.CARDS[0]


def register(base_path,probe_path,history_path,output,deadline):
    base=protocol_identity(base_path);probe=protocol_identity(probe_path);history=protocol_identity(history_path)
    if base['manifest_hash']!='242190d617125f75f960a70a12cbeb3b53b7fd4227e6a93dbe1c6ffbdeca7479' or probe['manifest_hash']!='00bc9f0cad3b4e8b3a82a674859493add471fe1dbcd5f69ae69f5acb0d17d65f' or history['manifest_hash']!='89d069f31a78466b06bab518a2f2aef75ec318e3ee5aa1541126f430a14b7eb5':raise ValueError('Exact195/diagnostic/history identities required')
    if not time.time()<deadline<=time.time()+12*3600:raise ValueError('Fresh quality deadline at most12h required')
    history_receipt=Path(history_path).parent/'gpu0/receipt.json';h=json.loads(history_receipt.read_text())
    if h['status']!='complete' or not all(c['versus']['historical'][key]['bitwise_equal'] for c in h['uncaptured'] for key in ('raw','decoded')):raise ValueError('Historical branch exact reproduction proof required')
    m={k:v for k,v in base.items() if k not in ('manifest_hash','created_unix')}
    m.update(kind=KIND,created_unix=time.time(),deadline_unix=deadline,assigned_gpus={str(k):v for k,v in diagnostic.CARDS.items()},
        shard_rule='Global frozen job ordinal modulo2 equals physical GPU index0/1; independent B1 services, never batched',
        previous_D2_registration=bound_file(base_path),cross_device_probe_manifest=bound_file(probe_path),legacy_history_manifest=bound_file(history_path),
        legacy_history_receipt=bound_file(history_receipt),legacy_bank=history['legacy_bank'],references=probe['references'],
        jobs=[dict(j,video=False) for j in base['jobs']],video_source_indices=[],video_policy='All new quality195 data/XYZ only, no MP4; old585 reused without re-inference',
        startup_gate=dict(initialization='Exact unchanged TransferPolicy.legacy_preflight historical same-bank six reference/candidate routes in original order, before any other model request',
            historical16='Each fixed reference repeated2; all raw50x32 and decoded50x7 bytes must equal historical and each other',D2='Same raw32 native/outer-wrapper full bytes equal; actual18AE×2=36, NFE2,nodes1,.5,0',
            tolerance='No numeric tolerance, padding exemption, case removal, fallback or card/configuration search'),
        repeat_count=2,cross_gpu_limitation='FixedGPU0/1 each independently require exact same-output legacy+16ref proof before quality; arbitrary initialization is not equivalent.',
        staged_execution='Per shard one complete new D2 case then runtime_gpu{card}/continue.json bound to first-result SHA; root de-duplicates old waiting queue before launch')
    sources=dict(base['source_files']);sources.update(probe['source_files']);sources.update(history['source_files']);sources[str(Path(__file__).resolve())]=sha(__file__);m['source_files']=sources
    m['manifest_hash']=digest(m);output=Path(output);output.mkdir(parents=True,exist_ok=False);write_json(output/'manifest.json',m);return m


def verify(path,job_identity=None):
    m=protocol_identity(path)
    if m.get('kind')!=KIND or m.get('assigned_gpus')!={str(k):v for k,v in diagnostic.CARDS.items()} or any(j['video'] for j in m['jobs']):raise ValueError('Fixed two-card data-only quality-v3 manifest required')
    for name,expected in m['source_files'].items():
        source=Path(name);source=source if source.is_absolute() else ROOT/source
        if sha(source)!=expected:raise ValueError('Frozen source changed: '+str(source))
    if job_identity is None:
        evidence=[m[k] for k in ('previous_D2_registration','cross_device_probe_manifest','legacy_history_manifest','legacy_history_receipt','legacy_bank','half_manifest','parent_manifest','audit_ledger')]+m['bound_files']
        evidence += [r[k] for r in m['references'] for k in ('result','observation','infer')]
    else:
        job=next(j for j in m['jobs'] if (j['group'],j['source_index'])==job_identity)
        evidence=[job['initial_observation'],job['first_input']]+[dict(path=r['path'],sha256=r['sha256']) for r in job['paired_results']]
    for item in evidence:
        if sha(item['path'])!=item['sha256']:raise ValueError('Frozen evidence changed: '+item['path'])
    return m


def all_equal(pairs):return all(pair[key]['bitwise_equal'] for pair in pairs for key in ('raw','decoded'))


def strict_gate(backend,m,directory,card=0):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);production=TransferPolicy(m,m['manifest_hash'],backend=backend)
    diagnostic.deadline_check(m['deadline_unix']);legacy=production.legacy_preflight(m['legacy_bank']['path']);write_json(directory/'legacy_verbatim.json',json_safe(legacy))
    if not legacy['passed'] or [c['mode'] for c in legacy['checks']]!=list(ORDER):raise RuntimeError('Exact unchanged legacy initialization failed')
    checks=[];inputs=[]
    for index,ref in enumerate(m['references']):
        with np.load(ref['observation']['path'],allow_pickle=False) as source:
            obs={k:source[k] for k in source.files if k!='instruction'};instruction=str(source['instruction'])
        with Path(ref['infer']['path']).open() as stream:old=json.loads(stream.readline())
        raw=native_libero_raw(obs);seed=old['seed'];noise=np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32)
        if diagnostic.input_digest(raw,instruction)!=old['response']['input_hash'] or diagnostic.value_digest(noise)!=old['response']['noise_hash']:raise ValueError('Historical input/noise differs')
        responses=[];pairs=[]
        for repeat in range(2):
            diagnostic.deadline_check(m['deadline_unix']);response=production.predict(ref['mode'],raw,instruction,seed,category='preflight');responses.append(response)
            path=directory/('reference%02d_repeat%d.npz'%(index,repeat))
            np.savez_compressed(path,raw=array(response['raw_actions']),decoded=array(response['actions']),historical_raw=array(old['response']['raw_actions']),historical_decoded=array(old['actions']))
            pairs.append(dict(raw=diagnostic.comparison(response['raw_actions'],old['response']['raw_actions']),decoded=diagnostic.comparison(response['actions'],old['actions'])))
            inputs.append(bound_file(path))
        repeat_pair=dict(raw=diagnostic.comparison(responses[0]['raw_actions'],responses[1]['raw_actions']),decoded=diagnostic.comparison(responses[0]['actions'],responses[1]['actions']))
        check=dict(reference=ref,repeats_vs_history=pairs,repeat_vs_repeat=repeat_pair,observed_calls=[r['observed_calls'] for r in responses],passed=all_equal(pairs+[repeat_pair]))
        checks.append(check);write_json(directory/'D10_progress.json',checks)
        if not check['passed']:raise RuntimeError('Strict historical16 bytewise gate failed; no quality')
    policy=V14D2Policy(m,backend=backend);d2_checks=[]
    for ref in m['references'][:2]:
        with np.load(ref['observation']['path'],allow_pickle=False) as source:
            obs={k:source[k] for k in source.files if k!='instruction'};instruction=str(source['instruction'])
        with Path(ref['infer']['path']).open() as stream:seed=json.loads(stream.readline())['seed']
        raw=native_libero_raw(obs);diagnostic.deadline_check(m['deadline_unix'])
        with route_observations() as observed:
            context,_=RawTritonBackend.prepare(backend,raw,instruction,(3,4,9,11),encode=True)
            actions,model_raw,costs=RawTritonBackend.sample(backend,context,seed,2)
        diagnostic.deadline_check(m['deadline_unix']);wrapped=policy.predict('v14_d2',raw,instruction,seed,category='preflight');calls=observed_counts(observed)
        pair=dict(raw=diagnostic.comparison(model_raw,wrapped['raw_actions']),decoded=diagnostic.comparison(actions,wrapped['actions']))
        expected=dict(vision_encoder_calls=1,vlm_prefix_calls=1,ae_layer_calls=36,learned_head_calls=0,verify_calls=0,constructor_full_record_calls=0)
        okay=all_equal([pair]) and all(calls[k]==v for k,v in expected.items()) and costs['actual_NFE']==2 and costs['nodes']==[1.,.5,0.]
        path=directory/('D2_'+ref['mode']+'.npz');np.savez_compressed(path,native_raw=array(model_raw),native_decoded=array(actions),wrapped_raw=array(wrapped['raw_actions']),wrapped_decoded=array(wrapped['actions']));inputs.append(bound_file(path))
        d2_checks.append(dict(mode=ref['mode'],comparison=pair,native_calls=calls,wrapper_calls=wrapped['observed_calls'],costs=costs,passed=okay))
        write_json(directory/'D2_progress.json',d2_checks)
        if not okay:raise RuntimeError('Strict D2 bitwise/actual-calls gate failed; no quality')
    gate=dict(passed=True,legacy_verbatim=legacy,D10=checks,D2=d2_checks,arrays=inputs,quality_requests=0,bytewise_only=True,assigned_gpu_index=card,assigned_gpu_uuid=m['assigned_gpus'][str(card)])
    write_json(directory/'gate.json',json_safe(gate));policy.gate_passed=True;policy.protocol.update(legacy_gate_passed=True,strict_bitwise16_gate=True,no_video=True,assigned_gpu_index=card)
    return policy,gate


def serve(a,m):
    uuid=m['assigned_gpus'][str(a.card)]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid:raise ValueError('Explicit assigned UUID required')
    output=Path(a.output)/('runtime_gpu%d'%a.card);path=output/'service.json'
    if path.exists():raise FileExistsError(path)
    receipt=dict(status='CPU_assembly',manifest_hash=m['manifest_hash'],server_pid=os.getpid(),assigned_gpu_uuid=uuid,assigned_gpu_index=a.card,no_video=True,deadline_unix=m['deadline_unix']);write_json(path,receipt)
    try:
        from preparations.idea1_v3_2.residual import require_idle_gpu
        initial=require_idle_gpu();receipt['initial_idle_guard']=dict(raw=initial,validated=diagnostic.validate_idle_guard(initial,a.card))
        backend=RawTritonBackend();cpu=audit_cpu_backend(backend)
        if cpu['cuda_initialized_before_guard']:raise RuntimeError('CUDA allocated before final guard')
        diagnostic.deadline_check(m['deadline_unix']);final=require_idle_gpu();receipt['final_preCUDA_guard']=dict(cpu=cpu,raw=final,validated=diagnostic.validate_idle_guard(final,a.card));write_json(path,receipt)
        policy,gate=strict_gate(backend,m,output/'strict_gate',a.card);policy.protocol.update(server_pid=os.getpid(),gpu_uuid=uuid)
        metadata=diagnostic.metadata(backend);receipt.update(status='ready',gate=gate,protocol=policy.protocol,runtime_metadata={k:v for k,v in metadata.items() if k!='backend_protocol'});write_json(path,json_safe(receipt))
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);server.bind(('127.0.0.1',a.port));server.listen(4);server.settimeout(2)
            while time.time()<m['deadline_unix']:
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
                            if request.pop('family',None)!='libero_transfer' or request.pop('environment',None) not in ('libero','libero_plus'):raise ValueError('Native LIBERO family required')
                            diagnostic.deadline_check(m['deadline_unix']);request.pop('mother_suite',None);request['raw']=native_libero_raw(request.pop('observation'));result=policy.predict(**request)
                        else:raise ValueError('Unknown RPC')
                        result=json_safe(dict(result,ok=True,manifest_hash=m['manifest_hash'],protocol_hash=m['manifest_hash']))
                    except Exception:result=dict(ok=False,failure_code='TECHNICAL',error=traceback.format_exc())
                    payload=pickle.dumps(result,protocol=4);connection.sendall(struct.pack('!I',len(payload))+payload)
                receipt['counters']=json_safe(policy.counters);write_json(path,json_safe(receipt))
                if stop:break
        receipt['status']='complete'
    except Exception:receipt.update(status='technical_failure',error=traceback.format_exc());raise
    finally:receipt['finished_unix']=time.time();write_json(path,json_safe(receipt))


def health(a,m,pid):
    p=rpc(a.port,dict(op='health'),timeout=3)['protocol']
    if p.get('server_pid')!=pid or p.get('manifest_hash')!=m['manifest_hash'] or p.get('gpu_uuid')!=m['assigned_gpus'][str(a.card)] or p.get('assigned_gpu_index')!=a.card or not p.get('strict_bitwise16_gate'):raise RuntimeError('Owned bitwise-gated service identity differs')
    return p


def shard_jobs(m,card):return [job for ordinal,job in enumerate(m['jobs']) if ordinal%2==card]


def queue(a,m,receipt,path,limit):
    count=0
    for job in shard_jobs(m,a.card):
        if completed(a.output,job,m['manifest_hash'])[0]:continue
        if count>=limit or time.time()>=m['deadline_unix']:break
        base=Path(a.output)/job['group']/('case%04d'%job['source_index'])/'v14_d2';base.mkdir(parents=True,exist_ok=True)
        indices=[int(p.name[7:]) for p in base.glob('attempt*') if p.is_dir() and p.name[7:].isdigit()];dest=base/('attempt%03d'%(max(indices,default=-1)+1))
        env=isolated_child_environment(job['case'],Path(a.output)/('config_gpu%d_'%a.card+job['group']),a.card,m['assigned_gpus'][str(a.card)])
        interpreter=PLUS/'.venv/bin/python' if job['group']=='perturbed350' else ROOT/'policies/libero-eval-v21/.venv/bin/python'
        cmd=[str(interpreter),'-m',__spec__.name,'--stage','episode','--output',str(Path(a.output).resolve()),'--attempt-output',str(dest.resolve()),'--group',job['group'],'--source-index',str(job['source_index']),'--port',str(a.port),'--card',str(a.card),'--launch']
        with (base/(dest.name+'.log')).open('x') as log:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT);row=dict(pid=child.pid,identity=process_identity(child.pid),command=cmd,output=str(dest),group=job['group'],source_index=job['source_index'])
            receipt['active_episode']=row;write_json(path,receipt);code=child.wait()
        row.update(exit_code=code,reaped=True);receipt.pop('active_episode',None);receipt['episodes'].append(row);write_json(path,receipt);count+=1
        if code:raise RuntimeError('Technical attempt saved; stop without policy-zero substitution')
    return count


def watch(a,m):
    output=Path(a.output);runtime=output/('runtime_gpu%d'%a.card);path=runtime/'watcher.json';uuid=m['assigned_gpus'][str(a.card)]
    if path.exists() or (runtime/'service.json').exists():raise FileExistsError('New execution receipts required; existing complete cases are otherwise resumable')
    diagnostic.deadline_check(m['deadline_unix']);receipt=dict(status='loading_assigned_gpu',owner=process_identity(os.getpid()),manifest_hash=m['manifest_hash'],assigned_gpu_uuid=uuid,assigned_gpu_index=a.card,shard_cases=len(shard_jobs(m,a.card)),deadline_unix=m['deadline_unix'],no_video=True,episodes=[]);write_json(path,receipt);service=None
    try:
        command=[sys.executable,'-m',__spec__.name,'--stage','serve','--output',str(output.resolve()),'--port',str(a.port),'--card',str(a.card),'--launch']
        with (runtime/'service.log').open('x') as log:
            service=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,JAX_PLATFORMS='cpu'),stdout=log,stderr=subprocess.STDOUT)
            receipt.update(service_pid=service.pid,service_command=command);write_json(path,receipt)
            while time.time()<m['deadline_unix']:
                if service.poll() is not None:raise RuntimeError('Bitwise startup gate failed; no quality')
                try:receipt['service_health']=health(a,m,service.pid);break
                except (OSError,ConnectionError,TimeoutError):time.sleep(2)
            else:raise TimeoutError('Startup deadline')
            receipt['status']='executing_one_case_for_review';write_json(path,receipt);queue(a,m,receipt,path,1)
            done=[r for j in shard_jobs(m,a.card) for r in completed(output,j,m['manifest_hash'])[0]]
            if not done:raise RuntimeError('No first complete quality before deadline')
            first=done[0];receipt.update(status='awaiting_first_case_review',first_completed=first,continuation_required=dict(manifest_hash=m['manifest_hash'],first_result_sha256=first['sha256'],continue_remaining=True));write_json(path,receipt)
            marker=runtime/'continue.json'
            while time.time()<m['deadline_unix']:
                if service.poll() is not None:raise RuntimeError('Service exited while awaiting review')
                if marker.exists():
                    if json.loads(marker.read_text())!=receipt['continuation_required']:raise ValueError('Continuation marker identity differs')
                    receipt['continuation']=bound_file(marker);break
                time.sleep(min(5,max(0,m['deadline_unix']-time.time())))
            else:receipt['status']='deadline_waiting_review';return
            receipt['status']='executing_remaining_D2_only';write_json(path,receipt);queue(a,m,receipt,path,195);receipt['status']='queue_scanned'
    except Exception:receipt.update(status='technical_failure',error=traceback.format_exc());raise
    finally:
        if service is not None and service.poll() is None:
            try:health(a,m,service.pid);receipt['shutdown_response']=rpc(a.port,dict(op='shutdown'),timeout=5)
            except Exception:receipt['shutdown_note']='Own service deadline retained; no external signal or renderer kill'
            service.wait();receipt['service_reaped']=True;receipt['service_exit_code']=service.returncode
        elif service is not None:service.wait();receipt['service_reaped']=True
        receipt['finished_unix']=time.time();write_json(path,receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,choices=('register','watch','serve','episode'));p.add_argument('--output',required=True)
    p.add_argument('--base-manifest');p.add_argument('--probe-manifest');p.add_argument('--history-manifest');p.add_argument('--deadline-unix',type=float)
    p.add_argument('--port',type=int);p.add_argument('--card',type=int,choices=(0,1),default=0);p.add_argument('--attempt-output');p.add_argument('--group');p.add_argument('--source-index',type=int);p.add_argument('--launch',action='store_true');a=p.parse_args();a.port=a.port or 19164+a.card
    if a.stage=='register':
        if not all((a.base_manifest,a.probe_manifest,a.history_manifest,a.deadline_unix)):p.error('Three parent manifests and fresh quality deadline required')
        m=register(a.base_manifest,a.probe_manifest,a.history_manifest,a.output,a.deadline_unix);print(json.dumps(dict(manifest_hash=m['manifest_hash'],target=195,assigned_gpus=m['assigned_gpus'],deadline_unix=m['deadline_unix'])));return
    ident=(a.group,a.source_index) if a.stage=='episode' else None;m=verify(Path(a.output)/'manifest.json',ident)
    if not a.launch:print(json.dumps(dict(dry_run=True,stage=a.stage,manifest_hash=m['manifest_hash'],assigned_gpu_uuid=m['assigned_gpus'][str(a.card)],target=len(shard_jobs(m,a.card)),no_video=True)));return
    if a.stage=='serve':serve(a,m)
    elif a.stage=='watch':watch(a,m)
    else:
        j=next(j for j in shard_jobs(m,a.card) if (j['group'],j['source_index'])==ident);args=argparse.Namespace(output=a.attempt_output,port=a.port,gpu=a.card,gpu_uuid=m['assigned_gpus'][str(a.card)],deadline_unix=m['deadline_unix'])
        value=episode(args,j,m);raise SystemExit(0 if value['status']=='complete' else 1)


if __name__=='__main__':main()

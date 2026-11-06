"""Drain parent GPU0 normally, then complete only frozen58 interrupted odd jobs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from .native_envs import ROOT,digest,sha,write_json
from .evaluation import protocol_identity
from .plus_v14_d2 import completed,bound_file
from .parallel_continue import process_identity
from .libero_plus_env import PLUS,isolated_child_environment
from .run_plus_v14_d2 import episode
from . import plus_v14_d2_quality_v3 as v3
from . import d2_cross_device_probe as diagnostic
from preparations.idea1_v3_2.transfer_client import rpc

PARENT='c823d3ccea915a4cca78a6282b479f290f1cfff7a0c58052d14081a7efad3290'
DEADLINE=1792762731
KIND='D2_GPU0_only_interrupted_odd58_continuation'


def alive(identity):
    current=process_identity(identity['pid'])
    return current is not None and current['state']!='Z' and current['start_ticks']==identity['start_ticks']


def register(parent_path,output):
    parent=v3.verify(parent_path)
    if parent['manifest_hash']!=PARENT or parent['deadline_unix']!=DEADLINE:raise ValueError('Exact c823 parent and original deadline required')
    diagnostic.deadline_check(DEADLINE);root=Path(parent_path).resolve().parent
    watcher=json.loads((root/'runtime_gpu0/watcher.json').read_text())
    if watcher['owner']['pid']!=3837565 or watcher['owner']['start_ticks']!='2512101766':raise ValueError('Expected owned parent GPU0 watcher differs')
    service=process_identity(watcher['service_pid'])
    if service is None:
        service_value=json.loads((root/'runtime_gpu0/service.json').read_text())
        service=dict(pid=watcher['service_pid'],start_ticks=service_value.get('start_ticks','unavailable_already_exited'))
    reused=[];pending=[];technical=[]
    for job in v3.shard_jobs(parent,1):
        good,bad=completed(root,job,PARENT);technical.extend(bound_file(p) for p in bad)
        if good:
            if json.loads(Path(good[0]['path']).read_text()).get('gpu_uuid')!=diagnostic.CARDS[1]:raise ValueError('Frozen old odd result was not executed on parentGPU1')
            reused.extend(good)
        else:pending.append(job)
    if len(reused)!=39 or len(pending)!=58:raise ValueError('Expected exact39 valid old odd results and58 pending jobs')
    if not any(j['group']=='perturbed350' and j['source_index']==154 for j in pending):raise ValueError('Interrupted case154 must remain pending')
    m={k:v for k,v in parent.items() if k not in ('manifest_hash','created_unix')}
    m.update(kind=KIND,parent_quality_manifest=bound_file(parent_path),parent_quality_manifest_hash=PARENT,parent_results_root=str(root),
        parent_gpu0_watcher=watcher['owner'],parent_gpu0_service=service,parent_gpu0_watcher_path=str(root/'runtime_gpu0/watcher.json'),
        original_global_jobs=parent['jobs'],jobs=pending,condition_count=58,perturbed_case_count=sum(j['group']=='perturbed350' for j in pending),original_case_count=sum(j['group']=='original40' for j in pending),
        parent_gpu1_reused_complete=reused,parent_gpu1_preserved_technical_attempts=technical,
        assigned_gpu_index=0,assigned_gpu_uuid=diagnostic.CARDS[0],assigned_gpus={'0':diagnostic.CARDS[0]},forbidden_to_scan_or_use_gpu_indices=[1,2,3],
        shard_rule='New unsharded queue comprises only58 frozen pending global-odd parent jobs; explicit GPU0 continuation, not old GPU1 output',
        start_condition='Parent GPU0 queue_scanned, all98 original even jobs complete, watcher/service identities exited/reaped, then exact GPU0 idle check',
        created_unix=time.time(),deadline_unix=DEADLINE,staged_execution='No repeated first-case gate: same verified frozen v3 startup legacy+16ref+D2 gate, then remaining58 sequentially',
        final_accounting='ParentGPU0 even98 + frozen parentGPU1 odd39 + thisGPU0 new58 =195; technical interruptions excluded')
    m['source_files']=dict(parent['source_files']);m['source_files'][str(Path(__file__).resolve())]=sha(__file__)
    m['manifest_hash']=digest(m);output=Path(output);output.mkdir(parents=True,exist_ok=False);write_json(output/'manifest.json',m);return m


def verify(path,job_identity=None):
    m=protocol_identity(path)
    if m.get('kind')!=KIND or m.get('condition_count')!=58 or m.get('assigned_gpu_uuid')!=diagnostic.CARDS[0] or m.get('deadline_unix')!=DEADLINE:raise ValueError('Fixed GPU0 continuation identity required')
    for name,expected in m['source_files'].items():
        source=Path(name);source=source if source.is_absolute() else ROOT/source
        if sha(source)!=expected:raise ValueError('Frozen source changed: '+str(source))
    evidence=[m['parent_quality_manifest']]
    if job_identity is None:evidence += m['bound_files']+[dict(path=r['path'],sha256=r['sha256']) for r in m['parent_gpu1_reused_complete']]+m['parent_gpu1_preserved_technical_attempts']
    else:
        job=next(j for j in m['jobs'] if (j['group'],j['source_index'])==job_identity)
        evidence += [job['initial_observation'],job['first_input']]+[dict(path=r['path'],sha256=r['sha256']) for r in job['paired_results']]
    for item in evidence:
        if sha(item['path'])!=item['sha256']:raise ValueError('Bound evidence changed: '+item['path'])
    return m


def parent_readiness(m):
    watcher=json.loads(Path(m['parent_gpu0_watcher_path']).read_text())
    if watcher.get('manifest_hash')!=PARENT or watcher.get('owner',{}).get('pid')!=m['parent_gpu0_watcher']['pid'] or watcher['owner']['start_ticks']!=m['parent_gpu0_watcher']['start_ticks']:raise ValueError('Parent watcher identity changed')
    if watcher['status']=='technical_failure':raise RuntimeError('ParentGPU0 technical failure: continuation prohibited')
    live=alive(m['parent_gpu0_watcher'])
    if watcher['status']!='queue_scanned':
        if not live:raise RuntimeError('ParentGPU0 exited without normal queue_scanned')
        return dict(ready=False,status=watcher['status'])
    parent=protocol_identity(m['parent_quality_manifest']['path']);rows=[]
    for job in v3.shard_jobs(parent,0):
        good,_=completed(m['parent_results_root'],job,PARENT)
        if not good:raise RuntimeError('Parent claimed queue_scanned without all98 even conditions')
        if json.loads(Path(good[0]['path']).read_text()).get('gpu_uuid')!=diagnostic.CARDS[0]:raise ValueError('Parent even result device differs from GPU0')
        rows+=good
    if len(rows)!=98:raise RuntimeError('Parent normal even98 coverage differs')
    service_live=alive(m['parent_gpu0_service'])
    return dict(ready=not live and not service_live and watcher.get('service_reaped') is True,
        status='parent_normal_drain' if not live and not service_live else 'awaiting_owned_process_exit',
        completed_even98=rows,watcher_receipt=bound_file(m['parent_gpu0_watcher_path']))


def sample_gpu0():
    uuid=diagnostic.CARDS[0];base=['nvidia-smi','-i',uuid]
    processes=subprocess.check_output(base+['--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True).strip()
    device=subprocess.check_output(base+['--query-gpu=index,uuid,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
    guard=dict(device=device,preexisting_compute_processes=processes)
    try:validated=diagnostic.validate_idle_guard(guard,0);idle=True
    except RuntimeError:validated=None;idle=False
    return dict(idle=idle,guard=guard,validated=validated,time_unix=time.time())


def queue(a,m,receipt,path):
    for job in m['jobs']:
        if completed(a.output,job,m['manifest_hash'])[0]:continue
        diagnostic.deadline_check(DEADLINE)
        base=Path(a.output)/job['group']/('case%04d'%job['source_index'])/'v14_d2';base.mkdir(parents=True,exist_ok=True)
        numbers=[int(p.name[7:]) for p in base.glob('attempt*') if p.is_dir() and p.name[7:].isdigit()];dest=base/('attempt%03d'%(max(numbers,default=-1)+1))
        env=isolated_child_environment(job['case'],Path(a.output)/('config_gpu0_'+job['group']),0,m['assigned_gpu_uuid'])
        interpreter=PLUS/'.venv/bin/python' if job['group']=='perturbed350' else ROOT/'policies/libero-eval-v21/.venv/bin/python'
        command=[str(interpreter),'-m',__spec__.name,'--stage','episode','--output',str(Path(a.output).resolve()),'--attempt-output',str(dest.resolve()),'--group',job['group'],'--source-index',str(job['source_index']),'--port',str(a.port),'--launch']
        with (base/(dest.name+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            row=dict(pid=child.pid,identity=process_identity(child.pid),command=command,output=str(dest),source_index=job['source_index'],group=job['group']);receipt['active_episode']=row;write_json(path,receipt)
            code=child.wait()
        row.update(exit_code=code,reaped=True);receipt.pop('active_episode',None);receipt['episodes'].append(row);write_json(path,receipt)
        if code:raise RuntimeError('Continuation technical failure preserved; no policy-zero substitution')


def watch(a,m):
    output=Path(a.output);path=output/'continuation.json'
    if path.exists() or (output/'runtime_gpu0/service.json').exists():raise FileExistsError('New continuation execution receipt required')
    receipt=dict(status='waiting_parent_gpu0_normal_drain',owner=process_identity(os.getpid()),manifest_hash=m['manifest_hash'],deadline_unix=DEADLINE,
        assigned_gpu_uuid=m['assigned_gpu_uuid'],forbidden_gpu_indices=[1,2,3],source_sha256=sha(__file__),episodes=[]);write_json(path,receipt);service=None
    try:
        while time.time()<DEADLINE:
            ready=parent_readiness(m);receipt['parent_readiness']=ready;write_json(path,receipt)
            if ready['ready']:break
            time.sleep(min(30,max(0,DEADLINE-time.time())))
        else:receipt['status']='deadline_waiting_parent';return
        receipt['status']='waiting_gpu0_idle_after_parent_exit';write_json(path,receipt)
        while time.time()<DEADLINE:
            sample=sample_gpu0();receipt['last_gpu0_idle_sample']=sample;write_json(path,receipt)
            if sample['idle']:break
            time.sleep(min(30,max(0,DEADLINE-time.time())))
        else:receipt['status']='deadline_waiting_gpu0_idle';return
        command=[sys.executable,'-m',__spec__.name,'--stage','serve','--output',str(output.resolve()),'--port',str(a.port),'--launch'];runtime=output/'runtime_gpu0';runtime.mkdir(exist_ok=True)
        with (runtime/'service.log').open('x') as log:
            service=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=m['assigned_gpu_uuid'],JAX_PLATFORMS='cpu'),stdout=log,stderr=subprocess.STDOUT)
            receipt.update(status='starting_strict_legacy16_D2_gate',service_pid=service.pid,service_command=command);write_json(path,receipt)
            while time.time()<DEADLINE:
                if service.poll() is not None:raise RuntimeError('Continuation startup strict gate failed')
                try:receipt['service_health']=v3.health(a,m,service.pid);break
                except (OSError,ConnectionError,TimeoutError):time.sleep(2)
            else:raise TimeoutError('Continuation startup deadline')
            receipt['status']='executing_frozen58_GPU0';write_json(path,receipt);queue(a,m,receipt,path)
            new=[r for job in m['jobs'] for r in completed(output,job,m['manifest_hash'])[0]]
            if len(new)!=58:raise RuntimeError('Continuation ended without58 valid results')
            receipt.update(status='queue_scanned',completed_new58=new,final_coverage=195,reused_parent_even98=98,reused_parent_odd39=39)
    except Exception:receipt.update(status='technical_failure',error=traceback.format_exc());raise
    finally:
        if service is not None and service.poll() is None:
            try:v3.health(a,m,service.pid);receipt['shutdown_response']=rpc(a.port,dict(op='shutdown'),timeout=5)
            except Exception:receipt['shutdown_note']='Own service deadline retained; no renderer/process signals'
            service.wait();receipt['service_reaped']=True
        elif service is not None:service.wait();receipt['service_reaped']=True
        receipt['finished_unix']=time.time();write_json(path,receipt)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,choices=('register','watch','serve','episode'));p.add_argument('--output',required=True)
    p.add_argument('--parent-manifest');p.add_argument('--port',type=int,default=19166);p.add_argument('--attempt-output');p.add_argument('--group');p.add_argument('--source-index',type=int);p.add_argument('--launch',action='store_true');a=p.parse_args();a.card=0
    if a.stage=='register':
        if not a.parent_manifest:p.error('--parent-manifest required')
        m=register(a.parent_manifest,a.output);print(json.dumps(dict(manifest_hash=m['manifest_hash'],pending=58,reused_odd=39,deadline_unix=DEADLINE)));return
    ident=(a.group,a.source_index) if a.stage=='episode' else None;m=verify(Path(a.output)/'manifest.json',ident)
    if not a.launch:print(json.dumps(dict(dry_run=True,stage=a.stage,manifest_hash=m['manifest_hash'],pending=58,gpu_uuid=m['assigned_gpu_uuid'],deadline_unix=DEADLINE)));return
    if a.stage=='watch':watch(a,m)
    elif a.stage=='serve':v3.serve(a,m)
    else:
        job=next(j for j in m['jobs'] if (j['group'],j['source_index'])==ident);args=argparse.Namespace(output=a.attempt_output,port=a.port,gpu=0,gpu_uuid=m['assigned_gpu_uuid'],deadline_unix=DEADLINE)
        result=episode(args,job,m);raise SystemExit(0 if result['status']=='complete' else 1)


if __name__=='__main__':main()

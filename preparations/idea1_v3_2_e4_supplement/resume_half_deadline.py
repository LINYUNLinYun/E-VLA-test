"""Approval-gated17:55UTC same-device half585 continuation, never a scheduler optimization.

Prepare/status are read-only toward existing processes. --launch additionally
requires a root-written revocable approval. No GPU sampling occurs until the
verified original renderer and model have naturally exited. No renderer child
is ever signalled. This file does not alter any frozen runtime or manifest.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from .native_envs import ROOT,digest,sha,write_json
from .half_plus import audit,source_results
from .evaluation import protocol_identity
from .parallel_continue import process_identity
from .policy_native_watch_v3 import rpc

TRIGGER=1792691700  #2026-10-22 17:55UTC
DEADLINE=1792713600 #2026-10-23 00:00UTC
UUID='GPU-555cde36-5e41-bc44-0c69-ea47358fd703'


def exact_process(expected,current):
    return current is not None and all(current.get(key)==expected.get(key) for key in ('pid','start_ticks','argv','cwd')) and current.get('state')!='Z'


def endpoint(protocol,expected_pid,uuid,manifest_hash):
    if (protocol.get('server_pid')!=expected_pid or protocol.get('gpu_uuid')!=uuid or protocol.get('manifest_hash')!=manifest_hash
        or protocol.get('family')!='libero_transfer' or protocol.get('legacy_gate_passed') is not True):
        raise RuntimeError('Endpoint PID/UUID/manifest/family/gate mismatch; no mutation authorized')
    return protocol


def sources(plan):
    for name,expected in plan['source_sha256'].items():
        if sha(name)!=expected: raise RuntimeError('Frozen source/receipt changed: '+name)


def approval(plan,path):
    value=json.loads(Path(path).read_text())
    if not (value.get('approved') is True and value.get('revoked') is False and value.get('plan_hash')==plan['plan_hash']
        and value.get('priority')=='plus_half585_only' and value.get('new_deadline_unix')==DEADLINE):
        raise RuntimeError('Approval absent/revoked or priority changed; stop without touching replacement processes')
    if Path(str(path)+'.cancel').exists(): raise RuntimeError('Explicit priority cancellation marker present')
    return value


def verify_old(plan):
    sources(plan)
    for key in ('worker','model'):
        if not exact_process(plan[key],process_identity(plan[key]['pid'])): raise RuntimeError('Original '+key+' changed/exited; fail-safe')
    return endpoint(rpc(plan['port'],dict(op='health'),timeout=10)['protocol'],plan['model']['pid'],UUID,plan['parent_manifest_hash'])


def prepare(a):
    half_root=Path(a.half_root);manifest=json.loads((half_root/'manifest.json').read_text())
    if manifest['manifest_hash']!=digest({k:v for k,v in manifest.items() if k!='manifest_hash'}): raise ValueError('Half identity invalid')
    parent=protocol_identity(manifest['parent_manifest_path']);worker=process_identity(a.worker_pid);model=process_identity(a.model_pid)
    for mapping in (parent['source_files'],parent['receipt_files'],manifest['source_files']):
        for name,expected in mapping.items():
            path=Path(name);path=path if path.is_absolute() else ROOT/path
            if sha(path)!=expected: raise ValueError('Existing frozen source/receipt mismatch: '+str(path))
    if not worker or 'supplement.half_plus ' not in worker['argv'] or not model or 'supplement.policy_plus --serve ' not in model['argv']:
        raise ValueError('Expected original half worker/model identities required')
    health=endpoint(rpc(a.port,dict(op='health'),timeout=10)['protocol'],a.model_pid,UUID,parent['manifest_hash'])
    paths=[Path(manifest['parent_manifest_path']),half_root/'manifest.json',half_root/'runtime_source_closure.json',
        Path(__file__),Path(__file__).with_name('policy_plus_mirror.py'),Path(__file__).with_name('policy_native_watch_v3.py'),
        Path(__file__).with_name('half_plus.py'),Path(__file__).with_name('parallel_continue.py')]
    for name in list(parent['source_files'])+list(parent['receipt_files']):
        path=Path(name);paths.append(path if path.is_absolute() else ROOT/path)
    plan=dict(schema=1,kind='approved_same_GPU3_half585_deadline_extension',prepared_unix=time.time(),half_root=str(half_root),half_manifest_hash=manifest['manifest_hash'],
        parent_manifest_hash=parent['manifest_hash'],parent_manifest_path=manifest['parent_manifest_path'],worker=worker,model=model,port=a.port,
        gpu=3,gpu_uuid=UUID,trigger_unix=TRIGGER,new_deadline_unix=DEADLINE,child_drain_timeout_s=180,idle_poll_interval_s=30,
        source_sha256={str(path.resolve()):sha(path) for path in paths},prepared_health=health,
        priority_revocation='Root must revoke approval/delete it/create approval.cancel or stop this guard when userpriority changes; chat messages are not observable by a background process',
        constraints='No newcard; no renderer signals; no crossGPUgate/outer; frozen samea416 native6route startup+CPUassembly/finalpreCUDAguard; all585audit is onlycompletion')
    plan['plan_hash']=digest(plan);output=Path(a.output);output.mkdir(parents=True,exist_ok=False);write_json(output/'plan.json',plan)
    write_json(output/'approval_template.json',dict(approved=False,revoked=False,priority='plus_half585_only',plan_hash=plan['plan_hash'],new_deadline_unix=DEADLINE))
    return plan


def children_of(pid):
    path=Path('/proc')/str(pid)/'task'/str(pid)/'children'
    return [process_identity(int(value)) for value in path.read_text().split()]


def alive(identity):
    current=process_identity(identity['pid'])
    return current is not None and current.get('start_ticks')==identity['start_ticks'] and current['state']!='Z'


def drain_old_worker(plan,receipt,directory,approve):
    expected=plan['worker'];approve();verify_old(plan)
    os.kill(expected['pid'],signal.SIGSTOP);receipt['worker_stopped']=True
    receipt.setdefault('signals',[]).append(dict(pid=expected['pid'],signal='SIGSTOP',unix=time.time()))
    try:
        time.sleep(.1);children=children_of(expected['pid'])
        # A child may finish between STOP and /proc inspection. Its zombie PID
        # still has this exact parent (cannot be reused), but cmdline is empty.
        # Recover only its exact active command from the verified workerreceipt.
        for child in children:
            if child and child['state']=='Z' and not child['argv']:
                path=Path(plan['half_root'])/('worker_gpu3_%d.json'%expected['pid'])
                current=json.loads(path.read_text());active=current.get('active',{})
                if current.get('pid')!=expected['pid'] or current.get('controller_start_ticks')!=expected['start_ticks'] or active.get('pid')!=child['pid']:
                    raise RuntimeError('Unidentified zombie child; no renderer signal')
                child['argv']=' '.join(active['command']);child['argv_source']='exact verified worker active command; zombie PPID stilloriginal'
        receipt['draining_children']=children
        for child in children:
            if not child or child['ppid']!=expected['pid'] or 'supplement.run_plus ' not in child['argv'] or '--child-index ' not in child['argv']:
                raise RuntimeError('Unexpected child; do not signal or drain a replacement')
        write_json(directory/'execution.json',receipt);until=time.monotonic()+180
        while any(alive(child) for child in children):
            approve()
            if time.monotonic()>=until: raise TimeoutError('Current episode did not finish naturally within180s; never killrenderer')
            time.sleep(.2)
        for child in children:
            argv=child['argv'].split();path=Path(argv[argv.index('--output')+1])/'result.json';value=json.loads(path.read_text())
            if value.get('status')!='complete' or value.get('manifest_hash')!=plan['parent_manifest_hash']:
                raise RuntimeError('Drained episode technical/nonidentity result; preserve and stop')
            receipt.setdefault('drained_results',[]).append(dict(pid=child['pid'],path=str(path),sha256=sha(path),success=value['success']))
        approve()
        if not exact_process(expected,process_identity(expected['pid'])): raise RuntimeError('Original worker changed before TERM')
        os.kill(expected['pid'],signal.SIGTERM);os.kill(expected['pid'],signal.SIGCONT)
        receipt['signals']+=[dict(pid=expected['pid'],signal='SIGTERM'),dict(pid=expected['pid'],signal='SIGCONT')]
        until=time.monotonic()+15
        while alive(expected):
            if time.monotonic()>until: raise RuntimeError('Original worker did not exit after TERM+CONT')
            time.sleep(.1)
        receipt.update(worker_stopped=False,original_worker_exited=True,renderer_signals_sent=False)
    except BaseException:
        if exact_process(expected,process_identity(expected['pid'])):
            os.kill(expected['pid'],signal.SIGCONT);receipt['signals'].append(dict(pid=expected['pid'],signal='SIGCONT',reason='guardfailure_resume_original'))
        receipt['worker_stopped']=False;raise


def stop_old_model(plan,approve):
    approve();sources(plan)
    if not exact_process(plan['model'],process_identity(plan['model']['pid'])): raise RuntimeError('Old model identity changed; do notshutdownreplacement')
    endpoint(rpc(plan['port'],dict(op='health'),timeout=10)['protocol'],plan['model']['pid'],UUID,plan['parent_manifest_hash'])
    rpc(plan['port'],dict(op='shutdown'),timeout=10)
    until=time.monotonic()+30
    while alive(plan['model']):
        if time.monotonic()>until: raise RuntimeError('Own original model did not exit after RPC; do not forcekill')
        time.sleep(.2)


def archive_claims(plan,directory,manifest):
    if alive(plan['worker']): raise RuntimeError('Never archive claims of liveowner')
    archived=[];base=Path(plan['half_root'])/'claims';target=directory/'archived_claims';target.mkdir()
    jobs={('%s_%04d'%(job['group'],job['source_index'])):job for job in manifest['jobs']}
    for claim in base.iterdir():
        if not claim.is_dir() or not (claim/'owner.json').exists() or (claim/'complete.json').exists(): continue
        owner=json.loads((claim/'owner.json').read_text());job=jobs.get(claim.name)
        if not job or any(owner.get(key)!=value for key,value in dict(pid=plan['worker']['pid'],start_ticks=plan['worker']['start_ticks'],gpu_uuid=UUID,
            half_manifest_hash=plan['half_manifest_hash'],case_hash=job['case_hash']).items()): continue
        pending=[]
        for mode in job['modes']:
            old,_=source_results(manifest['source_results_root'],job['group'],job['source_index'],job['case'],mode,plan['parent_manifest_hash'])
            new,_=source_results(plan['half_root'],job['group'],job['source_index'],job['case'],mode,plan['parent_manifest_hash'])
            if old and new: raise RuntimeError('Duplicate completed identity before archive')
            if not old and not new: pending.append(mode)
        if not pending: continue
        destination=target/claim.name
        if destination.exists(): raise RuntimeError('Archive overwrite forbidden')
        owner_sha=sha(claim/'owner.json');claim.rename(destination)
        archived.append(dict(original=str(claim),archive=str(destination),owner_sha256=owner_sha,pending_modes=pending,recoverable=True))
    write_json(directory/'claim_archives.json',archived);return archived


def gpu3_sample():
    command=['nvidia-smi','-i','3']
    device=subprocess.check_output(command+['--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    fields=[value.strip() for value in device.split(',')]
    processes=subprocess.check_output(command+['--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True).strip()
    if fields[0]!='3' or fields[1]!=UUID: raise RuntimeError('PhysicalGPU3 UUID changed; noalternative allowed')
    return dict(index=3,uuid=fields[1],memory_mib=int(fields[2]),utilization=int(fields[3]),compute_processes=processes,
        idle=not processes and int(fields[2])<=16 and int(fields[3])==0,unix=time.time())


def wait_idle(plan,directory,approve):
    while time.time()<DEADLINE:
        approve();first=gpu3_sample()
        with (directory/'post_shutdown_gpu_samples.jsonl').open('a') as stream: stream.write(json.dumps(first)+'\n')
        if first['idle']:
            second=gpu3_sample()
            with (directory/'post_shutdown_gpu_samples.jsonl').open('a') as stream: stream.write(json.dumps(second)+'\n')
            if second['idle'] and second['uuid']==first['uuid']: return second
        time.sleep(min(30,max(0,DEADLINE-time.time())))
    raise TimeoutError('No genuinely idle sameGPU3 before extensiondeadline')


def cleanup_new_model(process,identity,plan):
    if process is None or process.poll() is not None:return
    # Only signal our exact Popen process, never whichever endpoint nowexists.
    if not exact_process(identity,process_identity(process.pid)): raise RuntimeError('New ownedmodel identity changed; do not touch')
    try:
        endpoint(rpc(plan['port'],dict(op='health'),timeout=5)['protocol'],process.pid,UUID,plan['parent_manifest_hash'])
        rpc(plan['port'],dict(op='shutdown'),timeout=5);process.wait(timeout=30)
    except (ConnectionError,OSError,TimeoutError,subprocess.TimeoutExpired):
        if exact_process(identity,process_identity(process.pid)):
            process.terminate();process.wait(timeout=30)


def execute(plan,a):
    if plan['plan_hash']!=digest({k:v for k,v in plan.items() if k!='plan_hash'}): raise ValueError('Guard plan identity mismatch')
    manifest=json.loads((Path(plan['half_root'])/'manifest.json').read_text());approve=lambda:approval(plan,a.approval)
    approve();sources(plan);directory=Path(a.plan).parent/('execution_%d'%int(time.time()));directory.mkdir(exist_ok=False)
    receipt=dict(status='waiting_trigger_readonly',guard_pid=os.getpid(),plan_hash=plan['plan_hash'],new_deadline_unix=DEADLINE,events=[],gpu_scans_before_original_shutdown=0)
    model=worker=None;model_identity=None
    def cancel(signum,frame):raise InterruptedError('Guardcancelled; do not resumeCALVIN/B or alterpriority')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    try:
        next_readonly_check=0
        while time.time()<TRIGGER:
            approve()  # lightweight revocation check, at most60s latency
            if time.time()>=next_readonly_check:
                ledger=audit(manifest,plan['half_root']);write_json(directory/'last_waiting_audit.json',ledger)
                if ledger['completed']==585:
                    receipt.update(status='already_complete_no_restart',completed_unix=time.time());return
                verify_old(plan)  # expensive source/PID/health audit onlyevery5min
                receipt['last_readonly_check_unix']=time.time();next_readonly_check=time.time()+300
                write_json(directory/'execution.json',receipt)
            time.sleep(min(60,max(0,TRIGGER-time.time())))
        approve();ledger=audit(manifest,plan['half_root']);write_json(directory/'audit_at_trigger.json',ledger)
        if ledger['completed']==585:
            receipt.update(status='already_complete_no_restart',completed_unix=time.time());return
        verify_old(plan)  # mandatory freshidentity/source/endpoint check at trigger
        receipt['status']='draining_old_worker';write_json(directory/'execution.json',receipt)
        drain_old_worker(plan,receipt,directory,approve);stop_old_model(plan,approve)
        receipt['original_model_exited']=True;receipt['archived_claims']=archive_claims(plan,directory,manifest)
        write_json(directory/'execution_extension.json',dict(kind='same_GPU3_execution_deadline_extension',parent_manifest_hash=plan['parent_manifest_hash'],
            half_manifest_hash=plan['half_manifest_hash'],old_deadline_unix=1792692000,new_deadline_unix=DEADLINE,scientific_settings_unchanged=True,
            guard_plan_hash=plan['plan_hash'],claim_archive=str(directory/'claim_archives.json')))
        receipt['status']='waiting_same_GPU3_idle';write_json(directory/'execution.json',receipt)
        receipt['idle_allocation']=wait_idle(plan,directory,approve);approve();sources(plan)
        command=[str(ROOT/'policies/realtime-vla-flash/.venv/bin/python'),'-m','preparations.idea1_v3_2_e4_supplement.policy_plus_mirror',
            '--serve','--family','libero_transfer','--protocol',plan['parent_manifest_path'],'--port',str(plan['port']),
            '--output',str(directory/'service.json'),'--deadline-unix',str(DEADLINE),'--launch']
        with (directory/'service.log').open('w') as log:
            model=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES=UUID,JAX_PLATFORMS='cpu'),stdout=log,stderr=subprocess.STDOUT)
            model_identity=process_identity(model.pid);receipt.update(status='loading_same_device_model',new_model=model_identity,model_command=command);write_json(directory/'execution.json',receipt)
            while time.time()<DEADLINE:
                approve()
                if model.poll() is not None: raise RuntimeError('Own newmodel exited before sixroute/finalCUDA readiness')
                try:
                    health=endpoint(rpc(plan['port'],dict(op='health'),timeout=2)['protocol'],model.pid,UUID,plan['parent_manifest_hash'])
                    if not health.get('mirror_final_preCUDA_guard',{}).get('checked'): raise RuntimeError('CPUassembly/finalpreCUDA guard absent')
                    receipt['new_health']=health;break
                except (ConnectionError,OSError,TimeoutError): time.sleep(2)
            else: raise TimeoutError('Newmodel readiness deadline')
            approve();command=[str(PLUS_PYTHON),'-m','preparations.idea1_v3_2_e4_supplement.half_plus','--output',plan['half_root'],
                '--gpu','3','--gpu-uuid',UUID,'--port',str(plan['port']),'--deadline-unix',str(DEADLINE),'--launch']
            with (directory/'worker.log').open('w') as worker_log:
                worker=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)),stdout=worker_log,stderr=subprocess.STDOUT)
                receipt.update(status='resumed_half585',new_worker=process_identity(worker.pid),worker_command=command);write_json(directory/'execution.json',receipt)
                while worker.poll() is None:
                    approve()
                    if time.time()>DEADLINE+180: raise TimeoutError('Worker/child notexited afterdeadlinegrace; do notsignalrenderer')
                    time.sleep(5)
                worker.wait();receipt['new_worker_exit_code']=worker.returncode
            ledger=audit(manifest,plan['half_root']);write_json(directory/'final_audit.json',ledger)
            receipt.update(status='complete_585_verified' if worker.returncode==0 and ledger['completed']==585 else 'incomplete_not_scientific_completion',
                completed_conditions=ledger['completed'],remaining=ledger['remaining'])
    except BaseException:
        receipt.update(status='guard_failed_or_cancelled',error=traceback.format_exc());raise
    finally:
        # Never terminate a halfworker/renderer for cleanup. If it is live,
        # leave model's own deadline in force and explicitly report this state.
        if model is not None and (worker is None or worker.poll() is not None):
            try:
                cleanup_new_model(model,model_identity,plan);receipt['new_model_exit_code']=model.poll()
            except Exception:
                receipt['cleanup_identity_failure_no_foreign_action']=traceback.format_exc()
        elif model is not None: receipt['cleanup_deferred_live_worker_no_renderer_signals']=True
        receipt['completed_unix']=time.time();write_json(directory/'execution.json',receipt)


PLUS_PYTHON=Path('/policies/LIBERO-plus/.venv/bin/python')


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--half-root');p.add_argument('--output')
    p.add_argument('--worker-pid',type=int,default=3420227);p.add_argument('--model-pid',type=int,default=3371306);p.add_argument('--port',type=int,default=19143)
    p.add_argument('--plan');p.add_argument('--approval');p.add_argument('--status',action='store_true');p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.prepare:
        if not a.half_root or not a.output:p.error('--half-root/--output required')
        plan=prepare(a);print(json.dumps(dict(prepared=True,plan_hash=plan['plan_hash'],trigger_unix=TRIGGER,never_started=True)));return
    if not a.plan:p.error('--plan required')
    plan=json.loads(Path(a.plan).read_text())
    if a.status:
        ledger=audit(json.loads((Path(plan['half_root'])/'manifest.json').read_text()),plan['half_root'])
        print(json.dumps(dict(completed=ledger['completed'],remaining=ledger['remaining'],worker=process_identity(plan['worker']['pid']),model=process_identity(plan['model']['pid']),gpu_scanned=False)));return
    if not a.launch:print(json.dumps(dict(dry_run=True,plan_hash=plan['plan_hash'],trigger_unix=TRIGGER)));return
    if not a.approval:p.error('Root-written revocable --approval required')
    execute(plan,a)


if __name__=='__main__':main()

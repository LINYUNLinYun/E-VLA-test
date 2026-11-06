"""Versioned17:30 guard with bounded25-minute natural episode drain.

Reuses reviewed ownership/claim/idle/model helpers without editing their files.
No launch/revocation/signals occur during --prepare or without --launch. Timing
is frozen into the new plan, validated, and consumed by this execution loop.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from types import SimpleNamespace
from . import resume_half_deadline as base
from .native_envs import ROOT,digest,sha,write_json

TRIGGER=1792690200
OLD_DEADLINE=1792692000
NEW_DEADLINE=1792713600
UUID=base.UUID
audit=base.audit
sources=base.sources
verify_old=base.verify_old
process_identity=base.process_identity
exact_process=base.exact_process
children_of=base.children_of
alive=base.alive
rpc=base.rpc
endpoint=base.endpoint


def validate_timing(plan):
    if plan.get('trigger_unix')!=TRIGGER or plan.get('old_deadline_unix')!=OLD_DEADLINE or plan.get('new_deadline_unix')!=NEW_DEADLINE:
        raise ValueError('Only approved17:30/old18:00/new00:00UTC timing is supported')
    timeout=plan.get('child_drain_timeout_s',0);margin=plan.get('shutdown_safety_margin_s',0)
    if not 180<timeout<=1500 or margin<180 or plan['trigger_unix']+timeout>plan['old_deadline_unix']-margin:
        raise ValueError('Natural drain must be >180s, <=1500s and finish before olddeadline minus >=180s safety')
    if plan.get('heavy_check_interval_s')!=300 or not 0<plan.get('approval_poll_interval_s',0)<=60 or plan.get('idle_poll_interval_s')!=30:
        raise ValueError('Heavy5min/approval<=60s/idle>=30s cadence required')
    return dict(trigger_unix=plan['trigger_unix'],timeout_s=timeout,latest_drain_end_unix=plan['old_deadline_unix']-margin,
        shutdown_safety_margin_s=margin)


def prepare(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    capture_args=SimpleNamespace(**vars(a));capture_args.output=str(output/'readonly_identity_capture')
    captured=base.prepare(capture_args)  # only frozen SHA, /proc and health reads
    plan={key:value for key,value in captured.items() if key!='plan_hash'}
    plan.update(schema=2,kind='same_GPU3_half585_long_natural_drain_guard',trigger_unix=TRIGGER,old_deadline_unix=OLD_DEADLINE,
        new_deadline_unix=NEW_DEADLINE,child_drain_timeout_s=1500,shutdown_safety_margin_s=180,
        heavy_check_interval_s=300,approval_poll_interval_s=60,idle_poll_interval_s=30,
        base_identity_capture_plan_hash=captured['plan_hash'],identity_capture_only_not_execution='readonly_identity_capture contains original helper metadata, not timing used byv2',
        old_revocable_approval_pinned=False,guard_version=2)
    plan['source_sha256'][str(Path(__file__).resolve())]=sha(__file__)
    if any('root_approval' in path for path in plan['source_sha256']): raise ValueError('Revocable old approval must not enter frozen source closure')
    validate_timing(plan);plan['plan_hash']=digest(plan);write_json(output/'plan.json',plan)
    write_json(output/'approval_template.json',dict(approved=False,revoked=False,priority='plus_half585_only',plan_hash=plan['plan_hash'],
        trigger_unix=TRIGGER,new_deadline_unix=NEW_DEADLINE,child_drain_timeout_s=1500,shutdown_safety_margin_s=180))
    return plan


def drain_old_worker(plan,receipt,directory,approve):
    timing=validate_timing(plan);expected=plan['worker'];approve();verify_old(plan)
    remaining=timing['latest_drain_end_unix']-time.time()
    if remaining<=0: raise TimeoutError('Too late for safe shutdown margin; original worker untouched')
    cap=min(timing['timeout_s'],remaining);until=time.monotonic()+cap
    receipt['drain_timing']=dict(timing,started_unix=time.time(),actual_cap_s=cap)
    os.kill(expected['pid'],signal.SIGSTOP);receipt['worker_stopped']=True
    receipt.setdefault('signals',[]).append(dict(pid=expected['pid'],signal='SIGSTOP',unix=time.time()))
    try:
        time.sleep(.1);children=children_of(expected['pid'])
        for child in children:
            if child and child['state']=='Z' and not child['argv']:
                current=json.loads((Path(plan['half_root'])/('worker_gpu3_%d.json'%expected['pid'])).read_text());active=current.get('active',{})
                if current.get('pid')!=expected['pid'] or current.get('controller_start_ticks')!=expected['start_ticks'] or active.get('pid')!=child['pid']:
                    raise RuntimeError('Unidentified zombie child; renderer never signalled')
                child['argv']=' '.join(active['command']);child['argv_source']='verified exact worker active command forzombie'
        receipt['draining_children']=children
        for child in children:
            if not child or child['ppid']!=expected['pid'] or 'supplement.run_plus ' not in child['argv'] or '--child-index ' not in child['argv']:
                raise RuntimeError('Unexpected child; do not affect replacement process')
        write_json(directory/'execution.json',receipt)
        while any(alive(child) for child in children):
            approve()
            if time.monotonic()>=until or time.time()>=timing['latest_drain_end_unix']:
                raise TimeoutError('Natural drain cap or olddeadline safety boundary reached; never killchild')
            time.sleep(.2)
        if time.monotonic()>until or time.time()>timing['latest_drain_end_unix']:
            raise TimeoutError('Child exited only after drain/safety cap; resume original parent without restart')
        for child in children:
            argv=child['argv'].split();path=Path(argv[argv.index('--output')+1])/'result.json';value=json.loads(path.read_text())
            if value.get('status')!='complete' or value.get('manifest_hash')!=plan['parent_manifest_hash']:
                raise RuntimeError('Drained episode technical/nonidentity result preserved; no restart')
            receipt.setdefault('drained_results',[]).append(dict(pid=child['pid'],path=str(path),sha256=sha(path),success=value['success']))
        approve()
        if not exact_process(expected,process_identity(expected['pid'])): raise RuntimeError('Original worker changed before TERM')
        os.kill(expected['pid'],signal.SIGTERM);os.kill(expected['pid'],signal.SIGCONT)
        receipt['signals']+=[dict(pid=expected['pid'],signal='SIGTERM'),dict(pid=expected['pid'],signal='SIGCONT')]
        until_exit=time.monotonic()+15
        while alive(expected):
            if time.monotonic()>until_exit: raise RuntimeError('Original worker did not exit after TERM+CONT')
            time.sleep(.1)
        receipt.update(worker_stopped=False,original_worker_exited=True,renderer_signals_sent=False)
    except BaseException:
        if exact_process(expected,process_identity(expected['pid'])):
            os.kill(expected['pid'],signal.SIGCONT);receipt['signals'].append(dict(pid=expected['pid'],signal='SIGCONT',reason='drain_failure_resume_only_original'))
        receipt['worker_stopped']=False;raise


def execute(plan,a):
    timing=validate_timing(plan)
    if plan['plan_hash']!=digest({key:value for key,value in plan.items() if key!='plan_hash'}): raise ValueError('Guardv2 plan identity mismatch')
    manifest=json.loads((Path(plan['half_root'])/'manifest.json').read_text());approve=lambda:base.approval(plan,a.approval)
    approve();sources(plan);directory=Path(a.plan).parent/('execution_%d'%int(time.time()));directory.mkdir(exist_ok=False)
    receipt=dict(status='waiting_trigger_readonly',guard_version=2,guard_pid=os.getpid(),plan_hash=plan['plan_hash'],timing=timing,
        new_deadline_unix=plan['new_deadline_unix'],gpu_scans_before_original_shutdown=0)
    model=worker=None;model_identity=None;deadline=plan['new_deadline_unix']
    def cancel(signum,frame): raise InterruptedError('Guardv2cancelled; no renderer signals orpriority expansion')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    try:
        next_check=0
        while time.time()<plan['trigger_unix']:
            approve()
            if time.time()>=next_check:
                ledger=audit(manifest,plan['half_root']);write_json(directory/'last_waiting_audit.json',ledger)
                if ledger['completed']==585: receipt['status']='already_complete_no_restart';return
                verify_old(plan);receipt['last_readonly_check_unix']=time.time();next_check=time.time()+plan['heavy_check_interval_s']
                write_json(directory/'execution.json',receipt)
            time.sleep(min(plan['approval_poll_interval_s'],max(0,plan['trigger_unix']-time.time())))
        approve();ledger=audit(manifest,plan['half_root']);write_json(directory/'audit_at_trigger.json',ledger)
        if ledger['completed']==585: receipt['status']='already_complete_no_restart';return
        verify_old(plan);receipt['status']='draining_old_worker';write_json(directory/'execution.json',receipt)
        drain_old_worker(plan,receipt,directory,approve)
        # The final current episode may finish the entire585. Re-audit BEFORE
        # touching the old model; keep it untouched and avoid needless reload.
        ledger=audit(manifest,plan['half_root']);write_json(directory/'audit_after_drain.json',ledger)
        if ledger['completed']==585:
            receipt.update(status='complete_after_drain_original_model_untouched',completed_conditions=585);return
        base.stop_old_model(plan,approve);receipt['original_model_exited']=True
        receipt['archived_claims']=base.archive_claims(plan,directory,manifest)
        write_json(directory/'execution_extension.json',dict(kind='same_GPU3_long_drain_execution_extension',parent_manifest_hash=plan['parent_manifest_hash'],
            half_manifest_hash=plan['half_manifest_hash'],old_deadline_unix=plan['old_deadline_unix'],new_deadline_unix=deadline,
            trigger_unix=plan['trigger_unix'],child_drain_timeout_s=plan['child_drain_timeout_s'],shutdown_safety_margin_s=plan['shutdown_safety_margin_s'],
            scientific_settings_unchanged=True,guard_plan_hash=plan['plan_hash'],claim_archive=str(directory/'claim_archives.json')))
        receipt['status']='waiting_same_GPU3_idle';write_json(directory/'execution.json',receipt)
        receipt['idle_allocation']=base.wait_idle(plan,directory,approve);approve();sources(plan)
        command=[str(ROOT/'policies/realtime-vla-flash/.venv/bin/python'),'-m','preparations.idea1_v3_2_e4_supplement.policy_plus_mirror',
            '--serve','--family','libero_transfer','--protocol',plan['parent_manifest_path'],'--port',str(plan['port']),
            '--output',str(directory/'service.json'),'--deadline-unix',str(deadline),'--launch']
        with (directory/'service.log').open('w') as log:
            model=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES=UUID,JAX_PLATFORMS='cpu'),stdout=log,stderr=subprocess.STDOUT)
            model_identity=process_identity(model.pid);receipt.update(status='loading_same_device_model',new_model=model_identity,model_command=command);write_json(directory/'execution.json',receipt)
            while time.time()<deadline:
                approve()
                if model.poll() is not None: raise RuntimeError('New ownedmodel exited before startupnative6/finalCUDAguard')
                try:
                    health=endpoint(rpc(plan['port'],dict(op='health'),timeout=2)['protocol'],model.pid,UUID,plan['parent_manifest_hash'])
                    if not health.get('mirror_final_preCUDA_guard',{}).get('checked'): raise RuntimeError('CPUassembly/finalCUDAguard absent')
                    receipt['new_health']=health;break
                except (ConnectionError,OSError,TimeoutError):time.sleep(2)
            else: raise TimeoutError('Model readiness deadline')
            approve();command=[str(base.PLUS_PYTHON),'-m','preparations.idea1_v3_2_e4_supplement.half_plus','--output',plan['half_root'],
                '--gpu','3','--gpu-uuid',UUID,'--port',str(plan['port']),'--deadline-unix',str(deadline),'--launch']
            with (directory/'worker.log').open('w') as worker_log:
                worker=subprocess.Popen(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)),stdout=worker_log,stderr=subprocess.STDOUT)
                receipt.update(status='resumed_half585',new_worker=process_identity(worker.pid),worker_command=command);write_json(directory/'execution.json',receipt)
                while worker.poll() is None:
                    approve()
                    if time.time()>deadline+180: raise TimeoutError('Worker notexited afterdeadline grace; renderer notsignalled')
                    time.sleep(5)
                worker.wait();receipt['new_worker_exit_code']=worker.returncode
            ledger=audit(manifest,plan['half_root']);write_json(directory/'final_audit.json',ledger)
            receipt.update(status='complete_585_verified' if worker.returncode==0 and ledger['completed']==585 else 'incomplete_not_scientific_completion',
                completed_conditions=ledger['completed'],remaining=ledger['remaining'])
    except BaseException:
        receipt.update(status='guard_failed_or_cancelled',error=traceback.format_exc());raise
    finally:
        if model is not None and (worker is None or worker.poll() is not None):
            try: base.cleanup_new_model(model,model_identity,plan);receipt['new_model_exit_code']=model.poll()
            except Exception:receipt['cleanup_identity_failure_no_foreign_action']=traceback.format_exc()
        elif model is not None:receipt['cleanup_deferred_live_worker_no_renderer_signals']=True
        receipt['completed_unix']=time.time();write_json(directory/'execution.json',receipt)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--half-root');p.add_argument('--output')
    p.add_argument('--worker-pid',type=int,default=3420227);p.add_argument('--model-pid',type=int,default=3371306);p.add_argument('--port',type=int,default=19143)
    p.add_argument('--plan');p.add_argument('--approval');p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.prepare:
        if not a.half_root or not a.output:p.error('--half-root/--output required')
        plan=prepare(a);print(json.dumps(dict(prepared=True,plan_hash=plan['plan_hash'],timing=validate_timing(plan),never_started=True)));return
    if not a.plan:p.error('--plan required')
    plan=json.loads(Path(a.plan).read_text());validate_timing(plan)
    if not a.launch:print(json.dumps(dict(dry_run=True,plan_hash=plan['plan_hash'],timing=validate_timing(plan))));return
    if not a.approval:p.error('New root-written revocable --approval required')
    execute(plan,a)


if __name__=='__main__':main()

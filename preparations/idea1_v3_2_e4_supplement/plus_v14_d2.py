"""Freeze/reuse 585 historical results and dispatch only 195 new V14/D2 cases."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from .native_envs import ROOT,digest,sha,write_json
from .evaluation import protocol_identity,verify_service
from .libero_plus_env import PLUS,isolated_child_environment
from .parallel_continue import process_identity,same_process

VIDEOS=[0,50,100,150,200,250,300]


def bound_file(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=sha(path))


def register(half_path, ledger_path, output, deadline):
    half=protocol_identity(half_path);parent=protocol_identity(half['parent_manifest_path'])
    ledger=json.loads(Path(ledger_path).read_text())
    if ledger.get('half_manifest_hash')!=half['manifest_hash'] or ledger.get('completed')!=585:
        raise ValueError('Completed matching585 audit ledger required')
    if not time.time()<deadline<=time.time()+12*3600: raise ValueError('Deadline must be fresh and at most12h')
    pairs={};files=[]
    for row in ledger['selected_completed']:
        path=Path(row['path']);value=json.loads(path.read_text())
        if sha(path)!=row['sha256'] or value.get('manifest_hash')!=parent['manifest_hash'] or value.get('status')!='complete' or value.get('mode')!=row['mode'] or type(value.get('success')) is not bool:
            raise ValueError('Historical result identity mismatch: '+str(path))
        uid=(row['group'],row['source_index']);pairs.setdefault(uid,[]).append(dict(row))
        files.append(bound_file(path))
    jobs=[]
    for job in sorted(half['jobs'],key=lambda j:(j['group']!='perturbed350',j['source_index'])):
        rows=pairs[(job['group'],job['source_index'])]
        if len(rows)!=3 or {r['mode'] for r in rows}!={'full_d10','v14_d10',job['case']['matching_draft']}: raise ValueError('Exact3 historical conditions required')
        for row in rows:
            value=json.loads(Path(row['path']).read_text())
            if value['case_hash']!=job['case_hash'] or digest(value['case'])!=job['case_hash']: raise ValueError('Historical case mismatch')
        full=next(r for r in rows if r['mode']=='full_d10');base=Path(full['path']).parent
        initial=bound_file(base/'initial_native_observation.npz');infer=bound_file(base/'infer.jsonl')
        files.extend((initial,infer))
        jobs.append(dict(group=job['group'],source_index=job['source_index'],case=job['case'],case_hash=job['case_hash'],
                         modes=['v14_d2'],paired_results=rows,initial_observation=initial,first_input=infer,
                         video=job['group']=='perturbed350' and job['source_index'] in VIDEOS))
    if len(jobs)!=195 or sum(j['video'] for j in jobs)!=7: raise ValueError('Exact195 cases/seven preregistered videos required')
    startup=[]
    for mode in ('full_d10','v14_d10'):
        row=next(r for r in jobs[0]['paired_results'] if r['mode']==mode);base=Path(row['path']).parent
        startup.append(dict(mode=mode,observation_path=str((base/'initial_native_observation.npz').resolve()),infer_path=str((base/'infer.jsonl').resolve())))
        files.extend((bound_file(base/'initial_native_observation.npz'),bound_file(base/'infer.jsonl')))
    sources=dict(parent['source_files'])
    for name in ('plus_v14_d2.py','policy_plus_v14_d2.py','run_plus_v14_d2.py','policy_plus_mirror.py','parallel_continue.py'):
        path=Path(__file__).with_name(name).resolve();sources[str(path)]=sha(path)
    m=dict(schema=1,kind='native_LIBERO_Plus_half_V14_D2_supplement',family='libero_transfer',
           modes=['v14_d2'],H=50,R=5,D=2,nodes=[1.,.5,0.],ae_depth=18,skip_vlm_1based=[3,4,9,11],
           native_hz=20,settle_steps=5,native_max_steps=600,condition_count=195,perturbed_case_count=175,original_case_count=20,
           reused_condition_count=585,jobs=jobs,deadline_unix=deadline,
           half_manifest=bound_file(half_path),parent_manifest=bound_file(half['parent_manifest_path']),
           audit_ledger=bound_file(ledger_path),parent_manifest_hash=parent['manifest_hash'],half_manifest_hash=half['manifest_hash'],
           checkpoint_metadata=parent['checkpoint_metadata'],source_files=sources,bound_files=files,startup_inputs=startup,
           noise_protocol='unchanged int(digest([plus-evaluation-v1,case_hash,request_index])[:8],16); mode excluded',
           initial_comparison=dict(physical='exact array equality',image_max_abs=1,image_max_changed_channels=4,instruction='exact',initial_vector_sha256='exact'),
           startup_gate=dict(historical_d10_max_abs_raw=.02,historical_d10_max_abs_decoded=.02,first5_gripper_sign_equal=True,
                             allclose_atol=.002,allclose_rtol=.002,allclose_diagnostic_only=True,d2_native_max_abs=1e-5),
           cross_gpu_limitation='Historical585 GPU3; D2 may run on another idle assigned GPU. Same-input D10 numerical diagnostic budget preregistered; no historical rerun.',
           video_source_indices=VIDEOS,created_unix=time.time())
    m['manifest_hash']=digest(m);output=Path(output);output.mkdir(parents=True,exist_ok=False);write_json(output/'manifest.json',m)
    return m


def verify_manifest(path, job_identity=None):
    m=protocol_identity(path)
    if m.get('kind')!='native_LIBERO_Plus_half_V14_D2_supplement' or m.get('modes')!=['v14_d2'] or m.get('condition_count')!=195: raise ValueError('Independent D2 manifest required')
    for name,expected in m['source_files'].items():
        resolved=Path(name);resolved=resolved if resolved.is_absolute() else ROOT/resolved
        if sha(resolved)!=expected: raise ValueError('Frozen source changed: '+str(resolved))
    evidence=[m['half_manifest'],m['parent_manifest'],m['audit_ledger']]+m['bound_files']
    if job_identity is not None:
        job=next(j for j in m['jobs'] if (j['group'],j['source_index'])==job_identity)
        evidence=[job['initial_observation'],job['first_input']]+[dict(path=r['path'],sha256=r['sha256']) for r in job['paired_results']]
    for entry in evidence:
        if sha(entry['path'])!=entry['sha256']: raise ValueError('Frozen pairing evidence changed: '+entry['path'])
    return m


def completed(output,job,identity):
    rows=[];technical=[]
    for path in (Path(output)/job['group']/('case%04d'%job['source_index'])/'v14_d2').glob('attempt*/result.json'):
        v=json.loads(path.read_text())
        if v.get('status')=='complete':
            if v.get('manifest_hash')!=identity or v.get('case_hash')!=job['case_hash'] or v.get('mode')!='v14_d2' or type(v.get('success')) is not bool: raise ValueError('Completed D2 result identity mismatch')
            rows.append(dict(group=job['group'],source_index=job['source_index'],mode='v14_d2',path=str(path),sha256=sha(path),success=v['success'],steps=v['steps']))
        elif v.get('status')=='technical_failure': technical.append(str(path))
    if len(rows)>1: raise ValueError('Duplicate completed D2 condition')
    return rows,technical


def audit(m,output):
    rows=[];technical=[]
    for job in m['jobs']:
        good,bad=completed(output,job,m['manifest_hash']);rows+=good;technical+=bad
    return dict(status='audit_complete',manifest_hash=m['manifest_hash'],target=195,completed=len(rows),remaining=195-len(rows),
                reused_historical=585,selected_completed=rows,technical_attempts=technical,audited_unix=time.time())


def dispatch(a,m):
    if not time.time()<a.deadline_unix<=m['deadline_unix']: raise ValueError('Fresh bounded deadline required')
    health=verify_service(a.port,m['manifest_hash'],'libero_transfer')
    if health.get('gpu_uuid')!=a.gpu_uuid or health.get('D')!=2 or health.get('modes')!=['v14_d2'] or not health.get('legacy_gate_passed'): raise ValueError('Assigned D2 service gate/device mismatch')
    output=Path(a.output);claim=output/'worker_claim';owner=process_identity(os.getpid())
    try: claim.mkdir()
    except FileExistsError:
        old=json.loads((claim/'owner.json').read_text())
        if same_process(old): raise RuntimeError('Live worker owns sequential queue')
        claim.rename(output/('worker_claim_stale_%d_%d'%(old['pid'],time.time_ns())))
        claim.mkdir()
    write_json(claim/'owner.json',owner)
    receipt=dict(status='running',owner=owner,gpu=a.gpu,gpu_uuid=a.gpu_uuid,port=a.port,manifest_hash=m['manifest_hash'],deadline_unix=a.deadline_unix,results=[])
    receipt_path=output/('worker_%d.json'%os.getpid());write_json(receipt_path,receipt)
    new=0
    try:
        for job in m['jobs']:
            if completed(output,job,m['manifest_hash'])[0]: continue
            if time.time()>=a.deadline_unix or new>=a.max_new_cases: break
            base=output/job['group']/('case%04d'%job['source_index'])/'v14_d2';base.mkdir(parents=True,exist_ok=True)
            indices=[int(p.name[7:]) for p in base.glob('attempt*') if p.is_dir() and p.name[7:].isdigit()]
            directory=base/('attempt%03d'%(max(indices,default=-1)+1))
            env=isolated_child_environment(job['case'],output/('config_'+job['group']),a.gpu,a.gpu_uuid)
            interpreter=PLUS/'.venv/bin/python' if job['group']=='perturbed350' else ROOT/'policies/libero-eval-v21/.venv/bin/python'
            command=[str(interpreter),'-m','preparations.idea1_v3_2_e4_supplement.run_plus_v14_d2','--protocol',str(output/'manifest.json'),'--output',str(directory),'--group',job['group'],'--source-index',str(job['source_index']),'--port',str(a.port),'--gpu',str(a.gpu),'--gpu-uuid',a.gpu_uuid,'--deadline-unix',str(a.deadline_unix),'--launch']
            with (base/(directory.name+'.log')).open('w') as log:
                child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
                row=dict(group=job['group'],source_index=job['source_index'],output=str(directory),command=command,pid=child.pid,identity=process_identity(child.pid));receipt['active']=row;write_json(receipt_path,receipt)
                try: code=child.wait(timeout=max(1,a.deadline_unix-time.time()))
                except subprocess.TimeoutExpired:
                    # The episode checks deadline before its next request and
                    # closes its renderer naturally; never kill active MuJoCo.
                    receipt['status']='deadline_draining_owned_child';write_json(receipt_path,receipt)
                    child.wait()
                    code=124
            row.update(exit_code=code,reaped=True);receipt['results'].append(row);receipt.pop('active',None);write_json(receipt_path,receipt);new+=1
            if code: raise RuntimeError('Technical attempt preserved; stop for diagnosis, never policy zero')
        receipt['status']='queue_scanned'
    except Exception: receipt.update(status='technical_failure',error=traceback.format_exc());raise
    finally:
        receipt['completed_unix']=time.time();write_json(receipt_path,receipt);write_json(output/'live_ledger.json',audit(m,output))
        claim.rename(output/('worker_claim_finished_%d_%d'%(os.getpid(),time.time_ns())))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--register',action='store_true');p.add_argument('--half-manifest');p.add_argument('--audit-ledger')
    p.add_argument('--audit',action='store_true');p.add_argument('--deadline-unix',type=float);p.add_argument('--port',type=int);p.add_argument('--gpu',type=int);p.add_argument('--gpu-uuid');p.add_argument('--max-new-cases',type=int,default=195);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.register:
        if not a.half_manifest or not a.audit_ledger or not a.deadline_unix: p.error('Register requires half manifest/audit ledger/fresh deadline')
        m=register(a.half_manifest,a.audit_ledger,a.output,a.deadline_unix);print(json.dumps(dict(manifest_hash=m['manifest_hash'],target=195,reused=585)));return
    m=verify_manifest(Path(a.output)/'manifest.json')
    if a.audit: result=audit(m,a.output);write_json(Path(a.output)/'live_ledger.json',result);print(json.dumps(result));return
    if not a.launch: print(json.dumps(dict(dry_run=True,target=195,reused=585,manifest_hash=m['manifest_hash'])));return
    if any(v is None for v in (a.port,a.gpu,a.gpu_uuid,a.deadline_unix)): p.error('Explicit port/GPU/UUID/deadline required')
    dispatch(a,m)


if __name__=='__main__': main()

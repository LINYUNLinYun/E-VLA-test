"""Outcome-blind175+20 subset and small per-GPU frozen-child dispatcher.

Original a416 scientific/runtime files and historical results remain untouched.
New half manifest has585 conditions and explicitly references parent identities.
Multiple approved devices may share this queue using atomic whole-case claims;
each case's three conditions stay on one physical GPU, never model batching.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import time
from .native_envs import ROOT,digest,sha,write_json
from .evaluation import protocol_identity,verify_service
from .libero_plus_env import PLUS,isolated_child_environment
from .parallel_continue import process_identity


def subset(selection):
    cases=selection['selected'];chosen=set(i for i in range(len(cases)) if i%10<5);repairs=[]
    universe={(c['mother_suite'],c['mother_task']) for c in cases}
    def mother(i): return cases[i]['mother_suite'],cases[i]['mother_task']
    for missing in sorted(universe-{mother(i) for i in chosen}):
        candidate=next(i for i in range(len(cases)) if i not in chosen and mother(i)==missing)
        counts=Counter(mother(i) for i in chosen);case=cases[candidate]
        victims=[i for i in chosen if cases[i]['category']==case['category'] and cases[i]['difficulty_level']==case['difficulty_level']
            and cases[i]['mother_suite']==case['mother_suite'] and counts[mother(i)]>1]
        if not victims: raise ValueError('Cannot repair coverage while preserving exact cell/suite balance')
        victim=max(victims);chosen.remove(victim);chosen.add(candidate);repairs.append(dict(remove_source_index=victim,add_source_index=candidate,missing_mother=missing))
    if len(chosen)!=175 or len({mother(i) for i in chosen})!=40: raise ValueError('Half subset coverage/count mismatch')
    counts=Counter((cases[i]['category'],cases[i]['difficulty_level']) for i in chosen)
    if len(counts)!=35 or set(counts.values())!={5}: raise ValueError('Half subset must have5/cell')
    original=[suite*10+index for suite in range(4) for index in range(5)]
    return sorted(chosen),original,repairs


def modes(case): return ['full_d10','v14_d10',case['matching_draft']]


def source_results(root,group,index,case,mode,parent_hash):
    paths=list((Path(root)/group/('case%04d'%index)/mode).glob('attempt*/result.json'));valid=[];technical=[]
    for path in paths:
        value=json.loads(path.read_text())
        if value.get('status')=='complete':
            if value.get('manifest_hash')!=parent_hash or value.get('case_hash')!=digest(case) or value.get('mode')!=mode or type(value.get('success')) is not bool:
                raise ValueError('Completed source result identity mismatch: '+str(path))
            valid.append(dict(path=str(path),sha256=sha(path),success=value['success'],steps=value['steps'],gpu_uuid=value.get('gpu_uuid')))
        elif value.get('status')=='technical_failure': technical.append(str(path))
    if len(valid)>1: raise ValueError('Duplicate complete condition: '+str(paths))
    return valid,technical


def register(source_root,output):
    source_root=Path(source_root);protocol=protocol_identity(source_root/'manifest.json');selection=protocol['case_selection']
    perturbed,original,repairs=subset(selection);jobs=[]
    for group,key,indices in [('perturbed350','selected',perturbed),('original40','original_baselines',original)]:
        for index in indices:
            case=selection[key][index];jobs.append(dict(group=group,source_index=index,case=case,case_hash=digest(case),modes=modes(case)))
    manifest=dict(schema=1,kind='native_LIBERO_Plus_half_subset',parent_manifest_hash=protocol['manifest_hash'],
        parent_manifest_path=str(source_root/'manifest.json'),parent_manifest_sha256=sha(source_root/'manifest.json'),source_results_root=str(source_root),
        perturbed_case_count=175,original_case_count=20,condition_count=585,jobs=jobs,coverage_repairs=repairs,
        selection_rule='Outcome-blind: first5 of original frozen10 per actualcategory/difficulty cell; add missingmother via earliest unselected sourceindex, replacing largest selectedindex with repeatedmother in samecell/suite. Original first5 per suite.',
        selection_uses_outcomes=False,selected_mother_count=40,perturbed_suite_counts=dict(Counter(selection['selected'][i]['mother_suite'] for i in perturbed)),
        source_files={str(Path(__file__)):sha(__file__)},runtime='Unchanged frozen run_plus child/model/observations/language/noise/action contract; originala416 service identity retained',
        schedule='Fixed category×difficulty roundrobin across halfsubset, then originalsuite roundrobin; completedcases reused, case3modes sameGPU',
        historical_full_matrix_preserved=True)
    manifest['manifest_hash']=digest(manifest);output=Path(output);output.mkdir(parents=True,exist_ok=False);write_json(output/'manifest.json',manifest)
    ledger=audit(manifest,output);write_json(output/'initial_ledger.json',ledger);return manifest,ledger


def audit(manifest,output):
    parent=manifest['parent_manifest_hash'];source=manifest['source_results_root'];selected={};rows=[];extras=[];technical=[]
    for job in manifest['jobs']:
        for mode in job['modes']:
            uid=(job['group'],job['source_index'],mode);selected[uid]=job
            old,old_technical=source_results(source,*uid[:2],job['case'],mode,parent)
            new,new_technical=source_results(output,*uid[:2],job['case'],mode,parent)
            if old and new: raise ValueError('Half dispatcher duplicated historical completed condition')
            if old or new:
                rows.append(dict(group=uid[0],source_index=uid[1],mode=mode,reused_historical=bool(old),**(old or new)[0]))
            technical+=old_technical+new_technical
    protocol=protocol_identity(manifest['parent_manifest_path'])
    for group,key in [('perturbed350','selected'),('original40','original_baselines')]:
        for index,case in enumerate(protocol['case_selection'][key]):
            for mode in modes(case):
                if (group,index,mode) not in selected:
                    found,_=source_results(source,group,index,case,mode,parent)
                    if found: extras.append(dict(group=group,source_index=index,mode=mode,**found[0]))
    return dict(status='audit_complete',half_manifest_hash=manifest['manifest_hash'],parent_manifest_hash=parent,target=585,
        completed=len(rows),remaining=585-len(rows),reused_historical=sum(row['reused_historical'] for row in rows),
        selected_completed=rows,extra_historical_completed=extras,extra_historical_count=len(extras),technical_attempts=technical,
        by_mode={mode:dict(completed=sum((row['mode'].startswith('draft_') if mode=='matching_draft' else row['mode']==mode) for row in rows),
            success=sum(row['success'] for row in rows if (row['mode'].startswith('draft_') if mode=='matching_draft' else row['mode']==mode))) for mode in ('full_d10','v14_d10','matching_draft')},audited_unix=time.time())


def dispatch(a,manifest):
    output=Path(a.output);parent=manifest['parent_manifest_hash'];verify_service(a.port,parent,'libero_transfer')
    protocol=protocol_identity(manifest['parent_manifest_path'])
    for name,expected in protocol['source_files'].items():
        path=Path(name);path=path if path.is_absolute() else ROOT/path
        if sha(path)!=expected: raise ValueError('Frozen parent source changed: '+str(path))
    claims=output/'claims';claims.mkdir(exist_ok=True);identity=process_identity(os.getpid())
    receipt=dict(status='running',pid=os.getpid(),controller_start_ticks=identity['start_ticks'],gpu=a.gpu,gpu_uuid=a.gpu_uuid,
        port=a.port,half_manifest_hash=manifest['manifest_hash'],parent_manifest_hash=parent,deadline_unix=a.deadline_unix,source_sha256=sha(__file__),results=[])
    receipt_path=output/('worker_gpu%d_%d.json'%(a.gpu,os.getpid()));write_json(receipt_path,receipt)
    perturbed=[job for job in manifest['jobs'] if job['group']=='perturbed350'];original=[job for job in manifest['jobs'] if job['group']=='original40']
    categories=list(dict.fromkeys(job['case']['category'] for job in perturbed));ordered=[]
    for rank in range(5):
        for difficulty in range(1,6):
            for category in categories:
                cell=[job for job in perturbed if job['case']['category']==category and job['case']['difficulty_level']==difficulty]
                ordered.append(cell[rank])
    ordered+=sorted(original,key=lambda job:(job['source_index']%10,job['source_index']//10))
    try:
        for job in ordered:
            if time.time()>=a.deadline_unix: receipt['status']='deadline_reached';break
            case=job['case'];pending=[];completed_devices=set()
            for mode in job['modes']:
                old,_=source_results(manifest['source_results_root'],job['group'],job['source_index'],case,mode,parent)
                new,_=source_results(output,job['group'],job['source_index'],case,mode,parent)
                if old and new: raise ValueError('Duplicate selected condition across old/newresults')
                if old or new: completed_devices.add((old or new)[0]['gpu_uuid'])
                else: pending.append(mode)
            if not pending: continue
            if len(completed_devices)>1: raise ValueError('Historical case modes executed on multiple devices')
            if completed_devices and a.gpu_uuid not in completed_devices: continue
            claim=claims/('%s_%04d'%(job['group'],job['source_index']))
            try: claim.mkdir()
            except FileExistsError: continue
            write_json(claim/'owner.json',dict(pid=os.getpid(),start_ticks=identity['start_ticks'],gpu_uuid=a.gpu_uuid,
                half_manifest_hash=manifest['manifest_hash'],case_hash=job['case_hash'],created_unix=time.time()))
            for mode in pending:
                base=output/job['group']/('case%04d'%job['source_index'])/mode;base.mkdir(parents=True,exist_ok=True)
                indices=[int(path.name[7:]) for path in base.glob('attempt*') if path.is_dir() and path.name[7:].isdigit()]
                directory=base/('attempt%03d'%(max(indices,default=-1)+1))
                env=isolated_child_environment(case,output/('config_original_gpu%d'%a.gpu if job['group']=='original40' else 'config_plus_gpu%d'%a.gpu),a.gpu,a.gpu_uuid)
                interpreter=ROOT/'policies/libero-eval-v21/.venv/bin/python' if job['group']=='original40' else PLUS/'.venv/bin/python'
                command=[str(interpreter),'-m','preparations.idea1_v3_2_e4_supplement.run_plus','--protocol',manifest['parent_manifest_path'],
                    '--output',str(directory),'--port',str(a.port),'--gpu',str(a.gpu),'--gpu-uuid',a.gpu_uuid,'--deadline-unix',str(a.deadline_unix),
                    '--child-index',str(job['source_index']),'--child-mode',mode,'--launch']
                if job['group']=='original40': command.append('--baseline')
                if job['source_index']==0: command.append('--video')
                row=dict(group=job['group'],source_index=job['source_index'],mode=mode,output=str(directory),command=command,started_unix=time.time())
                with (base/(directory.name+'.log')).open('w') as log:
                    process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);row['pid']=process.pid
                    receipt['active']=row;write_json(receipt_path,receipt)
                    try: code=process.wait(timeout=max(1,a.deadline_unix-time.time())+15)
                    except subprocess.TimeoutExpired:
                        process.terminate()
                        try: process.wait(timeout=15)
                        except subprocess.TimeoutExpired: process.kill();process.wait()
                        code=124
                row.update(exit_code=code,reaped=True,completed_unix=time.time());receipt['results'].append(row);receipt.pop('active',None)
                write_json(receipt_path,receipt)
                if code: raise RuntimeError('Halfcondition technical/deadlinefailure preserved; not policyzero')
            write_json(claim/'complete.json',dict(status='complete',gpu_uuid=a.gpu_uuid,completed_unix=time.time()))
        else: receipt['status']='queue_scanned'
    except Exception:
        import traceback
        receipt.update(status='technical_failure',error=traceback.format_exc());raise
    finally:
        receipt['completed_unix']=time.time();write_json(receipt_path,receipt)
        write_json(output/('ledger_gpu%d_%d.json'%(a.gpu,os.getpid())),audit(manifest,output))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root');p.add_argument('--output',required=True);p.add_argument('--register',action='store_true')
    p.add_argument('--audit',action='store_true');p.add_argument('--gpu',type=int,default=3);p.add_argument('--gpu-uuid');p.add_argument('--port',type=int,default=19143)
    p.add_argument('--deadline-unix',type=float);p.add_argument('--launch',action='store_true');a=p.parse_args()
    if a.register:
        if not a.source_root: p.error('--source-root required')
        manifest,ledger=register(a.source_root,a.output);print(json.dumps(dict(manifest_hash=manifest['manifest_hash'],target=585,reused=ledger['reused_historical'],extra_historical=ledger['extra_historical_count'])));return
    manifest=json.loads((Path(a.output)/'manifest.json').read_text())
    if manifest['manifest_hash']!=digest({k:v for k,v in manifest.items() if k!='manifest_hash'}): raise ValueError('Half manifesthash mismatch')
    if a.audit:
        ledger=audit(manifest,a.output);write_json(Path(a.output)/'live_ledger.json',ledger)
        print(json.dumps({key:ledger[key] for key in ('target','completed','remaining','reused_historical','extra_historical_count','by_mode')}));return
    if not a.launch: print(json.dumps(dict(dry_run=True,manifest_hash=manifest['manifest_hash'],target=585)));return
    if not a.gpu_uuid or not a.deadline_unix: p.error('AssignedGPUUUID/deadline required')
    dispatch(a,manifest)


if __name__=='__main__': main()

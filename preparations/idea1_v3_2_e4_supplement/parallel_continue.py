"""Receipt-gated two-environment B1 service pool; frozen scientific child.

Safe handoff stops only validated owned dispatchers, lets their current episode
finish, then terminates those dispatchers. All completed episodes are reused.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from .native_envs import ROOT,digest,sha,write_json
from .evaluation import protocol_identity,verify_service
from .libero_plus_env import isolated_child_environment,PLUS


def process_identity(pid):
    root=Path('/proc')/str(pid)
    if not root.exists(): return None
    try:
        stat=(root/'stat').read_text();fields=stat[stat.rfind(')')+2:].split()
        argv=(root/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        try: cwd=str((root/'cwd').resolve())
        except FileNotFoundError: cwd=None
        return dict(pid=pid,state=fields[0],ppid=int(fields[1]),start_ticks=fields[19],argv=argv,cwd=cwd)
    except FileNotFoundError: return None


def same_process(identity):
    current=process_identity(identity['pid'])
    return current is not None and current['start_ticks']==identity['start_ticks']


def handoff(a,directory):
    outer=process_identity(a.old_outer_pid);dispatcher=process_identity(a.old_dispatcher_pid)
    if not outer or not dispatcher or 'supplement.plus_continue ' not in outer['argv'] or 'supplement.run_plus ' not in dispatcher['argv']:
        raise ValueError('Old process identity mismatch; no signal sent')
    if dispatcher['ppid']!=outer['pid'] or outer['cwd']!=str(ROOT) or dispatcher['cwd']!=str(ROOT):
        raise ValueError('Owned process ancestry/cwd mismatch')
    row=dict(status='stopping_dispatch_only',outer=outer,dispatcher=dispatcher,signals=[],children=[])
    write_json(directory/'handoff.json',row)
    # Freeze redispatch, not the episode. No signal is sent to renderer child.
    for identity in (outer,dispatcher):
        if not same_process(identity): raise ValueError('PID reused before STOP')
        os.kill(identity['pid'],signal.SIGSTOP);row['signals'].append(dict(pid=identity['pid'],signal='SIGSTOP',unix=time.time()))
    time.sleep(.1)
    children_file=Path('/proc')/str(dispatcher['pid'])/'task'/str(dispatcher['pid'])/'children'
    children=[int(value) for value in children_file.read_text().split()]
    try:
        for pid in children:
            identity=process_identity(pid)
            if not identity or 'supplement.run_plus ' not in identity['argv'] or '--child-index ' not in identity['argv']:
                raise ValueError('Unexpected old dispatcher child; preserve stopped controllers for operator review')
            row['children'].append(identity)
        write_json(directory/'handoff.json',row)
        while any(same_process(identity) and process_identity(identity['pid'])['state']!='Z' for identity in row['children']):
            if time.time()>=a.deadline_unix: raise TimeoutError('Episode completion handoff deadline')
            time.sleep(.2)
        for identity in row['children']:
            argv=identity['argv'].split();output=Path(argv[argv.index('--output')+1])
            path=output/'result.json'
            value=json.loads(path.read_text()) if path.exists() else {}
            row.setdefault('child_results',[]).append(dict(pid=identity['pid'],path=str(path),status=value.get('status'),success=value.get('success')))
            if value.get('status')!='complete': raise RuntimeError('Current old episode failed technically; do not silently replace old queue')
    except Exception:
        for identity in (dispatcher,outer):
            if same_process(identity): os.kill(identity['pid'],signal.SIGCONT)
        row['status']='handoff_failed_controllers_resumed';write_json(directory/'handoff.json',row);raise
    # Stopped Python processes must be continued after TERM. Current env child
    # has already completed its close()/result, and is never terminated here.
    for identity in (outer,dispatcher):
        if same_process(identity):
            os.kill(identity['pid'],signal.SIGTERM);os.kill(identity['pid'],signal.SIGCONT)
            row['signals'].extend([dict(pid=identity['pid'],signal='SIGTERM',unix=time.time()),dict(pid=identity['pid'],signal='SIGCONT',unix=time.time())])
    until=time.time()+15
    while any(same_process(identity) and process_identity(identity['pid'])['state']!='Z' for identity in (outer,dispatcher)):
        if time.time()>until: raise RuntimeError('Owned old dispatchers did not exit after TERM+CONT')
        time.sleep(.1)
    row.update(status='safe_handoff_complete',completed_unix=time.time(),
        final_states={str(identity['pid']):process_identity(identity['pid']) for identity in (outer,dispatcher)},
        renderer_child_signals_sent=False,reaping_note='Outer exec session remains owned by tool launcher; finished child zombies are reaped by their parent/init, not falsely waitpid-owned here')
    write_json(directory/'handoff.json',row);return row


def completed(base,case,mode,identity):
    rows=[]
    for path in base.glob('attempt*/result.json'):
        value=json.loads(path.read_text())
        if value.get('status')=='complete':
            if value.get('manifest_hash')!=identity or value.get('case_hash')!=digest(case) or value.get('mode')!=mode or type(value.get('success')) is not bool:
                raise ValueError('Completed case identity mismatch: '+str(path))
            rows.append(path)
    if len(rows)>1: raise ValueError('Duplicate completed case: '+str(base))
    return rows


def run_pool(a,protocol,directory):
    root=Path(a.root);selection=protocol['case_selection'];identity=protocol['manifest_hash'];claims=directory/'claims';claims.mkdir()
    # Fixed category round-robin independent of observed success/speed. Every
    # case retains its Full/V14/mother-draft order on the same assigned GPU.
    perturbed=selection['selected'];categories=list(dict.fromkeys(case['category'] for case in perturbed))
    by_category={category:[i for i,case in enumerate(perturbed) if case['category']==category] for category in categories}
    order=[i for offset in range(max(map(len,by_category.values()))) for category in categories for i in by_category[category][offset:offset+1]]
    jobs=[dict(group='perturbed350',index=i,case=perturbed[i],baseline=False) for i in order]
    originals=selection['original_baselines'];original_order=[suite*10+index for index in range(10) for suite in range(4)]
    jobs+=[dict(group='original40',index=i,case=originals[i],baseline=True) for i in original_order]
    receipt=dict(status='running',manifest_hash=identity,pid=os.getpid(),workers=2,model_batching=False,
        source_sha256=sha(__file__),job_order=[dict(group=j['group'],index=j['index']) for j in jobs],
        scheduling='Fixed category roundrobin for all350, then frozen original40roundrobin; case3modes stay one worker/GPU',
        results=[],active=[])
    write_json(directory/'pool.json',receipt);active=[];failed=False
    def launch(job,mode):
        case=job['case'];base=root/job['group']/('case%04d'%job['index'])/mode
        if completed(base,case,mode,identity):
            receipt['results'].append(dict(group=job['group'],index=job['index'],mode=mode,reused=True));return False
        claim=claims/('%s_%04d_%s'%(job['group'],job['index'],mode));claim.mkdir(exist_ok=False)
        write_json(claim/'owner.json',dict(pid=os.getpid(),controller_start=process_identity(os.getpid())['start_ticks'],manifest_hash=identity,created_unix=time.time()))
        if completed(base,case,mode,identity): raise ValueError('Case completed during claim; possible foreign dispatcher race')
        base.mkdir(parents=True,exist_ok=True)
        indices=[int(path.name[7:]) for path in base.glob('attempt*') if path.is_dir() and path.name[7:].isdigit()]
        output=base/('attempt%03d'%(max(indices,default=-1)+1))
        env=isolated_child_environment(case,root/job['group']/('config_original' if job['baseline'] else 'config_plus'),3,a.gpu_uuid)
        interpreter=ROOT/'policies/libero-eval-v21/.venv/bin/python' if job['baseline'] else PLUS/'.venv/bin/python'
        command=[str(interpreter),'-m','preparations.idea1_v3_2_e4_supplement.run_plus','--protocol',str(root/'manifest.json'),
            '--output',str(output),'--port',str(a.port),'--gpu','3','--gpu-uuid',a.gpu_uuid,'--deadline-unix',str(a.deadline_unix),
            '--child-index',str(job['index']),'--child-mode',mode,'--launch']
        if job['baseline']: command.append('--baseline')
        if job['index']==0: command.append('--video')
        log=(base/(output.name+'.log')).open('w');process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        active.append(dict(job=job,mode=mode,process=process,log=log,output=output,claim=claim,started_unix=time.time(),command=command))
        return True
    pending=list(jobs);case_modes={};case_started=set()
    try:
        while pending or active or any(case_modes.values()):
            # Finish a case's remaining modes before taking a new case. This
            # keeps two independent simulator slots, never >2 render children.
            while len(active)<2 and not failed and time.time()<a.deadline_unix:
                continuation=next((key for key,modes in case_modes.items() if modes and key not in [slot['job']['group']+str(slot['job']['index']) for slot in active]),None)
                if continuation is not None:
                    job,modes=case_modes[continuation][0],case_modes[continuation][1:]
                    # Stored value [job,mode,...]; pop one mode only.
                    mode=modes[0];case_modes[continuation]=[job]+modes[1:] if len(modes)>1 else []
                elif pending:
                    job=pending.pop(0);key=job['group']+str(job['index']);mode='full_d10'
                    case_modes[key]=[job,'v14_d10',job['case']['matching_draft']]
                else: break
                launch(job,mode)
            receipt['active']=[dict(group=slot['job']['group'],index=slot['job']['index'],mode=slot['mode'],pid=slot['process'].pid,output=str(slot['output'])) for slot in active]
            write_json(directory/'pool.json',receipt)
            if not active:
                if failed or time.time()>=a.deadline_unix: break
                continue
            time.sleep(.2)
            for slot in list(active):
                code=slot['process'].poll()
                if code is None: continue
                slot['process'].wait();slot['log'].close();active.remove(slot)
                path=slot['output']/'result.json';value=json.loads(path.read_text()) if path.exists() else {}
                row=dict(group=slot['job']['group'],index=slot['job']['index'],mode=slot['mode'],pid=slot['process'].pid,
                    exit_code=code,reaped=True,status=value.get('status'),success=value.get('success'),result=str(path),wall_s=time.time()-slot['started_unix'])
                write_json(slot['claim']/'completion.json',row);receipt['results'].append(row)
                if code or value.get('status')!='complete': failed=True
            if time.time()>=a.deadline_unix and active:
                # Frozen child checks every replan and owns renderer cleanup;
                # permit it to reach its same bounded deadline, then drain.
                if time.time()>a.deadline_unix+30:
                    for slot in active:
                        slot['process'].terminate()
                    failed=True
        receipt.update(status='technical_failure' if failed else ('deadline_reached' if pending or any(case_modes.values()) else 'complete'),completed_unix=time.time())
    finally:
        for slot in active:
            if slot['process'].poll() is None:
                slot['process'].terminate()
                try: slot['process'].wait(timeout=15)
                except subprocess.TimeoutExpired: slot['process'].kill();slot['process'].wait()
            slot['log'].close()
        write_json(directory/'pool.json',receipt)
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--profile-receipt',required=True)
    p.add_argument('--old-outer-pid',type=int,required=True);p.add_argument('--old-dispatcher-pid',type=int,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--port',type=int,default=19143);p.add_argument('--deadline-unix',type=float,required=True)
    p.add_argument('--launch',action='store_true');a=p.parse_args();profile=json.loads(Path(a.profile_receipt).read_text());protocol=protocol_identity(Path(a.root)/'manifest.json')
    if profile.get('status')!='complete' or profile.get('deploy_parallel') is not True or profile.get('manifest_hash')!=protocol['manifest_hash']:
        raise ValueError('Passed completed profile receipt required; sequential queue unchanged')
    if not a.launch: print(json.dumps(dict(dry_run=True,workers=2,model_batching=False)));return
    verify_service(a.port,protocol['manifest_hash'],'libero_transfer')
    directory=Path(a.root)/('parallel_execution_%d'%int(time.time()));directory.mkdir(exist_ok=False)
    write_json(directory/'authorization.json',dict(profile_receipt=a.profile_receipt,profile_sha256=sha(a.profile_receipt),manifest_hash=protocol['manifest_hash'],
        scope='Parent approved conditional two-worker only after strictgate +robust20percent profile',source_sha256=sha(__file__),deadline_unix=a.deadline_unix))
    handoff(a,directory);result=run_pool(a,protocol,directory);print(json.dumps(dict(status=result['status'],results=len(result['results']))))


if __name__=='__main__': main()

"""Resumable campaign supervisor; never stops processes belonging to others."""
import json
from collections import defaultdict
import os
from pathlib import Path
import shutil
import subprocess
import time
import fcntl
import socket
import re
from preparations.idea1_v3.io import atomic,append
from preparations.idea1_v3_1.common import environment
from preparations.idea1_v3.cli import verify_gpu
from .dispatch import ROOT,OUT,PY
from .analyze import run as analyze,summary
from .selection import first80,promote
from .annotate import case as annotate_case
from .manifest import cases


def alive(pid):
    try:
        command=Path(f'/proc/{pid}/cmdline').read_bytes()
        return b'preparations.idea1_v3_3.' in command and Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except (FileNotFoundError,ProcessLookupError): return False


def launch(gpu,model,manifest=None,shard=0,shards=1):
    leases=OUT/'external_gpu_leases.json'
    if leases.exists() and json.loads(leases.read_text()).get(str(gpu),{}).get('active'): raise RuntimeError('GPU reserved by external authorized lease')
    stamp=str(time.time_ns()); wd=OUT/'workers'/f'gpu{gpu}'; history=wd/'history'/stamp
    history.mkdir(parents=True,exist_ok=True)
    for name in ('identity.json','loaded_protocol.json','finished.json'):
        if (wd/name).exists(): shutil.copy2(wd/name,history/name)
    operation='stage' if manifest else 'dispatch_bound'
    command=[str(PY['pi0']),'-m','preparations.idea1_v3_3.'+operation,'--gpu',str(gpu),'--model',model]
    if manifest: command+=['--manifest',str(manifest),'--shard',str(shard),'--shards',str(shards)]
    logpath=OUT/f'campaign_gpu{gpu}_{stamp}.log'
    with logpath.open('a') as log: p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    append(OUT/'campaign_events.jsonl',dict(event='launch',gpu=gpu,model=model,pid=p.pid,manifest=str(manifest) if manifest else None,log=str(logpath)))
    return p.pid


def allocate(conditions,available,groups,target):
    work={}
    for model in ('pi0','xiaomi'):
        own=[c for c in conditions if c['model']==model]
        historical=[r['wall_s'] for rows in groups.values() for r in rows if r['config']['model']==model]
        cost=sum(historical)/len(historical) if historical else 12.
        expected={tuple(case[k] for k in ('suite','task','init')) for case in cases(model,expand=target==300)}
        work[model]=sum(len(expected-{tuple(r['case'][k] for k in ('suite','task','init')) for r in groups[c['id']]})*cost for c in own)
    if len(available)<sum(value>0 for value in work.values()):raise RuntimeError('Not enough authorized GPU leases for concurrent model stage; state preserved, sequential handoff required before resuming')
    if not work['xiaomi']:split=len(available)
    elif not work['pi0']:split=0
    else:split=max(1,min(len(available)-1,round(len(available)*work['pi0']/sum(work.values()))))
    return (('pi0',available[:split]),('xiaomi',available[split:])),work


def covered(groups,conditions,case_sets):
    for cfg in conditions:
        expected={tuple(c[k] for k in ('suite','task','init')) for c in case_sets[cfg['model']]}
        actual={tuple(r['case'][k] for k in ('suite','task','init')) for r in groups[cfg['id']]}
        if not expected<=actual:return False
    return True


def check_stage_exits(state):
    for device,worker in state['workers'].items():
        if alive(worker['pid']):continue
        wd=OUT/'stages'/state['phase']/f'gpu{device}'
        identity=wd/'identity.json';finished=wd/'finished.json'
        if not identity.exists() or json.loads(identity.read_text()).get('pid')!=worker['pid'] or not finished.exists() or finished.stat().st_mtime<identity.stat().st_mtime:
            raise RuntimeError(f"Stage worker {worker['pid']} GPU{device} exited without a normal finished receipt; preserve cases and diagnose/recover the same model")


def remaining_model_held(model,manifest,groups,workers):
    holders=[]
    for cfg in manifest['conditions']:
        if cfg['model']!=model or len(groups[cfg['id']])>=80:continue
        directory=OUT/'conditions'/cfg['id'];path=directory/'running.json'
        if not path.exists():return False,holders
        running=json.loads(path.read_text());pid=running['pid']
        if not any(w['pid']==pid and w['model']==model and alive(pid) for w in workers.values()):return False,holders
        with (directory/'owner.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);return False,holders
            except BlockingIOError:pass
        holders.append(dict(condition=cfg['id'],pid=pid,valid=len(groups[cfg['id']]),condition_lock_held=True))
    return True,holders


def retired_gpu(gpu,port):
    text=subprocess.check_output(['nvidia-smi','-i',str(gpu),'-q','-d','PIDS,MEMORY,UTILIZATION'],text=True)
    if not re.search(r'Processes\s*:\s*None',text):raise RuntimeError('GPU still has C/G contexts; do not start a new model')
    with socket.socket() as sock:
        # Match the actual server: a closed RPC connection may remain in
        # TIME_WAIT after the listener and GPU contexts have fully exited.
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(('127.0.0.1',port))
    return text


def main():
    state_path=OUT/'campaign_state.json'
    if state_path.exists(): state=json.loads(state_path.read_text())
    else: state=dict(phase='first80',workers={str(g):dict(pid=p,model=m) for g,p,m in ((0,32615,'pi0'),(1,32616,'pi0'),(2,32617,'xiaomi'),(3,32618,'xiaomi'))})
    cached={}
    def completed_records():
        for complete in OUT.glob('*/episodes/*/task*/init*/complete.json'):
            if '.invalid.' in str(complete) or str(complete) in cached: continue
            path=complete.with_name('result.json')
            r=json.loads(path.read_text()); r['_directory']=str(path.parent); annotate_case(r); cached[str(complete)]=r
        groups=defaultdict(list)
        for row in cached.values(): groups[row['condition']].append(row)
        return groups
    while True:
        try:
            previous_phase=state['phase']
            transitioning=set()
            for path in (OUT/'worker_handoffs').glob('*.json'):
                handoff=json.loads(path.read_text()); device=str(handoff['gpu']); current=state['workers'].get(device)
                if current and current['pid']==handoff['old_pid']:
                    if handoff.get('new_worker'): state['workers'][device]=handoff['new_worker']
                    elif time.time()<handoff['expires']: transitioning.add(device)
                    else: raise TimeoutError('Worker handoff did not finish: '+str(path))
            groups=completed_records(); m=json.loads((OUT/'manifest.json').read_text())
            failures=list((OUT/'conditions').glob('*/failed.json'))
            if failures:
                atomic(OUT/'campaign_alert.json',dict(time=time.time(),status='technical_failure_needs_diagnosis',files=[str(p) for p in failures])); print('ALERT technical_failure',flush=True); return
            if state['phase']=='first80':
                if covered(groups,m['conditions'],m['cases']):
                    if any(alive(w['pid']) for w in state['workers'].values()): time.sleep(5); continue
                    manifest=first80()
                    if manifest is None:
                        state.update(status='waiting_cost_calibration',time=time.time(),complete=sum(len(v) for v in groups.values()))
                        atomic(state_path,state);print('WAITING_COST_CALIBRATION: all scientific first80 preserved; automatic retry in30s',flush=True);time.sleep(30);continue
                    state.update(phase='bounded_followups',stage_manifest=str(manifest),workers={})
                    extra=json.loads(manifest.read_text())
                    leases=json.loads((OUT/'external_gpu_leases.json').read_text()) if (OUT/'external_gpu_leases.json').exists() else {}
                    available=[g for g in range(4) if not leases.get(str(g),{}).get('active')]
                    allocation,work=allocate(extra['conditions'],available,groups,80)
                    if any(c['model']=='xiaomi' for c in extra['conditions']) and any(c['model']=='pi0' for c in extra['conditions']):
                        # One exclusive Xiaomi service also owns its serial
                        # engineering pure80; remaining cards shard Pi cases.
                        allocation=(('pi0',available[:-1]),('xiaomi',available[-1:]))
                    append(OUT/'campaign_events.jsonl',dict(event='stage_case_allocation',stage='bounded_followups',estimated_work_s=work,allocation=allocation))
                    for model,gpus in allocation:
                        configs=[c for c in extra['conditions'] if c['model']==model]
                        needed=gpus if configs else []
                        for index,gpu in enumerate(needed): state['workers'][str(gpu)]=dict(pid=launch(gpu,model,manifest,index,len(needed)),model=model)
                else:
                    pi_todo=any(len(groups[c['id']])<80 and not (OUT/'conditions'/c['id']/'failed.json').exists() for c in m['conditions'] if c['model']=='pi0')
                    for device,w in list(state['workers'].items()):
                        if device in transitioning: continue
                        if not alive(w['pid']):
                            gpu=int(device)
                            wd=OUT/'workers'/f'gpu{gpu}'
                            identity=json.loads((wd/'identity.json').read_text())
                            finished=wd/'finished.json'
                            normal=finished.exists() and identity.get('pid')==w['pid'] and finished.stat().st_mtime >= (wd/'identity.json').stat().st_mtime
                            if not normal:
                                atomic(OUT/'campaign_alert.json',dict(time=time.time(),status='unexpected_worker_exit_needs_same_model_recovery',gpu=gpu,model=w['model'],pid=w['pid']))
                                print('ALERT unexpected_worker_exit',flush=True); return
                            model_complete=all(len(groups[c['id']])>=80 for c in m['conditions'] if c['model']==w['model'])
                            held,holders=remaining_model_held(w['model'],m,groups,state['workers'])
                            if not pi_todo or w['model']=='pi0' or not (model_complete or held): continue
                            retired=retired_gpu(gpu,9340+gpu)
                            atomic(OUT/f'gpu{gpu}_xiaomi_to_pi0_handoff.json',dict(time=time.time(),gpu=gpu,old_worker_pid=w['pid'],normal_finished=True,remaining_Xiaomi_holders=holders,
                                old_server_and_render_exited=True,port_free=9340+gpu,nvidia_smi=retired,runner_label_note='resume driver used v1 label; actual EGL mapping conversion/parity/sidecars are v2',
                                preflight='No C/G contexts and old RPC port unbound; retain external GPU1 idea2 lease'))
                            if w['model']=='xiaomi' and not (OUT/'xiaomi_full_native_probe.json').exists():
                                verify_gpu(gpu)
                                receipt=OUT/'xiaomi_full_native_probe.json'
                                env=environment(gpu)
                                probe=[str(PY['xiaomi']),'-m','preparations.idea1_v3_3.full_probe','--input',str(OUT/'xiaomi_full_d2/episodes/libero_spatial/task00/init00/input0.npz'),'--output',str(receipt)]
                                with (OUT/'xiaomi_full_native_probe.log').open('a') as log: subprocess.run(probe,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                                if not json.loads(receipt.read_text())['passed']: raise ValueError('Xiaomi Full equivalence failed')
                                for attempt in range(20):
                                    try:retired_gpu(gpu,9340+gpu);break
                                    except RuntimeError:
                                        if attempt==19:raise
                                        time.sleep(.5)
                            # Only the current campaign's own finished worker
                            # releases the resource. launch verifies ownership.
                            state['workers'][device]=dict(pid=launch(gpu,'pi0'),model='pi0')
            elif state['phase']=='bounded_followups':
                check_stage_exits(state)
                extra=json.loads(Path(state['stage_manifest']).read_text())
                if covered(groups,extra['conditions'],extra['cases']) and not any(alive(w['pid']) for w in state['workers'].values()):
                    manifest=promote()
                    if manifest is None:
                        state.update(status='waiting_cost_calibration',time=time.time());atomic(state_path,state);print('WAITING_COST_CALIBRATION: bounded data preserved; automatic retry in30s',flush=True);time.sleep(30);continue
                    state.update(phase='confirm300',stage_manifest=str(manifest),workers={})
                    confirmation=json.loads(manifest.read_text())
                    leases=json.loads((OUT/'external_gpu_leases.json').read_text()) if (OUT/'external_gpu_leases.json').exists() else {}
                    available=[g for g in range(4) if not leases.get(str(g),{}).get('active')]
                    allocation,work=allocate(confirmation['conditions'],available,groups,300)
                    append(OUT/'campaign_events.jsonl',dict(event='stage_case_allocation',stage='confirm300',estimated_work_s=work,allocation=allocation))
                    for model,gpus in allocation:
                        configs=[c for c in confirmation['conditions'] if c['model']==model]
                        needed=gpus if configs else []
                        for index,gpu in enumerate(needed): state['workers'][str(gpu)]=dict(pid=launch(gpu,model,manifest,index,len(needed)),model=model)
            elif state['phase']=='confirm300':
                check_stage_exits(state)
                confirmation=json.loads(Path(state['stage_manifest']).read_text())
                if covered(groups,confirmation['conditions'],confirmation['cases']) and not any(alive(w['pid']) for w in state['workers'].values()):
                    state['phase']='scientific_complete'
            state.update(time=time.time(),complete=sum(len(v) for v in groups.values()))
            atomic(state_path,state)
            if state['phase']!=previous_phase or state['phase']=='scientific_complete': analyze()
            else:
                atomic(OUT/'progress.json',dict(time=time.time(),total_complete=state['complete'],phase=state['phase'],conditions={cid:summary(rows) for cid,rows in groups.items()},
                    scope='Incremental completion stats; paired bootstrap and full routing recomputed only at stage boundaries'))
            print(json.dumps(dict(phase=state['phase'],complete=state['complete'],active={g:w for g,w in state['workers'].items() if alive(w['pid'])})),flush=True)
            if state['phase']=='scientific_complete': return
        except Exception as exc:
            atomic(OUT/'campaign_alert.json',dict(time=time.time(),error=repr(exc),phase=state.get('phase'))); raise
        time.sleep(30)


if __name__=='__main__': main()

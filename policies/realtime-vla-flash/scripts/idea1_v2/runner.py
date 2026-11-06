"""Budgeted subprocess execution. All CUDA/LIBERO imports remain in child entrypoints."""
from __future__ import annotations
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import numpy as np
from .core import NEW, REPO, SUITES, LIMITS, WEIGHT, SKIP, GPU_DISCOVERY_INTERVAL_SECONDS, Router, noise, rows, sha, write_json

ALT='ALT_V14x2_FULLx1_D3_R5'
PHASE='PHASE_ALT_V14x2_FULLx1_D3_R5'

@contextlib.contextmanager
def locked(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield

class Ledger:
    """Campaign-wide reservations survive new output attempts and process crashes."""
    def __init__(self,path=None):
        self.path=path or NEW/'status/work_ledger.json'
    def reserve(self,key,kind,actions=0,predictions=0,retry=False,fix_note=''):
        with locked(self.path.with_suffix('.lock')):
            data=json.loads(self.path.read_text()) if self.path.exists() else {'entries':[]}
            entries=data['entries'];previous=[e for e in entries if e['key']==key]
            if previous:
                if any(e.get('status')=='valid' for e in previous):raise RuntimeError('Valid key cannot repeat')
                if not retry or not fix_note or len(previous)>1 or previous[-1].get('status')!='invalid':
                    raise RuntimeError('Invalid retry requires explicit fix note, at most one per key')
                if kind!='diagnostic' and sum(e['retry'] and e['kind']!='diagnostic' for e in entries)>=8:raise RuntimeError('Global retry budget exhausted')
            elif retry:
                raise RuntimeError('Cannot retry an unseen key')
            if not retry or kind=='diagnostic':
                cap={'E2':13200,'E3':13200,'baseline':528,'diagnostic':256}[kind]
                used=sum(e['predictions'] for e in entries if e['kind']==kind and (kind=='diagnostic' or not e['retry']))
                if used+predictions>cap:raise RuntimeError('Prediction budget exhausted')
            token='%d-%d'%(time.time_ns(),os.getpid())
            entry={'token':token,'key':key,'kind':kind,'actions':actions,'predictions':predictions,
                   'retry':retry,'fix_note':fix_note,'status':'reserved','pid':os.getpid(),'time':time.time()}
            entry['owner_start_ticks']=(Path('/proc')/str(os.getpid())/'stat').read_text().split(') ',1)[1].split()[19]
            entries.append(entry);write_json(self.path,data)
            return token
    def annotate(self,token,**fields):
        with locked(self.path.with_suffix('.lock')):
            data=json.loads(self.path.read_text())
            entry=next(e for e in data['entries'] if e['token']==token)
            entry.update(fields);write_json(self.path,data)
    def finish(self,token,status,actual_actions=None,actual_predictions=None,**fields):
        with locked(self.path.with_suffix('.lock')):
            data=json.loads(self.path.read_text())
            entry=next(e for e in data['entries'] if e['token']==token)
            if entry['status']!='reserved':raise RuntimeError('Reservation already settled')
            if actual_actions is not None:entry['actions']=actual_actions
            if actual_predictions is not None:entry['predictions']=actual_predictions
            entry.update(fields,status=status);write_json(self.path,data)

    def reconcile(self,fix_note):
        if not fix_note:raise ValueError('Reconciliation requires diagnosis/fix note')
        recovered=[]
        with locked(self.path.with_suffix('.lock')):
            data=json.loads(self.path.read_text())
            for entry in data['entries']:
                if entry['status']!='reserved':continue
                try:
                    fields=(Path('/proc')/str(entry['pid'])/'stat').read_text().split(') ',1)[1].split()
                    live=fields[0] not in ('Z','X') and fields[19]==entry.get('owner_start_ticks')
                except FileNotFoundError:live=False
                if live:continue
                validated=Path(entry.get('directory','/nonexistent'))/'validated.json'
                if validated.exists():
                    result=json.loads(validated.read_text())
                    if result.get('valid') and result.get('directory')==entry.get('directory'):
                        entry.update(status='valid',actions=result['actual_actions'],predictions=result['actual_predictions'],success=result['success'],
                                     reconciliation_note='Recovered durable validation; '+fix_note)
                        recovered.append(entry['token']);continue
                # Conservative reservations retain their full spending charge.
                entry.update(status='invalid',error='dead owner reservation',reconciliation_note=fix_note)
                recovered.append(entry['token'])
            write_json(self.path,data)
        return recovered

def discover():
    root=NEW/'status'
    with locked(root/'discovery.lock'):
        path=root/'gpu_discovery.json'
        last=json.loads(path.read_text()) if path.exists() else {'time':0,'free':[]}
        # A caller may have just discovered an idle card before invoking smoke or
        # launch. Reuse that shared snapshot throughout the cooldown; do not turn
        # "hardware query not due" into "no cards idle". Lease acquisition still
        # serializes our workers, and the next hardware scan remains at +30s.
        if time.time()-last['time']<GPU_DISCOVERY_INTERVAL_SECONDS:
            return list(last['free']),last['time']+GPU_DISCOVERY_INTERVAL_SECONDS
        proc=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],capture_output=True,text=True,check=True)
        devices=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        busy={line.split(',')[0].strip() for line in proc.stdout.splitlines() if line.strip()}
        free=[]
        for line in devices.stdout.splitlines():
            index,uuid,memory=[x.strip() for x in line.split(',')]
            if uuid not in busy and int(memory)<1024:free.append(index)
        scanned_at=time.time()
        write_json(path,{'time':scanned_at,'free':free,'process_snapshot':proc.stdout,
                         'interval_seconds':GPU_DISCOVERY_INTERVAL_SECONDS,
                         'next_scan_time':scanned_at+GPU_DISCOVERY_INTERVAL_SECONDS})
        return free,scanned_at+GPU_DISCOVERY_INTERVAL_SECONDS

@contextlib.contextmanager
def lease(gpu):
    path=NEW/'status/leases'/('gpu%s.lock'%gpu)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield

def env(gpu):
    value=os.environ.copy()
    value.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu),MUJOCO_GL='egl',
                 LIBERO_CONFIG_PATH=str(REPO/'.libero'),JAX_PLATFORMS='cpu',
                 OPENPI_DATA_HOME='/root/nfs/embodied/benchmark/shared/checkpoints/openpi',
                 PYTHONPATH=':'.join([str(REPO/'src'),str(REPO),'/root/nfs/embodied/benchmark/LIBERO']),
                 PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',IDEA1_V2_ORCHESTRATED='1')
    return value

def command(args,action,python=None,**values):
    cmd=[str(python or REPO/'.venv/bin/python'),'-m','scripts.idea1_v2',action,'--launch',
         '--source',str(args.source),'--output',str(args.output)]
    for key,value in values.items():cmd.extend(['--'+key.replace('_','-'),str(value)])
    if action in ('worker','parity') and getattr(args,'retry_invalid',False):
        cmd.extend(['--retry-invalid','--fix-note',args.fix_note])
    return cmd

@contextlib.contextmanager
def server(args,gpu,suite,condition,mode):
    with socket.socket() as reserve:
        reserve.bind(('127.0.0.1',0));port=reserve.getsockname()[1]
    logs=args.output/'logs';logs.mkdir(parents=True,exist_ok=True)
    log=logs/('%s_%s_%d.server.log'%(condition,suite,time.time_ns()))
    with log.open('x') as stream:
        child=subprocess.Popen(command(args,'serve',suite=suite,condition=condition,mode=mode,port=port),cwd=REPO,env=env(gpu),stdout=stream,stderr=subprocess.STDOUT)
        try:
            start=time.monotonic()
            while True:
                if child.poll() is not None:raise RuntimeError('Server exited; see %s'%log)
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=1):break
                except OSError:
                    if time.monotonic()-start>600:raise TimeoutError('Server readiness timeout')
                    time.sleep(2)
            yield port
        finally:
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();child.wait()

def validate(directory,mode,condition,suite,task,init):
    entries=json.loads((directory/'episode_log.json').read_text())
    if len(entries)!=1:raise ValueError('Expected exactly one official episode')
    episode=entries[0]
    if (episode['task_id'],episode['episode_idx'])!=(task,init) or episode.get('failure_reason') or episode.get('episode_error'):
        raise ValueError('Missing identity or episode exception: invalid, never task failure')
    states=rows(directory/episode['telemetry_path']);infer=rows(directory/episode['infer_path'])
    actions=np.asarray([s['action'] for s in states[1:]])
    if [s['step'] for s in states]!=list(range(len(states))) or len(states)!=episode['frame_count']:
        raise ValueError('N+1 state/frame mismatch')
    if len(actions)>LIMITS[suite] or not np.allclose(np.diff([s['sim_time_s'] for s in states]),.05,atol=1e-8,rtol=0):
        raise ValueError('Action limit or control period mismatch')
    router=Router(mode=='phase')
    offset=0;full_actions=0;full_rounds=0
    for r,rec in enumerate(infer):
        meta=rec['idea1_v2'];t=int(rec['frame_idx'])
        if t!=offset or meta['trace_infer_id']!=r or meta['checkpoint_sha256']!=WEIGHT or meta['denoise_steps']!=3:
            raise ValueError('Inference protocol/index mismatch')
        model,reason,slot=router.choose(states[:t+1])
        if mode in ('full','v14'):model='Full' if mode=='full' else 'V14'
        if meta['model']!=model or meta['candidate_skip_layers_1based']!=([] if model=='Full' else list(SKIP)):
            raise ValueError('Recorded source model violates online routing')
        if mode=='phase' and (meta['route_reason']!=reason or meta['precision_latch']!=router.latch):raise ValueError('Phase provenance mismatch')
        if meta['noise_sha256']!=hashlib.sha256(noise(init,r).tobytes()).hexdigest():raise ValueError('Noise mismatch')
        n=min(5,len(actions)-t)
        if n<1 or not np.allclose(actions[t:t+n],np.asarray(rec['chunk_actions'])[:n],rtol=1e-4,atol=1e-5):raise ValueError('Executed prefix mismatch')
        path=Path(meta['full_input_path'])
        with np.load(path,allow_pickle=False) as data:
            if data['actions'].shape!=(50,7) or not np.isfinite(data['actions']).all():raise ValueError('Invalid prediction')
        confirmation={'actual_executed_length':n,'source_infer':str(directory/episode['infer_path'])}
        confirmed=path.with_suffix('.executed.json')
        if confirmed.exists():
            if json.loads(confirmed.read_text())!=confirmation:raise ValueError('Execution confirmation changed')
        else:write_json(confirmed,confirmation,exclusive=True)
        full_actions+=n*(model=='Full');full_rounds+=model=='Full';offset+=n
    if offset!=len(actions):raise ValueError('Execution coverage mismatch')
    video=directory/episode['video_path']
    probe=subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,codec_name,pix_fmt,r_frame_rate','-of','json',str(video)],capture_output=True,text=True,check=True)
    info=json.loads(probe.stdout)['streams'][0]
    if int(info['nb_read_frames'])!=len(states) or info['codec_name']!='h264' or info['pix_fmt']!='yuv420p':raise ValueError('Video frame/codec mismatch')
    numerator,denominator=map(int,info['r_frame_rate'].split('/'))
    if numerator/denominator!=20:raise ValueError('Video fps differs from control frequency')
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],capture_output=True,check=True)
    return dict(episode,valid=True,actual_actions=len(actions),actual_predictions=len(infer),full_rounds=full_rounds,
                full_actions=full_actions,directory=str(directory),video_decoded=True)

def episode(args,gpu,port,suite,task,init,condition,mode,kind,retry=False):
    key='%s/%s/%02d/%02d'%(condition,suite,task,init)
    ledger=Ledger()
    prior=json.loads(ledger.path.read_text())['entries'] if ledger.path.exists() else []
    retry=retry and any(e['key']==key for e in prior)
    token=ledger.reserve(key,kind,LIMITS[suite],LIMITS[suite]//5,retry,getattr(args,'fix_note',''))
    attempt='attempt_'+token
    directory=args.output/'closed_loop'/condition/suite/('task%02d'%task)/('init%02d'%init)/attempt
    ledger.annotate(token,directory=str(directory))
    log=args.output/'logs'/(attempt+'.client.log')
    log.parent.mkdir(parents=True,exist_ok=True)
    try:
        episode_started=time.perf_counter()
        with log.open('x') as stream:
            subprocess.run(command(args,'episode',REPO/'examples/libero/.venv/bin/python',suite=suite,task=task,init=init,
                                   condition=condition,mode=mode,port=port,episode_attempt=attempt),cwd=REPO,env=env(gpu),stdout=stream,stderr=subprocess.STDOUT,check=True)
        episode_wall_s=time.perf_counter()-episode_started
        result=validate(directory,mode,condition,suite,task,init)
        result['client_process_wall_s']=episode_wall_s
        result['client_process_wall_scope']='client setup, simulation and video encoding; excludes server start and validation'
        write_json(directory/'validated.json',result,exclusive=True)
        ledger.finish(token,'valid',result['actual_actions'],result['actual_predictions'],directory=str(directory),success=result['success'])
        return result
    except BaseException as exc:
        ledger.finish(token,'invalid',error=repr(exc),directory=str(directory))
        write_json(directory/'invalid.json',{'error':repr(exc),'reservation':token})
        raise

def baseline_compare(args,result,condition,suite):
    original=json.loads((args.source/'closed_loop'/condition/suite/'run/episode_log.json').read_text())
    old=next(r for r in original if r['task_id']==0 and r['episode_idx']==0)
    base=args.source/'closed_loop'/condition/suite/'run'
    current=Path(result['directory'])
    a=rows(base/old['telemetry_path']);b=rows(current/result['telemetry_path'])
    if old['success']!=result['success'] or len(a)!=len(b):raise ValueError('Baseline success/termination differs')
    for x,y in zip(a,b):
        if not np.allclose(x['eef_pos'],y['eef_pos'],rtol=0,atol=1e-5):raise ValueError('Baseline XYZ differs')
        if x['action'] is not None and not np.allclose(x['action'],y['action'],rtol=1e-4,atol=1e-5):raise ValueError('Baseline actions differ')

def parity(args):
    from .policy import make_backend
    from scripts.experiment10.common import CONVERTED
    if sha(CONVERTED/'base/base_weights.pkl')!=WEIGHT:raise ValueError('Checkpoint hash mismatch')
    records=[]
    metadata=[]
    for suite in SUITES:
        for task in (0,5):
            path=args.source/'full_observations'/suite/('task%02d'%task)/'init00/replan0000.npz'
            with np.load(path,allow_pickle=False) as data:
                metadata.append({'suite':suite,'task_description':str(data['prompt'].item())})
    backend=make_backend(metadata)
    token=Ledger().reserve('parity/'+str(args.output),'diagnostic',predictions=64,retry=args.retry_invalid,fix_note=args.fix_note)
    try:
        for meta in metadata:
            suite=meta['suite'];task=(0,5)[len(records)%2]
            path=args.source/'full_observations'/suite/('task%02d'%task)/'init00/replan0000.npz'
            # The source cache was float32; reconstruct client precision from measured state.
            from scripts.experiment11.replay_state import client_state_from_telemetry
            with np.load(path,allow_pickle=False) as data:
                inputs={'observation/image':data['agent_image'],'observation/wrist_image':data['wrist_image'],
                        'observation/state':data['robot_state'],'prompt':str(data['prompt'].item())}
            # restore helper interface is validated separately below; missing precision blocks smoke.
            episodes=json.loads((args.source/'closed_loop/Full18_D3_R5_FORMAL'/suite/'run/episode_log.json').read_text())
            episode=next(e for e in episodes if e['task_id']==task and e['episode_idx']==0)
            state=rows(args.source/'closed_loop/Full18_D3_R5_FORMAL'/suite/'run'/episode['telemetry_path'])[0]
            inputs['observation/state']=client_state_from_telemetry(state,inputs['observation/state'])
            random=noise(0,0);static={}
            for mask in ((),SKIP):
                backend.prepare(inputs,meta,random);static[mask]=backend.infer(mask,3)[1].copy()
            differences=[]
            for mask in ((),SKIP,SKIP,(),SKIP,()):
                backend.prepare(inputs,meta,random);actual=backend.infer(mask,3)[1].copy()
                if actual.shape!=(50,7) or not np.isfinite(actual).all() or not np.allclose(actual,static[mask],rtol=1e-4,atol=1e-5):raise ValueError('Switch parity failed')
                differences.append(float(np.max(np.abs(actual-static[mask]))))
            records.append(dict(meta,task=task,max_difference=max(differences)))
        Ledger().finish(token,'valid',actual_predictions=64)
    except BaseException as exc:
        Ledger().finish(token,'invalid',error=repr(exc));raise
    write_json(args.output/'smoke/parity.json',{'passed':True,'points':records,'logical_predictions':64,
                                             'backend_forward_counts':backend.forward_counts},exclusive=True)

def smoke(args):
    free,_=discover()
    if str(args.gpu) not in free:
        raise RuntimeError('Requested smoke GPU is not confirmed idle by shared 30-second discovery; wait for the next discovery window')
    with lease(args.gpu):
        # Run parity in model Python as a separate process; parent remains CPU-only.
        if not (args.output/'smoke/parity.json').exists():
            subprocess.run(command(args,'parity',gpu=args.gpu),cwd=REPO,env=env(args.gpu),check=True)
        reproduced=[]
        for suite in SUITES:
            for mode,baseline in (('full','Full18_D3_R5_FORMAL'),('v14','V14_A03_D3_R5_FORMAL')):
                condition='SMOKE_'+baseline
                valid=list((args.output/'closed_loop'/condition/suite/'task00/init00').glob('*/validated.json'))
                if valid:result=json.loads(valid[0].read_text())
                else:
                    with server(args,args.gpu,suite,condition,mode) as port:
                        result=episode(args,args.gpu,port,suite,0,0,condition,mode,'baseline',args.retry_invalid)
                baseline_compare(args,result,baseline,suite);reproduced.append(result)
        if not (args.output/'smoke/baselines.json').exists():
            write_json(args.output/'smoke/baselines.json',{'passed':True,'episodes':reproduced},exclusive=True)
        for suite in SUITES:
            if list((args.output/'closed_loop'/ALT/suite/'task00/init00').glob('*/validated.json')):continue
            with server(args,args.gpu,suite,ALT,'alternating') as port:episode(args,args.gpu,port,suite,0,0,ALT,'alternating','E2',args.retry_invalid)
        if not (args.output/'smoke/gates.json').exists():
            write_json(args.output/'smoke/gates.json',{'passed':True,'code_hashes':json.loads((args.output/'protocol/execution_manifest.json').read_text())['code_hashes'],
                                                'E2_smoke_reused':True},exclusive=True)

def check_completed(root,condition,suite):
    valid=list((root/'closed_loop'/condition/suite).glob('*/*/*/validated.json'))
    if not valid:return 0
    ledger=json.loads(Ledger().path.read_text())
    keys=set()
    for path in valid:
        row=json.loads(path.read_text());key='%s/%s/%02d/%02d'%(condition,suite,row['task_id'],row['episode_idx'])
        if key in keys:raise RuntimeError('Duplicate completed key')
        keys.add(key)
        if not any(e['key']==key and e['status']=='valid' and e.get('directory')==str(path.parent) for e in ledger['entries']):
            raise RuntimeError('Durable validation and ledger disagree; diagnose and reconcile before resume')
    return len(valid)

def worker(args):
    condition=PHASE if args.mode=='phase' else ALT
    kind='E3' if args.mode=='phase' else 'E2'
    if check_completed(args.output,condition,args.suite)==50:return
    with locked(args.output/'status'/('%s_%s.worker.lock'%(kind,args.suite))), lease(args.gpu):
        with server(args,args.gpu,args.suite,condition,args.mode) as port:
            for task in range(10):
                for init in range(5):
                    existing=list((args.output/'closed_loop'/condition/args.suite/('task%02d'%task)/('init%02d'%init)).glob('*/validated.json'))
                    if existing:
                        key='%s/%s/%02d/%02d'%(condition,args.suite,task,init)
                        ledger=json.loads(Ledger().path.read_text())
                        if not any(e['key']==key and e['status']=='valid' and e.get('directory')==str(existing[0].parent) for e in ledger['entries']):
                            raise RuntimeError('Validated artifact and ledger disagree; use reconcile after diagnosing dead owner')
                        continue
                    episode(args,args.gpu,port,args.suite,task,init,condition,args.mode,kind,args.retry_invalid)

class Adopted:
    def __init__(self,pid,root,condition,suite,start_ticks=None):
        self.pid,self.root,self.condition,self.suite,self.start_ticks=pid,root,condition,suite,start_ticks
    def poll(self):
        try:
            fields=(Path('/proc')/str(self.pid)/'stat').read_text().split(') ',1)[1].split()
            state=fields[0]
            if self.start_ticks is None or fields[19]!=self.start_ticks:return 1
            if state not in ('X','Z'):
                cmd=(Path('/proc')/str(self.pid)/'cmdline').read_bytes().split(b'\0')
                if b'scripts.idea1_v2' not in cmd or b'worker' not in cmd or str(self.root).encode() not in cmd:return 1
                return None
        except FileNotFoundError:pass
        return 0 if len(list((self.root/'closed_loop'/self.condition/self.suite).glob('*/*/*/validated.json')))==50 else 1

def launch(args):
    with locked(args.output/'status/scheduler.lock'):
        _launch(args)

def _launch(args):
    from .__main__ import manifest
    frozen=json.loads((args.output/'protocol/execution_manifest.json').read_text())
    if frozen['code_hashes']!=manifest(args.source)['code_hashes']:raise RuntimeError('Code changed after freeze; review and freeze a new attempt')
    gate=json.loads((args.output/'smoke/gates.json').read_text())
    if not gate['passed'] or gate['code_hashes']!=frozen['code_hashes']:raise RuntimeError('Smoke gates not passed for frozen code')
    if args.mode=='phase':
        completed=[json.loads(p.read_text()) for p in (args.output/'closed_loop'/ALT).glob('*/*/*/*/validated.json')]
        if len(completed)!=200 or sum(r['success'] for r in completed)<160 or not (args.output/'kalman/completion.json').exists():
            write_json(args.output/'E3_deferred.json',{'status':'deferred_by_preregistered_gate'});return
    condition=PHASE if args.mode=='phase' else ALT
    pending=[s for s in SUITES if check_completed(args.output,condition,s)<50]
    running={};available=[];next_scan=0
    process_file=args.output/'status'/('%s_workers.json'%condition)
    processes=json.loads(process_file.read_text()) if process_file.exists() else {}
    for gpu,value in list(processes.items()):
        adopted=Adopted(value['pid'],args.output,condition,value['suite'],value.get('start_ticks'))
        if value['suite'] in pending and adopted.poll() is None:
            running[gpu]=(adopted,value['suite'])
            if value['suite'] in pending:pending.remove(value['suite'])
        else:processes.pop(gpu,None)
    write_json(process_file,processes)
    while pending or running:
        for gpu,(proc,suite) in list(running.items()):
            code=proc.poll()
            if code is not None:
                del running[gpu]
                processes.pop(gpu,None);write_json(process_file,processes)
                if code:raise RuntimeError('Suite worker failed: %s; retain partial and diagnose before explicit retry'%suite)
                available.append(gpu)
        if pending and not available and len(running)<4 and time.time()>=next_scan:
            found,next_scan=discover();available.extend(g for g in found if g not in running)
        while pending and available and len(running)<4:
            gpu=available.pop(0);suite=pending.pop(0)
            proc=subprocess.Popen(command(args,'worker',suite=suite,gpu=gpu,mode=args.mode),cwd=REPO,env=env(gpu))
            running[gpu]=(proc,suite)
            start_ticks=(Path('/proc')/str(proc.pid)/'stat').read_text().split(') ',1)[1].split()[19]
            processes[gpu]={'pid':proc.pid,'suite':suite,'condition':condition,'start_ticks':start_ticks}
            write_json(process_file,processes)
        if pending or running:time.sleep(2)

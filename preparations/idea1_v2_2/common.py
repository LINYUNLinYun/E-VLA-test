"""Immutable scientific identities and atomic, process-safe attempt state."""
import contextlib, fcntl, hashlib, json, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_FILES=('common.py','runtime_client.py','runtime_server.py','runtime_data.py','runtime_inference.py','runtime_runner.py','runtime_models.py','schedules.py','pi0_adapter.py','provenance.py')
UPSTREAM_FILES=('preparations/idea1_v2_1/runtime_client.py','preparations/idea1_v2_1/runtime_models.py','preparations/idea1_v2_1/adapters.py','policies/realtime-vla-flash/packages/openpi-client/src/openpi_client/image_tools.py','policies/realtime-vla-flash/scripts/experiment6/trajectory_replay.py')
def evaluation_env(gpu=None):
    env=dict(os.environ); repo=ROOT/'policies/realtime-vla-flash'
    paths=[str(ROOT/'LIBERO'),str(repo/'packages/openpi-client/src')]
    env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',OMP_NUM_THREADS='1',LIBERO_CONFIG_PATH=str(repo/'.libero'),PYTHONUNBUFFERED='1',PYTHONPATH=':'.join(paths+[env.get('PYTHONPATH','')]))
    if gpu is not None: env.update(CUDA_VISIBLE_DEVICES=str(gpu),MUJOCO_EGL_DEVICE_ID=str(gpu))
    return env
def bootstrap_environment(gpu=None):
    import sys
    env=evaluation_env(gpu); os.environ.update(env)
    for path in env['PYTHONPATH'].split(':'):
        if path and path not in sys.path: sys.path.insert(0,path)
SUITES = ('libero_spatial','libero_object','libero_goal','libero_10')
MODELS = ('pi0','xiaomi-robotic0','db-cogact')
def prior_diagnostic_budget(manifest,model):
    record=manifest.get('prior_diagnostics',{}).get(model,{})
    return record.get('conservative_predictions',record.get('actual_predictions',0))
MASKS = {'pi0':('V14_A03',[3,4,9,11]),'xiaomi-robotic0':('skip12_short_runs',[4,5,6,8,9,10,29,30,31,33,34,35]),'db-cogact':('skip9_adjacent_pairs',[5,6,8,9,21,22,23,24,25])}
NODES = [('u50',.5),('m30',.3),('m70',.7),('m10',.1),('m25',.25),('m90',.9),('m75',.75),('m60',.6)]
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def atomic_json(path,value):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    with t.open('x') as f: json.dump(value,f,indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(t,p)
@contextlib.contextmanager
def lock(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX); yield
@contextlib.contextmanager
def exclusive(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('a+') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RuntimeError(f'Active owner holds lease: {path}') from exc
        yield f
@contextlib.contextmanager
def gpu_lease(gpu,allow_inherited=True):
    path=ROOT/'preparations/idea1_v2_2/readiness/gpu_leases'/f'gpu{gpu}.lock'
    inherited=os.environ.get('V22_GPU_LEASE_FD') if allow_inherited else None
    if inherited:
        fd=int(inherited)
        if os.fstat(fd).st_ino!=path.stat().st_ino: raise RuntimeError('Inherited GPU lease does not match physical device')
        yield fd; return
    with exclusive(path) as f:
        atomic_json(path.with_suffix('.json'),dict(pid=os.getpid(),start_ticks=start_ticks(os.getpid()),gpu=str(gpu),time=time.time()))
        yield f.fileno()
def start_ticks(pid):
    try:
        x=Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()
        return x[19] if x[0] not in ('Z','X') else None
    except FileNotFoundError: return None
def seed(suite,task,init,replan):
    return int(digest(['idea1-v22',1701,suite,task,init,replan])[:8],16)
def conditions(extension=False):
    out=[]
    for model in MODELS:
        for mask,skip in [('full',[]),MASKS[model]]:
            schedules=[]
            if model!='db-cogact': schedules += [(n,dict(kind='flow',m=m,num_steps=2)) for n,m in NODES]
            if model!='pi0':
                schedules += [(f'd{d}',dict(kind='native',m=None,num_steps=d)) for d in ((5,) if model=='xiaomi-robotic0' else (10,2))]
            if model=='db-cogact' and extension: schedules += [(n,dict(kind='ddim_custom',m=m,num_steps=2)) for n,m in NODES]
            for name,schedule in schedules:
                cid=f'{model}_{mask}_{name}'
                aliases=[f'{model}_{mask}_d2'] if model=='xiaomi-robotic0' and name=='u50' else []
                out.append(dict(id=cid,model=model,mask=mask,skip=skip,schedule=schedule,aliases=aliases,extension=model=='db-cogact' and schedule['kind']=='ddim_custom'))
    assert len(out)==(54 if extension else 38)
    return out
def planned(manifest,initial=False):
    for init in ([6] if initial else range(6,11)):
        for c in sorted(manifest['conditions'],key=lambda x:x['extension']):
            for suite in SUITES:
                for task in range(10):
                    row=dict(condition=c['id'],model=c['model'],suite=suite,task=task,init=init,protocol_hash=manifest['protocol_hashes'][c['model']],schedule=c['schedule'],mask=c['mask'])
                    row['key']=digest(row); yield row
class Ledger:
    def __init__(self,root): self.path=Path(root)/'episode_ledger.json'
    def read(self): return json.loads(self.path.read_text()) if self.path.exists() else {'entries':[]}
    def reserve(self,row,require_artifacts=False):
        with lock(self.path.with_suffix('.lock')):
            data=self.read(); old=[x for x in data['entries'] if x['key']==row['key']]
            for x in old:
                if x['status']=='valid':
                    if require_artifacts: self.verify_committed(x,row)
                    return None
            if any(x['status']=='reserved' and start_ticks(x['pid'])==x['start_ticks'] for x in old): return None
            for x in old:
                if x['status']=='reserved':
                    validated=Path(x.get('directory','/nonexistent'))/'validated.json'
                    if validated.exists():
                        result=json.loads(validated.read_text())
                        if result.get('valid') and result.get('episode_key')==row['key']:
                            x.update(status='valid',success=result['success'],validated_sha256=sha(validated),recovered=True)
                            if require_artifacts: self.verify_committed(x,row)
                            atomic_json(self.path,data); return None
                    x.update(status='invalid',error='Owner ended before atomic validation')
            if old and old[-1]['status']=='invalid' and not old[-1].get('retry_diagnosis'):
                atomic_json(self.path,data)
                raise RuntimeError('Invalid episode requires explicit diagnosis and retry authorization in ledger; automatic retry is forbidden')
            token=f'{os.getpid()}-{time.time_ns()}'
            data['entries'].append(dict(row,token=token,status='reserved',pid=os.getpid(),start_ticks=start_ticks(os.getpid())))
            atomic_json(self.path,data); return token
    def verify_committed(self,entry,row):
        path=Path(entry['directory'])/'validated.json'; value=json.loads(path.read_text())
        if sha(path)!=entry['validated_sha256'] or value.get('episode_key')!=row['key'] or value.get('valid') is not True or not value.get('artifacts'): raise ValueError('Committed validation evidence is missing/corrupt; preserve the original result and investigate')
        for name,h in value['artifacts'].items():
            if sha(path.parent/name)!=h: raise ValueError('Committed raw artifact changed; do not replace a valid episode')
    def authorize_retry(self,key,diagnosis):
        if not diagnosis.strip(): raise ValueError('A concrete diagnosis/fix note is required')
        with lock(self.path.with_suffix('.lock')):
            data=self.read(); old=[x for x in data['entries'] if x['key']==key]
            if not old or old[-1]['status']!='invalid' or any(x['status']=='valid' for x in old): raise ValueError('Only technical invalid episodes can retry')
            if len(old)>=3: raise ValueError('Three-attempt technical retry budget exhausted')
            old[-1].update(retry_diagnosis=diagnosis,retry_authorized_at=time.time()); atomic_json(self.path,data)
    def update(self,token,**fields):
        with lock(self.path.with_suffix('.lock')):
            data=self.read(); next(x for x in data['entries'] if x['token']==token).update(fields); atomic_json(self.path,data)

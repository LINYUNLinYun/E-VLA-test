"""Single scheduler with continuous C/G occupancy watchdog; never signals foreign PIDs."""
import datetime,hashlib,json,os,shlex,subprocess,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
R=Path(__file__).resolve().parent
W=R.parents[2]
matrix=Path(sys.argv[1]).resolve()
uuid=sys.argv[2]
cpus=set(map(int,sys.argv[3].split(','))) if len(sys.argv)>3 else set(range(16))
state_path=matrix.with_suffix('.execution.json')
state=dict(status='running',matrix=str(matrix),gpu_uuid=uuid,child_cpu_affinity=sorted(cpus),cases=[])
env=dict(os.environ,CUDA_DEVICE_MAX_CONNECTIONS='32',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH='.:policies/openpi/src',OPENPI_DATA_HOME=str(W/'shared/checkpoints/openpi'),JAX_PLATFORMS='cpu',JAX_PLATFORM_NAME='cpu')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save():
 p=state_path.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2)+'\n');p.replace(state_path)
def occupancy():
 xml=subprocess.check_output(['nvidia-smi','-i',uuid,'-q','-x'],text=True)
 g=ET.fromstring(xml).find('gpu')
 processes=[dict(pid=int(p.findtext('pid')),type=p.findtext('type'),memory=p.findtext('used_memory')) for p in g.findall('processes/process_info')]
 return dict(time_utc=utc(),processes=processes,memory_mib=float(g.findtext('fb_memory_usage/used').split()[0]),utilization=float(g.findtext('utilization/gpu_util').split()[0]))
def sources(case):
 paths=[p for p in (W/'preparations/idea2').glob('*.py') if not p.name.startswith('test_')]
 paths += [W/'policies/openpi/src/openpi/models_pytorch'/n for n in ['pi0_pytorch.py','gemma_pytorch.py']]
 paths += [W/(case['module'].replace('.','/')+'.py')]
 paths += [W/'results/idea2/progressive_kv_20261025'/n for n in ['resource_adapter.py','performance_bench.py']]
 paths += [R/'adapter.py']
 return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
for case in json.loads(matrix.read_text()):
 if (R/'STOP_ISOLATED_AFTER_CURRENT').exists():state['status']='stopped_at_boundary';save();raise SystemExit(0)
 dest=R/case['name']
 if dest.exists():raise RuntimeError(f'Refusing existing output {dest}')
 stable=0;idle=[]
 for attempt in range(30):
  o=occupancy();idle.append(o)
  if o['processes']:raise RuntimeError('GPU has a live C/G process: '+str(o))
  stable=stable+1 if o['memory_mib']<=256 and o['utilization']<=1 else 0
  if stable>=2:break
  time.sleep(1)
 else:raise RuntimeError('GPU did not settle idle')
 command=[str(W/'policies/openpi/.venv/bin/python'),'-u','-m',case['module'],'--checkpoint','shared/checkpoints/openpi/pi05_libero_pytorch_idea2','--gpu-uuid',uuid,'--output',str(dest),*case['args']]
 if case.get('nsys_output'):
  command=['nsys','profile','--trace=cuda,nvtx','--cuda-graph-trace=node:nvtx-precapture','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=false','-o',case['nsys_output'],*command]
 before=sources(case)
 snap=R/'source_snapshots'/case['name'];snap.mkdir(parents=True,exist_ok=False)
 for p in before:
  path=Path(p);target=snap/path.relative_to(W);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
 row=dict(name=case['name'],command=shlex.join(command),start_utc=utc(),status='running',runtime_sources_before=before,idle_observations=idle,watchdog=[])
 state['cases'].append(row);save();print('START',case['name'],flush=True)
 os.sched_setaffinity(0,cpus)
 with (R/(case['name']+'.log')).open('x') as log:
  p=subprocess.Popen(command,cwd=W,env=env,stdout=log,stderr=subprocess.STDOUT)
  row['pid']=p.pid
  os.sched_setaffinity(0,{16,17})
  expected=None
  last_saved=time.monotonic()
  while p.poll() is None:
   o=occupancy()
   procs=o['processes']
   ids={x['pid'] for x in procs}
   if expected is None and len(procs)==1 and procs[0]['type']=='C':expected=ids
   conflict=len(procs)>1 or any('G' in str(x['type']) for x in procs) or (expected is not None and ids and ids!=expected)
   o['expected_host_pids']=sorted(expected) if expected is not None else None
   o['conflict']=bool(conflict);row['watchdog'].append(o)
   if conflict:
    row['interference']=o;row['status']='excluded_gpu_interference'
    p.terminate()
    try:p.wait(timeout=30)
    except subprocess.TimeoutExpired:p.kill();p.wait()
    break
   if time.monotonic()-last_saved>=10:save();last_saved=time.monotonic()
   time.sleep(1)
  row['exit_code']=p.wait()
 row['end_utc']=utc();row['runtime_sources_unchanged']=before==sources(case)
 rp=dest/'result.json'
 d=json.loads(rp.read_text()) if rp.exists() else {}
 passed=row['exit_code']==0 and d.get('status')=='passed' and row['runtime_sources_unchanged'] and 'interference' not in row
 if not passed:
  row['status']=row['status'] if row['status'].startswith('excluded_') else 'failed'
  if d:
   d.update(status_before_execution_exclusion=d.get('status'),status=row['status'],performance_eligible=False,execution_receipt=str(state_path));rp.write_text(json.dumps(d,indent=2)+'\n')
  state['status']='failed';save();print('FAILED',case['name'],flush=True);raise SystemExit(1)
 row['status']='passed';save();print('PASSED',case['name'],flush=True)
state['status']='passed';save()

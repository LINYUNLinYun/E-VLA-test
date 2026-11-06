"""Sequential, fail-closed execution of an explicitly frozen experiment list."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'d10_dynamic_sm_20261024'))
from idle_settle import wait_for_idle

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
matrix = Path(sys.argv[1]).resolve()
state_path = matrix.with_suffix('.execution.json')
env = dict(os.environ, CUDA_DEVICE_MAX_CONNECTIONS='32', OMP_NUM_THREADS='2',
           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONPATH='.:policies/openpi/src',
           OPENPI_DATA_HOME=str(WORKSPACE/'shared/checkpoints/openpi'),
           JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu')
os.sched_setaffinity(0, set(range(16)))
state = dict(matrix=str(matrix), status='running', cpu_affinity=sorted(os.sched_getaffinity(0)), cases=[])

def save():
    temp = state_path.with_suffix('.tmp')
    temp.write_text(json.dumps(state, indent=2)+'\n')
    temp.replace(state_path)

def runtime_hashes():
    paths = list((WORKSPACE/'preparations/idea2').glob('*.py'))
    paths += [WORKSPACE/'policies/openpi/src/openpi/models_pytorch'/name
              for name in ('pi0_pytorch.py', 'gemma_pytorch.py')]
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if not path.name.startswith('test_')}

for case in json.loads(matrix.read_text()):
    if (ROOT/'STOP_AFTER_CURRENT').exists():
        state['status'] = 'stopped_at_case_boundary'
        save()
        print('STOPPED before', case['name'], flush=True)
        raise SystemExit(0)
    dest = ROOT / case['name']
    if dest.exists():
        raise RuntimeError(f'Refusing to overwrite or restart existing case {dest}')
    command = [str(WORKSPACE/'policies/openpi/.venv/bin/python'), '-u', '-m', case['module'],
               '--checkpoint', 'shared/checkpoints/openpi/pi05_libero_pytorch_idea2',
               '--gpu-uuid', 'GPU-599477fb-0507-40dc-14df-d491f7a7b202', '--output', str(dest),
               *case['args']]
    if case.get('nsys_output'):
        command = ['nsys', 'profile', '--trace=cuda,nvtx', '--cuda-graph-trace=node:nvtx-precapture',
                   '--sample=none', '--cpuctxsw=none', '--capture-range=cudaProfilerApi',
                   '--capture-range-end=stop', '--force-overwrite=false', '-o', case['nsys_output'], *command]
    row = dict(name=case['name'], command=shlex.join(command),
               start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='running')
    row['runtime_sources_before'] = runtime_hashes()
    state['cases'].append(row)
    save()
    try:
        row['idle_settle_observations'] = wait_for_idle()
    except Exception as error:
        row.update(status='failed_before_launch', error=repr(error))
        state['status'] = 'failed'
        save()
        raise
    print('START', case['name'], flush=True)
    begin = time.monotonic()
    with (ROOT/(case['name']+'.log')).open('x') as log:
        process = subprocess.Popen(command, cwd=WORKSPACE, env=env, stdout=log, stderr=subprocess.STDOUT)
        row['pid'] = process.pid
        save()
        row['exit_code'] = process.wait()
    row['wall_seconds'] = time.monotonic()-begin
    result_path = dest/'result.json'
    row['result_status'] = json.loads(result_path.read_text()).get('status') if result_path.exists() else None
    row['runtime_sources_unchanged'] = row['runtime_sources_before'] == runtime_hashes()
    row['status'] = 'passed' if row['exit_code']==0 and row['result_status']=='passed' and row['runtime_sources_unchanged'] else 'failed'
    if not row['runtime_sources_unchanged'] and result_path.exists():
        result = json.loads(result_path.read_text())
        result.update(execution_status_before_exclusion=result.get('status'), status='excluded_source_drift',
                      performance_eligible=False, provenance_exclusion='Runtime sources changed during execution')
        result_path.write_text(json.dumps(result, indent=2)+'\n')
    print(row['status'].upper(), case['name'], round(row['wall_seconds'], 1), flush=True)
    if row['status'] != 'passed':
        state['status'] = 'failed'
        save()
        raise SystemExit(1)
    save()
state['status'] = 'passed'
save()

"""Sequential experiment launcher; never interferes with foreign GPU work."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gpu-uuid', default='GPU-599477fb-0507-40dc-14df-d491f7a7b202')
parser.add_argument('queues', nargs='+')
options = parser.parse_args()
GPU = options.gpu_uuid


def idle():
    # Utilization is a trailing sample and can remain high after our last job
    # exits. Wait for the measurement to settle; never wait out another owner.
    for _ in range(15):
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                        '--format=csv,noheader'], text=True)
        if any(line.split(',')[0].strip() == GPU for line in apps.splitlines()):
            raise RuntimeError('Assigned GPU occupied; stop without touching its processes')
        state = subprocess.check_output(['nvidia-smi', '--id='+GPU,
                                         '--query-gpu=memory.used,utilization.gpu',
                                         '--format=csv,noheader,nounits'], text=True)
        memory, usage = map(float, state.strip().split(','))
        if memory <= 256 and usage <= 1:
            return
        time.sleep(1)
    raise RuntimeError('Assigned GPU did not become idle')


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=GPU, CUDA_DEVICE_MAX_CONNECTIONS='32',
           JAX_PLATFORMS='cpu', JAX_PLATFORM_NAME='cpu', OMP_NUM_THREADS='2',
           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
           OPENPI_DATA_HOME='/root/nfs/embodied/benchmark/shared/checkpoints/openpi',
           PYTHONPATH='.:policies/openpi/src')
for filename in options.queues:
    path = ROOT / filename
    data = json.loads(path.read_text())
    for job in data['jobs']:
        if job.get('state', job.get('status')) not in ('pending', 'prepared_not_run'):
            continue
        if '--gpu-uuid' in job['argv']:
            expected_gpu = job['argv'][job['argv'].index('--gpu-uuid') + 1]
            if expected_gpu != GPU:
                raise RuntimeError('Queue device differs from launcher device; prepare a matched-device queue explicitly')
        idle()
        env['CUDA_DEVICE_MAX_CONNECTIONS'] = str(job.get('connections', 32))
        job.update(state='running', started_unix=time.time(), gpu_uuid=GPU,
                   environment={k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_MAX_CONNECTIONS',
                               'JAX_PLATFORMS', 'JAX_PLATFORM_NAME', 'OMP_NUM_THREADS',
                               'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})
        save(path, data)
        print('START', job['name'], flush=True)
        with (ROOT / (job['name']+'.log')).open('w') as log:
            code = subprocess.call(job['argv'], env=env, stdout=log, stderr=subprocess.STDOUT)
        job.update(state='process_succeeded' if code == 0 else 'failed',
                   returncode=code, ended_unix=time.time())
        save(path, data)
        print('END', job['name'], code, flush=True)
        if code:
            sys.exit(code)
        if 'trace' in job:
            trace = job['trace']
            with (ROOT / (job['name']+'_export.log')).open('w') as log:
                code = subprocess.call(['nsys', 'export', '--type', 'sqlite', '--output', trace+'.sqlite',
                                        trace+'.nsys-rep'], stdout=log, stderr=subprocess.STDOUT)
            job.update(export_returncode=code, state='export_succeeded' if code == 0 else 'export_failed')
            save(path, data)
            if code:
                sys.exit(code)

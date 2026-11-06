"""Wait for other users, then lease idle cards and finish the bounded matrix."""
import argparse
import json
import os
import socket
import subprocess
import time
import traceback
from pathlib import Path
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment, lock
from preparations.idea1_v3_4.dispatch import render_environment
from preparations.idea1_v2_1.runtime_client import rpc
from .common import ROOT, OUT, MODULE, freeze, progress
from .resource_watch import scan

MODEL = ROOT/'policies/realtime-vla-flash/.venv/bin/python'
EVAL = ROOT/'policies/libero-eval-v21/.venv/bin/python'


def start(gpu, uuid):
    lease = lock(Path('/tmp')/f'embodied-benchmark-gpu-{uuid}.lock', blocking=False)
    try:
        lease.__enter__()
    except BlockingIOError:
        return None
    if (gpu, uuid) not in scan()['free']:
        lease.__exit__(None, None, None)
        return None
    wd = OUT/'workers'/f'gpu{gpu}'
    wd.mkdir(parents=True, exist_ok=True)
    env = environment(gpu)
    env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4')
    env['PYTHONPATH'] = str(ROOT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
    render, mapping = render_environment(env, gpu)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    atomic(wd/'identity.json', dict(gpu=gpu, uuid=uuid, port=port,
        controller_pid=os.getpid(), egl_mapping=mapping, time=time.time()))
    with (wd/'service.log').open('a') as stream:
        service = subprocess.Popen([str(MODEL), '-u', '-m', MODULE+'.server',
            '--port', str(port), '--output', str(wd)], cwd=ROOT, env=env,
            stdout=stream, stderr=subprocess.STDOUT)
    atomic(wd/'service_process.json', dict(pid=service.pid, port=port, time=time.time()))
    return dict(gpu=gpu, uuid=uuid, port=port, wd=wd, env=env, render=render,
        lease=lease, service=service, gate=None, evaluation=None,
        phase='loading', phase_start=time.time())


def finish(worker):
    for name in ('evaluation', 'gate'):
        process = worker[name]
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    service = worker['service']
    if service.poll() is None:
        try:
            rpc(worker['port'], dict(op='stop'))
        except (OSError, RuntimeError, ConnectionError):
            service.terminate()
        try:
            service.wait(timeout=20)
        except subprocess.TimeoutExpired:
            service.kill()
            service.wait()
    worker['lease'].__exit__(None, None, None)
    atomic(worker['wd']/'exits.json', dict(released=True, service_exit=service.returncode, time=time.time()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-gpus', type=int, default=4)
    args = parser.parse_args()
    manifest = freeze()
    workers, last_scan = [], 0.
    with lock(OUT/'controller.lock', blocking=False):
        atomic(OUT/'controller_process.json', dict(pid=os.getpid(), started=time.time()))
        try:
            while True:
                if (OUT/'stop_controller.json').exists():
                    return
                if time.time()-last_scan >= 30 and len(workers)<args.max_gpus:
                    try:
                        resource = scan()
                    except (OSError, subprocess.SubprocessError) as error:
                        atomic(OUT/'resource_scan_error.json', dict(error=repr(error), time=time.time()))
                        last_scan = time.time()
                        resource = dict(free=[])
                    for gpu, uuid in resource['free']:
                        if any(w['gpu']==gpu for w in workers):
                            continue
                        worker = start(gpu, uuid)
                        if worker is not None:
                            workers.append(worker)
                        if len(workers)>=args.max_gpus:
                            break
                    last_scan = time.time()
                for worker in workers:
                    wd, phase = worker['wd'], worker['phase']
                    if phase == 'done':
                        continue
                    if worker['service'].poll() is not None:
                        raise RuntimeError(f"Service exited on GPU {worker['gpu']}; inspect {wd/'service.log'}")
                    if phase == 'loading':
                        try:
                            rpc(worker['port'], dict(op='health'))
                        except (OSError, ConnectionError):
                            if time.time()-worker['phase_start']>600:
                                raise TimeoutError('Model load')
                            continue
                        with (wd/'preflight.log').open('a') as stream:
                            worker['gate'] = subprocess.Popen([str(MODEL), '-u', '-m', MODULE+'.preflight',
                                '--port', str(worker['port'])], cwd=ROOT, env=worker['env'],
                                stdout=stream, stderr=subprocess.STDOUT)
                        worker['phase'] = 'preflight'
                        worker['phase_start'] = time.time()
                    elif phase == 'preflight':
                        if worker['gate'].poll() is None:
                            if time.time()-worker['phase_start']>600:
                                raise TimeoutError('Preflight')
                            continue
                        if worker['gate'].returncode:
                            raise RuntimeError(f"New mechanism preflight failed on GPU {worker['gpu']}")
                        with (wd/'evaluate.log').open('a') as stream:
                            worker['evaluation'] = subprocess.Popen([str(EVAL), '-u', '-m', MODULE+'.evaluate',
                                '--port', str(worker['port']), '--worker', f"gpu{worker['gpu']}"],
                                cwd=ROOT, env=worker['render'], stdout=stream, stderr=subprocess.STDOUT)
                        atomic(wd/'evaluation_process.json', dict(pid=worker['evaluation'].pid, time=time.time()))
                        worker['phase'] = 'evaluating'
                    elif phase == 'evaluating' and worker['evaluation'].poll() is not None:
                        if worker['evaluation'].returncode:
                            raise RuntimeError(f"Evaluator failed on GPU {worker['gpu']}; inspect {wd/'evaluate.log'}")
                        finish(worker)
                        worker['phase'] = 'done'
                status = progress(manifest)
                completed = sum(r['completed'] for r in status.values())
                atomic(OUT/'status.json', dict(status='waiting_gpu' if not workers else 'running',
                    completed=completed, target=manifest['target_episodes'], progress=status,
                    workers=[dict(gpu=w['gpu'], phase=w['phase']) for w in workers], time=time.time()))
                if completed == manifest['target_episodes']:
                    atomic(OUT/'execution_complete.json', dict(time=time.time(), progress=status))
                    return
                if workers and all(w['phase']=='done' for w in workers):
                    raise RuntimeError('All evaluators ended before the matrix completed')
                time.sleep(3)
        finally:
            for worker in workers:
                if worker['phase'] != 'done':
                    finish(worker)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        atomic(OUT/'failure.json', dict(error=traceback.format_exc(), time=time.time()))
        raise

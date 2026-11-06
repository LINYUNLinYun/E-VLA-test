"""Single mixed-experiment scheduler, GPU UUID leases and serial B1 services."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time
import traceback

from .common import (ALLOWED_GPUS, ROOT, OUT, claim_group, finish_group, freeze,
                     pending, progress, read_manifest, release, verify_frozen)
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment, lock

MODEL = ROOT / 'policies/realtime-vla-flash/.venv/bin/python'
EVAL = ROOT / 'policies/libero-eval-v21/.venv/bin/python'
MODULE = 'preparations.idea1_v3_4_E2_mixed_cost'


def available(excluded=()):
    candidates = [gpu for gpu in ALLOWED_GPUS if gpu not in excluded]
    if not candidates:
        return [], {}
    selection = '--id=' + ','.join(map(str, candidates))
    apps = subprocess.check_output(['nvidia-smi', selection, '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    busy = {row.split(',')[0].strip() for row in apps.splitlines() if row.strip()}
    rows = subprocess.check_output(['nvidia-smi', selection, '--query-gpu=index,name,uuid,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    free, identities = [], {}
    for row in rows.splitlines():
        index, name, uuid, used = [value.strip() for value in row.split(',')]
        index = int(index)
        identities[index] = uuid
        if '4090' not in name:
            raise ValueError(f'GPU{index} is not the measured RTX4090 hardware class')
        if uuid not in busy and float(used) <= 1500:
            free.append(index)
    atomic(OUT / 'resource_scan.json', dict(time=time.time(), queried=candidates,
           free=free, gpus=rows, compute_apps=apps, excluded=list(excluded)))
    return free, identities


def free_port():
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def ready_groups():
    return sum(group['state'] == 'pending' for group in json.loads((OUT / 'queue.json').read_text())['groups'])


def wait_and_freeze(cost_table_path):
    if (OUT / 'manifest.json').exists():
        manifest = read_manifest()
        verify_frozen(manifest)
        return manifest
    while not cost_table_path.exists():
        failure = cost_table_path.parent / 'failure.json'
        if failure.exists():
            raise RuntimeError(f'hardware cost probe failed: {failure}')
        atomic(OUT / 'status.json', dict(status='waiting_measured_4090_costs', cost_table=str(cost_table_path), time=time.time()))
        time.sleep(5)
    manifest = freeze(cost_table_path)
    verify_frozen(manifest)
    return manifest


def terminate(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)


def fatal_failure():
    return any(json.loads(path.read_text()).get('fatal') for path in (OUT / 'technical_attempts').glob('*/*.json'))


def worker(gpu, uuid, stop):
    from preparations.idea1_v3_4.dispatch import render_environment
    from preparations.idea1_v2_1.runtime_client import rpc
    work = OUT / 'workers' / f'gpu{gpu}'
    work.mkdir(parents=True, exist_ok=True)
    with lock(work / 'owner.lock', blocking=False):
        for attempt in range(2):
            if stop.is_set() or not pending():
                return
            env = environment(gpu)
            env.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='4')
            env['PYTHONPATH'] = str(ROOT / 'policies/realtime-vla-flash/src') + ':' + env['PYTHONPATH']
            render, mapping = render_environment(env, gpu)
            port = free_port()
            atomic(work / 'identity.json', dict(gpu=gpu, gpu_uuid=uuid, egl_mapping=mapping,
                   render_cuda_mask=render['CUDA_VISIBLE_DEVICES'], pid=os.getpid()))
            service, evaluations, loaded = None, [], False
            try:
                with (work / f'service_attempt{attempt + 1}.log').open('w') as log:
                    tick = time.monotonic()
                    service = subprocess.Popen([str(MODEL), '-u', '-m', MODULE + '.server', '--port', str(port),
                                                '--output', str(work)], cwd=ROOT, env=env,
                                               stdout=log, stderr=subprocess.STDOUT)
                    atomic(work / 'process.json', dict(pid=service.pid, gpu=gpu, gpu_uuid=uuid,
                           port=port, attempt=attempt + 1, time=time.time()))
                    atomic(work / 'state.json', dict(status='loading_and_preflight', time=time.time()))
                    while not stop.is_set():
                        if service.poll() is not None:
                            raise RuntimeError('mixed service startup/preflight failed')
                        if time.monotonic() - tick > 1200:
                            raise TimeoutError('mixed service readiness exceeded twenty minutes')
                        try:
                            health = rpc(port, dict(op='health'))
                            break
                        except (OSError, ConnectionError):
                            time.sleep(1)
                    else:
                        return
                    actors = int(health['protocol']['environment_workers'])
                    if actors not in (1, 2):
                        raise ValueError('preflight isolation actor count')
                    loaded = True
                    startup = dict(status='evaluating', actors=actors, startup_s=time.monotonic() - tick, time=time.time())
                    atomic(work / 'startup.json', startup)
                    atomic(work / f'startup_attempt{attempt + 1}.json', startup)
                    while not stop.is_set() and pending():
                        group = claim_group(gpu)
                        if group is None:
                            atomic(work / 'state.json', dict(status='waiting_inflight_groups', time=time.time()))
                            time.sleep(2)
                            continue
                        rpc(port, dict(op='reset_all'))
                        atomic(work / 'state.json', dict(status='evaluating', group=group, actors=actors, time=time.time()))
                        evaluations, logs = [], []
                        try:
                            for actor in range(actors):
                                stream = (work / f"{group['key']}_actor{actor}_attempt{attempt + 1}.log").open('a')
                                logs.append(stream)
                                evaluations.append(subprocess.Popen([str(EVAL), '-u', '-m', MODULE + '.evaluate',
                                    '--port', str(port), '--gpu', str(gpu), '--actor', str(actor),
                                    '--group', group['key'], '--phase', group['phase']], cwd=ROOT, env=render,
                                    stdout=stream, stderr=subprocess.STDOUT))
                            while any(process.poll() is None for process in evaluations):
                                if stop.is_set():
                                    raise InterruptedError('implementation stopped by another worker')
                                if service.poll() is not None or any(process.poll() not in (None, 0) for process in evaluations):
                                    raise RuntimeError(f"service/evaluator failed in {group['key']}")
                                time.sleep(2)
                            if any(process.returncode for process in evaluations):
                                raise RuntimeError(f"evaluators failed in {group['key']}")
                            finish_group(group['key'])
                        finally:
                            for process in evaluations:
                                terminate(process)
                            for stream in logs:
                                stream.close()
                atomic(work / 'state.json', dict(status='complete' if not pending() else 'stopped', time=time.time()))
                return
            except Exception:
                atomic(work / f'error_attempt{attempt + 1}.json', dict(error=traceback.format_exc(), time=time.time()))
                resource_fallback = (work / 'single_actor_required.json').exists()
                if stop.is_set() or attempt or fatal_failure() or (not loaded and not resource_fallback):
                    stop.set()
                    raise
            finally:
                for process in evaluations + [service]:
                    terminate(process)
                release(gpu)
                atomic(work / 'exits.json', dict(service=None if service is None else service.poll(),
                       evaluations=[process.poll() for process in evaluations], time=time.time()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cost-table', type=Path, default=OUT / 'hardware_probe/cost_table.json')
    parser.add_argument('--exclude-gpus', type=int, nargs='*', default=[2])
    args = parser.parse_args()
    with lock(OUT / 'controller.lock', blocking=False):
        atomic(OUT / 'process.json', dict(pid=os.getpid(), time=time.time()))
        wait_and_freeze(args.cost_table.resolve())
        active, leases, stop, last_scan = {}, {}, threading.Event(), -1e9
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                try:
                    while pending() or active:
                        for gpu, future in list(active.items()):
                            if future.done():
                                future.result()
                                del active[gpu]
                                leases.pop(gpu).__exit__(None, None, None)
                        need = ready_groups()
                        if need and len(active) < 3 and time.monotonic() - last_scan >= 30:
                            last_scan = time.monotonic()
                            free, identities = available(set(args.exclude_gpus) | set(active))
                            for gpu in free:
                                if len(active) >= 3 or need <= 0:
                                    break
                                lease_path = Path('/tmp') / f'embodied-benchmark-gpu-{identities[gpu]}.lock'
                                lease = lock(lease_path, blocking=False)
                                try:
                                    lease.__enter__()
                                except BlockingIOError:
                                    continue
                                # Recheck only this candidate after acquiring the shared UUID lease.
                                try:
                                    free_now, _ = available(set(ALLOWED_GPUS) - {gpu})
                                except BaseException:
                                    lease.__exit__(None, None, None)
                                    raise
                                if gpu not in free_now:
                                    lease.__exit__(None, None, None)
                                    continue
                                leases[gpu] = lease
                                atomic(OUT / 'workers' / f'gpu{gpu}' / 'lease.json',
                                       dict(path=str(lease_path), gpu_uuid=identities[gpu], pid=os.getpid(), time=time.time()))
                                active[gpu] = executor.submit(worker, gpu, identities[gpu], stop)
                                need -= 1
                        counts = progress()
                        atomic(OUT / 'status.json', dict(status='running' if active else 'waiting_gpu',
                               phase='main40', progress=counts, new_completed=sum(row['new'] for row in counts.values()),
                               active_gpus=list(active), time=time.time()))
                        if pending() or active:
                            time.sleep(2)
                except BaseException:
                    stop.set()
                    raise
        finally:
            stop.set()
            for lease in leases.values():
                lease.__exit__(None, None, None)
        counts = progress()
        if any(row['completed'] != row['target'] for row in counts.values()):
            raise RuntimeError('queue finished without all four forty-case conditions')
        atomic(OUT / 'evaluation_complete.json', dict(passed=True, target_records=160, new_episodes=160,
               reused_episodes=0, progress=counts, time=time.time()))
        from .finish import main as finish
        if not finish()['passed']:
            raise RuntimeError('mixed final audit failed; no completion marker was written')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        atomic(OUT / 'failed.json', dict(error=traceback.format_exc(), time=time.time()))
        raise

"""One scheduler, measured-cost freeze, persistent services and pilot barrier."""
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

from .common import (ROOT, OUT, _read_frozen, claim_group, finish_group, freeze,
                     pending, progress, release, sha)
from .selection import freeze_budget
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import environment, lock

MODEL = ROOT / 'policies/realtime-vla-flash/.venv/bin/python'
EVAL = ROOT / 'policies/libero-eval-v21/.venv/bin/python'
MODULE = 'preparations.idea1_v3_4_E2_module_skip'


def available(excluded=()):
    """Called only by the single scheduler when unclaimed work needs a card."""
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    busy = {row.split(',')[0].strip() for row in apps.splitlines() if row.strip()}
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    free, identities = [], {}
    for row in rows.splitlines():
        index, uuid, used = [value.strip() for value in row.split(',')]
        index = int(index)
        identities[index] = uuid
        if index not in excluded and uuid not in busy and float(used) <= 1500:
            free.append(index)
    atomic(OUT / 'resource_scan.json', dict(time=time.time(), free=free, gpus=rows,
                                           compute_apps=apps, excluded=list(excluded)))
    return free, identities


def free_port():
    # Bind an OS-assigned local port for each fresh service; never reuse an old
    # experiment's hard-coded port. The service immediately rebinds this port.
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


def ready_groups():
    groups = json.loads((OUT / 'queue.json').read_text())['groups']
    pilot_ready = (OUT / 'pilot_mechanism_complete.json').exists() or not any(group['phase'] == 'pilot' for group in groups)
    return sum(group['state'] == 'pending' and (group['phase'] == 'pilot' or pilot_ready) for group in groups)


def verify_frozen(manifest):
    for filename, expected in manifest['source_files'].items():
        if sha(ROOT / filename) != expected:
            raise ValueError(f'frozen source changed: {filename}')
    for filename, expected in manifest['references'].items():
        if sha(ROOT / filename) != expected:
            raise ValueError(f'frozen reference changed: {filename}')
    if sha(manifest['cost_table_path']) != manifest['cost_table_file_sha256']:
        raise ValueError('frozen measured cost table changed')


def wait_and_freeze(cost_table_path):
    if (OUT / 'manifest.json').exists():
        manifest = _read_frozen(OUT / 'manifest.json')
        verify_frozen(manifest)
        return manifest
    while True:
        failure = OUT / 'benchmark_failure.json'
        if failure.exists() and not cost_table_path.exists():
            raise RuntimeError(f'cost profiling failed; inspect {failure}')
        if cost_table_path.exists():
            table = json.loads(cost_table_path.read_text())
            # Missing/unmeasured/partial costs are never silently manufactured.
            freeze_budget(table, 'B4')
            freeze_budget(table, 'B6')
            status = OUT / 'benchmark_status.json'
            if not status.exists() or json.loads(status.read_text()).get('status') == 'complete':
                manifest = freeze(cost_table_path)
                verify_frozen(manifest)
                return manifest
        atomic(OUT / 'status.json', dict(status='waiting_measured_cost_table',
                                         cost_table=str(cost_table_path), time=time.time()))
        time.sleep(5)


def _terminate(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)


def offline_if_ready(port, work):
    if not (OUT / 'pilot_complete.json').exists() or (OUT / 'pilot_mechanism_complete.json').exists():
        return
    try:
        with lock(OUT / 'offline.lock', blocking=False):
            if (OUT / 'pilot_mechanism_complete.json').exists():
                return
            from preparations.idea1_v2_1.runtime_client import rpc
            try:
                rpc(port, dict(op='reset_all'))
                atomic(work / 'state.json', dict(status='pilot_offline_mechanism', time=time.time()))
                result = rpc(port, dict(op='offline'))
                if result.get('passed') is not True:
                    raise ValueError('offline mechanism did not pass its technical checks')
                atomic(OUT / 'pilot_mechanism_complete.json', dict(passed=True, worker=str(work),
                       summary=result.get('summary'), time=time.time()))
            except Exception:
                atomic(OUT / 'offline_failure.json', dict(error=traceback.format_exc(), worker=str(work), time=time.time()))
                raise
    except BlockingIOError:
        return


def _fatal_episode_failure():
    return any(json.loads(path.read_text()).get('fatal')
               for path in (OUT / 'technical_attempts').glob('*/*.json'))


def worker(gpu, gpu_uuid, stop):
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
            atomic(work / 'identity.json', dict(gpu=gpu, gpu_uuid=gpu_uuid, egl_mapping=mapping,
                                                render_cuda_mask=render['CUDA_VISIBLE_DEVICES']))
            service, evaluators, loaded = None, [], False
            try:
                with (work / f'service_attempt{attempt + 1}.log').open('w') as log:
                    tick = time.monotonic()
                    service = subprocess.Popen([str(MODEL), '-u', '-m', MODULE + '.server', '--port', str(port),
                                                '--output', str(work)], cwd=ROOT, env=env,
                                               stdout=log, stderr=subprocess.STDOUT)
                    atomic(work / 'process.json', dict(pid=service.pid, gpu=gpu, gpu_uuid=gpu_uuid,
                                                       port=port, attempt=attempt + 1, time=time.time()))
                    atomic(work / 'state.json', dict(status='loading_and_preflight', time=time.time()))
                    while not stop.is_set():
                        if service.poll() is not None:
                            raise RuntimeError('service startup/preflight failed')
                        if time.monotonic() - tick > 1200:
                            raise TimeoutError('service readiness exceeded twenty minutes')
                        try:
                            health = rpc(port, dict(op='health'))
                            break
                        except (OSError, ConnectionError):
                            time.sleep(1)
                    else:
                        return
                    actors = int(health['protocol']['environment_workers'])
                    loaded = True
                    startup = dict(status='evaluating', actors=actors,
                                   startup_s=time.monotonic() - tick, time=time.time())
                    atomic(work / 'startup.json', startup)
                    atomic(work / f'startup_attempt{attempt + 1}.json', startup)
                    while not stop.is_set() and pending():
                        group = claim_group(gpu)
                        if group is None:
                            offline_if_ready(port, work)
                            atomic(work / 'state.json', dict(status='waiting_pilot_or_inflight_groups', time=time.time()))
                            time.sleep(2)
                            continue
                        rpc(port, dict(op='reset_all'))
                        atomic(work / 'state.json', dict(status='evaluating', group=group, actors=actors, time=time.time()))
                        evaluators, files = [], []
                        try:
                            for actor in range(actors):
                                logfile = (work / f"{group['key']}_actor{actor}_attempt{attempt + 1}.log").open('a')
                                files.append(logfile)
                                evaluators.append(subprocess.Popen(
                                    [str(EVAL), '-u', '-m', MODULE + '.evaluate', '--port', str(port),
                                     '--gpu', str(gpu), '--actor', str(actor), '--group', group['key'],
                                     '--phase', group['phase']], cwd=ROOT, env=render,
                                    stdout=logfile, stderr=subprocess.STDOUT))
                            while any(process.poll() is None for process in evaluators):
                                if stop.is_set():
                                    raise InterruptedError('another worker stopped the implementation')
                                if service.poll() is not None:
                                    raise RuntimeError('model service exited during evaluation')
                                if any(process.poll() not in (None, 0) for process in evaluators):
                                    raise RuntimeError(f"evaluator failed in {group['key']}")
                                time.sleep(2)
                            codes = [process.returncode for process in evaluators]
                            if any(codes):
                                raise RuntimeError(f"evaluators failed {group['key']}: {codes}")
                            finish_group(group['key'])
                        finally:
                            for process in evaluators:
                                _terminate(process)
                            for logfile in files:
                                logfile.close()
                atomic(work / 'state.json', dict(status='complete' if not pending() else 'stopped', time=time.time()))
                return
            except Exception:
                atomic(work / f'error_attempt{attempt + 1}.json', dict(error=traceback.format_exc(), time=time.time()))
                resource_fallback = (work / 'single_actor_required.json').exists()
                if stop.is_set() or attempt or (OUT / 'offline_failure.json').exists() or _fatal_episode_failure() or (not loaded and not resource_fallback):
                    stop.set()
                    raise
            finally:
                for process in evaluators + [service]:
                    _terminate(process)
                release(gpu)
                atomic(work / 'exits.json', dict(service=None if service is None else service.poll(),
                                                evaluations=[process.poll() for process in evaluators], time=time.time()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cost-table', type=Path, default=OUT / 'cost_table.json')
    parser.add_argument('--exclude-gpus', type=int, nargs='*', default=[])
    args = parser.parse_args()
    with lock(OUT / 'controller.lock', blocking=False):
        atomic(OUT / 'process.json', dict(pid=os.getpid(), time=time.time()))
        wait_and_freeze(args.cost_table.resolve())
        last_scan, active, stop = -1e9, {}, threading.Event()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            try:
                while pending() or active:
                    for gpu, future in list(active.items()):
                        if future.done():
                            future.result()
                            del active[gpu]
                    need = ready_groups()
                    if need and len(active) < 3 and time.monotonic() - last_scan >= 30:
                        last_scan = time.monotonic()
                        free, identities = available(set(args.exclude_gpus) | set(active))
                        for gpu in free:
                            if len(active) >= 3 or need <= 0:
                                break
                            active[gpu] = executor.submit(worker, gpu, identities[gpu], stop)
                            need -= 1
                    counts = progress()
                    atomic(OUT / 'status.json', dict(status='running' if active else 'waiting_gpu',
                          phase='small40' if (OUT / 'pilot_mechanism_complete.json').exists() else
                                ('pilot_mechanism' if (OUT / 'pilot_complete.json').exists() else 'pilot'),
                          progress=counts, new_completed=sum(row['new'] for row in counts.values()),
                          reused=sum(row['reused'] for row in counts.values()), active_gpus=list(active), time=time.time()))
                    if pending() or active:
                        time.sleep(2)
            except BaseException:
                stop.set()
                raise
        counts = progress()
        if any(row['completed'] != row['target'] for row in counts.values()):
            raise RuntimeError('queue finished without the frozen forty records per condition')
        atomic(OUT / 'evaluation_complete.json', dict(passed=True, progress=counts,
               target_records=480, reused_episodes=80, new_episodes=400, time=time.time()))
        atomic(OUT / 'status.json', dict(status='evaluation_complete', progress=counts, time=time.time()))
        from .finish import main as finish
        audit = finish()
        if not audit['passed']:
            raise RuntimeError('final matrix audit failed; no completion marker was written')


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        atomic(OUT / 'failed.json', dict(error=traceback.format_exc(), time=time.time()))
        raise

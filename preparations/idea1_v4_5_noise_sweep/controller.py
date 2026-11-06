"""One 30-second poller; natural idle GPU leases and persistent paired services."""
import argparse
import json
import os
import shutil
import subprocess
import time
import traceback

import numpy as np

from preparations.idea1_v2_1.runtime_client import rpc
from preparations.idea1_v3.io import atomic, digest
from preparations.idea1_v3_1.common import environment, lock
from preparations.idea1_v4_0 import controller as native
from .common import ROOT, OUT, MODULE, directory, key, sha

UUIDS = {}


def scan():
    apps = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    occupied = {line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}
    raw = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'], text=True)
    free = []
    for line in raw.splitlines():
        gpu, identity, used = [item.strip() for item in line.split(',')]
        if identity not in occupied and float(used) <= 1500:
            free.append((int(gpu), identity))
    free.sort(key=lambda item: (item[0] != 3, item[0]))
    atomic(OUT/'resource_scan.json', dict(free=free, apps=apps, time=time.time()))
    return dict(free=free)


def pinned_environment(gpu):
    env = environment(gpu)
    env.update(CUDA_VISIBLE_DEVICES=UUIDS[gpu], CUDA_DEVICE_ORDER='PCI_BUS_ID')
    return env


def progress(manifest):
    counts = {}
    for cfg in manifest['conditions']:
        rows = []
        for case in manifest['cases']:
            d = directory(cfg['id'], case)
            if (d/'complete.json').exists():
                row = json.loads((d/'result.json').read_text())
                assert row['valid'] and row['manifest_hash'] == manifest['manifest_hash']
                rows.append(row)
        counts[cfg['id']] = dict(completed=len(rows), target=len(manifest['cases']),
                                 success=sum(row['success'] for row in rows))
    return counts


def unclaimed_cases(manifest):
    """Inspect advisory claims only in the single scheduler, without taking work."""
    unclaimed = 0
    for case in manifest['cases']:
        if (OUT/'case_done'/f'{key(case)}.json').exists():
            continue
        try:
            with lock(OUT/'claims'/f'{key(case)}.lock', blocking=False):
                if not (OUT/'case_done'/f'{key(case)}.json').exists():
                    unclaimed += 1
        except BlockingIOError:
            pass
    return unclaimed


def cross_service(worker):
    with lock(OUT/'cross_service.lock'):
        reference = OUT/'cross_service_reference.npz'
        if not reference.exists():
            reference.write_bytes((worker['wd']/'gate_actions.npz').read_bytes())
            atomic(OUT/'cross_service_reference.json', dict(worker=worker['wd'].name, sha256=sha(reference)))
        with np.load(reference) as expected, np.load(worker['wd']/'gate_actions.npz') as got:
            assert set(expected.files) == set(got.files), 'Cross-service gate keys differ'
            checks = [dict(key=name, passed=bool(np.allclose(expected[name], got[name], atol=1e-5, rtol=1e-5)),
                           max_abs=float(np.abs(expected[name]-got[name]).max())) for name in expected.files]
        atomic(worker['wd']/'cross_service.json',
               dict(passed=all(row['passed'] for row in checks), checks=checks, atol=1e-5, rtol=1e-5))
        assert all(row['passed'] for row in checks), 'Cross-service numerical mismatch'



def start_worker(gpu, gpu_uuid, workers):
    # Retain the identity/gates of any earlier finished service on this GPU.
    # A fresh service gets a clean gpuN directory and can never inherit readiness.
    wd = OUT/'workers'/f'gpu{gpu}'
    if wd.exists():
        assert (wd/'exits.json').exists(), 'Unreleased service directory already exists'
        old = wd.with_name(f'gpu{gpu}_service_{time.time_ns()}')
        wd.rename(old)
        for previous in workers:
            if previous['wd'] == wd:
                previous['wd'] = old
    return native.start(gpu, gpu_uuid)


def start_evaluation(worker):
    cross_service(worker)
    with (worker['wd']/'evaluate.log').open('a') as stream:
        worker['evaluation'] = subprocess.Popen(
            [str(native.EVAL), '-u', '-m', MODULE+'.evaluate', '--port', str(worker['port']),
             '--worker', worker['wd'].name], cwd=ROOT, env=worker['render'],
            stdout=stream, stderr=subprocess.STDOUT)
    atomic(worker['wd']/'evaluation_process.json', dict(pid=worker['evaluation'].pid, time=time.time()))
    worker['phase'] = 'evaluating'
    worker['phase_start'] = time.time()


def log_tail(path):
    if not path.exists():
        return ''
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-65536))
        return stream.read().decode(errors='replace')


def resource_oom(worker):
    return any('out of memory' in log_tail(worker['wd']/name).lower()
               for name in ('service.log', 'evaluate.log'))


def recover_resource(worker, retries, cooldown, idle_since):
    """Retire only our interrupted service; preserve every completed episode."""
    state_path = worker['wd']/'evaluation_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else None
    native.finish(worker)
    archive = OUT/'attempts'/f'resource_conflict_{time.time_ns()}'/worker['wd'].name
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(worker['wd']), str(archive))
    if state:
        case = state['case']
        done = OUT/'case_done'/f'{key(case)}.json'
        if not done.exists():
            recovery_path = OUT/'case_recovery'/f'{key(case)}.json'
            assert not recovery_path.exists(), 'Same paired case exceeded one service recovery'
            atomic(recovery_path,
                   dict(reason='resource_oom', attempt=1, case=case,
                        previous_server_pid=worker['service'].pid,
                        valid_results_preserved=True, archive=str(archive.relative_to(OUT)), time=time.time()))
    gpu = worker['gpu']
    retries[gpu] = retries.get(gpu, 0)+1
    cooldown[gpu] = time.time()+120
    idle_since.pop(worker['uuid'], None)
    print(f'GPU {gpu} resource conflict archived; retained other services and valid episodes', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-gpus', type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads((OUT/'manifest.json').read_text())
    identity = manifest['manifest_hash']
    assert digest({name: value for name, value in manifest.items() if name != 'manifest_hash'}) == identity
    assert all(sha(ROOT/name) == expected for name, expected in manifest['source_files'].items()), 'Frozen source changed'
    max_gpus = min(args.max_gpus, manifest['max_gpus'])
    assert 1 <= max_gpus <= 4
    native.OUT, native.MODULE = OUT, MODULE
    native.scan, native.environment = scan, pinned_environment
    workers, last_scan, idle_since, retries, cooldown = [], 0., {}, {}, {}
    with lock(OUT/'controller.lock', blocking=False):
        atomic(OUT/'controller_process.json', dict(pid=os.getpid(), started=time.time()))
        try:
            while True:
                counts = progress(manifest)
                completed = sum(row['completed'] for row in counts.values())
                paired_done = sum((OUT/'case_done'/f'{key(case)}.json').exists() for case in manifest['cases'])
                # Wait for pair checks too; result files alone do not finish the matrix.
                if completed == manifest['target_episodes'] and paired_done == len(manifest['cases']):
                    for worker in workers:
                        if worker['phase'] != 'done':
                            evaluated = worker['evaluation'] is not None
                            native.finish(worker)
                            atomic(worker['wd']/'terminal.json', dict(evaluated=evaluated, released=True,
                                   reason='matrix_complete' if evaluated else 'matrix_complete_before_evaluation', time=time.time()))
                            worker['phase'] = 'done'
                    atomic(OUT/'execution_complete.json', dict(time=time.time(), progress=counts, paired_cases=paired_done))
                    break
                active = [worker for worker in workers if worker['phase'] != 'done']
                if time.time()-last_scan >= 30 and len(active) < max_gpus:
                    now = time.time()
                    try:
                        free = scan()['free']
                    except (OSError, subprocess.SubprocessError) as error:
                        atomic(OUT/'resource_scan_error.json', dict(error=repr(error), time=now))
                        free = []
                    free_ids = {identity for _, identity in free}
                    idle_since = {identity: since for identity, since in idle_since.items() if identity in free_ids}
                    available = unclaimed_cases(manifest)-sum(worker['phase'] == 'loading' for worker in active)
                    for gpu, gpu_uuid in free:
                        if any(worker['gpu'] == gpu for worker in active) or retries.get(gpu, 0) >= 2:
                            continue
                        idle_since.setdefault(gpu_uuid, now)
                        # A late cold load needs at least two unclaimed paired cases;
                        # the first service is always allowed to finish a small remainder.
                        if available < (1 if not active else 2):
                            continue
                        if now-idle_since[gpu_uuid] < 60 or now < cooldown.get(gpu, 0):
                            continue
                        UUIDS[gpu] = gpu_uuid
                        worker = start_worker(gpu, gpu_uuid, workers)
                        if worker is not None:
                            workers.append(worker)
                            active.append(worker)
                            available -= 1
                            print(f'Leased GPU {gpu} ({gpu_uuid})', flush=True)
                        if len(active) == max_gpus:
                            break
                    last_scan = now
                for worker in list(workers):
                    if worker['phase'] == 'done':
                        continue
                    if worker['service'].poll() is not None:
                        if resource_oom(worker):
                            recover_resource(worker, retries, cooldown, idle_since)
                            workers.remove(worker)
                            continue
                        raise RuntimeError(f"Service failed: {worker['wd']}")
                    if worker['phase'] == 'loading':
                        try:
                            rpc(worker['port'], dict(op='health'))
                        except (OSError, ConnectionError):
                            if time.time()-worker['phase_start'] > 600:
                                raise TimeoutError('Model load and preflight')
                            continue
                        start_evaluation(worker)
                    elif worker['phase'] == 'evaluating' and worker['evaluation'].poll() is not None:
                        if worker['evaluation'].returncode:
                            if resource_oom(worker):
                                recover_resource(worker, retries, cooldown, idle_since)
                                workers.remove(worker)
                                continue
                            raise RuntimeError(f"Evaluation failed: {worker['wd']}")
                        native.finish(worker)
                        atomic(worker['wd']/'terminal.json', dict(evaluated=True, released=True,
                                                                reason='evaluation_finished', time=time.time()))
                        worker['phase'] = 'done'
                counts = progress(manifest)
                completed = sum(row['completed'] for row in counts.values())
                active = [worker for worker in workers if worker['phase'] != 'done']
                atomic(OUT/'status.json', dict(status='running' if active else 'waiting_gpu',
                       completed=completed, target=manifest['target_episodes'], progress=counts,
                       workers=[dict(gpu=worker['gpu'], phase=worker['phase']) for worker in workers], time=time.time()))
                if workers and not active and completed < manifest['target_episodes'] and not any(retries.values()):
                    raise RuntimeError('Evaluators ended before all cases completed; inspect technical errors')
                time.sleep(3)
        finally:
            for worker in workers:
                if worker['phase'] != 'done':
                    native.finish(worker)
    from .finish import finish
    finish()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        atomic(OUT/'failure.json', dict(error=traceback.format_exc(), time=time.time()))
        raise

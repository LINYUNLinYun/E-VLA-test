"""Lease genuinely idle GPUs for four representative module-probe suites."""
import fcntl
import argparse
import os
import subprocess
import time

from scripts.experiment10.common import ROOT, RESULTS, SUITES, dump_json, claim_gpu, managed_gpus
from scripts.experiment10.run_pipeline import free_gpus, event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('submodule', 'branches'), default='submodule')
    args = parser.parse_args()
    stage = args.stage
    lock = (RESULTS/('.'+stage+'_queue.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pending = [s for s in SUITES if not (RESULTS/stage/s/'summary.json').exists()]
    active, failures, next_poll = {}, [], 0.
    previous = managed_gpus()
    seen_ready = set()
    while pending or active:
        current = managed_gpus()
        if previous-current:
            next_poll = 0.
        previous = current
        for gpu, (process, suite, stream, lease) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            stream.close(); lease.close()
            event(event=stage+'_finished', suite=suite, gpu=gpu, code=code)
            if code:
                failures.append({'suite': suite, 'code': code})
            del active[gpu]
            next_poll = 0.
        ready = [s for s in pending if stage == 'submodule' or (RESULTS/'shadow'/s/'summary.json').exists()]
        if set(ready)-seen_ready:
            next_poll = 0.
            seen_ready.update(ready)
        if ready and len(current | set(active)) < 4 and time.monotonic() >= next_poll:
            next_poll = time.monotonic()+300
            for gpu in free_gpus():
                if not ready or gpu in active:
                    continue
                lease = claim_gpu(gpu)
                if lease is None:
                    continue
                suite = ready.pop(0)
                pending.remove(suite)
                env = os.environ.copy()
                env.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'JAX_PLATFORMS': 'cpu',
                            'PYTHONUNBUFFERED': '1', 'PYTHONPATH': str(ROOT/'src')+':'+str(ROOT),
                            'OPENPI_DATA_HOME': '/root/nfs/embodied/benchmark/shared/checkpoints/openpi'})
                if stage == 'branches':
                    env.update({'MUJOCO_GL': 'egl', 'MUJOCO_EGL_DEVICE_ID': str(gpu),
                                'LIBERO_CONFIG_PATH': str(ROOT/'.libero'),
                                'PYTHONPATH': env['PYTHONPATH']+':/root/nfs/embodied/benchmark/LIBERO'})
                executable = ROOT/('examples/libero/.venv/bin/python' if stage == 'branches' else '.venv/bin/python')
                module = 'branch_prefixes' if stage == 'branches' else 'probe_submodules'
                stream = (RESULTS/'logs'/(stage+'_'+suite+'.log')).open('a')
                process = subprocess.Popen([str(executable), '-m',
                            'scripts.experiment10.'+module, '--suite', suite],
                            cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                active[gpu] = process, suite, stream, lease
                event(event=stage+'_started', suite=suite, gpu=gpu, pid=process.pid)
        dump_json(RESULTS/(stage+'_status.json'), {'pid': os.getpid(), 'pending': pending,
             'active': [{'gpu': g, 'pid': p.pid, 'suite': s} for g, (p,s,_,_) in active.items()],
             'failures': failures, 'complete': not pending and not active and not failures})
        if pending or active:
            time.sleep(2)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

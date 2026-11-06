"""Resumable four-GPU queue. Dry-run by default; --launch explicitly permits GPU jobs."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.experiment10.common import ROOT, RESULTS, SUITES, PROTOCOL, dump_json, managed_gpus


def free_gpus():
    response = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                               '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    applications = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], check=True, capture_output=True, text=True)
    occupied = {line.split(',')[0].strip() for line in applications.stdout.splitlines() if line.strip()}
    available = []
    for line in response.stdout.splitlines():
        fields = [v.strip() for v in line.split(',')]
        if len(fields) == 4 and all(fields[i].isdigit() for i in (0, 2, 3)):
            gpu, used, utilization = map(int, (fields[0], fields[2], fields[3]))
            if used <= 2048 and utilization <= 10 and fields[1] not in occupied:
                available.append(gpu)
    return available


def event(**value):
    record = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **value}
    with (RESULTS / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(record) + '\n')
    print(json.dumps(record), flush=True)


def command(stage, suite):
    if stage == 'collect':
        return [str(ROOT / 'examples/libero/.venv/bin/python'), '-m',
                'scripts.experiment10.collect_observations', '--suite', suite]
    return [str(ROOT / '.venv/bin/python'), '-m', 'scripts.experiment10.probe_vlm', '--suite', suite]


class AdoptedWorker:
    def __init__(self, pid, suite):
        self.pid, self.suite = pid, suite

    def poll(self):
        process = Path('/proc') / str(self.pid)
        try:
            if process.exists() and process.joinpath('stat').read_text().split(') ', 1)[1][0] != 'Z':
                return None
        except FileNotFoundError:
            pass
        manifest = RESULTS / 'observations' / self.suite / 'manifest.json'
        if manifest.exists():
            data = json.loads(manifest.read_text())
            if data.get('complete') and data.get('episode_count') == 50:
                return 0
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--collect-only', action='store_true')
    parser.add_argument('--adopt', action='append', default=[], help='Existing collector suite:gpu:pid')
    args = parser.parse_args()
    jobs = [('collect', s) for s in SUITES]
    if not args.collect_only:
        jobs += [('probe', s) for s in SUITES]
    if not args.launch:
        print(json.dumps({'protocol': PROTOCOL, 'jobs': [{'stage': a, 'suite': b, 'command': command(a, b)}
                                                        for a, b in jobs]}, indent=2))
        return
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dump_json(RESULTS / 'protocol.json', PROTOCOL)
        active, next_poll, failures = {}, 0.0, []
        for value in args.adopt:
            suite, gpu, pid = value.split(':')
            if suite not in SUITES or int(gpu) in active:
                raise ValueError('Invalid adopted worker')
            active[int(gpu)] = (AdoptedWorker(int(pid), suite), 'collect', suite, open(os.devnull, 'w'))
            jobs.remove(('collect', suite))
            event(event='worker_adopted', suite=suite, gpu=int(gpu), pid=int(pid))
        while jobs or active:
            for gpu, (process, stage, suite, stream) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stream.close()
                event(event='job_finished', gpu=gpu, stage=stage, suite=suite, code=code)
                if code:
                    failures.append({'stage': stage, 'suite': suite, 'code': code})
                    jobs = [j for j in jobs if not (j[0] == 'probe' and j[1] == suite)]
                del active[gpu]
                # A known released own card can be reused without an nvidia-smi polling interval.
                next_poll = 0.0
            if (jobs and len(managed_gpus() | set(active)) < PROTOCOL['max_workers'] and
                    time.monotonic() >= next_poll):
                available = [g for g in free_gpus() if g not in active]
                next_poll = time.monotonic() + 300
                for gpu in available:
                    eligible = [j for j in jobs if j[0] == 'collect' or
                                (RESULTS / 'observations' / j[1] / 'manifest.json').exists()]
                    if not eligible:
                        break
                    stage, suite = eligible[0]
                    jobs.remove((stage, suite))
                    output = RESULTS / 'logs' / (stage + '_' + suite + '.log')
                    output.parent.mkdir(parents=True, exist_ok=True)
                    stream = output.open('a')
                    env = os.environ.copy()
                    env.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'MUJOCO_EGL_DEVICE_ID': str(gpu),
                                'MUJOCO_GL': 'egl', 'LIBERO_CONFIG_PATH': str(ROOT / '.libero'),
                                'JAX_PLATFORMS': 'cpu', 'OPENPI_DATA_HOME': '/root/nfs/embodied/benchmark/shared/checkpoints/openpi',
                                'PYTHONPATH': ':'.join([str(ROOT), str(ROOT / 'src'),
                                        '/root/nfs/embodied/benchmark/LIBERO']), 'PYTHONUNBUFFERED': '1'})
                    process = subprocess.Popen(command(stage, suite), cwd=ROOT, env=env,
                                               stdout=stream, stderr=subprocess.STDOUT)
                    active[gpu] = (process, stage, suite, stream)
                    event(event='job_started', gpu=gpu, stage=stage, suite=suite, pid=process.pid, log=str(output))
            dump_json(RESULTS / 'pipeline_status.json', {'pid': os.getpid(), 'active': [
                {'gpu': g, 'pid': p.pid, 'stage': s, 'suite': q} for g, (p, s, q, _) in active.items()],
                'pending': jobs, 'failures': failures, 'initial_stages_complete': not jobs and not active})
            if jobs or active:
                time.sleep(2)  # local child completion check, NOT a GPU-state poll
        event(event='initial_stages_finished', failures=failures)
        if failures:
            raise SystemExit(1)


if __name__ == '__main__':
    main()

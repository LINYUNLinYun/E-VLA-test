"""Poll for an unused GPU every minute and run the remaining experiments."""
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CWD = ROOT.parents[2]
PREFERRED = 'GPU-599477fb-0507-40dc-14df-d491f7a7b202'
STATUS = ROOT / 'gpu_watch_status.json'


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def devices():
    def query(args):
        return list(csv.reader(subprocess.check_output(
            ['nvidia-smi', *args, '--format=csv,noheader,nounits'], text=True, timeout=15).splitlines()))
    occupied = {row[0].strip() for row in query(['--query-compute-apps=gpu_uuid,pid']) if row}
    rows = []
    for index, uuid, memory, utilization in query(['--query-gpu=index,uuid,memory.used,utilization.gpu']):
        rows.append(dict(index=int(index), uuid=uuid.strip(), memory_mib=float(memory),
                         utilization=float(utilization), occupied=uuid.strip() in occupied))
    return sorted(rows, key=lambda row: (row['uuid'] != PREFERRED, row['index']))


def available(row):
    return not row['occupied'] and row['memory_mib'] <= 256 and row['utilization'] <= 1


def prepare_queue(device):
    jobs = []
    for filename in ('remaining_measurement_queue.json', 'slo_followup_queue.json'):
        source = json.loads((ROOT / filename).read_text())
        jobs.extend(dict(job, source_queue=filename) for job in source['jobs'] if job['state'] == 'pending')
    if device['uuid'] != PREFERRED:
        # Preserve the GPU1 artifacts and add same-device controls/repetition.
        template = next(j for j in jobs if j['name'] == 'crossframe_capacity_repeat100')
        extra = []
        for name, period, pairs in [('crossframe_capacity_initial', 0, 100),
                                    ('crossframe_arrival77_w0_control', 77, 0),
                                    ('crossframe_arrival154_w0_control', 154, 0)]:
            argv = template['argv'].copy()
            if '--modes' in argv:
                argv = argv[:argv.index('--modes')]
            argv[argv.index('--output') + 1] = str(ROOT / name)
            argv[argv.index('--pairs') + 1] = str(pairs)
            argv.extend(['--arrival-period-ms', str(period)])
            extra.append(dict(name=name, state='pending', argv=argv,
                              source_queue='same_device_controls'))
        jobs = extra + jobs
        for job in jobs:
            job['original_name'] = job['name']
            job['name'] += '_gpu' + str(device['index'])
            job['argv'] = job['argv'].copy()
            job['argv'][job['argv'].index('--output') + 1] = str(ROOT / job['name'])
            job['argv'][job['argv'].index('--gpu-uuid') + 1] = device['uuid']
    path = ROOT / ('auto_queue_gpu' + str(device['index']) + '.json')
    if path.exists():
        raise RuntimeError('Auto queue already exists; inspect it before replacing evidence')
    save(path, dict(gpu_uuid=device['uuid'], jobs=jobs,
                    comparison_policy='Use within-case serial controls; never pool physical GPUs silently'))
    for filename in ('remaining_measurement_queue.json', 'slo_followup_queue.json'):
        source_path = ROOT / filename
        source = json.loads(source_path.read_text())
        mapping = {job.get('original_name', job['name']): job['name'] for job in jobs
                   if job.get('source_queue') == filename}
        for job in source['jobs']:
            if job['name'] in mapping and job['state'] == 'pending':
                job.update(state='delegated_to_auto_queue', auto_queue=str(path),
                           auto_case=mapping[job['name']])
        save(source_path, source)
    return path


def retry_occupancy_failure(path):
    data = json.loads(path.read_text())
    for index, job in enumerate(data['jobs']):
        if job['state'] != 'failed' or job.get('retry_queued'):
            continue
        result_path = Path(job['argv'][job['argv'].index('--output') + 1]) / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        error = result.get('error', '')
        if not any(marker in error for marker in ('GPU process occupancy changed', 'GPU is not exclusive/idle')):
            return False
        retry = dict(name=job['name'] + '_occupancy_retry', state='pending',
                     argv=job['argv'].copy(), retry_of=job['name'])
        retry['argv'][retry['argv'].index('--output') + 1] = str(ROOT / retry['name'])
        job['retry_queued'] = retry['name']
        data['jobs'].insert(index + 1, retry)
        save(path, data)
        return True
    # A pre-launch idle check may fail before any job changes state.
    return any(job['state'] == 'pending' for job in data['jobs'])


def main():
    lock = (ROOT / 'gpu_watch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    selected = queue = None
    if STATUS.exists():
        previous = json.loads(STATUS.read_text())
        if previous.get('selected_gpu') and previous.get('queue'):
            selected = previous['selected_gpu']
            queue = Path(previous['queue'])
    while True:
        try:
            rows = devices()
        except Exception as error:
            save(STATUS, dict(state='poll_error', pid=os.getpid(), checked_unix=time.time(),
                             error=repr(error), selected_gpu=selected, queue=str(queue) if queue else None))
            time.sleep(60)
            continue
        candidates = [row for row in rows if available(row) and
                      (selected is None or row['uuid'] == selected['uuid'])]
        state = dict(state='waiting', pid=os.getpid(), checked_unix=time.time(), poll_seconds=60,
                     devices=rows, selected_gpu=selected, queue=str(queue) if queue else None)
        save(STATUS, state)
        print(time.strftime('%F %T'), 'idle candidates:', [r['index'] for r in candidates], flush=True)
        if not candidates:
            time.sleep(60)
            continue
        if selected is None:
            selected = candidates[0]
            queue = prepare_queue(selected)
        state.update(state='running', selected_gpu=selected, queue=str(queue))
        save(STATUS, state)
        command = [sys.executable, '-u', str(ROOT / 'run_plan.py'),
                   '--gpu-uuid', selected['uuid'], queue.name]
        with (ROOT / 'auto_queue.log').open('a') as log:
            child = subprocess.Popen(command, cwd=CWD, stdout=log, stderr=subprocess.STDOUT)
            state['launcher_pid'] = child.pid
            save(STATUS, state)
            code = child.wait()
        state.update(returncode=code, ended_unix=time.time())
        if code == 0:
            state['state'] = 'queue_completed_review_results'
            save(STATUS, state)
            return
        if not retry_occupancy_failure(queue):
            state['state'] = 'experiment_error_needs_review'
            save(STATUS, state)
            return
        state['state'] = 'waiting_after_occupancy_failure'
        save(STATUS, state)
        time.sleep(60)


if __name__ == '__main__':
    main()

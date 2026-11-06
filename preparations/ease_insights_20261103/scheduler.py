"""A single natural-idle GPU poller for the independently frozen experiments.

Queue entries contain an argv list, dependencies, and a required completion file.
Only this process allocates GPUs. It never terminates unrelated processes.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def lock(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    stream = open(path, 'a')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        stream.close()
        raise
    return stream


def scan(allowed=None):
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader'], text=True, timeout=15)
    busy = {line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                  '--format=csv,noheader,nounits'], text=True, timeout=15)
    records = []
    for line in raw.splitlines():
        index, identity, memory, utilization = [part.strip() for part in line.split(',')]
        selected = allowed is None or int(index) in allowed
        records.append(dict(index=int(index), uuid=identity, memory_mib=float(memory),
                            utilization=float(utilization), allowed=selected,
                            idle=selected and identity not in busy and float(memory) < 500))
    return dict(time=time.time(), devices=records, compute_apps=apps)


def process_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def eligible(jobs, state):
    return [job for job in jobs
            if state.get(job['id'], {}).get('status', 'pending') == 'pending'
            and all(state.get(dep, {}).get('status') == 'complete' for dep in job.get('depends_on', []))
            and all(Path(p).is_file() for p in job.get('ready_files', []))]


def substitute(value, values):
    result = str(value)
    for name, replacement in values.items():
        result = result.replace('{' + name + '}', str(replacement))
    return result


def finish_owned(worker):
    process = worker['process']
    if process.poll() is None:
        # Each launched job has its own session and owns its server/client children.
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    worker['log'].close()
    worker['lease'].close()


def run(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    owner = lock(out/'controller.lock')
    state_path = out/'jobs_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    # A restarted scheduler must not duplicate a still-running job.
    for key, record in state.items():
        if record['status'] == 'running':
            if process_alive(record['pid']):
                raise RuntimeError(f"Existing job {key} is still live as PID {record['pid']}")
            record.update(status='interrupted', error='Previous process is no longer live; review partial outputs')
    allowed = None if args.gpus == 'all' else {int(x) for x in args.gpus.split(',')}
    active = {}
    last_scan = 0.
    idle_since = {}
    write(out/'controller_process.json', dict(pid=os.getpid(), started=time.time(), argv=os.sys.argv))
    print(json.dumps(dict(event='scheduler_started', pid=os.getpid(), queue=args.queue)), flush=True)
    try:
        while True:
            if (out/'stop_after_jobs.json').exists() and not active:
                break
            queue = json.loads(Path(args.queue).read_text())
            jobs = queue['jobs']
            ids = [job['id'] for job in jobs]
            if len(ids) != len(set(ids)):
                raise ValueError('Duplicate job ids')
            for job in jobs:
                if not set(job.get('depends_on', [])).issubset(ids):
                    raise ValueError(f"Unknown dependency for {job['id']}")
                state.setdefault(job['id'], dict(status='pending'))
                if state[job['id']]['status'] == 'complete' and not Path(job['complete_file']).is_file():
                    raise RuntimeError(f"Completed output missing for {job['id']}")
            for key, worker in list(active.items()):
                code = worker['process'].poll()
                if code is None:
                    continue
                complete_file = Path(worker['job']['complete_file'])
                successful = code == 0 and complete_file.is_file()
                state[key].update(status='complete' if successful else 'failed',
                                  returncode=code, finished=time.time(), completion_file=str(complete_file),
                                  completion_file_present=complete_file.is_file())
                print(json.dumps(dict(event='job_finished', job=key, **state[key])), flush=True)
                finish_owned(worker)
                del active[key]
            ready = eligible(jobs, state)
            if ready and len(active) < args.max_gpus and time.monotonic()-last_scan >= args.scan_seconds:
                last_scan = time.monotonic()
                try:
                    resource = scan(allowed)
                    write(out/'resource_scan.json', resource)
                except (OSError, subprocess.SubprocessError) as error:
                    write(out/'resource_scan_error.json', dict(error=repr(error), time=time.time()))
                    resource = dict(devices=[])
                for device in resource['devices']:
                    uuid = device['uuid']
                    if not device['idle']:
                        idle_since.pop(uuid, None)
                        continue
                    idle_since.setdefault(uuid, time.monotonic())
                    if time.monotonic()-idle_since[uuid] < args.idle_seconds:
                        continue
                    if not ready or len(active) >= args.max_gpus:
                        break
                    try:
                        lease = lock(Path('/tmp')/f'embodied-benchmark-gpu-{uuid}.lock')
                    except BlockingIOError:
                        continue
                    # Close the allocation race after the shared lock is acquired.
                    if not any(d['uuid'] == uuid and d['idle'] for d in scan(allowed)['devices']):
                        lease.close()
                        continue
                    job = ready.pop(0)
                    values = dict(gpu=device['index'], gpu_uuid=uuid, port=port())
                    argv = [substitute(arg, values) for arg in job['argv']]
                    env = dict(os.environ)
                    env.update({k: substitute(v, values) for k,v in job.get('env', {}).items()})
                    log = open(out/f"{job['id']}.log", 'a')
                    try:
                        process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True, pass_fds=(lease.fileno(),))
                    except BaseException:
                        lease.close()
                        log.close()
                        raise
                    state[job['id']].update(status='running', pid=process.pid, gpu=device['index'],
                                           gpu_uuid=uuid, port=values['port'], started=time.time(), argv=argv)
                    active[job['id']] = dict(process=process, lease=lease, log=log, job=job)
                    print(json.dumps(dict(event='job_started', job=job['id'], **state[job['id']])), flush=True)
                    write(state_path, state)
            write(state_path, state)
            pending = [k for k in ids if state[k]['status'] == 'pending']
            failed = [k for k in ids if state[k]['status'] in ('failed', 'interrupted')]
            status = 'running' if active else ('waiting_gpu' if eligible(jobs,state) else 'waiting_dependencies')
            if not active and not pending:
                status = 'failed' if failed else 'complete'
            write(out/'status.json', dict(status=status, jobs={k:state[k]['status'] for k in ids},
                  active_pids=[v['process'].pid for v in active.values()], failed=failed, time=time.time()))
            if status in ('complete', 'failed'):
                return 0 if status == 'complete' else 1
            if not active and failed and not eligible(jobs,state):
                return 1
            time.sleep(3)
    finally:
        for key, worker in active.items():
            finish_owned(worker)
            state[key].update(status='interrupted', finished=time.time())
        write(state_path, state)
        owner.close()
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--queue', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--gpus', default='all')
    p.add_argument('--max-gpus', type=int, default=2)
    p.add_argument('--scan-seconds', type=float, default=30.)
    p.add_argument('--idle-seconds', type=float, default=25.)
    args = p.parse_args()
    try:
        raise SystemExit(run(args))
    except Exception:
        write(Path(args.output)/'controller_failure.json', dict(error=traceback.format_exc(), time=time.time()))
        raise


if __name__ == '__main__':
    main()

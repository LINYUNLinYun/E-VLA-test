"""One-shot CPU recovery for a zero-case startup failure; frozen science unchanged."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/root/nfs/embodied/benchmark')
STAGE = ROOT / 'results/ease_insights_20261103/vlm_within_suite_v1/discovery'
OUT = Path(__file__).resolve().parent
UUID = 'GPU-599477fb-0507-40dc-14df-d491f7a7b202'
MANIFEST = '6db11c5a8f1cdb5eab549f406ece3ba7b14661b2699da407c1ac87073e6c8a7d'
FILE_SHA = 'e9dc20f236a74da7e10493488ea644f0050501159f0c2e390e77c8f2ab653ddb'
CONTROLLER_SHA = '10b6a2cbb31e5cb7ebf654d985f0a215ae6d46a1564f865b6a76be87dc697e5b'
WATCH_SHA = '4314b4b5d7eca4f88968b0d66da305a5cd2f6aff4829f363ae76b951b9495680'
DIAGNOSIS_SHA = 'a516d62b7315a82bae5c3798b94cad36ac0395af54a5faf344629b9ec71b76cf'
CONDITIONS = ('full_d10', 'jac_anchor', 'local_spatial', 'local_object', 'local_goal', 'local_long')


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def proc(pid):
    try:
        data = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'state': data[0], 'start_ticks': data[19]}
    except FileNotFoundError:
        return None


def require(value, message):
    if not value:
        raise RuntimeError(message)


def fixed_identity():
    for pid, start in ((385842, '2618445988'), (600088, '2622761021'),
                       (631915, '2623408624'), (632160, '2623410216')):
        p = proc(pid)
        require(p and p['state'] != 'Z' and p['start_ticks'] == start, 'Original process identity changed')
    old = proc(780012)
    require(old is None or old['state'] == 'Z', 'Previous borrower is still alive')
    require(sha(STAGE / 'manifest.json') == FILE_SHA, 'Frozen manifest file changed')
    require(sha(ROOT / 'preparations/ease_insights_20261103/vlm_within_suite_v1/borrowed_controller.py') == CONTROLLER_SHA,
            'Frozen borrower source changed')
    require(sha(OUT / 'diagnosis.json') == DIAGNOSIS_SHA, 'Zero-case diagnosis identity changed')
    diagnosis = read(OUT / 'diagnosis.json')
    require(diagnosis.get('zero_case_startup_failure_verified') is True and
            diagnosis.get('scientific_manifest_hash') == MANIFEST, 'Missing zero-case startup diagnosis')
    failed = Path(diagnosis['failed_run'])
    require(sha(failed / 'server.log') == diagnosis['server_log_sha256'] and
            sha(failed / 'identity.json') == diagnosis['identity_sha256'] and
            not (failed / 'protocol.json').exists(), 'Failed run changed or reached a serving protocol')


def complete():
    for cid in CONDITIONS:
        path = STAGE / cid / 'discovery_shard00of01_complete.json'
        if not path.exists():
            return False
        d = read(path)
        require(d.get('complete') is True and d.get('manifest_hash') == MANIFEST and
                d.get('condition') == cid and d.get('completed_after', d.get('already_completed')) == 72,
                'Existing completion marker differs')
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'singleton.lock').open('a') as singleton:
        fcntl.flock(singleton.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fixed_identity()
        require(not (OUT / 'launch_intent.json').exists(), 'A launch was already intended; manual identity reconciliation required')
        own = proc(os.getpid())
        write(OUT / 'waiter.json', {'pid': os.getpid(), **own, 'source_sha256': sha(Path(__file__)),
              'policy_seconds': [15, 2], 'created': time.time(), 'no_gpu_query': True})
        first = previous = None
        samples = []
        while True:
            fixed_identity()
            if complete():
                write(OUT / 'status.json', {'phase': 'stage_already_complete', 'time': time.time(), 'no_launch': True})
                return
            now = time.time()
            status = read(ROOT / 'results/ease_insights_20261103/gpu_release_watch/status.json')
            valid = (status.get('pid') == 385842 and str(status.get('proc_start_ticks')) == '2618445988' and
                     status.get('run_id') == '060b0172-ba52-4c9b-9c39-03c7fb604d69' and
                     status.get('source_sha256') == WATCH_SHA and status.get('error') is None and
                     status.get('poll_seconds') == 2 and
                     0 <= now - status.get('query_finished_at', 0) <= 3)
            device = next((d for d in status.get('devices', []) if d.get('gpu_uuid') == UUID), {})
            available = (valid and device.get('compute_process_count') == 1 and
                         device.get('host_compute_pids') == [1975205] and 0 < device.get('memory_mib', 0) <= 512)
            query = status.get('query_finished_at', 0)
            if not available or (previous is not None and (query < previous or query - previous > 3)):
                first = previous = None
                samples = []
            if available and query != previous:
                first = query if first is None else first
                samples.append({'query_finished_at': query, 'memory_mib': device['memory_mib'],
                                'compute_process_count': device['compute_process_count']})
                previous = query
            elapsed = query - first if available and first is not None else 0
            write(OUT / 'status.json', {'phase': 'waiting_owner_available', 'time': now,
                  'available_seconds': elapsed, 'device': device, 'fresh_cache': valid, 'policy_seconds': [15, 2]})
            if elapsed >= 15 and len(samples) >= 9:
                fixed_identity()
                registry_path = STAGE / 'controllers/controller_processes.json'
                registry = read(registry_path)
                old_entry = next(p for p in registry['processes'] if p['gpu_uuid'] == UUID)
                require(old_entry['pid'] == 780012, 'Borrower registry changed; refusing duplicate launch')
                tag = str(time.time_ns())
                write(OUT / ('previous_registry_' + tag + '.json'), registry)
                log = OUT / ('restarted_borrower_' + tag + '.log')
                argv = [str(ROOT / 'policies/realtime-vla-flash/.venv/bin/python'), '-B', '-u', '-m',
                        'preparations.ease_insights_20261103.vlm_within_suite_v1.borrowed_controller',
                        '--output', str(STAGE), '--gpu-uuid', UUID, '--launch']
                require(old_entry['argv'] == argv, 'Original borrower argv changed')
                intent = {'time': time.time(), 'argv': argv, 'log': str(log), 'availability_samples': samples,
                          'previous_borrower_pid': 780012, 'reconciliation_required_before_any_retry': True}
                fd = os.open(OUT / 'launch_intent.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(fd, 'w') as stream:
                    json.dump(intent, stream, indent=2)
                    stream.write('\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                with log.open('ab', buffering=0) as stream:
                    child = subprocess.Popen(argv, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                             stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
                actual = proc(child.pid)
                require(actual and actual['state'] != 'Z', 'New borrower exited immediately')
                entry = dict(old_entry, pid=child.pid, start_ticks=actual['start_ticks'], state=actual['state'],
                             log=str(log), started_unix=time.time(), recovery='foreign_oom_zero_case_startup',
                             owner_available_seconds=elapsed, available_query_start=first, available_query_end=query)
                registry['processes'] = [entry if p['gpu_uuid'] == UUID else p for p in registry['processes']]
                registry['time'] = time.time()
                write(registry_path, registry)
                receipt = {'phase': 'borrower_launched', 'time': time.time(), 'process': entry,
                           'availability_samples': samples, 'policy_seconds': [15, 2],
                           'original_gpu_holders_retained': True, 'scientific_sources_unchanged': True,
                           'completed_conditions_preserved': True, 'failed_run_preserved': True}
                write(OUT / ('launch_' + tag + '.json'), receipt)
                write(OUT / 'status.json', receipt)
                print(json.dumps({'phase': 'borrower_launched', 'pid': child.pid, 'available_seconds': elapsed}), flush=True)
                return
            time.sleep(2)


if __name__ == '__main__':
    main()

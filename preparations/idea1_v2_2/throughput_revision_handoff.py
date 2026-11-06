"""Stop a direct worker only at a ledger-proven committed episode boundary."""
import argparse
import os
from pathlib import Path
import select
import signal
import time

from common import atomic_json, exclusive, lock, sha, start_ticks
from throughput_revision import acceptance, read, verify_revision


def process_identity(pid):
    base = Path(f'/proc/{pid}')
    fields = (base/'stat').read_text().split(') ', 1)[1].split()
    return dict(pid=pid, start_ticks=fields[19], ppid=int(fields[1]), state=fields[0],
                argv=(base/'cmdline').read_bytes().decode().rstrip('\0').split('\0'),
                cwd=os.readlink(base/'cwd'))


def argument(argv, name):
    if argv.count(name) != 1:
        raise ValueError(f'Expected exactly one {name}')
    return argv[argv.index(name)+1]


def resolved_argument(process, name):
    value = Path(argument(process['argv'], name))
    return (Path(process['cwd'])/value).resolve() if not value.is_absolute() else value.resolve()


def validate_targets(attempt, model, gpu, worker, server, worker_ticks, server_ticks):
    attempt = Path(attempt).resolve()
    if worker['start_ticks'] != worker_ticks or server['start_ticks'] != server_ticks:
        raise ValueError('PID start ticks changed')
    if worker['state'] in ('Z', 'X') or server['state'] in ('Z', 'X') or server['ppid'] != worker['pid']:
        raise ValueError('Expected live direct worker/server parent relationship')
    for process, script in ((worker, 'runtime_runner.py'), (server, 'runtime_server.py')):
        scripts = [Path(value).name for value in process['argv']]
        if script not in scripts or argument(process['argv'], '--model') != model:
            raise ValueError('Target command/model mismatch')
        script_arg = next(value for value in process['argv'] if Path(value).name == script)
        path = Path(script_arg)
        path = (Path(process['cwd'])/path).resolve() if not path.is_absolute() else path.resolve()
        if path != Path(__file__).resolve().with_name(script):
            raise ValueError('Target is not the frozen runtime entrypoint')
    if ('worker' not in worker['argv'] or '--launch' not in worker['argv'] or
            resolved_argument(worker, '--attempt') != attempt or argument(worker['argv'], '--gpu') != str(gpu)):
        raise ValueError('Worker attempt/GPU mismatch')
    directory = attempt/'services'/f'{model}_gpu{gpu}'
    if resolved_argument(server, '--output-dir') != directory:
        raise ValueError('Server output directory mismatch')
    lease = read(directory/'service_lease.json')
    if (lease['pid'] != server['pid'] or lease['start_ticks'] != server_ticks or lease['gpu'] != str(gpu) or
            str(lease['port']) != argument(server['argv'], '--port')):
        raise ValueError('Server lease identity mismatch')
    parent = process_identity(worker['ppid'])
    if any(Path(value).name == 'runtime_runner.py' for value in parent['argv']):
        raise ValueError('Scheduler-owned worker requires a separately reviewed handoff')
    return dict(worker=worker, server=server, worker_parent=parent,
                service_lease_sha256=sha(directory/'service_lease.json'))


def has_reserved(entries, pid, ticks):
    return any(entry.get('status') == 'reserved' and entry.get('pid') == pid and
               entry.get('start_ticks') == ticks for entry in entries)


def wait_pidfd(fd, timeout=10):
    poll = select.poll()
    poll.register(fd, select.POLLIN)
    if not poll.poll(int(timeout*1000)):
        raise RuntimeError('Target did not exit after SIGTERM; do not release ownership or force-kill')


def handoff(revision, model, gpu, worker_pid, server_pid, worker_ticks, server_ticks, execute=False):
    revision = Path(revision).resolve()
    identity, _ = verify_revision(revision)
    acceptance_hash = acceptance(revision, model, 'source')
    attempt = Path(identity['source_attempt'])
    worker = process_identity(worker_pid)
    server = process_identity(server_pid)
    evidence = validate_targets(attempt, model, gpu, worker, server, worker_ticks, server_ticks)
    receipt = dict(schema=1, revision_manifest_sha256=sha(revision/'revision_manifest.json'),
        source_acceptance_sha256=acceptance_hash, model=model, gpu=gpu,
        target=evidence, execute=execute, timestamp=time.time())
    if not execute:
        return dict(receipt, status='inspected_only')
    # pidfds pin the precise processes even if numeric PIDs are later recycled.
    worker_fd, server_fd = os.pidfd_open(worker_pid), os.pidfd_open(server_pid)
    try:
        with lock(attempt/'episode_ledger.lock'):
            # The same LOCK_EX protects every original reserve and valid update.
            validate_targets(attempt, model, gpu, process_identity(worker_pid), process_identity(server_pid), worker_ticks, server_ticks)
            ledger = read(attempt/'episode_ledger.json')
            if has_reserved(ledger['entries'], worker_pid, worker_ticks):
                return dict(receipt, status='busy_active_episode_no_signal')
            snapshot = revision/'handoff_snapshots'/f'{model}_{time.time_ns()}.json'
            atomic_json(snapshot, ledger)
            receipt.update(ledger_sha256=sha(attempt/'episode_ledger.json'),
                           ledger_snapshot=str(snapshot), ledger_snapshot_sha256=sha(snapshot),
                           boundary='ledger LOCK_EX held; no target PID/startticks reserved entry')
            signal.pidfd_send_signal(worker_fd, signal.SIGTERM)
            wait_pidfd(worker_fd)
            receipt['worker_stopped_under_ledger_lock'] = True
        # The worker cannot start another episode after the transaction lock is released.
        if start_ticks(server_pid) == server_ticks:
            signal.pidfd_send_signal(server_fd, signal.SIGTERM)
            wait_pidfd(server_fd, 30)
        if start_ticks(worker_pid) is not None or start_ticks(server_pid) is not None:
            raise RuntimeError('A target is still alive; no new owner may start')
        # Demonstrate that every required ownership lock has actually been released.
        for path in [attempt/'leases'/f'{model}.shard{shard}.lock' for shard in (0, 1)] + [Path(__file__).resolve().parent/'readiness'/'gpu_leases'/f'gpu{gpu}.lock']:
            with exclusive(path):
                pass
        receipt.update(status='safe_handoff_complete', locks_released=True, completed=time.time())
        atomic_json(revision/f'{model}.handoff_complete.json', receipt)
        return receipt
    finally:
        os.close(worker_fd)
        os.close(server_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--server', type=int, required=True)
    parser.add_argument('--worker-start-ticks', required=True)
    parser.add_argument('--server-start-ticks', required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    print(__import__('json').dumps(handoff(args.revision, args.model, args.gpu, args.worker, args.server,
        args.worker_start_ticks, args.server_start_ticks, args.execute), indent=2))


if __name__ == '__main__':
    main()

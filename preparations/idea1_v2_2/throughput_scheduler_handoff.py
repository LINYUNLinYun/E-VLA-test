"""Independent, narrowly bound pi0 scheduler handoff; never changes signed v1 code."""
import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

from common import atomic_json, exclusive, lock, sha, start_ticks
from throughput_revision import acceptance, read, verify_revision
from throughput_revision_handoff import argument, process_identity, resolved_argument, wait_pidfd, has_reserved

HERE = Path(__file__).resolve().parent


def children_map():
    output = subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True)
    mapping = {}
    for line in output.splitlines():
        pid, ppid = map(int, line.split())
        mapping.setdefault(ppid, []).append(pid)
    return mapping


def live_children(pid, mapping=None):
    result = []
    for child in (mapping if mapping is not None else children_map()).get(pid, []):
        try:
            identity = process_identity(child)
        except FileNotFoundError:
            continue
        if identity['state'] not in ('Z', 'X'):
            result.append(identity)
    return result


def stable_identity(identity):
    return {key: identity[key] for key in ('pid', 'start_ticks', 'ppid', 'argv', 'cwd')}


def lock_fds(pid, paths):
    records = {}
    for fd in Path(f'/proc/{pid}/fd').iterdir():
        try:
            target = Path(os.readlink(fd))
        except FileNotFoundError:
            continue
        if target in paths:
            stat = fd.stat()
            expected = target.stat()
            info = fd.with_name(fd.name).parent.parent/'fdinfo'/fd.name
            lock_lines = [line for line in info.read_text().splitlines() if line.startswith('lock:')]
            if (stat.st_ino, stat.st_dev) != (expected.st_ino, expected.st_dev) or not any('FLOCK' in line and 'WRITE' in line for line in lock_lines):
                raise ValueError('Expected live advisory ownership lock')
            records[str(target)] = dict(fd=int(fd.name), inode=stat.st_ino, device=stat.st_dev, locks=lock_lines)
    if set(records) != {str(path) for path in paths}:
        raise ValueError('Required ownership lock descriptors are absent')
    return records


def compiler_tree(server_pid):
    mapping = children_map()
    queue = [server_pid]
    result = []
    expected_script = HERE.parents[1]/'policies/realtime-vla-flash/.venv/lib/python3.11/site-packages/torch/_inductor/compile_worker/__main__.py'
    while queue:
        parent = queue.pop()
        for identity in live_children(parent, mapping):
            argv = identity['argv']
            if (len(argv) < 3 or Path(argv[1]).resolve() != expected_script.resolve() or
                    f'--parent={server_pid}' not in argv or '--kind=fork' not in argv or
                    '--pickler=torch._inductor.compile_worker.subproc_pool.SubprocPickler' not in argv):
                raise ValueError(f"Unreviewed server descendant: {identity['pid']}")
            result.append(identity)
            queue.append(identity['pid'])
    return sorted(result, key=lambda item: item['pid'])


def inspect_chain(revision, scheduler_pid, worker_pid, server_pid):
    identity, _ = verify_revision(revision)
    attempt = Path(identity['source_attempt'])
    scheduler, worker, server = [process_identity(pid) for pid in (scheduler_pid, worker_pid, server_pid)]
    for process, script, stage in ((scheduler, 'runtime_runner.py', 'run'), (worker, 'runtime_runner.py', 'worker')):
        argv = process['argv']
        if (len(argv) < 3 or (Path(process['cwd'])/argv[1]).resolve() != HERE/script or argv[2] != stage or
                argument(argv, '--model') != 'pi0' or resolved_argument(process, '--attempt') != attempt or '--launch' not in argv):
            raise ValueError('Scheduler/worker stage, model or attempt differs')
    if worker['ppid'] != scheduler_pid or server['ppid'] != worker_pid or argument(worker['argv'], '--gpu') != '0' or argument(worker['argv'], '--shard') != '0':
        raise ValueError('Expected exact scheduler -> pi0 shard0/GPU0 worker -> server chain')
    argv = server['argv']
    service = attempt/'services'/'pi0_gpu0'
    lease = read(service/'service_lease.json')
    if (len(argv) < 2 or Path(argv[1]).resolve() != HERE/'runtime_server.py' or argument(argv, '--model') != 'pi0' or
            resolved_argument(server, '--output-dir') != service or lease['pid'] != server_pid or
            lease['start_ticks'] != server['start_ticks'] or lease['gpu'] != '0' or
            str(lease['port']) != argument(argv, '--port')):
        raise ValueError('Server command/lease differs')
    if [child['pid'] for child in live_children(scheduler_pid)] != [worker_pid]:
        raise ValueError('Scheduler must have exactly the one reviewed live worker')
    paths = [attempt/'leases'/'scheduler.lock', HERE/'readiness/gpu_leases/gpu0.lock']
    return dict(scheduler=scheduler, worker=worker, server=server,
        scheduler_lock_fds=lock_fds(scheduler_pid, paths),
        worker_lock_fds=lock_fds(worker_pid, [attempt/'leases'/'pi0.shard0.lock']),
        compiler_tree=compiler_tree(server_pid), service_lease_sha256=sha(service/'service_lease.json'))


def native_probe():
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Use the reviewed /usr/bin/python3 native pidfd interpreter')
    fd = os.pidfd_open(os.getpid())
    try:
        signal.pidfd_send_signal(fd, 0)
    finally:
        os.close(fd)
    return dict(executable=str(Path(sys.executable).resolve()), executable_sha256=sha(Path(sys.executable).resolve()),
                version=sys.version, native_pidfd_self_signal0=True)


def prepare(revision, destination, scheduler_pid, worker_pid, server_pid):
    revision, destination = Path(revision).resolve(), Path(destination).resolve()
    chain = inspect_chain(revision, scheduler_pid, worker_pid, server_pid)
    if any(chain[name]['state'] in ('T', 't', 'Z', 'X') for name in ('scheduler', 'worker', 'server')):
        raise ValueError('Expected running, live targets; do not adopt a pre-stopped scheduler')
    original, _ = verify_revision(revision)
    if (Path(original['source_attempt'])/'PAUSE').exists():
        raise ValueError('Existing PAUSE must be handled by its owner')
    destination.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema=1, kind='pi0_scheduler_boundary_handoff', throughput_revision=str(revision),
        throughput_manifest_sha256=sha(revision/'revision_manifest.json'),
        helper_source=str(Path(__file__).resolve()), helper_source_sha256=sha(__file__),
        interpreter=native_probe(), chain=chain, pause_before='absent', created=time.time())
    atomic_json(destination/'manifest.json', manifest)
    return manifest


def verify_plan(destination):
    destination = Path(destination).resolve()
    plan = read(destination/'manifest.json')
    revision = Path(plan['throughput_revision'])
    verify_revision(revision)
    if plan['schema'] != 1 or plan['kind'] != 'pi0_scheduler_boundary_handoff' or plan['helper_source_sha256'] != sha(__file__) or plan['throughput_manifest_sha256'] != sha(revision/'revision_manifest.json'):
        raise ValueError('Handoff source/throughput identity changed')
    if native_probe() != plan['interpreter']:
        raise ValueError('Handoff interpreter differs from reviewed probe')
    signed = read(destination/'acceptance.json')
    if (signed.get('schema') != 1 or signed.get('accepted') is not True or
            signed.get('reviewer_model') != 'gpt-6.1-sol' or signed.get('reasoning_effort') != 'medium' or
            signed.get('handoff_manifest_sha256') != sha(destination/'manifest.json') or not signed.get('evidence')):
        raise ValueError('Independent scheduler handoff acceptance missing/stale')
    acceptance(revision, 'pi0', 'source')
    return plan


def same_chain(actual, expected):
    for name in ('scheduler', 'worker', 'server'):
        if stable_identity(actual[name]) != stable_identity(expected[name]):
            raise ValueError(f'{name} process identity changed')
    if [stable_identity(item) for item in actual['compiler_tree']] != [stable_identity(item) for item in expected['compiler_tree']]:
        raise ValueError('Compiler subtree identity changed')
    if actual['service_lease_sha256'] != expected['service_lease_sha256']:
        raise ValueError('Service lease changed')


def wait_stopped(pid, ticks, timeout=5):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        process = process_identity(pid)
        if process['start_ticks'] != ticks:
            raise ValueError('Scheduler identity changed while stopping')
        if process['state'] == 'T':
            return
        time.sleep(.02)
    raise TimeoutError('Scheduler did not enter stopped state')


def pending_stop_handler(pid):
    status = Path(f'/proc/{pid}/status').read_text().splitlines()
    mask = sum(1 << (number-1) for number in (signal.SIGINT, signal.SIGTERM))
    return any(int(line.split()[1], 16) & mask for line in status if line.startswith(('SigPnd:', 'ShdPnd:')))


def execute(destination, timeout=1800):
    with exclusive(Path(destination).resolve()/'execute.lock'):
        return _execute(destination, timeout)


def _execute(destination, timeout=1800):
    destination = Path(destination).resolve()
    plan = verify_plan(destination)
    revision = Path(plan['throughput_revision'])
    original, _ = verify_revision(revision)
    attempt = Path(original['source_attempt'])
    expected = plan['chain']
    pids = {name: expected[name]['pid'] for name in ('scheduler', 'worker', 'server')}
    same_chain(inspect_chain(revision, **{f'{name}_pid': pid for name, pid in pids.items()}), expected)
    if (attempt/'PAUSE').exists():
        raise ValueError('PAUSE changed before handoff')
    fds = {}
    old_handlers = {}
    stopped = False
    worker_signaled = False
    state = dict(schema=1, attempt_id=str(time.time_ns()), handoff_manifest_sha256=sha(destination/'manifest.json'), events=[])
    def event(label, **data):
        state['events'].append(dict(label=label, timestamp=time.time(), **data))
        atomic_json(destination/'state.json', state)
        with (destination/'history.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(attempt_id=state['attempt_id'], **state['events'][-1]))+'\n')
            stream.flush()
            os.fsync(stream.fileno())
    try:
        def interrupted(number, frame):
            raise KeyboardInterrupt(f'Handoff operator signal {number}')
        for number in (signal.SIGINT, signal.SIGTERM):
            old_handlers[number] = signal.signal(number, interrupted)
        for name, pid in pids.items():
            fds[name] = os.pidfd_open(pid)
        before = inspect_chain(revision, **{f'{name}_pid': pid for name, pid in pids.items()})
        same_chain(before, expected)
        if any(before[name]['state'] in ('T', 't', 'Z', 'X') for name in pids):
            raise ValueError('Do not adopt an externally stopped or dead target')
        event('targets_pinned', chain=expected)
        # Mark intent first: an operator interruption may arrive after the kernel
        # applied STOP but before the following Python bytecode executes.
        stopped = True
        event('scheduler_stop_intent')
        signal.pidfd_send_signal(fds['scheduler'], signal.SIGSTOP)
        wait_stopped(pids['scheduler'], expected['scheduler']['start_ticks'])
        frozen = inspect_chain(revision, **{f'{name}_pid': pid for name, pid in pids.items()})
        same_chain(frozen, expected)
        event('scheduler_stopped', chain=frozen)
        deadline = time.monotonic()+timeout
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError('No committed episode boundary before handoff timeout')
            with lock(attempt/'episode_ledger.lock'):
                ledger = read(attempt/'episode_ledger.json')
                if not has_reserved(ledger['entries'], pids['worker'], expected['worker']['start_ticks']):
                    current = inspect_chain(revision, **{f'{name}_pid': pid for name, pid in pids.items()})
                    same_chain(current, expected)
                    if current['scheduler']['state'] != 'T' or (attempt/'PAUSE').exists():
                        raise ValueError('Scheduler state/PAUSE changed while waiting')
                    if [child['pid'] for child in live_children(pids['worker'])] == [pids['server']]:
                        snapshot = destination/'boundary_ledger.json'
                        atomic_json(snapshot, ledger)
                        event('committed_boundary', ledger_snapshot=str(snapshot), ledger_snapshot_sha256=sha(snapshot), chain=current)
                        # From this point, automatic continuation is permanently prohibited.
                        worker_signaled = True
                        signal.pidfd_send_signal(fds['worker'], signal.SIGTERM)
                        wait_pidfd(fds['worker'])
                        event('worker_dead_under_ledger_lock')
                        break
                    else:
                        raise ValueError('Unexpected worker child after committed episode boundary')
            time.sleep(.1)
        signal.pidfd_send_signal(fds['server'], signal.SIGTERM)
        wait_pidfd(fds['server'], 30)
        event('server_dead')
        deadline = time.monotonic()+30
        while any(start_ticks(item['pid']) == item['start_ticks'] for item in expected['compiler_tree']):
            if time.monotonic() > deadline:
                raise TimeoutError('Compiler subtree remained alive; no automatic descendant signal or scheduler continuation')
            time.sleep(.1)
        event('compiler_subtree_naturally_dead', identities=expected['compiler_tree'])
        scheduler = process_identity(pids['scheduler'])
        if stable_identity(scheduler) != stable_identity(expected['scheduler']) or scheduler['state'] != 'T' or live_children(pids['scheduler']):
            raise ValueError('Controller is not the pinned stopped controller without live workers')
        # SIGTERM/INT invoke the original global PAUSE handler; SIGKILL cannot.
        signal.pidfd_send_signal(fds['scheduler'], signal.SIGKILL)
        wait_pidfd(fds['scheduler'])
        event('stopped_scheduler_killed_without_handler')
        for pid in pids.values():
            if start_ticks(pid) is not None:
                raise ValueError('A reviewed main PID is still alive')
        for path in [attempt/'leases'/name for name in ('scheduler.lock', 'pi0.shard0.lock', 'pi0.shard1.lock')] + [HERE/'readiness/gpu_leases/gpu0.lock']:
            with exclusive(path):
                pass
        if (attempt/'PAUSE').exists():
            raise ValueError('PAUSE changed; preserve marker and investigate')
        event('ownership_locks_released_pause_unchanged')
        state.update(complete=True, no_live_reviewed_processes=True, pause_after='absent')
        atomic_json(destination/'complete.json', state)
        return state
    except BaseException as error:
        resumed = False
        if stopped and not worker_signaled:
            try:
                current = inspect_chain(revision, **{f'{name}_pid': pid for name, pid in pids.items()})
                same_chain(current, expected)
                if current['scheduler']['state'] == 'T' and not pending_stop_handler(pids['scheduler']) and not (attempt/'PAUSE').exists():
                    signal.pidfd_send_signal(fds['scheduler'], signal.SIGCONT)
                    resumed = True
            except BaseException:
                pass
        event('handoff_failed', error=repr(error), worker_signal_attempted=worker_signaled,
              scheduler_resumed=resumed, manual_recovery_required=stopped and not resumed)
        raise
    finally:
        for fd in fds.values():
            os.close(fd)
        for number, handler in old_handlers.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'execute'])
    parser.add_argument('--destination', required=True)
    parser.add_argument('--revision')
    parser.add_argument('--scheduler', type=int)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--server', type=int)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    if args.operation == 'prepare':
        if None in (args.revision, args.scheduler, args.worker, args.server):
            parser.error('prepare requires --revision, --scheduler, --worker and --server')
        prepare(args.revision, args.destination, args.scheduler, args.worker, args.server)
    else:
        execute(args.destination, args.timeout)


if __name__ == '__main__':
    main()

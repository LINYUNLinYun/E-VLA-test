"""Run new science while the original idle sessions retain their GPU leases.

This process never starts, stops, signals, or takes a lease from a holder.
Only its own model/worker Popen handles are cleaned up.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import time

from preparations.ease_insights_20261103 import gpu_resource_session as legacy
from preparations.ease_insights_20261103 import retry_resource_session_port as remote

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / 'results/ease_insights_20261103'
CAMPAIGN = EXPERIMENT / 'gpu_resource_campaign_v3'
NEW_ROOT = EXPERIMENT / 'vlm_within_suite_v1'
HELPER_SHA = '98e6fae9353f5766a661014756b33d92727fec8d61a46ce5e645bac6d43bbfd1'
OLD_REPORTS = {
    'reports/ease_insights_20261103/vlm_local_search_runtime_acceptance/full_confirmation/audit.json':
        '347c6bb41a5c6589336e7b58ac4cc174825d7ab6c1072fed973cc7eb59dcd66e',
    'results/ease_insights_20261103/vlm_local_search_v1/confirmation/analysis/analysis.json':
        '48400436855e9b78cd5ebfa61ae409b765699dcc96280522db16145a861f946b',
    'reports/ease_insights_20261103/vlm_local_confirmation_statistics/audit.json':
        '096681189368855d25425691f80114f5f5ed15fc76ea3adc32cd9327bc4724da',
}
BROKER_PID, BROKER_START = 600807, '2622765627'


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    remote.canonical(Path(path).parent).mkdir(parents=True, exist_ok=True)
    legacy.identity.durable_json(Path(path), value)


def verify_remote_lease(binding, session_pid, proc_root=Path('/proc')):
    """Read another owned process's FD proof, without opening/taking its lock."""
    path = Path(binding['lease_path'])
    stat = path.stat()
    require((stat.st_dev, stat.st_ino) == (binding['lease_device'], binding['lease_inode']),
            'Original GPU lease inode/device changed')
    proofs = []
    for number in binding['lease_fds']:
        fd = proc_root / str(session_pid) / 'fd' / str(number)
        current = fd.stat()
        info = (proc_root / str(session_pid) / 'fdinfo' / str(number)).read_text()
        require((current.st_dev, current.st_ino) == (stat.st_dev, stat.st_ino),
                'Original session no longer holds the recorded lease FD')
        require(any(line.startswith('lock:') and 'FLOCK' in line and 'WRITE' in line
                    for line in info.splitlines()), 'Original FD lacks exclusive FLOCK proof')
        proofs.append(dict(fd=number, device=current.st_dev, inode=current.st_ino))
    require(proofs, 'No remote lease FD proof')
    return proofs


def campaign_completed(campaign=CAMPAIGN):
    """Every activated old task is complete or has its one completed retry."""
    tasks, outcomes, commits = {}, {}, {}
    for directory in (campaign / 'transactions').iterdir():
        if not directory.is_dir():
            continue
        commit_path = directory / 'committed.json'
        require(commit_path.is_file(), 'Uncommitted old publication remains: ' + str(directory))
        commit = read(commit_path)
        require(commit.get('schema') == 'ease-gpu-resource-activation-v1' and
                commit.get('plan_sha256') == remote.PLAN_SHA256, 'Old activation identity differs')
        commits[str(commit_path)] = sha(commit_path)
    for path in (campaign / 'tasks').glob('*.json'):
        task = read(path); tid = task['id']; activation_path = Path(task['activation_file'])
        require(path.name == tid + '.json' and str(activation_path) in commits,
                'Unactivated or noncanonical old task remains')
        activation = read(activation_path)
        require(activation.get('task_sha256', {}).get(tid) == sha(path), 'Old task activation differs')
        outcome_path = campaign / 'outcomes' / (hashlib.sha256(tid.encode()).hexdigest() + '.json')
        outcome = read(outcome_path)
        require(outcome.get('task_id') == tid and outcome.get('task_sha256') == sha(path),
                'Old task is not durably consumed')
        tasks[tid], outcomes[tid] = task, outcome
    for tid, task in tasks.items():
        outcome = outcomes[tid]
        if outcome.get('phase') == 'attention':
            retry = tid + '__port_retry1'
            require(sorted(x for x in tasks if x.startswith(tid + '__port_retry')) == [retry],
                    'Old attention has extra or ambiguous technical retries')
            require(outcome.get('error') == remote.PORT_ERROR and retry in tasks and
                    outcomes[retry].get('phase') == 'complete', 'Unresolved old attention remains')
            require({k for k in task if task[k] != tasks[retry][k]} == {'id', 'activation_file'},
                    'Old technical retry changed science')
        else:
            require(outcome.get('phase') == 'complete', 'Old task remains executable')
        marker = read(task['complete_file'])
        require(marker.get('complete') is True and marker.get('manifest_hash') == task['manifest_hash']
                and marker.get('condition') == task['condition'], 'Old completion marker differs')
    require(tasks, 'Old campaign has no task evidence')
    return dict(tasks=len(tasks), commits=commits,
                task_hashes={tid: sha(campaign / 'tasks' / (tid + '.json')) for tid in tasks})


class HolderGuard:
    def __init__(self, verifier=None, check_old=True):
        self.verifier = verifier or remote.PortRetryController()
        self.specs = {s.gpu_uuid: s for s in self.verifier.b.slots}
        self.check_old = check_old
        self.queue_sha = None
        self.task_names = None

    def quick(self, gpu_uuid):
        require(gpu_uuid in self.specs, 'GPU is not one of the two original owned holders')
        if self.queue_sha is not None:
            require(sha(EXPERIMENT / 'queue.json') == self.queue_sha,
                    'Old scheduler queue changed after the borrower boundary')
            require(sorted(p.name for p in (self.verifier.b.campaign / 'tasks').glob('*.json')) == self.task_names,
                    'A new old-campaign task appeared while resources are borrowed')
        spec = self.specs[gpu_uuid]
        proof = self.verifier._verify_binding_and_processes(spec)
        require([p['pid'] for p in proof['children']] == [spec.keeper_pid],
                'Original holder has a scientific or unknown child')
        slot = self.verifier.b.campaign / 'sessions' / spec.name
        status = read(slot / 'status.json')
        require(status.get('phase') == 'idle_holding' and status.get('session_pid') == spec.session_pid
                and status.get('gpu_uuid') == gpu_uuid and status.get('attention') is None,
                'Original holder is no longer idle without attention')
        require(0 <= time.time() - status['time'] <= 15, 'Original holder heartbeat is stale')
        require(not (self.verifier.b.campaign / 'release').exists() and
                not (self.verifier.b.campaign / 'releases' / gpu_uuid).exists(), 'Original release requested')
        proof['remote_lease_fds'] = verify_remote_lease(proof['binding'], spec.session_pid,
                                                       self.verifier.b.proc_root)
        return proof

    def full(self, require_broker_stopped=True):
        require(sha(ROOT / 'preparations/ease_insights_20261103/retry_resource_session_port.py') == HELPER_SHA,
                'Pinned remote holder verifier changed')
        plan, plan_sha = self.verifier._verify_plan()
        old = {name: read(ROOT / name) for name in OLD_REPORTS}
        require(all(sha(ROOT / name) == digest for name, digest in OLD_REPORTS.items()),
                'Accepted completed-study report changed')
        acceptance, analysis, stats = list(old.values())
        require(acceptance.get('status') == stats.get('status') == 'pass' and
                analysis.get('complete') is True and analysis.get('valid_episodes') == 600 and
                analysis.get('conclusion') == 'not_reproduced' and stats.get('differences') == [],
                'Old scientific branch is not a valid completed negative study')
        supervisor = read(EXPERIMENT / 'vlm_local_search_v1/supervisor/state.json')
        require(supervisor.get('phase') == 'report_ready' and
                supervisor.get('conclusion') == 'not_reproduced' and
                all(v.get('status') == 'complete' for v in supervisor.get('actions', {}).values()),
                'Old supervisor can still advance science')
        supervisor_pid = read(EXPERIMENT / 'vlm_local_search_v1/supervisor/supervisor_process.json')['pid']
        observer = legacy.identity.LocalRuntime()
        require(observer.identity(supervisor_pid).get('state') in ('absent', 'Z', 'X', 'x'),
                'Old scientific supervisor is still live')
        broker_receipt = read(self.verifier.b.campaign / 'broker_process.json')
        require(broker_receipt.get('pid') == BROKER_PID and broker_receipt.get('start_ticks') == BROKER_START
                and broker_receipt.get('plan_sha256') == remote.PLAN_SHA256
                and broker_receipt.get('source_sha256') == plan['source_hashes'][
                    'preparations/ease_insights_20261103/gpu_resource_campaign.py'],
                'Old publisher receipt changed; do not assume a restarted broker is quiesced')
        broker = observer.identity(broker_receipt['pid'])
        if require_broker_stopped:
            require(broker.get('state') in ('absent', 'Z', 'X', 'x'), 'Old CPU broker must be quiesced before borrowing')
        completed = campaign_completed(self.verifier.b.campaign)
        queue_path = EXPERIMENT / 'queue.json'
        queue, states = read(queue_path), read(EXPERIMENT / 'scheduler/jobs_state.json')
        holder_jobs = {'ease_gpu_resource_session_v1_00', 'ease_gpu_resource_session_v1_01'}
        require({j['id'] for j in queue['jobs'] if j['id'] in holder_jobs} == holder_jobs,
                'Original resource-holder rows changed')
        require(all(states.get(j['id'], {}).get('status') == 'complete'
                    for j in queue['jobs'] if j['id'] not in holder_jobs),
                'Old scheduler queue still contains runnable science')
        for spec in self.specs.values():
            row = states['ease_gpu_resource_session_v1_' + spec.name[-2:]]
            outer_uuid, _, inner = legacy.identity.gate_parts(row['argv'])
            require(row.get('status') == 'running' and row.get('pid') == spec.session_pid
                    and row.get('gpu_uuid') == outer_uuid == spec.gpu_uuid
                    and inner == list(spec.session_argv), 'Original scheduler resource row changed')
        self.queue_sha = sha(queue_path)
        self.task_names = sorted(p.name for p in (self.verifier.b.campaign / 'tasks').glob('*.json'))
        holders = {uuid: self.quick(uuid) for uuid in self.specs}
        return dict(schema='ease-borrowed-idle-holders-proof-v1', time=time.time(),
                    source_hashes=plan['source_hashes'], plan_sha256=plan_sha,
                    old_reports=OLD_REPORTS, old_campaign=completed, holders=holders,
                    old_queue_sha256=self.queue_sha,
                    broker_stopped=broker.get('state') in ('absent', 'Z', 'X', 'x'),
                    original_lease_ownership_unchanged=True)


class BorrowedRuntime(legacy.Runtime):
    def __init__(self, guard, gpu_uuid):
        self.guard, self.gpu_uuid = guard, gpu_uuid

    def keeper_live(self, keeper):
        try:
            proof = self.guard.quick(self.gpu_uuid)
            return proof['keeper_ready'] == keeper['ready']
        except (OSError, ValueError, RuntimeError, remote.ValidationError):
            return False

    def start_keeper(self, *args, **kwargs):
        raise RuntimeError('Borrower cannot start another keeper')

    def close_keeper(self, *args, **kwargs):
        raise RuntimeError('Borrower cannot close an original keeper')

    def bind_scheduler(self, *args, **kwargs):
        raise RuntimeError('Borrower must use remote proof, not claim an inherited scheduler FD')


class BorrowedSession:
    def __init__(self, guard, gpu_uuid, port, manifest):
        self.guard, self.manifest = guard, manifest
        proof = guard.quick(gpu_uuid)
        self.args = argparse.Namespace(gpu_uuid=gpu_uuid, port=port)
        self.runtime = BorrowedRuntime(guard, gpu_uuid)
        self.keeper = dict(ready=proof['keeper_ready'], identity=proof['keeper'])
        self.binding = proof['binding']
        self.plan = dict(keeper_max_memory_mib=512)

    def check_keeper(self):
        self.guard.quick(self.args.gpu_uuid)

    def execution_revision(self, observation):
        return dict(schema='ease-external-borrower-science-v1', controller_pid=os.getpid(),
                    controller_source_sha256=sha(__file__), scientific_manifest_hash=self.manifest['manifest_hash'],
                    original_science_unchanged=False, original_holder_resources_unchanged=True,
                    original_scheduler_session_owns_lease=True, borrower_did_not_take_or_duplicate_lease=True,
                    holder_proof=self.guard.quick(self.args.gpu_uuid), assigned_gpu_consistency=observation,
                    keeper=self.keeper['ready'], scope='New science under an external CPU borrower; original holders remain idle')


@contextmanager
def lock(path, optional=False, shared=False):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # On this NFS mount LOCK_SH requires a readable FD; append-only gives
    # EBADF even though the same shared lock works on local temporary files.
    with Path(path).open('a+' if shared else 'a') as f:
        try:
            fcntl.flock(f.fileno(), (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            if not optional:
                raise
            yield False
            return
        yield True


def fresh_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def run(args):
    guard = HolderGuard()
    if args.launch:
        # The frozen legacy publisher takes EX on this same CPU singleton
        # before publishing its receipt or dispatching. Concurrent borrowers
        # hold SH for their whole lifetimes, preventing publisher restart.
        # This is not either GPU lease, whose FD ownership remains untouched.
        with lock(CAMPAIGN / 'broker.lock', shared=True):
            proof = guard.full(require_broker_stopped=True)
            return run_with_proof(args, guard, proof)
    proof = guard.full(require_broker_stopped=False)
    return run_with_proof(args, guard, proof)


def run_with_proof(args, guard, proof):
    if args.check_holders:
        return proof
    from . import common, runner, worker
    output = Path(args.output).resolve()
    require(output.parent == NEW_ROOT and output.name in ('discovery', 'confirmation'), 'New output escaped its frozen roots')
    manifest = common.read_manifest(output / 'manifest.json', verify_sources=True)
    require(args.gpu_uuid in guard.specs, 'Unknown borrowed GPU')
    if not args.launch:
        return dict(mode='dry_run', holder_proof=proof, manifest_hash=manifest['manifest_hash'],
                    planned_conditions=manifest['planned_conditions'], no_model_started=True)
    status = output / 'controllers' / args.gpu_uuid / 'status.json'
    with lock(NEW_ROOT / 'resources' / (args.gpu_uuid + '.borrower.lock')):
        results = []
        for cid in manifest['planned_conditions']:
            with lock(output / 'claims' / (cid + '.lock'), optional=True) as acquired:
                if acquired:
                    guard.full()
                    stage = common.stage_for(manifest)
                    marker = output / cid / (stage + '_shard00of01_complete.json')
                    if marker.exists():
                        done = read(marker)
                        require(done.get('complete') is True and done.get('manifest_hash') == manifest['manifest_hash']
                                and done.get('condition') == cid
                                and done.get('completed_after', done.get('already_completed')) == len(common.cases(manifest, stage)),
                                'Existing complete marker differs; do not overwrite it')
                        results.append(dict(condition=cid, existing_complete_marker=str(marker), unchanged=True))
                        continue
                    port = fresh_port()
                    session = BorrowedSession(guard, args.gpu_uuid, port, manifest)
                    opts = argparse.Namespace(output=str(output), condition=cid, stage=stage,
                        shard_index=0, shards=1, gpu_uuid=args.gpu_uuid, port=port,
                        launch=False, preflight_only=False)
                    prepared = runner.run(opts)
                    write(status, dict(phase='running', condition=cid, time=time.time(),
                                       manifest_hash=manifest['manifest_hash'], controller_pid=os.getpid()))
                    try:
                        result = legacy.Science(runner=runner, worker=worker).execute(prepared, session)
                    finally:
                        session.runtime.reap_scientific_children()
                    session.check_keeper()
                    session.runtime.scan_owned(args.gpu_uuid, session.keeper, 512)
                    results.append(dict(condition=cid, result=result))
        value = dict(phase='stage_assignment_complete', time=time.time(), controller_pid=os.getpid(),
                     manifest_hash=manifest['manifest_hash'], results=results,
                     holders_retained=True, scope='This borrower assignment only; other borrower may still be running')
        write(status, value)
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--check-holders', action='store_true')
    args = parser.parse_args()
    require(args.check_holders or (args.output and args.gpu_uuid), 'Supply output and borrowed GPU')
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()

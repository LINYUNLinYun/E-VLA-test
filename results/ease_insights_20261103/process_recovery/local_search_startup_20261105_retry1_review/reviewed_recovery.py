"""Review artifact. Default/check is read-only. Mutation commands were NOT executed.

Run only after root resolves and reviews the startup resource diagnosis. This
script preserves the prepared 5 s / 2 s policy and refuses an altered proposal.
It never signals a process or launches/restarts a scheduler.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
ROOT = Path('/root/nfs/embodied/benchmark')
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
R = ROOT / 'results/ease_insights_20261103'
SUP = R / 'vlm_local_search_v1/supervisor'
COOR = R / 'gate_opportunity_coordinator'
QUEUE = R / 'queue.json'
from preparations.ease_insights_20261103.pipeline import proc_identity, scheduler_match, file_lock, write

def read(path):
    return json.loads(Path(path).read_bytes())

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def dead(pid):
    p = proc_identity(pid)
    require(p['status'] in ('absent', 'dead'), f'PID {pid} is live/uncertain; preserve it: {p}')
    return p

def record(label, **details):
    path = OUT / 'execution_receipts' / f'{time.time_ns()}_{label}.json'
    write(path, dict(action=label, time=time.time(), **details))
    return path

def allocator():
    pinned = read(OUT / 'preparation_receipt.json')['live_allocator_unchanged']
    process = proc_identity(pinned['pid'])
    require(process['status'] == 'live' and process.get('start_ticks') == pinned['start_ticks'], 'Original live allocator identity changed')
    require(scheduler_match(process, QUEUE, R / 'scheduler'), 'Original scheduler argv paths changed')
    matches = []
    for d in Path('/proc').iterdir():
        if d.name.isdecimal():
            candidate = proc_identity(int(d.name))
            if candidate['status'] == 'live' and scheduler_match(candidate, QUEUE, R / 'scheduler'):
                matches.append(candidate)
    require(len(matches) == 1 and matches[0]['pid'] == pinned['pid'], 'Allocator is no longer unique')
    return process

def sources(require_zero_cases=False):
    for path, wanted in read(OUT / 'source_hashes.json').items():
        require(digest(path) == wanted, 'Reviewed source changed: ' + path)
    receipt = read(OUT / 'preparation_receipt.json')
    require(digest(OUT / 'queue.before.json') == receipt['queue_before_sha256'], 'Reviewed before queue changed')
    require(digest(OUT / 'queue.proposed.json') == receipt['queue_proposed_sha256'], 'Reviewed proposal changed; generate and review a new material package')
    audit = read(OUT / 'strict_stage_audit.json')
    require(audit['valid'] == 120 and audit['expected'] == 720 and not audit['invalid'] and not audit['unexpected'], 'Strict audit differs')
    for condition in ('full_d10', 'local_spatial'):
        require(audit['conditions'][condition]['valid'] == 0, 'An affected condition contains valid results')
        if require_zero_cases:
            require(not list((R / 'vlm_local_search_v1/discovery' / condition).glob('episodes/**/episode_receipt.json')), 'New affected-condition episode receipts appeared before requeue; re-audit instead of retrying blindly')
    require(digest(R / 'vlm_local_search_v1/discovery/manifest.json') == digest(OUT / 'archive/vlm_local_search_v1/discovery/manifest.json'), 'Frozen manifest changed')

def before_commit():
    sources(require_zero_cases=True)
    process = allocator()
    for pid in (338645, 415450, 432881, 432890):
        dead(pid)
    require(QUEUE.read_bytes() == (OUT / 'queue.before.json').read_bytes(), 'Live queue differs from reviewed before bytes')
    require((SUP / 'state.json').read_bytes() == (OUT / 'archive/vlm_local_search_v1/supervisor/state.json').read_bytes(), 'Supervisor state changed')
    require((COOR / 'state.json').read_bytes() == (OUT / 'archive/gate_opportunity_coordinator/state.json').read_bytes(), 'Coordinator state changed')
    state = read(COOR / 'state.json')
    require(state.get('pending', 'MISSING') is None, 'Coordinator pending is non-null/missing; preserve and audit its transaction, never erase it')
    require(state.get('phase') == 'supervisor_terminal', 'Unexpected coordinator phase')
    states = read(R / 'scheduler/jobs_state.json')
    receipt = read(OUT / 'preparation_receipt.json')
    require(all(states[j]['status'] == 'failed' for j in receipt['failed_job_ids']), 'Old job status changed')
    require(all(j not in states for j in receipt['retry_job_ids']), 'Retry ID already has history; inspect it')
    for job in read(OUT / 'queue.proposed.json')['jobs']:
        if job['id'] in receipt['retry_job_ids']:
            require(not Path(job['complete_file']).exists(), 'Affected completion marker appeared')
            flag = job['argv'].index('--status-file') + 1
            require(not Path(job['argv'][flag]).exists(), 'Retry gate receipt already exists')
    return process

def commit():
    # The scheduler remains live and discovers new IDs in its existing 3 s loop.
    with file_lock(QUEUE.with_name(QUEUE.name + '.lock')):
        with file_lock(SUP / 'supervisor.lock', blocking=False):
            with file_lock(COOR / 'coordinator.lock', blocking=False):
                process = before_commit()
                # First remove terminal supervisor error, then publish retry IDs.
                # No process is launched until the caller invokes the next command.
                write(SUP / 'state.json', read(OUT / 'supervisor.state.proposed.json'))
                try:
                    # pipeline.write preserves reviewed serializer/newline exactly.
                    write(QUEUE, read(OUT / 'queue.proposed.json'))
                    require(QUEUE.read_bytes() == (OUT / 'queue.proposed.json').read_bytes(), 'Committed queue bytes differ')
                except BaseException:
                    record('partial_commit_preserved', supervisor_state_sha256=digest(SUP / 'state.json'), queue_sha256=digest(QUEUE))
                    raise
                return record('queue_and_supervisor_committed', scheduler=process,
                    queue_sha256=digest(QUEUE), supervisor_state_sha256=digest(SUP / 'state.json'),
                    jobs_state_written=False, coordinator_state_written=False)

def sup_command():
    return [str(ROOT / 'policies/realtime-vla-flash/.venv/bin/python'), '-B', '-u', '-m',
        'preparations.ease_insights_20261103.vlm_local_search_v1.supervisor',
        '--root', str(R / 'vlm_local_search_v1/discovery'),
        '--confirmation-root', str(R / 'vlm_local_search_v1/confirmation'),
        '--scheduler-pid', '335208', '--poll-seconds', '10', '--launch']

def spawn(argv, log):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    with log.open('a') as stream:
        return subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)

def healthy_sup(expected_pid=None):
    receipt = read(SUP / 'supervisor_process.json')
    process = proc_identity(receipt['pid'])
    require(expected_pid is None or receipt['pid'] == expected_pid, 'Unexpected supervisor receipt PID')
    require(process['status'] == 'live', 'New supervisor is not live')
    require('preparations.ease_insights_20261103.vlm_local_search_v1.supervisor' in process['argv'], 'New supervisor module differs')
    state, status = read(SUP / 'state.json'), read(SUP / 'status.json')
    require(state['phase'] == status['phase'] == 'waiting_discovery' and status['terminal'] is False, 'New supervisor has not reached healthy nonterminal discovery wait')
    require(status['time'] >= receipt['started_unix'] and time.time() - status['time'] <= 30, 'New supervisor status is not fresh')
    scheduler = allocator()
    for report in (state['scheduler'], state['scheduler_observation']['process'], status['scheduler']['process']):
        require(report['pid'] == scheduler['pid'] and report['start_ticks'] == scheduler['start_ticks'], 'Supervisor has not adopted the unchanged allocator')
    return process

def launch_sup():
    sources()
    allocator()
    require(QUEUE.read_bytes() == (OUT / 'queue.proposed.json').read_bytes(), 'Expected retry queue is not live')
    require(read(SUP / 'state.json') == read(OUT / 'supervisor.state.proposed.json'), 'Expected nonterminal supervisor state is not live')
    dead(read(SUP / 'supervisor_process.json')['pid'])
    argv = sup_command()
    process = spawn(argv, SUP / 'daemon.log')
    record('supervisor_launched', pid=process.pid, argv=argv)
    deadline = time.monotonic() + 20
    error = None
    while time.monotonic() < deadline:
        try:
            observed = healthy_sup(process.pid)
            return record('supervisor_healthy', process=observed, scheduler=allocator())
        except Exception as exc:
            error = str(exc)
        if process.poll() is not None:
            break
        time.sleep(.25)
    raise RuntimeError('New supervisor was preserved for inspection; no automatic second launch: ' + str(error))

def rebind_coor():
    sources()
    allocator()
    supervisor = healthy_sup()
    require(supervisor['pid'] != 338645, 'Supervisor identity was not replaced')
    dead(338645)
    dead(415450)
    with file_lock(COOR / 'coordinator.lock', blocking=False):
        state = read(COOR / 'state.json')
        require((COOR / 'state.json').read_bytes() == (OUT / 'archive/gate_opportunity_coordinator/state.json').read_bytes(), 'Coordinator state changed; re-review instead of clearing it')
        require(state.get('pending', 'MISSING') is None, 'Non-null/missing pending must be audited, never erased')
        require(state.get('phase') == 'supervisor_terminal', 'Unexpected coordinator phase')
        require(state['controllers']['supervisor']['pid'] == 338645, 'Unknown old supervisor pin')
        from preparations.ease_insights_20261103.gate_opportunity_coordinator import Backend
        backend = Backend()
        backend.verify_helper()
        # controllers(None) probes OS/file identity only. It does not call
        # inventory(), gpus(), lease_available(), helper.operate(), or step().
        fresh = backend.controllers(previous=None)
        require(fresh['scheduler']['pid'] == 335208 and fresh['supervisor']['pid'] == supervisor['pid'], 'Fresh controllers differ from reviewed live processes')
        require(fresh['scheduler']['_scheduler_proof'] == state['controllers']['scheduler']['_scheduler_proof'], 'Allocator proof changed')
        write(COOR / 'state.json', read(OUT / 'coordinator.state.proposed.json'))
        result = record('coordinator_old_pin_removed', old_controllers=state['controllers'], fresh_identity_preflight=fresh,
            pending_preserved=None, scheduler_restarted=False)
    return result

def launch_coor():
    sources()
    allocator()
    healthy_sup()
    dead(read(COOR / 'process.json')['pid'])
    state = read(COOR / 'state.json')
    require(state.get('pending', 'MISSING') is None and 'controllers' not in state, 'Expected reviewed unbound coordinator state is absent')
    argv = ['/usr/bin/python3', '-B', '-u', '-m', 'preparations.ease_insights_20261103.gate_opportunity_coordinator', '--launch']
    process = spawn(argv, COOR / 'daemon.log')
    return record('coordinator_launched', pid=process.pid, argv=argv,
        behavior='First normal step binds the new supervisor and unchanged scheduler, then resumes normal inventory; inspect fresh state/status/process.')

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', nargs='?', default='check', choices=['check', 'commit', 'launch-supervisor', 'rebind-coordinator', 'launch-coordinator'])
    parser.add_argument('--resource-policy', choices=['root-reviewed-unchanged-5s2s'])
    args = parser.parse_args()
    if args.action != 'check':
        require(args.resource_policy == 'root-reviewed-unchanged-5s2s', 'Root must first resolve startup resource diagnosis and explicitly review unchanged 5s/2s policy; changed resource policy requires a new reviewed proposal')
    if args.action == 'check':
        print(json.dumps(dict(mode='read_only', scheduler=before_commit(), prepared=str(OUT)), indent=2))
    else:
        actions = {'commit': commit, 'launch-supervisor': launch_sup, 'rebind-coordinator': rebind_coor, 'launch-coordinator': launch_coor}
        print(actions[args.action]())

if __name__ == '__main__':
    main()

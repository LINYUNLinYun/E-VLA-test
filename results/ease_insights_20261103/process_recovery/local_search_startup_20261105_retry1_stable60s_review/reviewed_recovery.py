"""Review artifact. Default/check is read-only. Mutation commands were NOT executed.

Root selected continuous idle 60 s and poll 2 s for all five unfinished local
jobs. This script enforces that exact reviewed proposal and rejects old 5 s policy.
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

def proposal():
    """Reconstruct every allowed change and compare the complete queue objects."""
    before = read(OUT / 'queue.before.json')
    after = read(OUT / 'queue.proposed.json')
    expected = copy.deepcopy(before)
    receipt = read(OUT / 'preparation_receipt.json')
    states = read(OUT / 'archive/scheduler/jobs_state.json')
    affected, renamed = [], []
    for old, job in zip(before['jobs'], expected['jobs']):
        if old['id'] in receipt['failed_job_ids']:
            require(states[old['id']]['status'] == 'failed', 'Retry source was not a failed job')
            job['id'] = old['id'] + '__startup_retry1'
            index = job['argv'].index('--status-file') + 1
            job['argv'][index] = str(R / 'scheduler/launch_gates' / (job['id'] + '.json'))
            renamed.append(job['id'])
        if ('vlm_local_v1_discovery' in old['id'] and
                states.get(old['id'], {}).get('status') in ('pending', 'failed')):
            job.setdefault('env', {}).update(EASE_GPU_GATE_IDLE_SECONDS='60', EASE_GPU_GATE_POLL_SECONDS='2')
            job['launch_gate'].update(continuous_idle_seconds=60, poll_seconds=2)
            affected.append(job['id'])
    require(len(affected) == 5 and sorted(renamed) == sorted(receipt['retry_job_ids']), 'Wrong resource/retry scope')
    require(after == expected, 'Complete queue diff contains a change outside two IDs/receipts and five resource policies')
    changes = []
    def diff(a, b, path=''):
        require(type(a) == type(b), 'Unexpected value type change: ' + path)
        if isinstance(a, dict):
            require(a.keys() == b.keys(), 'Unexpected dictionary keys changed: ' + path)
            for key in a:
                diff(a[key], b[key], path + '/' + key)
        elif isinstance(a, list):
            require(len(a) == len(b), 'Queue list length changed: ' + path)
            for index, (x, y) in enumerate(zip(a, b)):
                diff(x, y, path + '/' + str(index))
        elif a != b:
            changes.append(dict(field=path, before=a, after=b))
    diff(before, after)
    require(len(changes) == 14 and changes == read(OUT / 'queue.complete_diff.json'), 'Complete reviewed queue diff differs')
    return dict(affected_resource_jobs=affected, renamed_jobs=renamed, scalar_difference_count=len(changes))

def sources(require_zero_cases=False):
    for path, wanted in read(OUT / 'source_hashes.json').items():
        require(digest(path) == wanted, 'Reviewed source changed: ' + path)
    receipt = read(OUT / 'preparation_receipt.json')
    require(digest(OUT / 'queue.before.json') == receipt['queue_before_sha256'], 'Reviewed before queue changed')
    require(digest(OUT / 'queue.proposed.json') == receipt['queue_proposed_sha256'], 'Reviewed proposal changed; generate and review a new material package')
    require(digest(OUT / 'supervisor.state.proposed.json') == receipt['supervisor_state_proposed_sha256'], 'Reviewed supervisor state proposal changed')
    require(digest(OUT / 'coordinator.state.proposed.json') == receipt['coordinator_state_proposed_sha256'], 'Reviewed coordinator state proposal changed')
    expected_sup = read(OUT / 'archive/vlm_local_search_v1/supervisor/state.json')
    require(expected_sup.get('phase') == 'needs_attention' and expected_sup.get('actions') == {}, 'Archived supervisor recovery scope differs')
    expected_sup['phase'] = 'waiting_discovery'
    for key in ('error', 'attention_unix'):
        expected_sup.pop(key, None)
    require(read(OUT / 'supervisor.state.proposed.json') == expected_sup, 'Supervisor state contains an unreviewed field change')
    expected_coor = read(OUT / 'archive/gate_opportunity_coordinator/state.json')
    require(expected_coor.get('pending', 'MISSING') is None, 'Archived pending is unknown/non-null; never erase it')
    expected_coor.pop('controllers')
    require(read(OUT / 'coordinator.state.proposed.json') == expected_coor, 'Coordinator state contains an unreviewed field change')
    proposal()
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
    for condition in ('local_object', 'local_goal', 'local_long'):
        job_id = 'vlm_local_v1_discovery_' + condition + '_all120'
        require(states[job_id]['status'] == 'pending', 'A pending resource-update job started or changed; inspect it before applying: ' + job_id)
    local_modules = {'preparations.ease_insights_20261103.vlm_local_search_v1.' + name
        for name in ('runner', 'server', 'worker')}
    for directory in Path('/proc').iterdir():
        if not directory.name.isdecimal():
            continue
        candidate = proc_identity(int(directory.name))
        argv = candidate.get('argv', [])
        if (candidate['status'] == 'live' and any(module in argv for module in local_modules)
                and str(R / 'vlm_local_search_v1/discovery') in argv):
            raise RuntimeError('A local discovery gate/runtime is already in flight; preserve it and re-review: ' + str(candidate))
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

def check_adoption():
    """CPU/file evidence only; never invokes resource scanners or signals."""
    sources()
    unchanged = allocator()
    require(QUEUE.read_bytes() == (OUT / 'queue.proposed.json').read_bytes(), 'Published queue differs from the reviewed proposal; inspect before claiming adoption')
    details = proposal()
    jobs = {j['id']: j for j in read(OUT / 'queue.proposed.json')['jobs']}
    deadline = time.monotonic() + 12
    last = None
    while time.monotonic() < deadline:
        states = read(R / 'scheduler/jobs_state.json')
        status = read(R / 'scheduler/status.json')
        known = status.get('jobs', {})
        if (all(job in states and job in known for job in details['renamed_jobs']) and
                all(job not in known for job in read(OUT / 'preparation_receipt.json')['failed_job_ids'])):
            break
        last = dict(scheduler_status=status.get('status'), current_job_ids=list(known))
        time.sleep(.25)
    else:
        raise RuntimeError('Scheduler has not yet acknowledged the published retry IDs; preserve running processes and inspect: ' + str(last))
    observations = []
    for job_id in details['affected_resource_jobs']:
        state, job = states[job_id], jobs[job_id]
        item = dict(job=job_id, scheduler_status=state['status'])
        if state['status'] == 'pending':
            item['evidence'] = 'Scheduler acknowledged current queue; resource env/metadata prospectively 60/2; no process assigned yet'
        else:
            require(state.get('pid') is not None, 'Nonpending affected job has no PID: ' + job_id)
            pid = state['pid']
            before = proc_identity(pid)
            recorded_argv = state.get('argv', [])
            require(recorded_argv and 'preparations.ease_insights_20261103.await_assigned_gpu' in recorded_argv, 'Scheduler did not record the expected gate invocation: ' + job_id)
            index = job['argv'].index('--status-file') + 1
            receipt_path = Path(job['argv'][index])
            item.update(process=before, gate_receipt_path=str(receipt_path))
            if before['status'] == 'live':
                require(before.get('start_ticks'), 'Missing OS start identity')
                # The gate keeps its PID/start_ticks across exec and removes the
                # resource env immediately before exec. Never require it on runner.
                if 'preparations.ease_insights_20261103.await_assigned_gpu' in before['argv']:
                    require(before['argv'] == recorded_argv, 'Live gate argv differs from scheduler assignment')
                    raw = (Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0')
                    env = dict(part.split(b'=', 1) for part in raw if b'=' in part)
                    require(env.get(b'EASE_GPU_GATE_IDLE_SECONDS') == b'60' and env.get(b'EASE_GPU_GATE_POLL_SECONDS') == b'2', 'An old/different resource policy is already in flight; process preserved: ' + job_id)
                    item['live_gate_env'] = dict(idle_seconds='60', poll_seconds='2')
                else:
                    inner = recorded_argv[recorded_argv.index('--') + 1:]
                    require(before['argv'] == inner, 'PID argv is neither the assigned gate nor its exact exec runner')
                after = proc_identity(pid)
                require(after['status'] == 'live' and after.get('start_ticks') == before['start_ticks'], 'Affected PID identity changed during observation')
                # An exact gate->runner exec may happen between the two samples.
                require(after['argv'] in (recorded_argv, recorded_argv[recorded_argv.index('--') + 1:]), 'Affected argv changed outside reviewed gate exec')
            require(receipt_path.exists(), 'Affected assigned job has no gate receipt yet; preserve and recheck: ' + job_id)
            gate = read(receipt_path)
            require(gate.get('pid') == pid and gate.get('gpu_uuid') == state.get('gpu_uuid'), 'Gate receipt identity differs from the scheduler assignment')
            require(gate.get('idle_threshold_seconds') == 60 and gate.get('poll_seconds') == 2, 'An old/different gate resource policy was observed; process preserved: ' + job_id)
            require(state.get('started', 0) - 1 <= receipt_path.stat().st_mtime, 'Gate receipt predates this scheduler launch')
            if before['status'] != 'live' or 'preparations.ease_insights_20261103.await_assigned_gpu' not in before.get('argv', []):
                require(gate.get('status') == 'launching' and gate.get('final_idle_check', {}).get('status') == 'idle', 'Executed runner lacks the reviewed final idle confirmation')
            item['gate_receipt'] = gate
        observations.append(item)
    current_allocator = allocator()
    require(current_allocator['start_ticks'] == unchanged['start_ticks'], 'Allocator changed during adoption observation')
    return dict(mode='read_only_post_publication', scheduler=current_allocator,
        scientific_completion_claim=False, resource_policy='stable60s-poll2', observations=observations)

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
    supervisor = healthy_sup()
    dead(read(COOR / 'process.json')['pid'])
    state = read(COOR / 'state.json')
    require(state.get('pending', 'MISSING') is None and 'controllers' not in state, 'Expected reviewed unbound coordinator state is absent')
    require(state == read(OUT / 'coordinator.state.proposed.json'), 'Live unbound coordinator state differs from the complete reviewed candidate')
    argv = ['/usr/bin/python3', '-B', '-u', '-m', 'preparations.ease_insights_20261103.gate_opportunity_coordinator', '--launch']
    started = time.time()
    process = spawn(argv, COOR / 'daemon.log')
    record('coordinator_launched', pid=process.pid, argv=argv,
        behavior='First normal step binds the new supervisor and unchanged scheduler, then resumes normal inventory; inspect fresh state/status/process.')
    deadline = time.monotonic() + 45
    error = None
    while time.monotonic() < deadline:
        try:
            receipt = read(COOR / 'process.json')
            observed = proc_identity(process.pid)
            current_state, status = read(COOR / 'state.json'), read(COOR / 'status.json')
            require(receipt['pid'] == process.pid and observed['status'] == 'live' and receipt['start_ticks'] == observed['start_ticks'], 'New coordinator OS/receipt identity is not yet confirmed')
            require(status['time'] >= started and status['phase'] == current_state['phase'], 'New coordinator status is not fresh/settled')
            require(status['phase'] not in ('needs_attention', 'stopped', 'supervisor_terminal'), 'New coordinator entered a terminal/attention phase')
            controllers = current_state['controllers']
            require(controllers['supervisor']['pid'] == supervisor['pid'] and controllers['supervisor']['start_ticks'] == supervisor['start_ticks'], 'Coordinator has not bound the new supervisor identity')
            unchanged = allocator()
            require(controllers['scheduler']['pid'] == unchanged['pid'] and controllers['scheduler']['start_ticks'] == unchanged['start_ticks'], 'Coordinator allocator binding differs')
            healthy_sup(supervisor['pid'])
            return record('coordinator_healthy', process=observed, status=status, controllers=controllers)
        except Exception as exc:
            error = str(exc)
        if process.poll() is not None:
            break
        time.sleep(.25)
    raise RuntimeError('New coordinator was preserved for inspection; no automatic second launch: ' + str(error))

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', nargs='?', default='check', choices=['check', 'check-adoption', 'commit', 'launch-supervisor', 'rebind-coordinator', 'launch-coordinator'])
    parser.add_argument('--resource-policy', choices=['root-reviewed-stable60s-poll2'])
    args = parser.parse_args()
    if args.action not in ('check', 'check-adoption'):
        require(args.resource_policy == 'root-reviewed-stable60s-poll2', 'Root must explicitly review the stable60s/poll2 proposal; old 5s/2s authorization is not accepted')
    if args.action == 'check':
        print(json.dumps(dict(mode='read_only', scheduler=before_commit(), proposal=proposal(),
            resource_policy='root-reviewed-stable60s-poll2', source_hash_count=len(read(OUT / 'source_hashes.json')),
            prepared=str(OUT)), indent=2))
    elif args.action == 'check-adoption':
        print(json.dumps(check_adoption(), indent=2))
    else:
        actions = {'commit': commit, 'launch-supervisor': launch_sup, 'rebind-coordinator': rebind_coor, 'launch-coordinator': launch_coor}
        print(actions[args.action]())

if __name__ == '__main__':
    main()

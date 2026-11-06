"""Review-bound, fixed-experiment CPU transaction installing persistent sessions.

Default only writes a fresh review/audit package. --apply requires a separately
reviewed full-byte package SHA. No GPU query or jobs_state write is performed.
Only the pinned allocator and exact owned childless CPU gates can be stopped.
Unknown/committed queue state is never rolled back. Re-enter the same package
to finish cleanup/adoption; a recorded spawn is never repeated.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import signal
import subprocess

from preparations.ease_insights_20261103 import reassign_waiting_gate_v2 as h
from preparations.ease_insights_20261103 import gpu_resource_campaign as campaign

PROJECT = Path('/root/nfs/embodied/benchmark')
ROOT = PROJECT/'results/ease_insights_20261103'
SCHEMA = 'ease-gpu-resource-campaign-install-transaction-v1'
SCHEDULER_PID, SCHEDULER_TICKS = 335208, '2615895606'
SUPERVISOR_PID, SUPERVISOR_TICKS = 595772, '2622724551'
COORDINATOR_PID, COORDINATOR_TICKS = 576212, '2622324631'
SCHEDULER_SHA = '0786dd02cb35dc7da40ee06dd515dc4481aa0d28e09d4d78baa5df1338345752'
GATE_SHA = '07aa0356850fa6d5199d5f3ba0b35632c4b5f500b163f5da4bb106da506a6e3f'
GATE_PINS = {
    'vlm_local_v1_discovery_local_spatial_all120__gate_policy_5s2s_ece0430b8b3a__startup_retry1__gate_policy_25s2s_c9279611e43c':
        (592976, '2622685261'),
    'vlm_local_v1_discovery_local_object_all120': (593473, '2622691348'),
}
ACK_TIMEOUT = 45.0
Refused = h.Refused
require = h.require


class Runtime(h.LocalRuntime):
    def gpu(self, uuid):
        raise Refused('Installer never queries GPUs')

    def lease_available(self, uuid):
        raise Refused('Installer never allocates/probes GPU leases')

    def start_scheduler(self, argv, log_path):
        with Path(log_path).open('a') as log:
            process = subprocess.Popen(argv, cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        return self.identity(process.pid)


def paths_for(args):
    require(PROJECT.resolve() == PROJECT and ROOT.resolve() == ROOT, 'Fixed project/experiment root is noncanonical')
    root = Path(args.campaign_root)
    require(root.is_absolute() and root.resolve() == root and root.parent == ROOT,
            'Only a direct canonical campaign under this fixed experiment is supported')
    audit = Path(args.output_audit)
    require(audit == root/'installation_audit' and audit.resolve() == audit, 'Audit must be campaign/installation_audit')
    paths = dict(campaign=root, audit=audit, queue=ROOT/'queue.json', states=ROOT/'scheduler/jobs_state.json',
                scheduler=ROOT/'scheduler', supervisor=ROOT/'vlm_local_search_v1/supervisor',
                coordinator=ROOT/'gate_opportunity_coordinator', restart=ROOT/'pipeline/scheduler_restart.lock')
    require(all(path.resolve() == path for path in paths.values()), 'A fixed installation artifact path is a symlink')
    return paths


def hashed(path):
    return h.digest(Path(path).read_bytes())


def stats(hashes):
    result = {}
    for name in hashes:
        value = Path(name).stat()
        result[name] = [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]
    return result


def scheduler_argv(paths, scan):
    return ['/usr/bin/python3', '-u', '-m', h.SCHEDULER, '--queue', str(paths['queue']),
            '--output', str(paths['scheduler']), '--max-gpus', '2', '--scan-seconds', str(scan), '--idle-seconds', '0']


def verify_bundle(paths):
    root = paths['campaign']; installation = h.read_json(root/'installation.json'); plan = h.read_json(root/'plan.json')
    campaign.verify_plan(plan, root/'plan.json')
    require(installation.get('schema') == 'ease-gpu-campaign-install-v1'
            and hashed(root/'plan.json') == installation['plan_sha256'], 'Installation/plan identity changed')
    require(hashed(PROJECT/'preparations/ease_insights_20261103/scheduler.py') == SCHEDULER_SHA
            and hashed(PROJECT/'preparations/ease_insights_20261103/await_assigned_gpu.py') == GATE_SHA,
            'Original scheduler or 25/2 gate source changed')
    before_raw, after_raw = (root/'queue.before.json').read_bytes(), (root/'queue.proposed.json').read_bytes()
    require(h.digest(before_raw) == installation['queue_before_sha256']
            and h.digest(after_raw) == installation['queue_proposed_sha256'], 'Prepared queue hashes changed')
    before, after = json.loads(before_raw), json.loads(after_raw)
    logical = installation['logical_jobs']; ids = {j['id'] for j in logical}
    require(len(logical) == len(ids) == 5, 'Exactly five unique unfinished logical jobs are required')
    require(all(sum(row == job for row in before['jobs']) == 1 for job in logical), 'Original logical queue rows differ')
    expected = copy.deepcopy(before)
    expected['jobs'] = campaign.slot_jobs(plan)+[j for j in expected['jobs'] if j['id'] not in ids]
    require(after == expected and installation['slot_job_ids'] == [j['id'] for j in campaign.slot_jobs(plan)],
            'Proposal changes more than five logical rows -> two persistent slots')
    require(not any(ids.intersection(j.get('depends_on', [])) for j in after['jobs']), 'Unhandled dependency on removed logical job')
    require(installation['activation_file'] == str(root/'transactions/initial/committed.json'), 'Initial activation path differs')
    tasks = {}
    for job in logical:
        task = campaign.task_for(job, plan, 'initial')
        path = root/'tasks'/(task['id']+'.json'); raw = path.read_bytes()
        require(json.loads(raw) == task and h.digest(raw) == installation['task_sha256'].get(task['id']),
                'Task does not transparently map its original scientific argv/output/manifest')
        tasks[str(path)] = h.digest(raw)
    require(len(tasks) == len(installation['task_sha256']) == 5, 'Unexpected/missing initial tasks')
    files = {str(root/name): hashed(root/name) for name in ('plan.json', 'installation.json', 'queue.before.json', 'queue.proposed.json')}
    files.update(tasks)
    files.update({str(PROJECT/name): value for name, value in plan['source_hashes'].items()})
    files[str(Path(__file__).resolve())] = hashed(__file__)
    # Hash source/manifest evidence before any STOP or shared queue/restart lock.
    for job in logical:
        profile = h.bound_profile(job, dict(root=ROOT))
        manifest_path = Path(profile['manifest_path']); files[str(manifest_path)] = hashed(manifest_path)
    return dict(installation=installation, plan=plan, before=before_raw, proposed=after_raw,
                hashes=files, stats=stats(files))


def coordinator_stopped(paths, runtime):
    directory = paths['coordinator']; receipt = h.read_json(directory/'process.json'); state = h.read_json(directory/'state.json')
    require((directory/'stop').is_file() and receipt.get('pid') == COORDINATOR_PID
            and receipt.get('start_ticks') == COORDINATOR_TICKS and state.get('pending') is None
            and state.get('phase') == 'stopped' and runtime.identity(COORDINATOR_PID).get('state') in h.DEAD_STATES,
            'Pinned opportunity coordinator must have stopped itself with pending=None')
    return {str(directory/name): hashed(directory/name) for name in ('process.json', 'state.json')}


def owned(actual, expected, allow_inner=False, stopped=False):
    argv = [expected['argv']]
    if allow_inner:
        argv.append(h.gate_parts(expected['argv'])[2])
    return (actual.get('pid') == expected['pid'] and actual.get('start_ticks') == expected['start_ticks']
            and actual.get('argv') in argv and actual.get('uid') == expected.get('uid') == os.getuid()
            and actual.get('ppid') == expected.get('ppid') and actual.get('state') not in h.DEAD_STATES
            and (not stopped or actual.get('state') == 'T'))


def capture(paths, runtime, bundle):
    states = h.read_json(paths['states']); selected = bundle['installation']['logical_jobs']; gates = {}
    scheduler_receipt = h.read_json(paths['scheduler']/'controller_process.json')
    scheduler = runtime.identity(SCHEDULER_PID)
    require(scheduler.get('start_ticks') == SCHEDULER_TICKS and scheduler.get('argv') == scheduler_argv(paths, 30)
            and scheduler.get('uid') == os.getuid() and scheduler.get('state') not in (*h.DEAD_STATES, 'T', 't')
            and scheduler.get('sigterm_default') and scheduler_receipt.get('pid') == SCHEDULER_PID
            and runtime.matches_started(scheduler, scheduler_receipt['started']), 'Original CPU allocator identity/configuration differs')
    require(scheduler_receipt.get('argv') == [str(PROJECT/'preparations/ease_insights_20261103/scheduler.py'),
            *h.module_invocation(scheduler['argv'])['args']], 'Original scheduler receipt argv differs')
    profile = h.bound_profile(selected[0], dict(root=ROOT))
    supervisor = h.controller(runtime, h.read_json(paths['supervisor']/'supervisor_process.json'), h.LOCAL_SUPERVISOR,
                              paths['queue'], paths['supervisor'], profile)
    require(supervisor['pid'] == SUPERVISOR_PID and supervisor['start_ticks'] == SUPERVISOR_TICKS,
            'Bound original supervisor identity changed')
    require(h.read_json(paths['supervisor']/'supervisor_process.json').get('argv') ==
            [str(PROJECT/'preparations/ease_insights_20261103/vlm_local_search_v1/supervisor.py'),
             *h.module_invocation(supervisor['argv'])['args']], 'Supervisor receipt argv differs')
    h.stable_supervisor_state(profile)
    for job in selected:
        state = states.get(job['id'], dict(status='pending'))
        require(state == bundle['installation']['logical_states_before'][job['id']], 'Prepared logical state changed')
        require(not Path(job['complete_file']).exists(), 'Logical completion marker appeared; reprepare without rerunning valid science')
        if state.get('status') == 'running':
            require(job['id'] in GATE_PINS, 'An unapproved logical job is running')
            pid, ticks = GATE_PINS[job['id']]; gate = runtime.identity(pid)
            _, status_path, _ = h.gate_parts(job['argv'])
            require(state['pid'] == pid and gate.get('start_ticks') == ticks
                    and gate.get('argv') == state.get('argv') == h.concrete(job['argv'], state)
                    and gate.get('uid') == os.getuid() and gate.get('ppid') == SCHEDULER_PID
                    and gate.get('state') not in (*h.DEAD_STATES, 'T', 't')
                    and not gate.get('children') and gate.get('sigterm_default')
                    and runtime.matches_started(gate, state['started']), 'Running process is not the pinned childless default-TERM CPU gate')
            h.gate_parts(gate['argv'])
            require(status_path == paths['scheduler']/'launch_gates'/(job['id']+'.json'), 'Gate receipt destination differs')
            gates[job['id']] = gate
        else:
            require(state.get('status') == 'pending' and not state.get('pid'), 'Unfinished logical task is outside pending/CPU-gate states')
    require(sorted(scheduler.get('children', [])) == sorted(str(g['pid']) for g in gates.values()),
            'Allocator has unknown/additional children; no STOP is authorized')
    require(not (paths['scheduler']/'stop_after_jobs.json').exists(), 'Scheduler stop marker is present')
    controller_files = {str(p): hashed(p) for p in (paths['scheduler']/'controller_process.json', paths['supervisor']/'supervisor_process.json')}
    return dict(scheduler=scheduler, supervisor=supervisor, profile=profile, gates=gates,
                controller_hashes=controller_files, coordinator_hashes=coordinator_stopped(paths, runtime))


def save(paths, runtime, audit, stage, **details):
    h.save_audit(paths, audit, stage, runtime, **details)


def verify_static(paths, runtime, audit):
    require(stats(audit['bundle_hashes']) == audit['bundle_stats'], 'Reviewed source/plan/task/manifest evidence changed')
    require(coordinator_stopped(paths, runtime) == audit['coordinator_hashes'], 'Stopped coordinator evidence changed')
    supervisor = runtime.identity(audit['supervisor']['pid'])
    require(owned(supervisor, audit['supervisor']) and supervisor['state'] not in ('T', 't'), 'Original supervisor changed/stopped')
    require(hashed(paths['supervisor']/'supervisor_process.json') == audit['controller_hashes'][str(paths['supervisor']/'supervisor_process.json')],
            'Supervisor receipt changed')
    h.stable_supervisor_state(audit['profile'])


def outcome(queue, audit):
    before, after = json.loads(audit['queue_before']), json.loads(audit['queue_proposed'])
    current = {job['id']: job for job in queue['jobs']}
    if len(current) != len(queue['jobs']):
        return 'unknown'
    logical = audit['installation']['logical_jobs']; slots = audit['installation']['slot_job_ids']
    if (all(job['id'] not in current for job in logical)
            and queue['jobs'][:len(after['jobs'])] == after['jobs']
            and {key: value for key, value in queue.items() if key != 'jobs'} ==
                {key: value for key, value in after.items() if key != 'jobs'}):
        return 'committed'
    if all(current.get(job['id']) == job for job in logical) and all(key not in current for key in slots):
        return 'not_committed'
    return 'unknown'


def authorization(process):
    require(process.get('state') not in (*h.DEAD_STATES, 'T', 't'), 'STOP precheck must be live and not already stopped')
    return {key: copy.deepcopy(process[key]) for key in ('pid', 'start_ticks', 'argv', 'uid', 'ppid', 'state')}


def role_process(audit, role):
    return audit['scheduler'] if role == 'scheduler' else audit['gates'][role]


def stopped_target(runtime, audit, role, dead_children=False):
    expected = role_process(audit, role); current = runtime.identity(expected['pid'])
    require(role in audit.get('stop_authorizations', {}) and audit['stop_authorizations'][role] == authorization(expected)
            and any(event.get('stage') == 'stop_intent' for event in audit.get('events', []))
            and owned(current, expected, stopped=True) and current.get('sigterm_default'), 'Target is not this transaction\'s exact stopped CPU process')
    if role != 'scheduler':
        h.gate_parts(current['argv']); require(not current.get('children'), 'Stopped gate has children')
    elif dead_children:
        allowed = {str(g['pid']): g for g in audit['gates'].values()}
        for pid in current.get('children', []):
            require(pid in allowed, 'Old allocator has an unknown child before TERM')
            child = runtime.identity(int(pid))
            require(child.get('state') in h.DEAD_STATES and (child.get('state') == 'absent'
                    or child.get('start_ticks') == allowed[pid]['start_ticks'])
                    and audit.get('exited_gate_proofs', {}).get(pid, {}).get('start_ticks') == allowed[pid]['start_ticks'],
                    'Old allocator child lacks this transaction\'s exact exited pidfd proof')
    else:
        require(sorted(current.get('children', [])) == sorted(str(g['pid']) for g in audit['gates'].values()),
                'Allocator spawned/changed a child before STOP quiescence')
    return current


def stop_role(paths, runtime, audit, role, fd):
    process = role_process(audit, role)
    audit.setdefault('stop_authorizations', {})[role] = authorization(process)
    save(paths, runtime, audit, 'stop_intent', signal_role=role, signal_name='SIGSTOP')
    verify_static(paths, runtime, audit)
    actual = runtime.identity(process['pid'])
    require(owned(actual, process) and actual['state'] not in ('T', 't') and not runtime.exited(fd), 'Process changed before own STOP')
    if role != 'scheduler':
        require(not actual.get('children') and actual.get('sigterm_default'), 'CPU gate became unsafe before STOP')
    runtime.send(fd, signal.SIGSTOP)
    deadline = runtime.monotonic()+2
    while runtime.identity(process['pid']).get('state') != 'T':
        require(not runtime.exited(fd) and runtime.monotonic() < deadline, 'Own STOP did not settle')
        runtime.sleep(.01)
    stopped_target(runtime, audit, role)
    save(paths, runtime, audit, 'stopped_verified', stopped_role=role)


def restore_precommit(paths, runtime, audit, fds):
    require(outcome(h.read_json(paths['queue']), audit) == 'not_committed'
            and not any(key in h.read_json(paths['states']) for key in audit['installation']['slot_job_ids']),
            'Committed/unknown queue cannot authorize precommit CONT')
    for role in [*audit['gates'], 'scheduler']:
        if role not in audit.get('stop_authorizations', {}):
            continue
        expected = role_process(audit, role); actual = runtime.identity(expected['pid'])
        require(audit['stop_authorizations'][role] == authorization(expected), 'Missing exact own STOP authorization')
        require(any(event.get('stage') == 'stop_intent' for event in audit.get('events', [])), 'No durable own STOP intent event')
        require(owned(actual, expected, allow_inner=role != 'scheduler'), 'Precommit restore identity changed; no signal')
        if actual['state'] != 'T':
            intent = audit.get('restore_intents', {}).get(role)
            target = {key: copy.deepcopy(actual[key]) for key in ('pid', 'start_ticks', 'argv', 'uid', 'ppid')}
            require(actual['state'] in ('S', 'R') and intent == dict(target, from_state='T', signal='SIGCONT'),
                    'Own STOP has not been observed and no exact durable CONT recovery intent exists; delayed STOP requires cleanup')
            continue
        fd = fds.get(role)
        if fd is None:
            fd = runtime.open_pidfd(expected['pid']); fds[role] = fd
        target = {key: copy.deepcopy(actual[key]) for key in ('pid', 'start_ticks', 'argv', 'uid', 'ppid')}
        audit.setdefault('restore_intents', {})[role] = dict(target, from_state='T', signal='SIGCONT')
        save(paths, runtime, audit, 'precommit_restore_intent', signal_role=role, signal_name='SIGCONT')
        require(outcome(h.read_json(paths['queue']), audit) == 'not_committed'
                and not any(key in h.read_json(paths['states']) for key in audit['installation']['slot_job_ids'])
                and owned(runtime.identity(expected['pid']), expected, allow_inner=role != 'scheduler', stopped=True)
                and not runtime.exited(fd), 'Precommit restore proof changed')
        fresh = runtime.identity(expected['pid'])
        require({key: fresh.get(key) for key in target} == target and fresh.get('state') == 'T',
                'Exact CONT recovery target changed after durable intent')
        runtime.send(fd, signal.SIGCONT)
    save(paths, runtime, audit, 'aborted_restored', commit_outcome='not_committed')


def terminate_role(paths, runtime, audit, role, fds):
    expected = role_process(audit, role); actual = runtime.identity(expected['pid'])
    if actual.get('state') in h.DEAD_STATES:
        if role != 'scheduler' and actual.get('state') != 'absent':
            require(actual.get('start_ticks') == expected['start_ticks'], 'Exited gate PID identity changed')
            fd = fds.get(role)
            if fd is None:
                fd = runtime.open_pidfd(expected['pid']); fds[role] = fd
            require(runtime.exited(fd), 'Exited gate lacks pidfd exit proof')
            audit.setdefault('exited_gate_proofs', {})[str(expected['pid'])] = dict(start_ticks=expected['start_ticks'], pidfd_exited=True)
            save(paths, runtime, audit, 'owned_gate_exited')
        return
    fd = fds.get(role)
    if fd is None:
        fd = runtime.open_pidfd(expected['pid']); fds[role] = fd
    for sig in (signal.SIGTERM, signal.SIGCONT):
        if runtime.exited(fd):
            break
        stopped_target(runtime, audit, role, dead_children=role == 'scheduler')
        save(paths, runtime, audit, 'terminating_owned_cpu', signal_role=role, signal_name=signal.Signals(sig).name)
        verify_static(paths, runtime, audit)
        require(outcome(h.read_json(paths['queue']), audit) == 'committed', 'Committed queue identity changed before cleanup signal')
        stopped_target(runtime, audit, role, dead_children=role == 'scheduler')
        require(not runtime.exited(fd), 'Owned pidfd exited before cleanup signal')
        runtime.send(fd, sig)
    deadline = runtime.monotonic()+3
    while not runtime.exited(fd):
        require(runtime.monotonic() < deadline, 'Owned CPU process exit not observed; re-enter same audit')
        runtime.sleep(.05)
    if role != 'scheduler':
        audit.setdefault('exited_gate_proofs', {})[str(expected['pid'])] = dict(start_ticks=expected['start_ticks'], pidfd_exited=True)
        save(paths, runtime, audit, 'owned_gate_exited')


def activate(paths, audit):
    root = paths['campaign']; installation = audit['installation']
    campaign.append_identical(root/'transactions/initial/committed.json', dict(schema=campaign.ACTIVATION_SCHEMA,
        task_sha256=installation['task_sha256'], plan_sha256=installation['plan_sha256'],
        queue_after_sha256=installation['queue_proposed_sha256']))
    campaign.append_identical(root/'deployed.json', dict(schema='ease-gpu-resource-deployment-v1',
        plan_sha256=installation['plan_sha256'], slot_job_ids=installation['slot_job_ids'],
        queue_after_sha256=installation['queue_proposed_sha256']))


def new_scheduler(runtime, paths, audit, process):
    current = runtime.identity(process['pid']); receipt = h.read_json(paths['scheduler']/'controller_process.json')
    require(h.same_process(current, process) and current.get('uid') == process.get('uid') == os.getuid()
            and current.get('argv') == audit['new_scheduler_argv']
            and current['state'] not in ('T', 't') and receipt.get('pid') == current['pid']
            and runtime.matches_started(current, receipt['started'])
            and receipt.get('argv') == [str(PROJECT/'preparations/ease_insights_20261103/scheduler.py'),
                                       *h.module_invocation(audit['new_scheduler_argv'])['args']],
            'Recorded replacement scheduler receipt/OS identity is not acknowledged')
    return current


def finish_under_restart_lock(paths, runtime, audit, fds):
    verify_static(paths, runtime, audit)
    require(outcome(h.read_json(paths['queue']), audit) == 'committed', 'Unknown committed proposal; preserve stopped CPU processes')
    for role in audit['gates']:
        terminate_role(paths, runtime, audit, role, fds)
    # Frozen supervisor sees fully-ready slots before the allocator gap.
    activate(paths, audit); save(paths, runtime, audit, 'activated')
    terminate_role(paths, runtime, audit, 'scheduler', fds)
    deadline = runtime.monotonic()+5
    while any(runtime.identity(p['pid']).get('state') != 'absent' for p in [audit['scheduler'], *audit['gates'].values()]):
        require(runtime.monotonic() < deadline, 'Original scheduler/gates await OS reap; do not spawn against stale running state')
        runtime.sleep(.05)
    if audit.get('new_scheduler') is not None:
        new_scheduler(runtime, paths, audit, audit['new_scheduler'])
        return
    if audit.get('spawn_intent_at') is not None:
        receipt = h.read_json(paths['scheduler']/'controller_process.json')
        require(receipt.get('pid') != audit['scheduler']['pid'], 'Spawn intent has no bound replacement receipt; no duplicate allocator spawn')
        candidate = runtime.identity(receipt['pid'])
        require(candidate.get('argv') == audit['new_scheduler_argv'] and candidate.get('uid') == os.getuid()
                and runtime.matches_started(candidate, receipt['started'])
                and receipt['started'] >= audit['spawn_intent_at']-1,
                'Ambiguous post-spawn identity; no second allocator will be started')
        audit['new_scheduler'] = candidate
        save(paths, runtime, audit, 'scheduler_recovered')
        new_scheduler(runtime, paths, audit, candidate)
        return
    save(paths, runtime, audit, 'scheduler_spawn_intent', spawn_intent_at=runtime.now())
    process = runtime.start_scheduler(audit['new_scheduler_argv'], paths['audit']/'scheduler.log')
    audit['new_scheduler'] = process
    save(paths, runtime, audit, 'scheduler_spawned')
    deadline = runtime.monotonic()+10
    while True:
        try:
            new_scheduler(runtime, paths, audit, process)
            break
        except (Refused, FileNotFoundError):
            require(runtime.identity(process['pid']).get('state') not in h.DEAD_STATES
                    and runtime.monotonic() < deadline, 'Replacement scheduler identity ack timed out; preserve same PID, never respawn')
            runtime.sleep(.05)
    save(paths, runtime, audit, 'scheduler_registered')


def finish_ack(paths, runtime, audit):
    deadline = runtime.monotonic()+ACK_TIMEOUT
    while True:
        verify_static(paths, runtime, audit)
        new_scheduler(runtime, paths, audit, audit['new_scheduler'])
        state = h.read_json(paths['supervisor']/'state.json'); status = h.read_json(paths['supervisor']/'status.json')
        if (state.get('scheduler', {}).get('pid') == audit['new_scheduler']['pid']
                and state.get('phase') == status.get('phase') == 'waiting_discovery' and status.get('terminal') is False
                and isinstance(status.get('time'), (int, float)) and status['time'] > audit['spawn_intent_at']):
            save(paths, runtime, audit, 'complete', supervisor_adopted_pid=audit['new_scheduler']['pid'])
            return audit
        require(runtime.monotonic() < deadline, 'Supervisor adoption ack timed out; re-enter same audit/PID, never spawn again')
        runtime.sleep(.2)


REVIEW_KEYS = ('schema', 'request', 'installation', 'queue_before', 'queue_proposed', 'bundle_hashes', 'bundle_stats',
               'scheduler', 'supervisor', 'profile', 'gates', 'controller_hashes', 'coordinator_hashes', 'new_scheduler_argv')


def review_value(audit):
    return {key: audit[key] for key in REVIEW_KEYS}


def read_review(args, paths, audit):
    require(isinstance(args.review_sha256, str) and re.fullmatch('[a-f0-9]{64}', args.review_sha256), 'Apply requires independently reviewed.json SHA256')
    raw = (paths['audit']/'reviewed.json').read_bytes()
    require(h.digest(raw) == args.review_sha256 and json.loads(raw) == review_value(audit), 'Full-byte independent review identity differs')
    require(audit.get('review_sha256', args.review_sha256) == args.review_sha256, 'Recovery review changed')
    return raw


def operate(args, runtime=None):
    runtime = runtime or Runtime(); paths = paths_for(args)
    request = dict(campaign_root=str(paths['campaign']), output_audit=str(paths['audit']))
    try:
        bundle = verify_bundle(paths)
    except BaseException:
        # A changed source refuses installation but must not strand a proven
        # own precommit STOP. This narrow recovery uses immutable reviewed
        # identities and queue absence, never the changed implementation plan.
        prior_path = paths['audit']/'audit.json'
        if args.apply and prior_path.exists():
            prior = h.read_json(prior_path)
            require(prior.get('schema') == SCHEMA and prior.get('request') == request, 'Source recovery request differs')
            read_review(args, paths, prior)
            if prior.get('stop_authorizations') and outcome(h.read_json(paths['queue']), prior) == 'not_committed':
                restored_fds = {}
                try:
                    with h.lock(paths['audit']/'transaction.lock'), h.lock(paths['restart']), h.lock(paths['queue'].with_name('queue.json.lock')):
                        restore_precommit(paths, runtime, prior, restored_fds)
                finally:
                    for fd in restored_fds.values(): runtime.close_pidfd(fd)
        raise
    paths['audit'].mkdir(exist_ok=True)
    fds = {}; audit = None
    with h.lock(paths['audit']/'transaction.lock'):
        audit_file = paths['audit']/'audit.json'
        if not audit_file.exists():
            require(not args.apply, 'Prepare and independently review dry-run before apply')
            require(paths['queue'].read_bytes() == bundle['before'], 'Live queue differs from prepared before bytes')
            inspected = capture(paths, runtime, bundle)
            audit = dict(schema=SCHEMA, request=request, installation=bundle['installation'], queue_before=bundle['before'].decode(),
                queue_proposed=bundle['proposed'].decode(), bundle_hashes=bundle['hashes'], bundle_stats=bundle['stats'],
                new_scheduler_argv=scheduler_argv(paths, 2), **inspected)
            h.durable_json(paths['audit']/'review.template.json', review_value(audit))
            save(paths, runtime, audit, 'dry_run')
            return audit
        audit = h.read_json(audit_file)
        require(audit.get('schema') == SCHEMA and audit.get('request') == request, 'Audit belongs to another install request')
        if not args.apply:
            require(audit['bundle_hashes'] == bundle['hashes'] and audit['bundle_stats'] == bundle['stats'], 'Reviewed bundle/source identity changed')
            require(audit['stage'] == 'dry_run', 'Transaction already began; same apply package is required')
            return audit
        reviewed_raw = read_review(args, paths, audit)
        if audit['bundle_hashes'] != bundle['hashes'] or audit['bundle_stats'] != bundle['stats']:
            try:
                if audit.get('stop_authorizations') and outcome(h.read_json(paths['queue']), audit) == 'not_committed':
                    with h.lock(paths['restart']), h.lock(paths['queue'].with_name('queue.json.lock')):
                        restore_precommit(paths, runtime, audit, fds)
            finally:
                for fd in fds.values(): runtime.close_pidfd(fd)
            raise Refused('Reviewed bundle/source identity changed; only proven own precommit STOP was restored')
        if audit.get('stage') in ('complete', 'aborted_restored'):
            return audit
        audit['review_sha256'] = args.review_sha256
        try:
            # Shared with BOTH frozen supervisor restart paths. After a proven
            # new receipt is durable, release this lock before supervisor ACK.
            with h.lock(paths['restart']), h.lock(paths['queue'].with_name('queue.json.lock')):
                current_outcome = outcome(h.read_json(paths['queue']), audit)
                if current_outcome == 'not_committed':
                    if audit.get('stop_authorizations'):
                        restore_precommit(paths, runtime, audit, fds)
                        return audit
                    require(paths['queue'].read_bytes() == audit['queue_before'].encode(), 'Reviewed queue changed before STOP')
                    inspected = capture(paths, runtime, bundle)
                    for key in ('scheduler', 'supervisor'):
                        require(owned(inspected[key], audit[key]), 'Reviewed controller changed')
                    require(set(inspected['gates']) == set(audit['gates'])
                            and all(owned(inspected['gates'][key], value) for key, value in audit['gates'].items())
                            and inspected['controller_hashes'] == audit['controller_hashes']
                            and inspected['coordinator_hashes'] == audit['coordinator_hashes'], 'Reviewed gate/controller receipt identity changed')
                    fds['scheduler'] = runtime.open_pidfd(audit['scheduler']['pid'])
                    stop_role(paths, runtime, audit, 'scheduler', fds['scheduler'])
                    for role, gate in audit['gates'].items():
                        fds[role] = runtime.open_pidfd(gate['pid'])
                        stop_role(paths, runtime, audit, role, fds[role])
                    save(paths, runtime, audit, 'commit_intent')
                    verify_static(paths, runtime, audit)
                    stopped_target(runtime, audit, 'scheduler')
                    for role in audit['gates']:
                        stopped_target(runtime, audit, role)
                    require(read_review(args, paths, audit) == reviewed_raw
                            and paths['queue'].read_bytes() == audit['queue_before'].encode()
                            and bundle['proposed'] == audit['queue_proposed'].encode(), 'Final queue CAS/review bytes changed')
                    states = h.read_json(paths['states'])
                    require(all(states.get(key, dict(status='pending')) == value for key, value
                                in audit['installation']['logical_states_before'].items()), 'Logical state changed before commit')
                    temporary = paths['audit']/'queue.commit.tmp'; h.durable_file(temporary, bundle['proposed'])
                    # The final temporary write can block; redo all identities.
                    verify_static(paths, runtime, audit); stopped_target(runtime, audit, 'scheduler')
                    for role in audit['gates']: stopped_target(runtime, audit, role)
                    require(paths['queue'].read_bytes() == audit['queue_before'].encode()
                            and read_review(args, paths, audit) == reviewed_raw and temporary.read_bytes() == bundle['proposed'],
                            'Final durable-write queue CAS differs from reviewed bytes')
                    runtime.replace_queue(temporary, paths['queue']); h.sync_directory(paths['queue'].parent)
                    save(paths, runtime, audit, 'committed', committed_at=runtime.now(), commit_outcome='committed')
                else:
                    require(current_outcome == 'committed', 'Unknown queue state; preserve stopped processes, no rollback/CONT')
                    audit.setdefault('committed_at', runtime.now())
                finish_under_restart_lock(paths, runtime, audit, fds)
            return finish_ack(paths, runtime, audit)
        except BaseException as error:
            try:
                current_outcome = outcome(h.read_json(paths['queue']), audit)
            except Exception:
                current_outcome = 'unknown'
            if current_outcome == 'not_committed' and audit.get('stop_authorizations'):
                try:
                    with h.lock(paths['restart']), h.lock(paths['queue'].with_name('queue.json.lock')):
                        restore_precommit(paths, runtime, audit, fds)
                except BaseException as restore_error:
                    save(paths, runtime, audit, 'needs_cleanup', error=repr(error), restore_error=repr(restore_error), commit_outcome=current_outcome)
            else:
                save(paths, runtime, audit, 'needs_cleanup', error=repr(error), commit_outcome=current_outcome)
            raise
        finally:
            for fd in fds.values(): runtime.close_pidfd(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-root', required=True)
    parser.add_argument('--output-audit', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--review-sha256')
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise InterruptedError(f'Installer interrupted by {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    result = operate(args)
    print(json.dumps(dict(stage=result['stage'], output_audit=args.output_audit,
                          new_scheduler_pid=result.get('new_scheduler', {}).get('pid'))))


if __name__ == '__main__':
    main()

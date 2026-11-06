"""One-shot, fixed-experiment refresh of an unstarted local-search CPU gate.

Default prepares queue.before.json, queue.proposed.json, audit.json and a review
template only. A separate reviewer must write reviewed.json and pass its SHA256
to --apply. No GPU query, lease probe, controller restart or jobs_state write is
performed. The original scheduler alone allocates the replacement. The pinned
opportunity coordinator must have stopped itself with no pending transaction.

After commit, repeat the same audit and review hash to finish cleanup. Never
roll back an unknown/committed queue. A precommit exec race only undoes our own
STOP on the exact recorded runner; it never terminates a scientific runner.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import re
import signal

from preparations.ease_insights_20261103 import reassign_waiting_gate_v2 as h
from preparations.ease_insights_20261103 import gate_opportunity_coordinator as coordinator


PROJECT = Path('/root/nfs/embodied/benchmark')
ROOT = PROJECT / 'results/ease_insights_20261103'
SCHEMA = 'ease-waiting-gate-policy-refresh-v1'
HELPER_SHA256 = '0238d949fd5585fc52dae19fc56c54b5b970cb4542714c4599024418c8169498'
COORDINATOR_SHA256 = '70ce6ef8a86153ad1f3cc6ffe0c4d05cc7e2d2dd334423ceb3af9598a7207432'
GATE_SHA256 = 'b4b5ca1bf95c8f43103e9a40b4de8b8cd8036bc6c870b12c2bf58dac6653e4cf'
MANIFEST_HASH = '032eacba3709cd363e528b4f4ae1acbc52b325c6c99fb42a0e0ef0043c2817a9'
SCHEDULER_PID, SCHEDULER_TICKS = 335208, '2615895606'
SUPERVISOR_PID, SUPERVISOR_TICKS = 338645, '2616104350'
COORDINATOR_PID, COORDINATOR_TICKS = 356386, '2617020452'
JOBS = {
    'vlm_local_v1_discovery_local_spatial_all120': (344836, '2616450623', 'local_spatial'),
    'vlm_local_v1_discovery_full_d10_all120__wait_reassign_5f9688a613c6': (361717, '2617253193', 'full_d10'),
}
POLICY_ENV = {'EASE_GPU_GATE_IDLE_SECONDS': '5', 'EASE_GPU_GATE_POLL_SECONDS': '2'}
ACK_TIMEOUT = 45.0
Refused = h.Refused
require = h.require


class LocalRuntime(coordinator.SystemRuntime):
    """Reuse pidfd/proc/atomic replace primitives, with GPU access disabled."""
    def gpu(self, uuid):
        raise Refused('Policy refresh never queries GPUs')

    def lease_available(self, uuid):
        raise Refused('Policy refresh never probes or allocates a GPU lease')


def paths_for(args):
    require(ROOT.resolve() == ROOT and PROJECT.resolve() == PROJECT, 'Noncanonical fixed experiment root')
    audit = Path(args.output_audit)
    require(audit.is_absolute() and audit.resolve() == audit and audit.parent == ROOT/'gate_policy_refreshes',
            'Audit must be a direct transaction directory in the fixed gate_policy_refreshes root')
    require(args.job_id in JOBS, 'Only the two pinned unstarted local-search gates are supported')
    paths = dict(root=ROOT, queue=ROOT/'queue.json', scheduler=ROOT/'scheduler',
                 states=ROOT/'scheduler/jobs_state.json', audit=audit)
    for path in (*paths.values(), ROOT/'gate_opportunity_coordinator', ROOT/'vlm_local_search_v1/supervisor'):
        require(path.resolve() == path, 'Noncanonical experiment artifact')
    return paths


def file_hash(path):
    return h.digest(Path(path).read_bytes())


def same_controller(actual, expected):
    return h.same_process(actual, expected) and actual.get('_scheduler_proof') == expected.get('_scheduler_proof')


def verify_sources(paths, audit=None):
    base = PROJECT/'preparations/ease_insights_20261103'
    require(Path(h.__file__).resolve() == base/'reassign_waiting_gate_v2.py'
            and file_hash(h.__file__) == HELPER_SHA256, 'Pinned v2 helper source changed')
    require(file_hash(base/'gate_opportunity_coordinator.py') == COORDINATOR_SHA256,
            'Pinned coordinator source changed')
    require(file_hash(base/'await_assigned_gpu.py') == GATE_SHA256, 'Reviewed 5/2-capable gate source changed')
    manifest_path = paths['root']/'vlm_local_search_v1/discovery/manifest.json'
    manifest = h.read_json(manifest_path)
    require(manifest.get('manifest_hash') == MANIFEST_HASH, 'Pinned scientific manifest changed')
    hashes = {str(manifest_path): file_hash(manifest_path), str(Path(__file__).resolve()): file_hash(__file__),
              str(base/'await_assigned_gpu.py'): file_hash(base/'await_assigned_gpu.py'),
              str(Path(h.__file__).resolve()): HELPER_SHA256,
              str(base/'gate_opportunity_coordinator.py'): COORDINATOR_SHA256}
    for group in ('source_files', 'local_runtime_sources', 'analysis_sources'):
        mapping = manifest.get(group)
        require(isinstance(mapping, dict) and mapping, 'Missing frozen source group: '+group)
        for name, expected in mapping.items():
            path = PROJECT/name
            require(path.is_relative_to(PROJECT) and path.resolve() == path and file_hash(path) == expected,
                    'Frozen source changed: '+name)
            hashes[str(path)] = expected
    if audit is not None:
        require(hashes == audit['source_hashes'], 'Reviewed source identities changed')
    return hashes


def source_stats(hashes):
    result = {}
    for name in hashes:
        value = Path(name).stat()
        result[name] = [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]
    return result


def verify_coordinator(paths, runtime, audit=None):
    directory = paths['root']/'gate_opportunity_coordinator'
    require((directory/'stop').is_file(), 'Coordinator stop file is required')
    receipt = h.read_json(directory/'process.json')
    require(receipt.get('pid') == COORDINATOR_PID and receipt.get('start_ticks') == COORDINATOR_TICKS
            and receipt.get('helper_sha256') == HELPER_SHA256
            and receipt.get('coordinator_sha256') == COORDINATOR_SHA256,
            'Original coordinator receipt changed')
    state = h.read_json(directory/'state.json')
    require(state.get('schema') == 'ease-gate-opportunity-coordinator-v1'
            and state.get('pending') is None and state.get('phase') == 'stopped',
            'Coordinator has not stopped itself with pending=None')
    require(runtime.identity(COORDINATOR_PID).get('state') in h.DEAD_STATES,
            'Original coordinator is still alive; no refresh is allowed')
    hashes = {str(directory/'process.json'): file_hash(directory/'process.json'),
              str(directory/'state.json'): file_hash(directory/'state.json')}
    if audit is not None:
        require(hashes == audit['coordinator_hashes'], 'Stopped coordinator evidence changed')
    return hashes


def no_cases(job, profile):
    require(not Path(job['complete_file']).exists(), 'Completion marker exists; no refresh is allowed')
    condition = h.option(h.gate_parts(job['argv'])[2], '--condition')
    directory = Path(profile['scientific_output'])/condition
    require(directory.resolve() == directory and (not directory.exists() or
            directory.is_dir() and not any(directory.iterdir())),
            'Condition has runner/case artifacts; only a never-started zero-case job is supported')


def receipt(runtime, path, pid, uuid):
    path = Path(path)
    before = path.stat(); value = h.read_json(path); after = path.stat()
    require((before.st_ino, before.st_mtime_ns, before.st_size) ==
            (after.st_ino, after.st_mtime_ns, after.st_size), 'Gate receipt changed during read')
    require(0 <= runtime.now()-after.st_mtime <= h.RECEIPT_MAX_AGE
            and value.get('status') in ('busy', 'waiting', 'check_error')
            and value.get('pid') == pid and value.get('gpu_uuid') == uuid,
            'Gate must have a fresh non-launching CPU-wait receipt')
    return value


def bound_controller(runtime, receipt_path, module, paths, profile=None):
    value = h.read_json(receipt_path)
    process = h.controller(runtime, value, module, paths['queue'],
                           paths['scheduler'] if module == h.SCHEDULER else Path(profile['output']), profile)
    pid, ticks = (SCHEDULER_PID, SCHEDULER_TICKS) if module == h.SCHEDULER else (SUPERVISOR_PID, SUPERVISOR_TICKS)
    require(process['pid'] == pid and process['start_ticks'] == ticks, 'Original controller identity changed')
    parsed = h.module_invocation(process['argv'])
    script = PROJECT/(module.replace('.', '/')+'.py')
    require(value.get('argv') == [str(script), *parsed['args']], 'Controller receipt argv differs from live invocation')
    if module == h.SCHEDULER:
        require(parsed['args'] == ['--queue', str(paths['queue']), '--output', str(paths['scheduler']),
                '--max-gpus', '2', '--scan-seconds', '30', '--idle-seconds', '0'],
                'Original scheduler allocation/scan configuration changed')
    else:
        require(h.option(process['argv'], '--scheduler-pid') == str(SCHEDULER_PID), 'Supervisor bound scheduler changed')
    return process


def inspect(args, paths, runtime, queue, states):
    ids = [j['id'] for j in queue['jobs']]
    require(len(ids) == len(set(ids)) and ids.count(args.job_id) == 1, 'Selected job must occur exactly once')
    job = next(j for j in queue['jobs'] if j['id'] == args.job_id)
    state = states.get(args.job_id, {})
    pid, ticks, condition = JOBS[args.job_id]
    require(state.get('status') == 'running' and state.get('pid') == pid, 'Pinned CPU gate is no longer running')
    uuid, status_path, inner = h.gate_parts(job['argv'])
    require(uuid == '{gpu_uuid}' and status_path == paths['scheduler']/'launch_gates'/(args.job_id+'.json')
            and job.get('launch_gate', {}).get('original_argv') == inner,
            'Original gate wrapper binding changed')
    require(h.option(inner, '--condition') == condition and job.get('expected_episodes') == 120
            and job['launch_gate'].get('continuous_idle_seconds') == 60
            and job['launch_gate'].get('poll_seconds') == 10
            and job['launch_gate'].get('assigned_gpu_only') is True
            and not set(POLICY_ENV).intersection(job.get('env', {})), 'Original job/policy is outside this one-shot refresh')
    profile = h.bound_profile(job, paths)
    require(profile['kind'] == 'local_search' and profile['stage'] == 'discovery'
            and profile['runner_module'] == h.LOCAL_RUNNER, 'Only exact local-search discovery is supported')
    no_cases(job, profile)
    require(not (paths['scheduler']/'stop_after_jobs.json').exists(), 'Scheduler stop marker is present')
    sched_receipt = paths['scheduler']/'controller_process.json'
    super_receipt = Path(profile['output'])/'supervisor_process.json'
    scheduler = bound_controller(runtime, sched_receipt, h.SCHEDULER, paths)
    supervisor = bound_controller(runtime, super_receipt, h.LOCAL_SUPERVISOR, paths, profile)
    h.stable_supervisor_state(profile)
    process = runtime.identity(pid)
    require(process.get('start_ticks') == ticks and process.get('state') not in (*h.DEAD_STATES, 'T', 't')
            and process.get('argv') == state.get('argv') == h.concrete(job['argv'], state)
            and process.get('uid') == os.getuid() and process.get('ppid') == scheduler['pid']
            and not process.get('children') and process.get('sigterm_default'), 'Pinned process is not its live owned childless CPU gate')
    require(runtime.matches_started(process, state['started']), 'Pinned gate start identity differs')
    h.gate_parts(process['argv'])
    h.assigned_uuid(state.get('gpu_uuid', ''))
    gate_receipt = receipt(runtime, status_path, pid, state['gpu_uuid'])
    return dict(job=job, state=state, gate=process, scheduler=scheduler, supervisor=supervisor, profile=profile,
                receipt_path=str(status_path), gate_receipt=gate_receipt,
                controller_hashes={str(p): file_hash(p) for p in (sched_receipt, super_receipt)})


def make_proposal(queue, states, inspected):
    old = inspected['job']
    identity = f"{old['id']}:{inspected['gate']['pid']}:{inspected['gate']['start_ticks']}:5:2"
    new_id = old['id']+'__gate_policy_5s2s_'+h.digest(identity.encode())[:12]
    require(new_id not in states and all(j['id'] != new_id for j in queue['jobs']), 'Replacement ID already exists')
    new = copy.deepcopy(old); new['id'] = new_id
    index = h.module_invocation(new['argv'])['args_index']+3
    new['argv'][index] = str(Path(inspected['receipt_path']).with_name(new_id+'.json'))
    require(not Path(new['argv'][index]).exists(), 'Replacement status receipt already exists')
    new.setdefault('env', {}).update(POLICY_ENV)
    new['launch_gate'].update(continuous_idle_seconds=5, poll_seconds=2)
    proposal = copy.deepcopy(queue)
    proposal['jobs'] = [copy.deepcopy(new) if j['id'] == old['id'] else j for j in proposal['jobs']]
    for job in proposal['jobs']:
        if 'depends_on' in job:
            job['depends_on'] = [new_id if dep == old['id'] else dep for dep in job['depends_on']]
    return proposal, new


REVIEW_FIELDS = ('schema', 'request', 'old_job_id', 'original_job', 'new_job', 'job_state', 'gate',
                 'scheduler', 'supervisor', 'profile', 'controller_hashes', 'coordinator_hashes', 'source_hashes', 'source_stats')


def review_value(audit, before, proposed):
    return dict({k: audit[k] for k in REVIEW_FIELDS}, before=before.decode('utf-8'), proposed=proposed.decode('utf-8'),
                before_sha256=h.digest(before), proposed_sha256=h.digest(proposed))


def verify_proposal(audit, before_raw, proposed_raw, paths):
    """Independent full-value validator; intentionally does not use make_proposal."""
    before, proposed = json.loads(before_raw), json.loads(proposed_raw)
    old, new = audit['original_job'], audit['new_job']
    require(old['id'] in JOBS and audit['old_job_id'] == old['id'], 'Unrecognized original job')
    pid, ticks, condition = JOBS[old['id']]
    require(audit['gate']['pid'] == pid and audit['gate']['start_ticks'] == ticks
            and h.option(h.gate_parts(old['argv'])[2], '--condition') == condition, 'Reviewed gate differs from pinned target')
    expected_id = old['id']+'__gate_policy_5s2s_'+h.digest(f"{old['id']}:{pid}:{ticks}:5:2".encode())[:12]
    require(new['id'] == expected_id, 'Replacement identity differs')
    expected = copy.deepcopy(old); expected['id'] = expected_id
    uuid, old_status, inner = h.gate_parts(old['argv'])
    new_uuid, new_status, new_inner = h.gate_parts(new['argv'])
    require(uuid == new_uuid == '{gpu_uuid}' and inner == new_inner == old['launch_gate']['original_argv']
            and old_status == paths['scheduler']/'launch_gates'/(old['id']+'.json')
            and new_status == paths['scheduler']/'launch_gates'/(new['id']+'.json'), 'Scientific argv or status binding changed')
    expected['argv'][h.module_invocation(expected['argv'])['args_index']+3] = str(new_status)
    expected['env'] = dict(old.get('env', {}), EASE_GPU_GATE_IDLE_SECONDS='5', EASE_GPU_GATE_POLL_SECONDS='2')
    expected['launch_gate']['continuous_idle_seconds'] = 5
    expected['launch_gate']['poll_seconds'] = 2
    require(new == expected, 'Proposal changed scientific job semantics')
    require(sum(j == old for j in before['jobs']) == 1 and not any(j['id'] == new['id'] for j in before['jobs']),
            'Before queue original/replacement identity differs')
    expected_queue = copy.deepcopy(before)
    expected_queue['jobs'] = [copy.deepcopy(new) if j['id'] == old['id'] else j for j in expected_queue['jobs']]
    for job in expected_queue['jobs']:
        if 'depends_on' in job:
            job['depends_on'] = [new['id'] if dep == old['id'] else dep for dep in job['depends_on']]
    require(proposed == expected_queue, 'Proposal changed unrelated queue/scientific semantics')
    require(h.digest(before_raw) == audit['queue_before_sha256']
            and h.digest(proposed_raw) == audit['queue_after_sha256'], 'Proposal artifact hashes differ')


def read_review(args, paths, audit):
    require(isinstance(args.review_sha256, str) and re.fullmatch('[0-9a-f]{64}', args.review_sha256),
            '--apply requires the independently reviewed.json SHA256')
    raw = (paths['audit']/'reviewed.json').read_bytes()
    require(h.digest(raw) == args.review_sha256, 'Independent review file hash differs')
    before = (paths['audit']/'queue.before.json').read_bytes()
    proposed = (paths['audit']/'queue.proposed.json').read_bytes()
    verify_proposal(audit, before, proposed, paths)
    require(json.loads(raw) == review_value(audit, before, proposed), 'Independent review does not bind complete proposal bytes and identities')
    if audit.get('review_sha256') is not None:
        require(audit['review_sha256'] == args.review_sha256, 'Recovery review hash changed')
    return raw, before, proposed


def verify_controllers(paths, runtime, audit, precommit=False):
    h.verify_controllers(runtime, audit)
    require(source_stats(audit['source_hashes']) == audit['source_stats'], 'Reviewed source metadata changed')
    verify_coordinator(paths, runtime, audit)
    for name, module, path in (('scheduler', h.SCHEDULER, paths['scheduler']/'controller_process.json'),
                               ('supervisor', h.LOCAL_SUPERVISOR, Path(audit['profile']['output'])/'supervisor_process.json')):
        process = bound_controller(runtime, path, module, paths, audit['profile'])
        require(same_controller(process, audit[name]) and file_hash(path) == audit['controller_hashes'][str(path)],
                'Bound controller receipt/start/argv changed')
    if precommit:
        no_cases(audit['original_job'], audit['profile'])


def finish_committed(args, paths, runtime, audit, fd):
    deadline = runtime.monotonic()+ACK_TIMEOUT
    while True:
        verify_controllers(paths, runtime, audit)
        require(h.retained_commit(h.read_json(paths['queue']), audit), 'Committed queue replacement changed; review required')
        exited = fd is None or runtime.exited(fd)
        if not exited:
            h.verified_stopped(runtime, audit)
        states = h.read_json(paths['states'])
        status = h.read_json(Path(audit['profile']['output'])/'status.json')
        require(status.get('terminal') is not True and status.get('phase') == audit['profile']['expected_phase'],
                'Supervisor left the matching nonterminal waiting phase')
        acknowledged = (audit['new_job']['id'] in states and isinstance(status.get('time'), (int, float))
                        and math.isfinite(status['time']) and audit['committed_at'] < status['time'] <= runtime.now()+1)
        if acknowledged:
            break
        require(runtime.monotonic() < deadline, 'Timed out awaiting scheduler and supervisor post-commit acknowledgements')
        runtime.sleep(.2)
    if exited:
        h.save_audit(paths, audit, 'complete', runtime, gate_exited=True)
        return audit
    with h.lock(paths['queue'].with_name('queue.json.lock')):
        verify_controllers(paths, runtime, audit); h.verified_stopped(runtime, audit)
        require(h.retained_commit(h.read_json(paths['queue']), audit), 'Committed queue changed before cleanup')
        require(audit.get('stop_authorization') == h.stop_authorization(audit)
                and any(e.get('stage') == 'stop_intent' for e in audit.get('events', [])), 'No durable own STOP authorization')
        h.save_audit(paths, audit, 'controllers_acknowledged', runtime,
                     scheduler_ack_job=audit['new_job']['id'], supervisor_ack_time=status['time'])
        h.save_audit(paths, audit, 'terminating_cpu_gate', runtime)
        verify_controllers(paths, runtime, audit); h.verified_stopped(runtime, audit)
        require(h.retained_commit(h.read_json(paths['queue']), audit), 'Committed queue changed before TERM')
        runtime.send(fd, signal.SIGTERM)
        if not runtime.exited(fd):
            verify_controllers(paths, runtime, audit); h.verified_stopped(runtime, audit)
            require(h.retained_commit(h.read_json(paths['queue']), audit), 'Committed queue changed before CONT')
            runtime.send(fd, signal.SIGCONT)
    deadline = runtime.monotonic()+3
    while not runtime.exited(fd):
        require(runtime.monotonic() < deadline, 'CPU gate exit was not observed; cleanup requires review')
        runtime.sleep(.05)
    h.save_audit(paths, audit, 'complete', runtime, gate_exited=True)
    return audit


def operate(args, runtime=None):
    runtime = runtime or LocalRuntime()
    runtime = coordinator.SchedulerIdentityRuntime(runtime, SCHEDULER_PID)
    paths = paths_for(args)
    paths['audit'].mkdir(parents=True, exist_ok=True)
    request = dict(job_id=args.job_id, queue=str(paths['queue']), jobs_state=str(paths['states']),
                   scheduler_output=str(paths['scheduler']), resource_env=POLICY_ENV)
    # Hash small frozen implementation sources once, before any queue lock or
    # STOP. Critical sections use exact inode/mtime/ctime/size snapshots only.
    source_hashes = verify_sources(paths)
    stats = source_stats(source_hashes)
    # Shared with v2 as well as a refresh-wide singleton: no second allocator.
    (ROOT/'gate_policy_refreshes').mkdir(exist_ok=True)
    with h.lock(ROOT/'gate_policy_refreshes/refresh.lock'), h.lock(paths['audit']/'transaction.lock'):
        audit_path = paths['audit']/'audit.json'
        if audit_path.exists():
            audit = h.read_json(audit_path)
            require(audit.get('schema') == SCHEMA and audit.get('request') == request, 'Audit is not this exact refresh request')
            runtime.expected = copy.deepcopy(audit['scheduler']['_scheduler_proof'])
            require(source_hashes == audit['source_hashes'] and stats == audit['source_stats'], 'Reviewed source identities changed')
            if not args.apply:
                require(audit['stage'] == 'dry_run', 'Existing transaction requires explicit apply/recovery')
                return audit
            reviewed_raw, before_raw, proposed_raw = read_review(args, paths, audit)
            if audit.get('stage') in ('complete', 'aborted_restored'):
                return audit
            try:
                outcome = h.commit_outcome(h.read_json(paths['queue']), audit)
            except (OSError, ValueError, KeyError, TypeError):
                outcome = 'unknown'
            if audit.get('committed_at') is not None or outcome == 'committed':
                require(outcome == 'committed', 'Commit outcome is ambiguous; preserve stopped target for review')
                require(audit.get('stop_authorization') == h.stop_authorization(audit)
                        and any(e.get('stage') == 'stop_intent' for e in audit.get('events', [])), 'No durable own STOP authorization')
                audit.setdefault('committed_at', runtime.now())
                current = runtime.identity(audit['gate']['pid'])
                exited = current.get('state') in h.DEAD_STATES
                require(exited or h.same_process(current, audit['gate']), 'Recovery gate identity changed; no signals')
                fd = None if exited else runtime.open_pidfd(current['pid'])
                try:
                    return finish_committed(args, paths, runtime, audit, fd)
                except BaseException as error:
                    h.save_audit(paths, audit, 'needs_cleanup', runtime, error=str(error)); raise
                finally:
                    if fd is not None:
                        runtime.close_pidfd(fd)
            if audit.get('stage') != 'dry_run':
                require(outcome == 'not_committed', 'Commit outcome is ambiguous; no precommit resume')
                return h.restore_precommit(paths, runtime, audit)
        else:
            require(not args.apply, 'Prepare a dry-run and independently review it before --apply')
            audit = None

        committed, stopped, fd = False, False, None
        temporary = paths['queue'].with_name(f'queue.json.refresh-policy.{os.getpid()}.tmp')
        try:
            with h.lock(paths['queue'].with_name('queue.json.lock')):
                current_raw = paths['queue'].read_bytes()
                queue, states = json.loads(current_raw), h.read_json(paths['states'])
                coordinator_hashes = verify_coordinator(paths, runtime)
                inspected = inspect(args, paths, runtime, queue, states)
                if not args.apply:
                    proposal, new = make_proposal(queue, states, inspected)
                    audit = dict(schema=SCHEMA, request=request, old_job_id=args.job_id, original_job=inspected['job'], new_job=new,
                        job_state=inspected['state'], **{k: inspected[k] for k in ('gate', 'scheduler', 'supervisor', 'profile', 'controller_hashes')},
                        receipt_path=inspected['receipt_path'], gate_receipt=inspected['gate_receipt'], source_hashes=source_hashes,
                        coordinator_hashes=coordinator_hashes, source_stats=stats, queue_before_sha256=h.digest(current_raw),
                        queue_after_sha256=h.digest(h.json_bytes(proposal)), allocator='existing_scheduler_only', gpu_queries=0)
                    proposed_raw = h.json_bytes(proposal)
                    verify_proposal(audit, current_raw, proposed_raw, paths)
                    h.durable_file(paths['audit']/'queue.before.json', current_raw)
                    h.durable_file(paths['audit']/'queue.proposed.json', proposed_raw)
                    h.durable_json(paths['audit']/'review.template.json', review_value(audit, current_raw, proposed_raw))
                    h.save_audit(paths, audit, 'dry_run', runtime)
                    return audit
                require(current_raw == before_raw, 'Queue changed after independent dry-run review')
                require(source_hashes == audit['source_hashes'] and coordinator_hashes == audit['coordinator_hashes'], 'Reviewed source/coordinator changed')
                for key, inspected_key in (('original_job', 'job'), ('job_state', 'state'), ('profile', 'profile'), ('controller_hashes', 'controller_hashes')):
                    require(audit[key] == inspected[inspected_key], 'Reviewed '+key+' changed')
                for name in ('gate', 'scheduler', 'supervisor'):
                    require(same_controller(inspected[name], audit[name]), 'Reviewed '+name+' changed')
                audit['review_sha256'] = args.review_sha256
                h.durable_file(temporary, proposed_raw)
                fd = runtime.open_pidfd(audit['gate']['pid'])
                h.save_audit(paths, audit, 'stop_intent', runtime, stop_authorization=h.stop_authorization(audit))
                verify_controllers(paths, runtime, audit, precommit=True)
                current = runtime.identity(audit['gate']['pid'])
                require(h.same_process(current, audit['gate']) and current['state'] not in ('T', 't')
                        and current.get('uid') == os.getuid() and current.get('ppid') == audit['scheduler']['pid']
                        and not current.get('children'), 'Gate identity changed before STOP')
                receipt(runtime, audit['receipt_path'], current['pid'], audit['job_state']['gpu_uuid'])
                require(not runtime.exited(fd), 'Pidfd target exited before STOP')
                stopped = True
                runtime.send(fd, signal.SIGSTOP)
                deadline = runtime.monotonic()+2
                while runtime.identity(audit['gate']['pid']).get('state') != 'T':
                    require(not runtime.exited(fd) and runtime.monotonic() < deadline, 'Gate did not stop promptly')
                    runtime.sleep(.01)
                require(h.same_process(runtime.identity(audit['gate']['pid']), audit['gate']),
                        'Gate exec identity changed during stop; exact precommit restoration required')
                h.verified_stopped(runtime, audit)
                h.save_audit(paths, audit, 'stopped_verified', runtime)
                h.save_audit(paths, audit, 'commit_intent', runtime)
                verify_controllers(paths, runtime, audit, precommit=True); h.verified_stopped(runtime, audit)
                final_review, final_before, final_proposed = read_review(args, paths, audit)
                require(final_review == reviewed_raw and final_before == before_raw and final_proposed == proposed_raw
                        and temporary.read_bytes() == proposed_raw and paths['queue'].read_bytes() == before_raw,
                        'Final queue CAS differs from independently reviewed complete bytes')
                require(h.read_json(paths['states']).get(args.job_id) == audit['job_state'], 'Scheduler job state changed before commit')
                runtime.replace_queue(temporary, paths['queue'])
                committed = True
                audit['committed_at'] = runtime.now(); audit['commit_outcome'] = 'committed'
                h.sync_directory(paths['queue'].parent)
                h.save_audit(paths, audit, 'committed_waiting_ack', runtime)
            return finish_committed(args, paths, runtime, audit, fd)
        except BaseException as error:
            if audit is not None and not committed:
                try:
                    outcome = h.commit_outcome(h.read_json(paths['queue']), audit)
                    committed = outcome != 'not_committed'
                except Exception:
                    committed, outcome = stopped, 'unknown'
                audit['commit_outcome'] = outcome
            if audit is not None:
                if committed:
                    if audit.get('commit_outcome') == 'committed':
                        audit.setdefault('committed_at', runtime.now())
                    h.save_audit(paths, audit, 'needs_cleanup', runtime, error=str(error))
                elif stopped and fd is not None and not runtime.exited(fd):
                    try:
                        h.restore_precommit(paths, runtime, audit, fd)
                    except BaseException as recovery_error:
                        h.save_audit(paths, audit, 'needs_cleanup', runtime, error=str(error), recovery_error=str(recovery_error))
                elif args.apply:
                    h.save_audit(paths, audit, 'aborted_restored', runtime, error=str(error))
            raise
        finally:
            if fd is not None:
                runtime.close_pidfd(fd)
            if temporary.exists():
                temporary.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', required=True, choices=sorted(JOBS))
    parser.add_argument('--output-audit', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--review-sha256')
    args = parser.parse_args(argv)

    def interrupted(signum, frame):
        raise InterruptedError(f'Interrupted by signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    result = operate(args)
    print(json.dumps(dict(stage=result['stage'], old_job_id=result['old_job_id'], new_job_id=result['new_job']['id'],
                          output_audit=args.output_audit), ensure_ascii=False))


if __name__ == '__main__':
    main()

"""CPU broker for persistent scheduler-owned GPU sessions.

This broker never allocates a GPU, writes scheduler job state, or sends signals.
It moves explicitly recognized pending scientific queue rows into an append-only
inbox. The original frozen supervisor advances from the original validated case
markers; the original scheduler owns the two long-lived session allocations.
Queue publication and task activation are separate, durable steps so an inbox
file cannot authorize execution before its queue row has been removed.
"""
import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

PROJECT = Path('/root/nfs/embodied/benchmark')
EXPERIMENT = PROJECT/'results/ease_insights_20261103'
RUNNER = 'preparations.ease_insights_20261103.vlm_local_search_v1.runner'
GATE = 'preparations.ease_insights_20261103.await_assigned_gpu'
SESSION = 'preparations.ease_insights_20261103.gpu_resource_session'
MODEL_PYTHON = PROJECT/'policies/realtime-vla-flash/.venv/bin/python'
PLAN_SCHEMA = 'ease-gpu-resource-campaign-v1'
TASK_SCHEMA = 'ease-gpu-resource-task-v1'
ACTIVATION_SCHEMA = 'ease-gpu-resource-activation-v1'
RESOURCE_ENV = {'EASE_GPU_GATE_IDLE_SECONDS', 'EASE_GPU_GATE_POLL_SECONDS'}
SOURCE_NAMES = [
    'preparations/ease_insights_20261103/gpu_owned_keeper.py',
    'preparations/ease_insights_20261103/gpu_resource_session.py',
    'preparations/ease_insights_20261103/gpu_resource_campaign.py',
    'preparations/ease_insights_20261103/await_assigned_gpu.py',
    'preparations/ease_insights_20261103/reassign_waiting_gate_v2.py',
    'preparations/ease_insights_20261103/reassign_waiting_gate.py',
    'preparations/ease_insights_20261103/scheduler.py',
    'preparations/ease_insights_20261103/vlm_local_search_v1/runner.py',
]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n').encode()


def read(path):
    return json.loads(Path(path).read_bytes())


def absolute(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path, 'Noncanonical path')
    return path


def durable(path, value, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = encode(value)
    if exclusive:
        with path.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    else:
        temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
        with temporary.open('wb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return raw


def append_identical(path, value):
    raw = encode(value)
    if Path(path).exists():
        require(Path(path).read_bytes() == raw, 'Append-only artifact differs: '+str(path))
    else:
        durable(path, value, exclusive=True)
    return digest(raw)


def append_raw_identical(path, raw):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == raw, 'Append-only raw archive differs: '+str(path))
    else:
        with path.open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return digest(raw)


def unwrapped(job):
    argv = job.get('argv', [])
    require(isinstance(argv, list) and all(isinstance(x, str) for x in argv), 'Invalid argv')
    if GATE in argv:
        require(len(argv) > 8 and argv[0] in ('/usr/bin/python3', str(MODEL_PYTHON))
                and argv[1:3] == ['-m', GATE] and argv[3:5] == ['--gpu-uuid', '{gpu_uuid}']
                and argv[5] == '--status-file' and argv[7] == '--', 'Unknown resource wrapper')
        argv = argv[8:]
        require(job.get('launch_gate', {}).get('original_argv') == argv, 'Original argv evidence differs')
    require(argv[:3] == [str(MODEL_PYTHON), '-m', RUNNER], 'Only frozen local-search runner tasks are admitted')
    return list(argv)


def options(argv):
    flags = {'--output', '--condition', '--gpu-uuid', '--port', '--stage', '--shard-index', '--shards'}
    result = {}; i = 3
    while i < len(argv):
        name = argv[i]
        if name == '--launch':
            require(name not in result, 'Duplicate launch flag'); result[name] = True; i += 1
        else:
            require(name in flags and name not in result and i+1 < len(argv), 'Unknown/duplicate runner argument')
            result[name] = argv[i+1]; i += 2
    require(result.get('--launch') is True and result.get('--gpu-uuid') == '{gpu_uuid}'
            and result.get('--port') == '{port}', 'Task must retain assigned UUID/port placeholders')
    require('--output' in result and '--condition' in result, 'Missing science arguments')
    return result


def scientific_manifest(output):
    # Lazy CPU-only imports: never create a model or CUDA context in the broker.
    from .vlm_local_search_v1.common import read_manifest, validate_output, stage_for
    from .vlm_local_search_v1.selection import check_scope
    manifest = read_manifest(Path(output)/'manifest.json', verify_sources=True)
    validate_output(Path(output), manifest)
    stage = stage_for(manifest)
    check_scope(Path(output), manifest, stage)
    return manifest, stage


def task_for(job, plan, transaction, manifest_reader=scientific_manifest):
    argv = unwrapped(job); args = options(argv); output = absolute(args['--output'])
    require(str(output) in plan['allowed_outputs'], 'Scientific output is not approved')
    require(job.get('env') is None or set(job['env']).issubset(RESOURCE_ENV), 'Nonempty scientific job env needs explicit support')
    manifest, stage = manifest_reader(output)
    require(job.get('scientific_phase') == manifest['phase'], 'Queue/manifest scientific phase differs')
    require(args.get('--stage', stage) == stage, 'Stage differs from manifest')
    condition = args['--condition']; index = int(args.get('--shard-index', 0)); shards = int(args.get('--shards', 1))
    require(shards == 1 and index == 0, 'Campaign admits only the frozen all-case shard')
    require(condition in manifest['planned_conditions'], 'Condition not in frozen plan')
    require(job.get('manifest_hash') == manifest['manifest_hash'], 'Queue/manifest identity differs')
    marker = output/condition/f'{stage}_shard{index:02d}of{shards:02d}_complete.json'
    require(absolute(job['complete_file']) == marker, 'Completion path differs from science')
    task_id = 'task_'+digest(encode({'logical_job_id': job['id'], 'argv': argv, 'transaction': transaction,
                                   'manifest_hash': manifest['manifest_hash']}))[:24]
    return dict(schema=TASK_SCHEMA, id=task_id, logical_job_id=job['id'], argv=argv,
        complete_file=str(marker), manifest_hash=manifest['manifest_hash'], output=str(output),
        condition=condition, stage=stage, shard_index=index, shards=shards, env={},
        activation_file=str(Path(plan['campaign_root'])/'transactions'/transaction/'committed.json'))


def slot_jobs(plan):
    root = Path(plan['campaign_root']); jobs = []
    for slot in range(2):
        job_id = f'ease_gpu_resource_session_v1_{slot:02d}'
        output = root/'sessions'/f'slot{slot:02d}'
        inner = [str(MODEL_PYTHON), '-B', '-u', '-m', SESSION, '--gpu-uuid', '{gpu_uuid}',
                 '--port', '{port}', '--plan', str(root/'plan.json'), '--output', str(output), '--launch']
        jobs.append(dict(id=job_id, argv=['/usr/bin/python3', '-m', GATE, '--gpu-uuid', '{gpu_uuid}',
            '--status-file', str(EXPERIMENT/'scheduler/launch_gates'/(job_id+'.json')), '--', *inner],
            env={'EASE_GPU_GATE_IDLE_SECONDS': '25', 'EASE_GPU_GATE_POLL_SECONDS': '2'},
            launch_gate=dict(continuous_idle_seconds=25, poll_seconds=2, assigned_gpu_only=True, original_argv=inner),
            complete_file=str(output/'complete.json'), depends_on=[],
            ready_files=[str(root/'plan.json'), str(root/'deployed.json')],
            resource_execution_revision=PLAN_SCHEMA, resource_campaign=str(root)))
    return jobs


def known_task(job):
    return isinstance(job.get('argv'), list) and RUNNER in job['argv']


def collect(queue, states, plan, transaction, manifest_reader=scientific_manifest):
    selected = []; tasks = []
    ids = [j['id'] for j in queue['jobs']]
    require(len(ids) == len(set(ids)), 'Duplicate queue IDs')
    for job in queue['jobs']:
        if not known_task(job) or states.get(job['id'], {}).get('status', 'pending') == 'complete':
            continue
        require(states.get(job['id'], {}).get('status', 'pending') == 'pending',
                'A logical science task is running/failed; preserve it for review')
        if not all(states.get(dep, {}).get('status') == 'complete' for dep in job.get('depends_on', [])):
            continue
        if not all(Path(p).is_file() for p in job.get('ready_files', [])):
            continue
        tasks.append(task_for(job, plan, transaction, manifest_reader)); selected.append(copy.deepcopy(job))
    selected_ids = {j['id'] for j in selected}
    require(not any(set(j.get('depends_on', [])) & selected_ids for j in queue['jobs'] if j['id'] not in selected_ids),
            'Unhandled downstream dependency on a moved logical job')
    proposed = copy.deepcopy(queue)
    proposed['jobs'] = [j for j in proposed['jobs'] if j['id'] not in selected_ids]
    return selected, tasks, proposed


def verify_plan(plan, path):
    require(plan.get('schema') == PLAN_SCHEMA and absolute(plan['campaign_root']) == absolute(path).parent,
            'Plan root/schema binding differs')
    expected = [str(EXPERIMENT/'vlm_local_search_v1'/s) for s in ('discovery', 'confirmation')]
    require(plan.get('allowed_outputs') == expected and plan.get('hold_until_explicit_release') is True,
            'Approved science roots/hold policy differs')
    for name, expected_hash in plan['source_hashes'].items():
        source = absolute(PROJECT/name)
        require(source.is_relative_to(PROJECT) and digest(source.read_bytes()) == expected_hash,
                'Resource/scientific source pin changed: '+name)


def prepare(root, queue_path=EXPERIMENT/'queue.json', state_path=EXPERIMENT/'scheduler/jobs_state.json',
            manifest_reader=None, waiting_gate_proofs=None):
    root = absolute(root); require(root.parent == EXPERIMENT, 'Campaign must be a direct experiment child')
    root.mkdir(exist_ok=False)
    for directory in ('tasks', 'claims', 'outcomes', 'transactions', 'sessions'):
        (root/directory).mkdir()
    plan = dict(schema=PLAN_SCHEMA, campaign_root=str(root), task_schema=TASK_SCHEMA,
        allowed_outputs=[str(EXPERIMENT/'vlm_local_search_v1'/s) for s in ('discovery', 'confirmation')],
        source_hashes={name: digest((PROJECT/name).read_bytes()) for name in SOURCE_NAMES},
        hold_until_explicit_release=True, keeper_max_memory_mib=512, keeper_allocation_mib=64,
        operation='Original runner resource orchestration is replaced; original frozen server/worker/science checks and case markers are retained.',
        scope='Two persistent allocations for the active user goal; empty inbox, CPU preparation and task attention do not release them.')
    append_identical(root/'plan.json', plan)
    before = Path(queue_path).read_bytes(); queue = json.loads(before); states = read(state_path)
    # Draft planning may include an exactly identified *CPU-only* gate. The
    # installer must stop and revalidate that identity before queue publication;
    # this copy never changes real scheduler state or authorizes a model signal.
    original_states = copy.deepcopy(states)
    for key, proof in (waiting_gate_proofs or {}).items():
        require(states.get(key, {}).get('status') == 'running'
                and states[key].get('pid') == proof.get('pid') and proof.get('module') == GATE
                and proof.get('start_ticks') and proof.get('children') == [], 'Invalid waiting-gate planning proof')
        states[key] = dict(status='pending')
    selected, tasks, proposed = collect(queue, states, plan, 'initial', manifest_reader or scientific_manifest)
    require(len(selected) == 5, 'Initial campaign must map exactly the five unfinished discovery conditions')
    require({t['condition'] for t in tasks} == {'full_d10', 'local_spatial', 'local_object', 'local_goal', 'local_long'},
            'Initial scientific scope differs')
    slots = slot_jobs(plan); require(not {j['id'] for j in slots} & {j['id'] for j in proposed['jobs']}, 'Slots already in queue')
    require(all(original_states.get(j['id'], {}).get('status') == 'complete' for j in proposed['jobs']),
            'Other unfinished queue work must be explicitly included or preserved before installing persistent allocations')
    proposed['jobs'] = slots + proposed['jobs']
    task_hashes = {t['id']: append_identical(root/'tasks'/(t['id']+'.json'), t) for t in tasks}
    (root/'queue.before.json').write_bytes(before); durable(root/'queue.proposed.json', proposed)
    proposal = dict(schema='ease-gpu-campaign-install-v1', prepared_at=time.time(), plan_sha256=digest((root/'plan.json').read_bytes()),
        queue_before_sha256=digest(before), queue_proposed_sha256=digest((root/'queue.proposed.json').read_bytes()),
        logical_jobs=selected, task_sha256=task_hashes, slot_job_ids=[j['id'] for j in slots],
        waiting_gate_proofs=waiting_gate_proofs or {},
        logical_states_before={j['id']: original_states.get(j['id'], dict(status='pending')) for j in selected},
        activation_file=str(root/'transactions/initial/committed.json'),
        deployment='Root-reviewed quiescent original-scheduler transaction required; this command does not deploy or signal.')
    durable(root/'installation.json', proposal, exclusive=True)
    return proposal


def slot_capacity_proof(queue, states, slot_ids, candidate_ids):
    """Every eligible cold allocation precedes science, or both slots are live."""
    positions = {j['id']: i for i, j in enumerate(queue['jobs'])}
    require(set(slot_ids).issubset(positions), 'Persistent allocation row disappeared')
    statuses = [states.get(key, {}).get('status', 'pending') for key in slot_ids]
    require(all(s in ('pending', 'running') for s in statuses), 'Allocation failed/exited; dispatch requires review')
    require(all(positions[slot] < positions[task] for slot in slot_ids for task in candidate_ids),
            'New logical task precedes a cold persistent allocation')
    # Running scheduler jobs keep their slot even while the outer gate waits.
    require(len(slot_ids) == 2, 'Original allocator must retain its two-slot limit')


def recover_publications(root, queue, states, current_raw=None, manifest_reader=None):
    """Complete only a proven committed queue transition, never an uncertain one.

    A crash after queue CAS but before activation must not strand scientific work.
    Extra rows may only be appended at the tail; the complete proposed prefix
    and non-job metadata must match, and every moved logical ID must stay absent.
    """
    root = Path(root); recovered = []
    current_raw = encode(queue) if current_raw is None else current_raw
    plan = read(root/'plan.json'); plan_hash = digest((root/'plan.json').read_bytes())
    current = {j['id']: j for j in queue['jobs']}
    require(len(current) == len(queue['jobs']), 'Duplicate current queue IDs')
    for record_path in sorted((root/'transactions').glob('dispatch_*/transaction.json')):
        directory = record_path.parent
        if (directory/'committed.json').exists() or (directory/'superseded_uncommitted.json').exists():
            continue
        record = read(record_path); before = read(directory/'queue.before.json'); after = read(directory/'queue.proposed.json')
        require(record['schema'] == 'ease-gpu-resource-dispatch-v1'
                and record['plan_sha256'] == plan_hash
                and record['queue_before_sha256'] == digest((directory/'queue.before.json').read_bytes())
                and record['queue_proposed_sha256'] == digest((directory/'queue.proposed.json').read_bytes()),
                'Dispatch recovery archive changed')
        logical = {j['id']: j for j in record['logical_jobs']}
        expected = copy.deepcopy(before); expected['jobs'] = [j for j in expected['jobs'] if j['id'] not in logical]
        require(expected == after and all(next((x for x in before['jobs'] if x['id'] == key), None) == row
                for key, row in logical.items()), 'Recovery proposal changed more than its logical rows')
        present = set(logical) & set(current)
        if present == set(logical):
            require(all(current[key] == row for key, row in logical.items()), 'Uncommitted logical row changed')
            if digest(current_raw) != record['queue_before_sha256']:
                append_identical(directory/'superseded_uncommitted.json',
                                 dict(reason='Queue changed before commit; inactive inbox is not execution authority.'))
            continue
        require(not present, 'Ambiguous partial logical-row removal; do not activate')
        require(queue['jobs'][:len(after['jobs'])] == after['jobs']
                and {k: v for k, v in queue.items() if k != 'jobs'}
                == {k: v for k, v in after.items() if k != 'jobs'},
                'Committed queue prefix or metadata no longer matches')
        require(all(states.get(key, {}).get('status', 'pending') == 'pending' for key in logical),
                'Original logical task ran during the uncertain transition')
        seen_logical = set()
        for task_id, expected_hash in record['task_sha256'].items():
            task_path = root/'tasks'/(task_id+'.json')
            task = read(task_path); original = logical.get(task.get('logical_job_id'))
            require(original is not None and task['logical_job_id'] not in seen_logical,
                    'Recovery tasks do not map one-to-one to logical jobs')
            seen_logical.add(task['logical_job_id'])
            expected_task = task_for(original, plan, directory.name, manifest_reader or scientific_manifest)
            require(digest(task_path.read_bytes()) == expected_hash and original is not None
                    and task == expected_task and task['id'] == task_id,
                    'Recovery task/activation binding changed')
        require(seen_logical == set(logical), 'Recovery task/logical-job mapping is incomplete')
        append_identical(directory/'committed.json', dict(schema=ACTIVATION_SCHEMA,
            task_sha256=record['task_sha256'], plan_sha256=record['plan_sha256'],
            queue_after_sha256=record['queue_proposed_sha256']))
        recovered.append(directory.name)
    return recovered


def publish(root, queue_path=EXPERIMENT/'queue.json', state_path=EXPERIMENT/'scheduler/jobs_state.json', manifest_reader=None):
    root = absolute(root); plan_path = root/'plan.json'; plan = read(plan_path); verify_plan(plan, plan_path)
    deployment = read(root/'deployed.json')
    require(deployment['plan_sha256'] == digest(plan_path.read_bytes()), 'Deployment/plan identity differs')
    if (root/'release').exists():
        return dict(phase='release_requested', moved=0)
    lock_path = Path(queue_path).with_name('queue.json.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = Path(queue_path).read_bytes(); queue = json.loads(before); states = read(state_path)
        transaction = 'dispatch_'+digest(before)[:24]
        recovered = recover_publications(root, queue, states, before, manifest_reader)
        selected, tasks, proposed = collect(queue, states, plan, transaction, manifest_reader or scientific_manifest)
        if not selected:
            return dict(phase='waiting_tasks', moved=0, recovered=recovered)
        slot_capacity_proof(queue, states, deployment['slot_job_ids'], [j['id'] for j in selected])
        directory = root/'transactions'/transaction; directory.mkdir(exist_ok=True)
        hashes = {t['id']: append_identical(root/'tasks'/(t['id']+'.json'), t) for t in tasks}
        append_raw_identical(directory/'queue.before.json', before)
        append_identical(directory/'queue.proposed.json', proposed)
        transaction_record = dict(schema='ease-gpu-resource-dispatch-v1', logical_jobs=selected,
            task_sha256=hashes, plan_sha256=digest(plan_path.read_bytes()), queue_before_sha256=digest(before),
            queue_proposed_sha256=digest(encode(proposed)))
        append_identical(directory/'transaction.json', transaction_record)
        require(Path(queue_path).read_bytes() == before, 'Queue CAS changed during publication')
        # Scheduler never starts a logical row behind the two persistent slots.
        fresh = read(state_path); slot_capacity_proof(queue, fresh, deployment['slot_job_ids'], [j['id'] for j in selected])
        require(all(fresh.get(j['id'], {}).get('status', 'pending') == 'pending' for j in selected),
                'Logical task began during dispatch; inactive inbox preserved, no duplicate execution')
        durable(queue_path, proposed)
        after = read(state_path)
        require(all(after.get(j['id'], {}).get('status', 'pending') == 'pending' for j in selected),
                'Post-CAS logical task race; do not activate inbox, inspect original task')
        append_identical(directory/'committed.json', dict(schema=ACTIVATION_SCHEMA, task_sha256=hashes,
            plan_sha256=digest(plan_path.read_bytes()), queue_after_sha256=digest(encode(proposed))))
        return dict(phase='dispatched', moved=len(tasks), transaction=transaction)


def launch_broker(root):
    root = absolute(root); plan = read(root/'plan.json'); verify_plan(plan, root/'plan.json')
    with (root/'broker.lock').open('a') as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = Path('/proc/self/stat').read_text(); fields = raw[raw.rfind(')')+2:].split()
        durable(root/'broker_process.json', dict(pid=os.getpid(), start_ticks=fields[19], started=time.time(),
            source_sha256=digest(Path(__file__).read_bytes()), plan_sha256=digest((root/'plan.json').read_bytes())))
        while True:
            try:
                result = publish(root)
                durable(root/'broker_status.json', dict(time=time.time(), **result))
                if result['phase'] == 'release_requested':
                    return
            except Exception as error:
                # Preserve inactive descriptors, current queue and all GPU holders.
                durable(root/'broker_status.json', dict(time=time.time(), phase='attention', error=str(error)))
                return
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--campaign', required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--prepare', action='store_true'); actions.add_argument('--launch-broker', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.campaign), indent=2))
    else:
        launch_broker(args.campaign)


if __name__ == '__main__':
    main()

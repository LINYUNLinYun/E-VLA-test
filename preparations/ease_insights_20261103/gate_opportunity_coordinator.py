"""Inspect by default; --launch coordinates the pinned v2 helper, never GPUs.

Fixed experiment only. Root must review and launch durably (setsid/nohup with
stdin=DEVNULL). A stop file stops future attempts, after pending cleanup. The
single scheduler remains the only allocator. needs_attention requires review;
restarting does not clear it. This module sends no process signals.
"""
import argparse
import copy
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from preparations.ease_insights_20261103 import reassign_waiting_gate_v2 as helper

PROJECT = Path('/root/nfs/embodied/benchmark')
ROOT = PROJECT / 'results/ease_insights_20261103'
HELPER = PROJECT / 'preparations/ease_insights_20261103/reassign_waiting_gate_v2.py'
HELPER_SHA256 = '0238d949fd5585fc52dae19fc56c54b5b970cb4542714c4599024418c8169498'
SCHEMA = 'ease-gate-opportunity-coordinator-v1'
POLL_SECONDS = 10.0
COOLDOWN_SECONDS = 45.0
ADOPTION_MAX_AGE = 75.0
SYSTEM_PYTHON = Path('/usr/bin/python3')
_SYSTEM_STAT = SYSTEM_PYTHON.stat()
SYSTEM_EXECUTABLE = dict(path=str(SYSTEM_PYTHON.resolve()), device=_SYSTEM_STAT.st_dev, inode=_SYSTEM_STAT.st_ino)
TERMINAL_AUDITS = ('complete', 'aborted_restored')
CLEANUP_STAGES = ('stop_intent', 'stopped_verified', 'commit_intent', 'committed_waiting_ack',
                  'controllers_acknowledged', 'terminating_cpu_gate', 'needs_cleanup', 'precommit_restore_intent')
RESOURCE_REFUSALS = frozenset((
    'Candidate GPU is busy or lacks complete idle evidence',
    'Candidate project GPU lease is busy',
    'Source GPU is already idle or lacks positive busy evidence; continue its existing gate',
    'GPU observations expired during preparation; retry later',
    'Gate must have a fresh busy receipt with idle_seconds=0',
    'Gate receipt changed during read; retry later',
    'Completion target already exists; no migration allowed',
))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


class SystemRuntime(helper.LocalRuntime):
    def executable(self, pid):
        path = Path('/proc')/str(pid)/'exe'
        target, stat = os.readlink(path), path.stat()
        return dict(path=target, device=stat.st_dev, inode=stat.st_ino)


class SchedulerIdentityRuntime:
    """A proven helper view of one scheduler, retaining its raw exec identity."""
    def __init__(self, runtime, scheduler_pid, expected=None):
        self.runtime, self.scheduler_pid = runtime, scheduler_pid
        self.expected = copy.deepcopy(expected['_scheduler_proof']) if expected is not None else None
        if self.expected is not None:
            require(self.expected['pid'] == scheduler_pid, 'Bound scheduler proof PID differs')

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def identity(self, pid):
        before = self.runtime.identity(pid)
        if pid != self.scheduler_pid:
            # A helper receipt change must not silently bind a second scheduler.
            argv = before.get('argv', [])
            candidate = [str(SYSTEM_PYTHON), *argv[1:]] if argv and argv[0] == 'python3' else argv
            require(not helper.exact_module(candidate, helper.SCHEDULER), 'Helper encountered an unbound scheduler PID')
            return before
        if before.get('state') in helper.DEAD_STATES:
            return before
        raw_argv = before.get('argv', [])
        viewed_argv = [str(SYSTEM_PYTHON), *raw_argv[1:]] if raw_argv and raw_argv[0] == 'python3' else raw_argv
        require(helper.exact_module(viewed_argv, helper.SCHEDULER), 'Unsupported raw scheduler invocation')
        executable = self.runtime.executable(pid)
        after = self.runtime.identity(pid)
        executable_after = self.runtime.executable(pid)
        require(all(before.get(key) == after.get(key) for key in ('pid', 'start_ticks', 'argv', 'uid'))
                and before.get('state') in ('S', 'R', 'D', 'I', 'W', 'P')
                and after.get('state') in ('S', 'R', 'D', 'I', 'W', 'P') and after.get('uid') == os.getuid(),
                'Scheduler raw identity changed during executable proof')
        require(executable == executable_after == SYSTEM_EXECUTABLE, 'Scheduler executable is not the fixed system Python')
        proof = dict(pid=pid, start_ticks=after['start_ticks'], raw_argv=copy.deepcopy(raw_argv),
                     uid=after['uid'], executable=copy.deepcopy(executable))
        require(self.expected is None or self.expected == proof, 'Bound scheduler raw argv/start/executable identity changed')
        return dict(after, argv=list(viewed_argv), _scheduler_proof=proof)


def same_controller(actual, expected):
    return helper.same_process(actual, expected) and actual.get('_scheduler_proof') == expected.get('_scheduler_proof')


class WaitingAdoption(RuntimeError):
    """A verified former scheduler exited; wait for supervisor adoption."""


class ReviewBoundRuntime:
    """Delegate all operations, guarding only the helper's final queue CAS.

    The reviewed bytes come from coordinator state, never the helper's mutable
    audit/proposal files. Rejection is handled by v2's own STOP restoration.
    """
    def __init__(self, runtime, queue, reviewed):
        require(isinstance(reviewed, dict) and reviewed.get('helper_sha256') == HELPER_SHA256,
                'Missing immutable reviewed proposal')
        self.runtime, self.queue = runtime, Path(queue)
        self.before = reviewed['before'].encode('utf-8')
        self.proposed = reviewed['proposed'].encode('utf-8')
        require(sha(self.before) == reviewed['before_sha256'] and sha(self.proposed) == reviewed['proposed_sha256'],
                'Immutable reviewed proposal hashes differ')

    def __getattr__(self, name):
        return getattr(self.runtime, name)

    def replace_queue(self, temporary, queue):
        require(Path(queue) == self.queue and Path(queue).resolve() == self.queue,
                'Commit queue differs from reviewed queue')
        require(Path(temporary).read_bytes() == self.proposed and self.queue.read_bytes() == self.before,
                'Final queue commit differs from immutable reviewed proposal')
        self.runtime.replace_queue(temporary, queue)


class Backend:
    """Real IO boundary; tests inject a temporary filesystem and fake runtime."""
    def __init__(self, root=ROOT, runtime=None):
        self.root = Path(root)
        self.output = self.root / 'gate_opportunity_coordinator'
        self.scheduler = self.root / 'scheduler'
        self.supervisor = self.root / 'vlm_local_search_v1/supervisor'
        self.queue = self.root / 'queue.json'
        self.states = self.scheduler / 'jobs_state.json'
        self.runtime = runtime or SystemRuntime()

    def now(self):
        return self.runtime.now()

    def read(self, path):
        return json.loads(Path(path).read_bytes())

    def write(self, path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        helper.durable_json(path, value)

    def verify_helper(self):
        require(self.root.resolve() == ROOT and ROOT.resolve() == ROOT and PROJECT.resolve() == PROJECT,
                'Noncanonical experiment path')
        require(Path(__file__).resolve() == PROJECT / 'preparations/ease_insights_20261103/gate_opportunity_coordinator.py',
                'Noncanonical coordinator source')
        require(Path(helper.__file__).resolve() == HELPER and Path(helper.operate.__code__.co_filename).resolve() == HELPER,
                'Unexpected helper operation source')
        require(sha(HELPER.read_bytes()) == HELPER_SHA256, 'Audited helper hash changed')
        for path in (self.output, self.scheduler, self.supervisor, self.queue, self.states, self.root/'gate_reassignments'):
            require(path.resolve() == path, 'Noncanonical experiment artifact path')

    def gpus(self):
        # Two all-device queries; timestamp starts before both, never after them.
        checked_at = self.now()
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used',
            '--format=csv,noheader,nounits'], text=True, timeout=15)
        apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'], text=True, timeout=15)
        result = {}
        for row in rows.splitlines():
            if not row.strip():
                continue
            uuid, raw = [x.strip() for x in row.split(',')]
            helper.assigned_uuid(uuid)
            memory = float(raw)
            require(uuid not in result and math.isfinite(memory) and memory >= 0, 'Invalid all-GPU inventory')
            result[uuid] = dict(gpu_uuid=uuid, memory_mib=memory, compute_process_count=0, checked_at=checked_at)
        require(result, 'Empty all-GPU inventory')
        for row in apps.splitlines():
            if not row.strip():
                continue
            uuid, pid = [x.strip() for x in row.split(',')]
            require(uuid in result and int(pid) > 0, 'Unknown all-GPU compute process')
            result[uuid]['compute_process_count'] += 1
        for item in result.values():
            item['idle'] = item['compute_process_count'] == 0 and item['memory_mib'] < 500
        return result

    def args(self, pending, apply=False):
        return argparse.Namespace(queue=str(self.queue), jobs_state=str(self.states), scheduler_output=str(self.scheduler),
            job_id=pending['job_id'], candidate_gpu_uuid=pending['candidate_gpu_uuid'],
            output_audit=pending['audit'], apply=apply)

    def controllers(self, previous=None):
        profile = dict(kind='local_search', module=helper.LOCAL_SUPERVISOR, output=str(self.supervisor),
            root=str(self.root/'vlm_local_search_v1/discovery'), confirmation_root=str(self.root/'vlm_local_search_v1/confirmation'))
        supervisor = helper.controller(self.runtime, self.read(self.supervisor/'supervisor_process.json'),
            helper.LOCAL_SUPERVISOR, self.queue, self.supervisor, profile)
        require(supervisor.get('state') in ('S', 'R', 'D', 'I', 'W', 'P'), 'Supervisor OS state is uncertain')
        if previous:
            require(helper.same_process(supervisor, previous['supervisor']), 'Original supervisor identity changed')
        old = previous['scheduler'] if previous else None
        try:
            receipt = self.read(self.scheduler/'controller_process.json')
        except FileNotFoundError:
            self.verify_idle_gap(old)
            raise WaitingAdoption('Exited scheduler has no replacement receipt yet')
        current = self.runtime.identity(receipt['pid'])
        hint = helper.option(supervisor['argv'], '--scheduler-pid')
        if current.get('state') in helper.DEAD_STATES:
            require(old is not None and receipt['pid'] == old['pid'], 'Unbound scheduler receipt is dead')
            self.verify_idle_gap(old)
            raise WaitingAdoption('Original scheduler exited; waiting for its supervisor to restart/adopt')
        require(current.get('state') in ('S', 'R', 'D', 'I', 'W', 'P'), 'Scheduler OS state is uncertain')
        pinned = old if old is not None and old['pid'] == receipt['pid'] else None
        controller_runtime = SchedulerIdentityRuntime(self.runtime, receipt['pid'], pinned)
        scheduler = helper.controller(controller_runtime, receipt, helper.SCHEDULER, self.queue, self.scheduler)
        require(float(helper.option(scheduler['argv'], '--idle-seconds', '25')) == 0,
                'Scheduler idle-seconds differs from fixed allocator contract')
        changed = old is not None and not same_controller(scheduler, old)
        hint_differs = hint is not None and int(hint) != scheduler['pid']
        if changed or hint_differs:
            former_pids = {int(hint)} if hint_differs else set()
            if changed:
                former_pids.add(old['pid'])
            for pid in former_pids:
                require(self.runtime.identity(pid).get('state') in helper.DEAD_STATES,
                        'Former scheduler is live or uncertain; no controller adoption')
            self.verify_adoption(scheduler, receipt)
        return dict(scheduler=scheduler, supervisor=supervisor)

    def verify_idle_gap(self, old):
        require(old is not None and self.runtime.identity(old['pid']).get('state') in helper.DEAD_STATES,
                'Missing scheduler receipt without confirmed former scheduler exit')
        for state in self.read(self.states).values():
            if state.get('status') == 'running':
                require(self.runtime.identity(state['pid']).get('state') in helper.DEAD_STATES,
                        'Scheduler exited while a recorded GPU job is live or uncertain')

    def verify_adoption(self, scheduler, receipt):
        state, status = self.read(self.supervisor/'state.json'), self.read(self.supervisor/'status.json')
        observations = (state.get('scheduler_observation', {}), status.get('scheduler', {}))
        for observation in observations:
            require(observation.get('status') != 'needs_attention', 'Supervisor reported uncertain scheduler adoption')
            for process in observation.get('processes', []):
                require(process.get('status') != 'unknown', 'Supervisor scheduler observation is unknown')
                pid = process.get('pid')
                if pid and pid != scheduler['pid']:
                    require(self.runtime.identity(pid).get('state') in helper.DEAD_STATES,
                            'Multiple live or uncertain scheduler identities during adoption')
        reports = (state.get('scheduler', {}), *(observation.get('process', {}) for observation in observations))
        raw = dict(pid=scheduler['pid'], start_ticks=scheduler['start_ticks'], argv=scheduler['_scheduler_proof']['raw_argv'])
        if not all(report.get('status') == 'live' and all(report.get(key) == raw[key]
                   for key in ('pid', 'start_ticks', 'argv')) for report in reports):
            raise WaitingAdoption('Supervisor has not yet adopted the new scheduler identity in state and status')
        timestamps = (state.get('updated_unix'), state.get('scheduler', {}).get('last_verified_unix'), status.get('time'))
        if not all(isinstance(value, (int, float)) and math.isfinite(value)
                   and receipt['started']-1 <= value <= self.now()+1 and self.now()-value <= ADOPTION_MAX_AGE
                   for value in timestamps):
            raise WaitingAdoption('Waiting for fresh supervisor scheduler adoption evidence')
        if state.get('phase') != status.get('phase') or status.get('terminal') is not False:
            raise WaitingAdoption('Supervisor state/status transition has not settled')

    def inspect_job(self, job_id, candidate, queue, states, controllers):
        pending = dict(job_id=job_id, candidate_gpu_uuid=candidate, audit=str(self.root/'gate_reassignments/unused'))
        paths = dict(root=self.root, queue=self.queue, states=self.states, scheduler=self.scheduler)
        runtime = SchedulerIdentityRuntime(self.runtime, controllers['scheduler']['pid'], controllers['scheduler'])
        inspected = helper.inspect_job(self.args(pending), paths, runtime, queue, states)
        require(same_controller(inspected['scheduler'], controllers['scheduler'])
                and helper.same_process(inspected['supervisor'], controllers['supervisor']), 'Controllers changed before transaction')
        return inspected

    def invoke(self, pending, apply):
        self.verify_helper()
        runtime = SchedulerIdentityRuntime(self.runtime, pending['scheduler']['pid'], pending['scheduler'])
        if apply:
            runtime = ReviewBoundRuntime(runtime, self.queue, pending.get('reviewed'))
        return helper.operate(self.args(pending, apply), runtime)


def local_gate(job):
    outer = helper.module_invocation(job.get('argv'))
    if outer is None or outer['module'] != helper.GATE:
        return False
    args = outer['args']
    require('--' in args, 'Running gate has no inner-command separator')
    inner = args[args.index('--')+1:]
    # Syntax-only classification may exclude a relative-Python legacy command;
    # this never authorizes it. Local commands still face the strict raw parser.
    viewed = [str(SYSTEM_PYTHON), *inner[1:]] if inner and inner[0] == 'python3' else inner
    invocation = helper.module_invocation(viewed)
    if invocation is not None:
        if invocation['module'] != helper.LOCAL_RUNNER:
            return False
        helper.gate_parts(job['argv'])
        return True
    for index, value in enumerate(viewed[1:], 1):
        if not value.startswith('-') and Path(value).suffix == '.py' and helper.module_invocation(
                [*viewed[:index], '-m', '_legacy_script_shape']) is not None:
            return False
    raise ValueError('Unrecognized inner invocation in running gate')


def inventory(backend):
    queue, states, gpus = backend.read(backend.queue), backend.read(backend.states), backend.gpus()
    require(isinstance(queue.get('jobs'), list) and isinstance(states, dict), 'Malformed queue/state')
    require(all(0 <= backend.now()-g['checked_at'] <= helper.GPU_PROOF_MAX_AGE for g in gpus.values()),
            'Stale all-GPU inventory')
    assigned = {s.get('gpu_uuid') for s in states.values() if s.get('status') == 'running'}
    candidates = sorted(uuid for uuid, g in gpus.items() if uuid not in assigned and g['idle'] is True
                        and g['compute_process_count'] == 0 and 0 <= g['memory_mib'] < 500)
    gates = []
    for job in queue['jobs']:
        state = states.get(job['id'], {})
        if state.get('status') != 'running':
            continue
        if not local_gate(job):
            continue
        process = backend.runtime.identity(state['pid'])
        # A gate can exec its runner between observations; it is never a target.
        if not helper.exact_module(process.get('argv'), helper.GATE):
            continue
        require(process.get('argv') == state.get('argv') and process.get('uid') == os.getuid()
                and backend.runtime.matches_started(process, state['started']), 'Owned gate identity differs')
        source = gpus.get(state.get('gpu_uuid'), {})
        if process.get('state') not in (*helper.DEAD_STATES, 'T', 't') and source.get('idle') is False:
            gates.append((state['started'], job['id']))
    return queue, states, candidates, sorted(gates)


def scheduler_has_priority(backend, queue, states):
    """Defer if the allocator has a spare slot and another ready queue job."""
    live = 0
    for job in queue['jobs']:
        state = states.get(job['id'], {})
        if state.get('status') != 'running':
            continue
        process = backend.runtime.identity(state['pid'])
        if process.get('state') in helper.DEAD_STATES:
            continue
        require(process.get('state') in ('S', 'R', 'D', 'I', 'T', 't', 'W', 'P')
                and backend.runtime.matches_started(process, state['started']), 'Running slot process identity is uncertain')
        ticks = state.get('start_ticks', state.get('proc_start_ticks'))
        require(ticks is None or str(ticks) == process.get('start_ticks'), 'Running slot PID start identity differs')
        live += 1
    ready = any(states.get(job['id'], {}).get('status', 'pending') == 'pending'
                and all(states.get(dep, {}).get('status') == 'complete' for dep in job.get('depends_on', []))
                and all(Path(path).is_file() for path in job.get('ready_files', [])) for job in queue['jobs'])
    return live < 2 and ready


def audit_may_need_cleanup(audit):
    return audit is not None and (audit.get('committed_at') is not None or audit.get('commit_outcome') in ('unknown', 'committed')
        or audit.get('stage') in CLEANUP_STAGES or any(e.get('stage') in CLEANUP_STAGES for e in audit.get('events', [])))


def verify_plan(backend, pending, audit):
    """Independent full-value proposal check, without helper.make_proposal()."""
    directory = Path(pending['audit'])
    before_raw, proposed_raw = (directory/'queue.before.json').read_bytes(), (directory/'queue.proposed.json').read_bytes()
    before, proposed = json.loads(before_raw), json.loads(proposed_raw)
    reviewed = dict(before=before_raw.decode('utf-8'), proposed=proposed_raw.decode('utf-8'),
        before_sha256=sha(before_raw), proposed_sha256=sha(proposed_raw), helper_sha256=HELPER_SHA256)
    require(pending.get('reviewed', reviewed) == reviewed, 'Helper proposal differs from immutable reviewed snapshot')
    require(audit.get('schema') == helper.SCHEMA and audit.get('request') == {
        'queue': str(backend.queue), 'jobs_state': str(backend.states), 'scheduler_output': str(backend.scheduler),
        'job_id': pending['job_id'], 'candidate_gpu_uuid': pending['candidate_gpu_uuid']}, 'Prepared request differs')
    require(sha(before_raw) == audit.get('queue_before_sha256') and sha(proposed_raw) == audit.get('queue_after_sha256'),
            'Prepared queue hashes differ')
    old, new = audit['original_job'], audit['new_job']
    require(old == pending['original_job'] and old['id'] == pending['job_id'] and local_gate(old), 'Prepared scientific job differs')
    old_uuid, old_receipt, old_inner = helper.gate_parts(old['argv'])
    new_uuid, new_receipt, new_inner = helper.gate_parts(new['argv'])
    require(old_uuid == new_uuid == '{gpu_uuid}' and old_inner == new_inner == old['launch_gate']['original_argv'],
            'Scientific inner argv changed')
    expected = copy.deepcopy(old)
    expected['id'] = new['id']
    require(new['id'].startswith(old['id']+'__wait_reassign_') and new['id'] != old['id'], 'Invalid replacement ID')
    require(old_receipt == backend.scheduler/'launch_gates'/(old['id']+'.json')
            and new_receipt == backend.scheduler/'launch_gates'/(new['id']+'.json'), 'Gate receipt binding changed')
    expected['argv'][helper.module_invocation(expected['argv'])['args_index']+3] = str(new_receipt)
    require(expected == new, 'Completion/config/ready-file/dependency semantics changed')
    require(sum(j == old for j in before['jobs']) == 1 and not any(j['id'] == new['id'] for j in before['jobs']),
            'Original/replacement queue identity differs')
    expected_queue = copy.deepcopy(before)
    expected_queue['jobs'] = [copy.deepcopy(new) if j['id'] == old['id'] else j for j in expected_queue['jobs']]
    for job in expected_queue['jobs']:
        if 'depends_on' in job:
            job['depends_on'] = [new['id'] if x == old['id'] else x for x in job['depends_on']]
    require(proposed == expected_queue, 'Queue proposal changes unrelated scientific semantics')
    profile = audit['profile']
    require(profile['kind'] == 'local_search' and profile['module'] == helper.LOCAL_SUPERVISOR
            and profile['runner_module'] == helper.LOCAL_RUNNER and profile['output'] == str(backend.supervisor)
            and profile['root'] == str(backend.root/'vlm_local_search_v1/discovery')
            and profile['confirmation_root'] == str(backend.root/'vlm_local_search_v1/confirmation'), 'Wrong local supervisor profile')
    for key in ('gate', 'scheduler', 'supervisor'):
        compare = same_controller if key == 'scheduler' else helper.same_process
        require(compare(audit[key], pending[key]), 'Prepared '+key+' identity differs')
    require(audit.get('allocator') == 'existing_scheduler_only' and audit.get('candidate_is_only_an_opportunity') is True,
            'Unexpected GPU allocator')
    return reviewed


class Coordinator:
    def __init__(self, backend, state=None):
        self.b = backend
        self.state = state if state is not None else dict(schema=SCHEMA, pending=None, cooldown_until=0, sequence=0)
        require(self.state.get('schema') == SCHEMA, 'Unrecognized coordinator state')

    def record(self, phase, **details):
        self.state['phase'] = phase
        self.state.update(details)
        self.b.write(self.b.output/'state.json', self.state)
        status = dict(schema=SCHEMA, phase=phase, time=self.b.now(),
            pending_audit=(self.state.get('pending') or {}).get('audit'), **details)
        self.b.write(self.b.output/'status.json', status)
        event = {k: v for k, v in status.items() if k not in ('pending', 'controllers')}
        with (self.b.output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event, ensure_ascii=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
        return phase

    def finish(self, outcome):
        self.state['pending'] = None
        return self.record(outcome, cooldown_until=self.b.now()+COOLDOWN_SECONDS)

    def attention(self, error):
        return self.record('needs_attention', error=str(error))

    def pending_audit(self):
        pending = self.state['pending']
        directory = Path(pending['audit'])
        require(directory.parent == self.b.root/'gate_reassignments' and directory.resolve() == directory,
                'Pending audit escaped fixed transaction root')
        path = directory/'audit.json'
        return self.b.read(path) if path.exists() else None

    def transaction(self):
        pending = self.state['pending']
        try:
            self.b.verify_helper()
            audit = self.pending_audit()
            if audit and audit.get('stage') in TERMINAL_AUDITS:
                verify_plan(self.b, pending, audit)
                return self.finish(audit['stage'])
            if audit_may_need_cleanup(audit):
                verify_plan(self.b, pending, audit)
                require(audit.get('commit_outcome') != 'unknown', 'Unknown commit outcome; preserve audit for review')
                require(pending.get('recovery_attempts', 0) < 1, 'Bounded helper recovery exhausted')
                pending['recovery_attempts'] = 1
                self.record('recovering')  # durable before the only recovery call
                result = self.b.invoke(pending, True)
            else:
                require(audit is not None or pending['phase'] == 'planned', 'Missing audit after prepared/apply intent')
                if (self.b.output/'stop').exists() or self.b.read(self.b.supervisor/'status.json').get('terminal') is True:
                    return self.finish('cancelled_before_apply')
                if audit:
                    require(audit.get('stage') in ('dry_run', 'prepared'), 'Unknown helper audit stage')
                    verify_plan(self.b, pending, audit)
                # A restart after dry-run repeats dry-run in the SAME directory.
                result = self.b.invoke(pending, False)
                require(result.get('stage') == 'dry_run', 'Unexpected dry-run result')
                reviewed = verify_plan(self.b, pending, result)
                pending.setdefault('reviewed', reviewed)
                pending['plan_sha256'] = reviewed['proposed_sha256']
                pending['phase'] = 'verified'
                self.record('prepared')
                require(self.b.queue.read_bytes() == pending['reviewed']['before'].encode('utf-8'),
                        'Queue changed after independent proposal review')
                if (self.b.output/'stop').exists() or self.b.read(self.b.supervisor/'status.json').get('terminal') is True:
                    return self.finish('cancelled_before_apply')
                pending['phase'] = 'applying'
                self.record('applying')
                result = self.b.invoke(pending, True)
            require(result.get('stage') in TERMINAL_AUDITS, 'Helper returned nonterminal audit')
            verify_plan(self.b, pending, result)
            return self.finish(result['stage'])
        except (Exception, KeyboardInterrupt) as error:
            try:
                audit = self.pending_audit()
            except Exception as read_error:
                return self.attention('Unreadable pending audit: '+str(read_error))
            if isinstance(error, helper.Refused) and str(error) in RESOURCE_REFUSALS and not audit_may_need_cleanup(audit):
                return self.finish('resource_lost')
            if audit and audit.get('stage') == 'aborted_restored':
                # Identity/controller errors still need attention, even if helper restored safely.
                return self.attention(error)
            if isinstance(error, (InterruptedError, KeyboardInterrupt)) and pending.get('interruption_retries', 0) < 1:
                pending['interruption_retries'] = pending.get('interruption_retries', 0)+1
                return self.record('recovery_pending', error=str(error))
            if isinstance(error, helper.Refused) and str(error) == 'Timed out awaiting scheduler and supervisor post-commit acknowledgements' and pending.get('recovery_attempts', 0) < 1:
                return self.record('recovery_pending', error=str(error))
            return self.attention(error)

    def step(self):
        if self.state.get('phase') == 'needs_attention':
            return 'needs_attention'
        if self.state.get('pending'):
            # Cleanup precedes stop markers and terminal controller status.
            return self.transaction()
        try:
            self.b.verify_helper()
            if (self.b.output/'stop').exists():
                return self.record('stopped')
            if self.b.read(self.b.supervisor/'status.json').get('terminal') is True:
                return self.record('supervisor_terminal')
            if self.b.now() < self.state.get('cooldown_until', 0):
                return 'cooldown'
            previous = self.state.get('controllers')
            controllers = self.b.controllers(previous)
            self.state['controllers'] = controllers
            if previous and not same_controller(controllers['scheduler'], previous['scheduler']):
                self.record('scheduler_adopted', scheduler_pid=controllers['scheduler']['pid'])
            queue, states, candidates, gates = inventory(self.b)
            if scheduler_has_priority(self.b, queue, states):
                if self.state.get('phase') != 'waiting_scheduler_capacity':
                    return self.record('waiting_scheduler_capacity')
                return 'waiting_scheduler_capacity'
            for _, job_id in gates:
                for candidate in candidates:
                    if not self.b.runtime.lease_available(candidate):
                        continue
                    try:
                        inspected = self.b.inspect_job(job_id, candidate, queue, states, controllers)
                    except helper.Refused as error:
                        if str(error) in RESOURCE_REFUSALS:
                            continue
                        raise
                    self.state['sequence'] += 1
                    name = f"opportunity_{int(self.b.now()*1000000)}_{self.state['sequence']:06d}"
                    pending = dict(job_id=job_id, candidate_gpu_uuid=candidate,
                        audit=str(self.b.root/'gate_reassignments'/name), original_job=inspected['job'], phase='planned',
                        **{key: inspected[key] for key in ('gate', 'scheduler', 'supervisor')})
                    self.state['pending'] = pending
                    self.record('planned')
                    return self.transaction()
            if self.state.get('phase') != 'waiting':
                return self.record('waiting')
            return 'waiting'
        except WaitingAdoption as error:
            if self.state.get('phase') != 'waiting_scheduler_adoption':
                return self.record('waiting_scheduler_adoption', reason=str(error))
            return 'waiting_scheduler_adoption'
        except Exception as error:
            return self.attention(error)


@contextmanager
def singleton(backend):
    backend.output.mkdir(parents=True, exist_ok=True)
    with (backend.output/'coordinator.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = backend.runtime.identity(os.getpid())
        receipt = dict(pid=os.getpid(), start_ticks=identity['start_ticks'], started=backend.now(),
            helper_sha256=HELPER_SHA256, coordinator_sha256=sha(Path(__file__).read_bytes()))
        backend.write(backend.output/'process.json', receipt)
        yield


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--once', action='store_true', help='With --launch, perform one coordinator step')
    args = parser.parse_args(argv)
    backend = Backend()
    backend.verify_helper()
    if not args.launch:
        # No writes, leases, inspect_job, controller helpers, or helper.operate.
        _, _, candidates, gates = inventory(backend)
        print(json.dumps(dict(mode='inspect', candidate_gpu_uuids=candidates, waiting_gate_ids=[x[1] for x in gates],
            leases='not probed', pending_state_exists=(backend.output/'state.json').exists())))
        return 0
    with singleton(backend):
        state_file = backend.output/'state.json'
        coordinator = Coordinator(backend, backend.read(state_file) if state_file.exists() else None)
        while True:
            phase = coordinator.step()
            if (args.once and phase != 'recovery_pending') or phase in ('needs_attention', 'stopped', 'supervisor_terminal'):
                print(json.dumps(dict(phase=phase, output=str(backend.output))))
                return 2 if phase == 'needs_attention' else 0
            if phase != 'recovery_pending':
                backend.runtime.sleep(POLL_SECONDS)


if __name__ == '__main__':
    raise SystemExit(main())

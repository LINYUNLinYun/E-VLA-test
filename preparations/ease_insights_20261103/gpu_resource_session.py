"""Persistent scheduler-owned GPU session; frozen local-search science is reused.

Only --launch starts children. The original scheduler assigns UUID/port and
passes its existing flock through the assigned-GPU gate. No other GPU or
process is allocated, signalled, adopted or killed. Empty inbox and attention
retain the keeper and lease until root explicitly creates a release file.
An identity-bound append-only root resume request can clear only nonfatal
attention after fresh source, scheduler/lease, child and keeper consistency
checks. Failed task outcomes remain unchanged; science retries need a new ID.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import socket
import subprocess
import sys
import time
import uuid

from preparations.ease_insights_20261103 import reassign_waiting_gate_v2 as identity
from preparations.ease_insights_20261103.await_assigned_gpu import assigned_uuid

PROJECT = Path('/root/nfs/embodied/benchmark')
EXPERIMENT = PROJECT/'results/ease_insights_20261103'
MODULE = 'preparations.ease_insights_20261103.gpu_resource_session'
KEEPER_MODULE = 'preparations.ease_insights_20261103.gpu_owned_keeper'
PLAN_SCHEMA = 'ease-gpu-resource-campaign-v1'
TASK_SCHEMA = 'ease-gpu-resource-task-v1'
SESSION_SCHEMA = 'ease-gpu-resource-session-v1'
RESUME_SCHEMA = 'ease-gpu-resource-resume-v1'
RESUME_RECEIPT_SCHEMA = 'ease-gpu-resource-resume-receipt-v1'
RESOURCE_SOURCES = (MODULE, KEEPER_MODULE, identity.LOCAL_RUNNER)
ALLOCATION_BYTES = 64*1024*1024


class KeeperFailure(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path, 'Noncanonical absolute path')
    return path


def load_plan(path):
    path = canonical(path)
    require(path.is_relative_to(EXPERIMENT), 'Campaign plan must be in the fixed experiment')
    raw = path.read_bytes(); plan = json.loads(raw)
    require(plan.get('schema') == PLAN_SCHEMA, 'Unknown campaign plan schema')
    root = canonical(plan['campaign_root'])
    require(root == path.parent and root.is_relative_to(EXPERIMENT), 'Campaign root/plan binding differs')
    allowed = [str(EXPERIMENT/'vlm_local_search_v1'/stage) for stage in ('discovery', 'confirmation')]
    require(plan.get('allowed_outputs') == allowed, 'Only fixed discovery/confirmation outputs are allowed')
    hashes = plan.get('source_hashes', {})
    for module in RESOURCE_SOURCES:
        require(module.replace('.', '/')+'.py' in hashes, 'Missing trusted source pin: '+module)
    require(isinstance(plan.get('keeper_max_memory_mib', 512), (int, float))
            and 64 <= plan.get('keeper_max_memory_mib', 512) <= 512, 'Keeper consistency memory bound is invalid')
    plan['_path'], plan['_sha256'] = str(path), sha(raw)
    verify_plan(plan)
    return plan


def verify_plan(plan):
    require(sha(Path(plan['_path']).read_bytes()) == plan['_sha256'], 'Append-only campaign plan changed')
    for name, expected in plan['source_hashes'].items():
        path = PROJECT/name
        require(path.resolve() == path and path.is_relative_to(PROJECT)
                and sha(path.read_bytes()) == expected, 'Trusted implementation source changed: '+name)


@contextmanager
def claim(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


class Runtime(identity.LocalRuntime):
    """All real child/GPU/pipe boundaries; CPU tests replace this object."""
    def write(self, path, value):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        identity.durable_json(path, value)

    def self_identity(self):
        return self.identity(os.getpid())

    def write_resume_receipt(self, path, value):
        """Resume decisions are append-only, unlike the live status file."""
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        require(path.parent.resolve() == path.parent and path.resolve() == path,
                'Resume receipt path is not canonical')
        with path.open('xb') as stream:
            stream.write(identity.json_bytes(value)); stream.flush(); os.fsync(stream.fileno())
        identity.sync_directory(path.parent)

    def validate_resume(self, request, active):
        """Validate an explicit operator request without releasing any resource."""
        fields = {'schema', 'action', 'plan_sha256', 'session_pid', 'proc_start_ticks',
                  'gpu_uuid', 'attention_sha256', 'reason'}
        require(isinstance(request, dict) and set(request) == fields
                and request['schema'] == RESUME_SCHEMA and request['action'] == 'resume_after_attention',
                'Unknown resume request schema/action/fields')
        require(active.attention is not None and active.fatal is None,
                'Resume requires nonfatal current attention')
        original = active.binding
        require(original and original.get('verified') is not False, 'Original session ownership was not verified')
        attention_hash = sha(json.dumps(active.attention, sort_keys=True).encode())
        own = original['session']
        require(request['plan_sha256'] == active.plan['_sha256']
                and isinstance(request['session_pid'], int) and not isinstance(request['session_pid'], bool)
                and request['session_pid'] == own['pid']
                and str(request['proc_start_ticks']) == own['start_ticks']
                and request['gpu_uuid'] == active.args.gpu_uuid
                and request['attention_sha256'] == attention_hash
                and isinstance(request['reason'], str) and request['reason'].strip(),
                'Resume plan/session/PID/start/UUID/attention/reason differs')
        require(active.keeper is not None and self.keeper_live(active.keeper), 'Original keeper is not live')
        require(self.children_reaped(), 'Scientific children are not all reaped')
        current = self.bind_scheduler(active.args.gpu_uuid)
        require(identity.same_process(current['session'], original['session'])
                and identity.same_process(current['scheduler'], original['scheduler'])
                and all(current[key] == original[key] for key in
                        ('job_id', 'lease_path', 'lease_device', 'lease_inode'))
                and set(current['lease_fds']) == set(original['lease_fds']),
                'Scheduler/session/inherited lease binding changed')
        verify_plan(active.plan)
        observation = self.scan_owned(active.args.gpu_uuid, active.keeper,
                                      active.plan.get('keeper_max_memory_mib', 512))
        require(active.fatal is None and active.attention is not None
                and sha(json.dumps(active.attention, sort_keys=True).encode()) == attention_hash
                and self.keeper_live(active.keeper) and self.children_reaped(),
                'Resume state changed during validation')
        return dict(verified=True, session_source_sha256=sha(Path(__file__).read_bytes()),
                    plan_sha256=active.plan['_sha256'], source_hashes=active.plan.get('source_hashes', {}),
                    inherited_scheduler_lease=current, keeper=active.keeper['ready'],
                    assigned_gpu_consistency=observation)

    def bind_scheduler(self, gpu_uuid):
        own = self.self_identity()
        scheduler_dir = EXPERIMENT/'scheduler'
        receipt = read(scheduler_dir/'controller_process.json')
        parent = self.identity(receipt['pid'])
        require(own['ppid'] == parent['pid'] and parent.get('state') not in identity.DEAD_STATES
                and identity.exact_module(parent.get('argv'), identity.SCHEDULER)
                and parent.get('uid') == own.get('uid') == os.getuid()
                and self.matches_started(parent, receipt['started']), 'Session is not an owned child of the recorded scheduler')
        states = read(scheduler_dir/'jobs_state.json')
        matched = [(key, state) for key, state in states.items()
                   if state.get('status') == 'running' and state.get('pid') == own['pid']]
        require(len(matched) == 1, 'Scheduler has not persisted this unique session PID')
        job_id, state = matched[0]
        outer_uuid, _, inner = identity.gate_parts(state['argv'])
        require(outer_uuid == state.get('gpu_uuid') == gpu_uuid and inner == own['argv']
                and self.matches_started(own, state['started']), 'Scheduler gate/session PID/start/UUID/argv binding differs')
        ticks = state.get('start_ticks', state.get('proc_start_ticks'))
        require(ticks is None or str(ticks) == own['start_ticks'], 'Recorded session start ticks differ')
        expected = Path('/tmp')/f'embodied-benchmark-gpu-{gpu_uuid}.lock'
        stat = expected.stat(); leases = []
        for entry in Path('/proc/self/fd').iterdir():
            try:
                current = entry.stat()
                proof = (Path('/proc/self/fdinfo')/entry.name).read_text()
            except (FileNotFoundError, OSError):
                continue
            if (current.st_dev, current.st_ino) == (stat.st_dev, stat.st_ino):
                require(any('FLOCK' in line and 'WRITE' in line for line in proof.splitlines() if line.startswith('lock:')),
                        'Inherited lease descriptor has no proven exclusive flock')
                leases.append(int(entry.name))
        require(leases, 'Missing inherited scheduler GPU flock descriptor')
        return dict(job_id=job_id, scheduler=parent, session=own, lease_path=str(expected),
                    lease_device=stat.st_dev, lease_inode=stat.st_ino, lease_fds=leases)

    def start_keeper(self, gpu_uuid, output):
        control_read, control_write = os.pipe(); ready_read, ready_write = os.pipe()
        nonce = uuid.uuid4().hex; own = self.self_identity()
        argv = ['/usr/bin/python3', '-m', KEEPER_MODULE, '--gpu-uuid', gpu_uuid,
                '--control-fd', str(control_read), '--ready-fd', str(ready_write), '--nonce', nonce,
                '--parent-pid', str(own['pid']), '--parent-start-ticks', own['start_ticks'], '--allocation-mib', '64']
        log = (Path(output)/'keeper.log').open('a')
        child = None
        try:
            child = subprocess.Popen(argv, cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, close_fds=True, pass_fds=(control_read, ready_write))
            os.close(control_read); control_read = None
            os.close(ready_write); ready_write = None
            deadline = time.monotonic()+30; raw = b''
            while b'\n' not in raw:
                require(child.poll() is None, 'Keeper exited before readiness')
                require(time.monotonic() < deadline, 'Keeper readiness timed out')
                readable, _, _ = select.select([ready_read], [], [], .2)
                if readable:
                    block = os.read(ready_read, 4096)
                    require(block, 'Keeper closed readiness without a receipt')
                    raw += block; require(len(raw) <= 65536, 'Keeper readiness exceeds bounded receipt size')
            ready = json.loads(raw.split(b'\n', 1)[0])
            process = self.identity(child.pid)
            require(ready.get('schema') == 'ease-gpu-owned-keeper-v1' and ready.get('phase') == 'ready'
                    and ready.get('gpu_uuid') == gpu_uuid and ready.get('pid') == child.pid
                    and str(ready.get('proc_start_ticks')) == process.get('start_ticks')
                    and ready.get('nonce') == nonce and ready.get('context_created') is True
                    and ready.get('allocated_bytes') == ALLOCATION_BYTES
                    and isinstance(ready.get('total_memory_bytes'), int) and ready['total_memory_bytes'] > ALLOCATION_BYTES
                    and isinstance(ready.get('driver_version'), int) and ready['driver_version'] > 0
                    and process.get('ppid') == own['pid'] and process.get('uid') == os.getuid()
                    and identity.exact_module(process.get('argv'), KEEPER_MODULE), 'Keeper readiness/owned identity proof failed')
            return dict(child=child, writer=control_write, identity=process, ready=ready, log=log)
        except BaseException:
            os.close(control_write)
            if child is not None:
                self.cleanup_child(child)
            log.close()
            raise
        finally:
            for fd in (control_read, ready_write, ready_read):
                if fd is not None:
                    os.close(fd)

    def keeper_live(self, keeper):
        return keeper['child'].poll() is None and identity.same_process(self.identity(keeper['child'].pid), keeper['identity'])

    def close_keeper(self, keeper):
        if not keeper.get('control_closed'):
            os.close(keeper['writer'])  # EOF releases only this child's allocation.
            keeper['control_closed'] = True
        try:
            keeper['child'].wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.cleanup_child(keeper['child'])
        keeper['log'].close()
        return dict(process_exited=keeper['child'].poll() is not None, exit_code=keeper['child'].returncode,
                    control_closed=True)

    def scan_owned(self, gpu_uuid, keeper, maximum):
        try:
            return self._scan_owned(gpu_uuid, keeper, maximum)
        except KeeperFailure:
            raise
        except Exception as error:
            raise KeeperFailure('Assigned GPU driver/ownership evidence unavailable: '+repr(error)) from error

    def _scan_owned(self, gpu_uuid, keeper, maximum):
        if not self.keeper_live(keeper):
            raise KeeperFailure('Owned keeper died or changed identity')
        selector = '--id='+gpu_uuid
        rows = subprocess.check_output(['nvidia-smi', selector, '--query-compute-apps=gpu_uuid,pid,used_memory',
            '--format=csv,noheader,nounits'], text=True, timeout=15).splitlines()
        rows = [row for row in rows if row.strip()]
        if len(rows) != 1:
            raise KeeperFailure('Assigned GPU has unknown/additional compute processes')
        observed_uuid, host_pid, memory = [part.strip() for part in rows[0].split(',')]
        used = float(memory)
        raw = subprocess.check_output(['nvidia-smi', selector, '--query-gpu=uuid,memory.used',
            '--format=csv,noheader,nounits'], text=True, timeout=15).strip().splitlines()
        require(len(raw) == 1, 'Assigned GPU memory evidence is incomplete')
        memory_uuid, total = [part.strip() for part in raw[0].split(',')]; total = float(total)
        consistent = (observed_uuid == memory_uuid == gpu_uuid and int(host_pid) > 0
                and math.isfinite(used) and math.isfinite(total) and 0 < used <= maximum
                and used <= total <= maximum and self.keeper_live(keeper))
        if not consistent:
            raise KeeperFailure('Assigned keeper/sole-small-compute consistency failed')
        return dict(gpu_uuid=gpu_uuid, compute_pid_observed=host_pid, compute_pid_namespace='unresolved',
                    used_memory_mib=used, total_memory_mib=total, attribution='owned_live_driver_keeper_plus_sole_bounded_compute_consistency')

    def popen(self, argv, env, handle):
        child = subprocess.Popen(argv, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL, stdout=handle,
                                 stderr=subprocess.STDOUT, close_fds=True, pass_fds=())
        self.__dict__.setdefault('owned_scientific_children', []).append(child)
        return child

    def children_reaped(self):
        return all(child.poll() is not None for child in self.__dict__.get('owned_scientific_children', []))

    def reap_scientific_children(self):
        for child in reversed(self.__dict__.get('owned_scientific_children', [])):
            self.cleanup_child(child)
        require(self.children_reaped(), 'Owned scientific child is not reaped')

    def cleanup_child(self, child):
        # This is always our Popen handle, never an inventory PID.
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait(timeout=30)
        else:
            child.wait()

    def reserve_port(self, port):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))


class Science:
    def __init__(self, runner=None, worker=None):
        if runner is None:
            from preparations.ease_insights_20261103.vlm_local_search_v1 import runner, worker
        self.r, self.w = runner, worker

    def prepare(self, task, plan, gpu_uuid, port):
        required = {'schema', 'id', 'logical_job_id', 'argv', 'complete_file', 'manifest_hash',
                    'output', 'condition', 'stage', 'shard_index', 'shards', 'env', 'activation_file'}
        require(set(task) == required and task['schema'] == TASK_SCHEMA, 'Task schema/fields differ')
        require(isinstance(task['id'], str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,200}', task['id'])
                and isinstance(task['logical_job_id'], str) and task['logical_job_id'], 'Invalid task/logical job ID')
        require(task['env'] == {}, 'Only the current original empty scientific job env is supported')
        options = identity.parse_options(task['argv'], identity.LOCAL_RUNNER,
            ('--output', '--condition', '--gpu-uuid', '--port', '--stage', '--shard-index', '--shards'), ('--launch',))
        require(options.get('--launch') is True and options.get('--gpu-uuid') == '{gpu_uuid}'
                and options.get('--port') == '{port}' and task['argv'][0] == str(self.r.MODEL_PYTHON),
                'Original runner must use the frozen Python and only assigned GPU/port and launch')
        output = canonical(task['output'])
        require(str(output) in plan['allowed_outputs'] and options.get('--output') == str(output)
                and options.get('--condition') == task['condition']
                and options.get('--stage', task['stage']) == task['stage']
                and int(options.get('--shard-index', 0)) == task['shard_index']
                and int(options.get('--shards', 1)) == task['shards'], 'Task and original scientific argv differ')
        manifest = self.r.read_manifest(output/'manifest.json', verify_sources=True)
        self.r.validate_output(output, manifest); self.r.planned_config(manifest, task['condition'])
        require(manifest['manifest_hash'] == task['manifest_hash'] and self.r.stage_for(manifest) == task['stage'],
                'Task scientific manifest/stage differs')
        scenes = self.r.cases(manifest, task['stage'], task['shard_index'], task['shards'])
        marker = output/task['condition']/f"{task['stage']}_shard{task['shard_index']:02d}of{task['shards']:02d}_complete.json"
        require(canonical(task['complete_file']) == marker, 'Task completion destination differs')
        args = argparse.Namespace(output=str(output), condition=task['condition'], stage=task['stage'],
            shard_index=task['shard_index'], shards=task['shards'], gpu_uuid=gpu_uuid, port=port, launch=False, preflight_only=False)
        # Obtain commands and log destination from the frozen pure dry-run path.
        commands = self.r.run(args)
        logs = Path(identity.option(commands['server'], '--output'))
        require(logs.is_relative_to(output/task['condition']/'runs') and logs.resolve() == logs,
                'Frozen dry-run returned an unexpected log destination')
        return dict(task=task, args=args, output=output, manifest=manifest, scenes=scenes,
                    marker=marker, logs=logs, server=commands['server'], worker=commands['worker'])

    def execute(self, prepared, session):
        r, w, rt = self.r, self.w, session.runtime
        p = prepared; args = p['args']; output = p['output']; manifest = p['manifest']; scenes = p['scenes']
        for scene in scenes:
            r.recover_case(output, args.condition, scene, manifest, w.finalize)
        missing = [scene for scene in scenes if not w.completed(output, args.condition, scene, manifest['manifest_hash'])]
        if not missing:
            receipt = dict(complete=True, manifest_hash=manifest['manifest_hash'], condition=args.condition, already_completed=len(scenes))
            r.atomic_json(p['marker'], receipt)
            return receipt
        binding = r.resolve_binding(args.gpu_uuid)
        with r.lock(output/args.condition/f'shard{args.shard_index:02d}of{args.shards:02d}.lock', blocking=False):
            session.check_keeper()
            observation = rt.scan_owned(args.gpu_uuid, session.keeper, session.plan.get('keeper_max_memory_mib', 512))
            rt.reserve_port(args.port)
            env = r.environment()
            env.update(CUDA_VISIBLE_DEVICES=args.gpu_uuid, MUJOCO_EGL_DEVICE_ID=str(binding['egl_index']),
                       EASE_VLM_GPU_UUID=args.gpu_uuid, XLA_PYTHON_CLIENT_PREALLOCATE='false')
            env['PYTHONPATH'] = str(PROJECT/'policies/realtime-vla-flash/src')+':'+env['PYTHONPATH']
            logs = p['logs']; logs.mkdir(parents=True)
            revision = session.execution_revision(observation)
            r.atomic_json(logs/'identity.json', dict(manifest_hash=manifest['manifest_hash'], condition=args.condition,
                gpu_binding=binding, controller_pid=os.getpid(), server_command=p['server'], worker_command=p['worker'],
                stateful_workers_per_server=1, technical_runtime_revision=r.runtime_revision(manifest),
                resource_execution_revision=revision,
                command_semantics='server_command and worker_command are the actual argv passed to Popen; worker case assignment is inferred from the immutable local-search phase; original backend and policy mechanics are reused',
                global_gpu_lock_owner='external scheduler', global_gpu_lock=session.binding['lease_path']))
            started = rt.monotonic(); children = []; handles = []
            try:
                handle = (logs/'server.log').open('a'); handles.append(handle)
                server = rt.popen(p['server'], env, handle); children.append(server)
                deadline = rt.monotonic()+1800
                while rt.monotonic() < deadline:
                    session.check_keeper()
                    require(server.poll() is None, 'Frozen server exited before health gate')
                    try:
                        protocol = r.rpc(args.port, dict(op='health'))['protocol']
                        require(protocol['server_pid'] == server.pid and protocol['manifest_hash'] == manifest['manifest_hash']
                                and protocol['condition_id'] == args.condition and protocol['gpu_uuid'] == args.gpu_uuid
                                and protocol['real_input_parity_passed'], 'Wrong service identity or failed real-input gate')
                        break
                    except (ConnectionError, OSError):
                        rt.sleep(1)
                else:
                    raise TimeoutError('Server preparation exceeded 1800 seconds')
                r.cross_service(output, logs, manifest['manifest_hash'])
                tick = rt.monotonic()
                handle = (logs/'worker.log').open('a'); handles.append(handle)
                client = rt.popen(p['worker'], env, handle); children.append(client)
                while client.poll() is None:
                    session.check_keeper(); require(server.poll() is None, 'Model server exited during rollout'); rt.sleep(1)
                require(client.returncode == 0, 'Frozen worker exited with a technical failure')
                after = sum(w.completed(output, args.condition, scene, manifest['manifest_hash']) for scene in scenes)
                require(after == len(scenes), 'Worker did not complete all assigned cases')
                receipt = dict(complete=True, manifest_hash=manifest['manifest_hash'], condition=args.condition,
                    gpu_binding=binding, completed_after=after, new_episodes=after-len(scenes)+len(missing),
                    closed_loop_wall_s=rt.monotonic()-tick, total_wall_s=rt.monotonic()-started, logs=str(logs),
                    resource_execution_revision=revision)
                r.atomic_json(logs/'result.json', receipt); r.atomic_json(p['marker'], receipt)
                return receipt
            finally:
                cleanup_errors = []
                for child in reversed(children):
                    try:
                        rt.cleanup_child(child)
                    except Exception as error:
                        cleanup_errors.append(repr(error))
                for handle in handles:
                    handle.close()
                require(not cleanup_errors, 'Owned child cleanup requires attention: '+repr(cleanup_errors))


class Session:
    def __init__(self, args, plan, runtime=None, science=None):
        self.args, self.plan = args, plan
        self.runtime, self.science = runtime or Runtime(), science or Science()
        self.root, self.output = Path(plan['campaign_root']), Path(args.output)
        self.keeper = None; self.binding = None; self.attention = None; self.fatal = None; self.seen = {}

    def record(self, phase, **details):
        value = dict(schema=SESSION_SCHEMA, phase=phase, gpu_uuid=self.args.gpu_uuid,
                     time=self.runtime.now(), session_pid=os.getpid(), **details)
        self.runtime.write(self.output/'status.json', value)
        return value

    def check_keeper(self):
        if self.keeper is None or not self.runtime.keeper_live(self.keeper):
            raise KeeperFailure('Fatal owned keeper/driver failure')

    def execution_revision(self, observation):
        value = dict(schema=SESSION_SCHEMA, source_sha256=sha(Path(__file__).read_bytes()), plan_sha256=self.plan['_sha256'],
                     original_science_unchanged=True, frozen_runner_launch_used=False,
                     command_builder='frozen runner.run(launch=False)', keeper=self.keeper['ready'],
                     allocation_bytes=ALLOCATION_BYTES, inherited_scheduler_lease=self.binding,
                     assigned_gpu_consistency=observation)
        value['revision_hash'] = sha(json.dumps(value, sort_keys=True).encode())
        return value

    def released(self):
        return (self.root/'release').is_file() or (self.root/'releases'/self.args.gpu_uuid).is_file()

    def request_signal(self, signum):
        self.attention = dict(scope='operator_signal', signal=signum, safe_boundary=True)

    def resume_requests(self):
        """Consume explicit requests once; a bad request only preserves the hold."""
        directory = self.root/'resumes'/self.output.name
        receipts = self.output/'resume_receipts'
        for path in sorted(directory.glob('*.json')):
            request_id = path.stem; raw = b''; proof = dict(verified=False,
                session_source_sha256=None, plan_sha256=self.plan['_sha256'])
            decision = 'rejected'; error = None; snapshot = self.attention
            attention_hash = sha(json.dumps(snapshot, sort_keys=True).encode())
            receipt_path = receipts/(request_id+'.json')
            try:
                require(re.fullmatch(r'[A-Za-z0-9_.-]{1,200}', request_id)
                        and directory.resolve() == directory and path.resolve() == path,
                        'Resume request is outside the canonical slot inbox')
                raw = path.read_bytes(); request_hash = sha(raw)
                require(len(raw) <= 65536, 'Resume request exceeds the bounded record size')
                if receipt_path.exists():
                    previous = read(receipt_path)
                    if previous.get('request_sha256') == request_hash:
                        continue
                    # Never overwrite a consumed decision or replay a changed ID.
                    changed_key = sha(request_id.encode()+b'\0'+request_hash.encode())
                    receipt_path = receipts/('changed_'+changed_key+'.json')
                    if receipt_path.exists():
                        continue
                    raise RuntimeError('Append-only consumed resume request bytes changed')
                proof['session_source_sha256'] = sha(Path(__file__).read_bytes())
                request = json.loads(raw)
                proof = self.runtime.validate_resume(request, self)
                require(path.resolve() == path and path.read_bytes() == raw,
                        'Resume request bytes changed during validation')
                decision = 'resumed'
            except Exception as failure:
                error = repr(failure)
            own = (self.binding or {}).get('session', {})
            receipt = dict(schema=RESUME_RECEIPT_SCHEMA, phase=decision, request_id=request_id,
                request_sha256=sha(raw), attention_sha256=attention_hash, plan_sha256=self.plan['_sha256'],
                session_pid=own.get('pid'), proc_start_ticks=own.get('start_ticks'), gpu_uuid=self.args.gpu_uuid,
                time=self.runtime.now(), error=error, source_proof=proof)
            try:
                self.runtime.write_resume_receipt(receipt_path, receipt)
            except Exception:
                # No durable consumption means no authority to clear attention.
                continue
            if (decision == 'resumed' and self.attention is snapshot and self.fatal is None
                    and sha(json.dumps(self.attention, sort_keys=True).encode()) == attention_hash):
                self.attention = None

    def finish_release(self):
        self.runtime.reap_scientific_children()
        cleanup = dict(process_exited=True, exit_code=None, keeper_was_started=False)
        if self.keeper is not None:
            cleanup = self.runtime.close_keeper(self.keeper)
        require(cleanup.get('process_exited') and self.runtime.children_reaped(),
                'Resource release lacks reaped owned server/worker/keeper confirmation')
        own = self.binding.get('session') or self.runtime.self_identity()
        phase = 'released_after_fatal' if self.fatal is not None or cleanup.get('exit_code') not in (None, 0) else 'released'
        complete = dict(schema='ease-gpu-resource-slot-release-v1', complete=True, phase=phase,
            scope='resource_slot_only', scientific_success=False, reason='explicit_root_release_file',
            plan_sha256=self.plan['_sha256'], session_id=self.output.name, gpu_uuid=self.args.gpu_uuid,
            pid=own['pid'], proc_start_ticks=own.get('start_ticks'), time=self.runtime.now(),
            keeper_cleanup=cleanup, owned_scientific_children_reaped=True, fatal_attention=self.fatal)
        self.runtime.write(self.output/'complete.json', complete)
        return self.record(phase, completion_file=str(self.output/'complete.json'))

    def step(self):
        if self.released():
            return self.record('release_requested')
        if self.attention is not None or self.fatal is not None:
            self.resume_requests()
        if self.fatal is not None:
            return self.record('fatal_attention', attention=self.fatal, scheduler_flock_retained=True)
        self.check_keeper()
        if self.attention is not None:
            return self.record('attention_holding', attention=self.attention)
        verify_plan(self.plan)
        for path in sorted((self.root/'tasks').glob('*.json')):
            require(path.resolve() == path and path.parent == self.root/'tasks', 'Task is not a canonical inbox file')
            raw = path.read_bytes(); task = json.loads(raw)
            require(path.name == task.get('id', '')+'.json', 'Task filename differs from append-only task ID')
            digest = sha(raw)
            require(task['id'] not in self.seen or self.seen[task['id']] == digest, 'Append-only task file changed')
            self.seen[task['id']] = digest
            activation = canonical(task['activation_file'])
            require(activation.name == 'committed.json' and activation.parent.parent == self.root/'transactions',
                    'Task activation is outside campaign transaction commits')
            if not activation.exists():
                continue
            activated = read(activation)
            require(activated.get('schema') == 'ease-gpu-resource-activation-v1'
                    and activated.get('task_sha256', {}).get(task['id']) == digest,
                    'Task was not activated by matching committed queue transaction bytes')
            key = sha(task['id'].encode()); outcome = self.root/'outcomes'/(key+'.json')
            if outcome.exists():
                previous = read(outcome)
                require(previous.get('task_sha256') == digest, 'Recorded task bytes changed')
                continue
            try:
                with claim(self.root/'claims'/(key+'.lock')):
                    if outcome.exists():
                        continue
                    self.record('running', task_id=task['id'], logical_job_id=task.get('logical_job_id'))
                    try:
                        prepared = self.science.prepare(task, self.plan, self.args.gpu_uuid, self.args.port)
                        result = self.science.execute(prepared, self)
                        self.runtime.write(outcome, dict(schema=SESSION_SCHEMA, phase='complete', task_id=task['id'],
                            task_sha256=digest, gpu_uuid=self.args.gpu_uuid, result=result, time=self.runtime.now()))
                    except Exception as error:
                        self.attention = dict(task_id=task['id'], error=repr(error))
                        if isinstance(error, KeeperFailure):
                            self.fatal = self.attention
                        self.runtime.write(outcome, dict(schema=SESSION_SCHEMA, phase='attention', task_id=task['id'],
                            task_sha256=digest, gpu_uuid=self.args.gpu_uuid, error=repr(error), time=self.runtime.now()))
                        if self.fatal is not None:
                            return self.record('fatal_attention', attention=self.fatal, scheduler_flock_retained=True)
                        self.record('attention_holding', attention=self.attention)
                        self.check_keeper()
                        return dict(phase='attention_holding')
                    self.check_keeper()
                    return self.record('task_complete', task_id=task['id'])
            except BlockingIOError:
                continue
        return self.record('idle_holding')

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        with claim(self.output/'session.lock'):
            try:
                self.binding = self.runtime.bind_scheduler(self.args.gpu_uuid)
            except Exception as error:
                self.binding = dict(verified=False)
                self.fatal = dict(error=repr(error), scope='scheduler_or_lease_ownership', gpu_context_retained=False)
            self.runtime.write(self.output/'binding.json', self.binding)
            if self.fatal is None:
                try:
                    self.keeper = self.runtime.start_keeper(self.args.gpu_uuid, self.output)
                except Exception as error:
                    self.fatal = dict(error=repr(error), scope='keeper_startup', gpu_context_retained=False)
            try:
                if self.keeper is not None:
                    self.runtime.write(self.output/'keeper.json', self.keeper['ready'])
                    self.record('ready_holding')
                while True:
                    try:
                        value = self.step()
                    except KeeperFailure as error:
                        self.fatal = dict(error=repr(error), scope='keeper_or_resource_ownership')
                        value = self.record('fatal_attention', attention=self.fatal, scheduler_flock_retained=True)
                    except Exception as error:
                        self.attention = dict(error=repr(error), scope='campaign_or_task_validation')
                        value = self.record('attention_holding', attention=self.attention)
                    if value['phase'] == 'release_requested':
                        try:
                            return self.finish_release()
                        except Exception as error:
                            self.fatal = dict(error=repr(error), scope='explicit_release_cleanup')
                            value = self.record('fatal_attention', attention=self.fatal, scheduler_flock_retained=True)
                    if value['phase'] != 'task_complete':
                        self.runtime.sleep(1)
            except BaseException as error:
                self.record('fatal', error=repr(error)); raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--port', required=True, type=int)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args(argv)
    assigned_uuid(args.gpu_uuid); require(1024 <= args.port < 65536, 'Invalid scheduler port')
    plan = load_plan(args.plan); output = canonical(args.output)
    require(output.parent == Path(plan['campaign_root'])/'sessions', 'Session output escaped fixed campaign sessions')
    if not args.launch:
        print(json.dumps(dict(mode='inspect', plan_sha256=plan['_sha256'], gpu_uuid=args.gpu_uuid,
                              task_inbox=str(Path(plan['campaign_root'])/'tasks')))); return

    active = Session(args, plan)

    def stop(signum, frame):
        # Never interrupt science or release a scheduler lease merely because
        # a signal arrived in one particular Python frame. Root release files
        # remain the sole graceful release mechanism.
        active.request_signal(signum)

    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    print(json.dumps(active.run()))


if __name__ == '__main__':
    main()

"""Two isolated LIBERO clients, one unchanged sequential scientific server."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from common import (ROOT, MODELS, Ledger, atomic_json, digest, evaluation_env,
                    exclusive, gpu_lease, lock, prior_diagnostic_budget, sha)
from runtime_data import verify_frozen, verify_parity, verify_acceptance
from runtime_runner import CLIENT, available_port, free_gpus, model_python

HERE = Path(__file__).resolve().parent
SOURCES = ('throughput_revision.py', 'throughput_revision_client.py', 'throughput_revision_handoff.py')


def read(path):
    return json.loads(Path(path).read_text())


def prepare(attempt, revision):
    attempt, revision = Path(attempt).resolve(), Path(revision).resolve()
    manifest = verify_frozen(attempt)
    revision.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema=1, kind='two_env_sequential_server_execution_revision',
                   source_attempt=str(attempt), manifest_hash=manifest['manifest_hash'],
                   manifest_file_sha256=sha(attempt/'condition_manifest.json'),
                   frozen_code_hashes=manifest['code_hashes'],
                   code_hashes={name: sha(HERE/name) for name in SOURCES},
                   environments=2, server='unchanged runtime_server.py; sequential accept/predict loop',
                   episode_client='one OS process per environment; unchanged run_episode/validation',
                   valid_episode_policy='preserve prior valid; canonical locked Ledger.reserve',
                   diagnostic_predictions_per_model=8, created=time.time())
    atomic_json(revision/'revision_manifest.json', receipt)
    return receipt


def verify_revision(revision):
    revision = Path(revision).resolve()
    receipt = read(revision/'revision_manifest.json')
    attempt = Path(receipt['source_attempt'])
    manifest = verify_frozen(attempt)
    if (receipt.get('schema') != 1 or receipt.get('environments') != 2 or
            receipt.get('kind') != 'two_env_sequential_server_execution_revision' or
            receipt['manifest_hash'] != manifest['manifest_hash'] or
            receipt['manifest_file_sha256'] != sha(attempt/'condition_manifest.json') or
            receipt['frozen_code_hashes'] != manifest['code_hashes']):
        raise ValueError('Throughput revision/scientific identity changed')
    if set(receipt['code_hashes']) != set(SOURCES):
        raise ValueError('Throughput source identity incomplete')
    for name, expected in receipt['code_hashes'].items():
        if sha(HERE/name) != expected:
            raise ValueError(f'Throughput source changed: {name}')
    return receipt, manifest


def acceptance(revision, model, phase, evidence_path=None):
    revision = Path(revision)
    path = revision/'acceptance'/f'{model}.{phase}.json'
    receipt = read(path)
    if (receipt.get('schema') != 1 or receipt.get('accepted') is not True or
            receipt.get('reviewer_model') != 'gpt-6.1-sol' or receipt.get('reasoning_effort') != 'medium' or
            receipt.get('phase') != phase or receipt.get('model') != model or
            receipt.get('revision_manifest_sha256') != sha(revision/'revision_manifest.json') or
            not receipt.get('evidence')):
        raise ValueError(f'Medium {phase} acceptance missing/stale')
    if evidence_path is not None and receipt.get('phase_receipt_sha256') != sha(evidence_path):
        raise ValueError('Acceptance does not bind phase receipt')
    return sha(path)


def reserve_gate(attempt, revision, model, manifest, parity):
    path = Path(attempt)/'readiness'/model/'diagnostic_budget.json'
    with lock(path.with_suffix('.lock')):
        budget = read(path) if path.exists() else dict(smoke_reserved=0, smoke_actual=0, entries=[])
        index_path = path.with_name('throughput_diagnostic_budget.json')
        index = read(index_path) if index_path.exists() else dict(schema=1, entries=[])
        revision_hash = sha(Path(revision)/'revision_manifest.json')
        if any(e.get('throughput_revision_sha256') == revision_hash for e in index['entries']):
            raise RuntimeError('This revision already reserved its gate; no implicit diagnostic retry')
        total = (parity['reserved_calls'] + budget['smoke_reserved'] + prior_diagnostic_budget(manifest, model)
                 + sum(entry['reserved_predictions'] for entry in index['entries']))
        if total + 8 > 256:
            raise RuntimeError(f'Diagnostic budget exhausted: {total}+8>256')
        index['entries'].append(dict(kind='throughput_equivalence',
            throughput_revision_sha256=revision_hash, reserved_predictions=8,
            status='reserved', not_formal=True, timestamp=time.time(),
            base_budget_path=str(path), base_budget_sha256=sha(path),
            parity_path=str(Path(attempt)/'parity'/f'{model}.json'),
            parity_sha256=sha(Path(attempt)/'parity'/f'{model}.json'), base_total=total))
        atomic_json(index_path, index)
    return total + 8


def gate_requests(manifest, model):
    conditions = [c for c in manifest['conditions'] if c['model'] == model]
    choices = [c for c in conditions if (c['schedule']['kind'] == 'native' and c['schedule']['num_steps'] == 2)] if model == 'db-cogact' else [c for c in conditions if c['schedule'].get('m') == .5]
    if len(choices) != 2 or {c['mask'] for c in choices} == {'full'}:
        raise ValueError('Gate requires Full and one variant at the same native/control schedule')
    point = manifest['observations'][model][0]
    seeds = [point['seed'], point['seed'] ^ 0x10203]
    return [dict(index=i, condition=condition, seed=seed, point=point)
            for i, (condition, seed) in enumerate((c, s) for s in seeds for c in choices)]


def compare_gate(directory):
    import numpy as np
    comparisons = []
    for index in range(4):
        left, right = [read(Path(directory)/phase/f'{index}.json') for phase in ('reference', 'interleaved')]
        for field in ('input_hash', 'noise_hash', 'raw_prediction_hash', 'seed', 'loaded_code_hashes'):
            if left[field] != right[field]:
                raise ValueError(f'Interleaved {field} differs for request {index}')
        for field in ('NFE', 'actual_NFE', 'schedule', 'skip', 'instrumentation_mode', 'cache_key'):
            if left['audit'].get(field) != right['audit'].get(field):
                raise ValueError(f'Interleaved audit {field} differs')
        maximum = 0.
        for key in ('noise_path', 'server_prediction_path'):
            with np.load(left[key], allow_pickle=False) as a, np.load(right[key], allow_pickle=False) as b:
                if a.files != b.files:
                    raise ValueError('Raw artifact fields differ')
                for field in a.files:
                    if not np.array_equal(a[field], b[field]):
                        raise ValueError(f'Interleaved raw artifact differs: {key}/{field}')
                    maximum = max(maximum, float(np.max(np.abs(a[field].astype(float)-b[field].astype(float)))))
        if left['actions'] != right['actions']:
            raise ValueError('Decoded actions differ')
        comparisons.append(dict(index=index, max_abs=maximum, exact=True,
            reference_sha256=sha(Path(directory)/'reference'/f'{index}.json'),
            interleaved_sha256=sha(Path(directory)/'interleaved'/f'{index}.json')))
    return comparisons


def stop_process(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def launch_client(revision, session, model, port, slot, mode, env, stream):
    return subprocess.Popen([str(CLIENT), str(HERE/'throughput_revision_client.py'), mode,
        '--revision', str(revision), '--session', str(session), '--model', model,
        '--port', str(port), '--slot', str(slot)], env=env, stdout=stream, stderr=subprocess.STDOUT)


def claim_pilot(revision, model, directory):
    marker = Path(revision)/f'{model}.pilot_session.json'
    if marker.exists():
        raise RuntimeError('This revision already owns a pilot session; no automatic repeat')
    atomic_json(marker, dict(session=str(directory), revision_manifest_sha256=sha(Path(revision)/'revision_manifest.json')))


def session(revision, model, gpu, phase):
    revision = Path(revision).resolve()
    identity, manifest = verify_revision(revision)
    attempt = Path(identity['source_attempt'])
    if model not in MODELS or phase not in ('gate', 'pilot', 'run'):
        raise ValueError('Unknown model/phase')
    required = {'gate': 'source', 'pilot': 'gate', 'run': 'pilot'}[phase]
    evidence = None if required == 'source' else revision/f'{model}.{required}_ready.json'
    accepted_sha = acceptance(revision, model, required, evidence)
    directory = revision/'sessions'/f'{model}_{phase}_{uuid.uuid4().hex}'
    directory.mkdir(parents=True)
    stop = directory/'STOP'
    def request_stop(*_):
        atomic_json(stop, dict(reason='manager signal: drain active episodes', timestamp=time.time()))
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    with ExitStack() as stack:
        # Refuse coexistence with either old sequential worker, including idle owners.
        for shard in (0, 1):
            stack.enter_context(exclusive(attempt/'leases'/f'{model}.shard{shard}.lock'))
        stack.enter_context(exclusive(revision/f'{model}.manager.lock'))
        fd = stack.enter_context(gpu_lease(gpu, allow_inherited=False))
        if gpu not in free_gpus():
            raise RuntimeError('GPU is busy before throughput server launch')
        if phase == 'pilot':
            claim_pilot(revision, model, directory)
        env = evaluation_env(gpu)
        env['V22_GPU_LEASE_FD'] = str(fd)
        port = available_port()
        server_log = stack.enter_context((directory/'server.log').open('a'))
        server = subprocess.Popen([str(model_python(model)), str(HERE/'runtime_server.py'),
            '--model', model, '--port', str(port), '--output-dir', str(directory/'server'), '--launch'],
            env=env, pass_fds=(fd,), stdout=server_log, stderr=subprocess.STDOUT)
        clients = []
        reference = None
        started, peak_memory = time.time(), 0.
        telemetry = directory/'gpu_memory.jsonl'
        try:
            from runtime_client import rpc
            deadline = time.monotonic() + 900
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Sequential server exited during startup')
                try:
                    health = rpc(port, dict(op='health'))
                    break
                except (OSError, RuntimeError):
                    if time.monotonic() > deadline or stop.exists():
                        raise RuntimeError('Server startup stopped/timed out')
                    time.sleep(1)
            parity = verify_parity(attempt, health['protocol'], model)
            verify_acceptance(attempt, model, parity)
            atomic_json(directory/'execution_receipt.json', dict(schema=1, model=model, phase=phase,
                revision_manifest_sha256=sha(revision/'revision_manifest.json'),
                phase_acceptance_sha256=accepted_sha, gpu=gpu, port=port,
                server_pid=server.pid, manager_pid=os.getpid(), environments=2,
                frozen_runtime_protocol=health['protocol'], started=started))
            if phase == 'gate':
                calls_before = health['calls']
                total = reserve_gate(attempt, revision, model, manifest, parity)
                atomic_json(directory/'requests.json', gate_requests(manifest, model))
                stream = stack.enter_context((directory/'reference.log').open('a'))
                reference = launch_client(revision, directory, model, port, 0, 'reference', env, stream)
                if reference.wait(timeout=900):
                    raise RuntimeError('Reference gate predictions failed; reserved budget retained')
                atomic_json(directory/'turn.json', dict(next=0))
                mode = 'interleaved'
            else:
                mode = phase
            for slot in range(2):
                stream = stack.enter_context((directory/f'client{slot}.log').open('a'))
                clients.append(launch_client(revision, directory, model, port, slot, mode, env, stream))
            evaluation_started = time.time()
            next_sample = 0.
            pilot_ready_written = False
            while any(client.poll() is None for client in clients):
                if server.poll() is not None:
                    atomic_json(stop, dict(reason='server exited', timestamp=time.time()))
                    for client in clients:
                        stop_process(client)
                    raise RuntimeError('Server exited; inspect and explicitly diagnose remaining reservations')
                if any(client.poll() not in (None, 0) for client in clients):
                    atomic_json(stop, dict(reason='client failed; drain peer', timestamp=time.time()))
                if phase == 'pilot' and not pilot_ready_written and all((directory/f'pilot{slot}.json').exists() for slot in range(2)):
                    receipt = dict(schema=1, model=model, phase='pilot',
                        revision_manifest_sha256=sha(revision/'revision_manifest.json'),
                        session=str(directory), results=[read(directory/f'pilot{slot}.json') for slot in range(2)],
                        evaluation_seconds=time.time()-evaluation_started,
                        episodes_per_hour=2*3600/(time.time()-evaluation_started),
                        end_to_end_wall_seconds=time.time()-started, peak_sampled_gpu_memory_mib=peak_memory,
                        pending_medium_acceptance=True)
                    atomic_json(revision/f'{model}.pilot_ready.json', receipt)
                    pilot_ready_written = True
                if time.monotonic() >= next_sample:
                    output = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-gpu=memory.used',
                        '--format=csv,noheader,nounits'], text=True)
                    memory = float(output.strip())
                    peak_memory = max(peak_memory, memory)
                    with telemetry.open('a') as output_stream:
                        output_stream.write(json.dumps(dict(timestamp=time.time(), memory_mib=memory))+'\n')
                    next_sample = time.monotonic()+5
                time.sleep(.2)
            if any(client.returncode != 0 for client in clients):
                raise RuntimeError('Throughput client failed; no automatic retry')
            if phase == 'gate':
                calls_after = rpc(port, dict(op='health'))['calls']
                if calls_after - calls_before != 8:
                    raise ValueError('Gate server logical call count differs from reserved eight')
                receipt = dict(schema=1, model=model, phase='gate',
                    revision_manifest_sha256=sha(revision/'revision_manifest.json'),
                    session=str(directory), comparisons=compare_gate(directory), actual_predictions=8,
                    total_diagnostic_reserved=total, exact=True, calls_before=calls_before, calls_after=calls_after,
                    reference_order=[0, 2, 1, 3], interleaved_order=[0, 1, 2, 3],
                    client_identities=[read(directory/f'client{slot}_identity.json') for slot in range(2)],
                    startup_forward_ledgers={str(path): sha(path) for path in (directory/'server').rglob('startup_forward_ledger.jsonl')})
                atomic_json(revision/f'{model}.gate_ready.json', receipt)
                budget_path = attempt/'readiness'/model/'diagnostic_budget.json'
                with lock(budget_path.with_suffix('.lock')):
                    index_path = budget_path.with_name('throughput_diagnostic_budget.json')
                    index = read(index_path)
                    entry = next(entry for entry in index['entries'] if entry['throughput_revision_sha256'] == sha(revision/'revision_manifest.json'))
                    entry.update(status='gate_complete', actual_predictions=8, gate_receipt_sha256=sha(revision/f'{model}.gate_ready.json'))
                    atomic_json(index_path, index)
            summary = dict(schema=1, phase=phase, model=model, session=str(directory),
                end_to_end_elapsed_seconds=time.time()-started, evaluation_seconds=time.time()-evaluation_started,
                peak_sampled_gpu_memory_mib=peak_memory,
                server_calls=rpc(port, dict(op='health'))['calls'],
                client_summaries=[read(directory/f'client{slot}_summary.json') for slot in range(2)],
                revision_manifest_sha256=sha(revision/'revision_manifest.json'))
            episodes = sum(row.get('valid', 0) for row in summary['client_summaries'])
            if phase == 'pilot':
                summary['evaluation_seconds'] = read(revision/f'{model}.pilot_ready.json')['evaluation_seconds']
            summary.update(valid=episodes, episodes_per_hour=episodes*3600/summary['evaluation_seconds'])
            atomic_json(directory/'complete.json', summary)
            return summary
        finally:
            if reference is not None and reference.poll() is None:
                stop_process(reference)
            # Normally clients drain themselves. Unexpected manager errors stop peers first.
            if any(client.poll() is None for client in clients):
                atomic_json(stop, dict(reason='manager failure; drain peer', timestamp=time.time()))
                for client in clients:
                    client.wait()
            stop_process(server)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'gate', 'pilot', 'run'])
    parser.add_argument('--revision', required=True)
    parser.add_argument('--attempt')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    if args.operation == 'prepare':
        if not args.attempt:
            parser.error('prepare requires --attempt')
        prepare(args.attempt, args.revision)
    else:
        if args.model is None or args.gpu is None:
            parser.error('gate/pilot/run require --model and --gpu')
        session(args.revision, args.model, args.gpu, args.operation)


if __name__ == '__main__':
    main()

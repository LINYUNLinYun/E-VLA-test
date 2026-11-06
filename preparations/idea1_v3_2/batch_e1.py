"""Bounded E1: bank numerical/throughput gates then paired actual-batch rollout."""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from preparations.idea1_v3_1.common import ROOT, PYTHONS, SUITES, atomic_json, environment, read_manifest
from preparations.idea1_v3_1.server import rpc
from .batch_policy import compare


def gpu_memory(gpu):
    command = ['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits']
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return dict(zip(('used_mib', 'total_mib'), map(int, result.stdout.strip().split(','))))


def worker(args):
    from preparations.idea1_v3_1.rollout import Environments, run_episode
    manifest = read_manifest(args.manifest)
    condition = next(c for c in manifest['conditions'] if c['id'] == 'pi0_full')
    pool = Environments(reuse=True)
    try:
        suite = SUITES[args.worker]
        for init in (0, 1):
            result = run_episode(Path(args.output)/args.arm/suite/f'init{init}', args.port, condition,
                suite, 0, init, pool, manifest['manifest_hash'], video=False, save_observations=False)
            print(json.dumps(result), flush=True)
    finally:
        pool.close()


def service(args):
    from .batch_policy import BatchPolicy
    from .batch_service import Aggregator, serve
    from preparations.idea1_v3_1.benchmark import load_banks
    from preparations.idea1_v3_1.models import raw_observation
    records, banks = load_banks([args.bank])
    policy = BatchPolicy()
    initial = records[0]
    from .batch_policy import noises
    tick = time.perf_counter()
    original = policy.policy.infer(raw_observation(initial['obs'], initial['instruction']), noise=noises([initial])[0])['actions']
    original_first_s = time.perf_counter()-tick
    decoded, raw, timing = policy.predict([initial])
    native = compare(original, decoded[0], atol=1e-5, rtol=1e-5)
    if not native['passed']:
        raise RuntimeError(f'Native B1 parity failed: {native}')
    manifest = read_manifest(args.manifest)
    protocol = dict(server_pid=os.getpid(), manifest_hash=manifest['manifest_hash'], condition_id='pi0_full',
        H=50, R=5, D=10, real_input_parity_passed=True, native_b1_parity=native, load_s=policy.load_s,
        bank=banks, backend='unchanged native JAX pi0 Full', batch_compiles=policy.compiles,
        original_policy_first_completed_s=original_first_s,
        native_b1_first_timing=timing, native_b1_first_raw_shape=list(raw.shape))
    atomic_json(Path(args.output)/'protocol.json', protocol)
    serve(args.port, Aggregator(policy, protocol, wait_ms=args.wait_ms))


def run(args):
    from preparations.idea1_v3_1.benchmark import load_banks
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started+args.deadline_seconds
    def check_budget():
        if time.monotonic() >= deadline:
            raise TimeoutError('E1 overall budget exhausted')
    receipt = dict(status='starting', started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        budget_seconds=args.deadline_seconds, controller_pid=os.getpid(), gpu=args.gpu,
        protocol='pi0 Full D10 H50 R5, B1/2/4, explicit per-example RNG noise; four env workers in every arm',
        official_scenes=[dict(suite=s, task=0, init=i) for s in SUITES for i in (0, 1)],
        arms=[], controls=[], bank_results=[])
    receipt['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT/'preparations/idea1_v3_2').glob('batch*.py'))}
    path = output/'result.json'
    def save():
        receipt['elapsed_s'] = time.monotonic()-started
        atomic_json(path, receipt)
    records, banks = load_banks([args.bank])
    receipt['banks'] = banks
    # Only the assigned physical GPU is inspected. A live compute user blocks launch.
    busy = subprocess.run(['nvidia-smi', '-i', str(args.gpu), '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout.strip()
    if busy:
        raise RuntimeError(f'Assigned GPU has compute PIDs: {busy}')
    command = [sys.executable, '-m', 'preparations.idea1_v3_2.batch_e1', '--service', '--bank', args.bank,
        '--manifest', args.manifest, '--output', args.output, '--port', str(args.port), '--wait-ms', str(args.wait_ms), '--launch']
    receipt['service_command'] = command
    save()
    children = []
    streams = []
    env = environment(args.gpu)
    env.update(XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.90')
    try:
        stream = (output/'service.log').open('x')
        streams.append(stream)
        server = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        children.append(server)
        receipt['service_pid'] = server.pid
        save()
        while True:
            if time.monotonic() > deadline or server.poll() is not None:
                raise RuntimeError('Service exited or load exceeded budget')
            try:
                protocol = rpc(args.port, dict(op='health'))['protocol']
                if protocol['server_pid'] != server.pid:
                    raise RuntimeError('Port belongs to another service')
                break
            except OSError:
                time.sleep(1)
        receipt['loaded_protocol'] = protocol
        receipt['cold_service_ready_s'] = time.monotonic()-started
        receipt['memory_after_load'] = gpu_memory(args.gpu)
        # Reuse exact same observations and explicit noise across all B values.
        references = []
        for record in records:
            check_budget()
            response = rpc(args.port, dict(op='control', records=[record]))
            references.append(response)
        n = len(records)-len(records)%4
        for size in (1, 2, 4):
            check_budget()
            first = rpc(args.port, dict(op='control', records=records[:size]))
            for _ in range(3):
                rpc(args.port, dict(op='control', records=records[:size]))
            rows, checks, durations = [], [], []
            begin = time.perf_counter()
            for offset in range(0, n, size):
                check_budget()
                tick = time.perf_counter()
                response = rpc(args.port, dict(op='control', records=records[offset:offset+size]))
                durations.append(time.perf_counter()-tick)
                for j in range(size):
                    checks.append(dict(index=offset+j,
                        decoded=compare(references[offset+j]['actions'][0], response['actions'][j]),
                        raw=compare(references[offset+j]['raw'][0], response['raw'][j])))
                rows.append(response['timing'])
            elapsed = time.perf_counter()-begin
            item = dict(batch_size=size, completed_examples=n, wall_s=elapsed,
                completed_examples_per_s=n/elapsed, first_request=first['timing'], warm_timings=rows,
                rpc_completed_ms_percentiles=(np.percentile(durations, [50, 95, 99])*1000).tolist(),
                checks=checks, numerical_passed=all(c['raw']['passed'] and c['decoded']['passed'] for c in checks),
                memory=gpu_memory(args.gpu),
                compile_stages=rpc(args.port, dict(op='health'))['protocol']['batch_compiles'])
            receipt['bank_results'].append(item)
            save()
        passed = [r for r in receipt['bank_results'] if r['numerical_passed']]
        winner = max(passed, key=lambda r: r['completed_examples_per_s'])['batch_size']
        receipt['selected_batch'] = winner
        if winner == 1:
            receipt['status'] = 'bank_complete_no_numerically_valid_batch_gain'
            receipt['closed_loop_skipped'] = 'B>1 did not pass numerical gate; approximate exploratory rollout requires an explicit recorded decision.'
            return
        # Dynamic aggregation can produce every tail size, including B3.
        receipt['tail_prewarm'] = []
        for size in range(1, winner+1):
            check_budget()
            for _ in range(3):
                receipt['tail_prewarm'].append(rpc(args.port, dict(op='control', records=records[:size]))['timing'])
        save()
        control_records = records[:4]
        reference = rpc(args.port, dict(op='control', records=control_records))
        for arm, maximum in [('b1_before', 1), ('selected', winner), ('b1_after', 1)]:
            if time.monotonic() > deadline-60:
                raise TimeoutError('Remaining budget insufficient for next closed-loop arm')
            configuration = rpc(args.port, dict(op='configure', maximum=maximum))
            check = rpc(args.port, dict(op='control', records=control_records))
            receipt['controls'].append(dict(arm=arm, when='before', decoded=compare(reference['actions'], check['actions']),
                                            raw=compare(reference['raw'], check['raw'])))
            if not receipt['controls'][-1]['raw']['passed']:
                raise RuntimeError('Fixed same-service control failed')
            arm_result = dict(arm=arm, maximum=maximum, workers=4, pids=[], completed_episodes=0)
            receipt['arms'].append(arm_result)
            begin = time.perf_counter()
            workers = []
            for i in range(4):
                cmd = [str(PYTHONS['environment']), '-m', 'preparations.idea1_v3_2.batch_e1', '--worker', str(i),
                    '--manifest', args.manifest, '--output', args.output, '--port', str(args.port), '--arm', arm, '--launch']
                stream = (output/f'{arm}_worker{i}.log').open('x')
                streams.append(stream)
                child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                children.append(child)
                workers.append(child)
                arm_result['pids'].append(child.pid)
            save()
            while any(w.poll() is None for w in workers):
                if time.monotonic() > deadline:
                    raise TimeoutError('Closed-loop deadline')
                if any(w.poll() not in (None, 0) for w in workers):
                    raise RuntimeError('Owned environment worker failed; inspect log')
                time.sleep(1)
            if any(w.returncode for w in workers):
                raise RuntimeError('Owned environment worker failed')
            arm_result['wall_s'] = time.perf_counter()-begin
            counts = rpc(args.port, dict(op='configure', maximum=1))['actual_batches']
            arm_result['actual_batch_histogram'] = dict(Counter(counts))
            results = [json.loads(p.read_text()) for p in (output/arm).glob('*/init*/result.json')]
            arm_result.update(completed_episodes=len(results), successes=sum(r['success'] for r in results),
                completed_actions=sum(r['actions'] for r in results), predictions=sum(r['predictions'] for r in results))
            arm_result['episodes_per_hour'] = len(results)*3600/arm_result['wall_s']
            check = rpc(args.port, dict(op='control', records=control_records))
            receipt['controls'].append(dict(arm=arm, when='after', decoded=compare(reference['actions'], check['actions']),
                                            raw=compare(reference['raw'], check['raw'])))
            save()
        receipt['status'] = 'completed'
    except BaseException as exc:
        receipt.update(status='failed_or_budget_limited', error=repr(exc))
        raise
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for stream in streams:
            stream.close()
        receipt['owned_process_exit_codes'] = {str(c.pid): c.returncode for c in children}
        save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', default='results/idea1_v3_1/input_bank.pkl')
    parser.add_argument('--manifest', default='results/idea1_v3_1/manifest_v2.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--port', type=int, default=19332)
    parser.add_argument('--deadline-seconds', type=int, default=2700)
    parser.add_argument('--wait-ms', type=float, default=40)
    parser.add_argument('--arm')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--service', action='store_true')
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    if not args.launch:
        print(json.dumps(dict(dry_run=True, **vars(args))))
        return
    if args.worker is not None:
        worker(args)
    elif args.service:
        service(args)
    else:
        run(args)


if __name__ == '__main__':
    main()

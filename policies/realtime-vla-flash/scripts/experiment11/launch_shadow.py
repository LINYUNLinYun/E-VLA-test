"""Dry-run or launch the bounded Experiment 4 Full-only shadow replay queue."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time

from scripts.experiment11.protocol import PROTOCOL, RESULTS, ROOT, SUITES, environment

os.environ['EXP11_RESULTS_ROOT'] = str(RESULTS)
os.environ['EXP11_PROTOCOL_JSON'] = json.dumps(PROTOCOL, separators=(',', ':'))

from scripts.experiment10.common import claim_gpu, dump_json, managed_gpus, sha256
from scripts.experiment10.run_pipeline import free_gpus


STATUS = RESULTS / 'experiment11_shadow_status.json'
LOCK = STATUS.with_name(STATUS.name + '.scheduler.lock')
BASELINE = 'Full18_D3_R5_FORMAL'
MAX_WORKERS = 4
POLL_SECONDS = 300
LOG_ROOT = RESULTS / 'logs/shadow'


def live_pid(pid: int) -> bool:
    try:
        state = (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1][0]
    except (FileNotFoundError, IndexError, ValueError):
        return False
    return state not in ('Z', 'X')


class AdoptedWorker:
    """Poll a worker owned by a previous scheduler process without reaping it."""

    def __init__(self, pid: int, job: dict):
        self.pid = pid
        self.job = job

    def poll(self):
        if live_pid(self.pid):
            return None
        return 0 if validate_job(self.job) else 1


def frozen_jobs() -> tuple[list[dict], str, str]:
    d3_status_path = RESULTS / 'experiment11_closed_loop_status.json'
    d3_status = json.loads(d3_status_path.read_text())
    audit_path = RESULTS / 'analysis/closed_loop_d3/completion_audit.json'
    if (not d3_status.get('complete') or d3_status.get('active') or d3_status.get('pending') or
            d3_status.get('failures') or not audit_path.is_file()):
        raise RuntimeError('The complete, clean D3 matrix and its audit are required before shadow replay')
    audit = json.loads(audit_path.read_text())
    if audit.get('condition_suite_runs_validated') != 124 or audit.get('episodes_validated') != 6200:
        raise RuntimeError('D3 audit does not prove 124 validated suite runs / 6200 episodes')

    selection_path = RESULTS / 'protocol/selected_shadow_manifest.json'
    selection = json.loads(selection_path.read_text())
    if selection.get('status') != 'eligible' or not selection.get('selected_new_masks'):
        raise RuntimeError('No new layer-skip mask passed the 80% gate; shadow is not eligible')
    if len(selection['selected_new_masks']) > 4:
        raise RuntimeError('Shadow selection exceeds the four-new-mask limit')
    candidate_manifest = RESULTS / 'protocol/candidate_manifest.json'
    manifest_hash = sha256(selection_path)
    audit_hash = sha256(audit_path)

    selected = list(selection['selected_new_masks']) + [selection['original_v13_reference']]
    if len({row['variant_id'] for row in selected}) != len(selected) or len(selected) > 5:
        raise RuntimeError('Shadow selection must be at most four new masks plus one original V13 reference')
    jobs = []
    for variant in selected:
        variant_id = str(variant['variant_id'])
        skip = [int(value) for value in variant['skip_layers']]
        if not skip or len(set(skip)) != len(skip) or any(value < 1 or value > 18 for value in skip):
            raise RuntimeError(f'Invalid frozen skip mask for {variant_id}: {skip}')
        for suite in SUITES:
            jobs.append({'job_id': f'{variant_id}__{suite}', 'variant_id': variant_id,
                         'suite': suite, 'skip_layers': skip,
                         'condition_name': str(variant['condition_name'])})
    return jobs, manifest_hash, audit_hash


def output_paths(job: dict) -> tuple[Path, Path, Path]:
    directory = RESULTS / 'shadow' / job['variant_id'] / job['suite']
    return directory / 'predictions.jsonl', directory / 'validated.json', directory / 'predictions.jsonl.partial'


def expected_prediction_count(suite: str) -> int:
    episode_path = RESULTS / 'closed_loop' / BASELINE / suite / 'run/episode_log.json'
    episodes = json.loads(episode_path.read_text())
    if len(episodes) != 50:
        raise RuntimeError(f'{suite}: expected 50 Full18 episodes, found {len(episodes)}')
    count = sum(int(row.get('infer_record_count', 0)) for row in episodes)
    if count <= 0 or any(int(row.get('infer_record_count', 0)) <= 0 for row in episodes):
        raise RuntimeError(f'{suite}: Full18 episode log has missing replan records')
    return count


def validate_job(job: dict) -> bool:
    predictions, marker_path, partial = output_paths(job)
    if not predictions.is_file() or not marker_path.is_file():
        return False
    marker = json.loads(marker_path.read_text())
    expected = expected_prediction_count(job['suite'])
    selection_path = RESULTS / 'protocol/selected_shadow_manifest.json'
    candidate_path = RESULTS / 'protocol/candidate_manifest.json'
    source_hashes = marker.get('source_hashes') or {}
    parity_error = marker.get('full_recomputed_parity_max_abs_error')
    return bool(
        marker.get('complete') is True and marker.get('variant_id') == job['variant_id'] and
        marker.get('suite') == job['suite'] and marker.get('skip_layers') == job['skip_layers'] and
        marker.get('condition_name') == job['condition_name'] and
        marker.get('episode_count') == 50 and marker.get('replan_prediction_count') == expected and
        marker.get('action_horizon') == 50 and marker.get('denoise_steps') == 3 and
        marker.get('replan_execute_prefix') == 5 and
        marker.get('environment_steps_executed_by_variant') == 0 and
        isinstance(parity_error, (int, float)) and math.isfinite(float(parity_error)) and
        source_hashes.get('full_episode_log_sha256') == sha256(
            RESULTS / 'closed_loop' / BASELINE / job['suite'] / 'run/episode_log.json') and
        source_hashes.get('selected_shadow_manifest_sha256') == sha256(selection_path) and
        source_hashes.get('candidate_manifest_sha256') == sha256(candidate_path) and
        marker.get('predictions_sha256') == sha256(predictions) and not partial.exists()
    )


def inspect_job_output(job: dict) -> bool:
    """Return True if validated; refuse to overwrite any partial/invalid output."""
    predictions, marker, partial = output_paths(job)
    any_output = predictions.exists() or marker.exists() or partial.exists()
    if not any_output:
        return False
    if not validate_job(job):
        raise RuntimeError(f"Existing shadow output is incomplete/invalid; audit before resume: {job['job_id']}")
    return True


def child_command(job: dict, gpu: int) -> list[str]:
    return [str(ROOT / '.venv/bin/python'), '-m', 'scripts.experiment11.shadow_worker',
            '--variant-id', job['variant_id'], '--suite', job['suite'], '--gpu-index', str(gpu)]


def run_manager(jobs: list[dict], selection_hash: str, audit_hash: str) -> int:
    lock_stream = LOCK.open('a')
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f'A shadow scheduler already owns {STATUS}') from exc

    previous = json.loads(STATUS.read_text()) if STATUS.exists() else None
    if previous and previous.get('complete'):
        raise RuntimeError('Shadow matrix is already complete; refusing duplicate launch')
    if previous and previous.get('failures'):
        raise RuntimeError('Previous shadow run has failures; audit them before any retry')
    if previous and (previous.get('selection_sha256') != selection_hash or
                     previous.get('d3_audit_sha256') != audit_hash):
        raise RuntimeError('Frozen selection or D3 audit changed since the prior shadow run')

    job_by_id = {job['job_id']: job for job in jobs}
    active: dict[int, tuple[object, dict, object]] = {}
    leases: dict[int, object] = {}
    adopted_ids: set[str] = set()
    if previous:
        for record in previous.get('active', []):
            job_id = str(record.get('job_id'))
            if job_id not in job_by_id:
                raise RuntimeError(f'Unknown active shadow job in prior status: {record}')
            job = job_by_id[job_id]
            pid, gpu = int(record['pid']), int(record['gpu'])
            if live_pid(pid):
                lease = claim_gpu(gpu)
                if lease is None:
                    raise RuntimeError(f'Cannot adopt live shadow worker on GPU {gpu}; lease is held')
                leases[gpu] = lease
                active[gpu] = (AdoptedWorker(pid, job), job, open(os.devnull, 'w'))
                adopted_ids.add(job_id)
            elif not inspect_job_output(job):
                raise RuntimeError(f'Prior shadow worker {pid} died without a validated result: {job_id}')

    completed_ids = {job['job_id'] for job in jobs if inspect_job_output(job)}
    pending = [job for job in jobs if job['job_id'] not in completed_ids and job['job_id'] not in adopted_ids]
    failures = []
    available: set[int] = set(globals().get('INITIAL_AVAILABLE', ()))
    previous_scan_epoch = float(previous.get('last_gpu_scan_epoch', 0.0)) if previous else 0.0
    discovery_clock = RESULTS / 'next_round_gpu_discovery.json'
    if discovery_clock.exists():
        previous_scan_epoch = max(previous_scan_epoch, float(json.loads(discovery_clock.read_text()).get('last_gpu_scan_epoch',0)))
    elapsed_since_scan = max(0.0, time.time() - previous_scan_epoch) if previous_scan_epoch else POLL_SECONDS
    next_scan = time.monotonic() + max(0.0, POLL_SECONDS - elapsed_since_scan)
    last_scan_epoch = previous_scan_epoch
    worker_limit = min(MAX_WORKERS, int(PROTOCOL['max_workers']))

    def write_status(state: str = 'running') -> None:
        dump_json(STATUS, {
            'schema': 1, 'experiment_id': 'experiment11_shadow_prefix_d3', 'state': state,
            'pid': os.getpid(), 'complete': not pending and not active and not failures,
            'selection_sha256': selection_hash, 'd3_audit_sha256': audit_hash,
            'last_gpu_scan_epoch': last_scan_epoch,
            'expected_jobs': len(jobs), 'validated_jobs': len(completed_ids),
            'active': [{'pid': process.pid, 'gpu': gpu, 'job_id': job['job_id'],
                        'variant_id': job['variant_id'], 'suite': job['suite']}
                       for gpu, (process, job, _) in sorted(active.items())],
            'pending': [job['job_id'] for job in pending], 'failures': failures,
        })

    write_status()
    while pending or active:
        for gpu, (process, job, stream) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            if hasattr(stream, 'close'):
                stream.close()
            del active[gpu]
            lease = leases.pop(gpu, None)
            if lease is not None:
                lease.close()
            if code == 0 and validate_job(job):
                completed_ids.add(job['job_id'])
                available.add(gpu)  # Reuse a GPU just released by our own worker.
                print(json.dumps({'event': 'shadow_job_validated', 'gpu': gpu,
                                  'job_id': job['job_id']}), flush=True)
            else:
                failures.append({'job_id': job['job_id'], 'gpu': gpu, 'exit_code': code,
                                 'reason': 'worker failed or output validation failed'})
                print(json.dumps({'event': 'shadow_job_failed', 'gpu': gpu,
                                  'job_id': job['job_id'], 'exit_code': code}), flush=True)

        if failures and not active:
            break

        managed = managed_gpus()
        # Only scan for externally available devices every five minutes. If all
        # four GPUs belong to this experiment, worker completion reuses its GPU.
        if pending and len(managed) < worker_limit and time.monotonic() >= next_scan and not failures:
            available.update(g for g in free_gpus() if g not in managed)
            last_scan_epoch = time.time()
            next_scan = time.monotonic() + POLL_SECONDS

        if not failures:
            for gpu in sorted(available - set(active) - managed):
                if not pending or len(active) >= worker_limit:
                    break
                job = pending.pop(0)
                lease = claim_gpu(gpu)
                if lease is None:
                    pending.insert(0, job)
                    available.discard(gpu)
                    continue
                _, _, partial = output_paths(job)
                if partial.exists() or inspect_job_output(job):
                    lease.close()
                    pending.insert(0, job)
                    available.discard(gpu)
                    raise RuntimeError(f"Refusing to overwrite existing shadow output for {job['job_id']}")
                log_path = LOG_ROOT / job['variant_id'] / f"{job['suite']}.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                stream = log_path.open('a', encoding='utf-8')
                env = environment()
                env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                process = subprocess.Popen(child_command(job, gpu), cwd=ROOT, env=env,
                                           stdout=stream, stderr=subprocess.STDOUT)
                active[gpu] = (process, job, stream)
                leases[gpu] = lease
                available.discard(gpu)
                print(json.dumps({'event': 'shadow_job_started', 'gpu': gpu,
                                  'job_id': job['job_id'], 'pid': process.pid}), flush=True)

        write_status('failed' if failures else 'running')
        if pending or active:
            time.sleep(2)

    complete = not failures and not pending and not active and len(completed_ids) == len(jobs)
    write_status('complete' if complete else 'failed')
    for lease in leases.values():
        lease.close()
    lock_stream.close()
    return 0 if complete else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true', help='Start the resumable shadow GPU scheduler')
    args = parser.parse_args()
    jobs, selection_hash, audit_hash = frozen_jobs()
    existing = sum(inspect_job_output(job) for job in jobs)
    print(json.dumps({'dry_run': not args.launch, 'job_count': len(jobs),
                      'validated_jobs': existing, 'pending_jobs': len(jobs) - existing,
                      'max_workers': min(MAX_WORKERS, int(PROTOCOL['max_workers'])),
                      'selection_sha256': selection_hash}, indent=2))
    if not args.launch:
        return 0
    return run_manager(jobs, selection_hash, audit_hash)


if __name__ == '__main__':
    raise SystemExit(main())

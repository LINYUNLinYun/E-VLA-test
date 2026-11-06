"""Single-sample Full/variant parity smoke for the frozen Experiment 11 shadow."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np

from scripts.experiment11.protocol import PROTOCOL, RESULTS, ROOT, SOURCE, SUITES, environment

os.environ['EXP11_RESULTS_ROOT'] = str(RESULTS)
os.environ['EXP11_PROTOCOL_JSON'] = json.dumps(PROTOCOL, separators=(',', ':'))

from scripts.experiment10.common import claim_gpu, dump_json, sha256
from scripts.experiment10.run_pipeline import free_gpus


def validate_attempt_id(value: str) -> str:
    if not value or value in {'.', '..'} or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', value):
        raise ValueError('attempt-id must be 1..64 safe filename characters')
    return value


def input_digest(data, prompt: str, suite: str) -> str:
    digest = hashlib.sha256()
    for name in ('agent_image', 'wrist_image', 'robot_state', 'noise'):
        digest.update(np.ascontiguousarray(data[name]).tobytes())
    digest.update(prompt.encode('utf-8'))
    digest.update(suite.encode('utf-8'))
    return digest.hexdigest()


def run_worker(attempt_id: str, gpu: int, variant_id: str, suite: str, task: int, init: int) -> int:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise RuntimeError('Shadow smoke worker is not bound to its leased GPU')
    if suite not in SUITES or not (0 <= task < 10 and 0 <= init < 5):
        raise ValueError('Smoke sample must use a standard suite, task0-9, init0-4')
    import torch
    from scripts.experiment10.backend import Backend

    torch.cuda.init()
    selected = json.loads((RESULTS / 'protocol/selected_shadow_manifest.json').read_text())
    if selected.get('status') != 'eligible':
        raise RuntimeError('A frozen eligible Shadow manifest is required')
    variants = {row['variant_id']: row for row in selected['selected_new_masks']}
    variants[selected['original_v13_reference']['variant_id']] = selected['original_v13_reference']
    if variant_id not in variants:
        raise RuntimeError(f'{variant_id} is not in the frozen Shadow manifest')
    candidate = variants[variant_id]
    skip = tuple(int(layer) for layer in candidate['skip_layers'])

    full_run = RESULTS / 'closed_loop' / 'Full18_D3_R5_FORMAL' / suite / 'run'
    episodes = json.loads((full_run / 'episode_log.json').read_text())
    episode = next(row for row in episodes
                   if int(row['task_id']) == task and int(row['episode_idx']) == init)
    infer_path = full_run / episode['infer_path']
    with infer_path.open(encoding='utf-8') as stream:
        saved_full = json.loads(next(stream))
    input_path = RESULTS / 'full_observations' / suite / f'task{task:02d}' / f'init{init:02d}' / 'replan0000.npz'
    checkpoint_hash = json.loads((RESULTS / 'protocol/candidate_manifest.json').read_text())['checkpoint_sha256']
    with np.load(input_path, allow_pickle=False) as data:
        input_meta = json.loads(str(data['metadata_json'].item()))
        prompt = str(data['prompt'].item())
        noise = np.asarray(data['noise'], dtype=np.float32).copy()
        digest = input_digest(data, prompt, suite)
        noise_hash = hashlib.sha256(noise.tobytes()).hexdigest()
        if (input_meta.get('suite') != suite or int(input_meta.get('task_id', -1)) != task or
                int(input_meta.get('episode_idx', -1)) != init or
                int(input_meta.get('infer_id', -1)) != int(saved_full['infer_id']) or
                input_meta.get('checkpoint_sha256') != checkpoint_hash or
                input_meta.get('skip_layers') != [] or input_meta.get('denoise_steps') != 3 or
                digest != input_meta.get('input_sha256') or
                noise_hash != input_meta.get('noise_sha256')):
            raise RuntimeError(f'Smoke input provenance mismatch: {input_path}')
        saved_meta = saved_full.get('experiment11', {})
        if (saved_meta.get('condition') != 'Full18_D3_R5_FORMAL' or
                saved_meta.get('checkpoint_sha256') != checkpoint_hash or
                saved_meta.get('full_input_sha256') != digest or
                saved_meta.get('noise_sha256') != noise_hash or
                int(saved_full.get('accepted_prefix_len', -1)) != 5):
            raise RuntimeError(f'Saved Full inference does not match the selected input: {infer_path}')
        observation = {
            'observation/image': data['agent_image'],
            'observation/wrist_image': data['wrist_image'],
            'observation/state': data['robot_state'], 'prompt': prompt,
        }

    source_rows = json.loads((SOURCE / 'observations' / suite / 'manifest.json').read_text())['observations']
    source_metadata = []
    for row in source_rows:
        if row.get('task_id') in range(10):
            with np.load(row['path'], allow_pickle=False) as data:
                source_metadata.append(json.loads(str(data['metadata_json'].item())))
    backend = Backend(source_metadata)
    meta = {'suite': suite, 'task_description': prompt}
    backend.prepare(observation, meta, noise)
    expected = np.asarray(saved_full['chunk_actions'], dtype=np.float32)
    if expected.shape != (50, 7) or not np.isfinite(expected).all():
        raise RuntimeError('Saved Full action chunk must be finite 50x7')

    _, full_before, full_timing = backend.infer((), 3)
    full_before = np.asarray(full_before, dtype=np.float32)
    if full_before.shape != expected.shape or not np.allclose(full_before, expected, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f'Full parity failed before variant inference for {suite}/{task}/{init}')
    _, variant_actions, variant_timing = backend.infer(skip, 3)
    variant_actions = np.asarray(variant_actions, dtype=np.float32)
    if variant_actions.shape != (50, 7) or not np.isfinite(variant_actions).all():
        raise RuntimeError('Variant smoke output must be finite 50x7')
    _, full_after, _ = backend.infer((), 3)
    full_after = np.asarray(full_after, dtype=np.float32)
    if full_after.shape != expected.shape or not np.allclose(full_after, expected, rtol=1e-4, atol=1e-5):
        raise RuntimeError(f'Variant inference changed Full output for {suite}/{task}/{init}')

    difference = variant_actions - full_before
    output_dir = RESULTS / 'shadow_smoke' / attempt_id
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / 'result.json'
    if result_path.exists():
        raise FileExistsError(f'Refusing to replace existing Shadow smoke result: {result_path}')
    dump_json(result_path, {
        'complete': True, 'attempt_id': attempt_id, 'gpu_index': gpu,
        'variant_id': variant_id, 'skip_layers': list(skip), 'suite': suite,
        'task_id': task, 'init_index': init, 'action_horizon': 50, 'denoise_steps': 3,
        'replan_execute_prefix': 5, 'environment_steps_executed_by_variant': 0,
        'input_path': str(input_path), 'input_sha256': digest, 'noise_sha256': noise_hash,
        'full_infer_id': int(saved_full['infer_id']), 'checkpoint_sha256': checkpoint_hash,
        'full_before_max_abs_parity_error': float(np.max(np.abs(full_before - expected))),
        'full_after_max_abs_parity_error': float(np.max(np.abs(full_after - expected))),
        'variant_vs_full': {
            'translation_rmse': float(np.sqrt(np.mean(difference[:, :3] ** 2))),
            'rotation_rmse': float(np.sqrt(np.mean(difference[:, 3:6] ** 2))),
            'gripper_rmse': float(np.sqrt(np.mean(difference[:, 6] ** 2))),
            'max_abs': float(np.max(np.abs(difference))),
        },
        'timing_ms': {'full_cuda_policy': float(full_timing['cuda_policy_ms']),
                      'variant_cuda_policy': float(variant_timing['cuda_policy_ms'])},
        'full_episode_log_sha256': sha256(full_run / 'episode_log.json'),
    })
    print(json.dumps({'event': 'shadow_smoke_complete', 'result': str(result_path),
                      'variant_id': variant_id, 'suite': suite,
                      'full_parity_max_abs': float(np.max(np.abs(full_before - expected))),
                      'full_repeat_max_abs': float(np.max(np.abs(full_after - expected)))}, indent=2), flush=True)
    return 0


def manager(attempt_id: str, variant_id: str, suite: str, task: int, init: int) -> int:
    attempt_id = validate_attempt_id(attempt_id)
    if suite not in SUITES or not (0 <= task < 10 and 0 <= init < 5):
        raise ValueError('Smoke sample must use a standard suite, task0-9, init0-4')
    output_dir = RESULTS / 'shadow_smoke' / attempt_id
    if output_dir.exists():
        raise FileExistsError(f'Shadow smoke attempt already exists: {output_dir}')
    output_dir.mkdir(parents=True)
    status_path = output_dir / 'status.json'
    state = {'attempt_id': attempt_id, 'state': 'waiting_for_idle_gpu', 'active': None,
             'failures': [], 'variant_id': variant_id, 'suite': suite, 'task_id': task, 'init_index': init}
    dump_json(status_path, state)
    lease = None
    while lease is None:
        for gpu in free_gpus():
            lease = claim_gpu(gpu)
            if lease is not None and gpu in free_gpus():
                break
            if lease is not None:
                lease.close()
                lease = None
        if lease is None:
            print(json.dumps({'event': 'shadow_smoke_waiting_for_idle_gpu', 'sleep_seconds': 300}), flush=True)
            time.sleep(300)
    env = environment()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env['EXP11_METADATA_KEY'] = 'experiment11'
    env['EXP11_SOURCE_RESULTS'] = str(SOURCE)
    command = [str(ROOT / '.venv/bin/python'), '-m', 'scripts.experiment11.shadow_smoke',
               '--worker', '--attempt-id', attempt_id, '--gpu-index', str(gpu),
               '--variant-id', variant_id, '--suite', suite,
               '--task-id', str(task), '--init-index', str(init)]
    log_path = output_dir / 'worker.log'
    try:
        with log_path.open('x', encoding='utf-8') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            state['state'] = 'running'
            state['active'] = {'pid': process.pid, 'gpu': gpu}
            dump_json(status_path, state)
            code = process.wait()
        state['active'] = None
        if code == 0 and (output_dir / 'result.json').is_file():
            state['state'] = 'complete'
            state['result'] = str(output_dir / 'result.json')
        else:
            state['state'] = 'failed'
            state['failures'] = [{'pid': process.pid, 'exit_code': code, 'log': str(log_path)}]
        dump_json(status_path, state)
        return code
    finally:
        lease.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true', help='Run one isolated Full/variant GPU smoke')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--attempt-id', required=True)
    parser.add_argument('--variant-id', default='V16_A02')
    parser.add_argument('--suite', choices=SUITES, default='libero_spatial')
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--init-index', type=int, default=0)
    parser.add_argument('--gpu-index', type=int)
    args = parser.parse_args()
    validate_attempt_id(args.attempt_id)
    if args.worker:
        if args.gpu_index is None:
            raise ValueError('Smoke worker requires --gpu-index')
        return run_worker(args.attempt_id, args.gpu_index, args.variant_id, args.suite,
                          args.task_id, args.init_index)
    preview = {'dry_run': not args.launch, 'attempt_id': args.attempt_id,
               'variant_id': args.variant_id, 'suite': args.suite,
               'task_id': args.task_id, 'init_index': args.init_index,
               'replan_input': 'Full replan0000; same state/prompt/noise',
               'sequence': ['Full D3', 'variant D3', 'Full D3 repeat'],
               'environment_steps_executed_by_variant': 0}
    print(json.dumps(preview, indent=2), flush=True)
    return manager(args.attempt_id, args.variant_id, args.suite, args.task_id, args.init_index) if args.launch else 0


if __name__ == '__main__':
    raise SystemExit(main())

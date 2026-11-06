#!/usr/bin/env python3
"""Paired Full18 vs Attention-L16 closed-loop screen across five official inits."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiment10.common import (  # noqa: E402
    RESULTS as EXP10_RESULTS, SUITES as EXP10_SUITES, claim_gpu, dump_json,
    managed_gpus, sha256,
)
from scripts.experiment10.run_pipeline import event, free_gpus  # noqa: E402

CONFIG_DEFAULT = Path(__file__).with_name('attn16_multi_init_config.json')
OUTPUT_ROOT = ROOT / 'results/pi0_attention_ffn_diagnosis/closed_loop_multi_init'
CONDITION_DEFAULT = 'AttnL16_T12_Init0to4_seed12345'
STATUS_DEFAULT = EXP10_RESULTS / 'attn16_multi_init_status.json'
SUITES = ('libero_90', 'libero_10')
BASE_PYTHON = ROOT / '.venv/bin/python'
CLIENT_PYTHON = ROOT / 'examples/libero/.venv/bin/python'
SERVER_SCRIPT = ROOT / 'scripts/experiment8/serve_submodule_policy.py'
CLIENT_SCRIPT = ROOT / 'scripts/experiment8/run_submodule_rollout.py'


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def reserve_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def server_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        'CUDA_VISIBLE_DEVICES': str(gpu),
        'JAX_PLATFORMS': 'cpu',
        'OPENPI_DATA_HOME': '/root/nfs/embodied/benchmark/shared/checkpoints/openpi',
        'PYTHONPATH': ':'.join((str(ROOT / 'src'), str(ROOT))),
        'PYTHONUNBUFFERED': '1',
        'TOKENIZERS_PARALLELISM': 'false',
    })
    return env


def client_env(gpu: int) -> dict[str, str]:
    env = server_env(gpu)
    env.update({
        'MUJOCO_GL': 'egl',
        'MUJOCO_EGL_DEVICE_ID': str(gpu),
        'LIBERO_CONFIG_PATH': str(ROOT / '.libero'),
        'PYTHONPATH': ':'.join((str(ROOT / 'src'), str(ROOT),
                                '/root/nfs/embodied/benchmark/LIBERO')),
    })
    return env


def wait_server(process: subprocess.Popen, port: int, role: str, started: float) -> None:
    from websockets.sync.client import connect

    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError('%s server exited before ready (code=%s)' % (role, code))
        try:
            with connect('ws://127.0.0.1:%d' % port, open_timeout=1,
                         compression=None, max_size=None) as connection:
                connection.recv()
                event(event='submodule_multi_init_server_ready', role=role, port=port,
                      elapsed_s=round(time.monotonic() - started, 2))
                return
        except (OSError, TimeoutError):
            time.sleep(2)
    raise TimeoutError('%s server did not become ready in 600 seconds' % role)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def candidate_dir_name(config: dict) -> str:
    layers = config['candidate_skip_attn_layers']
    return 'attn' + '_'.join(str(layer) for layer in layers)


def candidate_label(config: dict) -> str:
    layers = config['candidate_skip_attn_layers']
    return 'SkipAttnL' + '_'.join(str(layer) for layer in layers)


def suite_config_matches(summary_path: Path, suite: str, config: dict) -> bool:
    """Reuse only completed suite results whose suite-specific inputs still match."""
    if not summary_path.is_file():
        return False
    try:
        saved = json.loads(summary_path.read_text())
        old = saved.get('configuration', {})
        old_manifests = old.get('prompt_manifests', {})
        old_manifest = old_manifests.get(suite, old.get('prompt_manifest'))
        expected = {
            'checkpoint_sha256': config['checkpoint_sha256'],
            'base_triton_path': config['base_triton_path'],
            'noise_seed': config['noise_seed'], 'env_seed': config['env_seed'],
            'warmup_steps': config['warmup_steps'], 'replan_steps': config['replan_steps'],
            'init_indices': config['init_indices'],
            'tasks': {suite: config['tasks'][suite]},
            'candidate_skip_attn_layers': config['candidate_skip_attn_layers'],
            'candidate_skip_ffn_layers': config['candidate_skip_ffn_layers'],
            'runtime': config['runtime'],
        }
        actual = {key: old.get(key) for key in expected if key != 'tasks'}
        actual['tasks'] = {suite: old.get('tasks', {}).get(suite)}
        return (saved.get('complete') is True and actual == expected and
                Path(str(old_manifest)).resolve() == Path(config['prompt_manifests'][suite]).resolve())
    except (OSError, ValueError, TypeError, KeyError):
        return False


def validate_video(path: Path, expected_frames: int) -> None:
    probe = subprocess.run([
        'ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,pix_fmt,nb_read_frames', '-of', 'json', str(path),
    ], check=True, capture_output=True, text=True)
    streams = json.loads(probe.stdout).get('streams', [])
    if len(streams) != 1:
        raise RuntimeError('Expected one video stream in %s' % path)
    stream = streams[0]
    frames = int(stream.get('nb_read_frames', -1))
    if stream.get('codec_name') != 'h264' or stream.get('pix_fmt') != 'yuv420p' or frames != expected_frames:
        raise RuntimeError('Video protocol/frame mismatch in %s: %s' % (path, stream))
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def validate_pair(pair_root: Path, suite: str, task_id: int, init_index: int,
                  config: dict, decode_videos: bool = True) -> dict:
    baseline_dir = pair_root / 'baseline'
    candidate_dir = pair_root / candidate_dir_name(config)
    baseline = json.loads((baseline_dir / 'summary.json').read_text())
    candidate = json.loads((candidate_dir / 'summary.json').read_text())
    expected_common = {
        'official_init_index': init_index,
        'noise_seed': config['noise_seed'],
        'env_seed': config['env_seed'],
        'warmup_steps': config['warmup_steps'],
        'replan_steps': config['replan_steps'],
        'denoise_steps': 10,
        'runtime': config['runtime'],
    }
    for role, summary, expected_attn in (
        ('baseline', baseline, []),
        ('candidate', candidate, config['candidate_skip_attn_layers']),
    ):
        for key, value in expected_common.items():
            if summary.get(key) != value:
                raise RuntimeError('%s %s mismatch for %s' % (role, key, pair_root))
        if summary.get('skip_attn_layers') != expected_attn or summary.get('skip_ffn_layers') != config['candidate_skip_ffn_layers']:
            raise RuntimeError('%s intervention metadata mismatch for %s' % (role, pair_root))
        if summary.get('state_count') != summary.get('executed_actions', -1) + 1:
            raise RuntimeError('%s N+1 state count mismatch for %s' % (role, pair_root))
        if summary.get('video_frame_count') != summary.get('state_count'):
            raise RuntimeError('%s video/state frame mismatch for %s' % (role, pair_root))
    if baseline.get('init_state_sha256') != candidate.get('init_state_sha256'):
        raise RuntimeError('Paired official initial states differ: %s' % pair_root)
    if baseline.get('semantic_task_key') != candidate.get('semantic_task_key'):
        raise RuntimeError('Paired semantic task identities differ: %s' % pair_root)

    baseline_inferences = read_jsonl(baseline_dir / 'inferences.jsonl')
    candidate_inferences = read_jsonl(candidate_dir / 'inferences.jsonl')
    base_by_id = {int(row['infer_id']): row['metadata'] for row in baseline_inferences}
    cand_by_id = {int(row['infer_id']): row['metadata'] for row in candidate_inferences}
    shared_ids = sorted(set(base_by_id) & set(cand_by_id))
    if not shared_ids:
        raise RuntimeError('No shared inference IDs in paired episode %s' % pair_root)
    for infer_id in shared_ids:
        base_meta, cand_meta = base_by_id[infer_id], cand_by_id[infer_id]
        if base_meta.get('noise_sha256') != cand_meta.get('noise_sha256'):
            raise RuntimeError('Paired noise mismatch at infer %d in %s' % (infer_id, pair_root))
        if base_meta.get('noise_identity') != cand_meta.get('noise_identity'):
            raise RuntimeError('Paired noise identity mismatch at infer %d in %s' % (infer_id, pair_root))
        if base_meta.get('skip_attn_layers') != [] or cand_meta.get('skip_attn_layers') != config['candidate_skip_attn_layers']:
            raise RuntimeError('Server intervention metadata mismatch at infer %d' % infer_id)

    base_xyz = np.load(baseline_dir / 'trajectory.npz', allow_pickle=False)['eef_xyz']
    cand_xyz = np.load(candidate_dir / 'trajectory.npz', allow_pickle=False)['eef_xyz']
    shared_steps = min(len(base_xyz), len(cand_xyz))
    distances = np.linalg.norm(base_xyz[:shared_steps] - cand_xyz[:shared_steps], axis=1)
    if not len(distances) or not np.isfinite(distances).all():
        raise RuntimeError('Invalid paired EEF trajectory in %s' % pair_root)
    if decode_videos:
        validate_video(baseline_dir / 'rollout.mp4', int(baseline['video_frame_count']))
        validate_video(candidate_dir / 'rollout.mp4', int(candidate['video_frame_count']))

    return {
        'suite': suite,
        'task_id': task_id,
        'init_index': init_index,
        'baseline_success': int(bool(baseline['success'])),
        'candidate_success': int(bool(candidate['success'])),
        'outcome': ('both_success' if baseline['success'] and candidate['success'] else
                    'baseline_only' if baseline['success'] else
                    'candidate_only' if candidate['success'] else 'both_failure'),
        'baseline_steps': baseline['executed_actions'],
        'candidate_steps': candidate['executed_actions'],
        'shared_inference_count': len(shared_ids),
        'shared_state_count': shared_steps,
        'eef_deviation_mean_m': float(np.mean(distances)),
        'eef_deviation_p95_m': float(np.percentile(distances, 95)),
        'eef_deviation_max_m': float(np.max(distances)),
        'eef_max_deviation_state_index': int(np.argmax(distances)),
        'baseline_video_validated': True,
        'candidate_video_validated': True,
        'initial_state_sha256': baseline['init_state_sha256'],
    }


def client_command(suite: str, task_id: int, init_index: int, port: int,
                   output: Path, label: str, config: dict, skip_attn: list[int]) -> list[str]:
    command = [
        str(CLIENT_PYTHON), str(CLIENT_SCRIPT), '--suite', suite, '--task', str(task_id),
        '--official-init-index', str(init_index), '--noise-seed', str(config['noise_seed']),
        '--env-seed', str(config['env_seed']), '--warmup-steps', str(config['warmup_steps']),
        '--replan-steps', str(config['replan_steps']), '--condition-label', label,
        '--expected-runtime', config['runtime'], '--host', '127.0.0.1', '--port', str(port),
        '--output-dir', str(output),
    ]
    if skip_attn:
        command.extend(['--skip-attn-layers', ','.join(map(str, skip_attn))])
    if config['candidate_skip_ffn_layers']:
        command.extend(['--skip-ffn-layers', ','.join(map(str, config['candidate_skip_ffn_layers']))])
    return command


def run_worker(suite: str, gpus: tuple[int, int], config: dict,
               condition: str, config_hash: str) -> None:
    if suite not in SUITES or len(gpus) != 2 or gpus[0] == gpus[1]:
        raise ValueError('Invalid suite/GPU assignment')
    output = OUTPUT_ROOT / condition / suite
    logs = output / 'logs'
    provenance_dir = output / 'server_provenance'
    logs.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)
    baseline_port, candidate_port = reserve_port(), reserve_port()
    while candidate_port == baseline_port:
        candidate_port = reserve_port()
    server_base = [
        str(BASE_PYTHON), str(SERVER_SCRIPT), '--base-triton-path', config['base_triton_path'],
        '--prompt-manifest', config['prompt_manifests'][suite], '--host', '127.0.0.1',
    ]
    started = time.monotonic()
    processes, streams = [], []
    server_specs = (
        ('baseline', gpus[0], baseline_port, []),
        ('candidate', gpus[1], candidate_port, config['candidate_skip_attn_layers']),
    )
    provenance_paths = {}
    try:
        for role, gpu, port, skipped in server_specs:
            metadata_path = provenance_dir / ('%s_%s_pid%s.json' % (suite, role, os.getpid()))
            provenance_paths[role] = metadata_path
            command = server_base + ['--port', str(port), '--run-metadata-path', str(metadata_path)]
            if skipped:
                command += ['--skip-attn-layers', *map(str, skipped)]
            if role == 'candidate' and config['candidate_skip_ffn_layers']:
                command += ['--skip-ffn-layers', *map(str, config['candidate_skip_ffn_layers'])]
            stream = (logs / (role + '_server.log')).open('a')
            streams.append(stream)
            process = subprocess.Popen(command, env=server_env(gpu), cwd=ROOT,
                                       stdout=stream, stderr=subprocess.STDOUT)
            processes.append(process)
        wait_server(processes[0], baseline_port, 'baseline', started)
        wait_server(processes[1], candidate_port, 'candidate', started)
        for role, expected_skip in (
            ('baseline', []), ('candidate', config['candidate_skip_attn_layers']),
        ):
            provenance = json.loads(provenance_paths[role].read_text())
            runtime = provenance.get('runtime_config', {})
            if provenance.get('checkpoint_sha256') != config['checkpoint_sha256']:
                raise RuntimeError('%s server loaded a different checkpoint' % role)
            if Path(provenance.get('prompt_manifest_path', '')).resolve() != Path(config['prompt_manifests'][suite]).resolve():
                raise RuntimeError('%s server loaded a different prompt manifest' % role)
            if (runtime.get('eager_diagnostic') is not True or
                runtime.get('denoise_steps') != 10 or
                runtime.get('skip_attn_layers_1based') != expected_skip or
                runtime.get('skip_ffn_layers_1based') != config['candidate_skip_ffn_layers']):
                raise RuntimeError('%s server intervention/runtime provenance mismatch' % role)

        rows = []
        tasks = config['tasks'][suite]
        inits = config['init_indices']
        episode_total = len(tasks) * len(inits)
        progress_path = output / 'progress.json'
        for task_id in tasks:
            for init_index in inits:
                pair_root = output / 'pairs' / ('task%03d_init%02d' % (task_id, init_index))
                baseline_dir = pair_root / 'baseline'
                candidate_dir = pair_root / candidate_dir_name(config)
                base_summary, cand_summary = baseline_dir / 'summary.json', candidate_dir / 'summary.json'
                if base_summary.exists() and cand_summary.exists():
                    row = validate_pair(pair_root, suite, task_id, init_index, config)
                    rows.append(row)
                    continue
                if pair_root.exists():
                    raise FileExistsError('Partial prior pair preserved; refusing overwrite: %s' % pair_root)
                pair_root.mkdir(parents=True)
                base_log_path = pair_root / 'baseline_client.log'
                cand_log_path = pair_root / 'candidate_client.log'
                base_log = base_log_path.open('w')
                cand_log = cand_log_path.open('w')
                client_processes = []
                try:
                    for role, gpu, port, dest, label, skipped, stream in (
                        ('baseline', gpus[0], baseline_port, baseline_dir, 'Full18', [], base_log),
                        ('candidate', gpus[1], candidate_port, candidate_dir, candidate_label(config),
                         config['candidate_skip_attn_layers'], cand_log),
                    ):
                        command = client_command(suite, task_id, init_index, port, dest,
                                                 label, config, skipped)
                        env = client_env(gpu)
                        client_processes.append(subprocess.Popen(
                            command, env=env, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT))
                    while any(process.poll() is None for process in client_processes):
                        time.sleep(1)
                    codes = [process.returncode for process in client_processes]
                    if any(code != 0 for code in codes):
                        raise subprocess.CalledProcessError(max(code for code in codes if code is not None),
                                                           'paired client rollout')
                finally:
                    for process in client_processes:
                        if process.poll() is None:
                            process.terminate()
                    for process in client_processes:
                        if process.poll() is None:
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                    base_log.close()
                    cand_log.close()
                row = validate_pair(pair_root, suite, task_id, init_index, config)
                rows.append(row)
                dump_json(progress_path, {
                    'suite': suite, 'completed_pairs': len(rows), 'expected_pairs': episode_total,
                    'last_pair': {'task_id': task_id, 'init_index': init_index},
                    'videos_validated': 2 * len(rows), 'condition': condition,
                })
        csv_path = output / 'paired_episode_results.csv'
        fields = list(rows[0]) if rows else []
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        outcomes = {name: sum(row['outcome'] == name for row in rows)
                    for name in ('both_success', 'baseline_only', 'candidate_only', 'both_failure')}
        dump_json(output / 'suite_summary.json', {
            'complete': True, 'condition': condition, 'suite': suite,
            'episode_pairs': len(rows), 'baseline_successes': sum(row['baseline_success'] for row in rows),
            'candidate_successes': sum(row['candidate_success'] for row in rows),
            'paired_outcomes': outcomes,
            'mean_eef_deviation_p95_m': float(np.mean([row['eef_deviation_p95_m'] for row in rows])),
            'all_initial_states_paired': True,
            'all_shared_inference_noise_matched': True,
            'all_videos_h264_yuv420p_frame_count_and_decode_validated': True,
            'runtime': config['runtime'],
            'latency_claim': 'none; eager diagnostic path',
            'rows_csv': str(csv_path),
            'configuration': config,
            'configuration_sha256': config_hash,
            'runner_sha256': sha256(Path(__file__)),
        })
        event(event='submodule_multi_init_suite_validated', condition=condition,
              suite=suite, gpu_pair=list(gpus),
              episode_pairs=len(rows), paired_outcomes=outcomes)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for stream in streams:
            stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_DEFAULT)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--suite', choices=SUITES)
    parser.add_argument('--gpus', help='Two physical GPU IDs, e.g. 0,1')
    parser.add_argument('--status-path', type=Path, default=STATUS_DEFAULT)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    condition = str(config.get('experiment', ''))
    if config.get('schema') != 1 or not condition or any(ch not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_' for ch in condition):
        raise ValueError('Experiment configuration identity mismatch')
    if config['runtime'] != 'triton_eager' or not config['candidate_skip_attn_layers']:
        raise ValueError('Unexpected intervention/runtime; refusing to run')
    if any(suite not in SUITES for suite in config['tasks']):
        raise ValueError('Unexpected suite in config')
    for suite, tasks in config['tasks'].items():
        if not tasks or len(set(tasks)) != len(tasks) or any(not 0 <= int(task) < (10 if suite == 'libero_10' else 90) for task in tasks):
            raise ValueError('Invalid task selection for %s' % suite)
    if sorted(config['init_indices']) != list(range(5)):
        raise ValueError('This frozen screen requires official init indices 0..4')
    if (not Path(config['base_triton_path']).exists() or
        any(not Path(path).is_file() for path in config['prompt_manifests'].values())):
        raise FileNotFoundError('Model artifact or prompt manifest is missing')

    if args.worker:
        if args.suite not in SUITES or not args.gpus:
            raise ValueError('Worker mode requires --suite and --gpus')
        run_worker(args.suite, tuple(int(x) for x in args.gpus.split(',')),
                   config, condition, file_sha(args.config))
        return

    expected = {suite: len(tasks) * len(config['init_indices'])
                for suite, tasks in config['tasks'].items()}
    if not args.launch:
        print(json.dumps({
            'experiment': condition, 'conditions': ['Full18', candidate_label(config)],
            'episodes_by_suite_and_condition': expected,
            'paired_episode_count': sum(expected.values()),
            'total_policy_rollouts': 2 * sum(expected.values()),
            'runtime': config['runtime'], 'gpus_per_suite_worker': 2,
            'suites': list(config['tasks']), 'init_indices': config['init_indices'],
            'note': 'Diagnostic eager runtime; no production speed claim.',
        }, indent=2))
        return

    status_path = args.status_path.resolve()
    if EXP10_RESULTS.resolve() not in status_path.parents:
        raise ValueError('Status path must be under experiment10 results for shared lease tracking')
    status_path.parent.mkdir(parents=True, exist_ok=True)
    lock = status_path.with_name(status_path.name + '.scheduler.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError('Another scheduler owns %s' % status_path) from exc

    config_hash = file_sha(args.config)
    prior = json.loads(status_path.read_text()) if status_path.exists() else None
    if prior:
        try:
            state = (Path('/proc') / str(prior.get('pid', -1)) / 'stat').read_text().split(') ', 1)[1][0]
        except (OSError, IndexError):
            state = 'Z'
        if state not in ('Z', 'X'):
            raise RuntimeError('Prior scheduler PID %s is still live' % prior.get('pid'))
        if prior.get('config_sha256') != config_hash:
            raise RuntimeError('Refusing to resume with a changed configuration')
    pending = [suite for suite in config['tasks']
               if not suite_config_matches(OUTPUT_ROOT / condition / suite / 'suite_summary.json',
                                           suite, config)]
    failures, active, leases = [], {}, {}
    next_poll = 0.0
    previous = managed_gpus()
    event(event='submodule_multi_init_scheduler_started', condition=condition,
          status_path=str(status_path), pending=len(pending),
          gpus_per_worker=2, pairs=sum(expected.values()))
    while pending or active:
        current = managed_gpus()
        if previous - current:
            next_poll = 0.0
        previous = current
        for suite, record in list(active.items()):
            code = record['process'].poll()
            if code is None:
                continue
            record['stream'].close()
            for lease in record['leases']:
                lease.close()
            del active[suite]
            event(event='submodule_multi_init_worker_finished', condition=condition, suite=suite,
                  gpus=record['gpus'], code=code)
            if code:
                failures.append({'suite': suite, 'code': code})
            next_poll = 0.0
        current = managed_gpus()
        if pending and len(current) < 4 and time.monotonic() >= next_poll:
            available = [gpu for gpu in free_gpus() if gpu not in current]
            next_poll = time.monotonic() + 300
            while len(available) >= 2 and pending:
                pair = tuple((available.pop(0), available.pop(0)))
                pair_leases = [claim_gpu(gpu) for gpu in pair]
                if any(lease is None for lease in pair_leases):
                    for lease in pair_leases:
                        if lease is not None:
                            lease.close()
                    continue
                suite = pending.pop(0)
                suite_logs = (OUTPUT_ROOT / condition / suite / 'logs')
                suite_logs.mkdir(parents=True, exist_ok=True)
                stream = (suite_logs / 'worker.log').open('a')
                command = [str(BASE_PYTHON), str(Path(__file__).resolve()), '--config',
                           str(args.config.resolve()), '--worker', '--suite', suite,
                           '--gpus', ','.join(map(str, pair))]
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream,
                                           stderr=subprocess.STDOUT)
                active[suite] = {'process': process, 'gpus': list(pair),
                                 'leases': pair_leases, 'stream': stream}
                event(event='submodule_multi_init_worker_started', condition=condition, suite=suite,
                      gpus=list(pair), pid=process.pid)
        dump_json(status_path, {
            'pid': os.getpid(), 'config_sha256': config_hash,
            'pending': pending, 'failures': failures,
            'active': [{'gpu': row['gpus'][0], 'gpus': row['gpus'],
                        'pid': row['process'].pid, 'suite': suite}
                       for suite, row in active.items()],
            'complete': not pending and not active and not failures,
        })
        if pending or active:
            time.sleep(2)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

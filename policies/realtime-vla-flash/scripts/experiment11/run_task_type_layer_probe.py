"""Bounded real-Full-observation D3 single-layer probe; never creates an environment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np

from scripts.experiment11 import launch_shadow as scheduler
from scripts.experiment11.protocol import RESULTS, ROOT, SUITES, PROTOCOL, SOURCE
from scripts.experiment10.common import PHASE_MAP, dump_json, json_lines, sha256
from scripts.experiment11.call_budget import CallBudget, accounted_infer
from scripts.experiment11.shadow_worker import _input_digest

ATTEMPT = 'round2_20261011'
PHASES = ('approach', 'contact_grasp', 'transport', 'alignment_release')
LAYERS = tuple(range(1, 18))


def paths(attempt=ATTEMPT):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', attempt):
        raise ValueError('Invalid attempt')
    directory = RESULTS / 'task_type_layer_probe' / attempt
    return directory, directory / 'manifest.json', RESULTS / f'experiment11_task_type_probe_status_{attempt}.json'


def choose_replans(inference, telemetry):
    bystep = {int(r['step']): r for r in telemetry if r.get('when') in ('post_action', 'initial', None)}
    first = {}
    for row in sorted(inference, key=lambda r: int(r['infer_id'])):
        phase = PHASE_MAP.get(bystep.get(int(row['frame_idx']), {}).get('semantic_phase'), 'unknown')
        if phase in PHASES and phase not in first:
            first[phase] = row
    return first


def read_input(sample):
    with np.load(sample['input_path'], allow_pickle=False) as data:
        meta = json.loads(str(data['metadata_json'].item()))
        prompt = str(data['prompt'].item())
        digest = _input_digest(data, prompt, sample['suite'])
        noise = np.array(data['noise'], dtype=np.float32, copy=True)
        if (digest != sample['input_sha256'] or digest != meta.get('input_sha256') or
                meta.get('checkpoint_sha256') != sample['checkpoint_sha256'] or
                meta.get('skip_layers') != [] or meta.get('denoise_steps') != 3 or
                meta.get('suite') != sample['suite'] or
                int(meta.get('task_id', -1)) != sample['task_id'] or
                int(meta.get('episode_idx', -1)) != sample['episode_idx'] or
                int(meta.get('infer_id', -1)) != sample['replan_id'] or
                hashlib.sha256(noise.tobytes()).hexdigest() != sample['noise_sha256']):
            raise RuntimeError(f"Input provenance mismatch: {sample['sample_id']}")
        observation = {'observation/image': data['agent_image'].copy(),
                       'observation/wrist_image': data['wrist_image'].copy(),
                       'observation/state': data['robot_state'].copy(), 'prompt': prompt}
    return observation, noise


def freeze(attempt):
    directory, manifest_path, _ = paths(attempt)
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    candidates = json.loads((RESULTS / 'protocol/candidate_manifest.json').read_text())
    checkpoint = candidates['checkpoint_sha256']
    samples, missing, sources = [], [], {}
    for suite in SUITES:
        base = RESULTS / 'closed_loop' / scheduler.BASELINE / suite / 'run'
        episode_path = base / 'episode_log.json'
        episodes = json.loads(episode_path.read_text())
        if {(int(r['task_id']), int(r['episode_idx'])) for r in episodes} != {
                (t, i) for t in range(10) for i in range(5)} or len(episodes) != 50:
            raise RuntimeError(f'Full task/init coverage mismatch: {suite}')
        sources[str(episode_path)] = sha256(episode_path)
        for episode in episodes:
            infer_path, telemetry_path = base / episode['infer_path'], base / episode['telemetry_path']
            inference, telemetry = json_lines(infer_path), json_lines(telemetry_path)
            if len(inference) != int(episode['infer_record_count']):
                raise RuntimeError('Missing Full replan records')
            selected = choose_replans(inference, telemetry)
            sources[str(infer_path)], sources[str(telemetry_path)] = sha256(infer_path), sha256(telemetry_path)
            for phase in PHASES:
                if phase not in selected:
                    missing.append({'suite': suite, 'task_id': int(episode['task_id']),
                                    'episode_idx': int(episode['episode_idx']), 'phase': phase,
                                    'reason': 'no_replan_in_reached_phase'})
                    continue
                row = selected[phase]
                info = row['experiment11']
                if (info.get('condition') != scheduler.BASELINE or info.get('candidate_skip_layers_1based') != [] or
                        info.get('denoise_steps') != 3 or row.get('accepted_prefix_len') != 5):
                    raise RuntimeError('Full protocol mismatch')
                sample = {'sample_id': f'{suite}__task{int(episode["task_id"]):02d}__init{int(episode["episode_idx"]):02d}__{phase}',
                          'suite': suite, 'task_id': int(episode['task_id']), 'episode_idx': int(episode['episode_idx']),
                          'phase': phase, 'replan_id': int(row['infer_id']), 'environment_step': int(row['frame_idx']),
                          'input_path': info['full_input_path'], 'input_sha256': info['full_input_sha256'],
                          'noise_sha256': info['noise_sha256'], 'checkpoint_sha256': checkpoint,
                          'infer_path': str(infer_path), 'telemetry_path': str(telemetry_path),
                          'full_actions': row['chunk_actions'], 'full_success': bool(episode['success'])}
                observation, _ = read_input(sample)
                sample['prompt'] = observation['prompt']
                samples.append(sample)
    if len(samples) > 800 or len({s['sample_id'] for s in samples}) != len(samples):
        raise RuntimeError('Sample cap/unique key violation')
    manifest = {'schema': 1, 'attempt': attempt, 'protocol': PROTOCOL, 'checkpoint_sha256': checkpoint,
                'layers': list(LAYERS), 'samples': samples, 'missing_phase_cells': missing,
                'source_hashes': sources, 'budget_limit': 14400,
                'expected_returned_predictions': 18 * len(samples), 'environment_steps': 0}
    dump_json(manifest_path, manifest)
    return manifest


def verify_manifest(manifest_path):
    manifest = json.loads(manifest_path.read_text())
    if (manifest['layers'] != list(LAYERS) or len(manifest['samples']) > 800 or
            manifest['budget_limit'] != 14400 or manifest['protocol'] != PROTOCOL):
        raise RuntimeError('Frozen probe protocol changed')
    for name, digest in manifest['source_hashes'].items():
        if sha256(Path(name)) != digest:
            raise RuntimeError(f'Probe source changed: {name}')
    return manifest


def activity_summary(submodules, sample):
    import torch
    rows = []
    identity = {k: sample[k] for k in ('sample_id', 'suite', 'task_id', 'episode_idx', 'phase')}
    for layer, (before, attn, ffn) in enumerate(submodules, 1):
        for modality, selected in [('agent', slice(0, 256)), ('wrist', slice(256, 512)), ('language', slice(512, None))]:
            a, b, c = [t[selected].float() for t in (before, attn, ffn)]
            if not a.numel():
                continue
            for name, x, ref in [('H_in', a, None), ('H_attn', b, None), ('H_out', c, None),
                                 ('delta_attn', b-a, a), ('delta_ffn', c-b, b), ('delta_block', c-a, a)]:
                rms = x.square().mean().sqrt()
                energy = x.square().sum(-1)
                denominator = energy.sum()
                token_cosine = torch.tensor(float('nan'), device=x.device)
                relative = torch.tensor(float('nan'), device=x.device)
                if ref is not None:
                    norm_ref = ref.norm()
                    relative = torch.where(norm_ref > 1e-12, x.norm() / norm_ref.clamp_min(1e-12),
                                           torch.tensor(float('nan'), device=x.device))
                    target = ref + x
                    norms = ref.norm(dim=-1) * target.norm(dim=-1)
                    valid = norms > 1e-12
                    token_cosine = ((ref * target).sum(-1)[valid] / norms[valid]).mean()
                values = torch.stack([rms, x.mean(), x.std(unbiased=False), x.abs().max(),
                    (x == 0).float().mean(), (x.abs() <= .001*rms).float().mean(),
                    (x.abs() <= .01*rms).float().mean(), (x.abs() <= .1*rms).float().mean(),
                    energy.topk(max(1, int(np.ceil(.1 * len(energy))))).values.sum() / denominator.clamp_min(1e-12),
                    relative, token_cosine, torch.isfinite(x).all().float()]).cpu().numpy()
                if values[-1] != 1:
                    raise RuntimeError('Nonfinite activation')
                fields = ('rms', 'mean', 'std', 'max_abs', 'zero_fraction', 'near_zero_0.001',
                          'near_zero_0.01', 'near_zero_0.1', 'top10pct_token_energy_fraction',
                          'relative_residual', 'cosine')
                rows.append({**identity, 'layer': layer, 'modality': modality, 'tensor': name,
                             'execution_dtype': str(before.dtype), 'token_count': len(x),
                             **{k: (float(v) if np.isfinite(v) else None) for k, v in zip(fields, values)}})
    rows.append({**identity, 'layer': 18, 'modality': 'all', 'tensor': 'kv_only',
                 'reason': 'H18_equals_H17_by_implementation; excluded_from_hidden_ranking'})
    return rows


def capture_full(backend, sample, budget):
    import torch
    from scripts.spec.triton import pi0_infer as kernels
    from scripts.experiment10.backend import decoder
    call = budget.reserve({'kind': 'full_activity', 'sample_id': sample['sample_id'], 'steps': 3})
    try:
        w, b = backend.runtime.weights, backend.runtime.buffers
        images, state, noise = backend.inputs
        b['observation_images_normalized'].copy_(images)
        b['observation_state_normalized'].copy_(state)
        b['diffusion_noise'].copy_(noise)
        b['encoder_x'][512:].copy_(w['language_embeds'])
        kernels.vision_encoder(w, b, 2)
        hidden, modules = [], []
        kernels.transformer_encoder(w, b, b['encoder_x'].shape[0], hidden_trace=hidden, submodule_trace=modules)
        decoder(backend.runtime, backend.schedule(3))
        raw = b['diffusion_noise'].float().cpu().numpy().copy()
        commands = np.asarray(backend.output_transform({'state': np.asarray(backend.transformed['state']).copy(),
                                                       'actions': raw.copy()})['actions'], dtype=np.float32)
        parity = float(np.max(np.abs(commands - np.asarray(sample['full_actions']))))
        if parity > 1e-5 or not np.allclose(commands, sample['full_actions'], rtol=1e-4, atol=1e-5):
            raise RuntimeError(f'Instrumented Full D3 parity failed: {parity}')
        activity = activity_summary(modules, sample)
        budget.finish(call, 'complete', {'parity_max_abs_error': parity, 'returned_actions': 50,
                                       'internal_warmup_capture_forwards': 0, 'requested_schedule_denoising_steps': 3})
        return raw, commands, activity, parity
    except BaseException as exc:
        budget.finish(call, 'failed', {'exception': type(exc).__name__})
        raise


def job_directory(attempt, suite, task, retry_id=None):
    directory = paths(attempt)[0]
    if retry_id is not None:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', retry_id):
            raise ValueError('Invalid retry id')
        directory = directory / 'retries' / retry_id
    return directory / suite / f'task{task:02d}'


def verify_warmup_diagnostics(directory):
    for task in (4, 5):
        path = directory / f'parity_diagnostic_task{task:02d}_history_v1.json'
        record = json.loads(path.read_text())
        if (record.get('sample_id') != f'libero_object__task{task:02d}__init00__approach' or
                record.get('history_scope') != 'original task0 warmup plus last replan of init4 per preceding task; not complete replay' or
                record['paths']['target_after_history']['max_abs'] > 1e-5):
            raise RuntimeError('Original warmup/history parity proof missing or failed')


def original_suite_warmup(backend, suite, task, budget):
    source_manifest = SOURCE / 'observations' / suite / 'manifest.json'
    entries = json.loads(source_manifest.read_text())['observations']
    entry = next(e for e in entries if int(e['task_id']) == 0)
    source = Path(entry['path'])
    with np.load(source, allow_pickle=False) as data:
        observation = {'observation/image': data['agent_image'].copy(),
                       'observation/wrist_image': data['wrist_image'].copy(),
                       'observation/state': data['robot_state'].copy(), 'prompt': str(data['prompt'].item())}
        metadata = json.loads(str(data['metadata_json'].item()))
    noise = np.random.default_rng(12345).standard_normal((50,32)).astype(np.float32)
    backend.prepare(observation, metadata, noise)
    accounted_infer(backend, (), 3, budget=budget,
                    identity={'kind': 'original_suite_warmup', 'suite': suite,
                              'input_path': str(source), 'input_file_sha256': sha256(source)})
    transitions = []
    full_run = RESULTS / 'closed_loop/Full18_D3_R5_FORMAL' / suite / 'run'
    episodes = json.loads((full_run / 'episode_log.json').read_text())
    for preceding in range(task):
        episode = next(e for e in episodes if int(e['task_id']) == preceding and int(e['episode_idx']) == 4)
        infer_path = full_run / episode['infer_path']
        row = json_lines(infer_path)[-1]
        path = Path(row['experiment11']['full_input_path'])
        with np.load(path, allow_pickle=False) as data:
            observation = {'observation/image': data['agent_image'].copy(),
                           'observation/wrist_image': data['wrist_image'].copy(),
                           'observation/state': data['robot_state'].copy(), 'prompt': str(data['prompt'].item())}
            noise = data['noise'].copy()
        backend.prepare(observation, {'suite': suite, 'task_description': observation['prompt']}, noise)
        _, commands, _ = accounted_infer(backend, (), 3, budget=budget,
            identity={'kind': 'original_preceding_task_last_replan', 'suite': suite, 'task': preceding,
                      'input_path': str(path), 'input_file_sha256': sha256(path)})
        transitions.append({'task': preceding,'input_path': str(path), 'input_sha256': sha256(path),
                            'infer_path_sha256': sha256(infer_path), 'prompt_length': backend.length,
                            'parity_max_abs': float(np.max(np.abs(commands-np.asarray(row['chunk_actions']))))})
    return {'source_path': str(source), 'source_sha256': sha256(source),
            'source_manifest_sha256': sha256(source_manifest), 'prompt_length': backend.length,
            'noise_seed': 12345, 'denoise_steps': 3, 'preceding_task_last_replans': transitions,
            'scope': 'shape/history transitions; not full historical environment replay'}


def run_worker(attempt, suite, task, gpu, retry_id=None):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise RuntimeError('Worker is not bound to its leased GPU')
    import torch
    from scripts.experiment10.backend import Backend
    from scripts.experiment10.numerics import action_metrics
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    directory, manifest_path, _ = paths(attempt)
    manifest = verify_manifest(manifest_path)
    samples = [s for s in manifest['samples'] if s['suite'] == suite and s['task_id'] == task]
    job_dir = job_directory(attempt, suite, task, retry_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    target, marker = job_dir / 'predictions.jsonl', job_dir / 'validated.json'
    partial = target.with_name(target.name + '.partial')
    if any(p.exists() for p in (target, marker, partial)):
        raise RuntimeError('Refusing to overwrite probe output')
    budget = CallBudget(directory / 'call_budget.sqlite', 14400)
    metadata = [{'suite': suite, 'task_description': s['prompt']} for s in samples]
    if retry_id is not None:
        verify_warmup_diagnostics(directory)
        entries = json.loads((SOURCE / 'observations' / suite / 'manifest.json').read_text())['observations']
        for entry in entries:
            with np.load(entry['path'], allow_pickle=False) as data:
                metadata.append(json.loads(str(data['metadata_json'].item())))
    backend = Backend(metadata)
    if backend.weight_hash != manifest['checkpoint_sha256']:
        raise RuntimeError('Actual checkpoint does not match the frozen source')
    warmup = original_suite_warmup(backend, suite, task, budget) if retry_id is not None else None
    baseline, activations, max_parity, smoke = {}, [], 0.0, None
    for sample in samples:
        obs, noise = read_input(sample)
        backend.prepare(obs, {'suite': suite, 'task_description': sample['prompt']}, noise)
        raw, commands, rows, parity = capture_full(backend, sample, budget)
        baseline[sample['sample_id']] = raw, commands
        activations.extend(rows)
        max_parity = max(max_parity, parity)
        if smoke is None:
            smoke = accounted_infer(backend, (1,), 3, budget=budget,
                    identity={'kind': 'single_layer', 'sample_id': sample['sample_id'], 'layer': 1,
                              'smoke_reused_as_formal': True})
            if smoke[0].shape != (50,32) or smoke[1].shape != (50,7) or not np.isfinite(smoke[1]).all():
                raise RuntimeError('Full+L1 smoke output shape/finiteness failed')
            print(json.dumps({'event': 'full_l1_smoke_passed', 'suite': suite, 'task': task,
                              'full_parity': parity}), flush=True)
    with partial.open('x') as stream:
        for layer in LAYERS:
            for sample in samples:
                obs, noise = read_input(sample)
                backend.prepare(obs, {'suite': suite, 'task_description': sample['prompt']}, noise)
                if layer == 1 and sample['sample_id'] == samples[0]['sample_id']:
                    raw, commands, timing = smoke
                else:
                    raw, commands, timing = accounted_infer(backend, (layer,), 3, budget=budget,
                        identity={'kind': 'single_layer', 'sample_id': sample['sample_id'], 'layer': layer})
                full_raw, full_commands = baseline[sample['sample_id']]
                metrics = action_metrics(raw, full_raw, commands, full_commands)
                for h in (3, 5):
                    metrics.append({'horizon': h, 'normalized_continuous_rmse': float(np.sqrt(np.mean(
                        (raw[:h,:6] - full_raw[:h,:6])**2))),
                        'translation_command_rmse': float(np.sqrt(np.mean((commands[:h,:3]-full_commands[:h,:3])**2))),
                        'rotation_command_rmse': float(np.sqrt(np.mean((commands[:h,3:6]-full_commands[:h,3:6])**2))),
                        'gripper_flip_rate': float(np.mean(np.sign(commands[:h,6]) != np.sign(full_commands[:h,6])))})
                row = {k: sample[k] for k in ('sample_id', 'suite', 'task_id', 'episode_idx', 'phase',
                                              'input_sha256', 'noise_sha256', 'checkpoint_sha256')}
                row.update(layer=layer, metrics=metrics, full_actions=full_commands.tolist(),
                           variant_actions=commands.tolist(), cuda_policy_ms=float(timing['cuda_policy_ms']))
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            print(json.dumps({'suite': suite, 'task': task, 'layer_complete': layer, 'samples': len(samples)}), flush=True)
    partial.replace(target)
    dump_json(job_dir / 'activation_statistics.json', {'rows': activations})
    dump_json(marker, {'complete': True, 'suite': suite, 'task_id': task, 'samples': len(samples),
                      'prediction_records': len(samples)*17, 'layers': list(LAYERS),
                      'manifest_sha256': sha256(manifest_path), 'predictions_sha256': sha256(target),
                      'activation_sha256': sha256(job_dir / 'activation_statistics.json'),
                      'full_parity_max_abs_error': max_parity, 'environment_steps': 0,
                      'retry_id': retry_id, 'original_suite_warmup': warmup})
    return 0


def configure(attempt, retry_id=None):
    from scripts.experiment11.gpu_discovery import throttled_free_gpus
    directory, manifest_path, status = paths(attempt)
    if retry_id is not None:
        job_directory(attempt, SUITES[0], 0, retry_id)  # Validate before any state write.
        status = status.with_name(status.stem + '_' + retry_id + status.suffix)
    def jobs():
        manifest = verify_manifest(manifest_path)
        pairs = sorted({(s['suite'], s['task_id']) for s in manifest['samples']},
                       key=lambda pair: (SUITES.index(pair[0]), pair[1]))
        result = [{'job_id': f'{s}__task{t:02d}', 'variant_id': f'task{t:02d}', 'suite': s,
                   'task_id': t, 'skip_layers': list(LAYERS), 'condition_name': 'single_layer_D3'} for s,t in pairs]
        return result, sha256(manifest_path), sha256(RESULTS / 'analysis/closed_loop_d3/completion_audit.json')
    def outputs(job):
        original = job_directory(attempt, job['suite'], job['task_id'])
        # Only validated original results are reused; invalid/partial originals
        # remain untouched and cannot be mistaken for successful retry output.
        base = original
        if retry_id is not None and not validate_at(job, original):
            base = job_directory(attempt, job['suite'], job['task_id'], retry_id)
        return base / 'predictions.jsonl', base / 'validated.json', base / 'predictions.jsonl.partial'
    def validate_at(job, base):
        prediction, marker, partial = base/'predictions.jsonl', base/'validated.json', base/'predictions.jsonl.partial'
        if not prediction.is_file() or not marker.is_file() or partial.exists():
            return False
        m = json.loads(marker.read_text())
        manifest = json.loads(manifest_path.read_text())
        count = sum(s['suite']==job['suite'] and s['task_id']==job['task_id'] for s in manifest['samples'])
        return (m.get('complete') and m.get('suite') == job['suite'] and m.get('task_id') == job['task_id'] and
                m.get('samples') == count and m.get('prediction_records') == 17*count and m.get('layers') == list(LAYERS) and
                m.get('environment_steps') == 0 and m.get('manifest_sha256') == sha256(manifest_path) and
                m.get('predictions_sha256') == sha256(prediction) and
                m.get('activation_sha256') == sha256(marker.parent/'activation_statistics.json') and
                m.get('full_parity_max_abs_error', float('inf')) <= 1e-5)
    def validate(job):
        prediction, marker, partial = outputs(job)
        return validate_at(job, marker.parent)
    def command(job, gpu):
        result = [str(ROOT/'.venv/bin/python'), '-m', 'scripts.experiment11.run_task_type_layer_probe',
                '--attempt-id', attempt, '--worker-suite', job['suite'], '--task-id', str(job['task_id']),
                '--gpu-index', str(gpu)]
        if retry_id is not None:
            result += ['--retry-id', retry_id]
        return result
    scheduler.STATUS, scheduler.LOCK = status, status.with_name(status.name+'.scheduler.lock')
    scheduler.LOG_ROOT = RESULTS / 'logs/task_type_layer_probe' / attempt
    if retry_id is not None:
        scheduler.LOG_ROOT = scheduler.LOG_ROOT / retry_id
    scheduler.frozen_jobs, scheduler.output_paths = jobs, outputs
    scheduler.validate_job, scheduler.child_command = validate, command
    scheduler.free_gpus = throttled_free_gpus
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt-id', default=ATTEMPT)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker-suite', choices=SUITES)
    parser.add_argument('--task-id', type=int)
    parser.add_argument('--gpu-index', type=int)
    parser.add_argument('--retry-id', help='Separate status/output paths; original manifest and budget are reused')
    parser.add_argument('--reuse-completed-diagnostic', type=Path)
    args = parser.parse_args()
    if args.worker_suite:
        return run_worker(args.attempt_id, args.worker_suite, args.task_id, args.gpu_index, args.retry_id)
    if args.freeze:
        freeze(args.attempt_id)
    jobs = configure(args.attempt_id, args.retry_id)
    queue, selection_hash, audit_hash = jobs()
    manifest = json.loads(paths(args.attempt_id)[1].read_text())
    if args.launch and args.retry_id is not None:
        verify_warmup_diagnostics(paths(args.attempt_id)[0])
    scheduler.INITIAL_AVAILABLE = ()
    if args.reuse_completed_diagnostic is not None:
        source = args.reuse_completed_diagnostic.resolve()
        if source.parent != paths(args.attempt_id)[0].resolve() or not args.retry_id:
            raise RuntimeError('GPU reuse must reference this retry source attempt')
        record = json.loads(source.read_text())
        if (time.time()-record.get('completed_epoch',0)>30 or
                scheduler.live_pid(int(record.get('pid',0))) or
                record['paths']['target_after_history']['max_abs']>1e-5):
            raise RuntimeError('Recent successful own GPU completion is not proven')
        scheduler.INITIAL_AVAILABLE = (int(record['gpu']),)
    print(json.dumps({'dry_run': not args.launch, 'jobs': len(queue), 'samples': len(manifest['samples']),
                      'expected_calls': manifest['expected_returned_predictions'], 'hard_cap':14400,
                      'retry_id': args.retry_id, 'retry_extra_warmups_per_new_worker': '1+task_id' if args.retry_id else 0,
                      'ledger_path': str(paths(args.attempt_id)[0]/'call_budget.sqlite')}), flush=True)
    return scheduler.run_manager(queue, selection_hash, audit_hash) if args.launch else 0


if __name__ == '__main__':
    raise SystemExit(main())

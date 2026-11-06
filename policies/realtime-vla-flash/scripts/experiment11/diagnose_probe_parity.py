"""Bounded same-input diagnostic; shares the formal probe call ledger."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.experiment11.run_task_type_layer_probe import paths, read_input, capture_full
from scripts.experiment11.call_budget import CallBudget, accounted_infer
from scripts.experiment10.common import claim_gpu, dump_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', default='round2_20261011')
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--task', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--warm-history', action='store_true',
                        help='Replay the original task0 warmup and one last-observation transition per preceding task')
    parser.add_argument('--only-original-warmup', action='store_true')
    parser.add_argument('--reuse-completed-diagnostic', type=Path,
                        help='Event-based reuse of our own GPU within 30 seconds of a verified completed diagnostic')
    args = parser.parse_args()
    if args.only_original_warmup and not args.warm_history:
        raise ValueError('--only-original-warmup requires --warm-history')
    if args.output.exists():
        raise RuntimeError('Diagnostic output already exists')
    from scripts.experiment11.gpu_discovery import throttled_free_gpus
    if args.reuse_completed_diagnostic:
        prior = json.loads(args.reuse_completed_diagnostic.read_text())
        if (prior.get('gpu') != args.gpu or time.time()-prior.get('completed_epoch',0) > 30 or
                Path('/proc',str(prior.get('pid',0))).exists()):
            raise RuntimeError('Cannot verify recent own diagnostic completion')
    elif args.gpu not in throttled_free_gpus():
        raise RuntimeError('GPU not confirmed idle by shared discovery clock')
    lease = claim_gpu(args.gpu)
    if lease is None:
        raise RuntimeError('GPU lease unavailable')
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        import torch
        from scripts.experiment10.backend import Backend
        torch.set_grad_enabled(False)
        torch.set_num_threads(4)
        directory, manifest_path, _ = paths(args.attempt)
        manifest = json.loads(manifest_path.read_text())
        sample = next(s for s in manifest['samples'] if s['suite'] == 'libero_object'
                      and s['task_id'] == args.task)
        obs, noise = read_input(sample)
        budget = CallBudget(directory / 'call_budget.sqlite', 14400)
        from scripts.experiment11.protocol import SOURCE, RESULTS
        source_entries = json.loads((SOURCE / 'observations' / sample['suite'] / 'manifest.json').read_text())['observations']
        metadata = []
        for entry in source_entries:
            with np.load(entry['path'], allow_pickle=False) as data:
                metadata.append(json.loads(str(data['metadata_json'].item())))
        backend = Backend(metadata)
        if backend.weight_hash != manifest['checkpoint_sha256']:
            raise RuntimeError('Checkpoint mismatch')
        if args.warm_history:
            result = {'sample_id': sample['sample_id'], 'before': budget.summary(), 'paths': {}}
            entry = next(e for e in source_entries if e['task_id'] == 0)
            with np.load(entry['path'], allow_pickle=False) as data:
                warm = {'observation/image': data['agent_image'].copy(),
                        'observation/wrist_image': data['wrist_image'].copy(),
                        'observation/state': data['robot_state'].copy(), 'prompt': str(data['prompt'].item())}
                meta = json.loads(str(data['metadata_json'].item()))
            backend.prepare(warm, meta, np.random.default_rng(12345).standard_normal((50,32)).astype(np.float32))
            accounted_infer(backend, (), 3, budget=budget, identity={'kind': 'parity_history_original_warmup'})
            for task in range(0 if args.only_original_warmup else args.task):
                base = RESULTS / 'closed_loop/Full18_D3_R5_FORMAL' / sample['suite'] / 'run'
                episodes = json.loads((base / 'episode_log.json').read_text())
                episode = next(e for e in episodes if int(e['task_id']) == task and int(e['episode_idx']) == 4)
                rows = [json.loads(line) for line in (base / episode['infer_path']).read_text().splitlines()]
                row = rows[-1]
                with np.load(row['experiment11']['full_input_path'], allow_pickle=False) as data:
                    warm = {'observation/image': data['agent_image'].copy(),
                            'observation/wrist_image': data['wrist_image'].copy(),
                            'observation/state': data['robot_state'].copy(), 'prompt': str(data['prompt'].item())}
                    warm_noise = data['noise'].copy()
                backend.prepare(warm, {'suite': sample['suite'], 'task_description': warm['prompt']}, warm_noise)
                _, command, _ = accounted_infer(backend, (), 3, budget=budget,
                    identity={'kind': 'parity_history_last_task_observation', 'task': task, 'infer_id': row['infer_id']})
                result['paths'][f'task{task:02d}_last'] = {
                    'max_abs': float(np.max(np.abs(command-np.asarray(row['chunk_actions'])))),
                    'prompt_length': backend.length}
            backend.prepare(obs, {'suite': sample['suite'], 'task_description': sample['prompt']}, noise)
            _, command, _ = accounted_infer(backend, (), 3, budget=budget,
                identity={'kind': 'parity_history_target', 'sample_id': sample['sample_id']})
            result['paths']['target_after_history'] = {
                'max_abs': float(np.max(np.abs(command-np.asarray(sample['full_actions'])))),
                'prompt_length': backend.length}
            result['after'] = budget.summary()
            result.update(pid=os.getpid(), gpu=args.gpu, completed_epoch=time.time())
            result['history_scope'] = ('original task0 warmup only' if args.only_original_warmup else
                'original task0 warmup plus last replan of init4 per preceding task; not complete replay')
            dump_json(args.output, result)
            print(json.dumps(result), flush=True)
            return
        backend.prepare(obs, {'suite': sample['suite'], 'task_description': sample['prompt']}, noise)
        reference = np.asarray(sample['full_actions'])
        result = {'sample_id': sample['sample_id'], 'before': budget.summary(), 'paths': {}}
        # Fresh capture preserves the failing ordering. No tolerance change.
        try:
            _, commands, _, parity = capture_full(backend, sample, budget)
            result['paths']['fresh_capture'] = {'max_abs': parity}
        except RuntimeError as exc:
            result['paths']['fresh_capture'] = {'error': str(exc)}
        raw_graph, commands, _ = accounted_infer(backend, (), 3, budget=budget,
            identity={'kind': 'parity_diagnostic_graph', 'sample_id': sample['sample_id']})
        result['paths']['production_graph'] = {'max_abs': float(np.max(np.abs(commands-reference)))}
        call = budget.reserve({'kind': 'parity_diagnostic_eager', 'sample_id': sample['sample_id']})
        try:
            images, state, input_noise = backend.inputs
            b = backend.runtime.buffers
            b['observation_images_normalized'].copy_(images)
            b['observation_state_normalized'].copy_(state)
            b['diffusion_noise'].copy_(input_noise)
            backend.record((), 3)
            raw = b['diffusion_noise'].float().cpu().numpy().copy()
            commands = np.asarray(backend.output_transform({'state': backend.transformed['state'],
                                                           'actions': raw})['actions'])
            result['paths']['ordinary_eager'] = {
                'max_abs': float(np.max(np.abs(commands-reference))),
                'raw_vs_graph_max_abs': float(np.max(np.abs(raw-raw_graph)))}
            budget.finish(call, 'complete', result['paths']['ordinary_eager'])
        except BaseException:
            budget.finish(call, 'failed', {})
            raise
        try:
            _, commands, _, parity = capture_full(backend, sample, budget)
            result['paths']['post_graph_capture'] = {'max_abs': parity}
        except RuntimeError as exc:
            result['paths']['post_graph_capture'] = {'error': str(exc)}
        result['after'] = budget.summary()
        # Compare the service's native D3 runtime against the arbitrary-schedule
        # backend, rather than attributing a shared mismatch to instrumentation.
        call = budget.reserve({'kind': 'parity_diagnostic_native_d3', 'sample_id': sample['sample_id']})
        try:
            from scripts.spec.triton.pi0_infer import Pi0Inference
            embedding = backend.embeddings[sample['suite'], sample['prompt']]
            runtime = Pi0Inference(dict(backend.checkpoint, language_embeds=embedding),
                                   num_views=2, chunk_size=50, denoise_steps=3)
            raw = runtime.forward(*backend.inputs).float().cpu().numpy().copy()
            commands = np.asarray(backend.output_transform({'state': np.asarray(backend.transformed['state']).copy(),
                                                           'actions': raw.copy()})['actions'])
            result['paths']['native_d3'] = {
                'max_abs': float(np.max(np.abs(commands-reference))),
                'per_axis_max_abs': np.max(np.abs(commands-reference), axis=0).tolist(),
                'raw_vs_graph_max_abs': float(np.max(np.abs(raw-raw_graph))),
                'prompt_length': int(len(embedding))}
            budget.finish(call, 'complete', {**result['paths']['native_d3'],
                          'internal_warmup_capture_forwards': 4})
        except BaseException:
            budget.finish(call, 'failed', {})
            raise
        result['after'] = budget.summary()
        telemetry = [json.loads(line) for line in Path(sample['telemetry_path']).read_text().splitlines()]
        point = next(r for r in telemetry if int(r['step']) == sample['environment_step'])
        from scripts.experiment11.replay_state import client_state_from_telemetry
        original_state = client_state_from_telemetry(point, obs['observation/state'])
        result['state_roundtrip'] = {'cached_dtype': str(obs['observation/state'].dtype),
            'reconstructed_dtype': str(original_state.dtype),
            'max_cast_loss': float(np.max(np.abs(original_state-obs['observation/state'])))}
        for label, state in [('cached_state_float64', obs['observation/state'].astype(np.float64)),
                             ('telemetry_original_float64', original_state)]:
            backend.prepare(dict(obs, **{'observation/state': state}),
                            {'suite': sample['suite'], 'task_description': sample['prompt']}, noise)
            _, command, _ = accounted_infer(backend, (), 3, budget=budget,
                identity={'kind': 'parity_diagnostic_state', 'sample_id': sample['sample_id'], 'state_source': label})
            result['paths'][label] = {'max_abs': float(np.max(np.abs(command-reference))),
                                     'per_axis_max_abs': np.max(np.abs(command-reference), axis=0).tolist()}
        result['after'] = budget.summary()
        result.update(pid=os.getpid(), gpu=args.gpu, completed_epoch=time.time())
        dump_json(args.output, result)
        print(json.dumps(result), flush=True)
    finally:
        lease.close()


if __name__ == '__main__':
    main()

"""Paired VLM layer intervention and hidden/attn/FFN statistics on real phase observations."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import pickle
from pathlib import Path
import time

import numpy as np

from scripts.experiment10.common import (ROOT, RESULTS, CONVERTED, SUITES, NOISE_SEEDS,
                                         PROTOCOL, dump_json, sha256)
from scripts.experiment10.numerics import activity, representation, action_metrics


def write_csv(path, rows):
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def forward(runtime, images, state, noise, *, skip=(), capture=False, skip_attn=(), skip_ffn=()):
    import torch
    from scripts.spec.triton import pi0_infer as kernels
    w, b = runtime.weights, runtime.buffers
    with torch.inference_mode():
        b['observation_images_normalized'].copy_(images)
        b['observation_state_normalized'].copy_(state)
        b['diffusion_noise'].copy_(noise)
        b['encoder_x'][runtime.num_views * 256:].copy_(w['language_embeds'])
        kernels.vision_encoder(w, b, runtime.num_views)
        hidden, submodules = ([] if capture else None), ([] if capture else None)
        kernels.transformer_encoder(w, b, b['encoder_x'].shape[0],
                                    vlm_layer_indices=tuple(i for i in range(18) if i + 1 not in skip),
                                    hidden_trace=hidden, submodule_trace=submodules,
                                    skip_attn_layers=tuple(l-1 for l in skip_attn),
                                    skip_ffn_layers=tuple(l-1 for l in skip_ffn))
        kernels.transformer_decoder(w, b, b['encoder_x'].shape[0], denoise_steps=10)
        return b['diffusion_noise'].clone(), hidden, submodules


def activation_rows(meta, hidden, submodules):
    modalities = [('agent', slice(0, 256)), ('wrist', slice(256, 512)), ('language', slice(512, None))]
    rows = []
    identity = {k: meta[k] for k in ('sample_id', 'suite', 'task_id', 'episode_idx', 'semantic_phase', 'split')}
    for layer, (before, after_attn, after_ffn) in enumerate(submodules, 1):
        for modality, selected in modalities:
            inputs = before[selected]
            attn, ffn = after_attn[selected], after_ffn[selected]
            if not inputs.numel():
                continue
            representations = {'attn': representation(inputs, attn), 'ffn': representation(attn, ffn),
                               'block': representation(inputs, ffn)}
            for name, value in [('H_in', inputs), ('H_attn', attn), ('H_out', ffn),
                                ('delta_attn', attn.float() - inputs.float()),
                                ('delta_ffn', ffn.float() - attn.float()),
                                ('delta_block', ffn.float() - inputs.float())]:
                row = {**identity, 'layer': layer, 'modality': modality, 'tensor': name,
                       'execution_activation_dtype': str(before.dtype), **activity(value)}
                if name.startswith('delta_'):
                    row.update(representations[name.removeprefix('delta_')])
                rows.append(row)
    # L18 is an explicit special case; do not feed cosine=1 into candidate ranking.
    rows.append({**identity, 'layer': 18, 'modality': 'all', 'tensor': 'kv_only',
                 'valid': False, 'reason': 'H18_equals_H17_by_implementation; KV still used'})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, required=True)
    parser.add_argument('--sample-limit', type=int)
    args = parser.parse_args()
    import torch
    from scripts.experiment1.run_intermediate_action_probe import _load_prompt_embeddings
    from scripts.spec.spec_serve_policy import _build_policy_components
    from scripts.spec.triton import pi0_infer
    from scripts.spec.triton.triton_pi0_runtime import prepare_triton_inputs_from_transformed
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    manifest_path = RESULTS / 'observations' / args.suite / 'manifest.json'
    source = json.loads(manifest_path.read_text())
    if not source['complete'] or source['episode_count'] != 50:
        raise ValueError('Observation suite must contain 50 verified replay episodes')
    entries = source['observations'][:args.sample_limit] if args.sample_limit else source['observations']
    paths = [Path(row['path']) for row in entries]
    metadata = []
    for entry, path in zip(entries, paths):
        if sha256(path) != entry['sha256']:
            raise ValueError('Observation provenance mismatch')
        with np.load(path, allow_pickle=False) as data:
            metadata.append(json.loads(str(data['metadata_json'].item())))
    output = RESULTS / 'probe' / args.suite
    output.mkdir(parents=True, exist_ok=True)
    samples = output / 'samples'
    samples.mkdir(exist_ok=True)
    embeddings = _load_prompt_embeddings(CONVERTED, metadata)
    _, transform, output_transform = _build_policy_components(config_name='pi0_libero', checkpoint_dir=CONVERTED / 'base')
    weights_path = CONVERTED / 'base/base_weights.pkl'
    weight_hash = sha256(weights_path)
    with weights_path.open('rb') as stream:
        weights = pickle.load(stream)
    groups = {}
    for i, meta in enumerate(metadata):
        groups.setdefault(len(embeddings[meta['suite'], meta['task_description']]), []).append(i)
    completed = []
    for length, indices in sorted(groups.items()):
        checkpoint = dict(weights)
        first = metadata[indices[0]]
        checkpoint['language_embeds'] = embeddings[first['suite'], first['task_description']]
        runtime = pi0_infer.Pi0Inference(checkpoint, num_views=2, chunk_size=50, denoise_steps=10)
        for index in indices:
            meta, path = metadata[index], paths[index]
            target = samples / (meta['sample_id'] + '.npz')
            statistics_path = samples / (meta['sample_id'] + '.activity.csv')
            obs_hash = entries[index]['sha256']
            expected = {'observation_sha256': obs_hash, 'checkpoint_sha256': weight_hash,
                        'noise_seeds': list(NOISE_SEEDS), 'schema': 1, 'protocol': json.loads(json.dumps(PROTOCOL))}
            if target.exists() and statistics_path.exists():
                with np.load(target, allow_pickle=False) as old:
                    info = json.loads(str(old['metadata_json'].item()))
                    if any(info.get(k) != v for k, v in expected.items()) or not info.get('parity_passed'):
                        raise ValueError('Probe resume identity mismatch')
                completed.append(str(target))
                continue
            runtime.weights['language_embeds'].copy_(embeddings[meta['suite'], meta['task_description']].to('cuda'))
            with np.load(path, allow_pickle=False) as data:
                transformed = transform({'observation/image': data['agent_image'], 'observation/wrist_image': data['wrist_image'],
                                         'observation/state': data['robot_state'], 'prompt': str(data['prompt'].item())})
            def commands(raw):
                return np.asarray(output_transform({'state': np.asarray(transformed['state']).copy(),
                                  'actions': raw.float().cpu().numpy().copy()})['actions'], dtype=np.float32)
            baselines, variants, base_commands, variant_commands, noises, hashes = [], [], [], [], [], []
            start = time.monotonic()
            for seed_index, seed in enumerate(NOISE_SEEDS):
                rng = np.random.default_rng(np.random.SeedSequence([seed, int(obs_hash[:8], 16)]))
                noise_array = rng.standard_normal((50, 32), dtype=np.float32)
                hashes.append(__import__('hashlib').sha256(noise_array.tobytes()).hexdigest())
                images, state, noise = prepare_triton_inputs_from_transformed(transformed=transformed, device='cuda',
                                         action_horizon=50, action_dim=32, noise=noise_array)
                ordinary = runtime.forward(images, state, noise).clone()
                baseline, hidden, submodules = forward(runtime, images, state, noise, capture=seed_index == 0)
                if not torch.equal(ordinary, baseline):
                    dump_json(output / 'parity_failure.json', {'sample': meta['sample_id'], 'seed': seed,
                              'max_raw_difference': float((ordinary.float() - baseline.float()).abs().max())})
                    raise RuntimeError('Instrumented vs graph baseline parity failed; not increasing tolerance')
                if seed_index == 0:
                    write_csv(statistics_path, activation_rows(meta, hidden, submodules))
                del hidden, submodules
                candidate_raw, candidate_commands = [], []
                for layer in range(1, 19):
                    candidate, _, _ = forward(runtime, images, state, noise, skip=(layer,))
                    candidate_raw.append(candidate.float().cpu().numpy())
                    candidate_commands.append(commands(candidate))
                repeated, _, _ = forward(runtime, images, state, noise)
                if not torch.equal(repeated, baseline):
                    raise RuntimeError('Baseline changed after interventions')
                baselines.append(baseline.float().cpu().numpy())
                base_commands.append(commands(baseline))
                variants.append(np.stack(candidate_raw))
                variant_commands.append(np.stack(candidate_commands))
                noises.append(noise_array)
            info = {**meta, **expected, 'parity_passed': True, 'noise_sha256': hashes,
                    'layers_1based': list(range(1, 19)), 'prefix_noise_independent': True,
                    'activation_statistics_noise_seed': NOISE_SEEDS[0], 'seconds': time.monotonic() - start}
            temporary = target.with_name(target.stem + '.tmp.npz')
            np.savez_compressed(temporary, baseline_raw=np.stack(baselines), skip_raw=np.stack(variants),
                                baseline_commands=np.stack(base_commands), skip_commands=np.stack(variant_commands),
                                noise=np.stack(noises), metadata_json=np.asarray(json.dumps(info, sort_keys=True)))
            temporary.replace(target)
            completed.append(str(target))
            dump_json(output / 'progress.json', {'completed': len(completed), 'expected': len(entries),
                      'last_sample': meta['sample_id'], 'last_seconds': info['seconds']})
            print(json.dumps({'event': 'probe_sample_complete', 'completed': len(completed), 'expected': len(entries),
                              'sample_id': meta['sample_id'], 'seconds': info['seconds']}), flush=True)
        del runtime, checkpoint
        gc.collect()
        torch.cuda.empty_cache()
    if sha256(weights_path) != weight_hash:
        raise RuntimeError('Checkpoint changed')
    sensitivity, activations = [], []
    for filename in completed:
        with np.load(filename, allow_pickle=False) as data:
            info = json.loads(str(data['metadata_json'].item()))
            # NpzFile access decompresses each time: materialize once per sample.
            skip_raw, baseline_raw = data['skip_raw'], data['baseline_raw']
            skip_commands, baseline_commands = data['skip_commands'], data['baseline_commands']
            identity = {k: info[k] for k in ('sample_id', 'suite', 'task_id', 'episode_idx', 'semantic_phase', 'split')}
            for seed_index, seed in enumerate(NOISE_SEEDS):
                for layer in range(1, 19):
                    for metrics in action_metrics(skip_raw[seed_index, layer-1], baseline_raw[seed_index],
                               skip_commands[seed_index, layer-1], baseline_commands[seed_index]):
                        sensitivity.append({**identity, 'noise_seed': seed, 'layer': layer, **metrics})
        with Path(filename).with_suffix('.activity.csv').open() as stream:
            activations.extend(csv.DictReader(stream))
    write_csv(output / 'layer_action_sensitivity.csv', sensitivity)
    write_csv(output / 'activation_statistics.csv', activations)
    dump_json(output / 'summary.json', {'complete': True, 'sample_count': len(completed), 'samples': completed,
             'checkpoint_sha256': weight_hash, 'checkpoint_unchanged': True, 'protocol': PROTOCOL,
             'source_manifest_sha256': sha256(manifest_path), 'noise_seeds': NOISE_SEEDS,
             'gpu': torch.cuda.get_device_name(0), 'parity_passed': True,
             'partial_sample_limit': args.sample_limit})


if __name__ == '__main__':
    main()

"""Representative real-observation attn/ffn/both paired interventions, full Expert/D10."""
import argparse
import json
from pathlib import Path
import numpy as np

from scripts.experiment10.backend import Backend
from scripts.experiment10.common import RESULTS, SUITES, NOISE_SEEDS, dump_json, sha256
from scripts.experiment10.search_eval import load_entries
from scripts.experiment10.probe_vlm import forward, write_csv
from scripts.experiment10.numerics import action_metrics
from scripts.experiment10.update_details import update_details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, required=True)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    selection_path = RESULTS / 'submodule/selected_layers.json'
    selection = json.loads(selection_path.read_text())
    if selection['selection_split'] != 'search' or len(selection['selected']) > 4:
        raise RuntimeError('Invalid diagnostic selection')
    layers = [r['layer'] for r in selection['selected']]
    entries = [r for r in load_entries('search', coarse=True) if r['suite'] == args.suite]
    metadata = []
    for e in entries:
        with np.load(e['path'], allow_pickle=False) as data:
            metadata.append(json.loads(str(data['metadata_json'].item())))
    backend = Backend(metadata)
    order = sorted(range(len(entries)), key=lambda i: (len(backend.embeddings[args.suite, metadata[i]['task_description']]),
                                                       metadata[i]['sample_id']))
    output = RESULTS / 'submodule' / args.suite
    output.mkdir(parents=True, exist_ok=True)
    paths, metrics = [], []
    code_hashes = {name: sha256(Path(__file__).with_name(name)) for name in
                   ('probe_submodules.py', 'update_details.py', 'probe_vlm.py', 'backend.py', 'numerics.py')}
    for i in order:
        e, meta = entries[i], metadata[i]
        target = output / (meta['sample_id']+'.npz')
        paths.append(str(target))
        expected = {'selection_sha256': sha256(selection_path), 'observation_sha256': e['sha256'],
                    'checkpoint_sha256': backend.weight_hash, 'layers': layers, 'seeds': list(NOISE_SEEDS),
                    'numerical_detail_schema': 2, 'code_sha256': code_hashes}
        if not target.exists():
            if sha256(e['path']) != e['sha256']:
                raise RuntimeError('Input changed')
            with np.load(e['path'], allow_pickle=False) as data:
                observation = {'observation/image': data['agent_image'], 'observation/wrist_image': data['wrist_image'],
                               'observation/state': data['robot_state'], 'prompt': str(data['prompt'].item())}
            anchor = RESULTS / 'probe' / args.suite / 'samples' / (meta['sample_id']+'.npz')
            with np.load(anchor, allow_pickle=False) as data:
                noises = data['noise']
                anchor_info = json.loads(str(data['metadata_json'].item()))
            if anchor_info['checkpoint_sha256'] != backend.weight_hash:
                raise RuntimeError('Anchor checkpoint mismatch')
            for index, seed in enumerate(NOISE_SEEDS):
                expected_noise = np.random.default_rng(np.random.SeedSequence(
                    [seed, int(e['sha256'][:8], 16)])).standard_normal((50, 32), dtype=np.float32)
                if not np.array_equal(noises[index], expected_noise):
                    raise RuntimeError('Submodule noise provenance mismatch')
            raw, commands, references, reference_commands = [], [], [], []
            detail_rows, detail_arrays, shell_differences = [], {}, []
            for seed_index, noise in enumerate(noises):
                backend.prepare(observation, meta, noise)
                baseline, base_command, _ = backend.infer()
                eager, hidden, modules = forward(backend.runtime, *backend.inputs, capture=seed_index == 0)
                if not np.array_equal(baseline, eager.float().cpu().numpy()):
                    raise RuntimeError('Eager vs ordinary baseline parity failed')
                if seed_index == 0:
                    for layer in layers:
                        before, attn, after = modules[layer-1]
                        for modality, selected in [('agent', slice(0,256)), ('wrist', slice(256,512)), ('language', slice(512,None))]:
                            for module, residual in [('attn', attn.float()-before.float()),
                                                     ('ffn', after.float()-attn.float()), ('block', after.float()-before.float())]:
                                stats, tokens, channels = update_details(residual[selected])
                                key = 'L%s_%s_%s' % (layer, modality, module)
                                detail_arrays[key+'_token_norm'] = tokens
                                detail_arrays[key+'_channel_energy'] = channels
                                detail_rows.append({'sample_id': meta['sample_id'], 'layer': layer,
                                                    'modality': modality, 'module': module, **stats})
                del hidden, modules
                candidates, candidate_commands = [], []
                for layer in layers:
                    for mode in ('attn', 'ffn', 'both'):
                        candidate, _, _ = forward(backend.runtime, *backend.inputs,
                           skip_attn=(layer,) if mode in ('attn', 'both') else (),
                           skip_ffn=(layer,) if mode in ('ffn', 'both') else ())
                        array = candidate.float().cpu().numpy()
                        command = backend.output_transform({'state': np.asarray(backend.transformed['state']).copy(),
                                                            'actions': array.copy()})['actions']
                        candidates.append(array)
                        candidate_commands.append(command)
                    shell, _, _ = forward(backend.runtime, *backend.inputs, skip=(layer,))
                    shell_differences.append({'seed': NOISE_SEEDS[seed_index], 'layer': layer,
                        'max_both_vs_kv_shell_raw_difference': float(np.max(np.abs(candidates[-1]-shell.float().cpu().numpy())))})
                repeated, _, _ = forward(backend.runtime, *backend.inputs)
                if not np.array_equal(baseline, repeated.float().cpu().numpy()):
                    raise RuntimeError('Baseline changed after module interventions')
                raw.append(candidates); commands.append(candidate_commands)
                references.append(baseline); reference_commands.append(base_command)
            temporary = target.with_suffix('.tmp.npz')
            detail_arrays['metadata_json'] = np.asarray(json.dumps({**meta, **expected, 'noise_independent_prefix': True}))
            detail_target = output/(meta['sample_id']+'.update_details.npz')
            detail_temporary = detail_target.with_suffix('.tmp.npz')
            np.savez_compressed(detail_temporary, **detail_arrays)
            detail_temporary.replace(detail_target)
            write_csv(output/(meta['sample_id']+'.update_details.csv'), detail_rows)
            np.savez_compressed(temporary, raw=np.asarray(raw), commands=np.asarray(commands),
                baseline=np.asarray(references), baseline_commands=np.asarray(reference_commands), noise=noises,
                metadata_json=json.dumps({**meta, **expected, 'parity_passed': True,
                                         'both_vs_shell_checks': shell_differences}))
            temporary.replace(target)
        with np.load(target, allow_pickle=False) as data:
            info = json.loads(str(data['metadata_json'].item()))
            if any(info.get(k) != v for k, v in expected.items()) or not info['parity_passed']:
                raise RuntimeError('Submodule resume provenance mismatch')
            detail_path = output/(meta['sample_id']+'.update_details.npz')
            if not detail_path.exists():
                raise RuntimeError('Missing focused numerical details')
            with np.load(detail_path, allow_pickle=False) as detail:
                detail_info = json.loads(str(detail['metadata_json'].item()))
                if any(detail_info.get(k) != v for k, v in expected.items()):
                    raise RuntimeError('Focused numerical detail provenance mismatch')
                for layer in layers:
                    for modality in ('agent', 'wrist', 'language'):
                        for module in ('attn', 'ffn', 'block'):
                            key = 'L%s_%s_%s' % (layer, modality, module)
                            for suffix in ('_token_norm', '_channel_energy'):
                                vector = detail[key+suffix]
                                if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all() or np.any(vector < 0):
                                    raise RuntimeError('Invalid focused numerical vector: '+key+suffix)
                            if modality in ('agent', 'wrist') and len(detail[key+'_token_norm']) != 256:
                                raise RuntimeError('Visual patch layout mismatch')
            raw, commands = data['raw'], data['commands']
            baseline, base_command = data['baseline'], data['baseline_commands']
        for seed_index, seed in enumerate(NOISE_SEEDS):
            for j, layer in enumerate(layers):
                for k, mode in enumerate(('attn', 'ffn', 'both')):
                    for row in action_metrics(raw[seed_index, 3*j+k], baseline[seed_index],
                                              commands[seed_index, 3*j+k], base_command[seed_index]):
                        metrics.append({**{f: meta[f] for f in ('sample_id', 'suite', 'task_id', 'episode_idx', 'semantic_phase')},
                                        'layer': layer, 'mode': mode, 'noise_seed': seed, **row})
        dump_json(output/'progress.json', {'completed': len(paths), 'expected': len(entries)})
    write_csv(output/'submodule_action_metrics.csv', metrics)
    dump_json(output/'summary.json', {'complete': True, 'samples': paths, 'selection_sha256': sha256(selection_path),
              'checkpoint_sha256': backend.weight_hash, 'parity_passed': True,
              'sampling': 'search tasks init0 first observed replan per phase; three noises',
              'full_five_init_module_probe_complete': False, 'not_closed_loop_evidence': True})


if __name__ == '__main__':
    main()

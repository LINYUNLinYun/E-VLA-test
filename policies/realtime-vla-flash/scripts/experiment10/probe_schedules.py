"""Same-observation shadow probes for D1..D5 and existing draft masks (full expert layers)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from scripts.experiment10.backend import Backend
from scripts.experiment10.common import RESULTS, SUITES, NOISE_SEEDS, PROTOCOL, dump_json, sha256
from scripts.experiment10.numerics import action_metrics
from scripts.experiment10.probe_vlm import write_csv

CONDITIONS = [('D1', (), 1), ('D2', (), 2), ('D3', (), 3), ('D4', (), 4), ('D5', (), 5),
              ('V16_D10', (9, 10), 10), ('V13_D10', (3, 4, 7, 12, 15), 10),
              ('V13_D3', (3, 4, 7, 12, 15), 3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, required=True)
    parser.add_argument('--sample-limit', type=int)
    args = parser.parse_args()
    entries = json.loads((RESULTS / 'observations' / args.suite / 'manifest.json').read_text())['observations']
    if args.sample_limit:
        entries = entries[:args.sample_limit]
    metadata = []
    for entry in entries:
        with np.load(entry['path'], allow_pickle=False) as data:
            metadata.append(json.loads(str(data['metadata_json'].item())))
    backend = Backend(metadata)
    order = sorted(range(len(entries)), key=lambda i: (len(backend.embeddings[args.suite, metadata[i]['task_description']]),
                                                       metadata[i]['task_id'], metadata[i]['sample_id']))
    output = RESULTS / 'shadow' / args.suite
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in order:
        meta, entry = metadata[index], entries[index]
        target = output / (meta['sample_id'] + '.npz')
        paths.append(str(target))
        expected = {'observation_sha256': entry['sha256'], 'checkpoint_sha256': backend.weight_hash,
                    'conditions': [[name, list(skip), steps] for name, skip, steps in CONDITIONS],
                    'noise_seeds': list(NOISE_SEEDS), 'protocol': json.loads(json.dumps(PROTOCOL))}
        if target.exists():
            with np.load(target, allow_pickle=False) as old:
                saved = json.loads(str(old['metadata_json'].item()))
            if any(saved.get(k) != v for k, v in expected.items()):
                raise RuntimeError('Shadow resume identity mismatch')
            continue
        if sha256(entry['path']) != entry['sha256']:
            raise RuntimeError('Input observation changed')
        with np.load(entry['path'], allow_pickle=False) as data:
            observation = {'observation/image': data['agent_image'], 'observation/wrist_image': data['wrist_image'],
                           'observation/state': data['robot_state'], 'prompt': str(data['prompt'].item())}
        anchor = RESULTS / 'probe' / args.suite / 'samples' / (meta['sample_id'] + '.npz')
        base_raw, base_command, raw, commands, timing, differences = [], [], [], [], [], []
        started = time.monotonic()
        with np.load(anchor, allow_pickle=False) as reference:
            for seed_index, seed in enumerate(NOISE_SEEDS):
                backend.prepare(observation, meta, reference['noise'][seed_index])
                baseline, baseline_command, _ = backend.infer()
                differences.append(float(np.max(np.abs(baseline - reference['baseline_raw'][seed_index]))))
                candidate_raw, candidate_commands, candidate_timing = [], [], []
                for name, mask, steps in CONDITIONS:
                    a, b, c = backend.infer(mask, steps)
                    candidate_raw.append(a)
                    candidate_commands.append(b)
                    candidate_timing.append(c)
                repeated, _, _ = backend.infer()
                if not np.array_equal(repeated, baseline):
                    raise RuntimeError('D10 changed after schedule interventions')
                base_raw.append(baseline)
                base_command.append(baseline_command)
                raw.append(candidate_raw)
                commands.append(candidate_commands)
                timing.append(candidate_timing)
        info = {**meta, **expected, 'noise_source_probe_sha256': sha256(anchor),
                'cross_instance_d10_maxdiff': differences, 'within_instance_parity_passed': True,
                'cross_instance_differences_are_not_pruning_effects': True,
                'timings': timing, 'seconds': time.monotonic() - started}
        temporary = target.with_name(target.stem + '.tmp.npz')
        np.savez_compressed(temporary, baseline_raw=np.asarray(base_raw), baseline_commands=np.asarray(base_command),
                            candidate_raw=np.asarray(raw), candidate_commands=np.asarray(commands),
                            metadata_json=np.asarray(json.dumps(info, sort_keys=True)))
        temporary.replace(target)
        print(json.dumps({'event': 'shadow_sample_complete', 'completed': len(paths), 'expected': len(entries),
                          'sample': meta['sample_id'], 'seconds': info['seconds']}), flush=True)
        dump_json(output / 'progress.json', {'completed': len(paths), 'expected': len(entries), 'last_sample': meta['sample_id']})
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            info = json.loads(str(data['metadata_json'].item()))
            candidate_raw, baseline_raw = data['candidate_raw'], data['baseline_raw']
            candidate_commands, baseline_commands = data['candidate_commands'], data['baseline_commands']
            identity = {k: info[k] for k in ('sample_id', 'suite', 'task_id', 'episode_idx', 'semantic_phase', 'split')}
            for seed_index, seed in enumerate(NOISE_SEEDS):
                for condition_index, (name, mask, steps) in enumerate(CONDITIONS):
                    for metrics in action_metrics(candidate_raw[seed_index, condition_index],
                             baseline_raw[seed_index], candidate_commands[seed_index, condition_index],
                             baseline_commands[seed_index]):
                        rows.append({**identity, 'condition': name, 'noise_seed': seed, 'steps': steps, **metrics})
    write_csv(output / 'shadow_action_metrics.csv', rows)
    dump_json(output / 'summary.json', {'complete': True, 'sample_count': len(paths), 'samples': paths,
              'conditions': CONDITIONS, 'protocol': PROTOCOL, 'checkpoint_sha256': backend.weight_hash,
              'partial_sample_limit': args.sample_limit, 'not_closed_loop_evidence': True})


if __name__ == '__main__':
    main()

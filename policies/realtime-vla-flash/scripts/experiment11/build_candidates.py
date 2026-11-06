"""Freeze the evidence-tagged direct-closed-loop candidate manifest."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from scripts.experiment11.protocol import CONVERTED, PROTOCOL, RESULTS, SOURCE, SUITES
from scripts.experiment10.common import dump_json


MASKS = {
    'V16': [
        ('low_impact_spread', [3, 5]), ('low_impact_spread', [4, 10]),
        ('mixed_spread', [6, 11]), ('mixed_spread', [8, 12]),
        ('adjacent_challenge', [3, 4]), ('adjacent_challenge', [9, 10]),
        ('adjacent_challenge', [11, 12]), ('low_cosine_challenge', [13, 15]),
        ('low_cosine_challenge', [14, 16]), ('early_layer_challenge', [2, 4]),
    ],
    'V15': [
        ('low_impact_spread', [3, 5, 10]), ('mixed_spread', [4, 8, 12]),
        ('adjacent_challenge', [3, 4, 10]), ('adjacent_challenge', [6, 7, 11]),
        ('adjacent_challenge', [8, 9, 12]), ('mixed_spread', [3, 7, 11]),
        ('low_cosine_challenge', [13, 15, 16]), ('early_layer_challenge', [2, 5, 9]),
    ],
    'V14': [
        ('low_impact_spread', [3, 5, 8, 11]), ('mixed_spread', [4, 6, 10, 12]),
        ('adjacent_challenge', [3, 4, 9, 11]), ('adjacent_challenge', [5, 6, 10, 11]),
        ('low_cosine_challenge', [7, 9, 11, 13]), ('early_layer_challenge', [2, 4, 8, 12]),
    ],
    'V13': [
        ('custom_v13_one_swap_restore_L3_add_L13', [4, 7, 12, 13, 15]),
        ('custom_v13_one_swap_restore_L4_add_L14', [3, 7, 12, 14, 15]),
        ('custom_v13_one_swap_restore_L7_add_L6', [3, 4, 6, 12, 15]),
        ('custom_v13_one_swap_restore_L12_add_L11', [3, 4, 7, 11, 15]),
        ('nonadjacent_L2_retained', [3, 5, 8, 10, 12]),
    ],
}
CUSTOM_V13 = [3, 4, 7, 12, 15]
CHECKPOINT_SHA256 = 'c423a508742f8461fa17d8fb42951837b4419804842ed27a4890a9903042f891'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_layer_evidence():
    prior_path = SOURCE / 'phase_analysis/layer_prior.json'
    prior = json.loads(prior_path.read_text())
    rows = {}
    for suite in SUITES:
        source = SOURCE / 'probe' / suite / 'layer_action_sensitivity.csv'
        with source.open(newline='') as stream:
            for row in csv.DictReader(stream):
                if int(row['horizon']) != 4 or row['split'] != 'search':
                    continue
                layer = int(row['layer'])
                entry = rows.setdefault(layer, {'rmse': [], 'gripper_flip': []})
                value = float(row['normalized_continuous_rmse'])
                if value == value:
                    entry['rmse'].append(value)
                entry['gripper_flip'].append(float(row['gripper_flip_rate']))
    evidence = {}
    for item in prior['layers']:
        layer = int(item['layer'])
        values = rows.get(layer, {'rmse': [], 'gripper_flip': []})
        evidence[str(layer)] = {
            'robust_cosine_p10': item['robust_cosine_p10'],
            'cosine_cell_count': int(item['cosine_cell_count']),
            'single_layer_action_rmse_h4_mean': (sum(values['rmse']) / len(values['rmse'])
                                                 if values['rmse'] else None),
            'single_layer_gripper_flip_h4_mean': (sum(values['gripper_flip']) / len(values['gripper_flip'])
                                                  if values['gripper_flip'] else None),
            'action_samples_h4': len(values['rmse']),
        }
    return evidence, prior_path, [SOURCE / 'probe' / suite / 'layer_action_sensitivity.csv' for suite in SUITES]


def build():
    checkpoint = CONVERTED / 'base/base_weights.pkl'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha = sha256(checkpoint)
    if checkpoint_sha != CHECKPOINT_SHA256:
        raise RuntimeError('Converted checkpoint does not match the prior experiment: ' + checkpoint_sha)
    evidence, prior_path, action_paths = load_layer_evidence()
    sources = [prior_path] + action_paths
    source_hashes = {str(path.relative_to(SOURCE)): sha256(path) for path in sources}
    candidates = []
    seen = {tuple(CUSTOM_V13)}
    for family, entries in MASKS.items():
        expected_deletes = 18 - int(family[1:])
        if len(entries) != {'V16': 10, 'V15': 8, 'V14': 6, 'V13': 5}[family]:
            raise RuntimeError('Wrong family quota: ' + family)
        for index, (reason, skip) in enumerate(entries, 1):
            skip = sorted(skip)
            key = tuple(skip)
            if key in seen or len(skip) != expected_deletes:
                raise RuntimeError('Duplicate or wrong deletion budget: %s %s' % (family, skip))
            if any(layer in (1, 17, 18) for layer in skip) or any(layer < 1 or layer > 18 for layer in skip):
                raise RuntimeError('Protected/out-of-range layer in candidate: %s' % skip)
            seen.add(key)
            variant = '%s_A%02d' % (family, index)
            candidates.append({
                'variant_id': variant,
                'family': family,
                'skip_layers': skip,
                'retain_layers': [layer for layer in range(1, 19) if layer not in skip],
                'deleted_layer_count': expected_deletes,
                'proposal_class': reason,
                'proposal_rule': 'evidence-guided structured coverage; no offline combined-mask scoring or candidate elimination',
                'layer_evidence': {str(layer): evidence[str(layer)] for layer in skip},
                'exceptions': [
                    ('allows adjacent deletion; explicit structure challenge' if
                     any(b == a + 1 for a, b in zip(skip, skip[1:])) else None),
                    ('deletes early L2; explicit preference challenge' if 2 in skip else None),
                    ('deletes low-robust-cosine layer(s); explicit challenge' if
                     any(evidence[str(layer)]['robust_cosine_p10'] is not None and
                         evidence[str(layer)]['robust_cosine_p10'] < 0.98 for layer in skip) else None),
                ],
                'condition': {
                    'name': variant + '_D3_R5', 'skip': skip, 'steps': 3,
                    'mode': 'static', 'horizon': 5, 'replan_steps': 5,
                    'checkpoint_sha256': checkpoint_sha,
                },
            })
    if len(candidates) != 29:
        raise RuntimeError('Expected 29 new masks')
    baseline = {'name': 'Full18_D3_R5', 'skip': [], 'steps': 3, 'mode': 'static',
                'horizon': 5, 'replan_steps': 5, 'checkpoint_sha256': checkpoint_sha,
                'save_full_inputs': True}
    original = {'name': 'V13_Custom_D3_R5', 'skip': CUSTOM_V13, 'steps': 3,
                'mode': 'static', 'horizon': 5, 'replan_steps': 5,
                'checkpoint_sha256': checkpoint_sha}
    order = ['V13', 'V14', 'V15', 'V16']
    candidates.sort(key=lambda row: (order.index(row['family']), row['variant_id']))
    conditions = [baseline, original] + [item['condition'] for item in candidates]
    config = {'schema': 2, 'purpose': 'Direct four-suite static VLM layer-skip evaluation; D3, action horizon 50, execute/replan 5',
              'protocol': PROTOCOL, 'source_hashes': source_hashes,
              'candidate_manifest_sha256_pending': True, 'conditions': conditions}
    RESULTS.mkdir(parents=True, exist_ok=True)
    protocol_dir = RESULTS / 'protocol'
    protocol_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'schema': 1, 'frozen_before_closed_loop': True, 'candidate_count': len(candidates),
                'family_counts': {family: sum(c['family'] == family for c in candidates) for family in MASKS},
                'custom_v13_reference_skip_layers': CUSTOM_V13,
                'checkpoint_sha256': checkpoint_sha, 'proposal_sources_sha256': source_hashes,
                'protocol': PROTOCOL, 'candidates': candidates,
                'reference_conditions': [baseline, original],
                'selection_warning': 'Layer evidence proposes candidates only; it was not used to screen combinations or predict closed-loop success.'}
    manifest_path = protocol_dir / 'candidate_manifest.json'
    config_path = protocol_dir / 'closed_loop_config.json'
    if manifest_path.exists() or config_path.exists():
        raise FileExistsError('Frozen protocol already exists; inspect it instead of overwriting')
    dump_json(manifest_path, manifest)
    config['candidate_manifest_sha256_pending'] = sha256(manifest_path)
    dump_json(config_path, config)
    table_path = protocol_dir / 'candidate_manifest.md'
    lines = ['# Frozen direct closed-loop candidates', '',
             '| ID | Budget | Skip layers | Proposal class | Heuristic evidence / exception |',
             '|---|---:|---|---|---|']
    for item in candidates:
        rmse = [item['layer_evidence'][str(layer)]['single_layer_action_rmse_h4_mean'] for layer in item['skip_layers']]
        cos = [item['layer_evidence'][str(layer)]['robust_cosine_p10'] for layer in item['skip_layers']]
        exceptions = [x for x in item['exceptions'] if x]
        lines.append('| %s | %d | `%s` | %s | mean single-layer H4 RMSE %.4f; mean cosine p10 %s; %s |' %
                     (item['variant_id'], item['deleted_layer_count'], item['skip_layers'], item['proposal_class'],
                      sum(rmse) / len(rmse),
                      'n/a' if any(v is None for v in cos) else '%.4f' % (sum(cos) / len(cos)),
                      '; '.join(exceptions) if exceptions else 'no declared exception'))
    table_path.write_text('\n'.join(lines) + '\n')
    return manifest_path, config_path, table_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs and print the frozen candidate counts')
    args = parser.parse_args()
    paths = build()
    print(json.dumps({'manifest': str(paths[0]), 'config': str(paths[1]), 'table': str(paths[2]),
                      'candidate_count': 29, 'episodes_primary': 6200, 'diagnostic_episode_cap': 1400,
                      'gpu_jobs': 124, 'launch': False}, indent=2))


if __name__ == '__main__':
    main()

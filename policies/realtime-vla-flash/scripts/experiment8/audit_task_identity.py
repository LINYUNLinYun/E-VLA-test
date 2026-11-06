#!/usr/bin/env python3
"""Compare saved testset identities against the installed evaluation suites."""
import argparse
import hashlib
import json
from pathlib import Path

from libero.libero import benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    samples = json.loads(manifest.read_text())['samples']
    suites = {}
    rows = []
    for sample in samples:
        name = sample['suite_name']
        if name not in suites:
            suites[name] = benchmark.get_benchmark_dict()[name]()
        task = suites[name].get_task(int(sample['task_id']))
        state = manifest.parent / sample['init_state_path']
        actual_hash = hashlib.sha256(state.read_bytes()).hexdigest()
        row = {
            'sample_id': sample['sample_id'], 'runtime_task_name': task.name,
            'manifest_task_name': sample['task_name'],
            'name_matches': task.name == sample['task_name'],
            'language_matches': task.language.strip() == sample['task_description'].strip(),
            'demo_path_matches': suites[name].get_task_demonstration(int(sample['task_id'])) == sample['source_hdf5_path'],
            'state_hash_matches': actual_hash == sample['init_state_sha256'],
        }
        rows.append(row)
    checks = ['name_matches', 'language_matches', 'demo_path_matches', 'state_hash_matches']
    report = {
        'manifest': str(manifest), 'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
        'sample_count': len(rows),
        'mismatch_counts': {key: sum(not row[key] for row in rows) for key in checks},
        'scope': 'Task mapping and saved-state file integrity only; does not establish simulator XML compatibility or baseline success.',
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'rows'}, indent=2))
    return int(any(report['mismatch_counts'].values()))


if __name__ == '__main__':
    raise SystemExit(main())

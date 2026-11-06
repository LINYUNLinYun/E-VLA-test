#!/usr/bin/env python3
"""Run multi-noise probes serially on one explicitly selected idle GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--seeds', type=int, nargs='+', default=[12345,23456,34567])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    checkpoint = Path('/root/nfs/embodied/benchmark/shared/checkpoints/realtime-vla-flash/converted')
    payload = json.loads((args.observations / 'manifest.json').read_text())
    paths = [Path(path) for path in payload['observations']]
    groups = {'libero_90': [], 'libero_10': []}
    for path in paths:
        matched = [suite for suite in groups if path.name.startswith(suite + '_task')]
        if len(matched) != 1:
            raise ValueError(f'Unsupported observation identity: {path}')
        groups[matched[0]].append(path)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               OPENPI_DATA_HOME='/root/nfs/embodied/benchmark/shared/checkpoints/openpi')
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError('Seeds must be unique nonnegative integers')
    for seed in args.seeds:
        for suite, observations in groups.items():
            if not observations:
                continue
            command = [sys.executable, str(root / 'scripts/experiment8/collect_submodules.py'),
                       *map(str, observations), '--base-triton-path', str(checkpoint / 'base'),
                       '--prompt-manifest', str(checkpoint / ('draft' if suite == 'libero_90' else 'draft_10') / 'manifest.json'),
                       '--output-dir', str(args.output_dir / f'seed{seed}' / suite), '--seed', str(seed)]
            if args.dry_run:
                command.append('--dry-run')
            print(f'BEGIN seed={seed} suite={suite} observations={len(observations)}', flush=True)
            subprocess.run(command, env=env, cwd=root, check=True)
            print(f'COMPLETE seed={seed} suite={suite}', flush=True)


if __name__ == '__main__':
    main()

"""Create and optionally launch the three-condition single-init GPU smoke."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from scripts.experiment10.common import dump_json
from scripts.experiment11.protocol import PROTOCOL, RESULTS, ROOT, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--suite', choices=PROTOCOL['suites'], default='libero_object')
    parser.add_argument('--task-id', type=int, default=0)
    parser.add_argument('--init-index', type=int, default=0)
    parser.add_argument('--status-tag', default='smoke', choices=('smoke', 'smoke_retry1', 'smoke_retry2'))
    args = parser.parse_args()
    manifest = json.loads((RESULTS / 'protocol/candidate_manifest.json').read_text())
    config = json.loads((RESULTS / 'protocol/closed_loop_config.json').read_text())
    new_candidate = next(row for row in manifest['candidates'] if row['family'] == 'V13' and row['variant_id'] == 'V13_A05')
    names = {'Full18_D3_R5', 'V13_Custom_D3_R5', new_candidate['condition']['name']}
    conditions = []
    for condition in config['conditions']:
        if condition['name'] not in names:
            continue
        conditions.append({**condition, 'save_full_inputs': False, 'suites': [args.suite],
                           'task_ids': [args.task_id], 'init_indices': [args.init_index]})
    if len(conditions) != 3:
        raise RuntimeError('Smoke must cover Full18, original V13 and one new V13 mask')
    config_path = RESULTS / 'protocol/smoke_closed_loop_config.json'
    dump_json(config_path, {'schema': 1, 'purpose': 'One-task/init parity, queue and H264 validation only',
                            'parent_candidate_manifest_sha256': __import__('hashlib').sha256(
                                (RESULTS / 'protocol/candidate_manifest.json').read_bytes()).hexdigest(),
                            'conditions': conditions})
    command = [str(ROOT / '.venv/bin/python'), '-m', 'scripts.experiment10.run_closed_loop',
               '--conditions', str(config_path), '--status-path', str(RESULTS / ('experiment11_%s_status.json' % args.status_tag))]
    if args.launch:
        command.append('--launch')
    print(json.dumps({'command': command, 'suite': args.suite, 'task_id': args.task_id,
                      'init_index': args.init_index, 'episodes': 3, 'launch': args.launch}, indent=2))
    if args.launch:
        env = environment()
        env['EXP11_SOURCE_RESULTS'] = str(RESULTS.parent / 'experiment10_phase_draft_20261009')
        return subprocess.call(command, cwd=ROOT, env=env)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

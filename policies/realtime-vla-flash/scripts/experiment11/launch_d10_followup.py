"""Dry-run or launch the frozen post-D3 D10 diagnostic matrix."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

from scripts.experiment11.protocol import RESULTS, ROOT, environment


def process_live(pid: int) -> bool:
    try:
        state = (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1][0]
    except (FileNotFoundError, IndexError):
        return False
    return state not in ('Z', 'X')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true', help='Start/resume the D10 GPU scheduler')
    args = parser.parse_args()

    d3_status_path = RESULTS / 'experiment11_closed_loop_status.json'
    d3_status = json.loads(d3_status_path.read_text())
    d3_audit_path = RESULTS / 'analysis/closed_loop_d3/completion_audit.json'
    if (not d3_status.get('complete') or d3_status.get('active') or d3_status.get('pending') or
            d3_status.get('failures') or not d3_audit_path.is_file()):
        raise RuntimeError('D3 matrix and completion audit must be complete before launching D10')

    config_path = RESULTS / 'protocol/closed_loop_config_d10_followup.json'
    if not config_path.is_file():
        raise FileNotFoundError('Run build_d10_followup.py after D3 analysis first')
    config = json.loads(config_path.read_text())
    if (not config.get('conditions') or len(config['conditions']) > 7 or
            any(c.get('steps') != 10 or c.get('horizon') != 5 or c.get('replan_steps') != 5
                for c in config['conditions'])):
        raise RuntimeError('D10 config violates the frozen diagnostic boundary')

    status_path = RESULTS / 'experiment11_d10_closed_loop_status.json'
    if status_path.is_file():
        previous = json.loads(status_path.read_text())
        live = [row for row in previous.get('active', []) if process_live(int(row['pid']))]
        if live:
            raise RuntimeError(f'D10 scheduler already has live workers: {live}')
        if previous.get('complete'):
            raise RuntimeError('D10 matrix is already complete; refusing duplicate launch')

    command = [str(ROOT / '.venv/bin/python'), '-m', 'scripts.experiment10.run_closed_loop',
               '--conditions', str(config_path), '--status-path', str(status_path)]
    if args.launch:
        command.append('--launch')
    print(' '.join(command), flush=True)
    if not args.launch:
        return 0
    env = environment()
    env['EXP11_METADATA_KEY'] = 'experiment11'
    env['EXP11_SOURCE_RESULTS'] = str(RESULTS.parent / 'experiment10_phase_draft_20261009')
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == '__main__':
    raise SystemExit(main())

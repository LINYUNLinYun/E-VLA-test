"""Dry-run or launch the frozen Experiment 11 closed-loop matrix."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from scripts.experiment11.protocol import RESULTS, ROOT, environment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true', help='Start the resumable GPU scheduler')
    args = parser.parse_args()
    config = RESULTS / 'protocol/closed_loop_config_formal.json'
    if not config.is_file():
        raise FileNotFoundError('Freeze candidates first: scripts.experiment11.build_candidates')
    env = environment()
    env['EXP11_METADATA_KEY'] = 'experiment11'
    env['EXP11_SOURCE_RESULTS'] = str(RESULTS.parent / 'experiment10_phase_draft_20261009')
    command = [str(ROOT / '.venv/bin/python'), '-m', 'scripts.experiment10.run_closed_loop',
               '--conditions', str(config), '--status-path', str(RESULTS / 'experiment11_closed_loop_status.json')]
    if args.launch:
        command.append('--launch')
    print(' '.join(command), flush=True)
    return subprocess.call(command, cwd=ROOT, env=env)


if __name__ == '__main__':
    raise SystemExit(main())

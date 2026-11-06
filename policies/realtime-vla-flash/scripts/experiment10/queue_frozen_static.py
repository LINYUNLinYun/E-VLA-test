"""Queue up to two frozen static skip masks for the official five-init suites."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.experiment10.common import RESULTS, ROOT, dump_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-pid', type=int, required=True)
    args = parser.parse_args()
    lock = (RESULTS/'.frozen_static_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = RESULTS/'search/frozen_candidates.json'
    status = RESULTS/'frozen_static_queue_status.json'
    while not source.exists():
        try:
            stat = (Path('/proc')/str(args.search_pid)/'stat').read_text()
            if stat.split(') ', 1)[1][0] in ('Z', 'X'):
                dump_json(status, {'pid': os.getpid(), 'state': 'search_terminal_without_frozen_candidates',
                                   'search_pid': args.search_pid})
                raise SystemExit('Search exited before writing frozen candidates')
        except FileNotFoundError:
            dump_json(status, {'pid': os.getpid(), 'state': 'search_handle_missing_without_frozen_candidates',
                               'search_pid': args.search_pid})
            raise SystemExit('Search handle missing before frozen candidates')
        dump_json(status, {'pid': os.getpid(), 'state': 'waiting_for_frozen_candidates',
                           'search_pid': args.search_pid, 'source': str(source)})
        time.sleep(20)
    frozen = json.loads(source.read_text())
    if frozen.get('complete') is not True:
        raise SystemExit('Frozen candidate file is not marked complete')
    masks = frozen.get('static_masks', [])
    if not masks or len(masks) > 2:
        raise SystemExit('Expected one or two frozen static masks')
    baseline = json.loads((RESULTS/'closed_loop/Full18_D10/libero_object/run/validated.json').read_text())
    checkpoint_sha = baseline['condition']['checkpoint_sha256']
    conditions = []
    for index, mask in enumerate(masks, 1):
        mask = sorted(set(int(layer) for layer in mask))
        if not mask or any(layer < 1 or layer > 18 for layer in mask):
            raise ValueError('Invalid frozen mask')
        conditions.append({'name': 'FrozenStatic%02d' % index, 'skip': mask,
                           'steps': 10, 'mode': 'static', 'horizon': 12,
                           'checkpoint_sha256': checkpoint_sha})
    config = {'schema': 1, 'purpose': 'Frozen search-selected static masks; five official inits per task',
              'frozen_candidates_sha256': sha256(source),
              'search_split_only': True, 'conditions': conditions}
    config_path = RESULTS/'frozen_static_closed_loop_config.json'
    if config_path.exists():
        saved = json.loads(config_path.read_text())
        if saved != config:
            raise RuntimeError('Existing frozen closed-loop config disagrees; preserve for review')
    else:
        dump_json(config_path, config)
    command = [str(ROOT/'.venv/bin/python'), '-u', '-m', 'scripts.experiment10.run_closed_loop',
               '--conditions', str(config_path), '--launch']
    log_path = RESULTS/'logs/frozen_static_closed_loop.log'
    with log_path.open('a') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        dump_json(status, {'pid': os.getpid(), 'state': 'running', 'child_pid': child.pid,
                           'conditions': conditions, 'config_sha256': sha256(config_path),
                           'frozen_candidates_sha256': sha256(source),
                           'search_pid': args.search_pid})
        code = child.wait()
    dump_json(status, {'pid': os.getpid(), 'state': 'finished' if code == 0 else 'failed',
                       'child_pid': child.pid, 'returncode': code, 'conditions': conditions,
                       'config_sha256': sha256(config_path), 'frozen_candidates_sha256': sha256(source),
                       'search_pid': args.search_pid})
    raise SystemExit(code)


if __name__ == '__main__':
    main()

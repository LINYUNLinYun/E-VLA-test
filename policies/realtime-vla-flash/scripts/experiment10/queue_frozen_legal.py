"""After frozen challenger rollouts, evaluate up to two heuristic-legal masks."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.experiment10.common import RESULTS, ROOT, dump_json, sha256
from scripts.experiment10.search_rules import Rules, pareto_select


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--challenge-queue-pid', type=int, required=True)
    args = parser.parse_args()
    lock = (RESULTS/'.frozen_legal_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    challenge_status = RESULTS/'frozen_static_queue_status.json'
    status = RESULTS/'frozen_legal_queue_status.json'
    while True:
        data = json.loads(challenge_status.read_text())
        if data.get('pid') != args.challenge_queue_pid:
            raise RuntimeError('Unexpected challenge queue identity')
        if data.get('state') in ('finished', 'failed'):
            break
        try:
            stat = (Path('/proc')/str(args.challenge_queue_pid)/'stat').read_text()
            if stat.split(') ', 1)[1][0] in ('Z', 'X'):
                raise RuntimeError('Challenge queue exited before writing terminal status')
        except FileNotFoundError as error:
            raise RuntimeError('Challenge queue handle disappeared before terminal status') from error
        dump_json(status, {'pid': os.getpid(), 'state': 'waiting_for_challenge_rollouts',
                           'challenge_queue_pid': args.challenge_queue_pid})
        time.sleep(20)
    if data['state'] != 'finished' or data.get('returncode') != 0:
        raise SystemExit('Challenge rollouts did not complete successfully')

    prior_path = RESULTS/'phase_analysis/layer_prior.json'
    validation_path = RESULTS/'search/validation.json'
    prior = json.loads(prior_path.read_text())
    validation = json.loads(validation_path.read_text())
    if prior.get('selection_split') != 'search' or validation.get('role') != 'validation':
        raise RuntimeError('Legal mask selection provenance mismatch')
    rules = Rules(frozenset(x['layer'] for x in prior['layers'] if x['protected_low_cosine']))
    scored = {}
    for row in validation['candidates']:
        mask = tuple(row['skip_layers'])
        if not rules.violations(mask):
            scored[mask] = tuple(row['objectives'])
    selected = pareto_select(scored, width=2)
    if not selected:
        raise RuntimeError('No heuristic-legal validation candidates are available')
    baseline = json.loads((RESULTS/'closed_loop/Full18_D10/libero_object/run/validated.json').read_text())
    conditions = [{'name': 'FrozenLegal%02d' % index, 'skip': list(mask), 'steps': 10,
                   'mode': 'static', 'horizon': 12,
                   'checkpoint_sha256': baseline['condition']['checkpoint_sha256']}
                  for index, mask in enumerate(selected, 1)]
    config = {'schema': 1, 'purpose': 'Heuristic-legal masks, Pareto-selected from validation only',
              'search_split_only': True, 'validation_sha256': sha256(validation_path),
              'prior_sha256': sha256(prior_path), 'mask_violations': {c['name']: [] for c in conditions},
              'conditions': conditions}
    config_path = RESULTS/'frozen_legal_closed_loop_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise RuntimeError('Existing legal-mask config disagrees; preserve for review')
    else:
        dump_json(config_path, config)
    command = [str(ROOT/'.venv/bin/python'), '-u', '-m', 'scripts.experiment10.run_closed_loop',
               '--conditions', str(config_path), '--launch']
    log_path = RESULTS/'logs/frozen_legal_closed_loop.log'
    with log_path.open('a') as stream:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        dump_json(status, {'pid': os.getpid(), 'state': 'running', 'child_pid': child.pid,
                           'conditions': conditions, 'config_sha256': sha256(config_path),
                           'challenge_queue_pid': args.challenge_queue_pid})
        code = child.wait()
    dump_json(status, {'pid': os.getpid(), 'state': 'finished' if code == 0 else 'failed',
                       'child_pid': child.pid, 'returncode': code, 'conditions': conditions,
                       'config_sha256': sha256(config_path),
                       'challenge_queue_pid': args.challenge_queue_pid})
    raise SystemExit(code)


if __name__ == '__main__':
    main()

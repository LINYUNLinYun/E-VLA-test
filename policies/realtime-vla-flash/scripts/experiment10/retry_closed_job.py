"""Retry one verified terminal invalid batch, archiving it before fresh outputs."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.experiment10.common import ROOT, RESULTS, SUITES, claim_gpu, managed_gpus, dump_json
from scripts.experiment10.run_followup import is_live
from scripts.experiment10.run_pipeline import free_gpus, event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--condition-index', type=int, required=True)
    parser.add_argument('--suite', choices=SUITES, required=True)
    args = parser.parse_args()
    config = ROOT/'scripts/experiment10/closed_loop_config.json'
    condition = json.loads(config.read_text())['conditions'][args.condition_index]
    name = condition['name']+'_'+args.suite
    lock = (RESULTS/('.retry_'+name+'.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = RESULTS/('retry_'+name+'_status.json')
    destination = RESULTS/'closed_loop'/condition['name']/args.suite
    if (destination/'run/validated.json').exists():
        raise SystemExit('Batch already validated; do not rerun')
    main_status = json.loads((RESULTS/'closed_loop_status.json').read_text())
    if any(r['condition'] == condition['name'] and r['suite'] == args.suite and is_live(r['pid'])
           for r in main_status['active']):
        raise SystemExit('Original batch still live; do not restart')
    if not any(r['condition'] == condition['name'] and r['suite'] == args.suite for r in main_status['failures']):
        raise SystemExit('No verified failed batch to retry')
    previous, next_poll = managed_gpus(), 0.
    while True:
        current = managed_gpus()
        if previous-current:
            next_poll = 0.
        previous = current
        dump_json(status_path, {'pid': os.getpid(), 'state': 'waiting_for_idle_gpu', 'condition': condition})
        if len(current) < 4 and time.monotonic() >= next_poll:
            next_poll = time.monotonic()+300
            for gpu in free_gpus():
                lease = claim_gpu(gpu)
                if lease is None:
                    continue
                timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
                archive = RESULTS/'failed_attempts'/name/timestamp
                archive.parent.mkdir(parents=True, exist_ok=True)
                destination.rename(archive)  # preserve all failed evidence; no deletion
                destination.mkdir(parents=True)
                event(event='invalid_batch_archived', condition=condition['name'], suite=args.suite,
                      archive=str(archive))
                with (destination/'retry_worker.log').open('a') as stream:
                    child = subprocess.Popen([str(ROOT/'.venv/bin/python'), '-m', 'scripts.experiment10.run_closed_loop',
                        '--conditions', str(config), '--worker', '--condition-index', str(args.condition_index),
                        '--suite', args.suite, '--gpu', str(gpu)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    dump_json(status_path, {'pid': os.getpid(), 'state': 'running', 'archive': str(archive),
                              'active': [{'gpu': gpu, 'pid': child.pid}], 'condition': condition})
                    code = child.wait()
                lease.close()
                dump_json(status_path, {'pid': os.getpid(), 'state': 'finished' if code == 0 else 'failed',
                                       'returncode': code, 'archive': str(archive), 'condition': condition})
                raise SystemExit(code)
        time.sleep(2)


if __name__ == '__main__':
    main()

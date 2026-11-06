"""Wait on verified upstream handles, then launch the bounded search once."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time

from scripts.experiment10.common import RESULTS, ROOT, dump_json
from scripts.experiment10.run_followup import is_live


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-pid', type=int, required=True)
    args = parser.parse_args()
    lock = (RESULTS / '.search_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    prior = RESULTS / 'phase_analysis/layer_prior.json'
    status = RESULTS / 'search_queue_status.json'
    while not prior.exists():
        if not is_live(args.upstream_pid):
            dump_json(status, {'pid': os.getpid(), 'state': 'upstream_terminal_without_prior'})
            raise SystemExit('Upstream terminal without prior; inspect logs before retry')
        dump_json(status, {'pid': os.getpid(), 'state': 'waiting_for_prior',
                           'verified_upstream_pid': args.upstream_pid})
        time.sleep(10)  # Process/file checks only; does not poll GPUs.
    data = json.loads(prior.read_text())
    if not data['complete_four_suites'] or data['selection_split'] != 'search':
        raise SystemExit('Incomplete or non-search prior')
    frozen = RESULTS / 'search/frozen_candidates.json'
    if frozen.exists() and json.loads(frozen.read_text()).get('complete'):
        dump_json(status, {'pid': os.getpid(), 'state': 'already_finished'})
        return
    with (RESULTS / 'logs/search.log').open('a') as stream:
        child = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', '-m',
                                  'scripts.experiment10.run_search', '--launch'],
                                 cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        dump_json(status, {'pid': os.getpid(), 'state': 'running', 'child_pid': child.pid})
        code = child.wait()
    dump_json(status, {'pid': os.getpid(), 'state': 'finished' if code == 0 else 'failed',
                       'child_pid': child.pid, 'returncode': code})
    raise SystemExit(code)


if __name__ == '__main__':
    main()

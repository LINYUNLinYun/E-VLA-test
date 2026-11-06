"""Queue shadow jobs as each numerical suite ends; CPU phase analysis after all probes.

The upstream process and child PIDs are inspected, not inferred live from a lock.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.experiment10.common import ROOT, RESULTS, SUITES, dump_json, claim_gpu, managed_gpus
from scripts.experiment10.run_pipeline import free_gpus, event


def is_live(pid):
    try:
        state = (Path('/proc') / str(pid) / 'stat').read_text().split(') ', 1)[1][0]
        return state not in ('Z', 'X')
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-pid', type=int, required=True)
    args = parser.parse_args()
    pending, active, next_poll, analyzed, failures = list(SUITES), {}, 0., False, []
    seen_ready = set()
    previous_reserved = set()
    leases = {}
    while pending or active or not analyzed:
        upstream_status = RESULTS / 'pipeline_status.json'
        upstream = json.loads(upstream_status.read_text()) if upstream_status.exists() else {}
        reserved = {int(r['gpu']) for r in upstream.get('active', []) if is_live(r['pid'])}
        if previous_reserved - reserved:
            next_poll = 0.  # known upstream completion; do not wait five minutes
        previous_reserved = reserved
        for gpu, (process, suite, stream) in list(active.items()):
            code = process.poll()
            if code is not None:
                stream.close()
                event(event='shadow_finished', gpu=gpu, suite=suite, code=code)
                if code:
                    failures.append({'suite': suite, 'code': code})
                del active[gpu]
                leases.pop(gpu).close()
                next_poll = 0.
        # Own known completed jobs are events, so permit an immediate idle query.
        ready = [s for s in pending if (RESULTS / 'probe' / s / 'summary.json').exists()]
        if set(ready) - seen_ready:
            next_poll = 0.
            seen_ready.update(ready)
        known_owned = managed_gpus() | reserved | set(active)
        if ready and len(known_owned) < 4 and time.monotonic() >= next_poll:
            free = [g for g in free_gpus() if g not in reserved and g not in active]
            next_poll = time.monotonic() + 300
            for gpu, suite in zip(free, ready):
                lease = claim_gpu(gpu)
                if lease is None:
                    continue
                leases[gpu] = lease
                pending.remove(suite)
                path = RESULTS / 'logs' / ('shadow_' + suite + '.log')
                stream = path.open('a')
                env = os.environ.copy()
                env.update({'CUDA_VISIBLE_DEVICES': str(gpu), 'JAX_PLATFORMS': 'cpu', 'PYTHONUNBUFFERED': '1',
                            'OPENPI_DATA_HOME': '/root/nfs/embodied/benchmark/shared/checkpoints/openpi',
                            'PYTHONPATH': str(ROOT / 'src') + ':' + str(ROOT)})
                process = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-m', 'scripts.experiment10.probe_schedules',
                                            '--suite', suite], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                active[gpu] = (process, suite, stream)
                event(event='shadow_started', gpu=gpu, suite=suite, pid=process.pid)
        if not analyzed and all((RESULTS / 'probe' / s / 'summary.json').exists() for s in SUITES):
            response = subprocess.run([str(ROOT / '.venv/bin/python'), '-m', 'scripts.experiment10.analyze_phase'], cwd=ROOT)
            analyzed = True
            if response.returncode:
                failures.append({'stage': 'phase_analysis', 'code': response.returncode})
        if not is_live(args.upstream_pid) and any(not (RESULTS / 'probe' / s / 'summary.json').exists() for s in pending):
            event(event='upstream_incomplete', pending=pending)
            raise SystemExit('Upstream exited without required probes; preserving completed work')
        dump_json(RESULTS / 'followup_status.json', {'pid': os.getpid(), 'pending': pending, 'analyzed': analyzed,
                     'active': [{'gpu': g, 'pid': p.pid, 'suite': s} for g, (p, s, _) in active.items()],
                     'failures': failures, 'complete_shadow_stages': not pending and not active and analyzed})
        if pending or active or not analyzed:
            time.sleep(2)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

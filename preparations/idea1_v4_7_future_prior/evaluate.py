"""Case-level paired future priors: manifest conditions on one persistent service."""
import argparse
import json
import random
import time
import traceback
import uuid

import numpy as np

from preparations.idea1_v2_1.runtime_client import rpc
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import lock
from preparations.idea1_v3_1.rollout import Environments
from preparations.idea1_v4_4_one_step.rollout import episode
from .common import OUT, directory, key


def initial_check(cfgs, case):
    """All conditions bootstrap from identical Gaussian D1, including decoded output."""
    rows, identities, arrays = [], [], []
    for cfg in cfgs:
        d = directory(cfg['id'], case)
        with (d/'infer.jsonl').open() as stream:
            rows.append(json.loads(stream.readline()))
        identities.append(json.loads((d/'identity.json').read_text()))
        with np.load(d/'arrays.npz') as saved:
            arrays.append({name: saved[name].copy() for name in ('raw_0', 'actions_0')})
    assert len({r['input_hash'] for r in rows}) == 1, 'Paired initial observation differs'
    assert len({r['model_noise_sha256'] for r in rows}) == 1, 'Paired initial noise differs'
    assert len({r['seed'] for r in rows}) == 1, 'Paired initial generation seed differs'
    assert len({r['init_vector_sha256'] for r in identities}) == 1, 'Paired official initial state differs'
    server_pids = sorted({r['protocol']['server_pid'] for r in identities})
    recovery_path = OUT/'case_recovery'/f'{key(case)}.json'
    recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else None
    if len(server_pids) != 1:
        assert recovery and recovery['reason'] == 'resource_oom' and recovery['attempt'] == 1
        assert recovery['previous_server_pid'] in server_pids and len(server_pids) == 2
        assert recovery['case'] == case and recovery['valid_results_preserved']
        assert (OUT/recovery['archive']).is_dir(), 'Technical recovery archive missing'
    assert all(r['route']['initial_state'] == 'gaussian' and r['route']['D'] == 1 for r in rows)
    checks = []
    for cfg, got in zip(cfgs, arrays):
        for name in ('raw_0', 'actions_0'):
            expected = arrays[0][name]
            passed = expected.shape == got[name].shape and np.allclose(expected, got[name], atol=1e-5, rtol=1e-5)
            checks.append(dict(condition=cfg['id'], array=name, passed=bool(passed),
                               max_abs=float(np.max(np.abs(expected-got[name])))))
    assert all(row['passed'] for row in checks), 'Initial Gaussian D1 raw/decoded output differs'
    return dict(passed=True, atol=1e-5, rtol=1e-5, checks=checks,
                initial_input_hash=rows[0]['input_hash'],
                initial_noise_sha256=rows[0]['model_noise_sha256'],
                server_pid=identities[0]['protocol']['server_pid'], server_pids=server_pids,
                technical_recovery=recovery)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--worker', required=True)
    args = parser.parse_args()
    manifest = json.loads((OUT/'manifest.json').read_text())
    server_pid = rpc(args.port, dict(op='health'))['protocol']['server_pid']
    for index, case in enumerate(manifest['cases']):
        done = OUT/'case_done'/f'{key(case)}.json'
        if done.exists():
            continue
        try:
            with lock(OUT/'claims'/f'{key(case)}.lock', blocking=False):
                if done.exists():
                    continue
                cfgs = manifest['conditions']
                # A different PID requires an archived, bounded resource recovery.
                prior_pids = set()
                for cfg in cfgs:
                    d = directory(cfg['id'], case)
                    if (d/'complete.json').exists():
                        prior_pids.add(json.loads((d/'identity.json').read_text())['protocol']['server_pid'])
                if prior_pids and prior_pids != {server_pid}:
                    recovery_path = OUT/'case_recovery'/f'{key(case)}.json'
                    # Allow the single controller to record a just-detected OOM.
                    for _ in range(15):
                        if recovery_path.exists():
                            break
                        time.sleep(1)
                    recovery = json.loads(recovery_path.read_text())
                    assert recovery['reason'] == 'resource_oom' and recovery['attempt'] == 1
                    assert prior_pids == {recovery['previous_server_pid']}
                    assert recovery['case'] == case and recovery['valid_results_preserved']
                    assert (OUT/recovery['archive']).is_dir()
                offset = index % len(cfgs)
                for cfg in cfgs[offset:]+cfgs[:offset]:
                    d = directory(cfg['id'], case)
                    if (d/'complete.json').exists():
                        continue
                    previous_error = d/'technical_error.json'
                    consumed = json.loads(previous_error.read_text())['attempt'] if previous_error.exists() else 0
                    assert consumed < 2, 'Condition exhausted its single technical retry'
                    for attempt in range(consumed, 2):
                        if d.exists():
                            archive = OUT/'attempts'/f'{key(case)}_{cfg["id"]}_{uuid.uuid4().hex}'
                            archive.parent.mkdir(parents=True, exist_ok=True)
                            d.rename(archive)
                        random.seed(7)
                        np.random.seed(7)
                        pool = Environments(reuse=False)
                        atomic(OUT/'workers'/args.worker/'evaluation_state.json',
                               dict(case=case, condition=cfg['id'], attempt=attempt+1, time=time.time()))
                        try:
                            result = episode(d, args.port, case, cfg, pool, manifest['manifest_hash'])
                            print(json.dumps(dict(case=case, condition=cfg['id'],
                                                  success=result['success'], actions=result['actions'])), flush=True)
                            break
                        except Exception:
                            error = traceback.format_exc()
                            atomic(d/'technical_error.json', dict(attempt=attempt+1, error=error))
                            # A resource OOM needs a new service, not another call
                            # on the same CUDA state; the saved count follows it.
                            if attempt or 'out of memory' in error.lower():
                                raise
                        finally:
                            pool.close()
                checked = initial_check(cfgs, case)
                atomic(done, dict(checked, case=case, worker=args.worker, time=time.time()))
        except BlockingIOError:
            continue


if __name__ == '__main__':
    main()

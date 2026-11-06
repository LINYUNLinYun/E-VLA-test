"""Four conditions alternate within each same-task, same-GPU case group."""
import argparse
import json
from pathlib import Path
import time
import uuid

from .common import (OUT, begin_attempt, ck, claim_case, directory, finish_case,
                     read_manifest, record_technical_failure, register, source)
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_E2_module_skip.evaluate import identity, fatal_error


def ordered_conditions(manifest, case):
    offset = manifest['cases'].index(case) % len(manifest['conditions'])
    conditions = manifest['conditions']
    return conditions[offset:] + conditions[:offset]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--actor', type=int, required=True)
    parser.add_argument('--group', required=True)
    parser.add_argument('--phase', default='main40', choices=['main40'])
    args = parser.parse_args()
    from . import rollout
    from preparations.idea1_v2_1.runtime_client import rpc
    from preparations.idea1_v3_4_E2_max2_repair.evaluate import Fresh
    client = f'gpu{args.gpu}-actor{args.actor}'
    rollout.rpc = lambda port, request, *positional, **keywords: rpc(port, dict(request, client=client), *positional, **keywords)
    manifest = read_manifest()
    pool = Fresh()
    state = OUT / 'workers' / f'gpu{args.gpu}' / f'actor{args.actor}.json'
    try:
        while True:
            case = claim_case(args.group, client)
            if case is None:
                break
            expected = identity(manifest['controls'][ck(case)]['directory'])
            records = []
            for cfg in ordered_conditions(manifest, case):
                path, _ = source(manifest, cfg, case)
                if path is None:
                    path = directory(cfg['id'], case)
                    while True:
                        attempt = begin_attempt(cfg, case)
                        if path.exists():
                            path.rename(path.with_name(path.name + '.invalid.' + uuid.uuid4().hex))
                        atomic(state, dict(status='evaluating', group=args.group, case=case,
                               condition=cfg['id'], attempt=attempt, time=time.time()))
                        try:
                            result = rollout.episode(path, args.port, case, dict(cfg, episode_case=case), pool, manifest['manifest_hash'])
                            break
                        except Exception as error:
                            path.mkdir(parents=True, exist_ok=True)
                            fatal = fatal_error(error)
                            atomic(path / 'technical_error.json', dict(error=repr(error), attempt=attempt, fatal=fatal, time=time.time()))
                            recover = record_technical_failure(cfg, case, error, fatal=fatal)
                            if not recover or 'OutOfMemoryError' in str(error) or 'single_actor_required' in str(error):
                                raise
                else:
                    result = json.loads((path / 'result.json').read_text())
                actual = identity(path)
                if any(actual[field] != expected[field] for field in expected if field != 'first_input_hash'):
                    error = ValueError('official case/control identity mismatch')
                    atomic(path / 'pairing_error.json', dict(expected=expected, actual=actual, fatal=True))
                    record_technical_failure(cfg, case, error, fatal=True)
                    raise error
                register(cfg, case, path, result)
                records.append(dict(condition=cfg['id'], directory=str(path), initial_hash=actual['first_input_hash'],
                                    control_initial_hash=expected['first_input_hash'], initial_matches_control=actual['first_input_hash'] == expected['first_input_hash']))
                print(json.dumps(dict(condition=cfg['id'], case=case, success=result['success'], actions=result['actions'])), flush=True)
            atomic(OUT / 'case_done' / f'{ck(case)}.json', dict(case=case, records=records, gpu=args.gpu, client=client, time=time.time()))
            finish_case(args.group, case)
    finally:
        pool.close()
    atomic(state, dict(status='group_complete', group=args.group, time=time.time()))


if __name__ == '__main__':
    main()

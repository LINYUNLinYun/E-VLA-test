"""Same-task closed-loop evaluator, with a complete pilot barrier."""
import argparse
import json
import time
import uuid
from pathlib import Path

from .common import (OUT, _read_frozen, begin_attempt, ck, claim_case,
                     directory, finish_case, record_technical_failure, register, source)
from preparations.idea1_v3.io import atomic


def identity(path):
    path = Path(path)
    row = json.loads((path / 'identity.json').read_text())
    with (path / 'infer.jsonl').open() as stream:
        row['first_input_hash'] = json.loads(stream.readline())['input_hash']
    return {key: row[key] for key in ('init_vector_sha256', 'bddl_sha256', 'controller',
                                     'prompt', 'env_seed', 'settling', 'first_input_hash')}


def ordered_conditions(manifest, case, phase):
    if phase == 'pilot':
        if case not in manifest['pilot_cases']:
            raise ValueError('pilot attempted a case outside the frozen subset')
        return [cfg for cfg in manifest['conditions'] if cfg['condition'] == 'F-R+']
    if phase != 'small40' or not (OUT / 'pilot_mechanism_complete.json').exists():
        raise ValueError('all pilot episodes and offline mechanism must finish before the main matrix')
    configs = manifest['conditions']
    offset = manifest['cases'].index(case) % len(configs)
    return configs[offset:] + configs[:offset]


def runtime_config(manifest, cfg, case, phase):
    result = dict(cfg, episode_case=case,
                  pilot=phase == 'pilot' and cfg['condition'] == 'F-R+')
    if cfg['scoring'] == 'fixed':
        result['fixed_skip'] = manifest['fixed_episode_masks'][cfg['id']][ck(case)]
    return result


def fatal_error(error):
    """Scientific identity/routing failures stop; infrastructure gets one retry."""
    if isinstance(error, (ValueError, AssertionError, KeyError, TypeError)):
        return True
    message = str(error).lower()
    markers = ('mismatch', 'invalid model', 'history', 'non-increasing', 'assertionerror',
               'valueerror', 'keyerror', 'typeerror', 'unacknowledged', 'different frozen',
               'must share', 'retry budget exhausted')
    return any(marker in message for marker in markers)


def run_condition(manifest, cfg, case, phase, port, pool, state, gpu, client):
    path, reused = source(manifest, cfg, case)
    if path is not None:
        return path, reused, json.loads((path / 'result.json').read_text())
    path = directory(cfg['id'], case)
    while True:
        attempt = begin_attempt(cfg, case)
        if path.exists():
            path.rename(path.with_name(path.name + '.invalid.' + uuid.uuid4().hex))
        atomic(state, dict(status='evaluating', phase=phase, case=case,
                           condition=cfg['id'], attempt=attempt, gpu=gpu, client=client, time=time.time()))
        try:
            from .rollout import episode
            result = episode(path, port, case, runtime_config(manifest, cfg, case, phase),
                             pool, manifest['manifest_hash'])
            print(json.dumps(dict(phase=phase, condition=cfg['id'], case=case,
                                  success=result['success'], actions=result['actions'],
                                  wall_s=result['wall_s'])), flush=True)
            return path, False, result
        except Exception as error:
            path.mkdir(parents=True, exist_ok=True)
            fatal = fatal_error(error)
            atomic(path / 'technical_error.json', dict(error=repr(error), attempt=attempt,
                                                       fatal=fatal, time=time.time()))
            recover = record_technical_failure(cfg, case, error, fatal=fatal)
            # Resource failures require a fresh service and one-client preflight.
            if not recover or 'OutOfMemoryError' in str(error) or 'single_actor_required' in str(error):
                raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--actor', type=int, required=True)
    parser.add_argument('--group', required=True)
    parser.add_argument('--phase', choices=['pilot', 'small40'], required=True)
    args = parser.parse_args()
    from . import rollout
    from preparations.idea1_v2_1.runtime_client import rpc
    from preparations.idea1_v3_4_E2_max2_repair.evaluate import Fresh
    client = f'gpu{args.gpu}-actor{args.actor}'

    def isolated(port, request, *positional, **keywords):
        return rpc(port, dict(request, client=client), *positional, **keywords)

    rollout.rpc = isolated
    manifest = _read_frozen(OUT / 'manifest.json')
    pool = Fresh()
    state = OUT / 'workers' / f'gpu{args.gpu}' / f'actor{args.actor}.json'
    try:
        while True:
            case = claim_case(args.group, client)
            if case is None:
                break
            expected = identity(manifest['baseline']['B4_B-R+'][ck(case)]['directory'])
            records = []
            for cfg in ordered_conditions(manifest, case, args.phase):
                path, reused, result = run_condition(manifest, cfg, case, args.phase,
                                                     args.port, pool, state, args.gpu, client)
                actual = identity(path)
                fields = [key for key in expected if key != 'first_input_hash']
                if any(actual[key] != expected[key] for key in fields):
                    error = ValueError(f'official initial-state/protocol mismatch: {cfg["id"]} {ck(case)}')
                    atomic(path / 'pairing_error.json', dict(expected=expected, actual=actual, fatal=True))
                    if not reused:
                        record_technical_failure(cfg, case, error, fatal=True)
                    raise error
                register(cfg, case, path, result, reused=reused)
                row = dict(condition=cfg['id'], directory=str(path), reused=reused,
                           initial_hash=actual['first_input_hash'], baseline_initial_hash=expected['first_input_hash'],
                           initial_matches_baseline=actual['first_input_hash'] == expected['first_input_hash'])
                if args.phase == 'pilot':
                    row['saved_input_count'] = len(list(path.glob('input_*.npz')))
                    if row['saved_input_count'] > 15:
                        raise ValueError('pilot observation bank exceeded fifteen requests per episode')
                records.append(row)
            atomic(OUT / 'case_done' / args.phase / f'{ck(case)}.json',
                   dict(case=case, records=records, gpu=args.gpu, client=client, time=time.time()))
            finish_case(args.group, case)
    finally:
        pool.close()
    atomic(state, dict(status='group_complete', group=args.group, phase=args.phase, time=time.time()))


if __name__ == '__main__':
    main()

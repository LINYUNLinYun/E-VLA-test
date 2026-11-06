"""One isolated OS process per simulator; no shared legacy module globals."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from common import Ledger, atomic_json, bootstrap_environment, planned, sha, start_ticks
from throughput_revision import acceptance, read, verify_revision


def predict_gate(revision, session, model, port, slot, mode):
    from runtime_client import rpc, check_prediction
    from runtime_data import load_observation, audited_protocol
    from runtime_inference import save_prediction
    identity, _ = verify_revision(revision)
    protocol = audited_protocol(identity['source_attempt'], model, rpc(port, dict(op='health'))['protocol'])
    requests = read(session/'requests.json')
    indexes = [0, 2, 1, 3] if mode == 'reference' else range(slot, 4, 2)
    if mode != 'reference':
        atomic_json(session/f'client{slot}_identity.json', dict(pid=os.getpid(), start_ticks=start_ticks(os.getpid()), slot=slot))
    for index in indexes:
        if (session/'STOP').exists():
            raise RuntimeError('Gate stopped before next prediction')
        if mode != 'reference':
            deadline = time.monotonic()+900
            while read(session/'turn.json')['next'] != index:
                if (session/'STOP').exists() or time.monotonic() > deadline:
                    raise RuntimeError('Interleave turn stopped/timed out')
                time.sleep(.02)
        request = requests[index]
        obs, instruction, _ = load_observation(request['point'])
        condition = request['condition']
        # reset is deliberately exercised between clients; frozen reset is a no-op.
        rpc(port, dict(op='reset'))
        result = rpc(port, dict(op='predict', obs=obs, instruction=instruction,
            seed=request['seed'], skip=condition['skip'], schedule=condition['schedule'], category='throughput_equivalence'))
        row = save_prediction(session/mode/f'{index}.json', result,
            condition=condition['id'], point=0, seed=request['seed'])
        check_prediction(row, condition, protocol, request['seed'])
        if mode != 'reference':
            atomic_json(session/'turn.json', dict(next=index+1))
    if mode != 'reference':
        atomic_json(session/f'client{slot}_summary.json', dict(slot=slot, predictions=2, valid=0))


def episode_metrics(directory, elapsed):
    rows = [json.loads(line) for line in (directory/'infer.jsonl').read_text().splitlines()]
    timing = {}
    for row in rows:
        for key, value in row.get('timing', {}).items():
            if key.endswith('_ms') and isinstance(value, (int, float)):
                timing[key] = timing.get(key, 0.) + value
    return dict(episode_wall_seconds=elapsed, predictions=len(rows), summed_prediction_timing_ms=timing)


def pilot_authorization(revision, session, model, slot, token, row, directory, validated_hash):
    marker = read(revision/f'{model}.pilot_session.json')
    ready = read(revision/f'{model}.pilot_ready.json')
    if marker['session'] != str(session) or ready['session'] != str(session):
        raise ValueError('Pilot authorization refers to another session')
    if (len(ready['results']) != 2 or len({result['key'] for result in ready['results']}) != 2 or
            len({result['token'] for result in ready['results']}) != 2):
        raise ValueError('Pilot requires two distinct fresh episode candidates')
    matches = [result for result in ready['results'] if result['slot'] == slot]
    if len(matches) != 1:
        raise ValueError('Pilot authorization does not identify this environment')
    result = matches[0]
    if (result['token'] != token or result['key'] != row['key'] or
            result['candidate_directory'] != str(directory) or result['validated_sha256'] != validated_hash):
        raise ValueError('Pilot authorization does not identify this candidate')
    return acceptance(revision, model, 'pilot', revision/f'{model}.pilot_ready.json')


def episodes(revision, session, model, port, slot, mode):
    from runtime_client import run_episode
    identity, manifest = verify_revision(revision)
    attempt = Path(identity['source_attempt'])
    ledger = Ledger(attempt)
    condition_map = {c['id']: c for c in manifest['conditions']}
    completed = []
    started = time.time()
    for row in planned(manifest):
        if row['model'] != model:
            continue
        if (attempt/'PAUSE').exists() or (session/'STOP').exists():
            break
        token = ledger.reserve(row, require_artifacts=True)
        if token is None:
            continue
        directory = attempt/'episodes'/row['condition']/row['suite']/f"task{row['task']:02d}"/f"init{row['init']:02d}"/token
        # A pilot's unapproved validated.json must never trigger old Ledger recovery.
        ledger_directory = session/'escrow'/token if mode == 'pilot' else directory
        ledger.update(token, directory=str(ledger_directory),
            execution_revision=str(revision), execution_revision_sha256=sha(revision/'revision_manifest.json'),
            execution_mode='two_process_envs_sequential_server', throughput_session=str(session), env_slot=slot,
            pilot_candidate_directory=str(directory) if mode == 'pilot' else None)
        try:
            episode_start = time.time()
            result = run_episode(directory, port, condition_map[row['condition']], row)
            metrics = episode_metrics(directory, time.time()-episode_start)
            validated_hash = sha(directory/'validated.json')
            if mode == 'pilot':
                atomic_json(session/f'pilot{slot}.json', dict(slot=slot, token=token, key=row['key'],
                    candidate_directory=str(directory), escrow_directory=str(ledger_directory),
                    validated_sha256=validated_hash, success=result['success'], metrics=metrics))
                deadline = time.monotonic()+1800
                while True:
                    if (session/'STOP').exists() or time.monotonic() > deadline:
                        raise RuntimeError('Pilot stopped/timed out before medium acceptance; candidate retained')
                    if (revision/f'{model}.pilot_ready.json').exists() and (revision/'acceptance'/f'{model}.pilot.json').exists():
                        pilot_authorization(revision, session, model, slot, token, row, directory, validated_hash)
                        break
                    time.sleep(.5)
            if sha(directory/'validated.json') != validated_hash:
                raise ValueError('Pilot validation changed before commit')
            ledger.update(token, directory=str(directory), status='valid', success=result['success'],
                validated_sha256=validated_hash, throughput_metrics=metrics,
                throughput_acceptance_sha256=pilot_authorization(revision, session, model, slot, token, row, directory, validated_hash) if mode == 'pilot' else acceptance(revision, model, 'pilot', revision/f'{model}.pilot_ready.json'))
            completed.append(dict(key=row['key'], token=token, **metrics))
        except BaseException:
            ledger.update(token, status='invalid', error=traceback.format_exc())
            atomic_json(session/'STOP', dict(reason='client episode failed', slot=slot, token=token, timestamp=time.time()))
            raise
        if mode == 'pilot':
            break
    if mode == 'pilot' and len(completed) != 1:
        raise RuntimeError('Pilot requires one fresh canonical key per environment')
    atomic_json(session/f'client{slot}_summary.json', dict(slot=slot, valid=len(completed),
        elapsed_seconds=time.time()-started, episodes=completed))


def main():
    bootstrap_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['reference', 'interleaved', 'pilot', 'run'])
    parser.add_argument('--revision', type=Path, required=True)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--slot', type=int, choices=[0, 1], required=True)
    args = parser.parse_args()
    if args.mode in ('reference', 'interleaved'):
        predict_gate(args.revision, args.session, args.model, args.port, args.slot, args.mode)
    else:
        episodes(args.revision, args.session, args.model, args.port, args.slot, args.mode)


if __name__ == '__main__':
    main()

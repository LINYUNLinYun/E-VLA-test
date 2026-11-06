"""Isolated original LIBERO client with new cache-revision provenance and escrow."""
import argparse
from pathlib import Path
import time
import traceback

from common import Ledger, atomic_json, bootstrap_environment, planned, sha
from pi0_prompt_cache_revision import acceptance, read, verify_revision


def run(revision, session, port, slot, phase):
    from runtime_client import run_episode
    from throughput_revision_client import episode_metrics
    identity, manifest = verify_revision(revision)
    attempt = Path(identity['source_attempt'])
    ledger = Ledger(attempt)
    conditions = {item['id']: item for item in manifest['conditions']}
    completed = []
    for row in planned(manifest):
        if row['model'] != 'pi0':
            continue
        if (session/'STOP').exists() or (attempt/'PAUSE').exists():
            break
        token = ledger.reserve(row, require_artifacts=True)
        if token is None:
            continue
        directory = attempt/'episodes'/row['condition']/row['suite']/f"task{row['task']:02d}"/f"init{row['init']:02d}"/token
        escrow = session/'escrow'/token
        ledger.update(token, directory=str(escrow if phase == 'pilot' else directory),
            execution_revision=str(revision), execution_revision_sha256=sha(revision/'revision_manifest.json'),
            execution_mode='pi0_two_env_prompt_length_cache2', throughput_session=str(session), env_slot=slot,
            pilot_candidate_directory=str(directory) if phase == 'pilot' else None)
        try:
            start = time.time()
            result = run_episode(directory, port, conditions[row['condition']], row)
            metrics = episode_metrics(directory, time.time()-start)
            validated_hash = sha(directory/'validated.json')
            if phase == 'pilot':
                candidate = dict(slot=slot, token=token, key=row['key'], candidate_directory=str(directory),
                    escrow_directory=str(escrow), validated_sha256=validated_hash, success=result['success'], metrics=metrics)
                atomic_json(session/f'pilot{slot}.json', candidate)
                deadline = time.monotonic()+1800
                while True:
                    ready_path = revision/'pi0.pilot_ready.json'
                    if ready_path.exists() and (revision/'acceptance/pi0.pilot.json').exists():
                        ready = read(ready_path)
                        if ready['session'] != str(session) or candidate not in ready['results']:
                            raise ValueError('Pilot signature does not identify this exact candidate/session')
                        acceptance(revision, 'pilot', ready_path)
                        break
                    if (session/'STOP').exists() or time.monotonic() > deadline:
                        raise RuntimeError('Unapproved pilot stopped; candidate remains preserved')
                    time.sleep(.5)
            signed = acceptance(revision, 'pilot', revision/'pi0.pilot_ready.json')
            if sha(directory/'validated.json') != validated_hash:
                raise ValueError('Validation changed before atomic commit')
            ledger.update(token, status='valid', directory=str(directory), success=result['success'],
                validated_sha256=validated_hash, throughput_metrics=metrics, throughput_acceptance_sha256=signed)
            completed.append(dict(key=row['key'], token=token, **metrics))
        except BaseException:
            ledger.update(token, status='invalid', error=traceback.format_exc())
            atomic_json(session/'STOP', dict(reason='cache client failed', slot=slot, timestamp=time.time()))
            raise
        if phase == 'pilot':
            break
    if phase == 'pilot' and len(completed) != 1:
        raise RuntimeError('Two fresh pilot keys are required')
    atomic_json(session/f'client{slot}_summary.json', dict(slot=slot, valid=len(completed), episodes=completed))


def main():
    bootstrap_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', type=Path, required=True)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--slot', type=int, choices=(0, 1), required=True)
    parser.add_argument('--phase', choices=('pilot', 'run'), required=True)
    args = parser.parse_args()
    run(args.revision, args.session, args.port, args.slot, args.phase)


if __name__ == '__main__':
    main()

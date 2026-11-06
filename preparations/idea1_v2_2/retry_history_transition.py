"""Explicitly authorized native Ledger transition wrapper; plan is read-only default."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
from contextlib import nullcontext

from common import Ledger, digest, exclusive, lock, planned, sha, start_ticks
from independent_media_audit import unique_write
from retry_history_audit import chain_for, manifest_identity

EXPECTED_ERROR = 'Invalid episode requires explicit diagnosis and retry authorization in ledger; automatic retry is forbidden'


def read(path):
    return json.loads(Path(path).read_text())


def register_receipt(attempt,receipt_path,phase):
    """Seal this completed phase only; never launches the next recovery step."""
    index_path=attempt/'retry_recovery_index.json'
    with lock(attempt/'retry_recovery_index.lock'):
        previous=read(index_path) if index_path.exists() else dict(schema=1,kind='explicit_retry_recovery_index',entries=[])
        if previous.get('schema')!=1 or previous.get('kind')!='explicit_retry_recovery_index':raise ValueError('Unknown recovery index')
        identities=set()
        for reference in previous['entries']:
            if sha(reference['path'])!=reference['sha256']:raise ValueError('Old recovery receipt bytes changed')
            value=read(reference['path']);identity=(value['key'],value['old_token'],reference['phase'])
            kind={'transition_only':'technical_retry_transition','authorize_retry':'technical_retry_authorization'}.get(reference['phase'])
            if identity in identities or kind is None or value.get('kind')!=kind or value.get('passed') is not True:
                raise ValueError('Duplicate/unknown/failed registered recovery phase')
            identities.add(identity)
        value=read(receipt_path);identity=(value['key'],value['old_token'],phase)
        expected_kind={'transition_only':'technical_retry_transition','authorize_retry':'technical_retry_authorization'}.get(phase)
        if expected_kind is None or value.get('kind')!=expected_kind or value.get('passed') is not True or value.get('source_sha256')!=sha(__file__):
            raise ValueError('New registration is not this source successful exact phase')
        if identity in identities:raise ValueError('Recovery phase already registered; never overwrite')
        def seal_index_bytes(raw):
            import hashlib
            path=attempt/'retry_recovery/index_history'/f'{hashlib.sha256(raw).hexdigest()}.json'
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                if path.read_bytes()!=raw:raise ValueError('Immutable recovery index snapshot changed')
            else:
                with path.open('xb') as stream:stream.write(raw)
        if index_path.exists():seal_index_bytes(index_path.read_bytes())
        updated=dict(previous,entries=previous['entries']+[dict(phase=phase,path=str(receipt_path),sha256=sha(receipt_path))])
        from common import atomic_json
        atomic_json(index_path,updated)
        seal_index_bytes(index_path.read_bytes())


def plan(attempt, key, token, phase='transition_only', _locked=False):
    attempt = Path(attempt).resolve()
    manifest = manifest_identity(attempt/'condition_manifest.json')
    matches = [row for row in planned(manifest['data']) if row['key'] == key]
    if len(matches) != 1 or int(key[:8],16)%2 != 0:
        raise ValueError('Target is not exactly one original shard0 canonical key')
    canonical = matches[0]
    with (lock(attempt/'episode_ledger.lock') if _locked else nullcontext()):
        raw = (attempt/'episode_ledger.json').read_bytes()
        chain = chain_for(json.loads(raw), key)
        if len(chain) != 1 or chain[0]['token'] != token or any(chain[0].get(field) != value for field,value in canonical.items()):
            raise ValueError('Target attempt chain changed or is not its exact canonical row')
        row = chain[0]
        if any(field in row for field in ('retry_diagnosis','retry_authorized_at','directory','validated_sha256')):
            raise ValueError('Target has prior authorization/artifact identity; refuse state-only transition')
        if row['status'] != ('reserved' if phase == 'transition_only' else 'invalid'):
            raise ValueError('Target is not in the exact requested pre-transition status')
        current = start_ticks(row['pid'])
        expected = attempt/'episodes'/row['condition']/row['suite']/f"task{row['task']:02d}"/f"init{row['init']:02d}"/token
        preflight = dict(owner_start_ticks=current, expected_directory=str(expected), expected_directory_present=expected.exists(),
            validated_present=Path('/nonexistent/validated.json').exists() or Path('/nonexistent/validated.json').is_symlink())
        if (expected.is_symlink() or any(path.is_symlink() for path in expected.parents)
                or current == row['start_ticks'] or preflight['expected_directory_present'] or preflight['validated_present']):
            raise ValueError('Owner live or possible episode artifact exists; refuse recovery')
    return dict(schema=1, attempt=str(attempt), phase=phase, key=key, old_token=token, canonical=canonical,
        canonical_row_sha256=digest(canonical), manifest_file_sha256=manifest['file_sha256'], chain=chain,
        chain_sha256=digest(chain), preflight=preflight, source_sha256=sha(__file__),
        ledger_snapshot_sha256=__import__('hashlib').sha256(raw).hexdigest(), cause='unknown; only absence of episode entry/artifacts is established'), raw


def verify_approval(path, current, phase):
    approval = read(path)
    source = approval['source_acceptance']
    if sha(source['path']) != source['sha256']:
        raise ValueError('Independent recovery source signature changed')
    accepted = read(source['path'])
    if (accepted.get('schema') != 1 or accepted.get('accepted') is not True or accepted.get('reviewer_model') != 'gpt-6.1-sol'
            or accepted.get('reasoning_effort') != 'medium' or accepted['source_hashes'].get(Path(__file__).name) != sha(__file__)
            or approval.get('schema') != 1 or approval.get('accepted') is not True or approval.get('authorized_by') != 'root'
            or approval.get('target_single_owner') is not True or approval.get('phase') != phase
            or approval.get('attempt') != current['attempt']
            or approval.get('key') != current['key'] or approval.get('old_token') != current['old_token']
            or approval.get('expected_chain_sha256') != current['chain_sha256']
            or approval.get('manifest_file_sha256') != current['manifest_file_sha256']
            or approval.get('source_sha256') != sha(__file__)):
        raise ValueError('Missing exact root phase/key/token/source/chain recovery authority')
    return approval


def execute(attempt, key, token, approval_path, evidence, phase='transition_only', transition_receipt=None):
    attempt, evidence = Path(attempt).resolve(), Path(evidence).resolve()
    initial, _ = plan(attempt,key,token,phase)
    with exclusive(attempt/'leases'/f"{initial['canonical']['model']}.shard0.lock"), exclusive(attempt/'leases'/f'retry_{key}.lock'):
        current, before = plan(attempt,key,token,phase,_locked=True)
        approval = verify_approval(approval_path,current,phase)
        from runtime_data import verify_frozen
        verify_frozen(attempt)
        if evidence.parent != attempt/'retry_recovery':
            raise ValueError('Recovery evidence must be a new direct child of attempt/retry_recovery')
        evidence.mkdir(parents=True, exist_ok=False)
        def save_snapshot(name, raw):
            path=evidence/name
            with path.open('xb') as stream:stream.write(raw)
            return dict(path=str(path),sha256=sha(path))
        before_reference=save_snapshot('before_ledger.json',before)
        unique_write(evidence/'plan.json',current)
        record=dict(schema=1,kind='technical_retry_transition' if phase=='transition_only' else 'technical_retry_authorization',
            attempt=str(attempt),
            key=key,old_token=token,manifest_file_sha256=current['manifest_file_sha256'],canonical_row_sha256=current['canonical_row_sha256'],
            source=str(Path(__file__).resolve()),source_sha256=sha(__file__),approval=dict(path=str(Path(approval_path).resolve()),sha256=sha(approval_path)),
            before_snapshot=before_reference,preflight=current['preflight'],cause=current['cause'],started_epoch=time.time())
        try:
            if phase=='transition_only':
                try:
                    returned=Ledger(attempt).reserve(current['canonical'],require_artifacts=True)
                except RuntimeError as error:
                    if str(error)!=EXPECTED_ERROR:raise
                    record['native_exception']=str(error)
                else:
                    raise RuntimeError(f'Native reserve unexpectedly returned {returned!r}; do not rollback or continue')
                expected=[dict(current['chain'][0],status='invalid',error='Owner ended before atomic validation')]
            else:
                if not approval.get('diagnosis','').strip() or not transition_receipt:
                    raise ValueError('Retry authorization needs explicit diagnosis and prior transition receipt')
                transition=read(transition_receipt)
                if (transition.get('kind')!='technical_retry_transition' or transition.get('passed') is not True
                        or transition['key']!=key or transition['old_token']!=token
                        or chain_for(read(transition['after_snapshot']['path']),key)!=current['chain']
                        or sha(transition['after_snapshot']['path'])!=transition['after_snapshot']['sha256']):
                    raise ValueError('Retry authorization does not follow its exact state-only transition')
                command=[sys.executable,str(Path(__file__).with_name('runtime_runner.py')),'retry','--attempt',str(attempt),
                    '--key',key,'--diagnosis',approval['diagnosis'],'--launch']
                result=subprocess.run(command,capture_output=True,timeout=60)
                for name,data in [('stdout',result.stdout),('stderr',result.stderr)]:
                    with (evidence/name).open('xb') as stream:stream.write(data)
                record['original_cli']=dict(command=command,exit_code=result.returncode,
                    script_sha256=sha(Path(__file__).with_name('runtime_runner.py')),
                    python=dict(path=str(Path(sys.executable).resolve()),sha256=sha(sys.executable)),
                    stdout=dict(path=str(evidence/'stdout'),sha256=sha(evidence/'stdout')),
                    stderr=dict(path=str(evidence/'stderr'),sha256=sha(evidence/'stderr')))
                if result.returncode!=0:raise RuntimeError('Original retry CLI failed; preserve state and stop')
                record['transition_receipt']=dict(path=str(Path(transition_receipt).resolve()),sha256=sha(transition_receipt))
                record['before_snapshot']=transition['before_snapshot']
                record['preflight']=transition['preflight']
                record['diagnosis']=approval['diagnosis']
                expected=None
            with lock(attempt/'episode_ledger.lock'):
                after=(attempt/'episode_ledger.json').read_bytes()
                actual=chain_for(json.loads(after),key)
            record['after_snapshot']=save_snapshot('after_ledger.json',after)
            if phase=='transition_only':
                if actual!=expected:raise ValueError('Native transition changed target token/count/fields unexpectedly')
            else:
                if len(actual)!=1:raise ValueError('Original retry authorization appended/changed target attempts')
                moment=actual[0].get('retry_authorized_at')
                expected=[dict(current['chain'][0],retry_diagnosis=approval['diagnosis'],retry_authorized_at=moment)]
                if actual!=expected or not isinstance(moment,(int,float)) or moment<=0:
                    raise ValueError('Original retry CLI target delta differs from authorization fields only')
                record['retry_authorized_at']=moment
            record.update(passed=True,ended_epoch=time.time(),other_keys_may_have_legitimate_concurrent_commits=True)
            unique_write(evidence/'receipt.json',record)
            register_receipt(attempt,evidence/'receipt.json',phase)
        except BaseException as error:
            if isinstance(error,subprocess.TimeoutExpired):
                outputs={}
                for name,data in [('stdout',error.stdout or b''),('stderr',error.stderr or b'')]:
                    path=evidence/name
                    if not path.exists():
                        with path.open('xb') as stream:stream.write(data)
                    outputs[name]=dict(path=str(path),sha256=sha(path))
                record['original_cli']=dict(command=command,exit_code='timeout',**outputs)
            try:
                with lock(attempt/'episode_ledger.lock'):
                    failed_bytes=(attempt/'episode_ledger.json').read_bytes()
                record['after_failure_snapshot']=save_snapshot('after_failure_ledger.json',failed_bytes)
                record['after_failure_target_chain']=chain_for(json.loads(failed_bytes),key)
            except BaseException as evidence_error:
                record['after_failure_snapshot_error']=repr(evidence_error)
            record.update(passed=False,error=repr(error),ended_epoch=time.time())
            unique_write(evidence/'failure.json',record)
            raise
    return evidence/'receipt.json'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt',required=True);parser.add_argument('--key',required=True);parser.add_argument('--token',required=True)
    parser.add_argument('--phase',choices=('transition_only','authorize_retry'),default='transition_only')
    parser.add_argument('--execute',action='store_true');parser.add_argument('--approval');parser.add_argument('--evidence');parser.add_argument('--transition-receipt')
    args=parser.parse_args()
    if not args.execute:print(json.dumps(plan(args.attempt,args.key,args.token,args.phase)[0],indent=2))
    else:
        if not args.approval or not args.evidence:parser.error('execute requires exact approval and new evidence directory')
        print(execute(args.attempt,args.key,args.token,args.approval,args.evidence,args.phase,args.transition_receipt))


if __name__=='__main__':main()

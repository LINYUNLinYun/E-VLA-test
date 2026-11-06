"""History-aware controller over unchanged, signed independent media functions."""
import argparse
import hashlib
import os
from pathlib import Path
import time
import uuid

import independent_media_audit as media
import retry_history_audit as history
from common import exclusive, sha


def prepare(attempt, evidence, base_evidence):
    attempt,evidence=media.evidence_path(attempt,evidence)
    _,base=media.evidence_path(attempt,base_evidence)
    media.verify_source(attempt,base)
    if evidence==base:raise ValueError('History controller must have a new evidence directory')
    evidence.mkdir(parents=True,exist_ok=False)
    identity=dict(schema=1,kind='history_aware_media_controller',attempt=str(attempt),
        manifest_sha256=sha(attempt/'condition_manifest.json'),source=str(Path(__file__).resolve()),source_sha256=sha(__file__),
        resolver_sha256=sha(history.__file__),base_helper_sha256=sha(media.__file__),base_evidence=str(base),
        base_manifest_sha256=sha(base/'source_manifest.json'),base_acceptance_sha256=sha(base/'source_acceptance.json'),
        execution='calls actual unchanged base binding/audit_one, not original CLI/snapshot execution',
        workers=1,max_items=50,nice=19,threads=1)
    media.unique_write(evidence/'source_manifest.json',identity)
    return identity


def verify_source(attempt,evidence):
    attempt,evidence=media.evidence_path(attempt,evidence)
    identity=media.read(evidence/'source_manifest.json')
    accepted=media.read(evidence/'source_acceptance.json')
    base=Path(identity['base_evidence']).resolve()
    if (identity.get('kind')!='history_aware_media_controller' or identity.get('schema')!=1 or identity['attempt']!=str(attempt)
            or identity['source']!=str(Path(__file__).resolve()) or identity['source_sha256']!=sha(__file__)
            or identity['resolver_sha256']!=sha(history.__file__) or identity['base_helper_sha256']!=sha(media.__file__)
            or identity['manifest_sha256']!=sha(attempt/'condition_manifest.json')
            or identity['base_manifest_sha256']!=sha(base/'source_manifest.json')
            or identity['base_acceptance_sha256']!=sha(base/'source_acceptance.json')
            or accepted.get('schema')!=1 or accepted.get('accepted') is not True
            or accepted.get('reviewer_model')!='gpt-6.1-sol' or accepted.get('reasoning_effort')!='medium'
            or accepted.get('source_manifest_sha256')!=sha(evidence/'source_manifest.json')):
        raise ValueError('History media controller/base/source acceptance changed')
    media.evidence_path(attempt,base)
    original=media.verify_source(attempt,base)
    return identity,base,original


def audit(attempt,evidence,max_items=50,keys=None,final=False):
    if not isinstance(max_items,int) or not 1<=max_items<=50:raise ValueError('Bounded batch size must be 1..50')
    attempt,evidence=media.evidence_path(attempt,evidence)
    identity,base,original=verify_source(attempt,evidence)
    with exclusive(evidence/'audit.lock'),exclusive(base/'audit.lock'):
        raw=(attempt/'episode_ledger.json').read_bytes()
        manifest=history.manifest_identity(attempt/'condition_manifest.json')
        resolved=history.resolve(manifest,raw,history.recovery_references(attempt))
        index=resolved['history_index'];valid=resolved['valid_by_key']
        recovery_index=attempt/'retry_recovery_index.json'
        if recovery_index.exists():index['recovery_index_file']=history.recovery_index_reference(attempt)
        if final and (resolved['unresolved'] or len(valid)!=index['canonical_count'] or keys):
            raise ValueError('Cannot seal media with unresolved/incomplete actual attempt history')
        batch=evidence/'batches'/uuid.uuid4().hex;batch.mkdir(parents=True,exist_ok=False)
        with (batch/'episode_ledger.snapshot.json').open('xb') as stream:stream.write(raw)
        media.unique_write(batch/'retry_history_index.json',index)
        receipts={}
        for path in (base/'receipts').glob('*.json'):
            saved=media.read(path);key=saved['binding']['key']
            if key not in valid or path.name!=key+'.json' or key in receipts:
                raise ValueError('Old media store contains unknown/duplicate current canonical keys')
            row=valid[key]
            if saved.get('passed') is not True:
                raise ValueError('Old failed media evidence requires explicit investigation, never implicit retry')
            for field in ('key','condition','model','suite','task','init','protocol_hash','schedule','mask','token','validated_sha256'):
                if saved['binding'][field]!=row[field]:raise ValueError('Old media receipt token/validation/scientific tuple changed')
            if (saved['source_manifest_sha256']!=sha(base/'source_manifest.json')
                    or saved['source_acceptance_sha256']!=sha(base/'source_acceptance.json') or saved['before']!=saved['after']):
                raise ValueError('Old media receipt source/acceptance/stability changed')
            media.verify_receipt_outputs(base,original,saved)
            receipts[key]=dict(path=str(path),sha256=sha(path),source='unchanged original media helper')
        requested=list(keys) if keys else (sorted(valid) if final else sorted(set(valid)-set(receipts)))
        if len(requested)!=len(set(requested)) or any(key not in valid for key in requested):raise ValueError('Duplicate/nonvalid requested media key')
        processed=0;current=[]
        for key in requested:
            existed=key in receipts
            if not existed and processed>=max_items:continue
            try:
                path=media.audit_one(attempt,base,original,manifest['data'],valid[key])
            except BaseException as error:
                media.unique_write(batch/'failure.json',dict(passed=False,key=key,error=repr(error),history_index_sha256=sha(batch/'retry_history_index.json')))
                raise
            receipts[key]=dict(path=str(path),sha256=sha(path),source='unchanged original media helper')
            current.append(key);processed+=not existed
        verify_source(attempt,evidence)
        if final and (set(current)!=set(valid) or (attempt/'episode_ledger.json').read_bytes()!=raw):
            raise ValueError('Final media history/current tuple coverage changed or incomplete')
        result=dict(schema=1,kind='history_aware_independent_media_coverage',partial=not final,complete=final,
            source_manifest_sha256=sha(evidence/'source_manifest.json'),source_acceptance_sha256=sha(evidence/'source_acceptance.json'),
            base_manifest_sha256=sha(base/'source_manifest.json'),base_acceptance_sha256=sha(base/'source_acceptance.json'),
            ledger_snapshot=str(batch/'episode_ledger.snapshot.json'),ledger_snapshot_sha256=sha(batch/'episode_ledger.snapshot.json'),
            history_index=str(batch/'retry_history_index.json'),history_index_sha256=sha(batch/'retry_history_index.json'),
            canonical_count=index['canonical_count'],effective_valid_count=len(valid),unresolved_count=len(resolved['unresolved']),
            covered_count=len(receipts),new_decodes=processed,current_batch_reverified_keys=current,
            historical_receipts_not_all_current_rehashed=not final,receipts=receipts,
            actual_execution='new history-aware controller calling original strict media functions; never original snapshot() on projected data',
            independent_final_C_acceptance_required=True)
        output=batch/'index.json';media.unique_write(output,result);return output


def verify_final_index(attempt,index_path):
    attempt=Path(attempt).resolve();index_path=Path(index_path).resolve()
    evidence=index_path.parents[2]
    identity,base,original=verify_source(attempt,evidence)
    index=media.read(index_path)
    if (index.get('kind')!='history_aware_independent_media_coverage' or index.get('partial') is not False
            or index.get('complete') is not True or index.get('historical_receipts_not_all_current_rehashed') is not False
            or index['source_manifest_sha256']!=sha(evidence/'source_manifest.json')
            or index['source_acceptance_sha256']!=sha(evidence/'source_acceptance.json')
            or index['base_manifest_sha256']!=sha(base/'source_manifest.json')
            or index['base_acceptance_sha256']!=sha(base/'source_acceptance.json')):
        raise ValueError('Partial/stale/wrong-kind media index cannot seal final coverage')
    raw=(attempt/'episode_ledger.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=index['ledger_snapshot_sha256'] or sha(index['ledger_snapshot'])!=index['ledger_snapshot_sha256']:
        raise ValueError('Final media index does not bind exact real complete current ledger')
    manifest=history.manifest_identity(attempt/'condition_manifest.json')
    resolved=history.resolve(manifest,raw,history.recovery_references(attempt))
    expected=resolved['history_index']
    if (attempt/'retry_recovery_index.json').exists():
        expected['recovery_index_file']=history.recovery_index_reference(attempt)
    if (resolved['unresolved'] or sha(index['history_index'])!=index['history_index_sha256']
            or media.read(index['history_index'])!=expected or set(index['receipts'])!=set(resolved['valid_by_key'])
            or set(index['current_batch_reverified_keys'])!=set(resolved['valid_by_key'])
            or index['covered_count']!=len(resolved['valid_by_key']) or index['unresolved_count']!=0):
        raise ValueError('Media final coverage omitted actual retry history/current canonical keys')
    for key,row in resolved['valid_by_key'].items():
        reference=index['receipts'][key];path=Path(reference['path']).resolve()
        if path!=base/'receipts'/f'{key}.json' or sha(path)!=reference['sha256']:
            raise ValueError('Final media receipt identity changed')
        old=media.read(path);bound,before=media.binding(attempt,manifest['data'],row)
        if (old.get('passed') is not True or old['binding']!=bound or old['before']!=before or old['after']!=before
                or old['source_manifest_sha256']!=sha(base/'source_manifest.json')
                or old['source_acceptance_sha256']!=sha(base/'source_acceptance.json')):
            raise ValueError('Final media current tuple/source is not the sealed decoded receipt')
        media.verify_receipt_outputs(base,original,old)
    verify_source(attempt,evidence)
    if (attempt/'episode_ledger.json').read_bytes()!=raw:raise ValueError('Ledger changed during final media revalidation')
    return dict(index_path=str(index_path),index_sha256=sha(index_path),ledger_sha256=index['ledger_snapshot_sha256'],
        history_index_sha256=index['history_index_sha256'],covered_count=index['covered_count'],
        source_manifest_sha256=index['source_manifest_sha256'],source_acceptance_sha256=index['source_acceptance_sha256'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=('prepare','audit','finalize'));parser.add_argument('--attempt',required=True)
    parser.add_argument('--evidence',required=True);parser.add_argument('--base-evidence');parser.add_argument('--max-items',type=int,default=50)
    parser.add_argument('--key',action='append');args=parser.parse_args()
    if args.operation=='prepare':
        if not args.base_evidence:parser.error('prepare requires base-evidence')
        prepare(args.attempt,args.evidence,args.base_evidence)
    else:
        os.nice(max(0,19-os.nice(0)))
        print(audit(args.attempt,args.evidence,args.max_items,args.key,final=args.operation=='finalize'))


if __name__=='__main__':main()

"""Read-only canonical retry history resolution; never rewrites/projects a ledger."""
import hashlib
import json
import math
from pathlib import Path

from common import digest, planned, sha


def manifest_identity(path):
    raw = Path(path).read_bytes()
    return dict(data=json.loads(raw), file_sha256=hashlib.sha256(raw).hexdigest(),path=str(Path(path).resolve()))


def evidence_bytes(reference, files):
    path = Path(reference['path']).resolve()
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != reference['sha256']:
        raise ValueError('Retry evidence bytes changed')
    files[str(path)] = actual
    return raw


def read_evidence(reference, files):
    return json.loads(evidence_bytes(reference, files))


def chain_for(ledger, key):
    return [row for row in ledger['entries'] if row['key'] == key]


def load_authorizations(manifest, references):
    found, files = {}, {}
    for reference in references:
        receipt = read_evidence(reference, files)
        if receipt.get('schema') != 1 or receipt.get('kind') != 'technical_retry_authorization' or receipt.get('passed') is not True:
            raise ValueError('Unknown retry recovery receipt')
        identity = (receipt['key'], receipt['old_token'])
        if identity in found or receipt['manifest_file_sha256'] != manifest['file_sha256']:
            raise ValueError('Duplicate/wrong-manifest retry authorization evidence')
        source = Path(receipt['source']).resolve()
        if source != Path(__file__).with_name('retry_history_transition.py').resolve() or sha(source) != receipt['source_sha256']:
            raise ValueError('Retry transition source changed')
        files[str(source)] = receipt['source_sha256']
        before = chain_for(read_evidence(receipt['before_snapshot'], files), receipt['key'])
        after = chain_for(read_evidence(receipt['after_snapshot'], files), receipt['key'])
        transition = read_evidence(receipt['transition_receipt'], files)
        if (transition.get('schema') != 1 or transition.get('passed') is not True or transition.get('kind') != 'technical_retry_transition'
                or transition['key'] != receipt['key'] or transition['old_token'] != receipt['old_token']
                or transition['source_sha256'] != receipt['source_sha256'] or transition['before_snapshot'] != receipt['before_snapshot']
                or transition['preflight'] != receipt['preflight']):
            raise ValueError('Retry authorization lacks its exact preserved transition chain')
        invalid = chain_for(read_evidence(transition['after_snapshot'], files), receipt['key'])
        for record, phase, chain in ((transition,'transition_only',before),(receipt,'authorize_retry',invalid)):
            approval = read_evidence(record['approval'], files)
            accepted = read_evidence(approval['source_acceptance'], files)
            if (approval.get('schema') != 1 or approval.get('accepted') is not True or approval.get('authorized_by') != 'root'
                    or approval.get('target_single_owner') is not True or approval.get('phase') != phase
                    or approval.get('key') != receipt['key'] or approval.get('old_token') != receipt['old_token']
                    or approval.get('attempt') != str(Path(manifest['path']).parent) or record['attempt'] != approval['attempt'] or receipt['attempt'] != approval['attempt']
                    or approval.get('expected_chain_sha256') != digest(chain) or approval.get('source_sha256') != receipt['source_sha256']
                    or approval.get('manifest_file_sha256') != manifest['file_sha256']
                    or accepted.get('schema') != 1 or accepted.get('accepted') is not True or accepted.get('reviewer_model') != 'gpt-6.1-sol'
                    or accepted.get('reasoning_effort') != 'medium'
                    or accepted['source_hashes'].get('retry_history_transition.py') != receipt['source_sha256']):
                raise ValueError('Recovery root/source authority does not bind exact phase/key/token/chain')
        cli = receipt['original_cli']
        script = Path(__file__).with_name('runtime_runner.py').resolve()
        attempt = Path(receipt['attempt']).resolve()
        command = cli['command']
        expected = [command[0],str(script),'retry','--attempt',str(attempt),'--key',receipt['key'],
            '--diagnosis',receipt['diagnosis'],'--launch']
        if command != expected or cli['exit_code'] != 0 or cli['script_sha256'] != sha(script):
            raise ValueError('Original retry CLI command/script/exit evidence differs')
        python=Path(cli['python']['path']).resolve()
        if python!=Path(command[0]).resolve() or sha(python)!=cli['python']['sha256']:
            raise ValueError('Original retry interpreter identity changed')
        files[str(python)]=cli['python']['sha256']
        files[str(script)] = cli['script_sha256']
        for channel in ('stdout','stderr'):
            if Path(cli[channel]['path']).resolve()!=Path(reference['path']).resolve().parent/channel:
                raise ValueError('Original retry output escaped its preserved evidence directory')
            evidence_bytes(cli[channel],files)
        if (not before or len(before) != len(after) or before[-1]['token'] != receipt['old_token']
                or after[-1]['token'] != receipt['old_token'] or before[:-1] != after[:-1]
                or after[-1]['status'] != 'invalid' or not receipt.get('diagnosis', '').strip()
                or after[-1].get('retry_diagnosis') != receipt['diagnosis']
                or after[-1].get('retry_authorized_at') != receipt['retry_authorized_at']):
            raise ValueError('Retry authorization snapshots do not show exact old-token authorization')
        original = dict(before[-1])
        if original['status'] == 'reserved':
            preflight = receipt['preflight']
            if (preflight.get('owner_start_ticks') == original['start_ticks'] or preflight.get('validated_present') is not False
                    or preflight.get('expected_directory_present') is not False or original.get('directory')
                    or original.get('validated_sha256') or original.get('retry_diagnosis')):
                raise ValueError('Reserved recovery lacked strict dead-owner/unexecuted preconditions')
            original.update(status='invalid', error='Owner ended before atomic validation')
        elif original['status'] != 'invalid':
            raise ValueError('Only an explicit technical invalid attempt may authorize retry')
        if invalid != before[:-1]+[original]:
            raise ValueError('State-only transition changed original target fields/token count')
        original.update(retry_diagnosis=receipt['diagnosis'], retry_authorized_at=receipt['retry_authorized_at'])
        if original != after[-1]:
            raise ValueError('Recovery changed fields beyond the original invalid/authorization transition')
        found[identity] = dict(receipt=receipt, after_chain=after, reference=dict(path=str(Path(reference['path']).resolve()), sha256=reference['sha256']))
    return found, files


def resolve(manifest, ledger_bytes, recovery_receipts=()):
    if set(manifest) != {'data', 'file_sha256','path'}:
        raise ValueError('Resolver requires exact manifest bytes SHA plus parsed data')
    scientific = manifest['data']
    canonical = {row['key']:row for row in planned(scientific)}
    ledger = json.loads(ledger_bytes)
    authorizations, files = load_authorizations(manifest, recovery_receipts)
    chains, tokens = {}, set()
    for position, row in enumerate(ledger['entries']):
        key = row['key']
        if key not in canonical or any(row.get(field) != value for field,value in canonical[key].items()):
            raise ValueError('History scientific key/protocol/schedule/mask differs from canonical plan')
        if row['token'] in tokens or row['status'] not in ('reserved', 'invalid', 'valid'):
            raise ValueError('Duplicate attempt token or unknown status')
        tokens.add(row['token'])
        chains.setdefault(key, []).append((position,row))
    valid, unresolved, keys, used = {}, [], {}, set()
    for key, canonical_row in canonical.items():
        attempts = chains.get(key, [])
        rows = [row for _,row in attempts]
        if len(rows) > 3:
            raise ValueError('Original three-attempt technical retry budget exceeded')
        effective, transitions = None, []
        for number, row in enumerate(rows):
            if effective is not None:
                raise ValueError('Attempt appended after an already valid episode')
            if row['status'] == 'valid':
                if not row.get('directory') or not row.get('validated_sha256') or not isinstance(row.get('success'),bool):
                    raise ValueError('Valid attempt lacks committed artifact identity')
                effective = row
            if number+1 < len(rows):
                if row['status'] != 'invalid' or not row.get('retry_diagnosis','').strip():
                    raise ValueError('New attempt lacks original explicit technical retry authorization')
                moment = row.get('retry_authorized_at')
                if not isinstance(moment,(int,float)) or isinstance(moment,bool) or not math.isfinite(moment) or moment <= 0:
                    raise ValueError('Invalid original retry authorization timestamp')
                evidence = authorizations.get((key,row['token']))
                if not evidence or evidence['after_chain'] != rows[:number+1]:
                    raise ValueError('Missing/wrong-token/drifted retry authorization evidence')
                if evidence['receipt'].get('canonical_row_sha256') != digest(canonical_row):
                    raise ValueError('Recovery receipt belongs to another scientific tuple')
                transitions.append(evidence['reference'])
                used.add((key,row['token']))
        if effective is not None:
            valid[key] = effective
        else:
            unresolved.append(dict(key=key, reason='not_started' if not rows else 'latest_'+rows[-1]['status'],
                token=None if not rows else rows[-1]['token']))
        if rows:
            keys[key] = dict(attempts=[dict(ledger_position=position, entry=row) for position,row in attempts],
                transitions=transitions, effective_valid=None if effective is None else dict(token=effective['token'],validated_sha256=effective['validated_sha256']))
    # An authorization on the latest invalid row is retained even before its new attempt starts.
    for identity, evidence in authorizations.items():
        if identity not in used:
            key, token = identity
            rows = [row for _,row in chains.get(key, [])]
            if not rows or rows[-1]['token'] != token or rows != evidence['after_chain']:
                raise ValueError('Recovery authorization evidence is not represented in the real complete ledger')
            keys[key]['transitions'].append(evidence['reference'])
    index = dict(schema=1, kind='real_complete_retry_history', manifest_path=manifest['path'], manifest_file_sha256=manifest['file_sha256'],
        canonical_manifest_hash=scientific['manifest_hash'], ledger_sha256=hashlib.sha256(ledger_bytes).hexdigest(),
        resolver_source_sha256=sha(__file__), canonical_count=len(canonical), valid_count=len(valid),
        unresolved=unresolved, keys=keys, verified_evidence_file_hashes=files,recovery_receipts=list(recovery_receipts),
        policy='actual complete append-order ledger, never a projected or rewritten ledger; task failures remain valid')
    return dict(valid_by_key=valid, unresolved=unresolved, history_index=index)


def recovery_references(attempt):
    path = Path(attempt)/'retry_recovery_index.json'
    if not path.exists():
        return []
    value = json.loads(path.read_text())
    if value.get('schema') != 1 or value.get('kind') != 'explicit_retry_recovery_index':
        raise ValueError('Unknown recovery receipt index')
    references=[];seen=set()
    for reference in value['entries']:
        path=Path(reference['path']).resolve()
        if sha(path)!=reference['sha256']:raise ValueError('Registered retry receipt changed')
        receipt=json.loads(path.read_text())
        phase=reference['phase']
        kind={'transition_only':'technical_retry_transition','authorize_retry':'technical_retry_authorization'}.get(phase)
        identity=(receipt['key'],receipt['old_token'],phase)
        if kind is None or receipt.get('kind')!=kind or receipt.get('passed') is not True or identity in seen:
            raise ValueError('Duplicate/unknown/failed recovery index phase')
        seen.add(identity)
        if phase=='authorize_retry':references.append(dict(path=str(path),sha256=reference['sha256']))
    return references


def recovery_index_reference(attempt):
    path=Path(attempt)/'retry_recovery_index.json'
    if not path.exists():return None
    raw=path.read_bytes();value=hashlib.sha256(raw).hexdigest()
    preserved=Path(attempt)/'retry_recovery/index_history'/f'{value}.json'
    if not preserved.exists() or preserved.read_bytes()!=raw:
        raise ValueError('Current recovery index lacks its immutable original bytes snapshot')
    return dict(path=str(preserved),sha256=value)

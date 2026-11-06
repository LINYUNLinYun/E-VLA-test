"""Read-only verification of actual throughput receipt hashes, including history."""
import argparse
import hashlib
import json
from pathlib import Path

from common import atomic_json, digest, lock, planned, prior_diagnostic_budget, sha
from throughput_revision import read, verify_revision as verify_legacy_revision, compare_gate, gate_requests


BASE_AUDITOR_SOURCE_SHA256 = 'bf88f2441ad3a25560c2cd5a07c3ea864f116b515eebad121680c647ce67db27'


def verify_revision(revision):
    if read(Path(revision)/'revision_manifest.json').get('kind') == 'pi0_prompt_length_runtime_cache2':
        from pi0_prompt_cache_revision import verify_revision as verify_cache
        return verify_cache(revision)
    return verify_legacy_revision(revision)


def verify_gate(index, model, manifest, runtime):
    if read(index.revision/'revision_manifest.json').get('kind') == 'pi0_prompt_length_runtime_cache2':
        from pi0_prompt_cache_audit_helpers import verify_gate as verify_cache_gate
        return verify_cache_gate(index, model, manifest, runtime)
    return verify_legacy_gate(index, model, manifest, runtime)


class ReceiptIndex:
    def __init__(self, revision, models):
        self.revision = Path(revision).resolve()
        self.revision_hash = sha(self.revision/'revision_manifest.json')
        self.files, self.receipts, self.corrections = {}, {}, []
        for model in sorted(models):
            for phase in ('source', 'gate', 'pilot'):
                self.add_receipt(self.revision/'acceptance'/f'{model}.{phase}.json', model, phase)
            for path in sorted((self.revision/'acceptance').glob(f'{model}.*.evidence_path_correction.json')):
                self.add_correction(path, model)

    def record(self, path):
        path = Path(path).resolve()
        value = sha(path)
        if str(path) in self.files and self.files[str(path)] != value:
            raise ValueError('An immutable indexed file changed during audit')
        self.files[str(path)] = value
        return value

    def load(self, path):
        path = Path(path).resolve()
        data = path.read_bytes()
        value = hashlib.sha256(data).hexdigest()
        if str(path) in self.files and self.files[str(path)] != value:
            raise ValueError('An immutable indexed file changed during audit')
        self.files[str(path)] = value
        return json.loads(data)

    def add_receipt(self, path, model, phase):
        receipt = self.load(path)
        if (receipt.get('schema') != 1 or receipt.get('accepted') is not True or
                receipt.get('reviewer_model') != 'gpt-6.1-sol' or receipt.get('reasoning_effort') != 'medium' or
                receipt.get('model') != model or receipt.get('phase') != phase or
                receipt.get('revision_manifest_sha256') != self.revision_hash or not receipt.get('evidence')):
            raise ValueError(f'Invalid acceptance identity: {path}')
        if phase != 'source':
            ready = self.revision/f'{model}.{phase}_ready.json'
            if receipt.get('phase_receipt_sha256') != self.record(ready):
                raise ValueError(f'Acceptance phase-ready identity drift: {path}')
        for evidence, expected in receipt.get('evidence_sha256', {}).items():
            source = Path(evidence)
            source = source if source.is_absolute() else self.revision/source
            if self.record(source) != expected:
                raise ValueError(f'Signed evidence drift: {source}')
        value = self.record(path)
        existing = self.receipts.setdefault(value, dict(model=model, phase=phase,
            revision_manifest_sha256=self.revision_hash, paths=[], receipt=receipt))
        if existing['model'] != model or existing['phase'] != phase:
            raise ValueError('Receipt hash has conflicting identity')
        resolved = str(Path(path).resolve())
        if resolved not in existing['paths']:
            existing['paths'].append(resolved)
        return value

    def add_correction(self, path, model):
        correction = self.load(path)
        if (correction.get('schema') != 1 or correction.get('kind') != 'display_evidence_path_correction' or
                correction.get('reviewer_model') != 'gpt-6.1-sol' or correction.get('reasoning_effort') != 'medium' or
                correction.get('scientific_acceptance_unchanged') is not True):
            raise ValueError('Unknown/unreviewed receipt correction')
        names = [correction['original_receipt'], correction['corrected_receipt']]
        if any(Path(name).name != name for name in names) or names[1] != f'{model}.pilot.json':
            raise ValueError('Correction must point to local preserved pilot receipts')
        old, new = [Path(path).parent/name for name in names]
        old_bytes, new_bytes = old.read_bytes(), new.read_bytes()
        before, after = correction['change']['from'], correction['change']['to']
        if (not before or before == after or '/episodes/' not in before or '/episodes/' not in after or
                old_bytes.count(before.encode()) == 0 or old_bytes.replace(before.encode(), after.encode()) != new_bytes):
            raise ValueError('Receipt correction changed undeclared bytes')
        left, right = json.loads(old_bytes), json.loads(new_bytes)
        if {k: v for k, v in left.items() if k != 'evidence'} != {k: v for k, v in right.items() if k != 'evidence'}:
            raise ValueError('Correction changed scientific acceptance fields')
        old_hash = self.add_receipt(old, model, 'pilot')
        new_hash = self.add_receipt(new, model, 'pilot')
        if (old_hash != correction['original_receipt_sha256'] or new_hash != correction['corrected_receipt_sha256'] or
                correction['phase_receipt_sha256'] != left['phase_receipt_sha256']):
            raise ValueError('Correction does not bind original/corrected bytes and phase-ready')
        self.corrections.append(dict(path=str(Path(path).resolve()), sha256=self.record(path),
            original_sha256=old_hash, corrected_sha256=new_hash, model=model, phase='pilot'))

    def resolve(self, value, model, phase):
        if value not in self.receipts:
            raise ValueError(f'Unknown acceptance SHA: {value}')
        found = self.receipts[value]
        if found['model'] != model or found['phase'] != phase:
            raise ValueError('Acceptance SHA resolves to the wrong model/phase')
        # Resolve using actual preserved bytes, never a hash-only allowlist.
        if not any(sha(path) == value for path in found['paths']):
            raise ValueError('Preserved receipt changed after indexing')
        return found['receipt']

    def equivalent(self, left, right, model, phase):
        self.resolve(left, model, phase)
        self.resolve(right, model, phase)
        return left == right or any({left, right} == {item['original_sha256'], item['corrected_sha256']}
            and item['model'] == model and item['phase'] == phase for item in self.corrections)


def verify_legacy_gate(index, model, manifest, runtime):
    from runtime_client import check_prediction
    from runtime_data import audited_protocol
    ready = index.load(index.revision/f'{model}.gate_ready.json')
    session = Path(ready['session']).resolve()
    if (ready.get('model') != model or ready.get('phase') != 'gate' or
            ready.get('revision_manifest_sha256') != index.revision_hash or ready.get('exact') is not True or
            ready.get('actual_predictions') != 8 or ready['calls_after']-ready['calls_before'] != 8 or
            ready['reference_order'] != [0, 2, 1, 3] or ready['interleaved_order'] != [0, 1, 2, 3]):
        raise ValueError('Invalid eight-call equivalence gate identity')
    if session.parent != index.revision/'sessions':
        raise ValueError('Gate session is outside its revision')
    requests = read(session/'requests.json')
    if requests != gate_requests(manifest, model):
        raise ValueError('Gate requests differ from frozen observation/mask/seed rules')
    index.record(session/'requests.json')
    protocol = audited_protocol(Path(read(index.revision/'revision_manifest.json')['source_attempt']), model, runtime)
    for phase in ('reference', 'interleaved'):
        for request in requests:
            path = session/phase/f"{request['index']}.json"
            row = index.load(path)
            check_prediction(row, request['condition'], protocol, request['seed'])
            if row['input_hash'] != request['point']['content_hash'] or row['loaded_code_hashes'] != runtime['scientific_code_hashes']:
                raise ValueError('Gate input/scientific code identity differs')
            index.record(path)
            for key in ('noise_path', 'server_prediction_path'):
                artifact = Path(row[key]).resolve()
                if not artifact.is_relative_to(session/'server'):
                    raise ValueError('Gate raw artifact came from another session')
                index.record(artifact)
    if compare_gate(session) != ready['comparisons']:
        raise ValueError('Gate exact raw/action/noise comparison changed')
    identities = ready['client_identities']
    if len(identities) != 2 or len({item['pid'] for item in identities}) != 2 or {item['slot'] for item in identities} != {0, 1}:
        raise ValueError('Gate did not use two distinct client processes')
    for item in identities:
        path = session/f"client{item['slot']}_identity.json"
        if read(path) != item or not item['start_ticks']:
            raise ValueError('Gate client process identity changed')
        index.record(path)
    for path, expected in ready.get('startup_forward_ledgers', {}).items():
        if not Path(path).resolve().is_relative_to(session/'server') or index.record(path) != expected:
            raise ValueError('Startup forward evidence changed')
    summaries = []
    for slot in (0, 1):
        path = session/f'client{slot}_summary.json'
        summary = read(path)
        if summary != dict(slot=slot, predictions=2, valid=0):
            raise ValueError('Gate client summary is not exactly two reserved predictions')
        index.record(path)
        summaries.append(summary)
    complete = read(session/'complete.json')
    if complete['model'] != model or complete['phase'] != 'gate' or complete['session'] != str(session) or complete['client_summaries'] != summaries or complete['revision_manifest_sha256'] != index.revision_hash or complete['server_calls'] != ready['calls_after']:
        raise ValueError('Gate session completion differs from the exact gate')
    index.record(session/'complete.json')
    attempt = Path(read(index.revision/'revision_manifest.json')['source_attempt'])
    budget_path = attempt/'readiness'/model/'throughput_diagnostic_budget.json'
    budget = read(budget_path)
    matching = [entry for entry in budget['entries'] if entry['throughput_revision_sha256'] == index.revision_hash]
    if len(matching) != 1 or matching[0].get('status') != 'gate_complete' or matching[0].get('reserved_predictions') != 8 or matching[0].get('actual_predictions') != 8 or matching[0].get('gate_receipt_sha256') != sha(index.revision/f'{model}.gate_ready.json'):
        raise ValueError('Cumulative diagnostic reservation does not bind this exact eight-call gate')
    entry = matching[0]
    if entry['base_total']+8 != ready['total_diagnostic_reserved'] or ready['total_diagnostic_reserved'] > 256:
        raise ValueError('Gate conservative diagnostic budget differs/exceeds cap')
    if index.record(entry['base_budget_path']) != entry['base_budget_sha256'] or index.record(entry['parity_path']) != entry['parity_sha256']:
        raise ValueError('Original signed diagnostic/parity budget identity changed')
    running = (index.load(entry['base_budget_path'])['smoke_reserved'] + index.load(entry['parity_path'])['reserved_calls']
               + prior_diagnostic_budget(manifest, model))
    for reservation in budget['entries']:
        if reservation['base_total'] != running or reservation['reserved_predictions'] != 8:
            raise ValueError('Cumulative throughput diagnostic reservation sequence changed')
        running += reservation['reserved_predictions']
    if running > 256:
        raise ValueError('Combined diagnostic reservations exceed the model cap')
    index.record(budget_path)
    index.record(index.revision/f'{model}.gate_ready.json')


def inference_fingerprint(row, client=False):
    keys = ('call', 'category', 'input_hash', 'noise_hash', 'raw_prediction_hash', 'seed', 'loaded_code_hashes', 'noise_path')
    value = {key: row[key] for key in keys}
    value['prediction_path'] = row['server_prediction_path'] if client else row['prediction_path']
    value['audit'] = {key: row['audit'].get(key) for key in ('NFE', 'actual_NFE', 'schedule', 'skip', 'cache_key', 'instrumentation_mode')}
    return digest(value)


def server_calls(path):
    """Fingerprint a complete-line prefix of a live append-only prediction ledger."""
    calls, checksum, length = {}, hashlib.sha256(), 0
    with Path(path).open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                break
            row = json.loads(line)
            if row['call'] in calls:
                raise ValueError('Duplicate logical call in one sequential server ledger')
            calls[row['call']] = inference_fingerprint(row)
            checksum.update(line)
            length += len(line)
    return calls, dict(path=str(Path(path).resolve()), prefix_bytes=length, prefix_sha256=checksum.hexdigest())


def verify_index_snapshot(index):
    if sha(__file__) != index['auditor_source_sha256']:
        raise ValueError('Auditor source changed after snapshot')
    if (sha(Path(__file__).with_name('throughput_provenance_audit.py')) != BASE_AUDITOR_SOURCE_SHA256 or
            sha(Path(__file__).with_name('pi0_prompt_cache_audit_helpers.py')) != index['cache_audit_helper_source_sha256']):
        raise ValueError('Preserved base auditor/cache helper source changed')
    for revision in index['revisions'].values():
        for path, expected in revision['verified_file_hashes'].items():
            if sha(path) != expected:
                raise ValueError(f'Indexed source changed after audit: {path}')
    for session in index['sessions'].values():
        for prefix in (session.get('server_prediction_ledger_prefix'), session.get('prompt_cache_ledger_prefix')):
            if not prefix:
                continue
            checksum = hashlib.sha256()
            remaining = prefix['prefix_bytes']
            with Path(prefix['path']).open('rb') as stream:
                while remaining:
                    block = stream.read(min(1024*1024, remaining))
                    if not block:
                        raise ValueError('Server prediction ledger prefix was truncated')
                    checksum.update(block)
                    remaining -= len(block)
            if checksum.hexdigest() != prefix['prefix_sha256']:
                raise ValueError('Server prediction ledger prefix changed')
            if not index['partial'] and Path(prefix['path']).stat().st_size != prefix['prefix_bytes']:
                raise ValueError('Final server prediction ledger changed length')


def audit(attempt, partial=True):
    """Audit a stable ledger snapshot; this does not replace the full artifact audit."""
    from runtime_data import verify_frozen, verify_parity, verify_acceptance
    attempt = Path(attempt).resolve()
    manifest = verify_frozen(attempt)
    if sha(Path(__file__).with_name('throughput_provenance_audit.py')) != BASE_AUDITOR_SOURCE_SHA256:
        raise ValueError('Preserved legacy auditor source changed')
    with lock(attempt/'episode_ledger.lock'):
        snapshot = (attempt/'episode_ledger.json').read_bytes()
    ledger = json.loads(snapshot)
    valid = [entry for entry in ledger['entries'] if entry['status'] == 'valid']
    if len({entry['key'] for entry in valid}) != len(valid):
        raise ValueError('Duplicate valid canonical key')
    pending = [entry for entry in ledger['entries'] if entry['status'] in ('reserved', 'invalid')]
    canonical = {row['key']: row for row in planned(manifest)}
    if not partial and (pending or set(canonical) != {entry['key'] for entry in valid}):
        raise ValueError('Final provenance sealing requires all canonical keys and no reserved/invalid entries')
    entries = [entry for entry in valid if entry.get('execution_revision_sha256') or entry.get('execution_revision')]
    indexes, session_records, slot_owners, entry_records, call_maps = {}, {}, {}, [], {}
    revision_paths = {str(Path(entry['execution_revision']).resolve()) for entry in entries}
    for path in (attempt/'throughput_revisions').glob('*/revision_manifest.json'):
        if read(path).get('kind') in ('two_env_sequential_server_execution_revision', 'pi0_prompt_length_runtime_cache2') and any((path.parent/'sessions').glob('*/execution_receipt.json')):
            revision_paths.add(str(path.parent.resolve()))
    for revision_path in sorted(revision_paths):
        revision = Path(revision_path).resolve()
        identity, scientific = verify_revision(revision)
        if Path(identity['source_attempt']).resolve() != attempt or scientific != manifest:
            raise ValueError('Throughput revision belongs to another scientific attempt')
        cache_revision = identity.get('kind') == 'pi0_prompt_length_runtime_cache2'
        models = {'pi0'} if cache_revision else set(manifest['protocol_hashes'])
        index = ReceiptIndex(revision, models)
        index.record(revision/'revision_manifest.json')
        index.record(attempt/'condition_manifest.json')
        for name, expected in identity['code_hashes'].items():
            if index.record(Path(__file__).resolve().parent/name) != expected:
                raise ValueError('Throughput execution source identity changed')
        indexes[str(revision)] = index
        # Validate every recorded session acceptance, including long-running exec
        # receipts whose acceptance predates an evidence-path correction.
        for path in sorted((revision/'sessions').glob('*/execution_receipt.json')):
            execution = index.load(path)
            model, phase = execution['model'], execution['phase']
            if model not in models:
                raise ValueError('Recorded session has an unknown scientific model')
            if phase not in ('gate', 'pilot', 'run') or execution.get('schema') != 1 or execution.get('environments') != (0 if cache_revision and phase == 'gate' else 2) or execution.get('revision_manifest_sha256') != index.revision_hash:
                raise ValueError('Throughput session execution identity changed')
            index.resolve(execution['phase_acceptance_sha256'], model, {'gate': 'source', 'pilot': 'gate', 'run': 'pilot'}[phase])
            runtime = execution['frozen_runtime_protocol']
            verify_frozen(attempt, runtime, model)
            verify_acceptance(attempt, model, verify_parity(attempt, runtime, model))
            native_protocol = path.parent/'server'/'native_protocol.json'
            service_lease = path.parent/'server'/'service_lease.json'
            lease = index.load(service_lease)
            if index.load(native_protocol) != runtime or lease['protocol'] != runtime or lease['pid'] != execution['server_pid'] or str(lease['gpu']) != str(execution['gpu']) or lease['port'] != execution['port']:
                raise ValueError('Native protocol/service lease differs from execution receipt')
            index.record(native_protocol)
            index.record(service_lease)
            session_records[str(path.parent.resolve())] = dict(model=model, phase=phase,
                receipt_path=str(path.resolve()), receipt_sha256=index.record(path),
                acceptance_sha256=execution['phase_acceptance_sha256'], revision=str(revision),
                server_pid=execution['server_pid'], manager_pid=execution['manager_pid'], gpu=execution['gpu'])
        for model in sorted(models):
            gate = read(revision/f'{model}.gate_ready.json')
            gate_session = session_records.get(str(Path(gate['session']).resolve()))
            if not gate_session or gate_session['model'] != model or gate_session['phase'] != 'gate':
                raise ValueError('Gate-ready session is not the accepted model gate')
            verify_gate(index, model, manifest, read(gate_session['receipt_path'])['frozen_runtime_protocol'])
            pilot = index.load(revision/f'{model}.pilot_ready.json')
            marker = index.load(revision/f'{model}.pilot_session.json')
            candidates = pilot['results']
            if (pilot.get('model') != model or pilot.get('phase') != 'pilot' or
                    pilot.get('revision_manifest_sha256') != index.revision_hash or marker['session'] != pilot['session'] or
                    marker['revision_manifest_sha256'] != index.revision_hash or len(candidates) != 2 or
                    {row['slot'] for row in candidates} != {0, 1} or len({row['key'] for row in candidates}) != 2 or
                    len({row['token'] for row in candidates}) != 2):
                raise ValueError('Pilot candidates/session identity changed')
            pilot_session = session_records.get(str(Path(pilot['session']).resolve()))
            if not pilot_session or pilot_session['model'] != model or pilot_session['phase'] != 'pilot':
                raise ValueError('Pilot-ready session is not the accepted model pilot')
            for candidate in candidates:
                if candidate['escrow_directory'] != str(Path(pilot['session'])/'escrow'/candidate['token']):
                    raise ValueError('Pilot escrow was outside its declared session/token')
                source = read(Path(pilot['session'])/f"pilot{candidate['slot']}.json")
                if source != candidate:
                    raise ValueError('Pilot candidate differs from signed ready receipt')
                index.record(Path(pilot['session'])/f"pilot{candidate['slot']}.json")
                matching = [entry for entry in entries if entry['key'] == candidate['key'] and entry['token'] == candidate['token']]
                if len(matching) != 1:
                    raise ValueError('Signed pilot candidate is not exactly one valid ledger entry')
                entry = matching[0]
                if (entry['model'] != model or entry['env_slot'] != candidate['slot'] or
                        Path(entry['throughput_session']).resolve() != Path(pilot['session']).resolve() or
                        entry['directory'] != candidate['candidate_directory'] or entry['validated_sha256'] != candidate['validated_sha256'] or
                        entry['success'] != candidate['success']):
                    raise ValueError('Pilot candidate/ledger binding differs')
            index.record(revision/f'{model}.pilot_session.json')
    for entry in entries:
        revision = str(Path(entry['execution_revision']).resolve())
        index = indexes[revision]
        model = entry['model']
        if entry['execution_revision_sha256'] != index.revision_hash or entry.get('execution_mode') != ('pi0_two_env_prompt_length_cache2' if read(index.revision/'revision_manifest.json').get('kind') == 'pi0_prompt_length_runtime_cache2' else 'two_process_envs_sequential_server'):
            raise ValueError('Ledger execution revision/mode differs')
        if entry['key'] not in canonical or any(entry.get(key) != value for key, value in canonical[entry['key']].items()):
            raise ValueError('Ledger key/scientific fields differ from frozen canonical key')
        session = str(Path(entry['throughput_session']).resolve())
        record = session_records.get(session)
        if not record or record['revision'] != revision or record['model'] != model or record['phase'] not in ('pilot', 'run'):
            raise ValueError('Ledger references an unknown/wrong-model execution session')
        if record['phase'] == 'pilot':
            pilot = read(index.revision/f'{model}.pilot_ready.json')
            if not any(candidate['key'] == entry['key'] and candidate['token'] == entry['token'] for candidate in pilot['results']):
                raise ValueError('Unapproved extra candidate claims the pilot session')
        slot = entry['env_slot']
        if type(slot) is not int or slot not in (0, 1) or str(entry['pid']) != entry['token'].split('-')[0] or not entry['start_ticks']:
            raise ValueError('Ledger environment/process/token identity invalid')
        owner = (entry['pid'], entry['start_ticks'])
        previous = slot_owners.setdefault((session, slot), owner)
        if previous != owner or any(owner == other for (other_session, other_slot), other in slot_owners.items() if other_session == session and other_slot != slot):
            raise ValueError('Environment slot/process identities conflict')
        index.resolve(entry['throughput_acceptance_sha256'], model, 'pilot')
        current_pilot = sha(index.revision/'acceptance'/f'{model}.pilot.json')
        if not index.equivalent(entry['throughput_acceptance_sha256'], current_pilot, model, 'pilot'):
            raise ValueError('Ledger pilot acceptance has no declared equivalence chain')
        if record['phase'] == 'run' and not index.equivalent(record['acceptance_sha256'], entry['throughput_acceptance_sha256'], model, 'pilot'):
            raise ValueError('Run session and ledger signatures differ without a declared correction')
        expected_directory = attempt/'episodes'/entry['condition']/entry['suite']/f"task{entry['task']:02d}"/f"init{entry['init']:02d}"/entry['token']
        if Path(entry['directory']).resolve() != expected_directory.resolve():
            raise ValueError('Formal artifacts are outside the canonical parent episode directory')
        path = expected_directory/'validated.json'
        value = index.load(path)
        if index.record(path) != entry['validated_sha256'] or value.get('valid') is not True or value.get('video_verified') is not True or value.get('episode_key') != entry['key']:
            raise ValueError('Validated artifact/ledger identity changed')
        if any(value.get(field) != entry[field] for field in ('model', 'suite', 'task', 'init', 'success')):
            raise ValueError('Validated scientific fields differ from ledger')
        infer_path = expected_directory/'infer.jsonl'
        if index.record(infer_path) != value['artifacts']['infer.jsonl']:
            raise ValueError('Validated inference log changed')
        if session not in call_maps:
            call_maps[session], prefix = server_calls(Path(session)/'server'/'prediction_ledger.jsonl')
            record['server_prediction_ledger_prefix'] = prefix
        for line in infer_path.read_text().splitlines():
            inference = json.loads(line)
            if call_maps[session].get(inference['call']) != inference_fingerprint(inference, client=True):
                raise ValueError('Episode prediction does not bind to its declared sequential server session')
        if read(index.revision/'revision_manifest.json').get('kind') == 'pi0_prompt_length_runtime_cache2':
            from pi0_prompt_cache_audit_helpers import verify_episode_cache
            verify_episode_cache(index, entry, record, infer_path)
        entry_records.append(dict(key=entry['key'], token=entry['token'], model=model, session=session,
            env_slot=slot, acceptance_sha256=entry['throughput_acceptance_sha256'],
            validated_path=str(path.resolve()), validated_sha256=entry['validated_sha256']))
    for session, record in session_records.items():
        if record['phase'] == 'gate':
            continue
        directory = Path(session)
        revision_index = indexes[record['revision']]
        actual = [entry for entry in entries if str(Path(entry['throughput_session']).resolve()) == session]
        if not (directory/'complete.json').exists():
            if not partial:
                raise ValueError('Final execution provenance requires completed client/session summaries')
            record['active_snapshot'] = True
            continue
        complete = read(directory/'complete.json')
        if complete.get('model') != record['model'] or complete.get('phase') != record['phase'] or complete.get('session') != session or complete.get('revision_manifest_sha256') != revision_index.revision_hash:
            raise ValueError('Session completion identity changed')
        summaries = []
        for slot in (0, 1):
            path = directory/f'client{slot}_summary.json'
            summary = read(path)
            expected = {(entry['key'], entry['token']) for entry in actual if entry['env_slot'] == slot}
            observed = [(entry['key'], entry['token']) for entry in summary['episodes']]
            if summary['slot'] != slot or summary['valid'] != len(expected) or len(observed) != len(set(observed)) or set(observed) != expected:
                raise ValueError('Client summary key/token/slot differs from committed ledger')
            revision_index.record(path)
            summaries.append(summary)
        if complete['client_summaries'] != summaries or complete['valid'] != len(actual):
            raise ValueError('Manager/client completion summaries differ')
        record['complete_sha256'] = revision_index.record(directory/'complete.json')
    return dict(schema=2, kind='throughput_and_pi0_cache_provenance_index', partial=partial,
        base_auditor_source_sha256=BASE_AUDITOR_SOURCE_SHA256, cache_audit_helper_source_sha256=sha(Path(__file__).with_name('pi0_prompt_cache_audit_helpers.py')),
        scientific_artifact_audit_required=True, source_attempt=str(attempt),
        parent_manifest_sha256=sha(attempt/'condition_manifest.json'),
        ledger_snapshot_sha256=hashlib.sha256(snapshot).hexdigest(), ledger_snapshot=ledger,
        valid_scientific_entries=len(valid), pending_entries=len(pending), throughput_entries=entry_records,
        sessions=session_records, revisions={revision: dict(revision_manifest_sha256=index.revision_hash,
            hash_to_preserved_receipt=index.receipts, corrections=index.corrections, verified_file_hashes=index.files)
            for revision, index in indexes.items()}, auditor_source_sha256=sha(__file__))


def write_index(attempt, output, partial=True):
    output = Path(output).resolve()
    result = audit(attempt, partial)
    verify_index_snapshot(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('Never overwrite a prior provenance snapshot')
    atomic_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    write_index(args.attempt, args.output, partial=not args.final)


if __name__ == '__main__':
    main()


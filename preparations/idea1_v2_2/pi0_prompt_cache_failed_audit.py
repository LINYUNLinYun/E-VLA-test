"""Strictly scoped preserved failed diagnostic evidence; never authorizes episodes."""
import json
from pathlib import Path
import numpy as np

from common import sha, prior_diagnostic_budget
from throughput_revision import read

FAILURE_SHA256 = '3cc392bcaa3f0061ada6a05882e07136512590ce171230e4b4c7f87c4c349dea'


def declared_failures(revision_paths):
    result = {}
    for path in revision_paths:
        identity = read(Path(path)/'revision_manifest.json')
        controller = identity.get('gate_controller')
        if not controller:
            continue
        from pi0_prompt_cache_order_revision import validate_order
        validate_order(identity)
        failed = str(Path(controller['failed_revision']).resolve())
        if failed not in revision_paths or failed == str(Path(path).resolve()):
            raise ValueError('Declared failed diagnostic revision is absent or self-referential')
        failed_identity = read(Path(failed)/'revision_manifest.json')
        if failed_identity.get('kind') != 'pi0_prompt_length_runtime_cache2' or failed_identity.get('gate_controller'):
            raise ValueError('Unknown failed diagnostic revision kind')
        report = Path(failed)/'independent_gate_failure.json'
        if sha(report) != FAILURE_SHA256:
            raise ValueError('Unknown or changed independent failed diagnostic report')
        result[failed] = dict(declared_by=str(Path(path).resolve()), report=str(report), report_sha256=FAILURE_SHA256)
    return result


def verify_failed(index, manifest, ledger, declaration):
    from runtime_client import check_prediction
    from runtime_data import audited_protocol, verify_frozen, verify_parity
    from pi0_prompt_cache_audit_helpers_v3 import verify_revision, verify_patch, cache_events
    revision = index.revision
    identity, _ = verify_revision(revision)
    report = index.load(declaration['report'])
    if index.record(declaration['report']) != declaration['report_sha256']:
        raise ValueError('Failed diagnostic declaration drift')
    if any(str(Path(entry.get('execution_revision', '/absent')).resolve()) == str(revision)
           or entry.get('execution_revision_sha256') == index.revision_hash for entry in ledger['entries']):
        raise ValueError('Failed-only revision has a formal ledger entry of any status')
    sessions = list((revision/'sessions').glob('*/execution_receipt.json'))
    if len(sessions) != 1 or any((revision/'acceptance'/f'pi0.{phase}.json').exists() or
            (revision/f'pi0.{phase}_ready.json').exists() for phase in ('gate', 'pilot')):
        raise ValueError('Failed diagnostic has unexpected sessions or successful phase claims')
    execution = index.load(sessions[0])
    if (execution['schema'] != 1 or execution['phase'] != 'gate' or execution['model'] != 'pi0' or execution['environments'] != 0
            or execution['gate_request_clients'] != 1 or execution['revision_manifest_sha256'] != index.revision_hash):
        raise ValueError('Failed-only execution is not exactly its single gate')
    index.resolve(execution['phase_acceptance_sha256'], 'pi0', 'source')
    session = sessions[0].parent
    reference = revision/'gate_reference'/session.name
    lease = index.load(session/'server/service_lease.json')
    if (lease['protocol'] != execution['frozen_runtime_protocol'] or lease['pid'] != execution['server_pid']
            or str(lease['gpu']) != str(execution['gpu']) or lease['port'] != execution['port']):
        raise ValueError('Failed diagnostic service identity differs')
    reference_execution = index.load(reference/'gate_execution_receipt.json')
    if reference_execution['reference'] is not True or reference_execution['revision_manifest_sha256'] != index.revision_hash:
        raise ValueError('Failed diagnostic reference execution differs')
    protocol = audited_protocol(identity['source_attempt'], 'pi0', execution['frozen_runtime_protocol'])
    verify_patch(index, session/'server')
    events, _ = cache_events(session/'server')
    if set(events) != set(range(1, 9)):
        raise ValueError('Failed cache gate has missing or additional dispatches')
    comparisons = []
    for request in identity['requests']:
        number = request['index']
        rows = []
        for root, order in ((reference, identity['reference_order']), (session, identity['cached_order'])):
            row = index.load(root/'outputs'/f'{number}.json')
            check_prediction(row, request['condition'], protocol, request['seed'])
            if row['call'] != order.index(number)+1 or row['input_hash'] != request['point']['content_hash']:
                raise ValueError('Failed diagnostic actual order/input differs')
            if row['loaded_code_hashes'] != execution['frozen_runtime_protocol']['scientific_code_hashes']:
                raise ValueError('Failed diagnostic scientific code differs')
            for key in ('noise_path', 'server_prediction_path'):
                path = Path(row[key]).resolve()
                if not path.is_relative_to(reference if root == reference else session/'server'):
                    raise ValueError('Failed diagnostic raw escaped its original server')
                index.record(path)
            with np.load(row['server_prediction_path']) as raw:
                if not np.array_equal(raw['actions'], np.asarray(row['actions'])):
                    raise ValueError('Failed diagnostic decoded actions differ from persisted raw artifact')
            if root == session and row['execution_prompt_cache'] != events[row['call']]:
                raise ValueError('Failed diagnostic dispatch differs from saved output')
            rows.append(row)
        a, b = rows
        decoded = np.abs(np.asarray(a['actions'])-np.asarray(b['actions']))
        with np.load(a['server_prediction_path']) as left, np.load(b['server_prediction_path']) as right:
            raw = np.abs(left['raw_prediction'].astype(float)-right['raw_prediction'].astype(float))
            raw_exact = np.array_equal(left['raw_prediction'], right['raw_prediction'])
        with np.load(a['noise_path']) as left, np.load(b['noise_path']) as right:
            if left.files != right.files or any(not np.array_equal(left[key], right[key]) for key in left.files):
                raise ValueError('Failed diagnostic noise arrays differ')
        comparisons.append(dict(index=number, prompt_length=request['prompt_length'], skip=request['condition']['skip'],
            seed=request['seed'], reference_call=a['call'], cached_call=b['call'],
            same_input=a['input_hash']==b['input_hash'], same_noise=a['noise_hash']==b['noise_hash'],
            same_cache=a['audit']['cache_key']==b['audit']['cache_key'], decoded_max_abs=float(decoded.max()),
            decoded_max_offset=list(np.unravel_index(decoded.argmax(), decoded.shape)),
            raw_max_abs=float(raw.max()), raw_exact=bool(raw_exact)))
    if (report.get('schema') != 1 or report.get('accepted') is not False or report.get('reviewer_model') != 'gpt-6.1-sol'
            or report.get('reasoning_effort') != 'medium' or report['revision_manifest_sha256'] != index.revision_hash
            or report['actual_logical_predictions'] != 16 or report['formal_entries_created'] != 0
            or report['comparisons'] != comparisons or [row['index'] for row in comparisons if not row['raw_exact']] != [2, 4, 6]):
        raise ValueError('Preserved failure report differs from independently recomputed sixteen outputs')
    for path, expected in report['verified_file_hashes'].items():
        if not Path(path).resolve().is_relative_to(revision) or index.record(path) != expected:
            raise ValueError('Preserved failed diagnostic evidence changed')
    for server in (reference, session/'server'):
        native = index.load(server/'native_protocol.json')
        expected_protocol = reference_execution['frozen_runtime_protocol'] if server == reference else execution['frozen_runtime_protocol']
        if native != expected_protocol:
            raise ValueError('Failed diagnostic loaded protocol differs from execution')
        verify_frozen(identity['source_attempt'], native, 'pi0')
        verify_parity(identity['source_attempt'], native, 'pi0')
        rows = [json.loads(line) for line in (server/'prediction_ledger.jsonl').read_text().splitlines()]
        if len(rows) != 8 or {row['call'] for row in rows} != set(range(1, 9)):
            raise ValueError('Failed diagnostic did not perform exactly eight calls on each server')
        outputs = reference/'outputs' if server == reference else session/'outputs'
        by_call = {row['call']: row for row in rows}
        for number in range(8):
            saved = index.load(outputs/f'{number}.json')
            actual = by_call[saved['call']]
            for key in ('input_hash', 'noise_hash', 'raw_prediction_hash', 'seed', 'audit', 'loaded_code_hashes', 'prediction_path'):
                if saved[key] != actual[key]:
                    raise ValueError('Failed diagnostic saved output differs from actual server call')
        index.record(server/'prediction_ledger.jsonl')
    base = Path(identity['source_attempt'])/'readiness/pi0/diagnostic_budget.json'
    total = (index.load(base)['smoke_reserved'] + index.load(Path(identity['source_attempt'])/'parity/pi0.json')['reserved_calls']
        + prior_diagnostic_budget(manifest, 'pi0') + sum(row['reserved_predictions'] for row in index.load(base.with_name('throughput_diagnostic_budget.json'))['entries']))
    found = []
    for row in index.load(base.with_name('prompt_cache_diagnostic_budget.json'))['entries']:
        if row['base_total'] != total or row['reserved_predictions'] != 16:
            raise ValueError('Failed diagnostic budget sequence changed')
        total += 16
        if row['revision_manifest_sha256'] == index.revision_hash:
            found.append(total)
    if found != [report['conservative_after_attempt']] or total > 256:
        raise ValueError('Failed diagnostic reservation missing or over budget')
    index.record(revision/'revision_manifest.json')
    for name, expected in identity['code_hashes'].items():
        if index.record(Path(__file__).with_name(name)) != expected:
            raise ValueError('Failed diagnostic original execution source changed')

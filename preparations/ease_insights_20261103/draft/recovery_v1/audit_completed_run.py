"""CPU-only recovery audit; never launch workers or change scientific outputs.

A complete run can be reconciled into the scheduler using the original receipt.
An unresolved run remains unresolved: this tool cannot manufacture missing timing,
rewrite receipts, append requests, or authorize rerunning a complete native case.
Its ledger has neither manifest.json nor cases.json, so analyze.py cannot discover
it as a second scientific run.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def identity(case):
    return case['suite'], case['task_id'], case['init_id']


def case_key(case):
    return '%s_t%02d_i%02d' % identity(case)


def audit(run, frozen):
    run, frozen = Path(run).resolve(), Path(frozen).resolve()
    manifest = read_json(run / 'manifest.json')
    expected = read_json(run / 'cases.json')
    run_result = read_json(run / 'result.json', {})
    client_complete = read_json(run / 'client_complete.json', {})
    freeze = read_json(frozen)['source_sha256']
    errors = []
    expected_keys = [identity(c) for c in expected]
    if len(expected_keys) != len(set(expected_keys)):
        errors.append('duplicate_frozen_cases')
    if manifest.get('cases') != expected:
        errors.append('run_manifest_case_matrix_mismatch')
    code_checks = {}
    for name, sha in freeze.items():
        path = Path(name)
        actual = digest(path) if path.exists() else None
        recorded = manifest.get('code_sha256', {}).get(name)
        code_checks[name] = dict(expected_sha256=sha, current_sha256=actual,
                                 run_manifest_sha256=recorded, match=actual == sha == recorded)
        if not code_checks[name]['match']:
            errors.append('frozen_runtime_hash_mismatch:' + name)

    request_path = run / 'requests.jsonl'
    request_bytes = request_path.read_bytes() if request_path.exists() else b''
    requests = defaultdict(list)
    malformed = []
    byte_offset = 0
    for line_number, line in enumerate(request_bytes.splitlines(keepends=True), 1):
        try:
            row = json.loads(line)
            key = identity(row)
        except (ValueError, TypeError, KeyError) as exc:
            malformed.append(dict(line=line_number, byte_start=byte_offset,
                                  byte_end=byte_offset + len(line), error=str(exc)))
        else:
            requests[key].append(dict(line=line_number, byte_start=byte_offset,
                                      byte_end=byte_offset + len(line), row=row))
        byte_offset += len(line)
    if malformed:
        errors.append('malformed_request_records')
    if request_bytes and not request_bytes.endswith(b'\n'):
        errors.append('unterminated_request_tail')
    extra = sorted(set(requests) - set(expected_keys))
    if extra:
        errors.append('requests_outside_frozen_cases')

    cases = []
    for case in expected:
        key = case_key(case)
        case_dir = run / 'episodes' / key
        receipt_path = case_dir / 'result.json'
        native_path = case_dir / 'native' / 'episode_log.json'
        receipt = read_json(receipt_path, {})
        episodes = read_json(native_path, [])
        rows = requests[identity(case)]
        records = [r['row'] for r in rows]
        problems = []
        native = episodes[0] if len(episodes) == 1 else {}
        native_complete = bool(native) and native.get('failure_reason') is None
        native_complete = native_complete and (
            native.get('task_suite_name'), native.get('task_id'), native.get('episode_idx')) == identity(case)
        native_complete = native_complete and (
            native.get('executed_action_count') == native.get('trajectory_length'))
        # A valid unsuccessful scientific result must have reached its fixed cap.
        native_complete = native_complete and (native.get('success') is True or
            native.get('executed_action_count') == native.get('max_steps'))
        if not native_complete:
            problems.append('native_episode_not_complete')
        if [r.get('replan_id') for r in records] != list(range(len(records))):
            problems.append('request_ids_not_contiguous')
        if len(records) != native.get('infer_calls'):
            problems.append('native_request_count_mismatch')
        if any(r.get('mode') != manifest['mode'] or r.get('source_suite') != manifest['source_suite'] for r in records):
            problems.append('request_condition_mismatch')
        if manifest['mode'] == 'draft' and any(r.get('route_calls') != dict(full=0, draft=1, verify=0) for r in records):
            problems.append('pure_draft_route_violation')
        if receipt.get('status') == 'complete':
            if any(receipt.get(k) != case[k] for k in ('suite', 'task_id', 'init_id')):
                problems.append('receipt_identity_mismatch')
            if receipt.get('mode') != manifest['mode'] or receipt.get('source_suite') != manifest['source_suite']:
                problems.append('receipt_condition_mismatch')
            if receipt.get('episode') != native:
                problems.append('receipt_native_mismatch')
            if receipt.get('success') != native.get('success'):
                problems.append('receipt_success_mismatch')
            if len(records) != receipt.get('rpc_calls'):
                problems.append('receipt_request_count_mismatch')
            classification = 'complete_original' if not problems else 'complete_receipt_requires_review'
        elif native_complete:
            classification = 'native_complete_missing_receipt_do_not_rerun'
            problems.append('missing_complete_outer_receipt')
        else:
            classification = 'technical_interruption_or_not_started'
            problems.append('missing_complete_outer_receipt')
        if problems:
            errors.append('unresolved_case:' + key)
        cases.append(dict(case, case_key=key, classification=classification,
            success=native.get('success') if native_complete else None,
            request_count=len(records), request_first_line=rows[0]['line'] if rows else None,
            request_last_line=rows[-1]['line'] if rows else None,
            request_byte_start=rows[0]['byte_start'] if rows else None,
            request_byte_end=rows[-1]['byte_end'] if rows else None,
            original_receipt=str(receipt_path), receipt_sha256=digest(receipt_path) if receipt_path.exists() else None,
            original_native_log=str(native_path), native_sha256=digest(native_path) if native_path.exists() else None,
            earliest_native_scientific_result_preserved=bool(native_complete), issues=problems))

    successes = sum(c['success'] is True for c in cases)
    for name, receipt in [('run', run_result), ('client', client_complete)]:
        if receipt.get('status') != 'complete' or receipt.get('cases') != len(expected) or receipt.get('successes') != successes:
            errors.append(name + '_complete_receipt_mismatch')
    if run_result.get('total_cases') != len(expected):
        errors.append('run_total_cases_mismatch')
    complete = not errors
    return dict(schema='draft_recovery_audit_v1', audited_unix=time.time(),
        status='complete_original_verified' if complete else 'requires_explicit_recovery',
        authoritative_run=str(run), frozen_manifest=str(frozen), frozen_runtime=code_checks,
        expected_cases=len(expected), verified_complete_cases=sum(c['classification'] == 'complete_original' for c in cases),
        successes=successes, classifications=dict(Counter(c['classification'] for c in cases)),
        requests=dict(path=str(request_path), sha256=hashlib.sha256(request_bytes).hexdigest(),
            bytes=len(request_bytes), lines=len(request_bytes.splitlines()), valid_records=sum(map(len, requests.values())),
            ends_with_newline=request_bytes.endswith(b'\n'), malformed=malformed, unexpected_case_keys=extra),
        original_run_result=run_result, original_client_complete=client_complete,
        source_receipt_sha256={p.name: digest(p) for p in (run / 'result.json', run / 'client_complete.json') if p.exists()},
        cases=cases, issues=errors,
        next_action='reuse_original_run_receipt_without_gpu' if complete else 'inspect_unresolved_cases_without_overwriting_originals',
        merge_receipt_required=not complete, gpu_launch_performed=False,
        original_outputs_modified=False, duplicate_analysis_run_created=False,
        analysis_roots=[str(run)],
        recovery_rule='Complete native scientific outcomes remain authoritative, including failures. Never rerun them to replace missing outer timing. Incomplete attempts require a separate technical archive and fresh request log; no recovery is launched by this audit.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--frozen-manifest', required=True)
    parser.add_argument('--ledger', required=True)
    args = parser.parse_args()
    run, ledger = Path(args.run).resolve(), Path(args.ledger).resolve()
    if ledger == run or run in ledger.parents:
        parser.error('Ledger must be outside the original scientific run')
    if ledger.name in ('manifest.json', 'cases.json', 'result.json', 'client_complete.json'):
        parser.error('Ledger must not impersonate a runtime or analysis-discovery receipt')
    result = audit(run, args.frozen_manifest)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    temp = ledger.with_name(ledger.name + '.tmp')
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(ledger)
    print(json.dumps({k: result[k] for k in ('status', 'authoritative_run', 'expected_cases', 'verified_complete_cases', 'successes', 'issues', 'next_action')}))
    return 0 if not result['issues'] else 2


if __name__ == '__main__':
    sys.exit(main())

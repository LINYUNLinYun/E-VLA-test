"""Archive a technical attempt on CPU, then resume through the frozen runner.

prepare never starts a process. run requires scheduler allocation and invokes the
unchanged runner with every original case; its client skips complete receipts.
Complete native outcomes lacking an outer receipt are deliberately not retried.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

from audit_completed_run import audit, case_key, digest, identity, read_json


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def verify_files(hashes):
    changed = [p for p, sha in hashes.items() if not Path(p).is_file() or digest(Path(p)) != sha]
    if changed:
        raise RuntimeError('Protected evidence changed: ' + repr(changed))


def assert_processes_stopped(run):
    for key, pid in read_json(run / 'processes.json', {}).items():
        if key.endswith('_pid') and Path('/proc', str(pid)).exists():
            raise RuntimeError('Original process still exists; inspect before recovery: %s=%s' % (key, pid))


def archive_file(source, destination, expected_sha=None):
    expected_sha = expected_sha or digest(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    if digest(destination) != expected_sha:
        raise RuntimeError('Archive hash mismatch: ' + str(destination))
    return expected_sha


def filtered_requests(raw, retained_cases, expected_cases):
    """Retain original complete-case bytes verbatim, quarantine technical records."""
    kept, excluded = [], []
    lines = raw.splitlines(keepends=True)
    offset = 0
    for number, line in enumerate(lines, 1):
        try:
            row = json.loads(line)
            key = case_key(row)
        except (ValueError, KeyError, TypeError):
            if number != len(lines):
                raise RuntimeError('Malformed non-tail request requires manual inspection')
            key = None
        if key is not None and key not in expected_cases:
            raise RuntimeError('Unexpected scientific case in request log: ' + key)
        if key in retained_cases:
            if not line.endswith(b'\n'):
                raise RuntimeError('Complete-case request lacks newline; cannot append without changing it')
            kept.append(line)
        else:
            excluded.append(dict(line=number, byte_start=offset, byte_end=offset + len(line), case_key=key))
        offset += len(line)
    return b''.join(kept), excluded


def technical_receipts(directory):
    records = []
    for path in sorted(directory.rglob('result.json')):
        receipt = read_json(path, {})
        if receipt.get('status') == 'technical_failure':
            records.append(dict(case_key=case_key(receipt), receipt=str(path),
                receipt_sha256=digest(path), wall_seconds=receipt.get('wall_seconds', 0.)))
    return records


def attempt_entry(number, result_path, manifest_path, completed_cases, technical_cases):
    receipt, manifest = read_json(result_path), read_json(manifest_path)
    if 'recovery' in receipt:
        raise RuntimeError('A cumulative receipt cannot be counted as a raw service attempt')
    return dict(attempt=number, result=str(result_path), result_sha256=digest(result_path),
        manifest=str(manifest_path), manifest_sha256=digest(manifest_path), gpu=manifest.get('gpu'),
        status=receipt['status'], wall_seconds=receipt.get('wall_seconds', 0.),
        preparation_seconds=receipt.get('preparation_seconds', 0.),
        closed_loop_seconds=post_ready_seconds(receipt), completed_case_keys=completed_cases,
        technical_cases=technical_cases, technical_case_wall_seconds=sum(r['wall_seconds'] for r in technical_cases))


def previous_history(plan):
    """Flatten legacy v1 provenance once; later recoveries retain immutable leaves."""
    old, manifest, archive = plan['original_result'], plan['original_manifest'], Path(plan['archive'])
    provenance = manifest.get('recovery_provenance', {})
    origins = {key: provenance.get('case_service_attempt', {}).get(key, 0)
               for key in plan['retained_case_keys']}
    history = old.get('recovery', {}).get('attempt_history')
    if history is None:
        prior = provenance.get('attempts') or [dict(attempt=0, manifest=str(archive / 'attempt0/manifest.json.saved'))]
        history = []
        for index, item in enumerate(prior):
            manifest_path = Path(item['manifest'])
            # Legacy v1 stores the first failed episode beside attempt0. The
            # most recent failed episode is archived by this preparation.
            technical = technical_receipts(archive / 'technical_episodes' if index == len(prior) - 1
                                            else manifest_path.parent.parent / 'technical_episodes')
            history.append(attempt_entry(item['attempt'], manifest_path.with_name('result.json.saved'),
                manifest_path, [key for key, origin in origins.items() if origin == item['attempt']], technical))
    for item in history:
        verify_files({item['result']: item['result_sha256'], item['manifest']: item['manifest_sha256']})
        verify_files({r['receipt']: r['receipt_sha256'] for r in item['technical_cases']})
    ids = {item['attempt'] for item in history}
    if len(ids) != len(history) or not set(origins.values()).issubset(ids):
        raise RuntimeError('Invalid historical case-to-service mapping')
    for field in ('wall_seconds', 'preparation_seconds'):
        if abs(sum(item[field] for item in history) - old.get(field, 0.)) > 1e-6:
            raise RuntimeError('Historical raw attempts do not reconcile to cumulative ' + field)
    return history, origins


def prepare(run, frozen, ledger):
    run, frozen, ledger = map(lambda p: Path(p).resolve(), (run, frozen, ledger))
    if ledger == run or run in ledger.parents:
        raise RuntimeError('Recovery ledger and archive must be outside original run')
    plan = read_json(ledger)
    if plan is not None:
        if plan['run'] != str(run) or plan['frozen_manifest'] != str(frozen):
            raise RuntimeError('Ledger belongs to another recovery')
        verify_files(plan['protected_case_files_sha256'])
        if plan['status'] == 'prepared':
            if (run / 'result.json').exists():
                raise RuntimeError('Prepared run already has a result; do not prepare over a started attempt')
            if digest(run / 'requests.jsonl') != plan['retained_request_sha256']:
                raise RuntimeError('Prepared request log changed')
            return plan
        if plan['status'] != 'preparing':
            raise RuntimeError('Recovery has already started: ' + plan['status'])
    else:
        assert_processes_stopped(run)
        original_audit = audit(run, frozen)
        original_result = read_json(run / 'result.json', {})
        if original_result.get('status') != 'technical_failure':
            raise RuntimeError('Only an inspected technical_failure may be prepared')
        complete = [c for c in original_audit['cases'] if c['classification'] == 'complete_original']
        pending = [c for c in original_audit['cases'] if c['classification'] == 'technical_interruption_or_not_started']
        if len(complete) + len(pending) != original_audit['expected_cases'] or not pending:
            raise RuntimeError('Unresolved complete native/receipt evidence cannot be rerun')
        allowed = {'run_complete_receipt_mismatch', 'client_complete_receipt_mismatch',
                   'malformed_request_records', 'unterminated_request_tail'}
        allowed.update('unresolved_case:' + c['case_key'] for c in pending)
        if any(issue not in allowed for issue in original_audit['issues']):
            raise RuntimeError('Audit requires inspection: ' + repr(original_audit['issues']))
        protected = {str(p): digest(p) for c in complete
                     for p in (run / 'episodes' / c['case_key']).rglob('*') if p.is_file()}
        technical_files = {str(p.relative_to(run / 'episodes')): digest(p) for c in pending
                           for p in (run / 'episodes' / c['case_key']).rglob('*') if p.is_file()}
        raw = (run / 'requests.jsonl').read_bytes()
        retained, excluded = filtered_requests(raw, {c['case_key'] for c in complete},
                                               {c['case_key'] for c in original_audit['cases']})
        archive = ledger.parent / (ledger.stem + '_archive')
        if archive.exists():
            raise RuntimeError('Archive already exists without its preparation ledger')
        controls = {p.name: digest(p) for p in run.iterdir() if p.is_file()}
        plan = dict(schema='draft_resume_v1', status='preparing', prepared_unix=time.time(),
            run=str(run), frozen_manifest=str(frozen), archive=str(archive), original_audit=original_audit,
            original_result=original_result, original_manifest=read_json(run / 'manifest.json'),
            retained_case_keys=[c['case_key'] for c in complete], pending_case_keys=[c['case_key'] for c in pending],
            protected_case_files_sha256=protected, original_control_sha256=controls,
            technical_case_files_sha256=technical_files,
            retained_request_sha256=hashlib.sha256(retained).hexdigest(), retained_request_bytes=len(retained),
            excluded_request_records=excluded, archived_technical_cases=[],
            gpu_launch_performed=False, original_complete_cases_modified=False)
        write_json(ledger, plan)

    archive = Path(plan['archive'])
    for name, sha in plan['original_control_sha256'].items():
        archive_file(run / name, archive / 'attempt0' / (name + '.saved'), sha)
    artifacts_manifest = run / 'artifacts/manifest.json'
    if artifacts_manifest.exists():
        archive_file(artifacts_manifest, archive / 'attempt0/artifacts_manifest.json.saved')
    raw = (archive / 'attempt0/requests.jsonl.saved').read_bytes()
    retained, excluded = filtered_requests(raw, set(plan['retained_case_keys']),
        {case_key(c) for c in plan['original_manifest']['cases']})
    if hashlib.sha256(retained).hexdigest() != plan['retained_request_sha256']:
        raise RuntimeError('Retained request plan mismatch')
    for key in plan['pending_case_keys']:
        source, destination = run / 'episodes' / key, archive / 'technical_episodes' / key
        if source.exists():
            if destination.exists():
                raise RuntimeError('Both original and archived technical case exist: ' + key)
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
        if destination.exists() and key not in plan['archived_technical_cases']:
            plan['archived_technical_cases'].append(key)
    verify_files({str(archive / 'technical_episodes' / p): sha
                  for p, sha in plan['technical_case_files_sha256'].items()})
    if (run / 'requests.jsonl').read_bytes() != retained:
        temp = run / 'requests.jsonl.recovery_tmp'
        temp.write_bytes(retained)
        temp.replace(run / 'requests.jsonl')
    original_result = run / 'result.json'
    if original_result.exists():
        if digest(original_result) != plan['original_control_sha256']['result.json']:
            raise RuntimeError('Original root receipt changed during preparation')
        original_result.unlink()  # The byte-identical .saved archive was verified above.
    verify_files(plan['protected_case_files_sha256'])
    history, origins = previous_history(plan)
    plan.update(status='prepared', prepared_completed_unix=time.time(),
        scientific_cases_retained=len(plan['retained_case_keys']), scientific_cases_to_execute=len(plan['pending_case_keys']),
        prior_attempt_history=history, retained_case_service_attempt=origins,
        next_service_attempt=max(item['attempt'] for item in history) + 1,
        prior_total_technical_case_wall_seconds=sum(item['technical_case_wall_seconds'] for item in history))
    write_json(ledger, plan)
    return plan


def post_ready_seconds(receipt):
    if receipt.get('closed_loop_seconds') is not None:
        return float(receipt['closed_loop_seconds'])
    return max(0., receipt.get('completed_unix', receipt['started_unix']) - receipt['started_unix']
               - receipt.get('preparation_seconds', 0.))


def combined_receipt(plan, attempt, case_audit, attempt_manifest_path, history=None):
    old = plan['original_result']
    prior = plan['prior_attempt_history']
    history = history or prior
    failed_case_wall = sum(item['technical_case_wall_seconds'] for item in prior)
    result = dict(attempt)
    result.update(started_unix=old['started_unix'],
        preparation_seconds=old.get('preparation_seconds', 0.) + attempt.get('preparation_seconds', 0.),
        wall_seconds=old.get('wall_seconds', 0.) + attempt.get('wall_seconds', 0.),
        closed_loop_seconds=post_ready_seconds(old) + post_ready_seconds(attempt),
        recovery=dict(schema='draft_resume_v1', retained_cases=len(plan['retained_case_keys']),
            new_complete_cases=case_audit['verified_complete_cases'] - len(plan['retained_case_keys']),
            latest_attempt_result=str(Path(plan['archive']) / 'attempt1/result.json.saved'),
            latest_attempt_manifest=str(attempt_manifest_path),
            first_attempt_result=prior[0]['result'], first_attempt_manifest=prior[0]['manifest'],
            previous_aggregate_result=str(Path(plan['archive']) / 'attempt0/result.json.saved'),
            previous_aggregate_manifest=str(Path(plan['archive']) / 'attempt0/manifest.json.saved'),
            first_attempt_wall_seconds=prior[0]['wall_seconds'],
            latest_attempt_wall_seconds=attempt.get('wall_seconds'),
            prior_technical_case_wall_seconds=failed_case_wall,
            total_technical_case_wall_seconds=sum(item['technical_case_wall_seconds'] for item in history),
            attempt_history=history,
            prior_failed_attempt_wall_includes_valid_cases=True,
            active_wall_seconds=old.get('wall_seconds', 0.) + attempt.get('wall_seconds', 0.),
            calendar_elapsed_seconds=attempt.get('completed_unix', time.time()) - old['started_unix'],
            wall_definition='Sum of all active attempt wall times, including preparation and teardown; excludes waiting between attempts.',
            closed_loop_definition='Sum of post-readiness active intervals, including technical failure and client exit detection; excludes queue waiting.',
            technical_cost_definition='Explicit failed case wall is separately measured; the prior failed attempt also contains 36 valid scientific cases.',
            gpu_history=[item['gpu'] for item in history],
            preserved_original_case_hashes=True, preserved_request_prefix_bytes=plan['retained_request_bytes']))
    return result


def run_prepared(ledger, allocation):
    ledger = Path(ledger).resolve()
    plan = read_json(ledger)
    if not plan or plan['status'] != 'prepared':
        raise RuntimeError('Run requires a prepared, unused recovery ledger')
    run, archive = Path(plan['run']), Path(plan['archive'])
    assert_processes_stopped(run)
    verify_files(plan['protected_case_files_sha256'])
    verify_files({p: c['expected_sha256'] for p, c in plan['original_audit']['frozen_runtime'].items()})
    if (run / 'result.json').exists() or digest(run / 'requests.jsonl') != plan['retained_request_sha256']:
        raise RuntimeError('Prepared root receipt/request state changed')
    runner_path = next(Path(p) for p in plan['original_audit']['frozen_runtime'] if Path(p).name == 'run.py')
    sys.path.insert(0, str(runner_path.parent))
    spec = importlib.util.spec_from_file_location('frozen_draft_resume_runner', runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    original_atomic = runner.atomic_json
    raw_result_path = archive / 'attempt1/result.json.saved'

    def intercept(path, value):
        if Path(path).resolve() == run / 'result.json':
            write_json(raw_result_path, value)
            if value.get('status') == 'complete':
                value = dict(value, status='recovery_pending_validation')
        original_atomic(path, value)

    runner.atomic_json = intercept
    args = [str(runner_path), '--output', str(run), '--cases', str(run / 'cases.json'),
            '--mode', plan['original_manifest']['mode'], '--source-suite', plan['original_manifest']['source_suite']]
    for key, value in vars(allocation).items():
        if key in ('gpu', 'gpu_uuid', 'egl_index', 'port', 'ready_timeout') and value is not None:
            args += ['--' + key.replace('_', '-'), str(value)]
    plan.update(status='running', resumed_unix=time.time(), wrapper_pid=os.getpid(),
                resume_command=args, gpu_launch_performed=True)
    write_json(ledger, plan)
    prior_argv, error = sys.argv, None
    try:
        sys.argv = args
        runner.main()
    except BaseException:
        error = traceback.format_exc()
    finally:
        sys.argv = prior_argv
    attempt = read_json(raw_result_path, {})
    try:
        verify_files(plan['protected_case_files_sha256'])
        prefix = (run / 'requests.jsonl').read_bytes()[:plan['retained_request_bytes']]
        if hashlib.sha256(prefix).hexdigest() != plan['retained_request_sha256']:
            raise RuntimeError('Original complete-case request prefix changed')
        for name in ('manifest.json', 'cases.json', 'render_identity.json', 'processes.json',
                     'client_complete.json', 'client.log', 'server.log', 'progress.json'):
            path = run / name
            if path.exists():
                archive_file(path, archive / 'attempt1' / (name + '.saved'))
        attempt_manifest_path = archive / 'attempt1/manifest.json.saved'
        manifest = read_json(run / 'manifest.json')
        if manifest['cases'] != plan['original_manifest']['cases'] or manifest['protocol'] != plan['original_manifest']['protocol']:
            raise RuntimeError('Resumed science protocol/cases changed')
        case_audit = audit(run, plan['frozen_manifest'])
        if attempt.get('status') == 'complete' and any(issue != 'run_complete_receipt_mismatch' for issue in case_audit['issues']):
            raise RuntimeError('Resume completion failed audit: ' + repr(case_audit['issues']))
        if not attempt or 'started_unix' not in attempt:
            raise RuntimeError('Frozen runner produced no attempt receipt')
        new_complete = [c['case_key'] for c in case_audit['cases'] if c['classification'] == 'complete_original'
                        and c['case_key'] not in plan['retained_case_keys']]
        latest_technical = []
        for key in plan['pending_case_keys']:
            path = run / 'episodes' / key / 'result.json'
            receipt = read_json(path, {})
            if receipt.get('status') == 'technical_failure':
                saved = archive / 'attempt1/technical_receipts' / (key + '.json.saved')
                archive_file(path, saved)
                latest_technical.append(dict(case_key=key, receipt=str(saved), receipt_sha256=digest(saved),
                                             wall_seconds=receipt.get('wall_seconds', 0.)))
        entry = attempt_entry(plan['next_service_attempt'], raw_result_path, attempt_manifest_path,
                              new_complete, latest_technical)
        history = plan['prior_attempt_history'] + [entry]
        manifest['recovery_provenance'] = dict(
            case_service_attempt=plan['retained_case_service_attempt'] |
                                 {key: plan['next_service_attempt'] for key in new_complete},
            attempts=[{k: item[k] for k in ('attempt', 'gpu', 'manifest', 'manifest_sha256', 'result', 'result_sha256')}
                      for item in history],
            incomplete_case_keys=[key for key in plan['pending_case_keys'] if key not in new_complete],
            gpu_field_definition='Top-level GPU is latest attempt; case_service_attempt preserves individual provenance.')
        write_json(run / 'manifest.json', manifest)
        result = combined_receipt(plan, attempt, case_audit, attempt_manifest_path, history)
        if error:
            result.update(status='technical_failure', recovery_wrapper_error=error)
        write_json(run / 'result.json', result)
        final_issues = [issue for issue in case_audit['issues'] if issue != 'run_complete_receipt_mismatch']
        plan.update(status=result['status'], completed_unix=time.time(), final_result=result,
                    final_verified_cases=case_audit['verified_complete_cases'], validation_issues=final_issues)
        write_json(ledger, plan)
    except BaseException:
        error = (error or '') + '\n' + traceback.format_exc()
        plan.update(status='recovery_validation_failure', completed_unix=time.time(), error=error)
        write_json(ledger, plan)
        if 'started_unix' not in attempt:
            attempt = dict(status='technical_failure', started_unix=plan['resumed_unix'],
                completed_unix=time.time(), wall_seconds=time.time() - plan['resumed_unix'],
                total_cases=len(plan['original_manifest']['cases']), mode=plan['original_manifest']['mode'],
                source_suite=plan['original_manifest']['source_suite'], output=str(run))
            write_json(raw_result_path, attempt)
        failure = combined_receipt(plan, attempt, dict(verified_complete_cases=len(plan['retained_case_keys'])),
                                   archive / 'attempt1/manifest.json.saved')
        failure.update(status='technical_failure', recovery_wrapper_error=error)
        write_json(run / 'result.json', failure)
    if error:
        raise RuntimeError(error)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--run', required=True)
    prep.add_argument('--frozen-manifest', required=True)
    prep.add_argument('--ledger', required=True)
    launch = sub.add_parser('run')
    launch.add_argument('--ledger', required=True)
    launch.add_argument('--gpu')
    launch.add_argument('--gpu-uuid')
    launch.add_argument('--egl-index', type=int)
    launch.add_argument('--port', type=int)
    launch.add_argument('--ready-timeout', type=int, default=900)
    args = parser.parse_args()
    if args.command == 'prepare':
        plan = prepare(args.run, args.frozen_manifest, args.ledger)
    else:
        if not (args.gpu or args.gpu_uuid):
            parser.error('Scheduler-verified GPU allocation is required')
        plan = run_prepared(args.ledger, args)
    print(json.dumps({key: plan.get(key) for key in ('status', 'run', 'retained_case_keys', 'pending_case_keys', 'final_verified_cases')}))


if __name__ == '__main__':
    main()

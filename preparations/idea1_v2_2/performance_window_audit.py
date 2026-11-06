"""Read-only verification of immutable throughput window comparisons; no sampling."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import shlex

from common import sha, planned


def read(path):
    return json.loads(Path(path).read_text())


def unique_rows(rows):
    if len(rows) != len({row['key'] for row in rows}) or len(rows) != len({row['token'] for row in rows}):
        raise ValueError('Duplicate comparison record key/token')


def verify_time(data, receipts):
    for label, receipt in zip(('baseline', 'endpoint'), receipts):
        if data[label]['snapshot_read_completed_utc'] != receipt['read_completed_utc']:
            raise ValueError('Comparison timestamp differs from immutable snapshot receipt')
    elapsed = (datetime.fromisoformat(receipts[1]['read_completed_utc'])-datetime.fromisoformat(receipts[0]['read_completed_utc'])).total_seconds()
    if elapsed != data['elapsed_s'] or elapsed <= 0:
        raise ValueError('Window elapsed time differs')
    return elapsed


def verify_rates(data, manifest, endpoint, elapsed):
    targets = {}
    for row in planned(manifest):
        targets[row['model']] = targets.get(row['model'], 0)+1
    def number(actual, expected):
        if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Observed rate or remaining ETA differs')
    etas = {}
    for model, count in data['per_model'].items():
        valid = sum(row['status'] == 'valid' and row['model'] == model for row in endpoint['entries'])
        reserved = sum(row['status'] == 'reserved' and row['model'] == model for row in endpoint['entries'])
        remaining = targets[model]-valid
        if (count['expected_episodes'] != targets[model] or count['full_scope_valid_at_endpoint'] != valid
                or count['full_scope_reserved_at_endpoint'] != reserved or count['remaining_not_yet_valid'] != remaining):
            raise ValueError('Manifest target or full-scope remaining count differs')
        for source, target, scale in [('new_valid', 'observed_valid_per_hour', 3600), ('new_actions', 'observed_actions_per_second', 1), ('new_predictions', 'observed_predictions_per_second', 1)]:
            number(count[target], count[source]/elapsed*scale)
        etas[model] = remaining/(count['new_valid']/elapsed*3600)
        number(count['projected_hours_remaining_at_observed_rate'], etas[model])
    for source, target, scale in [('new_valid', 'observed_valid_per_hour', 3600), ('new_actions', 'observed_actions_per_second', 1), ('new_predictions', 'observed_predictions_per_second', 1)]:
        number(data['aggregate'][target], data['aggregate'][source]/elapsed*scale)
    number(data['aggregate']['max_projected_remaining_hours'], max(etas.values()))
    if data['aggregate']['bottleneck_model_by_projected_remaining_hours'] != max(etas, key=etas.get):
        raise ValueError('Bottleneck model differs')


def check_slots(model, counts, rows, sessions):
    mapping = sessions['client_pids']
    if set(mapping) != {'0', '1'} or set(counts['slots']) != {'0', '1'}:
        raise ValueError('Slot keys must be 0/1, never client PIDs')
    if counts['client_pids'] != mapping or len(set(mapping.values())) != 2:
        raise ValueError('Client PID mapping changed or is inverted')
    for slot, pid in mapping.items():
        bound = counts['slots'][slot]
        selected = [row for row in rows if row['slot'] == slot]
        if bound['pid'] != pid or any(row['client_pid'] != pid for row in selected):
            raise ValueError('Slot/PID identity mismatch')
        for target, field in [('new_valid', None), ('actions', 'actions'), ('predictions', 'predictions'), ('frames', 'frames')]:
            expected = len(selected) if field is None else sum(row[field] for row in selected)
            if bound[target] != expected:
                raise ValueError('Slot increment does not match bound records')
        if (len(bound['new_keys']) != len(set(bound['new_keys'])) or len(bound['new_tokens']) != len(set(bound['new_tokens']))
                or set(bound['new_keys']) != {row['key'] for row in selected} or set(bound['new_tokens']) != {row['token'] for row in selected}):
            raise ValueError('Slot key/token set differs')
    for field, total in [('new_valid', 'new_valid'), ('actions', 'new_actions'), ('predictions', 'new_predictions'), ('frames', 'new_frames')]:
        if sum(row[field] for row in counts['slots'].values()) != counts[total]:
            raise ValueError(f'{model} slot sum differs from model total')


def verify(baseline, endpoint, comparison):
    baseline, endpoint = Path(baseline).resolve(), Path(endpoint).resolve()
    comparison = Path(comparison).resolve()
    data = read(comparison)
    files = {str(comparison): sha(comparison)}
    snapshots, metrics, bindings, receipts = [], [], [], []
    for directory, label in ((baseline, 'baseline'), (endpoint, 'endpoint')):
        receipt = read(directory/'snapshot_receipt.json')
        receipts.append(receipt)
        for name in ('episode_ledger.snapshot.json', 'derived_metrics.json', 'result_metric_bindings.json', 'snapshot_receipt.json', 'result_metric_bindings_receipt.json'):
            files[str(directory/name)] = sha(directory/name)
        if receipt['sha256'] != sha(directory/'episode_ledger.snapshot.json') or data[label]['snapshot_sha256'] != receipt['sha256']:
            raise ValueError('Immutable snapshot hash differs')
        for name, key in [('derived_metrics.json', 'derived_metrics'), ('result_metric_bindings.json', 'result_metric_bindings')]:
            if data[label][key+'_sha256'] != sha(directory/name) or Path(data[label][key+'_path']).resolve() != directory/name:
                raise ValueError('Comparison source binding differs')
        br = read(directory/'result_metric_bindings_receipt.json')
        b = read(directory/'result_metric_bindings.json')
        if br['sha256'] != sha(directory/'result_metric_bindings.json') or b['ledger_snapshot_sha256'] != receipt['sha256']:
            raise ValueError('Result metric binding receipt differs')
        l = read(directory/'episode_ledger.snapshot.json')
        entries = {(row['key'], row['token']): row for row in l['entries']}
        if len(entries) != len(l['entries']) or len({row['key'] for row in b['records']}) != len(b['records']):
            raise ValueError('Duplicate snapshot or result key')
        m = read(directory/'derived_metrics.json')['active_two_env_sessions']
        for row in b['records']:
            entry = entries[row['key'], row['token']]
            session = m[row['model']]
            pid = session['client_pids'][row['slot']]
            argv = shlex.split(session['process_identity'][str(pid)]['cmdline'])
            if (entry['status'] != 'valid' or entry['model'] != row['model'] or str(entry['env_slot']) != row['slot']
                    or entry['pid'] != row['client_pid'] or row['client_pid'] != pid
                    or Path(entry['throughput_session']).name != session['session_id']
                    or argv[argv.index('--slot')+1] != row['slot'] or Path(argv[argv.index('--session')+1]).name != session['session_id']
                    or row['ledger_validated_sha256'] != entry['validated_sha256']):
                raise ValueError('Binding does not match actual ledger slot/PID/session')
        snapshots.append(l); metrics.append(m); bindings.append(b)
    elapsed = verify_time(data, receipts)
    old_valid = {row['key'] for row in snapshots[0]['entries'] if row['status'] == 'valid'}
    if not old_valid <= {row['key'] for row in snapshots[1]['entries'] if row['status'] == 'valid'}:
        raise ValueError('Baseline valid keys were downgraded or removed')
    new_entries = {row['key']: row for row in snapshots[1]['entries'] if row['status'] == 'valid' and row['key'] not in old_valid}
    new_rows = [row for row in bindings[1]['records'] if row['key'] in new_entries]
    if {row['key'] for row in new_rows} != set(new_entries) or data['new_valid_total'] != len(new_rows):
        raise ValueError('Comparison misses or duplicates newly valid snapshot entries')
    def identities(rows, model=None):
        fields = ('key', 'token', 'slot', 'client_pid', 'actions', 'frames', 'predictions', 'wall_s', 'record_io_s',
                  'result_json_sha256', 'condition', 'suite', 'task', 'init', 'success')
        return {(row.get('model', model), *(row[field] for field in fields)) for row in rows}
    unique_rows(data['new_valid_records_flat'])
    if identities(data['new_valid_records_flat']) != identities(new_rows):
        raise ValueError('Comparison flat record set differs')
    analysis = {}
    for model, current in metrics[1].items():
        prior = metrics[0][model]
        for field in ('client_pids', 'session_id', 'manager_pid', 'server_pid'):
            if current[field] != prior[field] or data['per_model'][model][field] != current[field]:
                raise ValueError('Window changed active process/session identity')
        for pid in current['client_pids'].values():
            if current['process_identity'][str(pid)]['start_ticks'] != prior['process_identity'][str(pid)]['start_ticks']:
                raise ValueError('Client PID was reused across window')
        rows = [row for row in new_rows if row['model'] == model]
        counts = data['per_model'][model]
        unique_rows(counts['new_valid_records'])
        check_slots(model, counts, rows, current)
        if identities(counts['new_valid_records'], model) != identities(rows):
            raise ValueError('Per-model record set differs')
        analysis[model] = {}
        for slot, pid in current['client_pids'].items():
            selected = [row for row in rows if row['slot'] == slot]
            wall = sum(row['wall_s'] for row in selected)
            io = sum(row['record_io_s'] for row in selected)
            analysis[model][slot] = dict(pid=pid, session=current['session_id'], new_valid=len(selected),
                sum_result_wall_s=wall, sum_record_io_s=io, elapsed_s=elapsed,
                completion_attributed_residual_s=elapsed-wall,
                completion_attributed_wall_fraction=wall/elapsed)
    for row in new_rows:
        path = Path(row['result_path'])
        if sha(path) != row['result_json_sha256']:
            raise ValueError('Bound small result.json changed')
        result = read(path)
        for field in ('actions', 'predictions', 'frames', 'wall_s', 'record_io_s', 'model', 'suite', 'task', 'init', 'success'):
            if result[field] != row[field]:
                raise ValueError('Result metric differs from original small result')
        files[str(path)] = row['result_json_sha256']
    for field, target in [('actions', 'new_actions'), ('predictions', 'new_predictions'), ('frames', 'new_frames')]:
        if sum(row[field] for row in new_rows) != data['aggregate'][target]:
            raise ValueError('Aggregate increment differs')
    if len(new_rows) != data['aggregate']['new_valid']:
        raise ValueError('Aggregate valid count differs')
    manifest_path = endpoint.parent.parent/'condition_manifest.json'
    if Path(data['condition_manifest']['path']).resolve() != manifest_path or sha(manifest_path) != data['condition_manifest']['file_sha256']:
        raise ValueError('Comparison condition manifest binding differs')
    files[str(manifest_path)] = sha(manifest_path)
    verify_rates(data, read(manifest_path), snapshots[1], elapsed)
    return dict(schema=1, comparison_sha256=sha(comparison), verified_file_hashes=files,
        new_valid=len(new_rows), elapsed_s=elapsed, slots=analysis,
        caveats=['Completion-attributed whole-episode durations include pre-baseline carry-in and omit endpoint in-flight work; residual is not measured idle time.',
                 'record_io_s is included within result wall_s, not additive. Result wall excludes post-result validation and ledger commit.',
                 'Residual can include reset/setup, validation/commit/reservation, scheduling/queueing and boundary effects; no stack or lock evidence proves NFS/global-lock causation.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--endpoint', required=True)
    parser.add_argument('--comparison', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.baseline, args.endpoint, args.comparison), indent=2))

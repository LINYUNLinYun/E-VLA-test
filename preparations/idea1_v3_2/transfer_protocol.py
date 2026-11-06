"""Immutable preregistration and complete-key rules for zero-shot E4 transfer."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODES = ('full_d10', 'v14_d10', 'draft_goal', 'draft_object', 'draft_spatial', 'draft_10')
ENVIRONMENTS = ('calvin', 'simpler', 'pusht')
CAPS = {'calvin': 360, 'simpler': 80, 'pusht': 300}
CASE_COUNTS = {'calvin': 8, 'simpler': 5, 'pusht': 5}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x') as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    else:
        temp = path.with_name(path.name+'.temporary')
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)); temp.replace(path)


def root_path(value):
    path = Path(value).resolve()
    if path.parent != ROOT/'results/idea1_v3_2' or not path.name.startswith('e4_transfer_'):
        raise ValueError('Use a distinct results/idea1_v3_2/e4_transfer_<UTC> root')
    return path


def cases():
    return [dict(environment=env, case=i, seed=i, max_steps=CAPS[env])
            for env in ENVIRONMENTS for i in range(CASE_COUNTS[env])]


def key(environment, case, mode):
    if environment not in ENVIRONMENTS or mode not in MODES or not 0 <= int(case) < CASE_COUNTS[environment]:
        raise ValueError('Case/mode is outside frozen protocol')
    return f'{environment}/case{int(case):02d}/{mode}'


def load_manifest(path, verify_sources=True):
    m = json.loads(Path(path).read_text())
    payload = {k: v for k, v in m.items() if k != 'manifest_hash'}
    if digest(payload) != m.get('manifest_hash'):
        raise ValueError('Immutable manifest hash mismatch')
    if verify_sources:
        for filename, expected in m['source_files'].items():
            if sha(ROOT/filename) != expected:
                raise ValueError('Frozen source changed: '+filename)
        for filename, expected in m['receipt_files'].items():
            if sha(filename) != expected:
                raise ValueError('Frozen preflight receipt changed: '+filename)
    return m


def complete_result(result, manifest, case, mode, gpu_uuid=None):
    """Only matching, actual completed native episodes are resumable."""
    return (result.get('status') == 'complete' and isinstance(result.get('success'), bool)
            and result.get('environment') == case['environment'] and int(result.get('case', -1)) == case['case']
            and result.get('mode') == mode and result.get('manifest_hash') == manifest['manifest_hash']
            and (gpu_uuid is None or result.get('gpu_uuid') == gpu_uuid))


def aggregate_environments(paths, destination):
    """Combine final per-environment bindings without dropping per-card proof."""
    result = dict(status='complete', environments={}, case_lists={}, source_files={},
                  consumed_receipts={}, binding_receipts={})
    for filename in paths:
        path = Path(filename).resolve(); row = json.loads(path.read_text())
        if row.get('status') != 'complete':
            raise ValueError('Environment preflight not complete: '+str(path))
        result['consumed_receipts'][str(path)] = sha(path)
        for code, expected in row.get('source_files', {}).items():
            old = result['source_files'].get(code)
            if old is not None and old != expected:
                raise ValueError('Preflight sources differ between environment receipts')
            result['source_files'][code] = expected
        for name, status in row.get('environments', {}).items():
            result['binding_receipts'].setdefault(name, []).append(dict(path=str(path), receipt=row))
            if name in result['environments'] and result['environments'][name]['status'] != status['status']:
                raise ValueError('Conflicting support statuses; resolve explicit binding before freeze')
            result['environments'][name] = status
            if row.get('cases'):
                if name in result['case_lists'] and result['case_lists'][name] != row['cases']:
                    raise ValueError('Native cases differ between bindings')
                result['case_lists'][name] = row['cases']
    if set(result['environments']) != set(ENVIRONMENTS):
        raise ValueError('All three environments need explicit support evidence')
    write_json(destination, result, exclusive=True)
    return result


def freeze(output, environment_receipt, policy_receipt, source_reference, gpus):
    output = root_path(output); output.mkdir(parents=True, exist_ok=True)
    if not 1 <= len(gpus) <= 2 or len(set(gpus)) != len(gpus):
        raise ValueError('One or two distinct explicitly assigned GPU shards are required')
    env = json.loads(Path(environment_receipt).read_text())
    policy = json.loads(Path(policy_receipt).read_text())
    source = json.loads(Path(source_reference).read_text())
    if env.get('status') != 'complete' or policy.get('status') != 'complete':
        raise ValueError('Completed environment bindings and policy preflight are required before freeze')
    if not env.get('environments'):
        raise ValueError('Environment preflight must explicitly inventory supported/unsupported environments')
    supported = {}
    for name in ENVIRONMENTS:
        row = env['environments'].get(name)
        if row is None or row.get('status') not in ('supported', 'unsupported'):
            raise ValueError('Missing explicit environment support status: '+name)
        if row['status'] == 'supported' and not row.get('success_oracle'):
            raise ValueError('Native success oracle not specified: '+name)
        supported[name] = row
    if not policy.get('checkpoint_metadata') or not policy.get('prompt_proof'):
        raise ValueError('Checkpoint metadata and new-instruction prompt proof must precede freeze')
    if tuple(policy.get('modes', [])) != MODES:
        raise ValueError('Policy must expose all six frozen conditions')
    if not source:
        raise ValueError('Real source LIBERO reference missing')
    files = sorted((ROOT/'preparations/idea1_v3_2').glob('transfer*.py'))
    source_files = {str(p.relative_to(ROOT)): sha(p) for p in files}
    for receipt in (env, policy):
        for field in ('source_files', 'source_sha256'):
            for filename, expected in receipt.get(field, {}).items():
                p = Path(filename)
                if not p.is_absolute():
                    p = ROOT/p
                if sha(p) != expected:
                    raise ValueError('Preflight dependency source changed: '+str(p))
                source_files[str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)] = expected
    case_rows = cases()
    for i, row in enumerate(case_rows):
        native_cases = env.get('case_lists', {}).get(row['environment'])
        if native_cases is None or len(native_cases) != CASE_COUNTS[row['environment']]:
            raise ValueError('All native case instructions/caps must precede freeze: '+row['environment'])
        native = next((c for c in native_cases if c.get('case') == row['case']), None)
        if native is None or not all(k in native for k in ('instruction', 'task', 'seed', 'max_steps', 'native_hz')):
            raise ValueError('Incomplete native case metadata')
        if native['max_steps'] != CAPS[row['environment']]:
            raise ValueError('Native cap differs from fixed plan')
        row.update(native)
        row['shard'] = i % len(gpus); row['gpu'] = gpus[i % len(gpus)]
        row['support_status'] = supported[row['environment']]['status']
    manifest = dict(schema=1, experiment='idea1 v3.2 E4 frozen zero-shot transfer',
                    created_utc=datetime.now(timezone.utc).isoformat(), modes=list(MODES), cases=case_rows,
                    gpus=gpus, source_files=source_files,
                    receipt_files={str(Path(p).resolve()): sha(p) for p in (environment_receipt, policy_receipt, source_reference)},
                    environment_receipt=env, policy_receipt=policy, source_reference_path=str(Path(source_reference).resolve()),
                    H=50, execute=5, seed_rule='same case seed and replan RNG seed across conditions; no mode in seed',
                    model_controls={'full_d10': dict(D=10, skip=[]), 'v14_d10': dict(D=10, skip=[3, 4, 9, 11]),
                                    'drafts': 'All four released suite heads, pure draft, no VLM/action-expert/verify'},
                    selection='All six conditions retained; no training, head selection or adaptation on target environments',
                    budget=dict(seconds=5400, started_utc='2026-10-22T05:00:00+00:00',
                                deadline_unix=datetime.fromisoformat('2026-10-22T06:30:00+00:00').timestamp(),
                                origin='self-imposed exploratory cap including preparation; not a user-specified E4 budget'),
                    metrics=dict(success='Native oracle, separate denominator per environment; technical/unsupported never zero-policy outcomes',
                                 pusht='Explicit 2D stress projection, not directly comparable to arm task success',
                                 latency='Same case/condition on same GPU shard; compare only within shard; cold/compile separate'),
                    videos='case0 in every environment, every condition; full-stream decode validation; fixed selection independent of outcome',
                    interpretation='Full baseline zero/unsupported adapter makes generalization inference inconclusive; no causal model-capacity claims')
    design = ROOT/'docs/experiments/idea1/idea1_v3_2_e4_zero_shot_protocol.md'
    manifest['canonical_design'] = dict(path=str(design), sha256=sha(design))
    manifest['manifest_hash'] = digest(manifest)
    write_json(output/'manifest.json', manifest, exclusive=True)
    return manifest


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output'); p.add_argument('--environment-receipt'); p.add_argument('--policy-receipt')
    p.add_argument('--source-reference'); p.add_argument('--gpus', nargs='+', type=int)
    p.add_argument('--aggregate-environments', nargs='+')
    p.add_argument('--freeze', action='store_true')
    a = p.parse_args()
    if a.aggregate_environments:
        if not a.environment_receipt:
            p.error('Aggregation requires --environment-receipt destination')
        result = aggregate_environments(a.aggregate_environments, a.environment_receipt)
        print(json.dumps(dict(status='complete', environments={k: v['status'] for k, v in result['environments'].items()})))
        return
    if not a.freeze:
        print(json.dumps(dict(dry_run=True, gpu_calls=0, modes=MODES, cases=cases(), episodes=108,
                              freeze_requires='completed environment binding/success inventory, checkpoint metadata, new prompt proof, source reference, one or two assigned GPUs')))
        return
    if not all((a.output, a.environment_receipt, a.policy_receipt, a.source_reference, a.gpus)):
        p.error('Freeze requires all receipt paths, output and one or two GPUs')
    print(json.dumps(freeze(a.output, a.environment_receipt, a.policy_receipt, a.source_reference, a.gpus), indent=2))


if __name__ == '__main__':
    main()

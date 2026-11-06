"""Explicit new-kind benchmark routing with sampled all-type process evidence."""
import json
from pathlib import Path

from common import atomic_json, sha
from benchmark_revision import read
from benchmark_all_process_revision import verify_revision, authorize
from benchmark_all_process_guard import validate_sample
from benchmark_revision_analysis import install_analysis_route


def verify_call_boundary(samples, guard, name, seen):
    pre, post = samples[guard['before_sequence']], samples[guard['after_sequence']]
    if (pre['label'] != name+':before' or post['label'] != name+':after'
            or not pre['monotonic_end'] <= guard['started'] <= guard['ended'] <= post['monotonic_start']
            or guard['before_sequence'] >= guard['after_sequence']):
        raise ValueError('Timed call is not bounded by its actual all-process observations')
    for sequence in (guard['before_sequence'], guard['after_sequence']):
        if sequence in seen:
            raise ValueError('Benchmark reused an exclusivity boundary')
        seen.add(sequence)


def verify_attempt_events(revision, root, model, phase):
    revision, root = Path(revision).resolve(), Path(root).resolve()
    if not root.is_relative_to(revision/'attempts'/model) or root.parent != revision/'attempts'/model or (root/'invalid_attempt.json').exists():
        raise ValueError('Attempt is invalid or belongs to another model/revision')
    completion = root/('gate_complete.json' if phase == 'allocation_gate' else 'revision_complete.json')
    proof, execution = read(completion), read(root/'execution_receipt.json')
    approved, path = authorize(revision, model, phase)
    if (proof.get('passed') is not True or proof['phase'] != phase or proof['model'] != model
            or execution['model'] != model or execution['phase'] != phase
            or execution['revision_manifest_sha256'] != sha(revision/'revision_manifest.json')
            or proof['revision_manifest_sha256'] != execution['revision_manifest_sha256']
            or execution['authorization_path'] != str(path) or execution['authorization_sha256'] != sha(path)
            or execution['uuid'] != approved['uuid'] or proof['uuid'] != execution['uuid']
            or proof['execution_receipt_sha256'] != sha(root/'execution_receipt.json') or proof['binding_sha256'] != sha(root/'binding.json')):
        raise ValueError('Per-model execution/completion authority differs')
    binding = read(root/'binding.json')
    events = binding['events']
    if [row['label'] for row in events] != ['empty_baseline', 'context_created', 'context_stable', 'challenge_allocated', 'challenge_released', 'causal_identity_bound']:
        raise ValueError('Causal binding sequence incomplete')
    pid, local, uuid = execution['nvml_pid'], execution['local'], execution['uuid']
    if events[0]['processes'] or events[0]['graphics']:
        raise ValueError('Baseline was not empty across all process types')
    for row in events:
        if row['uuid'] != uuid or row['local'] != local or row['graphics']:
            raise ValueError('Causal all-type identity changed')
    for row in events[1:]:
        if set(map(int, row['processes'])) != {pid}:
            raise ValueError('Causal compute process identity differs')
    key = str(pid)
    if events[3]['processes'][key]-events[2]['processes'][key] < 64 or events[3]['processes'][key]-events[4]['processes'][key] < 64:
        raise ValueError('Causal memory allocation/release challenge failed')
    samples, previous = {}, None
    labels = ['final'] if phase == 'allocation_gate' else ['startup']+[f'point{point:02d}' for point in range(32)]+['final']
    if len(proof['monitor_chunks']) != len(labels):
        raise ValueError('Missing complete startup/point/final monitor chunks')
    if {Path(item['path']).resolve() for item in proof['monitor_chunks']} != set((root/'monitor').glob('*.json')):
        raise ValueError('Unaccounted or missing monitor chunk files')
    for position, (chunk, label) in enumerate(zip(proof['monitor_chunks'], labels)):
        file = Path(chunk['path']).resolve()
        if file.parent != root/'monitor' or file.name != f'{position:04d}_{label}.json' or sha(file) != chunk['sha256']:
            raise ValueError('Monitor evidence missing, changed, or escaped attempt')
        response = read(file)
        if response.get('error') or not response.get('events'):
            raise ValueError('Incomplete/failed monitoring evidence')
        for row in response['events']:
            validate_sample(row, uuid, pid, previous)
            if row['local'] != local or row['sequence'] != len(samples)+1:
                raise ValueError('Monitor identity/sequence gap')
            samples[row['sequence']] = row
            previous = row
        if response.get('sequence') != previous['sequence']:
            raise ValueError('Monitor chunk completion sequence differs from actual events')
    if not samples or samples[1]['label'] != 'monitor_start' or samples[len(samples)]['label'] != 'final':
        raise ValueError('No complete start-through-final monitor coverage')
    return execution, proof, samples


def verify_benchmark(revision):
    from runtime_client import check_prediction
    from runtime_data import audited_protocol
    revision = Path(revision).resolve()
    identity = verify_revision(revision)
    manifest, result = read(revision/'condition_manifest.json'), {}
    for model in identity['required_models']:
        selection = read(revision/'selected'/f'{model}.json')
        root = Path(selection['attempt']).resolve()
        if selection['model'] != model or selection['revision_manifest_sha256'] != sha(revision/'revision_manifest.json') or selection['completion_sha256'] != sha(root/'revision_complete.json'):
            raise ValueError('Selected benchmark model result changed')
        execution, proof, samples = verify_attempt_events(revision, root, model, 'benchmark')
        approval, _ = authorize(revision, model, 'benchmark')
        verify_attempt_events(revision, Path(approval['gate_receipt']).parent, model, 'allocation_gate')
        directory = root/'benchmark'/model
        conds = [c for c in manifest['conditions'] if c['model'] == model]
        if (read(directory/'complete.json') != dict(conditions=len(conds), points=32, warmup=5, measurement=20, exclusive=True)
                or sha(directory/'complete.json') != proof['scientific_complete_sha256']):
            raise ValueError('Scientific benchmark scope incomplete')
        expected = {f"p{point:02d}_r{repeat:02d}_{c['id']}.json" for point in range(32) for repeat in range(25) for c in conds}
        if {path.name for path in (directory/'samples').iterdir()} != expected:
            raise ValueError('Missing/unexpected benchmark sample set')
        protocol = audited_protocol(root, model)
        seen_boundaries, noise = set(), {}
        for condition in conds:
            for point in range(32):
                source = manifest['observations'][model][point]
                for repeat in range(25):
                    name = f"p{point:02d}_r{repeat:02d}_{condition['id']}"
                    row = read(directory/'samples'/f'{name}.json')
                    check_prediction(row, condition, protocol, source['seed'])
                    guard = row['all_process_guard']
                    verify_call_boundary(samples, guard, name, seen_boundaries)
                    if (row['condition'] != condition['id'] or row['point'] != point or row['repeat'] != repeat
                            or row['independent_forward'] is not True or row['input_hash'] != source['content_hash']
                            or row['phase'] != ('warmup' if repeat < 5 else 'measurement')
                            or row['audit'].get('instrumentation_mode') != 'minimal'
                            or row['timing'].get('measurement_mode') != 'minimal_action_generation'
                            or row['timing'].get('shared_condition') is not False
                            or not isinstance(row['calls_before'], int) or isinstance(row['calls_before'], bool)
                            or row['calls_after'] != row['calls_before']+1):
                        raise ValueError('Canonical independent minimal benchmark protocol differs')
                    for field in ('action_generation_ms', 'server_total_ms', 'client_roundtrip_ms'):
                        if not isinstance(row['timing'][field], (int, float)) or not 0 < row['timing'][field] < float('inf'):
                            raise ValueError('Invalid benchmark timing')
                    for field in ('noise_path', 'server_prediction_path'):
                        if not Path(row[field]).resolve().is_relative_to(root/'native'/model):
                            raise ValueError('Benchmark reused raw artifacts from another attempt')
                    if point in noise and noise[point] != row['noise_hash']:
                        raise ValueError('Benchmark initial noise not paired')
                    noise[point] = row['noise_hash']
        observed_boundaries = {sequence for sequence, row in samples.items() if row['label'].endswith((':before', ':after'))}
        if observed_boundaries != seen_boundaries:
            raise ValueError('Unexpected/unaccounted monitored benchmark calls')
        result[model] = dict(samples=len(expected), attempt=str(root),
            files={str(path.relative_to(root)):sha(path) for path in root.rglob('*') if path.is_file()})
    if sum(item['samples'] for item in result.values()) != 43200:
        raise ValueError('Final benchmark must contain all 43,200 original planned calls')
    return identity, result


def create_view(revision, view):
    revision, view = Path(revision).resolve(), Path(view).resolve()
    identity, benchmarks = verify_benchmark(revision)
    parent = Path(identity['source_attempt'])
    view.mkdir(parents=True, exist_ok=False)
    routes = {}
    for name in ('condition_manifest.json', 'episode_ledger.json', 'episodes', 'offline', 'parity', 'runtime_acceptance.json', 'native'):
        source = parent/name
        (view/name).symlink_to(source, target_is_directory=source.is_dir())
        routes[name] = str(source)
    (view/'protocols').symlink_to(revision/'protocols', target_is_directory=True)
    (view/'benchmark').mkdir()
    for model, item in benchmarks.items():
        (view/'benchmark'/model).symlink_to(Path(item['attempt'])/'benchmark'/model, target_is_directory=True)
    routes.update(protocols=str(revision/'protocols'), benchmark={model:str(view/'benchmark'/model) for model in benchmarks})
    receipt = dict(schema=2, kind='sampled_all_process_benchmark_view', source_attempt=str(parent), revision=str(revision),
        revision_manifest_sha256=sha(revision/'revision_manifest.json'), parent_manifest_sha256=sha(parent/'condition_manifest.json'),
        routes=routes, benchmark_inputs=benchmarks, overlay_code_sha256=sha(__file__),
        model_authorizations={model:sha(revision/'acceptance'/f'{model}.benchmark.json') for model in benchmarks},
        excluded_historical_benchmarks=[str(parent/'benchmark'), str(parent/'benchmark_revisions/pid_causal_v1')],
        isolation_claim=identity['isolation_claim'], observer_effect=identity['observer_effect'])
    atomic_json(view/'benchmark_revision_view_receipt.json', receipt)
    atomic_json(parent/'benchmark_revision_index.json', dict(schema=2, revision=str(revision), analysis_view=str(view),
        revision_manifest_sha256=sha(revision/'revision_manifest.json'), view_receipt_sha256=sha(view/'benchmark_revision_view_receipt.json'),
        models={model:str(Path(item['attempt'])/'benchmark'/model) for model,item in benchmarks.items()}, historical_benchmark_excluded=True))
    return view

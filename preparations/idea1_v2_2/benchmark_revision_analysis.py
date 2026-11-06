"""Explicitly route final analysis to a wholly separate benchmark revision."""
import argparse
import json
from pathlib import Path

from common import atomic_json, sha
from benchmark_revision import read, verify_revision


def verify_benchmark(revision):
    from runtime_client import check_prediction
    from runtime_data import audited_protocol
    revision = Path(revision).resolve()
    identity = verify_revision(revision)
    manifest = read(revision/'condition_manifest.json')
    result = {}
    for model in identity['required_models']:
        directory = revision/'benchmark'/model
        execution = read(directory/'execution_receipt.json')
        complete = read(directory/'revision_complete.json')
        if (execution.get('schema') != 1 or complete.get('exclusive') is not True or
                execution['revision_manifest_sha256'] != sha(revision/'revision_manifest.json') or
                execution['acceptance_sha256'] != sha(revision/'benchmark_revision_acceptance.json') or
                execution['model'] != model or execution['gate_only'] is not False):
            raise ValueError('Benchmark execution receipt is stale')
        for name, field in [('identity_events.jsonl', 'identity_events_sha256'),
                            ('execution_receipt.json', 'execution_receipt_sha256'), ('complete.json', 'complete_sha256')]:
            if sha(directory/name) != complete[field]:
                raise ValueError(f'Benchmark completion receipt changed: {name}')
        events = [json.loads(line) for line in (directory/'identity_events.jsonl').read_text().splitlines()]
        labels = [e['label'] for e in events]
        required_prefix = ['empty_baseline', 'context_created', 'context_stable', 'challenge_allocated',
                           'challenge_released', 'causal_identity_bound']
        if labels[:6] != required_prefix or events[0]['processes'] != {}:
            raise ValueError('Missing CUDA/NVML causal binding sequence')
        pid = str(execution['nvml_pid'])
        for event in events:
            if event['uuid'] != execution['uuid'] or event['local'] != execution['local'] or str(event['gpu']) != execution['gpu']:
                raise ValueError('GPU/local identity changed')
        if any(set(e['processes']) != {pid} for e in events[1:]):
            raise ValueError('GPU exclusive PID set changed')
        if events[3]['processes'][pid] - events[2]['processes'][pid] < 64 or events[3]['processes'][pid] - events[4]['processes'][pid] < 64:
            raise ValueError('CUDA allocation/release challenge failed')
        if labels[6:] != ['benchmark_snapshot'] + ['point_boundary'] * 32 + ['benchmark_snapshot']:
            raise ValueError('Missing complete pre-point/final exclusivity coverage')
        if any(a['timestamp'] > b['timestamp'] for a, b in zip(events, events[1:])):
            raise ValueError('Identity event clock moved backwards')
        for filename, event in [('exclusivity.json', events[6]), ('final_exclusivity.json', events[-1])]:
            snapshot = read(directory/filename)
            if snapshot['uuid'] != event['uuid'] or snapshot['pids'] != [int(pid)] or snapshot['timestamp'] != event['timestamp']:
                raise ValueError('Exclusivity snapshot differs from event log')
        conds = [c for c in manifest['conditions'] if c['model'] == model]
        if read(directory/'complete.json') != dict(conditions=len(conds), points=32, warmup=5, measurement=20, exclusive=True):
            raise ValueError('Benchmark complete marker has wrong coverage')
        expected = {f"p{point:02d}_r{repeat:02d}_{c['id']}.json"
                    for c in conds for point in range(32) for repeat in range(25)}
        actual = {p.name for p in (directory/'samples').iterdir()}
        if actual != expected:
            raise ValueError(f'Incomplete/unexpected {model} benchmark samples: {len(actual)}/{len(expected)}')
        protocol = audited_protocol(revision, model)
        noise_by_point = {}
        native_root = (revision/'native'/model).resolve()
        for condition in conds:
            for point in range(32):
                source = manifest['observations'][model][point]
                for repeat in range(25):
                    path = directory/'samples'/f"p{point:02d}_r{repeat:02d}_{condition['id']}.json"
                    if not events[7 + point]['timestamp'] <= path.stat().st_mtime <= events[8 + point]['timestamp']:
                        raise ValueError('Sample file is outside its checked point interval')
                    sample = read(path)
                    check_prediction(sample, condition, protocol, source['seed'])
                    if (sample['condition'] != condition['id'] or sample['point'] != point or
                            sample['repeat'] != repeat or sample['independent_forward'] is not True or
                            sample['input_hash'] != source['content_hash'] or
                            sample['phase'] != ('warmup' if repeat < 5 else 'measurement')):
                        raise ValueError('Sample canonical identity mismatch')
                    for component in ('action_generation_ms', 'server_total_ms', 'client_roundtrip_ms'):
                        value = sample['timing'][component]
                        if not isinstance(value, (int, float)) or not 0 < value < float('inf'):
                            raise ValueError('Invalid benchmark timing')
                    if (sample['timing'].get('measurement_mode') != 'minimal_action_generation' or
                            sample['timing'].get('shared_condition') is not False or
                            sample['audit'].get('instrumentation_mode') != 'minimal'):
                        raise ValueError('Benchmark must use independent minimal instrumentation')
                    before, after = sample['calls_before'], sample['calls_after']
                    if not isinstance(before, int) or isinstance(before, bool) or after != before + 1:
                        raise ValueError('Benchmark did not perform exactly one independent forward')
                    for key in ('noise_path', 'server_prediction_path'):
                        if not Path(sample[key]).resolve().is_relative_to(native_root):
                            raise ValueError('Benchmark reused native artifacts from another attempt')
                    if point in noise_by_point and sample['noise_hash'] != noise_by_point[point]:
                        raise ValueError('Benchmark initial noise not paired')
                    noise_by_point[point] = sample['noise_hash']
        hashes = {str(p.relative_to(revision)): sha(p) for p in directory.rglob('*') if p.is_file()}
        result[model] = dict(samples=len(actual), files=hashes)
    return identity, result


def create_view(revision, view):
    revision, view = Path(revision).resolve(), Path(view).resolve()
    identity, benchmarks = verify_benchmark(revision)
    parent = Path(identity['source_attempt']).resolve()
    view.mkdir(parents=True, exist_ok=False)
    # Input-only symlinks. All derived analysis and C audit artifacts live in view.
    names = ['condition_manifest.json', 'episode_ledger.json', 'episodes', 'offline', 'parity',
             'runtime_acceptance.json', 'native']
    routes = {}
    for name in names:
        source = parent/name
        (view/name).symlink_to(source, target_is_directory=source.is_dir())
        routes[name] = str(source)
    (view/'protocols').symlink_to(revision/'protocols', target_is_directory=True)
    (view/'benchmark').symlink_to(revision/'benchmark', target_is_directory=True)
    routes.update(protocols=str(revision/'protocols'), benchmark=str(revision/'benchmark'))
    receipt = dict(schema=1, kind='explicit_benchmark_revision_analysis_view', source_attempt=str(parent),
                   revision=str(revision), revision_manifest_sha256=sha(revision/'revision_manifest.json'),
                   acceptance_sha256=sha(revision/'benchmark_revision_acceptance.json'),
                   parent_manifest_sha256=sha(parent/'condition_manifest.json'), routes=routes,
                   benchmark_inputs=benchmarks, excluded_historical_benchmark=str(parent/'benchmark'),
                   overlay_code_sha256=sha(__file__),
                   artifact_directory_routing='original audit validates real artifacts, then return equivalent lexical view/episodes alias to analysis')
    atomic_json(view/'benchmark_revision_view_receipt.json', receipt)
    atomic_json(parent/'benchmark_revision_index.json', dict(
        schema=1, revision=str(revision), analysis_view=str(view),
        revision_manifest_sha256=receipt['revision_manifest_sha256'],
        view_receipt_sha256=sha(view/'benchmark_revision_view_receipt.json'),
        models={model: str(revision/'benchmark'/model) for model in identity['required_models']},
        historical_benchmark_excluded=True))
    return view


def install_analysis_route(module, view, parent):
    """Preserve original validation, expose equivalent lexical paths for relative URLs."""
    view, parent = Path(view).resolve(), Path(parent).resolve()
    original_audit = module.audit
    def routed_audit(root, strict=True, check_outputs=True):
        if Path(root).resolve() != view:
            raise ValueError('Analysis route used outside its declared view')
        result, valid = original_audit(root, strict, check_outputs)
        routed = {}
        for key, entry in valid.items():
            actual = Path(entry['directory']).resolve()
            relative = actual.relative_to(parent/'episodes')
            alias = view/'episodes'/relative
            if alias.resolve() != actual:
                raise ValueError('Analysis artifact alias differs from validated source')
            routed[key] = dict(entry, directory=str(alias))
        return result, routed
    module.audit = routed_audit
    return original_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--view', required=True)
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    view = create_view(args.revision, args.view)
    if args.analyze:
        import runtime_analysis
        install_analysis_route(runtime_analysis, view, read(Path(args.revision)/'revision_manifest.json')['source_attempt'])
        runtime_analysis.analyze(view)
        runtime_analysis.audit(view, strict=True)


if __name__ == '__main__':
    main()

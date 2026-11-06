"""Cache-only additions for the explicitly versioned final provenance auditor."""
import ast
import hashlib
import json
from pathlib import Path

from common import sha, prior_diagnostic_budget
from throughput_revision import read
from pi0_prompt_cache_revision import compare_gate, verify_revision as original_verify_revision, MEMORY_CAP_MIB


def verify_revision(revision):
    identity, manifest = original_verify_revision(revision)
    if identity.get('gate_controller'):
        from pi0_prompt_cache_order_revision import validate_order
        validate_order(identity)
    return identity, manifest


def verify_patch(index, server):
    identity, _ = verify_revision(index.revision)
    path = Path(server)/'runtime_execution_patch.json'
    patch = index.load(path)
    if (patch.get('schema') != 1 or patch.get('kind') != 'bounded_prompt_length_runtime_cache' or
            patch.get('capacity') != 2 or patch.get('revision_manifest_sha256') != index.revision_hash or
            patch['original_prepare'] != identity['original_prepare'] or
            patch['patch_source_sha256'] != identity['code_hashes']['pi0_prompt_cache.py'] or
            Path(patch['patch_source']).resolve() != Path(__file__).with_name('pi0_prompt_cache.py').resolve()):
        raise ValueError('Actual loaded cache patch differs from reviewed source/function identity')
    source = Path(__file__).with_name('runtime_models.py')
    lines = source.read_text().splitlines(keepends=True)
    cls = next(node for node in ast.parse(''.join(lines)).body if isinstance(node, ast.ClassDef) and node.name == 'NativePolicy')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'predict')
    original = patch['original_predict']
    if (original['module'] != 'runtime_models' or original['qualname'] != 'NativePolicy.predict' or
            Path(original['source_file']).resolve() != source.resolve() or
            original['source_sha256'] != hashlib.sha256(''.join(lines[method.lineno-1:method.end_lineno]).encode()).hexdigest()):
        raise ValueError('Actual original prediction function identity differs')
    index.record(source)


def cache_events(server):
    events, checksum, length = {}, hashlib.sha256(), 0
    path = Path(server)/'prompt_cache_events.jsonl'
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                break
            event = json.loads(line)
            if event['call'] in events or len(event['resident_lengths']) > 2 or event['capacity'] != 2:
                raise ValueError('Duplicate cache call or resident runtime bound exceeded')
            if event['hit'] and (event['startup_forward_calls'] or event['startup_action_NFE']):
                raise ValueError('A purported cache hit rebuilt the runtime')
            if event['cuda_peak_reserved_mib'] >= MEMORY_CAP_MIB or event['cuda_peak_allocated_mib'] >= MEMORY_CAP_MIB:
                raise ValueError('Cache allocator peak exceeded reviewed cap')
            events[event['call']] = event
            checksum.update(line)
            length += len(line)
    return events, dict(path=str(path.resolve()), prefix_bytes=length, prefix_sha256=checksum.hexdigest())


def verify_gate(index, model, manifest, runtime):
    from runtime_client import check_prediction
    from runtime_data import audited_protocol, verify_frozen, verify_parity
    if model != 'pi0':
        raise ValueError('Cache revision may not authorize another model')
    identity, _ = verify_revision(index.revision)
    if identity.get('gate_controller'):
        index.record(identity['gate_controller']['source'])
        index.record(Path(identity['gate_controller']['failed_revision'])/'revision_manifest.json')
    ready = index.load(index.revision/'pi0.gate_ready.json')
    session, reference = Path(ready['session']).resolve(), Path(ready['reference_session']).resolve()
    if (ready.get('actual_predictions') != 16 or ready.get('exact') is not True or
            ready['model'] != 'pi0' or ready['phase'] != 'gate' or ready['revision_manifest_sha256'] != index.revision_hash or
            ready['memory_cap_mib'] != MEMORY_CAP_MIB or ready['peak_sampled_gpu_memory_mib'] >= MEMORY_CAP_MIB or
            ready['cached_order'] != identity['cached_order'] or ready['reference_order'] != identity['reference_order'] or
            session.parent != index.revision/'sessions' or reference.parent != index.revision/'gate_reference'):
        raise ValueError('Sixteen-call cache gate identity/resource policy differs')
    execution = index.load(session/'execution_receipt.json')
    if execution['environments'] != 0 or execution['gate_request_clients'] != 1:
        raise ValueError('Cache gate must describe its actual single sequential request client')
    verify_patch(index, session/'server')
    events, prefix = cache_events(session/'server')
    if set(events) != set(range(1, 9)):
        raise ValueError('Cache gate did not execute exactly eight cached logical calls')
    runtimes = {}
    reference_execution = index.load(reference/'gate_execution_receipt.json')
    if reference_execution['reference'] is not True or reference_execution['revision_manifest_sha256'] != index.revision_hash:
        raise ValueError('Unpatched reference execution identity differs')
    verify_frozen(identity['source_attempt'], reference_execution['frozen_runtime_protocol'], 'pi0')
    verify_parity(identity['source_attempt'], reference_execution['frozen_runtime_protocol'], 'pi0')
    protocol = audited_protocol(identity['source_attempt'], 'pi0', runtime)
    for request in identity['requests']:
        number = request['index']
        for root in (reference, session):
            path = root/'outputs'/f'{number}.json'
            row = index.load(path)
            check_prediction(row, request['condition'], protocol, request['seed'])
            if row['input_hash'] != request['point']['content_hash'] or row['loaded_code_hashes'] != runtime['scientific_code_hashes']:
                raise ValueError('Cache/reference input/scientific identity differs')
            for key in ('noise_path', 'server_prediction_path'):
                if not Path(row[key]).resolve().is_relative_to(reference if root == reference else session/'server'):
                    raise ValueError('Gate raw artifacts came from another session')
                index.record(row[key])
            order = identity['reference_order'] if root == reference else identity['cached_order']
            if row['call'] != order.index(number)+1:
                raise ValueError('Actual reference/cache call order differs from manifest')
            startup = sum(item['forward_calls'] for item in row['audit'].get('startup_events', []))
            expected_startup = 4 if root == reference and identity.get('gate_controller') else (4 if number < 2 else 0)
            if startup != expected_startup:
                raise ValueError('Reference/cache startup accounting is not two cold lengths only')
            if root == session:
                event = row['execution_prompt_cache']
                if event != events[row['call']]:
                    raise ValueError('Cached result differs from actual runtime dispatch log')
                old = runtimes.setdefault(event['prompt_length'], event['runtime_id'])
                if old != event['runtime_id']:
                    raise ValueError('A length unexpectedly changed runtime during the gate')
    if len(runtimes) != 2 or len(set(runtimes.values())) != 2 or compare_gate(index.revision, session, reference) != ready['comparisons']:
        raise ValueError('Cache/reference exact outputs or isolated runtimes differ')
    index.record(session/'server/prompt_cache_events.jsonl')
    for server in (reference, session/'server'):
        index.record(server/'startup_forward_ledger.jsonl')
        index.record(server/'native_protocol.json')
        index.record(server/'service_lease.json')
        lines = (server/'prediction_ledger.jsonl').read_text().splitlines()
        if len(lines) != 8 or {json.loads(line)['call'] for line in lines} != set(range(1, 9)):
            raise ValueError('Reference/cache server recorded extra or missing logical predictions')
        index.record(server/'prediction_ledger.jsonl')
    samples = [json.loads(line)['memory_mib'] for line in (session/'gpu_memory.jsonl').read_text().splitlines()]
    if len(samples) != 16 or max(samples) != ready['peak_sampled_gpu_memory_mib'] or max(samples) >= MEMORY_CAP_MIB:
        raise ValueError('Gate total GPU sampled memory evidence differs')
    index.record(session/'gpu_memory.jsonl')
    original = Path(identity['source_attempt'])/'readiness/pi0/diagnostic_budget.json'
    throughput = original.with_name('throughput_diagnostic_budget.json')
    budget = original.with_name('prompt_cache_diagnostic_budget.json')
    base = (index.load(original)['smoke_reserved'] + index.load(Path(identity['source_attempt'])/'parity/pi0.json')['reserved_calls']
            + prior_diagnostic_budget(manifest, 'pi0') + sum(item['reserved_predictions'] for item in index.load(throughput)['entries']))
    matched = []
    for entry in index.load(budget)['entries']:
        if entry['base_total'] != base or entry['reserved_predictions'] != 16:
            raise ValueError('Cache diagnostic reservation sequence differs')
        base += 16
        if entry['revision_manifest_sha256'] == index.revision_hash:
            matched.append(entry)
    if (base > 256 or len(matched) != 1 or matched[0]['base_total']+16 != ready['total_diagnostic_reserved'] or
            matched[0].get('status') != 'gate_complete' or matched[0].get('actual_predictions') != 16 or
            matched[0].get('gate_receipt_sha256') != sha(index.revision/'pi0.gate_ready.json')):
        raise ValueError('Cache gate lacks a bounded completed sixteen-call reservation')


def verify_episode_cache(index, entry, record, infer_path):
    session = Path(entry['throughput_session'])
    if not hasattr(index, '_prompt_cache_event_snapshots'):
        index._prompt_cache_event_snapshots = {}
    key = str(session)
    if key not in index._prompt_cache_event_snapshots:
        verify_patch(index, session/'server')
        index._prompt_cache_event_snapshots[key] = cache_events(session/'server')
    events, prefix = index._prompt_cache_event_snapshots[key]
    record['prompt_cache_ledger_prefix'] = prefix
    for line in Path(infer_path).read_text().splitlines():
        row = json.loads(line)
        event = events.get(row['call'])
        if not event or row.get('execution_prompt_cache') != event or any(event[key] != row[key] for key in ('input_hash', 'noise_hash', 'seed')) or event['skip'] != row['audit']['skip'] or event['schedule'] != row['audit']['schedule'] or event['prediction_path'] != row['server_prediction_path']:
            raise ValueError('Formal inference does not bind to its actual cache dispatch')

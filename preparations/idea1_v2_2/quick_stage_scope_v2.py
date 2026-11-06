"""Explicit quick-stage subset of the real, complete, preserved ledger snapshot."""
import hashlib
import json
from pathlib import Path
import numpy as np
from common import ROOT, SUITES, planned, sha, digest
from retry_history_audit import resolve, manifest_identity, recovery_references

INITS = (6, 7, 8)
EXPECTED = 6480


def read(path): return json.loads(Path(path).read_text())


def verify_protocol_receipts(root):
    """Reuse frozen scientific gates, checking their small identity/coverage receipts."""
    from runtime_data import parity_identity, verify_acceptance
    root=Path(root); manifest=read(root/'condition_manifest.json')
    if digest({k:v for k,v in manifest.items() if k!='manifest_hash'})!=manifest['manifest_hash']:
        raise ValueError('Frozen manifest identity changed')
    for name,expected in manifest['code_hashes'].items():
        if sha(Path(__file__).parent/name)!=expected: raise ValueError('Frozen source changed')
    for model,expected in manifest['protocol_hashes'].items():
        protocol=read(root/'protocols'/f'{model}.json'); runtime=read(root/'protocols'/f'{model}.runtime.json')
        parity=read(root/'parity'/f'{model}.json')
        if (digest(protocol)!=expected or parity.get('passed') is not True
                or parity.get('identity')!=parity_identity(manifest,runtime,model)
                or not 0<parity.get('actual_predictions',0)<=parity.get('reserved_calls',0)<=256
                or len(parity.get('checks',[]))!=4 or not all(check.get('passed') is True for check in parity['checks'])):
            raise ValueError('Frozen protocol/parity receipt identity changed')
        verify_acceptance(root,model,parity)
    return manifest


def select(manifest, ledger_bytes, recovery=()):
    history = resolve(manifest, ledger_bytes, recovery)
    canonical = {row['key']: row for row in planned(manifest['data'])}
    main = {key:row for key,row in canonical.items() if row['init'] in INITS}
    if len(canonical)!=10800 or len(main)!=EXPECTED or len(manifest['data']['conditions'])!=54:
        raise ValueError('Quick scope must be exactly 54 conditions x4 suites x10 tasks x3 init')
    missing = set(main)-history['valid_by_key'].keys()
    if missing or any(item['reason']!='not_started' for item in history['unresolved']):
        raise ValueError('Missing main key or unresolved attempted history; drain/recovery evidence required')
    entries = list(json.loads(ledger_bytes)['entries'])
    if any(row['status']=='reserved' for row in entries):
        raise ValueError('Drain snapshot still contains reservations')
    valid = history['valid_by_key']
    extras = {key:row for key,row in valid.items() if key not in main}
    return dict(main={key:valid[key] for key in main}, extras=extras, history=history['history_index'],
        original_full_complete=len(valid)==len(canonical), original_planned=10800)


def physical_directory(attempt, entry):
    raw=Path(entry['directory'])
    actual=(raw if raw.is_absolute() else ROOT/raw).resolve()
    expected=Path(attempt).resolve()/'episodes'/entry['condition']/entry['suite']/f"task{entry['task']:02d}"/f"init{entry['init']:02d}"/entry['token']
    if expected.resolve()!=expected or actual!=expected:
        raise ValueError('Original ledger directory is not its exact canonical physical token directory')
    return actual


def build(attempt, ledger_snapshot, drain_receipt):
    attempt, ledger_snapshot, drain_receipt = map(lambda x:Path(x).resolve(), (attempt,ledger_snapshot,drain_receipt))
    envelope=manifest_identity(attempt/'condition_manifest.json')
    raw=ledger_snapshot.read_bytes(); selected=select(envelope,raw,recovery_references(attempt))
    drain=read(drain_receipt); entries=json.loads(raw)['entries']
    snapshot=drain.get('drain_snapshot',{}); sessions=drain.get('formal_sessions',[])
    process=drain.get('process_and_lease_checks',{})
    if (drain.get('schema')!=1 or drain.get('receipt_type')!='quick_scope_formal_drain_validation'
            or drain.get('attempt')!=str(attempt)
            or (attempt/snapshot.get('path','')).resolve()!=ledger_snapshot
            or snapshot.get('sha256')!=hashlib.sha256(raw).hexdigest()
            or snapshot.get('bytes')!=len(raw) or snapshot.get('entries')!=len(entries)
            or snapshot.get('valid_rows')!=len(selected['main'])+len(selected['extras'])
            or snapshot.get('reserved_rows')!=0 or snapshot.get('canonical_ledger_lock')!='episode_ledger.lock'
            or drain.get('frozen_manifest',{}).get('sha256')!=envelope['file_sha256']
            or drain.get('quick_scope',{}).get('inits')!=list(INITS)
            or drain.get('quick_scope',{}).get('valid_keys')!=EXPECTED
            or drain.get('preserved_supplemental_scope',{}).get('valid_rows_outside_quick_scope')!=len(selected['extras'])
            or len(sessions)!=4 or any(s.get('exit_code')!=0 for s in sessions)
            or len({s.get('session_id') for s in sessions})!=4
            or {(s.get('model'),s.get('shard')) for s in sessions}!={('pi0',0),('pi0',1),('db-cogact',0),('db-cogact',1)}
            or any(process.get(field) is not True for field in (
                'all_eight_worker_server_pids_absent_after_sessions_terminated',
                'all_four_model_shard_locks_nonblocking_acquired_then_released',
                'all_four_gpu_locks_nonblocking_acquired_then_released'))):
        raise ValueError('Drain receipt does not bind this exact complete snapshot and natural drain')
    bound={str(envelope['path']):envelope['file_sha256'], str(ledger_snapshot):hashlib.sha256(raw).hexdigest(),
           str(drain_receipt):sha(drain_receipt)}
    if not drain_receipt.is_file() or ledger_snapshot==attempt/'episode_ledger.json':
        raise ValueError('An independent preserved drain snapshot/receipt is required')
    all_valid={**selected['main'],**selected['extras']}; records={}
    protocols={model:read(attempt/'protocols'/f'{model}.json') for model in envelope['data']['protocol_hashes']}
    for key,entry in all_valid.items():
        directory=physical_directory(attempt,entry)
        if not directory.is_relative_to(attempt/'episodes'):
            raise ValueError('Episode directory escapes original attempt')
        path=directory/'validated.json'; validated=read(path)
        result_path=directory/'result.json'; result=read(result_path)
        protocol=protocols[entry['model']]; r=protocol.get('R_by_suite',protocol['R'])
        r=r[entry['suite']] if isinstance(r,dict) else r
        if (sha(path)!=entry['validated_sha256'] or validated.get('episode_key')!=key
                or validated.get('valid') is not True or validated.get('video_verified') is not True
                or validated.get('success')!=entry['success'] or result.get('success')!=entry['success']
                or sha(result_path)!=validated['artifacts']['result.json']
                or any(validated.get(field)!=entry[field] for field in ('model','suite','task','init'))
                or validated.get('actions')!=result.get('actions')
                or validated.get('frames')!=validated.get('actions',-2)+1
                or validated.get('frames')!=result.get('frames')
                or validated.get('H')!=protocol['H'] or validated.get('R')!=r
                or validated.get('actions',-1)>envelope['data']['native_action_limits']['limits'][entry['suite']]):
            raise ValueError('Committed top-level validation/result identity differs')
        bound[str(path)]=entry['validated_sha256']; bound[str(result_path)]=sha(result_path)
        records[key]=dict(entry,result=validated,resolved_directory=str(directory))
    extras_counts={}
    for entry in selected['extras'].values():
        label=entry['condition']+'|'+entry['suite']; extras_counts[label]=extras_counts.get(label,0)+1
    return dict(schema=1,kind='user_authorized_quick_stage_v2',source_attempt=str(attempt),
        manifest_sha256=envelope['file_sha256'],manifest_hash=envelope['data']['manifest_hash'],
        ledger_snapshot=str(ledger_snapshot),ledger_snapshot_sha256=hashlib.sha256(raw).hexdigest(),
        drain_receipt=str(drain_receipt),drain_receipt_sha256=sha(drain_receipt),
        main_inits=list(INITS),main_expected=EXPECTED,per_condition=120,per_condition_suite=30,
        main_keys=sorted(selected['main']),extra_keys=sorted(selected['extras']),
        extras_actual_denominators=extras_counts,valid_records=records,retry_history=selected['history'],
        original_planned=10800,original_full_complete=selected['original_full_complete'],
        quick_stage_complete=True,verified_top_level_file_hashes=bound,
        source_sha256=sha(__file__),verification_scope='All canonical/history and top-level validated/result receipts checked once; no full nested raw/video rehash or decode; worker validations reused; independent media coverage reported separately')


def check_prediction_metadata(record,condition,protocol,expected_seed=None):
    audit=record.get('audit',{})
    if audit.get('NFE')!=condition['schedule']['num_steps'] or audit.get('schedule')!=condition['schedule'] or audit.get('skip')!=condition['skip']: raise ValueError('Actual NFE/schedule/skip does not match frozen condition')
    if expected_seed is not None and record.get('seed')!=expected_seed: raise ValueError('Generation seed differs from frozen rule')
    actions=np.asarray(record.get('actions',record.get('chunk_actions')))
    if actions.shape!=(protocol['H'],7) or not np.isfinite(actions).all(): raise ValueError('Full H action shape/value invalid')
    if audit.get('instrumentation_mode')=='minimal':
        if record.get('category') not in ('measurement','warmup','benchmark') or record.get('timing',{}).get('measurement_mode')!='minimal_action_generation': raise ValueError('Minimal audit used outside independent timing')
    else:
        actual=audit.get('actual_timesteps') or audit.get('timesteps') or []
        if len(actual)!=condition['schedule']['num_steps'] or audit.get('actual_NFE')!=condition['schedule']['num_steps']: raise ValueError('Actual timestep/NFE count missing')
    from schedules import flow_schedule
    if condition['model']!='db-cogact':
        expected=flow_schedule(condition['model'],condition['schedule']['m'],condition['schedule']['num_steps'])
        if not np.allclose(audit.get('nodes',[]),expected['nodes'],atol=1e-7) or not np.allclose(audit.get('dt',[]),expected['dt'],atol=1e-7): raise ValueError('Flow update nodes/dt differ from frozen formula')
        if audit.get('instrumentation_mode')!='minimal':
            values=np.asarray(expected['timesteps'],dtype=np.float32)
            if condition['model']=='xiaomi-robotic0':
                bits=values.view(np.uint32); values=((bits+0x7fff+((bits>>16)&1))&0xffff0000).view(np.float32)
            if not np.allclose(actual,values,rtol=0,atol=1e-7): raise ValueError('Actual model timestep coordinate differs')
    else:
        mapping=list(reversed(audit.get('timestep_map',[])))
        if audit.get('timesteps')!=mapping or audit.get('nodes')!=mapping+['clean'] or audit.get('alpha_clean')!=1.: raise ValueError('DDIM mapping/clean endpoint differs')
        if condition['schedule']['kind']=='ddim_custom' and mapping[1]!=round(condition['schedule']['m']*mapping[0]): raise ValueError('DDIM intermediate training index differs')
    evidence=protocol.get('v22_audited_schedules',{}).get(json.dumps(condition['schedule'],sort_keys=True))
    if evidence:
        for field in ('nodes','dt','timestep_map','alpha_cumprod'):
            if field in evidence and audit.get(field)!=evidence[field]: raise ValueError('Schedule differs from parity-audited sampler')
    elif audit.get('instrumentation_mode')=='minimal': raise ValueError('Minimal measurement lacks separately audited schedule evidence')


def load_scope(root):
    root=Path(root); scope=read(root/'quick_stage_scope.json')
    if (scope.get('kind')!='user_authorized_quick_stage_v2' or scope.get('main_inits')!=list(INITS)
            or scope.get('main_expected')!=EXPECTED or scope.get('source_sha256')!=sha(__file__)
            or sha(root/'condition_manifest.json')!=scope['manifest_sha256']
            or sha(root/'episode_ledger.json')!=scope['ledger_snapshot_sha256']
            or sha(scope['drain_receipt'])!=scope['drain_receipt_sha256']):
        raise ValueError('Quick scope/source/snapshot binding changed')
    main=scope['main_keys']; extra=scope['extra_keys']; records=scope['valid_records']
    if (len(main)!=EXPECTED or len(set(main))!=EXPECTED or len(extra)!=len(set(extra))
            or set(main)&set(extra) or set(records)!=set(main)|set(extra)
            or any(records[key]['init'] not in INITS for key in main)
            or any(records[key]['init'] in INITS for key in extra)):
        raise ValueError('Quick main/extras coverage changed')
    history=select(manifest_identity(root/'condition_manifest.json'), (root/'episode_ledger.json').read_bytes(),
                   scope['retry_history']['recovery_receipts'])
    if (set(history['main'])!=set(main) or set(history['extras'])!=set(extra)
            or history['history']!=scope['retry_history']):
        raise ValueError('Scope does not represent the preserved full append-order history')
    for key,entry in {**history['main'],**history['extras']}.items():
        if records[key].get('resolved_directory')!=str(physical_directory(scope['source_attempt'],entry)):
            raise ValueError('Resolved directory differs from original canonical physical token')
        if any(records[key].get(field)!=value for field,value in entry.items()):
            raise ValueError('Scope effective valid row differs from full snapshot')
    return scope


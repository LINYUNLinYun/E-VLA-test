"""Read-only service identity checks for camera paths under retained resource poolV3."""
import hashlib
import json
from pathlib import Path
ROOT=Path('/root/nfs/embodied/benchmark')
PLAN_SHA='bccfc4c41d630bb4c9609a67e0972a119abb0d529ad367760dfc8302be6b6ff0'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def service(common,root,manifest,run,registry,resource):
    result,identity,protocol,gate,cross=[read(run/name) for name in ('result.json','identity.json','protocol.json','gate.json','cross_service.json')]
    cid=run.parent.parent.name
    require(cid in manifest['planned_conditions'] and result.get('complete') is True,'Service not completed/planned')
    require(all(v.get('manifest_hash')==manifest['manifest_hash'] for v in (result,identity,protocol,gate)), 'Mixed service manifest')
    require(result['condition']==identity['condition']==protocol['condition_id']==cid,'Mixed service condition')
    require(protocol['config']==common.planned_config(manifest,cid) and protocol['checkpoint_sha256']==manifest['checkpoints']['pi0']['sha256'],'Wrong service checkpoint/config')
    require(gate['passed'] and all(c['passed'] for c in gate['checks']) and cross['passed'] and all(c['passed'] for c in cross['checks']), 'Service numerical gate failed')
    qk=read(run/'qk_receipt.json')
    require(qk.get('passed') is True and qk.get('unsafe_legacy_kernel_executed') is False
        and qk.get('atol')==.0009765625 and qk.get('rtol')==.0078125,'QK gate changed')
    for key,tail in (('server_command','.server'),('worker_command','.worker')):
        require(common.MODULE+tail in identity[key], 'Wrong executable module')
    command=identity['worker_command'];options={command[i]:command[i+1] for i in range(len(command)-1) if command[i].startswith('--')}
    shard,shards=int(options['--shard-index']),int(options['--shards'])
    require(result.get('completed_after',result.get('already_completed'))==len(common.cases(manifest,common.stage_for(manifest),shard,shards)),'Service case-shard count differs')
    revision=identity['resource_execution_revision'];uuid=protocol['gpu_uuid']
    owner=next(r for r in registry['processes'] if r['gpu_uuid']==uuid)
    require(identity['controller_pid']==revision['controller_pid']==owner['pid'],'Wrong resource owner')
    require(revision['schema']=='ease-resource-pool-science-v3' and revision['scientific_manifest_hash']==manifest['manifest_hash']
        and revision['controller_source_sha256']==sha(ROOT/'preparations/ease_insights_20261103/resource_pool_v3/guardian.py')
        and revision['scientific_runtime_unchanged'] is False and revision['original_scientific_manifests_changed'] is False,'Resource revision identity differs')
    require(revision['resource_amendment']==dict(path=str(resource/'plan.json'),sha256=PLAN_SHA),'Resource plan differs')
    keeper=revision['reservoir_keeper'];staged=read(resource/'sessions'/uuid/'staging_ready.json')
    require(all(keeper[k]==staged[k] for k in ('gpu_uuid','pid','proc_start_ticks','nonce'))
        and keeper['allocated_bytes']==8192*1024**2 and keeper['context_created'] and keeper.get('error') is None,'Serving8GiB keeper is not the identity-bound staged context')
    proof=revision['holder_proof'];binding=proof['binding']
    expected_session=631907 if uuid=='GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1' else 631915
    expected_small=632159 if expected_session==631907 else 632160
    require(proof['scheduler']['pid']==600088 and proof['session']['pid']==expected_session
        and proof['keeper']['pid']==expected_small and proof['keeper_ready']['gpu_uuid']==uuid
        and binding['session']['pid']==expected_session and binding['lease_fds']==([4] if expected_session==631907 else [6])
        and revision['borrower_did_not_take_or_duplicate_lease'] and revision['original_scheduler_session_owns_lease'],'Original lease/holder binding differs')
    cost=read(run/'fixed_input_cost.json')
    require(cost['manifest_hash']==manifest['manifest_hash'] and cost['total_warmup_requests']==4*len(manifest['conditions'])*10
        and cost['total_measured_requests']==4*len(manifest['conditions'])*20,'Fixed-input cost probe size differs')
    return dict(condition=cid,run=str(run),shard=shard,shards=shards,controller_pid=owner['pid'],gpu_uuid=uuid,
        gate_checks=len(gate['checks']),cross_service_checks=len(cross['checks']),
        artifacts={n:sha(run/n) for n in ('result.json','identity.json','protocol.json','gate.json','cross_service.json','qk_receipt.json','fixed_input_cost.json')})


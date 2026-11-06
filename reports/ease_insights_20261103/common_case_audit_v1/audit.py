"""Read-only reconstruction of every accepted calibration/diversity case and service."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

ROOT=Path('/root/nfs/embodied/benchmark')
sys.path.insert(0,str(ROOT))
ALLOWED={'baseline_recovery_v1','vlm_diversity_v2'}
PLAN_SHA='2dd03053395d180605f5907960c04afdd05c71eb7eedb1131ad46e225733aff8'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def case(common,worker,root,manifest,cid,scene):
    require(worker.completed(root,cid,scene,manifest['manifest_hash']),'Missing/changed accepted case')
    directory=worker.episode_dir(root,cid,scene);path=directory/'episode_receipt.json';original=read(path)
    reference=root/'initial_observations'/scene['suite']/f"task{scene['task']:02d}"/f"init{scene['init']:02d}.json"
    require(reference.exists() and Path(original['initial_gate']['reference'])==reference,'Initial reference absent/escaped')
    before=sha(path);writes=[];saved=worker.atomic_json
    # Reuse the full numerical/route/action/physical validator without rewriting
    # the receipt or changing its recorded worker PID. Initial references exist.
    worker.atomic_json=lambda destination,value:writes.append((Path(destination),value))
    try: rebuilt=worker.finalize(root,cid,scene,manifest)
    finally:worker.atomic_json=saved
    require(len(writes)==1 and writes[0][0]==path and writes[0][1]==rebuilt,'Unexpected validator write intent')
    # JSON object keys (e.g. the exported NFE Counter) are strings on disk.
    # Normalize this representation without dropping or rounding any field.
    rebuilt=json.loads(json.dumps(rebuilt))
    ignore={'worker_pid'}
    require({k:v for k,v in original.items() if k not in ignore}=={k:v for k,v in rebuilt.items() if k not in ignore},'Reconstructed receipt differs')
    require(sha(path)==before,'Read-only reconstruction changed a receipt')
    return dict(condition=cid,case=scene,receipt=str(path),sha256=before,success=original['success'],predictions=original['predictions'])


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
    require(revision['schema']=='ease-resource-pool-science-v2' and revision['scientific_manifest_hash']==manifest['manifest_hash']
        and revision['controller_source_sha256']==sha(ROOT/'preparations/ease_insights_20261103/resource_pool_v2/guardian.py')
        and revision['scientific_runtime_unchanged'] is False and revision['original_scientific_manifests_changed'] is False,'Resource revision identity differs')
    require(revision['resource_amendment']==dict(path=str(resource/'plan.json'),sha256=PLAN_SHA),'Resource plan differs')
    keeper=revision['reservoir_keeper'];staged=read(resource/'sessions'/uuid/'staging_ready.json')
    require(all(keeper[k]==staged[k] for k in ('gpu_uuid','pid','proc_start_ticks','nonce'))
        and keeper['allocated_bytes']==8192*1024**2 and keeper['context_created'] and keeper.get('error') is None,'Serving keeper is not the staged8GiB context')
    proof=revision['holder_proof'];binding=proof['binding']
    expected_session=631907 if uuid=='GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1' else 631915
    expected_small=632159 if expected_session==631907 else 632160
    require(proof['scheduler']['pid']==600088 and proof['session']['pid']==expected_session
        and proof['keeper']['pid']==expected_small and proof['keeper_ready']['gpu_uuid']==uuid
        and binding['session']['pid']==expected_session and binding['lease_fds']==([4] if expected_session==631907 else [6])
        and revision['borrower_did_not_take_or_duplicate_lease'] and revision['original_scheduler_session_owns_lease'],'Original lease/holder binding differs')
    cost=read(run/'fixed_input_cost.json')
    require(cost['manifest_hash']==manifest['manifest_hash'] and cost['total_warmup_requests']==4*len(manifest['conditions'])*6
        and cost['total_measured_requests']==4*len(manifest['conditions'])*12,'Fixed-input cost probe size differs')
    return dict(condition=cid,run=str(run),shard=shard,shards=shards,controller_pid=owner['pid'],gpu_uuid=uuid,
        gate_checks=len(gate['checks']),cross_service_checks=len(cross['checks']),
        artifacts={n:sha(run/n) for n in ('result.json','identity.json','protocol.json','gate.json','cross_service.json','qk_receipt.json','fixed_input_cost.json')})


def audit(root,cohort,output):
    require(cohort in ALLOWED,'Unapproved cohort')
    common=importlib.import_module('preparations.ease_insights_20261103.'+cohort+'.common')
    worker=importlib.import_module('preparations.ease_insights_20261103.'+cohort+'.worker')
    manifest=common.read_manifest(root/'manifest.json',verify_sources=True);scenes=common.cases(manifest)
    resource=ROOT/'results/ease_insights_20261103/resource_pool_v2'
    require(sha(resource/'plan.json')==PLAN_SHA,'Resource plan changed')
    registry=read(resource/'launches.json')
    cases=[];services=[];failures=[]
    for cid in manifest['planned_conditions']:
        for scene in scenes:
            try:cases.append(case(common,worker,root,manifest,cid,scene))
            except Exception as error:failures.append(dict(condition=cid,case=scene,error=repr(error)))
    for run in sorted(root.glob('*/runs/*')):
        if not (run/'result.json').exists():
            failures.append(dict(run=str(run),error='No completed service receipt'));continue
        try:services.append(service(common,root,manifest,run,registry,resource))
        except Exception as error:failures.append(dict(run=str(run),error=repr(error)))
    planned={(cid,scene['suite'],scene['task'],scene['init']) for cid in manifest['planned_conditions'] for scene in scenes}
    for receipt in root.glob('*/episodes/*/task*/init*/episode_receipt.json'):
        if not receipt.parent.name[4:].isdigit():continue
        row=read(receipt);s=row['case']
        require((row['condition'],s['suite'],s['task'],s['init']) in planned,'Unexpected scientific case')
    expected=len(planned)
    if len(cases)!=expected or len(services)!=2*len(manifest['planned_conditions']):
        failures.append(dict(error='Incomplete case or service coverage',expected_cases=expected,actual_cases=len(cases),services=len(services)))
    value=dict(status='pass' if not failures else 'fail',cohort=cohort,manifest_hash=manifest['manifest_hash'],
        expected_cases=expected,cases=len(cases),requests=sum(c['predictions'] for c in cases),
        completed_services=len(services),failures=failures,case_evidence=cases,service_evidence=services,
        auditor_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),time=time.time(),
        scientific_artifacts_modified=False,source_files_checked=len(manifest['source_files']),
        scope='Complete read-only case reconstruction plus completed-service numerical/resource identities; no exclusive-GPU timing or historical live-PID claim')
    common.atomic_json(output,value)
    return {k:v for k,v in value.items() if k not in ('case_evidence','service_evidence')}


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--cohort',choices=sorted(ALLOWED),required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();value=audit(args.root.resolve(),args.cohort,args.output.resolve());print(json.dumps(value));raise SystemExit(0 if value['status']=='pass' else 1)

"""Read-only acceptance:240 new camera cases,80 original Full controls,six services."""
import importlib.util
import json
from pathlib import Path
import sys
import time
ROOT=Path('/root/nfs/embodied/benchmark');sys.path.insert(0,str(ROOT))
from preparations.ease_insights_20261103.camera_paths_v3 import common,worker
from preparations.ease_insights_20261103.camera_paths_v3.cost_records import cost_records
from preparations.ease_insights_20261103.vlm_diversity_v2 import common as old,worker as old_worker
from reports.ease_insights_20261103.camera_paths_v3.service_audit import service


def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def audit():
    root=ROOT/'results/ease_insights_20261103/camera_paths_v3/discovery'
    receipts=[p for p in root.glob('*/episodes/**/episode_receipt.json') if p.parent.name[4:].isdigit() and p.parent.name.startswith('init')]
    require(len(receipts)==240 and len(list(root.glob('*/discovery_shard*of02_complete.json')))==6,
        'Await all240 new cases and six completed shards before reading outcomes')
    m=common.read_manifest(root/'manifest.json',verify_sources=True);previous=old.read_manifest(common.REFERENCE_ROOT/'manifest.json',verify_sources=True)
    require(common.sha(common.REFERENCE_ROOT/'manifest.json')==common.REFERENCE_MANIFEST_SHA,'Original Full manifest changed')
    tool=ROOT/'reports/ease_insights_20261103/common_case_audit_v1/audit.py'
    require(common.sha(tool)=='2fec7afe6e80cca6286e52a467d42be9e525f6125da76d0b63873985b4a6d113','Accepted case validator changed')
    spec=importlib.util.spec_from_file_location('camera_case_validator',tool);validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    scenes=common.cases(m);owned=[];preserved=[];services=[];failures=[];pairing={}
    def check(case_common,case_worker,case_root,manifest,cid,scene,new):
        try:
            value=validator.case(case_common,case_worker,case_root,manifest,cid,scene)
            value.update(source_condition=cid,source_manifest_hash=manifest['manifest_hash'],new_case=new)
            initial=read(case_worker.episode_dir(case_root,cid,scene)/'episode_receipt.json')['initial_gate']
            k=scene['suite'],scene['task'],scene['init'];fields={f:initial[f] for f in ('input_hash','seed','physical_assets')}
            if k in pairing:require(pairing[k]==fields,'Original/new physical first input or noise differs')
            else:pairing[k]=fields
            (owned if new else preserved).append(value)
        except Exception as error:failures.append(dict(condition=cid,case=scene,new_case=new,error=repr(error)))
    for cid in m['planned_conditions']:
        for scene in scenes:check(common,worker,root,m,cid,scene,True)
    for scene in scenes:check(old,old_worker,common.REFERENCE_ROOT,previous,'full_d10',scene,False)
    resource=ROOT/'results/ease_insights_20261103/resource_pool_v4';registry=read(resource/'launches.json')
    for run in sorted(root.glob('*/runs/*')):
        try:
            value=service(common,root,m,run,registry,resource);gate=read(run/'gate.json')
            require(all(gate.get(f) is True for f in ('native_no_AE_bypass_parity_passed','actual_AE_visits_and_NFE_passed','actual_single_view_batch_passed','nontrivial_mixed_view_oracle_passed')),
                'Missing camera/native arithmetic gates');services.append(value)
        except Exception as error:failures.append(dict(run=str(run),error=repr(error)))
    costs=cost_records(root,m)
    if costs['valid_services']!=6 or costs['excluded_or_pending']:failures.append(dict(error='Fixed-input cost calls fail native AE/camera/schedule validation',costs=costs))
    if (len(owned),len(preserved),len(services),len(pairing))!=(240,80,6,80):
        failures.append(dict(error='Incomplete required coverage',new=len(owned),preserved=len(preserved),services=len(services),paired_cases=len(pairing)))
    value=dict(status='pass' if not failures else 'fail',manifest_hash=m['manifest_hash'],new_cases=len(owned),preserved_control_cells=len(preserved),
        preserved_controls_are_new_replications=False,new_requests=sum(c['predictions'] for c in owned),preserved_recorded_requests=sum(c['predictions'] for c in preserved),
        completed_new_services=len(services),failures=failures,new_case_evidence=owned,preserved_case_evidence=preserved,service_evidence=services,
        actual_AE_NFE_and_camera_batches_validated_for_new_cases_and_cost_calls=not failures,preserved_old_AE_internal_counters_observed=False,
        scientific_records_modified=False,auditor_source=dict(path=str(Path(__file__).resolve()),sha256=common.sha(__file__)),time=time.time(),
        scope='Complete read-only case reconstruction, original Full identity/physical pairing, actual AE18/NFE and vision batch/cache checks; completed-service numerical/cost/resource identities')
    common.atomic_json(Path(__file__).with_name('discovery_acceptance.json'),value)
    return {k:v for k,v in value.items() if k not in ('new_case_evidence','preserved_case_evidence','service_evidence')}


if __name__=='__main__':
    result=audit();print(json.dumps(result));raise SystemExit(0 if result['status']=='pass' else 1)

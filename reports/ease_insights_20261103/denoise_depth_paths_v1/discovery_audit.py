"""Read-only reconstruction of320 new depth/time records and160 two-origin controls."""
import importlib.util
import json
from pathlib import Path
import sys
import time
ROOT=Path('/root/nfs/embodied/benchmark');sys.path.insert(0,str(ROOT))
from preparations.ease_insights_20261103.denoise_depth_paths_v1 import common,worker
from preparations.ease_insights_20261103.denoise_depth_paths_v1.cost_records import cost_records
from reports.ease_insights_20261103.denoise_depth_paths_v1.service_audit import service


def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def audit():
    root=ROOT/'results/ease_insights_20261103/denoise_depth_paths_v1/discovery'
    receipts=[p for p in root.glob('*/episodes/**/episode_receipt.json') if p.parent.name.startswith('init') and p.parent.name[4:].isdigit()]
    require(len(receipts)==320 and len(list(root.glob('*/discovery_shard*of02_complete.json')))==8,'Await all320 cases and eight completed shards')
    m=common.read_manifest(root/'manifest.json',verify_sources=True);tool=ROOT/'reports/ease_insights_20261103/common_case_audit_v1/audit.py'
    require(common.sha(tool)=='2fec7afe6e80cca6286e52a467d42be9e525f6125da76d0b63873985b4a6d113','Accepted reconstruction changed')
    spec=importlib.util.spec_from_file_location('depth_time_case_validator',tool);validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    owned=[];preserved=[];services=[];failures=[];paired={};scenes=common.cases(m)
    def check(cm,w,origin,manifest,cid,alias,scene,new):
        try:
            value=validator.case(cm,w,origin,manifest,cid,scene)
            value.update(analysis_condition=alias,source_condition=cid,source_manifest_hash=manifest['manifest_hash'],new_case=new)
            initial=read(w.episode_dir(origin,cid,scene)/'episode_receipt.json')['initial_gate'];key=scene['suite'],scene['task'],scene['init']
            fields={k:initial[k] for k in ('input_hash','seed','physical_assets')}
            if key in paired:require(paired[key]==fields,'Two-origin/new physical first input or noise differs')
            else:paired[key]=fields
            (owned if new else preserved).append(value)
        except Exception as error:failures.append(dict(condition=alias,source_condition=cid,case=scene,new_case=new,error=repr(error)))
    for cid in m['planned_conditions']:
        for scene in scenes:check(common,worker,root,m,cid,cid,scene,True)
    for alias in common.REFERENCES:
        origin,old,cm,w,cid=common.reference(alias,verify_sources=True)
        for scene in scenes:check(cm,w,origin,old,cid,alias,scene,False)
    resource=ROOT/'results/ease_insights_20261103/resource_pool_v4';registry=read(resource/'launches.json')
    for run in sorted(root.glob('*/runs/*')):
        try:
            value=service(common,root,m,run,registry,resource);gate=read(run/'gate.json')
            require(all(gate.get(k) is True for k in ('original_adapter_no_AE_bypass_parity_passed','actual_AE_visits_and_NFE_passed','exact_D2_D3_D4_D10_node_receipts_passed')),'Missing adapter/depth/time gate')
            services.append(value)
        except Exception as error:failures.append(dict(run=str(run),error=repr(error)))
    cost=cost_records(root,m)
    if cost['valid_services']!=8 or cost['excluded_or_pending']:failures.append(dict(error='Invalid fixed-input cost calls',costs=cost))
    if (len(owned),len(preserved),len(services),len(paired))!=(320,160,8,80):failures.append(dict(error='Incomplete coverage',new=len(owned),preserved=len(preserved),services=len(services),paired_cases=len(paired)))
    value=dict(status='pass' if not failures else 'fail',manifest_hash=m['manifest_hash'],new_cases=len(owned),preserved_control_cells=len(preserved),
        preserved_controls_are_new_replications=False,new_requests=sum(c['predictions'] for c in owned),preserved_recorded_requests=sum(c['predictions'] for c in preserved),
        completed_new_services=len(services),failures=failures,new_case_evidence=owned,preserved_case_evidence=preserved,service_evidence=services,
        actual_AE_NFE_and_layer_visits_validated_for_new_cases_and_cost_calls=not failures,
        scientific_records_modified=False,auditor_source=dict(path=str(Path(__file__).resolve()),sha256=common.sha(__file__)),time=time.time(),
        scope='Complete new-record and unchanged two-origin control reconstruction,physical/noise pairing,actual AE/NFE/time nodes,and completed-service numerical/cost/resource identities')
    common.atomic_json(Path(__file__).with_name('discovery_acceptance.json'),value)
    return {k:v for k,v in value.items() if k not in ('new_case_evidence','preserved_case_evidence','service_evidence')}


if __name__=='__main__':
    value=audit();print(json.dumps(value));raise SystemExit(0 if value['status']=='pass' else 1)

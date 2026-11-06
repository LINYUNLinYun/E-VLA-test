"""Preparation: accept360 genuinely fresh confirmation records, with no reused quality cells."""
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path('/root/nfs/embodied/benchmark');sys.path.insert(0,str(ROOT))
from preparations.ease_insights_20261103.camera_paths_v3 import common,worker
from preparations.ease_insights_20261103.camera_paths_v3.cost_records import cost_records
from reports.ease_insights_20261103.camera_paths_v3.service_audit import service as verify_service


def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def audit():
    root=ROOT/'results/ease_insights_20261103/camera_paths_v3/confirmation'
    m=common.read_manifest(root/'manifest.json',verify_sources=True)
    require(common.stage_for(m)=='confirm' and m['analytical_conditions']==m['planned_conditions']
        and 'reference_conditions' not in m and m['confirmation_inits']==[47,48,49],'Confirmation may not reuse old quality cells')
    receipts=[p for p in root.glob('*/episodes/**/episode_receipt.json') if p.parent.name[4:].isdigit()]
    require(len(receipts)==360 and len(list(root.glob('*/confirm_shard*of02_complete.json')))==6,
            'Await all360 cases and6 completed shards')
    tool=ROOT/'reports/ease_insights_20261103/common_case_audit_v1/audit.py'
    require(common.sha(tool)=='2fec7afe6e80cca6286e52a467d42be9e525f6125da76d0b63873985b4a6d113','Accepted case validator changed')
    spec=importlib.util.spec_from_file_location('module_confirmation_case_validator',tool);validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    cases=[];services=[];failures=[];paired={}
    for cid in m['planned_conditions']:
        for scene in common.cases(m):
            try:
                value=validator.case(common,worker,root,m,cid,scene)
                initial=read(worker.episode_dir(root,cid,scene)/'episode_receipt.json')['initial_gate']
                key=(scene['suite'],scene['task'],scene['init']);identity={k:initial[k] for k in ('input_hash','seed','physical_assets')}
                if key in paired:require(paired[key]==identity,'Confirmation physical input/noise differs across paths')
                else:paired[key]=identity
                cases.append(value)
            except Exception as error:failures.append(dict(condition=cid,case=scene,error=repr(error)))
    resource=ROOT/'results/ease_insights_20261103/resource_pool_v4';registry=read(resource/'launches.json')
    for run in sorted(root.glob('*/runs/*')):
        try:
            checked=verify_service(common,root,m,run,registry,resource)
            gate=read(run/'gate.json')
            require(all(gate.get(f) is True for f in ('native_no_AE_bypass_parity_passed','actual_AE_visits_and_NFE_passed','actual_single_view_batch_passed','nontrivial_mixed_view_oracle_passed')),
                'Missing camera/native execution gate')
            services.append(checked)
        except Exception as error:failures.append(dict(run=str(run),error=repr(error)))
    costs=cost_records(root,m)
    if costs['valid_services']!=6 or costs['excluded_or_pending']:failures.append(dict(error='Confirmation native AE/camera cost calls invalid',costs=costs))
    if (len(cases),len(services),len(paired))!=(360,6,120):failures.append(dict(error='Incomplete coverage',cases=len(cases),services=len(services),paired_cases=len(paired)))
    value=dict(status='pass' if not failures else 'fail',manifest_hash=m['manifest_hash'],new_cases=len(cases),preserved_control_cells=0,
        cases=len(cases),completed_services=len(services),requests=sum(c['predictions'] for c in cases),failures=failures,
        case_evidence=cases,service_evidence=services,scientific_records_modified=False,
        auditor_source=dict(path=str(Path(__file__).resolve()),sha256=common.sha(__file__)),time=time.time(),
        scope='Complete fresh47-49 confirmation record reconstruction, actual AE/NFE, physical/noise pairing and service identities; no reused quality outcomes')
    common.atomic_json(Path(__file__).with_name('confirmation_acceptance.json'),value)
    return {k:v for k,v in value.items() if k not in ('case_evidence','service_evidence')}


if __name__=='__main__':
    value=audit();print(json.dumps(value));raise SystemExit(0 if value['status']=='pass' else 1)

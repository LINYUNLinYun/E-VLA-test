"""Read-only acceptance of320 new cases,160 unchanged controls and eight services."""
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT=Path('/root/nfs/embodied/benchmark');sys.path.insert(0,str(ROOT))
from preparations.ease_insights_20261103.module_tradeoff_v1 import common,worker
from preparations.ease_insights_20261103.vlm_diversity_v2 import common as old,worker as old_worker


def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def audit():
    root=ROOT/'results/ease_insights_20261103/module_tradeoff_v1/discovery'
    current_receipts=[p for p in root.glob('*/episodes/**/episode_receipt.json') if p.parent.name.startswith('init') and p.parent.name[4:].isdigit()]
    require(len(list(root.glob('*/discovery_shard*of02_complete.json')))==8
        and len(current_receipts)==320,
        'Await all320 new cases and8 completed shards before reading outcomes')
    m=common.read_manifest(root/'manifest.json',verify_sources=True)
    require(common.sha(root/'manifest.json')=='80ea5dfd59fc51f90aa045a63b7c47f9ca66475da612eecb194e709429586e02','Module manifest changed')
    previous=old.read_manifest(common.REFERENCE_ROOT/'manifest.json',verify_sources=True)
    require(common.sha(common.REFERENCE_ROOT/'manifest.json')==common.REFERENCE_MANIFEST_SHA,'Control source manifest changed')
    tool=ROOT/'reports/ease_insights_20261103/common_case_audit_v1/audit.py'
    require(common.sha(tool)=='2fec7afe6e80cca6286e52a467d42be9e525f6125da76d0b63873985b4a6d113','Accepted reconstruction validator changed')
    spec=importlib.util.spec_from_file_location('accepted_module_case_validator',tool);validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    scenes=common.cases(m);owned=[];preserved=[];services=[];failures=[];pairing={}
    def check(case_common,case_worker,case_root,manifest,cid,alias,scene,new):
        try:
            value=validator.case(case_common,case_worker,case_root,manifest,cid,scene)
            value.update(analysis_condition=alias,source_condition=cid,source_manifest_hash=manifest['manifest_hash'],new_case=new)
            path=case_worker.episode_dir(case_root,cid,scene)/'episode_receipt.json'
            initial=read(path)['initial_gate'];key=(scene['suite'],scene['task'],scene['init'])
            fields={k:initial[k] for k in ('input_hash','seed','physical_assets')}
            if key in pairing:require(pairing[key]==fields,'Preserved/new physical input or noise mismatch')
            else:pairing[key]=fields
            (owned if new else preserved).append(value)
        except Exception as error:failures.append(dict(condition=alias,source_condition=cid,case=scene,new_case=new,error=repr(error)))
    for cid in m['planned_conditions']:
        for scene in scenes:check(common,worker,root,m,cid,cid,scene,True)
    for alias,cid in m['reference_conditions'].items():
        for scene in scenes:check(old,old_worker,common.REFERENCE_ROOT,previous,cid,alias,scene,False)
    resource=ROOT/'results/ease_insights_20261103/resource_pool_v2';registry=read(resource/'launches.json')
    for run in sorted(root.glob('*/runs/*')):
        try:
            value=validator.service(common,root,m,run,registry,resource)
            gate=read(run/'gate.json');require(gate['native_no_AE_bypass_parity_passed'] and gate['actual_AE_visits_and_NFE_passed'],'Missing module execution/numerical gates')
            # Validate the measured benchmark calls too, including the two
            # preserved algorithm definitions under the new instrumentation.
            costs=read(run/'fixed_input_cost.json')
            for suite in costs['suites']:
                for block in suite['conditions']:
                    cfg=next(c for c in m['conditions'] if c['id']==block['condition'])
                    for row in block['warmup']+block['measured']:
                        refresh=cfg['id']=='full_d10' or row['round']%3==0
                        skip=[] if refresh else cfg['ae_skip'];nfe=10 if refresh else 2
                        timing=row['model_timing'];audit=timing['ae_execution_audit']
                        worker.validate_ae(dict(timing=timing,route=dict(ae_execution_audit=audit,ae_bypassed_layers=skip)),skip,nfe)
            services.append(value)
        except Exception as error:failures.append(dict(run=str(run),error=repr(error)))
    if (len(owned),len(preserved),len(services),len(pairing))!=(320,160,8,80):
        failures.append(dict(error='Incomplete required coverage',new=len(owned),preserved=len(preserved),services=len(services),paired_cases=len(pairing)))
    value=dict(status='pass' if not failures else 'fail',manifest_hash=m['manifest_hash'],new_cases=len(owned),preserved_control_cells=len(preserved),
        preserved_controls_are_new_replications=False,new_requests=sum(c['predictions'] for c in owned),
        preserved_recorded_requests=sum(c['predictions'] for c in preserved),completed_new_services=len(services),
        failures=failures,new_case_evidence=owned,preserved_case_evidence=preserved,service_evidence=services,
        actual_AE_NFE_and_layer_visits_validated_for_new_cases_and_cost_calls=not failures and len(owned)==320 and len(services)==8,
        preserved_old_AE_internal_counters_observed=False,
        scientific_records_modified=False,auditor_source=dict(path=str(Path(__file__).resolve()),sha256=common.sha(__file__)),time=time.time(),
        scope='Complete new-case reconstruction plus unchanged-source reference reconstruction, physical/noise pairing and new-service numerical/cost/resource identities')
    path=Path(__file__).with_name('discovery_acceptance.json');common.atomic_json(path,value)
    return {k:v for k,v in value.items() if k not in ('new_case_evidence','preserved_case_evidence','service_evidence')}


if __name__=='__main__':
    result=audit();print(json.dumps(result));raise SystemExit(0 if result['status']=='pass' else 1)

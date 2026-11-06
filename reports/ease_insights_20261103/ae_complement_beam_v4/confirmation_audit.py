"""Read-only reconstruction of720 rollouts, including cross-stream physical pairing."""
import importlib.util
import json
from pathlib import Path
import sys
import time
ROOT=Path('/root/nfs/embodied/benchmark');sys.path.insert(0,str(ROOT))
from preparations.ease_insights_20261103.ae_complement_beam_v4 import common,worker
from preparations.ease_insights_20261103.ae_complement_beam_v4.noise import model_seed
from preparations.ease_insights_20261103.ae_complement_beam_v4.prepare_confirmation import read_bundle,ROOTS,BUNDLE
from preparations.ease_insights_20261103.ae_complement_beam_v4.cost_records import cost_records
from reports.ease_insights_20261103.ae_complement_beam_v4.service_audit import service
REPORT=Path(__file__).resolve().parent


def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def audit():
    bundle=read_bundle(verify_sources=True)
    tool=ROOT/'reports/ease_insights_20261103/common_case_audit_v1/audit.py'
    require(common.sha(tool)=='2fec7afe6e80cca6286e52a467d42be9e525f6125da76d0b63873985b4a6d113','Accepted case validator changed')
    spec=importlib.util.spec_from_file_location('cross_module_v4_case_validator',tool);validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    resource=ROOT/'results/ease_insights_20261103/resource_pool_v4';registry=read(resource/'launches.json')
    accepted=[];services=[];failures=[];paired={};physical={};stream_costs=[]
    for repeat,root in enumerate(ROOTS):
        m=common.read_manifest(root/'manifest.json',verify_sources=True)
        require(m['noise_repeat']==repeat and common.stage_for(m)=='confirm' and 'reference_conditions' not in m,'Wrong independent confirmation identity')
        receipts=[p for p in root.glob('*/episodes/*/task*/init*/episode_receipt.json') if p.parent.name[4:].isdigit()]
        markers=list(root.glob('*/confirm_shard*of02_complete.json'))
        require(len(receipts)==360 and len(markers)==6,'Await all360 cases and6 completed shards in BOTH noise streams')
        expected={worker.episode_dir(root,cid,scene)/'episode_receipt.json' for cid in common.CONDITIONS for scene in common.cases(m)}
        require(set(receipts)==expected,'Missing or unexpected fresh outcome paths')
        for marker in markers:
            value=read(marker);require(value['complete'] and value['manifest_hash']==m['manifest_hash'],'Wrong completed shard manifest')
        for cid in common.CONDITIONS:
            for scene in common.cases(m):
                try:
                    value=validator.case(common,worker,root,m,cid,scene)
                    receipt=read(value['receipt']);initial=receipt['initial_gate'];key=(scene['suite'],scene['task'],scene['init'])
                    identity={k:initial[k] for k in ('input_hash','seed','physical_assets')}
                    require(initial['seed']==model_seed(repeat,scene['suite'],scene['task'],scene['init'],0),'Wrong fixed model-noise stream')
                    if (repeat,key) in paired:require(paired[repeat,key]==identity,'Conditions differ in initial input/noise/physical identity')
                    else:paired[repeat,key]=identity
                    shared={k:identity[k] for k in ('input_hash','physical_assets')}
                    if key in physical:require(physical[key]==shared,'Noise streams do not share the same untouched physical state/input')
                    else:physical[key]=shared
                    accepted.append(dict(value,noise_repeat=repeat))
                except Exception as error:failures.append(dict(noise_repeat=repeat,condition=cid,case=scene,error=repr(error)))
        for run in sorted(root.glob('*/runs/*')):
            try:
                value=service(common,root,m,run,registry,resource);gate=read(run/'gate.json')
                require(all(gate.get(k) is True for k in ('native_no_AE_bypass_parity_passed','actual_AE_visits_and_NFE_passed')),'Missing mixed-depth/time native numerical gate')
                services.append(dict(value,noise_repeat=repeat))
            except Exception as error:failures.append(dict(noise_repeat=repeat,run=str(run),error=repr(error)))
        costs=cost_records(root,m);stream_costs.append(dict(noise_repeat=repeat,records=costs))
        if costs['valid_services']!=6 or costs['excluded_or_pending']:failures.append(dict(noise_repeat=repeat,error='Need6 complete validated native cost services',costs=costs))
    for key in physical:
        if (0,key) in paired and (1,key) in paired:
            require(paired[0,key]['seed']!=paired[1,key]['seed'],'Two predetermined streams unexpectedly share initial model seed')
    if (len(accepted),len(services),len(paired),len(physical))!=(720,12,240,120):
        failures.append(dict(error='Incomplete confirmation coverage',cases=len(accepted),services=len(services),paired=len(paired),physical_states=len(physical)))
    value=dict(status='pass' if not failures else 'fail',bundle_hash=bundle['bundle_hash'],
        manifest_hashes=[s['manifest_hash'] for s in bundle['streams']],new_cases=len(accepted),preserved_control_cells=0,
        rollouts_per_algorithm=240,unique_physical_states_per_algorithm=len(physical),paired_noise_state_cells=len(paired),
        completed_services=len(services),requests=sum(c['predictions'] for c in accepted),failures=failures,
        case_evidence=accepted,service_evidence=services,cost_records=stream_costs,
        bundle_evidence=dict(path=str(BUNDLE),sha256=common.sha(BUNDLE)),
        auditor_source=dict(path=str(Path(__file__).resolve()),sha256=common.sha(__file__)),time=time.time(),
        scientific_records_modified=False,
        scope='All720 fresh47-49 rollouts reconstructed, actual VLM/AE/NFE and action receipts checked;120 unique physical states per algorithm and two fixed noise streams are explicitly separated')
    common.atomic_json(REPORT/'confirmation_acceptance.json',value)
    return {k:v for k,v in value.items() if k not in ('case_evidence','service_evidence','cost_records')}


if __name__=='__main__':
    value=audit();print(json.dumps(value));raise SystemExit(0 if value['status']=='pass' else 1)

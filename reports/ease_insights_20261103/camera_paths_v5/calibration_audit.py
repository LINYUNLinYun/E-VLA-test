"""Accept8 repeated native controls and real packed-prefix GPU gates."""
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np
ROOT=Path('/root/nfs/embodied/benchmark');sys.path.insert(0,str(ROOT))
from preparations.ease_insights_20261103.camera_paths_v5 import common,worker
from preparations.ease_insights_20261103.camera_paths_v5.cost_records import cost_records
from reports.ease_insights_20261103.camera_paths_v5.service_audit import service


def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def main():
    root=common.FAMILY_ROOT/'calibration';m=common.read_manifest(root/'manifest.json',verify_sources=True)
    require(len(list(root.glob('*/calibration_shard*of02_complete.json')))==4,'Await four completed control services')
    tool=ROOT/'reports/ease_insights_20261103/common_case_audit_v1/audit.py'
    require(common.sha(tool)=='2fec7afe6e80cca6286e52a467d42be9e525f6125da76d0b63873985b4a6d113','Case reconstruction tool changed')
    spec=importlib.util.spec_from_file_location('fresh_camera_control_case_validator',tool);validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    cases=[validator.case(common,worker,root,m,cid,scene) for cid in m['planned_conditions'] for scene in common.cases(m)]
    resource=ROOT/'results/ease_insights_20261103/resource_pool_v4';registry=read(resource/'launches.json')
    services=[service(common,root,m,run,registry,resource) for run in sorted(root.glob('*/runs/*'))]
    require(len(cases)==8 and len(services)==4,'Incomplete native controls')
    for accepted in services:
        gate=read(Path(accepted['run'])/'gate.json')
        require(all(gate.get(k) is True for k in ['original_vision_parity_passed','packed_prefix_identity_passed','independent_masked_prefix_reference_passed','actual_compact_prefix_and_original_RoPE_passed','native_no_AE_bypass_parity_passed','actual_AE_visits_and_NFE_passed']),'Real numeric/vision/native gate absent')
    costs=cost_records(root,m);require(costs['valid_services']==4 and not costs['excluded_or_pending'],'Native camera/NFE cost records invalid')
    ratios={c:statistics.median(s['full_over_policy_cycle'][c] for s in costs['services']) for c in common.CONDITIONS}
    candidates={c:ratios[c] for c in common.NEW_CONDITIONS};matched=max(candidates.values())<=1.10*min(candidates.values())
    joint=dict(median_same_service_ratios=ratios,pair_cost_factor=max(candidates.values())/min(candidates.values()),pair_matched=matched,completed_services=4)
    joint['pass']=matched and all(v>=1.15 for v in candidates.values())
    bridges=[]
    for alias in m['planned_conditions']:
        old_root,old,cm,w,cid=common.reference(alias,verify_sources=True)
        for scene in common.cases(m):
            before_case=validator.case(cm,w,old_root,old,cid,scene);source=w.episode_dir(old_root,cid,scene);destination=worker.episode_dir(root,alias,scene)
            before,after=read(source/'episode_receipt.json'),read(destination/'episode_receipt.json')
            require(all(before[k]==after[k] for k in ['success','executed_steps','predictions']),'Known native control outcomes or trajectory lengths differ')
            require(all(before['initial_gate'][k]==after['initial_gate'][k] for k in ['case','input_hash','seed','physical_assets']),'Known control physical input/noise differs')
            x=[json.loads(line) for line in (source/'infer.jsonl').read_text().splitlines()];y=[json.loads(line) for line in (destination/'infer.jsonl').read_text().splitlines()]
            require(len(x)==len(y) and all(a['input_hash']==b['input_hash'] and a['model_noise_sha256']==b['model_noise_sha256'] for a,b in zip(x,y)),'Known control per-request inputs/noise differ')
            with np.load(source/'arrays.npz',allow_pickle=False) as a,np.load(destination/'arrays.npz',allow_pickle=False) as b:
                require(set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files),'Known native raw/decoded action arrays are not bitwise identical')
            bridges.append(dict(condition=alias,source_condition=cid,case=scene,original_case=before_case,known_control=dict(path=str(destination/'episode_receipt.json'),sha256=common.sha(destination/'episode_receipt.json')),bitwise_actions_and_raw_identical=True))
    value=dict(status='pass',time=time.time(),manifest_hash=m['manifest_hash'],known_controls=8,independent_quality_cases=0,
        original_vision_parity_passed=True,packed_prefix_identity_passed=True,independent_masked_prefix_reference_passed=True,actual_compact_prefix_and_original_RoPE_passed=True,
        old_Full_bridge_passed=True,old_DenseD2_bridge_passed=True,all_eight_controls_bitwise_identical=True,joint_cost_qualification=joint,
        cases=cases,services=services,bridges=bridges,cost_calls_validated=4*(96+192),cost_records=costs,
        source=dict(path=str(Path(__file__).resolve()),sha256=common.sha(__file__)),
        scope='Finite real prefix/numerical/dispatch gates,knownnativecontrolbridgeandfixed-inputcosts only;no newtask-category preference or independent quality evidence')
    common.atomic_json(Path(__file__).with_name('calibration_acceptance.json'),value)
    print(json.dumps({k:v for k,v in value.items() if k not in ['cases','services','bridges','cost_records']}))


if __name__=='__main__':main()

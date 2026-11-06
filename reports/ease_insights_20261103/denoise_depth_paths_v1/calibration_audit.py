"""Read-only acceptance of eight known Full/AE16 controls from two source roots."""
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path('/root/nfs/embodied/benchmark');sys.path.insert(0,str(ROOT))
from preparations.ease_insights_20261103.denoise_depth_paths_v1 import common,worker
from preparations.ease_insights_20261103.denoise_depth_paths_v1.cost_records import cost_records
from reports.ease_insights_20261103.denoise_depth_paths_v1.service_audit import service


def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def main():
    root=ROOT/'results/ease_insights_20261103/denoise_depth_paths_v1/calibration';m=common.read_manifest(root/'manifest.json',verify_sources=True)
    require(len(list(root.glob('*/calibration_shard*of02_complete.json')))==4,'Await four completed calibration shards')
    tool=ROOT/'reports/ease_insights_20261103/common_case_audit_v1/audit.py'
    require(common.sha(tool)=='2fec7afe6e80cca6286e52a467d42be9e525f6125da76d0b63873985b4a6d113','Accepted case reconstruction changed')
    spec=importlib.util.spec_from_file_location('depth_time_calibration_validator',tool);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    cases=[audit.case(common,worker,root,m,cid,s) for cid in m['planned_conditions'] for s in common.cases(m)]
    resource=ROOT/'results/ease_insights_20261103/resource_pool_v4';registry=read(resource/'launches.json')
    services=[service(common,root,m,run,registry,resource) for run in sorted(root.glob('*/runs/*'))]
    require(len(cases)==8 and len(services)==4,'Eight-control calibration incomplete')
    for accepted in services:
        gate=read(Path(accepted['run'])/'gate.json')
        require(all(gate.get(k) is True for k in ('original_adapter_no_AE_bypass_parity_passed','actual_AE_visits_and_NFE_passed','exact_D2_D3_D4_D10_node_receipts_passed')),'Missing adapter/depth/time numerical gates')
    costs=cost_records(root,m);require(costs['valid_services']==4 and not costs['excluded_or_pending'],'Calibration cost-call validation failed')
    bridges=[]
    for alias in m['planned_conditions']:
        original,old,old_common,old_worker,source_cid=common.reference(alias,verify_sources=True)
        for scene in common.cases(m):
            old_case=audit.case(old_common,old_worker,original,old,source_cid,scene)
            source=old_worker.episode_dir(original,source_cid,scene);target=worker.episode_dir(root,alias,scene)
            before,after=read(source/'episode_receipt.json'),read(target/'episode_receipt.json')
            require(all(before[k]==after[k] for k in ('success','executed_steps','predictions')),'Known control outcome or trajectory length changed')
            require(all(before['initial_gate'][k]==after['initial_gate'][k] for k in ('case','input_hash','seed','physical_assets')),'Control physical first input/noise differs')
            a=[json.loads(s) for s in (source/'infer.jsonl').read_text().splitlines()];b=[json.loads(s) for s in (target/'infer.jsonl').read_text().splitlines()]
            require(len(a)==len(b) and all(x['model_noise_sha256']==y['model_noise_sha256'] and x['input_hash']==y['input_hash'] for x,y in zip(a,b)),'Control per-request input/noise differs')
            with np.load(source/'arrays.npz',allow_pickle=False) as x,np.load(target/'arrays.npz',allow_pickle=False) as y:
                require(set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files),'Known control raw/actions are not bitwise identical')
            bridges.append(dict(analysis_condition=alias,source_condition=source_cid,source_manifest_hash=old['manifest_hash'],case=scene,
                original_receipt=old_case,known_control_receipt=dict(path=str(target/'episode_receipt.json'),sha256=common.sha(target/'episode_receipt.json')),bitwise_actions_and_raw_identical=True))
    value=dict(status='pass',manifest_hash=m['manifest_hash'],known_controls=8,independent_quality_cases=0,
        GPU_adapter_parity_passed=True,actual_AE_visits_and_NFE_passed=True,exact_D2_D3_D4_D10_node_receipts_passed=True,
        old_Full_bridge_passed=True,old_AE16_bridge_passed=True,all_eight_controls_bitwise_identical=True,cases=cases,services=services,bridges=bridges,
        cost_calls_validated=4*(144+288),cost_records=costs,auditor_source=dict(path=str(Path(__file__).resolve()),sha256=common.sha(__file__)),time=time.time(),
        scope='Finite original-adapter arithmetic, actual dispatch/cost receipts and repeated-control trajectory bridge only; no independent quality or complementary-path evidence')
    common.atomic_json(Path(__file__).with_name('calibration_acceptance.json'),value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('cases','services','bridges','cost_records')}))


if __name__=='__main__':main()

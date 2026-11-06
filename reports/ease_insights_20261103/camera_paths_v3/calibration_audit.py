"""Accept module calibration only with actual AE counters and preserved Full bridge."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path('/root/nfs/embodied/benchmark');sys.path.insert(0,str(ROOT))
from preparations.ease_insights_20261103.camera_paths_v3 import common,worker


def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def main():
    tool=ROOT/'reports/ease_insights_20261103/common_case_audit_v1/audit.py'
    require(common.sha(tool)=='2fec7afe6e80cca6286e52a467d42be9e525f6125da76d0b63873985b4a6d113','Accepted case validator changed')
    spec=importlib.util.spec_from_file_location('accepted_case_auditor',tool);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    root=ROOT/'results/ease_insights_20261103/camera_paths_v3/calibration'
    m=common.read_manifest(root/'manifest.json',verify_sources=True)
    cases=[audit.case(common,worker,root,m,'full_d10',scene) for scene in common.cases(m)]
    resource=ROOT/'results/ease_insights_20261103/resource_pool_v4';registry=read(resource/'launches.json')
    from reports.ease_insights_20261103.camera_paths_v3.service_audit import service
    services=[service(common,root,m,run,registry,resource) for run in sorted(root.glob('full_d10/runs/*'))]
    require(len(cases)==4 and len(services)==2,'Module calibration incomplete')
    for service in services:
        gate=read(Path(service['run'])/'gate.json')
        require(gate.get('native_no_AE_bypass_parity_passed') is True and gate.get('actual_AE_visits_and_NFE_passed') is True
            and gate.get('actual_single_view_batch_passed') is True and gate.get('nontrivial_mixed_view_oracle_passed') is True,'Missing GPU native/AE/camera parity gate')
    from preparations.ease_insights_20261103.vlm_diversity_v2 import common as old,worker as old_worker
    old_m=old.read_manifest(common.REFERENCE_ROOT/'manifest.json',verify_sources=True)
    bridges=[]
    for scene in common.cases(m):
        require(old_worker.completed(common.REFERENCE_ROOT,'full_d10',scene,old_m['manifest_hash']),'Preserved Full case invalid')
        source=old_worker.episode_dir(common.REFERENCE_ROOT,'full_d10',scene)
        target=worker.episode_dir(root,'full_d10',scene)
        prior,current=read(source/'episode_receipt.json'),read(target/'episode_receipt.json')
        require(prior['success']==current['success'] and prior['executed_steps']==current['executed_steps']
            and prior['predictions']==current['predictions'],'Known Full control outcome/length changed')
        require(all(prior['initial_gate'][k]==current['initial_gate'][k] for k in ('case','input_hash','seed','physical_assets')),'Full physical input/noise bridge changed')
        before=[json.loads(s) for s in (source/'infer.jsonl').read_text().splitlines()]
        after=[json.loads(s) for s in (target/'infer.jsonl').read_text().splitlines()]
        require(all(a['model_noise_sha256']==b['model_noise_sha256'] and a['input_hash']==b['input_hash'] for a,b in zip(before,after)),'Full per-request input/noise changed')
        with np.load(source/'arrays.npz',allow_pickle=False) as a,np.load(target/'arrays.npz',allow_pickle=False) as b:
            require(set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files),'Full action/raw arrays are not bitwise identical')
        bridges.append(dict(case=scene,source_receipt=dict(path=str(source/'episode_receipt.json'),sha256=common.sha(source/'episode_receipt.json')),
            new_receipt=dict(path=str(target/'episode_receipt.json'),sha256=common.sha(target/'episode_receipt.json')),bitwise_actions_and_raw_identical=True,requests=current['predictions']))
    value=dict(status='pass',manifest_hash=m['manifest_hash'],known_Full_controls=4,independent_quality_cases=0,
        GPU_native_parity_passed=True,actual_AE_visits_and_NFE_passed=True,actual_single_view_batch_passed=True,nontrivial_mixed_view_oracle_passed=True,old_Full_bridge_passed=True,
        all_four_controls_bitwise_identical=True,cases=cases,services=services,bridges=bridges,
        source_reference_manifest_hash=old_m['manifest_hash'],source_reference_manifest_sha256=common.sha(common.REFERENCE_ROOT/'manifest.json'),
        auditor_source=dict(path=str(Path(__file__).resolve()),sha256=common.sha(__file__)),time=time.time(),
        scope='Finite numerical/runtime bridge and known-control reproduction only; no independent quality or complementary-path evidence')
    path=ROOT/'reports/ease_insights_20261103/camera_paths_v3/calibration_acceptance.json';common.atomic_json(path,value)
    print(json.dumps({k:v for k,v in value.items() if k not in ('cases','services','bridges')}))


if __name__=='__main__':main()

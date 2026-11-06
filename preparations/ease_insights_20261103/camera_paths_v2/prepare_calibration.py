"""Freeze four repeated Full bridge controls and the new four-path GPU gates."""
import copy
import json
from pathlib import Path
from .common import ROOT,SUITES,LIMITS,conditions,sha,digest,atomic_json,read_manifest,validate_output
from .protocol import freeze

PACKAGE=Path(__file__).resolve().parent
OUTPUT=ROOT/'results/ease_insights_20261103/camera_paths_v2/calibration'


def prepare():
    target=OUTPUT/'manifest.json'
    if target.exists():
        m=read_manifest(target,verify_sources=True);return dict(reused=True,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target))
    from preparations.ease_insights_20261103.module_tradeoff_v1.common import read_manifest as read_module
    from preparations.ease_insights_20261103.vlm_diversity_v2.common import read_manifest as read_original
    from preparations.ease_insights_20261103.vlm_diversity_v2.prepare import unused
    prior_path=ROOT/'results/ease_insights_20261103/module_tradeoff_v1/discovery/manifest.json'
    base=read_module(prior_path,verify_sources=True)
    summary=ROOT/'reports/ease_insights_20261103/module_tradeoff_v1/completed_summary.json'
    previous=json.loads(summary.read_text())
    if previous.get('status')!='completed_no_candidate':raise ValueError('Previous module branch is not complete without a candidate')
    original=read_original(ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery/manifest.json',verify_sources=True)
    heldout=unused(original,[47,48,49]);pin=freeze()
    keys=('checkpoints','bank','environment','noise','quality_environment_reuse','technical_max_attempts','official_init_availability','health_evidence')
    m={k:copy.deepcopy(base[k]) for k in keys}
    local={str(p.relative_to(ROOT)):sha(p) for p in sorted(PACKAGE.glob('*.py'))}
    sources=dict(base['source_files']);sources.update(local)
    for p in sorted((ROOT/'reports/ease_insights_20261103/camera_paths_v2').glob('*.py')):
        sources[str(p.relative_to(ROOT))]=sha(p)
    resource=ROOT/'results/ease_insights_20261103/resource_pool_v4/plan.json'
    resource_plan=json.loads(resource.read_text())
    for name,d in resource_plan['source_hashes'].items():
        if sha(ROOT/name)!=d:raise ValueError('Resource source changed')
        sources[name]=d
    science=dict(module='camera',H=50,R=5,depth=18,vision_layers=27,AE_layers=18,draft_D=5,refresh_D=10,
        schedule='Full-D10 followed by four Fast-D5 requests',compensation='hold other-view last-own-Full raw features',
        selector_calls=0,gradient_selector=False,training=False,environment_seed=7,settling_actions=10,policy_action_limits=dict(LIMITS),
        interpretation='Periodic camera fast paths; no standalone path or adaptive-selector benefit claim')
    m.update(revision='ease-camera-paths-calibration-v2',phase='camera_paths_calibration_v2',output_root=str(OUTPUT),
        science=science,conditions=conditions(),planned_conditions=['full_d10'],suites=list(SUITES),tasks=[0],
        case_table=[dict(suite=s,task=0) for s in SUITES],calibration_inits=[45],cases_per_condition=4,total_calibration_episodes=4,
        protocol_evidence=pin,design_evidence=pin,local_runtime_sources=local,source_files=sources,analysis_sources=local,
        resource_execution_revision=dict(name='resource-pool-v4',plan=dict(path=str(resource),sha256=sha(resource)),
            original_GPU_leases_unchanged=True,idle_mib=16384,running_placeholder_mib=8192),
        preflight=dict(status='required_before_rollout',atol=1e-5,rtol=1e-5,
            requirements=['native no-bypass D5/D10 parity','actual AE18 visits and native NFE','actual batch1 vision buffers','same-observation camera reconstruction',
                'nontrivial mixed-view oracle parity','Full+fourFast own-cache schedule and reset','paired physical first input and noise']),
        confirmation_initial_state_prior_use_audit=heldout,
        calibration_case_scope='Four repeated known Full controls, never independent quality evidence',
        preserved_reference=dict(root=str(ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery'),
            manifest_path=str(ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery/manifest.json'),
            manifest_sha256='a4a996b03059874ceb10a21e1099985508051d7ae0392af944e739911cd1494b',case_records_unchanged=True),
        fixed_input_cost=dict(conditions='All four fixed camera definitions',warmup_requests=160,measured_requests=320,
            scope='Native completed-policy mean five-request cycle; four old-bank inputs; no exclusive-GPU or quality claim'))
    m['manifest_hash']=digest(m);validate_output(OUTPUT,m);atomic_json(target,m);read_manifest(target,verify_sources=True)
    return dict(reused=False,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target),episodes=4,GPU_started=False)


if __name__=='__main__':print(json.dumps(prepare()))

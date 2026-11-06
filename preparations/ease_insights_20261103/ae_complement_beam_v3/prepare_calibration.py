"""Freeze eight known controls and six-path depth/time numerical/cost calibration."""
import copy
import json
from pathlib import Path
from .common import ROOT,SUITES,LIMITS,REFERENCES,conditions,sha,digest,atomic_json,read_manifest,validate_output,reference
from .protocol import REPORT,freeze

PACKAGE=Path(__file__).resolve().parent
OUTPUT=ROOT/'results/ease_insights_20261103/ae_complement_beam_v3/calibration'


def prepare():
    target=OUTPUT/'manifest.json'
    if target.exists():
        m=read_manifest(target,verify_sources=True);return dict(reused=True,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target))
    from preparations.ease_insights_20261103.denoise_depth_paths_v1.common import read_manifest as read_anchor
    from preparations.ease_insights_20261103.vlm_diversity_v2.common import read_manifest as read_original
    from preparations.ease_insights_20261103.vlm_diversity_v2.prepare import unused
    base=read_anchor(ROOT/'results/ease_insights_20261103/denoise_depth_paths_v1/discovery/manifest.json',verify_sources=True)
    prior=ROOT/'reports/ease_insights_20261103/ae_complement_beam_v2/completed_summary.json';ended=json.loads(prior.read_text())
    if ended.get('status')!='completed_no_candidate' or ended.get('new_cases')!=160 or ended.get('confirmation_launched'):
        raise ValueError('Previous AE14/D2 proxy-selected pair must be fully accepted without confirmation')
    original=read_original(ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery/manifest.json',verify_sources=True)
    heldout=unused(original,[47,48,49])
    for alias in REFERENCES:reference(alias,verify_sources=True)
    pin=freeze();keys=('checkpoints','bank','environment','noise','quality_environment_reuse','technical_max_attempts','official_init_availability','health_evidence')
    m={k:copy.deepcopy(base[k]) for k in keys}
    local={str(p.relative_to(ROOT)):sha(p) for p in sorted(PACKAGE.glob('*.py'))};sources=dict(base['source_files']);sources.update(local)
    for p in sorted(REPORT.glob('*.py')):sources[str(p.relative_to(ROOT))]=sha(p)
    resource=ROOT/'results/ease_insights_20261103/resource_pool_v4/plan.json';resource_plan=json.loads(resource.read_text())
    for name,pin_sha in resource_plan['source_hashes'].items():
        if sha(ROOT/name)!=pin_sha:raise ValueError('Resource source changed')
        sources[name]=pin_sha
    m.update(revision='ease-AE14-neighbor-calibration-v3',phase='AE14_neighbor_calibration_v3',output_root=str(OUTPUT),
        conditions=conditions(),planned_conditions=['full_d10','ae14_base'],suites=list(SUITES),tasks=[0],
        case_table=[dict(suite=s,task=0) for s in SUITES],calibration_inits=[45],cases_per_condition=4,total_calibration_episodes=8,
        protocol_evidence=pin,design_evidence=pin,local_runtime_sources=local,source_files=sources,analysis_sources=local,
        science=dict(module='AE14_local_layer_swaps',H=50,R=5,depth=18,VLM_layers=18,vision_layers=27,cameras=2,
            refresh_D=10,fast_D_options=[3],schedule='FullD10 followed by two AE14/D3 Fast requests',
            compensation='No VLM omission or residual update; empty-mask own-Full timestamps only',selector_calls=0,gradient_selector=False,
            training=False,environment_seed=7,settling_actions=10,policy_action_limits=dict(LIMITS)),
        preflight=dict(status='required_before_rollout',atol=1e-5,rtol=1e-5,
            requirements=['original adapter uniform Euler D2/D3/D4 and fusedD10 AE18 parity','actual ordered AE visits and internal NFE',
                'exact Euler receipts for every schedule','full VLM prefix,period/reset','all eight known Full/AE14 controls bitwise identical to both source roots','paired physical initial input/noise']),
        resource_execution_revision=dict(name='resource-pool-v4',plan=dict(path=str(resource),sha256=sha(resource)),original_GPU_leases_unchanged=True,idle_mib=16384,running_placeholder_mib=8192),
        confirmation_initial_state_prior_use_audit=heldout,reference_origins=copy.deepcopy(REFERENCES),
        calibration_case_scope='Eight explicitly repeated known controls from two original roots, not independent quality evidence',
        fixed_input_cost=dict(conditions='All six fixed depth/time definitions',warmup_requests=144,measured_requests=288,
            scope='Original-adapter completed policy cost on four old-bank inputs; no layer-count equivalence, exclusivity or quality claim'))
    m['manifest_hash']=digest(m);validate_output(OUTPUT,m);atomic_json(target,m);read_manifest(target,verify_sources=True)
    return dict(reused=False,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target),known_controls=8,GPU_started=False)


if __name__=='__main__':print(json.dumps(prepare()))

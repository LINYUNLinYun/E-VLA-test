"""12 known controls and new joint native-cost evidence; no fresh states yet."""
import copy
import json
from pathlib import Path
from .common import *
from .protocol import REPORT,freeze

PACKAGE=Path(__file__).resolve().parent
OUTPUT=FAMILY_ROOT/'calibration'


def prepare():
    target=OUTPUT/'manifest.json'
    if target.exists():
        m=read_manifest(target,verify_sources=True);return dict(reused=True,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target))
    from preparations.ease_insights_20261103.ae_complement_beam_v3.common import read_manifest as read_previous
    from preparations.ease_insights_20261103.vlm_diversity_v2.common import read_manifest as read_original
    from preparations.ease_insights_20261103.vlm_diversity_v2.prepare import unused
    base=read_previous(ROOT/'results/ease_insights_20261103/ae_complement_beam_v3/discovery/manifest.json',verify_sources=True)
    original=read_original(ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery/manifest.json',verify_sources=True)
    heldout=unused(original,[47,48,49])
    for alias in REFERENCES:reference(alias,verify_sources=True)
    pin=freeze();protocol=json.loads(Path(pin['path']).read_text())
    keys=['checkpoints','bank','environment','noise','quality_environment_reuse','technical_max_attempts','official_init_availability','health_evidence','resource_execution_revision']
    m={k:copy.deepcopy(base[k]) for k in keys};local={str(p.relative_to(ROOT)):sha(p) for p in sorted(PACKAGE.glob('*.py'))};sources=dict(base['source_files']);sources.update(local)
    for p in sorted(REPORT.glob('*.py')):sources[str(p.relative_to(ROOT))]=sha(p)
    evidence=ROOT/'reports/ease_insights_20261103/cross_module_completed_audit_v1/audit.py';sources[str(evidence.relative_to(ROOT))]=sha(evidence)
    m.update(revision='ease-cross-module-calibration-v4',phase='cross_module_calibration_v4',output_root=str(OUTPUT),
        conditions=conditions(),planned_conditions=list(CONDITIONS),suites=list(SUITES),tasks=[0],case_table=[dict(suite=s,task=0) for s in SUITES],
        calibration_inits=[45],cases_per_condition=4,total_calibration_episodes=12,noise_repeat=0,
        protocol_evidence=pin,design_evidence=pin,historical_discovery_evidence=protocol['historical_discovery_evidence'],
        local_runtime_sources=local,source_files=sources,analysis_sources=local,reference_origins=copy.deepcopy(REFERENCES),
        science=dict(module='VLM_AE_allocation',H=50,R=5,depth=18,vision_layers=27,cameras=2,refresh_D=10,fast_D_options=[2,3],
            schedule='FullD10 then two config-bound Fast requests',training=False,environment_seed=7,settling_actions=10,policy_action_limits=dict(LIMITS)),
        preflight=dict(status='required_before_rollout',atol=1e-5,rtol=1e-5),confirmation_initial_state_prior_use_audit=heldout,
        calibration_case_scope='12 repeated known controls,zero independent quality cases',fixed_input_cost=dict(warmup_requests=72,measured_requests=144,conditions='All three definitions in every completed service'))
    m['manifest_hash']=digest(m);validate_output(OUTPUT,m);atomic_json(target,m);read_manifest(target,verify_sources=True)
    return dict(reused=False,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target),known_controls=12,GPU_started=False)


if __name__=='__main__':print(json.dumps(prepare()))

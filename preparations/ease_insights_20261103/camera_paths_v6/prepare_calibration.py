"""Freeze8 known native controls after CPU checks;no model/environment starts here."""
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
    from preparations.ease_insights_20261103.camera_paths_v5.common import read_manifest as prior_manifest
    old=prior_manifest(ROOT/'results/ease_insights_20261103/camera_paths_v5/calibration/manifest.json',verify_sources=True)
    for alias in REFERENCES:reference(alias,verify_sources=True)
    pin=freeze();protocol=json.loads(Path(pin['path']).read_text())
    keys=('checkpoints','bank','environment','noise','quality_environment_reuse','technical_max_attempts','health_evidence','resource_execution_revision','official_init_availability')
    m={k:copy.deepcopy(old[k]) for k in keys}
    for row in m['official_init_availability']['cases']:
        require(sha(row['bddl_path'])==row['bddl_sha256'] and sha(row['init_file'])==row['init_file_sha256'],'Official physical assets changed')
    local={str(p.relative_to(ROOT)):sha(p) for p in sorted(PACKAGE.glob('*.py'))};sources=dict(old['source_files']);sources.update(local)
    for folder in [REPORT,*[ROOT/f'reports/ease_insights_20261103/prefix_kv_preparation_v{i}' for i in range(1,6)]]:
        for p in sorted(folder.glob('*.py')):sources[str(p.relative_to(ROOT))]=sha(p)
    m.update(revision='ease-full-storage-image-edge-calibration-v6',phase='masked_attention_calibration_v6',output_root=str(OUTPUT),
        conditions=conditions(),planned_conditions=['full_d10','dense_d2'],suites=list(SUITES),tasks=[0],case_table=[dict(suite=s,task=0) for s in SUITES],
        calibration_inits=[45],cases_per_condition=4,total_calibration_episodes=8,noise_repeat=0,
        protocol_evidence=pin,design_evidence=pin,antecedent_evidence=protocol['antecedent_evidence'],
        local_runtime_sources=local,analysis_sources=local,source_files=sources,
        science=dict(H=50,R=5,VLM_layers=18,AE_layers=18,vision_Full=[27,27],vision_Fast=[27,27],Full_D=10,Fast_D=2,refresh_D=10,period=2,
            training=False,environment_seed=7,settling_actions=10,policy_action_limits=dict(LIMITS),both_images_fresh=True,vision_cache=False,
            prefix_patterns=dict(PATTERNS),Fast_visible_visual_keys_per_camera=128,original_QK_V_storage=True,prefix_compute_reduction_claimed=False,all_text_retained=True,original_decoder_absolute_RoPE=True),
        preflight=dict(status='required_before_rollout',atol=1e-5,rtol=1e-5,partial_reference_atol=1e-5,partial_reference_rtol=1e-5,partial_gripper_all50_exact=True),
        reference_origins=copy.deepcopy(REFERENCES),calibration_case_scope='8knowncontrolrepeats,zeroindependentqualitycases',heldout_scope=protocol['state_exposure_scope'],
        fixed_input_cost=dict(conditions='All4prefixdefinitionsineachcompletedservice',warmup_requests=96,measured_requests=192))
    m['manifest_hash']=digest(m);validate_output(OUTPUT,m);atomic_json(target,m);read_manifest(target,verify_sources=True)
    return dict(reused=False,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target),known_controls=8,source_files=len(sources),GPU_started=False)


if __name__=='__main__':print(json.dumps(prepare()))

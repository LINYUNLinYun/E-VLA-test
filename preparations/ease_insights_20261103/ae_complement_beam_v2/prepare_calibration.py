"""Freeze eight repeated controls before160 new pair rollouts."""
import copy
import json
from pathlib import Path
from .common import *
from .protocol import freeze, REPORT

PACKAGE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'results/ease_insights_20261103/ae_complement_beam_v2/calibration'


def prepare():
    target = OUTPUT / 'manifest.json'
    if target.exists():
        m = read_manifest(target, verify_sources=True)
        return dict(reused=True, manifest_hash=m['manifest_hash'], manifest_file_sha256=sha(target))
    from preparations.ease_insights_20261103.denoise_depth_paths_v1.common import read_manifest as read_prior
    from preparations.ease_insights_20261103.vlm_diversity_v2.common import read_manifest as read_original
    from preparations.ease_insights_20261103.vlm_diversity_v2.prepare import unused
    prior = read_prior(ROOT / 'results/ease_insights_20261103/denoise_depth_paths_v1/discovery/manifest.json', verify_sources=True)
    screen = ROOT / 'results/ease_insights_20261103/ae_complement_beam_v1/manifest.json'
    screen_manifest = json.loads(screen.read_text())
    for alias in REFERENCES: reference(alias, verify_sources=True)
    original = read_original(ROOT / 'results/ease_insights_20261103/vlm_diversity_v2/discovery/manifest.json', verify_sources=True)
    heldout = unused(original, [47, 48, 49])
    pin = freeze()
    keys = ('checkpoints', 'bank', 'environment', 'noise', 'quality_environment_reuse', 'technical_max_attempts',
            'official_init_availability', 'health_evidence', 'resource_execution_revision')
    value = {k: copy.deepcopy(prior[k]) for k in keys}
    local = {str(p.relative_to(ROOT)): sha(p) for p in sorted(PACKAGE.glob('*.py'))}
    sources = dict(screen_manifest['source_files'])
    sources.update(local)
    for p in sorted(REPORT.glob('*.py')): sources[str(p.relative_to(ROOT))] = sha(p)
    for p in [Path(pin['path']), ROOT / 'reports/ease_insights_20261103/ae_complement_beam_v1/independent_audit.py']:
        sources[str(p.relative_to(ROOT))] = sha(p)
    value.update(revision='ease-AE-joint-mask-calibration-v2', phase='AE_joint_mask_calibration_v2', output_root=str(OUTPUT),
        conditions=conditions(), planned_conditions=['full_d10','ae16_d2'], suites=list(SUITES), tasks=[0],
        case_table=[dict(suite=s, task=0) for s in SUITES], calibration_inits=[45], cases_per_condition=4, total_calibration_episodes=8,
        protocol_evidence=pin, design_evidence=pin, proxy_selection_evidence=proxy_evidence(), proxy_preregistered_directions=copy.deepcopy(DIRECTIONS),
        local_runtime_sources=local, source_files=sources, analysis_sources=local, reference_origins=copy.deepcopy(REFERENCES),
        science=dict(module='AE16_joint_masks', H=50, R=5, depth=18, VLM_layers=18, vision_layers=27, cameras=2,
            refresh_D=10, fast_D_options=[2], schedule='FullD10 followed by two AE16D2 requests',
            compensation='No VLM omission or residual update; empty-mask own-Full timestamps only', selector_calls=0, gradient_selector=False,
            training=False, environment_seed=7, settling_actions=10, policy_action_limits=dict(LIMITS)),
        preflight=dict(status='required_before_rollout', atol=1e-5, rtol=1e-5,
            requirements=['Native fullAE18 D2/D10 parity,actual ordered AE/NFE/time nodes','FullVLM prefix,period/reset',
                          'Eight known Full/AE16 controls bitwise identical across both source roots','All four definitions on each service']),
        confirmation_initial_state_prior_use_audit=heldout,
        calibration_case_scope='Eight repeated known controls; zero independent quality evidence',
        fixed_input_cost=dict(conditions='All four frozen definitions', warmup_requests=96, measured_requests=192,
                              scope='Native three-request policy cycles, four old-bank inputs; not exclusive-GPU or closed-loop speed'))
    value['manifest_hash'] = digest(value)
    validate_output(OUTPUT, value)
    atomic_json(target, value)
    read_manifest(target, verify_sources=True)
    return dict(reused=False, manifest_hash=value['manifest_hash'], manifest_file_sha256=sha(target), known_controls=8, GPU_started=False)


if __name__ == '__main__': print(json.dumps(prepare()))

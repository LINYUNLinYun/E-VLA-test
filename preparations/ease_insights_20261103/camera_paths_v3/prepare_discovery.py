"""Freeze240 new camera cases after calibration; retain80 original Full identities."""
import copy
import json
from pathlib import Path
from .common import ROOT,CONDITIONS,NEW_CONDITIONS,SUITES,conditions,sha,digest,atomic_json,read_manifest,validate_output
from .protocol import REPORT,freeze

PACKAGE=Path(__file__).resolve().parent
OUTPUT=ROOT/'results/ease_insights_20261103/camera_paths_v3/discovery'


def prepare():
    target=OUTPUT/'manifest.json'
    if target.exists():
        m=read_manifest(target,verify_sources=True);return dict(reused=True,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target))
    cal=read_manifest(OUTPUT.parent/'calibration/manifest.json',verify_sources=True)
    accepted=REPORT/'calibration_acceptance.json';proof=json.loads(accepted.read_text())
    if (proof.get('status')!='pass' or proof.get('manifest_hash')!=cal['manifest_hash']
        or not proof.get('all_four_controls_bitwise_identical') or not proof.get('actual_single_view_batch_passed')
        or not proof.get('nontrivial_mixed_view_oracle_passed')):raise ValueError('Camera calibration bridge not accepted')
    from preparations.ease_insights_20261103.vlm_diversity_v2.common import read_manifest as read_old
    from preparations.ease_insights_20261103.vlm_diversity_v2.prepare import unused
    old_root=ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery'
    old=read_old(old_root/'manifest.json',verify_sources=True);heldout=unused(old,[47,48,49])
    prior_audit=ROOT/'reports/ease_insights_20261103/vlm_diversity_v2/full_discovery_audit.json'
    if sha(prior_audit)!='41d70066f98bbe69ddb003a13cc87495e8e46fff4b8bdb5942e84ce0af31a383':raise ValueError('Accepted original Full case audit changed')
    keys=('checkpoints','bank','environment','noise','science','quality_environment_reuse','technical_max_attempts','preflight','official_init_availability','health_evidence','resource_execution_revision')
    m={k:copy.deepcopy(cal[k]) for k in keys}
    for row in m['official_init_availability']['cases']:
        if sha(row['bddl_path'])!=row['bddl_sha256'] or sha(row['init_file'])!=row['init_file_sha256']:raise ValueError('Official camera case asset changed')
    local={str(p.relative_to(ROOT)):sha(p) for p in sorted(PACKAGE.glob('*.py'))}
    sources=dict(cal['source_files']);sources.update(local)
    for p in sorted(REPORT.glob('*.py')):sources[str(p.relative_to(ROOT))]=sha(p)
    pin=freeze()
    m.update(revision='ease-camera-paths-discovery-v3',phase='camera_paths_discovery_v3',output_root=str(OUTPUT),
        conditions=conditions(),planned_conditions=list(NEW_CONDITIONS),analytical_conditions=list(CONDITIONS),reference_conditions={'full_d10':'full_d10'},
        suites=list(SUITES),tasks=list(range(10)),case_table=[dict(suite=s,task=t) for s in SUITES for t in range(10)],
        discovery_inits=[45,46],cases_per_condition=80,total_new_discovery_episodes=240,preserved_reference_episodes=80,analytical_cells=320,
        protocol_evidence=pin,design_evidence=pin,calibration_evidence=dict(path=str(accepted),sha256=sha(accepted)),
        preserved_reference=dict(root=str(old_root),manifest_path=str(old_root/'manifest.json'),manifest_hash=old['manifest_hash'],manifest_sha256=sha(old_root/'manifest.json'),
            case_audit=dict(path=str(prior_audit),sha256=sha(prior_audit)),records_unchanged=True,new_independent_replications=0),
        local_runtime_sources=local,source_files=sources,analysis_sources=local,confirmation_initial_state_prior_use_audit=heldout,
        discovery_reuse_disclosure='The80 Full outcomes on init45-46 are known and referenced with original identities. Only three new camera/control configurations generate240 new cases.',
        initial_pairing='Every new case first observation/noise/physical identity must agree with original Full before selection.',
        pair_cost_match_formula='max(median_completed_service_Full_over_path_ratios)/min(same_ratios)<=1.10; each>=1.15',
        new_scientific_episode_cap=604,independent_validation=dict(status='locked_until_complete_qualified_camera_pair_selection',inits=[47,48,49],episodes=360),
        fixed_input_cost=dict(conditions='All four camera definitions on each completed service',warmup_requests=160,measured_requests=320,
            scope='Native mean five-request cycle on four old-bank inputs; no exclusive-GPU or quality claim'))
    m['manifest_hash']=digest(m);validate_output(OUTPUT,m);atomic_json(target,m);read_manifest(target,verify_sources=True)
    return dict(reused=False,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target),new_episodes=240,preserved_Full_cells=80,GPU_started=False)


if __name__=='__main__':print(json.dumps(prepare()))

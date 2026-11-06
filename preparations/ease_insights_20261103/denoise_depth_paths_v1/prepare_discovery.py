"""Freeze320 new depth/time cases after the eight-control bridge passes."""
import copy
import json
from pathlib import Path
from .common import ROOT,SUITES,CONDITIONS,NEW_CONDITIONS,REFERENCE_CONDITIONS,REFERENCES,conditions,reference,sha,digest,atomic_json,read_manifest,validate_output
from .protocol import REPORT,freeze

PACKAGE=Path(__file__).resolve().parent
OUTPUT=ROOT/'results/ease_insights_20261103/denoise_depth_paths_v1/discovery'


def prepare():
    target=OUTPUT/'manifest.json'
    if target.exists():
        m=read_manifest(target,verify_sources=True);return dict(reused=True,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target))
    cal=read_manifest(OUTPUT.parent/'calibration/manifest.json',verify_sources=True)
    accepted=REPORT/'calibration_acceptance.json';proof=json.loads(accepted.read_text())
    if proof.get('status')!='pass' or proof.get('manifest_hash')!=cal['manifest_hash'] or not proof.get('all_eight_controls_bitwise_identical'):
        raise ValueError('Await the eight-known-control numeric/source bridge')
    from preparations.ease_insights_20261103.vlm_diversity_v2.common import read_manifest as read_original
    from preparations.ease_insights_20261103.vlm_diversity_v2.prepare import unused
    old=read_original(ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery/manifest.json',verify_sources=True);heldout=unused(old,[47,48,49])
    for alias in REFERENCES:reference(alias,verify_sources=True)
    keys=('checkpoints','bank','environment','noise','science','quality_environment_reuse','technical_max_attempts','preflight','official_init_availability','health_evidence','resource_execution_revision')
    m={k:copy.deepcopy(cal[k]) for k in keys}
    for row in m['official_init_availability']['cases']:
        if sha(row['bddl_path'])!=row['bddl_sha256'] or sha(row['init_file'])!=row['init_file_sha256']:raise ValueError('Official depth/time case assets changed')
    local={str(p.relative_to(ROOT)):sha(p) for p in sorted(PACKAGE.glob('*.py'))};sources=dict(cal['source_files']);sources.update(local)
    for p in sorted(REPORT.glob('*.py')):sources[str(p.relative_to(ROOT))]=sha(p)
    pin=freeze()
    m.update(revision='ease-denoise-depth-discovery-v1',phase='denoise_depth_discovery_v1',output_root=str(OUTPUT),
        conditions=conditions(),planned_conditions=list(NEW_CONDITIONS),analytical_conditions=list(CONDITIONS),reference_conditions=REFERENCE_CONDITIONS,
        reference_origins=copy.deepcopy(REFERENCES),suites=list(SUITES),tasks=list(range(10)),case_table=[dict(suite=s,task=t) for s in SUITES for t in range(10)],
        discovery_inits=[45,46],cases_per_condition=80,total_new_discovery_episodes=320,preserved_reference_episodes=160,analytical_cells=480,
        protocol_evidence=pin,design_evidence=pin,calibration_evidence=dict(path=str(accepted),sha256=sha(accepted)),
        local_runtime_sources=local,source_files=sources,analysis_sources=local,confirmation_initial_state_prior_use_audit=heldout,
        discovery_reuse_disclosure='Full80 and AE16D2 80 retain their two original roots/manifests/condition IDs. They are known exploratory controls,not160 new replications.',
        initial_pairing='All four new paths must match both original controls in official physical identity,first observation and explicit noise for each suite/task/init.',
        new_scientific_episode_cap=688,independent_validation=dict(status='locked_until_complete_qualified_selection',inits=[47,48,49],episodes=360),
        pair_cost_match_formula='max(median_completed_service_Full_over_path_ratios)/min(same_ratios)<=1.10; each ratio>=1.15',
        fixed_input_cost=dict(conditions='All six depth/time definitions on each completed service',warmup_requests=144,measured_requests=288,
            scope='Whole three-request original-adapter policy cycles,four old-bank inputs; actual AE/NFE/nodes checked; no exclusive-GPU or closed-loop speed claim'))
    m['manifest_hash']=digest(m);validate_output(OUTPUT,m);atomic_json(target,m);read_manifest(target,verify_sources=True)
    return dict(reused=False,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target),new_episodes=320,preserved_control_cells=160,GPU_started=False)


if __name__=='__main__':print(json.dumps(prepare()))

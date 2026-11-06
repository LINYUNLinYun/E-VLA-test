"""Freeze320 new module-path cases, keeping160 prior control records intact."""
import copy
import json
from pathlib import Path
from .common import ROOT,CONDITIONS,NEW_CONDITIONS,SUITES,conditions,sha,digest,atomic_json,read_manifest,validate_output
from .protocol import REPORT,freeze

PACKAGE=Path(__file__).resolve().parent
OUTPUT=ROOT/'results/ease_insights_20261103/module_tradeoff_v1/discovery'


def prepare():
    target=OUTPUT/'manifest.json'
    if target.exists():
        m=read_manifest(target,verify_sources=True)
        return dict(reused=True,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target))
    cal_path=OUTPUT.parent/'calibration/manifest.json';cal=read_manifest(cal_path,verify_sources=True)
    accepted=REPORT/'calibration_acceptance.json';proof=json.loads(accepted.read_text())
    if proof.get('status')!='pass' or proof.get('manifest_hash')!=cal['manifest_hash'] or not proof.get('all_four_controls_bitwise_identical'):
        raise ValueError('Need accepted native/runtime bridge before reference reuse')
    from preparations.ease_insights_20261103.vlm_diversity_v2.common import read_manifest as read_old
    from preparations.ease_insights_20261103.vlm_diversity_v2.prepare import unused
    old_root=ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery'
    old=read_old(old_root/'manifest.json',verify_sources=True);heldout_unused=unused(old,[47,48,49])
    prior_audit=ROOT/'reports/ease_insights_20261103/vlm_diversity_v2/full_discovery_audit.json'
    if sha(prior_audit)!='41d70066f98bbe69ddb003a13cc87495e8e46fff4b8bdb5942e84ce0af31a383':
        raise ValueError('Accepted prior400-case audit changed')
    keys=('checkpoints','bank','environment','noise','science','quality_environment_reuse','technical_max_attempts','preflight','official_init_availability','health_evidence','resource_execution_revision')
    m={k:copy.deepcopy(cal[k]) for k in keys}
    for row in m['official_init_availability']['cases']:
        if sha(row['bddl_path'])!=row['bddl_sha256'] or sha(row['init_file'])!=row['init_file_sha256']:
            raise ValueError('Official assets changed before new module cases')
    local={str(p.relative_to(ROOT)):sha(p) for p in sorted(PACKAGE.glob('*.py'))}
    sources=dict(cal['source_files']);sources.update(local)
    tool='reports/ease_insights_20261103/common_case_audit_v1/audit.py';sources[tool]=sha(ROOT/tool)
    pin=freeze()
    m.update(revision='ease-module-tradeoff-discovery-v1',phase='module_tradeoff_discovery_v1',output_root=str(OUTPUT),
        conditions=conditions(),planned_conditions=list(NEW_CONDITIONS),analytical_conditions=list(CONDITIONS),
        reference_conditions={'full_d10':'full_d10','vlm_c':'spread_c'},
        suites=list(SUITES),tasks=list(range(10)),case_table=[dict(suite=s,task=t) for s in SUITES for t in range(10)],
        discovery_inits=[45,46],cases_per_condition=80,total_new_discovery_episodes=320,preserved_reference_episodes=160,analytical_cells=480,
        protocol_evidence=pin,design_evidence=pin,calibration_evidence=dict(path=str(accepted),sha256=sha(accepted)),
        preserved_reference=dict(root=str(old_root),manifest_path=str(old_root/'manifest.json'),manifest_hash=old['manifest_hash'],manifest_sha256=sha(old_root/'manifest.json'),
            case_audit=dict(path=str(prior_audit),sha256=sha(prior_audit)),records_unchanged=True,new_independent_replications=0),
        local_runtime_sources=local,source_files=sources,analysis_sources=local,
        confirmation_initial_state_prior_use_audit=heldout_unused,
        discovery_reuse_disclosure='Initial states and Full/spread_c outcomes are already known. Only four new module paths generate new discovery cases; preserved160 cells retain their original source identities.',
        initial_pairing='Before selection, every new case first observation/noise/physical identity must agree with preserved Full and spread_c for the same suite/task/init.',
        pair_cost_match_formula='max(median_completed_service_Full_over_path_ratios)/min(same_ratios)<=1.10; each ratio>=1.15',
        new_scientific_episode_cap=684,independent_validation=dict(status='locked_until_complete_qualified_selection',inits=[47,48,49],episodes=360),
        fixed_input_cost=dict(conditions='All six module definitions on each completed service',warmup_requests=144,measured_requests=288,
            scope='Four old-bank inputs; native completed-policy costs including dispatcher; no exclusive-GPU or quality claim'))
    m['manifest_hash']=digest(m);validate_output(OUTPUT,m);atomic_json(target,m);read_manifest(target,verify_sources=True)
    return dict(reused=False,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target),new_episodes=320,preserved_control_cells=160,GPU_started=False)


if __name__=='__main__':print(json.dumps(prepare()))

"""Only four known Full controls plus six-condition numerical/cost profiling."""
import copy
import json
from pathlib import Path
import time
from .common import ROOT,MODULE,SUITES,LIMITS,conditions,sha,digest,atomic_json,read_manifest,validate_output
from .protocol import freeze,REPORT

PACKAGE=Path(__file__).resolve().parent
OUTPUT=ROOT/'results/ease_insights_20261103/module_tradeoff_v1/calibration'


def prepare():
    path=OUTPUT/'manifest.json'
    if path.exists():
        m=read_manifest(path,verify_sources=True)
        return dict(reused=True,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(path))
    from preparations.ease_insights_20261103.vlm_diversity_v2.common import read_manifest as read_prior
    from preparations.ease_insights_20261103.vlm_diversity_v2.prepare import unused
    prior_path=ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery/manifest.json'
    base=read_prior(prior_path,verify_sources=True)
    summary=ROOT/'reports/ease_insights_20261103/vlm_diversity_v2/completed_summary.json'
    ended=json.loads(summary.read_text())
    if ended.get('status')!='completed_no_candidate' or ended.get('episodes')!=400:
        raise ValueError('Previous bounded branch is not complete')
    confirmation_unused=unused(base,[47,48,49])
    pin=freeze()
    keys=('checkpoints','bank','environment','noise','science','quality_environment_reuse','technical_max_attempts','preflight','official_init_availability','health_evidence')
    m={k:copy.deepcopy(base[k]) for k in keys}
    local={str(p.relative_to(ROOT)):sha(p) for p in sorted(PACKAGE.glob('*.py'))}
    sources=dict(base['source_files']);sources.update(local)
    m.update(revision='ease-module-tradeoff-calibration-v1',phase='module_tradeoff_calibration_v1',output_root=str(OUTPUT),
        conditions=conditions(),planned_conditions=['full_d10'],suites=list(SUITES),tasks=[0],
        case_table=[dict(suite=s,task=0) for s in SUITES],calibration_inits=[45],cases_per_condition=4,total_calibration_episodes=4,
        protocol_evidence=pin,design_evidence=pin,local_runtime_sources=local,source_files=sources,analysis_sources=local,
        resource_execution_revision=copy.deepcopy(base['resource_execution_revision']),
        confirmation_initial_state_prior_use_audit=confirmation_unused,
        calibration_case_scope='Four explicitly repeated known Full controls for runtime/parity bridging, not independent quality outcomes',
        preserved_reference=dict(root=str(prior_path.parent),manifest_path=str(prior_path),manifest_hash=base['manifest_hash'],manifest_sha256=sha(prior_path),
            prior_acceptance=dict(path=str(summary),sha256=sha(summary)),case_records_unchanged=True),
        fixed_input_cost=dict(conditions='All six fixed VLM/AE definitions',warmup_requests=144,measured_requests=288,
            scope='Completed native policy, four old-bank inputs; no latency exclusivity or quality claim'))
    m['manifest_hash']=digest(m);validate_output(OUTPUT,m);atomic_json(path,m);read_manifest(path,verify_sources=True)
    return dict(reused=False,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(path),episodes=4,GPU_started=False)


if __name__=='__main__':print(json.dumps(prepare()))

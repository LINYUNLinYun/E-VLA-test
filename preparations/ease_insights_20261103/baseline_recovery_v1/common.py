"""Prospective Full-only health check after the unusable LIBERO90 baseline."""
import hashlib
import json
from pathlib import Path
from preparations.idea1_v3_1.common import ROOT, SUITES, LIMITS, atomic_json, digest
from preparations.idea1_v3_1.common import read_manifest as _read_manifest

MODULE = 'preparations.ease_insights_20261103.baseline_recovery_v1'
MODEL_PYTHON = ROOT/'policies/realtime-vla-flash/.venv/bin/python'
ENV_PYTHON = ROOT/'policies/libero-eval-v21/.venv/bin/python'
GATE_SUITES = SUITES
CONDITIONS = ('full_d10',)
PHASE = 'common_suite_full_calibration_v1'
HEALTH_RULE = dict(minimum_total_successes=32, total_cases=40,
                   minimum_suite_successes=6, cases_per_suite=10,
                   role='Health gate only; no complementary-path inference or task filtering')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditions():
    return [dict(id='full_d10', model='pi0', mode='full', compensation=None,
                 period=2, H=50, R=5, D=10, refresh_D=10, skip=[])]


def validate_condition(config):
    if config != conditions()[0]:
        raise ValueError('Calibration permits only the unchanged FullD10/R5 condition')
    return ()


def stage_for(manifest):
    if manifest.get('phase') != PHASE or manifest.get('calibration_inits') != [44]:
        raise ValueError('Calibration requires the fixed independent init44')
    return 'calibration'


def validate_manifest(manifest):
    stage_for(manifest)
    table = [dict(suite=s, task=t) for s in SUITES for t in range(10)]
    if (manifest.get('conditions') != conditions() or manifest.get('planned_conditions') != list(CONDITIONS)
        or manifest.get('suites') != list(SUITES) or manifest.get('tasks') != list(range(10))
        or manifest.get('case_table') != table or manifest.get('cases_per_condition') != 40
        or manifest.get('total_calibration_episodes') != 40 or manifest.get('health_rule') != HEALTH_RULE
        or manifest.get('science',{}).get('policy_action_limits') != LIMITS
        or manifest.get('science',{}).get('settling_actions') != 10):
        raise ValueError('Fixed40-case calibration universe, health rule or action budget differs')
    return manifest


def cases(manifest, stage=None, shard_index=0, shards=1):
    validate_manifest(manifest)
    if stage not in (None, 'calibration') or type(shards) is not int or type(shard_index) is not int or not 0 <= shard_index < shards:
        raise ValueError('Invalid calibration stage or shard')
    return [dict(**row, init=44) for i,row in enumerate(manifest['case_table']) if i % shards == shard_index]


def planned_config(manifest, condition):
    validate_manifest(manifest)
    if condition != 'full_d10':
        raise ValueError('Only Full is planned for this calibration')
    return conditions()[0]


def validate_output(output, manifest):
    validate_manifest(manifest)
    if Path(output).resolve() != Path(manifest['output_root']).resolve():
        raise ValueError('Output differs from frozen calibration destination')
    for key in ('protocol_evidence','design_evidence'):
        evidence=manifest[key]
        if sha(evidence['path']) != evidence['sha256']:
            raise ValueError('Calibration evidence changed: '+key)


def read_manifest(path, verify_sources=False):
    value=_read_manifest(path,verify_sources=verify_sources)
    validate_output(Path(path).resolve().parent,value)
    return value


def runtime_revision(manifest):
    value=dict(name='common-suite-full-calibration-v1', scientific_manifest_hash=manifest['manifest_hash'],
        phase=manifest['phase'],protocol_evidence=manifest['protocol_evidence'],
        design_evidence=manifest['design_evidence'],wrapper_source_files=manifest['local_runtime_sources'],
        backend_reused='preparations.ease_insights_20261103.vlm.backend.Backend',
        policy_action_limits=dict(LIMITS),settling_actions=10,legacy_sources_changed=False,
        no_task_filtering=True,scope='Independent health calibration; no fast-path outcome claim')
    value['revision_hash']=digest(value)
    return value

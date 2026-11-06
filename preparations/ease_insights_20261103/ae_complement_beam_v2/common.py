"""One proxy-selected AE16 pair, fixed directions and independent world stages."""
import json
from pathlib import Path
from preparations.idea1_v3_1.common import ROOT, SUITES, LIMITS, atomic_json, digest
from preparations.idea1_v3_1.common import read_manifest as original_read_manifest
from preparations.ease_insights_20261103.denoise_depth_paths_v1.common import REFERENCES, reference, sha

MODULE = 'preparations.ease_insights_20261103.ae_complement_beam_v2'
MODEL_PYTHON = ROOT / 'policies/realtime-vla-flash/.venv/bin/python'
ENV_PYTHON = ROOT / 'policies/libero-eval-v21/.venv/bin/python'
GATE_SUITES = SUITES
PATHS = dict(ae16_d2=(2, 8), path_a=(2, 13), path_b=(7, 14))
CONDITIONS = ('full_d10', *PATHS)
NEW_CONDITIONS = ('path_a', 'path_b')
REFERENCE_CONDITIONS = {'full_d10': 'full_d10', 'ae16_d2': 'ae_16'}
DIRECTIONS = [dict(suite='libero_object', better='path_a', worse='path_b'),
              dict(suite='libero_spatial', better='path_b', worse='path_a')]
SCREEN_SELECTION = ROOT / 'results/ease_insights_20261103/ae_complement_beam_v1/analysis/selection.json'
SCREEN_SELECTION_SHA = '8bdd13b5c091387ecba67c721df8452543cccc03a9305e08fde5646cceb42c11'


def require(value, message):
    if not value:
        raise ValueError(message)


def conditions():
    return [dict(id=cid, model='pi0', mode='full' if cid == 'full_d10' else 'periodic',
                 compensation=None if cid == 'full_d10' else 'extrapolate', period=2, H=50, R=5,
                 D=10 if cid == 'full_d10' else 2, refresh_D=10, skip=[], ae_skip=list(PATHS.get(cid, ())))
            for cid in CONDITIONS]


def validate_condition(config):
    require(config in conditions(), 'Condition differs from the frozen joint-mask pair/control definitions')
    return (), tuple(config['ae_skip'])


def proxy_evidence():
    require(sha(SCREEN_SELECTION) == SCREEN_SELECTION_SHA, 'Frozen offline selection changed')
    value = json.loads(SCREEN_SELECTION.read_text())
    payload = dict(value)
    declared = payload.pop('selection_hash')
    require(digest(payload) == declared and value['complete_observations'] == 110 and value['physical_episodes'] == 0, 'Not a complete offline selection')
    chosen = value['selection']['selected']
    require(value['selection']['status'] == 'proxy_candidate' and chosen['masks'] == [[2, 13], [7, 14]], 'Proxy masks changed')
    require([(d['suite'], d['lower_error_mask']) for d in chosen['directions']] ==
            [('libero_object', [2, 13]), ('libero_spatial', [7, 14])], 'Proxy directions changed')
    require(sha(value['analysis']['path']) == value['analysis']['sha256'], 'Offline analysis changed')
    return dict(path=str(SCREEN_SELECTION), sha256=SCREEN_SELECTION_SHA, selection_hash=declared)


def stage_for(manifest):
    for stage, field, inits in [('calibration', 'calibration_inits', [45]),
                               ('discovery', 'discovery_inits', [45, 46]), ('confirm', 'confirmation_inits', [47, 48, 49])]:
        if manifest.get('phase') == 'AE_joint_mask_' + stage + '_v2' and manifest.get(field) == inits:
            return stage
    raise ValueError('Wrong joint-mask world stage or initial assignment')


def validate_manifest(value):
    stage = stage_for(value)
    tasks = [0] if stage == 'calibration' else list(range(10))
    require(value['conditions'] == conditions() and value['suites'] == list(SUITES) and value['tasks'] == tasks and
            value['case_table'] == [dict(suite=s, task=t) for s in SUITES for t in tasks], 'Wrong world case/config universe')
    require(value['science']['policy_action_limits'] == LIMITS and value['science']['settling_actions'] == 10 and
            value['proxy_preregistered_directions'] == DIRECTIONS, 'Physical contract or proxy directions differ')
    if stage == 'calibration':
        require(value['planned_conditions'] == ['full_d10', 'ae16_d2'] and value['cases_per_condition'] == 4 and
                value['total_calibration_episodes'] == 8, 'Exactly eight known controls required')
    elif stage == 'discovery':
        require(value['planned_conditions'] == list(NEW_CONDITIONS) and value['cases_per_condition'] == 80 and
                value['total_new_discovery_episodes'] == 160 and value['reference_conditions'] == REFERENCE_CONDITIONS and
                value['analytical_conditions'] == list(CONDITIONS), 'Require160 new plus160 original control cells')
    else:
        require(value['planned_conditions'] == ['full_d10', *NEW_CONDITIONS] and value['cases_per_condition'] == 120 and
                value['total_confirmation_episodes'] == 360, 'Exactly Full and the frozen pair on120 independent cases')
    return value


def cases(manifest, stage=None, shard_index=0, shards=1):
    validate_manifest(manifest)
    actual = stage_for(manifest)
    require(stage in (None, actual) and type(shards) is int and type(shard_index) is int and 0 <= shard_index < shards, 'Wrong world shard')
    field = dict(calibration='calibration_inits', discovery='discovery_inits', confirm='confirmation_inits')[actual]
    return [dict(**row, init=i) for n, row in enumerate(manifest['case_table']) if n % shards == shard_index for i in manifest[field]]


def planned_config(manifest, cid):
    validate_manifest(manifest)
    require(cid in manifest['planned_conditions'], 'Condition not planned at this world stage')
    return next(c for c in conditions() if c['id'] == cid)


def validate_output(output, manifest):
    validate_manifest(manifest)
    require(Path(output).resolve() == Path(manifest['output_root']).resolve(), 'Wrong joint-mask output')
    require(manifest['proxy_selection_evidence'] == proxy_evidence(), 'Not the frozen offline proxy choice')
    for key in ('protocol_evidence', 'design_evidence', 'health_evidence'):
        ev = manifest[key]
        require(sha(ev['path']) == ev['sha256'], 'World design/health evidence changed: ' + key)
    health = json.loads(Path(manifest['health_evidence']['path']).read_text())
    require(health['health_passed'] is True and health['episodes'] == 40, 'Common-suite Full health gate required')
    if stage_for(manifest) != 'calibration':
        ev = manifest['calibration_evidence']
        require(sha(ev['path']) == ev['sha256'], 'Calibration acceptance changed')
        value = json.loads(Path(ev['path']).read_text())
        require(value['status'] == 'pass' and value['all_eight_controls_bitwise_identical'], 'Known Full/AE16 bridge required')
    if stage_for(manifest) == 'confirm':
        ev = manifest['selection_evidence']
        require(sha(ev['path']) == ev['sha256'], 'World selection changed')
        selected = json.loads(Path(ev['path']).read_text())
        body = dict(selected)
        identity = body.pop('selection_hash')
        require(digest(body) == identity == ev['selection_hash'] and selected['selection']['status'] == 'selected', 'Qualified complete world discovery required')
        require(manifest['preregistered_hypotheses'] == selected['selection']['selected']['directions'], 'Confirmation directions differ')


def read_manifest(path, verify_sources=False):
    value = original_read_manifest(path, verify_sources)
    validate_output(Path(path).resolve().parent, value)
    return value


def runtime_revision(manifest):
    value = dict(name='AE16-joint-mask-world-v2', scientific_manifest_hash=manifest['manifest_hash'], phase=manifest['phase'],
                 wrapper_source_files=manifest['local_runtime_sources'], protocol_evidence=manifest['protocol_evidence'],
                 original_sampler_sources_changed=False, actual_NFE_exported=True, actual_AE_layer_visits_exported=True,
                 policy_action_limits=dict(LIMITS), settling_actions=10,
                 scope='FullVLM18/two fresh vision27 branches; FullAE18D10 then two AE16D2 requests. Fixed equal Fast depth and NFE. Empty VLM mask leaves residual compensation unused.')
    value['revision_hash'] = digest(value)
    return value

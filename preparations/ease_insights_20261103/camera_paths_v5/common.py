"""Fixed fresh-camera allocations and separately frozen calibration/study contracts."""
from preparations.idea1_v3_1.common import ROOT,SUITES,LIMITS,atomic_json,digest
from preparations.ease_insights_20261103.denoise_depth_paths_v1.common import sha

from reports.ease_insights_20261103.prefix_kv_preparation_v5.common import FULL,PATTERNS,CONDITIONS,conditions,validate_condition

import importlib
import json
from pathlib import Path
from preparations.idea1_v3_1.common import read_manifest as original_read_manifest
MODULE='preparations.ease_insights_20261103.camera_paths_v5'
FAMILY_ROOT=ROOT/'results/ease_insights_20261103/camera_paths_v5'
MODEL_PYTHON=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
ENV_PYTHON=ROOT/'policies/libero-eval-v21/.venv/bin/python'
GATE_SUITES=SUITES
NEW_CONDITIONS=tuple(PATTERNS)
REFERENCES={
 'full_d10':dict(root=str(ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery'),cohort='vlm_diversity_v2',condition='full_d10',manifest_sha256='a4a996b03059874ceb10a21e1099985508051d7ae0392af944e739911cd1494b'),
 'dense_d2':dict(root=str(ROOT/'results/ease_insights_20261103/denoise_depth_paths_v1/discovery'),cohort='denoise_depth_paths_v1',condition='dense_d2',manifest_sha256='df34a6e280492733ee57862368cd41fdbd54cb114ae385feb06d0fc6bb076ddf')}


def require(value,message):
    if not value:raise ValueError(message)


def reference(alias,verify_sources=True):
    spec=REFERENCES[alias];root=Path(spec['root']);require(sha(root/'manifest.json')==spec['manifest_sha256'],'Preserved source manifest changed')
    prefix='preparations.ease_insights_20261103.'+spec['cohort'];cm=importlib.import_module(prefix+'.common');w=importlib.import_module(prefix+'.worker')
    m=cm.read_manifest(root/'manifest.json',verify_sources=verify_sources);cfg=next(c for c in m['conditions'] if c['id']==spec['condition'])
    require(cfg['H']==50 and cfg['R']==5 and cfg['D']==(10 if alias=='full_d10' else 2) and cfg['period']==2
        and cfg['skip']==[] and cfg.get('ae_skip',[])==[] and cfg['refresh_D']==10,'Old control active arithmetic differs')
    require(cfg['mode']==('full' if alias=='full_d10' else 'periodic')
        and cfg['compensation'] in (None,'extrapolate'),'Old control schedule differs')
    return root,m,cm,w,spec['condition']


def stage_for(m):
    for stage,key,initials in [('calibration','calibration_inits',[45]),('discovery','discovery_inits',[45,46]),('confirm','confirmation_inits',[10,11,12,13,14])]:
        if m.get('phase')=='prefix_attention_'+stage+'_v5' and m.get(key)==initials:return stage
    raise ValueError('Incorrect fresh-vision stage or initial-state assignment')


def validate_manifest(m):
    stage=stage_for(m);tasks=[0] if stage=='calibration' else list(range(10))
    require(m['conditions']==conditions() and m['suites']==list(SUITES) and m['tasks']==tasks
        and m['case_table']==[dict(suite=s,task=t) for s in SUITES for t in tasks]
        and m['science']['settling_actions']==10 and m['science']['policy_action_limits']==LIMITS
        and m['noise_repeat']==0,'Fresh-vision task/config/physical universe differs')
    if stage=='calibration':
        require(m['planned_conditions']==['full_d10','dense_d2'] and m['cases_per_condition']==4 and m['total_calibration_episodes']==8,'Need eight known Full/DenseD2 controls')
    elif stage=='discovery':
        require(m['planned_conditions']==list(NEW_CONDITIONS) and m['cases_per_condition']==80
            and m['total_new_discovery_episodes']==160 and m['analytical_cells']==320
            and m['reference_conditions']=={'full_d10':'full_d10','dense_d2':'dense_d2'},'Need160 new plus160 preserved control cells')
    else:
        require(m['planned_conditions']==list(CONDITIONS) and m['cases_per_condition']==200 and m['total_confirmation_episodes']==800,'Need800 independent within-family confirmation rollouts includingDenseD2 control')
    return m


def cases(m,stage=None,shard_index=0,shards=1):
    validate_manifest(m);actual=stage_for(m)
    require(stage in (None,actual) and type(shard_index) is int and type(shards) is int and 0<=shard_index<shards,'Wrong fresh-vision task shard')
    key={'calibration':'calibration_inits','discovery':'discovery_inits','confirm':'confirmation_inits'}[actual]
    return [dict(**row,init=i) for n,row in enumerate(m['case_table']) if n%shards==shard_index for i in m[key]]


def planned_config(m,cid):
    validate_manifest(m);require(cid in m['planned_conditions'],'Unplanned fresh-vision condition')
    return next(c for c in conditions() if c['id']==cid)


def validate_output(output,m):
    validate_manifest(m);stage=stage_for(m);expected=FAMILY_ROOT/('confirmation' if stage=='confirm' else stage)
    require(Path(output).resolve()==expected and m['output_root']==str(expected),'Wrong fresh-vision output binding')
    for key in ['protocol_evidence','design_evidence','health_evidence','antecedent_evidence']:
        value=m[key];require(sha(value['path'])==value['sha256'],'Fresh-vision evidence changed: '+key)
    health=json.loads(Path(m['health_evidence']['path']).read_text());require(health['health_passed'] and health['episodes']==40,'Need accepted Full health')
    previous=json.loads(Path(m['antecedent_evidence']['path']).read_text());require(previous['status']=='complete_no_candidate' and previous['new_quality_cases']==160,'Previous fresh-vision branch not finished as registered')
    for alias in REFERENCES:require(sha(Path(REFERENCES[alias]['root'])/'manifest.json')==REFERENCES[alias]['manifest_sha256'],'Original control manifest changed')
    if stage!='calibration':
        ev=m['calibration_evidence'];require(sha(ev['path'])==ev['sha256'],'Fresh-vision calibration changed');proof=json.loads(Path(ev['path']).read_text())
        require(proof['status']=='pass' and proof['all_eight_controls_bitwise_identical'] and proof['packed_prefix_identity_passed']
            and proof['independent_masked_prefix_reference_passed'] and proof['joint_cost_qualification']['pass'],'Need real vision parity,oracle,known controls and matched-cost qualification')
    if stage=='confirm':
        ev=m['selection_evidence'];require(sha(ev['path'])==ev['sha256'],'Fresh-vision selection changed');selected=json.loads(Path(ev['path']).read_text());payload=dict(selected);declared=payload.pop('selection_hash')
        require(digest(payload)==declared==ev['selection_hash'] and selected['selection']['status']=='selected','Need frozen qualified visual pair')
        choice=selected['selection']['selected'];require(choice['paths']==list(NEW_CONDITIONS) and m['preregistered_hypotheses']==choice['directions'],'Frozen visual directions differ')


def read_manifest(path,verify_sources=False):
    m=original_read_manifest(path,verify_sources);validate_output(Path(path).resolve().parent,m);return m


def runtime_revision(m):
    value=dict(name='full-fresh-vision-packed-AE-prefix-v5',scientific_manifest_hash=m['manifest_hash'],phase=m['phase'],
        wrapper_source_files=m['local_runtime_sources'],protocol_evidence=m['protocol_evidence'],
        actual_AE_layer_visits_exported=True,actual_NFE_exported=True,actual_vision_layers_exported=True,actual_prefix_key_lengths_exported=True,packing_and_restore_cost_exported=True,
        policy_action_limits=dict(LIMITS),settling_actions=10,original_sampler_sources_changed=False,
        scope='Both cameras fresh/native27+27andVLM18prefill;AE18FastD2center128orchecker128imagekeyspercamera,alltextandoriginalabsoluteRoPE.FullD10allkeys.Noadditionaltraining.')
    value['revision_hash']=digest(value);return value

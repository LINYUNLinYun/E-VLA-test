"""Frozen equal-budget camera paths and separate calibration/discovery stages."""
import hashlib
import json
from pathlib import Path
from preparations.idea1_v3_1.common import ROOT,SUITES,LIMITS,atomic_json,digest
from preparations.idea1_v3_1.common import read_manifest as _read_manifest

MODULE='preparations.ease_insights_20261103.camera_paths_v1'
MODEL_PYTHON=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
ENV_PYTHON=ROOT/'policies/libero-eval-v21/.venv/bin/python'
GATE_SUITES=SUITES
CAMERAS={'both_fresh_d5':None,'agent_fresh_d5':0,'wrist_fresh_d5':1}
NEW_CONDITIONS=tuple(CAMERAS)
PAIR_PATHS=('agent_fresh_d5','wrist_fresh_d5')
CONDITIONS=('full_d10',*NEW_CONDITIONS)
REFERENCE_ROOT=ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery'
REFERENCE_MANIFEST_SHA='a4a996b03059874ceb10a21e1099985508051d7ae0392af944e739911cd1494b'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditions():
    return [dict(id=cid,model='pi0',mode='full' if cid=='full_d10' else 'periodic',
        compensation=None,period=4,H=50,R=5,D=10 if cid=='full_d10' else 5,refresh_D=10,
        skip=[],ae_skip=[],fresh_camera=CAMERAS.get(cid),
        vision_cache='none' if cid in ('full_d10','both_fresh_d5') else 'last_own_Full_raw_features') for cid in CONDITIONS]


def validate_condition(config):
    if config not in conditions():raise ValueError('Camera condition differs from four fixed definitions')
    return (),()


def stage_for(manifest):
    phase=manifest.get('phase')
    for stage,key,inits in [('calibration','calibration_inits',[45]),('discovery','discovery_inits',[45,46]),('confirm','confirmation_inits',[47,48,49])]:
        if phase=='camera_paths_'+stage+'_v1' and manifest.get(key)==inits:return stage
    raise ValueError('Not a fixed camera stage/initial-state assignment')


def validate_manifest(m):
    stage=stage_for(m);tasks=[0] if stage=='calibration' else list(range(10))
    if (m.get('conditions')!=conditions() or m.get('suites')!=list(SUITES) or m.get('tasks')!=tasks
        or m.get('case_table')!=[dict(suite=s,task=t) for s in SUITES for t in tasks]
        or m.get('science',{}).get('policy_action_limits')!=LIMITS or m.get('science',{}).get('settling_actions')!=10):
        raise ValueError('Camera task/config/action-budget universe differs')
    planned=m.get('planned_conditions')
    if stage=='calibration':
        if planned!=['full_d10'] or m.get('cases_per_condition')!=4 or m.get('total_calibration_episodes')!=4:
            raise ValueError('Camera calibration is four repeated known Full controls')
    elif stage=='discovery':
        if (planned!=list(NEW_CONDITIONS) or m.get('cases_per_condition')!=80 or m.get('total_new_discovery_episodes')!=240
            or m.get('reference_conditions')!={'full_d10':'full_d10'} or m.get('analytical_cells')!=320):
            raise ValueError('Camera discovery is three new paths times80, plus80 preserved Full controls')
    elif (planned!=['full_d10',*PAIR_PATHS] or m.get('cases_per_condition')!=120 or m.get('total_confirmation_episodes')!=360):
        raise ValueError('Camera confirmation requires Full and the frozen camera pair on120 unused cases')
    return m


def cases(manifest,stage=None,shard_index=0,shards=1):
    validate_manifest(manifest);actual=stage_for(manifest)
    if (stage not in (None,actual) or type(shards) is not int or type(shard_index) is not int
        or not 0<=shard_index<shards):raise ValueError('Invalid camera case shard')
    field={'calibration':'calibration_inits','discovery':'discovery_inits','confirm':'confirmation_inits'}[actual]
    return [dict(**row,init=i) for n,row in enumerate(manifest['case_table']) if n%shards==shard_index for i in manifest[field]]


def planned_config(manifest,condition):
    validate_manifest(manifest)
    if condition not in manifest['planned_conditions']:raise ValueError('Condition not scheduled for this camera stage')
    return next(c for c in conditions() if c['id']==condition)


def validate_output(output,manifest):
    validate_manifest(manifest)
    if Path(output).resolve()!=Path(manifest['output_root']).resolve():raise ValueError('Camera output differs')
    for key in ('protocol_evidence','design_evidence','health_evidence'):
        ev=manifest[key]
        if sha(ev['path'])!=ev['sha256']:raise ValueError('Camera evidence changed: '+key)
    health=json.loads(Path(manifest['health_evidence']['path']).read_text())
    if health.get('health_passed') is not True or health.get('episodes')!=40:raise ValueError('Completed common-suite Full health evidence required')
    if stage_for(manifest)!='calibration':
        ev=manifest['calibration_evidence']
        if sha(ev['path'])!=ev['sha256']:raise ValueError('Camera calibration evidence changed')
        accepted=json.loads(Path(ev['path']).read_text())
        if (accepted.get('status')!='pass' or not accepted.get('GPU_native_parity_passed')
            or not accepted.get('old_Full_bridge_passed')):raise ValueError('Camera numerical/runtime bridge not accepted')
    if stage_for(manifest)=='discovery' and sha(REFERENCE_ROOT/'manifest.json')!=REFERENCE_MANIFEST_SHA:
        raise ValueError('Preserved Full manifest changed')
    if stage_for(manifest)=='confirm':
        ev=manifest['selection_evidence'];value=json.loads(Path(ev['path']).read_text());payload=dict(value);key=payload.pop('selection_hash',None)
        if sha(ev['path'])!=ev['sha256'] or key!=ev['selection_hash'] or digest(payload)!=key:raise ValueError('Camera selection changed')
        chosen=value['selection']
        if (chosen['status']!='selected' or chosen['selected']['paths']!=list(PAIR_PATHS)
            or manifest['preregistered_hypotheses']!=chosen['selected']['directions']):raise ValueError('Camera confirmation differs from frozen directions')


def read_manifest(path,verify_sources=False):
    value=_read_manifest(path,verify_sources=verify_sources);validate_output(Path(path).resolve().parent,value);return value


def runtime_revision(manifest):
    value=dict(name='camera-paths-observed-single-view-v1',scientific_manifest_hash=manifest['manifest_hash'],phase=manifest['phase'],
        protocol_evidence=manifest['protocol_evidence'],design_evidence=manifest['design_evidence'],wrapper_source_files=manifest['local_runtime_sources'],
        native_sampler_reused='preparations.idea1_v2_2.pi0_adapter.Pi0Adapter.sample',original_sampler_sources_changed=False,
        actual_NFE_exported=True,actual_AE_layer_visits_exported=True,actual_vision_input_batch_exported=True,
        policy_action_limits=dict(LIMITS),settling_actions=10,
        scope='VLM18/AE18; FullD10 two fresh views; four FastD5 rounds with one fresh view and one held last-own-Full feature map; both-fresh D5 control')
    value['revision_hash']=digest(value);return value

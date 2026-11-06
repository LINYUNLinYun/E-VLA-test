"""Prospective module axes and independent calibration/discovery/confirmation."""
import hashlib
import json
from pathlib import Path
from preparations.idea1_v3_1.common import ROOT,SUITES,LIMITS,atomic_json,digest
from preparations.idea1_v3_1.common import read_manifest as _read_manifest

MODULE='preparations.ease_insights_20261103.module_tradeoff_v1'
MODEL_PYTHON=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
ENV_PYTHON=ROOT/'policies/libero-eval-v21/.venv/bin/python'
GATE_SUITES=SUITES
PATHS=dict(vlm_c=((4,7,10,13,15,17),()),ae_16=((),(2,8)),ae_14=((),(2,6,7,8)),
           ae_12=((),(2,4,6,7,8,9)),hybrid_15_15=((2,3,17),(2,7,8)))
CONDITIONS=('full_d10',*PATHS)
NEW_CONDITIONS=('ae_16','ae_14','ae_12','hybrid_15_15')
REFERENCE_ROOT=ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery'
REFERENCE_MANIFEST_SHA='a4a996b03059874ceb10a21e1099985508051d7ae0392af944e739911cd1494b'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditions():
    return [dict(id=cid,model='pi0',mode='full' if cid=='full_d10' else 'periodic',
        compensation=None if cid=='full_d10' else 'extrapolate',period=2,H=50,R=5,D=10 if cid=='full_d10' else 2,
        refresh_D=10,skip=list(PATHS.get(cid,((),()))[0]),ae_skip=list(PATHS.get(cid,((),()))[1])) for cid in CONDITIONS]


def validate_condition(config):
    if config not in conditions():raise ValueError('Module condition differs from six fixed definitions')
    return tuple(config['skip']),tuple(config['ae_skip'])


def stage_for(manifest):
    phase=manifest.get('phase')
    for stage,key,inits in [('calibration','calibration_inits',[45]),('discovery','discovery_inits',[45,46]),('confirm','confirmation_inits',[47,48,49])]:
        if phase=='module_tradeoff_'+stage+'_v1' and manifest.get(key)==inits:return stage
    raise ValueError('Not a fixed module stage/initial-state assignment')


def validate_manifest(m):
    stage=stage_for(m);tasks=[0] if stage=='calibration' else list(range(10))
    table=[dict(suite=s,task=t) for s in SUITES for t in tasks]
    if (m.get('conditions')!=conditions() or m.get('suites')!=list(SUITES) or m.get('tasks')!=tasks
        or m.get('case_table')!=table or m.get('science',{}).get('policy_action_limits')!=LIMITS
        or m.get('science',{}).get('settling_actions')!=10):raise ValueError('Module task/config/action-budget universe differs')
    planned=m.get('planned_conditions')
    if stage=='calibration':
        if planned!=['full_d10'] or m.get('cases_per_condition')!=4 or m.get('total_calibration_episodes')!=4:
            raise ValueError('Calibration is exactly four known Full controls, not independent quality evidence')
    elif stage=='discovery':
        if planned!=list(NEW_CONDITIONS) or m.get('cases_per_condition')!=80 or m.get('total_new_discovery_episodes')!=320:
            raise ValueError('Discovery requires exactly four new paths times80 cases')
        if m.get('reference_conditions')!={'full_d10':'full_d10','vlm_c':'spread_c'}:
            raise ValueError('Only exact preserved prior Full/spread_c cases may be referenced')
    elif (not isinstance(planned,list) or len(planned)!=3 or planned[0]!='full_d10' or planned[1:]!=sorted(set(planned[1:]))
          or any(c not in PATHS for c in planned[1:]) or m.get('cases_per_condition')!=120 or m.get('total_confirmation_episodes')!=360):
        raise ValueError('Confirmation requires Full and two selected paths on120 unused cases')
    return m


def cases(manifest,stage=None,shard_index=0,shards=1):
    validate_manifest(manifest);actual=stage_for(manifest)
    if stage not in (None,actual) or type(shards) is not int or type(shard_index) is not int or not 0<=shard_index<shards:
        raise ValueError('Invalid module case shard')
    field={'calibration':'calibration_inits','discovery':'discovery_inits','confirm':'confirmation_inits'}[actual]
    return [dict(**row,init=i) for n,row in enumerate(manifest['case_table']) if n%shards==shard_index for i in manifest[field]]


def planned_config(manifest,condition):
    validate_manifest(manifest)
    if condition not in manifest['planned_conditions']:raise ValueError('Condition not scheduled for this module stage')
    return next(c for c in conditions() if c['id']==condition)


def validate_output(output,manifest):
    validate_manifest(manifest)
    if Path(output).resolve()!=Path(manifest['output_root']).resolve():raise ValueError('Module output differs')
    for key in ('protocol_evidence','design_evidence','health_evidence'):
        ev=manifest[key]
        if sha(ev['path'])!=ev['sha256']:raise ValueError('Module evidence changed: '+key)
    health=json.loads(Path(manifest['health_evidence']['path']).read_text())
    if health.get('health_passed') is not True or health.get('episodes')!=40:raise ValueError('Need completed common-suite Full health acceptance')
    if stage_for(manifest)!='calibration':
        ev=manifest['calibration_evidence']
        if sha(ev['path'])!=ev['sha256']:raise ValueError('Module numerical calibration evidence changed')
        cal=json.loads(Path(ev['path']).read_text())
        if cal.get('status')!='pass' or not cal.get('GPU_native_parity_passed') or not cal.get('old_Full_bridge_passed'):
            raise ValueError('Module runtime/calibration bridge not accepted')
    if stage_for(manifest)=='discovery':
        if sha(REFERENCE_ROOT/'manifest.json')!=REFERENCE_MANIFEST_SHA:raise ValueError('Preserved baseline manifest changed')
    if stage_for(manifest)=='confirm':
        ev=manifest['selection_evidence'];selected=json.loads(Path(ev['path']).read_text());payload=dict(selected);key=payload.pop('selection_hash',None)
        if sha(ev['path'])!=ev['sha256'] or key!=ev['selection_hash'] or digest(payload)!=key:raise ValueError('Module selection changed')
        chosen=selected['selection']
        if chosen['status']!='selected' or manifest['planned_conditions']!=['full_d10',*chosen['selected']['paths']] or manifest['preregistered_hypotheses']!=chosen['selected']['directions']:
            raise ValueError('Module confirmation differs from frozen pair/directions')


def read_manifest(path,verify_sources=False):
    value=_read_manifest(path,verify_sources=verify_sources);validate_output(Path(path).resolve().parent,value);return value


def runtime_revision(manifest):
    value=dict(name='module-tradeoff-observed-AE-v1',scientific_manifest_hash=manifest['manifest_hash'],phase=manifest['phase'],
        protocol_evidence=manifest['protocol_evidence'],design_evidence=manifest['design_evidence'],wrapper_source_files=manifest['local_runtime_sources'],
        native_sampler_reused='preparations.idea1_v2_2.pi0_adapter.Pi0Adapter.sample',original_sampler_sources_changed=False,
        actual_NFE_exported=True,actual_AE_layer_visits_exported=True,policy_action_limits=dict(LIMITS),settling_actions=10,
        scope='Explicit VLM/AE axes; FullD10 keepsAE18, FastD2 uses configured AE bypass; no layer-count cost-equivalence claim')
    value['revision_hash']=digest(value);return value

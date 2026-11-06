"""Fixed depth/time candidates, explicit two-origin controls and separate stages."""
import hashlib
import importlib
import json
from pathlib import Path
from preparations.idea1_v3_1.common import ROOT,SUITES,LIMITS,atomic_json,digest
from preparations.idea1_v3_1.common import read_manifest as _read_manifest

MODULE='preparations.ease_insights_20261103.ae_complement_beam_v3'
MODEL_PYTHON=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
ENV_PYTHON=ROOT/'policies/libero-eval-v21/.venv/bin/python'
GATE_SUITES=SUITES
PATHS=dict(ae14_base=(3,(2,6,7,8)),path_a=(3,(2,4,7,8)),path_b=(3,(2,4,6,8)),
           path_c=(3,(2,4,6,7)),path_d=(3,(2,6,7,9)))
CONDITIONS=('full_d10',*PATHS)
NEW_CONDITIONS=('path_a','path_b','path_c','path_d')
REFERENCE_CONDITIONS={'full_d10':'full_d10','ae14_base':'ae14_d3'}
REFERENCES={
    'full_d10':dict(root=str(ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery'),cohort='vlm_diversity_v2',condition='full_d10',
        manifest_sha256='a4a996b03059874ceb10a21e1099985508051d7ae0392af944e739911cd1494b',
        case_audit=dict(path=str(ROOT/'reports/ease_insights_20261103/vlm_diversity_v2/full_discovery_audit.json'),sha256='41d70066f98bbe69ddb003a13cc87495e8e46fff4b8bdb5942e84ce0af31a383')),
    'ae14_base':dict(root=str(ROOT/'results/ease_insights_20261103/denoise_depth_paths_v1/discovery'),cohort='denoise_depth_paths_v1',condition='ae14_d3',
        manifest_sha256='df34a6e280492733ee57862368cd41fdbd54cb114ae385feb06d0fc6bb076ddf',
        case_audit=dict(path=str(ROOT/'reports/ease_insights_20261103/denoise_depth_paths_v1/discovery_acceptance.json'),sha256='e08a96d40b91f3cc8347a44de844edffeee49c94d3f4fceba16ffdd34760079f'))}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditions():
    return [dict(id=cid,model='pi0',mode='full' if cid=='full_d10' else 'periodic',
        compensation=None if cid=='full_d10' else 'extrapolate',period=2,H=50,R=5,
        D=10 if cid=='full_d10' else PATHS[cid][0],refresh_D=10,skip=[],ae_skip=[] if cid=='full_d10' else list(PATHS[cid][1])) for cid in CONDITIONS]


def validate_condition(config):
    if config not in conditions():raise ValueError('Depth/time condition differs from six fixed definitions')
    return (),tuple(config['ae_skip'])


def reference(alias,verify_sources=True):
    spec=REFERENCES[alias];root=Path(spec['root'])
    if sha(root/'manifest.json')!=spec['manifest_sha256'] or sha(spec['case_audit']['path'])!=spec['case_audit']['sha256']:
        raise ValueError('Preserved depth/time control identity changed: '+alias)
    prefix='preparations.ease_insights_20261103.'+spec['cohort']
    common=importlib.import_module(prefix+'.common');worker=importlib.import_module(prefix+'.worker')
    manifest=common.read_manifest(root/'manifest.json',verify_sources=verify_sources)
    old=dict(next(c for c in manifest['conditions'] if c['id']==spec['condition']))
    old['id']=alias;old.setdefault('ae_skip',[])
    if old!=next(c for c in conditions() if c['id']==alias):raise ValueError('Preserved control active algorithm differs: '+alias)
    return root,manifest,common,worker,spec['condition']


def stage_for(manifest):
    for stage,key,inits in [('calibration','calibration_inits',[45]),('discovery','discovery_inits',[45,46]),('confirm','confirmation_inits',[47,48,49])]:
        if manifest.get('phase')=='AE14_neighbor_'+stage+'_v3' and manifest.get(key)==inits:return stage
    raise ValueError('Not a fixed depth/time stage/initial-state assignment')


def validate_manifest(m):
    stage=stage_for(m);tasks=[0] if stage=='calibration' else list(range(10));planned=m.get('planned_conditions')
    if (m.get('conditions')!=conditions() or m.get('suites')!=list(SUITES) or m.get('tasks')!=tasks
        or m.get('case_table')!=[dict(suite=s,task=t) for s in SUITES for t in tasks]
        or m.get('science',{}).get('policy_action_limits')!=LIMITS or m.get('science',{}).get('settling_actions')!=10):
        raise ValueError('Depth/time task/config/physical universe differs')
    if stage=='calibration':
        if planned!=['full_d10','ae14_base'] or m.get('cases_per_condition')!=4 or m.get('total_calibration_episodes')!=8:
            raise ValueError('Depth/time calibration is eight repeated known controls from two origins')
    elif stage=='discovery':
        if (planned!=list(NEW_CONDITIONS) or m.get('cases_per_condition')!=80 or m.get('total_new_discovery_episodes')!=320
            or m.get('reference_conditions')!=REFERENCE_CONDITIONS or m.get('analytical_cells')!=480):
            raise ValueError('Depth/time discovery requires320 new plus160 original control cells')
    elif (not isinstance(planned,list) or len(planned)!=3 or planned[0]!='full_d10' or planned[1:]!=sorted(set(planned[1:]))
        or any(c not in PATHS for c in planned[1:]) or m.get('cases_per_condition')!=120 or m.get('total_confirmation_episodes')!=360):
        raise ValueError('Depth/time confirmation requires Full and two selected paths on120 unused cases')
    return m


def cases(manifest,stage=None,shard_index=0,shards=1):
    validate_manifest(manifest);actual=stage_for(manifest)
    if stage not in (None,actual) or type(shards) is not int or type(shard_index) is not int or not 0<=shard_index<shards:
        raise ValueError('Invalid depth/time case shard')
    field={'calibration':'calibration_inits','discovery':'discovery_inits','confirm':'confirmation_inits'}[actual]
    return [dict(**row,init=i) for n,row in enumerate(manifest['case_table']) if n%shards==shard_index for i in manifest[field]]


def planned_config(manifest,condition):
    validate_manifest(manifest)
    if condition not in manifest['planned_conditions']:raise ValueError('Condition not planned in this depth/time stage')
    return next(c for c in conditions() if c['id']==condition)


def validate_output(output,manifest):
    validate_manifest(manifest)
    if Path(output).resolve()!=Path(manifest['output_root']).resolve():raise ValueError('Depth/time output differs')
    for key in ('protocol_evidence','design_evidence','health_evidence'):
        ev=manifest[key]
        if sha(ev['path'])!=ev['sha256']:raise ValueError('Depth/time evidence changed: '+key)
    health=json.loads(Path(manifest['health_evidence']['path']).read_text())
    if health.get('health_passed') is not True or health.get('episodes')!=40:raise ValueError('Accepted common-suite Full health required')
    for spec in REFERENCES.values():
        if sha(Path(spec['root'])/'manifest.json')!=spec['manifest_sha256'] or sha(spec['case_audit']['path'])!=spec['case_audit']['sha256']:
            raise ValueError('Original control manifest/audit changed')
    if stage_for(manifest)!='calibration':
        ev=manifest['calibration_evidence']
        if sha(ev['path'])!=ev['sha256']:raise ValueError('Depth/time calibration evidence changed')
        proof=json.loads(Path(ev['path']).read_text())
        if (proof.get('status')!='pass' or not proof.get('GPU_adapter_parity_passed')
            or not proof.get('old_Full_bridge_passed') or not proof.get('old_AE14_bridge_passed')):raise ValueError('Eight-control adapter/runtime bridge not accepted')
    if stage_for(manifest)=='confirm':
        ev=manifest['selection_evidence'];value=json.loads(Path(ev['path']).read_text());payload=dict(value);key=payload.pop('selection_hash',None)
        if sha(ev['path'])!=ev['sha256'] or key!=ev['selection_hash'] or digest(payload)!=key:raise ValueError('Depth/time selection changed')
        chosen=value['selection']
        if (chosen['status']!='selected' or manifest['planned_conditions']!=['full_d10',*chosen['selected']['paths']]
            or manifest['preregistered_hypotheses']!=chosen['selected']['directions']):raise ValueError('Depth/time confirmation differs from frozen pair/directions')


def read_manifest(path,verify_sources=False):
    value=_read_manifest(path,verify_sources=verify_sources);validate_output(Path(path).resolve().parent,value);return value


def runtime_revision(manifest):
    value=dict(name='AE14-neighbor-observed-AE-v3',scientific_manifest_hash=manifest['manifest_hash'],phase=manifest['phase'],
        protocol_evidence=manifest['protocol_evidence'],design_evidence=manifest['design_evidence'],wrapper_source_files=manifest['local_runtime_sources'],
        native_sampler_reused='preparations.idea1_v2_2.pi0_adapter.Pi0Adapter.sample',original_sampler_sources_changed=False,
        actual_NFE_exported=True,actual_AE_layer_visits_exported=True,policy_action_limits=dict(LIMITS),settling_actions=10,
        scope='FullVLM18/vision27 and two fresh cameras; all Fast paths retain14 AE layers with uniform EulerD3, one swap from the known anchor. FullD10 AE18; empty VLM masks leave residual compensation unused. Logical layer counts do not substitute for measured cost.')
    value['revision_hash']=digest(value);return value

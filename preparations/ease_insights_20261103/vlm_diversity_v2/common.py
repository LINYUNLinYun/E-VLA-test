"""Four prospectively defined, geometrically separated masks at a fixed budget."""
import hashlib
import json
from pathlib import Path
from preparations.idea1_v3_1.common import ROOT, SUITES, LIMITS, atomic_json, digest
from preparations.idea1_v3_1.common import read_manifest as _read_manifest

MODULE='preparations.ease_insights_20261103.vlm_diversity_v2'
MODEL_PYTHON=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
ENV_PYTHON=ROOT/'policies/libero-eval-v21/.venv/bin/python'
GATE_SUITES=SUITES
DISCOVERY_PHASE='diversity_discovery_v2'
CONFIRMATION_PHASE='diversity_confirmation_v2'
MASKS=dict(spread_a=(2,4,6,8,10,12),spread_b=(3,6,9,12,14,16),
           spread_c=(4,7,10,13,15,17),spread_d=(4,6,8,11,13,15))
CONDITIONS=('full_d10',*MASKS)
ANALYSIS_FILES=[f'preparations/ease_insights_20261103/vlm_diversity_v2/{name}' for name in ('selection.py','analyze.py')]+[
    f'preparations/ease_insights_20261103/{name}' for name in ('vlm/analyze.py','vlm/analyze_confirmation.py','pipeline.py')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditions():
    return [dict(id=cid,model='pi0',mode='full' if cid=='full_d10' else 'periodic',
        compensation=None if cid=='full_d10' else 'extrapolate',period=2,H=50,R=5,
        D=10 if cid=='full_d10' else 2,refresh_D=10,skip=list(MASKS.get(cid,()))) for cid in CONDITIONS]


def validate_condition(config):
    if config not in conditions():
        raise ValueError('Condition differs from the five-condition broad-mask whitelist')
    return tuple(config['skip'])


def stage_for(manifest):
    if manifest.get('phase')==DISCOVERY_PHASE and manifest.get('discovery_inits')==[45,46]:
        return 'discovery'
    if manifest.get('phase')==CONFIRMATION_PHASE and manifest.get('confirmation_inits')==[47,48,49]:
        return 'confirm'
    raise ValueError('Diversity study requires fixed discovery45-46 or confirmation47-49')


def validate_manifest(manifest):
    stage=stage_for(manifest)
    if (manifest.get('conditions')!=conditions() or manifest.get('suites')!=list(SUITES)
        or manifest.get('tasks')!=list(range(10))
        or manifest.get('case_table')!=[dict(suite=s,task=t) for s in SUITES for t in range(10)]
        or manifest.get('science',{}).get('policy_action_limits')!=LIMITS
        or manifest.get('science',{}).get('settling_actions')!=10):
        raise ValueError('Diversity study task/condition/action-budget universe differs')
    planned=manifest.get('planned_conditions')
    if stage=='discovery':
        if planned!=list(CONDITIONS) or manifest.get('cases_per_condition')!=80 or manifest.get('total_discovery_episodes')!=400:
            raise ValueError('Discovery must cover5x40x2=400 cases')
    elif (not isinstance(planned,list) or len(planned)!=3 or planned[0]!='full_d10'
          or planned[1:]!=sorted(set(planned[1:])) or any(c not in MASKS for c in planned[1:])
          or manifest.get('cases_per_condition')!=120 or manifest.get('total_confirmation_episodes')!=360):
        raise ValueError('Confirmation requires Full plus exactly two selected paths,3x40x3=360 cases')
    return manifest


def cases(manifest,stage=None,shard_index=0,shards=1):
    validate_manifest(manifest);actual=stage_for(manifest)
    if stage not in (None,actual) or type(shards) is not int or type(shard_index) is not int or not 0<=shard_index<shards:
        raise ValueError('Invalid diversity-study stage or shard')
    inits=manifest['discovery_inits' if actual=='discovery' else 'confirmation_inits']
    return [dict(**row,init=i) for n,row in enumerate(manifest['case_table']) if n%shards==shard_index for i in inits]


def planned_config(manifest,condition):
    validate_manifest(manifest)
    if condition not in manifest['planned_conditions']:
        raise ValueError('Condition not planned for this stage')
    return next(c for c in conditions() if c['id']==condition)


def validate_output(output,manifest):
    validate_manifest(manifest)
    if Path(output).resolve()!=Path(manifest['output_root']).resolve():
        raise ValueError('Output differs from the frozen diversity-study destination')
    for key in ('protocol_evidence','design_evidence','health_evidence'):
        ev=manifest[key]
        if sha(ev['path'])!=ev['sha256']:
            raise ValueError('Diversity-study evidence changed: '+key)
    health=json.loads(Path(manifest['health_evidence']['path']).read_text())
    if health.get('status')!='complete' or health.get('episodes')!=40 or health.get('health_passed') is not True:
        raise ValueError('The new search requires the completed, prospectively accepted Full calibration')
    if stage_for(manifest)=='confirm':
        ev=manifest['selection_evidence'];selected=json.loads(Path(ev['path']).read_text())
        payload=dict(selected);identity=payload.pop('selection_hash',None)
        if sha(ev['path'])!=ev['sha256'] or identity!=ev['selection_hash'] or digest(payload)!=identity:
            raise ValueError('Frozen selection changed')
        choice=selected['selection']
        if (choice['status']!='selected' or manifest['planned_conditions']!=['full_d10',*choice['selected']['paths']]
            or manifest['preregistered_hypotheses']!=choice['selected']['directions']):
            raise ValueError('Confirmation differs from the frozen pair and directions')


def read_manifest(path,verify_sources=False):
    value=_read_manifest(path,verify_sources=verify_sources)
    validate_output(Path(path).resolve().parent,value)
    return value


def runtime_revision(manifest):
    value=dict(name='vlm-broad-mask-diversity-v2',scientific_manifest_hash=manifest['manifest_hash'],
        phase=manifest['phase'],protocol_evidence=manifest['protocol_evidence'],design_evidence=manifest['design_evidence'],
        wrapper_source_files=manifest['local_runtime_sources'],backend_reused='preparations.ease_insights_20261103.vlm.backend.Backend',
        mechanics_reused='preparations.ease_insights_20261103.vlm.policy.Policy.predict/reset/acknowledge',
        policy_action_limits=dict(LIMITS),settling_actions=10,legacy_sources_changed=False,
        fixed_mask_budget=6,nominal_VLM_retained=12,actual_fast_hidden_updates=11,
        scope='New broad masks, same D10-D2-D2 and own-path residual extrapolation; suite preferences remain hypotheses')
    value['revision_hash']=digest(value)
    return value

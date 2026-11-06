"""Fixed cross-module pair, known-control bridge and two fresh noise replicates."""
import importlib
import json
from pathlib import Path
from preparations.idea1_v3_1.common import ROOT,SUITES,LIMITS,atomic_json,digest
from preparations.idea1_v3_1.common import read_manifest as original_read_manifest
from preparations.ease_insights_20261103.denoise_depth_paths_v1.common import sha

MODULE='preparations.ease_insights_20261103.ae_complement_beam_v4'
FAMILY_ROOT=ROOT/'results/ease_insights_20261103/ae_complement_beam_v4'
MODEL_PYTHON=ROOT/'policies/realtime-vla-flash/.venv/bin/python'
ENV_PYTHON=ROOT/'policies/libero-eval-v21/.venv/bin/python'
GATE_SUITES=SUITES
CONDITIONS=('full_d10','vlm_c','ae14_d3')
DIRECTIONS=[dict(suite='libero_spatial',better='ae14_d3',worse='vlm_c'),dict(suite='libero_10',better='vlm_c',worse='ae14_d3')]
REFERENCES={
 'full_d10':dict(root=str(ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery'),cohort='vlm_diversity_v2',condition='full_d10',manifest_sha256='a4a996b03059874ceb10a21e1099985508051d7ae0392af944e739911cd1494b'),
 'vlm_c':dict(root=str(ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery'),cohort='vlm_diversity_v2',condition='spread_c',manifest_sha256='a4a996b03059874ceb10a21e1099985508051d7ae0392af944e739911cd1494b'),
 'ae14_d3':dict(root=str(ROOT/'results/ease_insights_20261103/denoise_depth_paths_v1/discovery'),cohort='denoise_depth_paths_v1',condition='ae14_d3',manifest_sha256='df34a6e280492733ee57862368cd41fdbd54cb114ae385feb06d0fc6bb076ddf')}


def require(v,message):
    if not v:raise ValueError(message)


def conditions():
    return [dict(id=cid,model='pi0',mode='full' if cid=='full_d10' else 'periodic',
                 compensation=None if cid=='full_d10' else 'extrapolate',period=2,H=50,R=5,
                 D=10 if cid=='full_d10' else 2 if cid=='vlm_c' else 3,refresh_D=10,
                 skip=[4,7,10,13,15,17] if cid=='vlm_c' else [],ae_skip=[2,6,7,8] if cid=='ae14_d3' else []) for cid in CONDITIONS]


def validate_condition(config):
    require(config in conditions(),'Not the fixed cross-module configuration')
    return tuple(config['skip']),tuple(config['ae_skip'])


def reference(alias,verify_sources=True):
    spec=REFERENCES[alias];root=Path(spec['root']);require(sha(root/'manifest.json')==spec['manifest_sha256'],'Original source manifest changed')
    prefix='preparations.ease_insights_20261103.'+spec['cohort'];cm=importlib.import_module(prefix+'.common');w=importlib.import_module(prefix+'.worker');m=cm.read_manifest(root/'manifest.json',verify_sources=verify_sources)
    cfg=dict(next(c for c in m['conditions'] if c['id']==spec['condition']));cfg['id']=alias;cfg.setdefault('ae_skip',[])
    require(cfg==next(c for c in conditions() if c['id']==alias),'Source active algorithm differs')
    return root,m,cm,w,spec['condition']


def stage_for(m):
    if m.get('phase')=='cross_module_calibration_v4' and m.get('calibration_inits')==[45]:return 'calibration'
    if m.get('phase')=='cross_module_confirm_v4' and m.get('confirmation_inits')==[47,48,49]:return 'confirm'
    raise ValueError('Incorrect cross-module stage/physical initial states')


def validate_manifest(m):
    stage=stage_for(m);tasks=[0] if stage=='calibration' else list(range(10));repeat=m.get('noise_repeat')
    require(m['conditions']==conditions() and m['planned_conditions']==list(CONDITIONS) and m['suites']==list(SUITES) and m['tasks']==tasks
            and m['case_table']==[dict(suite=s,task=t) for s in SUITES for t in tasks],'Cross-module task/config universe differs')
    require(type(repeat) is int and repeat in (0,1) and m['science']['settling_actions']==10 and m['science']['policy_action_limits']==LIMITS,'Noise/physical contract differs')
    if stage=='calibration':require(repeat==0 and m['cases_per_condition']==4 and m['total_calibration_episodes']==12,'Need12 repeated known controls with original noise')
    else:require(m['cases_per_condition']==120 and m['total_confirmation_episodes']==360 and m['preregistered_hypotheses']==DIRECTIONS,'Need360 episodes per noise stream and exactly fixed Spatial/Long directions')
    return m


def cases(m,stage=None,shard_index=0,shards=1):
    validate_manifest(m);actual=stage_for(m);require(stage in (None,actual) and type(shards) is int and type(shard_index) is int and 0<=shard_index<shards,'Wrong case shard')
    field='calibration_inits' if actual=='calibration' else 'confirmation_inits'
    return [dict(**row,init=i) for n,row in enumerate(m['case_table']) if n%shards==shard_index for i in m[field]]


def planned_config(m,cid):
    validate_manifest(m);require(cid in m['planned_conditions'],'Unplanned cross-module condition')
    return next(c for c in conditions() if c['id']==cid)


def validate_output(output,m):
    validate_manifest(m);stage=stage_for(m)
    expected=FAMILY_ROOT/'calibration' if stage=='calibration' else FAMILY_ROOT/'confirmation' if m['noise_repeat']==0 else FAMILY_ROOT
    require(Path(output).resolve()==expected and m['output_root']==str(expected),'Wrong physical output/noise stream binding')
    for key in ['protocol_evidence','design_evidence','health_evidence','historical_discovery_evidence']:
        value=m[key];require(sha(value['path'])==value['sha256'],'Evidence changed: '+key)
    health=json.loads(Path(m['health_evidence']['path']).read_text());require(health['health_passed'] and health['episodes']==40,'Full health acceptance missing')
    if stage=='confirm':
        ev=m['calibration_evidence'];require(sha(ev['path'])==ev['sha256'],'Joint calibration changed');accepted=json.loads(Path(ev['path']).read_text())
        require(accepted['status']=='pass' and accepted['all_twelve_controls_bitwise_identical'] and accepted['joint_cost_qualification']['pass'],'Need12-control bitwise bridge and joint same-service cost gate')


def read_manifest(path,verify_sources=False):
    m=original_read_manifest(path,verify_sources);validate_output(Path(path).resolve().parent,m);return m


def runtime_revision(m):
    v=dict(name='cross-module-original-adapter-v4',scientific_manifest_hash=m['manifest_hash'],phase=m['phase'],
           wrapper_source_files=m['local_runtime_sources'],protocol_evidence=m['protocol_evidence'],noise_repeat=m['noise_repeat'],
           actual_NFE_exported=True,actual_AE_layer_visits_exported=True,original_sampler_sources_changed=False,
           policy_action_limits=dict(LIMITS),settling_actions=10,
           scope='VLM12/AE18 D2 versus VLM18/AE14 D3; one FullD10 then two Fast requests. Actual policy costs must match; no same-layer-count/NFE claim.')
    v['revision_hash']=digest(v);return v

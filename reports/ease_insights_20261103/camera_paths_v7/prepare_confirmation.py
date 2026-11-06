"""Only an independently accepted complete discovery pair unlocks800 fresh rollouts."""
import copy
import json
from pathlib import Path
import sys
ROOT=Path('/root/nfs/embodied/benchmark');sys.path.insert(0,str(ROOT))
from preparations.ease_insights_20261103.camera_paths_v7 import common as c
from reports.ease_insights_20261103.camera_paths_v7.discovery_analysis import read_selection
REPORT=Path(__file__).resolve().parent
DISCOVERY=c.FAMILY_ROOT/'discovery';OUTPUT=c.FAMILY_ROOT/'confirmation'


def prepare():
    target=OUTPUT/'manifest.json'
    if target.exists():
        m=c.read_manifest(target,verify_sources=True)
        return dict(reused=True,manifest_hash=m['manifest_hash'],manifest_file_sha256=c.sha(target))
    base=c.read_manifest(DISCOVERY/'manifest.json',verify_sources=True)
    pipeline_path=REPORT/'discovery_pipeline_acceptance.json';pipeline=json.loads(pipeline_path.read_text())
    c.require(pipeline['status']=='pass' and pipeline['manifest_hash']==base['manifest_hash'] and pipeline['new_cases']==160
        and pipeline['preserved_control_cells']==160 and all(c.sha(p)==h for p,h in pipeline['artifacts'].items()),'Need complete accepted discovery')
    selection_path=DISCOVERY/'analysis/selection.json';selection=read_selection(selection_path);choice=selection['selection'].get('selected')
    c.require(selection['selection']['status']=='selected' and choice and choice['paths']==list(c.NEW_CONDITIONS)
        and selection['discovery_manifest_hash']==base['manifest_hash'],'No qualified pair;confirmation stays locked')
    audits={}
    for key,name in [('cases','discovery_acceptance.json'),('statistics','independent_statistics.json')]:
        path=REPORT/name;v=json.loads(path.read_text())
        c.require(v['status']=='pass' and v['manifest_hash']==base['manifest_hash'] and v['new_cases']==160 and v['preserved_control_cells']==160,'Incomplete independent discovery audit')
        if key=='statistics':c.require(v['differences']==[],'Independent discovery arithmetic differs')
        audits[key]=dict(path=str(path),sha256=c.sha(path))
    m=copy.deepcopy(base)
    for name in ('manifest_hash','discovery_inits','total_new_discovery_episodes','preserved_reference_episodes','analytical_cells','reference_conditions','reference_origins','discovery_reuse_disclosure','independent_validation'):
        m.pop(name,None)
    sources={str(p.relative_to(ROOT)):c.sha(p) for p in sorted(REPORT.glob('*.py'))};m['source_files'].update(sources)
    m.update(revision='ease-fixed-component-confirmation-v7',phase='component_paths_confirm_v7',output_root=str(OUTPUT),
        planned_conditions=list(c.CONDITIONS),analytical_conditions=list(c.CONDITIONS),confirmation_inits=[10,11,12,13,14],cases_per_condition=200,total_confirmation_episodes=800,analysis_sources=sources,
        selection_evidence=dict(path=str(selection_path),sha256=c.sha(selection_path),selection_hash=selection['selection_hash'],frozen_selection=selection['selection'],evidence=selection['evidence']),
        preregistered_hypotheses=copy.deepcopy(choice['directions']),
        discovery_reference=dict(root=str(DISCOVERY),manifest_hash=base['manifest_hash'],manifest_sha256=c.sha(DISCOVERY/'manifest.json'),accepted_new_cases=160,preserved_control_cells=160,audits=audits,
            pipeline=dict(path=str(pipeline_path),sha256=c.sha(pipeline_path))),
        heldout_preparation=dict(outcomes_inspected_before_freeze=False,independent_initial_states=[10,11,12,13,14],controls_reused_for_confirmation=False,
            scope='Within-familyholdoutrelative45-46;allofficialindiceshavehistoricalR12exposure.Notgloballynewstatesorunseentasks.'),
        initial_pairing='Allfourconditionsfreshlyexecuteeachphysicalcasewiththeoriginalcondition-independentnoise0;no old quality controls are reused.',
        fixed_input_cost=dict(conditions='All4fixeddefinitionsineachof8completedservices',warmup_requests=96,measured_requests=192,
            scope='Whole native policy including fresh vision,current-observation KV,residual capture/patch,AE,decode and transfer;RPC/environment excluded.'))
    m['manifest_hash']=c.digest(m);c.validate_output(OUTPUT,m);c.atomic_json(target,m);c.read_manifest(target,verify_sources=True)
    return dict(reused=False,manifest_hash=m['manifest_hash'],manifest_file_sha256=c.sha(target),new_episodes=800,source_files=len(m['source_files']),GPU_started=False)


if __name__=='__main__':print(json.dumps(prepare()))

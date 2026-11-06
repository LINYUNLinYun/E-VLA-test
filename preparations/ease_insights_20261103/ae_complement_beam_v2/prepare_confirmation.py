"""Only a fully audited, frozen qualified pair unlocks the unused47-49 cases."""
import copy
import json
from pathlib import Path
from .common import ROOT,sha,digest,atomic_json,read_manifest,validate_output
from .selection import read_selection
from .protocol import REPORT

PACKAGE=Path(__file__).resolve().parent
DISCOVERY=ROOT/'results/ease_insights_20261103/ae_complement_beam_v2/discovery'
OUTPUT=DISCOVERY.parent/'confirmation'


def prepare():
    target=OUTPUT/'manifest.json'
    if target.exists():
        m=read_manifest(target,verify_sources=True)
        return dict(reused=True,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target))
    base=read_manifest(DISCOVERY/'manifest.json',verify_sources=True)
    selection_path=DISCOVERY/'analysis/selection.json';selected=read_selection(selection_path)
    choice=selected['selection'].get('selected')
    if selected['selection']['status']!='selected' or not choice:
        raise ValueError('No qualified module pair; keep confirmation locked')
    if selected['discovery_manifest_hash']!=base['manifest_hash']:
        raise ValueError('Selection belongs to another module discovery')
    audits={}
    for key,name in [('cases','discovery_acceptance.json'),('statistics','independent_statistics.json')]:
        path=REPORT/name;value=json.loads(path.read_text())
        if value.get('status')!='pass' or value.get('manifest_hash')!=base['manifest_hash']:
            raise ValueError('Need complete accepted module '+key+' audit')
        if value.get('new_cases')!=160 or value.get('preserved_control_cells')!=160:
            raise ValueError('Audit does not separate160 new and160 preserved cases')
        if key=='statistics' and value.get('differences')!=[]:
            raise ValueError('Independent module statistics differ')
        audits[key]=dict(path=str(path),sha256=sha(path))
    from preparations.ease_insights_20261103.vlm_diversity_v2.common import read_manifest as read_old
    from preparations.ease_insights_20261103.vlm_diversity_v2.prepare import unused
    prior=read_old(ROOT/'results/ease_insights_20261103/vlm_diversity_v2/discovery/manifest.json',verify_sources=True)
    heldout=unused(prior,[47,48,49])
    m=copy.deepcopy(base)
    for key in ('manifest_hash','discovery_inits','total_new_discovery_episodes','preserved_reference_episodes',
                'analytical_cells','reference_conditions','preserved_reference','reference_origins','discovery_reuse_disclosure'):
        m.pop(key,None)
    planned=['full_d10',*choice['paths']]
    local={str(p.relative_to(ROOT)):sha(p) for p in sorted(PACKAGE.glob('*.py'))}
    sources=dict(base['source_files']);sources.update(local)
    m.update(revision='ease-AE-joint-mask-confirmation-v2',phase='AE_joint_mask_confirm_v2',output_root=str(OUTPUT),
        planned_conditions=planned,analytical_conditions=planned,confirmation_inits=[47,48,49],cases_per_condition=120,
        total_confirmation_episodes=360,local_runtime_sources=local,source_files=sources,analysis_sources=local,
        selection_evidence=dict(path=str(selection_path),sha256=sha(selection_path),selection_hash=selected['selection_hash'],
            frozen_selection=selected['selection'],evidence=selected['evidence']),
        preregistered_hypotheses=copy.deepcopy(choice['directions']),
        discovery_reference=dict(root=str(DISCOVERY),manifest_hash=base['manifest_hash'],manifest_sha256=sha(DISCOVERY/'manifest.json'),
            accepted_new_cases=160,preserved_control_cells=160,audits=audits),
        confirmation_initial_state_prior_use_audit=heldout,
        heldout_preparation=dict(outcomes_inspected_before_freeze=False,independent_initial_states=[47,48,49],
            controls_reused_for_confirmation=False),
        initial_pairing='All three conditions execute the same unused official initial states and explicit noise; no old quality outcome enters confirmation')
    m['manifest_hash']=digest(m);validate_output(OUTPUT,m);atomic_json(target,m);read_manifest(target,verify_sources=True)
    return dict(reused=False,manifest_hash=m['manifest_hash'],manifest_file_sha256=sha(target),new_episodes=360,GPU_started=False)


if __name__=='__main__':print(json.dumps(prepare()))

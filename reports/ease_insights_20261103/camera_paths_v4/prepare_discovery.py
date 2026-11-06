"""Freeze160 fresh-camera discovery cases only after the8-control and cost gates."""
import copy
import json
from pathlib import Path
import sys
ROOT=Path('/root/nfs/embodied/benchmark');sys.path.insert(0,str(ROOT))
from preparations.ease_insights_20261103.camera_paths_v4 import common as c
REPORT=Path(__file__).resolve().parent
OUTPUT=c.FAMILY_ROOT/'discovery'


def initialize_original_inputs(m):
    origins={alias:c.reference(alias,verify_sources=True) for alias in c.REFERENCES}
    evidence=[]
    for scene in c.cases(m):
        paired=None;sources=[]
        for alias,(root,old,cm,w,cid) in origins.items():
            path=w.episode_dir(root,cid,scene)/'episode_receipt.json';row=json.loads(path.read_text())
            c.require(row['valid'] and row['manifest_hash']==old['manifest_hash'] and row['condition']==cid and row['case']==scene,'Original control identity differs')
            fields={k:row['initial_gate'][k] for k in ('case','input_hash','seed','physical_assets')}
            if paired is not None:c.require(fields==paired,'Full/DenseD2 first physical input or noise differs')
            paired=fields;sources.append(dict(path=str(path),sha256=c.sha(path),source_condition=cid,source_manifest_hash=old['manifest_hash'],analysis_alias=alias))
        path=OUTPUT/'initial_observations'/scene['suite']/f"task{scene['task']:02d}"/f"init{scene['init']:02d}.json"
        value=dict(paired,manifest_hash=m['manifest_hash'],condition='derived_original_control_reference',source_identities=sources,original_records_modified=False)
        if path.exists():c.require(json.loads(path.read_text())==value,'Derived physical input cache changed')
        else:c.atomic_json(path,value)
        evidence.append(dict(case=scene,path=str(path),sha256=c.sha(path),original_sources=sources))
    c.atomic_json(REPORT/'original_controls_prelaunch.json',dict(status='paired_metadata_passed',manifest_hash=m['manifest_hash'],original_cases=160,paired_initials=80,quality_values_used=False,new_quality_cases=0,evidence=evidence))


def prepare():
    target=OUTPUT/'manifest.json'
    if target.exists():
        m=c.read_manifest(target,verify_sources=True);initialize_original_inputs(m)
        return dict(reused=True,manifest_hash=m['manifest_hash'],manifest_file_sha256=c.sha(target))
    cal=c.read_manifest(c.FAMILY_ROOT/'calibration/manifest.json',verify_sources=True)
    accepted=REPORT/'calibration_acceptance.json';proof=json.loads(accepted.read_text())
    c.require(proof['status']=='pass' and proof['manifest_hash']==cal['manifest_hash'] and proof['all_eight_controls_bitwise_identical']
        and proof['GPU_original_vision_parity_passed'] and proof['partial_vision_oracle_passed'] and proof['joint_cost_qualification']['pass'],'Await real GPU,known-control and matched-cost qualification')
    for row in cal['official_init_availability']['cases']:
        c.require(c.sha(row['bddl_path'])==row['bddl_sha256'] and c.sha(row['init_file'])==row['init_file_sha256'],'Official physical assets changed')
    m=copy.deepcopy(cal)
    for key in ('manifest_hash','calibration_inits','total_calibration_episodes','calibration_case_scope'):m.pop(key,None)
    report_sources={str(p.relative_to(ROOT)):c.sha(p) for p in sorted(REPORT.glob('*.py'))}
    m['source_files'].update(report_sources)
    m.update(revision='ease-fresh-camera-depth-discovery-v4',phase='fresh_vision_discovery_v4',output_root=str(OUTPUT),
        planned_conditions=list(c.NEW_CONDITIONS),analytical_conditions=list(c.CONDITIONS),reference_conditions={a:a for a in c.REFERENCES},
        discovery_inits=[45,46],tasks=list(range(10)),case_table=[dict(suite=s,task=t) for s in c.SUITES for t in range(10)],
        cases_per_condition=80,total_new_discovery_episodes=160,preserved_reference_episodes=160,analytical_cells=320,
        calibration_evidence=dict(path=str(accepted),sha256=c.sha(accepted)),analysis_sources=report_sources,
        discovery_reuse_disclosure='Original Full80 and DenseD2 80 retain source manifests/condition IDs;not new replications.',
        initial_pairing='Both candidate paths must match both original controls before prediction: official state,BDDL,prompt,first observation and explicit noise.',
        independent_validation=dict(status='locked_until_complete_qualified_selection',inits=[10,11,12,13,14],conditions=list(c.CONDITIONS),episodes=800),
        new_scientific_episode_cap_including_known_controls=968,
        fixed_input_cost=dict(conditions='All4fixeddefinitionsineachcompletedservice',warmup_requests=96,measured_requests=192,
            scope='Real fresh vision plus original VLM/AE/nodes/decoding/transfer;RPC/environment excluded,no exclusiveGPU or closedloop speed claim'))
    m['manifest_hash']=c.digest(m);c.validate_output(OUTPUT,m);c.atomic_json(target,m);c.read_manifest(target,verify_sources=True);initialize_original_inputs(m)
    return dict(reused=False,manifest_hash=m['manifest_hash'],manifest_file_sha256=c.sha(target),new_episodes=160,preserved_control_cells=160,source_files=len(m['source_files']),GPU_started=False)


if __name__=='__main__':print(json.dumps(prepare()))

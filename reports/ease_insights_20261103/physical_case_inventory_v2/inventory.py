"""Metadata-only inventory across canonical, FLASH-flat and native episode schemas."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
ROOT=Path('/root/nfs/embodied/benchmark');RESULTS=ROOT/'results/ease_insights_20261103'
REPORT=Path(__file__).resolve().parent
CURRENT=RESULTS/'ae_complement_beam_v4'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def extract(value,path):
    candidates=[]
    if isinstance(value,dict):
        for row in (value.get('case'),value):
            if not isinstance(row,dict):continue
            suite=row.get('suite',row.get('task_suite_name',row.get('target_suite')))
            task=row.get('task',row.get('task_id'));initial=row.get('init',row.get('init_id',row.get('episode_idx')))
            if isinstance(suite,str) and type(task) is int and type(initial) is int:candidates.append((suite,task,initial))
    text=str(path)
    canonical=re.search(r'/episodes/(libero_[a-z0-9_]+)/task(\d+)/init(\d+)(?:/|$)',text)
    flat=re.search(r'/episodes/(libero_[a-z0-9_]+)_t(\d+)_i(\d+)(?:/|$)',text)
    for match in (canonical,flat):
        if match:candidates.append((match[1],int(match[2]),int(match[3])))
    unique=set(candidates)
    if len(unique)>1:raise ValueError('Conflicting case identities: '+str(path))
    return next(iter(unique)) if unique else None


def run():
    base=read(RESULTS/'vlm_diversity_v2/discovery/manifest.json');catalog=base['initial_state_prior_use_audit']['catalog']
    if sha(catalog['path'])!=catalog['sha256']:raise ValueError('Physical alias catalog changed')
    with Path(catalog['path']).open() as f:table={(r['suite'],int(r['task'])):r['full_bddl_identity'] for r in csv.DictReader(f)}
    targets={table[s,t] for s in SUITES for t in range(10)}
    command=['rg','--files','--hidden','--no-ignore','-g','identity.json','-g','result.json','-g','episode_receipt.json',
        '-g','first_policy_observation.npz','-g','input_0000.npz','-g','initial_state.npz',str(RESULTS)]
    scan=subprocess.run(command,text=True,capture_output=True,check=True)
    if scan.stderr.strip():raise RuntimeError(scan.stderr)
    rows=[];excluded=0;unknown=[];by_init={i:set() for i in range(50)};documents={i:0 for i in range(50)}
    for name in sorted(set(scan.stdout.splitlines())):
        path=Path(name)
        if path.is_relative_to(CURRENT):excluded+=1;continue
        value=read(path) if path.suffix=='.json' else None;scene=extract(value,path)
        if scene is None:continue
        s,t,i=scene
        if not 0<=i<50:continue
        alias=table.get((s,t))
        if alias in targets:
            by_init[i].add(alias);documents[i]+=1
            rows.append(dict(path=name,case=dict(suite=s,task=t,init=i),physical_task_identity=alias,
                metadata_sha256=sha(path) if path.suffix=='.json' else None,content_read=path.suffix=='.json'))
        elif s in SUITES:unknown.append(dict(path=name,case=dict(suite=s,task=t,init=i)))
    current_clear=all(documents[i]==0 for i in (47,48,49)) and not unknown
    available=[i for i in range(16,30) if documents[i]==0]
    value=dict(status='metadata_inventory_passed' if current_clear else 'review_required',time=time.time(),
        roots=[str(RESULTS)],source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        catalog=catalog,excluded_current_confirmation_cohort=str(CURRENT),excluded_current_documents=excluded,
        existing_prior_case_documents_by_init=documents,physical_tasks_with_prior_documents_by_init={i:len(v) for i,v in by_init.items()},
        candidate_indices16_to29_with_zero_prior_documents=available,
        current47_49_no_prior_documents_in_all_three_supported_schemas=current_clear,
        unresolved=unknown,case_metadata=rows,success_values_accessed=False,
        raw_NPZ_contents_read=False,scientific_artifacts_modified=False,new_GPU_model_or_environment_started=False,
        scope='Known EASE result roots and physical aliases only; canonical scene,FLASH flat and native case formats plus first-input path exposures. Zero matches is not a claim about unknown external experiments. Current v4 outcomes are wholly excluded.')
    REPORT.mkdir(parents=True,exist_ok=True);(REPORT/'inventory.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:value[k] for k in ['status','candidate_indices16_to29_with_zero_prior_documents','current47_49_no_prior_documents_in_all_three_supported_schemas','existing_prior_case_documents_by_init']}))


if __name__=='__main__':run()

"""Read historical native episode identities only, never success values or actions."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path('/root/nfs/embodied/benchmark');sys.path.insert(0,str(ROOT))
from reports.ease_insights_20261103.physical_case_inventory_v2.inventory import extract
REPORT=Path(__file__).resolve().parent
HISTORY=ROOT/'policies/realtime-vla-flash/results'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    scan=subprocess.run(['rg','--files','-g','episode_log.json',str(HISTORY)],text=True,capture_output=True,check=True)
    if scan.stderr.strip():raise RuntimeError(scan.stderr)
    files=[];counts={i:0 for i in range(50)};tasks={i:set() for i in range(50)};unknown=[];empty=[];identities=0
    for name in sorted(set(scan.stdout.splitlines())):
        path=Path(name);data=path.read_bytes();value=json.loads(data)
        rows=value if isinstance(value,list) else value.get('episodes') if isinstance(value,dict) else None
        if not isinstance(rows,list):unknown.append(dict(path=name,error='Unrecognized native episode table'));continue
        if not rows:empty.append(name)
        by_init={};unknown_rows=0
        for index,row in enumerate(rows):
            scene=extract(row,path)
            if scene is None:
                unknown_rows+=1;continue
            s,t,i=scene
            if s not in SUITES or not 0<=t<10:continue
            identities+=1;by_init[i]=by_init.get(i,0)+1
            if 0<=i<50:counts[i]+=1;tasks[i].add((s,t))
        if unknown_rows:unknown.append(dict(path=name,unrecognized_rows=unknown_rows,rows=len(rows)))
        files.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest(),rows=len(rows),target_case_metadata_by_init=by_init))
    ease_path=ROOT/'reports/ease_insights_20261103/physical_case_inventory_v2/inventory.json';ease=json.loads(ease_path.read_text())
    candidates=[i for i in range(16,30) if not counts[i] and not ease['existing_prior_case_documents_by_init'][str(i)]]
    clear47_49=all(not counts[i] for i in (47,48,49)) and ease['current47_49_no_prior_documents_in_all_three_supported_schemas'] and not unknown
    result=dict(status='metadata_inventory_passed' if not unknown else 'incomplete_schema_review_required',time=time.time(),
        root=str(HISTORY),source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        extractor=dict(path=str(ROOT/'reports/ease_insights_20261103/physical_case_inventory_v2/inventory.py'),sha256=sha(ROOT/'reports/ease_insights_20261103/physical_case_inventory_v2/inventory.py')),
        native_episode_logs=len(files),target_case_identities=identities,unknown_schemas=unknown,empty_logs=len(empty),
        historical_case_rows_by_init=counts,physical_tasks_by_init={i:len(v) for i,v in tasks.items()},
        EASE_inventory=dict(path=str(ease_path),sha256=sha(ease_path)),
        candidate_indices16_to29_clear_in_both_known_inventories=candidates,
        current47_49_clear_in_both_known_inventories=clear47_49,
        source_files=files,success_values_accessed=False,raw_actions_or_observations_read=False,new_GPU_or_model_or_environment_started=False,
        scope='Known EASE metadata plus all historical native episode_log.json files in this Pi0 policy result tree. Currentv4 outcomes excluded by the EASE inventory. Unknown external systems,training exposure and other unrecognized formats are not claimed absent.')
    (REPORT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return {k:result[k] for k in ['status','native_episode_logs','target_case_identities','unknown_schemas','historical_case_rows_by_init','candidate_indices16_to29_clear_in_both_known_inventories','current47_49_clear_in_both_known_inventories']}


if __name__=='__main__':print(json.dumps(audit()))

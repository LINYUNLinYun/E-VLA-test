"""CPU pair checks for the forty-episode diagnostic, including physical identity gates."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from .draft_diagnostic_audit import read,dump,csvout,HEADS


def analyze(directory):
    directory=Path(directory);plan=read(directory/'protocol_plan.json');records={};missing=[]
    for row in plan['episodes']:
        path=directory/('episode%03d'%row['episode_index'])/'result.json'
        if not path.exists() or read(path).get('status')!='complete':missing.append(row['episode_index']);continue
        records[row['episode_index']]=read(path)
    pairs=[]
    def pair(a,b,factor,physical_stage):
        if a['episode_index'] not in records or b['episode_index'] not in records:return
        first,second=records[a['episode_index']],records[b['episode_index']]
        keys=('gpu_uuid','server_pid','manifest_hash')
        same_service=all(first['service_identity'][k]==second['service_identity'][k] for k in keys)
        physical_equal=first[physical_stage+'_physical_hashes']==second[physical_stage+'_physical_hashes']
        observation_equal=first['initial_observation_hashes']==second['initial_observation_hashes']
        # Init changes the full official initial state; it is intentionally not a robot-only claim.
        valid=same_service and (physical_equal if factor!='init' else True)
        result=dict(head=a['head'],task=a['case']['task_index'],factor=factor,first=a['episode_index'],second=b['episode_index'],
            first_label=f"init{a['init_index']}:{a['settle']}:{a['prompt']}",second_label=f"init{b['init_index']}:{b['settle']}:{b['prompt']}",
            first_success=int(first['success']),second_success=int(second['success']),first_steps=first['steps'],second_steps=second['steps'],
            same_service=same_service,physical_gate_stage=physical_stage,physical_equal=physical_equal,initial_observation_equal=observation_equal,
            valid_pair=valid,claim='official init-vector effect, includes object scene state' if factor=='init' else
                'settle bundle effect (duration and gripper action together)' if factor=='settle' else 'prompt-only effect' )
        if factor=='prompt':result['valid_pair']=valid and observation_equal
        pairs.append(result)
    groups=defaultdict(list)
    for row in plan['episodes']:groups[(row['head'],row['case']['task_index'])].append(row)
    for group in groups.values():
        lookup={(r['init_index'],r['settle'],r['prompt']):r for r in group}
        for settle in ('new','old'):pair(lookup[(0,settle,'legacy_filename')],lookup[(6,settle,'legacy_filename')],'init','reset')
        for init in (0,6):pair(lookup[(init,'old','legacy_filename')],lookup[(init,'new','legacy_filename')],'settle','reset')
        pair(lookup[(0,'new','legacy_filename')],lookup[(0,'new','effective_bddl')],'prompt','initial')
    cells=[]
    for head in HEADS.values():
        for init,settle,prompt in ((0,'new','legacy_filename'),(0,'old','legacy_filename'),(6,'new','legacy_filename'),(6,'old','legacy_filename'),(0,'new','effective_bddl')):
            group=[r for r in records.values() if (r['design']['head'],r['design']['init_index'],r['design']['settle'],r['design']['prompt'])==(head,init,settle,prompt)]
            cells.append(dict(head=head,init=init,settle=settle,prompt=prompt,n=len(group),success=sum(r['success'] for r in group)))
    if pairs:csvout(directory/'paired_comparisons.csv',pairs)
    csvout(directory/'factorial_cells.csv',cells)
    result=dict(status='complete' if not missing else 'partial',completed=len(records),target=40,missing=missing,
        physical_pair_failures=[p for p in pairs if not p['valid_pair']],
        limitations=['Two metadata-selected tasks per head; no new suite accuracy claim.',
                     'Init0 versus6 changes the whole official initial state, not robot pose alone.',
                     'Old/new settle changes duration and gripper command jointly.',
                     'Unified600 cap is a controlled diagnostic, not the historical score protocol.'])
    dump(directory/'analysis.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    print(json.dumps(analyze(p.parse_args().directory),indent=2))

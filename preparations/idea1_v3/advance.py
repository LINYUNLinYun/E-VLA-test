"""Result-based selection and immutable incremental manifests, default dry run."""
import argparse
import csv
import json
from pathlib import Path
import types

from preparations.idea1_v3.cli import make_manifest,config_id
from preparations.idea1_v3.branches import manifest as branch_manifest
from preparations.idea1_v3.io import atomic


def completed(root):
    rows=[]
    for marker in sorted((Path(root)/'episodes').glob('*/*/*/*/task*/init*/complete.json')):
        record=json.loads(marker.read_text())
        result=json.loads((marker.parent/'result.json').read_text())
        if not record.get('valid') or not result.get('valid') or result['key']!=record['key']:
            raise ValueError(f'Invalid committed episode {marker}')
        rows.append(dict(result,directory=str(marker.parent.resolve())))
    return rows


def rank_calibration(root):
    data=json.loads((Path(root)/'manifest.json').read_text())
    if data['stage']!='calibration': raise ValueError('E1 selection requires calibration source')
    results=completed(root); output=[]
    for condition in data['conditions']:
        rows=[x for x in results if x['condition']==condition['id']]
        if len(rows)!=8: raise ValueError(f"Calibration candidate {condition['id']} has {len(rows)}/8 valid episodes; selection not frozen")
        wall=actions=0
        for row in rows:
            for line in (Path(row['directory'])/'infer.jsonl').read_text().splitlines():
                inference=json.loads(line); wall+=inference['timing']['policy_wall_ms']
            actions+=row['actions']
        output.append(dict(condition=condition['id'],config=condition['config'],successes=sum(x['success'] for x in rows),
                           episodes=len(rows),policy_ms_per_action=wall/actions))
    selected=[]
    for d in (2,10):
        subset=[x for x in output if x['config']['draft_steps']==d]
        best=min(subset,key=lambda x:(-x['successes'],x['policy_ms_per_action'],x['config']['tau'],x['config']['period']))
        reference=dict(experiment='E1',draft_steps=d,tau=.15,period=2)
        selected.append(reference)
        if best['config']!=reference: selected.append(best['config'])
    return dict(calibration=str(Path(root).resolve()),candidates=output,selected=selected,
        rule='success count descending, policy ms per actual action ascending, tau ascending, P ascending',
        calibration_is_separate_from_init6_10=True)


def rank_e2(root):
    data=json.loads((Path(root)/'manifest.json').read_text()); rows=completed(root)
    scores=[]
    for condition in data['conditions']:
        subset=[x for x in rows if x['condition']==condition['id'] and x['init']==6]
        if len(subset)!=40: raise ValueError(f"E2 {condition['id']} {len(subset)}/40; wait for full balanced screen")
        scores.append(dict(mask=condition['config']['mask'],successes=sum(x['success'] for x in subset),episodes=40))
    from preparations.idea1_v3.pi05 import MASKS
    masks=[x for x in scores if x['mask']!='Full']
    best=min(masks,key=lambda x:(-x['successes'],len(MASKS[x['mask']]),x['mask']))
    eligible=[x for x in masks if best['successes']-x['successes']<=4]
    pruned=min(eligible,key=lambda x:(-len(MASKS[x['mask']]),-x['successes'],x['mask']))
    selected=list(dict.fromkeys(['Full',best['mask'],pruned['mask']]))
    return dict(screen=str(Path(root).resolve()),candidates=scores,selected_masks=selected,
        rule='best screen success, plus most skipped within ten percentage points of best; deduplicate; Full retained')


def build(args):
    source=Path(args.source).resolve(); selection=None
    if args.command=='e1-select': return rank_calibration(source)
    if args.command=='e2-select': return rank_e2(source)
    if args.command=='e1-prepare':
        selection=rank_calibration(source); configs=selection['selected']; all_tasks=[]; conditions=[]
        for config in configs:
            native=make_manifest(types.SimpleNamespace(stage='E1',config=json.dumps(config),execution='A',expand=False,shards=args.shards))
            all_tasks.extend(native['tasks']); conditions.extend(native['conditions'])
        native.update(tasks=all_tasks,conditions=conditions,selection=selection,source_attempts=[str(source)],stage='E1')
    elif args.command=='e1-expand':
        first=json.loads((source/'manifest.json').read_text())
        if first['stage']!='E1' or any(row['init']!=6 for row in first['tasks']): raise ValueError('E1 expansion must reference selected initial40-per-condition manifest')
        if len(completed(source))!=len(first['tasks']): raise ValueError('Wait for all source init6 episodes to complete before expansion')
        tasks=[]
        for condition in first['conditions']:
            native=make_manifest(types.SimpleNamespace(stage='E1',config=json.dumps(condition['config']),execution='A',expand=True,shards=args.shards))
            tasks.extend(row for row in native['tasks'] if row['init']>6)
        native.update(tasks=tasks,conditions=first['conditions'],source_attempts=[str(source)])
    elif args.command=='e4-expand':
        first=json.loads((source/'manifest.json').read_text())
        if first['stage']!='E4' or len(first['tasks'])!=40: raise ValueError('E4 expansion must reference original forty-key manifest')
        if len(completed(source))!=40: raise ValueError('Wait for all forty source E4 episodes before expansion')
        native=make_manifest(types.SimpleNamespace(stage='E4',config=None,execution='A',expand=True,shards=args.shards))
        expected={x['key'] for x in first['tasks']}
        initial={x['key'] for x in native['tasks'] if x['init']==6}
        if expected!=initial: raise ValueError('E4 source scientific key/checkpoint mismatch')
        native['tasks']=[x for x in native['tasks'] if x['init']>6]
        native['source_attempts']=[str(source)]
        native['reused_initial_keys']=sorted(initial)
    elif args.command=='e2-expand':
        selection=rank_e2(source)
        native=branch_manifest(types.SimpleNamespace(experiment='E2',parent=None,
            only_mask=','.join(selection['selected_masks']),expand=True,shards=args.shards))
        native.update(selection=selection,source_attempts=[str(source)])
    elif args.command=='e3-prepare':
        selection=rank_calibration(source)
        best=min((x for x in selection['candidates'] if x['config']['draft_steps']==2),
            key=lambda x:(-x['successes'],x['policy_ms_per_action'],x['config']['tau'],x['config']['period']))['config']
        native=branch_manifest(types.SimpleNamespace(experiment='E3',parent=json.dumps(best),only_mask=None,
            expand=False,shards=args.shards))
        native.update(parent_selection=selection,parent_condition=config_id(best),source_attempts=[str(source)])
        if args.parent_attempt:
            parent_root=Path(args.parent_attempt).resolve()
            scientific=json.loads((parent_root/'manifest.json').read_text())
            if best not in [condition['config'] for condition in scientific['conditions']]:
                raise ValueError('Explicit E1 parent attempt does not contain frozen selected D2 config')
            native['source_attempts'].append(str(parent_root))
    elif args.command=='e3-expand':
        first=json.loads((source/'manifest.json').read_text())
        if first['stage']!='E3' or len(first['tasks'])!=40: raise ValueError('E3 expansion requires initial40 source')
        if len(completed(source))!=40: raise ValueError('Wait for all forty source E3 episodes before expansion')
        parent=first['conditions'][0]['config']['parent']
        native=branch_manifest(types.SimpleNamespace(experiment='E3',parent=json.dumps(parent),only_mask=None,
            expand=True,shards=args.shards))
        native.update(source_attempts=[str(source)],parent_condition=first.get('parent_condition'))
    else: raise ValueError(args.command)
    # Owners span the final union, not independently repeated per condition.
    for index,row in enumerate(native['tasks']): row['owner']=index%args.shards
    native['incremental_only']=args.command.endswith('expand')
    return native


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('e1-select','e2-select','e1-prepare','e1-expand','e4-expand','e2-expand','e3-prepare','e3-expand'))
    parser.add_argument('--source',type=Path,required=True); parser.add_argument('--attempt',type=Path)
    parser.add_argument('--shards',type=int,default=1); parser.add_argument('--launch',action='store_true')
    parser.add_argument('--parent-attempt',type=Path)
    args=parser.parse_args(); value=build(args)
    if not args.launch:
        print(json.dumps(value,indent=2)); return
    if args.command.endswith('select'):
        atomic(args.source/'summaries'/f'{args.command}.json',value)
    else:
        if args.attempt is None: raise ValueError('new --attempt required')
        if (args.attempt/'manifest.json').exists(): raise FileExistsError('immutable manifest already exists')
        atomic(args.attempt/'manifest.json',value)
    print(json.dumps(dict(command=args.command,tasks=len(value.get('tasks',[])),selected=value.get('selected',value.get('selected_masks')))))

if __name__=='__main__': main()

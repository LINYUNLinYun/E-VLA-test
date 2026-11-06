"""Frozen A->B->C selection and missing-init-only immutable expansion."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from preparations.idea1_v3.analysis import collect
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_supplement.e1_cli import manifest as e1_manifest


def rank(rows,experiment):
    groups={}
    for row in rows:
        if row['experiment']!=experiment or row['init']!=6 or row['analysis_stage'] in ('P0','calibration'): continue
        group=groups.setdefault(row['condition'],[]); group.append(row)
    ranked=[]
    for condition,episodes in groups.items():
        keys={(e['suite'],e['task']) for e in episodes}
        if len(episodes)!=40 or len(keys)!=40: raise ValueError(f'{condition}: need exactly complete common init6 forty')
        total=actions=0
        for episode in episodes:
            actions+=episode['actions']
            for line in (Path(episode['directory'])/'infer.jsonl').read_text().splitlines(): total+=json.loads(line)['timing']['policy_wall_ms']
        ranked.append(dict(condition=condition,config=episodes[0]['config'],successes=sum(e['success'] for e in episodes),episodes=40,
            actual_actions=actions,policy_ms=total,policy_ms_per_action=total/actions,
            source_attempts=sorted({e['source_attempt'] for e in episodes}),filter='init6 only; P0/calibration excluded'))
    return ranked


def parameter_order(row):
    c=row['config']; return c.get('tau',0),float('inf') if c.get('period') is None else c['period'],row['condition']


def quality(row): return (-row['successes'],row['policy_ms_per_action'],*parameter_order(row))
def speed(row): return (row['policy_ms_per_action'],-row['successes'],*parameter_order(row))


def select_parents(ranked):
    eligible=[r for r in ranked if r['config'].get('gripper_rule','strict')=='strict' and r['config'].get('reject_steps',10)==10]
    first=min(eligible,key=quality)
    other=sorted([r for r in eligible if r['successes']>=32 and r['condition']!=first['condition']],key=speed)
    if not other: raise ValueError('No independent second >=80% parent; report no selection rather than extend grid')
    return [first,other[0]]


def expansion(source,conditions,shards):
    data=json.loads((source/'manifest.json').read_text()); rows,_=collect([source])
    selected=[c for c in data['conditions'] if c['id'] in conditions]
    if len(selected)!=len(set(conditions)): raise ValueError('Unknown/duplicate selected condition')
    tasks=[]
    for c in selected:
        completed=[r for r in rows if r['condition']==c['id'] and r['init']==6]
        if len(completed)!=40: raise ValueError('Selected source40 not actually complete')
        originals=[r for r in data['tasks'] if r['condition']==c['id'] and r['init']==6]
        if len(originals)!=40: raise ValueError('Expansion source must first40 manifest')
        # Same task's inits adjacent, preserving prompt/model residency.
        for row in originals:
            for init in range(7,11):
                identity={k:v for k,v in row.items() if k not in ('key','owner','execution')}; identity['init']=init
                tasks.append(dict(identity,key=digest(identity),owner=(len(tasks)//4)%shards,execution='A'))
    return dict(data,shards=shards,conditions=selected,tasks=tasks,source_attempts=[str(source)],incremental_only='init7..10; init6 referenced, never rerun')


def build(args):
    if args.command=='expand':
        if len(args.conditions)>2: raise ValueError('At most two expansion candidates per experiment')
        return expansion(args.source[0],args.conditions,args.shards)
    rows,attempts=collect(args.source)
    def parent_candidate(c):
        return c.get('experiment')=='E1' and c.get('gripper_rule','strict')=='strict' and c.get('reject_steps',10)==10 and c.get('draft_steps')==2 and (c.get('revision')=='supplement-e1-router-v1' or (c.get('tau') in (.10,.15) and c.get('period')==2))
    if args.command=='phase-b': rows=[r for r in rows if parent_candidate(r['config'])]
    ranked=rank(rows,'E1' if args.command in ('phase-b','phase-c','e1-select') else 'E2')
    if args.command in ('phase-b','e1-select','e2-select'):
        expected={c['id'] for _,m in attempts for c in m['conditions'] if (parent_candidate(c['config']) if args.command=='phase-b' else c['config'].get('revision')==('supplement-e1-router-v1' if args.command=='e1-select' else 'supplement-pi05-prefix-mask-native-suffix-v1'))}
        if not expected or not expected.issubset({r['condition'] for r in ranked}): raise ValueError('Missing required complete first40 condition; selection must wait')
    if args.command=='phase-b':
        # Only A new strict configs plus the frozen old .10/.15 P2 references.
        allowed=[r for r in ranked if r['config'].get('revision')=='supplement-e1-router-v1' or (r['config'].get('tau') in (.10,.15) and r['config'].get('period')==2 and r['config'].get('draft_steps')==2)]
        parents=select_parents(allowed)
        configs=[dict(p['config'],gripper_rule=g,reject_steps=10) for p in parents for g in ('confirmed_disagreement','off')]
        receipt=dict(rule='highest success then cost; independent >=80% fastest then success; tau/P tie-break none last',ranked=sorted(allowed,key=quality),parents=parents,selection_denominator='common init6 forty only, never old200')
    elif args.command=='phase-c':
        selection=json.loads(args.parent_selection.read_text()); parents=selection['parents']; configs=[]; choices=[]
        for parent in parents:
            pc=parent['config']
            candidates=[r for r in ranked if r['config'].get('tau')==pc['tau'] and r['config'].get('period')==pc['period'] and r['config'].get('gripper_rule') in ('confirmed_disagreement','off') and r['config'].get('reject_steps',10)==10]
            if len(candidates)!=2: raise ValueError('Need both relaxed40 for each frozen parent')
            best=min(candidates,key=quality); choices.append(best)
            configs.extend([dict(pc,gripper_rule='strict',reject_steps=2),dict(best['config'],reject_steps=2)])
        receipt=dict(parents=parents,best_relaxed=choices,ranked=ranked,rule='each parent strict and best relaxed by success/cost; rejection D2 only')
    elif args.command=='e1-select':
        new=[r for r in ranked if r['config'].get('revision')=='supplement-e1-router-v1']
        winners=[]
        for minimum in (36,32):
            eligible=[r for r in new if r['successes']>=minimum]
            if eligible:
                best=min(eligible,key=speed)
                if best['condition'] not in {r['condition'] for r in winners}: winners.append(best)
        return dict(ranked=new,selected=winners,rule='sample >=90% fastest and >=80% fastest by policy_ms/actual_action; deduplicate, no fill; shared-load screening, not causal speedup')
    else:
        winners=[]
        for count in (4,5):
            candidates=[r for r in ranked if len(r['config'].get('skip',[]))==count and r['successes']>=32]
            if candidates: winners.append(min(candidates,key=lambda r:(-r['successes'],r['policy_ms_per_action'],r['config']['mask'])))
        return dict(ranked=ranked,selected=winners,rule='per skip4/5: >=80% only, highest success then lowest policy_ms/action then stable mask name; none eligible means no expansion')
    data=e1_manifest(SimpleNamespace(phase_a=False,configs=json.dumps(configs),shards=args.shards))
    data['selection_receipt']=receipt
    data['source_attempts']=[str(root) for root in args.source if json.loads((root/'manifest.json').read_text()).get('revision','').startswith('supplement-')]
    data['reference_attempts']=[str(root) for root in args.source if root.as_posix() not in data['source_attempts']]
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('phase-b','phase-c','e1-select','e2-select','expand'))
    p.add_argument('--source',type=Path,action='append',required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--parent-selection',type=Path); p.add_argument('--conditions',nargs='+',default=[]); p.add_argument('--shards',type=int,default=2); p.add_argument('--launch',action='store_true')
    args=p.parse_args(); args.source=[r.resolve() for r in args.source]
    if args.command=='phase-c' and not args.parent_selection: p.error('--parent-selection B selection JSON required')
    if args.command=='expand' and (len(args.source)!=1 or not args.conditions): p.error('expand needs one first40 source and explicit selected conditions')
    data=build(args)
    if not args.launch: print(json.dumps(data,indent=2)); return
    if args.output.exists(): raise FileExistsError('Never overwrite frozen selection/output')
    if args.command in ('e1-select','e2-select'): atomic(args.output,data)
    else:
        atomic(args.output/'manifest.json',data)
        if 'selection_receipt' in data: atomic(args.output/'selection.json',data['selection_receipt'])


if __name__=='__main__': main()

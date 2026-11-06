"""Bounded explicit later screens and missing-init-only D2 expansion."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from preparations.idea1_v3.analysis import collect
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_pi05_d2_search.cli import manifest
from preparations.idea1_pi05_d2_search.policy import REVISION

ODD_PRIORITY=(6,15,4,7,12,5,16,8,10,13,18,17,14,11,9,3)


def choose(ranked):
    eligible=[r for r in ranked if r['successes']>=26 and r['config']['skip']]
    winners=[]
    if eligible:
        n=max(len(r['config']['skip']) for r in eligible)
        winners.append(min([r for r in eligible if len(r['config']['skip'])==n],key=lambda r:(-r['successes'],r['policy_ms_per_action'],r['config']['mask'])))
        winners.append(min(eligible,key=lambda r:(r['policy_ms_per_action'],-r['successes'],r['config']['mask'])))
        backup=[r for r in eligible if r['successes']>=30]
        if backup: winners.append(min(backup,key=lambda r:(r['policy_ms_per_action'],-r['successes'],r['config']['mask'])))
    unique={}
    for row in winners:
        if row['config']['skip']: unique.setdefault(row['condition'],row)
    return list(unique.values())


def odd_choice(ranked):
    eligible=[r for r in ranked if r['successes']>=26]
    if not eligible: return None
    n=max(len(r['config']['skip']) for r in eligible)
    if n not in range(2,15,2): return None
    parent=min([r for r in eligible if len(r['config']['skip'])==n],key=lambda r:(-r['successes'],r['policy_ms_per_action'],r['config']['mask']))
    extra=next(i for i in ODD_PRIORITY if i not in parent['config']['skip'])
    return dict(parent=parent,extra_layer=extra,name=f'n{n+1}_adaptive',skip=sorted([*parent['config']['skip'],extra]))


def build(args):
    rows,attempts=collect(args.source)
    conditions={}
    for _,m in attempts:
        if m.get('revision')!=REVISION: raise ValueError('Only explicit new D2 search sources allowed')
        for c in m['conditions']:
            signature=tuple(c['config']['skip'])
            if signature in conditions and conditions[signature]['config']!=c['config']: raise ValueError('Conflicting names/config for identical mask')
            conditions[signature]=c
    if args.command=='screen':
        data=manifest(SimpleNamespace(masks=args.masks,shards=args.shards))
        if len(conditions)+len(data['conditions'])>20: raise ValueError('Global screen cap20 conditions')
        if any(tuple(c['config']['skip']) in conditions for c in data['conditions']): raise ValueError('Existing semantic mask cannot be rescreened')
        if set(c['id'] for c in data['conditions'])&{c['id'] for c in conditions.values()}: raise ValueError('Condition name collision')
        data.update(source_attempts=[str(s) for s in args.source],search_conditions_including_sources=len(conditions)+len(data['conditions']))
        return data
    if args.command in ('odd','select'):
        from preparations.idea1_v3_supplement.advance import rank
        ranked=rank(rows,'E2')
        if {r['condition'] for r in ranked}!={c['id'] for c in conditions.values()}: raise ValueError('Every source condition must have actual complete first40')
        if args.command=='odd':
            if len(conditions)!=19: raise ValueError('Adaptive odd only after the fixed19 complete')
            decision=odd_choice(ranked)
            if decision is None: return dict(no_new_screen=True,rule='highest passing n not even2..14, or none passing',ranked=ranked)
            if tuple(decision['skip']) in conditions: raise ValueError('Odd mask already tested; no renamed rescreen')
            data=manifest(SimpleNamespace(masks=json.dumps({decision['name']:decision['skip']}),shards=args.shards))
            data.update(source_attempts=[str(s) for s in args.source],odd_decision=decision,search_conditions_including_sources=20)
            return data
        if len(conditions) not in (19,20): raise ValueError('Selection waits for fixed19 and conditional odd completion')
        if len(conditions)==19 and odd_choice(ranked) is not None: raise ValueError('Required odd boundary screen not yet completed')
        selected=choose(ranked); full=next(c['id'] for c in conditions.values() if not c['config']['skip'])
        return dict(ranked=ranked,selected=selected,expansion_conditions=[full,*[r['condition'] for r in selected]],
            rule='overall>=26/40 only; highest passing n quality/cost/name; overall fastest quality/name; >=30/40 fastest backup; dedup<=3 no fill',
            source_attempts=[str(s) for s in args.source],per_suite_gate=False)
    if not args.conditions or len(args.conditions)>4: raise ValueError('FullD2 plus at most3 explicitly selected candidates required')
    chosen=[c for c in conditions.values() if c['id'] in args.conditions]
    if len(chosen)!=len(set(args.conditions)): raise ValueError('Unknown/duplicate selected condition')
    if not any(not c['config']['skip'] for c in chosen): raise ValueError('Expansion must include new FullD2 quality reference')
    tasks=[]
    for c in chosen:
        original=[r for r in rows if r['condition']==c['id'] and r['init']==6]
        if len(original)!=40 or len({(r['suite'],r['task']) for r in original})!=40: raise ValueError('Actual complete first40 required')
        if c['config']['skip'] and sum(r['success'] for r in original)<26: raise ValueError('Candidate overall<65%; no per-suite gate')
        if any(r['condition']==c['id'] and r['init'] in range(7,11) for r in rows): raise ValueError('Already expanded; reference existing valid data rather than rerun')
        source_rows=[]
        for _,m in attempts: source_rows.extend(r for r in m['tasks'] if r['condition']==c['id'] and r['init']==6)
        unique={r['key']:r for r in source_rows}
        if len(unique)!=40: raise ValueError('Need original first40 scientific rows')
        for row in unique.values():
            for init in range(7,11):
                identity={k:v for k,v in row.items() if k not in ('key','owner','execution')}; identity['init']=init
                tasks.append(dict(identity,key=digest(identity),owner=(len(tasks)//4)%args.shards,execution='A'))
    return dict(schema=1,stage='E2',search='pi05_skip_n_D2',revision=REVISION,shards=args.shards,conditions=chosen,tasks=tasks,
        source_attempts=[str(s) for s in args.source],incremental_only='init7..10, original init6 referenced; maxFullD2+3 candidates',quality_threshold=dict(overall=.65,per_suite_gate=False))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('screen','odd','select','expand')); p.add_argument('--source',type=Path,action='append',required=True); p.add_argument('--masks'); p.add_argument('--conditions',nargs='+'); p.add_argument('--output',type=Path,required=True); p.add_argument('--shards',type=int,default=4); p.add_argument('--launch',action='store_true')
    args=p.parse_args(); args.source=[p.resolve() for p in args.source]
    if args.shards<1: p.error('positive shards required')
    if args.command=='screen' and not args.masks: p.error('explicit frozen --masks JSON required')
    data=build(args)
    if not args.launch:
        print(json.dumps(dict(dry_run=True,**data) if 'tasks' not in data else dict(dry_run=True,conditions=data['conditions'],tasks=len(data['tasks']),shards=data['shards']),indent=2)); return
    if args.output.exists(): raise FileExistsError('Immutable new attempt required')
    if args.command=='select': atomic(args.output,data)
    elif data.get('no_new_screen'): atomic(args.output/'odd_decision.json',data)
    else:
        atomic(args.output/'manifest.json',data)
        if 'odd_decision' in data: atomic(args.output/'odd_decision.json',data['odd_decision'])


if __name__=='__main__': main()

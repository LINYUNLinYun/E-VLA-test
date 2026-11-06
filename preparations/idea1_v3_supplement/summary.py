"""Small-result/infer supplement summaries; explicit references stay separate."""
import argparse
import collections
import json
import math
from pathlib import Path
import numpy as np
from preparations.idea1_v3.analysis import collect,write_csv


def summarize(rows,output):
    output.mkdir(parents=True,exist_ok=True)
    by=collections.defaultdict(list)
    for e in rows:
        if e['config'].get('revision','').startswith('supplement-'): by[(e['experiment'],e['condition'],e['suite'])].append(e)
    successes=[]; segments=[]; costs=[]; routes=[]; configs=[]; trajectories=[]
    for (stage,condition,suite),episodes in sorted(by.items()):
        successes.append(dict(stage=stage,condition=condition,execution='A',suite=suite,successes=sum(e['success'] for e in episodes),episodes=len(episodes),inits=json.dumps(sorted({e['init'] for e in episodes}))))
        for label,subset in [('selection_init6',[e for e in episodes if e['init']==6]),('confirmation_init7_10',[e for e in episodes if e['init'] in range(7,11)])]:
            if subset: segments.append(dict(stage=stage,condition=condition,suite=suite,segment=label,successes=sum(e['success'] for e in subset),episodes=len(subset)))
        requests=[]; routegroups=collections.defaultdict(list); total_actions=sum(e['actions'] for e in episodes)
        for e in episodes:
            infer=[json.loads(s) for s in (Path(e['directory'])/'infer.jsonl').read_text().splitlines()]
            if sum(i['executed_length'] for i in infer)!=e['actions']: raise ValueError('Executed length journal does not match result actions')
            for i in infer:
                requests.append(i['timing']['policy_wall_ms']); r=i['route']
                routegroups[(r['path'],r.get('full_reason',''),r.get('full_steps','N/A'))].append(i)
            if e['init']==6 and e['task'] in (0,5): trajectories.append(dict(stage=stage,condition=condition,suite=suite,task=e['task'],init=6,success=e['success'],directory=e['directory'],telemetry=str(Path(e['directory'])/'telemetry.jsonl')))
        costs.append(dict(stage=stage,condition=condition,execution='A',suite=suite,episodes=len(episodes),actual_actions=total_actions,requests=len(requests),policy_ms=sum(requests),policy_ms_per_actual_action=sum(requests)/total_actions,
            request_p50_ms=float(np.percentile(requests,50)),request_p95_ms=float(np.percentile(requests,95)),ae_compilation_structure='native scanned suffix' if stage=='E2' else 'Triton native',boundary='complete request policy wall incl rejected draft/verify/immediate Full; model startup separate'))
        for key,items in sorted(routegroups.items(),key=str):
            routes.append(dict(stage=stage,condition=condition,suite=suite,route=key[0],full_reason=key[1],full_steps=key[2],requests=len(items),policy_ms=sum(i['timing']['policy_wall_ms'] for i in items),actual_actions=sum(i['executed_length'] for i in items),
                accepted_length_counts=json.dumps(dict(collections.Counter(i['route'].get('accepted_length','N/A') for i in items))),actual_R_counts=json.dumps(dict(collections.Counter(i['executed_length'] for i in items)))))
    for stage,condition in sorted({(e['experiment'],e['condition']) for e in rows if e['config'].get('revision','').startswith('supplement-')}):
        subset=[e for e in rows if e['condition']==condition]
        configs.append(dict(stage=stage,condition=condition,config=json.dumps(subset[0]['config'],sort_keys=True),source_attempts=json.dumps(sorted({e['source_attempt'] for e in subset}))))
    for name,values in [('success_by_condition_suite',successes),('success_by_segment',segments),('latency_and_throughput',costs),('routing',routes),('scientific_configs',configs),('trajectory_index',trajectories)]: write_csv(output/(name+'.csv'),values)
    overall=[]
    for stage,condition in sorted({(r['stage'],r['condition']) for r in successes}):
        sub=[r for r in successes if (r['stage'],r['condition'])==(stage,condition)]; s=sum(r['successes'] for r in sub); n=sum(r['episodes'] for r in sub)
        c=[r for r in costs if (r['stage'],r['condition'])==(stage,condition)]; ms=sum(r['policy_ms'] for r in c); actions=sum(r['actual_actions'] for r in c)
        overall.append(dict(stage=stage,condition=condition,successes=s,episodes=n,success_rate=s/n,policy_ms=ms,actual_actions=actions,policy_ms_per_actual_action=ms/actions,quality_tier_90=s/n>=.9,quality_tier_85=s/n>=.85,quality_tier_80=s/n>=.8))
        z=1.959963984540054; p=s/n; scale=1+z*z/n
        center=(p+z*z/(2*n))/scale; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/scale
        overall[-1].update(wilson95_low=center-half,wilson95_high=center+half,quality_tiers='observed sample fractions, not guaranteed population lower bounds')
    write_csv(output/'overall_quality_cost.csv',overall)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--attempt',type=Path,action='append',required=True); p.add_argument('--reference-attempt',type=Path,action='append',default=[]); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); rows,attempts=collect(args.attempt); summarize(rows,args.output)
    startup=[]
    for root,manifest in attempts:
        if not manifest.get('revision','').startswith('supplement-'): continue
        for path in sorted((root/'workers').glob('worker*/events.jsonl')):
            for line in path.read_text().splitlines():
                event=json.loads(line)
                if event.get('kind')=='startup': startup.append(dict(source_attempt=str(root),worker=path.parent.name,seconds=event['seconds'],config=json.dumps(event['config'],sort_keys=True),boundary='model construction/load/initial startup readiness; first request compilation may remain in policy_ms'))
    write_csv(args.output/'model_startup_events.csv',startup)
    references=[]
    if args.reference_attempt:
        old,_=collect(args.reference_attempt)
        for e in old:
            if e['analysis_stage'] in ('P0','calibration') or e['init']!=6: continue
            c=e['config']
            if e['experiment']=='E1' and not(c.get('draft_steps')==2 and c.get('tau') in (.1,.15) and c.get('period')==2): continue
            references.append(dict(stage=e['experiment'],condition=e['condition'],suite=e['suite'],task=e['task'],init=6,success=e['success'],source_attempt=e['source_attempt'],key=e['key'],boundary='existing quality reference; separate revision, no same-revision latency ratio'))
    write_csv(args.output/'old_reference_keys_init6.csv',references)
    (args.output/'summary_receipt.json').write_text(json.dumps(dict(attempts=[str(a) for a,_ in attempts],source_visit_deduplicated=True,old_references_separate=True,quality='sum successes / sum valid episodes; all scientific failures retained',cost='sum complete policy wall / sum actual actions; cold startup separate; shared load not causal speedup'),indent=2))


if __name__=='__main__': main()

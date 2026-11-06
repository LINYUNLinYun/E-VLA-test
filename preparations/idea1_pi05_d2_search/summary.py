"""New D2 success/cost CSVs; old FullD10 quality remains a separate reference."""
import argparse
import collections
import json
import math
from pathlib import Path
import numpy as np
from preparations.idea1_v3.analysis import collect,write_csv
from preparations.idea1_pi05_d2_search.policy import REVISION


def interval(s,n):
    z=1.959963984540054; p=s/n; scale=1+z*z/n
    center=(p+z*z/(2*n))/scale; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/scale
    return center-half,center+half


def summarize(rows,output):
    output.mkdir(parents=True,exist_ok=True); groups=collections.defaultdict(list)
    for row in rows:
        if row['config'].get('revision')!=REVISION: raise ValueError('Only explicit new search rows in main table')
        p=row['protocol']
        if p['D']!=2 or p['sample_kwargs']['num_steps']!=2 or p['denoise_nodes']!=[1.,.5,0.] or p['H']!=10 or p['R']!=5: raise ValueError('Actual runtime protocol is not native D2/H10/R5')
        groups[(row['condition'],row['suite'])].append(row)
    suites=[]; segments=[]; costs=[]; configs=[]; trajectories=[]; requests=[]; outcomes=[]
    for (condition,suite),episodes in sorted(groups.items()):
        config=episodes[0]['config']; s=sum(e['success'] for e in episodes); n=len(episodes)
        suites.append(dict(stage='E2',condition=condition,execution='A',suite=suite,successes=s,episodes=n,success_rate=s/n,inits=json.dumps(sorted({e['init'] for e in episodes})),per_suite_gate=False,skip_n=len(config['skip']),D=2))
        for label,subset in [('selection_init6',[e for e in episodes if e['init']==6]),('confirmation_init7_10',[e for e in episodes if e['init'] in range(7,11)])]:
            if subset: segments.append(dict(condition=condition,suite=suite,segment=label,successes=sum(e['success'] for e in subset),episodes=len(subset)))
        times=[]; lengths=[]
        for e in episodes:
            outcomes.append(dict(condition=condition,suite=suite,task=e['task'],init=e['init'],key=e['key'],success=e['success'],actual_actions=e['actions'],termination=e.get('termination'),source_attempt=e['source_attempt'],directory=e['directory']))
            infer=[json.loads(line) for line in (Path(e['directory'])/'infer.jsonl').read_text().splitlines()]
            if sum(i['executed_length'] for i in infer)!=e['actions']: raise ValueError('Action denominator differs from journal')
            for i in infer:
                if i['route']['denoise_steps']!=2: raise ValueError('Inference native D2 route missing/mismatched')
                ms=i['timing']['policy_wall_ms']; times.append(ms); lengths.append(i['executed_length'])
                requests.append(dict(condition=condition,suite=suite,task=e['task'],init=e['init'],key=e['key'],round=i['round'],policy_ms=ms,actual_actions=i['executed_length'],denoise_steps=2,mask=config['mask'],native_policy_calls=1,full_fallback_calls='N/A'))
            if e['init']==6 and e['task'] in (0,5): trajectories.append(dict(stage='E2',condition=condition,suite=suite,task=e['task'],init=6,success=e['success'],directory=e['directory'],telemetry=str(Path(e['directory'])/'telemetry.jsonl'),baseline_condition='pi05D2_FullD2',pairing='explicit new FullD2; same task/init, validate init_vector_hash before plotting'))
        costs.append(dict(stage='E2',condition=condition,execution='A',suite=suite,episodes=n,actual_actions=sum(lengths),requests=len(times),policy_ms=sum(times),policy_ms_per_actual_action=sum(times)/sum(lengths),request_p50_ms=float(np.percentile(times,50)),request_p95_ms=float(np.percentile(times,95)),ae_compilation_structure='original native scanned suffix, complete18',native_policy_calls=len(times),full_fallback_calls='N/A',actual_R_counts=json.dumps(dict(collections.Counter(lengths)))))
    overall=[]
    for condition in sorted({r['condition'] for r in suites}):
        sub=[r for r in suites if r['condition']==condition]; c=[r for r in costs if r['condition']==condition]
        s=sum(r['successes'] for r in sub); n=sum(r['episodes'] for r in sub); ms=sum(r['policy_ms'] for r in c); actions=sum(r['actual_actions'] for r in c); low,high=interval(s,n)
        overall.append(dict(condition=condition,skip_n=sub[0]['skip_n'],successes=s,episodes=n,success_rate=s/n,overall_ge65_sample=s/n>=.65,per_suite_gate=False,wilson95_low=low,wilson95_high=high,policy_ms=ms,actual_actions=actions,policy_ms_per_actual_action=ms/actions,quality_rule='observed overall >=65%, not a population CI guarantee; each suite reported independently'))
        e=next(r for r in rows if r['condition']==condition)
        configs.append(dict(condition=condition,config=json.dumps(e['config'],sort_keys=True),actual_protocol=json.dumps(e['protocol'],sort_keys=True),source_attempts=json.dumps(sorted({r['source_attempt'] for r in rows if r['condition']==condition}))))
    tasks=[]
    for condition,suite,task in sorted({(e['condition'],e['suite'],e['task']) for e in outcomes}):
        sub=[e for e in outcomes if (e['condition'],e['suite'],e['task'])==(condition,suite,task)]
        tasks.append(dict(condition=condition,suite=suite,task=task,successes=sum(e['success'] for e in sub),episodes=len(sub)))
    for name,values in [('success_by_condition_suite',suites),('success_by_segment',segments),('latency_and_throughput',costs),('overall_quality_cost',overall),('scientific_configs',configs),('trajectory_index',trajectories),('native_request_costs',requests),('episode_outcomes',outcomes),('success_by_task',tasks)]: write_csv(output/(name+'.csv'),values)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--attempt',type=Path,action='append',required=True); p.add_argument('--old-full-d10',type=Path,action='append',default=[]); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); rows,attempts=collect(args.attempt); summarize(rows,args.output)
    refs=[]
    if args.old_full_d10:
        old,_=collect(args.old_full_d10)
        for e in old:
            c=e['config']
            if e['experiment']!='E2' or c.get('mask')!='Full' or c.get('denoise_steps',10)!=10: continue
            refs.append(dict(condition=e['condition'],suite=e['suite'],task=e['task'],init=e['init'],success=e['success'],key=e['key'],source_attempt=e['source_attempt'],boundary='existing FullD10 quality only; separate from new D2 main denominator and same-card new timing'))
    write_csv(args.output/'old_full_d10_quality_keys.csv',refs)
    reference_summary=[]
    for segment in ('common_init6','existing_all'):
        subset=[r for r in refs if segment=='existing_all' or r['init']==6]
        for suite in ['overall',*sorted({r['suite'] for r in subset})]:
            selected=[r for r in subset if suite=='overall' or r['suite']==suite]
            if selected: reference_summary.append(dict(reference='existing FullD10',segment=segment,suite=suite,successes=sum(r['success'] for r in selected),episodes=len(selected),boundary='separate old-quality reference, not new D2 main denominator or a timing ratio'))
    write_csv(args.output/'old_full_d10_quality_summary.csv',reference_summary)
    startup=[]; invalid=[]
    for root,_ in attempts:
        for path in sorted((root/'workers').glob('worker*/events.jsonl')):
            for line in path.read_text().splitlines():
                e=json.loads(line)
                if e.get('kind')=='startup': startup.append(dict(source_attempt=str(root),worker=path.parent.name,seconds=e['seconds'],config=json.dumps(e['config'],sort_keys=True)))
                elif e.get('kind')=='invalid': invalid.append(dict(source_attempt=str(root),worker=path.parent.name,key=e['key'],error=e.get('error'),classification='technical invalid, not scientific failure'))
    write_csv(args.output/'model_startup_events.csv',startup); write_csv(args.output/'technical_invalid_events.csv',invalid)
    (args.output/'summary_receipt.json').write_text(json.dumps(dict(source_attempts=[str(r) for r,_ in attempts],visited_deduplicated=True,quality_gate='overall only >=65%; suite never gated',selection_vs_confirmation='init6 / init7..10 separate; combined200 is not independent selection confirmation',timing='closed-loop complete native policy wall / actual actions; startup separately; shared-load screen cost not causal speed ratio',oldFullD10='quality reference separate; no new quality replay'),indent=2))


if __name__=='__main__': main()

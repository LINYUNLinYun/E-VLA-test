"""Incremental results, paired task-cluster intervals and routing summaries."""
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from .manifest import SUITES
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'results/idea1_v3_3'


def records():
    groups=defaultdict(list)
    for p in OUT.glob('*/episodes/*/task*/init*/result.json'):
        if '.invalid.' in str(p) or not (p.parent/'complete.json').exists(): continue
        r=json.loads(p.read_text()); r['_directory']=str(p.parent)
        groups[r['condition']].append(r)
    return groups


def summary(rows):
    n=len(rows); actions=sum(r['actions'] for r in rows); policy=sum(r['policy_wall_s'] for r in rows); wall=sum(r['wall_s'] for r in rows)
    tasks=defaultdict(list)
    for r in rows: tasks[(r['case']['suite'],r['case']['task'])].append(float(r['success']))
    macro={s:float(np.mean([np.mean(v) for (suite,t),v in tasks.items() if suite==s])) for s in SUITES if any(suite==s for suite,t in tasks)}
    return dict(n=n,successes=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/n if n else None,
        suite_task_macro_success=float(np.mean(list(macro.values()))) if macro else None,suite_task_macro=macro,
        actions=actions,policy_wall_s=policy,ms_per_action=1000*policy/actions if actions else None,
        episode_wall_s_sum=wall,episodes_per_worker_hour=3600*n/wall if wall else None,
        requests=sum(r['predictions'] for r in rows),mean_steps=actions/n if n else None,
        suites={s:dict(n=sum(r['case']['suite']==s for r in rows),successes=sum(r['success'] for r in rows if r['case']['suite']==s)) for s in SUITES})


def pairing(rows,baseline):
    def key(r): return tuple(r['case'][x] for x in ('suite','task','init'))
    base={key(r):r for r in baseline}; paired=[(r,base[key(r)]) for r in rows if key(r) in base]
    if not paired: return dict(n=0)
    task=defaultdict(list)
    for r,b in paired: task[(r['case']['suite'],r['case']['task'])].append(float(r['success'])-float(b['success']))
    rng=np.random.default_rng(3301); draws=[]
    for _ in range(2000):
        values=[]
        for suite in SUITES:
            a=[np.mean(v) for (s,t),v in task.items() if s==suite]
            if a: values.append(float(np.mean(rng.choice(a,len(a),replace=True))))
        draws.append(np.mean(values))
    macro_delta=float(np.mean([np.mean([np.mean(v) for (s,t),v in task.items() if s==suite]) for suite in SUITES if any(s==suite for s,t in task)]))
    return dict(n=len(paired),recoveries=sum(r['success'] and not b['success'] for r,b in paired),new_failures=sum(not r['success'] and b['success'] for r,b in paired),
        raw_episode_delta=float(np.mean([float(r['success'])-float(b['success']) for r,b in paired])),delta=macro_delta,ci95=np.percentile(draws,[2.5,97.5]).tolist())


def run():
    groups=records(); m=json.loads((OUT/'manifest.json').read_text()); result={}
    configs={c['id']:c for c in m['conditions']}
    for name in ('bounded_followups','confirm300'):
        path=OUT/'stages'/name/'manifest.json'
        if path.exists():configs.update({c['id']:c for c in json.loads(path.read_text())['conditions']})
    for cfg in configs.values():
        cid=cfg['id']; rows=groups[cid]; r=summary(rows)
        baseline='xiaomi_pure_d2' if cfg['model']=='xiaomi' else 'pi0_parent'
        r.update(config=cfg,paired_vs=baseline,paired=pairing(rows,groups[baseline]))
        result[cid]=r
    atomic(OUT/'progress.json',dict(time=time.time(),total_complete=sum(len(v) for v in groups.values()),conditions=result))
    routing=[]
    for cid,rows in groups.items():
        paths=Counter(); reasons=Counter(); blocked=Counter(); lengths=Counter(); count=0; policy=0
        for row in rows:
            for line in (Path(row['_directory'])/'infer.jsonl').read_text().splitlines():
                r=json.loads(line); count+=1; paths[r['route']['path']]+=1; reasons[r['route'].get('full_reason')]+=1
                blocked.update(r['route'].get('gate_blocked',[])); lengths[r['executed_length']]+=1; policy+=r['timing']['policy_wall_ms']
        routing.append(dict(condition=cid,requests=count,paths=dict(paths),full_reasons={str(k):v for k,v in reasons.items()},gate_blocks=dict(blocked),execute_lengths=dict(lengths),policy_ms=policy))
    atomic(OUT/'routing_summary.json',routing)
    return result


if __name__=='__main__':
    result=run()
    print('\n'.join(f"{k}: {v['successes']}/{v['n']} {v['ms_per_action']} ms/action" for k,v in result.items() if v['n']))

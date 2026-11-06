"""Independent800-case task arithmetic and native-cost reconstruction.

Does not import production analysis/statistics. Raw action validation is supplied
by the separately pinned complete case audit;this pass rereads accepted outcomes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import numpy as np
ROOT=Path('/root/nfs/embodied/benchmark');REPORT=Path(__file__).resolve().parent
STUDY=ROOT/'results/ease_insights_20261103/camera_paths_v7/confirmation'
IDS=('full_d10','dense_d2','attn6_d2','ffn6_d2');PATHS=IDS[2:]
INITIALS=(10,11,12,13,14);SEED=20261121


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def canonical(value,field):
    payload=dict(value);declared=payload.pop(field)
    require(hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()==declared,'Canonical identity changed')
    return declared


def adjusted(values):
    require(len(values)==2,'Need exactly two fixed tests');order=sorted(range(2),key=lambda i:(values[i],i));result=[0.,0.];last=0.
    for rank,i in enumerate(order):last=max(last,min(1.,(2-rank)*values[i]));result[i]=last
    return result


def direction(values):
    require(len(values)==50 and all(type(v) is int and v in (-1,0,1) for v in values),'Need50 paired differences clustered into ten five-state tasks')
    counts=[sum(values[5*t:5*t+5]) for t in range(10)];total=sum(counts);extreme=0
    for bits in range(1024):extreme+=sum(v if bits&(1<<t) else -v for t,v in enumerate(counts))>=total
    cases=np.random.default_rng(SEED).choice(np.asarray(values,np.int64),size=(20000,50),replace=True).sum(axis=1)/50
    tasks=np.random.default_rng(SEED).choice(np.asarray(counts,np.int64),size=(20000,10),replace=True).sum(axis=1)/50
    def interval(draws):return dict(low=float(np.quantile(draws,.025,method='linear')),high=float(np.quantile(draws,.975,method='linear')))
    a=sum(v==1 for v in values);b=sum(v==-1 for v in values);n=a+b
    return dict(difference_count=total,task_difference_counts=counts,p_one_sided=extreme/1024,
        case_interval=interval(cases),task_interval=interval(tasks),sensitivity_case_p=sum(math.comb(n,k) for k in range(a,n+1))/2**n if n else 1.,
        loo_positive=all(total-v>0 for v in counts))


def verdict(directions,counts,ratios):
    p=adjusted([d['p_one_sided'] for d in directions]);matched=max(ratios[c] for c in PATHS)<=1.10*min(ratios[c] for c in PATHS)
    qualified={c:counts[c]>=150 and counts['full_d10']-counts[c]<=30 and ratios[c]>=1.15 and matched for c in PATHS}
    for d,value in zip(directions,p):d.update(primary_holm2_p=value,direction_reproduced=d['difference_count']>0 and value<.05 and d['task_interval']['low']>0 and d['loo_positive'])
    conclusion='complementarity_reproduced' if all(d['direction_reproduced'] for d in directions) and all(qualified.values()) else 'not_reproduced'
    return conclusion,qualified,matched


def audit(analysis_path,case_audit_path,output):
    launch=read(REPORT/'prelaunch_confirmation.json');m=read(STUDY/'manifest.json');mh=canonical(m,'manifest_hash')
    require(sha(STUDY/'manifest.json')==launch['manifest_file_sha256'] and mh==launch['manifest_hash'],'Raw frozen confirmation manifest changed')
    require(m['planned_conditions']==list(IDS) and m['confirmation_inits']==list(INITIALS) and m['noise_repeat']==0
        and m['total_confirmation_episodes']==800 and 'reference_conditions' not in m,'Fresh independent case universe differs')
    for name,pin in m['source_files'].items():require(sha(ROOT/name)==pin,'Frozen source changed: '+name)
    selection=read(m['selection_evidence']['path']);canonical(selection,'selection_hash')
    require(sha(m['selection_evidence']['path'])==m['selection_evidence']['sha256'] and selection['selection']==m['selection_evidence']['frozen_selection'],'Frozen selected pair changed')
    frozen=m['preregistered_hypotheses'];require(len(frozen)==2 and len({d['suite'] for d in frozen})==2 and {d['better'] for d in frozen}==set(PATHS)
        and all({d['better'],d['worse']}==set(PATHS) for d in frozen),'Need exactly the two frozen opposite directions')
    accepted,reported=read(case_audit_path),read(analysis_path)
    require(accepted['status']=='pass' and accepted['new_cases']==800 and accepted['preserved_control_cells']==0 and accepted['manifest_hash']==mh,'Need complete fresh case acceptance')
    require(reported['complete'] and reported['valid_episodes']==800 and reported['manifest_hash']==mh,'Need complete production analysis')
    pins={row['receipt']:row['sha256'] for row in accepted['new_case_evidence']};require(len(pins)==800,'Need800 accepted receipt pins')
    records={};counts={c:0 for c in IDS};evidence=[]
    for cid in IDS:
        for suite in m['suites']:
            for task in range(10):
                for initial in INITIALS:
                    folder=STUDY/cid/'episodes'/suite/f'task{task:02d}'/f'init{initial:02d}';path=folder/'episode_receipt.json'
                    require(pins.get(str(path))==sha(path),'Accepted receipt changed');r=read(path);result=read(folder/'result.json')
                    require(r['valid'] and r['manifest_hash']==mh and r['condition']==cid and r['case']==dict(suite=suite,task=task,init=initial)
                        and type(r['success']) is bool and r['success']==result['success'] and sha(folder/'result.json')==r['artifacts']['result.json'],'Fresh outcome identity differs')
                    records[cid,suite,task,initial]=r['success'];counts[cid]+=int(r['success']);evidence.append(dict(path=str(path),sha256=sha(path)))
    rebuilt=[dict(frozen_direction=d,**direction([int(records[d['better'],d['suite'],t,i])-int(records[d['worse'],d['suite'],t,i]) for t in range(10) for i in INITIALS])) for d in frozen]
    costs=[]
    for service in accepted['service_evidence']:
        path=Path(service['run'])/'fixed_input_cost.json';require(sha(path)==service['artifacts']['fixed_input_cost.json'],'Accepted native cost changed')
        obj=read(path);require(obj['manifest_hash']==mh and [s['suite'] for s in obj['suites']]==m['suites'],'Cost identity differs');sums={c:0. for c in IDS}
        for suite in obj['suites']:
            require(sorted(b['condition'] for b in suite['conditions'])==sorted(IDS),'Cost condition set differs')
            for block in suite['conditions']:
                rows=block['measured'];require(len(rows)==12 and all(math.isfinite(r['policy_wall_ms']) and r['policy_wall_ms']>0 for r in rows),'Invalid native cost sample')
                sums[block['condition']]+=math.fsum(r['policy_wall_ms'] for r in rows)
        costs.append(dict(path=str(path),sha256=sha(path),ratios={c:sums['full_d10']/sums[c] for c in IDS}))
    require(len(costs)==8 and len({v['path'] for v in costs})==8,'Need eight distinct completed cost services')
    ratios={c:statistics.median(s['ratios'][c] for s in costs) for c in IDS};conclusion,qualified,matched=verdict(rebuilt,counts,ratios)
    sensitivity=adjusted([d['sensitivity_case_p'] for d in rebuilt]);differences=[]
    def compare(field,a,b,tol=0.):
        if not (abs(a-b)<=tol if tol else a==b):differences.append(dict(field=field,reconstructed=a,reported=b))
    for n,(a,b) in enumerate(zip(rebuilt,reported['primary_directions'])):
        info=b['inference'];compare(f'direction{n}',a['frozen_direction'],b['frozen_direction']);compare(f'direction{n}/gap',a['difference_count'],b['success_difference_count'])
        compare(f'direction{n}/task_counts',a['task_difference_counts'],info['primary_task_block']['task_difference_counts']);compare(f'direction{n}/task_p',a['p_one_sided'],info['primary_task_block']['p_one_sided'])
        compare(f'direction{n}/Holm2',a['primary_holm2_p'],info['primary_holm2_p']);compare(f'direction{n}/McNemar',a['sensitivity_case_p'],info['sensitivity_case_mcnemar_p'],1e-12)
        compare(f'direction{n}/sensitivity_Holm2',sensitivity[n],info['sensitivity_holm2_p'],1e-12)
        for label,field in [('case','case_interval'),('task_cluster','task_interval')]:
            for bound in ('low','high'):compare(f'direction{n}/{label}/{bound}',a[field][bound],info['confidence_intervals'][label][bound],1e-12)
        compare(f'direction{n}/LOO',a['loo_positive'],info['every_leave_one_task_out_positive']);compare(f'direction{n}/reproduced',a['direction_reproduced'],info['direction_reproduced'])
    for cid in IDS:
        total=next(v for v in reported['condition_summaries'] if v['level']=='all' and v['condition']==cid);compare(cid+'/total',counts[cid],total['successes'])
        for suite in m['suites']:
            row=next(v for v in reported['condition_summaries'] if v['level']=='suite' and v['condition']==cid and v['suite']==suite)
            compare(cid+'/'+suite,sum(records[cid,suite,t,i] for t in range(10) for i in INITIALS),row['successes'])
    for cid in PATHS:
        g=reported['overall_candidate_eligibility'][cid];compare(cid+'/usable',counts[cid]>=150 and counts['full_d10']-counts[cid]<=30,g['usable'])
        compare(cid+'/cost',ratios[cid],g['median_fixed_input_ratio'],1e-10);compare(cid+'/cost_pass',ratios[cid]>=1.15,g['cost_pass']);compare(cid+'/pair_matched',matched,g['pair_cost_pass'])
    compare('conclusion',conclusion,reported['conclusion']);compare('pair_cost_pass',matched,reported['pair_cost_pass'])
    value=dict(status='pass' if not differences else 'fail',manifest_hash=mh,new_cases=800,preserved_control_cells=0,differences=differences,success_counts=counts,
        directions=rebuilt,cost_ratios=ratios,pair_cost_pass=matched,candidate_qualified=qualified,conclusion=conclusion,case_evidence=evidence,cost_evidence=costs,
        production_statistics_imported=False,production_analysis=dict(path=str(analysis_path),sha256=sha(analysis_path)),case_audit=dict(path=str(case_audit_path),sha256=sha(case_audit_path)),
        source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),time=time.time(),scope='Independently reread accepted fresh outcomes and raw native cost sums;ten-task exact test,integer bootstrap,Holm2 andquality/cost verdict.Raw action reconstruction is covered by the pinned complete case audit.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(value,indent=2)+'\n')
    return {k:v for k,v in value.items() if k not in ('case_evidence','cost_evidence')}


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--case-audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=audit(a.analysis,a.case_audit,a.output);print(json.dumps(result));raise SystemExit(0 if result['status']=='pass' else 1)

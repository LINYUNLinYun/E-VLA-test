"""Independent720-case arithmetic; never imports production statistics or analysis."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time
import numpy as np
ROOT=Path('/root/nfs/embodied/benchmark');FAMILY=ROOT/'results/ease_insights_20261103/ae_complement_beam_v4'
ROOTS=(FAMILY/'confirmation',FAMILY);REPORT=Path(__file__).resolve().parent
SEED=20261111
CONDITIONS=('full_d10','vlm_c','ae14_d3')
DIRECTIONS=[dict(suite='libero_spatial',better='ae14_d3',worse='vlm_c'),dict(suite='libero_10',better='vlm_c',worse='ae14_d3')]


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def canonical(value,field):
    payload=dict(value);declared=payload.pop(field)
    require(hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()==declared,'Canonical identity changed')
    return declared


def adjusted(values):
    require(len(values)==2,'Exactly two tests')
    order=sorted(range(2),key=lambda i:(values[i],i));result=[0.,0.];previous=0.
    for rank,i in enumerate(order):
        previous=max(previous,min(1.,(2-rank)*values[i]));result[i]=previous
    return result


def direction(values):
    require(len(values)==60 and all(type(v) is int and v in (-1,0,1) for v in values),'Need60 heldout paired rollout differences')
    counts=[sum(values[6*t:6*t+6]) for t in range(10)];observed=sum(counts);extreme=0
    # Independent binary-sign enumeration rather than the production iterator.
    for assignment in range(1024):
        score=sum(v if assignment&(1<<t) else -v for t,v in enumerate(counts))
        extreme+=score>=observed
    case_draws=np.random.default_rng(SEED).choice(np.asarray(values,np.int64),size=(20000,60),replace=True).sum(axis=1)/60
    task_draws=np.random.default_rng(SEED).choice(np.asarray(counts,np.int64),size=(20000,10),replace=True).sum(axis=1)/60
    def interval(draws):return dict(low=float(np.quantile(draws,.025,method='linear')),high=float(np.quantile(draws,.975,method='linear')))
    a=sum(v==1 for v in values);b=sum(v==-1 for v in values);n=a+b
    sensitivity=sum(math.comb(n,k) for k in range(a,n+1))/2**n if n else 1.
    return dict(difference_count=observed,task_difference_counts=counts,p_one_sided=extreme/1024,
        case_interval=interval(case_draws),task_interval=interval(task_draws),sensitivity_case_p=sensitivity,
        loo_positive=all(observed-v>0 for v in counts))


def audit():
    bundle=read(REPORT/'confirmation_freeze.json');declared=canonical(bundle,'bundle_hash')
    acceptance=read(REPORT/'confirmation_acceptance.json');production=read(REPORT/'confirmation_analysis.json')
    require(acceptance['status']=='pass' and acceptance['new_cases']==720 and acceptance['bundle_hash']==declared,'Need720 accepted fresh rollouts')
    require(production['complete'] and production['valid_episodes']==720 and production['bundle_hash']==declared,'Need complete production analysis')
    pins={r['receipt']:r['sha256'] for r in acceptance['case_evidence']};require(len(pins)==720,'Need720 accepted receipt pins')
    records={};counts={c:0 for c in CONDITIONS};evidence=[];manifests=[]
    for repeat,root in enumerate(ROOTS):
        m=read(root/'manifest.json');mh=canonical(m,'manifest_hash');manifests.append(m)
        require(sha(root/'manifest.json')==bundle['streams'][repeat]['manifest_file_sha256'] and mh==bundle['streams'][repeat]['manifest_hash'],'Bundle stream identity differs')
        require(m['noise_repeat']==repeat and m['preregistered_hypotheses']==DIRECTIONS
            and m['planned_conditions']==list(CONDITIONS) and m['confirmation_inits']==[47,48,49] and 'reference_conditions' not in m,'Independent fresh design differs')
        for c in CONDITIONS:
            for s in m['suites']:
                for t in range(10):
                    for i in (47,48,49):
                        directory=root/c/'episodes'/s/f'task{t:02d}'/f'init{i:02d}';p=directory/'episode_receipt.json';require(pins.get(str(p))==sha(p),'Accepted receipt changed');r=read(p)
                        require(r['manifest_hash']==mh and r['condition']==c and r['case']==dict(suite=s,task=t,init=i) and type(r['success']) is bool,'Fresh outcome identity differs')
                        for name,pin in r['artifacts'].items():require(Path(name).name==name and sha(directory/name)==pin,'Fresh artifact changed')
                        records[c,repeat,s,t,i]=r['success'];counts[c]+=int(r['success']);evidence.append(dict(path=str(p),sha256=sha(p)))
    require(len(records)==720,'Duplicate/missing rollout identities')
    reconstructed=[]
    for frozen in DIRECTIONS:
        a,b,s=frozen['better'],frozen['worse'],frozen['suite']
        values=[int(records[a,r,s,t,i])-int(records[b,r,s,t,i]) for t in range(10) for r in (0,1) for i in (47,48,49)]
        reconstructed.append(dict(frozen_direction=frozen,**direction(values)))
    primary=adjusted([d['p_one_sided'] for d in reconstructed]);sensitivity=adjusted([d['sensitivity_case_p'] for d in reconstructed])
    differences=[]
    def compare(field,a,b,tolerance=0.):
        same=abs(a-b)<=tolerance if tolerance else a==b
        if not same:differences.append(dict(field=field,reconstructed=a,reported=b))
    for index,(actual,reported) in enumerate(zip(reconstructed,production['primary_directions'])):
        compare(f'direction{index}',actual['frozen_direction'],reported['frozen_direction']);info=reported['inference']
        compare(f'direction{index}/difference_count',actual['difference_count'],reported['success_difference_count'])
        compare(f'direction{index}/task_counts',actual['task_difference_counts'],info['primary_task_block']['task_difference_counts'])
        compare(f'direction{index}/task_p',actual['p_one_sided'],info['primary_task_block']['p_one_sided'])
        compare(f'direction{index}/Holm2',primary[index],info['primary_holm2_p'])
        compare(f'direction{index}/case_sensitivity',actual['sensitivity_case_p'],info['sensitivity_case_mcnemar_p'],1e-12)
        compare(f'direction{index}/sensitivity_Holm2',sensitivity[index],info['sensitivity_holm2_p'],1e-12)
        for label,field in [('case','case_interval'),('task_cluster','task_interval')]:
            for bound in ('low','high'):compare(f'direction{index}/{label}/{bound}',actual[field][bound],info['confidence_intervals'][label][bound],1e-12)
        compare(f'direction{index}/loo_positive',actual['loo_positive'],info['every_leave_one_task_out_positive'])
        reproduced=actual['difference_count']>0 and primary[index]<.05 and actual['task_interval']['low']>0 and actual['loo_positive']
        actual.update(primary_holm2_p=primary[index],direction_reproduced=reproduced)
        compare(f'direction{index}/reproduced',reproduced,info['direction_reproduced'])
        s,a,b=actual['frozen_direction']['suite'],actual['frozen_direction']['better'],actual['frozen_direction']['worse']
        for stream in reported['by_noise_stream']:
            r=stream['noise_repeat'];gap=sum(int(records[a,r,s,t,i])-int(records[b,r,s,t,i]) for t in range(10) for i in (47,48,49))
            compare(f'direction{index}/noise{r}/gap',gap,stream['success_difference_count'])
    ratios_by_run=[]
    for root,m in zip(ROOTS,manifests):
        for path in sorted(root.glob('*/runs/*/fixed_input_cost.json')):
            require(read(path.parent/'result.json').get('complete') is True,'Incomplete cost service')
            obj=read(path);require(obj['manifest_hash']==m['manifest_hash'],'Wrong cost identity')
            sums={c:0. for c in CONDITIONS}
            for suite in obj['suites']:
                for block in suite['conditions']:sums[block['condition']]+=math.fsum(r['policy_wall_ms'] for r in block['measured'])
            ratios_by_run.append({c:sums['full_d10']/v for c,v in sums.items()})
    require(len(ratios_by_run)==12,'Need12 completed fresh services')
    ratios={c:statistics.median(r[c] for r in ratios_by_run) for c in CONDITIONS}
    paths=CONDITIONS[1:];matched=max(ratios[c] for c in paths)<=1.10*min(ratios[c] for c in paths)
    eligibility={c:counts[c]>=180 and counts['full_d10']-counts[c]<=36 and ratios[c]>=1.15 and matched for c in paths}
    compare('counts',counts,production['success_counts'])
    for c in CONDITIONS:
        for s in manifests[0]['suites']:
            count=sum(records[c,r,s,t,i] for r in (0,1) for t in range(10) for i in (47,48,49))
            compare(c+'/'+s+'/count',count,production['condition_summaries'][c]['by_suite'][s]['successes'])
    for c in paths:
        prod=production['overall_candidate_eligibility'][c]
        compare(c+'/successes',counts[c],prod['successes']);compare(c+'/ratio',ratios[c],prod['median_fixed_input_ratio'],1e-10)
        compare(c+'/full_rate',counts['full_d10']/240,prod['full_success_rate'],1e-12)
        compare(c+'/usable',counts[c]>=180 and counts['full_d10']-counts[c]<=36,prod['usable'])
        compare(c+'/cost_pass',ratios[c]>=1.15,prod['cost_pass']);compare(c+'/pair_cost_pass',matched,prod['pair_cost_pass'])
    conclusion='complementarity_reproduced' if all(d['direction_reproduced'] for d in reconstructed) and all(eligibility.values()) else 'not_reproduced'
    compare('conclusion',conclusion,production['conclusion'])
    value=dict(status='pass' if not differences else 'fail',bundle_hash=declared,new_cases=720,preserved_control_cells=0,
        differences=differences,success_counts=counts,directions=reconstructed,cost_ratios=ratios,pair_cost_pass=matched,
        candidate_qualified=eligibility,conclusion=conclusion,case_evidence=evidence,
        production_analysis=dict(path=str(REPORT/'confirmation_analysis.json'),sha256=sha(REPORT/'confirmation_analysis.json')),
        auditor_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),time=time.time(),
        production_statistics_imported=False,scope='Independent ten-task exact randomization, integer-sum bootstrap, quality/cost verdict and per-noise descriptive gaps; repeat noise is not a new task')
    (REPORT/'confirmation_independent_statistics.json').write_text(json.dumps(value,indent=2)+'\n')
    return {k:v for k,v in value.items() if k!='case_evidence'}


if __name__=='__main__':
    result=audit();print(json.dumps(result));raise SystemExit(0 if result['status']=='pass' else 1)

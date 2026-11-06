"""Independent fresh-case confirmation arithmetic; never imports production statistics."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import time
import numpy as np

ROOT=Path('/root/nfs/embodied/benchmark')
STUDY=ROOT/'results/ease_insights_20261103/camera_paths_v3/confirmation'
REPORT=Path(__file__).resolve().parent
SEED=20261107


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def adjusted(values):
    order=sorted(range(len(values)),key=lambda i:(values[i],i));result=[0.]*len(values);previous=0.
    for rank,i in enumerate(order):
        previous=max(previous,min(1.,(len(values)-rank)*values[i]));result[i]=previous
    return result


def direction(values):
    require(len(values)==30,'Need30 heldout paired cases')
    task_counts=[sum(values[3*t:3*t+3]) for t in range(10)];observed=sum(task_counts)
    extreme=sum(sum(s*v for s,v in zip(signs,task_counts))>=observed for signs in itertools.product((-1,1),repeat=10))
    # Integer sums precede division. This is the same whole-task bootstrap,
    # avoiding a float cancellation being classified as strictly above zero.
    case_draws=np.random.default_rng(SEED).choice(np.asarray(values,np.int64),size=(20000,30),replace=True).sum(axis=1)/30
    task_draws=np.random.default_rng(SEED).choice(np.asarray(task_counts,np.int64),size=(20000,10),replace=True).sum(axis=1)/30
    def interval(draws):return dict(low=float(np.quantile(draws,.025,method='linear')),high=float(np.quantile(draws,.975,method='linear')))
    a_only=sum(v==1 for v in values);b_only=sum(v==-1 for v in values);n=a_only+b_only
    sensitivity=sum(math.comb(n,k) for k in range(a_only,n+1))/2**n if n else 1.
    return dict(difference_count=observed,task_difference_counts=task_counts,p_one_sided=extreme/1024,
        case_interval=interval(case_draws),task_interval=interval(task_draws),sensitivity_case_p=sensitivity,
        loo_positive=all(observed-v>0 for v in task_counts))


def audit():
    m=read(STUDY/'manifest.json');payload=dict(m);declared=payload.pop('manifest_hash')
    require(hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()==declared,'Confirmation manifest changed')
    require(m['confirmation_inits']==[47,48,49] and len(m['planned_conditions'])==3 and 'reference_conditions' not in m,'Fresh confirmation scope differs')
    acceptance=read(REPORT/'confirmation_acceptance.json');production=read(STUDY/'analysis/analysis.json')
    require(acceptance.get('status')=='pass' and acceptance.get('new_cases')==360 and acceptance.get('manifest_hash')==declared,'Need accepted360 fresh cases')
    require(production.get('complete') is True and production.get('valid_episodes')==360 and production['manifest_hash']==declared,'Need completed production confirmation')
    records={};counts={c:0 for c in m['planned_conditions']};evidence=[]
    for c in m['planned_conditions']:
        for s in m['suites']:
            for t in range(10):
                for i in (47,48,49):
                    directory=STUDY/c/'episodes'/s/f'task{t:02d}'/f'init{i:02d}';p=directory/'episode_receipt.json';r=read(p)
                    require(r['manifest_hash']==declared and r['condition']==c and r['case']==dict(suite=s,task=t,init=i) and type(r['success']) is bool,'Fresh outcome identity differs')
                    for name,digest in r['artifacts'].items():require(sha(directory/name)==digest,'Confirmation artifact changed')
                    records[c,s,t,i]=r['success'];counts[c]+=int(r['success']);evidence.append(dict(path=str(p),sha256=sha(p)))
    reconstructed=[]
    for frozen in m['preregistered_hypotheses']:
        a,b,s=frozen['better'],frozen['worse'],frozen['suite']
        values=[int(records[a,s,t,i])-int(records[b,s,t,i]) for t in range(10) for i in (47,48,49)]
        reconstructed.append(dict(frozen_direction=frozen,**direction(values)))
    primary=adjusted([d['p_one_sided'] for d in reconstructed]);sensitivity=adjusted([d['sensitivity_case_p'] for d in reconstructed])
    differences=[]
    def compare(field,a,b,tolerance=0.):
        same=abs(a-b)<=tolerance if tolerance else a==b
        if not same:differences.append(dict(field=field,reconstructed=a,reported=b))
    for index,(actual,reported) in enumerate(zip(reconstructed,production['primary_directions'])):
        compare(f'direction{index}',actual['frozen_direction'],reported['frozen_direction'])
        info=reported['inference'];compare(f'direction{index}/difference_count',actual['difference_count'],reported['success_difference_count'])
        compare(f'direction{index}/task_p',actual['p_one_sided'],info['primary_task_block']['p_one_sided'])
        compare(f'direction{index}/Holm2',primary[index],info['primary_holm2_p'])
        compare(f'direction{index}/case_sensitivity',actual['sensitivity_case_p'],info['sensitivity_case_mcnemar_p'],1e-12)
        compare(f'direction{index}/sensitivity_Holm2',sensitivity[index],info['sensitivity_holm2_p'],1e-12)
        for label,field in [('case','case_interval'),('task_cluster','task_interval')]:
            for bound in ('low','high'):compare(f'direction{index}/{label}/{bound}',actual[field][bound],info['confidence_intervals'][label][bound],1e-12)
        reproduced=actual['difference_count']>0 and primary[index]<.05 and actual['task_interval']['low']>0 and actual['loo_positive']
        actual.update(primary_holm2_p=primary[index],direction_reproduced=reproduced)
        compare(f'direction{index}/reproduced',reproduced,info['direction_reproduced'])
    ratios_by_run=[]
    for path in sorted(STUDY.glob('*/runs/*/fixed_input_cost.json')):
        require(read(path.parent/'result.json').get('complete') is True,'Incomplete cost service')
        sums={c['id']:0. for c in m['conditions']}
        for suite in read(path)['suites']:
            for block in suite['conditions']:sums[block['condition']]+=sum(r['policy_wall_ms'] for r in block['measured'])
        ratios_by_run.append({c:sums['full_d10']/v for c,v in sums.items()})
    require(len(ratios_by_run)==6,'Need6 completed confirmation services')
    ratios={c['id']:statistics.median(r[c['id']] for r in ratios_by_run) for c in m['conditions']}
    paths=m['planned_conditions'][1:];matched=max(ratios[c] for c in paths)<=1.10*min(ratios[c] for c in paths)
    eligibility={c:counts[c]>=90 and counts['full_d10']-counts[c]<=18 and ratios[c]>=1.15 and matched for c in paths}
    for c in paths:
        prod=production['overall_candidate_eligibility'][c]
        compare(c+'/successes',counts[c],prod['successes']);compare(c+'/ratio',ratios[c],prod['median_fixed_input_ratio'],1e-10)
        compare(c+'/full_rate',counts['full_d10']/120,prod['full_success_rate'],1e-12)
        compare(c+'/usable',counts[c]>=90 and counts['full_d10']-counts[c]<=18,prod['usable'])
        compare(c+'/cost_pass',ratios[c]>=1.15,prod['cost_pass'])
        compare(c+'/pair_cost_pass',matched,prod['pair_cost_pass'])
    conclusion='complementarity_reproduced' if all(d['direction_reproduced'] for d in reconstructed) and all(eligibility.values()) else 'not_reproduced'
    compare('conclusion',conclusion,production['conclusion'])
    value=dict(status='pass' if not differences else 'fail',manifest_hash=declared,new_cases=360,preserved_control_cells=0,
        differences=differences,success_counts=counts,directions=reconstructed,cost_ratios=ratios,pair_cost_pass=matched,
        candidate_qualified=eligibility,conclusion=conclusion,case_evidence=evidence,
        production_analysis=dict(path=str(STUDY/'analysis/analysis.json'),sha256=sha(STUDY/'analysis/analysis.json')),
        auditor_source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),time=time.time(),
        production_statistics_imported=False,scope='Independent fresh-case task randomization, integer-sum bootstrap and cost/quality verdict; zero-boundary discrepancies are preserved')
    output=REPORT/'confirmation_independent_statistics.json';output.write_text(json.dumps(value,indent=2)+'\n')
    return {k:v for k,v in value.items() if k!='case_evidence'}


if __name__=='__main__':
    result=audit();print(json.dumps(result));raise SystemExit(0 if result['status']=='pass' else 1)

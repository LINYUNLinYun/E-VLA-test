"""Post hoc task-block sensitivity on the two already-frozen pure-head directions.

Read160 original pure-head result JSONs. No pair,task family or direction is
selected here. Preserve the original preregistered case-level analysis.
"""
import hashlib
import itertools
import json
from pathlib import Path
import time

ROOT=Path('/root/nfs/embodied/benchmark');OUT=Path(__file__).resolve().parent
STATS=ROOT/'reports/ease_insights_20261103/draft_statistics/statistics.json'
STATS_SHA='9e644f34bdb929557b4bda3a6dd036504fbbae6bad2245a870659374c9e623ad'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)


def signflip(counts):
    require(len(counts)==10 and all(type(v) is int and -4<=v<=4 for v in counts),'Need ten integer four-initial-state task differences')
    observed=sum(counts)
    extreme=sum(sum(s*v for s,v in zip(signs,counts))>=observed for signs in itertools.product((-1,1),repeat=10))
    return dict(p_one_sided=extreme/1024,extreme_assignments=extreme,assignments=1024,task_difference_counts=counts,
        observed_difference_count=observed,observed_mean_difference=observed/40,zero_task_ties_included=True)


def holm(values):
    require(len(values)==2,'Only the existing fixed two-direction family is allowed')
    order=sorted(range(2),key=lambda i:(values[i],i));out=[0.,0.];running=0.
    for rank,i in enumerate(order):running=max(running,min(1.,(2-rank)*values[i]));out[i]=running
    return out


def audit():
    require(sha(STATS)==STATS_SHA,'Frozen confirmation statistics changed');stats=json.loads(STATS.read_text())
    require(stats['status']=='complete' and stats['family_size']==2 and stats['no_reselection'] is True,'Original confirmation family differs')
    pins={STATS:STATS_SHA}
    for pathkey,shakey in [('input_path','input_sha256'),('protocol_path','protocol_sha256'),('selection_path','selection_sha256')]:
        p=Path(stats[pathkey]);require(sha(p)==stats[shakey],'Frozen input/protocol/selection changed');pins[p]=stats[shakey]
    rows=json.loads(Path(stats['input_path']).read_text());require(len(rows)==400,'Original confirmation count differs')
    pure={};evidence=[]
    for r in rows:
        if r['mode']!='draft':continue
        p=Path(r['result']).resolve();require(p.is_relative_to(ROOT),'Raw result escaped the workspace')
        raw=json.loads(p.read_text());key=r['target_suite'],r['source_suite'],r['task_id'],r['init_id']
        require(key not in pure and r['task_id'] in range(10) and r['init_id'] in (21,22,23,24),'Duplicate or unexpected pure case')
        require(raw['status']=='complete' and raw['mode']=='draft' and raw['suite']==r['target_suite'] and raw['source_suite']==r['source_suite']
            and raw['task_id']==r['task_id'] and raw['init_id']==r['init_id'] and type(raw['success']) is bool
            and raw['success']==raw['episode']['success'] and int(raw['success'])==r['success'],'Original pure outcome/identity differs')
        pure[key]=int(raw['success']);evidence.append(dict(path=str(p),sha256=sha(p),target_suite=r['target_suite'],source_suite=r['source_suite'],task=r['task_id'],init=r['init_id']))
    require(len(pure)==160 and len(stats['pairs'])==2,'Need all160 original pure-head outcomes')
    directions=[]
    for pair in stats['pairs']:
        target=pair['target_suite'];foreign=pair['foreign_source_suite'];tasks=[];matched_total=foreign_total=0
        require(pair['complete'] is True and len(pair['primary']['task_differences'])==10,'Incomplete frozen comparison')
        for task in range(10):
            a=sum(pure[target,target,task,i] for i in (21,22,23,24));b=sum(pure[target,foreign,task,i] for i in (21,22,23,24));count=a-b
            original=next(r for r in pair['primary']['task_differences'] if r['task_id']==task)
            require(original['n_paired']==4 and original['difference']==count/4,'Recomputed task difference disagrees with original statistics')
            tasks.append(dict(task=task,matching_successes=a,foreign_successes=b,n=4,difference_count=count));matched_total+=a;foreign_total+=b
        counts=[t['difference_count'] for t in tasks];test=signflip(counts)
        require(test['observed_mean_difference']==pair['primary']['case_mean_difference'],'Overall frozen difference changed')
        directions.append(dict(target_suite=target,matching_source=target,foreign_source=foreign,matching_successes=matched_total,foreign_successes=foreign_total,
            n=40,task_clusters=10,positive_tasks=sum(v>0 for v in counts),negative_tasks=sum(v<0 for v in counts),tied_tasks=sum(v==0 for v in counts),
            by_task=tasks,task_block_test=test,original_task_cluster_CI95=pair['primary']['task_cluster_bootstrap_ci95'],
            every_leave_one_task_out_positive=all(sum(counts)-v>0 for v in counts)))
    adjusted=holm([d['task_block_test']['p_one_sided'] for d in directions])
    for d,p in zip(directions,adjusted):d['task_block_Holm2_p']=p
    value=dict(status='pass',analysis_type='Post hoc task-level sensitivity on the original fixed two contrasts; not a new preregistered primary test',time=time.time(),
        original_preregistered_case_tests_unchanged=True,new_validation_episodes=0,raw_pure_result_JSONs_verified=160,directions=directions,
        task_block_null='The ten task-level paired differences are sign-exchangeable under the null; initial states are grouped within task. This is an assumption, not randomized task sampling.',
        limitations=['Same ten tasks per target and original init21-24; no unseen-task or all-trained-draft generalization.',
            'Adds a clustering sensitivity analysis,not new independent data; original case-level McNemar/Holm and task bootstrap intervals remain unchanged.',
            'Checks original result identities/outcomes and frozen derived/protocol/selection inputs; does not independently revalidate every controller/action array or historical training/normalization provenance.',
            'Operational released-head portability under the common Pi0 inference contract; cannot isolate every training/preprocessing cause.'],
        input_pins={str(p):h for p,h in pins.items()},raw_result_evidence=evidence,source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        GPU_or_model_or_environment_started=False,current_path_outcomes_read=False,scientific_outcomes_modified=False)
    (OUT/'analysis.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ('raw_result_evidence','input_pins')}))


if __name__=='__main__':audit()

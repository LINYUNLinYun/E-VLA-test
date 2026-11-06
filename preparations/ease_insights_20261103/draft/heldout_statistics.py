"""Prespecified paired confirmation statistics; no GPU, reselection, or request pseudoreplication."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import SUITES,atomic_json

PROTOCOL_PATH=HERE/'heldout_statistics_protocol.json'
INITS=(21,22,23,24)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_mcnemar_greater(matching_only,foreign_only):
    b,c=int(matching_only),int(foreign_only)
    if min(b,c)<0:raise ValueError('Discordant counts must be nonnegative')
    n=b+c
    return 1. if not n else sum(math.comb(n,k) for k in range(b,n+1))/2**n


def holm(pvalues):
    if any(not 0<=p<=1 for p in pvalues):raise ValueError('Invalid p-value')
    order=sorted(range(len(pvalues)),key=lambda i:(pvalues[i],i))
    adjusted=[None]*len(pvalues);running=0.
    for rank,i in enumerate(order):
        running=max(running,(len(pvalues)-rank)*pvalues[i])
        adjusted[i]=min(1.,running)
    return adjusted


def rng_for(label):
    seed=int.from_bytes(hashlib.sha256(('ease-heldout-statistics-v1|'+label).encode()).digest()[:8],'little')
    return np.random.default_rng(seed)


def paired_effect(differences,tasks,label,replicates=20000):
    d=np.asarray(differences,dtype=float);tasks=np.asarray(tasks,dtype=int)
    if len(d)!=len(tasks) or not np.isfinite(d).all():raise ValueError('Invalid paired differences')
    if not len(d):
        return dict(n_paired=0,n_tasks=0,case_mean_difference=None,task_mean_difference=None,
                    paired_case_bootstrap_ci95=None,task_cluster_bootstrap_ci95=None,task_differences=[])
    distinct=sorted(set(tasks.tolist()))
    task_means=np.array([d[tasks==t].mean() for t in distinct])
    cases=rng_for(label+'|case').integers(0,len(d),size=(replicates,len(d)))
    clusters=rng_for(label+'|task').integers(0,len(task_means),size=(replicates,len(task_means)))
    return dict(n_paired=len(d),n_tasks=len(distinct),case_mean_difference=float(d.mean()),
        task_mean_difference=float(task_means.mean()),
        paired_case_bootstrap_ci95=np.quantile(d[cases].mean(axis=1),[.025,.975]).tolist(),
        task_cluster_bootstrap_ci95=np.quantile(task_means[clusters].mean(axis=1),[.025,.975]).tolist(),
        task_differences=[dict(task_id=t,n_paired=int((tasks==t).sum()),difference=float(v)) for t,v in zip(distinct,task_means)],
        bootstrap_replicates=replicates,interval='paired percentile95%',seed_label=label)


def selected_pairs(selection):
    if selection.get('status')=='none':return []
    if selection.get('status')!='selected':raise ValueError('Selection must already be frozen as selected or none')
    pairs=selection['selected_pairs']
    if len(pairs) not in (1,2):raise ValueError('Expected one or two prespecified pairs')
    targets=set()
    for p in pairs:
        target,foreign=p['target_suite'],p['foreign_source_suite']
        if target not in SUITES or foreign not in SUITES or target==foreign or target in targets:
            raise ValueError('Invalid or duplicate-target prespecified pair')
        targets.add(target)
    return pairs


def condition_specs(pair):
    target,foreign=pair['target_suite'],pair['foreign_source_suite']
    return [('draft',target,'matching_pure'),('draft',foreign,'foreign_pure'),
            ('flash',target,'matching_FLASH'),('flash',foreign,'foreign_FLASH'),('full',None,'Full')]


def audit_matrix(rows,pairs):
    allowed=set()
    for pair in pairs:
        for mode,source,role in condition_specs(pair):
            allowed.add((mode,source,pair['target_suite']))
    index={};full_sources=set()
    for row in rows:
        mode,source,target=row['mode'],row['source_suite'],row['target_suite']
        init,task=int(row['init_id']),int(row['task_id'])
        if init not in INITS or task not in range(10):raise ValueError('Discovery or unplanned case in held-out input')
        keysource=None if mode=='full' else source
        if (mode,keysource,target) not in allowed:raise ValueError('Unprescribed scientific condition in held-out input')
        if mode=='full':full_sources.add(source)
        key=(mode,keysource,target,task,init)
        if key in index:raise ValueError('Duplicate held-out scientific condition')
        if row['success'] not in (0,1,False,True):raise ValueError('Invalid binary episode success')
        index[key]=row
    if len(full_sources)>1:raise ValueError('Multiple Full source artifacts in one held-out analysis')
    audits=[]
    for pair in pairs:
        target=pair['target_suite'];conditions=[]
        for mode,source,role in condition_specs(pair):
            missing=[dict(target_suite=target,task_id=t,init_id=i) for t in range(10) for i in INITS
                     if (mode,source,target,t,i) not in index]
            conditions.append(dict(role=role,mode=mode,source_suite=source,expected_n=40,n=40-len(missing),missing=missing))
        audits.append(dict(target_suite=target,foreign_source_suite=pair['foreign_source_suite'],conditions=conditions,
                           complete=all(c['n']==40 for c in conditions)))
    return index,audits


def ratio(a,b):return a/b if b else None


def metric(row,name):
    if name=='success':return float(row['success'])
    if name=='acceptance_rate':
        rounds=row['verified_rounds']
        return ratio(row['accepted_actions'],12*rounds)
    if name=='rejection_full_fraction':return ratio(row['rejection_full_rounds'],row['requests'])
    if name=='policy_ms_per_executed_action':return ratio(row['policy_total_ms'],row['env_steps'])
    if name=='client_rpc_ms_per_executed_action':return ratio(row['client_rpc_total_ms'],row['env_steps'])
    raise ValueError(name)


def context(rows):
    n=len(rows);actions=sum(r['env_steps'] for r in rows);requests=sum(r['requests'] for r in rows)
    return dict(n=n,successes=sum(int(r['success']) for r in rows),executed_actions=actions,requests=requests,
        episode_wall_seconds=sum(r['wall_seconds'] for r in rows),
        mean_episode_wall_seconds=ratio(sum(r['wall_seconds'] for r in rows),n),
        pooled_policy_ms_per_executed_action=ratio(sum(r['policy_total_ms'] for r in rows),actions),
        pooled_client_rpc_ms_per_executed_action=ratio(sum(r['client_rpc_total_ms'] for r in rows),actions))


def metric_effect(index,pair,mode,inits,name,subset,replicates):
    target,foreign=pair['target_suite'],pair['foreign_source_suite'];differences=[];tasks=[];undefined=[]
    matching=[];foreign_rows=[]
    for task in range(10):
        for init in inits:
            a=index[(mode,target,target,task,init)];b=index[(mode,foreign,target,task,init)]
            matching.append(a);foreign_rows.append(b)
            av,bv=metric(a,name),metric(b,name)
            if av is None or bv is None or not math.isfinite(av) or not math.isfinite(bv):
                undefined.append(dict(task_id=task,init_id=init,matching_value=av,foreign_value=bv));continue
            differences.append(av-bv);tasks.append(task)
    label='|'.join((target,foreign,mode,subset,name))
    result=paired_effect(differences,tasks,label,replicates)
    result.update(metric=name,mode=mode,subset=subset,expected_pairs=10*len(inits),undefined_pairs=undefined,
        difference_direction='matching minus foreign',
        favorable_direction='positive' if name in ('success','acceptance_rate') else 'negative',
        matching_context=context(matching),foreign_context=context(foreign_rows),confirmatory_p_value=None,
        statistical_unit='paired episode; no request-level independent samples',warm_state_claim=False)
    return result


def compute_statistics(rows,selection,replicates=20000):
    pairs=selected_pairs(selection)
    if not pairs:return dict(status='no_prespecified_positive_pairs',pairs=[],no_reselection=True)
    index,audits=audit_matrix(rows,pairs)
    complete=all(a['complete'] for a in audits)
    if not complete:
        return dict(status='incomplete_not_tested',pairs=[dict(a,primary_status='incomplete_not_tested',primary=None,secondary=[]) for a in audits],
                    family_size=len(pairs),expected_conditions=200*len(pairs),observed_conditions=len(rows),
                    reason='No primary tests, bootstrap intervals, or replicated label before all prespecified conditions are complete',no_reselection=True)
    outputs=[];pvalues=[]
    for pair,audit in zip(pairs,audits):
        target,foreign=pair['target_suite'],pair['foreign_source_suite']
        matching=[index[('draft',target,target,t,i)] for t in range(10) for i in INITS]
        foreign_rows=[index[('draft',foreign,target,t,i)] for t in range(10) for i in INITS]
        b=sum(bool(a['success']) and not bool(c['success']) for a,c in zip(matching,foreign_rows))
        c=sum(bool(c['success']) and not bool(a['success']) for a,c in zip(matching,foreign_rows))
        pvalue=exact_mcnemar_greater(b,c);pvalues.append(pvalue)
        primary=metric_effect(index,pair,'draft',INITS,'success','all4_init21_24',replicates)
        primary.update(matching_only_successes=b,foreign_only_successes=c,
            both_successes=sum(bool(a['success']) and bool(c['success']) for a,c in zip(matching,foreign_rows)),
            both_failures=sum(not bool(a['success']) and not bool(c['success']) for a,c in zip(matching,foreign_rows)),
            exact_mcnemar_one_sided_p=pvalue,confirmatory_p_value=pvalue,primary_direction='matching > foreign',
            scope='Case-level fixed-task confirmation; task-cluster CI is robustness information, not unseen-task generalization')
        secondary=[metric_effect(index,pair,'flash',INITS,name,'all4_init21_24',replicates)
                   for name in ('success','acceptance_rate','rejection_full_fraction')]
        for mode in ('draft','flash'):
            for subset,inits in [('all4_init21_24',INITS),('repeat_task_init22_24',(22,23,24))]:
                secondary += [metric_effect(index,pair,mode,inits,name,subset,replicates)
                              for name in ('policy_ms_per_executed_action','client_rpc_ms_per_executed_action')]
        full_context={subset:context([index[('full',None,target,t,i)] for t in range(10) for i in inits])
                      for subset,inits in [('all4_init21_24',INITS),('repeat_task_init22_24',(22,23,24))]}
        outputs.append(dict(audit,primary=primary,secondary=secondary,Full_context=full_context))
    adjusted=holm(pvalues)
    for output,pvalue in zip(outputs,adjusted):
        output['primary']['holm_adjusted_one_sided_p']=pvalue
        output['primary']['holm_family_size']=len(pairs)
        output['primary_status']=('replicated' if pvalue<=.05 and output['primary']['case_mean_difference']>0 else 'not_replicated')
        output['task_cluster_interval_excludes_zero']=output['primary']['task_cluster_bootstrap_ci95'][0]>0
    return dict(status='complete',pairs=outputs,family_size=len(pairs),expected_conditions=200*len(pairs),observed_conditions=len(rows),
                alpha=.05,primary_test='one-sided exact McNemar, Holm across frozen pairs',
                secondary_inference='descriptive paired effects and bootstrap intervals only',no_reselection=True,
                timing_caution='Per-action complete policy/RPC costs include zero-action rejection requests. Repeat-task is not guaranteed warm. Mean request latency is not actual acceleration; GPU stages are not end-to-end.')


def markdown_report(report):
    lines=['# Independent held-out confirmation','', 'Status: **'+report['status']+'**. No pair is chosen using these results.','']
    for pair in report['pairs']:
        lines += ['## '+pair['target_suite']+': matching versus '+pair['foreign_source_suite'],'',
                  'Primary status: **'+pair.get('primary_status','not_tested')+'**.','',
                  '| Condition | Valid / planned | Missing |','|---|---:|---:|']
        for condition in pair['conditions']:
            lines.append('| %s | %d/40 | %d |'%(condition['role'],condition['n'],len(condition['missing'])))
        primary=pair.get('primary')
        if primary is None:
            lines += ['', 'Inference is withheld until the entire prespecified matrix is complete. Missing case identities are listed in statistics.json.',''];continue
        lines += ['', 'Pure success: matching %d/40, foreign %d/40; matching-only=%d, foreign-only=%d.'%(
            primary['matching_context']['successes'],primary['foreign_context']['successes'],primary['matching_only_successes'],primary['foreign_only_successes']),
            'One-sided exact McNemar p=%.6g; Holm-adjusted p=%.6g.'%(primary['exact_mcnemar_one_sided_p'],primary['holm_adjusted_one_sided_p']),
            'Case mean gap=%.4f, paired-case95%% CI=%s; task mean gap=%.4f, task-cluster95%% CI=%s.'%(
                primary['case_mean_difference'],primary['paired_case_bootstrap_ci95'],primary['task_mean_difference'],primary['task_cluster_bootstrap_ci95']),
            '', '| Secondary metric | Mode / subset | Paired n | Matching − foreign | Case95% CI | Task-cluster95% CI |',
            '|---|---|---:|---:|---|---|']
        for metric_row in pair['secondary']:
            value=metric_row['case_mean_difference']
            lines.append('| %s | %s / %s | %d | %s | %s | %s |'%(metric_row['metric'],metric_row['mode'],metric_row['subset'],metric_row['n_paired'],
                'undefined' if value is None else '%.5g'%value,metric_row['paired_case_bootstrap_ci95'],metric_row['task_cluster_bootstrap_ci95']))
        lines += ['', 'Positive differences favor matching for success/acceptance; negative differences favor matching for rejection fraction and cost. Full baseline, both success counts, episode walls, and request/action totals are in statistics.json.','']
    lines += ['McNemar is a case-level analysis on repeated initial states of fixed tasks. Task-cluster intervals use ten clusters and do not establish generalization to unseen tasks. Secondary metrics are descriptive; requests are never independent samples.',
              '', 'FLASH need not lose success for draft generalization to matter: Full can preserve success while verification or service cost deteriorates. Cost uses complete policy/RPC milliseconds per executed action, including zero-action rejection requests; per-request latency is not actual acceleration. Repeat-task init22–24 is not guaranteed compilation-free.','']
    return '\n'.join(lines)


def fixture_rows():
    pair=dict(target_suite='libero_goal',foreign_source_suite='libero_spatial')
    rows=[]
    for mode,source,role in condition_specs(pair):
        for t in range(10):
            for i in INITS:
                success=1 if mode=='full' or source=='libero_goal' else int(t>=5)
                rows.append(dict(mode=mode,source_suite=source or 'libero_spatial',target_suite='libero_goal',task_id=t,init_id=i,
                    success=success,env_steps=12,requests=2,policy_total_ms=100.,client_rpc_total_ms=110.,wall_seconds=2.,
                    verified_rounds=int(mode=='flash'),accepted_actions=6 if mode=='flash' else 0,rejection_full_rounds=int(mode=='flash')))
    return rows,dict(status='selected',selected_pairs=[pair])


def self_test():
    assert exact_mcnemar_greater(5,0)==1/32 and exact_mcnemar_greater(0,5)==1.
    assert exact_mcnemar_greater(0,0)==1.
    assert np.allclose(holm([.01,.04]),[.02,.04]) and np.allclose(holm([.03,.03]),[.06,.06])
    d=np.repeat([1.]*5+[-1.]*5,4);tasks=np.repeat(np.arange(10),4)
    effect=paired_effect(d,tasks,'fixture_cluster',4000)
    case_width=np.diff(effect['paired_case_bootstrap_ci95'])[0];task_width=np.diff(effect['task_cluster_bootstrap_ci95'])[0]
    assert task_width>case_width and effect['case_mean_difference']==effect['task_mean_difference']==0.
    rows,selection=fixture_rows()
    missing=compute_statistics(rows[:-1],selection,200)
    assert missing['status']=='incomplete_not_tested' and missing['pairs'][0]['primary'] is None
    complete=compute_statistics(rows,selection,500)
    assert complete['pairs'][0]['primary_status']=='replicated'
    assert complete['pairs'][0]['primary']['matching_only_successes']==20
    assert all(r['n_paired'] in (30,40) for r in complete['pairs'][0]['secondary'])
    second=[]
    for row in rows:
        copied=dict(row,target_suite='libero_object')
        if row['mode']!='full':copied['source_suite']='libero_object' if row['source_suite']=='libero_goal' else 'libero_goal'
        second.append(copied)
    two_selection=dict(status='selected',selected_pairs=selection['selected_pairs']+[dict(target_suite='libero_object',foreign_source_suite='libero_goal')])
    two=compute_statistics(rows+second,two_selection,200)
    assert two['family_size']==2 and len(two['pairs'])==2
    assert all(abs(r['primary']['holm_adjusted_one_sided_p']-2*r['primary']['exact_mcnemar_one_sided_p'])<1e-15 for r in two['pairs'])
    assert compute_statistics(rows+second[:-1],two_selection,100)['status']=='incomplete_not_tested'
    for row in rows:row['success']=1
    zero=compute_statistics(rows,selection,200)
    assert zero['pairs'][0]['primary']['exact_mcnemar_one_sided_p']==1. and zero['pairs'][0]['primary_status']=='not_replicated'
    assert zero['pairs'][0]['primary']['paired_case_bootstrap_ci95']==[0.,0.]
    print(json.dumps(dict(status='passed',tests=['one-sided-direction','zero-discordance','Holm','task-clustering','incomplete-not-tested','complete-direction','secondary-case-denominators','two-pair-family','withhold-until-both-pairs-complete'])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection')
    p.add_argument('--analysis',help='Held-out analyze.py output; only episodes.json is used')
    p.add_argument('--output')
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test();return
    if not a.selection or not a.analysis or not a.output:p.error('--selection, --analysis and --output are required')
    selection_path=Path(a.selection);selection=json.loads(selection_path.read_text())
    episodes_path=Path(a.analysis)/'episodes.json'
    rows=json.loads(episodes_path.read_text()) if episodes_path.exists() else []
    protocol=json.loads(PROTOCOL_PATH.read_text())
    report=compute_statistics(rows,selection,protocol['effect_intervals']['replicates'])
    report.update(protocol_path=str(PROTOCOL_PATH.resolve()),protocol_sha256=sha(PROTOCOL_PATH),
        selection_path=str(selection_path.resolve()),selection_sha256=sha(selection_path),
        input_path=str(episodes_path.resolve()),input_sha256=sha(episodes_path) if episodes_path.exists() else None,
        producer_sha256=sha(__file__),bootstrap_replicates=protocol['effect_intervals']['replicates'])
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'statistics.json',report)
    (output/'README.md').write_text(markdown_report(report))
    print(json.dumps(dict(status=report['status'],output=str(output.resolve()),pairs=len(report['pairs']),
        primary_statuses=[r.get('primary_status') for r in report['pairs']],no_gpu_launched=True,no_selection_changed=True)))

if __name__=='__main__':main()

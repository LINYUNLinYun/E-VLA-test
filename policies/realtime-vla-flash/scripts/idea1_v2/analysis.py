"""Paired success analysis; task-cluster bootstrap stratified by suite."""
import json
from pathlib import Path
import numpy as np
from .core import SUITES, write_json
from .kalman import table
from .runner import ALT, PHASE

def completed(root,condition):
    result={}
    for path in (root/'closed_loop'/condition).glob('*/*/*/*/validated.json'):
        row=json.loads(path.read_text())
        if row.get('valid') is not True:raise ValueError('Invalid record mislabeled validated: '+str(path))
        key=(row['task_suite_name'],row['task_id'],row['episode_idx'])
        if key in result:raise ValueError('Duplicate valid episode identity')
        result[key]=row
    return result

def run(root,source):
    successes=[];tasks=[];latencies=[];bootstrap=[]
    rng=np.random.default_rng(20261014)
    for condition in (ALT,PHASE):
        current=completed(root,condition)
        if not current:continue
        for baseline in ('Full18_D3_R5_FORMAL','V14_A03_D3_R5_FORMAL'):
            old={}
            for suite in SUITES:
                for row in json.loads((source/'closed_loop'/baseline/suite/'run/episode_log.json').read_text()):
                    old[(suite,row['task_id'],row['episode_idx'])]=row
            for suite in (*SUITES,'overall'):
                keys=[k for k in current if suite=='overall' or k[0]==suite]
                if not keys:continue
                a=np.array([bool(current[k]['success']) for k in keys]);b=np.array([bool(old[k]['success']) for k in keys])
                successes.append({'condition':condition,'baseline':baseline,'suite':suite,'successes':int(a.sum()),'denominator':len(keys),
                                  'baseline_successes':int(b.sum()),'difference_pp':float((a.mean()-b.mean())*100),
                                  'both_success':int((a&b).sum()),'new_failure':int((~a&b).sum()),'recovered_success':int((a&~b).sum()),'both_failure':int((~a&~b).sum())})
            for suite in SUITES:
                for task in range(10):
                    keys=[k for k in current if k[:2]==(suite,task)]
                    if keys:tasks.append(dict(condition=condition,baseline=baseline,suite=suite,task=task,denominator=len(keys),
                                              successes=sum(bool(current[k]['success']) for k in keys),baseline_successes=sum(bool(old[k]['success']) for k in keys)))
            if len(current)==200:
                samples={s:[] for s in (*SUITES,'overall')}
                for _ in range(2000):
                    differences=[]
                    for suite in SUITES:
                        drawn_tasks=rng.integers(0,10,10)
                        difference=np.mean([int(current[(suite,int(t),i)]['success'])-int(old[(suite,int(t),i)]['success']) for t in drawn_tasks for i in range(5)])
                        differences.append(difference);samples[suite].append(difference*100)
                    samples['overall'].append(np.mean(differences)*100)
                for suite,values in samples.items():
                    bootstrap.append({'condition':condition,'baseline':baseline,'suite':suite,'cluster':'task; stratified suite; paired','replicates':2000,'seed':20261014,
                                      'difference_pp_ci025':float(np.quantile(values,.025)),'difference_pp_ci975':float(np.quantile(values,.975))})
        for key,row in current.items():
            directory=Path(row['directory'])
            from .core import rows
            infer=rows(directory/row['infer_path'])
            times=[r['client_roundtrip_ms'] for r in infer]
            latency={'condition':condition,'suite':key[0],'task':key[1],'init':key[2],
                              'rounds':len(infer),'actions':row['actual_actions'],'full_rounds':row['full_rounds'],'full_actions':row['full_actions'],
                              'full_round_fraction':row['full_rounds']/len(infer),'full_action_fraction':row['full_actions']/row['actual_actions'],
                              'request_total_ms':sum(times),'request_mean_ms':float(np.mean(times)),
                              'request_p50_ms':float(np.percentile(times,50)),'request_p95_ms':float(np.percentile(times,95)),
                              'client_process_wall_s':row.get('client_process_wall_s'),
                              'graph_rebuild_rounds':sum(bool(r['idea1_v2'].get('prompt_graph_rebuild')) for r in infer)}
            for metric in ('cuda_policy_ms','prepare_ms','sample_actions_ms'):
                measured=[float(r['idea1_v2'][metric]) for r in infer]
                latency.update({metric+'_total':sum(measured),metric+'_mean':float(np.mean(measured)),
                                metric+'_p50':float(np.percentile(measured,50)),metric+'_p95':float(np.percentile(measured,95))})
            latencies.append(latency)
    if successes:table(root/'analysis/success.csv',successes,exclusive=False)
    if tasks:table(root/'analysis/success_tasks.csv',tasks,exclusive=False)
    if latencies:table(root/'analysis/latency.csv',latencies,exclusive=False)
    if bootstrap:table(root/'analysis/success_bootstrap.csv',bootstrap,exclusive=False)
    from .diagnostics import run as diagnose
    diagnose(root,source)
    result={'E2_valid':len(completed(root,ALT)),'E3_valid':len(completed(root,PHASE)),'paired_statistics':bool(successes)}
    write_json(root/'analysis/status.json',result)
    return result

"""Complete720-rollout analysis of two prospectively frozen suite contrasts."""
import argparse
import json
import statistics
from pathlib import Path
from .common import *
from .prepare_confirmation import BUNDLE,ROOTS,read_bundle
from .protocol import REPORT
from .statistics import direction_inference,apply_verdict
from .cost_records import cost_records


def cell_key(cid,repeat,scene):return (cid,repeat,scene['suite'],scene['task'],scene['init'])


def load_complete(bundle_path=BUNDLE):
    bundle=read_bundle(bundle_path,verify_sources=True)
    accepted_path=REPORT/'confirmation_acceptance.json';accepted=json.loads(accepted_path.read_text())
    require(accepted['status']=='pass' and accepted['new_cases']==720 and accepted['bundle_hash']==bundle['bundle_hash'],
        'Inference stays locked until all720 cases pass reconstruction')
    pins={r['receipt']:r['sha256'] for r in accepted['case_evidence']}
    require(len(pins)==720,'Accepted evidence must bind720 unique receipts')
    records={};manifests=[]
    for repeat,root in enumerate(ROOTS):
        m=read_manifest(root/'manifest.json');manifests.append(m)
        for cid in CONDITIONS:
            for scene in cases(m):
                directory=root/cid/'episodes'/scene['suite']/f"task{scene['task']:02d}"/f"init{scene['init']:02d}"
                path=directory/'episode_receipt.json';require(pins.get(str(path))==sha(path),'Accepted receipt changed after reconstruction');r=json.loads(path.read_text())
                require(r['manifest_hash']==m['manifest_hash'] and r['condition']==cid and r['case']==scene
                    and type(r['success']) is bool and r['valid'],'Mixed confirmation outcome')
                for name,pin in r['artifacts'].items():
                    require(Path(name).name==name and sha(directory/name)==pin,'Confirmation artifact changed')
                records[cell_key(cid,repeat,scene)]=r
    require(len(records)==720,'Need all720 unique rollout identities')
    return bundle,manifests,records


def paired_summary(a,b,scene_keys,records):
    outcomes=[(records[(a,*k)]['success'],records[(b,*k)]['success']) for k in scene_keys]
    onlya=sum(x and not y for x,y in outcomes);onlyb=sum(y and not x for x,y in outcomes)
    return dict(a=a,b=b,n_paired=len(outcomes),a_success=sum(x for x,y in outcomes),b_success=sum(y for x,y in outcomes),
        both_success=sum(x and y for x,y in outcomes),both_failure=sum(not x and not y for x,y in outcomes),
        a_only=onlya,b_only=onlyb,success_difference_count=onlya-onlyb,success_difference=(onlya-onlyb)/len(outcomes))


def analyze(bundle_path=BUNDLE):
    bundle,manifests,records=load_complete(bundle_path);directions=[]
    for d in DIRECTIONS:
        keys=[(r,d['suite'],t,i) for t in range(10) for r in (0,1) for i in (47,48,49)]
        values=[int(records[(d['better'],*k)]['success'])-int(records[(d['worse'],*k)]['success']) for k in keys]
        task_ids=[k[2] for k in keys]
        directions.append(dict(frozen_direction=d,**paired_summary(d['better'],d['worse'],keys,records),
            by_task=[dict(task=t,**paired_summary(d['better'],d['worse'],[k for k in keys if k[2]==t],records)) for t in range(10)],
            by_noise_stream=[dict(noise_repeat=r,**paired_summary(d['better'],d['worse'],[k for k in keys if k[0]==r],records)) for r in (0,1)],
            inference=direction_inference(values,task_ids)))
    services=[];cost_by_stream=[]
    for repeat,(root,m) in enumerate(zip(ROOTS,manifests)):
        costs=cost_records(root,m)
        require(costs['valid_services']==6 and not costs['excluded_or_pending'],'Need six complete numerical/cost services per stream')
        for s in costs['services']:
            require(json.loads((Path(s['path']).parent/'result.json').read_text()).get('complete') is True,'Incomplete cost service')
            services.append(dict(s,noise_repeat=repeat))
        cost_by_stream.append(dict(noise_repeat=repeat,records=costs))
    require(len(services)==12,'Need12 completed fresh cost services')
    ratios={c:statistics.median(s['full_over_policy_cycle'][c] for s in services) for c in CONDITIONS}
    pair_factor=max(ratios[c] for c in CONDITIONS[1:])/min(ratios[c] for c in CONDITIONS[1:]);matched=pair_factor<=1.10
    counts={c:sum(r['success'] for k,r in records.items() if k[0]==c) for c in CONDITIONS};eligibility={}
    for c in CONDITIONS[1:]:
        eligibility[c]=dict(successes=counts[c],n=240,success_rate=counts[c]/240,full_success_rate=counts['full_d10']/240,
            usable=counts[c]>=180 and counts['full_d10']-counts[c]<=36,median_fixed_input_ratio=ratios[c],
            cost_pass=ratios[c]>=1.15,pair_cost_pass=matched)
    verdict=apply_verdict(directions,{c:g['usable'] and g['cost_pass'] and matched for c,g in eligibility.items()})
    def summarize(rows):
        n=len(rows);wall=sum(r['wall_s'] for r in rows);actions=sum(r['executed_steps'] for r in rows);requests=sum(r['predictions'] for r in rows)
        policy=sum(r['model_request_ms_total'] for r in rows);full=sum(r['counts']['full'] for r in rows)
        return dict(n=n,successes=sum(r['success'] for r in rows),actions=actions,predictions=requests,
            full_requests=full,fast_requests=requests-full,full_request_fraction=full/requests,
            native_policy_ms_total=policy,native_policy_ms_per_executed_action=policy/actions,
            rpc_ms_total=sum(r['rpc_ms_total'] for r in rows),summed_episode_wall_s=wall,
            episode_per_summed_worker_hour=n*3600/wall,
            wall_scope='Sum of completed episode times excludes preparation; it is not parallel experiment makespan')
    summaries={c:dict(overall=summarize([r for k,r in records.items() if k[0]==c]),
        by_suite={s:summarize([r for k,r in records.items() if k[0]==c and k[2]==s]) for s in SUITES},
        by_stream={str(rep):summarize([r for k,r in records.items() if k[0]==c and k[1]==rep]) for rep in (0,1)}) for c in CONDITIONS}
    runs=[]
    for repeat,root in enumerate(ROOTS):
        for p in sorted(root.glob('*/runs/*/result.json')):runs.append(dict(noise_repeat=repeat,path=str(p),sha256=sha(p),receipt=json.loads(p.read_text())))
    result=dict(revision='ease-cross-module-complete720-analysis-v4',bundle_hash=bundle['bundle_hash'],
        manifest_hashes=[m['manifest_hash'] for m in manifests],complete=True,valid_episodes=720,expected_episodes=720,
        unique_physical_states_per_algorithm=120,rollouts_per_algorithm=240,
        primary_directions=directions,overall_candidate_eligibility=eligibility,conclusion=verdict,
        success_counts=counts,condition_summaries=summaries,
        fixed_input_cost=dict(valid_services=12,services=services,by_stream=cost_by_stream,median_full_over_policy_cycle=ratios,
            pair_cost_factor=pair_factor,pair_cost_pass=matched,scope='Native policy cost on old fixed-input bank; no exclusive-GPU or closed-loop acceleration claim'),
        preparation_and_service_runs=runs,protocol_evidence=bundle['protocol_evidence'],
        analysis_protocol=dict(primary='One-sided exact ten-task signflip including zero ties;Holm2',
            bootstrap_replicates=20000,bootstrap_seed=20261111,quantile='linear percentile',
            observations_per_task=6,task_clusters_per_targeted_suite=10,
            sensitivity='Paired rollout McNemar/CI is descriptive only; noise streams are repeated physical states',
            thresholds='Both positive,Holm2<.05,task CI lower>0,every taskLOO>0;both paths>=.75 and>=Full-.15;ratio>=1.15,pairwithin10%',no_reselection=True),
        evidence=dict(bundle=dict(path=str(bundle_path),sha256=sha(bundle_path)),acceptance=dict(path=str(accepted_path()),sha256=sha(accepted_path())),
            source=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__))),
        limitations='Post-hoc cross-branch discovery followed by untouched-state fixed-pair replication. No global routing necessity,training/generalization universality,online selector,phase switching,or study-wide5% error claim.',
        scientific_outcomes_modified=False,goal_complete=False)
    atomic_json(REPORT/'confirmation_analysis.json',result)
    return {k:result[k] for k in ('complete','valid_episodes','success_counts','conclusion')}


def accepted_path():return REPORT/'confirmation_acceptance.json'


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--bundle',type=Path,default=BUNDLE);args=p.parse_args();print(json.dumps(analyze(args.bundle)))

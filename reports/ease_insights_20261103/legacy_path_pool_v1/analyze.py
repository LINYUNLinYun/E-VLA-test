"""Hypothesis generation from historicalR12 variants; never current720 confirmation."""
import collections
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path('/root/nfs/embodied/benchmark');sys.path.insert(0,str(ROOT))
from preparations.idea1_v3_1.common import atomic_json,digest
REPORT=Path(__file__).resolve().parent
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def select(counts):
    require('Full18' in counts,'MissingFullreference')
    for cid,by_suite in counts.items():
        require(set(by_suite)==set(SUITES) and all(len(v)==10 and all(type(x) is int and 0<=x<=10 for x in v) for v in by_suite.values()),'Needfour suites,ten tasks,ten initial states per task')
    totals={c:sum(sum(v) for v in suites.values()) for c,suites in counts.items()}
    eligible=sorted(c for c in counts if c!='Full18' and totals[c]>=300 and totals['Full18']-totals[c]<=60)
    ranked=[]
    for a,b in itertools.combinations(eligible,2):
        for sa,sb in itertools.permutations(SUITES,2):
            da=[x-y for x,y in zip(counts[a][sa],counts[b][sa])];db=[x-y for x,y in zip(counts[b][sb],counts[a][sb])]
            ga,gb=sum(da),sum(db)
            if min(ga,gb)<10 or min(sum(v>0 for v in da),sum(v>0 for v in db))<4:continue
            if not all(sum(vs)-v>0 for vs in (da,db) for v in vs):continue
            candidate=dict(paths=[a,b],directions=[dict(suite=sa,better=a,worse=b),dict(suite=sb,better=b,worse=a)],
                direction_difference_counts=[ga,gb],positive_tasks=[sum(v>0 for v in da),sum(v>0 for v in db)],
                minimum_gap=min(ga,gb)/100,sum_gaps=(ga+gb)/100,minimum_overall_success=min(totals[a],totals[b])/400)
            rank=(-min(ga,gb),-(ga+gb),-min(totals[a],totals[b]),a,b,sa,sb);ranked.append((rank,candidate))
    ranked.sort(key=lambda row:row[0])
    return dict(status='selected' if ranked else 'no_candidate',selected=ranked[0][1] if ranked else None,
        quality_eligible_paths=eligible,success_counts=totals,ranked_candidates=[c for r,c in ranked],
        scope='LegacyR12 exploratory success-profile selection;numericparity,noisepairing,andmatchedcostsarenotcertified')


def load(protocol,initials,conditions):
    records={};files=[];manifest_pins=[];expected={(c,s,t,i) for c in conditions for s in SUITES for t in range(10) for i in initials}
    for chunk in protocol['source_chunks']:
        if chunk['variant'] not in conditions or chunk['start']!=min(initials):continue
        path=Path(chunk['path']);data=path.read_bytes();require(hashlib.sha256(data).hexdigest()==chunk['sha256'],'Historicalsourcechanged')
        metadata=path.with_name('manifest.json');manifest=read(metadata)
        require(manifest['replan_steps']==12 and manifest['task_suite_name']==chunk['suite'],'Legacyruncontractdiffers')
        rows=json.loads(data);require(type(rows) is list and len(rows)==100,'Incompletelegacychunk')
        for index,row in enumerate(rows):
            key=(chunk['variant'],row['task_suite_name'],row['task_id'],row['episode_idx'])
            require(key in expected and key not in records and type(row['success']) is bool,'Duplicate,unplanned,orinvalidlegacyoutcome')
            require(row['num_steps_wait']==10,'Legacysettlingdiffers')
            records[key]=dict(success=row['success'],failure_reason=row.get('failure_reason'),max_steps=row.get('max_steps'),
                actions=row.get('executed_action_count'),logged_policy_ms=row.get('policy_time_sum_ms'),source=str(path),source_row=index)
        files.append(dict(path=str(path),sha256=chunk['sha256'],rows=100));manifest_pins.append(dict(path=str(metadata),sha256=sha(metadata)))
    require(set(records)==expected,'Missinglegacyphysicalcaseidentity')
    counts={c:{s:[sum(int(records[c,s,t,i]['success']) for i in initials) for t in range(10)] for s in SUITES} for c in conditions}
    descriptive={}
    for c in conditions:
        rows=[r for k,r in records.items() if k[0]==c];good=all(type(r['actions']) is int and r['actions']>0 and isinstance(r['logged_policy_ms'],(int,float)) and r['logged_policy_ms']>=0 for r in rows)
        descriptive[c]=dict(failure_reasons=dict(collections.Counter(str(r['failure_reason']) for r in rows if not r['success'])),
            logged_action_limits=sorted({r['max_steps'] for r in rows}),
            logged_policy_ms_per_executed_action=sum(r['logged_policy_ms'] for r in rows)/sum(r['actions'] for r in rows) if good else None,
            timing_scope='Originalepisode_log.policy_time_sum_ms field only;no native-eventor packing scope certification,no same-inputcostclaim')
    return counts,records,dict(files=files,manifests=manifest_pins),descriptive


def direction(counts,frozen):
    a,b,s=frozen['better'],frozen['worse'],frozen['suite'];d=[x-y for x,y in zip(counts[a][s],counts[b][s])];total=sum(d)
    extreme=sum(sum(sign*x for sign,x in zip(signs,d))>=total for signs in itertools.product((-1,1),repeat=10))
    draws=np.random.default_rng(20261115).choice(np.asarray(d,np.int64),size=(20000,10),replace=True).sum(axis=1)/100
    return dict(frozen_direction=frozen,difference_count=total,difference=total/100,task_difference_counts=d,
        positive_tasks=sum(x>0 for x in d),exploratory_task_signflip_p=extreme/1024,
        exploratory_task_bootstrap=dict(low=float(np.quantile(draws,.025,method='linear')),high=float(np.quantile(draws,.975,method='linear')),seed=20261115,replicates=20000),
        every_task_LOO_positive=all(total-x>0 for x in d),inferential_status='Retrospective sensitivity only;notcurrentR5ornewconfirmatoryevidence')


def run():
    preread=read(REPORT/'analysis_prerun.json');protocol_path=REPORT/'protocol_v2.json'
    require(sha(protocol_path)==preread['protocol']['sha256'] and all(sha(ROOT/name)==pin for name,pin in preread['source_pins'].items()),'Frozenexplorationplanorsourcechanged')
    require(all(sha(path)==pin for path,pin in preread['native_manifest_pins'].items()),'Legacy native manifest changed')
    target=REPORT/'completed_exploration.json'
    if target.exists():return read(target)
    protocol=read(protocol_path);counts,records,evidence,timing=load(protocol,protocol['discovery_initials'],protocol['variants'])
    discovery=dict(status='complete_historical_discovery',initials=protocol['discovery_initials'],cases=len(records),success_counts_by_task=counts,
        logged_timing_and_failures=timing,evidence=evidence,protocol=preread['protocol'],current720_used=False)
    atomic_json(REPORT/'discovery.json',discovery)
    selected=select(counts);selection_path=REPORT/'selection.json'
    if selection_path.exists():
        choice=read(selection_path);payload=dict(choice);declared=payload.pop('selection_hash');require(digest(payload)==declared and choice['selection']==selected,'Priorlegacyselectionchanged')
    else:
        choice=dict(revision='legacy-r12-selection-v1',time=time.time(),protocol=preread['protocol'],
            discovery=dict(path=str(REPORT/'discovery.json'),sha256=sha(REPORT/'discovery.json')),selection=selected,verification_quality_read=False)
        choice['selection_hash']=digest(choice);atomic_json(selection_path,choice)
    verification=None
    if selected['status']=='selected':
        fixed=selected['selected'];conditions=['Full18',*fixed['paths']]
        vc,vr,ve,vt=load(protocol,protocol['verification_initials'],conditions)
        directions=[direction(vc,d) for d in fixed['directions']];order=sorted(range(2),key=lambda i:(directions[i]['exploratory_task_signflip_p'],i));previous=0.
        for rank,i in enumerate(order):previous=max(previous,min(1.,(2-rank)*directions[i]['exploratory_task_signflip_p']));directions[i]['exploratory_Holm2_p']=previous
        totals={c:sum(sum(v) for v in suites.values()) for c,suites in vc.items()}
        qualifies={c:totals[c]>=300 and totals['Full18']-totals[c]<=60 for c in fixed['paths']}
        pattern=all(d['difference_count']>0 and d['exploratory_Holm2_p']<.05 and d['exploratory_task_bootstrap']['low']>0 and d['every_task_LOO_positive'] for d in directions) and all(qualifies.values())
        verification=dict(status='complete_retrospective_sensitivity',cases=len(vr),success_counts=totals,success_counts_by_task=vc,directions=directions,
            candidate_quality=qualifies,pattern_persisted_under_descriptive_tests=pattern,evidence=ve,logged_timing_and_failures=vt,
            selection_hash=choice['selection_hash'],not_new_independent_confirmation=True,current720_quality_not_used=True)
        atomic_json(REPORT/'retrospective_verification.json',verification)
    result=dict(status='complete_historical_exploration',time=time.time(),discovery_cases=len(records),selection_status=selected['status'],
        selected=selected['selected'],quality_eligible_paths=selected['quality_eligible_paths'],discovery_success_counts=selected['success_counts'],
        verification_summary=None if verification is None else {k:verification[k] for k in ['cases','success_counts','directions','candidate_quality','pattern_persisted_under_descriptive_tests']},
        artifacts={str(p):sha(p) for p in [protocol_path,REPORT/'analysis_prerun.json',REPORT/'discovery.json',selection_path]+([REPORT/'retrospective_verification.json'] if verification else [])},
        current720_parameters_or_outcomes_used=False,new_GPU_model_or_environment_started=False,new_quality_cases=0,
        independent_fastpath_evidence=False,goal_complete=False,
        scope='OldR12candidatehints only. These results may motivate new paired-noiseR5 experiments,butcannotreplacecurrent720registrationorprovethegoal.')
    atomic_json(target,result);return result


if __name__=='__main__':
    result=run();print(json.dumps({k:v for k,v in result.items() if k!='artifacts'}))

"""CPU-only frozen historical FFN importance search; no success/quality loading."""
import hashlib
import io
import itertools
import json
from pathlib import Path
import time
import zipfile
import numpy as np
REPORT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def need(v,msg):
    if not v:raise ValueError(msg)
def save(p,v):
    tmp=Path(str(p)+'.tmp');tmp.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');tmp.replace(p)
MASKS=[m for m in itertools.combinations(range(17),6) if all(b-a>1 for a,b in zip(m,m[1:]))]
INDICATOR=np.asarray([[int(i in m) for i in range(17)] for m in MASKS],np.float64)

def recipe(table,suites,tasks):
    means=np.asarray([np.mean([table[s,t] for t in tasks],axis=0) for s in suites])
    costs=means@INDICATOR.T;pool=costs.mean(axis=0);minimum=float(pool.min())
    eligible=np.flatnonzero(pool<=minimum*1.25+1e-15)
    selected={s:int(min(eligible,key=lambda i:(costs[j,i],MASKS[i]))) for j,s in enumerate(suites)}
    return selected,dict(minimum_pooled_score=minimum,eligible_masks=len(eligible),mask_count=len(MASKS))

def preference(table,suite,tasks,a,b):
    # Smaller scores are preferred. All physical initials already stay in task.
    better=np.asarray([table[suite,t]@INDICATOR[a] for t in tasks]);worse=np.asarray([table[suite,t]@INDICATOR[b] for t in tasks])
    differences=worse-better;total=float(differences.sum());denom=float(worse.mean())
    relative=float(differences.mean())/denom if denom>0 else 0.
    return dict(suite=suite,better_mask=[i+1 for i in MASKS[a]],worse_mask=[i+1 for i in MASKS[b]],
        per_task_score_differences=[float(v) for v in differences],mean_score_difference=float(differences.mean()),relative_gap=relative,
        positive_tasks=int((differences>0).sum()),every_LOO_positive=all(total-float(v)>0 for v in differences))

def analyze():
    need(not (REPORT/'analysis.json').exists(),'Do not replace an existing proxy decision')
    protocol=read(REPORT/'protocol.json');inventory=read(REPORT/'inventory.json')
    need(sha(REPORT/'inventory.json')==protocol['inventory']['sha256'],'Inventory changed')
    need(all(sha(p)==h for p,h in protocol['sources'].items()),'Frozen search source changed')
    need(sha(inventory['source_manifest']['path'])==inventory['source_manifest']['sha256'],'Old manifest changed')
    raw={};member_evidence=[]
    for row in inventory['cases']:
        need(sha(row['identity_path'])==row['identity_sha256'],'Old profile identity changed')
        with zipfile.ZipFile(row['arrays_path']) as z:data=z.read(row['selected_member'])
        need(hashlib.sha256(data).hexdigest()==row['selected_member_sha256'],'Selected score member changed')
        scores=np.load(io.BytesIO(data),allow_pickle=False).astype(np.float64)
        need(scores.shape==(17,) and np.isfinite(scores).all() and (scores>=0).all() and scores.sum()>0,'Invalid score vector')
        c=row['case'];raw[c['suite'],c['task'],c['init']]=scores/scores.sum()
        member_evidence.append(dict(case=c,selected_member_sha256=row['selected_member_sha256']))
    suites=sorted({s for s,t,i in raw});tasks=[1,3,5,7,9]
    need(len(raw)==40,'Not all40 profiles')
    table={(s,t):np.mean([raw[s,t,i] for i in (19,20)],axis=0) for s in suites for t in tasks}
    masks,search=recipe(table,suites,tasks);crossings=[];comparisons=[]
    for x,y in itertools.combinations(suites,2):
        a,b=masks[x],masks[y];directions=[preference(table,x,tasks,a,b),preference(table,y,tasks,b,a)]
        eligible=a!=b and all(d['relative_gap']>=.10 and d['positive_tasks']>=4 and d['every_LOO_positive'] for d in directions)
        candidate=dict(target_suites=[x,y],masks=[[i+1 for i in MASKS[v]] for v in (a,b)],directions=directions,proxy_eligible=eligible)
        comparisons.append(candidate)
        if eligible:crossings.append(candidate)
    crossings.sort(key=lambda c:(-min(d['relative_gap'] for d in c['directions']),-sum(d['relative_gap'] for d in c['directions']),c['target_suites'],c['masks']))
    selected=crossings[0] if crossings else None;cv=[];qualified=False
    if selected:
        x,y=selected['target_suites']
        for task in tasks:
            fitted,info=recipe(table,suites,[t for t in tasks if t!=task])
            dx=float(table[x,task]@(INDICATOR[fitted[y]]-INDICATOR[fitted[x]]))
            dy=float(table[y,task]@(INDICATOR[fitted[x]]-INDICATOR[fitted[y]]))
            cv.append(dict(heldout_task=task,score_differences=[dx,dy],training_masks=[[i+1 for i in MASKS[fitted[s]]] for s in (x,y)]))
        qualified=all(sum(r['score_differences'][i]>0 for r in cv)>=4 for i in range(2))
    # Separate loop recomputes the full-data recipe by explicit scalar sums.
    brute_pool=[np.mean([sum(float(table[s,t][i]) for i in mask) for s in suites for t in tasks]) for mask in MASKS]
    good=[i for i,p in enumerate(brute_pool) if p<=min(brute_pool)*1.25+1e-15]
    brute={s:min(good,key=lambda j:(np.mean([sum(float(table[s,t][i]) for i in MASKS[j]) for t in tasks]),MASKS[j])) for s in suites}
    need(brute==masks,'Independent scalar recipe differs')
    value=dict(time=time.time(),status='proxy_candidate_proposed' if qualified else 'no_proxy_candidate',
        records=40,task_profiles=20,source_score_members=member_evidence,
        suite_chosen_masks={s:[i+1 for i in MASKS[i]] for s,i in masks.items()},search=search,comparisons=comparisons,
        eligible_proxy_crossings=len(crossings),selected_full_data_pair=selected,cross_validation=cv,
        proposed_pair=selected if qualified else None,independent_scalar_recipe_matches=True,
        protocol=dict(path=str(REPORT/'protocol.json'),sha256=sha(REPORT/'protocol.json')),
        normalized_layer_profiles=[dict(suite=s,task=t,scores=table[s,t].tolist()) for s in suites for t in tasks],
        success_files_read=False,raw_action_arrays_read=False,current_C7_outcomes_read=False,new_GPU_jobs_started=False,
        actual_quality_or_complementarity_evidence=False,scope='Candidate search using old initial-Full Jacobian profiles only; no performance inference or actual path replication.')
    save(REPORT/'analysis.json',value)
    print(json.dumps({k:value[k] for k in ['status','records','task_profiles','suite_chosen_masks','search','eligible_proxy_crossings','independent_scalar_recipe_matches']}))
if __name__=='__main__':analyze()

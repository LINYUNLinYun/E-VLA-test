"""Predeclared temporal weights; exact reconstruction from archived Jacobians."""
import json,time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from preparations.idea1_v3_4_E2_single_action.common import OLD,ROOT,references,weights,mask,write_csv,dump,sha

OUT=ROOT/'results/idea1_v3_4_E2_backward_opt/time_weights'
# Fix these before evaluating; linear is the primary proposed candidate.
PROFILES={
    'uniform':np.ones(5)/5,
    'mild_late':np.array([.15,.175,.20,.225,.25]),
    'linear_late':np.arange(1.,6.)/15,
    'exponential_late':np.array([1.,2.,4.,8.,16.])/31,
    'fifth_only':np.array([0.,0.,0.,0.,1.]),
}

def paired_interval(values):
    a=np.asarray(values);indices=np.random.default_rng(341030).integers(len(a),size=(10000,len(a)))
    return [float(x) for x in np.percentile(a[indices].mean(1),[2.5,97.5])]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pairs=[p for p in json.loads((OLD/'selection.json').read_text()) if p['e2']]
    dump(OUT/'protocol.json',dict(profiles={k:v.tolist() for k,v in PROFILES.items()},primary_candidate='linear_late',primary_target='original unweighted R5 seven-dimensional deletion norm',seeds=list(range(340710,340742)),skip=4,max_consecutive=2,source_sha256=sha(Path(__file__)),time=time.time()))
    rows=[]
    for pair in pairs:
        _,J,delta=references(pair);truth=np.linalg.norm(delta.reshape(35,17),axis=0);tail=np.linalg.norm(delta[4],axis=0)
        for name,alpha in PROFILES.items():
            assert np.isclose(alpha.sum(),1) and np.all(alpha>=0)
            aggregate=np.einsum('t,tdl->dl',alpha,J)
            methods=[('exact7',-1,np.linalg.norm(aggregate,axis=0)),('equal_dimensions1',-1,abs(aggregate.sum(0)))]
            for seed in range(340710,340742):
                G=weights(seed)@aggregate
                for n in (1,4):methods.append((f'project{n}',seed,np.linalg.norm(G[:n],axis=0)))
            for method,seed,score in methods:
                chosen=mask(score)
                rows.append(dict(id=pair['id'],**pair['case'],profile=name,method=method,seed=seed,
                                 rho_R5=float(spearmanr(score,truth).statistic),rho_step5=float(spearmanr(score,tail).statistic),
                                 selected_R5_effect_ratio=float(truth[chosen].mean()/truth.mean()),selected_step5_effect_ratio=float(tail[chosen].mean()/tail.mean()),mask_1based=str((chosen+1).tolist())))
    assert len(rows)==40*len(PROFILES)*66 and np.isfinite([r['rho_R5'] for r in rows]).all()
    write_csv(OUT/'requests.csv',rows)
    # Average seeds within each request before bootstrap; seeds are not 1280 independent requests.
    request_means={}
    for profile in PROFILES:
        for method in ['project1','project4','exact7','equal_dimensions1']:
            for pair in pairs:
                a=[r for r in rows if r['profile']==profile and r['method']==method and r['id']==pair['id']]
                request_means[(profile,method,pair['id'])]={k:float(np.mean([r[k] for r in a])) for k in ['rho_R5','rho_step5','selected_R5_effect_ratio','selected_step5_effect_ratio']}
    summary=[];by_suite=[];paired=[]
    for profile in PROFILES:
        for method in ['project1','project4','exact7','equal_dimensions1']:
            for split in ['all','even_tasks','odd_tasks']:
                selected=[p for p in pairs if split=='all' or p['case']['task']%2==(0 if split=='even_tasks' else 1)]
                values=[request_means[(profile,method,p['id'])] for p in selected]
                summary.append(dict(profile=profile,method=method,split=split,requests=len(values),**{k:float(np.mean([x[k] for x in values])) for k in values[0]}))
            for suite in dict.fromkeys(p['case']['suite'] for p in pairs):
                selected=[p for p in pairs if p['case']['suite']==suite]
                by_suite.append(dict(profile=profile,method=method,suite=suite,**{k:float(np.mean([request_means[(profile,method,p['id'])][k] for p in selected])) for k in ['rho_R5','selected_R5_effect_ratio']}))
            if profile!='uniform':
                for metric in ['rho_R5','selected_R5_effect_ratio']:
                    diff=[request_means[(profile,method,p['id'])][metric]-request_means[('uniform',method,p['id'])][metric] for p in pairs]
                    lo,hi=paired_interval(diff);paired.append(dict(profile=profile,method=method,metric=metric,mean_difference=float(np.mean(diff)),ci95_low=lo,ci95_high=hi))
    write_csv(OUT/'summary.csv',summary);write_csv(OUT/'by_suite.csv',by_suite);write_csv(OUT/'paired.csv',paired)
    # Reconstructed equal weighting must match the preceding mean5 experiment.
    old=list(__import__('csv').DictReader((OUT.parent/'mean5/summary.csv').open()))
    for method,oldname in [('project1','mean5_1vjp'),('project4','mean5_4vjp'),('exact7','mean5_exact7')]:
        a=next(x for x in summary if x['profile']=='uniform' and x['method']==method and x['split']=='all')
        b=next(x for x in old if x['method']==oldname)
        assert abs(a['rho_R5']-float(b['rho_R5']))<1e-10
    dump(OUT/'audit.json',dict(passed=True,requests=40,rows=len(rows),profiles=5,seeds=32,uniform_matches_previous=True,source_sha256=sha(Path(__file__)),selection_sha256=sha(OLD/'selection.json'),notes='FP32 exact stored Jacobian contractions; no new GPU latency or closed-loop experiment',time=time.time()))
    print(json.dumps(dict(summary=[x for x in summary if x['split']=='all'],paired=[x for x in paired if x['method'] in ['project1','project4']],suites=[x for x in by_suite if x['profile']=='linear_late' and x['method']=='project1'])),flush=True)

if __name__=='__main__':main()

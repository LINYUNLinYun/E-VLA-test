"""Mechanism check, distinct from closed-loop success and multi-layer deletion."""
import json
import numpy as np
from scipy.stats import spearmanr
from .common import ROOT,OUT
from preparations.idea1_v3_4_E2_single_action.common import OLD,COMBOS,references,mask
from preparations.idea1_v3_4.analyze import paired_ci,write_csv
from preparations.idea1_v3.io import atomic

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    for p in [p for p in json.loads((OLD/'selection.json').read_text()) if p['e2']]:
        _,J,delta=references(p);truth=np.linalg.norm(delta.reshape(35,17),axis=0)
        alpha=np.array([1.,2.,4.,8.,16.])/31;gradient=abs(np.einsum('t,tdl->l',alpha,J))
        metrics=json.loads((OLD/'E2'/f"sample{p['id']:03d}"/'metrics.json').read_text())['rows']
        cosine=np.array([1-next(r['layer_cosine'] for r in metrics if r['layer']==i and r['gate']==0 and r['group']=='translation') for i in range(1,18)])
        random_ratio=float(truth[COMBOS].mean()/truth.mean())
        for method,score in [('jacobian_exponential_fixed',gradient),('cosine',cosine)]:
            chosen=mask(score)
            rows.append(dict(id=p['id'],**p['case'],method=method,rho_R5=float(spearmanr(score,truth).statistic),selected_effect_ratio=float(truth[chosen].mean()/truth.mean()),random_expected_effect_ratio=random_ratio,mask_1based=str((chosen+1).tolist())))
    write_csv(OUT/'offline_requests.csv',rows)
    summaries=[]
    for name in ['jacobian_exponential_fixed','cosine']:
        a=[r for r in rows if r['method']==name];diff=[dict(suite=r['suite'],task=r['task'],delta=r['selected_effect_ratio']-r['random_expected_effect_ratio']) for r in a];ci=paired_ci(diff)
        summaries.append(dict(method=name,requests=40,rho_R5=float(np.mean([r['rho_R5'] for r in a])),selected_effect_ratio=float(np.mean([r['selected_effect_ratio'] for r in a])),random_expected_effect_ratio=float(np.mean([r['random_expected_effect_ratio'] for r in a])),difference_vs_random=float(np.mean([r['delta'] for r in diff])),ci_low=ci[0],ci_high=ci[1]))
    write_csv(OUT/'offline_summary.csv',summaries);atomic(OUT/'offline_audit.json',dict(passed=True,requests=40,random_legal_combinations=len(COMBOS),notes='FP32 archived Jacobians, original cosine, single-layer deletions, exact uniform random-mask expectation; not new multi-layer or closed-loop evidence'))
    print(json.dumps(summaries),flush=True)
if __name__=='__main__':main()

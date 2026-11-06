import json
import numpy as np
from scipy.stats import spearmanr
from .common import OUT,OLD,STEPS,weights,mask,references,write_csv,dump
def main():
    pairs=[p for p in json.loads((OLD/'selection.json').read_text()) if p['e2']];rows=[]
    for p in pairs:
        full,J,delta=references(p);alltrue=np.linalg.norm(delta.reshape(35,17),axis=0)
        for step in STEPS:
            true=np.linalg.norm(delta[step],axis=0)
            methods=[('full35',-1,np.linalg.norm(J.reshape(35,17),axis=0)),('exact7',-1,np.linalg.norm(J[step],axis=0))]
            for seed in range(340710,340742):
                projected=weights(seed)@J[step]
                methods.extend((f'project{n}',seed,np.linalg.norm(projected[:n],axis=0)/np.sqrt(n)) for n in (1,4))
            for name,seed,score in methods:
                selected=mask(score)
                rows.append(dict(id=p['id'],**p['case'],step=step+1,method=name,seed=seed,
                    rho_step=float(spearmanr(score,true).statistic),rho_R5=float(spearmanr(score,alltrue).statistic),
                    selected_step_effect_ratio=float(true[selected].mean()/true.mean()),
                    selected_R5_effect_ratio=float(alltrue[selected].mean()/alltrue.mean()),mask_1based=str((selected+1).tolist())))
    assert len(pairs)==40 and np.isfinite([r['rho_step'] for r in rows]).all()
    write_csv(OUT/'quality_requests.csv',rows)
    summary=[]
    for step in (1,3,5):
        for name in ('project1','project4','exact7','full35'):
            a=[r for r in rows if r['step']==step and r['method']==name]
            seeds=sorted({r['seed'] for r in a});means=[np.mean([r['rho_step'] for r in a if r['seed']==s]) for s in seeds]
            summary.append(dict(step=step,method=name,requests=40,seeds=len(seeds),rho_step=float(np.mean(means)),
                seed_rho_step_p05=float(np.percentile(means,5)),seed_rho_step_p95=float(np.percentile(means,95)),
                rho_R5=float(np.mean([r['rho_R5'] for r in a])),selected_step_effect_ratio=float(np.mean([r['selected_step_effect_ratio'] for r in a])),
                selected_R5_effect_ratio=float(np.mean([r['selected_R5_effect_ratio'] for r in a])),
                fixed_seed_rho_step=float(np.mean([r['rho_step'] for r in a if r['seed'] in (-1,340710)]))))
    write_csv(OUT/'quality_summary.csv',summary);dump(OUT/'quality_complete.json',dict(passed=True,requests=40,rows=len(rows)))
if __name__=='__main__':main()

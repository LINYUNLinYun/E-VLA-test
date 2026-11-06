"""CPU comparison of averaging five actions versus independent output projections."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from preparations.idea1_v3_4_E2_single_action.common import OLD,ROOT,references,weights,mask,write_csv,dump,sha

OUT=ROOT/'results/idea1_v3_4_E2_backward_opt/mean5'

def main():
    OUT.mkdir(parents=True,exist_ok=True);rows=[]
    pairs=[p for p in json.loads((OLD/'selection.json').read_text()) if p['e2']]
    for pair in pairs:
        _,J,delta=references(pair);truth=np.linalg.norm(delta.reshape(35,17),axis=0)
        methods=[('full35_exact',-1,np.linalg.norm(J.reshape(35,17),axis=0)),
                 ('mean5_exact7',-1,np.linalg.norm(J.mean(0),axis=0)),
                 ('mean5_equal_dimensions_1vjp',-1,abs(J.mean(0).sum(0)))]
        for seed in range(340710,340742):
            R=weights(seed);R35=np.random.default_rng(seed).choice([-1.,1.],(4,35))
            for name,G in [('first',R@J[0]),('mean5',R@J.mean(0)),('all5_independent',R35@J.reshape(35,17))]:
                for n in (1,4):methods.append((f'{name}_{n}vjp',seed,np.linalg.norm(G[:n],axis=0)))
        for name,seed,score in methods:
            chosen=mask(score)
            rows.append(dict(id=pair['id'],method=name,seed=seed,rho_R5=float(spearmanr(score,truth).statistic),selected_R5_effect_ratio=float(truth[chosen].mean()/truth.mean()),mask_1based=str((chosen+1).tolist())))
    summary=[]
    for name in dict.fromkeys(x['method'] for x in rows):
        a=[x for x in rows if x['method']==name];seeds=sorted(set(x['seed'] for x in a));v=[np.mean([x['rho_R5'] for x in a if x['seed']==seed]) for seed in seeds]
        summary.append(dict(method=name,requests=40,seeds=len(seeds),rho_R5=float(np.mean(v)),seed_rho_p05=float(np.percentile(v,5)),seed_rho_p95=float(np.percentile(v,95)),selected_R5_effect_ratio=float(np.mean([x['selected_R5_effect_ratio'] for x in a]))))
    assert len(pairs)==40 and np.isfinite([x['rho_R5'] for x in rows]).all()
    write_csv(OUT/'requests.csv',rows);write_csv(OUT/'summary.csv',summary)
    dump(OUT/'audit.json',dict(passed=True,source_sha256=sha(Path(__file__)),selection_sha256=sha(OLD/'selection.json'),requests=40,rows=len(rows),notes='FP32 archived exact Jacobians; no new GPU latency or closed-loop measurement'))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()

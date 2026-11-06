import csv,json
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_selection_async.runtime import OUT,REFERENCE
from preparations.idea1_v3_4_selection_async.core import choose
from preparations.idea1_v3_4_selection_async.common import write_csv


def run():
    p=OUT/'elementwise_batch_probe';summary=json.loads((p/'complete.json').read_text())
    checks=[]
    with np.load(p/'gradients.npz') as data,np.load(REFERENCE) as ref:
        for method in ('rms','rms_rope','rms_rope_glu'):
            for i in range(8):
                raw=data[f'{method}_raw_{i}'];g=data[f'{method}_gradient_{i}'];expected=ref[f'serial_gradient_{i}']
                mask,layers,scores=choose(g,6);em,el,es=choose(expected,6)
                checks.append(dict(method=method,sample=i,raw_max_abs=float(np.max(np.abs(raw-ref[f'serial_raw_{i}']))),
                    gradient_max_abs=float(np.max(np.abs(g-expected))),
                    gradient_passed=bool(np.allclose(g,expected,atol=1e-4,rtol=1e-3)),
                    layer_score_max_abs=float(np.max(np.abs(layers-el))),
                    layer_score_passed=bool(np.allclose(layers,el,atol=1e-4,rtol=1e-3)),
                    v12_mask_passed=mask==em,v14_mask_passed=choose(g,4)[0]==choose(expected,4)[0]))
    prep=list(csv.DictReader((p/'preparation.csv').open()));out=[]
    for r in summary['summary']:
        cc=[c for c in checks if c['method']==r['method']];pp=[c for c in prep if c['method']==r['method']]
        out.append(dict(r,raw_max_abs=max(c['raw_max_abs'] for c in cc),
            gradient_passes=sum(c['gradient_passed'] for c in cc),layer_score_passes=sum(c['layer_score_passed'] for c in cc),
            v14_mask_changes=sum(not c['v14_mask_passed'] for c in cc),
            capture_mean_s=float(np.mean([float(c['capture_s']) for c in pp])),
            peak_allocated_mib=max(float(c['peak_allocated_mib']) for c in pp),
            peak_reserved_mib=max(float(c['peak_reserved_mib']) for c in pp)))
    write_csv(OUT/'fusion_summary.csv',out);atomic(OUT/'fusion_analysis.json',dict(summary=out,checks=checks))
    return out


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False))

"""Projection blind spots, measured at the shared startup Full observation."""
import json
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_4_selection_async.common import OUT,directory,ck,write_csv


def first_gradient(path):
    with np.load(path/'arrays.npz') as archive:
        key=next(k for k in archive.files if k.startswith('gate_gradient_job0_'))
        return archive[key].copy()


def rms(g,mask):return float(np.sqrt(np.mean(g[:,np.asarray(mask)-1].sum(1)**2)))


def run():
    m=json.loads((OUT/'closed_loop_manifest.json').read_text());result=[];summaries=[]
    for case in m['cases']:
        original=directory('v12_p3_q4_serial_async',case);full=first_gradient(original)
        for q in (1,2):
            for label in ('layer','combo'):
                condition=f'v12_p3_q{q}_{label}';d=directory(condition,case);gradient=first_gradient(d)
                rows=[json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()]
                events=[e for r in rows for e in r['selector_events']]
                events+=json.loads((d/'terminal.json').read_text())['selector_events']
                event=next(e for e in events if e['id']==0);mask=event['selected_skip']
                result.append(dict(condition=condition,case=ck(case),q=q,mask=str(mask),
                    own_signed_rms=rms(gradient,mask),heldout_signed_rms=rms(full[q:],mask),
                    full_four_signed_rms=rms(full,mask),
                    first_gradient_max_abs_vs_serial_prefix=float(np.max(np.abs(gradient-full[:q]))),
                    gradient_prefix_passed=bool(np.allclose(gradient,full[:q],atol=1e-4,rtol=1e-3))))
    for condition in sorted({r['condition'] for r in result}):
        group=[r for r in result if r['condition']==condition]
        summaries.append(dict(condition=condition,n=len(group),
            own_signed_rms_mean=float(np.mean([r['own_signed_rms'] for r in group])),
            heldout_signed_rms_mean=float(np.mean([r['heldout_signed_rms'] for r in group])),
            full_four_signed_rms_mean=float(np.mean([r['full_four_signed_rms'] for r in group])),
            gradient_prefix_passes=sum(r['gradient_prefix_passed'] for r in group)))
    write_csv(OUT/'projection_blindspots.csv',result);write_csv(OUT/'projection_blindspots_summary.csv',summaries)
    data=dict(summary=summaries,note='Held-out frozen projections are an additional first-order check, not nonlinear action-error truth.')
    atomic(OUT/'projection_blindspots.json',data);return data


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False))

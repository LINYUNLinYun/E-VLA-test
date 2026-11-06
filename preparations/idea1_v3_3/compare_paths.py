"""Outcome/action/path equivalence on every common real case, no simulation."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from .analyze import OUT,records,summary


def run(left='E4_tau0.8',right='E4_periodic'):
    groups=records()
    def key(r):return tuple(r['case'][k] for k in ('suite','task','init'))
    a={key(r):r for r in groups[left]};b={key(r):r for r in groups[right]};checks=[];verifies=0;full_accept=0
    for case in sorted(a.keys()&b.keys()):
        x,y=a[case],b[case];p,q=Path(x['_directory']),Path(y['_directory'])
        i=[json.loads(l) for l in (p/'infer.jsonl').read_text().splitlines()];j=[json.loads(l) for l in (q/'infer.jsonl').read_text().splitlines()]
        for r in i:
            if 'distance' in r['route']:
                verifies+=1;full_accept+=r['route'].get('accepted_length')==10
        with np.load(p/'arrays.npz') as aa,np.load(q/'arrays.npz') as bb:
            fields=sorted(k for k in aa.files if k.startswith('actions_'));other=sorted(k for k in bb.files if k.startswith('actions_'))
            proposed=fields==other and all(np.array_equal(aa[k],bb[k]) for k in fields)
            executed=len(i)==len(j) and all(r['executed_length']==s['executed_length'] and np.array_equal(aa[f'actions_{n}'][:r['executed_length']],bb[f'actions_{n}'][:s['executed_length']]) for n,(r,s) in enumerate(zip(i,j)))
        rhythm=lambda logs:[('full' if r['route']['path']=='full' else 'draft',r['executed_length']) for r in logs]
        checks.append(dict(case=list(case),proposal_arrays_exact=proposed,executed_prefixes_exact=executed,Full_draft_rhythm_exact=rhythm(i)==rhythm(j),success_equal=x['success']==y['success'],
            left_result_sha256=hashlib.sha256((p/'result.json').read_bytes()).hexdigest(),right_result_sha256=hashlib.sha256((q/'result.json').read_bytes()).hexdigest(),
            left_arrays_sha256=hashlib.sha256((p/'arrays.npz').read_bytes()).hexdigest(),right_arrays_sha256=hashlib.sha256((q/'arrays.npz').read_bytes()).hexdigest()))
    ls=summary(list(a.values()));rs=summary(list(b.values()))
    result=dict(time=time.time(),left=left,right=right,n=len(checks),left_n=len(a),right_n=len(b),
        proposal_all_exact=sum(c['proposal_arrays_exact'] for c in checks),executed_all_exact=sum(c['executed_prefixes_exact'] for c in checks),rhythm_all_exact=sum(c['Full_draft_rhythm_exact'] for c in checks),outcomes_equal=sum(c['success_equal'] for c in checks),
        verified_draft_proposals=verifies,verified_full10_acceptances=full_accept,checks=checks,
        left_observed_ms_per_action=ls['ms_per_action'],right_observed_ms_per_action=rs['ms_per_action'],
        cost_scope='Original complete closed-loop wall/action observations. Multi-GPU cost must additionally be stratified by physical GPU; no bank or engineering wall substituted.')
    atomic(OUT/'E4_tau08_periodic_all_shared_comparison.json',result)
    return result


if __name__=='__main__':run()

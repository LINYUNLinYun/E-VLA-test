"""Explicit handling of one non-reproducible, pre-scoring observation drift.

No rollout, numerical tolerance, original outcome, or frozen source is changed.
The affected pair is excluded from the primary paired comparison, retained in
descriptive totals and in a separately labelled all-40 sensitivity analysis.
"""
import json
import numpy as np
from preparations.idea1_v3_4_E2_comp_error_closedloop.common import OUT

PREFIX_EXCEPTION=('v14_comp_error_extrapolate',('libero_spatial',7,19))

def validate_prefix_exception(name,key,d,od,qs,oq,r,old_result):
    assert (name,key)==PREFIX_EXCEPTION
    assert r['success'] is True and old_result['success'] is True
    with np.load(d/'arrays.npz') as a,np.load(od/'arrays.npz') as b:
        assert all(np.isfinite(a[k]).all() for k in a.files)
        for i in range(4):
            assert qs[i]['input_hash']==oq[i]['input_hash'] and qs[i]['seed']==oq[i]['seed']
            assert np.array_equal(a[f'raw_{i}'],b[f'raw_{i}'])
            assert np.array_equal(a[f'actions_{i}'],b[f'actions_{i}'])
    for i in range(6):assert qs[i]['route']['selected_skip']==oq[i]['route']['selected_skip']
    assert qs[4]['input_hash']!=oq[4]['input_hash']
    a=[json.loads(s) for s in (d/'telemetry.jsonl').read_text().splitlines()][:21]
    b=[json.loads(s) for s in (od/'telemetry.jsonl').read_text().splitlines()][:21]
    assert a==b
    replay=json.loads((OUT/'prefix_replay/comparison.json').read_text())
    assert all(x['input_hash']==x['old_hash'] for rows in replay['records'].values() for x in rows)
    assert all(not x['different_fields'] for x in replay['comparisons'])
    return True

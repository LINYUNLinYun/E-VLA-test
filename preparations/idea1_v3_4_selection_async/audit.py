import json
from pathlib import Path
import numpy as np
from .core import choose


def episode(config,directory):
    d=Path(directory);r=json.loads((d/'result.json').read_text())
    rows=[json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()]
    terminal=json.loads((d/'terminal.json').read_text());events=[];full_steps=[];launches=[]
    assert r['valid'] and len(rows)==r['predictions'] and sum(x['executed_length'] for x in rows)==r['actions']
    assert r['counts']['kalman']==0
    with np.load(d/'arrays.npz') as arrays:
        for i,row in enumerate(rows):
            t=row['route'];assert row['round']==i
            assert t['selection_stride']==config['selection_stride']
            if t['path']=='full':
                full_index=len(full_steps);full_steps.append(row['step'])
                due=config['mode']=='verified' and full_index%config['selection_stride']==0
                assert t['actual_full_index']==full_index and t['mask_refresh']==due
                assert t['steady_vjp_count']==(config['projections'] if due else 0)
                if due:launches.append((i,row['step'],full_index))
                if config['mode']=='verified':assert t['history_full_steps_after']==full_steps[-2:]
            else:
                assert t['path']=='fast_verified' and t['verify_performed'] and t['accepted_length']>0
                assert len(t['draft_skip'])==6 and all(b-a>1 for a,b in zip(t['draft_skip'],t['draft_skip'][1:]))
                assert t['draft_history_full_steps']==full_steps[-2:]
                assert t['captured_layers']==list(range(1,18))
                assert t['draft_residual_age']==row['step']-full_steps[-1]
                if len(full_steps)>1:assert t['draft_full_gap']==full_steps[-1]-full_steps[-2]
                else:assert t['draft_full_gap'] is None and t['draft_extrapolation_ratio']==0.
                assert t['mask_origin_step'] in [s for _,s,_ in launches]
            assert row['executed_length']<=row['execute_length']<=5
            assert row['counts']['full']==len(full_steps) and row['counts']['selector']==len(launches)
            for event in row['selector_events']:events.append((event,f"request{i}"))
        for event in terminal['selector_events']:events.append((event,'terminal'))
        assert len(events)==len(launches)==r['counts']['completed_selector']
        assert sorted(event['id'] for event,_ in events)==list(range(len(launches)))
        for event,suffix in events:
            assert (event['origin_round'],event['origin_step'],event['actual_full_index'])==launches[event['id']]
            g=arrays[f"gate_gradient_job{event['id']}_{suffix}"]
            mask,layers,scores=choose(g,6,config['aggregation'])
            assert mask==event['selected_skip'] and g.shape==(config['projections'],17)
            np.testing.assert_allclose(layers,arrays[f"layer_score_job{event['id']}_{suffix}"],atol=1e-4,rtol=1e-3)
            np.testing.assert_allclose(scores,arrays[f"combination_score_job{event['id']}_{suffix}"],atol=1e-4,rtol=1e-3)
            for row in rows:
                t=row['route']
                if t['draft_attempted'] and t['mask_origin_step']==event['origin_step']:
                    assert t['draft_skip']==mask
        assert r['counts']['vjp']==len(launches)*config['projections']
    return dict(passed=True,condition=config['id'],case=r['case'],requests=len(rows),
        actual_fulls=len(full_steps),selectors=len(events),success=r['success'])

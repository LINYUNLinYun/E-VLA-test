"""Freeze periodic-arrival confirmation after the two capacity runs finish."""
from pathlib import Path
import json
import math

root = Path(__file__).resolve().parent
values = []
for name in ['d10_crossframe_formal', 'd10_crossframe_repeat']:
    path = root/name/'result.json'
    if not path.exists():
        raise SystemExit('Pending capacity result: '+name)
    r = json.loads(path.read_text())
    if r.get('status') != 'passed' or not r.get('paired_performance_eligible'):
        raise SystemExit('Capacity confirmation not passed: '+name)
    values += [g['trimmed_interior']['output_interval_ms']['mean']
               for g in r['groups'] if g['phase']=='paired' and g['mode']=='lookahead_serial']
assert len(values)==200
near = math.ceil(1.05*sum(values)/len(values))
selection = dict(formula='ceil(1.05 * mean of F1 paired-trajectory interior mean intervals across both independent runs))',
                 reference_cases=['d10_crossframe_formal','d10_crossframe_repeat'],
                 reference_trajectories=len(values), reference_interval_ms=sum(values)/len(values),
                 near_capacity_period_ms=near, fixed_periods_ms=[33,62],
                 lookahead_delay_ms=0, lookahead_window_ms=0,
                 budgets=[0,.02,.05,.1], default_budget=.05,
                 semantics='Independent periodic capture clock with latest-available selection; no future-frame wait. Replayed observations, not online robot control.')
path = root/'slo_selection.json'
if path.exists():
    assert json.loads(path.read_text())==selection, 'Selection must not change after freeze'
else:
    path.write_text(json.dumps(selection,indent=2)+'\n')
cases=[]
for repetition in ['formal','repeat']:
    periods = [33,62,near] if repetition=='formal' else [near,62,33]
    modes=['fresh_serial','lookahead_serial','encoder_overlap','combined']
    if repetition=='repeat':
        modes.reverse()
    for period in dict.fromkeys(periods):
        cases.append(dict(name=f'd10_slo_{period}ms_{repetition}',module='preparations.idea2.layer_crossframe_bench',
                          args=['--num-steps','10','--ae-layout','whole','--warmup','20',
                                '--pairs','0','--continuous-count','100','--reset-every','50',
                                '--hoist-modulation','--arrival-period-ms',str(period),
                                '--modes',*modes]))
(root/'slo_confirmation_matrix.json').write_text(json.dumps(cases,indent=2)+'\n')
print(json.dumps(selection,indent=2))

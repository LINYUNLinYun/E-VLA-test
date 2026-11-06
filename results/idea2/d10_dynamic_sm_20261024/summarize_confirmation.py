"""CPU-only summary. Failed, incomplete, profiled cases never enter timing tables."""
from pathlib import Path
import csv
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parents[2] / 'reports/idea2_d10_dynamic_sm_20261024'
OUT.mkdir(exist_ok=True, parents=True)

def dump_csv(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open('w') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)

def read(name):
    path = ROOT / name / 'result.json'
    if not path.exists():
        return None
    r = json.loads(path.read_text())
    if r.get('status') != 'passed' or not r.get('performance_eligible'):
        return None
    return r

def stats(values, prefix):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {}
    return {prefix + '_n': len(a), prefix + '_mean': float(a.mean()),
            prefix + '_p50': float(np.median(a)), prefix + '_p95': float(np.quantile(a, .95))}

def paired(values, prefix):
    a = np.asarray(values, dtype=float)
    result = stats(a, prefix)
    # Pair/trajectory is the unit, not thousands of correlated replay frames.
    rng = np.random.default_rng(20261024)
    means = a[rng.integers(0, len(a), (5000, len(a)))].mean(axis=1)
    result.update({prefix + '_ci95_low': float(np.quantile(means, .025)),
                   prefix + '_ci95_high': float(np.quantile(means, .975))})
    return result

cross_rows, cross_pairs, old_rows, old_pairs, slo_rows, evidence = [], [], [], [], [], []
for name in ['d10_crossframe_formal', 'd10_crossframe_repeat']:
    r = read(name)
    if r is None:
        continue
    groups = [g for g in r['groups'] if g['phase'] == 'paired']
    for mode, summary in r['summary'].items():
        selected = [g for g in groups if g['mode'] == mode]
        cont = next(g for g in r['groups'] if g['mode'] == mode and g['phase'] == 'continuous')
        row = dict(case=name, mode=mode, paired_trajectories=len(selected), frames_per_pair=r['args']['pair_length'],
                   continuous_frames=cont['count'], continuous_all_work_hz=cont['all_work_hz'],
                   continuous_completed_prefills=cont['completed_prefills'],
                   continuous_consumed_snapshots=cont['consumed_snapshots'],
                   peak_allocated_bytes=max(g.get('peak_allocated_bytes', 0) for g in r['groups'] if g['mode'] == mode),
                   peak_reserved_bytes=max(g.get('peak_reserved_bytes', 0) for g in r['groups'] if g['mode'] == mode))
        row.update(stats([g['all_work_trajectory_ms'] for g in selected], 'trajectory_ms'))
        row.update(stats([g['trimmed_interior']['output_interval_ms']['mean'] for g in selected], 'interior_interval_ms'))
        for key, value in summary.items():
            if value:
                row.update({key+'_'+k: value[k] for k in ['mean', 'p50', 'p95']})
        cross_rows.append(row)
    pairs = {}
    for g in groups:
        pairs.setdefault(g['pair'], {})[g['mode']] = g
    for base in ('fresh_serial', 'lookahead_serial', 'encoder_overlap'):
        for mode in r['summary']:
            if mode == base:
                continue
            data = [(x[base]['all_work_trajectory_ms'], x[mode]['all_work_trajectory_ms']) for x in pairs.values()]
            row = dict(case=name, baseline=base, candidate=mode)
            row.update(paired([a/b for a,b in data], 'speedup'))
            row.update(paired([a-b for a,b in data], 'saved_trajectory_ms'))
            cross_pairs.append(row)
    evidence.append(dict(case=name, measurements=len(r['rows']), all_correct=all(x['correct'] for x in r['rows']),
                         max_abs=max(x['normalized_error']['max_abs'] for x in r['rows']),
                         max_mae=max(x['normalized_error']['mae'] for x in r['rows']),
                         recaptures=max(x['graph_recaptures'] for x in r['rows']),
                         lifecycle_passed=r.get('lifecycle_audit', {}).get('passed'),
                         kv_age_set=sorted(set(x['observation_age'] for x in r['rows'])),
                         matching_pairs=all(len(set(g['input_signature_sha256'] for g in x.values()))==1 for x in pairs.values())))

for name in ['d10_oldkv_formal', 'd10_oldkv_repeat', 'd10_async_serial_formal', 'd10_async_serial_repeat']:
    r = read(name)
    if r is None:
        continue
    for measurement in r['rows']:
        ae_start=measurement['ae_completion_event_ms']-measurement['ae_gpu_event_ms']
        prefix_start=measurement['prefill_completion_event_ms']-measurement['prefill_gpu_event_ms']
        measurement['gpu_scheduling_span_ms']=measurement['both_completion_event_ms']-min(ae_start,prefix_start)
    for label, summary in r['summary'].items():
        row = dict(case=name, group=label)
        for key, value in summary.items():
            if isinstance(value, dict) and {'mean','p50','p95'} <= value.keys():
                row.update({key+'_'+k: value[k] for k in ['mean','p50','p95']})
            elif isinstance(value, (float, int, bool, str)):
                row[key] = value
        phase,mode=label.split('/',1)
        row.update(stats([x['gpu_scheduling_span_ms'] for x in r['rows'] if x['phase']==phase and x['mode']==mode], 'gpu_scheduling_span_ms'))
        row['gpu_scheduling_span_definition']='Earliest branch start Event to both done, including dispatch/queue/resource waits; not kernel busy. Both-completion origin also includes Host input preparation.'
        old_rows.append(row)
    pairs = {}
    for row in r['rows']:
        if row['phase'] == 'paired':
            pairs.setdefault((row['group'],row['pair']), {})[row['mode']] = row
    for group in sorted(set(k[0] for k in pairs)):
        members = [x for k,x in pairs.items() if k[0]==group]
        modes = set(members[0])
        base = ('graph_async_serial' if 'graph_async_serial' in modes else
                'graph_serial' if 'graph_serial' in modes else 'graph_ae_first')
        candidate = next(x for x in modes if x != base)
        out = dict(case=name, group=group, baseline=base, candidate=candidate)
        for metric in ['cpu_cpu_ms','both_completion_event_ms','gpu_scheduling_span_ms','ae_gpu_event_ms','prefill_gpu_event_ms']:
            out.update(paired([x[base][metric]-x[candidate][metric] for x in members], metric+'_saved'))
            out.update(stats([x[base][metric] for x in members], 'baseline_'+metric))
            out.update(stats([x[candidate][metric] for x in members], 'candidate_'+metric))
        out.update(paired([x[base]['cpu_cpu_ms']/x[candidate]['cpu_cpu_ms'] for x in members], 'cpu_speedup'))
        old_pairs.append(out)
    evidence.append(dict(case=name, measurements=len(r['rows']), all_correct=all(x['correct'] for x in r['rows']),
                         max_abs=max(x['normalized_error']['max_abs'] for x in r['rows']),
                         max_mae=max(x['normalized_error']['mae'] for x in r['rows']),
                         recaptures=max(x['steady_recapture'] for x in r['rows']),
                         kv_age_set=sorted(set(x['observation_age'] for x in r['rows'])),
                         freshness_all=all(g['freshness_gate']['passed'] for g in r['groups']),
                         same_graph_all=(all(g['same_graph_selective_passed'] for g in r['groups']
                                             if g.get('same_graph_selective_comparison'))
                                         if any(g.get('same_graph_selective_comparison') for g in r['groups'])
                                         else bool(r.get('wrapper')) and all(x.get('graph_engine_ordinal')==1 for x in r['rows'])),
                         wrapper_source_unchanged=r.get('wrapper',{}).get('source_unchanged'),
                         dependency_gates=sorted(set(x['prefix_dependency_gate'] for x in r['rows'] if 'prefix_dependency_gate' in x))))

for path in sorted(ROOT.glob('d10_slo_*/result.json')):
    r = read(path.parent.name)
    if r is None:
        continue
    groups = {g['mode']:g for g in r['groups'] if g['phase']=='continuous'}
    for mode, summary in r['summary'].items():
        g = groups[mode]
        selected = [x for x in r['rows'] if x['phase']=='continuous' and x['mode']==mode]
        row = dict(case=path.parent.name, mode=mode, arrival_period_ms=r['args']['arrival_period_ms'],
                   lookahead_delay_ms=r['args']['lookahead_delay_ms'], frames=g['count'],
                   all_work_hz=g['all_work_hz'], completed_prefills=g['completed_prefills'], consumed_snapshots=g['consumed_snapshots'],
                   capture_p95_ms=summary['capture_to_action_ms']['p95'], capture_mean_ms=summary['capture_to_action_ms']['mean'],
                   kv_source_age_p95_ms=summary['capture_to_action_ms']['p95'],
                   kv_age_definition='Current-frame KV: source capture to action; same milliseconds as capture latency. Camera-frame count is a quantized diagnostic.',
                   camera_age_frames_p95=float(np.quantile([x['camera_age_frames_at_output'] for x in selected],.95)),
                   kv_conditioning_offsets=sorted(set(x['kv_conditioning_offset_frames'] for x in selected)),
                   lookahead_admitted_fraction=sum(bool(x['lookahead_admitted']) for x in selected)/len(selected),
                   recaptures=max(x['graph_recaptures'] for x in selected),
                   max_abs=max(x['normalized_error']['max_abs'] for x in selected))
        row.update(g['arrival_counts'])
        for base in ['fresh_serial','lookahead_serial']:
            c = r['baseline_comparisons'][base][mode]
            row[base+'_throughput_ratio']=c['all_work_hz_ratio']
            row[base+'_capture_p95_ratio']=c['capture_p95_ratio']
            base_age=r['summary'][base]['capture_to_action_ms']['p95']
            for budget in [0,.02,.05,.1]:
                row[base+'_latency_budget_'+str(budget)]=c['capture_p95_budget_checks'][str(budget)]
                row[base+'_age_budget_'+str(budget)]=row['kv_source_age_p95_ms'] <= base_age*(1+budget)
        slo_rows.append(row)

dump_csv('crossframe_formal_summary.csv', cross_rows)
dump_csv('crossframe_formal_paired.csv', cross_pairs)
dump_csv('oldkv_formal_summary.csv', old_rows)
dump_csv('oldkv_formal_paired.csv', old_pairs)
dump_csv('slo_summary.csv', slo_rows)
(OUT/'confirmation_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
print(json.dumps(dict(crossframe_rows=len(cross_rows), oldkv_rows=len(old_rows), slo_rows=len(slo_rows), evidence=evidence),indent=2))

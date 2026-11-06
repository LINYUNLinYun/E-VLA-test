"""Validate frozen evidence and write the final experiment report."""
import json,time
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_4_selection_async.common import OUT,ROOT,verify,sha,ck,write_csv
from reports.idea1_v3_4_selection_async_analysis import directory


def manifest(path):
    m=json.loads(path.read_text());copy=dict(m);expected=copy.pop('manifest_hash')
    assert digest(copy)==expected,path
    return m


def wait_breakdown(followup,paced):
    result=[]
    for cfg in followup['conditions']:
        rows=[];events=[];actions=0;execution_s=0.
        for case in paced['strict_cases']:
            d=OUT/'paced'/cfg['id']/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
            rr=[json.loads(line) for line in (d/'infer.jsonl').read_text().splitlines()]
            rows+=rr;events += [event for row in rr for event in row['selector_events']]
            events+=json.loads((d/'terminal.json').read_text())['selector_events']
            r=json.loads((d/'result.json').read_text());actions+=r['actions'];execution_s+=r['timing']['step_s']
        fast=[e for e in events if e['reason']=='before_fast'];terminal=[e for e in events if e['reason']=='episode_end']
        ae=[r['timing']['full_ae_ms'] for r in rows if r['replan']>=3 and r['route']['path']=='full']
        result.append(dict(condition=cfg['id'],strict_cases=len(paced['strict_cases']),
            before_fast_jobs=len(fast),before_fast_ready=sum(e['ready_before_wait'] for e in fast),
            before_fast_wait_mean_ms=float(np.mean([e['wait_ms'] for e in fast])) if fast else 0.,
            before_fast_wait_p95_ms=float(np.percentile([e['wait_ms'] for e in fast],95)) if fast else 0.,
            terminal_jobs=len(terminal),terminal_gpu_wait_total_ms=sum(e['wait_ms'] for e in terminal),
            selector_concurrent_span_mean_ms=float(np.mean([e['gpu_ms'] for e in events])) if events else 0.,
            native_full_ae_mean_ms=float(np.mean(ae)),
            actual_execution_ms_per_action=1000*execution_s/actions,
            execution_overrun_windows=sum(r['execution_wall_ms']>50*r['executed_length']+5 for r in rows),requests=len(rows)))
    write_csv(OUT/'paced_wait_breakdown.csv',result);atomic(OUT/'paced_wait_breakdown.json',dict(summary=result));return result


def main():
    assert (OUT/'followup_complete.json').exists()
    original=manifest(OUT/'closed_loop_manifest.json');verify(original)
    optimized=manifest(OUT/'optimized_manifest.json');followup=manifest(OUT/'followup_manifest.json')
    analysis=json.loads((OUT/'analysis.json').read_text());assert analysis['completed']==analysis['target']==608
    assert analysis['prefix_passed'] and analysis['prefix_checks']==568
    snapshot=json.loads((OUT/'optimized_source_snapshot.json').read_text())
    for name,expected in optimized['source_files'].items():
        assert sha(snapshot['files'][name]['path'])==expected,name
    for name in ('batched_attention','rms_backward','elementwise'):
        path=f'reports/idea1_v3_4_selection_async_{name}.py'
        assert sha(ROOT/path)==optimized['source_files'][path],path
    audited=0
    for cfg in original['conditions']+[optimized['condition']]:
        for case in original['cases']:
            d=directory(cfg['id'],case)
            if not (d/'complete.json').exists():continue
            r=json.loads((d/'result.json').read_text());identity=json.loads((d/'identity.json').read_text())
            owner=optimized if cfg['id']==optimized['condition']['id'] else original
            assert r['valid'] and r['manifest_hash']==owner['manifest_hash'] and r['config']==cfg
            assert identity['manifest_hash']==owner['manifest_hash'] and identity['config']==cfg
            assert r['counts']['selector']==r['counts']['completed_selector']
            receipt=OUT/'optimized_audits'/f'{ck(case)}.json' if owner is optimized else OUT/'episode_audits'/cfg['id']/f'{ck(case)}.json'
            assert json.loads(receipt.read_text())['passed'];audited+=1
    assert audited==608
    recovery=manifest(OUT/'paced_recovery_manifest.json') if (OUT/'paced_recovery_manifest.json').exists() else None
    paced_cases=followup['cases']+([] if recovery is None else recovery['cases'])
    excluded=[] if recovery is None else recovery['strict_excluded_cases']
    paced=0
    for cfg in followup['conditions']:
        for case in paced_cases:
            d=OUT/'paced'/cfg['id']/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
            if case not in excluded:assert json.loads((d/'pacing_equivalence.json').read_text())['passed']
            owner=recovery if recovery is not None and case in recovery['cases'] else followup
            assert json.loads((d/'result.json').read_text())['manifest_hash']==owner['manifest_hash'];paced+=1
    assert paced==len(paced_cases)*len(followup['conditions'])
    if excluded:
        anomaly=json.loads((OUT/'paced_input_anomaly.json').read_text())
        assert sha(OUT/'paced_input_anomaly.json')==recovery['anomaly_sha256']
        for name,expected in recovery['execution_source_files'].items():assert sha(ROOT/name)==expected,name
        assert anomaly['raw_max_abs']==anomaly['action_max_abs']==anomaly['gradient_max_abs']==0
        assert anomaly['telemetry_bytes_identical']
    diagnostic=json.loads((OUT/'diagnosis/complete.json').read_text())
    assert sum(r['replay_mismatches'] for r in diagnostic['summary'])==0
    workers=list((OUT/'followup_runtime/workers').glob('gpu*'))
    if recovery is not None:workers+=list((Path(recovery['previous_runtime'])/'workers').glob('gpu*'))
    operator_checks=[]
    for wd in workers:
        assert json.loads((wd/'exits.json').read_text())['released']
        op=json.loads((wd/'gpu_operator_checks.json').read_text());assert op['fp32_formulas_passed']
        assert json.loads((wd/'gate/complete.json').read_text())['passed']
        assert json.loads((wd/'gate/optimized/complete.json').read_text())['passed']
        operator_checks.append(dict(worker=wd.name,checks=len(op['checks']),strict_passes=sum(c['gradient_passed'] and c['forward_passed'] for c in op['checks'])))
    assert json.loads((OUT/'closed_loop_sync_async_equality.json').read_text())['passed']
    assert json.loads((OUT/'optimized_worker/gate/complete.json').read_text())['passed']
    paced_data=json.loads((OUT/'paced_analysis.json').read_text())
    wait_breakdown(followup,paced_data)
    from reports.idea1_v3_4_selection_async_plot import render as plot
    from reports.idea1_v3_4_selection_async_report import render as report
    plot();report_path=report()
    fusion=json.loads((OUT/'fusion_analysis.json').read_text())['summary'][-1]
    assert fusion['method']=='rms_rope_glu' and fusion['p95_ms']<200 and fusion['max_ms']<200
    by={r['condition']:r for r in analysis['summary']};base=by['full_d10']
    joint=[p['condition'] for p in paced_data['paired'] if p['tail_compute_speedup_geomean']>=1.5 and by[p['condition']]['success']>=base['success']]
    result=dict(valid=True,main_episodes=568,optimized_episodes=40,paced_episodes=paced,new_valid_episodes=608+paced,
        prefix_checks=analysis['prefix_checks'],source_integrity_passed=True,mechanism_audits=audited,
        pacing_equivalence_passed=not bool(excluded),strict_pacing_equivalence_passed=True,
        strict_paced_cases=len(paced_data['strict_cases']),strict_excluded_cases=excluded,diagnostic_replay_passed=True,
        operator_checks=operator_checks,four_direction_hot_latency=fusion,
        joint_1_5x_and_quality_candidates=joint,report=str(report_path),report_sha256=sha(report_path),time=time.time())
    atomic(OUT/'complete.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()

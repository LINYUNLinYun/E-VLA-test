"""CPU-only progressive-KV evidence tables; failed/excluded cases never ranked."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT.parents[2]/'reports/idea2_progressive_kv_20261025'
OUT.mkdir(parents=True,exist_ok=True)


def stats(values,prefix):
    a=np.asarray([x for x in values if x is not None],float)
    if not len(a):return {}
    a=a[np.isfinite(a)]
    return {prefix+'_n':len(a),prefix+'_mean':float(a.mean()),prefix+'_p50':float(np.median(a)),prefix+'_p95':float(np.quantile(a,.95))}


def write(name,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


summary=[];pairs=[];slo=[];evidence=[];commands=[]
receipts={}
for receipt_path in ROOT.glob('*.execution.json'):
    receipt=json.loads(receipt_path.read_text())
    for case in receipt.get('cases',[]):
        if case.get('status')=='passed' and case.get('command'):
            receipts[case['name']]=case
for path in sorted(ROOT.glob('*/result.json')):
    d=json.loads(path.read_text())
    if d.get('status')!='passed':continue
    name=path.parent.name
    commands.append(dict(case=name,command=receipts.get(name,{}).get('command',d.get('command')),
                         performance_eligible=d.get('performance_eligible',False),result=str(path)))
    if 'slo_' in name:
        for g in d['groups']:
            selected=[r for r in d['rows'] if r['lead']==g['lead'] and r['variant']==g['variant']]
            row=dict(case=name,period_ms=d['args']['arrival_period_ms'],gpu_uuid=d['args']['gpu_uuid'],
                     ae_sm_min=d.get('resource_adapter',{}).get('ae_sm_min',0),**g)
            for metric in ('capture_to_action_ms','admission_to_action_ms','current_kv_source_age_ms','old_kv_source_age_ms',
                           'ae_ms','vlm_ms','exposed_ae_after_vlm_ms','ae_finish_minus_next_encoder_ms'):
                row.update(stats([x.get(metric) for x in selected],metric))
            row['old_bootstrap_requests_excluded']=sum(bool(x.get('old_source_is_bootstrap')) for x in selected)
            row['new_layer_fraction_mean']=float(np.mean([x['frontiers'] for x in selected])/18)
            base=next((s for s in d['summary'].values() if s['lead']==18 and s['variant']=='no_lookahead'),None)
            p95=row['capture_to_action_ms_p95']
            row['capture_p95_vs_fresh_serial_ratio']=p95/base['capture_to_action_ms']['p95'] if base else None
            for budget in [0,.02,.05,.1]:row['capture_budget_'+str(budget)]=(p95<=base['capture_to_action_ms']['p95']*(1+budget)) if base else None
            row['budget_scope']='Within this same-GPU run' if base else 'No unrestricted local baseline; absolute ages only'
            slo.append(row)
        continue
    if d.get('args',{}).get('profile'):continue
    measured=[r for r in d.get('rows',[]) if r['phase']!='verification']
    for lead,variant,phase in sorted(set((r['lead'],r['variant'],r['phase']) for r in measured)):
        selected=[r for r in measured if (r['lead'],r['variant'],r['phase'])==(lead,variant,phase)]
        groups=[g for g in d['groups'] if (g['lead'],g['variant'],g['phase'])==(lead,variant,phase)]
        row=dict(case=name,gpu_uuid=d['args']['gpu_uuid'],lead=lead,variant=variant,phase=phase,requests=len(selected),groups=len(groups),
            eligible=d.get('performance_eligible',False),
            normalized_only=not d.get('performance_driver') and not d.get('resource_adapter'),
            ae_sm_min=d.get('resource_adapter',{}).get('ae_sm_min',0),
            encoder_after_step=d.get('resource_adapter',{}).get('encoder_after_step',1),
            peak_allocated_bytes=d.get('peak_allocated_bytes'),peak_reserved_bytes=d.get('peak_reserved_bytes'))
        row.update(stats([g['all_work_ms'] for g in groups],'trajectory_ms'))
        row.update(stats([g['all_work_hz'] for g in groups],'all_work_hz'))
        for metric in ('cpu_cpu_ms','vision_ms','vlm_ms','ae_ms','exposed_ae_after_vlm_ms','model_after_encoder_ms',
                       'ae_finish_minus_next_encoder_ms','ae_start_after_vlm_start_ms','decode_host_ms'):
            row.update(stats([r.get(metric) for r in selected],metric))
        row.update(stats([r['admission_to_action_ms'] for r in selected],'loop_iteration_host_ms'))
        row['mean_frontiers_by_step']=np.mean([r['frontiers'] for r in selected],axis=0).tolist()
        row['new_layer_fraction_mean']=float(np.mean([r['frontiers'] for r in selected])/18)
        next_rows=[r for r in selected if r.get('next_encoder')]
        row.update(stats([max(0,r['ae_finish_minus_next_encoder_ms']) for r in next_rows],
                         'exposed_ae_after_current_vlm_and_next_encoder_ms'))
        summary.append(row)
    paired_groups={}
    for g in d.get('groups',[]):
        if g['phase']=='paired':paired_groups.setdefault(g['pair'],{})[(g['lead'],g['variant'])]=g
    if paired_groups:
        keys=set.intersection(*(set(x) for x in paired_groups.values()))
        for base in [(18,'host_only'),(18,'encoder')]:
            if base not in keys:continue
            for candidate in sorted(keys-{base}):
                a=np.array([x[base]['all_work_ms'] for x in paired_groups.values()])
                b=np.array([x[candidate]['all_work_ms'] for x in paired_groups.values()])
                rng=np.random.default_rng(20261025)
                ratio=a/b
                ci=ratio[rng.integers(0,len(ratio),(5000,len(ratio)))].mean(axis=1)
                pairs.append(dict(case=name,baseline_lead=base[0],baseline_variant=base[1],lead=candidate[0],variant=candidate[1],
                    paired_groups=len(a),baseline_trajectory_ms_mean=float(a.mean()),candidate_trajectory_ms_mean=float(b.mean()),
                    speedup_mean=float(ratio.mean()),speedup_ci95_low=float(np.quantile(ci,.025)),speedup_ci95_high=float(np.quantile(ci,.975)),
                    saved_trajectory_ms_mean=float((a-b).mean())))
    ver=[r for r in d.get('rows',[]) if 'numerical' in r]
    evidence.append(dict(case=name,source_unchanged=d.get('source_unchanged'),measurements=len(measured),
        verification_requests=len(ver),max_oracle_error=max((r['numerical']['matched_schedule']['max_abs'] for r in ver),default=None),
        max_oracle_mae=max((r['numerical']['matched_schedule']['mae'] for r in ver),default=None),
        max_vs_fresh_error=max((r['numerical']['versus_fresh']['max_abs'] for r in ver),default=None),
        all_frontiers_valid=all(len(r['frontiers'])==10 and r['frontiers']==sorted(r['frontiers']) and min(r['frontiers'])>=r['lead'] for r in measured),
        recaptures=max((r['graph_recaptures'] for r in measured),default=0)))
write('performance_summary.csv',summary);write('paired_trajectories.csv',pairs);write('slo_summary.csv',slo)
(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
(OUT/'reproduction_index.json').write_text(json.dumps(commands,indent=2)+'\n')
lines=['# 已完成实验的实际命令','','重跑前修改输出目录与nsys文件名，重新协调独占GPU。',
       '正式队列固定CPU0–15，闭环GPU0固定CPU32–47。','',
       '```bash','export CUDA_DEVICE_MAX_CONNECTIONS=32','export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2',
       'export PYTHONPATH=.:policies/openpi/src','export OPENPI_DATA_HOME=/root/nfs/embodied/benchmark/shared/checkpoints/openpi',
       'export JAX_PLATFORMS=cpu JAX_PLATFORM_NAME=cpu','```','']
for c in commands:
    if c['command']:lines.extend(['## '+c['case'],'','```bash',c['command'],'```',''])
lines.extend(['## 闭环与离线分析','','```bash',
 'policies/openpi/.venv/bin/python results/idea2/progressive_kv_20261025/closedloop.py --execute --output <新的目录> --gpu-uuid GPU-c7bded27-f806-bb96-48ac-6c3ca5cf20d1 --egl-device 3 --port 19085 --cpus 32-47',
 'policies/openpi/.venv/bin/python results/idea2/progressive_kv_20261025/summarize.py','```',''])
(OUT/'reproduction_commands.md').write_text('\n'.join(lines))
print(json.dumps(dict(performance_rows=len(summary),paired_comparisons=len(pairs),slo_rows=len(slo),completed_cases=len(commands)),indent=2))

import csv,json
from pathlib import Path
root=Path('results/idea2/cewp_20261021');manifest=json.loads((root/'request_profile_manifest.json').read_text());rows=[];checks={};total=0
checks['all_jobs_exit_zero']=all(p['status']=='done' and p['profile_exit_code']==0 for p in manifest)
for p in manifest:
 audit=json.loads((root/p['name']/'audit.json').read_text());result=json.loads((root/p['name']/'result.json').read_text())
 for mode,m in audit['modes'].items():
  prefix=p['name']+'/'+mode;total+=m['total_kernels'];samples=[r for r in result['rows'] if r['mode']==mode];agg=next(iter(m['aggregate'].values()));g=m['global_measured_window'];await_previous=result['args'].get('await_previous_prefix',False)
  checks[prefix+'/correlations_complete']=m['all_kernel_correlations_matched']
  checks[prefix+'/measured_accounted']=m['all_measured_kernels_accounted']
  checks[prefix+'/numerical_exact']=all(r['correct'] and all(r[e]['max_abs']==0 and r[e]['mae']==0 for e in ('normalized_error','decoded_error')) for r in samples)
  checks[prefix+'/no_device_sync']=m['steady_device_synchronize_calls']==0
  checks[prefix+'/no_capture']=m['capture_begin_calls']==0
  checks[prefix+'/pool_max_two']=all(r['pool']['live_snapshots']<=2 for r in samples)
  submissions=sum(r.get('prefill_cpu_submit_ms') is not None for r in samples)
  if 'serial' in mode:checks[prefix+'/serial_no_overlap']=g['overlap_ms']==0
  if await_previous:
   checks[prefix+'/prefill_every_request']=submissions==len(samples)
   checks[prefix+'/age_one']=all(r['observation_age']==1 for r in samples)
   checks[prefix+'/global_ae_vlm_disjoint']=m['global_stage_overlap']['prefill/vlm']['overlap_ms']==0
  rows.append(dict(plan=p['name'],mode=mode,steps=result['args']['steps'],await_previous_prefix=await_previous,hoist_modulation=result['args'].get('hoist_modulation',False),requests=len(samples),prefill_submissions=submissions,max_observation_age=max(r['observation_age'] for r in samples),ae_kernel_calls=m['stage_kernel_counts']['ae'],prefill_kernel_calls=sum(v for k,v in m['stage_kernel_counts'].items() if k.startswith('prefill')),global_overlap_fraction=g['overlap_fraction'],same_request_overlap_fraction=agg['overlap_fraction'],global_ae_vision_overlap_ms=m['global_stage_overlap'].get('prefill/vision',{}).get('overlap_ms'),global_ae_vlm_overlap_ms=m['global_stage_overlap'].get('prefill/vlm',{}).get('overlap_ms'),mean_ae_kernel_busy_ms=agg['modules']['ae']['mean_kernel_busy_ms'],mean_prefill_kernel_busy_ms=agg['modules']['prefill']['mean_kernel_busy_ms'],total_kernels=m['total_kernels'],matched_unique_kernels=m['matched_unique_kernel_rows']))
with (root/'request_trace_comparison.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
receipt=dict(plans=len(manifest),traces=len(rows),total_kernels=total,checks=checks,passed=all(checks.values()),scope='These are profiler diagnostics, not performance measurements. Sparse Prefill work in ungated modes is preserved, not counted as speedup. Await-previous modes require one Prefill per request and age exactly one.',figures=['request_kernel_timeline.png','request_kernel_timeline_best.png'])
(root/'request_trace_delivery.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('plans',receipt['plans'],'traces',receipt['traces'],'kernels',total,'passed',receipt['passed'],'checks',len(checks));print('failed',[k for k,v in checks.items() if not v])

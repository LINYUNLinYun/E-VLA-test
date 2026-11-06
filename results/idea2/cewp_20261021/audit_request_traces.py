"""Audit all kernel correlations, then compare matching launch geometries."""
import json,sqlite3,sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from preparations.idea2.scheduling_trace import duration,intersection
root=Path('results/idea2/cewp_20261021')
manifest=json.loads((root/'request_profile_manifest.json').read_text())
reports=[]
for plan in manifest:
 if plan['status']!='done':continue
 mode_reports={}
 for item in plan['traces']:
  db=sqlite3.connect(item['sqlite']);ranges=db.execute('SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL ORDER BY n.start,n.end DESC').fetchall()
  sections=[r for r in ranges if r[3] and r[3].startswith('cewp/continuous/')];starts=[r[0] for r in sections]
  modules=defaultdict(list)
  for b,e,tid,label in ranges:
   if label in ('ae','prefill','prefill/vision','prefill/vlm'):modules[tid].append((b,e,label))
  stage_intervals=defaultdict(list);signature_stats=defaultdict(list);stage_counts=defaultdict(int);unassigned=defaultdict(int);measured=0;matched=0;dev_context_streams=defaultdict(set)
  query='''SELECT k.start,k.end,k.deviceId,k.contextId,k.streamId,r.start,r.end,r.globalTid,s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId AND k.globalPid=(r.globalTid & -16777216) JOIN StringIds s ON k.demangledName=s.id ORDER BY r.start'''
  for row in db.execute(query):
   matched+=1;b,e,dev,ctx,stream,launch,lend,tid,name,*geom=row
   si=bisect_right(starts,launch)-1
   if si<0 or lend>sections[si][1]:continue
   measured+=1
   module=next((label for mb,me,label in reversed(modules[tid]) if mb<=launch and lend<=me),None)
   if module is None:unassigned[name]+=1;continue
   stage_counts[module]+=1;dev_context_streams[dev,ctx,module].add(stream)
   stage_intervals[sections[si][3],dev,ctx,module].append((b,e,stream))
   signature_stats[(module,name,*geom)].append((e-b)/1000)
  total=db.execute('select count(*) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()[0]
  matched_unique=db.execute('SELECT COUNT(DISTINCT k.rowid) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId AND k.globalPid=(r.globalTid & -16777216)').fetchone()[0]
  summary=json.loads(Path(item['analysis']).read_text())
  stage_overlap={}
  for stage in ('prefill/vision','prefill/vlm'):
   overlap=stage_busy=ae_busy=0
   for (section,dev,ctx,module),values in stage_intervals.items():
    if module!=stage:continue
    aes=stage_intervals.get((section,dev,ctx,'ae'),[])
    if aes:
     assert {r[2] for r in aes}.isdisjoint({r[2] for r in values})
     overlap+=intersection([(r[0],r[1]) for r in aes],[(r[0],r[1]) for r in values])
     ae_busy+=duration([(r[0],r[1]) for r in aes])
    stage_busy+=duration([(r[0],r[1]) for r in values])
   if stage_busy:stage_overlap[stage]=dict(overlap_ms=overlap/1e6,stage_busy_ms=stage_busy/1e6,ae_busy_ms=ae_busy/1e6,overlap_over_stage_busy=overlap/stage_busy)
  window=defaultdict(list);stage_window=defaultdict(list)
  for (section,dev,ctx,module),values in stage_intervals.items():
   window[dev,ctx,'ae' if module=='ae' else 'prefill'].extend(values)
   stage_window[dev,ctx,module].extend(values)
  global_overlap=global_union=0
  for dev,ctx in {(d,c) for d,c,m in window}:
   ae=window.get((dev,ctx,'ae'),[]);pre=window.get((dev,ctx,'prefill'),[])
   assert {r[2] for r in ae}.isdisjoint({r[2] for r in pre})
   global_overlap+=intersection([(r[0],r[1]) for r in ae],[(r[0],r[1]) for r in pre])
   global_union+=duration([(r[0],r[1]) for r in ae+pre])
  global_stage_overlap={}
  for stage in ('prefill/vision','prefill/vlm'):
   overlap=busy=0
   for (dev,ctx,module),rs in stage_window.items():
    if module!=stage:continue
    aes=window.get((dev,ctx,'ae'),[])
    overlap+=intersection([(r[0],r[1]) for r in aes],[(r[0],r[1]) for r in rs]);busy+=duration([(r[0],r[1]) for r in rs])
   if busy:global_stage_overlap[stage]=dict(overlap_ms=overlap/1e6,stage_busy_ms=busy/1e6,overlap_over_stage_busy=overlap/busy)
  global_window=dict(overlap_ms=global_overlap/1e6,kernel_union_ms=global_union/1e6,overlap_fraction=global_overlap/global_union if global_union else 0,scope='All measured AE/Prefill kernels; includes overlap between different request submissions, unlike per-request aggregate')
  stats=[]
  for sig,v in signature_stats.items():stats.append(dict(module=sig[0],name=sig[1],geometry=list(sig[2:]),calls=len(v),mean_us=sum(v)/len(v),total_ms=sum(v)/1000))
  report=dict(mode=item['mode'],sqlite=item['sqlite'],total_kernels=total,matched_launch_kernels=matched,matched_unique_kernel_rows=matched_unique,global_measured_window=global_window,global_stage_overlap=global_stage_overlap,measured_section_kernels=measured,branch_kernels=sum(stage_counts.values()),unassigned_measured_kernel_names=dict(unassigned),stage_kernel_counts=dict(stage_counts),stage_overlap=stage_overlap,device_context_streams=[dict(device=d,context=c,module=m,streams=sorted(s)) for (d,c,m),s in dev_context_streams.items()],aggregate=summary['aggregate'],kernel_signatures=stats,cuda_api=summary['cuda_api'])
  report['all_kernel_correlations_matched']=total==matched==matched_unique
  report['all_measured_kernels_accounted']=measured==sum(stage_counts.values())+sum(unassigned.values())
  report['steady_device_synchronize_calls']=sum(x['calls'] for x in summary['cuda_api'] if 'DeviceSynchronize' in x['name'])
  report['capture_begin_calls']=sum(x['calls'] for x in summary['cuda_api'] if 'BeginCapture' in x['name'])
  mode_reports[item['mode']]=report
 comparisons=[]
 for serial in [m for m in mode_reports if 'serial' in m]:
  base=mode_reports[serial];lookup={(r['module'],r['name'],tuple(r['geometry'])):r for r in base['kernel_signatures']}
  for concurrent,current in mode_reports.items():
   if 'serial' in concurrent:continue
   if concurrent.startswith('graph')!=serial.startswith('graph'):continue
   matches=[]
   for r in current['kernel_signatures']:
    ref=lookup.get((r['module'],r['name'],tuple(r['geometry'])))
    if ref:matches.append(dict(module=r['module'],name=r['name'],geometry=r['geometry'],serial_calls=ref['calls'],concurrent_calls=r['calls'],serial_mean_us=ref['mean_us'],concurrent_mean_us=r['mean_us'],mean_duration_ratio=r['mean_us']/ref['mean_us'],concurrent_total_ms=r['total_ms']))
   matches.sort(key=lambda r:r['concurrent_total_ms'],reverse=True)
   comparisons.append(dict(serial=serial,concurrent=concurrent,matched_launch_signature_types=len(matches),kernel_duration_comparison=matches))
 result=dict(plan=plan['name'],modes=mode_reports,comparisons=comparisons,caveat='Profiler elapsed intervals include stalls. Name plus launch geometry is matched, but is not a hardware-counter occupancy measurement. Continuous runs may submit differing numbers of Prefills; compare per-call means and inspect counts. Profile times are never end-to-end performance.')
 (root/plan['name']/'audit.json').write_text(json.dumps(result,indent=2)+'\n');reports.append(result)
small=[]
for r in reports:
 for mode,m in r['modes'].items():
  small.append(dict(plan=r['plan'],mode=mode,aggregate=m['aggregate'],global_measured_window=m['global_measured_window'],total_kernels=m['total_kernels'],matched=m['matched_launch_kernels'],unassigned_measured=sum(m['unassigned_measured_kernel_names'].values()),device_sync=m['steady_device_synchronize_calls'],capture_begin=m['capture_begin_calls']))
(root/'request_trace_summary.json').write_text(json.dumps(small,indent=2)+'\n')
for r in small:
 print(r['plan'],r['mode'],[(k,round(v['overlap_fraction']*100,2)) for k,v in r['aggregate'].items()],r['matched'], '/',r['total_kernels'])

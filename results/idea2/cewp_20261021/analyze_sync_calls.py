import json,sqlite3
from collections import defaultdict
from pathlib import Path
root=Path('results/idea2/cewp_20261021');manifest=json.loads((root/'request_profile_manifest.json').read_text());out=[]
for plan in manifest:
 if plan['status']!='done':continue
 for item in plan['traces']:
  d=sqlite3.connect(item['sqlite']);ranges=d.execute("SELECT n.start,n.end,n.globalTid FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL AND COALESCE(n.text,s.value) LIKE 'cewp/continuous/%'").fetchall();timings=defaultdict(list)
  for b,e,t,name in d.execute('SELECT r.start,r.end,r.globalTid,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id'):
   if any(rb<=b and e<=re and t==rt for rb,re,rt in ranges) and any(part in name for part in ('Synchronize','WaitEvent','EventQuery','GraphLaunch')):timings[name].append((e-b)/1000)
  out.append(dict(plan=plan['name'],mode=item['mode'],requests=len(ranges),api={name:dict(calls=len(values),total_host_ms=sum(values)/1000,mean_host_us=sum(values)/len(values),host_ms_per_request=sum(values)/1000/len(ranges)) for name,values in timings.items()}))
(root/'request_sync_api.json').write_text(json.dumps(dict(results=out,caveat='Profiler-perturbed host API durations; EventSynchronize includes intentional result-completion waiting, not removable overhead. CUDA StreamWaitEvent is a GPU dependency enqueue; its host call duration is not GPU wait duration.'),indent=2)+'\n')
for r in out:
 print(r['plan'],r['mode'],r['api'])

"""Graph launch correlation -> stage NVTX -> actual per-stream kernel intervals."""
import argparse,json,sqlite3
from pathlib import Path
from bisect import bisect_right
from collections import defaultdict
from preparations.idea2.scheduling_trace import union,duration,intersection
p=argparse.ArgumentParser();p.add_argument('prefix');p.add_argument('--output',required=True);a=p.parse_args()
results={}
for path in sorted(Path(a.prefix).parent.glob(Path(a.prefix).name+'*.sqlite')):
 db=sqlite3.connect(path)
 ranges=db.execute('SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL ORDER BY n.start,n.end DESC').fetchall()
 sections=[r for r in ranges if r[3] and r[3].startswith('stage/')]
 starts=[r[0] for r in sections]
 modules=defaultdict(list)
 for b,e,t,l in ranges:
  if l in ('ae','vlm'):modules[t].append((b,e,l))
 groups=defaultdict(list);unassigned=0
 for b,e,dev,ctx,stream,launch,lend,tid in db.execute('SELECT k.start,k.end,k.deviceId,k.contextId,k.streamId,r.start,r.end,r.globalTid FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId AND k.globalPid=(r.globalTid & -16777216) ORDER BY r.start'):
  si=bisect_right(starts,launch)-1
  if si<0 or lend>sections[si][1]:continue
  matches=[m for mb,me,m in modules[tid] if mb<=launch and lend<=me]
  if not matches:
   # Synthetic common release _sleep is intentionally not a branch kernel.
   unassigned+=1;continue
  groups[sections[si][3]].append((b,e,dev,ctx,stream,matches[-1]))
 for name,ks in groups.items():
  overlap=0;mods={b:[r for r in ks if r[-1]==b] for b in ('ae','vlm')}
  for dc in {(r[2],r[3]) for r in ks}:
   streams={b:{r[4] for r in rs if r[2:4]==dc} for b,rs in mods.items()}
   assert all(len(s)<=1 for s in streams.values())
   if streams['ae'] and streams['vlm'] and streams['ae']!=streams['vlm']:
    overlap+=intersection(*[[(r[0],r[1]) for r in mods[b] if r[2:4]==dc] for b in ('ae','vlm')])
  total=duration([(r[0],r[1]) for r in ks]);results[name]=dict(trace=str(path),kernel_count=len(ks),overlap_ms=overlap/1e6,kernel_union_ms=total/1e6,overlap_fraction=overlap/total if total else 0,modules={b:dict(kernel_busy_ms=duration([(r[0],r[1]) for r in rs])/1e6,streams=sorted({r[4] for r in rs}),kernels=len(rs)) for b,rs in mods.items()},synthetic_release_kernels_excluded=unassigned)
Path(a.output).write_text(json.dumps(results,indent=2)+'\n')
for name,r in results.items():print(name,round(r['overlap_fraction']*100,2),'%',round(r['overlap_ms'],5),'ms')

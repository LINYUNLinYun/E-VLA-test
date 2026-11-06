"""Render actual nsys kernel bars, never NVTX envelopes."""
import bisect
import collections
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parent
DB = ROOT.parents[2] / 'nsys/idea2_experiment/scheduling_final_20261021.1.sqlite'
c = sqlite3.connect(DB)
ranges = collections.defaultdict(list)
sections = {}
for a,b,t,label in c.execute('select n.start,n.end,n.globalTid,coalesce(n.text,s.value) from NVTX_EVENTS n left join StringIds s on n.textId=s.id where n.end is not null order by n.start,n.end desc'):
    if label in ('ae','prefill'):
        ranges[t].append((a,b,label))
    elif label in ('pair/original/1','pair/ae_first/1','pair/threaded_prefill_first/1'):
        sections[label] = (a,b)
starts = {t:[v[0] for v in rows] for t,rows in ranges.items()}
values = collections.defaultdict(list)
for a,b,s,la,lb,t in c.execute('select k.start,k.end,k.streamId,r.start,r.end,r.globalTid from CUPTI_ACTIVITY_KIND_KERNEL k join CUPTI_ACTIVITY_KIND_RUNTIME r on k.correlationId=r.correlationId and k.globalPid=(r.globalTid & -16777216)'):
    section = next((k for k,(start,end) in sections.items() if start<=la and lb<=end),None)
    if section is None:
        continue
    for i in range(bisect.bisect_right(starts.get(t,[]),la)-1,-1,-1):
        ra,rb,module = ranges[t][i]
        if ra<=la and lb<=rb:
            values[section].append((a,b,s,module))
            break
fig,axes=plt.subplots(3,2,figsize=(14,6),gridspec_kw={'width_ratios':[2.5,1.2]})
colors={'ae':'#d65437','prefill':'#2478b8'}
for row,label in enumerate(('pair/original/1','pair/ae_first/1','pair/threaded_prefill_first/1')):
    kernels=values[label]
    origin=min(r[0] for r in kernels)
    left=[r for r in kernels if r[3]=='ae']
    right=[r for r in kernels if r[3]=='prefill']
    overlaps=[]
    left=sorted(left);right=sorted(right)
    i=j=0
    while i<len(left) and j<len(right):
        a,b,stream,_=left[i]
        ra,rb,rstream,_=right[j]
        if stream!=rstream and max(a,ra)<min(b,rb):
            overlaps.append((max(a,ra),min(b,rb)))
        if b<=rb:
            i+=1
        else:
            j+=1
    center=(overlaps[len(overlaps)//2][0]-origin)/1e6 if overlaps else ((min(r[0] for r in left)-origin)/1e6)
    for col,ax in enumerate(axes[row]):
        for module,y in (('ae',0),('prefill',1)):
            selected=[r for r in kernels if r[3]==module]
            segments=[[((a-origin)/1e6,y),((b-origin)/1e6,y)] for a,b,_,_ in selected]
            ax.add_collection(LineCollection(segments,colors=colors[module],linewidths=8))
        ax.set_ylim(-.6,1.6)
        ax.set_yticks([0,1],['AE','Prefill'])
        ax.set_xlim((0,(max(r[1] for r in kernels)-origin)/1e6) if col==0 else (max(0,center-5),center+5))
        ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('GPU time from first kernel (ms)')
        ax.set_title(label.replace('pair/','') + (' | full' if col==0 else ' | 10 ms detail'),fontsize=10)
fig.suptitle('Actual CUDA kernels from Nsight Systems (bars = recorded kernels; copies excluded)',fontsize=12)
fig.tight_layout()
fig.savefig(ROOT/'actual_kernel_timeline.png',dpi=180)
print(ROOT/'actual_kernel_timeline.png')

"""Draw actual CUPTI kernel intervals, selected by launch->NVTX correlation."""
import argparse,json,sqlite3
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);a=p.parse_args();items=json.loads(Path(a.config).read_text())
colors={'ae':'#D55E00','prefill':'#777777','prefill/vision':'#009E73','prefill/vlm':'#0072B2'}
fig,axes=plt.subplots(len(items),1,figsize=(12,3.0*len(items)),squeeze=False,layout='constrained');receipt=[]
for ax,item in zip(axes[:,0],items):
 db=sqlite3.connect(item['sqlite']);ranges=db.execute('SELECT n.start,n.end,n.globalTid,COALESCE(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id WHERE n.end IS NOT NULL ORDER BY n.start,n.end DESC').fetchall()
 selected=[r for r in ranges if r[3]==item['nvtx']]
 assert len(selected)==1,(item['sqlite'],selected)
 sb,se,st,_=selected[0];mods=defaultdict(list)
 for b,e,t,label in ranges:
  if label in colors:mods[t].append((b,e,label))
 ks=[]
 for b,e,dev,ctx,stream,launch,lend,tid in db.execute('SELECT k.start,k.end,k.deviceId,k.contextId,k.streamId,r.start,r.end,r.globalTid FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId AND k.globalPid=(r.globalTid & -16777216) ORDER BY r.start'):
  if not (tid==st and sb<=launch and lend<=se):continue
  module=next((l for mb,me,l in reversed(mods[tid]) if mb<=launch and lend<=me),None)
  if module:ks.append((b,e,dev,ctx,stream,module))
 assert ks
 assert len({(r[2],r[3]) for r in ks})==1
 origin=min(r[0] for r in ks);end=max(r[1] for r in ks)
 stream_names={r[4] for r in ks};lanes=[('ae',1),('prefill',0)]
 for branch,y in lanes:
  for stage,color in colors.items():
   if (stage=='ae')!=(branch=='ae'):continue
   selected=[r for r in ks if r[5]==stage]
   if selected:ax.broken_barh([((r[0]-origin)/1e6,(r[1]-r[0])/1e6) for r in selected],(y-.22,.44),facecolors=color,edgecolors='none',linewidth=0)
  rs=[r for r in ks if (r[5]=='ae')==(branch=='ae')];streams=sorted({r[4] for r in rs});assert len(streams)<=1
  ax.text(1.003,y,f'{len(rs):,} kernels\nstream {streams}',transform=ax.get_yaxis_transform(),va='center',fontsize=9)
 ax.set_yticks([0,1],['Prefill','Action Expert']);ax.set_ylim(-.6,1.6);ax.set_xlim(-.5,(end-origin)/1e6+.5);ax.grid(axis='x',alpha=.25);ax.set_axisbelow(True);ax.set_title(item['label']+' — '+item['nvtx'],fontsize=11,loc='left');ax.set_xlabel('GPU time from first branch kernel (ms; profiler-perturbed)')
 receipt.append(dict(**item,kernels=len(ks),device_context=list({(r[2],r[3]) for r in ks})[0],streams=sorted(stream_names),origin_ns=origin,end_ns=end,method='Only CUPTI_ACTIVITY_KIND_KERNEL intervals; graph child kernels attributed via correlationId to cudaGraphLaunch inside exact NVTX section. Memcpy and Host NVTX spans not drawn.'))
limit=max((r['end_ns']-r['origin_ns'])/1e6 for r in receipt)+.5
for ax in axes[:,0]:ax.set_xlim(-.5,limit)
fig.legend(handles=[Patch(color=c,label=l) for l,c in colors.items()],loc='outside lower center',ncols=4,fontsize=10)
fig.suptitle('π0.5 D10 — actual GPU kernel timelines, request index 2\nNot steady latency: node tracing perturbs dispatch; pre-launch CPU waits excluded',fontsize=13)
fig.savefig(a.output,dpi=180)
Path(a.output).with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(a.output)

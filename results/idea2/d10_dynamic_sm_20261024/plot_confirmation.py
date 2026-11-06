"""Standalone scientific figures from confirmed measurements only."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
out=root.parents[2]/'reports/idea2_d10_dynamic_sm_20261024'
rows=list(csv.DictReader((out/'crossframe_formal_summary.csv').open()))
assert {r['case'] for r in rows}=={'d10_crossframe_formal','d10_crossframe_repeat'}
modes=['fresh_serial','lookahead_serial','encoder_overlap','combined']
labels=['F0: serial','F1: Host lookahead','F2: + Encoder overlap','F4: + layer overlap']
colors=['#64748b','#0284c7','#0d9488','#7c3aed']
fig,ax=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for i,(mode,label,color) in enumerate(zip(modes,labels,colors)):
    rr=[r for r in rows if r['mode']==mode]
    times=np.array([float(r['trajectory_ms_mean'])/1000 for r in rr])
    ax[0].bar(i,times.mean(),color=color,alpha=.8)
    ax[0].scatter([i-.08,i+.08],times,color='black',s=14,zorder=3)
    ax[0].text(i,times.max()+.035,f'{times.mean():.3f}',ha='center',fontsize=10)
    ax[1].scatter([float(r['admission_to_action_ms_p95']) for r in rr],
                  [float(r['continuous_all_work_hz']) for r in rr],color=color,label=label,s=60)
ax[0].set_xticks(range(4),['F0','F1','F2','F4'])
ax[0].set_ylim(0,1.9)
ax[0].set_ylabel('Complete 16-frame trajectory (s), lower is better')
ax[0].set_title('100 paired trajectories per run; fill/drain included')
ax[1].set_xlabel('Admission-to-action P95 (ms), lower is better')
ax[1].set_ylabel('Completed requests/s, higher is better')
ax[1].set_title('100 continuous requests per mode and run')
ax[1].legend(fontsize=9,loc='lower right')
for a in ax:
    a.spines[['top','right']].set_visible(False)
    a.grid(axis='y',alpha=.2)
fig.suptitle('D10 / fresh KV: Host preparation, GPU overlap, and latency trade-off',fontsize=13)
fig.savefig(out/'crossframe_confirmation.png',dpi=180)
fig.savefig(out/'crossframe_confirmation.svg')
plt.close(fig)
print(out/'crossframe_confirmation.png')

import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from preparations.idea1_v3_4_selection_async.common import OUT


def render():
    data=list(csv.DictReader((OUT/'overlap/timings.csv').open()))
    fig,ax=plt.subplots(figsize=(9.5,4.7),layout='constrained')
    colors=dict(native='#3366AA',execution='#D3D8DE',selector='#159A9C',wait='#D65454')
    for i,q in enumerate((4,2,1)):
        rr=[r for r in data if r['q']==str(q) and r['priority']=='high' and r['execution_ms']=='250']
        mean=lambda key:float(np.mean([float(r[key]) for r in rr]))
        ready=mean('action_ready_ms');execution=mean('actual_execution_ms')
        graph_end=mean('graph_finished_ms')
        graph_start=graph_end-mean('graph_gpu_ms');y=4-2*i
        ax.broken_barh([(0,ready)],(y+.12,.5),facecolors=colors['native'])
        ax.broken_barh([(ready,execution)],(y+.12,.5),facecolors=colors['execution'])
        wait=mean('selector_wait_ms')
        if wait>.1:ax.broken_barh([(ready+execution,wait)],(y+.12,.5),facecolors=colors['wait'])
        ax.broken_barh([(graph_start,graph_end-graph_start)],(y-.48,.4),facecolors=colors['selector'])
        ax.text(405,y+.30,f"wait {mean('selector_wait_ms'):.2f} ms",va='center',fontsize=10)
        ax.text(graph_end+4,y-.28,f"{mean('graph_gpu_ms'):.0f} ms",va='center',fontsize=9,color='#147477')
    ax.axvline(200,color='#7B8088',linestyle='--',linewidth=1)
    ax.text(202,5.0,'200 ms',color='#606770',fontsize=9)
    ax.set(yticks=[4.05,2.05,.05],yticklabels=['Four directions','Two directions','One direction'],
        xlabel='Milliseconds after native VLM completion',xlim=(0,505),ylim=(-.9,5.3))
    ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    ax.set_title('Original SDPA selector overlaps native AE and a 250 ms execution window',loc='left',fontsize=12,pad=15)
    ax.legend(handles=[Patch(color=colors[k],label=label) for k,label in (
        ('native','Staging + native AE'),('execution','Action execution'),('selector','Selector CUDA span'),('wait','Remaining wait'))],
        loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,frameon=False,fontsize=9)
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/'overlap_250ms.png',dpi=180,bbox_inches='tight')
    fig.savefig(folder/'overlap_250ms.svg',bbox_inches='tight');plt.close(fig)
    return folder/'overlap_250ms.png'


if __name__=='__main__':print(render())

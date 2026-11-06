"""Standalone evidence figure; different-cohort comparisons remain labeled."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root):
    root=Path(root)
    data=json.loads((root/'e1_conclusion.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    values=[r['examples_per_s'] for r in data['bank']]
    bars=axes[0].bar(['B1','B2*','B4*'],values,color=['#64748b','#278b9b','#278b9b'],width=.6)
    axes[0].bar_label(bars,labels=[f'{v:.2f}' for v in values],padding=5)
    axes[0].set(ylim=(0,22),ylabel='Completed examples / second',title='Fixed bank: 100 real observations')
    axes[0].text(.5,-.2,'* B2/B4 fail strict and relaxed numerical gates',ha='center',transform=axes[0].transAxes,fontsize=9)
    speedups=[data['wait40_closed_loop'][0]['throughput_vs_measured_b1']]+[
        r['throughput_vs_measured_b1'] for r in data['short_wait_closed_loop'][:2]]
    labels=['max B4 / 40 ms\ncohort A','max B2 / 0 ms\ncohort B','max B4 / 5 ms\ncohort B']
    bars=axes[1].bar(labels,speedups,color=['#d78b50','#278b9b','#278b9b'],width=.6)
    axes[1].bar_label(bars,labels=[f'{v:.3f}x' for v in speedups],padding=5)
    axes[1].axhline(1,color='#334155',linewidth=1,linestyle='--')
    axes[1].set(ylim=(0,1.2),ylabel='Episode throughput / same-service B1',title='Closed loop: 8 fixed scenes per arm')
    axes[1].text(.5,-.2,'Approximate outputs; work counts differ; no confidence intervals',ha='center',transform=axes[1].transAxes,fontsize=9)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15)
        ax.set_axisbelow(True)
    fig.suptitle('Native pi0 Full: bank batching gains did not yield a substantial evaluation gain',fontsize=12)
    fig.subplots_adjust(left=.075,right=.985,top=.83,bottom=.25,wspace=.32)
    fig.savefig(root/'e1_batch_evidence.png',dpi=180)
    fig.savefig(root/'e1_batch_evidence.pdf')
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root',nargs='?',default='results/idea1_v3_2')
    plot(parser.parse_args().root)

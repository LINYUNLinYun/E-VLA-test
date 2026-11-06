"""Show small submodule changes that are hidden by a full cosine axis."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('analysis_dir',type=Path)
    args=parser.parse_args()
    with (args.analysis_dir/'submodule_observation_metrics.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    grouped=defaultdict(list)
    for r in rows:
        grouped[(r['module'],int(r['layer']),r['modality'],r['sample_id'])].append(float(r['cosine_mean']))
    task_means=defaultdict(list)
    for (mod,layer,view,task),values in grouped.items():
        task_means[(mod,layer,view)].append(float(np.mean(values)))
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for col,mod in enumerate(('attn','ffn')):
        for view,color in [('agent_view','#1f77b4'),('wrist_view','#2ca02c'),('language','#ff7f0e')]:
            layers=np.arange(1,18)
            values=np.asarray([np.mean(task_means[(mod,l,view)]) for l in layers])
            axes[0,col].plot(layers,values,'o-',label=view,color=color,markersize=4)
            axes[1,col].plot(layers[1:12],values[1:12],'o-',label=view,color=color,markersize=4)
        axes[0,col].set_title(mod.upper()+': all hidden-update layers')
        axes[0,col].set_ylim(0.2,1.01)
        axes[0,col].set_xticks(np.arange(1,18,2))
        axes[1,col].set_title(mod.upper()+': L2–L12 zoom (different y-axis)')
        axes[1,col].set_ylim(0.978,1.0005)
        axes[1,col].set_xticks(np.arange(2,13))
        for row in range(2):
            axes[row,col].set_xlabel('VLM layer (1-based)')
            axes[row,col].set_ylabel('Task-equal mean token cosine')
            axes[row,col].grid(alpha=.2)
            axes[row,col].legend(fontsize=8)
    fig.suptitle('Attention / FFN residual updates: full range and near-identity zoom\n3 tasks, 12 observations; candidate safety requires action and closed-loop checks')
    fig.savefig(args.analysis_dir/'submodule_cosine_zoom.png',dpi=180)
    fig.savefig(args.analysis_dir/'submodule_cosine_zoom.pdf')
    plt.close(fig)


if __name__=='__main__':
    main()

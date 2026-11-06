#!/usr/bin/env python3
"""Plot evidence-backed furthest stages for historical new failures only."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-analysis', type=Path, required=True)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    with args.paired_analysis.open() as stream:
        rows = [row for row in csv.DictReader(stream) if row['cohort']=='new_failure' and row['phase_source']=='simulator_telemetry']
    conditions = ['V16','V14','V12','V10-Mid']
    stages = ['initial','approach','contact','grasp_or_manipulate','transport','goal_alignment','release_or_final_action','success']
    palette = ['#dce9f5','#a6c8e2','#76b7b2','#59a14f','#edc948','#f28e2b','#e15759','#b07aa1']
    labels = ['Initial','Approach','Contact','Grasp / manipulate','Transport','Goal alignment','Final action','Success']
    counts = Counter((row['condition_id'],row['candidate_furthest_phase']) for row in rows)
    unknown = {stage for _,stage in counts} - set(stages)
    if unknown:
        raise ValueError(f'Unknown stages: {unknown}')
    fig, ax = plt.subplots(figsize=(11,4.8),layout='constrained')
    left = np.zeros(len(conditions))
    for stage,color,label in zip(stages,palette,labels):
        values = np.asarray([counts[(condition,stage)] for condition in conditions])
        if not values.any():
            continue
        ax.barh(conditions,values,left=left,color=color,label=label,height=.64,edgecolor='white',linewidth=1)
        for index,value in enumerate(values):
            if value:
                ax.text(left[index]+value/2,index,str(value),ha='center',va='center',fontsize=11)
        left += values
    for index,total in enumerate(left):
        ax.text(total+.25,index,f'n={int(total)}',va='center',color='#555555')
    ax.invert_yaxis()
    ax.set_xlim(0,left.max()+3)
    ax.set_xlabel('Episodes (new failures relative to Historical Control / D3)')
    ax.set_title('Furthest semantic stage reached — not causal error onset',loc='left',fontsize=14,pad=16)
    ax.spines[['top','right','left']].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis='x',alpha=.15)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,frameon=False)
    args.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output_prefix.with_suffix('.png'),dpi=200,bbox_inches='tight')
    fig.savefig(args.output_prefix.with_suffix('.pdf'),bbox_inches='tight')
    plt.close(fig)
    args.output_prefix.with_suffix('.json').write_text(json.dumps({'cohort':'historical_new_failures',
        'evidence':'simulator_telemetry','episode_count':len(rows),'counts':[
        {'condition':condition,'stage':stage,'count':count} for (condition,stage),count in sorted(counts.items())],
        'warning':'Furthest reached stage is not the causal onset of model error; denominator excludes other outcome cohorts.'},indent=2)+'\n')


if __name__=='__main__':
    main()

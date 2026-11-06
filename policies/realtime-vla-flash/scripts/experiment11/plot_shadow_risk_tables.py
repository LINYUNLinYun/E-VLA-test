"""Supplement existing risk tables with directional, held-state and tail plots."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.experiment11.protocol import RESULTS
from scripts.experiment10.common import dump_json, sha256


def main():
    root = RESULTS/'analysis/next_round/round2_20261011/shadow_risk'
    output = root/'supplemental_figures';output.mkdir(exist_ok=True)
    names = ('gripper_direction_state_phase.csv','prefix_controller_metrics.csv','offset1_50_controller_metrics.csv')
    tables = {name:list(csv.DictReader((root/name).open())) for name in names}
    variants = ('V16_A02','V15_A08','V14_A03','V13_Custom')
    phases = ('approach','contact_grasp','transport','alignment_release')
    directions = ('full_open_variant_close','full_close_variant_open')
    for condition in ('all','stably_grasped','lifted_10mm','full_closed_and_stably_grasped'):
        selected_directions = directions if condition!='full_closed_and_stably_grasped' else (directions[1],)
        fig,axes = plt.subplots(1,len(selected_directions),figsize=(6*len(selected_directions),5),layout='constrained',squeeze=False)
        for ax,direction in zip(axes[0],selected_directions):
            cells = [];counts = []
            for variant in variants:
                values,n = [],[]
                for phase in phases:
                    matches = [r for r in tables[names[0]] if r['variant_id']==variant and r['phase']==phase
                               and r['condition']==condition and r['direction']==direction]
                    values.append(100*float(matches[0]['task_equal_rate']) if matches else np.nan)
                    n.append(int(matches[0]['n_full_executed_offsets']) if matches else 0)
                cells.append(values);counts.append(n)
            image = ax.imshow(np.ma.masked_invalid(cells),cmap='magma',aspect='auto',vmin=0)
            fig.colorbar(image,ax=ax,label='Task-equal disagreement rate (%)')
            ax.set_xticks(range(4),phases,rotation=25,ha='right');ax.set_yticks(range(4),variants)
            ax.set_title(direction)
            for i in range(4):
                for j in range(4):
                    label = 'N/A' if not counts[i][j] else f'{cells[i][j]:.2f}%\nn={counts[i][j]}'
                    ax.text(j,i,label,ha='center',va='center',color='white',fontsize=8,
                            bbox={'facecolor':'black','alpha':.25,'edgecolor':'none'})
        fig.suptitle(f'Full realized offsets1–5 / state: {condition}; n=available step pairs, not tasks')
        fig.savefig(output/f'gripper_direction_phase_{condition}.png',dpi=180);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,metric,unit in zip(axes,('translation_target_difference_mm_mean','rotation_increment_vector_difference_deg_mean'),('mm','deg')):
        x = np.arange(4)
        for field,style,label in [('task_equal_mean','o-','Task-equal mean'),('sample_p50','s--','Sample P50'),
                                  ('sample_p95','^--','Sample P95')]:
            values = [float(next(r[field] for r in tables[names[1]] if r['variant_id']==v and r['suite']=='ALL'
                                  and int(r['horizon'])==5 and r['metric']==metric)) for v in variants]
            ax.plot(x,values,style,label=label)
        ax.set_xticks(x,variants,rotation=20);ax.set(ylabel=unit,title='Prefix1–5 mean '+metric)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Distribution of per-replan prefix means; sample quantiles ≠ task-weighted mean')
    fig.savefig(output/'prefix5_distribution_quantiles.png',dpi=180);plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for v in variants:
        for axis,metric in zip(axes,('translation_signed_x_mm','translation_signed_y_mm','translation_signed_z_mm')):
            rows = sorted((r for r in tables[names[2]] if r['variant_id']==v and r['suite']=='ALL' and r['metric']==metric),key=lambda r:int(r['offset']))
            axis.plot([int(r['offset']) for r in rows],[float(r['task_equal_mean']) for r in rows],label=v)
            axis.axhline(0,color='gray',linewidth=.7);axis.set(xlabel='Predicted offset',ylabel='mm',title=metric)
            axis.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Signed controller-target increment bias; not measured EEF displacement')
    fig.savefig(output/'signed_translation_bias_offsets1_50.png',dpi=180);plt.close(fig)
    dump_json(output/'completion_audit.json',{'complete':True,'source_hashes':{str(root/n):sha256(root/n) for n in names},
         'output_hashes':{str(p):sha256(p) for p in output.glob('*.png')},'new_predictions':0,'environment_steps':0,
         'scope':'visualization_of_previously_validated_tables;no_new_equivalence_or_causal_claim'})
    print(json.dumps({'complete':True,'figures':len(list(output.glob('*.png')))}))


if __name__=='__main__':
    main()

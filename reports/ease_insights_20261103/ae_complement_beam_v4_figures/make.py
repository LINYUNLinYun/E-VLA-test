"""Render only accepted full720 results; no partial-outcome reads or analysis changes."""
import csv
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

ROOT=Path('/root/nfs/embodied/benchmark')
REPORT=ROOT/'reports/ease_insights_20261103/ae_complement_beam_v4'
OUTPUT=Path(__file__).resolve().parent
IDS=('full_d10','vlm_c','ae14_d3')
COLORS={'full_d10':'#747c84','vlm_c':'#3479ad','ae14_d3':'#c87539'}
LABELS={'full_d10':'Full D10','vlm_c':'VLM path','ae14_d3':'AE path'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def accepted():
    path=REPORT/'confirmation_pipeline_acceptance.json'
    if not path.exists():return None
    pipeline=read(path)
    require(pipeline['status']=='pass' and pipeline['new_cases']==720,'Need all720 accepted cases')
    for name,pin in pipeline['artifacts'].items():require(sha(name)==pin,'Accepted pipeline artifact changed')
    analysis=read(REPORT/'confirmation_analysis.json');audit=read(REPORT/'confirmation_independent_statistics.json')
    require(analysis['complete'] and analysis['valid_episodes']==720 and audit['status']=='pass' and not audit['differences'],
        'Need complete independently reconstructed statistics')
    require(analysis['bundle_hash']==audit['bundle_hash']==pipeline['bundle_hash']
        and analysis['conclusion']==audit['conclusion']==pipeline['conclusion'],'Mixed replication identities')
    require(len(analysis['primary_directions'])==2 and analysis['fixed_input_cost']['valid_services']==12,'Need fixed two contrasts and12 completed services')
    for d in analysis['primary_directions']:
        require(d['n_paired']==60 and len(d['by_task'])==10 and len(d['by_noise_stream'])==2,'Wrong task/noise sampling units')
    return analysis,pipeline


def write_csv(path,header,rows):
    with path.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)


def render(data,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10.5,'axes.labelsize':9,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14.2,5.2),gridspec_kw={'width_ratios':[1.5,1.15,1.]})
    fig.subplots_adjust(left=.055,right=.985,top=.78,bottom=.28,wspace=.42)
    fig.suptitle('Prespecified R5 fast-path replication: '+data['conclusion'].replace('_',' '),x=.5,y=.99,fontsize=14,fontweight='bold')
    fig.text(.5,.91,'VLM path = VLM12 / AE18, Fast D2     |     AE path = VLM18 / AE14, Fast D3     |     H50 / R5, Full + two Fast',ha='center',fontsize=9)
    group_keys=['libero_spatial','libero_object','libero_goal','libero_10','overall'];group_names=['Spatial','Object','Goal','Long','All tasks']
    rows=[];x=np.arange(5);width=.23
    for index,cid in enumerate(IDS):
        summaries=data['condition_summaries'][cid]
        cells=[summaries['overall'] if key=='overall' else summaries['by_suite'][key] for key in group_keys]
        values=[v['successes']/v['n'] for v in cells];positions=x+(index-1)*width
        axes[0].bar(positions,values,width=.205,color=COLORS[cid],label=LABELS[cid])
        for px,v,cell,key in zip(positions,values,cells,group_keys):
            axes[0].text(px,v+.025,f"{cell['successes']}/{cell['n']}",ha='center',va='bottom',fontsize=6.8)
            rows.append([cid,key,cell['n'],cell['successes'],v])
    axes[0].set_xticks(x,group_names);axes[0].set_ylim(0,1.14);axes[0].set_yticks([0,.25,.5,.75,1]);axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel('Closed-loop success rate');axes[0].set_title('A  All cases, including failures',loc='left',pad=28)
    axes[0].legend(frameon=False,ncol=3,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.105),handlelength=1.1,columnspacing=.8)
    axes[0].grid(axis='y',alpha=.18);axes[0].set_axisbelow(True)
    contrasts=[];taskwise=[];endpoints=[]
    for position,suite in [(1,'libero_spatial'),(0,'libero_10')]:
        d=next(v for v in data['primary_directions'] if v['frozen_direction']['suite']==suite)
        info=d['inference'];interval=info['confidence_intervals']['task_cluster'];sign=1 if d['frozen_direction']['better']=='ae14_d3' else -1
        delta=sign*d['success_difference']*100;low,high=(interval['low']*100,interval['high']*100) if sign==1 else (-interval['high']*100,-interval['low']*100)
        endpoints.extend([abs(delta),abs(low),abs(high)]);axes[1].hlines(position,low,high,color='#675a86',linewidth=3)
        axes[1].plot([low,high],[position,position],'|',color='#675a86',markersize=10);axes[1].scatter([delta],[position],color='#675a86',s=46,zorder=3)
        axes[1].text(.02,position+.23,f"Holm p = {info['primary_holm2_p']:.3g}",transform=axes[1].get_yaxis_transform(),fontsize=8)
        axes[1].text(.02,position-.27,f"AE - VLM = {delta:+.1f} pp",transform=axes[1].get_yaxis_transform(),fontsize=8)
        contrasts.append([suite,delta,low,high,info['primary_holm2_p'],info['direction_reproduced']])
        for t in d['by_task']:taskwise.append([suite,t['task'],sign*t['success_difference_count'],t['n_paired'],sign*t['success_difference']*100])
    extent=max(15,5+10*np.ceil(max(endpoints)/10));axes[1].set_xlim(-extent,extent);axes[1].set_ylim(-.55,1.55)
    axes[1].set_yticks([1,0],['Spatial','Long']);axes[1].axvline(0,color='#666666',linewidth=1,linestyle='--')
    axes[1].set_xlabel('AE path minus VLM path (percentage points)');axes[1].set_title('B  Task-clustered paired differences\n95% bootstrap intervals',loc='left',pad=10)
    axes[1].grid(axis='x',alpha=.16)
    costs=data['fixed_input_cost'];cost_rows=[]
    for index,cid in enumerate(IDS[1:]):
        values=[s['full_over_policy_cycle'][cid] for s in costs['services']];jitter=np.linspace(-.12,.12,len(values))
        axes[2].scatter(index+jitter,values,s=22,color=COLORS[cid],alpha=.55,zorder=2)
        median=costs['median_full_over_policy_cycle'][cid];axes[2].scatter([index],[median],marker='D',color='black',s=42,zorder=3)
        axes[2].text(index,median+.035,f'{median:.3f}',ha='center',fontsize=9)
        for s,v in zip(costs['services'],values):cost_rows.append([cid,s['noise_repeat'],s['gpu_uuid'],s['path'],v])
    axes[2].axhline(1.15,color='#666666',linestyle='--',linewidth=1,label='1.15 gate')
    axes[2].set_xticks([0,1],['VLM path','AE path']);axes[2].set_xlim(-.45,1.45)
    maximum=max(row[-1] for row in cost_rows);axes[2].set_ylim(.97,max(1.3,maximum+.11));axes[2].set_ylabel('Full / path native policy cost')
    axes[2].set_title('C  Same-service fixed-input probes\n12 completed services',loc='left',pad=10)
    axes[2].text(.5,.04,f"Pair factor = {costs['pair_cost_factor']:.3f} (gate <= 1.10)",transform=axes[2].transAxes,ha='center',fontsize=8)
    axes[2].legend(loc='upper left',frameon=False,fontsize=8);axes[2].grid(axis='y',alpha=.16);axes[2].set_axisbelow(True)
    captions=[
        '720 rollouts: 3 conditions x 40 tasks x 3 official initial states x 2 fixed policy-noise streams; 120 physical cases and 240 rollouts per condition.',
        'B: ten whole-task clusters per suite, six paired observations per task; one-sided exact task signflip with Holm2. C: native policy cost, not closed-loop speed.',
        'Initials 47-49 are held out relative to R5 discovery 45-46; the same official initial states were used in historical R12 experiments.'
    ]
    for y,line in zip([.16,.105,.05],captions):fig.text(.055,y,line,ha='left',va='center',fontsize=8.2,color='#444444')
    for suffix in ('png','pdf','svg'):fig.savefig(output/f'fastpath_replication.{suffix}',dpi=220,facecolor='white')
    plt.close(fig)
    write_csv(output/'successes.csv',['condition','group','rollouts','successes','success_rate'],rows)
    write_csv(output/'paired_differences.csv',['suite','AE_minus_VLM_pp','task_cluster_low95_pp','task_cluster_high95_pp','primary_Holm2_p','registered_direction_reproduced'],contrasts)
    write_csv(output/'task_differences.csv',['suite','task','AE_minus_VLM_success_count','paired_rollouts','AE_minus_VLM_pp'],taskwise)
    write_csv(output/'native_costs.csv',['condition','noise_repeat','gpu_uuid','cost_artifact','Full_over_path_policy_ratio'],cost_rows)


def main():
    pair=accepted()
    if pair is None:return dict(status='awaiting_full720_pipeline_acceptance',artifacts_created=False)
    data,pipeline=pair;inputs=[REPORT/name for name in ['confirmation_pipeline_acceptance.json','confirmation_analysis.json','confirmation_independent_statistics.json','prior_configuration_audit.json']]
    pins={str(p):sha(p) for p in inputs};source=sha(__file__);target=OUTPUT/'provenance.json'
    if target.exists():
        old=read(target);require(old['source_sha256']==source and old['input_pins']==pins and all(sha(p)==h for p,h in old['outputs'].items()),'Existingfigureprovenancechanged;useanewfigureversion')
        return dict(status='already_complete',conclusion=old['conclusion'])
    OUTPUT.mkdir(parents=True,exist_ok=True);render(data,OUTPUT)
    outputs=[OUTPUT/name for name in ['fastpath_replication.png','fastpath_replication.pdf','fastpath_replication.svg','successes.csv','paired_differences.csv','task_differences.csv','native_costs.csv']]
    value=dict(status='generated_from_accepted_full720',generated_UTC=datetime.now(timezone.utc).isoformat(),bundle_hash=data['bundle_hash'],
        conclusion=data['conclusion'],source_sha256=source,input_pins=pins,outputs={str(p):sha(p) for p in outputs},
        new_quality_cases=0,scientific_inputs_modified=False,figures_visually_inspected=False,
        scope='Accepted720-case success rates,task-clustered fixed contrasts and fixed-input native cost profiles;no globally unseen-state,closed-loop acceleration,online routing or study-wide5percent claim')
    target.write_text(json.dumps(value,indent=2)+'\n')
    return dict(status=value['status'],conclusion=value['conclusion'],figure=str(OUTPUT/'fastpath_replication.png'))


if __name__=='__main__':print(json.dumps(main()))

"""Static scientific figures from analyze.py outputs; missing data never become zero."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from common import ROOT, SUITES, atomic_json

LABELS=('Spatial','Object','Goal','Long')
SHORT=dict(zip(SUITES,LABELS))
MATRICES=[
    ('pure_success','draft','success_rate','Pure draft success','Successful episodes / completed episodes'),
    ('flash_success','flash','success_rate','FLASH success','Successful episodes / completed episodes'),
    ('flash_acceptance','flash','acceptance_rate','FLASH verified action acceptance','Accepted prefix actions / (12 × verified rounds)'),
    ('flash_rejection_full','flash','rejection_full_fraction','FLASH rejection-triggered Full share','Rejection-triggered Full rounds / all policy requests'),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_analysis(folder):
    folder=Path(folder)
    summary_path=folder/'summary.json'; episode_path=folder/'episodes.json'
    summary=json.loads(summary_path.read_text())
    episodes=json.loads(episode_path.read_text())
    hashes={str(path.resolve()):digest(path) for path in (summary_path,episode_path)}
    if (folder/'audit.json').exists():
        hashes[str((folder/'audit.json').resolve())]=digest(folder/'audit.json')
    seen=set()
    for r in summary:
        key=(r['mode'],r['source_suite'],r['target_suite'])
        if key in seen:raise ValueError('Duplicate summary condition: %r'%(key,))
        seen.add(key)
        if int(r['n'])<=0:raise ValueError('A summary row with n=0 must be omitted, not plotted as observed')
    return summary,episodes,hashes


def matrix_data(summary,mode,field,expected_n):
    values=np.full((4,4),np.nan); counts=np.zeros((4,4),dtype=int); cells=[]
    rows={(r['source_suite'],r['target_suite']):r for r in summary if r['mode']==mode}
    for i,source in enumerate(SUITES):
        for j,target in enumerate(SUITES):
            row=rows.get((source,target))
            value=None
            if row:
                counts[i,j]=int(row['n'])
                if counts[i,j]>expected_n:raise ValueError('More observations than the expected matrix denominator')
                value=row.get(field)
                if value is not None:
                    if not math.isfinite(float(value)) or not 0<=float(value)<=1:
                        raise ValueError('Metric outside [0,1]')
                    values[i,j]=float(value)*100
            cells.append(dict(mode=mode,metric=field,source_suite=source,target_suite=target,
                n=int(counts[i,j]),expected_n=expected_n,value=value,
                status='missing' if row is None else ('unavailable_metric' if value is None else 'observed')))
    return values,counts,cells


def style_axis(ax):
    ax.set_xticks(range(4),LABELS);ax.set_yticks(range(4),LABELS)
    ax.set_xlabel('Target evaluation suite',labelpad=8);ax.set_ylabel('Draft checkpoint source',labelpad=8)
    ax.set_xticks(np.arange(-.5,4,1),minor=True);ax.set_yticks(np.arange(-.5,4,1),minor=True)
    ax.grid(which='minor',color='#d4d8df',linewidth=.9);ax.tick_params(which='minor',bottom=False,left=False)
    for spine in ax.spines.values():spine.set_visible(False)


def draw_matrix(ax,summary,mode,field,title,expected_n):
    values,counts,cells=matrix_data(summary,mode,field,expected_n)
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('white')
    picture=ax.imshow(np.ma.masked_invalid(values),vmin=0,vmax=100,cmap=cmap,interpolation='nearest')
    style_axis(ax);ax.set_title(title,pad=12,fontweight='bold')
    for i in range(4):
        for j in range(4):
            value=values[i,j]
            if np.isnan(value):
                ax.text(j,i,'missing' if counts[i,j]==0 else 'N/A\nn=%d/%d'%(counts[i,j],expected_n),
                    ha='center',va='center',fontsize=9,color='#777777')
            else:
                ax.text(j,i,'%.1f%%\nn=%d/%d'%(value,counts[i,j],expected_n),ha='center',va='center',
                    fontsize=10,fontweight='medium',color='white' if value<55 else '#101820',linespacing=1.5)
            if i==j:
                ax.add_patch(Rectangle((j-.475,i-.475),.95,.95,fill=False,edgecolor='#d55e00',linewidth=1.7))
    return picture,cells


def save_figure(fig,out,stem,input_hashes,title):
    provenance='; '.join('%s sha256=%s'%(Path(p).name,h) for p,h in input_hashes.items())
    fig.savefig(out/(stem+'.png'),dpi=200,bbox_inches='tight',facecolor='white',
                metadata={'Software':'matplotlib','Description':provenance})
    fig.savefig(out/(stem+'.pdf'),bbox_inches='tight',facecolor='white',
                metadata={'Title':title,'Subject':provenance,'Creator':'EASE plot_results.py'})
    plt.close(fig)


def discovery_figures(summary,episodes,out,input_hashes,expected_n):
    if any(int(r['init_id'])!=20 for r in episodes):
        raise ValueError('Discovery figures accept only init20; held-out data use the separate replication entrypoint')
    expected=36*expected_n
    completed=sum(int(r['n']) for r in summary)
    draft_completed=sum(int(r['n']) for r in summary if r['mode'] in ('draft','flash'))
    grid_complete=all(matrix_data(summary,m,'success_rate',expected_n)[1].min()==expected_n for m in ('draft','flash'))
    complete=grid_complete and completed==expected
    status='EXPLORATORY DISCOVERY — '+('COMPLETE' if complete else 'INCOMPLETE')
    progress='%d/%d total episodes; %d/%d draft-condition episodes'%(completed,expected,draft_completed,32*expected_n)
    summary_hash=next(h for p,h in input_hashes.items() if Path(p).name=='summary.json')
    footer=('Orange outline: source matches target. White cells: missing, never zero. Cell n = episodes.\n'
            'Same-target comparisons only; discovery init20. Input summary SHA256: '+summary_hash[:16]+'…')
    all_cells=[]
    fig,axes=plt.subplots(2,2,figsize=(12.8,11.8))
    fig.subplots_adjust(top=.85,bottom=.18,hspace=.40,wspace=.30)
    for ax,(stem,mode,field,title,definition) in zip(axes.flat,MATRICES):
        image,cells=draw_matrix(ax,summary,mode,field,title,expected_n);all_cells+=cells
        cb=fig.colorbar(image,ax=ax,fraction=.043,pad=.035);cb.set_label('%')
        ax.text(.5,-.23,definition,ha='center',va='top',transform=ax.transAxes,fontsize=8)
    fig.suptitle('EASE: fixed draft source × target suite',fontsize=17,fontweight='bold',y=.97)
    fig.text(.5,.928,status,ha='center',fontsize=14,fontweight='bold',color='#9d261d' if not complete else '#52616b')
    fig.text(.5,.897,progress,ha='center',fontsize=11)
    fig.text(.5,.035,footer,ha='center',fontsize=9,linespacing=1.5)
    save_figure(fig,out,'discovery_overview',input_hashes,status)
    extra=('flash_all_full','flash','full_round_fraction','FLASH total Full-route share',
           'All Full rounds / all requests; includes initial and periodic refresh')
    for stem,mode,field,title,definition in MATRICES+[extra]:
        fig,ax=plt.subplots(figsize=(7.1,6.9));fig.subplots_adjust(top=.77,bottom=.22,left=.14,right=.88)
        image,cells=draw_matrix(ax,summary,mode,field,title,expected_n)
        fig.colorbar(image,ax=ax,fraction=.046,pad=.035,label='%')
        fig.suptitle(status,fontsize=13,fontweight='bold',color='#9d261d' if not complete else '#52616b',y=.96)
        fig.text(.5,.905,progress,ha='center',fontsize=10)
        fig.text(.5,.125,definition,ha='center',fontsize=9)
        fig.text(.5,.055,footer,ha='center',fontsize=8,linespacing=1.4)
        save_figure(fig,out,stem,input_hashes,title)
        if field=='full_round_fraction':all_cells+=cells
    return dict(status='complete_exploratory_discovery' if complete else 'incomplete_exploratory_discovery',
                completed_episodes=completed,expected_episodes=expected,metric_cells=all_cells)


def replication_figures(analysis,selection_path,out,hashes,expected_n):
    obsolete='independent_replication' if not analysis or not selection_path else 'independent_replication_pending'
    for suffix in ('.png','.pdf'):
        stale=out/(obsolete+suffix)
        if stale.exists():stale.unlink()
    if not analysis or not selection_path:
        fig,ax=plt.subplots(figsize=(9.5,4.6));ax.axis('off')
        ax.text(.5,.72,'INDEPENDENT REPLICATION — NO DATA',ha='center',va='center',fontsize=17,fontweight='bold',color='#9d261d')
        ax.text(.5,.48,'Awaiting frozen source–target pair selection and init21–24 results.',ha='center',fontsize=12)
        ax.text(.5,.28,'Matching source / foreign source / Full\nSuccess with Wilson intervals; complete policy/RPC cost per executed action; episode wall\nNo discovery result is displayed as independent confirmation.',ha='center',fontsize=11,linespacing=1.7)
        save_figure(fig,out,'independent_replication_pending',hashes,'Independent replication: no data')
        return dict(status='pending',not_started_or_not_supplied=True)
    summary,episodes,new_hashes=load_analysis(analysis);hashes.update(new_hashes)
    selection_path=Path(selection_path);hashes[str(selection_path.resolve())]=digest(selection_path)
    selection=json.loads(selection_path.read_text());selected=selection['selected_pairs']
    if selection.get('status')=='none':
        return dict(status='not_selected',reason='No positive discovery gap; no independent pair fabricated')
    if not selected:raise ValueError('Selected pair list is empty')
    if any(int(r['init_id']) not in (21,22,23,24) for r in episodes):
        raise ValueError('Independent figure contains discovery or other initial states')
    required_costs={'policy_ms_per_executed_action','client_rpc_ms_per_executed_action','mean_episode_wall_seconds','successes'}
    if any(not required_costs.issubset(row) for row in summary):
        raise ValueError('Replication summary lacks per-executed-action service costs; regenerate it with the updated analyze.py')
    lookup={(r['mode'],r['source_suite'],r['target_suite']):r for r in summary}
    row_specs=[(pair,mode) for pair in selected for mode in ('draft','flash')]
    fig,axes=plt.subplots(len(row_specs),3,figsize=(17.0,4.1*len(row_specs)),squeeze=False)
    fig.subplots_adjust(top=.91,bottom=.075,hspace=.58,wspace=.30)
    plotted=[];all_complete=True
    for axes_row,(pair,mode) in zip(axes,row_specs):
        target,foreign=pair['target_suite'],pair['foreign_source_suite']
        full=[r for r in summary if r['mode']=='full' and r['target_suite']==target]
        if len(full)>1:raise ValueError('Ambiguous Full baseline for target '+target)
        conditions=[lookup.get((mode,target,target)),lookup.get((mode,foreign,target)),full[0] if full else None]
        labels=['Matching\n'+SHORT[target],'Foreign\n'+SHORT[foreign],'Full']
        colors=['#247a88','#cb7447','#626b73']
        for j,(row,label,color) in enumerate(zip(conditions,labels,colors)):
            n=int(row['n']) if row else 0
            if n>expected_n:raise ValueError('Replication count exceeds40 per condition')
            all_complete &= n==expected_n
            if row:
                sr=100*float(row['success_rate']);low=100*float(row['success_ci_low']);high=100*float(row['success_ci_high'])
                axes_row[0].bar(j,sr,color=color,width=.6)
                axes_row[0].errorbar(j,sr,yerr=np.array([[sr-low],[high-sr]]),fmt='none',ecolor='#202830',capsize=4)
                axes_row[0].text(j,104,'success %d/%d\nn=%d/%d'%(row['successes'],n,n,expected_n),ha='center',fontsize=8.5,linespacing=1.3)
                latency=row.get('policy_ms_per_executed_action')
                if latency is not None:axes_row[1].bar(j,float(latency),color=color,width=.6)
                rpc=row.get('client_rpc_ms_per_executed_action')
                if rpc is not None:axes_row[1].plot(j,rpc,marker='D',markersize=7,color='#1b2630',markerfacecolor='none',linestyle='none')
                axes_row[1].text(j,1.015,'n=%d/%d'%(n,expected_n),ha='center',transform=axes_row[1].get_xaxis_transform(),fontsize=9)
                wall=row.get('mean_episode_wall_seconds')
                if wall is not None:axes_row[2].bar(j,float(wall),color=color,width=.6)
                axes_row[2].text(j,1.015,'success %d/%d'%(row['successes'],n),ha='center',transform=axes_row[2].get_xaxis_transform(),fontsize=9)
                plotted.append(dict(target_suite=target,comparison_mode=mode,role=('matching','foreign','Full')[j],
                    n=n,successes=row['successes'],success_rate=row['success_rate'],policy_ms_per_executed_action=latency,client_rpc_ms_per_executed_action=rpc,
                    mean_episode_wall_seconds=wall,policy_metric='sum(complete native policy-call wall ms) / sum(actually executed actions), including zero-action rejection requests',cold_calls_included=True))
            else:
                for ax in axes_row:ax.text(j,.06,'missing',ha='center',transform=ax.get_xaxis_transform(),color='#777777')
                plotted.append(dict(target_suite=target,comparison_mode=mode,role=('matching','foreign','Full')[j],n=0,status='missing'))
        for ax in axes_row:
            ax.set_xticks(range(3),labels);ax.set_xlim(-.6,2.6);ax.spines[['top','right']].set_visible(False)
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        axes_row[0].set_ylim(0,122);axes_row[0].set_ylabel('Success rate (%)')
        axes_row[1].set_ylabel('Milliseconds / executed action')
        axes_row[2].set_ylabel('Seconds / episode')
        title='%s target | %s'%(SHORT[target],'Pure draft' if mode=='draft' else 'FLASH')
        axes_row[0].set_title(title+' — independent success',fontweight='bold',fontsize=11)
        axes_row[1].set_title(title+' — service cost per action',fontweight='bold',fontsize=11)
        axes_row[2].set_title(title+' — real episode wall',fontweight='bold',fontsize=11)
    status='INDEPENDENT REPLICATION — '+('COMPLETE' if all_complete else 'INCOMPLETE')
    fig.suptitle(status,fontsize=17,fontweight='bold',y=.976,color='#9d261d' if not all_complete else '#243544')
    fig.text(.5,.943,'Held-out init21–24 only; same-target matching/foreign comparisons. Full baseline repeated for context.',ha='center',fontsize=10)
    fig.text(.5,.026,'Success: Wilson95% intervals. Service cost: complete native policy-call ms per executed action; hollow diamond: client RPC ms per executed action.\n'
             'All requests count, including rejections with zero actions. Cold calls included. Mean request latency alone is not actual speedup; GPU stages are not end-to-end.\n'
             'Discovery data are excluded. Full input SHA256 values are recorded in figure_provenance.json.',ha='center',fontsize=9,linespacing=1.45)
    save_figure(fig,out,'independent_replication',hashes,status)
    return dict(status='complete' if all_complete else 'incomplete',plotted_conditions=plotted)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',required=True,help='Discovery analyze.py output directory')
    p.add_argument('--output',default=str(ROOT/'reports/ease_insights_20261103/draft_figures'))
    p.add_argument('--expected-n',type=int,default=10)
    p.add_argument('--replication-analysis')
    p.add_argument('--selection')
    p.add_argument('--replication-expected-n',type=int,default=40)
    a=p.parse_args()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
        'pdf.fonttype':42,'ps.fonttype':42,'figure.dpi':120})
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    summary,episodes,hashes=load_analysis(a.analysis)
    discovery=discovery_figures(summary,episodes,out,hashes,a.expected_n)
    replication=replication_figures(a.replication_analysis,a.selection,out,hashes,a.replication_expected_n)
    atomic_json(out/'matrix_cells.json',discovery.pop('metric_cells'))
    provenance=dict(producer=str(Path(__file__).resolve()),producer_sha256=digest(__file__),input_sha256=hashes,
        discovery=discovery,replication=replication,
        axes=dict(rows='source checkpoint',columns='target evaluation suite',suite_order=list(SUITES)),
        missing_data='White cells labeled missing; never imputed as zero',
        fallback_definition='rejection_full_fraction excludes initial and periodic Full; all-Full-route share saved separately',
        timing_definition='complete native policy/RPC ms per actually executed action, charging rejected zero-action requests; real episode wall and success counts alongside. Mean request latency alone is not actual speedup. GPU stage total_ms is not end-to-end.',
        statistical_scope='Discovery is exploratory; independent graphs require only init21–24',
        outputs=sorted(p.name for p in out.iterdir() if p.suffix in ('.png','.pdf')))
    atomic_json(out/'figure_provenance.json',provenance)
    note=('These incomplete figures are a mapping/rendering check, not a final scientific conclusion.' if discovery['status'].startswith('incomplete') else 'Discovery remains exploratory; assess the selected differences using independent confirmation.')
    text='# Draft source generalization figures\n\nStatus: **'+discovery['status']+'**, '+str(discovery['completed_episodes'])+'/'+str(discovery['expected_episodes'])+' episodes. '+note+'\n\n'
    text+='Rows are draft source; columns are fixed target suite. Every observed cell shows completed episode count. White missing cells are not failures or zero rates. The orange outline identifies the target-matching source.\n\n'
    text+='`flash_rejection_full` counts rejection-triggered Full requests; `flash_all_full` additionally includes initial/periodic refresh. Acceptance is conditional on actual verified draft rounds and is never reported for pure draft.\n\n'
    text+='Independent confirmation is pending unless separate held-out analysis and frozen selection are supplied. Its primary cost plot uses complete native policy-call milliseconds per actually executed action, with client RPC milliseconds per action marked separately. Rejected requests that execute zero actions still incur full cost. Mean request latency alone cannot establish acceleration; GPU stages are not end-to-end. Real episode wall and success counts appear alongside service cost. Cold requests and changed closed-loop trajectories remain included.\n\n'
    text+='Regenerate discovery: `python3 preparations/ease_insights_20261103/draft/plot_results.py --analysis <analyze-output> --output reports/ease_insights_20261103/draft_figures`. Add `--replication-analysis <heldout-analyze-output> --selection <frozen-selection.json>` for the independent comparison. Discovery accepts only init20; replication accepts only init21–24.\n\n'
    text+='Input file SHA256 values and metric definitions: [figure_provenance.json](figure_provenance.json). Exact plotted cells: [matrix_cells.json](matrix_cells.json). PNG and PDF are emitted for every figure.\n'
    (out/'README.md').write_text(text)
    print(json.dumps(dict(output=str(out.resolve()),discovery=discovery,replication_status=replication['status'],artifacts=len(provenance['outputs']))))

if __name__=='__main__':main()

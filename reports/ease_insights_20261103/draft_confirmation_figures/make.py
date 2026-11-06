"""Export accepted released-head transfer evidence without reading live path outcomes."""
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path('/root/nfs/embodied/benchmark');OUT=Path(__file__).resolve().parent
STATS=ROOT/'reports/ease_insights_20261103/draft_statistics/statistics.json'
AUDIT=ROOT/'reports/ease_insights_20261103/draft_confirmed_audit/audit.json'
PINS={STATS:'9e644f34bdb929557b4bda3a6dd036504fbbae6bad2245a870659374c9e623ad',
    AUDIT:'b183ec20b8622fc99ebf45a6cd300f6be19188eda34109ebc21b845bcfc4459b'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    for path,pin in PINS.items():
        if sha(path)!=pin:raise ValueError('Accepted transfer source changed: '+str(path))
    stats=json.loads(STATS.read_text());audit=json.loads(AUDIT.read_text())
    if stats['status']!='complete' or audit['status']!='independent_CPU_metrics_verified':raise ValueError('Need complete accepted confirmation')
    rows=audit['condition_metrics'];targets=['libero_object','libero_spatial'];names=['Object target','Spatial target']
    if len(rows)!=10 or sum(r['n'] for r in rows)!=400 or any(r['n']!=40 for r in rows):raise ValueError('Confirmation coverage differs')
    def row(target,role):
        mode='draft' if role.endswith('pure') else ('full' if role=='Full' else 'flash')
        selected=[r for r in rows if r['target_suite']==target and r['mode']==mode and
            (mode=='full' or (r['source_suite']==target)==role.startswith('matching'))]
        if len(selected)!=1:raise ValueError('Ambiguous figure role')
        return selected[0]
    for pair in stats['pairs']:
        success=next(r for r in pair['secondary'] if r['mode']=='flash' and r['subset']=='all4_init21_24' and r['metric']=='success')
        for role,key in [('matching_FLASH','matching_context'),('foreign_FLASH','foreign_context')]:
            a=row(pair['target_suite'],role);b=success[key]
            if a['successes']!=b['successes'] or abs(a['policy_ms_per_executed_action']-b['pooled_policy_ms_per_executed_action'])>1e-12:
                raise ValueError('Independent metrics and primary statistics disagree')
    roles=['matching_pure','foreign_pure','matching_FLASH','foreign_FLASH','Full']
    fields=['target_suite','role','source_suite','n','successes','acceptance_rate','rejection_full_request_fraction','native_infer_ms_per_executed_action','client_rpc_ms_per_executed_action']
    with (OUT/'confirmed_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for target in targets:
            for role in roles:
                r=row(target,role)
                writer.writerow(dict(target_suite=target,role=role,source_suite='' if role=='Full' else r['source_suite'],n=r['n'],successes=r['successes'],
                    acceptance_rate=r['acceptance_rate'],rejection_full_request_fraction=r['rejection_full_fraction'] if r['mode']=='flash' else '',
                    native_infer_ms_per_executed_action=r['policy_ms_per_executed_action'],client_rpc_ms_per_executed_action=r['client_rpc_ms_per_executed_action']))
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,2,figsize=(12.8,8.4));blue='#2b6f9e';orange='#d57b55';grey='#84919a';x=np.arange(2)
    def bars(ax,values,labels,colors,width=.24,counts=None):
        for k,(v,label,color) in enumerate(zip(values,labels,colors)):
            offset=(k-(len(values)-1)/2)*width;ax.bar(x+offset,v,width,color=color,label=label)
            for i,value in enumerate(v):
                text=(f'{counts[k][i]}/40' if counts is not None else f'{value:.2f}')
                ax.text(x[i]+offset,max(value,0)+(.8 if counts is not None else .18),text,ha='center',va='bottom',fontsize=8)
        ax.set_xticks(x,names);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax=axs[0,0];counts=[[row(t,role)['successes'] for t in targets] for role in ('matching_pure','foreign_pure')]
    bars(ax,[[c/40*100 for c in group] for group in counts],['Matching source','Foreign source'],[blue,orange],counts=counts)
    ax.set_ylim(0,113);ax.set_ylabel('Success (%)');ax.set_title('A  Pure released draft head',loc='left',fontweight='bold');ax.legend(loc='upper right',fontsize=8)
    ax.text(.01,-.23,'Holm2 case tests: Object p=1.82e-12; Spatial p=2.91e-11',transform=ax.transAxes,fontsize=8)
    ax=axs[0,1];counts=[[row(t,role)['successes'] for t in targets] for role in ('matching_FLASH','foreign_FLASH','Full')]
    bars(ax,[[c/40*100 for c in group] for group in counts],['Matching FLASH','Foreign FLASH','Full'],[blue,orange,grey],counts=counts)
    ax.set_ylim(0,113);ax.set_ylabel('Success (%)');ax.set_title('B  Full fallback preserves system success',loc='left',fontweight='bold');ax.legend(loc='lower left',fontsize=8)
    ax=axs[1,0];xx=np.arange(4);width=.32
    for sign,role,label,color in [(-1,'matching_FLASH','Matching FLASH',blue),(1,'foreign_FLASH','Foreign FLASH',orange)]:
        values=[row(t,role)['acceptance_rate']*100 for t in targets]+[row(t,role)['rejection_full_fraction']*100 for t in targets]
        ax.bar(xx+sign*width/2,values,width,color=color,label=label)
        for i,value in enumerate(values):ax.text(xx[i]+sign*width/2,value+1,f'{value:.1f}',ha='center',va='bottom',fontsize=8)
    ax.set_xticks(xx,['Object\nacceptance','Spatial\nacceptance','Object\nreject→Full','Spatial\nreject→Full']);ax.set_ylim(0,86);ax.set_ylabel('Percent; distinct denominators')
    ax.set_title('C  Action acceptance and request fallback',loc='left',fontweight='bold');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax=axs[1,1];values=[[row(t,role)['policy_ms_per_executed_action'] for t in targets] for role in ('matching_FLASH','foreign_FLASH','Full')]
    bars(ax,values,['Matching FLASH','Foreign FLASH','Full'],[blue,orange,grey]);ax.set_ylim(0,18);ax.set_ylabel('Native infer ms / executed action')
    ax.set_title('D  Observed complete service cost',loc='left',fontweight='bold')
    fig.suptitle('Released FLASH head transfer on fixed target suites',fontweight='bold',y=.99)
    fig.text(.03,.015,'Confirmation: ten fixed tasks per target, init21–24, n=40 per condition; 400 episodes. Common Pi0 inference contract.\n'
        'Acceptance is action-budget weighted; reject→Full is a request fraction. Timing includes zero-action rejection requests, cold compilation and shared load.\n'
        'Operational released-head portability evidence; training/preprocessing provenance remains incomplete. No unseen-task, all-draft or controlled deployment-speed claim.',fontsize=8)
    fig.tight_layout(rect=(0,.105,1,.96),h_pad=3.2,w_pad=2)
    for suffix in ('png','pdf','svg'):fig.savefig(OUT/('draft_transfer_confirmation.'+suffix),dpi=220,bbox_inches='tight')
    plt.close(fig)
    artifacts={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.suffix in ('.png','.pdf','.svg','.csv')}
    (OUT/'provenance.json').write_text(json.dumps(dict(inputs={str(p):v for p,v in PINS.items()},script_sha256=sha(__file__),artifacts=artifacts,
        normalization_note='Full source_suite metadata is not a draft-head origin and is left blank in the derived CSV.',
        current_path_outcomes_read=False,scientific_records_modified=False),indent=2)+'\n')
    print(json.dumps(dict(artifacts=list(artifacts),source_conditions=len(rows),confirmation_episodes=sum(r['n'] for r in rows))))


if __name__=='__main__':main()

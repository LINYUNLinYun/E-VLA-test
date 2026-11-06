"""Reporting-only summary figures from a completed quick-stage view; no raw/media IO."""
import argparse
import csv
import json
import math
from pathlib import Path
from common import SUITES, atomic_json, sha

BLUE = '#1f77b4'
ORANGE = '#ff7f0e'
COMPONENTS = ('action_generation_ms','server_total_ms','client_roundtrip_ms')


def read(path): return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline='') as stream: return list(csv.DictReader(stream))


def build_data(manifest,scope,success,latency):
    """Pure small-metadata mapping, also used by the CPU regression test."""
    conditions=manifest['conditions']; by={c['id']:c for c in conditions}
    if len(by)!=len(conditions) or scope.get('main_inits')!=[6,7,8]:
        raise ValueError('Duplicate condition or wrong main init cohort')
    keys=scope['main_keys']; records=scope['valid_records']
    if len(keys)!=len(set(keys)) or len(keys)!=120*len(conditions):
        raise ValueError('Main cohort must contain 120 unique episodes per condition')
    main=[]; identities=set()
    for key in keys:
        row=records[key]; identity=(row['condition'],row['suite'],row['task'],row['init'])
        if (row['key']!=key or row['condition'] not in by or row['model']!=by[row['condition']]['model']
                or row['status']!='valid' or not isinstance(row['success'],bool)
                or row['result']['success']!=row['success'] or row['suite'] not in SUITES
                or row['init'] not in (6,7,8) or row['task'] not in range(10) or identity in identities):
            raise ValueError('Noncanonical/duplicate/nonvalid main episode in reporting input')
        identities.add(identity); main.append(row)
    expected={(c,s,t,i) for c in by for s in SUITES for t in range(10) for i in (6,7,8)}
    if identities!=expected: raise ValueError('Main reporting cohort is not balanced')
    heat={}; seen=set(); suites=(*SUITES,'overall')
    for row in success:
        condition,suite=row['condition'],row['suite']; identity=(condition,suite)
        if condition not in by or suite not in suites or identity in seen:
            raise ValueError('Unknown/duplicate success CSV row')
        seen.add(identity); denominator=120 if suite=='overall' else 30
        selected=[e for e in main if e['condition']==condition and (suite=='overall' or e['suite']==suite)]
        wins=sum(e['success'] for e in selected)
        if (int(row['denominator'])!=denominator or len(selected)!=denominator
                or int(row['successes'])!=wins or not math.isclose(float(row['success_rate']),wins/denominator,abs_tol=1e-12)):
            raise ValueError('Success CSV values/denominators differ from main cohort')
        heat[identity]=dict(successes=wins,denominator=denominator,percent=100*wins/denominator)
    if seen!={(c,s) for c in by for s in suites}: raise ValueError('Missing success CSV cells')
    timing={}
    for row in latency:
        if row['component'] not in COMPONENTS: continue
        identity=(row['condition'],row['component'])
        if row['condition'] not in by or identity in timing: raise ValueError('Unknown/duplicate latency CSV row')
        condition=by[row['condition']]; denominator=640 if condition['model']=='db-cogact' else 160
        value=float(row['mean_ms'])
        if int(row['denominator'])!=denominator or not math.isfinite(value) or value<=0:
            raise ValueError('Wrong mixed latency denominator or nonpositive/nonfinite timing')
        timing[identity]=dict(mean_ms=value,denominator=denominator)
    if set(timing)!={(c,k) for c in by for k in COMPONENTS}: raise ValueError('Missing timing components')
    models=list(dict.fromkeys(c['model'] for c in conditions)); heatmaps=[]; comparisons=[]; failures=[]
    for model in models:
        group=[c for c in conditions if c['model']==model]
        heatmaps.append(dict(model=model,conditions=[c['id'] for c in group],masks=[c['mask'] for c in group],
            suites=list(suites),cells=[[heat[(c['id'],suite)] for suite in suites] for c in group]))
        for c in group:
            schedule=c['schedule']
            chosen=schedule['kind']=='native' or (model!='db-cogact' and schedule['num_steps']==2 and schedule.get('m')==.5)
            if chosen:
                comparisons.append(dict(condition=c['id'],model=model,mask=c['mask'],schedule=schedule,
                    timing={k:timing[(c['id'],k)] for k in COMPONENTS}))
        for condition in group:
            selected=[e for e in main if e['condition']==condition['id']]
            failed=[e for e in selected if not e['success']]; stages={}
            for entry in failed:
                phase=entry['result'].get('furthest_phase') or 'unknown'
                stages[phase]=stages.get(phase,0)+1
            failures.append(dict(condition=condition['id'],model=model,mask=condition['mask'],schedule=condition['schedule'],
                valid_episodes=len(selected),valid_failures=len(failed),
                furthest_counts=stages,interpretation='Furthest reached stage among valid failures, not failure cause'))
    return dict(schema=1,kind='quick_stage_reporting_only',main_episodes=len(main),main_inits=[6,7,8],
        success_heatmaps=heatmaps,latency_comparisons=comparisons,failure_furthest=failures,
        failure_figure_conditions=[row['condition'] for row in comparisons],
        excluded_extras=len(scope.get('extra_keys',[])),
        failure_source='Main canonical valid records with success=False only; not mixed-success failure_stage_counts.csv',
        latency_units='milliseconds, observed mean; no observer-effect subtraction',
        original_full_complete=False,independent_C01_C08_all_passed=False)


def render(view,output=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    view=Path(view).resolve(); output=Path(output).resolve() if output else view/'summary_figures'
    if output.parent!=view: raise ValueError('Reporting output must be a new direct child of the quick view')
    paths={name:view/name for name in ('condition_manifest.json','quick_stage_scope.json','success_by_condition_suite.csv','latency_breakdown.csv')}
    before={str(p):sha(p) for p in paths.values()}
    data=build_data(read(paths['condition_manifest.json']),read(paths['quick_stage_scope.json']),
        csv_rows(paths['success_by_condition_suite.csv']),csv_rows(paths['latency_breakdown.csv']))
    output.mkdir(parents=False,exist_ok=False)
    atomic_json(output/'plotted_data.json',data)
    groups=data['success_heatmaps']; height=max(5,max(len(g['conditions']) for g in groups)*.28+1.8)
    fig,axes=plt.subplots(1,len(groups),figsize=(5*len(groups),height),squeeze=False,layout='constrained')
    for ax,group in zip(axes[0],groups):
        values=np.asarray([[cell['percent'] for cell in row] for row in group['cells']])
        image=ax.imshow(values,vmin=0,vmax=100,cmap='YlGnBu',aspect='auto')
        ax.set_xticks(range(5),[s.replace('libero_','')+'\n'+('n=120' if s=='overall' else 'n=30') for s in group['suites']],fontsize=8)
        ax.set_yticks(range(len(group['conditions'])),group['conditions'],fontsize=7)
        for label,mask in zip(ax.get_yticklabels(),group['masks']): label.set_color(BLUE if mask=='full' else ORANGE)
        for i,row in enumerate(values):
            for j,value in enumerate(row): ax.text(j,i,f'{value:.0f}%',ha='center',va='center',fontsize=6,color='white' if value>65 else 'black')
        ax.set_title(group['model']+'; init 6/7/8')
    fig.colorbar(image,ax=axes.ravel().tolist(),label='Success rate (%)',shrink=.65)
    fig.suptitle('Balanced quick cohort: 120 episodes/condition; row labels blue=Full, orange=skip')
    fig.savefig(output/'01_success_heatmaps.png',dpi=150); plt.close(fig)
    comparisons=data['latency_comparisons']; fig,axes=plt.subplots(1,3,figsize=(15,max(4,len(comparisons)*.4+1)),sharey=True,layout='constrained')
    labels=[f"{r['model']} / {r['condition']} (D={r['schedule']['num_steps']})" for r in comparisons]
    for ax,component in zip(axes,COMPONENTS):
        values=[r['timing'][component]['mean_ms'] for r in comparisons]
        ax.barh(range(len(values)),values,color=[BLUE if r['mask']=='full' else ORANGE for r in comparisons])
        ax.set_yticks(range(len(labels)),labels,fontsize=7); ax.set_xlabel('Observed mean (ms)'); ax.set_title(component)
        ax.invert_yaxis(); ax.grid(axis='x',alpha=.2)
        ax.margins(x=.2)
        for i,value in enumerate(values): ax.text(value,i,f' {value:.2f}',va='center',fontsize=7)
    fig.legend(handles=[Patch(color=BLUE,label='Full'),Patch(color=ORANGE,label='Skip/variant')],loc='outside lower center',ncol=2)
    fig.suptitle('Native D vs D2/u50 where available; pi0 has only D2 here. DB n=640; pi0/X n=160 per condition')
    fig.savefig(output/'02_independent_latency_comparison.png',dpi=150); plt.close(fig)
    core={row['condition'] for row in comparisons}
    failures=[row for row in data['failure_furthest'] if row['condition'] in core]
    phases=sorted({p for row in failures for p in row['furthest_counts']},key=lambda p:(p=='unknown',p))
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained'); bottom=np.zeros(len(failures)); hatches=['','//','\\\\','xx','..','++','oo','--']
    for n,phase in enumerate(phases):
        counts=np.asarray([row['furthest_counts'].get(phase,0) for row in failures])
        ax.bar(range(len(failures)),counts,bottom=bottom,color=[BLUE if r['mask']=='full' else ORANGE for r in failures],
            edgecolor='white',hatch=hatches[n%len(hatches)],alpha=max(.35,1-n*.09))
        bottom+=counts
    ax.set_xticks(range(len(failures)),[row['model']+'\n'+row['condition'] for row in failures],fontsize=7,rotation=20)
    ax.set_ylabel('Valid failed episodes per condition (count)'); ax.set_title('Core native/D2/u50 conditions: furthest reached among valid failures; NOT cause')
    ax.margins(y=.15)
    for i,row in enumerate(failures): ax.text(i,row['valid_failures'],f"n={row['valid_failures']}",ha='center',va='bottom',fontsize=8)
    if phases: ax.legend(handles=[Patch(facecolor='lightgrey',edgecolor='grey',hatch=hatches[n%len(hatches)],label=p) for n,p in enumerate(phases)],title='Furthest stage',loc='upper left',bbox_to_anchor=(1,1))
    else: ax.text(.5,.5,'No valid failures',transform=ax.transAxes,ha='center')
    fig.savefig(output/'03_valid_failure_furthest_stage.png',dpi=150); plt.close(fig)
    for path,expected in before.items():
        if sha(path)!=expected: raise ValueError('Reporting inputs changed while rendering')
    outputs={p.name:sha(p) for p in output.iterdir() if p.is_file()}
    receipt=dict(schema=1,kind='reporting_only_not_new_scientific_acceptance',source_sha256=sha(__file__),
        inputs=before,outputs=outputs,main_episodes=data['main_episodes'],excluded_extras=data['excluded_extras'],
        scope='Existing CSV and scope metadata only; no predictions, raw, video, or scientific result modification',
        independent_C01_C08_all_passed=False)
    atomic_json(output/'reporting_receipt.json',receipt)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--view',required=True); parser.add_argument('--output')
    args=parser.parse_args(); print(render(args.view,args.output))


if __name__=='__main__': main()

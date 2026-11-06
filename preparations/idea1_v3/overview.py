"""Two final overview figures from explicitly named existing summary bundles.

Example (render once after final summaries are ready):
  python3 -m preparations.idea1_v3.overview --summary E1=/path/to/summary \
    --summary E2=/path/to/summary --summary E3=/path/to/summary \
    --summary E4=/path/to/summary --benchmark E1-D2=/path/to/bench \
    --output /path/to/final_figures --launch

Each summary bundle reads only success_by_condition_suite.csv and
latency_and_throughput.csv, filtered to its explicitly named E stage. Each bench
bundle reads only benchmark_summary.json, latency_by_route_setup.csv and its
optional pi05_stage_summary.json. No episode, checkpoint, raw or video reads.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np

SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
PANELS=('pi0 verification / Triton H50 R<=5','pi0.5 / native JAX H10 R<=5','FLASH learned / Triton H50 R<=5')


def panel(stage):
    return PANELS[1] if stage=='E2' else (PANELS[2] if stage=='E4' else PANELS[0])


def read_csv(path):
    with Path(path).open() as stream: return list(csv.DictReader(stream))


def parse_named(value):
    if '=' not in value: raise ValueError('Use NAME=/explicit/bundle/path')
    name,path=value.split('=',1)
    return name,Path(path).resolve()


def aggregate_success(rows,stage):
    result={}
    for row in rows:
        if row.get('stage')!=stage: continue  # Explicitly excludes P0/calibration and parent stages.
        suite=row['suite']
        if suite not in SUITES: raise ValueError('Only suite rows may enter aggregation, never an existing overall row')
        key=(stage,row['condition'],row.get('execution','A'))
        value=result.setdefault(key,dict(stage=stage,condition=row['condition'],execution=key[2],suites={}))
        success,n=int(row['successes']),int(row['episodes'])
        if not 0<=success<=n or n<=0: raise ValueError('Invalid success/n denominator')
        if suite in value['suites']: raise ValueError('Duplicate suite aggregate: use one consolidated summary per stage')
        value['suites'][suite]=dict(successes=success,episodes=n,inits=row.get('inits','unknown'))
    for value in result.values():
        value['successes']=sum(x['successes'] for x in value['suites'].values())
        value['episodes']=sum(x['episodes'] for x in value['suites'].values())
        value['success_rate']=value['successes']/value['episodes']
        value['panel']=panel(stage)
        value['coverage']=('selected200' if value['episodes']==200 else 'screen40') if stage=='E2' and value['episodes'] in (40,200) else f"n={value['episodes']}"
    return result


def aggregate_cost(rows,stage):
    result={}; seen=set()
    for row in rows:
        if row.get('stage')!=stage: continue
        key=(stage,row['condition'],row.get('execution','A'))
        identity=(*key,row['suite'])
        if identity in seen: raise ValueError('Duplicate suite cost aggregate')
        seen.add(identity)
        value=result.setdefault(key,dict(policy_ms=0.,actual_actions=0,episodes=0,ae_structures=set()))
        ms=float(row['policy_ms']); actions=int(row['actual_actions'])
        if not math.isfinite(ms) or ms<0 or actions<=0: raise ValueError('Invalid closed-loop policy cost')
        value['policy_ms']+=ms; value['actual_actions']+=actions; value['episodes']+=int(row['episodes'])
        if row.get('ae_compilation_structure'): value['ae_structures'].add(row['ae_compilation_structure'])
    for value in result.values(): value['policy_ms_per_actual_action']=value['policy_ms']/value['actual_actions']
    return result


def load_summary(stage,directory):
    if stage not in ('E1','E2','E3','E4'): raise ValueError('Summary name must explicitly select E1/E2/E3/E4')
    quality=aggregate_success(read_csv(directory/'success_by_condition_suite.csv'),stage)
    cost=aggregate_cost(read_csv(directory/'latency_and_throughput.csv'),stage)
    for key,row in quality.items():
        if key not in cost or cost[key]['episodes']!=row['episodes']: raise ValueError('Quality and cost coverage differ')
        row.update(cost[key]); row['ae_structures']=sorted(row['ae_structures']); row['source_summary']=str(directory)
    return quality


def load_benchmark(label,directory):
    summary=json.loads((directory/'benchmark_summary.json').read_text())
    config=summary['config']; stage=config['experiment']
    measured=[row for row in read_csv(directory/'latency_by_route_setup.csv') if row['kind']=='measurement']
    n=sum(int(row['n']) for row in measured)
    if n<=0 or n!=summary['measured_requests']: raise ValueError('Benchmark measured-request denominator differs')
    # Only means can be reconstructed exactly from route-group summaries.
    # Never average route medians/p95 values and call them an overall quantile.
    mean=sum(float(row['mean_ms'])*int(row['n']) for row in measured)/n
    diagnostic=None; path=directory/'pi05_stage_summary.json'
    if path.exists(): diagnostic=json.loads(path.read_text())
    return dict(label=label,stage=stage,panel=panel(stage),config=config,n=n,mean_request_ms=mean,
        boundary='fixed-observation sequence replay; virtual ack for E1/E3; not closed-loop route weights',
        setup='repeat0 excluded; measured requests may include prompt/runtime reconstruction',
        source_benchmark=str(directory),execution_safety=summary.get('execution_safety'),
        stage_revision=summary.get('pi05_stage_revision'),diagnostic=diagnostic)


def short_label(row):
    name=row['condition']
    match=re.match(r'(E[13])_D(\d+)_tau([\d.]+)_P(\d+)',name)
    if match:
        prefix='history ' if match[1]=='E3' else ''
        name=f"{prefix}D{match[2]} tau={match[3]} P{match[4]}"
    elif row['stage']=='E2': name=name.removeprefix('E2_')
    elif row['stage']=='E4': name='pure learned draft'
    return f"{name} [{row['coverage']}]"


def figures(rows,benchmarks,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups={name:sorted([row for row in rows if row['panel']==name],key=lambda row:row['condition']) for name in PANELS}
    active=[name for name in PANELS if groups[name] or any(b['panel']==name for b in benchmarks)]
    if not active: raise ValueError('No scientific main rows available')
    heights=[max(1.5,len(groups[name])*.55) for name in active]
    fig,axes=plt.subplots(len(active),1,figsize=(12,sum(heights)+2.4),squeeze=False,gridspec_kw={'height_ratios':heights})
    for axis,name in zip(axes[:,0],active):
        entries=groups[name]; matrix=np.full((max(1,len(entries)),5),np.nan)
        for i,row in enumerate(entries):
            for j,suite in enumerate(SUITES):
                if suite in row['suites']:
                    value=row['suites'][suite]; matrix[i,j]=value['successes']/value['episodes']
                    axis.text(j,i,f"{value['successes']}/{value['episodes']}",ha='center',va='center',fontsize=9)
            matrix[i,4]=row['success_rate']; axis.text(4,i,f"{row['successes']}/{row['episodes']}",ha='center',va='center',fontsize=9,fontweight='bold')
        image=axis.imshow(np.ma.masked_invalid(matrix),vmin=0,vmax=1,cmap='YlGnBu',aspect='auto',alpha=.55)
        axis.set_xticks(range(5),['spatial','object','goal','10','overall'])
        axis.set_yticks(range(max(1,len(entries))),[short_label(row) for row in entries] if entries else ['no main results'])
        axis.set_title(name,fontsize=11,loc='left')
        for i in range(len(entries)):
            for j in range(4):
                if np.isnan(matrix[i,j]): axis.text(j,i,'missing',ha='center',va='center',fontsize=8)
    fig.suptitle('New main success / valid episodes',fontsize=14)
    fig.text(.01,.015,'P0/calibration excluded. Overall = sum(successes) / sum(n), not mean suite rate. E2 screen40 and selected200 retain actual coverage.\nHistorical controls are not mixed into these denominators. Shared CPU/card load and execution revisions differ; no cross-backend ranking.',fontsize=8)
    fig.tight_layout(rect=(0,.075,1,.95)); fig.savefig(output/'success_overview.png',dpi=180); plt.close(fig)

    diagnostics=[b for b in benchmarks if b['stage']=='E2' and b.get('diagnostic')]
    count=len(active)+(1 if diagnostics else 0)
    fig=plt.figure(figsize=(14,3*len(active)+(2.8 if diagnostics else 0)+1.4))
    grid=fig.add_gridspec(count,2,height_ratios=[1]*len(active)+([.8] if diagnostics else []))
    for index,name in enumerate(active):
        left=fig.add_subplot(grid[index,0]); right=fig.add_subplot(grid[index,1])
        entries=groups[name]
        for i,row in enumerate(entries):
            left.scatter(row['policy_ms_per_actual_action'],row['success_rate'],s=40,
                label=f"{i+1}: {short_label(row)} {row['successes']}/{row['episodes']}")
            left.annotate(str(i+1),(row['policy_ms_per_actual_action'],row['success_rate']),
                xytext=(5,3+i%2*6),textcoords='offset points',fontsize=8)
        left.set_title(name+' | closed loop (mixed first-use/setup)',fontsize=9,loc='left')
        left.set_xlabel('sum(policy wall ms) / actual executed actions'); left.set_ylabel('success / valid episodes')
        left.set_ylim(-.04,1.12); left.grid(alpha=.2)
        if entries: left.legend(fontsize=6,loc='lower left',framealpha=.7)
        replay=[b for b in benchmarks if b['panel']==name]
        if replay:
            labels=[f"{b['label']} (n={b['n']})" for b in replay]
            right.barh(range(len(replay)),[b['mean_request_ms'] for b in replay],color='#c28b4c',alpha=.7)
            right.set_yticks(range(len(replay)),labels,fontsize=8)
            for i,b in enumerate(replay): right.text(b['mean_request_ms'],i,f" {b['mean_request_ms']:.1f}",va='center',fontsize=8)
            right.margins(x=.2)
        else: right.text(.05,.5,'Representative replay not yet available',transform=right.transAxes)
        right.set_title('Separate boundary: fixed-sequence request replay',fontsize=10,loc='left')
        right.set_xlabel('weighted mean ms / measured request; not deployment speedup')
        right.grid(axis='x',alpha=.2)
    if diagnostics:
        axis=fig.add_subplot(grid[-1,:]); names=('input_ms','image_embedding_ms','vlm_prefill_ms','ae_d10_ms','decode_ms')
        markers=('D','o','s','^','x')
        colors=('#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd')
        for metric,marker,color in zip(names,markers,colors):
            labelled=False
            for i,b in enumerate(diagnostics):
                value=b['diagnostic']['stages'].get(metric)
                if value:
                    axis.scatter(value['mean'],i,marker=marker,color=color,s=35,label=metric.removesuffix('_ms') if not labelled else None)
                    labelled=True
        axis.set_yticks(range(len(diagnostics)),[f"{b['label']} [{b['diagnostic'].get('stage_revision',b.get('stage_revision','unknown revision'))}]" for b in diagnostics],fontsize=7)
        axis.set_xlabel('diagnostic stage mean ms; input includes preprocessing/noise/sync (not CPU-only); DO NOT SUM as native total')
        axis.set_title('E2 only: diagnostic components; v2 scanned AE is not the old masked unrolled native total',fontsize=9,loc='left')
        axis.legend(fontsize=7,ncol=5,loc='upper right'); axis.grid(axis='x',alpha=.2)
    fig.suptitle('Quality and cost: model/protocol and timing boundaries kept separate',fontsize=13)
    fig.text(.01,.012,'Closed loop includes rejected draft + immediate Full and actual short-prefix replanning. Server startup is separate; request first-use/setup remains included.\nReplay uses saved real observations, virtual acknowledgements where applicable; 100 measured requests can still include runtime/shape setup (not pure hot latency).\nE2 Full native AE scan vs masked unrolled AE confounds old total latency; v2 scanned-suffix components are diagnostic only. Shared CPU/cards/revisions: no causal speed ratios.',fontsize=8)
    fig.tight_layout(rect=(0,.08,1,.95)); fig.savefig(output/'quality_cost_overview.png',dpi=180); plt.close(fig)


def self_test():
    rows=[dict(stage='E2',condition='E2_test',execution='A',suite='libero_spatial',successes='1',episodes='1'),
          dict(stage='E2',condition='E2_test',execution='A',suite='libero_goal',successes='1',episodes='9'),
          dict(stage='P0',condition='ignored',suite='libero_goal',successes='99',episodes='99'),
          dict(stage='calibration',condition='ignored',suite='libero_goal',successes='99',episodes='99')]
    quality=aggregate_success(rows,'E2')[('E2','E2_test','A')]
    assert quality['successes']==2 and quality['episodes']==10 and quality['success_rate']==.2
    costs=[dict(stage='E2',condition='E2_test',execution='A',suite='libero_spatial',policy_ms='10',actual_actions='1',episodes='1'),
           dict(stage='E2',condition='E2_test',execution='A',suite='libero_goal',policy_ms='150',actual_actions='9',episodes='9')]
    cost=aggregate_cost(costs,'E2')[('E2','E2_test','A')]
    assert cost['policy_ms_per_actual_action']==16 and cost['episodes']==10
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        (directory/'benchmark_summary.json').write_text(json.dumps(dict(config=dict(experiment='E2'),measured_requests=100)))
        with (directory/'latency_by_route_setup.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=['kind','n','mean_ms']); writer.writeheader()
            writer.writerows([dict(kind='setup_warmup',n=20,mean_ms=10000),dict(kind='measurement',n=10,mean_ms=10),dict(kind='measurement',n=90,mean_ms=20)])
        benchmark=load_benchmark('test',directory)
        assert benchmark['n']==100 and benchmark['mean_request_ms']==19
    assert panel('E2')!=panel('E4') and panel('E1')==panel('E3')
    print('CPU fixture: stage filters; overall2/10 not suite-average; cost160/10 actions; replay weighted19ms/100 excluding setup; model boundaries: OK')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',action='append',default=[],help='E1|E2|E3|E4=/explicit/summary/directory')
    parser.add_argument('--benchmark',action='append',default=[],help='short-label=/explicit/benchmark/directory')
    parser.add_argument('--output',type=Path); parser.add_argument('--launch',action='store_true'); parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test: self_test(); return
    if not args.summary: parser.error('At least one explicit --summary is required')
    records={}; inputs=[]
    for value in args.summary:
        stage,path=parse_named(value); incoming=load_summary(stage,path)
        if set(records)&set(incoming): raise ValueError('Overlapping stage/condition bundles: supply one consolidated final summary per stage')
        records.update(incoming); inputs.append(dict(stage=stage,summary=str(path)))
    benchmarks=[load_benchmark(*parse_named(value)) for value in args.benchmark]
    rows=list(records.values())
    if not args.launch:
        print(json.dumps(dict(dry_run=True,conditions=len(rows),benchmark_conditions=len(benchmarks),inputs=inputs,
            figures=['success_overview.png','quality_cost_overview.png'],episode_recomputation=False),indent=2)); return
    if args.output is None: parser.error('--output required with --launch')
    args.output.mkdir(parents=True,exist_ok=True)
    figures(rows,benchmarks,args.output)
    (args.output/'overview_inputs.json').write_text(json.dumps(dict(summaries=inputs,benchmarks=benchmarks,
        aggregation='sum successes / sum episodes; sum policy ms / sum actual actions; replay means weighted by request count',
        stage_filters=['E1','E2','E3','E4'],excluded=['P0','calibration','historical controls']),indent=2))
    print(json.dumps(dict(output=str(args.output),figures=2)))

if __name__=='__main__': main()

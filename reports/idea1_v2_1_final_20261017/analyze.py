"""Read-only E1 report reproduction; CPU only, no model/simulator imports."""
import csv, hashlib, json
import html
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/root/nfs/embodied/benchmark')
SRC=ROOT/'policies/realtime-vla-flash/results/idea1_v2_1_multimodel_kalman/run_20261016T045100Z/e1'
OUT=Path(__file__).resolve().parent
SUITES=['libero_spatial','libero_object','libero_goal','libero_10']
MODELS=['xiaomi-robotic0','db-cogact']
def read(p): return json.loads(Path(p).read_text())
def save(name,x): (OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvout(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def rel(p):return str(Path(p).relative_to(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def longest(mask):
    best=run=0;last=-2
    for k in mask:run=run+1 if k==last+1 else 1;best=max(best,run);last=k
    return best
def ci(delta):
    # Each suite consists of ten tasks, five paired initial states each.
    a=np.asarray(delta).reshape(4,10,5).mean(2)
    rng=np.random.default_rng(1701)
    idx=rng.integers(0,10,(2000,4,10))
    boot=np.take_along_axis(np.broadcast_to(a,(2000,4,10)),idx,axis=2).mean((1,2))*100
    return np.percentile(boot,[2.5,97.5]).tolist()
def figsave(name):plt.tight_layout();plt.savefig(OUT/name,dpi=170,bbox_inches='tight');plt.close()

def main():
    summary=[];suite_rows=[];task_rows=[];episode_rows=[];fails=[];audits={};probes_meta={};features=[];impacts=[];corr=[];hashes=[];allrecords={};conditions={};latency=[]
    for model in MODELS:
        d=SRC/model;ledger=read(d/'ledger.json');candidates=read(d/'candidates.json');p=read(d/'calibration/probes.json');depth=36 if model==MODELS[0] else 28
        cs={'full':{'mask':[],'algorithm':'full'},**{c['id']:c for c in candidates}};conditions[model]=list(cs)
        records={};missingfiles=[];sizes=Counter();phase=Counter();inconsistent=[]
        for e in ledger['entries']:
            if e['kind']!='formal':continue
            v=read(Path(e['directory'])/'validated.json');_,condition,s,t,i=e['key'].split('/');key=(condition,s,int(t),int(i))
            assert key not in records
            assert v['valid'] and e['status']=='valid'
            if e['success']!=v['success']:inconsistent.append(e['key'])
            records[key]={**v,'condition':condition,'ledger_key':e['key']}
            for name in ['video.mp4','telemetry.jsonl','infer.jsonl']:
                path=Path(e['directory'])/name
                if not path.is_file() or path.stat().st_size==0:missingfiles.append(rel(path))
                else:sizes[name]+=path.stat().st_size
            phase[v['furthest_phase']]+=not v['success']
        allrecords[model]=records
        expected={(c,s,t,i) for c in cs for s in SUITES for t in range(10) for i in range(5)}
        audits[model]={'formal':len(records),'exact_key_coverage':set(records)==expected,'missing_keys':len(expected-set(records)),'unexpected_keys':len(set(records)-expected),'ledger_status_counts':dict(Counter(e['status'] for e in ledger['entries'])),'ledger_kind_counts':dict(Counter(e['kind'] for e in ledger['entries'])),'ledger_validated_inconsistencies':inconsistent,'missing_or_empty_media':missingfiles,'formal_file_bytes':dict(sizes),'original_video_decoded_count':sum(r.get('video_decoded',False) for r in records.values()),'original_audited_count':sum(r.get('audited',False) for r in records.values()),'current_audit':'stat only; reused original per-episode validation decode/hash evidence, no new video decode or media hash'}
        for condition,c in cs.items():
            rs=[records[(condition,s,t,i)] for s in SUITES for t in range(10) for i in range(5)]
            fs=[records[('full',s,t,i)] for s in SUITES for t in range(10) for i in range(5)]
            delta=[int(r['success'])-int(f['success']) for r,f in zip(rs,fs)];lo,hi=ci(delta)
            nf=sum(f['success'] and not r['success'] for r,f in zip(rs,fs));rec=sum(not f['success'] and r['success'] for r,f in zip(rs,fs));fullsuccess=sum(f['success'] for f in fs)
            summary.append(dict(model=model,condition=condition,algorithm=c['algorithm'],n=200,success=sum(r['success'] for r in rs),sr_pct=sum(r['success'] for r in rs)/2,skip_count=len(c['mask']),skip_pct=100*len(c['mask'])/depth,mask_1based=json.dumps(c['mask']),longest_consecutive_skip=longest(c['mask']),protected_overlap=json.dumps(sorted(set(c['mask'])&set(p['protected_layers']+[depth]))),paired_delta_pp=np.mean(delta)*100,bootstrap_low_pp=lo,bootstrap_high_pp=hi,new_failures=nf,recoveries=rec,sr_ratio_to_full=sum(r['success'] for r in rs)/fullsuccess,full_success_retention=(fullsuccess-nf)/fullsuccess))
            for s in SUITES:
                group=[r for r in rs if r['suite']==s];sd=np.array([int(r['success'])-int(records[('full',s,r['task'],r['init'])]['success']) for r in group]).reshape(10,5).mean(1);draw=np.random.default_rng(1701).choice(sd,(2000,10)).mean(1)*100;slo,shi=np.percentile(draw,[2.5,97.5]);suite_rows.append(dict(model=model,condition=condition,suite=s,n=50,success=sum(r['success'] for r in group),sr_pct=2*sum(r['success'] for r in group),paired_delta_pp=float(sd.mean()*100),bootstrap_low_pp=float(slo),bootstrap_high_pp=float(shi)))
                for t in range(10):
                    g=[r for r in group if r['task']==t];task_rows.append(dict(model=model,condition=condition,suite=s,task=t,n=5,success=sum(r['success'] for r in g),sr_pct=20*sum(r['success'] for r in g)))
            for r in rs:
                f=records[('full',r['suite'],r['task'],r['init'])]
                episode_rows.append(dict(model=model,condition=condition,suite=r['suite'],task=r['task'],init=r['init'],success=r['success'],full_success=f['success'],furthest_phase=r['furthest_phase'],video=rel(Path(r['directory'])/'video.mp4'),telemetry=rel(Path(r['directory'])/'telemetry.jsonl')))
            fc=Counter(r['furthest_phase'] for r in rs if not r['success'])
            for ph,n in fc.items():fails.append(dict(model=model,condition=condition,furthest_phase=ph,failures=n,total_failures=sum(fc.values()),interpretation='furthest phase reached among failures; not the causal failure time'))
        probes_meta[model]={k:v for k,v in p.items() if k!='points'};probes_meta[model]['points_count']=len(p['points'])
        probes_meta[model]['source_episode_count']=len({str(Path(q['source']['input_path']).parents[1]) for q in p['points']})
        probes_meta[model]['source_inputs']=[{'index':q['index'],'input_path':rel(q['source']['input_path']),'phase':q['source'].get('phase'),'step':q['source']['step']} for q in p['points']]
        groups=defaultdict(list)
        for q in p['points']:
            for f in q['features']:groups[(f['layer'],f['kind'],f['tokens'])].append(f)
        for (layer,kind,tokens),g in groups.items():
            features.append(dict(model=model,layer=layer,kind=kind,tokens=tokens,n_observations=len(g),token_count_min=min(v['count'] for v in g),token_count_max=max(v['count'] for v in g),cosine_mean=float(np.mean([v['cosine'] for v in g])),update_rms_mean=float(np.mean([v['update_rms'] for v in g])),input_rms_mean=float(np.mean([v['input_rms'] for v in g])),relative_update_rms_mean=float(np.mean([v['update_rms']/max(v['input_rms'],1e-12) for v in g]))))
        for layer in range(1,depth+1):
            values=[q['impacts'][layer-1] for q in p['points'] if q['impacts'][layer-1] is not None]
            impacts.append(dict(model=model,layer=layer,n=len(values),action_command_rms_mean=float(np.mean(values)) if values else None,action_command_rms_p95=float(np.percentile(values,95)) if values else None))
        # Correlations across eligible layer means, descriptive only.
        for kind in ['hidden','attn','ffn']:
            for metric in ['cosine_mean','update_rms_mean','relative_update_rms_mean']:
                layers=range(4 if model==MODELS[0] else 3,depth);x=[];y=[]
                for layer in layers:
                    g=[f for f in features if f['model']==model and f['layer']==layer and f['kind']==kind];x.append(np.mean([f[metric] for f in g]));y.append(next(f['action_command_rms_mean'] for f in impacts if f['model']==model and f['layer']==layer))
                rank=lambda z:np.argsort(np.argsort(z))
                corr.append(dict(model=model,kind=kind,metric=metric,n_layers=len(x),pearson=float(np.corrcoef(x,y)[0,1]),spearman=float(np.corrcoef(rank(x),rank(y))[0,1]),scope='eligible-layer means; token groups equal weight; descriptive, no causality'))
        for path in [d/'ledger.json',d/'candidates.json',d/'calibration/probes.json',d/'readiness/native_protocol.json',d/'readiness/parity.json',d/'native_reference_comparison.json',d/'benchmark/manifest.json',d/'benchmark/completion_audit.json']:
            hashes.append(dict(path=rel(path),exists=path.exists(),sha256=sha(path) if path.exists() else None))
        # Fixed task0/init0 in each suite, complete condition set. Closed-loop parallel logs only.
        for condition in cs:
            vals=defaultdict(list);n_episodes=0
            for s in SUITES:
                r=records[(condition,s,0,0)];n_episodes+=1
                with (Path(r['directory'])/'infer.jsonl').open() as f:
                    for line in f:
                        event=json.loads(line)
                        for k,v in {**event.get('timing',{}),'client_roundtrip_ms':event.get('client_roundtrip_ms')}.items():
                            if isinstance(v,(int,float)):vals[k].append(v)
            for k,v in vals.items():latency.append(dict(model=model,condition=condition,component=k,n_predictions=len(v),n_episodes=n_episodes,p50_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),scope='parallel closed-loop task0/init0 each suite; variable observations; not fixed-observation or exclusive speedup'))
    csvout('e1_summary.csv',summary);csvout('e1_by_suite.csv',suite_rows);csvout('e1_by_task.csv',task_rows);csvout('e1_episodes.csv',episode_rows);csvout('failure_furthest_phase.csv',fails);csvout('probe_features.csv',features);csvout('probe_impacts.csv',impacts);csvout('probe_correlations.csv',corr);csvout('parallel_latency_sample.csv',latency)
    save('e1_summary.json',summary);save('audit.json',audits);save('probe_coverage.json',probes_meta);save('source_hashes.json',hashes)
    csvout('suite_at_least_90.csv',[r for r in suite_rows if r['sr_pct']>=90])
    budget=[]
    for model in MODELS:
        for k in sorted(set(r['skip_count'] for r in summary if r['model']==model and r['skip_count'])):
            rs=[r for r in summary if r['model']==model and r['skip_count']==k];best=max(r['sr_pct'] for r in rs)
            for rank,r in enumerate(sorted(rs,key=lambda r:(-r['sr_pct'],r['condition'])),1):budget.append(dict(model=model,skip_count=k,condition=r['condition'],algorithm=r['algorithm'],sr_pct=r['sr_pct'],rank_within_tested_budget=rank,gap_to_tested_best_pp=r['sr_pct']-best,scope='finite tested heuristic masks; posthoc descriptive ranking'))
    csvout('algorithm_by_budget.csv',budget)
    plot_summary(summary,suite_rows,features,impacts,fails)
    trajectories=plot_trajectories(allrecords,summary)
    benchmark=analyze_benchmark(summary)
    tables=['|Model|Condition|Success / 200|SR %|Δ pp [95% task bootstrap]|New fail / recovered|Full success retained|','|---|---|---:|---:|---|---|---:|']
    for r in summary:tables.append(f"|{r['model']}|{r['condition']}|{r['success']}|{r['sr_pct']:.1f}|{r['paired_delta_pp']:+.1f} [{r['bootstrap_low_pp']:.1f}, {r['bootstrap_high_pp']:.1f}]|{r['new_failures']} / {r['recoveries']}|{100*r['full_success_retention']:.1f}%|")
    (OUT/'analysis_tables.md').write_text('\n'.join(tables)+'\n')
    save('index.json',dict(source=rel(SRC),summary='e1_summary.json',audit='audit.json',probes='probe_coverage.json',benchmark=benchmark,trajectory_count=len(trajectories),figures=sorted(p.name for p in OUT.glob('*.png')),notes=['Bootstrap 2000 repetitions seed1701; stratify suites, resample tasks retaining five initial states; intervals are exploratory/unadjusted for mask selection.','Full-success retention=(Full success minus new failures)/Full successes; sr_ratio_to_full is a separate marginal SR ratio.','Xiaomi no exclusive benchmark; sampled parallel timings do not establish speedup.','Feature RMS in hidden activation units; action impact is RMS across decoded native 7D action commands (mixed coordinate scales, not physical mm); XYZ meters and same-time EEF distance millimeters.','Cosine ranking equally averages all hidden feature records across token groups and observations.','Furthest phase is retrospective maximum reached, not failure cause or onset.']))

def plot_summary(summary,suites,features,impacts,fails):
    fig,axs=plt.subplots(2,2,figsize=(15,10),gridspec_kw={'width_ratios':[1,1.3]})
    for j,m in enumerate(MODELS):
        rs=[r for r in summary if r['model']==m];labels=[r['condition'].replace('skip','') for r in rs]
        axs[j,0].barh(labels,[r['sr_pct'] for r in rs]);axs[j,0].set_xlim(0,102);axs[j,0].set_xlabel('Success (%) / 200 episodes');axs[j,0].set_title(m);axs[j,0].invert_yaxis()
        mat=[[next(r['sr_pct'] for r in suites if r['model']==m and r['condition']==c['condition'] and r['suite']==s) for s in SUITES] for c in rs]
        im=axs[j,1].imshow(mat,vmin=0,vmax=100,cmap='viridis',aspect='auto');axs[j,1].set_xticks(range(4),[s.replace('libero_','') for s in SUITES]);axs[j,1].set_yticks(range(len(rs)),labels)
        for y in range(len(rs)):
            for x in range(4):axs[j,1].text(x,y,f'{mat[y][x]:.0f}',ha='center',va='center',fontsize=8,color='white' if mat[y][x]<65 else 'black')
    fig.subplots_adjust(hspace=.3,wspace=.6,right=.88);cax=fig.add_axes([.91,.25,.018,.5]);fig.colorbar(im,cax=cax,label='Suite SR (%) / 50 episodes');fig.savefig(OUT/'e1_success_suite_heatmap.png',dpi=170,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,4))
    for ax,m in zip(axs,MODELS):
        for algorithm in dict.fromkeys(r['algorithm'] for r in summary if r['model']==m):
            rs=[r for r in summary if r['model']==m and r['algorithm']==algorithm];ax.plot([r['skip_pct'] for r in rs],[r['sr_pct'] for r in rs],'o-',label=algorithm)
        ax.set(title=m,xlabel='Skipped hidden-update layers / total depth (%)',ylabel='Success (%)');ax.legend(fontsize=7)
    figsave('e1_sr_vs_budget.png')
    fig,axs=plt.subplots(2,3,figsize=(15,8))
    for j,m in enumerate(MODELS):
        for tokens in ['visual','text','readout']:
            rs=[r for r in features if r['model']==m and r['kind']=='hidden' and r['tokens']==tokens];axs[j,0].plot([r['layer'] for r in rs],[r['cosine_mean'] for r in rs],label=tokens)
        for kind in ['hidden','attn','ffn']:
            ls=sorted(set(r['layer'] for r in features if r['model']==m));ys=[np.mean([r['relative_update_rms_mean'] for r in features if r['model']==m and r['layer']==l and r['kind']==kind]) for l in ls];axs[j,1].plot(ls,ys,label=kind)
        rs=[r for r in impacts if r['model']==m and r['n']];axs[j,2].plot([r['layer'] for r in rs],[r['action_command_rms_mean'] for r in rs],'.-');axs[j,0].set_ylabel(m+'\nMean cosine (unitless)');axs[j,1].set_yscale('log');axs[j,1].set_ylabel('Branch or residual RMS / input RMS');axs[j,2].set_ylabel('Single-layer action impact RMS\n(native 7D command units; mixed scales)')
        for ax in axs[j]:ax.set_xlabel('Original layer (1-based)')
        axs[j,0].legend(fontsize=8);axs[j,1].legend(fontsize=8)
    figsave('probe_layer_features.png')
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for ax,m in zip(axs,MODELS):
        depth=36 if m==MODELS[0] else 28
        for l in range(4 if m==MODELS[0] else 3,depth):
            x=np.mean([r['cosine_mean'] for r in features if r['model']==m and r['kind']=='hidden' and r['layer']==l]);y=next(r['action_command_rms_mean'] for r in impacts if r['model']==m and r['layer']==l);ax.scatter(x,y);ax.annotate(str(l),(x,y),fontsize=6)
        ax.set(title=m,xlabel='Mean hidden input/output cosine (unitless)',ylabel='Mean single-layer action impact (native command RMS)')
    figsave('probe_cosine_vs_impact.png')
    phases=['initial','approach','contact','grasp_or_manipulate','transport','goal_alignment']
    fig,axs=plt.subplots(1,2,figsize=(15,6))
    for ax,m in zip(axs,MODELS):
        cs=[r['condition'] for r in summary if r['model']==m];bottom=np.zeros(len(cs))
        for ph in phases+sorted(set(r['furthest_phase'] for r in fails)-set(phases)):
            v=np.array([sum(r['failures'] for r in fails if r['model']==m and r['condition']==c and r['furthest_phase']==ph) for c in cs]);ax.barh(cs,v,left=bottom,label=ph);bottom+=v
        ax.invert_yaxis();ax.set_title(m);ax.set_xlabel('Failed episodes by furthest phase reached');ax.legend(fontsize=7)
    figsave('failure_furthest_phase.png')

def plot_trajectories(records,summary):
    chosen={MODELS[0]:['skip8_impact_dispersed','skip10_impact_dispersed','skip12_short_runs'],MODELS[1]:['skip7_impact_dispersed','skip8_adjacent_pairs','skip9_adjacent_pairs']};rows=[]
    for m,cs in chosen.items():
        rr=records[m]
        for c in cs:
            for status in ['both_success','new_failure']:
                keys=[(s,t,i) for cc,s,t,i in rr if cc==c and rr[('full',s,t,i)]['success'] and rr[(c,s,t,i)]['success']==(status=='both_success')]
                if not keys:continue
                s,t,i=sorted(keys)[0];a=rr[('full',s,t,i)];b=rr[(c,s,t,i)];states=[]
                for r in [a,b]:
                    with (Path(r['directory'])/'telemetry.jsonl').open() as f:states.append([json.loads(l) for l in f])
                fig,axs=plt.subplots(1,4,figsize=(16,4));xyz=[np.array([v['eef_pos'] for v in q]) for q in states]
                for ax,(l,r) in zip(axs[:3],[(0,1),(0,2),(1,2)]):
                    ax.plot(xyz[0][:,l],xyz[0][:,r],'.-',color='tab:blue',alpha=.6,markersize=2,label='Full');ax.plot(xyz[1][:,l],xyz[1][:,r],'--',color='tab:orange',alpha=.6,label=c);ax.set_xlabel('XYZ'[l]+' (m)');ax.set_ylabel('XYZ'[r]+' (m)');ax.axis('equal')
                aa={r['step']:r for r in states[0]};bb={r['step']:r for r in states[1]};common=sorted(set(aa)&set(bb));dist=np.array([np.linalg.norm(np.array(aa[k]['eef_pos'])-bb[k]['eef_pos'])*1000 for k in common]);times=np.array(common)/20
                axs[3].plot(times,dist);axs[3].set_xlabel('Same elapsed control time (s)');axs[3].set_ylabel('EEF distance (mm)');axs[0].legend(fontsize=6)
                add_full_phase_background(axs[3],states[0],common,fig)
                sr=next(r['sr_pct'] for r in summary if r['model']==m and r['condition']==c);fig.suptitle(f'{m} | {c} SR {sr:.1f}% | {s} task{t:02} init{i:02} | Full success; variant {b["success"]}')
                name=f'trajectory_{m}_{c}_{status}.png';fig.tight_layout(rect=(0,.11,1,.9));fig.savefig(OUT/name,dpi=170,bbox_inches='tight');plt.close(fig)
                rows.append(dict(model=m,condition=c,selection_status=status,suite=s,task=t,init=i,condition_sr_pct=sr,full_success=True,variant_success=b['success'],full_actions=a['actions'],variant_actions=b['actions'],full_control_duration_s=a['actions']/20,variant_control_duration_s=b['actions']/20,figure=name,full_video=rel(Path(a['directory'])/'video.mp4'),variant_video=rel(Path(b['directory'])/'video.mp4'),full_telemetry=rel(Path(a['directory'])/'telemetry.jsonl'),variant_telemetry=rel(Path(b['directory'])/'telemetry.jsonl'),same_time_points=len(common),same_time_eef_p95_mm=float(np.percentile(dist,95)),selection_rule='lexicographically sorted suite/task/init; first qualifying pair; no time warping'))
    csvout('trajectory_index.csv',rows);write_gallery(rows);return rows

def add_full_phase_background(ax,full_states,common,fig):
    """Full-only coarse phases on common physical time; never align by phase."""
    from matplotlib.patches import Patch
    palette={'initial/approach':'#a6cee3','contact/grasp':'#fdbf6f','transport':'#b2df8a','alignment/release':'#cab2d6'}
    merge={'initial':'initial/approach','approach':'initial/approach','contact':'contact/grasp','grasp_or_manipulate':'contact/grasp','transport':'transport','goal_alignment':'alignment/release','release_or_final_action':'alignment/release','success':'alignment/release'}
    full={r['step']:r for r in full_states};phases=[merge.get(full[k].get('semantic_phase')) for k in common]
    start=0
    for stop in range(1,len(common)+1):
        if stop==len(common) or phases[stop]!=phases[start]:
            phase=phases[start]
            if phase in palette:ax.axvspan(common[start]/20,common[stop]/20 if stop<len(common) else common[-1]/20,color=palette[phase],alpha=.22,zorder=0)
            if start:ax.axvline(common[start]/20,color='grey',alpha=.35,linewidth=.65,zorder=1)
            start=stop
    fig.legend(handles=[Patch(facecolor=color,alpha=.45,label=label) for label,color in palette.items()],loc='lower center',bbox_to_anchor=(.5,.005),ncol=4,fontsize=8,title='Background on EEF-distance panel: Full phases (same simulation time)',title_fontsize=8,frameon=False)

def redraw_selected_trajectories():
    """Re-render only the indexed twelve pairs and retain their deterministic IDs."""
    with (OUT/'trajectory_index.csv').open() as stream:rows=list(csv.DictReader(stream))
    records={m:{} for m in MODELS}
    for r in rows:
        for side,condition in [('full','full'),('variant',r['condition'])]:
            directory=(ROOT/r[side+'_telemetry']).parent;v=read(directory/'validated.json');records[r['model']][(condition,r['suite'],int(r['task']),int(r['init']))]={**v,'condition':condition}
    updated=plot_trajectories(records,read(OUT/'e1_summary.json'))
    assert {(r['model'],r['condition'],r['suite'],int(r['task']),int(r['init'])) for r in rows}=={(r['model'],r['condition'],r['suite'],r['task'],r['init']) for r in updated}
    # Source input hashes remain independent of report-generation code/assets.
    for item in read(OUT/'source_hashes.json'):
        if item['exists']:assert sha(ROOT/item['path'])==item['sha256']
    generated=[Path(__file__),OUT/'trajectory_index.csv',OUT/'media_index.md',OUT/'gallery.html']+[OUT/r['figure'] for r in updated]
    save('trajectory_render_hashes.json',dict(full_phase_background=True,source_hashes_reverified=True,files=[dict(path=rel(p),sha256=sha(p)) for p in generated]))
    index=read(OUT/'index.json');index['trajectory_render_hashes']='trajectory_render_hashes.json';save('index.json',index)

def write_gallery(rows):
    md=['# Real paired media index','','Same task/init and physical control time; independent closed-loop rollouts have different observations and lengths. EEF difference uses common steps only, no time warping. Video playback clocks may differ from simulator clocks; physical durations below use actions / 20 Hz.','']
    page=['<!doctype html><html><meta charset="utf-8"><title>E1 paired rollout gallery</title><style>body{font:16px system-ui;max-width:1300px;margin:30px auto}section{margin:45px 0}video{width:48%;vertical-align:top}img{width:100%}small{display:block}</style><h1>E1 paired rollout gallery</h1><p>Same task/init and physical control time; independent closed-loop observations and lengths. No time warping. Physical durations = actions / 20 Hz; playback clocks can differ.</p>']
    for r in rows:
        title=f"{r['model']} · {r['condition']} (SR {r['condition_sr_pct']:.1f}%) · {r['suite']} task{r['task']} init{r['init']} · {r['selection_status']}"
        duration=f"Full: success, {r['full_actions']} actions / {r['full_control_duration_s']:.2f}s; variant success={r['variant_success']}, {r['variant_actions']} actions / {r['variant_control_duration_s']:.2f}s."
        md.extend([f'## {title}','',duration,'',f"![Measured EEF]({r['figure']})",'',f"[Full video](../../{r['full_video']}) · [Variant video](../../{r['variant_video']}) · [Full telemetry](../../{r['full_telemetry']}) · [Variant telemetry](../../{r['variant_telemetry']})",''])
        page.extend([f'<section><h2>{html.escape(title)}</h2><p>{html.escape(duration)}</p>',f'<video controls preload="metadata" src="../../{html.escape(r["full_video"])}"></video> <video controls preload="metadata" src="../../{html.escape(r["variant_video"])}"></video>',f'<small><a href="../../{r["full_telemetry"]}">Full telemetry</a> · <a href="../../{r["variant_telemetry"]}">Variant telemetry</a></small><img src="{r["figure"]}" alt="Measured XYZ and same-time EEF distance"></section>'])
    (OUT/'media_index.md').write_text('\n'.join(md));(OUT/'gallery.html').write_text('\n'.join(page)+ '</html>')

def analyze_benchmark(summary):
    d=SRC/'db-cogact/benchmark';samples=defaultdict(lambda:defaultdict(list));keys=set();observations=set();phases=Counter()
    with (d/'samples.jsonl').open() as f:
        for line in f:
            r=json.loads(line);assert r['key'] not in keys;keys.add(r['key']);phases[r['phase']]+=1
            if r['phase']!='measured':continue
            observations.add(r['observation'])
            for k,v in r['timing'].items():samples[r['condition']][k].append(v)
    rows=[]
    for c,metrics in samples.items():
        for k,v in metrics.items():
            base=np.median(samples['full'][k]);rows.append(dict(model='db-cogact',condition=c,component=k,n=len(v),p50_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),full_p50_div_variant_p50=float(base/np.median(v)),median_reduction_pct=float(100*(1-np.median(v)/base)),scope='existing exclusive fixed-observation measured benchmark; diagnostic hooks active'))
    csvout('db_exclusive_benchmark.csv',rows)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,k in zip(axs,['vlm_ms','client_roundtrip_ms']):
        for r in rows:
            if r['component']!=k:continue
            sr=next(s['sr_pct'] for s in summary if s['model']=='db-cogact' and s['condition']==r['condition']);focus=r['condition'] in ['full','skip7_impact_dispersed','skip8_impact_dispersed','skip9_adjacent_pairs'];ax.scatter(r['p50_ms'],sr,color='tab:blue' if focus else 'grey',s=32 if focus else 15,alpha=1 if focus else .6)
            if focus:
                offset={'full':(-42,-15),'skip7_impact_dispersed':(4,-12),'skip8_impact_dispersed':(4,-12),'skip9_adjacent_pairs':(4,-17)}[r['condition']];ax.annotate(r['condition'].replace('skip',''),(r['p50_ms'],sr),xytext=offset,textcoords='offset points',fontsize=7)
        ax.set_xlabel('Exclusive '+k+' p50 (ms)');ax.set_ylabel('Closed-loop success (%)');ax.set_title('DB-CogACT: separate benchmark and formal episodes')
    fig.suptitle('All 13 conditions shown; four highlighted labels, remaining masks indexed in CSV');figsave('db_benchmark_tradeoff.png')
    completion=read(d/'completion_audit.json');info=dict(db_completion=completion,observed_sample_phases=dict(phases),unique_sample_keys=len(keys),measured_observation_count=len(observations),sample_sha256=sha(d/'samples.jsonl'),manifest_sha256=sha(d/'manifest.json'),xiaomi='benchmark directory absent; no exclusive speedup conclusion');assert info['sample_sha256']==completion['samples_sha256'];assert info['manifest_sha256']==completion['source_manifest_sha256']
    save('benchmark_audit.json',info);return info

if __name__=='__main__':main()

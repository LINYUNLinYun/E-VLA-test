"""All formal actual EEF telemetry, streaming extraction; no video decode/inference."""
from analyze import *
def extract(directory):
    out=[]
    with (Path(directory)/'telemetry.jsonl').open() as f:
        for line in f:
            r=json.loads(line);out.append((r['step'],r['eef_pos'],r.get('semantic_phase'),any(v is True for v in r.get('robot_grasps',{}).values())))
    return out
def main():
    pairs=[];episodes=[];phasecounts=[]
    for m in MODELS:
        entries=[e for e in read(SRC/m/'ledger.json')['entries'] if e['kind']=='formal' and e['status']=='valid'];cache={};full_entries={}
        for e in entries:
            _,c,s,t,i=e['key'].split('/')
            if c=='full':cache[(s,t,i)]=extract(e['directory']);full_entries[(s,t,i)]=e
        print(m,'Full cache ready',len(cache),flush=True)
        for ix,e in enumerate(entries):
            _,c,s,t,i=e['key'].split('/');full=cache[(s,t,i)];variant=full if c=='full' else extract(e['directory']);losssteps=[b[0] for a,b in zip(variant,variant[1:]) if a[3] and not b[3]]
            episodes.append(dict(model=m,condition=c,suite=s,task=int(t),init=int(i),success=e['success'],grasp_to_not_grasp_event=bool(losssteps),grasp_to_not_grasp_event_count=len(losssteps),first_grasp_to_not_grasp_step=losssteps[0] if losssteps else None,interpretation='grasp transition candidate includes normal placement/release; not failure cause'))
            if c=='full':continue
            aa={r[0]:r for r in full};bb={r[0]:r for r in variant};common=sorted(set(aa)&set(bb));dist=np.array([np.linalg.norm(np.array(aa[k][1])-bb[k][1])*1000 for k in common]);hit=None
            for j in range(max(0,len(common)-4)):
                if common[j+4]-common[j]==4 and np.all(dist[j:j+5]>50):hit=j;break
            k=common[hit] if hit is not None else None;f=full_entries[(s,t,i)]
            pairs.append(dict(model=m,condition=c,suite=s,task=int(t),init=int(i),full_success=f['success'],variant_success=e['success'],common_states=len(common),full_states=len(full),variant_states=len(variant),same_time_eef_p50_mm=float(np.median(dist)),same_time_eef_p95_mm=float(np.percentile(dist,95)),first_sustained_gt50mm_step=k,first_sustained_gt50mm_time_s=k/20 if k is not None else None,first_deviation_full_phase=aa[k][2] if k is not None else None,first_deviation_variant_phase=bb[k][2] if k is not None else None,variant_grasp_to_not_grasp_event=bool(losssteps),variant_grasp_to_not_grasp_event_count=len(losssteps)))
            if ix%500==0:print(m,'processed',ix,'/',len(entries),flush=True)
    csvout('pair_trajectory_metrics.csv',pairs);csvout('failure_first_deviation.csv',[r for r in pairs if not r['variant_success']]);csvout('all_episode_grasp_events.csv',episodes)
    agg=[];grasp=[]
    for m in MODELS:
        for c in dict.fromkeys(r['condition'] for r in episodes if r['model']==m):
            for success in [True,False]:
                rs=[r for r in pairs if r['model']==m and r['condition']==c and r['variant_success']==success];es=[r for r in episodes if r['model']==m and r['condition']==c and r['success']==success]
                if rs:
                    triggered=[r for r in rs if r['first_sustained_gt50mm_step'] is not None];agg.append(dict(model=m,condition=c,variant_success=success,n_episodes=len(rs),mean_episode_eef_p50_mm=float(np.mean([r['same_time_eef_p50_mm'] for r in rs])),mean_episode_eef_p95_mm=float(np.mean([r['same_time_eef_p95_mm'] for r in rs])),sustained_deviation_episodes=len(triggered),sustained_deviation_pct=100*len(triggered)/len(rs),median_trigger_step=float(np.median([r['first_sustained_gt50mm_step'] for r in triggered])) if triggered else None))
                    for side in ['full','variant']:
                        for phase,n in Counter(r[f'first_deviation_{side}_phase'] for r in triggered).items():phasecounts.append(dict(model=m,condition=c,variant_success=success,phase_source=side,phase=phase,episodes=n))
                if es:grasp.append(dict(model=m,condition=c,success=success,n_episodes=len(es),grasp_to_not_grasp_event_episodes=sum(r['grasp_to_not_grasp_event'] for r in es),event_episode_pct=100*np.mean([r['grasp_to_not_grasp_event'] for r in es])))
    csvout('trajectory_metric_summary.csv',agg);csvout('first_deviation_phase_counts.csv',phasecounts);csvout('grasp_event_by_success.csv',grasp)
    fig,axs=plt.subplots(2,2,figsize=(16,11))
    for j,m in enumerate(MODELS):
        cs=list(dict.fromkeys(r['condition'] for r in pairs if r['model']==m));y=np.arange(len(cs))
        for success,offset,color in [(True,-.18,'tab:blue'),(False,.18,'tab:orange')]:
            vals=[next((r['mean_episode_eef_p95_mm'] for r in agg if r['model']==m and r['condition']==c and r['variant_success']==success),np.nan) for c in cs];axs[j,0].barh(y+offset,vals,height=.34,color=color,label='variant success' if success else 'variant failure')
            vals=[next((r['sustained_deviation_pct'] for r in agg if r['model']==m and r['condition']==c and r['variant_success']==success),np.nan) for c in cs];axs[j,1].barh(y+offset,vals,height=.34,color=color,label='variant success' if success else 'variant failure')
        for ax in axs[j]:ax.set_yticks(y,cs,fontsize=8);ax.invert_yaxis();ax.set_title(m);ax.legend(fontsize=8)
        axs[j,0].set_xlabel('Mean of episode same-time actual EEF p95 distance (mm)');axs[j,1].set_xlabel('Episodes with >50 mm for 5 consecutive states (%)')
    fig.suptitle('Actual EEF telemetry, independent closed-loop rollouts; common simulation steps only\nDescriptive threshold: 5 states span 0.2 s at 20 Hz; no time warping, not causal diagnosis')
    figsave('trajectory_deviation_summary.png')
    fig,axs=plt.subplots(1,2,figsize=(15,6))
    for ax,m in zip(axs,MODELS):
        cs=list(dict.fromkeys(r['condition'] for r in episodes if r['model']==m));y=np.arange(len(cs))
        for success,offset,color in [(True,-.18,'tab:blue'),(False,.18,'tab:orange')]:
            vals=[next((r['event_episode_pct'] for r in grasp if r['model']==m and r['condition']==c and r['success']==success),np.nan) for c in cs];ax.barh(y+offset,vals,height=.34,color=color,label='success' if success else 'failure')
        ax.set_yticks(y,cs,fontsize=8);ax.invert_yaxis();ax.set(title=m,xlabel='Episodes with grasp → not-grasp event (%)');ax.legend()
    fig.suptitle('Event candidate includes ordinary placement / release; not evidence of failure cause');figsave('grasp_event_by_success.png')
    save('trajectory_metric_audit.json',dict(formal_episodes=len(episodes),paired_variants=len(pairs),expected_formal=5200,expected_pairs=4800,complete=len(episodes)==5200 and len(pairs)==4800,threshold='strict >50mm, five consecutive telemetry states / 0.2s span at20Hz',weighting='summary means equally weight episodes, each episode has its own p50/p95 over common simulation steps; not pooled-state quantiles',units='actual measured EEF Euclidean separation mm; unlike E4 command displacement mm',null_rule='no threshold trigger or absent success group remains null / absent; no inference of causal failure'))
    index=read(OUT/'index.json');index['figures']=sorted(p.name for p in OUT.glob('*.png'));index['all_trajectory_metrics']='pair_trajectory_metrics.csv';index['trajectory_metric_audit']='trajectory_metric_audit.json';index['e4_oracle']='e4_oracle_summary.json';index['gallery']='gallery.html';save('index.json',index)
if __name__=='__main__':main()

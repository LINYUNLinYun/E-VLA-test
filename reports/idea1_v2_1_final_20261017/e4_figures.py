"""Render existing E4 tables and summarize stored oracle diagnostics; no inference."""
from analyze import *
E4=SRC.parent/'e4_revision3/e4_kalman'
def loadcsv(p):
    with p.open() as f:return list(csv.DictReader(f))
def render_horizon():
    rows=[r for r in loadcsv(E4/'decomposition/offset_interval_summary.csv') if r['metric']=='l2_mm' and r['window'].isdigit()]
    pairs=list(dict.fromkeys(r['pair'] for r in rows));fig,axs=plt.subplots(1,2,figsize=(14,5))
    for pair in pairs:
        rr=sorted([r for r in rows if r['pair']==pair],key=lambda r:int(r['window']));x=[int(r['window']) for r in rr]
        for ax in axs:
            line=ax.plot(x,[float(r['equal_mean']) for r in rr],label=pair)[0];ax.plot(x,[float(r['sample_p95']) for r in rr],'--',color=line.get_color(),alpha=.55)
    for ax,n in zip(axs,[12,50]):
        ax.axvspan(1,5,color='grey',alpha=.15);ax.set_xlim(1,n);ax.set_xlabel('Action offset (1-based)');ax.set_ylabel('Command-space L2 difference (mm)');ax.grid(alpha=.2)
        visible_max=max(float(r[k]) for r in rows if int(r['window'])<=n for k in ['equal_mean','sample_p95']);ax.set_ylim(0,visible_max*1.1)
    axs[0].legend(fontsize=8);axs[0].set_title('Offsets 1–12');axs[1].set_title('Offsets 1–50')
    fig.suptitle('Solid: episode → task → suite equal mean; dashed: pooled sample p95 (different weights)\nShaded: nominal offsets 1–5 (final executed prefix may be shorter)')
    figsave('e4_horizon.png')

def main():
    render_horizon()
    rows=loadcsv(E4/'supplemental/phase_geometry_collapsed.csv');ss=['all']+SUITES;phases=sorted(set(r['phase'] for r in rows));geoms=sorted(set(r['geometry'] for r in rows));vmax=max(float(r['equal_l2_mm']) for r in rows)
    fig=plt.figure(figsize=(22,14));gs=fig.add_gridspec(3,2);axes=[fig.add_subplot(gs[0,:])]+[fig.add_subplot(gs[i//2+1,i%2]) for i in range(4)]
    for ax,s in zip(axes,ss):
        mat=np.full((len(phases),len(geoms)),np.nan);lookup={}
        for r in rows:
            if r['suite']==s:mat[phases.index(r['phase']),geoms.index(r['geometry'])]=float(r['equal_l2_mm']);lookup[(r['phase'],r['geometry'])]=r
        im=ax.imshow(mat,vmin=0,vmax=vmax,cmap='viridis',aspect='auto');ax.set_xticks(range(len(geoms)),[g.replace('_',' ') for g in geoms],fontsize=10);ax.set_yticks(range(len(phases)),phases,fontsize=10);ax.set_title(s,fontsize=14)
        for (p,g),r in lookup.items():
            val=float(r['equal_l2_mm']);star='*' if r['descriptive_only']=='True' else '';ax.text(geoms.index(g),phases.index(p),f'{val:.2f}{star}\nn={r["replans"]}; t={r["tasks"]}; e={r["episodes"]}',ha='center',va='center',fontsize=9,color='white' if val<vmax*.55 else 'black')
    fig.subplots_adjust(left=.12,right=.9,hspace=.45,wspace=.42,top=.91,bottom=.05);cax=fig.add_axes([.92,.15,.015,.65]);fig.colorbar(im,cax=cax,label='Kalman–Full L2 mean (mm)');fig.suptitle('E4 phase × historical geometry | same color scale\nCell: hierarchical mean; n=replans, t=tasks, e=episodes; *=source low-coverage descriptive flag; blank=missing',fontsize=15);fig.savefig(OUT/'e4_phase_geometry.png',dpi=170,bbox_inches='tight');plt.close(fig)
    records=read(E4/'evaluation/records.json');summ=[];sources=[]
    for p in sorted((E4/'decomposition').glob('*_time_oracle.json')):
        oracle=read(p);assert len(oracle)==len(records);metrics=['zero_error_mm','residual_mm','improvement_mm','residual_lateral_mm'];paired=list(zip(records,oracle))
        for suite in ['all']+SUITES:
            group=[(r,o) for r,o in paired if suite=='all' or r['suite']==suite];out=dict(pair=p.name.replace('_time_oracle.json',''),suite=suite,n_samples=len(group),n_episodes=len({(r['suite'],r['task'],r['init']) for r,o in group}),n_tasks=len({(r['suite'],r['task']) for r,o in group}),best_shift_sample_counts=dict(Counter(str(o['best_shift']) for r,o in group)),unidentifiable_samples=sum(o.get('unidentifiable',False) for r,o in group))
            for metric in metrics:
                eps=defaultdict(list)
                for r,o in group:
                    if o.get(metric) is not None:eps[(r['suite'],r['task'],r['init'])].append(o[metric])
                tasks=defaultdict(list)
                for (s,t,i),v in eps.items():tasks[(s,t)].append(np.mean(v))
                suites=defaultdict(list)
                for (s,t),v in tasks.items():suites[s].append(np.mean(v))
                out[metric]=float(np.mean([np.mean(v) for v in suites.values()]))
            summ.append(out)
        sources.append(dict(path=rel(p),sha256=sha(p)))
    save('e4_oracle_summary.json',dict(rows=summ,source_hashes=sources,records_sha256=sha(E4/'evaluation/records.json'),weighting='replans within episode → episodes within task → tasks within suite → suites; shift counts are pooled raw samples',interpretation='posthoc oracle, nondeployable; positive shift matches predicted later offset to Full current offset; ±3 offsets; reference offsets stored zero-based [4..9] = one-based [5..10]'))
    save('e4_figure_sources.json',[dict(path=rel(p),sha256=sha(p)) for p in [E4/'decomposition/offset_interval_summary.csv',E4/'supplemental/phase_geometry_collapsed.csv']])
if __name__=='__main__':main()

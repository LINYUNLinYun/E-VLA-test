"""Standalone scientific plots from passed artifacts only (CPU)."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

R=Path(__file__).resolve().parent
P=R.parents[2]/'reports/idea2_progressive_kv_20261025'
P.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

def read(name):
    path=R/name/'result.json'
    if not path.exists():return None
    value=json.loads(path.read_text())
    return value if value.get('status')=='passed' else None

def save(fig,name):
    fig.savefig(P/(name+'.png'),dpi=180,bbox_inches='tight')
    fig.savefig(P/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)

formal=[read('formal_same_task_gpu2'),read('repeat_same_task_gpu2')]
if all(formal):
    fig,ax=plt.subplots(1,2,figsize=(11,4.2),gridspec_kw={'width_ratios':[1,1.2]})
    leads=[0,1,3,6,18]
    for index,d in enumerate(formal):
        y=[d['summary'][f'lead{lead}/encoder']['trajectory_ms']['mean'] for lead in leads]
        ax[0].plot(range(5),y,'o-',label=f'Run {index+1}')
    ax[0].set_xticks(range(5),['0','1','3','6','18 (fresh)'])
    ax[0].set_xlabel('Minimum VLM lead (KV-ready layers)')
    ax[0].set_ylabel('8-request all-work time (ms)')
    ax[0].set_title('Same GPU, same Graphs, same Host lookahead')
    ax[0].legend();ax[0].grid(alpha=.2)
    h=np.array([np.mean([d['summary'][f'lead{lead}/encoder']['mean_new_layers_by_step'] for d in formal],axis=0) for lead in leads])
    im=ax[1].imshow(h,vmin=0,vmax=18,cmap='viridis',aspect='auto')
    ax[1].set_yticks(range(5),[f'lead {x}' for x in leads]);ax[1].set_xticks(range(10),range(1,11))
    ax[1].set_xlabel('AE denoising iteration');ax[1].set_title('New KV layers used (of 18)')
    fig.colorbar(im,ax=ax[1],label='New layers')
    fig.tight_layout();save(fig,'lead_sweep')

controls=[read('gpu0_full_control_formal'),read('gpu0_full_control_formal_repeat')]
sm16=[read('gpu0_sm16_lead3_formal'),read('gpu0_sm16_lead3_repeat')]
screen32=read('gpu0_resource_sm32_lead3_screen_clean')
if all(controls+sm16) and screen32:
    sets=[controls,sm16,[screen32]];labels=['Full GPU\nlead 3','AE16 / Prefix112\nlead 3','AE32 / Prefix96\nlead 3 (screen)']
    fields=['trajectory_ms','exposed_ae_after_vlm_ms','ae_finish_minus_next_encoder_ms']
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key,title in zip(axes,fields,['8-request all-work time','AE tail after current VLM','AE end minus next Encoder end']):
        means=[np.mean([d['summary']['lead3/encoder'][key]['mean'] for d in ds]) for ds in sets]
        ax.bar(range(3),means,color=['#4477AA','#228833','#CCBB44'])
        for i,y in enumerate(means):ax.text(i,y+(.8 if y>=0 else -.8),f'{y:.1f}',ha='center',va='bottom' if y>=0 else 'top')
        ax.set_xticks(range(3),labels,fontsize=8);ax.set_ylabel('ms');ax.set_title(title);ax.axhline(0,color='black',lw=.7)
    fig.suptitle('GPU0 only; full/AE16 confirmed twice, AE32 is 20-group screening',fontsize=11)
    fig.tight_layout();save(fig,'resource_tradeoff')

trace_dirs=[R/'trace_analysis'/n for n in ['gpu0_full','gpu0_sm16','gpu0_sm32']]
if all((p/'report.json').exists() for p in trace_dirs):
    fig,axes=plt.subplots(3,1,figsize=(12,7),sharex=True)
    colors={'vlm':'#4477AA','ae':'#EE6677','next_vision':'#228833'}
    for ax,directory,title in zip(axes,trace_dirs,['Full GPU, lead 3','AE16 / Prefix112, lead 3','AE32 / Prefix96, lead 3']):
        report=json.loads((directory/'report.json').read_text())
        phases={x['phase']:x for x in report['phases'] if x['request']==3}
        zero=phases['vlm']['start_ns']
        ks=list(csv.DictReader((directory/'kernels.csv').open()))
        for y,phase in enumerate(['vlm','ae','next_vision']):
            if phase not in phases:continue
            p=phases[phase];stage={'vlm':'VLM','ae':'AE','next_vision':'Vision'}[phase]
            intervals=[]
            for k in ks:
                if k['stage']==stage and int(k['start'])>=p['start_ns'] and int(k['end'])<=p['end_ns']:
                    intervals.append(((int(k['start'])-zero)/1e6,(int(k['end'])-int(k['start']))/1e6))
            ax.broken_barh([((p['start_ns']-zero)/1e6,(p['end_ns']-p['start_ns'])/1e6)],(y-.3,.6),facecolors=colors[phase],alpha=.15)
            ax.broken_barh(intervals,(y-.25,.5),facecolors=colors[phase])
        ax.axvline((phases['vlm']['end_ns']-zero)/1e6,ls='--',lw=.8,color='#4477AA')
        ax.set_yticks([0,1,2],['Current VLM','Current AE','Next Encoder'])
        ax.set_title(title,loc='left');ax.grid(axis='x',alpha=.15)
    axes[-1].set_xlabel('ms relative to current VLM first Kernel; request 3 in each trace')
    fig.suptitle('Actual CUPTI Kernel intervals (diagnostic profiles, not formal latency)',fontsize=12)
    fig.tight_layout();save(fig,'kernel_timeline')

slo_path=P/'slo_summary.csv'
if slo_path.exists():
    rows=list(csv.DictReader(slo_path.open()))
    normal=[x for x in rows if x['ae_sm_min']=='0']
    if normal:
        fig,axes=plt.subplots(1,2,figsize=(11,4.2))
        modes=[(18,'no_lookahead'),(18,'encoder'),(0,'encoder'),(1,'encoder'),(3,'encoder'),(6,'encoder')]
        labels=['Fresh serial','Fresh+Encoder','Lead0+Encoder','Lead1+Encoder','Lead3+Encoder','Lead6+Encoder']
        for ax,period in zip(axes,[33,62]):
            p95=[]; per_run=[]
            for lead,variant in modes:
                selected=[x for x in normal if int(x['lead'])==lead and x['variant']==variant and float(x['period_ms'])==period]
                values=[float(x['capture_to_action_ms_p95']) for x in selected]
                per_run.append(values)
                p95.append(np.mean(values))
            ax.barh(range(len(modes)),p95,color=['#BBBBBB','#4477AA','#66CCEE','#AA3377','#228833','#CCBB44'])
            ax.set_yticks(range(len(modes)),labels);ax.invert_yaxis()
            for i,values in enumerate(per_run):
                ax.plot(values,[i]*len(values),'ko',ms=4,label='Individual runs' if i==0 else None)
            ax.axvline(p95[0]*1.05,ls='--',color='red',label='Mean fresh serial +5%')
            ax.set_xlabel('Capture-to-action P95 (ms)');ax.set_title(f'{period} ms camera arrivals, GPU2');ax.legend(fontsize=8)
        fig.tight_layout();save(fig,'camera_slo')
print('Plots written to',P)

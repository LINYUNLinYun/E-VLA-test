"""CPU figures and self-contained media navigation; never creates simulator states."""
import html
import json
import os
from .core import OLD, write_json

def run(root,command,source=OLD):
    if command=='render':
        import csv
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from .trajectories import render
        if any((root/'closed_loop').glob('*/*/*/*/*/validated.json')):
            render(root,source)
        data=list(csv.DictReader((root/'analysis/kalman_offsets.csv').open())) if (root/'analysis/kalman_offsets.csv').exists() else []
        figures=root/'figures';figures.mkdir(exist_ok=True)
        for horizon in ((50,12) if data else ()):
            fig,ax=plt.subplots()
            for pair in ('Kalman-Full','HoldLast-Full','V14-Full','Kalman-V14'):
                selected=[r for r in data if r['split']=='evaluation' and r['pair']==pair and int(r['offset'])<=horizon]
                ax.plot([int(r['offset']) for r in selected],[float(r['equal_mean_mm']) for r in selected],label=pair)
            ax.set(xlabel='Prediction offset (control actions)',ylabel='Translation command L2 (mm)')
            ax.legend();fig.tight_layout()
            for suffix in ('png','svg'):
                destination=figures/('kalman_offsets_%d.%s'%(horizon,suffix))
                fig.savefig(destination)
            plt.close(fig)
        from .report import read,run as report
        counts=read(root,'phase_counts.csv')
        if counts:
            from .diagnostics import PHASES
            fig,ax=plt.subplots(figsize=(11,5))
            phases=(*PHASES,'unknown')
            names=sorted({r['condition'] for r in counts});width=.8/len(names)
            for i,name in enumerate(names):
                values=[sum(int(r['episodes']) for r in counts if r['condition']==name and r['furthest_phase']==p and r['success']=='False') for p in phases]
                ax.bar(__import__('numpy').arange(len(phases))+i*width,values,width=width,label=name,alpha=.75)
            ax.set_xticks(__import__('numpy').arange(len(phases))+.4-width/2,phases,rotation=25,ha='right')
            ax.set(ylabel='Failed episodes',title='Furthest observed phase (not failure onset)');ax.legend(fontsize=7);ax.grid(axis='y',alpha=.2);fig.tight_layout()
            for suffix in ('png','svg'):fig.savefig(figures/('failure_stage_counts.'+suffix))
            plt.close(fig)
        success=[r for r in read(root,'success.csv') if r['baseline']=='Full18_D3_R5_FORMAL' and r['suite']!='overall']
        if success:
            fig,ax=plt.subplots(figsize=(10,5))
            labels=[r['condition']+'\n'+r['suite'] for r in success]
            x=__import__('numpy').arange(len(success))
            ax.bar(x-.2,[100*int(r['baseline_successes'])/int(r['denominator']) for r in success],.4,label='Full',color='tab:blue',alpha=.7)
            ax.bar(x+.2,[100*int(r['successes'])/int(r['denominator']) for r in success],.4,label='New condition',color='tab:orange',alpha=.7)
            ax.set_xticks(x,labels,rotation=20,ha='right');ax.set(ylabel='Success (%)',ylim=(0,100));ax.legend();fig.tight_layout()
            for suffix in ('png','svg'):fig.savefig(figures/('success_rates.'+suffix))
            plt.close(fig)
        if (root/'analysis/failure_diagnostics.json').exists():
            from .diagnostics import clips
            clips(root)
        report(root)
        media=root/'media';media.mkdir(exist_ok=True)
        paths=sorted(root.rglob('*.png'))+sorted(root.rglob('*.mp4'))
        cases_path=media/'cases.json'
        if cases_path.exists():
            from pathlib import Path
            paths+=sorted({Path(c['full_video']) for c in json.loads(cases_path.read_text())['cases']})
        index=['# idea1 v2 媒体入口','', '动作目标增量不等于实际末端位移。','']
        gallery=['<!doctype html><meta charset="utf-8"><title>idea1 v2</title>']
        for p in paths:
            relative=os.path.relpath(p,media)
            if p.suffix=='.png':
                index.append('![%s](%s)'%(p.name,relative));gallery.append('<img width="600" src="%s">'%html.escape(relative))
            else:
                index.append('<video controls src="%s"></video>'%html.escape(relative));gallery.append('<video width="600" controls src="%s"></video>'%html.escape(relative))
        for name in ('index.md','viewer.md'):
            with (media/name).open('w') as stream:stream.write('\n'.join(index))
        with (media/'gallery.html').open('w') as stream:stream.write('\n'.join(gallery))
        return {'figures':'created','media_index':'created','plugin_playback_test':'not_run'}
    from .audit import run as audit
    return audit(root)

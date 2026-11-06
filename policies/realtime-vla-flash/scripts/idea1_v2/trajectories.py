"""Fixed-rule case selection and measured XYZ plots; no simulator replay."""
import json
from pathlib import Path
import numpy as np
from .analysis import completed
from .core import SUITES, rows, write_json
from .runner import ALT, PHASE

def render(root,source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    baseline={};v14={}
    for suite in SUITES:
        for name,target in (('Full18_D3_R5_FORMAL',baseline),('V14_A03_D3_R5_FORMAL',v14)):
            directory=source/'closed_loop'/name/suite/'run'
            for row in json.loads((directory/'episode_log.json').read_text()):
                target[(suite,row['task_id'],row['episode_idx'])]=(row,directory)
    selected=[]
    for condition in (ALT,PHASE):
        current=completed(root,condition)
        for suite in SUITES:
            candidates=[]
            for key,row in sorted(current.items()):
                if key[0]!=suite:continue
                full,full_dir=baseline[key]
                a=np.asarray([s['eef_pos'] for s in rows(full_dir/full['telemetry_path'])])
                b=np.asarray([s['eef_pos'] for s in rows(Path(row['directory'])/row['telemetry_path'])])
                n=min(len(a),len(b));difference=np.linalg.norm(a[:n]-b[:n],axis=1)
                candidates.append({'key':key,'row':row,'full':full,'full_dir':full_dir,'a':a,'b':b,
                                   'max':float(difference.max()),'mean':float(difference.mean())})
            rules=[]
            failures=[c for c in candidates if c['full']['success'] and not c['row']['success']]
            if failures:rules.append(('full_success_new_failure_max_same_step_xyz',min(failures,key=lambda c:(-c['max'],c['key']))))
            recoveries=[c for c in candidates if not v14[c['key']][0]['success'] and c['row']['success']]
            if recoveries:rules.append(('v14_failure_new_success_lexicographic',min(recoveries,key=lambda c:c['key'])))
            both=[c for c in candidates if c['full']['success'] and c['row']['success']]
            if both:
                median=float(np.median([c['mean'] for c in both]))
                rules.append(('both_success_closest_median_xyz',min(both,key=lambda c:(abs(c['mean']-median),c['key']))))
            seen=set()
            for rule,case in rules:
                if case['key'] in seen:continue
                seen.add(case['key']);key=case['key'];a,b=case['a'],case['b']
                fig=plt.figure(figsize=(12,9))
                ax=fig.add_subplot(221,projection='3d')
                ax.plot(*a.T,color='tab:blue',linestyle='-',marker='.',markevery=5,alpha=.65,label='Full')
                ax.plot(*b.T,color='tab:orange',linestyle='--',marker='^',markevery=5,alpha=.65,label=condition)
                ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)');ax.legend()
                infer=rows(Path(case['row']['directory'])/case['row']['infer_path'])
                takeover=[r['frame_idx'] for r in infer if r['idea1_v2']['model']=='Full' and r['frame_idx']<len(b)]
                if takeover:ax.scatter(*b[takeover].T,color='black',marker='x',label='Full call')
                for plot,(x,y) in enumerate(((0,1),(0,2),(1,2)),222):
                    ax=fig.add_subplot(plot)
                    ax.plot(a[:,x],a[:,y],color='tab:blue',linestyle='-',marker='.',markevery=5,alpha=.65)
                    ax.plot(b[:,x],b[:,y],color='tab:orange',linestyle='--',marker='^',markevery=5,alpha=.65)
                    if takeover:ax.scatter(b[takeover,x],b[takeover,y],color='black',marker='x')
                    for t in range(0,len(b),25):ax.annotate('%.2fs'%(t*.05),(b[t,x],b[t,y]),fontsize=6)
                    ax.set(xlabel='%s (m)'%'XYZ'[x],ylabel='%s (m)'%'XYZ'[y]);ax.set_aspect('equal',adjustable='datalim')
                fig.suptitle('%s task%02d/init%02d: %s\nMeasured XYZ; black x = Full call, association only'%(suite,key[1],key[2],rule))
                fig.tight_layout()
                directory=root/'figures/trajectories'/condition/suite;directory.mkdir(parents=True,exist_ok=True)
                stem='task%02d_init%02d'%(key[1],key[2])
                for suffix in ('png','svg'):
                    destination=directory/(stem+'.'+suffix)
                    fig.savefig(destination)
                plt.close(fig)
                selected.append({'condition':condition,'suite':suite,'task':key[1],'init':key[2],'rule':rule,
                                 'max_same_step_xyz_m':case['max'],'mean_same_step_xyz_m':case['mean'],
                                 'figure':str((directory/(stem+'.png')).relative_to(root)),
                                 'full_video':str(case['full_dir']/case['full']['video_path']),
                                 'new_video':str(Path(case['row']['directory'])/case['row']['video_path']),
                                 'interpretation':'descriptive association; cases do not estimate frequency'})
    write_json(root/'media/cases.json',{'cases':selected,'selection':'preregistered; <=3 per suite/condition; deduplicated'})
    return {'selected_cases':len(selected)}

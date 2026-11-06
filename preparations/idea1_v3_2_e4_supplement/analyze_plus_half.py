"""Read-only paired half-Plus summary; no model, environment, or renderer calls."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import textwrap

import numpy as np

from .half_plus import audit
from .native_envs import digest

PARENT='a4168640cd0a455cd50489ee10bcc3295511f9561ad8710b334817f16ecb9f20'
HALF='bd68b9a703729efb8ad649eab0a7f155fcffa98f9ecd44fac2bd359c347d1b6c'
LABELS=('full_d10','v14_d10','matching_draft')
REPRESENTATIVES=(0,50,100,150,200,250,300)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matched_jobs(manifest,ledger):
    by={(r['group'],r['source_index'],r['mode']):r for r in ledger['selected_completed']}
    if len(by)!=len(ledger['selected_completed']):raise ValueError('Duplicate selected condition')
    selected=[]
    for job in manifest['jobs']:
        rows=[by.get((job['group'],job['source_index'],mode)) for mode in job['modes']]
        if all(row is not None for row in rows):
            if len({row.get('gpu_uuid') for row in rows})!=1:raise ValueError('Three conditions span different GPU identities')
            selected.append((job,dict(zip(LABELS,rows))))
    return selected


def completion_gate(completed,paired,allow_partial):
    final=completed==585 and paired==195
    if not final and not allow_partial:raise RuntimeError('Not all585 identity-valid/195three-way cases complete; use --allow-partial')
    return final


def summaries(manifest,matched):
    rows=[]
    for group in ('perturbed350','original40'):
        expected=[j for j in manifest['jobs'] if j['group']==group]
        available=[pair for pair in matched if pair[0]['group']==group]
        dimensions=['overall','mother_suite']+(['category','difficulty_level'] if group=='perturbed350' else [])
        for dimension in dimensions:
            values=['all'] if dimension=='overall' else sorted({str(j['case'][dimension]) for j in expected})
            for value in values:
                inside=lambda j:dimension=='overall' or str(j['case'][dimension])==value
                pairs=[pair for pair in available if inside(pair[0])];n=len(pairs)
                full=np.array([r['full_d10']['success'] for _,r in pairs],dtype=int)
                for label in LABELS:
                    success=np.array([r[label]['success'] for _,r in pairs],dtype=int);delta=success-full
                    rows.append(dict(group=group,dimension=dimension,value=value,condition=label,
                        expected_cases=sum(inside(j) for j in expected),paired_cases=n,successes=int(success.sum()),
                        success_rate=float(success.mean()) if n else None,
                        paired_difference_vs_full=float(delta.mean()) if n else None,
                        wins_vs_full=int((delta>0).sum()),losses_vs_full=int((delta<0).sum())))
    return rows


def trajectory(result_path):
    directory=Path(result_path).parent
    result=json.loads(Path(result_path).read_text());hz=float(result['native_info']['native_hz'])
    with np.load(directory/'initial_native_observation.npz') as initial:xyz=[initial['robot0_eef_pos'].tolist()]
    steps=[0]
    for line in (directory/'telemetry.jsonl').read_text().splitlines():
        row=json.loads(line);steps.append(row['step']);xyz.append(row['actual']['eef_xyz_world'])
    if len(steps)!=result['steps']+1 or np.any(np.diff(steps)!=1):raise ValueError('Telemetry step coverage mismatch')
    return np.array(steps)/hz,np.array(xyz,dtype=float)


def common_distance(t1,p1,t2,p2):
    # Only interpolate inside BOTH observed time ranges. Never pad an ended run.
    low=max(t1[0],t2[0]);high=min(t1[-1],t2[-1])
    times=np.unique(np.r_[t1[(t1>=low)&(t1<=high)],t2[(t2>=low)&(t2<=high)]])
    first=np.column_stack([np.interp(times,t1,p1[:,i]) for i in range(3)])
    second=np.column_stack([np.interp(times,t2,p2[:,i]) for i in range(3)])
    return times,np.linalg.norm(first-second,axis=1)*1000.


def csv_write(path,rows):
    if not rows:return
    with Path(path).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def comparison_labels(results,other):
    labels={}
    for key in ('full_d10',other):
        result=json.loads(Path(results[key]['path']).read_text())
        name='Full' if key=='full_d10' else ('V14' if key=='v14_d10' else 'Matching draft')
        labels[key]='%s: %s, %d steps'%(name,'success' if result['success'] else 'failure',result['steps'])
    full=json.loads(Path(results['full_d10']['path']).read_text())
    return labels,full['native_info']['instruction']


def make_plots(out,summary,matched,status):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'full_d10':'#2474b4','v14_d10':'#ec8a23','matching_draft':'#299348'}
    rows=[r for r in summary if r['group']=='perturbed350' and r['dimension'] in ('overall','category')]
    values=list(dict.fromkeys(r['value'] for r in rows));fig,ax=plt.subplots(figsize=(11,4))
    for i,mode in enumerate(LABELS):
        subset=[next(r for r in rows if r['value']==v and r['condition']==mode) for v in values]
        ax.bar(np.arange(len(values))+(i-1)*.24,[100*r['success_rate'] if r['success_rate'] is not None else np.nan for r in subset],.24,color=colors[mode],alpha=.75,label={'full_d10':'Full','v14_d10':'V14','matching_draft':'Matching draft'}[mode])
    ax.set_xticks(range(len(values)));ax.set_xticklabels([v+'\npaired n='+str(next(r['paired_cases'] for r in rows if r['value']==v)) for v in values],rotation=20,ha='right')
    paired=next(r['paired_cases'] for r in rows if r['dimension']=='overall')
    ax.set_title('%s — perturbed half subset: three-way paired n=%d/175'%(status,paired))
    ax.set_ylim(0,100);ax.set_ylabel('Success rate (%)');ax.legend(loc='upper left',bbox_to_anchor=(1.01,1));fig.tight_layout();fig.savefig(out/'success_paired.png',dpi=140);plt.close(fig)
    images=[]
    for job,results in matched:
        if job['group']!='perturbed350' or job['source_index'] not in REPRESENTATIVES:continue
        for other in ('v14_d10','matching_draft'):
            labels,instruction=comparison_labels(results,other)
            contrast='Full vs V14' if other=='v14_d10' else 'Full vs matching draft'
            title='%s — %s / fixed source_index=%d\n%s'%(contrast,job['case']['category'],job['source_index'],textwrap.fill(instruction,80))
            tf,pf=trajectory(results['full_d10']['path']);to,po=trajectory(results[other]['path'])
            t,distance=common_distance(tf,pf,to,po);fig,axes=plt.subplots(4,1,figsize=(9,9),sharex=True)
            for axis in range(3):
                axes[axis].plot(tf,pf[:,axis],color=colors['full_d10'],alpha=.65,label=labels['full_d10'])
                axes[axis].plot(to,po[:,axis],color=colors[other],alpha=.65,linestyle='--',marker='.',markevery=max(1,len(to)//25),label=labels[other])
                axes[axis].set_ylabel('XYZ'[axis]+' (world m)')
            axes[0].legend(fontsize=8);axes[0].set_title(title,fontsize=10)
            axes[3].plot(t,distance,color=colors[other],alpha=.75);axes[3].set_ylabel('Path separation (mm)');axes[3].set_xlabel('Native time (s): XYZ = each complete record; distance = common interval only\nClosed-loop path / phase difference, not same-observation prediction error; no padding',fontsize=9)
            fig.tight_layout();name='trajectory_%03d_%s.png'%(job['source_index'],other);fig.savefig(out/name,dpi=140);plt.close(fig);images.append(name)
            fig,axes=plt.subplots(1,2,figsize=(11,5))
            for ax,(x,y) in zip(axes,((0,1),(0,2))):
                for key,points in [('full_d10',pf),(other,po)]:
                    ax.plot(points[:,x],points[:,y],color=colors[key],alpha=.65,linestyle='-' if key=='full_d10' else '--',marker='.',markevery=max(1,len(points)//30),label=labels[key])
                    ax.scatter(points[0,x],points[0,y],color=colors[key],marker='o',s=45,alpha=.7)
                    ax.scatter(points[-1,x],points[-1,y],color=colors[key],marker='X',s=55,alpha=.7)
                ax.set_xlabel('XYZ'[x]+' (world m)');ax.set_ylabel('XYZ'[y]+' (world m)');ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
            axes[0].legend(fontsize=8);fig.suptitle(title+'\nCircle = start; X = end; complete actual paths, no padding',fontsize=10)
            fig.tight_layout();name='spatial_%03d_%s.png'%(job['source_index'],other);fig.savefig(out/name,dpi=140);plt.close(fig);images.append(name)
    return images


def analyze(halfroot,output,allow_partial=False,plots=True):
    halfroot=Path(halfroot).resolve();manifest_path=halfroot/'manifest.json';manifest=json.loads(manifest_path.read_text())
    if manifest.get('manifest_hash')!=HALF or manifest.get('parent_manifest_hash')!=PARENT or digest({k:v for k,v in manifest.items() if k!='manifest_hash'})!=HALF:
        raise ValueError('Exact half/parent scientific identities required')
    ledger=audit(manifest,halfroot);matched=matched_jobs(manifest,ledger)
    final=completion_gate(ledger['completed'],len(matched),allow_partial)
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'audit_ledger.json').write_text(json.dumps(ledger,indent=2,ensure_ascii=False))
    summary=summaries(manifest,matched);csv_write(out/'success_paired.csv',summary)
    matched_keys={(j['group'],j['source_index']) for j,_ in matched}
    inputs=[]
    for kind,rows in [('selected',ledger['selected_completed']),('excluded_extra_historical',ledger['extra_historical_completed'])]:
        for row in rows:
            actual_sha=sha(row['path'])
            if actual_sha!=row['sha256']:raise ValueError('Result changed during audit')
            inputs.append(dict(kind=kind,group=row['group'],source_index=row['source_index'],mode=row['mode'],path=row['path'],sha256=actual_sha,
                included_in_paired_comparison=kind=='selected' and (row['group'],row['source_index']) in matched_keys))
    for path in ledger['technical_attempts']:inputs.append(dict(kind='technical_attempt_excluded',path=path,sha256=sha(path)))
    videos=[]
    by={(r['group'],r['source_index'],r['mode']):r for r in ledger['selected_completed']}
    for job in manifest['jobs']:
        if not (job['group']=='perturbed350' and job['source_index'] in REPRESENTATIVES):continue
        for mode in job['modes']:
            row=by.get((job['group'],job['source_index'],mode));video=Path(row['path']).parent/'video.mp4' if row else None
            videos.append(dict(group=job['group'],source_index=job['source_index'],category=job['case']['category'],mode=mode,
                type='actual existing rollout MP4',status='available' if video and video.is_file() else 'missing_not_generated',
                relative_path=os.path.relpath(video,out) if video and video.is_file() else '',sha256=sha(video) if video and video.is_file() else ''))
    csv_write(out/'videos.csv',videos)
    figures=make_plots(out,summary,matched,'FINAL' if final else 'PARTIAL') if plots else []
    report=dict(status='final' if final else 'PARTIAL',created_utc=datetime.now(timezone.utc).isoformat(),parent_manifest_hash=PARENT,half_manifest_hash=HALF,
        script_sha256=sha(__file__),half_manifest_file_sha256=sha(manifest_path),completed_identity_valid_conditions=ledger['completed'],target_conditions=585,
        paired_complete_cases=len(matched),expected_paired_cases=195,excluded_extra_historical_count=ledger['extra_historical_count'],
        reused_historical_selected_conditions=ledger['reused_historical'],new_selected_completed_conditions=ledger['completed']-ledger['reused_historical'],
        audit_ledger_sha256=sha(out/'audit_ledger.json'),
        paired_case_set=[dict(group=j['group'],source_index=j['source_index'],case_hash=j['case_hash']) for j,_ in matched],
        registered_case_set=[dict(group=j['group'],source_index=j['source_index'],case_hash=j['case_hash']) for j in manifest['jobs']],
        input_results=inputs,summary=summary,videos=videos,trajectory_figures=figures,
        scientific_limits=['Outcome-blind half subset, not all10030 Plus tasks.','Perturbed175 and original20 always separate; partial comparisons use only common three-way complete cases.','Closed-loop actual EEF trajectory separation is NOT same-observation action prediction error.','Telemetry lacks object/contact ground truth; gripper alone cannot establish contact/stages.','No new model, simulation, renderer, screenshot, or video creation.'])
    report['audit_source_sha256']=sha(Path(__file__).with_name('half_plus.py'))
    snapshot=out/'source_snapshot';snapshot.mkdir()
    for name in ('analyze_plus_half.py','test_analyze_plus_half.py'):shutil.copyfile(Path(__file__).with_name(name),snapshot/name)
    report['analysis_source_snapshot_sha256']={p.name:sha(p) for p in snapshot.iterdir()}
    report['all_audit_input_result_sha256']={str(p):sha(p) for base in (halfroot,Path(manifest['source_results_root'])) for p in base.rglob('result.json')}
    report['audit_snapshot_note']='Read-only live-queue snapshot. Running results may change later; all scanned result JSON hashes retained. Only identity-valid completed rows enter paired comparison.'
    # Hash every telemetry/initial NPZ actually used by representative plots.
    report['trajectory_input_sha256']={str(p):sha(p) for j,rs in matched if plots and j['group']=='perturbed350' and j['source_index'] in REPRESENTATIVES for row in rs.values() for p in [Path(row['path']).parent/'telemetry.jsonl',Path(row['path']).parent/'initial_native_observation.npz']}
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False))
    text=['# Plus half '+report['status'],'',
        '已完成身份有效条件 %d/585；三条件配齐 %d/195 case。额外历史 %d 条不计入。'%(ledger['completed'],len(matched),ledger['extra_historical_count']),
        '这里只比较三条件共同完成的 case，扰动175与原始20分开，不能外推全部10,030任务。','',
        '|样本组|配齐case|Full|V14|对应draft|','|---|---:|---:|---:|---:|']
    for group in ('perturbed350','original40'):
        rs=[r for r in summary if r['group']==group and r['dimension']=='overall'];label='扰动175' if group=='perturbed350' else '原始20';text.append('|%s|%d|%s|'%(label,rs[0]['paired_cases'],'|'.join('%d/%d (%.1f%%)'%(r['successes'],r['paired_cases'],100*r['success_rate']) if r['success_rate'] is not None else '未配齐' for r in rs)))
    text+=['','[分类、难度、母任务suite及配对差 CSV](success_paired.csv)','',
        '轨迹采用预先固定的各category最小source_index：0/50/100/150/200/250/300；缺少配齐结果时不挑替代样本。闭环路径距离只在双方已有的公共时间内计算，不是同观测预测误差；没有接触GT，不据夹爪推断接触阶段。','',
        '![三条件共同样本成功率](success_paired.png)' if plots else '未生成图。']
    for job in manifest['jobs']:
        if job['group']!='perturbed350' or job['source_index'] not in REPRESENTATIVES:continue
        text+=['','### %s · 固定 source_index=%d'%(job['case']['category'],job['source_index']),'']
        found=[f for f in figures if '_%03d_'%job['source_index'] in f]
        text+=['![%s](%s)'%('Full vs V14' if 'v14_d10' in f else 'Full vs matching draft',f) for f in found] if found else ['三条件尚未配齐，不替换代表case。']
    text+=['','真实现有视频（缺失不补造）：','']
    text+=['- %s / %s: %s'%(v['source_index'],v['mode'],'[MP4](%s)'%v['relative_path'] if v['relative_path'] else '缺失（未生成）') for v in videos]
    docs=Path(__file__).resolve().parents[2]/'docs/experiments/idea1'
    text+=['','既有补测[plan](%s) / [status](%s)。本报告只读离线汇总；当前队列状态见[half live ledger](%s)。'%(os.path.relpath(docs/'idea1_v3_2_e4_supplement_plan.md',out),os.path.relpath(docs/'idea1_v3_2_e4_supplement_status.md',out),os.path.relpath(halfroot/'live_ledger.json',out))]
    (out/'README.md').write_text('\n'.join(text)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--half-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--allow-partial',action='store_true');p.add_argument('--no-plots',action='store_true');a=p.parse_args()
    r=analyze(a.half_root,a.output,a.allow_partial,not a.no_plots)
    print(json.dumps({k:r[k] for k in ('status','completed_identity_valid_conditions','target_conditions','paired_complete_cases','excluded_extra_historical_count')}))


if __name__=='__main__':main()

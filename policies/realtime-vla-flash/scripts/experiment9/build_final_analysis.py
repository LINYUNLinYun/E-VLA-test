"""Stream the audited round, analyze paired outcomes and XYZ, and build the final report."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import html
import json
import re
from pathlib import Path
import subprocess
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_standard_vlm_pruning as old

ROOT, RESULTS = old.ROOT, old.RESULTS
OUT = RESULTS / 'analysis' / 'final'
REPORT = ROOT / 'experiment9_final_analysis_report.md'
AE = ('SkipAE-L2', 'SkipAE-L8')
CONDS = old.ALL_CONDITIONS + AE
RANK = {s: i for i, s in enumerate(old.STAGES)}


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def path_for(row, field):
    return Path(row['run_dir']) / row[field]


def trajectory(row):
    steps, xyz, phases = [], [], []
    with path_for(row, 'telemetry_path').open() as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            steps.append(int(item['step']))
            xyz.append(item['eef_pos'])
            phases.append(item.get('semantic_phase', 'unknown'))
    steps, xyz = np.asarray(steps), np.asarray(xyz, dtype=float)
    if len(steps) != row['executed_action_count'] + 1 or not np.array_equal(steps, np.arange(len(steps))) or not np.isfinite(xyz).all():
        raise ValueError(f'Invalid full XYZ trajectory: {row["run_dir"]} {row["episode_idx"]}')
    return steps, xyz, phases


def load_rows():
    audit = json.loads((RESULTS / 'audit/round_completion.json').read_text())
    if audit['missing_chunks'] or audit['selected_episodes'] != 28000:
        raise ValueError('Main round is not complete')
    rows = {}
    for directory in audit['selected_runs']:
        manifest = json.loads((Path(directory) / 'manifest.json').read_text())
        condition = manifest.get('condition')
        episodes = json.loads((Path(directory) / 'episode_log.json').read_text())
        if condition not in old.ALL_CONDITIONS:
            # Run-directory slugs are unambiguous even if manifests use an internal slug.
            condition = next(c for c in old.ALL_CONDITIONS if f'_{c.replace("-", "_")}_libero_' in Path(directory).name)
        for ep in episodes:
            ep.update(run_dir=directory, condition=condition, suite=ep['task_suite_name'])
            rows[(condition, ep['suite'], ep['task_id'], ep['episode_idx'])] = ep
    screen = json.loads((RESULTS / 'action_expert_layer_probe/closed_loop/screening_summary.json').read_text())
    if not screen['complete'] or screen['episodes'] != 800:
        raise ValueError('Conditional action-layer screening is incomplete')
    for run in screen['results']:
        directory = Path(run['run_dir'])
        condition = 'SkipAE-L2' if '_SkipAE_L2_' in directory.name else 'SkipAE-L8'
        for ep in json.loads((directory / 'episode_log.json').read_text()):
            ep.update(run_dir=str(directory), condition=condition, suite=ep['task_suite_name'])
            rows[(condition, ep['suite'], ep['task_id'], ep['episode_idx'])] = ep
    assert len(rows) == 28800
    return rows


def analyze(rows):
    baselines, stages, deviations, warm_times = {}, {}, [], {}
    timing_sources = Counter()
    for n, (key, row) in enumerate(rows.items(), 1):
        steps, xyz, phases = trajectory(row)
        stages[key] = set(phases)
        if key[0] == 'Full18':
            baselines[key[1:]] = (steps, xyz, phases)
        else:
            bsteps, bxyz, bphases = baselines[key[1:]]
            length = min(len(xyz), len(bxyz))
            assert np.allclose(xyz[0], bxyz[0], atol=1e-8, rtol=0), key
            distance = np.linalg.norm(xyz[:length] - bxyz[:length], axis=1)
            base = rows[('Full18', *key[1:])]
            outcome = ('both_success' if base['success'] and row['success'] else
                       'new_failure' if base['success'] else
                       'recovery' if row['success'] else 'both_failure')
            item = dict(condition=key[0], suite=key[1], task_id=key[2], episode_idx=key[3],
                        outcome=outcome, baseline_success=bool(base['success']), variant_success=bool(row['success']),
                        baseline_steps=len(bxyz)-1, variant_steps=len(xyz)-1, overlap_states=length,
                        mean_deviation_m=float(distance.mean()), p95_deviation_m=float(np.quantile(distance, .95)),
                        max_deviation_m=float(distance.max()), max_step=int(distance.argmax()),
                        variant_furthest_phase=row.get('furthest_semantic_phase', 'unknown'))
            for threshold in (.01, .02, .05):
                starts = np.flatnonzero(np.convolve((distance > threshold).astype(int), np.ones(5, dtype=int), 'valid') == 5) if length >= 5 else []
                onset = int(starts[0]) if len(starts) else None
                label = f'{int(threshold*100)}cm'
                item[f'onset_{label}_step'] = onset
                item[f'onset_{label}_baseline_phase'] = bphases[onset] if onset is not None else 'none'
                item[f'onset_{label}_variant_phase'] = phases[onset] if onset is not None else 'none'
            deviations.append(item)
        times = defaultdict(list)
        with path_for(row, 'infer_path').open() as f:
            for line in f:
                infer = json.loads(line)
                # Exclude the first inference in every episode; compilation/setup can dominate it.
                if int(infer['infer_id']) == 0:
                    continue
                for name in ('policy_time_ms', 'encoder_ms', 'vlm_prefill_ms', 'total_ms'):
                    if infer.get(name) is not None:
                        times[name].append(float(infer[name]))
                value = infer.get('action_denoise_ms')
                source = 'action_denoise_ms'
                if value is None and infer.get('is_full_pipeline_round') == 1:
                    value, source = infer.get('full_fallback_ms'), 'full_fallback_ms'
                if value is not None:
                    times['decoder_ms'].append(float(value)); timing_sources[source] += 1
        warm_times[key] = {k: float(np.median(v)) for k, v in times.items() if v}
        if n % 1000 == 0:
            print(f'Streamed full telemetry/inference: {n}/28800', flush=True)
    old.write_csv(OUT / 'paired_xyz_all.csv', deviations)
    (OUT / 'timing_sources.json').write_text(json.dumps(dict(timing_sources), indent=2))
    return stages, deviations, warm_times


def summarize(rows, stages, deviations, times):
    suites, tasks, failures, reach, cohorts = [], [], [], [], []
    for c in CONDS:
        for suite in old.SUITES:
            keys = [k for k in rows if k[:2] == (c, suite)]
            base_keys = [('Full18', *k[1:]) for k in keys]
            success = sum(bool(rows[k]['success']) for k in keys)
            base_success = sum(bool(rows[k]['success']) for k in base_keys)
            lows, highs = old.wilson(success, len(keys))
            new = sum(rows[b]['success'] and not rows[k]['success'] for k, b in zip(keys, base_keys))
            recover = sum(not rows[b]['success'] and rows[k]['success'] for k, b in zip(keys, base_keys))
            diffs = []
            for task in range(10):
                group = [k for k in keys if k[2] == task]
                rate = np.mean([bool(rows[k]['success']) for k in group])
                base_rate = np.mean([bool(rows[('Full18', *k[1:])]['success']) for k in group])
                diffs.append(rate-base_rate)
                tasks.append(dict(condition=c, suite=suite, task_id=task, episodes=len(group), success_rate=rate,
                                  matched_baseline_rate=base_rate, delta_pp=100*(rate-base_rate), task_description=rows[group[0]]['task_description']))
            blo, bhi = old.cluster_bootstrap(diffs, seed=20261005+CONDS.index(c)*10+old.SUITES.index(suite))
            item = dict(condition=c, suite=suite, successes=success, episodes=len(keys), success_rate=success/len(keys),
                        wilson95_low=lows, wilson95_high=highs, matched_baseline_successes=base_success,
                        matched_baseline_episodes=len(base_keys), delta_pp=100*(success-base_success)/len(keys),
                        task_cluster_delta95_low_pp=100*blo, task_cluster_delta95_high_pp=100*bhi,
                        new_failures=int(new), recoveries=int(recover), mcnemar_p=old.mcnemar_exact(new,recover))
            for metric in ('policy_time_ms','encoder_ms','vlm_prefill_ms','decoder_ms','total_ms'):
                values = [times[k][metric] for k in keys if metric in times[k]]
                item[f'warm_episode_median_{metric}'] = float(np.median(values)) if values else float('nan')
            suites.append(item)
            for stage, index in RANK.items():
                reached = sum(bool(rows[k]['success']) if stage=='success' else
                              any(p in RANK and RANK[p]>=index for p in stages[k]) for k in keys)
                reach.append(dict(condition=c,suite=suite,stage=stage,reached=reached,episodes=len(keys),reach_rate=reached/len(keys)))
            for cohort in ('all_failures', 'new_failures'):
                eligible = [k for k in keys if not rows[k]['success'] and (cohort=='all_failures' or rows[('Full18',*k[1:])]['success'])]
                buckets = Counter()
                for k in eligible:
                    highest = max((RANK[p] for p in stages[k] if p in RANK), default=-1)
                    bucket = 'pre_contact' if highest < RANK['contact'] else 'contact_before_grasp' if highest < RANK['grasp_or_manipulate'] else 'grasp_or_later'
                    buckets[bucket]+=1
                for bucket in ('pre_contact','contact_before_grasp','grasp_or_later'):
                    failures.append(dict(condition=c,suite=suite,cohort=cohort,bucket=bucket,count=buckets[bucket],failures=len(eligible),fraction=buckets[bucket]/len(eligible) if eligible else 0))
            for d in deviations:
                if d['condition']==c and d['suite']==suite:
                    cohorts.append(d)
    for name, data in [('suite_success',suites),('task_success',tasks),('failure_stage_counts',failures),('stage_reach',reach)]:
        old.write_csv(OUT / f'{name}.csv',data)
    return suites,tasks,failures,reach


def figures(suites, failures):
    lookup={(r['condition'],r['suite']):r for r in suites}
    fig, ax=plt.subplots(figsize=(9,9),layout='constrained')
    mat=np.array([[lookup[c,s]['success_rate'] for s in old.SUITES] for c in CONDS])
    im=ax.imshow(mat,vmin=0,vmax=1,cmap='RdYlGn',aspect='auto')
    ax.set_xticks(range(4),['Object','Spatial','Goal','LIBERO-10']);ax.set_yticks(range(len(CONDS)),CONDS)
    for i,c in enumerate(CONDS):
        for j,s in enumerate(old.SUITES):
            ax.text(j,i,f'{mat[i,j]:.1%}\n(n={lookup[c,s]["episodes"]})',ha='center',va='center',fontsize=8)
    ax.set_title('Standard LIBERO: full round + action-layer screening');fig.colorbar(im,ax=ax,label='Success rate')
    fig.savefig(OUT/'success_heatmap.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(15,12),layout='constrained')
    colors=('#4c78a8','#f2a541','#c65757')
    for s,ax in zip(old.SUITES,axes.flat):
        left=np.zeros(len(CONDS))
        for bucket,color in zip(('pre_contact','contact_before_grasp','grasp_or_later'),colors):
            vals=[next(r['count'] for r in failures if r['condition']==c and r['suite']==s and r['cohort']=='new_failures' and r['bucket']==bucket) for c in CONDS]
            ax.barh(range(len(CONDS)),vals,left=left,color=color,label=bucket.replace('_',' '));left+=vals
        ax.set_yticks(range(len(CONDS)),CONDS,fontsize=8);ax.invert_yaxis();ax.set_title(s);ax.set_xlabel('New failures vs matched baseline (count)');ax.grid(axis='x',alpha=.2)
        for i,v in enumerate(left):
            ax.text(v+1,i,str(int(v)),va='center',fontsize=8)
    axes.flat[0].legend(fontsize=8);fig.suptitle('Failure location: only baseline-success / variant-failure pairs')
    fig.savefig(OUT/'new_failure_stages.png',dpi=160);plt.close(fig)
    # Task-balanced action-expert hidden metrics, not treating repeated seeds as independent tasks.
    hidden=read_csv(RESULTS/'action_expert_layer_probe/probe/layer_hidden_similarity.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for step in (0,2,5,9):
        values=[]
        for layer in range(1,19):
            grouped=defaultdict(list)
            for r in hidden:
                if int(r['flow_step'])==step and int(r['layer'])==layer:
                    grouped[r['suite'],r['task_id']].append(float(r['adjacent_cosine']))
            values.append(np.mean([np.mean(v) for v in grouped.values()]))
        axes[0].plot(range(1,19),values,marker='o',label=f'flow step {step}')
    axes[0].legend();axes[0].set(xlabel='Action expert layer (1-based)',ylabel='Adjacent cosine',title='Real action-token hidden states')
    summary=json.loads((RESULTS/'action_expert_layer_probe/probe/summary.json').read_text())
    ranking=sorted(summary['ranking'],key=lambda r:r['layer'])
    means=np.array([r['task_mean_nmse'] for r in ranking]); lo=np.array([r['bootstrap95_low'] for r in ranking]);hi=np.array([r['bootstrap95_high'] for r in ranking])
    axes[1].errorbar(range(1,19),means,yerr=[means-lo,hi-means],fmt='o',capsize=2);axes[1].set_yscale('log')
    axes[1].axhline(.01,color='orange',ls='--',label='screening mean threshold');axes[1].legend();axes[1].set(xlabel='Skipped expert layer',ylabel='First-12 command NMSE',title='Task-cluster bootstrap 95% intervals')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(OUT/'action_expert_probe.png',dpi=180);plt.close(fig)


def render_cases(rows, deviations):
    cases=[]
    for c in CONDS[1:]:
        for suite in old.SUITES:
            group=[d for d in deviations if d['condition']==c and d['suite']==suite and d['outcome']=='new_failure']
            if not group:group=[d for d in deviations if d['condition']==c and d['suite']==suite and d['outcome']=='both_success']
            if not group:group=[d for d in deviations if d['condition']==c and d['suite']==suite]
            group.sort(key=lambda d:(d['mean_deviation_m'],d['task_id'],d['episode_idx']))
            case=dict(group[len(group)//2]);case['selection']='median mean-XYZ deviation within new-failure cohort, else both-success, else any'
            case['slug']=f'{c.replace("-","_")}__{suite}__task{case["task_id"]:02d}_ep{case["episode_idx"]:02d}'
            cases.append(case)
    plots=OUT/'xyz_plots';videos=OUT/'xyz_videos';plots.mkdir(exist_ok=True);videos.mkdir(exist_ok=True)
    def render(case):
        row=rows[case['condition'],case['suite'],case['task_id'],case['episode_idx']]
        base=rows['Full18',case['suite'],case['task_id'],case['episode_idx']]
        prefix=plots/case['slug']
        subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/experiment8/plot_paired_xyz.py'),str(path_for(base,'telemetry_path')),str(path_for(row,'telemetry_path')),'--baseline-label','Full18 baseline','--variant-label',case['condition'],'--output-prefix',str(prefix)],check=True,stdout=subprocess.DEVNULL)
        case['plot']=str(prefix.with_suffix('.png').relative_to(ROOT))
        # One paired video per variant; all four suites have plots and source video links.
        case['video']=''
        if case['suite']=='libero_10':
            dest=videos/f'{case["slug"]}.mp4'
            subprocess.run([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/experiment8/render_xyz_video.py'),'--baseline-video',str(path_for(base,'video_path')),'--variant-video',str(path_for(row,'video_path')),'--baseline-telemetry',str(path_for(base,'telemetry_path')),'--variant-telemetry',str(path_for(row,'telemetry_path')),'--baseline-label','Full18 baseline','--variant-label',case['condition'],'--alignment','state-synchronous','--output',str(dest)],check=True,stdout=subprocess.DEVNULL)
            case['video']=str(dest.relative_to(ROOT))
        case['baseline_video']=str(path_for(base,'video_path').relative_to(ROOT));case['variant_video']=str(path_for(row,'video_path').relative_to(ROOT))
        print(f'Rendered {case["slug"]}',flush=True)
        return case
    with ThreadPoolExecutor(max_workers=4) as executor:
        cases=list(executor.map(render,cases))
    old.write_csv(OUT/'xyz_case_index.csv',cases)
    cards=[]
    for c in cases:
        img=ROOT/c['plot'];vid=ROOT/c['video'] if c['video'] else None
        import os
        cards.append(f'<article><h3>{html.escape(c["condition"])} / {c["suite"]} / task {c["task_id"]} / init {c["episode_idx"]} ({c["outcome"]})</h3><p>mean/p95/max={c["mean_deviation_m"]:.3f}/{c["p95_deviation_m"]:.3f}/{c["max_deviation_m"]:.3f} m; first sustained 2cm divergence step={c["onset_2cm_step"]}, baseline phase={c["onset_2cm_baseline_phase"]}</p><img src="{os.path.relpath(img,OUT)}">'+(f'<video controls preload="metadata" src="{os.path.relpath(vid,OUT)}"></video>' if vid else '')+f'<p><a href="{os.path.relpath(ROOT/c["baseline_video"],OUT)}">Baseline source video</a> · <a href="{os.path.relpath(ROOT/c["variant_video"],OUT)}">Variant source video</a></p></article>')
    (OUT/'xyz_gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>Full round XYZ gallery</title><style>body{font:15px sans-serif;margin:2rem}article{margin:2rem 0}img,video{display:block;width:min(1000px,95vw)}</style>'+''.join(cards))
    return cases


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=load_rows()
    # Guarantee baseline trajectories are available before any variant is streamed.
    rows=dict(sorted(rows.items(),key=lambda kv:(kv[0][0]!='Full18',CONDS.index(kv[0][0]),kv[0][1:])))
    stages,deviations,times=analyze(rows)
    suites,tasks,failures,reach=summarize(rows,stages,deviations,times)
    figures(suites,failures)
    (OUT/'statistics_complete.json').write_text(json.dumps({'episodes':len(rows),'paired_xyz_episodes':len(deviations),'suite_cells':len(suites),'full_telemetry_validated':True},indent=2))
    cases=render_cases(rows,deviations)
    write_report(suites,tasks,failures,reach,deviations,cases)
    validate_artifacts()
    print(f'Report: {REPORT}',flush=True)


def write_report(suites,tasks,failures,reach,deviations,cases):
    # Kept separate from numerical analysis so report prose can be regenerated without rereading telemetry.
    from final_report_text import build_report
    REPORT.write_text(build_report(suites,tasks,failures,reach,deviations,cases),encoding='utf-8')


def validate_artifacts():
    cases=read_csv(OUT/'xyz_case_index.csv')
    assert len(cases)==60
    streams=[]
    for case in cases:
        assert (ROOT/case['plot']).is_file()
        if case['video']:
            path=ROOT/case['video']
            meta=json.loads(path.with_suffix('.json').read_text())
            stream=meta['encoding_validation']
            expected=max(int(case['baseline_steps']),int(case['variant_steps']))+1
            assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
            assert int(stream['nb_read_frames'])==expected
            assert len(meta['frame_index'])==expected
            assert meta['player_validation'].startswith('ffprobe and full ffmpeg decode passed')
            streams.append({'path':str(path.relative_to(ROOT)),'frames':expected,'full_decode_passed':True})
    assert len(streams)==15
    links=re.findall(r'\]\(([^)]+)\)',REPORT.read_text())
    missing=[x for x in links if not (ROOT/x).exists()]
    # This file is created below; the report intentionally links its own audit.
    missing=[x for x in missing if x!=str((OUT/'artifact_validation.json').relative_to(ROOT))]
    assert not missing,missing
    suites=read_csv(OUT/'suite_success.csv')
    assert sum(int(r['episodes']) for r in suites)==28800
    (OUT/'artifact_validation.json').write_text(json.dumps({'episodes':28800,'plot_count':len(cases),'report_links_checked':len(links),'missing_links':missing,'paired_videos':streams,'target_user_player_confirmed':False},indent=2)+'\n')


if __name__=='__main__':
    main()

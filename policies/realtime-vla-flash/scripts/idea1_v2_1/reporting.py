"""Frozen post-selection E4 statistics, figures and independently checked receipts."""
import csv
import json
from pathlib import Path
import numpy as np
from ..idea1_v2.core import sha, write_json


def table(path, rows):
    if not rows:
        return
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cluster_interval(values, metadata, seed=20261016, replicates=2000):
    """Paired hierarchy bootstrap: suite fixed, tasks and then episodes resampled."""
    rng = np.random.default_rng(seed)
    suites = sorted({m['suite'] for m in metadata})
    cells = {}
    for value, m in zip(values, metadata):
        if np.isfinite(value):
            cells.setdefault((m['suite'], m['task'], m['init']), []).append(value)
    draws = np.zeros(replicates)
    for suite in suites:
        tasks = sorted({k[1] for k in cells if k[0] == suite})
        task_draws = []
        for task in tasks:
            episodes = [np.mean(v) for k, v in cells.items() if k[:2] == (suite, task)]
            task_draws.append(np.mean(rng.choice(episodes, (replicates, len(episodes))), axis=1))
        task_draws = np.asarray(task_draws).T
        sampled = rng.integers(len(tasks), size=(replicates, len(tasks)))
        draws += np.take_along_axis(task_draws, sampled, axis=1).mean(axis=1)/len(suites)
    return np.quantile(draws, [.025, .975]).tolist()


def finish(root, metadata, arrays, summaries, status):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .pipeline import equal_mean, load_json
    out = root/'decomposition'
    figures = out/'figures'
    figures.mkdir()
    provenance = dict(source_prediction_sha256=status['prediction_sha256'], winner_sha256=status['winner_sha256'])
    table(out/'offset_interval_summary.csv', summaries)
    pairs = ['OldKalman-Full', 'Kalman-Full', 'HoldLast-Full', 'V14-Full', 'Kalman-V14']
    details = {}
    for pair in pairs:
        with np.load(out/(pair+'.npz'), allow_pickle=False) as archive:
            details[pair] = {field: archive[field] for field in archive.files}
    k = details['Kalman-Full']
    oracles = load_json(out/'Kalman-Full_time_oracle.json')
    n = len(metadata)
    prefix = np.asarray([min(5, m['valid_executed_prefix']) for m in metadata])
    def prefix_mean(field, data=k):
        return np.array([np.nanmean(data[field][i, :prefix[i]]) if np.isfinite(data[field][i, :prefix[i]]).any() else np.nan for i in range(n)])
    metrics = {field: prefix_mean(field) for field in ('l2_mm', 'along_mm', 'lateral_mm', 'magnitude_difference_mm')}
    metrics['oracle_residual_mm'] = np.asarray([o['residual_mm'] for o in oracles])
    metrics['oracle_improvement_mm'] = np.asarray([o['improvement_mm'] for o in oracles])
    # Every offset and interval, every pair, hierarchical confidence intervals.
    ci = []
    for pair in pairs:
        data = details[pair]
        for label, lo, hi in [('executed_1_5', 0, 5), ('diagnostic_6_12', 5, 12), ('diagnostic_13_50', 12, 50)]:
            for field in ('l2_mm', 'along_mm', 'lateral_mm'):
                values = np.array([np.nanmean(data[field][i, lo:min(hi, prefix[i]) if label == 'executed_1_5' else hi]) if np.isfinite(data[field][i, lo:min(hi, prefix[i]) if label == 'executed_1_5' else hi]).any() else np.nan for i in range(n)])
                low, high = cluster_interval(values, metadata)
                ci.append(dict(pair=pair, window=label, metric=field, ci025=low, ci975=high,
                               bootstrap='suite-stratified task then episode; replans retained within episode', replicates=2000, seed=20261016, **provenance))
    for other in ('OldKalman-Full', 'V14-Full', 'HoldLast-Full'):
        difference = metrics['l2_mm']-prefix_mean('l2_mm', details[other])
        low, high = cluster_interval(difference, metadata)
        ci.append(dict(pair='Kalman minus '+other, window='executed_1_5', metric='l2_mm', ci025=low, ci975=high,
                       bootstrap='paired suite-stratified task then episode', replicates=2000, seed=20261016, **provenance))
    table(out/'cluster_bootstrap.csv', ci)
    # Scene dimensions remain separate; all sensitivities use preregistered labels.
    cross = []
    configurations = ['primary']+sorted(metadata[0]['scene']['sensitivity'])
    for configuration in configurations:
        groups = {}
        for i, m in enumerate(metadata):
            scene = m['scene']
            geo = scene['geometry'] if configuration == 'primary' else scene['sensitivity'][configuration]
            for suite in ('all', m['suite']):
                groups.setdefault((suite, scene['phase'], geo['group'], bool(scene['transition'])), []).append(i)
        for (suite, phase, geometry, transition), indices in sorted(groups.items()):
            ids = set((metadata[i]['suite'], metadata[i]['task'], metadata[i]['init']) for i in indices)
            tasks = set(k[:2] for k in ids)
            for window, lo, hi in [('executed_1_5', 0, 5)]+[(str(h+1), h, h+1) for h in range(50)]:
                row = dict(configuration=configuration, suite=suite, phase=phase, geometry=geometry,
                           transition=transition, window=window, replans=len(indices), episodes=len(ids), tasks=len(tasks),
                           suites=len({k[0] for k in ids}), descriptive_only=len(tasks)<5 or len(ids)<20, **provenance)
                for field in ('l2_mm', 'along_mm', 'lateral_mm'):
                    cells, samples = {}, []
                    for i in indices:
                        m = metadata[i]
                        x = k[field][i, lo:min(hi, prefix[i]) if window == 'executed_1_5' else hi]
                        x = x[np.isfinite(x)]
                        if len(x):
                            cells.setdefault((m['suite'], m['task'], m['init']), []).append(x.mean())
                            samples.extend(x)
                    row[field] = equal_mean(cells)
                    row[field+'_p95'] = float(np.percentile(samples, 95)) if samples else None
                row['oracle_residual_mm'] = float(np.mean([oracles[i]['residual_mm'] for i in indices]))
                cross.append(row)
    table(out/'phase_geometry_offset.csv', cross)
    # Adjustment is an association diagnostic; categorical task fixed effects and
    # magnitude, saturation, history controls are fitted only after winner freeze.
    phases = sorted({m['scene']['phase'] for m in metadata})
    geometries = sorted({m['scene']['geometry']['group'] for m in metadata})
    tasks = sorted({(m['suite'], m['task']) for m in metadata})
    numeric = np.array([[np.linalg.norm(np.clip(arrays['full'][i, :prefix[i]], -1, 1)*50, axis=1).mean(),
                         m['clip_fraction'], np.log1p(m['history_count'])] for i,m in enumerate(metadata)])
    numeric = (numeric-numeric.mean(axis=0))/np.maximum(numeric.std(axis=0), 1e-12)
    names = ['intercept','amplitude_z','clip_fraction_z','log_history_z']+['phase='+s for s in phases[1:]]+['geometry='+s for s in geometries[1:]]+['task='+str(t) for t in tasks[1:]]+['transition']
    design = np.array([[1, *numeric[i], *[int(m['scene']['phase']==s) for s in phases[1:]],
                        *[int(m['scene']['geometry']['group']==s) for s in geometries[1:]],
                        *[int((m['suite'],m['task'])==t) for t in tasks[1:]], int(m['scene']['transition'])] for i,m in enumerate(metadata)])
    coefficients, _, rank, _ = np.linalg.lstsq(design, metrics['l2_mm'], rcond=None)
    table(out/'controlled_association.csv', [dict(term=name, coefficient_mm=float(value), rank=int(rank), columns=len(names),
            baseline_phase=phases[0], baseline_geometry=geometries[0], interpretation='descriptive OLS association; no causal inference or independent-replan CI', **provenance) for name,value in zip(names,coefficients)])
    # Signed axes and validity/stop fractions are explicit, never filled with zero.
    axes = []
    for pair,data in details.items():
        for axis,name in enumerate('xyz'):
            values = np.concatenate([data['error_mm'][i,:prefix[i],axis] for i in range(n)])
            axes.append(dict(pair=pair, axis=name, mean_signed_mm=float(values.mean()), mean_absolute_mm=float(abs(values).mean()),
                             positive_fraction=float((values>0).mean()), negative_fraction=float((values<0).mean()), p95_absolute_mm=float(np.percentile(abs(values),95)), **provenance))
    table(out/'signed_axes.csv', axes)
    valid = dict(low_motion_fraction=float(k['reference_low_motion'].mean()), direction_undefined_fraction=float((~k['direction_valid']).mean()),
                 tangent_undefined_fraction=float((~k['tangent_valid']).mean()), time_shift_unidentifiable_fraction=float(np.mean([o['unidentifiable'] for o in oracles])), **provenance)
    write_json(out/'validity_fractions.json', valid, exclusive=True)
    for end in (12, 50):
        fig, ax = plt.subplots(figsize=(8,4))
        for pair in pairs[:4]:
            rows = [r for r in summaries if r['pair']==pair and r['metric']=='l2_mm' and r['window'].isdigit() and int(r['window'])<=end]
            ax.plot(range(1,end+1), [r['equal_mean'] for r in rows], label=pair)
            ax.fill_between(range(1,end+1), [r['sample_p50'] for r in rows], [r['sample_p95'] for r in rows], alpha=.08)
        ax.axvspan(.5,5.5,color='gray',alpha=.12); ax.set(xlabel='Prediction offset (0.05 s/action)',ylabel='Command increment L2 difference (mm)')
        ax.legend(); fig.tight_layout(); fig.savefig(figures/f'offset_1_{end}.png',dpi=150); plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(13,7))
    for ax,(field,values) in zip(axs.flat,metrics.items()):
        ax.hist(values[np.isfinite(values)],bins=60); ax.set(title=field,xlabel='mm (command / proxy)')
    fig.tight_layout(); fig.savefig(figures/'decomposition_distributions.png',dpi=150); plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(14,8))
    for ax,suite in zip(axs.flat,['all']+sorted({m['suite'] for m in metadata})):
        image=np.full((len(phases),len(geometries)),np.nan); labels={}
        for y,phase in enumerate(phases):
            for x,geo in enumerate(geometries):
                rows=[r for r in cross if r['configuration']=='primary' and r['suite']==suite and r['phase']==phase and r['geometry']==geo and r['window']=='executed_1_5']
                if rows:
                    total=sum(r['replans'] for r in rows)
                    image[y,x]=sum(r['l2_mm']*r['replans'] for r in rows)/total
                    labels[y,x]=str(total)
        ax.imshow(image,aspect='auto');ax.set(title=suite,xticks=range(len(geometries)),xticklabels=geometries,yticks=range(len(phases)),yticklabels=phases)
        ax.tick_params(axis='x',rotation=30)
        for (y,x),label in labels.items(): ax.text(x,y,label,ha='center',color='white')
    axs.flat[-1].axis('off');fig.suptitle('Prefix L2 (sample-weighted cell display); annotations = replan count; see CSV for equal means and coverage')
    fig.tight_layout();fig.savefig(figures/'phase_geometry.png',dpi=150);plt.close(fig)
    # Fixed six category selection; deterministic index ties, global deduplication.
    near=np.array([m['scene']['geometry']['group']=='near_straight' for m in metadata])
    choices=[('lateral_max',metrics['lateral_mm'],None,False),('along_lead_max',metrics['along_mm'],None,False),
             ('along_lag_max',metrics['along_mm'],None,True),('time_alignment_improvement',metrics['oracle_improvement_mm'],None,False),
             ('near_straight_good',metrics['l2_mm'],near,True),('near_straight_bad',metrics['l2_mm'],near,False)]
    trajectories=load_json(root/'evaluation/eef_trajectories.json')
    trajectory_index={(r['suite'],r['task'],r['init']):r for r in trajectories}
    cases=[];seen=set();full_index=[]
    for category,values,mask,ascending in choices:
        ids=[i for i in range(n) if np.isfinite(values[i]) and (mask is None or mask[i])]
        ids.sort(key=lambda i:(values[i] if ascending else -values[i],i))
        full_index.extend(dict(category=category,rank=j+1,index=i,value=float(values[i])) for j,i in enumerate(ids))
        selected=0
        for i in ids:
            if i in seen:continue
            seen.add(i);selected+=1;m=metadata[i];trajectory=trajectory_index[(m['suite'],m['task'],m['init'])]
            row=dict(category=category,index=i,value=float(values[i]),**m,source_video=trajectory['video_candidates'],telemetry_path=trajectory['telemetry_path'])
            cases.append(row)
            fig=plt.figure(figsize=(12,4));ax=fig.add_subplot(121,projection='3d')
            xyz=np.asarray(trajectory['eef_xyz_m']);ax.plot(*xyz.T,color='blue',label='Full actual EEF');ax.scatter(*xyz[m['step']],color='black');ax.set_title('Actual executed Full EEF (m)');ax.legend()
            ax=fig.add_subplot(122,projection='3d')
            for key,label in [('full','Full'),('v14','V14'),('clipped_mean','Kalman')]:
                proxy=np.vstack([np.zeros(3),np.cumsum(np.clip(arrays[key][i],-1,1)*50,axis=0)])
                ax.plot(*proxy.T,label=label)
            ax.set_title('Control target path proxies (mm)');ax.legend();fig.tight_layout();fig.savefig(figures/f'case_{i}.png',dpi=120);plt.close(fig)
            if selected==3:break
    write_json(out/'cases.json',cases,exclusive=True);table(out/'case_selection_full_index.csv',full_index)
    html='<html><body><h1>E4 actual Full videos and separate command proxies</h1>'
    for c in cases:
        html+=f'<h2>{c["category"]}: {c["suite"]} task{c["task"]} init{c["init"]} replan{c["replan"]}</h2><img width="900" src="figures/case_{c["index"]}.png">'
        for video in c['source_video']:html+=f'<p><a href="{video}">Original Full video</a></p><video controls width="640" src="{video}"></video>'
    (out/'gallery.html').write_text(html+'</body></html>')
    # Independently recompute every L2 summary from saved arrays and metadata.
    checked=0
    for row in summaries:
        if row['metric']!='l2_mm':continue
        if row['window'].isdigit():lo=int(row['window'])-1;hi=lo+1
        else:lo,hi={'executed_1_5':(0,5),'diagnostic_6_12':(5,12),'diagnostic_13_50':(12,50)}[row['window']]
        a,b={'OldKalman-Full':('old_kalman','full'),'Kalman-Full':('clipped_mean','full'),'HoldLast-Full':('hold_last','full'),'V14-Full':('v14','full'),'Kalman-V14':('clipped_mean','v14')}[row['pair']]
        raw=np.linalg.norm(50*(np.clip(arrays[a],-1,1)-np.clip(arrays[b],-1,1)),axis=2)
        cells={};samples=[]
        for i,m in enumerate(metadata):
            values=raw[i,lo:min(hi,prefix[i]) if row['window']=='executed_1_5' else hi]
            if len(values):cells.setdefault((m['suite'],m['task'],m['init']),[]).append(values.mean());samples.extend(values)
        if not np.isclose(equal_mean(cells),row['equal_mean'],atol=1e-10):raise ValueError('Independent summary audit failed')
        if len(samples)!=row['valid_samples'] or not np.isclose(np.percentile(samples,95),row['sample_p95']):raise ValueError('Independent denominator/tail audit failed')
        checked+=1
    winner=load_json(root/'winner.json');reproduction=load_json(root/'reproduction.json')
    primary={r['pair']:r['equal_mean'] for r in summaries if r['window']=='executed_1_5' and r['metric']=='l2_mm'}
    missing=['Human semantic-phase video verification remains unperformed; fixed cases and original video links supplied.']
    report=f'''# E4 command-space Kalman confirmation

Full source reproduction: {reproduction['points']} points; maximum raw mean/covariance discrepancy {reproduction['max_array_difference']:.3g}. Calibration uses 634 points / 20 episodes, five held-out official init folds. Evaluation uses {n} points / 180 episodes / 36 suite-task identities. Previously inspected historical data are reused; this is not a blind test.

Frozen selection status: {winner['status']}; {winner['candidate_count']} distinct candidates, at most 211 preregistered slots. Winner: `{winner['parameters']}`. Calibration prefix score {winner['calibration_score_mm']:.5f} mm vs old {winner['old_reference_score_mm']:.5f} mm; boundary winner {winner.get('boundary_winner')}. Shape matrices use training calibration episodes only; folds, eigenvalues, hashes and tail checks are in calibration receipts.

Evaluation executable-prefix 1–5 episode→task→suite equal means: {json.dumps(primary)} mm. Kalman change against old: {primary['Kalman-Full']-primary['OldKalman-Full']:+.5f} mm; against V14: {primary['Kalman-Full']-primary['V14-Full']:+.5f} mm. Paired hierarchical bootstrap uncertainty is in cluster_bootstrap.csv. Offset 6–50 are future prediction diagnostics, not executed Kalman motion. Full is a same-observation reference, not uniquely correct action truth.

The signed along-path and lateral quantities describe cumulative control-target path proxies. Amplitude error can produce apparent along-path lead; it does not prove temporal lead. Oracle time-shift selection is post hoc, with equal fixed overlap and no deployment claim. Unidentifiable/low-motion fractions: {json.dumps(valid)}. Axis, amplitude, direction, and tails are stored separately, with undefined directions retained as NaN.

Phase and actual EEF straightness are independent post hoc labels. phase_geometry_offset.csv provides all primary/sensitivity cells, suite facets, transition status and coverage. Cells below five tasks or twenty episodes support descriptive statements only. controlled_association.csv adjusts amplitude, clipping, history and suite/task; its coefficients are association diagnostics, not causal estimates. Sparse coverage, noisy phase labels and only four calibration tasks limit extrapolation. Figures shade sample p50–p95; these are not confidence bands.

Fixed-rule deduplicated cases and complete ranking indices are provided in gallery.html/cases.json. Blue curves are actual Full EEF telemetry; separate Full/V14/Kalman curves are command target proxies. Original Full videos are linked. No Kalman policy was executed, no new model calls or GPU experiments were run. Human video validation is still required before declaring every E4 requirement complete.

Independent saved-array audit checked {checked} L2 means/tails/denominators. Final status: computational analysis complete; human phase validation pending. See completion_audit.json for exact remaining items.
'''
    (root/'report.md').write_text(report)
    audit=dict(status='computational_complete_human_review_pending', experiments_complete=False, source_points=6979, calibration_points=634,evaluation_points=n,
               evaluation_unique_keys=len({(m['suite'],m['task'],m['init'],m['replan']) for m in metadata}),
               calibration_task=0,evaluation_tasks=sorted({m['task'] for m in metadata}), candidates=winner['candidate_count'],
               independent_l2_summary_checks=checked, source_reproduction_max_difference=reproduction['max_array_difference'],
               model_calls=0,environment_actions=0,gpu_jobs=0,missing=missing,**provenance)
    audit['artifacts']={str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('completion_audit.json','missing_or_deferred.json')}
    write_json(root/'completion_audit.json',audit,exclusive=True)
    write_json(root/'missing_or_deferred.json',dict(status=audit['status'],missing=missing),exclusive=True)
    return audit

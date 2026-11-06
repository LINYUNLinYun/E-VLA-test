"""Complete-coverage numerical/functional probe analysis, without GPU or env."""
import argparse
from collections import defaultdict
import json

import numpy as np

from scripts.experiment11.protocol import RESULTS, SUITES
from scripts.experiment11 import run_task_type_layer_probe as runner
from scripts.experiment11.analyze_failure_events import write_csv
from scripts.experiment11.shadow_risk_metrics import differences, task_cluster_summary
from scripts.experiment10.common import dump_json, sha256, json_lines

ACTION_FIELDS = ('normalized_rmse_h4','normalized_rmse_h5','translation_target_mm_h5',
                 'rotation_increment_deg_h5','full_open_variant_close_h5','full_close_variant_open_h5')
NUMERIC_FIELDS = ('rms','mean','std','max_abs','zero_fraction','near_zero_0.001','near_zero_0.01',
                  'near_zero_0.1','top10pct_token_energy_fraction','relative_residual','cosine')


def episode_task_means(rows, fields):
    """Each episode first, then each task; unobserved phases are not zero-filled."""
    episodes = defaultdict(list)
    for row in rows:
        episodes[(row['suite'],row['task_id'],row['episode_idx'])].append(row)
    tasks = defaultdict(list)
    for (suite,task,_), subset in episodes.items():
        values = np.asarray([[np.nan if r.get(k) is None else r[k] for k in fields] for r in subset],float)
        valid = np.isfinite(values).sum(axis=0)
        mean = np.divide(np.nansum(values,axis=0),valid,out=np.full(len(fields),np.nan),where=valid>0)
        tasks[(suite,task)].append(mean)
    result = {}
    for key, values in tasks.items():
        values = np.asarray(values);valid = np.isfinite(values).sum(axis=0)
        result[key] = np.divide(np.nansum(values,axis=0),valid,out=np.full(len(fields),np.nan),where=valid>0)
    return result, len(episodes)


def ranks(values):
    """Average ranks for ties, not arbitrary layer-ID tie breaks."""
    a = np.asarray(values,float)
    return np.asarray([1+np.sum(a<v)+(np.sum(a==v)-1)/2 for v in a])


def spearman(a,b):
    a,b = np.asarray(a,float),np.asarray(b,float)
    good = np.isfinite(a)&np.isfinite(b);a,b = ranks(a[good]),ranks(b[good])
    if len(a)<3 or np.std(a)==0 or np.std(b)==0:
        return None,int(len(a))
    return float(np.corrcoef(a,b)[0,1]),int(len(a))


def centered_interaction(matrix):
    """Remove additive layer/suite main effects; equal suite weighting."""
    matrix=np.asarray(matrix,float)
    row_centered=matrix-matrix.mean(axis=-1,keepdims=True)
    return row_centered-row_centered.mean(axis=-2,keepdims=True)


def action_summary(rows, identity):
    tasks, episodes = episode_task_means(rows,ACTION_FIELDS)
    keys = sorted(tasks);values = np.asarray([tasks[k] for k in keys])
    if not np.isfinite(values).all():
        raise RuntimeError('Nonfinite functional metric')
    if len(keys)>=3:
        mean,ci,_ = task_cluster_summary(values,[f'{s}:{t}' for s,t in keys],strata=np.asarray([s for s,_ in keys]))
    else:
        mean,ci = values.mean(axis=0),np.full((2,len(ACTION_FIELDS)),np.nan)
    return [{**identity,'metric':field,'mean':float(mean[i]),
             'ci_low':float(ci[0,i]) if np.isfinite(ci[0,i]) else None,
             'ci_high':float(ci[1,i]) if np.isfinite(ci[1,i]) else None,
             'tasks':len(keys),'episodes':episodes,'samples':len(rows),'exploratory':True}
            for i,field in enumerate(ACTION_FIELDS)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt-id',default='round2_20261011')
    parser.add_argument('--retry-id',default='warmup_retry1')
    args = parser.parse_args()
    jobs = runner.configure(args.attempt_id,args.retry_id)()[0]
    missing = [j['job_id'] for j in jobs if not runner.scheduler.validate_job(j)]
    if missing:
        raise RuntimeError(f'Probe coverage incomplete; {len(missing)} jobs missing: {missing}')
    directory,manifest_path,_ = runner.paths(args.attempt_id)
    manifest = runner.verify_manifest(manifest_path)
    expected = {s['sample_id']:s for s in manifest['samples']}
    output = RESULTS/'analysis/next_round'/args.attempt_id/'task_type_probe'
    output.mkdir(parents=True,exist_ok=True)
    actions, numerical, seen, activation_seen = [],[],set(),set()
    hashes = {str(manifest_path):sha256(manifest_path)}
    for job in jobs:
        prediction,marker,_ = runner.scheduler.output_paths(job)
        activity = marker.parent/'activation_statistics.json'
        for p in (prediction,marker,activity):
            hashes[str(p)] = sha256(p)
        for row in json_lines(prediction):
            sample = expected[row['sample_id']];key = (row['sample_id'],row['layer'])
            if key in seen or row['layer'] not in runner.LAYERS:
                raise RuntimeError('Duplicate/unexpected layer prediction')
            seen.add(key)
            for field in ('suite','task_id','episode_idx','phase','input_sha256','noise_sha256','checkpoint_sha256'):
                if row[field] != sample[field]:
                    raise RuntimeError('Prediction identity mismatch')
            full = np.asarray(row['full_actions']);reference = np.asarray(sample['full_actions'])
            if np.max(np.abs(full-reference))>1e-5:
                raise RuntimeError('Saved Full parity failed')
            diff = differences(full,row['variant_actions'])
            metrics = {int(m['horizon']):m for m in row['metrics']}
            identity = {k:row[k] for k in ('sample_id','suite','task_id','episode_idx','phase','layer')}
            motion = float(np.mean(np.linalg.norm(np.clip(full[:5,:3],-1,1),axis=1))*50)
            actions.append({**identity,'normalized_rmse_h4':metrics[4]['normalized_continuous_rmse'],
                'normalized_rmse_h5':metrics[5]['normalized_continuous_rmse'],
                'translation_target_mm_h5':float(diff[:5,3].mean()),
                'rotation_increment_deg_h5':float(diff[:5,4].mean()),
                'full_open_variant_close_h5':float(diff[:5,5].mean()),
                'full_close_variant_open_h5':float(diff[:5,6].mean()),'full_motion_mm_h5':motion})
        for row in json.loads(activity.read_text())['rows']:
            if row['layer']==18:
                continue
            key = (row['sample_id'],row['layer'],row['modality'],row['tensor'])
            if key in activation_seen or row['sample_id'] not in expected:
                raise RuntimeError('Duplicate/unexpected activation statistic')
            activation_seen.add(key);numerical.append(row)
    if seen != {(sid,l) for sid in expected for l in runner.LAYERS}:
        raise RuntimeError('Missing functional cells')
    activation_expected = {(sid,l,m,t) for sid in expected for l in runner.LAYERS
        for m in ('agent','wrist','language') for t in ('H_in','H_attn','H_out','delta_attn','delta_ffn','delta_block')}
    if activation_seen != activation_expected:
        raise RuntimeError('Missing numerical cells')
    coverage=[]
    for suite in SUITES:
        for task in range(10):
            for phase in runner.PHASES:
                selected=[s for s in expected.values() if s['suite']==suite and s['task_id']==task and s['phase']==phase]
                episodes={s['episode_idx'] for s in selected}
                if len(selected)!=len(episodes):
                    raise RuntimeError('More than one selected point per episode/phase')
                coverage.append({'suite':suite,'task_id':task,'phase':phase,'valid_episodes':len(episodes),
                                 'planned_episodes':5,'missing_episodes':5-len(episodes),
                                 'sampling':'first real replan in reached Full phase;no synthetic fill'})
    write_csv(output/'probe_phase_coverage.csv',coverage)
    motion_by_sample = {r['sample_id']:r['full_motion_mm_h5'] for r in actions}
    thresholds = np.quantile(list(motion_by_sample.values()),[1/3,2/3])
    for row in actions:
        row['motion_bin'] = ('low','medium','high')[int(np.searchsorted(thresholds,row['full_motion_mm_h5'],side='right'))]
    write_csv(output/'sample_layer_action_metrics.csv',actions)
    groups = defaultdict(list)
    for row in actions:
        for suite in ('all',row['suite']):
            for phase in ('all',row['phase']):
                groups[(row['layer'],suite,phase,'all')].append(row)
            groups[(row['layer'],suite,'all',row['motion_bin'])].append(row)
    summaries = []
    for (layer,suite,phase,motion),rows in groups.items():
        summaries.extend(action_summary(rows,{'layer':layer,'suite':suite,'phase':phase,'motion_bin':motion}))
    write_csv(output/'layer_suite_phase_motion_summary.csv',summaries)
    # Repeated interventions are paired within tasks; task identities across
    # suites are not paired. These are exploratory descriptive interactions.
    interactions=[];rng=np.random.default_rng(20261011)
    for phase,motion in [('all','all')]+[(p,'all') for p in runner.PHASES]+[('all',m) for m in ('low','medium','high')]:
        for field in ('normalized_rmse_h4','translation_target_mm_h5'):
            matrices=[];task_counts=[]
            for suite in SUITES:
                layer_maps=[]
                for layer in runner.LAYERS:
                    rows=[r for r in actions if r['suite']==suite and r['layer']==layer and
                          (phase=='all' or r['phase']==phase) and (motion=='all' or r['motion_bin']==motion)]
                    layer_maps.append(episode_task_means(rows,(field,))[0])
                keys=sorted(set.intersection(*(set(m) for m in layer_maps)))
                task_counts.append(len(keys))
                matrices.append(np.asarray([[m[k][0] for m in layer_maps] for k in keys]))
            if any(n==0 for n in task_counts):
                continue
            point=centered_interaction(np.stack([m.mean(axis=0) for m in matrices],axis=-1))
            boot=np.stack([m[rng.integers(0,len(m),size=(2000,len(m)))].mean(axis=1) for m in matrices],axis=-1)
            boot=centered_interaction(boot);ci=np.quantile(boot,[.025,.975],axis=0)
            for li,layer in enumerate(runner.LAYERS):
                for si,suite in enumerate(SUITES):
                    interactions.append({'layer':layer,'suite':suite,'phase':phase,'motion_bin':motion,'metric':field,
                        'centered_interaction':float(point[li,si]),'tasks':task_counts[si],
                        'ci_low':float(ci[0,li,si]) if min(task_counts)>=3 else None,
                        'ci_high':float(ci[1,li,si]) if min(task_counts)>=3 else None,
                        'weighting':'episode->task;equal suite centering;paired layers within task',
                        'scope':'exploratory;unadjusted intervals;task/motion distributions not randomized'})
    write_csv(output/'layer_suite_phase_interactions.csv',interactions)
    # Task-level leave-one-out rank stability for the primary h4 criterion.
    loo = []
    for suite in ('all',*SUITES):
        task_matrices = []
        for layer in runner.LAYERS:
            rows = [r for r in actions if r['layer']==layer and (suite=='all' or r['suite']==suite)]
            task_matrices.append(episode_task_means(rows,('normalized_rmse_h4',))[0])
        keys = sorted(set.intersection(*(set(m) for m in task_matrices)))
        matrix = np.asarray([[m[k][0] for m in task_matrices] for k in keys])
        rank = ranks(matrix.mean(axis=0))
        leave = np.asarray([ranks(np.delete(matrix,i,axis=0).mean(axis=0)) for i in range(len(keys))])
        for i,layer in enumerate(runner.LAYERS):
            loo.append({'suite':suite,'layer':layer,'tasks':len(keys),'rank':float(rank[i]),
                        'loo_rank_min':float(leave[:,i].min()),'loo_rank_max':float(leave[:,i].max()),
                        'loo_top3_fraction':float(np.mean(leave[:,i]<=3))})
    write_csv(output/'leave_one_task_out_ranks.csv',loo)
    # Numeric means are episode->task means; absent/zero denominators stay missing.
    numeric_groups = defaultdict(list)
    for row in numerical:
        numeric_groups[(row['layer'],row['modality'],row['tensor'],row['suite'],row['phase'])].append(row)
    numeric_summary = []
    for (layer,modality,tensor,suite,phase),rows in numeric_groups.items():
        tasks,episodes = episode_task_means(rows,NUMERIC_FIELDS)
        values = np.asarray(list(tasks.values()))
        for i,field in enumerate(NUMERIC_FIELDS):
            valid = values[:,i][np.isfinite(values[:,i])]
            numeric_summary.append({'layer':layer,'modality':modality,'tensor':tensor,'suite':suite,'phase':phase,
                                    'metric':field,'mean':float(valid.mean()) if len(valid) else None,
                                    'tasks_valid':len(valid),'episodes':episodes,'samples':len(rows)})
    write_csv(output/'numerical_layer_modality_suite_phase.csv',numeric_summary)
    associations = []
    influence = {(r['sample_id'],r['layer']):r['normalized_rmse_h4'] for r in actions}
    for suite in ('all',*SUITES):
        for modality in ('agent','wrist','language'):
            for tensor in ('delta_attn','delta_ffn','delta_block'):
                rows = [dict(r,action_influence=influence[(r['sample_id'],r['layer'])]) for r in numerical
                        if r['modality']==modality and r['tensor']==tensor and (suite=='all' or r['suite']==suite)]
                for feature in ('rms','relative_residual','cosine','top10pct_token_energy_fraction'):
                    pooled = []
                    for layer in runner.LAYERS:
                        tasks,_ = episode_task_means([r for r in rows if r['layer']==layer],(feature,'action_influence'))
                        pooled.extend(tasks.values())
                        values = np.asarray(list(tasks.values()))
                        rho,n = spearman(values[:,0],values[:,1])
                        associations.append({'suite':suite,'layer':layer,'modality':modality,'tensor':tensor,
                            'feature':feature,'spearman':rho,'task_layer_pairs':n,
                            'scope':'within-layer;task-aggregated;exploratory;not causal'})
                    values = np.asarray(pooled);rho,n = spearman(values[:,0],values[:,1])
                    associations.append({'suite':suite,'layer':'all','modality':modality,'tensor':tensor,'feature':feature,
                                         'spearman':rho,'task_layer_pairs':n,'scope':'exploratory;layer-confounded;not causal'})
    write_csv(output/'numerical_action_associations.csv',associations)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric in ('normalized_rmse_h4','translation_target_mm_h5'):
        fig,ax = plt.subplots(figsize=(8,6),layout='constrained')
        grid = np.asarray([[next(r['mean'] for r in summaries if r['layer']==layer and r['suite']==suite
                                and r['phase']=='all' and r['motion_bin']=='all' and r['metric']==metric)
                            for suite in SUITES] for layer in runner.LAYERS])
        im=ax.imshow(grid,aspect='auto',cmap='magma');fig.colorbar(im,ax=ax,label=metric)
        ax.set_xticks(range(4),SUITES,rotation=25,ha='right');ax.set_yticks(range(17),[f'L{l}' for l in runner.LAYERS])
        ax.set_title('Single-layer skip influence; episode then task equal weighting')
        fig.savefig(output/f'{metric}_layer_suite.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
    grid=np.asarray([[next(r['centered_interaction'] for r in interactions if r['layer']==l and r['suite']==s
                    and r['phase']=='all' and r['motion_bin']=='all' and r['metric']=='normalized_rmse_h4')
                    for s in SUITES] for l in runner.LAYERS])
    limit=max(float(np.max(np.abs(grid))),1e-12)
    image=ax.imshow(grid,aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit)
    fig.colorbar(image,ax=ax,label='Additive-centered action RMSE interaction')
    ax.set_xticks(range(4),SUITES,rotation=25,ha='right');ax.set_yticks(range(17),[f'L{l}' for l in runner.LAYERS])
    ax.set_title('Descriptive layer × suite modulation; not causal specialization')
    fig.savefig(output/'layer_suite_centered_interaction.png',dpi=180);plt.close(fig)
    for feature in ('relative_residual','cosine'):
        fig,axes = plt.subplots(3,2,figsize=(11,13),layout='constrained')
        for mi,modality in enumerate(('agent','wrist','language')):
            for ti,tensor in enumerate(('delta_attn','delta_ffn')):
                grid = []
                for layer in runner.LAYERS:
                    line = []
                    for suite in SUITES:
                        rows = [r for r in numerical if r['layer']==layer and r['suite']==suite
                                and r['modality']==modality and r['tensor']==tensor]
                        tasks,_ = episode_task_means(rows,(feature,))
                        values = np.asarray(list(tasks.values()))[:,0]
                        valid = values[np.isfinite(values)]
                        line.append(float(valid.mean()) if len(valid) else np.nan)
                    grid.append(line)
                ax = axes[mi,ti]
                image = ax.imshow(grid,aspect='auto',cmap='viridis')
                fig.colorbar(image,ax=ax,label=feature)
                ax.set(title=f'{modality}: {tensor}',xlabel='suite',ylabel='layer')
                ax.set_xticks(range(4),SUITES,rotation=25,ha='right')
                ax.set_yticks(range(17),[f'L{l}' for l in runner.LAYERS])
        fig.suptitle('Full activations; episode then task means; numerical activity is not pruning proof')
        fig.savefig(output/f'attention_ffn_{feature}_heatmap.png',dpi=160);plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,tensor in zip(axes,('delta_attn','delta_ffn')):
        x,y,color = [],[],[]
        for layer in runner.LAYERS:
            rows = [dict(r,action_influence=influence[(r['sample_id'],layer)]) for r in numerical
                    if r['layer']==layer and r['modality']=='agent' and r['tensor']==tensor]
            tasks,_ = episode_task_means(rows,('relative_residual','action_influence'))
            for values in tasks.values():
                if np.isfinite(values).all():
                    x.append(values[0]);y.append(values[1]);color.append(layer)
        dots = ax.scatter(x,y,c=color,cmap='viridis',s=12,alpha=.4,vmin=1,vmax=17)
        fig.colorbar(dots,ax=ax,label='layer')
        ax.set(xlabel='Relative residual norm (agent tokens)',ylabel='Single-layer action RMSE h4',title=tensor)
        ax.grid(alpha=.2)
    fig.suptitle('Task × layer points; pooled relationship is layer-confounded, not causal')
    fig.savefig(output/'numerical_vs_action_influence.png',dpi=180);plt.close(fig)
    dump_json(output/'completion_audit.json',{'complete':True,'samples':len(expected),'functional_records':len(actions),
              'numeric_records':len(numerical),'source_hashes':hashes,'motion_bin_thresholds_mm':thresholds.tolist(),
              'missing_phase_cells':len(manifest['missing_phase_cells']),'new_predictions':0,'environment_steps':0,
              'limitations':['exploratory_tasks','association_not_causal','L18_KV_only_excluded','no_new_module_skip'],
              'outputs':{p.name:sha256(p) for p in output.iterdir() if p.is_file() and p.name!='completion_audit.json'}})
    print(json.dumps({'complete':True,'samples':len(expected),'analysis':str(output)}),flush=True)


if __name__=='__main__':
    main()

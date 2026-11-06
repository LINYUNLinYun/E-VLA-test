"""CPU-only, reproducible final audit/plots of completed Experiment10 and attention pairs.

Run from repository root: python3 scripts/experiment10/build_final_report.py
No model execution, GPU polling, or changes to original experimental artifacts.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'results/experiment10_phase_draft_20261009'
OUT = DATA / 'final_analysis'
REPORT = ROOT / 'experiment10_final_analysis_report.md'
SUITES = ['libero_object', 'libero_spatial', 'libero_goal', 'libero_10']
SHORT = ['Object', 'Spatial', 'Goal', 'LIBERO-10']
PHASES = ['approach', 'contact_grasp', 'transport', 'alignment_release']
PHASE_MAP = {'initial': 'approach', 'approach': 'approach', 'contact': 'contact_grasp',
             'grasp_or_manipulate': 'contact_grasp', 'transport': 'transport',
             'goal_alignment': 'alignment_release', 'release_or_final_action': 'alignment_release'}
MAIN = ['Full18_D10', 'Full18_D1', 'Full18_D2', 'Full18_D3', 'Full18_D4', 'Full18_D5',
        'FrozenStatic01', 'FrozenStatic02', 'FrozenLegal01', 'FrozenLegal02',
        'Recover_D3_H8', 'Recover_V16_D10_H8']
PROVENANCE = {}
plt.rcParams.update({'figure.dpi': 140, 'savefig.dpi': 180, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': False, 'figure.facecolor': 'white'})


def record(path):
    path = Path(path)
    PROVENANCE[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read_json(path):
    return json.loads(record(path).read_text())


def read_csv(path, **kw):
    return pd.read_csv(record(path), **kw)


def write_csv(df, name):
    df.to_csv(OUT / name, index=False)


def save(fig, name):
    fig.savefig(OUT / (name + '.png'), bbox_inches='tight')
    fig.savefig(OUT / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def link(path, label=None):
    return f'[{label or Path(path).name}]({os.path.relpath(path, ROOT)})'


def image(name, label):
    return f'![{label}]({os.path.relpath(OUT / (name + ".png"), ROOT)})'


def table(headers, rows):
    headers = [str(value).replace('|', '\\|') for value in headers]
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def bootstrap_pairs(pair, task_weighted=False):
    """Task-cluster, suite-stratified paired differences; not episode IID intervals."""
    rng = np.random.default_rng(20261010)
    draws = np.zeros(10000)
    means, weights = [], []
    for suite, group in pair.groupby('suite', sort=True):
        vals = group.groupby('task_id')['difference'].mean().to_numpy()
        weight = len(vals) if task_weighted else 1
        means.append(vals.mean()); weights.append(weight)
        draws += weight * vals[rng.integers(0, len(vals), (10000, len(vals)))].mean(axis=1)
    draws /= sum(weights)
    low, high = np.quantile(draws, [.025, .975]) * 100
    return np.average(means, weights=weights) * 100, low, high


def trajectories(path):
    """Read true measured XYZ; retain small arrays, not all object-state JSON."""
    xyz, phases, steps, gripper, contacts = [], [], [], [], []
    with Path(path).open() as stream:
        for line in stream:
            d = json.loads(line)
            xyz.append(d['eef_pos'])
            phases.append(PHASE_MAP.get(d.get('semantic_phase'), d.get('semantic_phase', 'unknown')))
            steps.append(d.get('step', d.get('state_index', len(steps))))
            gripper.append(d.get('gripper_width_proxy', np.nan))
            contacts.append(any(d.get('robot_contacts', {}).values()) or
                            any(d.get('robot_grasps', {}).values()))
    return {'xyz': np.array(xyz), 'phase': np.array(phases), 'step': np.array(steps),
            'gripper': np.array(gripper, dtype=float), 'contact': np.array(contacts)}


def first_deviation(b, v):
    shared, bi, vi = np.intersect1d(b['step'], v['step'], return_indices=True)
    if not len(shared):
        raise RuntimeError('No shared trajectory steps')
    errors = np.linalg.norm(b['xyz'][bi] - v['xyz'][vi], axis=1)
    above = errors > .02
    onset = None
    for i in range(len(above) - 2):
        if above[i:i+3].all() and np.diff(shared[i:i+3]).tolist() == [1, 1]:
            onset = i
            break
    contact = b['step'][b['contact']]
    first_contact = int(contact[0]) if len(contact) else None
    return {'shared_states': len(shared), 'xyz_mean_m': float(errors.mean()),
            'xyz_p95_m': float(np.quantile(errors, .95)), 'xyz_max_m': float(errors.max()),
            'onset_step': int(shared[onset]) if onset is not None else None,
            'baseline_phase_at_onset': str(b['phase'][bi[onset]]) if onset is not None else 'no_onset',
            'baseline_first_contact_step': first_contact,
            'onset_relative_to_contact': ('no_onset' if onset is None else 'no_baseline_contact'
                 if first_contact is None else 'pre_contact' if shared[onset] < first_contact else 'at_or_after_contact')}


def xyz_plot(b, v, name, title):
    fig = plt.figure(figsize=(13, 7.4))
    ax = fig.add_subplot(221, projection='3d')
    for d, color, style, marker, label in [(b, '#2274b5', '-', 'o', 'Baseline'),
                                          (v, '#f28e2b', '--', '^', 'Variant')]:
        p = d['xyz']
        ax.plot(*p.T, color=color, ls=style, alpha=.65, lw=1.5, marker=marker,
                markevery=max(1, len(p)//18), ms=3.5, label=label)
        ax.scatter(*p[0], color=color, s=25)
    ax.set(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)', title='Measured EEF: world coordinates')
    ax.legend(loc='upper left', fontsize=8)
    p = np.concatenate([b['xyz'], v['xyz']]); span = np.maximum(np.ptp(p, axis=0), .025)
    ax.set_box_aspect(span)
    ax2 = fig.add_subplot(222)
    for d, color, style, marker in [(b, '#2274b5', '-', 'o'), (v, '#f28e2b', '--', '^')]:
        ax2.plot(d['xyz'][:, 0], d['xyz'][:, 1], color=color, ls=style, alpha=.65,
                 marker=marker, markevery=max(1, len(d['xyz'])//18), ms=3)
    ax2.set(xlabel='X (m)', ylabel='Y (m)', title='XY projection'); ax2.set_aspect('equal', adjustable='datalim')
    ax3 = fig.add_subplot(223)
    shared, bi, vi = np.intersect1d(b['step'], v['step'], return_indices=True)
    err = np.linalg.norm(b['xyz'][bi]-v['xyz'][vi], axis=1) * 1000
    ax3.plot(shared, err, color='#5f4690'); ax3.axhline(20, ls=':', color='gray', label='20 mm threshold')
    for phase in PHASES:
        indices = np.where(b['phase'] == phase)[0]
        if len(indices): ax3.axvline(b['step'][indices[0]], color='gray', alpha=.3)
    ax3.set(xlabel='Environment action step (20 Hz)', ylabel='XYZ deviation (mm)', title='Shared-step overlap only')
    ax3.legend(fontsize=8)
    ax4 = fig.add_subplot(224)
    for d, color, style, label in [(b, '#2274b5', '-', 'Baseline'), (v, '#f28e2b', '--', 'Variant')]:
        ax4.plot(d['step'], d['gripper'], color=color, ls=style, alpha=.8, label=label)
    ax4.set(xlabel='Environment action step', ylabel='Gripper width proxy (m)', title='Gripper state (not command)')
    ax4.legend(fontsize=8); fig.suptitle(title, fontsize=12); fig.tight_layout()
    save(fig, name)


def synchronized_video(example):
    """Shared-state interval only: no frozen ending or interpolated simulation frames."""
    if example['composite_video']:
        return
    target = OUT / (Path(example['plot']).stem + '_synchronized.mp4')
    # Both inputs encode step0 initial state then one frame per executed action at 20Hz.
    filters = ("[0:v]setpts=PTS-STARTPTS,drawtext=text='Baseline Full18':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=0x2274b5@0.7[b];"
               "[1:v]setpts=PTS-STARTPTS,drawtext=text='Variant':x=8:y=8:fontsize=16:fontcolor=white:box=1:boxcolor=0xf28e2b@0.7[v];"
               "[b][v]hstack=inputs=2:shortest=1,drawtext=text='Shared environment step %{n}':x=8:y=h-25:fontsize=14:fontcolor=white:box=1:boxcolor=black@0.6[out]")
    if not target.exists():
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',example['baseline_video'],'-i',example['variant_video'],
                        '-filter_complex',filters,'-map','[out]','-an','-c:v','libx264','-threads','2',
                        '-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],check=True)
    probe = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries',
                            'stream=codec_name,pix_fmt,nb_read_frames,r_frame_rate','-of','json',str(target)],
                           capture_output=True,text=True,check=True)
    stream = json.loads(probe.stdout)['streams'][0]
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p' and stream['r_frame_rate']=='20/1'
    counts=[]
    for source in [example['baseline_video'],example['variant_video']]:
        result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames',
                               '-of','json',source],capture_output=True,text=True,check=True)
        counts.append(int(json.loads(result.stdout)['streams'][0]['nb_frames']))
    assert int(stream['nb_read_frames']) == min(counts), 'Synchronization frame-count mismatch'
    subprocess.run(['ffmpeg','-v','error','-xerror','-threads','2','-i',str(target),'-f','null','-'],
                   capture_output=True,check=True)
    example['composite_video']=str(target)
    example['composite_validation']={'full_decode':True,'frames':int(stream['nb_read_frames']),
                                     'interval':'shared steps only; full tails remain in original videos',**stream}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    record(Path(__file__))
    search=read_json(DATA/'search/global_search.json')
    assert len(search['candidates'])==239 and search['actual_main_evaluations']==139 and search['actual_challenge_evaluations']==100
    for name,n in [('refined_search.json',18),('validation.json',12)]:
        assert len(read_json(DATA/'search'/name)['candidates'])==n
    frozen=read_json(DATA/'search/frozen_candidates.json')
    assert len({tuple(mask) for mask in frozen['phase_masks'].values()})==1
    read_json(DATA/'search/frozen_mask_classification.json')
    power=read_json(DATA/'phase_analysis/discrete_test_power_audit.json')
    assert all(not f['any_holm_rejection_mathematically_possible'] for f in power['families'])
    # Only authoritative completed suite validators enter the closed-loop denominator.
    episodes, coverage = [], []
    for path in sorted((DATA / 'closed_loop').glob('*/*/run/validated.json')):
        valid = read_json(path)
        if not valid.get('complete'): continue
        rows = read_json(path.parent / 'episode_log.json')
        assert len(rows) == valid['episodes'] == valid['full_video_decodes']
        assert len({(x['task_id'], x['episode_idx']) for x in rows}) == len(rows)
        assert not any(x.get('failure_reason') for x in rows)
        cond, suite = path.parents[2].name, path.parents[1].name
        coverage.append({'condition': cond, 'suite': suite, 'episodes': len(rows),
                         'successes': sum(bool(x['success']) for x in rows), 'skip': valid['condition'].get('skip', []),
                         'steps': valid['condition'].get('steps'), 'mode': valid['condition'].get('mode')})
        for x in rows:
            episodes.append({'condition': cond, 'suite': suite, 'task_id': x['task_id'],
                             'init': x['episode_idx'], 'success': bool(x['success']),
                             'highest_phase': PHASE_MAP.get(x.get('furthest_semantic_phase'), x.get('furthest_semantic_phase', 'unknown')),
                             'telemetry': str(path.parent / x['telemetry_path']),
                             'video': str(path.parent / x['video_path']), 'actions': x['executed_action_count']})
    ep = pd.DataFrame(episodes); cov = pd.DataFrame(coverage)
    assert len(ep)==3160 and len(cov)==124
    write_csv(cov, 'completed_coverage.csv'); write_csv(ep, 'closed_loop_episodes.csv')
    missing = set(MAIN) - set(ep.condition)
    if missing: raise RuntimeError('Missing expected main conditions: '+str(missing))
    base = ep[ep.condition == 'Full18_D10']
    paired_rows, onset_rows, cache, examples = [], [], {}, []
    def get_trace(path):
        if path not in cache: cache[path] = trajectories(path)
        return cache[path]
    for cond in MAIN[1:]:
        pair = ep[ep.condition == cond].merge(base, on=['suite', 'task_id', 'init'], suffixes=('_v', '_b'), validate='one_to_one')
        assert len(pair) == 200
        pair['difference'] = pair.success_v.astype(int) - pair.success_b.astype(int)
        diff, lo, hi = bootstrap_pairs(pair)
        ev = pair[pair.task_id >= 8]
        paired_rows.append({'condition': cond, 'episodes': len(pair), 'baseline_successes': int(pair.success_b.sum()),
                            'variant_successes': int(pair.success_v.sum()), 'difference_pp': diff, 'ci_low_pp': lo,
                            'ci_high_pp': hi, 'new_failures': int((pair.success_b & ~pair.success_v).sum()),
                            'recoveries': int((~pair.success_b & pair.success_v).sum()),
                            'eval_baseline_successes': int(ev.success_b.sum()), 'eval_variant_successes': int(ev.success_v.sum()),
                            'eval_episodes': len(ev)})
        for row in pair.itertuples():
            b, v = get_trace(row.telemetry_b), get_trace(row.telemetry_v)
            assert np.max(np.abs(b['xyz'][0]-v['xyz'][0])) < 1e-7, 'Initial state mismatch'
            deviation = first_deviation(b, v)
            onset_rows.append({'condition': cond, 'suite': row.suite, 'task_id': row.task_id, 'init': row.init,
                               'baseline_success': row.success_b, 'variant_success': row.success_v, **deviation})
        # Deterministic worst new-failure example; diagnostic extremes, not average evidence.
        if cond in MAIN[1:]:
            subset = pd.DataFrame(onset_rows)
            subset = subset[(subset.condition == cond) & subset.baseline_success & ~subset.variant_success]
            if subset.empty: subset = pd.DataFrame(onset_rows).query('condition == @cond')
            chosen = subset.sort_values(['xyz_p95_m', 'suite', 'task_id', 'init']).iloc[-1]
            row = pair[(pair.suite == chosen.suite) & (pair.task_id == chosen.task_id) & (pair.init == chosen.init)].iloc[0]
            name = f'xyz_{cond}_{chosen.suite}_t{chosen.task_id:02d}_i{chosen.init:02d}'
            xyz_plot(get_trace(row.telemetry_b), get_trace(row.telemetry_v), name,
                     f'{cond} | {chosen.suite} task {chosen.task_id} init {chosen.init} | baseline={int(row.success_b)}, variant={int(row.success_v)}')
            composites = list((DATA / 'closed_loop_analysis/xyz' / cond / chosen.suite).glob(f'task{chosen.task_id:02d}_init{chosen.init:02d}*synchronized.mp4'))
            examples.append({'condition': cond, 'suite': chosen.suite, 'task_id': int(chosen.task_id), 'init': int(chosen.init),
                             'plot': str(OUT / (name+'.png')), 'baseline_video': row.video_b, 'variant_video': row.video_v,
                             'composite_video': str(composites[0]) if composites else None})
    paired = pd.DataFrame(paired_rows); onset = pd.DataFrame(onset_rows)
    write_csv(paired, 'paired_success_task_bootstrap.csv'); write_csv(onset, 'paired_xyz_and_onset.csv')
    print(f'Audited {len(ep)} closed-loop episodes; analyzed {len(onset)} true XYZ pairs', flush=True)

    rates = cov[cov.condition.isin(MAIN)].pivot(index='condition', columns='suite', values='successes').reindex(index=MAIN, columns=SUITES) / 50 * 100
    fig, ax = plt.subplots(figsize=(8, 7)); im = ax.imshow(rates, vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(4), SHORT); ax.set_yticks(range(len(MAIN)), MAIN)
    for i in range(len(MAIN)):
        for j in range(4): ax.text(j, i, f'{rates.iloc[i,j]:.0f}%\n{int(rates.iloc[i,j]/2)}/50', ha='center', va='center', fontsize=8,
                                     color='white' if rates.iloc[i,j] > 65 else '#20252b')
    ax.set_title('Completed standard suites: five official initial states/task'); fig.colorbar(im, ax=ax, label='Task success (%)')
    save(fig, 'success_heatmap')
    fig, ax = plt.subplots(figsize=(9, 5.8)); y = np.arange(len(paired))
    ax.errorbar(paired.difference_pp, y, xerr=np.array([paired.difference_pp-paired.ci_low_pp, paired.ci_high_pp-paired.difference_pp]),
                fmt='o', color='#2274b5', capsize=3)
    ax.set_yticks(y, paired.condition); ax.invert_yaxis(); ax.axvline(0, color='gray', ls=':')
    ax.set(xlabel='Paired success difference vs D10 (percentage points)', title='95% task-cluster bootstrap; exploratory, unadjusted')
    save(fig, 'paired_success_intervals')
    stages = ep[ep.condition.isin(MAIN) & ~ep.success].groupby(['condition', 'highest_phase']).size().unstack(fill_value=0).reindex(MAIN, fill_value=0)
    stages = stages.reindex(columns=PHASES+['unknown'], fill_value=0)
    write_csv(stages.reset_index(), 'failure_highest_stage_counts.csv')
    fig, ax = plt.subplots(figsize=(10, 6.5)); left = np.zeros(len(stages))
    for phase, color in zip(stages.columns, ['#86b7d8','#f6b26b','#a4c89e','#b7a0cc','#aaaaaa']):
        ax.barh(stages.index, stages[phase], left=left, label=phase, color=color); left += stages[phase]
    for i, value in enumerate(left): ax.text(value+.8, i, str(int(value)), va='center', fontsize=8)
    ax.invert_yaxis(); ax.set(xlabel='Failed episodes (counts, not percentages)', title='Highest reached stage is NOT the first error location')
    ax.legend(loc='lower right', fontsize=8); save(fig, 'failure_highest_stage')
    new = onset[onset.baseline_success & ~onset.variant_success]
    onset_counts = new.groupby(['condition', 'onset_relative_to_contact']).size().unstack(fill_value=0).reindex(MAIN[1:], fill_value=0)
    write_csv(onset_counts.reset_index(), 'new_failure_onset_counts.csv')
    fig, ax = plt.subplots(figsize=(10, 5.5)); left = np.zeros(len(onset_counts))
    for column, color in [('pre_contact','#2274b5'),('at_or_after_contact','#f28e2b'),('no_onset','#999999'),('no_baseline_contact','#c9bfd5')]:
        vals = onset_counts[column] if column in onset_counts else np.zeros(len(onset_counts))
        ax.barh(onset_counts.index, vals, left=left, color=color, label=column); left += vals
    ax.invert_yaxis(); ax.set(xlabel='New failures where baseline succeeded', title='First XYZ divergence >20 mm for 3 consecutive states')
    ax.legend(fontsize=8); save(fig, 'new_failure_xyz_onset')

    prior = pd.DataFrame(read_json(DATA/'phase_analysis/layer_prior.json')['layers'])
    write_csv(prior, 'layer_prior.csv')
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(prior.layer, prior.robust_cosine_p10, 'o-', color='#2274b5'); axes[0].axhline(.98,color='gray',ls=':')
    axes[0].set(ylabel='Robust cosine p10', title='Block input/output cosine and measured single-layer action intervention')
    axes[1].bar(prior.layer, prior.single_layer_task_mean_rmse, color=['#f28e2b' if x else '#86b7d8' for x in prior.protected_low_cosine])
    axes[1].set(xlabel='VLM layer (1-based; L18 KV-only)', ylabel='H4 normalized action RMSE'); axes[1].set_xticks(range(1,19))
    save(fig, 'layer_cosine_action')
    phase = read_csv(DATA/'phase_analysis/task_phase_sensitivity.csv')
    phase = phase[phase.split == 'search']
    # Task -> suite -> cross-suite equal aggregation, no extra weight for frequent observations.
    pm = phase.groupby(['suite','semantic_phase','layer']).normalized_continuous_rmse.mean().groupby(['semantic_phase','layer']).mean().unstack().reindex(PHASES)
    write_csv(pm.reset_index(), 'phase_layer_sensitivity.csv')
    fig, ax = plt.subplots(figsize=(12, 3.2)); im=ax.imshow(np.log10(pm.clip(lower=1e-5)), aspect='auto', cmap='magma')
    ax.set_xticks(range(len(pm.columns)), pm.columns); ax.set_yticks(range(4),PHASES)
    ax.set(xlabel='Layer',title='Descriptive stage/layer H4 sensitivity (search tasks only)'); fig.colorbar(im,ax=ax,label='log10 normalized RMSE')
    save(fig,'phase_layer_sensitivity')

    numerical, associations, modules, manifests, activity, energy = [], [], [], [], [], []
    for suite in SUITES:
        manifests.append(read_json(DATA/'observations'/suite/'manifest.json'))
        summary=read_json(DATA/'probe'/suite/'summary.json')
        assert summary['checkpoint_unchanged'] and summary['parity_passed']
        # Saved task means retain the actual activation numerics, including modality differences.
        n=read_csv(DATA/'numerical_updates'/suite/'task_phase_updates.csv');n['suite']=suite;numerical.append(n)
        a=read_csv(DATA/'numerical_updates'/suite/'activity_association_stability.csv');a['suite']=suite;associations.append(a)
        m=read_csv(DATA/'submodule'/suite/'submodule_task_equal_action_summary.csv');modules.append(m)
        act=read_csv(DATA/'probe'/suite/'activation_statistics.csv',usecols=['split','task_id','episode_idx','semantic_phase',
                     'layer','modality','tensor','zero_fraction','near_zero_0.01','top10pct_token_energy_fraction'])
        act=act[(act.split=='search') & act.tensor.isin(['delta_attn','delta_ffn','delta_block'])]
        features=['zero_fraction','near_zero_0.01','top10pct_token_energy_fraction']
        act=act.groupby(['task_id','episode_idx','layer','modality','tensor'])[features].mean().groupby(['task_id','layer','modality','tensor']).mean().groupby(['layer','modality','tensor']).mean().reset_index()
        act['suite']=suite;activity.append(act)
        en=read_csv(DATA/'submodule'/suite/'submodule_task_equal_numerical_summary.csv');en['suite']=suite;energy.append(en)
    num=pd.concat(numerical,ignore_index=True)
    # Only search tasks for candidate conclusions; no hidden use of frozen evaluation samples.
    num=num[num.task_id<6]
    metrics=['relative_attn_update','relative_ffn_update','cancellation_ratio','attn_ffn_cosine_derived']
    ng=num.groupby(['suite','task_id','layer','modality'])[metrics].mean().groupby(['suite','layer','modality']).mean().groupby(['layer','modality']).mean().reset_index()
    write_csv(ng,'numerical_layer_modality.csv')
    act=pd.concat(activity).groupby(['layer','modality','tensor'])[features].mean().reset_index()
    write_csv(act,'activation_sparsity_energy.csv')
    energy=pd.concat(energy,ignore_index=True);write_csv(energy,'submodule_token_channel_energy.csv')
    fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
    for ax,metric,title in zip(axes.flat,metrics,['Attention residual / H_in RMS','FFN residual / H_attn RMS','||a+f|| / (||a||+||f||)','Derived cosine(attn update, FFN update)']):
        for modality,color in [('agent','#2274b5'),('wrist','#f28e2b'),('language','#5f4690')]:
            d=ng[ng.modality==modality];ax.plot(d.layer,d[metric],'-o',ms=3,color=color,label=modality)
        ax.set_title(title,fontsize=10);ax.set_xticks(range(1,19,2));ax.set_xlabel('Layer');ax.legend(fontsize=7)
    save(fig,'attn_ffn_numerical')
    assoc=pd.concat(associations,ignore_index=True);write_csv(assoc,'activity_association_stability.csv')
    mod=pd.concat(modules,ignore_index=True);write_csv(mod,'submodule_interventions.csv')
    fig,axes=plt.subplots(1,4,figsize=(15,4),sharey=True)
    for suite,ax,label in zip(SUITES,axes,SHORT):
        d=mod[mod.suite==suite].groupby(['layer','mode']).normalized_continuous_rmse.mean().unstack()
        for mode,color in [('attn','#2274b5'),('ffn','#f28e2b'),('both','#5f4690')]:
            if mode in d: ax.plot(d.index,d[mode],'o-',label=mode,color=color)
        ax.set(title=label,xlabel='Selected layer');ax.set_yscale('log');ax.set_xticks(d.index)
    axes[0].set_ylabel('Normalized action RMSE (phase-cell mean)');axes[-1].legend(fontsize=8)
    fig.suptitle('Selected submodule interventions: uneven phase/task coverage, descriptive only')
    save(fig,'submodule_interventions')
    timing=[]
    for suite in SUITES:
        t=read_json(DATA/'dedicated_timing'/suite/'summary.json');assert t['complete']
        for cond,v in t['metrics'].items():
            timing.append({'condition':cond,'suite':suite,'pipeline_ms':v['pipeline_wall_ms']['task_phase_equal_median_ms'],
                           'measurements':v['pipeline_wall_ms']['n']})
    timing=pd.DataFrame(timing);write_csv(timing,'dedicated_latency.csv')
    tm=timing.groupby('condition').pipeline_ms.mean(); baseline_ms=tm['Full18_D10']
    fig,ax=plt.subplots(figsize=(8,5))
    for cond in MAIN[:6]:
        rate=float(ep[ep.condition==cond].success.mean()*100)
        ax.scatter(tm[cond],rate,s=80,color='#2274b5' if cond=='Full18_D10' else '#f28e2b')
        ax.annotate(cond.replace('Full18_',''),(tm[cond],rate),xytext=(5,6),textcoords='offset points')
    ax.set(xlabel='Dedicated pipeline wall time (ms; suite-equal median mean)',ylabel='Closed-loop success (%)',
           title='Denoising reduction: inference timing, NOT task completion wall time')
    ax.grid(alpha=.2);save(fig,'latency_success')
    parallel=[read_json(p) for p in sorted((DATA/'parallel_verify_horizons').glob('*/summary.json'))]
    parallel=[d for d in parallel if d.get('complete')]
    par=pd.DataFrame([{k:d[k] for k in ['condition','horizon','episodes','parallel_successes','baseline_successes','draft_prefix_ratio',
            'parallel_mean_episode_roundtrip_total_ms','baseline_mean_episode_roundtrip_total_ms','parallel_vs_baseline_mean_episode_roundtrip_ratio']} for d in parallel])
    par['steps']=par.condition.str.extract(r'ParallelVerifyD(\d+)').astype(int)
    write_csv(par,'parallel_verify_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for steps,color in [(3,'#2274b5'),(5,'#f28e2b'),(7,'#5f4690')]:
        d=par[par.steps==steps].sort_values('horizon')
        axes[0].plot(d.horizon,d.parallel_successes/40*100,'o-',label=f'D{steps}',color=color)
        axes[1].plot(d.horizon,d.parallel_vs_baseline_mean_episode_roundtrip_ratio,'o-',label=f'D{steps}',color=color)
    axes[0].axhline(87.5,color='gray',ls=':',label='D10 35/40');axes[1].axhline(1,color='gray',ls=':')
    axes[0].set(ylabel='Task success (%)',title='Frozen exploratory tasks 8/9');axes[1].set(ylabel='Episode cumulative wait / D10',title='Below 1 is faster; two GPUs required')
    for ax in axes: ax.set_xlabel('Draft prefix horizon');ax.set_xticks([4,8,12]);ax.legend(fontsize=8)
    save(fig,'parallel_verify_tradeoff')
    gate=read_json(DATA/'kalman_gate/kalman_gate_summary.json')
    recovery=read_json(DATA/'recovery_horizon/recovery_horizon_summary.json')
    gs=pd.DataFrame([{'condition':k,**{f:v[f] for f in ['episodes','gate_successes','baseline_successes','total_policy_inference_ratio_vs_baseline']}} for k,v in gate['conditions'].items()])
    rs=pd.DataFrame([{'condition':k,**{f:v[f] for f in ['horizon','episodes','gate_successes','baseline_successes','total_policy_inference_ratio_vs_baseline']}} for k,v in recovery['conditions'].items()])
    write_csv(gs,'kalman_gate_summary.csv');write_csv(rs,'recovery_horizon_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,d,title in [(axes[0],gs,'Kalman D3 gate: tasks 8/9'),(axes[1],rs,'One early D3 prefix then Full18: tasks 6/7')]:
        ax.bar(d.condition,d.total_policy_inference_ratio_vs_baseline,color='#86b7d8');ax.axhline(1,color='gray',ls=':')
        for i,row in enumerate(d.itertuples()):ax.text(i,row.total_policy_inference_ratio_vs_baseline+.01,f'{row.gate_successes}/{row.episodes}',ha='center',fontsize=8)
        ax.tick_params(axis='x',rotation=30,labelsize=8);ax.set(title=title,ylabel='Summed server policy inference / D10')
    save(fig,'gate_recovery_cost')
    branches=pd.concat([read_csv(DATA/'branch_analysis'/s/'paired_prefix_xyz.csv') for s in SUITES],ignore_index=True)
    bc=branches[branches.horizon==8].groupby(['variant','suite','task_id','semantic_phase']).mean_xyz_error_m.mean().reset_index()
    branch_summary=bc.groupby(['variant','suite']).mean_xyz_error_m.mean().groupby('variant').mean()*1000
    write_csv(bc,'prefix_h8_task_phase_xyz.csv')

    supplemental=[];supp_pairs=[]
    for layer in [16,17]:
        root=ROOT/f'results/pi0_attention_ffn_diagnosis/closed_loop_multi_init/AttnL{layer}_T12_Init0to4_seed12345'
        for suite in ['libero_90','libero_10']:
            pairs=read_csv(root/suite/'paired_episode_results.csv')
            pairs=pairs.rename(columns={f'attn{layer}_success':'candidate_success',f'attn{layer}_steps':'candidate_steps',
                                      f'attn{layer}_video_validated':'candidate_video_validated'})
            for column in ['baseline_success', 'candidate_success']:
                assert pairs[column].isin([0, 1, False, True]).all()
                pairs[column] = pairs[column].astype(bool)
            assert len(pairs)==(40 if suite=='libero_90' else 20)
            assert pairs.baseline_video_validated.all() and pairs.candidate_video_validated.all()
            for row in pairs.itertuples():
                pairroot=root/suite/'pairs'/f'task{row.task_id:03d}_init{row.init_index:02d}'
                states=[]
                for dirname in ['baseline',f'attn{layer}']:
                    summary=read_json(pairroot/dirname/'summary.json')
                    assert summary['runtime']=='triton_eager' and summary['denoise_steps']==10
                    assert summary['official_init_index']==row.init_index and summary['init_source']=='official'
                    assert summary['init_state_sha256']==row.initial_state_sha256
                    assert summary['state_count']==summary['executed_actions']+1==summary['video_frame_count']
                    assert summary['skip_ffn_layers']==[]
                    assert summary['skip_attn_layers']==([] if dirname=='baseline' else [layer])
                    with np.load(pairroot/dirname/'trajectory.npz',allow_pickle=False) as trajectory:
                        xyz=trajectory['eef_xyz']
                        assert xyz.shape==(summary['state_count'],3) and np.isfinite(xyz).all()
                        states.append(xyz[0])
                assert np.max(np.abs(states[0]-states[1]))<1e-7
            pair=pairs.copy();pair['condition']=f'AttnL{layer}';supp_pairs.append(pair)
            outcomes={'both_success':int((pairs.baseline_success & pairs.candidate_success).sum()),
                      'baseline_only':int((pairs.baseline_success & ~pairs.candidate_success).sum()),
                      'candidate_only':int((~pairs.baseline_success & pairs.candidate_success).sum()),
                      'both_failure':int((~pairs.baseline_success & ~pairs.candidate_success).sum())}
            supplemental.append({'condition':f'AttnL{layer}','suite':suite,'pairs':len(pairs),
                                  'baseline_successes':int(pairs.baseline_success.sum()),'candidate_successes':int(pairs.candidate_success.sum()),**outcomes})
        pairs=pd.concat(supp_pairs[-2:]);pairs['difference']=pairs.candidate_success.astype(int)-pairs.baseline_success.astype(int)
        diff,lo,hi=bootstrap_pairs(pairs, task_weighted=True)
        supplemental.append({'condition':f'AttnL{layer}','suite':'pooled_12_tasks','pairs':60,
            'baseline_successes':int(pairs.baseline_success.sum()),'candidate_successes':int(pairs.candidate_success.sum()),
            'both_success':int((pairs.baseline_success & pairs.candidate_success).sum()),
            'baseline_only':int((pairs.baseline_success & ~pairs.candidate_success).sum()),
            'candidate_only':int((~pairs.baseline_success & pairs.candidate_success).sum()),
            'both_failure':int((~pairs.baseline_success & ~pairs.candidate_success).sum()),
            'pooled_difference_pp':diff,'pooled_ci_low_pp':lo,'pooled_ci_high_pp':hi})
        failed=pairs[pairs.baseline_success & ~pairs.candidate_success]
        row=failed.sort_values(['eef_deviation_p95_m','suite','task_id','init_index']).iloc[-1]
        pairroot=root/row.suite/'pairs'/f'task{row.task_id:03d}_init{row.init_index:02d}'
        bdir,vdir=pairroot/'baseline',pairroot/f'attn{layer}'
        bs,vs=read_json(bdir/'summary.json'),read_json(vdir/'summary.json')
        assert bs['init_state_sha256']==vs['init_state_sha256']
        b,v=trajectories(bdir/'telemetry.jsonl'),trajectories(vdir/'telemetry.jsonl')
        name=f'xyz_AttnL{layer}_{row.suite}_t{row.task_id:03d}_i{row.init_index:02d}'
        xyz_plot(b,v,name,f'Attention L{layer} only | {row.suite} task {row.task_id} init {row.init_index} | baseline success / variant failure')
        examples.append({'condition':f'AttnL{layer}','suite':row.suite,'task_id':int(row.task_id),'init':int(row.init_index),
                         'plot':str(OUT/(name+'.png')),'baseline_video':str(bdir/'rollout.mp4'),'variant_video':str(vdir/'rollout.mp4'),'composite_video':None})
    supp=pd.DataFrame(supplemental);write_csv(supp,'attention_multi_init_results.csv');write_csv(pd.concat(supp_pairs),'attention_multi_init_pairs.csv')
    fig,axes=plt.subplots(1,2,figsize=(10,4.3))
    for ax,suite in zip(axes,['libero_90','libero_10']):
        d=supp[supp.suite==suite];x=np.arange(2)
        ax.bar(x-.18,d.baseline_successes/d.pairs*100,width=.36,color='#2274b5',label='Paired baseline')
        ax.bar(x+.18,d.candidate_successes/d.pairs*100,width=.36,color='#f28e2b',label='Attention skip')
        for i,r in enumerate(d.itertuples()):
            ax.text(i-.18,r.baseline_successes/r.pairs*100+1,f'{r.baseline_successes}/{r.pairs}',ha='center',fontsize=8)
            ax.text(i+.18,r.candidate_successes/r.pairs*100+1,f'{r.candidate_successes}/{r.pairs}',ha='center',fontsize=8)
        ax.set_xticks(x,d.condition);ax.set_ylim(0,110);ax.set(title=suite,ylabel='Success (%)');ax.legend(fontsize=8)
    fig.suptitle('Diagnostic eager runtime: do NOT pool with standard Triton suite baseline');save(fig,'attention_multi_init_success')
    for example in examples:
        synchronized_video(example)

    # Report assembled from measured tables; all original inputs remain untouched.
    sections=[]
    sections.append('''# 最终实验汇总与分析报告：层裁剪、低步数草稿与恢复/验证机制

更新：2026-10-10。范围：idea1 主线 Experiment 10 的已执行实验，以及 Attention L16/L17 五初始状态补充实验。本文是**已执行实验的最终归档**，不是声称研究方案中的全部扩展验证都已完成。层编号从 1 开始；不修改 idea1.md。

## 1. 核心结论

1. **当前最值得推进的是完整 VLM + Action Expert 降噪步数减少，而不是大幅跳 VLM 层。** D2/D4 为 190/200（95%），D3 为 186/200（93%），D5 为 188/200（94%）；D10 基线为 186/200（93%）。点估计更高不等于显著提高，也没有预注册非劣界限。
2. **高余弦只是提出候选的证据，不是可删证明。** 严格满足启发式的五层组合 [3,5,7,9,12] 只有 70/200（35%）；两层组合 [3,10] 为 184/200（92%），但 Goal 只有 41/50（82%）。因此不能宣称找到了统一最优 skip。
3. **“不同阶段层敏感度不同”目前有描述性线索，但没有足够的统计证据支持主张，更没有验证阶段路由优越性。** 四阶段选出的 mask 完全相同，当前阶段置换检验在样本量与多重比较规则下根本不具备拒绝零假设的分辨率。
4. **并行验证尚未实现可靠的任务级加速。** D3/H8 请求时间优于串行验证，但每任务累计等待为基线的 1.38 倍；D7/H12 接近 1 倍，却只有 32/40 成功，低于同子集基线 35/40。高动作接受率不等于任务成功。
5. **Attention L16 仅可作为继续验证候选；L17 不宜通用删除。** 12 任务 × 5 状态：L16 为 40/60 vs 配对基线 39/60，L17 为 31/60 vs 39/60。这是诊断 eager 模式，不是标准套件成绩。

## 2. 实验过程、协议和数据来源

先从完整模型真实 baseline 轨迹提取阶段观测，再做单层/子模块干预和数值探针；按 search→validation→冻结候选闭环筛选；随后开展恢复前缀长度、Kalman 门控和双 GPU 并行验证。没有使用随机初始化权重或随机输入代替真实机器人观测。

标准闭环：Object / Spatial / Goal / LIBERO-10，官方 init0–4，环境 seed=7、warmup=10、replan=12；训练后同一 checkpoint。search 为 task0–5，validation 为 task6–7，冻结探索评估为 task8–9。任务曾参与历史实验，因此 task8–9 不是严格首次见到的外部测试集。200 条总体成绩含 search/validation 任务，评估子集另外列出。

离线噪声种子为 12345/23456/34567；同一观测与重复噪声不视为独立任务。模型执行 BF16，数值统计转 FP32；权重 hash 不变、baseline 与记录路径 parity 均通过。参数不会随场景改变，“权重非零比例/数值变化”不能衡量场景敏感度；本轮测量的是激活、残差更新和真实跳过干预后的动作变化。

四卡用于相互独立 suite/probe/候选并行；并行验证单 worker 占两卡。无空卡时等待 300 秒；已由本实验持有的卡按任务完成事件派发，不高频扫描。专门计时按套件在同一张 4090 上轮转条件，避免拿不同 GPU 的拥塞时间混作模型差异。
''')
    total_observations=sum(x['sample_count'] for x in manifests)
    sections.append(table(['证据块','实际覆盖','解释'],[
        ['真实观测/单层离线探针',f'{total_observations} 观测；{total_observations*3} 观测-噪声块','399/392/342/486；来自 200 条 baseline，不是 1619 个独立任务'],
        ['标准闭环',f'{len(ep)} 条；{len(cov)} 个条件×套件单元','12 条件×200 + 19 条件×40；每条有成功标志/telemetry/视频'],
        ['专门计时',f'{int(timing.measurements.sum())} 次','8 条件×4 套件×60；少量 validation 阶段观测，非吞吐/任务总耗时'],
        ['Attention 补充','120 对、240 条 rollout','L16/L17 各 60 对；基线重复运行不当作额外独立任务'],
        ['视频校验',f'标准 {len(ep)} 条 + 补充 240 条','原始 validator 全帧解码已通过；不是全部人工观看']]))
    sections.append('''
### 2.1 过程中的修正和审计边界

L16 首次 LIBERO-90 运行发生 prompt-cache manifest 不匹配，失败 attempt 单独保留，修复各套件 manifest 后只纳入最终完整配对数据。部分 40 条任务子集曾被旧 validator 当成 50 条而标记失败，后来依据明确 task_ids 重验；运行错误不计作正常任务失败。GPU 归属状态文件发现逻辑扩展为匹配 `*status*.json`，避免漏掉带后缀状态文件。

本报告重新检查 completed validator、episode 唯一 task/init 键、分母、异常标志，并逐对读取 2200 条主条件 XYZ 对比数据验证配对初始 XYZ 一致。阶段/数值统计重新聚合；视频完整解码结论引用原始 validator，不声称这次重新解码了全部视频。历史文字若与原始 validator 冲突，以本报告重新统计的原始数据为准。

## 3. 相似度：怎么测、可说明什么

对真实观测中每层 block 输入 H_in 与输出 H_out，对 agent/wrist 视觉 token、language token 分别计算每个有效 token 的余弦后平均。表中 robust p10：仅 search 数据，先按 suite/task/phase/modality 对观测取中位数，再取这些单元的 10% 分位数。它是**同一层输入→输出的余弦**，不是任意两个层之间的完整 18×18 两两余弦矩阵；原始探针未保存所有 hidden tensor，不能凭现有统计伪造全层两两矩阵。

单层敏感度：固定同一观测和动作噪声，跳过指定 VLM block 的 hidden update，保留每层 KV shell；比较前 4 个动作、前 6 维连续动作在模型归一化坐标中的 RMSE，不包含夹爪维度（夹爪翻转单独记录）。先平均噪声/init，按任务/阶段汇总，再任务等权；先验表为 suite/task 单元均值，各 suite 有同等任务数。它不是米制轨迹误差，米制误差见第 7 节。
''')
    sections.append(image('layer_cosine_action','逐层余弦与动作干预'))
    sections.append(table(['层','robust cosine p10','H4 动作 RMSE','候选解释'],[
        [int(r.layer),f'{r.robust_cosine_p10:.4f}' if pd.notna(r.robust_cosine_p10) else '不可定义',
         f'{r.single_layer_task_mean_rmse:.4f}', '结构/低余弦保护' if r.protected_structural or r.protected_low_cosine else '仅可优先做组合干预'] for r in prior.itertuples()]))
    sections.append('''
L3–L5 是早期候选，但 L7 虽然余弦很高，动作干预更大；L17 余弦很低、动作扰动却较小，说明 hidden 变化幅度与动作影响不是同一件事。L18 是 KV-only 特例：hidden 无更新不能解释为整层可以删除。搜索保留 L1/L2/L17/L18；低余弦 L13–L16 保护；高余弦 L3–L12 仅用于候选提出。

### 3.1 场景/阶段敏感度的证据强度
''')
    sections.append(image('phase_layer_sensitivity','层与阶段描述性敏感度'))
    sections.append('''
阶段比较固定同 task、同 init；对单层影响扣除该 task/phase 的整体层平均变化，以尽量区分“所有层都更难”和“某层相对特别敏感”。408 个阶段比较分为 4 套件、每套件 102 比较。每次只有 5–6 任务，双侧符号置换最小可能 p=0.03125（5 任务为 0.0625），远大于 Holm 首步阈值 0.05/102≈0.000490；全部 Holm 拒绝数为 0，**这是检验分辨率不足，不是证明阶段敏感度相同**。

阶段与轨迹进展、观测分布共同变化，尚未做控制变量的场景扰动；描述性热图不能推出因果阶段路由。后续更合理的是先冻结少量层×阶段主要对比和效应阈值，再对独立任务复现，而非把重复噪声当额外样本来获得显著性。

## 4. Attention / FFN 数值与干预

记录 a=H_attn−H_in、f=H_out−H_attn、b=a+f 的 RMS、分位数、近零比例与 token/channel 能量集中度。相对更新分别为 RMS(a)/RMS(H_in)、RMS(f)/RMS(H_attn)；抵消比为 RMS(b)/(RMS(a)+RMS(f))。由同形状残差 RMS 恒等式推导 cosine(a,f)=(b²−a²−f²)/(2af)，不是额外保存了所有原始残差向量；零分母保留缺失。下图为 search 任务、阶段/任务/套件等权描述。
''')
    sections.append(image('attn_ffn_numerical','Attention FFN 更新和抵消'))
    sections.append(table(['层 / agent token','Attn 相对 RMS','FFN 相对 RMS','抵消比','derived cosine(a,f)'],[
        [int(r.layer),f'{r.relative_attn_update:.3f}',f'{r.relative_ffn_update:.3f}',f'{r.cancellation_ratio:.3f}',f'{r.attn_ffn_cosine_derived:.3f}']
        for r in ng[(ng.modality=='agent') & ng.layer.isin([1,3,5,7,12,16,17])].itertuples()]))
    sections.append(table(['层 / agent block 更新','精确零比例','|值|≤0.01×RMS','最高10% token能量'],[
        [int(row['layer']),f'{row["zero_fraction"]*100:.2f}%',f'{row["near_zero_0.01"]*100:.2f}%',
         f'{row["top10pct_token_energy_fraction"]*100:.1f}%']
        for _,row in act[(act.modality=='agent') & (act.tensor=='delta_block') & act.layer.isin([1,3,5,12,16,17])].iterrows()]))
    sections.append('''具体数值表明：agent 的 L3/L5 Attention 相对更新约 1.9%/3.0%，FFN 约 3.8%/4.6%，适合作为轻量干预候选；L12 的 Attention/FFN 更新方向负相关（derived cosine≈−0.49），存在抵消，不能据较小整 block 更新推断两个子模块都无用。L17 的视觉 FFN 相对更新超过 2，最高 10% agent token 集中约 89% block 更新能量，反映集中强更新而非零输出。注意这些是聚合描述，不是对所有任务逐样本的保证。\n''')
    sections.append('精确零比例包含 BF16 激活差分的量化效应，不是权重参数稀疏率，也不能直接作为剪枝安全判据。\n')
    sections.append(image('submodule_interventions','子模块单独干预'))
    valid_assoc=assoc[assoc.ci025.notna() & assoc.ci975.notna()]
    crosses=int(((valid_assoc.ci025<=0)&(valid_assoc.ci975>=0)).sum())
    sections.append(f'''数值活跃度可以揭示更新集中或抵消，却不能直接推出“剩余 token/channel 或模块是多余的”。活跃度与干预影响的相关性共有 {len(valid_assoc)} 个可用探索单元，其中 {crosses} 个 task-bootstrap 区间覆盖 0；小样本、重复对比且未经多重校正，不可筛选某个显著单元来证明机制。

子模块图来自各 suite 选定层、search init0 的代表阶段观测与 3 个噪声，不是每层×所有任务的完全均衡设计；缺少某阶段的 task 不能补 0。Attention、FFN、整 block 跳过的作用不同，不能把单模块结论移植为整层安全性。L1 是诊断强干预对照，不是删除建议。尚无完整 QKV entropy、FFN gate 开闭率或权重更新证据。
''')
    sections.append('## 5. 标准套件闭环：成功率、泛化与代价\n')
    sections.append(table(['条件 / VLM skip','Object','Spatial','Goal','LIBERO-10','总体'],[
        [cond+' / '+str(cov[cov.condition==cond].iloc[0]['skip'])]+[f'{int(rates.loc[cond,s]/2)}/50 ({rates.loc[cond,s]:.0f}%)' for s in SUITES]+
        [f'{int(ep[ep.condition==cond].success.sum())}/200 ({ep[ep.condition==cond].success.mean()*100:.1f}%)'] for cond in MAIN]))
    sections.append(image('success_heatmap','套件成功率'))
    sections.append('全部主条件均为 200 条，不是某些旧记录中的 400 条。Recover 条件表示执行一次短前缀后重新观测并恢复 Full18，不是整条轨迹都用裁剪模型。\n')
    sections.append(table(['条件','配对差 pp','任务聚类 95% pp','新失败 / 恢复','task8–9 子集 vs baseline'],[
        [r.condition,f'{r.difference_pp:+.1f}',f'[{r.ci_low_pp:+.1f}, {r.ci_high_pp:+.1f}]',f'{r.new_failures} / {r.recoveries}',f'{r.eval_variant_successes}/{r.eval_episodes} vs {r.eval_baseline_successes}/{r.eval_episodes}'] for r in paired.itertuples()]))
    sections.append(image('paired_success_intervals','配对任务聚类区间'))
    sections.append('''统计以 task 为聚类单元、suite 内重采样 10,000 次，四 suite 等权；不是把同任务五状态当 IID。所有区间均探索性、未做跨条件多重校正，也未预注册非劣界限。仅 2 个 task/suite 的 task8–9 子集不支持强泛化主张。

### 5.1 Layer-skip 搜索实际做了什么

采用启发式合法独立集、beam/Pareto、少量随机补充与同预算 local swap；候选排名使用真实组合 forward，不把单层分数相加当组合得分。合法删除数 2/3/4/5 分别只有 36/56/35/6 种，6 层删除不存在合法解。全局实际评估 239 个候选：139 主规则候选、100 单规则挑战；原定预算 90/10，但因合法空间与去重/搜索停止，**实现比例为 58.2/41.8，而不是 90/10**。代码挑战预算取名义预算的 10%，没有按实际主候选数重新缩减。

随后 refined 阶段 18 个候选、validation 阶段 12 个候选。FrozenStatic01=[2,4,6,8,10,12] 与 FrozenStatic02=[2,12] 是刻意放宽 L2 的挑战，不是严格遵守“起始两层保护”的最优解；严格合法对照另有 [3,5,7,9,12] 和 [3,10]。V13 原始随机 mask [3,4,7,12,15] 违反相邻删除及低余弦保护，不应用来支持规则有效。

四阶段最终 mask 都是 [2,4,6,8,10,12]，并未产生差异化阶段路由。未穷举整个参数空间、未证明全局最优；更重要的是闭环显示离线组合排名不能替代成功率。

### 5.2 超过 90% 的条件在哪些 suite

D2/D3/D4/D5 在 Object/Spatial/Goal 都超过 90%；LIBERO-10 分别 86/84/88/88%，没有超过 90%。[3,10] 在 Object/Spatial 为 98/100%，Goal 为 82%，LIBERO-10 为 88%；[2,12] 在 Spatial/Goal 为 98/94%，Object 恰好 90%、LIBERO-10 为 80%。这说明“任务限定轻裁剪”可能成立，但不是全套件统一无损。
''')
    sections.append(image('latency_success','单次推理与成功率'))
    sections.append(table(['条件','专门 pipeline ms','相对 D10 降幅','同轮闭环成功率'],[
        [cond,f'{tm[cond]:.2f}',f'{(1-tm[cond]/baseline_ms)*100:.1f}%',f'{ep[ep.condition==cond].success.mean()*100:.1f}%'] for cond in MAIN[:6]]))
    sections.append('''pipeline 为预处理+模型 wall time，排除环境步进/网络/控制，不是整条任务完成时间。D1 虽最短，成功率降至 76.5%，尤其 LIBERO-10 17/50（34%）；D2/D3 更值得评估。冻结 VLM mask 没有对应的专门计时，不用历史请求日志冷启动混造其时延。

## 6. 短前缀恢复、门控与双 GPU 验证
''')
    sections.append('同状态短分支 H8 的 measured EEF 平均偏差（suite 内 task/phase 等权，再 suite 等权）：'+ '；'.join(f'{k} {v:.2f} mm' for k,v in branch_summary.items())+'。它是少量 validation init0 锚点的局部结果，不是任务成功率。\n')
    sections.append(image('gate_recovery_cost','门控与恢复成本'))
    sections.append(table(['恢复 H（tasks6–7）','成功 / baseline','累计 server inference 比'],[
        [r.horizon,f'{r.gate_successes}/40 vs {r.baseline_successes}/40',f'{r.total_policy_inference_ratio_vs_baseline:.3f}×'] for r in rs.itertuples()]))
    sections.append('''一次早期 D3 前缀后恢复 Full18 在 H8 为 40/40 vs 39/40，200 条补充为 191/200 vs 186/200，但只是初步显示可恢复；H8/H12 的累计 server inference 节省仅约 1–2%，日志受采样和系统扰动影响，没有证明实用净加速。每条仅一次草稿不能摊薄总推理成本，也没有反复进入精细操作阶段的干预。
''')
    sections.append(table(['Kalman 条件（tasks8–9）','成功 / baseline','累计 server inference 比'],[
        [r.condition,f'{r.gate_successes}/40 vs {r.baseline_successes}/40',f'{r.total_policy_inference_ratio_vs_baseline:.3f}×'] for r in gs.itertuples()]))
    sections.append('''Kalman 门控所有阈值均未节省累计推理；Q/R 与方差 floor 敏感性未完成。预测动作一致性不验证目标状态或动作语义正确性，不适合作为安全保证。
''')
    sections.append(image('parallel_verify_tradeoff','并行验证任务成功与累计等待'))
    sections.append(table(['并行配置（tasks8–9）','成功 / baseline','草稿接受比例','累计请求等待比'],[
        [r.condition,f'{r.parallel_successes}/40 vs {r.baseline_successes}/40',f'{r.draft_prefix_ratio*100:.1f}%',f'{r.parallel_vs_baseline_mean_episode_roundtrip_ratio:.3f}×'] for r in par.sort_values(['steps','horizon']).itertuples()]))
    sections.append('''并行 D3/H8 请求均值 44.51 ms，相对串行验证约 71.2 ms 改善，但仍需要 1052 次 replans，每任务累计等待 1166 ms vs 基线 845 ms。D7/H12 约 843 ms vs 845 ms（0.998×），只差约 2 ms，不能宣称可靠加速，而且损失 3/40 成功。client roundtrip 与上一表 server inference 不同口径；未计入双 GPU 的总吞吐/占用成本，任务累计等待也不等于环境整体 wall time。缩短前缀增加 replans，可抵消单次请求改进。

## 7. 失败阶段与真实 XYZ 轨迹

最高到达阶段回答“失败前进展到哪里”，不回答“错误从哪里开始”。下图保留原阶段分析，并另算 baseline 成功、变体失败的配对轨迹：按相同环境步比较，XYZ 偏差连续 3 状态超过 2 cm 的首次时刻，再相对 baseline 实际接触/抓取事件分类。该阈值是描述性定义，尚未做阈值敏感性分析；未出现阈值不等于无动作/姿态/夹爪错误。
''')
    sections.append(image('failure_highest_stage','失败最高阶段'))
    sections.append(image('new_failure_xyz_onset','新增失败的首次持续XYZ偏差'))
    sections.append(table(['条件','新增失败','接触前','接触后/当时','baseline无接触事件','未持续>2cm'],[
        [cond,int(len(new[new.condition==cond]))]+[int(onset_counts.loc[cond].get(c,0)) for c in ['pre_contact','at_or_after_contact','no_baseline_contact','no_onset']] for cond in MAIN[1:]]))
    sections.append('''新增失败中，[2,4,6,8,10,12] 有 152/157 在 baseline 接触前首次持续偏离，另 5 条 baseline 无接触事件；合法五层组合有 111/116 接触前偏离，另 5 条无接触事件。反之 D3/D4/D5 的接触当时或之后首次偏离为 7/11、5/7、5/8。这支持“大幅跳层并不只是精细操作出问题”，也提示低步数应重点检查接触后控制，但不能把时序相关写成机制因果。baseline 无接触事件可能是非抓取任务或接触标注边界，独立列出，不能硬归为接触前。\n''')
    sections.append('''不能把“接近物体后才失败”直接等同为“精细操作层被删”：同一步比较包含动作节奏差异，不是动态时间对齐；轨迹偏差只覆盖双方都存在的状态，早结束尾部不补 0。XYZ 看不出完整姿态或夹爪语义，仅支持诊断线索。

以下展示真实 EEF world XYZ（米）而非相机截图；baseline 蓝色实线/圆点，变体橙色虚线/三角点，alpha=0.65，稀疏标记；同时有 XY 投影、共享步偏差与夹爪宽度。示例按新增失败中的最高 p95 偏差确定性选取，属于诊断极端样本，不冒充总体平均。视频与图共享 task/init，原视频 H264/yuv420p 全帧校验；缺少既有同步视频的示例新增左右同步视频，并以 ffprobe 帧数/20fps/H264/yuv420p 和 ffmpeg 全帧解码核查。新左右视频只展示共同步区间，不冻结结尾/不插值，完整尾部保留在原视频链接；左蓝 baseline、右橙 variant，可用步编号对照 XYZ 偏差曲线。
''')
    for ex in examples:
        sections.append(f"### {ex['condition']} / {ex['suite']} task{ex['task_id']} init{ex['init']}\n\n"+
                        f"![真实末端轨迹]({os.path.relpath(ex['plot'],ROOT)})\n\n"+
                        link(ex['baseline_video'],'baseline 视频')+' · '+link(ex['variant_video'],'变体视频')+
                        (' · '+link(ex['composite_video'],'同步对比视频') if ex['composite_video'] else '')+'\n')
    sections.append('## 8. Attention L16/L17 补充五状态：不要与标准 baseline 混算\n')
    sections.append(table(['条件','套件','baseline','Attention skip','baseline-only / candidate-only'],[
        [r.condition,r.suite,f'{r.baseline_successes}/{r.pairs}',f'{r.candidate_successes}/{r.pairs}',f'{r.baseline_only} / {r.candidate_only}'] for r in supp.itertuples()]))
    sections.append(image('attention_multi_init_success','Attention配对成功率'))
    pooled=supp[supp.suite=='pooled_12_tasks']
    sections.append(table(['条件','12任务配对差 pp','任务聚类95% pp'],[
        [r.condition,f'{r.pooled_difference_pp:+.1f}',f'[{r.pooled_ci_low_pp:+.1f}, {r.pooled_ci_high_pp:+.1f}]'] for r in pooled.itertuples()]))
    sections.append('此区间按 suite 内任务重采样，按 8:4 的任务数加权合并，匹配 60 对的总体口径；探索性未校正，不能仅凭区间或点估计宣布 L16 安全。\n')
    sections.append('''补充覆盖 LIBERO-90 的 task0/1/19/22/33/38/43/70，以及 LIBERO-10 的 task0/1/3/8，每任务 init0–4。baseline/variant 共享 official initial state hash，env7、warmup10、replan12、noise12345，完整 Action Expert D10；仅跳 Attention，不跳 FFN。运行模式为 triton_eager 诊断路径。

L16 在 LIBERO-90 从 22/40 到 21/40，LIBERO-10 从 17/20 到 19/20，合并净 +1，但新增失败 5、恢复 6，不能称普遍无损或显著改进。L17 在 LIBERO-90 从 22/40 到 14/40，新增失败 10、恢复 2；LIBERO-10 净不变却也有 1 次新增失败与 1 次恢复，说明总正确数会掩盖配对改变。整 block 的离线 L17 低动作扰动不等于 Attention-only 运行可靠。

基线 39/60（65%）偏低这一诊断路径现象未完成根因闭环。与标准套件 186/200 不是同任务集合/同 runtime，不能用两者差值归因于裁剪；历史 21/100 Control 的根因也不能由本轮高标准成绩反向宣布已解决。应分离协议、任务难度、预处理/runtime parity与具体成功判据逐项验证，不把它写成论文 baseline。

## 9. 与上一轮全量标准评测的交叉核对

上一轮 Experiment 9 与本轮不合并分母。其全量报告见 [Experiment 9 最终报告](experiment9_final_analysis_report.md)：标准 Full18 1888/2000（94.4%）；D3/D5 1898/2000（94.9%）/1897/2000（94.85%），支持低降噪步数优先方向。V12 在 Object 为 0/500，不支持“删 6 层但 Object 仍可 90%+”的猜测；V13-Custom 总体 875/2000（43.75%）。

上一轮还做了 Action Expert **单层** L2/L8 裁剪：356/400（89%）/354/400（88.5%），Goal 81%/84%、LIBERO-10 均 80%，不支持直接部署。这与本轮 D2–D5 **减少降噪迭代次数、不删除专家层** 是不同实验。历史数字仅引用既有报告，不冒充本轮重新遍历 28,800 条原始数据。

## 10. 下一步建议与完成边界

优先顺序：先把完整 VLM 的 D2/D3/D4 与 D10 做小规模独立任务/独立 seed 确认和真实端到端计时；再验证任务限定两层 mask，而不是继续扩大五六层 mask。冻结候选、接受标准和非劣界限后再测，避免筛选获胜者后用同一数据证明。

阶段主线保留，但调整证据路线：预注册少量层×阶段对比、同 init 配对、控制场景扰动；以干预动作和闭环失败作为主结果，激活/抵消/能量集中度作为解释。轻量门控应测净收益和误接受率，加入目标状态/姿态/夹爪错误诊断，不能只最大化动作接受率。

并行验证目前更像一致性诊断工具；暂不优先投入更多双卡吞吐预算。短前缀恢复有“错误可恢复”的证据，但净收益小；若探索真正 async，必须单独测 stale observation、执行重叠、取消/回退、系统吞吐及双卡成本，现有 request ledger 单元测试不是已完成异步机器人试验。

| 项目 | 状态 | 不能越界的结论 |
| --- | --- | --- |
| 四套件五状态 probe / 数值 / 单层干预 | 已完成 | 非随机数据，但不是完整任意层两两矩阵 |
| 启发式候选真实组合搜索与冻结闭环 | 已完成本轮范围 | 非全局最优；实际挑战占比偏离原计划 |
| Attention/FFN 代表样本与 L16/L17 五状态 | 已完成本轮范围 | 选层/阶段覆盖不均，诊断模式不可与标准成绩混算 |
| 短前缀恢复 / Kalman / 并行验证 H4/8/12 | 已完成当前筛选 | 没有可靠净加速证明 |
| 逐阶段不同 mask 的动态闭环 | 未完成有效验证 | 当前四阶段 mask 相同，不能支持阶段路由优势 |
| 真实执行重叠 async 与 50–200ms 延迟扰动 | 未完成 | 基础设施不是系统实验 |
| Kalman Q/R、方差 floor、2cm onset 阈值敏感性 | 未完成 | 门控/首次偏差当前定义只作探索描述 |
| 全层两两 hidden 矩阵、完整 QKV/FFN gate 表征 | 未采集或未完成 | 不伪造缺失数据 |
| 历史 Control 21% / eager 65% 的原因闭环 | 尚未定位 | 高标准基线不等于已找到旧 bug |
| 本轮每 suite 50 init 全量扩展 | 按研究筛选目标未开展 | 不是遗漏失败，不能写成 500 条全量 |

## 11. 复核入口

实验方案位于工作区根目录 [idea1 实验方案](../../idea1_experiment_plan.md)。可复现脚本为 [build_final_report.py](scripts/experiment10/build_final_report.py)，执行 `python3 scripts/experiment10/build_final_report.py`。无需 GPU，不启动新实验。全部新 CSV、PNG/PDF、输入文件 SHA256 和轨迹/视频索引保存在本报告的 final_analysis 子目录。
''')
    sections.append('\n'.join(link(OUT/f,f) for f in ['completed_coverage.csv','paired_success_task_bootstrap.csv','layer_prior.csv','numerical_layer_modality.csv',
        'submodule_interventions.csv','paired_xyz_and_onset.csv','failure_highest_stage_counts.csv','parallel_verify_summary.csv',
        'attention_multi_init_results.csv','attention_multi_init_pairs.csv','example_index.json','provenance.json']))
    REPORT.write_text('\n\n'.join(sections)+'\n')
    (OUT/'example_index.json').write_text(json.dumps(examples,indent=2))
    (OUT/'provenance.json').write_text(json.dumps({'report_scope':'executed experiment10 + attention supplemental',
        'closed_loop_episodes':len(ep),'supplemental_rollouts':240,'observation_count':total_observations,
        'bootstrap_seed':20261010,'bootstrap_draws':10000,'input_sha256':PROVENANCE},indent=2))
    print(f'Wrote {REPORT}; figures={len(list(OUT.glob("*.png")))}',flush=True)


if __name__ == '__main__':
    main()

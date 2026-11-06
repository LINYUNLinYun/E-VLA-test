"""Episode/suite-balanced descriptive analysis of completed shadow probes only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 241024
BOOTSTRAPS = 2000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def estimate(frame, column):
    """Mean within episode, then suite; stratified episode cluster bootstrap."""
    ep = frame.groupby(['suite','episode'], sort=True)[column].mean()
    rng = np.random.default_rng(SEED)
    suite_means, boot = [], []
    for suite in ep.index.get_level_values('suite').unique():
        values = ep.loc[suite].to_numpy(float)
        suite_means.append(values.mean())
        boot.append(values[rng.integers(0,len(values),size=(BOOTSTRAPS,len(values)))].mean(axis=1))
    samples = np.mean(boot,axis=0)
    low, high = np.quantile(samples,[.025,.975])
    return dict(mean=float(np.mean(suite_means)), ci_low=float(low), ci_high=float(high),
                episodes=len(ep), pairs=int(frame[['episode','pair_id']].drop_duplicates().shape[0]))


def summarize(frame, groups, values):
    rows = []
    for key, part in frame.groupby(groups, sort=True):
        key = key if isinstance(key,tuple) else (key,)
        identity = dict(zip(groups,key))
        for value in values:
            rows.append(dict(**identity, metric=value, **estimate(part,value)))
    return pd.DataFrame(rows)


def read_run(path):
    path = Path(path)
    manifest = json.loads((path/'manifest.json').read_text())
    # Reject before opening the periodically rewritten CSVs.
    if manifest.get('status') != 'complete' or manifest.get('completed_pairs') != manifest.get('num_pairs'):
        raise ValueError('Refusing incomplete/running probe: '+str(path))
    selection = json.loads((path/'selection.json').read_text())
    if sha(path/'selection.json') != manifest['selection_sha256']:
        raise ValueError('Selection digest changed')
    actions = pd.read_csv(path/'action_metrics.csv')
    layers = pd.read_csv(path/'layer_metrics.csv')
    for frame in (actions,layers):
        if set(frame.pair_id) != set(range(len(selection))):
            raise ValueError('Incomplete pair coverage')
        frame['suite'] = frame.episode.str.split('/').str[0]
        frame['run'] = 'D'+str(manifest['denoising'])
    if len(actions) != len(selection)*6*6 or len(layers) != len(selection)*2*2*18*3:
        raise ValueError('Unexpected metric row count')
    if not all(s['first_two_layers_bitwise_equal'] for s in manifest['sanity']) or not manifest['same_seed_full_repeat']['bitwise_equal']:
        raise ValueError('Run sanity failed')
    # Independent reconstruction of action metrics from saved decoded arrays.
    errors, raw_arrays = [], 0
    for pair_id in range(len(selection)):
        with np.load(path/f'actions_pair{pair_id:03d}.npz',allow_pickle=False) as arrays:
            base = arrays['full_current']
            for candidate in actions.candidate.unique():
                candidate_actions = arrays[candidate]
                raw = arrays[candidate+'_raw']
                if candidate_actions.shape != (50,7) or raw.shape != (50,32) or not np.isfinite(raw).all() or not np.isfinite(candidate_actions).all():
                    raise ValueError('Invalid saved action array')
                raw_arrays += 1
                for horizon in (5,50):
                    for component,sl in (('translation',slice(0,3)),('rotation',slice(3,6)),('gripper',slice(6,7))):
                        row = actions[(actions.pair_id==pair_id)&(actions.candidate==candidate)&(actions.horizon==horizon)&(actions.component==component)].iloc[0]
                        d = candidate_actions[:horizon,sl]-base[:horizon,sl]
                        expected = dict(mean_l2=np.linalg.norm(d,axis=1).mean(), rmse=np.sqrt(np.mean(d*d)), max_abs=np.abs(d).max())
                        if component == 'translation':
                            delta = np.clip(candidate_actions[:horizon,:3],-1,1)-np.clip(base[:horizon,:3],-1,1)
                            expected['clipped_controller_translation_mm_proxy'] = np.linalg.norm(delta,axis=1).mean()*50
                        if component == 'gripper':
                            expected['sign_mismatch_fraction'] = np.mean((candidate_actions[:horizon,6]>=0)!=(base[:horizon,6]>=0))
                        errors.extend(abs(float(row[k])-float(v)) for k,v in expected.items())
    max_error = max(errors)
    if max_error > 1e-6:
        raise ValueError('Independent action metric reconstruction differs: '+str(max_error))
    receipt = dict(run=str(path.resolve()), denoising=manifest['denoising'], pairs=len(selection), raw_arrays_verified=raw_arrays,
                   independent_action_metric_max_abs_error=max_error,
                   files={name:sha(path/name) for name in ('manifest.json','selection.json','action_metrics.csv','layer_metrics.csv')})
    return actions,layers,receipt


def produce(actions,layers,dest):
    # Store selected metrics as long values for uniform summaries and deltas.
    action_values = []
    for component,metric,label in [('translation','clipped_controller_translation_mm_proxy','translation_clipped_command_mm_proxy'),
                                  ('rotation','mean_l2','rotation_native_coordinate_mean_l2'),
                                  ('gripper','sign_mismatch_fraction','gripper_sign_mismatch_fraction'),
                                  ('gripper','mean_l2','gripper_raw_mean_l2')]:
        part = actions[actions.component==component].copy()
        part['quantity'],part['value'] = label,part[metric]
        action_values.append(part[['run','suite','episode','pair_id','candidate','horizon','quantity','value']])
    values = pd.concat(action_values,ignore_index=True)
    abs_summary = summarize(values,['run','candidate','horizon','quantity'],['value'])
    abs_summary.to_csv(dest/'action_summary.csv',index=False)
    summarize(values,['run','suite','candidate','horizon','quantity'],['value']).to_csv(dest/'action_by_suite.csv',index=False)
    baseline = values[values.candidate=='v14_current'].drop(columns='candidate').rename(columns={'value':'v14_value'})
    paired = values.merge(baseline,on=['run','suite','episode','pair_id','horizon','quantity'],validate='many_to_one')
    paired['delta_vs_v14'] = paired.value-paired.v14_value
    paired['win_vs_v14'] = (paired.value < paired.v14_value).astype(float)
    paired['tie_vs_v14'] = np.isclose(paired.value,paired.v14_value,atol=1e-12,rtol=0).astype(float)
    paired.to_csv(dest/'action_paired_values.csv',index=False)
    summarize(paired,['run','candidate','horizon','quantity'],['delta_vs_v14','win_vs_v14','tie_vs_v14']).to_csv(dest/'action_paired_summary.csv',index=False)
    summarize(paired,['run','suite','candidate','horizon','quantity'],['delta_vs_v14','win_vs_v14','tie_vs_v14']).to_csv(dest/'action_paired_by_suite.csv',index=False)
    ls = summarize(layers,['run','candidate','kv','layer','region'],['relative_l2','cosine','rms_ratio'])
    ls.to_csv(dest/'layer_summary.csv',index=False)
    band = layers.copy()
    band['band'] = pd.cut(band.layer,[0,4,11,18],labels=['early_L1_4','middle_L5_11','late_L12_18'])
    summarize(band,['run','candidate','kv','region','band'],['relative_l2','cosine','rms_ratio']).to_csv(dest/'layer_bands.csv',index=False)
    vlm = layers[layers.candidate=='current_v14'].drop(columns='candidate')
    old = layers[layers.candidate=='previous_full'].merge(vlm,on=['run','suite','episode','pair_id','kv','layer','region','current_request'],suffixes=('_old','_v14'))
    old['old_lower_relative_l2'] = (old.relative_l2_old < old.relative_l2_v14).astype(float)
    old['old_higher_cosine'] = (old.cosine_old > old.cosine_v14).astype(float)
    summarize(old,['run','kv','layer','region'],['old_lower_relative_l2','old_higher_cosine']).to_csv(dest/'layer_old_win_fraction.csv',index=False)
    trend = []
    from scipy.stats import spearmanr
    for key,part in layers[layers.layer>=5].groupby(['run','suite','episode','pair_id','candidate','kv','region']):
        trend.append(dict(zip(['run','suite','episode','pair_id','candidate','kv','region'],key), spearman_layer_vs_relative_l2=float(spearmanr(part.layer,part.relative_l2).statistic)))
    summarize(pd.DataFrame(trend),['run','candidate','kv','region'],['spearman_layer_vs_relative_l2']).to_csv(dest/'layer_trend_descriptive.csv',index=False)
    make_figures(ls,abs_summary,dest)


def make_figures(layers,actions,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(current_v14='#cf6b26', previous_full='#277c9d')
    runs = sorted(layers.run.unique())
    fig,axes = plt.subplots(2,2,figsize=(11,7),sharex=True)
    for row,kv in enumerate(('K','V')):
        for col,metric in enumerate(('relative_l2','cosine')):
            ax=axes[row,col]
            for run in runs:
                for candidate in colors:
                    part = layers[(layers.run==run)&(layers.kv==kv)&(layers.metric==metric)&(layers.region=='all')&(layers.candidate==candidate)].sort_values('layer')
                    ax.plot(part.layer,part['mean'],color=colors[candidate],linestyle='-' if run=='D2' else '--',label=run+' '+candidate)
            for skip in (3,4,9,11): ax.axvline(skip,color='gray',alpha=.18,lw=1)
            ax.set(title=kv+' '+metric,xlabel='AE prefix KV layer (1-based)',ylabel='Episode/suite-balanced mean')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Recorded Full trajectory: candidate prefix KV vs current Full; all tokens\nSkip [3,4,9,11]; L18 materializes KV only (KV remains consumed by AE)',fontsize=11)
    fig.tight_layout();fig.savefig(dest/'layer_kv.png',dpi=180);plt.close(fig)
    part = actions[(actions.horizon==5)&(actions.quantity=='translation_clipped_command_mm_proxy')&(actions.candidate!='full_current')]
    candidates = ['v14_current','v14_then_old_from_L10','v14_then_old_from_L13','v14_then_old_from_L16','all_old']
    fig,ax=plt.subplots(figsize=(10,5))
    x=np.arange(len(candidates));width=.35
    for i,run in enumerate(runs):
        rows=part[part.run==run].set_index('candidate').loc[candidates]
        pos=x+(i-(len(runs)-1)/2)*width
        ax.bar(pos,rows['mean'],width,label=run)
        ax.errorbar(pos,rows['mean'],yerr=np.maximum(0,np.array([rows['mean']-rows.ci_low,rows.ci_high-rows['mean']])),fmt='none',ecolor='black',capsize=3)
    ax.set_xticks(x,['Current V14','Old from L10','Old from L13','Old from L16','All old'])
    ax.set(ylabel='Clipped translation command difference proxy (mm)',title='First 5 commands vs current Full, same current state/noise\nSuite/episode-balanced means; 95% episode cluster bootstrap intervals')
    ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(dest/'action_first5_translation.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--output',default='results/kv_reuse_probe/analysis_20261024')
    args=p.parse_args()
    # All status checks occur before output mutation or metric reads.
    for run in args.runs:
        if json.loads((Path(run)/'manifest.json').read_text()).get('status')!='complete':
            raise ValueError('Refusing incomplete/running probe: '+run)
    loaded=[read_run(run) for run in args.runs]
    if len({item[2]['denoising'] for item in loaded})!=len(loaded):
        raise ValueError('Pass one run per denoising depth')
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    produce(pd.concat([x[0] for x in loaded]),pd.concat([x[1] for x in loaded]),dest)
    receipt=dict(status='complete',source_sha256=sha(__file__),bootstrap_replicates=BOOTSTRAPS,seed=SEED,
                 weighting='Mean pair values within episode; mean episodes within suite; equal suite mean. Stratified resampling episodes within each suite preserves pair clusters.',
                 interpretation='Exploratory uncorrected comparisons; skip mask [3,4,9,11] was previously selected. No closed-loop, speed or EEF displacement claim. D2/D10 sample sets differ; differences across D are not paired D effects. Trend is descriptive, not layer-independent significance.',
                 inputs=[x[2] for x in loaded])
    (dest/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()

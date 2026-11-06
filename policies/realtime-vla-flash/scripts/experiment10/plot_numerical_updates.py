"""Plot task-equal update diagnostics with explicit phase/task coverage."""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.experiment10.common import RESULTS, SUITES, PHASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=SUITES, required=True)
    args = parser.parse_args()
    directory = RESULTS / 'numerical_updates' / args.suite
    data = pd.read_csv(directory / 'module_cancellation.csv')
    data = data[data.split == 'search']
    metrics = ['relative_attn_update', 'relative_ffn_update', 'cancellation_ratio',
               'attn_ffn_cosine_derived']
    # Observations -> episodes -> tasks -> suite; do not overweight long rollouts.
    episodes = data.groupby(['task_id', 'episode_idx', 'semantic_phase', 'layer', 'modality'])[metrics].mean()
    tasks = episodes.groupby(['task_id', 'semantic_phase', 'layer', 'modality']).mean()
    mean = tasks.groupby(['semantic_phase', 'layer', 'modality']).mean()
    counts = tasks.groupby(['semantic_phase', 'layer', 'modality']).count()
    tasks.to_csv(directory / 'task_phase_updates.csv')
    for metric in metrics:
        fig, axes = plt.subplots(1, 3, figsize=(13, 7), constrained_layout=True)
        for axis, modality in zip(axes, ['agent', 'wrist', 'language']):
            subset = mean.xs(modality, level='modality')[metric]
            table = subset.unstack('semantic_phase').reindex(index=range(1, 18), columns=PHASES)
            coverage = counts.xs(modality, level='modality')[metric].unstack('semantic_phase').reindex_like(table)
            values = table.to_numpy()
            options = {'cmap': 'viridis'}
            if metric == 'attn_ffn_cosine_derived':
                options = {'cmap': 'coolwarm', 'vmin': -1, 'vmax': 1}
            elif metric == 'cancellation_ratio':
                options.update(vmin=0, vmax=1)
            artist = axis.imshow(np.ma.masked_invalid(values), aspect='auto', **options)
            for row in range(17):
                for column in range(len(PHASES)):
                    count = coverage.iloc[row, column]
                    axis.text(column, row, '—' if pd.isna(count) else 'n='+str(int(count)),
                              ha='center', va='center', fontsize=6, color='black',
                              bbox={'facecolor': 'white', 'alpha': .55, 'edgecolor': 'none', 'pad': .3})
            axis.set_xticks(range(len(PHASES)), PHASES, rotation=35, ha='right')
            axis.set_yticks(range(17), ['L'+str(i) for i in range(1, 18)])
            axis.set_title(modality)
            fig.colorbar(artist, ax=axis, shrink=.65)
        fig.suptitle(args.suite+' · '+metric+'\nSearch tasks only; task-equal mean; n=valid tasks; L18 KV-only excluded')
        fig.savefig(directory / (metric+'.png'), dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    main()

"""Same-init discordances and task-clustered exploratory success differences."""
import json
import numpy as np

from scripts.experiment10.common import RESULTS, SUITES, dump_json, sha256, split


def main():
    summary = json.loads((RESULTS/'closed_loop_analysis/validated_results.json').read_text())
    complete = {(r['condition'], r['suite']) for r in summary['suite_rows'] if r['status'] == 'validated_complete'}
    output = []
    for suite in SUITES:
        base_path = RESULTS/'closed_loop/Full18_D10'/suite/'run/episode_log.json'
        baseline = {(e['task_id'], e['episode_idx']): e['success'] for e in json.loads(base_path.read_text())}
        for condition in ('Full18_D1', 'Full18_D2', 'Full18_D4'):
            if (condition, suite) not in complete:
                continue
            path = RESULTS/'closed_loop'/condition/suite/'run/episode_log.json'
            candidate = {(e['task_id'], e['episode_idx']): e['success'] for e in json.loads(path.read_text())}
            if candidate.keys() != baseline.keys():
                raise RuntimeError('Paired matrix differs')
            for role in ('all_exploratory', 'search', 'validation', 'evaluation'):
                keys = [k for k in baseline if role == 'all_exploratory' or split(k[0]) == role]
                tasks = sorted({k[0] for k in keys})
                effects = np.asarray([np.mean([int(candidate[k])-int(baseline[k]) for k in keys if k[0] == task])
                                      for task in tasks])
                rng = np.random.default_rng(20261009)
                draws = rng.integers(0, len(tasks), size=(10000, len(tasks)))
                interval = np.quantile(effects[draws].mean(axis=1), [.025, .975])
                counts = {'both_success': 0, 'baseline_only_success': 0, 'variant_only_success': 0, 'both_failure': 0}
                for key in keys:
                    b, v = baseline[key], candidate[key]
                    label = 'both_success' if b and v else 'baseline_only_success' if b else 'variant_only_success' if v else 'both_failure'
                    counts[label] += 1
                output.append({'suite': suite, 'condition': condition, 'role': role, 'episodes': len(keys),
                               'tasks': len(tasks), 'paired_success_difference_variant_minus_baseline': float(effects.mean()),
                               'task_bootstrap95': interval.tolist(), 'discordances': counts,
                               'baseline_sha256': sha256(base_path), 'variant_sha256': sha256(path),
                               'interpretation': 'exploratory; tiny validation/evaluation task counts; no noninferiority claim'})
    dump_json(RESULTS/'closed_loop_analysis/paired_success.json', {'rows': output,
              'bootstrap_unit': 'task; all five inits retained together', 'bootstrap_repeats': 10000})
    print(len(output), 'paired summaries', flush=True)


if __name__ == '__main__':
    main()

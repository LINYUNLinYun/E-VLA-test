"""Audit discrete exact-test resolution, without changing post-hoc significance rules."""
import pandas as pd

from scripts.experiment10.common import RESULTS, dump_json, sha256


def main():
    path = RESULTS/'phase_analysis/phase_interaction.csv'
    frame = pd.read_csv(path)
    rows = []
    for (suite, role), group in frame.groupby(['suite', 'split']):
        size = len(group)
        # Two-sided absolute statistic: each extreme sign assignment has its negative.
        best_resolution = 2**(1-int(group.task_count.max()))
        rows.append({'suite': suite, 'split': role, 'family_comparisons': size,
                     'minimum_tasks': int(group.task_count.min()), 'maximum_tasks': int(group.task_count.max()),
                     'smallest_possible_two_sided_p': best_resolution,
                     'holm_first_step_alpha': .05/size,
                     'any_holm_rejection_mathematically_possible': best_resolution <= .05/size,
                     'observed_holm_rejections': int((group.p_holm < .05).sum()),
                     'leave_one_task_out_sign_stable': int(group.leave_one_task_out_sign_stable.sum())})
    dump_json(RESULTS/'phase_analysis/discrete_test_power_audit.json', {'families': rows,
              'source_sha256': sha256(path), 'alpha': .05,
              'interpretation': 'No rejection is not proof of equal sensitivities; report effect sizes and replication.',
              'posthoc_correction_changed': False, 'additional_evaluation_automatically_launched': False})
    print(rows, flush=True)


if __name__ == '__main__':
    main()

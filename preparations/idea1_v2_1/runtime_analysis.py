"""Recompute E1 denominators, task-cluster paired bootstrap and measured XYZ plots."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
try:
    from .runtime_client import SUITES, sha, write_json
except ImportError:
    from runtime_client import SUITES, sha, write_json


def csv_file(path, rows):
    if not rows:
        return
    with Path(path).open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap(pairs, repetitions=2000):
    """Sample tasks within each suite, retaining all init pairs of each task."""
    if not pairs:
        return (None, None)
    by_suite = {}
    for suite, task, full, skip in pairs:
        by_suite.setdefault(suite, {}).setdefault(task, []).append(float(skip)-float(full))
    if set(by_suite) != set(SUITES):
        return (None, None)
    rng = np.random.default_rng(1701)
    draws = []
    for _ in range(repetitions):
        means = []
        for suite in SUITES:
            groups = list(by_suite[suite].values())
            indices = rng.integers(0, len(groups), len(groups))
            means.append(np.mean([np.mean(groups[i]) for i in indices]))
        draws.append(np.mean(means) * 100)
    return tuple(float(v) for v in np.percentile(draws, [2.5, 97.5]))


def trajectory_plot(full, variant, output, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    a = np.asarray([r['eef_pos'] for r in full])
    b = np.asarray([r['eef_pos'] for r in variant])
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (left, right) in zip(axes[:3], [(0, 1), (0, 2), (1, 2)]):
        ax.plot(a[:, left], a[:, right], '.-', color='tab:blue', alpha=.6, label='Full measured EEF')
        ax.plot(b[:, left], b[:, right], '--', color='tab:orange', alpha=.6, label='Skip measured EEF')
        ax.set_xlabel('XYZ'[left] + ' (m)')
        ax.set_ylabel('XYZ'[right] + ' (m)')
        ax.axis('equal')
    n = min(len(a), len(b))
    delta = np.linalg.norm(a[:n] - b[:n], axis=1) * 1000
    axes[3].plot(np.arange(n)*.05, delta)
    axes[3].set_xlabel('Same physical time (s)')
    axes[3].set_ylabel('Measured EEF deviation (mm)')
    previous = None
    for row in variant:
        if row['semantic_phase'] != previous:
            axes[3].axvline(row['step'] * .05, color='grey', alpha=.3)
            axes[3].text(row['step'] * .05, 0, row['semantic_phase'], rotation=90, fontsize=6)
            previous = row['semantic_phase']
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=140)
    plt.close(fig)
    sustained = next((i for i in range(max(0, n-4)) if np.all(delta[i:i+5] > 50)), None)
    return dict(first_sustained_50mm_step=sustained, same_time_eef_p95_mm=float(np.percentile(delta, 95)))


def analyze(root):
    root = Path(root)
    ledger = json.loads((root / 'ledger.json').read_text())
    rows, sources = [], []
    for entry in ledger['entries']:
        if entry['status'] != 'valid':
            continue
        path = Path(entry['directory']) / 'validated.json'
        row = json.loads(path.read_text())
        row['condition'] = entry['key'].split('/')[1]
        sources.append(dict(path=str(path), sha256=sha(path)))
        if row['kind'] == 'formal':
            rows.append(row)
    index = {(r['condition'], r['suite'], r['task'], r['init']): r for r in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate formal identity')
    identity = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    out = root / 'analysis' / identity[:16]
    out.mkdir(parents=True, exist_ok=False)
    summary, detailed, latency, failures, trajectories = [], [], [], [], []
    (out / 'trajectories').mkdir()
    for condition in sorted({r['condition'] for r in rows}):
        condition_rows = [r for r in rows if r['condition'] == condition]
        pairs = []
        for row in condition_rows:
            full = index.get(('full', row['suite'], row['task'], row['init']))
            if full:
                pairs.append((row['suite'], row['task'], full['success'], row['success']))
        low, high = paired_bootstrap(pairs)
        suite_means = [np.mean([r['success'] for r in condition_rows if r['suite'] == s]) for s in SUITES if any(r['suite'] == s for r in condition_rows)]
        full_rate = np.mean([p[2] for p in pairs]) if pairs else None
        rate = float(np.mean(suite_means))
        summary.append(dict(condition=condition, completed=len(condition_rows), planned=200,
                       success=sum(r['success'] for r in condition_rows), suite_equal_SR=rate if len(suite_means) == 4 else None,
                       observed_suites=len(suite_means),
                       paired_n=len(pairs), paired_delta_pp=float(np.mean([p[3]-p[2] for p in pairs])*100) if pairs else None,
                       bootstrap_low_pp=low, bootstrap_high_pp=high,
                       retention=(np.mean([p[3] for p in pairs])/full_rate) if full_rate else None,
                       new_failures=sum(p[2] and not p[3] for p in pairs), recoveries=sum(not p[2] and p[3] for p in pairs),
                       source_id=identity))
        for suite in SUITES:
            for task in [-1] + list(range(10)):
                group = [r for r in condition_rows if r['suite'] == suite and (task == -1 or r['task'] == task)]
                detailed.append(dict(condition=condition, suite=suite, task=task, completed=len(group),
                                     planned=50 if task == -1 else 5, success=sum(r['success'] for r in group),
                                     SR=np.mean([r['success'] for r in group]) if group else None, source_id=identity))
        timings = {}
        for row in condition_rows:
            directory = Path(row['directory'])
            for line in (directory / 'infer.jsonl').read_text().splitlines():
                infer = json.loads(line)
                for name, value in dict(infer.get('timing', {}), client_roundtrip_ms=infer['client_roundtrip_ms']).items():
                    if isinstance(value, (int, float)):
                        timings.setdefault(name, []).append(value)
            states = [json.loads(l) for l in (directory / 'telemetry.jsonl').read_text().splitlines()]
            comparison = index.get(('full', row['suite'], row['task'], row['init']))
            if condition != 'full' and comparison:
                full_states = [json.loads(l) for l in (Path(comparison['directory']) / 'telemetry.jsonl').read_text().splitlines()]
                figure = out / 'trajectories' / f"{condition}_{row['suite']}_task{row['task']:02d}_init{row['init']:02d}.png"
                metrics = trajectory_plot(full_states, states, figure, f"{condition} {row['suite']} task{row['task']} init{row['init']}")
                trajectories.append(dict(condition=condition, suite=row['suite'], task=row['task'], init=row['init'],
                                         full_video=str(Path(comparison['directory']) / 'video.mp4'),
                                         variant_video=str(directory / 'video.mp4'), figure=str(figure), **metrics,
                                         source_id=identity))
            failures.append(dict(condition=condition, suite=row['suite'], task=row['task'], init=row['init'],
                                 success=row['success'], furthest_phase=row['furthest_phase'],
                                 grasp_loss_candidate=any(any(v is True for v in a.get('robot_grasps', {}).values()) and
                                     not any(v is True for v in b.get('robot_grasps', {}).values()) for a, b in zip(states, states[1:])),
                                 source_id=identity))
        for name, values in timings.items():
            latency.append(dict(condition=condition, component=name, scope='parallel_closed_loop', n=len(values),
                                p50_ms=float(np.percentile(values, 50)), p95_ms=float(np.percentile(values, 95)), source_id=identity))
    csv_file(out / 'success_by_mask.csv', summary)
    csv_file(out / 'success_by_task_suite.csv', detailed)
    csv_file(out / 'latency_breakdown.csv', latency)
    csv_file(out / 'phase_failure_counts.csv', failures)
    csv_file(out / 'trajectory_comparisons.csv', trajectories)
    candidates_path = root / 'candidates.json'
    expected = set()
    if candidates_path.exists():
        conditions = ['full'] + [c['id'] for c in json.loads(candidates_path.read_text())]
        expected = {(c, s, t, i) for c in conditions for s in SUITES for t in range(10) for i in range(5)}
    valid_keys = set(index)
    calibration = [r for r in ledger['entries'] if r['status'] == 'valid' and r['kind'] == 'calibration']
    references = [r for r in ledger['entries'] if r['status'] == 'valid' and r['kind'] == 'reference']
    parity = root / 'readiness/parity.json'
    write_json(out / 'completion_audit.json', dict(source_id=identity, sources=sources,
               formal_completed=len(rows), formal_planned=2600, complete=False,
               exact_formal_key_coverage=bool(expected) and expected == valid_keys,
               missing_formal_keys=sorted(expected - valid_keys), unexpected_formal_keys=sorted(valid_keys-expected),
               calibration_completed=len(calibration), calibration_planned=8,
               references_completed=len(references), references_planned=4,
               parity_passed=parity.exists() and json.loads(parity.read_text()).get('passed', False),
               invalid=sum(r['status'] == 'invalid' for r in ledger['entries']),
               valid_failures_retried=False, root=str(root), missing=['exclusive_fixed_observation_microbenchmark'] ))
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    print(analyze(parser.parse_args().root))

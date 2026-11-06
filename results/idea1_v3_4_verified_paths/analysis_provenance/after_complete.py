"""Read-only post-run analysis; never launches or changes an evaluation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from collections import Counter

ROOT = Path('/root/nfs/embodied/benchmark')
OUT = ROOT / 'results/idea1_v3_4_verified_paths'
JOB = Path('/tmp/idea1_verified_analysis_job.json')
RESULT = Path('/tmp/idea1_verified_final_analysis.json')
sys.path.insert(0, str(ROOT))


def read_json(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def csv_rows(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def key(row):
    return row['suite'], int(row['task']), int(row['init'])


def flag(value):
    assert value in ('True', 'False')
    return value == 'True'


def live_controller(pid):
    proc = Path('/proc') / str(pid)
    if not proc.exists():
        return False
    try:
        command = (proc / 'cmdline').read_bytes().replace(b'\x00', b' ').decode()
        status = (proc / 'status').read_text()
    except FileNotFoundError:
        return False
    state = next(s for s in status.splitlines() if s.startswith('State:'))
    return 'preparations.idea1_v3_4_verified_paths.controller' in command and 'Z' not in state


def analyze():
    manifest = read_json(OUT / 'manifest.json')
    audit = read_json(OUT / 'audit.json')
    assert audit['passed'] and audit['target_records'] == 1000
    assert audit['kalman_calls'] == 0 and audit['profile_passed']
    assert all(sha(ROOT / p) == h for p, h in manifest['source_files'].items())
    assert all(sha(ROOT / p) == h for p, h in manifest['references'].items())
    plan = ROOT / 'docs/experiments/idea1/idea1_v3_4_verified_paths_plan.md'
    assert sha(plan) == manifest['plan_sha256']
    summary = csv_rows('summary.csv')
    episodes = csv_rows('episodes.csv')
    requests = csv_rows('requests.csv')
    expected = {key(c) for c in manifest['cases']}
    configs = {c['id']: c for c in manifest['conditions']}
    assert len(expected) == 200 and len(episodes) == 1000
    assert len(summary) == 5 and set(configs) == {r['condition'] for r in summary}
    for name, cfg in configs.items():
        if cfg['mode'] == 'verified':
            assert cfg['selection'] == 'jacobian' and cfg['projections'] == 4
            assert cfg['adjacency_rule'] == 'spaced' and cfg['period'] in (3, 5)
            assert cfg['compensation'] == 'extrapolate' and not cfg['kalman']
        ee = [r for r in episodes if r['condition'] == name]
        assert len(ee) == 200 and {key(r) for r in ee} == expected
        reported = next(r for r in summary if r['condition'] == name)
        assert int(reported['success']) == sum(flag(r['success']) for r in ee)
        assert int(reported['reused']) == sum(flag(r['reused']) for r in ee)
        qq = [r for r in requests if r['condition'] == name]
        assert len(qq) == sum(int(r['predictions']) for r in ee)
        assert sum(int(r['executed']) for r in qq) == sum(int(r['actions']) for r in ee)
        assert abs(sum(float(r['policy_ms']) for r in qq) / 1000
                   - sum(float(r['policy_wall_s']) for r in ee)) < 1e-5
    fresh = {key(r) for r in episodes if r['condition'] == 'full_d10' and not flag(r['reused'])}
    assert 160 <= len(fresh) <= 200
    base = [r for r in episodes if r['condition'] == 'full_d10' and key(r) in fresh]
    base_cost = sum(float(r['policy_wall_s']) * 1000 for r in base) / sum(int(r['actions']) for r in base)
    costs = {r['condition']: r for r in csv_rows('costs.csv')}
    conditions = []
    for name in configs:
        ee = [r for r in episodes if r['condition'] == name]
        qq = [r for r in requests if r['condition'] == name]
        ff = [r for r in qq if flag(r['full'])]
        vv = [r for r in qq if flag(r['verify'])]
        hot_full = [r for r in ff if float(r['selector_capture_ms']) == 0]
        matched = [r for r in ee if key(r) in fresh]
        matched_requests = [r for r in qq if key(r) in fresh]
        total = sum(float(r['policy_ms']) for r in qq)
        cold = sum(float(r['selector_capture_ms']) for r in qq)
        selector = sum(float(r['selector_ms']) for r in qq)
        matched_ms = sum(float(r['policy_wall_s']) * 1000 for r in matched)
        matched_cold = sum(float(r['selector_capture_ms']) for r in matched_requests)
        actions = sum(int(r['actions']) for r in matched)
        speed = base_cost / (matched_ms / actions)
        assert abs(speed - float(costs[name]['speedup'])) < 1e-9
        categories = Counter(r['category'] for r in qq)
        row = dict(
            condition=name, n=200, success=sum(flag(r['success']) for r in ee),
            requests=len(qq), matched_new_baseline_cases=len(fresh),
            measured_speedup_including_selector_capture=speed,
            cost_ratio_vs_baseline_including_selector_capture=1 / speed,
            speedup_excluding_explicit_selector_capture=base_cost / ((matched_ms - matched_cold) / actions),
            policy_ms_per_request=total / len(qq),
            policy_ms_per_request_minus_explicit_selector_capture=(total - cold) / len(qq),
            selector_capture_seconds_per_episode=cold / 1000 / 200,
            selector_fraction_including_capture=selector / total,
            selector_fraction_after_subtracting_its_capture=(selector - cold) / (total - cold),
            full_fraction=len(ff) / len(qq),
            fast_action_fraction=sum(int(r['executed']) for r in qq if r['category'].startswith('fast_'))
                                 / sum(int(r['executed']) for r in qq),
            zero_accept_per_verify=sum(int(r['accepted_length']) == 0 for r in vv) / len(vv) if vv else None,
            verify_ms_per_call=statistics.mean(float(r['verify_ms']) for r in vv) if vv else None,
            selector_hot_ms_per_full=statistics.mean(float(r['selector_ms']) for r in hot_full) if hot_full else None,
            categories={k: dict(n=v, request_fraction=v / len(qq)) for k, v in sorted(categories.items())},
        )
        conditions.append(row)
    # A same-q, same-budget, period-5 comparison to the completed no-verifier run.
    # This estimates the combined verifier and early-Full-routing change.
    from preparations.idea1_v3_4_verified_paths.evaluate import header
    from preparations.idea1_v3_4_E2_selection_200.finish import paired
    old_root = ROOT / 'results/idea1_v3_4_E2_cycle5_q4_v12'
    assert read_json(old_root / 'audit.json')['passed']
    old_name = 'v12_jacobian_q4_extrapolate'
    new_name = 'v12_extrapolate_d2_verify_p5'
    old, new = {}, {}
    for c in manifest['cases']:
        kk = key(c)
        suffix = Path('episodes') / c['suite'] / f"task{c['task']:02d}" / f"init{c['init']:02d}"
        old[kk] = header(old_root / old_name / suffix)
        new[kk] = header(OUT / new_name / suffix)
        assert {k: v for k, v in old[kk]['identity'].items() if k != 'first_input_hash'} == {
            k: v for k, v in new[kk]['identity'].items() if k != 'first_input_hash'
        }
    verification_pairs = []
    prefix_exceptions = []
    for scope in ('strict', 'all200'):
        row, ex = paired('q4_v12_p5_without_verification', new_name, old, new, expected, scope, True)
        verification_pairs.append(row)
        if scope == 'strict':
            prefix_exceptions = ex
    files = ['manifest.json', 'audit.json', 'complete.json', 'summary.csv', 'episodes.csv',
             'requests.csv', 'costs.csv', 'paired.csv', 'paired_sensitivity.csv',
             'calls_and_ratios.csv', 'profile_summary.csv', 'prefix_audit.json']
    result = dict(
        complete=True, time=time.time(), analysis_source_sha256=sha(__file__),
        experiment_audit_sha256=sha(OUT / 'audit.json'),
        input_hashes={name: sha(OUT / name) for name in files},
        selection=manifest['selection'], conditions=conditions,
        paired=csv_rows('paired.csv'), paired_all200=csv_rows('paired_sensitivity.csv'),
        calls_and_ratios=csv_rows('calls_and_ratios.csv'), profiles=csv_rows('profile_summary.csv'),
        verification_and_early_full_paired=verification_pairs,
        verification_comparison_prefix_exceptions=prefix_exceptions,
        limitations=[
            'Subtracting explicit selector graph capture does not remove all other startup costs.',
            'Closed-loop cost ratios compare different trajectories and request mixtures.',
            'Verification comparison includes its early-Full routing and cache-history changes.',
            'Exploratory confidence intervals are not adjusted for multiple comparisons.',
            'Findings apply to online four-VJP, spaced selection, not all Jacobian methods.',
        ],
    )
    atomic(RESULT, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    controller = read_json(OUT / 'process.json')['pid']
    while True:
        if (OUT / 'complete.json').exists() and (OUT / 'audit.json').exists():
            assert read_json(OUT / 'audit.json')['passed']
            atomic(JOB, dict(status='analyzing', pid=os.getpid(), controller=controller, time=time.time()))
            result = analyze()
            atomic(JOB, dict(status='complete', pid=os.getpid(), result=str(RESULT), time=time.time()))
            print(json.dumps(dict(status='complete', result=str(RESULT), conditions=result['conditions'])), flush=True)
            return
        live = live_controller(controller)
        progress = read_json(OUT / 'status.json')
        state = dict(status='waiting_experiment' if live else 'controller_stopped_before_completion',
                     pid=os.getpid(), controller=controller, controller_live=live,
                     completed=progress.get('completed'), target=1000, time=time.time())
        atomic(JOB, state)
        if args.once or not live:
            print(json.dumps(state), flush=True)
            return
        time.sleep(30)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        atomic(JOB, dict(status='failed', pid=os.getpid(), error=traceback.format_exc(), time=time.time()))
        raise

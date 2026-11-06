"""Frozen forty-case module-skip matrix and bounded, atomic task queue."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from preparations.idea1_v3.io import atomic, digest
from preparations.idea1_v3_1.common import lock
from .selection import RANDOM_SEED, fixed_episode_mask, freeze_budget

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/idea1_v3_4_E2_module_skip'
BASE = ROOT / 'results/idea1_v3_4_E2_cycle5_spaced'
PLAN = ROOT / 'docs/experiments/idea1/idea1_v3_4_E2_module_skip_plan.md'
BASE_MANIFEST_HASH = '463b7f060b35d6fb1d9674d67e62622b8ca6e2b2cd2275227e016593e77156c8'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
PHASES = ('pilot', 'small40')
METHODS = ('B-R+', 'B-E+', 'F-R0', 'F-R+', 'F-E+', 'F-Fixed+')
MAX_TECHNICAL_ATTEMPTS = 2


def config(budget, condition):
    family = 'block' if condition.startswith('B-') else 'ffn'
    scoring = 'fixed' if condition == 'F-Fixed+' else ('error' if '-E' in condition else 'raw')
    return dict(id=f'{budget}_{condition}', condition=condition, budget=budget,
                family=family, scoring=scoring,
                method={'raw': 'jacobian', 'error': 'comp_error', 'fixed': 'random_fixed'}[scoring],
                compensation='none' if condition == 'F-R0' else 'extrapolate',
                block_skip_count={'B4': 4, 'B6': 6}[budget],
                adjacency_rule='spaced', random_seed=RANDOM_SEED,
                model='pi0', mode='periodic', period=5, full_steps=10, fast_steps=2,
                H=50, R=5, n=40, phase='small40', reused=condition == 'B-R+',
                weights=[1 / 31, 2 / 31, 4 / 31, 8 / 31, 16 / 31])


CONDITIONS = [config(budget, condition) for budget in ('B4', 'B6') for condition in METHODS]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ck(case):
    return f"{case['suite']}_task{case['task']:02d}_init{case['init']:02d}"


def directory(name, case):
    return OUT / name / 'episodes' / case['suite'] / f"task{case['task']:02d}" / f"init{case['init']:02d}"


def source(manifest, cfg, case):
    """Completed records, including valid failures, are never rerun."""
    if cfg.get('reused'):
        row = manifest['baseline'][cfg['id']][ck(case)]
        return Path(row['directory']), True
    path = directory(cfg['id'], case)
    if not (path / 'complete.json').exists():
        return None, False
    result = json.loads((path / 'result.json').read_text())
    if not result.get('valid') or result.get('case') != case:
        raise ValueError(f'invalid completed episode: {path}')
    if result.get('manifest_hash') != manifest['manifest_hash'] or result.get('condition') != cfg['id']:
        raise ValueError(f'completed episode has a different frozen identity: {path}')
    return path, False


def _read_frozen(path):
    manifest = json.loads(Path(path).read_text())
    expected = manifest.pop('manifest_hash')
    if digest(manifest) != expected:
        raise ValueError(f'manifest hash mismatch: {path}')
    manifest['manifest_hash'] = expected
    return manifest


def _baseline(old, conditions):
    if old['manifest_hash'] != BASE_MANIFEST_HASH:
        raise ValueError('cycle-five reference manifest is not the declared baseline')
    entries = {}
    for cfg in conditions:
        if not cfg['reused']:
            continue
        old_id = f"v{14 if cfg['budget'] == 'B4' else 12}_jacobian_extrapolate"
        rows = entries[cfg['id']] = {}
        for case in old['small_cases']:
            path = BASE / old_id / 'episodes' / case['suite'] / f"task{case['task']:02d}" / f"init{case['init']:02d}"
            result = json.loads((path / 'result.json').read_text())
            if not (path / 'complete.json').exists() or not result.get('valid') or result.get('case') != case:
                raise ValueError(f'no valid reusable cycle-five baseline: {path}')
            if result.get('manifest_hash') != BASE_MANIFEST_HASH or result.get('condition') != old_id:
                raise ValueError(f'reference episode identity mismatch: {path}')
            files = ('result.json', 'identity.json', 'infer.jsonl', 'complete.json', 'arrays.npz')
            rows[ck(case)] = dict(directory=str(path), condition=old_id,
                                 success=bool(result['success']), hashes={name: sha(path / name) for name in files})
    return entries


def _make_groups(cases):
    # task1/7 include the declared pilots; each task owns two independent init
    # clients. Cases remain fixed and are never replaced based on outcomes.
    groups = []
    for task in (1, 7):
        for suite in SUITES:
            pilot = next(case for case in cases if case['suite'] == suite and case['task'] == task and case['init'] == 19)
            groups.append(dict(key=f'module_pilot_{suite}_task{task:02d}', phase='pilot',
                               suite=suite, task=task, state='pending',
                               cases=[dict(case=pilot, state='pending')]))
    for task in (1, 7, 3, 5, 9):
        for suite in SUITES:
            subset = [case for case in cases if case['suite'] == suite and case['task'] == task]
            if len(subset) != 2:
                raise ValueError('small-case task groups must contain the two fixed inits')
            groups.append(dict(key=f'module_skip_{suite}_task{task:02d}', phase='small40',
                               suite=suite, task=task, state='pending',
                               cases=[dict(case=case, state='pending') for case in subset]))
    return groups


def freeze(cost_table_path):
    """Write the matrix only once, after measured costs are available.

    Freezing does not run a model, allocate a GPU, or start an evaluator.
    """
    cost_table_path = Path(cost_table_path).resolve()
    table = json.loads(cost_table_path.read_text())  # missing costs cannot be fabricated
    budgets = {budget: freeze_budget(table, budget) for budget in ('B4', 'B6')}
    old = _read_frozen(BASE / 'manifest.json')
    cases = old['small_cases']
    expected = {(suite, task, init) for suite in SUITES for task in (1, 3, 5, 7, 9) for init in (19, 20)}
    if len(cases) != 40 or {(c['suite'], c['task'], c['init']) for c in cases} != expected:
        raise ValueError('reference small cases do not match the predeclared forty cases')
    conditions = [dict(cfg, budget_hash=budgets[cfg['budget']]['budget_hash']) for cfg in CONDITIONS]
    baseline = _baseline(old, conditions)
    pilots = [case for case in cases if case['task'] in (1, 7) and case['init'] == 19]
    if len(pilots) != 8:
        raise ValueError('pilot subset must contain eight cases per F-R+ budget')
    source_files = {str(path.relative_to(ROOT)): sha(path) for path in sorted(Path(__file__).parent.glob('*.py'))}
    shared_sources = (
        'preparations/idea1_v3/policy.py',
        'preparations/idea1_v3/safe_qk.py',
        'preparations/idea1_v3/io.py',
        'preparations/idea1_v3_1/common.py',
        'preparations/idea1_v3_1/rollout.py',
        'preparations/idea1_v2_1/runtime_client.py',
        'preparations/idea1_v2_1/runtime_models.py',
        'preparations/idea1_v2_2/pi0_adapter.py',
        'preparations/idea1_v2_2/schedules.py',
        'preparations/idea1_v3_4_E2_E4_single/torch_model.py',
        'preparations/idea1_v3_4_E2_online_time_weights/selector.py',
        'preparations/idea1_v3_4_E2_cycle3_full10_jacobian/model.py',
        'preparations/idea1_v3_4_E2_cycle3/common.py',
        'preparations/idea1_v3_4_E2_cycle3/policy.py',
        'preparations/idea1_v3_4_E2_adjacency_200/common.py',
        'preparations/idea1_v3_4_v12_compensation/backend.py',
        'preparations/idea1_v3_4/dispatch.py',
        'preparations/idea1_v3_4_E2_max2_repair/evaluate.py',
        'policies/realtime-vla-flash/scripts/experiment10/backend.py',
        'policies/realtime-vla-flash/scripts/experiment1/run_intermediate_action_probe.py',
        'policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py',
    )
    source_files.update({name: sha(ROOT / name) for name in shared_sources})
    references = {str((BASE / 'manifest.json').relative_to(ROOT)): sha(BASE / 'manifest.json')}
    manifest = dict(schema_version=1, conditions=conditions, cases=cases, small_cases=cases,
                    expansion_cases=old['cases'], pilot_cases=pilots, baseline=baseline,
                    budgets=budgets, cost_table=table, cost_table_path=str(cost_table_path),
                    cost_table_file_sha256=sha(cost_table_path), source_files=source_files,
                    references=references, plan_sha256=sha(PLAN),
                    target_records=480, reused_episodes=80, new_episodes=400,
                    phases=list(PHASES), pilot_barrier='16 F-R+ episodes and offline mechanism complete before other conditions',
                    condition_order='pilot: B4_F-R+, B6_F-R+; small40: rotate manifest conditions by frozen case index',
                    max_gpus=3, max_actors=2,
                    max_technical_attempts=MAX_TECHNICAL_ATTEMPTS,
                    random_seed=RANDOM_SEED, schedule=['Full-D10'] + ['Fast-D2'] * 4,
                    layer_range=[1, 17], last_layer='original current-observation KV path',
                    protocol=dict(model='pi0', precision='BF16', backend='native Triton',
                                  period=5, full_steps=10, fast_steps=2, H=50, R=5,
                                  env_seed=7, settling=10, hz=20, weights=[1, 2, 4, 8, 16],
                                  weight_denominator=31, projected_raw_steps=5, raw_dimensions=7,
                                  request_noise='inherit case/replan mapping from cycle5 baseline'),
                    baseline_protocol=[cfg for cfg in old['conditions'] if cfg['id'] in
                                       ('v14_jacobian_extrapolate', 'v12_jacobian_extrapolate')],
                    optional_mixed_stage='not included in this frozen 480-record matrix',
                    stage_limits=dict(profile_requests=5000, offline_requests=1000,
                                      pilot_input_replans=15, pilot_input_total=240),
                    fixed_episode_masks={cfg['id']: {ck(case): fixed_episode_mask(budgets[cfg['budget']], case)
                                                   for case in cases}
                                         for cfg in conditions if cfg['scoring'] == 'fixed'},
                    time=time.time())
    manifest['manifest_hash'] = digest(manifest)
    # Keep reference hashing outside the exclusive section; no evaluator exists
    # before the manifest/queue/progress commit is complete.
    with lock(OUT / 'freeze.lock'):
        if (OUT / 'manifest.json').exists():
            raise FileExistsError('module-skip manifest already frozen')
        rows = {}
        for cfg in conditions:
            reused = baseline.get(cfg['id'], {})
            rows[cfg['id']] = dict(target=40, completed=len(reused),
                                   success=sum(row['success'] for row in reused.values()),
                                   reused=len(reused), new=0, seen=list(reused))
        if cost_table_path != (OUT / 'cost_table.json').resolve():
            atomic(OUT / 'cost_table.json', table)
        atomic(OUT / 'queue.json', dict(groups=_make_groups(cases)))
        atomic(OUT / 'progress.json', dict(conditions=rows))
        atomic(OUT / 'manifest.json', manifest)
    return manifest


def register(cfg, case, path, result, reused=None):
    if not result.get('valid') or result.get('case') != case:
        raise ValueError('only valid fixed-identity episodes may be registered')
    if reused is None:
        reused = bool(cfg.get('reused'))
    with lock(OUT / 'progress.lock'):
        progress_data = json.loads((OUT / 'progress.json').read_text())
        row = progress_data['conditions'][cfg['id']]
        key = ck(case)
        if key in row['seen']:
            return
        atomic(OUT / 'completions' / cfg['id'] / f'{key}.json',
               dict(case=case, directory=str(path), success=bool(result['success']),
                    actions=result['actions'], wall_s=result['wall_s'], reused=bool(reused), time=time.time()))
        row['seen'].append(key)
        row['completed'] += 1
        row['success'] += int(result['success'])
        row['reused' if reused else 'new'] += 1
        atomic(OUT / 'progress.json', progress_data)


def progress():
    return {key: {field: value for field, value in row.items() if field != 'seen'}
            for key, row in json.loads((OUT / 'progress.json').read_text())['conditions'].items()}


def claim_group(gpu):
    with lock(OUT / 'queue.lock'):
        queue = json.loads((OUT / 'queue.json').read_text())
        pilots = [group for group in queue['groups'] if group['phase'] == 'pilot']
        pilot_ready = not pilots or (OUT / 'pilot_mechanism_complete.json').exists()
        for group in queue['groups']:
            if group['state'] == 'pending' and (group['phase'] == 'pilot' or pilot_ready):
                group.update(state='running', gpu=gpu, started=time.time())
                atomic(OUT / 'queue.json', queue)
                return {key: group[key] for key in ('key', 'phase', 'suite', 'task')}
    return None


def claim_case(group, client):
    with lock(OUT / 'queue.lock'):
        queue = json.loads((OUT / 'queue.json').read_text())
        row = next(row for row in queue['groups'] if row['key'] == group)
        if row['state'] != 'running':
            raise ValueError('cases may only be claimed from a running group')
        running = [case for case in row['cases'] if case['state'] == 'running']
        if any(case.get('client') == client for case in running):
            raise ValueError('client already owns a running case')
        if len(running) >= 2:
            return None
        for case in row['cases']:
            if case['state'] == 'pending':
                case.update(state='running', client=client, started=time.time())
                atomic(OUT / 'queue.json', queue)
                return case['case']
    return None


def finish_case(group, case):
    with lock(OUT / 'queue.lock'):
        queue = json.loads((OUT / 'queue.json').read_text())
        row = next(row for row in queue['groups'] if row['key'] == group)
        item = next(item for item in row['cases'] if item['case'] == case)
        item.update(state='complete', finished=time.time())
        atomic(OUT / 'queue.json', queue)


def finish_group(group):
    with lock(OUT / 'queue.lock'):
        queue = json.loads((OUT / 'queue.json').read_text())
        row = next(row for row in queue['groups'] if row['key'] == group)
        if not all(case['state'] == 'complete' for case in row['cases']):
            raise ValueError('cannot complete a group with unfinished cases')
        row.update(state='complete', finished=time.time())
        atomic(OUT / 'queue.json', queue)
        pilots = [group for group in queue['groups'] if group['phase'] == 'pilot']
        if pilots and all(group['state'] == 'complete' for group in pilots) and not (OUT / 'pilot_complete.json').exists():
            atomic(OUT / 'pilot_complete.json', dict(passed=True, cases=8, episodes=16,
                   conditions=['B4_F-R+', 'B6_F-R+'], maximum_saved_inputs=240, time=time.time()))



def release(gpu):
    """Requeue only interrupted cases; completed episode keys stay reusable."""
    with lock(OUT / 'queue.lock'):
        queue = json.loads((OUT / 'queue.json').read_text())
        for group in queue['groups']:
            if group.get('gpu') == gpu and group['state'] == 'running':
                group['state'] = 'pending'
                for case in group['cases']:
                    if case['state'] == 'running':
                        case['state'] = 'pending'
        atomic(OUT / 'queue.json', queue)


def begin_attempt(cfg, case):
    """Persist before execution: one initial try and at most one recovery."""
    path = OUT / 'technical_attempts' / cfg['id'] / f'{ck(case)}.json'
    with lock(OUT / 'attempts.lock'):
        prior = json.loads(path.read_text()) if path.exists() else {}
        count = prior.get('attempts', 0)
        if prior.get('fatal'):
            raise RuntimeError(f"fatal implementation failure: {cfg['id']} {ck(case)}")
        if count >= MAX_TECHNICAL_ATTEMPTS:
            raise RuntimeError(f"technical retry budget exhausted: {cfg['id']} {ck(case)}")
        count += 1
        atomic(path, dict(attempts=count, time=time.time()))
    return count


def record_technical_failure(cfg, case, error, fatal=False):
    """Record infrastructure errors separately from valid task failures."""
    path = OUT / 'technical_attempts' / cfg['id'] / f'{ck(case)}.json'
    with lock(OUT / 'attempts.lock'):
        row = json.loads(path.read_text())
        row.update(error=repr(error), fatal=bool(fatal), failed=time.time())
        atomic(path, row)
    return not fatal and row['attempts'] < MAX_TECHNICAL_ATTEMPTS


def pending():
    return any(group['state'] != 'complete' for group in json.loads((OUT / 'queue.json').read_text())['groups'])


def phase_done(phase):
    if phase not in PHASES:
        raise ValueError(f'unknown phase {phase}')
    groups = json.loads((OUT / 'queue.json').read_text())['groups']
    return all(group['state'] == 'complete' for group in groups if group['phase'] == phase)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cost-table', required=True)
    freeze(parser.parse_args().cost_table)

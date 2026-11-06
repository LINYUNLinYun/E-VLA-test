"""Frozen mixed A/F weighted-vs-unweighted forty-case comparison."""
import hashlib
import json
from pathlib import Path
import time

from preparations.idea1_v3.io import atomic, digest
from preparations.idea1_v3_1.common import lock

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/idea1_v3_4_E2_mixed_cost'
BASE = ROOT / 'results/idea1_v3_4_E2_module_skip'
PLAN = ROOT / 'docs/experiments/idea1/idea1_v3_4_E2_mixed_cost_plan.md'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
ALLOWED_GPUS = (0, 1, 3)
PHASES = ('main40',)
MAX_TECHNICAL_ATTEMPTS = 2


def config(budget, weighted):
    name = 'M-E-W+' if weighted else 'M-E-U+'
    return dict(id=f'{budget}_{name}', condition=name, budget=budget, family='mixed',
                scoring='error', cost_weighted=bool(weighted),
                method='comp_error_cost_weighted' if weighted else 'comp_error_unweighted',
                compensation='extrapolate', adjacency_rule='spaced', random_seed=341101,
                model='pi0', mode='periodic', period=5, full_steps=10, fast_steps=2,
                H=50, R=5, n=40, phase='main40', weights=[1 / 31, 2 / 31, 4 / 31, 8 / 31, 16 / 31])


CONDITIONS = [config(budget, weighted) for budget in ('B4', 'B6') for weighted in (False, True)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ck(case):
    return f"{case['suite']}_task{case['task']:02d}_init{case['init']:02d}"


def directory(name, case):
    return OUT / name / 'episodes' / case['suite'] / f"task{case['task']:02d}" / f"init{case['init']:02d}"


def read_manifest(path=OUT / 'manifest.json'):
    manifest = json.loads(Path(path).read_text())
    expected = manifest.pop('manifest_hash')
    if digest(manifest) != expected:
        raise ValueError(f'frozen manifest hash mismatch: {path}')
    manifest['manifest_hash'] = expected
    return manifest


def verify_frozen(manifest):
    for filename, expected in manifest['source_files'].items():
        if sha(ROOT / filename) != expected:
            raise ValueError(f'frozen source changed: {filename}')
    for filename, expected in manifest['references'].items():
        if sha(ROOT / filename) != expected:
            raise ValueError(f'frozen reference changed: {filename}')
    if sha(PLAN) != manifest['plan_sha256']:
        raise ValueError('frozen plan changed')
    if sha(manifest['cost_table_path']) != manifest['cost_table_file_sha256']:
        raise ValueError('frozen hardware cost table changed')
    for budget in manifest['budgets'].values():
        if sha(budget['candidate_path']) != budget['candidate_sha256']:
            raise ValueError('frozen mixed candidate file changed')


def freeze(cost_table_path):
    from .selection import freeze_mixed_budget
    cost_table_path = Path(cost_table_path).resolve()
    table = json.loads(cost_table_path.read_text())
    if table.get('measured') is not True or table.get('units') != 'ms':
        raise ValueError('measured native RTX4090 cost table in ms required')
    old = read_manifest(BASE / 'manifest.json')
    if not json.loads((BASE / 'audit.json').read_text()).get('passed'):
        raise ValueError('module-skip reference audit has not passed')
    cases = old['small_cases']
    expected = {(suite, task, init) for suite in SUITES for task in (1, 3, 5, 7, 9) for init in (19, 20)}
    if len(cases) != 40 or {(c['suite'], c['task'], c['init']) for c in cases} != expected:
        raise ValueError('case cohort differs from the fixed forty cases')
    OUT.mkdir(parents=True, exist_ok=True)
    with lock(OUT / 'freeze.lock'):
        if (OUT / 'manifest.json').exists():
            raise FileExistsError('mixed-cost manifest already frozen')
        budgets = {budget: freeze_mixed_budget(table, budget) for budget in ('B4', 'B6')}
        controls = {ck(case): old['baseline']['B4_B-R+'][ck(case)] for case in cases}
        files = dict(old['source_files'])
        files['preparations/idea1_v2_2/common.py'] = sha(ROOT / 'preparations/idea1_v2_2/common.py')
        files.update({str(path.relative_to(ROOT)): sha(path) for path in sorted(Path(__file__).parent.glob('*.py'))})
        references = {str((BASE / name).relative_to(ROOT)): sha(BASE / name)
                      for name in ('manifest.json', 'audit.json', 'complete.json')}
        conditions = [dict(cfg, budget_hash=budgets[cfg['budget']]['budget_hash']) for cfg in CONDITIONS]
        manifest = dict(schema_version=1, experiment='mixed_cost', conditions=conditions,
            cases=cases, small_cases=cases, controls=controls, budgets=budgets,
            cost_table=table, cost_table_path=str(cost_table_path), cost_table_file_sha256=sha(cost_table_path),
            source_files=files, references=references, plan_sha256=sha(PLAN),
            target_records=160, new_episodes=160, reused_episodes=0,
            max_gpus=3, max_actors=2, allowed_gpus=list(ALLOWED_GPUS), excluded_gpus=[2],
            phases=list(PHASES), max_technical_attempts=MAX_TECHNICAL_ATTEMPTS,
            condition_order='rotate four frozen conditions by case index; every task group stays on one GPU',
            schedule=['Full-D10'] + ['Fast-D2'] * 4,
            protocol=dict(model='pi0', precision='BF16', backend='native Triton', H=50, R=5,
                          env_seed=7, settling=10, hz=20, period=5, full_steps=10, fast_steps=2,
                          weights=[1, 2, 4, 8, 16], weight_denominator=31,
                          gate_shape=[17, 2], module_order=['attn', 'ffn'],
                          layer_range=[1, 17], last_layer='original current-observation KV path',
                          histories='own episode, two previous real Full A/F residual banks',
                          noise_mapping='preparations.idea1_v2_2.common.seed(suite,task,init,replan)'),
            objectives=dict(U='min sum(s_m)', W='min sum(s_m / c_m)',
                            shared_constraints='same measured total cost window and union-layer spacing',
                            AF_budget='directly measured joint AF cost',
                            AF_score='local sum of individual module scores, not a whole-block Jacobian'),
            old_results_use='case identity and historical quality context only; no direct episode registration',
            time=time.time())
        manifest['manifest_hash'] = digest(manifest)
        groups = [dict(key=f'mixed_{suite}_task{task:02d}', phase='main40', suite=suite, task=task,
                       state='pending', cases=[dict(case=c, state='pending') for c in cases
                                               if c['suite'] == suite and c['task'] == task])
                  for task in (1, 3, 5, 7, 9) for suite in SUITES]
        atomic(OUT / 'queue.json', dict(groups=groups))
        atomic(OUT / 'progress.json', dict(conditions={cfg['id']: dict(target=40, completed=0,
               success=0, new=0, reused=0, seen=[]) for cfg in conditions}))
        atomic(OUT / 'manifest.json', manifest)
    return manifest


def source(manifest, cfg, case):
    path = directory(cfg['id'], case)
    if not (path / 'complete.json').exists():
        return None, False
    result = json.loads((path / 'result.json').read_text())
    if not result.get('valid') or result.get('case') != case or result.get('condition') != cfg['id'] or result.get('manifest_hash') != manifest['manifest_hash']:
        raise ValueError(f'completed mixed episode has incompatible identity: {path}')
    return path, False


def register(cfg, case, path, result):
    if not result.get('valid') or result.get('case') != case:
        raise ValueError('only valid fixed-case episodes are registered')
    with lock(OUT / 'progress.lock'):
        progress_data = json.loads((OUT / 'progress.json').read_text())
        row = progress_data['conditions'][cfg['id']]
        key = ck(case)
        if key in row['seen']:
            return
        atomic(OUT / 'completions' / cfg['id'] / f'{key}.json', dict(case=case, directory=str(path),
               success=bool(result['success']), actions=result['actions'], wall_s=result['wall_s'], time=time.time()))
        row['seen'].append(key)
        row['completed'] += 1
        row['success'] += int(result['success'])
        row['new'] += 1
        atomic(OUT / 'progress.json', progress_data)


def progress():
    return {key: {field: value for field, value in row.items() if field != 'seen'}
            for key, row in json.loads((OUT / 'progress.json').read_text())['conditions'].items()}


def claim_group(gpu):
    with lock(OUT / 'queue.lock'):
        queue = json.loads((OUT / 'queue.json').read_text())
        for group in queue['groups']:
            if group['state'] == 'pending' and group.get('gpu', gpu) == gpu:
                group.update(state='running', gpu=gpu, started=time.time())
                atomic(OUT / 'queue.json', queue)
                return {key: group[key] for key in ('key', 'phase', 'suite', 'task')}
    return None


def claim_case(group, client):
    with lock(OUT / 'queue.lock'):
        queue = json.loads((OUT / 'queue.json').read_text())
        row = next(row for row in queue['groups'] if row['key'] == group)
        if row['state'] != 'running':
            raise ValueError('case claimed outside a running task group')
        running = [case for case in row['cases'] if case['state'] == 'running']
        if any(case.get('client') == client for case in running):
            raise ValueError('client already owns a case')
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
        next(item for item in row['cases'] if item['case'] == case).update(state='complete', finished=time.time())
        atomic(OUT / 'queue.json', queue)


def finish_group(group):
    with lock(OUT / 'queue.lock'):
        queue = json.loads((OUT / 'queue.json').read_text())
        row = next(row for row in queue['groups'] if row['key'] == group)
        if not all(case['state'] == 'complete' for case in row['cases']):
            raise ValueError('unfinished task group')
        row.update(state='complete', finished=time.time())
        atomic(OUT / 'queue.json', queue)


def release(gpu):
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
    path = OUT / 'technical_attempts' / cfg['id'] / f'{ck(case)}.json'
    with lock(OUT / 'attempts.lock'):
        previous = json.loads(path.read_text()) if path.exists() else {}
        count = previous.get('attempts', 0)
        if previous.get('fatal') or count >= MAX_TECHNICAL_ATTEMPTS:
            raise RuntimeError(f"technical retry exhausted or fatal: {cfg['id']} {ck(case)}")
        atomic(path, dict(attempts=count + 1, time=time.time()))
    return count + 1


def record_technical_failure(cfg, case, error, fatal=False):
    path = OUT / 'technical_attempts' / cfg['id'] / f'{ck(case)}.json'
    with lock(OUT / 'attempts.lock'):
        row = json.loads(path.read_text())
        row.update(error=repr(error), fatal=bool(fatal), failed=time.time())
        atomic(path, row)
    return not fatal and row['attempts'] < MAX_TECHNICAL_ATTEMPTS


def pending():
    return any(group['state'] != 'complete' for group in json.loads((OUT / 'queue.json').read_text())['groups'])

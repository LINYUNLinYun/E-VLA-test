import json
import time
from pathlib import Path
from preparations.idea1_v3.io import atomic, digest
from preparations.idea1_v4_0.common import conditions as old_conditions, key, sha

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/idea1_v4_1_warmstart'
OLD = ROOT/'results/idea1_v4_0'
MODULE = 'preparations.idea1_v4_1_warmstart'
PLAN = ROOT/'docs/experiments/idea1/idea1_v4_1_warmstart_plan.md'


def directory(condition, case):
    return OUT/'episodes'/condition/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"


def grid():
    result = []
    for steps in (1, 2):
        for start in (.05, .10, .20, .35, .50, .75):
            for noise in (0., .5, 1.):
                name = f'kf_d{steps}_t{round(start*100):02d}_n{round(noise*100):03d}'
                result.append(dict(id=name, fast_steps=steps, start_time=start, noise_scale=noise))
    return result


def config(candidate):
    base = dict(old_conditions()[1])
    base.update(candidate, initial='kalman', warmstart=True, fast_time_embedding='openpi_linspace')
    return base


def control():
    return dict(old_conditions()[1], id='v12_d2_time_fixed_control', warmstart=False,
        fast_time_embedding='openpi_linspace')


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    old = json.loads((OLD/'manifest.json').read_text())
    cases = old['cases']
    tune = [c for c in cases if c['task'] in (0, 2)]
    confirm = [c for c in cases if c not in tune]
    bank = []
    sources = {}
    for case in tune:
        d = OLD/'episodes/v12_d2_control'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
        rows = [json.loads(s) for s in (d/'infer.jsonl').read_text().splitlines()]
        chosen = [r['replan'] for r in rows if r['route']['draft_attempted'] and r.get('input_path')][:6]
        assert chosen
        bank.append(dict(case=case, directory=str(d), requests=chosen))
        for file in ('identity.json', 'infer.jsonl', 'telemetry.jsonl', 'arrays.npz', 'complete.json'):
            sources[str(d/file)] = sha(d/file)
    own = {str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    for name in ('policy.py','sampling.py','rules.py','selection.py','rollout.py','common.py'):
        p = ROOT/'preparations/idea1_v4_0'/name
        own[str(p.relative_to(ROOT))] = sha(p)
    m = dict(cases=cases, tune_cases=tune, confirm_cases=confirm, grid=grid(), replay_bank=bank,
        replay_samples=sum(len(b['requests']) for b in bank), source_files=own, replay_sources=sources,
        old_manifest_sha256=sha(OLD/'manifest.json'), plan_sha256=sha(PLAN),
        phase_rule=old['phase_rule'], pace_hz=20, env_seed=7, settling=10, max_gpus=4,
        selection=dict(replay='Rank by mean first-five decoded seven-command MSE to same-observation Full-D10; retain best two per NFE',
            closed_loop='On eight tuning cases choose best per NFE by success count, then pooled hot tail ms/action, then id',
            confirm='Freeze two champions before running remaining 32 cases; report tuning and confirmation separately'),
        initial_formula='x_start=(1-start_time)*KF_raw + noise_scale*start_time*explicit_gaussian; dt=-start_time/NFE',
        reference='Fresh time-corrected Gaussian D2 on all40 cases; previous Full40 and legacy D2_40 are historical references',
        fast_time_embedding='OpenPI/D10-export linspace(0,1,512); previous custom D1/D2 used arange(0,1024,2)/1024',
        pre_gpu_review=dict(path=str(OUT/'attempts/pre_gpu_time_embedding_review'),
            no_gpu_results_existed=True,corrected_before_parameter_selection=True),
        unchanged='V12 skip6, nonadjacent layers1..17, original four projections and layer-RMS sum, every actual Full refresh, period3, residual extrapolation, verifier .10/.05 tau.10 gripper protection',
        graphs='Only selector retains CUDA Graph; encoder, AE and verifier stay eager',
        timestamp=time.time())
    m['manifest_hash'] = digest(m)
    atomic(path, m)
    return m


def read():
    return json.loads((OUT/'manifest.json').read_text())


def stage_cases(stage, manifest):
    return manifest['tune_cases' if stage == 'tune' else 'confirm_cases']


def stage_configs(stage):
    filename = 'replay_selection.json' if stage == 'tune' else 'champions.json'
    chosen = json.loads((OUT/filename).read_text())['candidates']
    configs = [config(c) for c in chosen]
    return [control()]+configs


def stage_progress(stage, manifest):
    if not (OUT/('replay_selection.json' if stage == 'tune' else 'champions.json')).exists():
        return dict(completed=0, target=None)
    configs, cases = stage_configs(stage), stage_cases(stage, manifest)
    counts = {cfg['id']:dict(completed=0, success=0, target=len(cases)) for cfg in configs}
    for cfg in configs:
        for case in cases:
            p = directory(cfg['id'], case)
            if (p/'complete.json').exists():
                r = json.loads((p/'result.json').read_text())
                assert r['valid'] and r['manifest_hash']==manifest['manifest_hash']
                counts[cfg['id']]['completed'] += 1
                counts[cfg['id']]['success'] += int(r['success'])
    return dict(completed=sum(v['completed'] for v in counts.values()), target=len(configs)*len(cases), conditions=counts)

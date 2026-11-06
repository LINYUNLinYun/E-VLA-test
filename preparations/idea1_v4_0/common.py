"""Small frozen matrix and lightweight progress for exactly three experiments."""
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import numpy as np
from preparations.idea1_v3.io import atomic, digest
from preparations.idea1_v3_1.common import SUITES
from .rules import motion_features, skip_budget

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/idea1_v4_0'
MODULE = 'preparations.idea1_v4_0'
PLAN = ROOT/'docs/experiments/idea1/idea1_v4_0_plan.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(case):
    return f"{case['suite']}_task{case['task']:02d}_init{case['init']:02d}"


def directory(condition, case):
    return OUT/'episodes'/condition/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"


def conditions():
    base = dict(mode='verified', period=3, budget=12, skip_count=6, fast_steps=2,
        scoring='original', initial='gaussian', dynamic=False, n=40,
        projections=4, compensation='extrapolate', selection='jacobian',
        selection_stride=1, batched=True, aggregation='layer_rms_sum', asynchronous=True,
        attention='original', cuda_graph=True, tau=.10, protect_gripper=True,
        model='pi0', H=50, R=5, full_steps=10, native_inference_graph=False,
        verify_graph=False, gradient_backend='rms_rope_glu')
    return [dict(base, id='full_d10', mode='full', period=1, compensation='none', selection='none'),
        dict(base, id='v12_d2_control'),
        dict(base, id='v12_d1_gaussian', fast_steps=1),
        dict(base, id='v12_d2_kalman', initial='kalman'),
        dict(base, id='v12_d1_kalman', initial='kalman', fast_steps=1),
        dict(base, id='v12_d2_error_score', scoring='compensation_error'),
        dict(base, id='dynamic_skip5_8_d2', dynamic=True)]


def calibrate():
    source = ROOT/'results/idea1_v3_4_selection_async/closed_loop/full_d10/episodes'
    samples, provenance = [], []
    for suite in SUITES:
        for task in (0, 2):
            path = source/suite/f'task{task:02d}'/'init19'/'telemetry.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            commands, poses = [], []
            for row in rows:
                if row['action'] is None:
                    continue
                command = np.asarray(row['action'], np.float64)
                command[:6] = np.clip(command[:6], -1, 1)
                commands.append(command)
                poses.append(row['eef_quat'])
                if len(commands) % 5 == 0 and len(commands) >= 5:
                    f = motion_features(commands, poses)
                    if not f['gripper']:
                        samples.append(f)
            provenance.append(dict(path=str(path), sha256=sha(path)))
    if len(samples) < 20:
        raise ValueError('Insufficient historical causal motion windows')
    risk = np.asarray([f['risk'] for f in samples])
    thresholds = dict(zip(('q25', 'q50', 'q85'), map(float, np.quantile(risk, [.25, .50, .85]))))
    distribution = Counter(skip_budget(f, thresholds)[0] for f in samples)
    if distribution[7]+distribution[8] < len(samples)*.25:
        raise ValueError('Dynamic rule degenerates before evaluation')
    return dict(thresholds=thresholds, calibration_windows=len(samples),
        calibration_skip_distribution=dict(distribution), sources=provenance,
        recipe='q25/q50/q85 of past-five-command decision-machine risk on eight prior Full trajectories; gripper change forces skip5',
        disclosure='Historical explored cases; no outcomes used to select thresholds; not a held-out quality benchmark')


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    cases = [dict(suite=s, task=t, init=19) for t in (0, 2, 1, 3, 4, 5, 6, 7, 8, 9) for s in SUITES]
    source = {str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    imports = ('preparations/idea1_v3_4_selection_async/core.py',
        'preparations/idea1_v3_4_selection_async/rollout.py',
        'preparations/idea1_v3_4_selection_async/runtime.py',
        'preparations/idea1_v2_2/pi0_adapter.py', 'preparations/idea1_v3/policy.py',
        'preparations/idea1_v3_4_E2_comp_error_cost/model.py',
        'reports/idea1_v3_4_selection_async_batched_attention.py',
        'reports/idea1_v3_4_selection_async_rms_backward.py',
        'reports/idea1_v3_4_selection_async_elementwise.py',
        'preparations/idea1_v3_3/kalman_frozen.json',
        'policies/realtime-vla-flash/scripts/idea1_v2_1/filter.py')
    source.update({name:sha(ROOT/name) for name in imports})
    manifest = dict(cases=cases, conditions=conditions(), experiments=dict(
        E1=['v12_d2_control','v12_d1_gaussian','v12_d2_kalman','v12_d1_kalman'],
        E2=['v12_d2_control','v12_d2_error_score'],
        E3=['v12_d2_control','dynamic_skip5_8_d2']),
        source_files=source, idea1_sha256=sha(ROOT/'idea1.md'), plan_sha256=sha(PLAN),
        phase_rule=calibrate(), pace_hz=20, env_seed=7, settling=10,
        target_episodes=280, max_gpus=4, case_count=40,
        kernel_policy='Native encoder, AE and verifier stay eager; selector fused four-direction CUDA Graph retained',
        kalman_policy='Direct deterministic Kalman initial state at original native D1/D2 time nodes; xyz filter frozen, rotation/gripper held, unused 25 dimensions zero; no shortened-time or added-noise branch',
        timestamp=time.time())
    manifest['manifest_hash'] = digest(manifest)
    atomic(path, manifest)
    return manifest


def progress(manifest=None):
    m = manifest or json.loads((OUT/'manifest.json').read_text())
    result = {}
    for cfg in m['conditions']:
        completed, successes = 0, 0
        for case in m['cases']:
            path = directory(cfg['id'], case)
            if (path/'complete.json').exists():
                row = json.loads((path/'result.json').read_text())
                if not row['valid']:
                    raise ValueError(f'Invalid completed episode: {path}')
                completed += 1
                successes += int(row['success'])
        result[cfg['id']] = dict(completed=completed, success=successes, target=40)
    return result

"""Supplement independent baseline replay and grouped diagnostics after main run."""
from collections import Counter
import csv
import json
import time
import numpy as np
try:
    from .predictor_experiment import (ROOT, OLD, OUT, SOURCE, SUITES, Config, Kalman, CommandKalman,
                                      Parameters, sha, save, table, macro)
except ImportError:
    from predictor_experiment import (ROOT, OLD, OUT, SOURCE, SUITES, Config, Kalman, CommandKalman,
                                      Parameters, sha, save, table, macro)


def run():
    if not (OUT/'completion.json').is_file():
        raise RuntimeError('Main experiment must finish first')
    # Preserve the exact original execution source after import-compatibility
    # edits. Reconstruct only that known import edit, then verify original hash.
    manifest = json.loads((OUT/'manifest.json').read_text())
    snapshot = OUT/'source_snapshot'; snapshot.mkdir(exist_ok=True)
    snapshot_checks = {}
    for relative, expected in manifest['code_hashes'].items():
        p = ROOT/relative; content = p.read_text()
        if p.name == 'predictor_experiment.py':
            content = content.replace('try:\n    from .predictor import Config, candidates, predict_history\nexcept ImportError:\n    from predictor import Config, candidates, predict_history',
                                      'from predictor import Config, candidates, predict_history')
        actual = __import__('hashlib').sha256(content.encode()).hexdigest()
        if actual != expected:
            raise ValueError('Cannot reconstruct original source snapshot')
        (snapshot/p.name).write_text(content)
        snapshot_checks[p.name] = actual
    winner = json.loads((OLD/'winner.json').read_text())
    old_parameters = json.loads((OLD/'reproduction.json').read_text())['old_parameters']
    meta = json.loads((OUT/'records.json').read_text())
    arrays = np.load(OUT/'predictions.npz', allow_pickle=False)
    sample_checks = []
    for suite in SUITES:
        path = SOURCE/'shadow_supplement_v14/round2_20261011/V14_A03'/suite/'predictions.jsonl'
        source_records = [r for r in map(json.loads, path.read_text().splitlines()) if r['task_id'] == 1 and r['episode_idx'] == 0]
        source_records.sort(key=lambda r: r['environment_action_step'])
        telemetry_path = __import__('pathlib').Path(source_records[0]['full_infer_path']).with_name('telemetry.jsonl')
        telemetry = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
        actions = np.asarray([r['action'] for r in telemetry[1:]])
        old = Kalman(old_parameters['q0'], old_parameters['r0'], old_parameters['dt'])
        frozen = CommandKalman(Parameters(**winner['parameters']), winner['shape_q'], winner['shape_r'])
        indices = [i for i, m in enumerate(meta) if m['suite'] == suite and m['task'] == 1 and m['init'] == 0]
        previous, oldmax, newmax, holdmax = 0, 0., 0., 0.
        for r, i in zip(source_records, indices):
            t = int(r['environment_action_step'])
            for action in actions[previous:t]:
                old.observe(action); frozen.observe(action)
            previous = t
            oldmean = old.forecast()[0]
            frozenmean = frozen.forecast()[0]
            hold = np.tile(np.clip(actions[t-1, :3], -1, 1) if t else np.zeros(3), (50, 1))
            oldmax = max(oldmax, float(np.max(np.abs(oldmean-arrays['OldKalman'][i]))))
            newmax = max(newmax, float(np.max(np.abs(np.clip(frozenmean, -1, 1)-arrays['KalmanFrozen'][i]))))
            holdmax = max(holdmax, float(np.max(np.abs(hold-arrays['HoldLast'][i]))))
        if max(oldmax, newmax, holdmax) > 1e-9:
            raise ValueError('Independent frozen baseline replay failed')
        sample_checks.append(dict(suite=suite, task=1, init=0, replans=len(indices),
                                  old_kalman_max_abs=oldmax, frozen_kalman_max_abs=newmax, hold_max_abs=holdmax,
                                  source_telemetry_sha256=sha(telemetry_path)))
    innovations = []
    for method in ('WLP', 'ConstantTurn', 'VAR', 'KalmanFrozen'):
        a = arrays[method+'_innovation']*50
        norm = np.linalg.norm(a, axis=1)
        valid = np.isfinite(norm)
        innovations.append(dict(method=method, defined_replans=int(valid.sum()),
                                mean_norm_proxy_mm=macro(norm, meta),
                                norm_p95_proxy_mm=float(np.percentile(norm[valid], 95)),
                                definition='Last already-observed command minus its prior one-step prediction; frozen Kalman original innovation preserved'))
    time_rows = list(csv.DictReader((OUT/'time_shift_oracle.csv').open()))
    shift_summary = []
    for method in sorted({r['method'] for r in time_rows}):
        for phase in ['ALL']+sorted({m['phase'] for m in meta}):
            rows = [r for r in time_rows if r['method'] == method and (phase == 'ALL' or r['phase'] == phase)]
            mm = [dict(suite=r['suite'], task=int(r['task']), init=int(r['init'])) for r in rows]
            shifts = [int(r['best_shift']) for r in rows]
            shift_summary.append(dict(method=method, phase=phase, replans=len(rows),
                                      zero_macro_mm=macro(np.array([float(r['zero_error_mm']) for r in rows]), mm),
                                      oracle_residual_macro_mm=macro(np.array([float(r['residual_mm']) for r in rows]), mm),
                                      oracle_improvement_macro_mm=macro(np.array([float(r['improvement_mm']) for r in rows]), mm),
                                      unidentifiable_fraction=sum(r['unidentifiable'] == 'True' for r in rows)/len(rows),
                                      shift_histogram=json.dumps(dict(sorted(Counter(shifts).items())))))
    table(OUT/'innovation_summary.csv', innovations)
    table(OUT/'time_shift_summary.csv', shift_summary)
    with (OUT/'record_windows.csv').open() as stream:
        records = list(csv.DictReader(stream))
    suite_rows = []
    for method in sorted({r['method'] for r in records}):
        for window in ('1_5', '6_12', '13_50'):
            for suite in SUITES:
                rs = [r for r in records if r['method'] == method and r['window'] == window and r['suite'] == suite]
                mm = [dict(suite=r['suite'], task=int(r['task']), init=int(r['init'])) for r in rs]
                suite_rows.append(dict(method=method, window=window, suite=suite,
                                       l2_macro_mm=macro(np.array([float(r['l2_mm']) for r in rs]), mm), replans=len(rs)))
    table(OUT/'suites.csv', suite_rows)
    audit = dict(status='passed', replay_samples=sample_checks, original_source_snapshot_sha256=snapshot_checks,
                 sample_scope='4 complete episodes, deterministic suite/task1/init0, all replans/all50 horizons',
                 no_future_interface='predict_history takes only executed actions/cutoff/config/horizon; unit future mutation checks all51 configs',
                 source_code_sha256={str(p.relative_to(ROOT)): sha(p) for p in __import__('pathlib').Path(__file__).parent.glob('*predictor*.py')},
                 innovations=innovations)
    save(OUT/'supplement_audit.json', audit)
    completion = json.loads((OUT/'completion.json').read_text())
    completion['supplement_audit'] = 'passed'
    completion['outputs'] = {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'completion.json'}
    save(OUT/'completion.json', completion)
    print(json.dumps(dict(status='passed', baseline_replay=sample_checks, innovations=innovations), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()

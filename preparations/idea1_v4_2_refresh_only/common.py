import hashlib
import json
import time
from pathlib import Path
from preparations.idea1_v3.io import atomic, digest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/idea1_v4_2_refresh_only'
MODULE = 'preparations.idea1_v4_2_refresh_only'
PLAN = ROOT/'docs/experiments/idea1/idea1_v4_2_refresh_only_plan.md'
REPORT = ROOT/'docs/experiments/idea1/idea1_v4_2_refresh_only_report.md'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
MASK = (2, 3, 5, 14, 16, 17)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(case):
    return f"{case['suite']}_task{case['task']:02d}_init{case['init']:02d}"


def directory(name, case):
    return OUT/name/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"


def conditions():
    base = dict(model='pi0', H=50, R=5, D=2, skip=list(MASK), refresh_interval=3,
                full_ae=False, verifier=False, fast_time_embedding='openpi_linspace')
    return [dict(base, id='v12_d2_only', compensation=None),
            dict(base, id='v12_d2_refresh_extrapolate', compensation='extrapolate')]


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    sources = list(Path(__file__).parent.glob('*.py'))
    sources += [ROOT/p for p in (
        'preparations/idea1_v3_4_v12_compensation/backend.py',
        'preparations/idea1_v3/policy.py', 'preparations/idea1_v3/safe_qk.py',
        'preparations/idea1_v4_1_warmstart/sampling.py',
        'preparations/idea1_v4_1_warmstart/schedule.py',
        'preparations/idea1_v3_4/rollout.py', 'preparations/idea1_v3_1/rollout.py',
        'preparations/idea1_v2_2/pi0_adapter.py')]
    manifest = dict(conditions=conditions(),
        cases=[dict(suite=s, task=t, init=19) for t in range(10) for s in SUITES],
        source_files={str(p.relative_to(ROOT)):sha(p) for p in sources},
        plan_sha256=sha(PLAN), target_episodes=80, max_gpus=2,
        env_seed=7, settling=10, control_hz=20,
        refresh='At replans 0,3,6,... run full VLM only, store own latest two true residual sets; then rerun V12 and execute its D2 actions',
        extrapolation='Fresh refresh has ratio0; other rounds use actual age/gap; hold before second refresh',
        time_embedding='Both arms use corrected OpenPI linspace frequencies; previous custom-D2 results are historical only',
        graphs='No encoder, AE, verifier or selector CUDA Graph',
        timing='Refresh includes a second vision/VLM preparation; all actual work is counted; no deployment speedup claim from this quick quality test',
        started=time.time())
    manifest['manifest_hash'] = digest(manifest)
    atomic(path, manifest)
    return manifest


def progress(manifest):
    result = {}
    for cfg in manifest['conditions']:
        rows = []
        for case in manifest['cases']:
            d = directory(cfg['id'], case)
            if (d/'complete.json').exists():
                row = json.loads((d/'result.json').read_text())
                assert row['valid'] and row['manifest_hash'] == manifest['manifest_hash']
                rows.append(row)
        result[cfg['id']] = dict(completed=len(rows), target=40, success=sum(r['success'] for r in rows))
    return result

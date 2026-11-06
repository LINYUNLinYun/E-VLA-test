import json
import time
from pathlib import Path
from preparations.idea1_v3.io import atomic, digest
from preparations.idea1_v4_2_refresh_only.common import ROOT, MASK, SUITES, sha, key

OUT = ROOT/'results/idea1_v4_4_one_step'
MODULE = 'preparations.idea1_v4_4_one_step'
PLAN = ROOT/'docs/experiments/idea1/idea1_v4_4_one_step_plan.md'
REPORT = ROOT/'docs/experiments/idea1/idea1_v4_4_one_step_report.md'
OLD = ROOT/'results/idea1_v4_2_refresh_only'


def directory(name, case):
    return OUT/name/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"


def conditions():
    base = dict(model='pi0', H=50, R=5, D=1, skip=list(MASK), refresh_interval=3,
                full_ae=False, verifier=False, compensation='extrapolate',
                fast_time_embedding='openpi_linspace')
    return [dict(base, id='v12_d1_gaussian_refresh', initial='gaussian', start_time=1., noise_scale=1., kalman_dimensions=0),
            dict(base, id='v12_d1_kf6_refresh', initial='kalman', start_time=.75, noise_scale=1., kalman_dimensions=6)]


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    sources = list(Path(__file__).parent.glob('*.py'))
    sources += [ROOT/p for p in (
        'preparations/idea1_v4_2_refresh_only/common.py',
        'preparations/idea1_v3_4_v12_compensation/backend.py',
        'preparations/idea1_v3/policy.py', 'preparations/idea1_v3/safe_qk.py',
        'preparations/idea1_v3/history.py',
        'preparations/idea1_v4_1_warmstart/sampling.py',
        'preparations/idea1_v4_1_warmstart/schedule.py',
        'preparations/idea1_v4_3_kalman6/rules.py',
        'preparations/idea1_v4_0/rules.py',
        'preparations/idea1_v3_3/kalman_frozen.json',
        'policies/realtime-vla-flash/scripts/idea1_v2_1/filter.py',
        'preparations/idea1_v3_4/rollout.py', 'preparations/idea1_v3_1/rollout.py',
        'preparations/idea1_v2_2/pi0_adapter.py')]
    old_manifest = json.loads((OLD/'manifest.json').read_text())
    assert json.loads((OLD/'complete.json').read_text())['passed']
    reference_files = {}
    for case in old_manifest['cases']:
        d = OLD/'v12_d2_refresh_extrapolate'/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
        for name in ('result.json', 'identity.json', 'complete.json'):
            reference_files[str(d/name)] = sha(d/name)
    manifest = dict(conditions=conditions(), cases=old_manifest['cases'],
        source_files={str(p.relative_to(ROOT)):sha(p) for p in sources},
        reference_files=reference_files, reference_manifest_hash=old_manifest['manifest_hash'],
        reference_id='v12_d2_refresh_extrapolate', reference_root=str(OLD),
        plan_sha256=sha(PLAN), target_episodes=80, max_gpus=4,
        env_seed=7, settling=10, control_hz=20, wall_pacing=False,
        bootstrap='Both conditions use Gaussian D1 until actual commands are acknowledged; only first request in normal rollout',
        refresh='Full VLM only at replans0/3/6; then V12+D1 always; no Full AE, verification, gripper protection or fallback',
        extrapolation='Own latest two histories, actual age/gap; fresh ratio0, first cycle hold',
        mask_source='Same fixed V12 subpath as v4.2; no dynamic selection or gradient computation',
        graphs='No encoder, AE, verifier or selector CUDA Graph',
        parameter_source='Frozen v4.3 six-dimensional D1 champion tau=.75 rho=1; no new search',
        timing='Serial CPU predictor is counted but is not claimed to be an unavoidable GPU/deployment cost',
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

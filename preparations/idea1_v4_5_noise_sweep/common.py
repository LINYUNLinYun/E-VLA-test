import json
import time
from pathlib import Path
from preparations.idea1_v3.io import atomic, digest
from preparations.idea1_v4_2_refresh_only.common import ROOT, MASK, SUITES, sha, key
from preparations.idea1_v4_4_one_step.common import conditions as previous_conditions

OUT = ROOT/'results/idea1_v4_5_noise_sweep'
MODULE = 'preparations.idea1_v4_5_noise_sweep'
PLAN = ROOT/'docs/experiments/idea1/idea1_v4_5_noise_sweep_plan.md'
REPORT = ROOT/'docs/experiments/idea1/idea1_v4_5_noise_sweep_report.md'
OLD = ROOT/'results/idea1_v4_4_one_step'
TAUS = (.05, .10, .20, .35, .50, .75)


def all_conditions():
    base = previous_conditions()[1]
    return [dict(base, id=f'v12_d1_kf6_t{round(tau*100):02d}_refresh', start_time=tau) for tau in TAUS]


def conditions():
    return [cfg for cfg in all_conditions() if cfg['start_time'] != .75]


def allowed_conditions():
    return [previous_conditions()[0]]+all_conditions()


def directory(name, case):
    return OUT/name/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/'manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    old_manifest = json.loads((OLD/'manifest.json').read_text())
    assert json.loads((OLD/'complete.json').read_text())['passed']
    assert all(sha(ROOT/p)==h for p,h in old_manifest['source_files'].items())
    d2root = ROOT/'results/idea1_v4_2_refresh_only'
    d2manifest = json.loads((d2root/'manifest.json').read_text())
    assert json.loads((d2root/'complete.json').read_text())['passed']
    references = []
    for kind, root, original, canonical, parent in (
        ('gaussian_d1', OLD, 'v12_d1_gaussian_refresh', 'v12_d1_gaussian_refresh', old_manifest),
        ('kalman_d1', OLD, 'v12_d1_kf6_refresh', 'v12_d1_kf6_t75_refresh', old_manifest),
        ('gaussian_d2', d2root, 'v12_d2_refresh_extrapolate', 'v12_d2_refresh_extrapolate', d2manifest)):
        assert parent['cases']==old_manifest['cases']
        cfg = next(c for c in parent['conditions'] if c['id']==original)
        references.append(dict(kind=kind, root=str(root), source_id=original, id=canonical,
            config=cfg, manifest_hash=parent['manifest_hash']))
    reference_files = {}
    for ref in references:
        for case in old_manifest['cases']:
            d=Path(ref['root'])/ref['source_id']/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
            for name in ('result.json', 'identity.json', 'complete.json', 'infer.jsonl', 'arrays.npz'):
                reference_files[str(d/name)] = sha(d/name)
    reference_files[str(OLD/'cross_service_reference.npz')]=sha(OLD/'cross_service_reference.npz')
    sources = list(Path(__file__).parent.glob('*.py'))
    dependencies = set(old_manifest['source_files'])
    dependencies.update(('preparations/idea1_v4_4_one_step/policy.py',
        'preparations/idea1_v4_4_one_step/common.py',
        'preparations/idea1_v4_4_one_step/rollout.py'))
    sources += [ROOT/p for p in sorted(dependencies)]
    manifest = dict(conditions=conditions(), all_conditions=all_conditions(), cases=old_manifest['cases'],
        source_files={str(p.relative_to(ROOT)):sha(p) for p in sources}, references=references,
        reference_files=reference_files, reference_manifest_hash=old_manifest['manifest_hash'],
        plan_sha256=sha(PLAN), target_episodes=200, reused_episodes=120, max_gpus=4,
        env_seed=7, settling=10, control_hz=20, wall_pacing=False,
        tau_grid=list(TAUS), noise_scale=1., actual_noise_std='sigma=tau with rho=1',
        scientific_scope='Joint start-time/Kalman-weight/noise-scale/one-step-interval sweep; not a noise-only causal comparison',
        bootstrap='Every condition starts with identical Gaussian D1; only acknowledged executed commands enter Kalman thereafter',
        refresh=old_manifest['refresh'], extrapolation=old_manifest['extrapolation'],
        mask_source=old_manifest['mask_source'], graphs=old_manifest['graphs'], timing=old_manifest['timing'],
        reference_matching='Retain all valid success outcomes. First input/init/noise/raw original atol=rtol=1e-5 defines eligibility for strict historical paired comparisons; disclose actual matched n without tolerance changes or reruns',
        scheduling='One poller every30s, stable idle60s then UUID cooperative lease; max4 persistent B1 services; effective failures retained, technical recovery at most once',
        started=time.time())
    manifest['manifest_hash']=digest(manifest)
    atomic(path, manifest)
    return manifest


def progress(manifest):
    result = {}
    for cfg in manifest['conditions']:
        rows = []
        for case in manifest['cases']:
            d=directory(cfg['id'],case)
            if (d/'complete.json').exists():
                row=json.loads((d/'result.json').read_text())
                assert row['valid'] and row['manifest_hash']==manifest['manifest_hash']
                rows.append(row)
        result[cfg['id']]=dict(completed=len(rows), target=40, success=sum(r['success'] for r in rows))
    return result

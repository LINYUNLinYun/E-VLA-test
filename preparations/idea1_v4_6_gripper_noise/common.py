import json
import time
from pathlib import Path
from preparations.idea1_v3.io import atomic, digest
from preparations.idea1_v4_2_refresh_only.common import ROOT, MASK, SUITES, sha, key
from preparations.idea1_v4_5_noise_sweep.common import all_conditions as old_kalman_conditions
from preparations.idea1_v4_4_one_step.common import conditions as old_controls

OUT=ROOT/'results/idea1_v4_6_gripper_noise'
MODULE='preparations.idea1_v4_6_gripper_noise'
PLAN=ROOT/'docs/experiments/idea1/idea1_v4_6_gripper_noise_plan.md'
REPORT=ROOT/'docs/experiments/idea1/idea1_v4_6_gripper_noise_report.md'
OLD=ROOT/'results/idea1_v4_5_noise_sweep'
TAUS=(.05,.10)


def hold_conditions():
    return [cfg for cfg in old_kalman_conditions() if cfg['start_time'] in TAUS]


def conditions():
    return [dict(cfg,id=f"v12_d1_kf6_t{round(cfg['start_time']*100):02d}_gripper_noise",
        gripper_initial='standard_gaussian',gripper_noise_std=1.) for cfg in hold_conditions()]


def allowed_conditions():
    return [old_controls()[0]]+hold_conditions()+conditions()


def directory(name,case):
    return OUT/name/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'manifest.json'
    if path.exists():return json.loads(path.read_text())
    parent=json.loads((OLD/'manifest.json').read_text())
    assert json.loads((OLD/'complete.json').read_text())['passed']
    assert all(sha(ROOT/p)==h for p,h in parent['source_files'].items())
    cases=[c for c in parent['cases'] if c['task'] in (0,2)]
    assert len(cases)==8
    references=[]
    for cfg in hold_conditions():
        references.append(dict(id=cfg['id'],kind='hold_gripper',root=str(OLD),source_id=cfg['id'],
            config=cfg,manifest_hash=parent['manifest_hash']))
    g_root=ROOT/'results/idea1_v4_4_one_step'
    g_manifest=json.loads((g_root/'manifest.json').read_text())
    g_cfg=old_controls()[0]
    references.append(dict(id=g_cfg['id'],kind='gaussian_d1',root=str(g_root),source_id=g_cfg['id'],
        config=g_cfg,manifest_hash=g_manifest['manifest_hash']))
    reference_files={str(OLD/'cross_service_reference.npz'):sha(OLD/'cross_service_reference.npz')}
    for ref in references:
        for case in cases:
            d=Path(ref['root'])/ref['source_id']/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
            for name in ('result.json','identity.json','complete.json','infer.jsonl','arrays.npz'):
                reference_files[str(d/name)]=sha(d/name)
    sources=list(Path(__file__).parent.glob('*.py'))
    dependencies=set(parent['source_files'])|{'preparations/idea1_v4_5_noise_sweep/common.py','preparations/idea1_v4_5_noise_sweep/policy.py','preparations/idea1_v4_5_noise_sweep/finish.py'}
    sources.extend(ROOT/p for p in sorted(dependencies))
    manifest=dict(conditions=conditions(),cases=cases,target_episodes=16,reused_episodes=24,max_gpus=2,
        references=references,reference_files=reference_files,source_files={str(p.relative_to(ROOT)):sha(p) for p in sources},
        parent_manifest_hash=parent['manifest_hash'],plan_sha256=sha(PLAN),tau_grid=list(TAUS),
        env_seed=7,settling=10,control_hz=20,wall_pacing=False,
        initial='Same normalized x_tau=(1-tau)*KF+tau*epsilon for motion0:6 and padding7:; overwrite only raw gripper dim6 with the same epsilon6, unit standard deviation and zero old-gripper coefficient',
        global_time='A single original AE time condition tau and Euler dt=-tau for all dimensions; gripper sigma1 at low tau is an intentional heterogeneous-noise heuristic, not a guaranteed matched training-flow distribution',
        bootstrap=parent['bootstrap'],refresh=parent['refresh'],extrapolation=parent['extrapolation'],mask_source=parent['mask_source'],graphs=parent['graphs'],timing=parent['timing'],
        scope='Quick fixed8case check per low tau to test escape from zero; no automatic full40 expansion or tau retuning; grip switching alone is not task success',
        scheduling=parent['scheduling'],reference_matching=parent['reference_matching'],started=time.time())
    manifest['manifest_hash']=digest(manifest);atomic(path,manifest);return manifest


def progress(manifest):
    result={}
    for cfg in manifest['conditions']:
        rows=[]
        for case in manifest['cases']:
            d=directory(cfg['id'],case)
            if (d/'complete.json').exists():
                r=json.loads((d/'result.json').read_text());assert r['valid'] and r['manifest_hash']==manifest['manifest_hash'];rows.append(r)
        result[cfg['id']]=dict(completed=len(rows),target=len(manifest['cases']),success=sum(r['success'] for r in rows))
    return result

import json
import time
from pathlib import Path
from preparations.idea1_v3.io import atomic, digest
from preparations.idea1_v4_2_refresh_only.common import ROOT, MASK, SUITES, sha, key
from preparations.idea1_v4_4_one_step.common import conditions as old_controls
from preparations.idea1_v4_5_noise_sweep.common import all_conditions as old_kalman_conditions
from preparations.idea1_v4_6_gripper_noise.common import conditions as gripper_noise_conditions

OUT=ROOT/'results/idea1_v4_7_future_prior'
MODULE='preparations.idea1_v4_7_future_prior'
PLAN=ROOT/'docs/experiments/idea1/idea1_v4_7_future_prior_plan.md'
REPORT=ROOT/'docs/experiments/idea1/idea1_v4_7_future_prior_report.md'
OLD=ROOT/'results/idea1_v4_6_gripper_noise'


def conditions():
    base=dict(old_controls()[0],start_time=.1,noise_scale=1.,gripper_initial='future_chunk')
    return [dict(base,id='v12_d1_kf6_future_grip_t10',initial='kalman',kalman_dimensions=6,prior_source='kalman6_future_grip'),
            dict(base,id='v12_d1_shift_chunk_t10',initial='shifted_chunk',kalman_dimensions=0,prior_source='shifted_chunk')]


def directory(name,case):
    return OUT/name/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'manifest.json'
    if path.exists():return json.loads(path.read_text())
    parent=json.loads((OLD/'manifest.json').read_text())
    assert json.loads((OLD/'complete.json').read_text())['passed']
    assert all(sha(ROOT/p)==h for p,h in parent['source_files'].items())
    cases=parent['cases'];assert len(cases)==8
    hold=next(c for c in old_kalman_conditions() if c['start_time']==.1)
    noise=next(c for c in gripper_noise_conditions() if c['start_time']==.1)
    reference_specs=[(ROOT/'results/idea1_v4_5_noise_sweep',hold,'hold_gripper'),
                     (ROOT/'results/idea1_v4_4_one_step',old_controls()[0],'gaussian_d1'),
                     (OLD,noise,'unit_gripper_noise')]
    references=[];reference_files={}
    for root,cfg,kind in reference_specs:
        old_manifest=json.loads((root/'manifest.json').read_text())
        references.append(dict(id=cfg['id'],kind=kind,root=str(root),source_id=cfg['id'],config=cfg,manifest_hash=old_manifest['manifest_hash']))
        for case in cases:
            d=root/cfg['id']/'episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
            for name in ('result.json','identity.json','complete.json','infer.jsonl','arrays.npz'):
                reference_files[str(d/name)]=sha(d/name)
    sources=list(Path(__file__).parent.glob('*.py'))
    dependencies=set(parent['source_files'])|{'preparations/idea1_v4_6_gripper_noise/common.py','preparations/idea1_v4_6_gripper_noise/policy.py','preparations/idea1_v4_6_gripper_noise/finish.py'}
    sources.extend(ROOT/p for p in sorted(dependencies))
    manifest=dict(conditions=conditions(),cases=cases,target_episodes=16,reused_episodes=24,max_gpus=2,
        references=references,reference_files=reference_files,source_files={str(p.relative_to(ROOT)):sha(p) for p in sources},
        parent_manifest_hash=parent['manifest_hash'],plan_sha256=sha(PLAN),tau_grid=[.1],
        env_seed=7,settling=10,control_hz=20,wall_pacing=False,
        initial='All dimensions use x_tau=.9*encoded_prior+.1*epsilon, global t=.1 and dt=-.1. A uses six-dimensional executed-command KF plus shifted previous predicted gripper; B shifts all seven decoded predicted command dimensions. Both encode in current observation coordinates with zero25dim padding.',
        shift='Only after real ack; shift previous full H50 decoded output by actual executed r1..5; repeat last predicted command to pad tail. Unexecuted predictions are priors only, never Kalman observations. B has no extra prediction clipping; A inherits frozen KF clipping.',
        bootstrap=parent['bootstrap'],refresh=parent['refresh'],extrapolation=parent['extrapolation'],mask_source=parent['mask_source'],graphs=parent['graphs'],timing=parent['timing'],
        scope='Fixed same8case exploratory A/B; no automatic40case expansion or parameter search. Not an independent confirmation set.',
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

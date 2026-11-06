import json
import time
from pathlib import Path
from preparations.idea1_v3.io import atomic, digest
from preparations.idea1_v4_0.common import conditions as old_conditions, key, sha

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/idea1_v4_3_kalman6'
OLD = ROOT/'results/idea1_v4_1_warmstart'
MODULE = 'preparations.idea1_v4_3_kalman6'
PLAN = ROOT/'docs/experiments/idea1/idea1_v4_3_kalman6_plan.md'
REPORT = ROOT/'docs/experiments/idea1/idea1_v4_3_kalman6_report.md'


def directory(condition, case):
    return OUT/'episodes'/condition/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"


def grid():
    return [dict(id=f'kf6_d{steps}_t{round(start*100):02d}_n{round(noise*100):03d}',
                 fast_steps=steps, start_time=start, noise_scale=noise)
            for steps in (1,2) for start in (.05,.10,.20,.35,.50,.75) for noise in (0.,.5,1.)]


def anchors():
    return [dict(id='kf6_d1_t75_n100', fast_steps=1, start_time=.75, noise_scale=1.),
            dict(id='kf6_d2_t05_n100', fast_steps=2, start_time=.05, noise_scale=1.)]


def unique(candidates):
    return list({c['id']:c for c in candidates}.values())


def config(candidate):
    return dict(old_conditions()[1], **dict(candidate, initial='kalman', warmstart=True,
        fast_time_embedding='openpi_linspace', kalman_dimensions=6))


def control():
    result = dict(old_conditions()[1])
    result.update(id='gaussian_d1_time_fixed_control',fast_steps=1,warmstart=False,
        fast_time_embedding='openpi_linspace',kalman_dimensions=0)
    return result


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'manifest.json'
    if path.exists():
        return json.loads(path.read_text())
    old=json.loads((OLD/'manifest.json').read_text())
    assert json.loads((OLD/'complete.json').read_text())['valid']
    own={str(p.relative_to(ROOT)):sha(p) for p in Path(__file__).parent.glob('*.py')}
    for relative in ('preparations/idea1_v4_1_warmstart/policy.py',
        'preparations/idea1_v4_1_warmstart/sampling.py','preparations/idea1_v4_1_warmstart/schedule.py',
        'preparations/idea1_v4_0/rules.py','preparations/idea1_v4_0/selection.py',
        'preparations/idea1_v4_0/rollout.py','preparations/idea1_v3/history.py',
        'preparations/idea1_v3_3/kalman_frozen.json',
        'policies/realtime-vla-flash/scripts/idea1_v2_1/filter.py'):
        own[relative]=sha(ROOT/relative)
    references=[]
    old_configs=[dict(old_conditions()[1],id='v12_d2_time_fixed_control',warmstart=False,fast_time_embedding='openpi_linspace')]
    for anchor in anchors():
        c=dict(old_conditions()[1]);c.update(anchor,id=anchor['id'].replace('kf6_','kf_'),initial='kalman',warmstart=True,fast_time_embedding='openpi_linspace')
        old_configs.append(c)
    for cfg in old_configs:
        references.append(dict(id='previous_'+cfg['id'],config=cfg,root=str(OLD/'episodes'/cfg['id'])))
    reference_files={}
    for ref in references:
        for case in old['cases']:
            d=Path(ref['root'])/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
            for name in ('result.json','identity.json','complete.json','infer.jsonl','terminal.json','arrays.npz'):
                reference_files[str(d/name)]=sha(d/name)
    manifest=dict(cases=old['cases'],tune_cases=old['tune_cases'],confirm_cases=old['confirm_cases'],
        grid=grid(),anchors=anchors(),replay_bank=old['replay_bank'],replay_samples=old['replay_samples'],
        source_files=own,replay_sources=old['replay_sources'],reference_files=reference_files,references=references,
        plan_sha256=sha(PLAN),previous_manifest_hash=old['manifest_hash'],phase_rule=old['phase_rule'],
        pace_hz=20,env_seed=7,settling=10,max_gpus=4,max_new_episodes=216,
        kalman=dict(dimensions=6,space='clipped normalized executed xyz and rotation delta controller commands; no absolute Euler/quaternion extrapolation',
            translation='unchanged frozen 3D filter',rotation='independent 3D filter, same frozen q_u/q_w/g/beta, identity Q/R shapes',
            gripper='hold last acknowledged command',unused='25 zero prior dimensions'),
        selection=dict(replay='Same 48 observations: rank six-dimensional candidates by first-five decoded seven-command MSE; retain two per NFE and always retain matched old-parameter anchors',
            closed_loop='On eight tuning cases select one six-dimensional candidate per NFE by successes, then pooled hot policy ms/action, then ID',
            confirm='Run both fixed-parameter anchors and any distinct frozen winners on remaining32; fresh corrected Gaussian D1 on all40'),
        unchanged='V12 dynamic skip6, nonadjacent layers1..17, original four projections, period3, Full residual extrapolation, verifier .10/.05 tau.10 gripper protection',
        graphs='Only selector CUDA Graph; encoder/AE/verifier eager',timestamp=time.time())
    manifest['manifest_hash']=digest(manifest)
    atomic(path,manifest)
    return manifest


def read():
    return json.loads((OUT/'manifest.json').read_text())


def stage_cases(stage, manifest):
    return manifest['tune_cases' if stage=='tune' else 'confirm_cases']


def stage_configs(stage):
    name='replay_selection.json' if stage=='tune' else 'champions.json'
    candidates=json.loads((OUT/name).read_text())['candidates']
    if stage=='confirm':
        candidates=unique(anchors()+candidates)
    return [control()]+[config(c) for c in candidates]


def stage_progress(stage, manifest):
    name='replay_selection.json' if stage=='tune' else 'champions.json'
    if not (OUT/name).exists():
        return dict(completed=0,target=None)
    cases=stage_cases(stage,manifest)
    counts={c['id']:dict(completed=0,success=0,target=len(cases)) for c in stage_configs(stage)}
    for cfg in stage_configs(stage):
        for case in cases:
            d=directory(cfg['id'],case)
            if (d/'complete.json').exists():
                r=json.loads((d/'result.json').read_text())
                assert r['valid'] and r['manifest_hash']==manifest['manifest_hash']
                counts[cfg['id']]['completed']+=1
                counts[cfg['id']]['success']+=int(r['success'])
    return dict(completed=sum(c['completed'] for c in counts.values()),target=len(cases)*len(counts),conditions=counts)

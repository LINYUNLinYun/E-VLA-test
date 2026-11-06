"""Frozen small6-arm experiment; no GPU discovery or launch."""
import hashlib
import json
from pathlib import Path
from preparations.idea1_v3.io import atomic,digest
from preparations.idea1_v3_3.manifest import SUITES

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/idea1_e2_tuning_20261025'


def main():
    cs=[dict(id='parent',experiment='baseline',kind='parent'),dict(id='alternate',experiment='E2',kind='alternate',alternate=True),
        dict(id='old_c5',experiment='E2',kind='abc',angle=30.,change=.10,pose=5.),dict(id='relaxed_abc',experiment='E2',kind='abc',angle=45.,change=.15,pose=10.),
        dict(id='error_abs',experiment='E2',kind='metric',metric='forecast_error_rms_max5',threshold=.15,pose=5.),
        dict(id='error_relative',experiment='E2',kind='metric',metric='forecast_error_relative_max5',threshold=.50,pose=5.)]
    for c in cs:c.update(model='pi0',tau=.10)
    cases=[dict(suite=s,task=t,init=8) for t in [0,5,1,2,3,4,6,7,8,9] for s in SUITES]
    base=[0,1,5,2,4,3];schedule=[dict(case=case,condition_order=[cs[(v+i)%6]['id'] for v in base]) for i,case in enumerate(cases)]
    paths=[ROOT/'docs/experiments/idea1/idea1_e2_tuning_20261025_plan.md',OUT/'historical_analysis.json',OUT/'historical_features.jsonl',ROOT/'preparations/idea1_v3_3/kalman_frozen.json',ROOT/'policies/realtime-vla-flash/scripts/idea1_v2_1/filter.py']
    manifest=dict(status='frozen_scientific_design',conditions=cs,cases=cases,schedule=schedule,profile_case_indices=list(range(8)),budget_valid=240,technical_attempt_limit_per_episode=2,
        counterbalance='Six-arm Williams base[0,1,5,2,4,3], rotate by case index mod6.40 not divisible6: each condition occupies each position6 or7 times. Every case all6 arms share one physical GPU/service.',
        costs='Primary server predict wall+entire acknowledge wall /actual executed actions, same-case same physicalGPU. Original predict-only retained separately; RPC/env/reset/IO and common warmup/load outside primary policy wall.',
        metric='Before each frozen Kalman.observe: prior=H F_forecast x; z=clip(actual sent command XYZ,-1,1), error=||z-prior||/sqrt3. Original observe unchanged. Max last5 errors; relative denominator=max(.02,RMS of last5 actually sent XYZ commands). No future/labels.',
        common_constraints='History>=5, no recent gripper sign changes, adjacent rotation command L2<=.05; c strict< for main/new gates. Original alternate keeps its original bypass. All arms no consecutive actualKF, verify tau.1, P2, strict gripper.',
        warmup='Before EVERY arm fixed initial-observation Full10,V14D2,dual verifier and FP32 command-encode verifier preparation, then3 canonical Full10 repetitions; last2 must match bitwise. This diagnostic prep never enters the environment and is timed separately. Actual startup Full must match canonical output; initial input/full signatures must match all arms in each case.',
        environment='Fresh offscreen environment each condition, official init8, seed7,10 settling,20Hz, native limits; one B1 worker/service per authorized GPU. Avoid inherited unrecorded solver/controller state between conditions. No GPU launch until explicit resource assignment.',
        seed='Unchanged v3.3 seed(suite,task,init,replan), independent of condition/order/GPU; all new40 fixed before observation.',
        hashed_inputs=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    manifest['manifest_hash']=digest(manifest);target=OUT/'manifest.json'
    if target.exists() and json.loads(target.read_text())!=manifest:raise ValueError('Design immutable')
    atomic(target,manifest);print(target)


if __name__=='__main__':main()

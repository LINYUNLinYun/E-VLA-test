"""Freeze distinguishable c exploration candidates using parent state only."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v3.io import atomic,digest
from .manifest import cases
from .rules_c1 import gate,pose_angle
from .dispatch import ROOT,OUT


def main():
    parent=json.loads((OUT/'manifest.json').read_text()); source={}; values=[]; candidate=[]
    levels=[(10,.02),(20,.05),(30,.1)]
    stats={str(a):dict(nonFull=0,eligible=0,angles=[],c_pass={str(t):0 for t in (2,5,10)}) for a,b in levels}
    for case in cases('pi0'):
        directory=OUT/'pi0_parent/episodes'/case['suite']/f"task{case['task']:02d}"/f"init{case['init']:02d}"
        if not (directory/'complete.json').exists(): raise ValueError('Parent80 incomplete')
        paths=[directory/'telemetry.jsonl',directory/'infer.jsonl']
        for p in paths: source[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        states=[json.loads(line) for line in paths[0].read_text().splitlines()]
        requests=[json.loads(line) for line in paths[1].read_text().splitlines()]
        for end in range(5,len(states)):
            angle=pose_angle([r['eef_quat'] for r in states[end-4:end+1]])
            if angle is not None: values.append(angle)
        for r in requests:
            if 'distance' not in r['route']: continue
            step=r['step']; history=states[1:step+1]
            commands=[s['action'] for s in history]; quaternions=[s['eef_quat'] for s in history]
            for a,b in levels:
                s=stats[str(a)];s['nonFull']+=1
                if gate(commands,quaternions,a,b,None)[0]:
                    angle=pose_angle(quaternions[-5:]);s['eligible']+=1;s['angles'].append(angle)
                    for theta in (2,5,10):s['c_pass'][str(theta)]+=int(angle<theta)
    for s in stats.values():
        s['eligible_angle_percentiles_degrees']={str(q):float(np.percentile(s['angles'],q)) for q in (25,50,75,90,95)};s.pop('angles')
    plan=ROOT/'docs/experiments/idea1/idea1_v3_3_experiment_plan.md'
    receipt=dict(status='exploration_candidates_frozen; final_theta_c_undetermined',time=time.time(),candidates_degrees=[2,5,10],operator='<',quaternion='simulation world EEF body xyzw',
        reference='max_{i<j} shortest-arc angle over last5 completed control-step observations; all pairwise references, not model forecasts',
        source_parent_manifest_hash=parent['manifest_hash'],parent_cases=cases('pi0'),source_files=source,
        all_completed_windows=dict(n=len(values),percentiles_degrees={str(q):float(np.percentile(values,q)) for q in (25,50,75,90,95,99)}),
        nonFull_candidate_points=stats,selection='Retain original draft2/5/10 only after confirming meaningful separation at real parent a/b-eligible request points; no success labels or E2 results consumed. Not an optimal/deployment threshold or predicted E2 Kalman rate.',
        plan_sha256=hashlib.sha256(plan.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    out=OUT/'E2_calibration.json'
    if out.exists():raise FileExistsError(out)
    atomic(out,receipt)
    files=['rules_c1.py','server_c1.py','calibrate_e2.py','test_rules_c1.py']
    m=dict(parent_manifest_hash=parent['manifest_hash'],supersedes='Only the E2 draft definitions in the original immutable manifest; E1/E3/E4 unchanged',
        scientific_revision='strict-c1',conditions=[c for c in parent['conditions'] if c['experiment']=='E2'],cases=parent['cases'],plan_sha256=receipt['plan_sha256'],
        calibration_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),original_source_files=parent['source_files'],
        gate_source_files={str(Path(__file__).with_name(n).relative_to(ROOT)):hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in files})
    m['manifest_hash']=digest(m);atomic(OUT/'E2_manifest.json',m)
    print(json.dumps(stats,ensure_ascii=False))


if __name__=='__main__': main()

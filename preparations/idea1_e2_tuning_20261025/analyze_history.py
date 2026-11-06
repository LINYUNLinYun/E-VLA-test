"""One bounded CPU replay of old parent/c5/alternate80; no new labels used."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from preparations.idea1_v3.io import atomic,write_rows
from .features import feature_after_observe,metrics,common_blockers

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/idea1_e2_tuning_20261025'
sys.path.insert(0,str(ROOT/'policies/realtime-vla-flash'))
from scripts.idea1_v2_1.filter import Parameters,CommandKalman


def main():
    OUT.mkdir(parents=True,exist_ok=True);frozen=json.loads((ROOT/'preparations/idea1_v3_3/kalman_frozen.json').read_text());rows=[];sources=[]
    for cid in ['pi0_parent','E2_a30_b0.1_c5','E2_alternate']:
        count=0
        for p in sorted((ROOT/'results/idea1_v3_3'/cid/'episodes').glob('*/task*/init*/result.json')):
            result=json.loads(p.read_text())
            if result['case']['init'] not in (6,7) or '.invalid.' in str(p) or not p.with_name('complete.json').exists():continue
            count+=1;tele=[json.loads(l) for l in p.with_name('telemetry.jsonl').read_text().splitlines()];inf=[json.loads(l) for l in p.with_name('infer.jsonl').read_text().splitlines()]
            k=CommandKalman(Parameters(**frozen['parameters']),frozen['shape_q'],frozen['shape_r']);history=[];commands=[];quats=[];request={r['step']:r for r in inf};last=False
            for step in range(result['actions']+1):
                if step:
                    action=np.asarray(tele[step]['action']);prior=k.H@k.F_forecast@k.x
                    pp=k.F@k.P@k.F.T+k.Q;S=k.H@pp@k.H.T+k.R
                    k.observe(action);f=feature_after_observe(k,action,prior);f['nis_rms']=float(np.sqrt(k.innovation@np.linalg.solve(S,k.innovation)/3));history.append(f);commands.append(action);quats.append(tele[step]['eef_quat'])
                r=request.get(step)
                if r:
                    # Frozen parent-priority opportunities, including rejected
                    # draft->Full calls. No future command/state is inspected.
                    if 'distance' in r['route']:
                        m=metrics(commands,quats,history)
                        if m['ready']:rows.append(dict(condition=cid,case=result['case'],step=step,replan=r['replan'],historical_previous_Kalman=last,
                            common_blockers=common_blockers(commands,quats),nis_rms_max5=max(x['nis_rms'] for x in history[-5:]),**m))
                    last=r['route']['path']=='kalman_verified'
            sources.append(dict(condition=cid,case=result['case'],telemetry_path=str(p.with_name('telemetry.jsonl')),telemetry_sha256=hashlib.sha256(p.with_name('telemetry.jsonl').read_bytes()).hexdigest(),infer_sha256=hashlib.sha256(p.with_name('infer.jsonl').read_bytes()).hexdigest()))
        if count!=80:raise ValueError('Historical cohort not exact80: '+cid)
    stats={}
    for cid in sorted({r['condition'] for r in rows}):
        own=[r for r in rows if r['condition']==cid];common=[r for r in own if not r['common_blockers']]
        stats[cid]=dict(opportunities=len(own),common_safe=len(common),distributions={name:{str(q):float(np.quantile([r[name] for r in common],q)) for q in [.25,.5,.75,.8,.9,.95,.99]} for name in ['a_turn_degrees','b_command_change','c_world_pose_degrees','innovation_rms_max5','nis_rms_max5','forecast_error_rms_max5','forecast_error_relative_max5']})
    write_rows(OUT/'historical_features.jsonl',rows)
    atomic(OUT/'historical_analysis.json',dict(time=time.time(),status='CPU_only',cohorts=stats,sources=sources,
        label_use='No success labels used for parameter calibration. Only old init6/7, no new init8 data read.',
        scope='Counterfactual opportunity coverage on old histories, not expected closed-loop Kalman utilization or quality. NIS uses frozen arbitrary command covariance and is not a calibrated chi-square statistic.',
        feature_file_sha256=hashlib.sha256((OUT/'historical_features.jsonl').read_bytes()).hexdigest()))
    print(json.dumps(stats,indent=2))


if __name__=='__main__':main()

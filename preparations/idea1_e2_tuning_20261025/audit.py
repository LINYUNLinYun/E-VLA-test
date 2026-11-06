"""Single final causal replay, exact case coverage and timing audit."""
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import LIMITS
from preparations.idea1_v2_2.common import seed
from preparations.idea1_v3.policy import verification_seed
from .design import ROOT,OUT
from .analyze import records
from .features import decide,metrics,feature_after_observe
sys.path.insert(0,str(ROOT/'policies/realtime-vla-flash'))
from scripts.idea1_v2_1.filter import Parameters,CommandKalman


def main():
    m=json.loads((OUT/'manifest.json').read_text());groups=records();expected={tuple(c[k] for k in ['suite','task','init']) for c in m['cases']};frozen=json.loads((ROOT/'preparations/idea1_v3_3/kalman_frozen.json').read_text())
    rows=[];checks=[];routing={};initials=defaultdict(list)
    for cfg in m['conditions']:
        own=groups[cfg['id']]
        if len(own)!=40 or {tuple(r['case'][k] for k in ['suite','task','init']) for r in own}!=expected:raise ValueError('Exact40 coverage mismatch')
        counts=Counter();blocks=Counter()
        for result in own:
            d=Path(result['_directory']);identity=json.loads((d/'identity.json').read_text());tele=[json.loads(l) for l in (d/'telemetry.jsonl').read_text().splitlines()];inf=[json.loads(l) for l in (d/'infer.jsonl').read_text().splitlines()]
            if len(tele)!=result['actions']+1 or len(inf)!=result['predictions']:raise ValueError('Episode count alignment')
            if not result['success'] and result['actions']!=LIMITS[result['case']['suite']]:raise ValueError('Effective failure before native limit')
            binding=result['engineering_binding'];assert binding['model_gpu']==binding['render_gpu'] in [2,3]
            initials[tuple(result['case'][k] for k in ['suite','task','init'])].append((result['initial_hash'],json.dumps(result['startup_signature'],sort_keys=True),binding['model_gpu'],identity['protocol']['server_pid']))
            k=CommandKalman(Parameters(**frozen['parameters']),frozen['shape_q'],frozen['shape_r']);commands=[];q=[];history=[];step=0;last=False;draft_rounds=0
            with np.load(d/'arrays.npz') as arrays:
                for index,r in enumerate(inf):
                    assert step==r['step'];n=r['executed_length'];a=arrays[f'actions_{index}'];raw=arrays[f'raw_{index}']
                    assert r['seed']==seed(result['case']['suite'],result['case']['task'],result['case']['init'],index) and r['verification_seed']==verification_seed(r['seed'])
                    assert a.shape==(50,7) and raw.shape==(50,32) and np.isfinite(a).all() and np.isfinite(raw).all() and 0<n<=5
                    assert np.array_equal(np.asarray([s['action'] for s in tele[step+1:step+n+1]]),a[:n].astype(float))
                    route=r['route'];counts[route['path']]+=1;counts['requests']+=1;counts['actions']+=n
                    diagnostic=metrics(commands,q,history)
                    if 'distance' in route:
                        if cfg['kind']=='parent':assert route.get('rejected_path',route['path'])!='kalman_verified'
                        else:
                            allowed,blocked,value=decide(cfg,commands,q,history);proposal=route.get('rejected_path',route['path'])=='kalman_verified'
                            assert proposal==(allowed and not last) and route['gate_blocked']==blocked and route['consecutive_kalman_block']==last
                            if cfg['kind']=='metric':assert np.isclose(value['metric_value'],route['tuning_features']['metrics']['metric_value'],atol=1e-12,rtol=1e-12)
                            blocks.update(blocked);counts['KF_proposals']+=proposal
                    if route['path']=='kalman_verified':
                        assert not last and route['vision_calls']==route['vlm_calls']==0 and r['draft_noise_hash'] is None
                        assert np.array_equal(a[:n,3:],np.broadcast_to(commands[-1][3:],(n,4)))
                        counts['KF_actions']+=n;counts['KF_executed_requests']+=1
                    if route['path']=='full':
                        if route.get('full_reason')=='periodic':assert draft_rounds==2
                        draft_rounds=0
                    else:draft_rounds+=1;assert draft_rounds<=2
                    rows.append(dict(condition=cfg['id'],**result['case'],replan=index,step=step,path=route['path'],gate_blocks='|'.join(route.get('gate_blocked',[])),metric_used=cfg['kind']=='metric',
                        a_turn=diagnostic.get('a_turn_degrees'),b_change=diagnostic.get('b_command_change'),c_pose=diagnostic.get('c_world_pose_degrees'),
                        error_abs=diagnostic.get('forecast_error_rms_max5'),error_relative=diagnostic.get('forecast_error_relative_max5'),
                        executed_length=n,predict_policy_ms=r['timing']['policy_wall_ms'],ack_policy_ms=r['ack_policy_wall_ms'],seed=r['seed'],input_hash=r['input_hash']))
                    for t in tele[step+1:step+n+1]:
                        action=np.asarray(t['action']);prior=k.H@k.F_forecast@k.x;k.observe(action);history.append(feature_after_observe(k,action,prior));commands.append(action);q.append(t['eef_quat'])
                    step+=n;last=route['path']=='kalman_verified'
            total=sum(r['timing']['policy_wall_ms']+r['ack_policy_wall_ms'] for r in inf)/1000
            assert np.isclose(total,result['policy_wall_s'],atol=1e-9,rtol=1e-12)
            assert np.isclose(result['predict_policy_wall_s']+result['ack_policy_wall_s'],result['policy_wall_s'])
            checks.append(dict(condition=cfg['id'],case=result['case'],actions=step,passed=True,result_sha256=hashlib.sha256((d/'result.json').read_bytes()).hexdigest(),identity_sha256=hashlib.sha256((d/'identity.json').read_bytes()).hexdigest()))
        routing[cfg['id']]=dict(counts,blocked=dict(blocks))
    assert all(len(v)==6 and len(set(v))==1 for v in initials.values())
    with (OUT/'requests_and_causal_features.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic(OUT/'routing_summary.json',routing)
    atomic(OUT/'execution_audit.json',dict(time=time.time(),passed=True,episodes=len(checks),requests=len(rows),checks=checks,case_blocks_initial_RGB_state_startup_Full_same_service_exact=40,
        gate_replay='Only completed telemetry before each request; metric RHS exactly matches recorded runtime value; Kalman/gripper/rotation and P2 verified.',
        timing='Result policy wall equals all predict+ack server wall. Offline feature columns on non-metric arms are diagnostic only and were not charged as runtime computation.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('AUDIT PASSED',len(checks),len(rows))


if __name__=='__main__':main()

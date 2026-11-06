"""One final streaming audit; preserve scientific failures and original files."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from preparations.idea1_v3.io import atomic
from preparations.idea1_v3_1.common import LIMITS
from preparations.idea1_v2_2.common import seed
from preparations.idea1_v3.policy import verification_seed
from .analyze import OUT,records
from .rules_c1 import gate
from .rules_c1 import pose_angle
from preparations.idea1_v3.io import write_rows


def e2(groups):
    output=[];conditions={}
    for cid,episodes in groups.items():
        if not episodes or episodes[0]['config']['experiment']!='E2':continue
        count=0;source=[];blocks=Counter();unique=Counter();stats=Counter()
        for episode in episodes:
            directory=Path(episode['_directory']);states=[json.loads(l) for l in (directory/'telemetry.jsonl').read_text().splitlines()];rows=[json.loads(l) for l in (directory/'infer.jsonl').read_text().splitlines()];cfg=episode['config'];last=False;n=0
            for r in rows:
                route=r['route'];step=r['step']
                if 'distance' in route:
                    commands=[s['action'] for s in states[1:step+1]];q=[s['eef_quat'] for s in states[1:step+1]]
                    allowed,blocked=gate(commands,q,cfg.get('angle',20),cfg.get('change',.05),cfg.get('pose',5))
                    if cfg.get('alternate'):allowed=len(commands)>=5;blocked=[] if allowed else ['history']
                    expected=allowed and not last;proposed=route.get('rejected_path',route['path'])=='kalman_verified'
                    if expected!=proposed or route['gate_blocked']!=blocked or route['consecutive_kalman_block']!=last:raise ValueError('E2 causal block/selection mismatch '+str(directory))
                    blocks.update(blocked)
                    if len(blocked)==1:unique[blocked[0]]+=1
                    stats['gate_all_pass']+=not blocked;stats['jointly_blocked']+=bool(blocked);stats['Kalman_proposals']+=expected
                    stats['Kalman_executed_requests']+=route['path']=='kalman_verified';stats['Kalman_executed_actions']+=r['executed_length'] if route['path']=='kalman_verified' else 0
                    output.append(dict(condition=cid,case=episode['case'],step=step,replan=r['replan'],executed_length=r['executed_length'],history_steps=[max(1,step-4),step],c_world_xyzw_max_pairwise_degrees=pose_angle(q[-5:]) if len(q)>=5 else None,theta_c_degrees=cfg.get('pose',5),c_operator='<',alternate_bypasses_gate=bool(cfg.get('alternate')),blocked=blocked,previous_actually_Kalman=last,expected_Kalman_proposal=expected,actual_path=route['path'],rejected_path=route.get('rejected_path'),passed=True))
                    count+=1;n+=1
                last=route['path']=='kalman_verified'
            source.append(dict(case=episode['case'],decisions=n,telemetry_sha256=hashlib.sha256((directory/'telemetry.jsonl').read_bytes()).hexdigest(),infer_sha256=hashlib.sha256((directory/'infer.jsonl').read_bytes()).hexdigest()))
        conditions[cid]=dict(episodes=len(episodes),decisions=count,mismatches=0,passed=True,individual_blocks=dict(blocks),sole_blocker=dict(unique),joint_stats=dict(stats),total_executed_actions=sum(r['actions'] for r in episodes),sources=source)
    path=OUT/'E2_gate_decisions.jsonl'
    if path.exists():path.unlink()
    write_rows(path,output)
    atomic(OUT/'E2_gate_audit.json',dict(time=time.time(),passed=True,conditions=conditions,decisions=len(output),mismatches=0,decision_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        rule='Recompute a/b/common constraints/strict-c1 from telemetry completed steps1..request.step only; then AND not previous actually executed Kalman. Compare ordered gate_blocked, consecutive flag, proposal including rejected_path.'))


def e4(groups):
    result={}
    for cid,episodes in groups.items():
        if not episodes or episodes[0]['config']['experiment']!='E4':continue
        counts=Counter();cases=[]
        for episode in episodes:
            path=Path(episode['_directory'])/'infer.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines()];consecutive=0;errors=[]
            for r in rows:
                route=r['route'];counts['requests']+=1
                if any(route.get(k,False) for k in ('verify_gripper','post_verify_gripper','pending_gripper','gripper_full')):errors.append('gripper protection flag')
                if episode['config'].get('mode')=='periodic':
                    if r['counts'].get('verify',0)!=0 or any(k.startswith('verify_') for k in r['timing']) or 'distance' in route:errors.append('periodic control invoked verifier')
                    counts['verified_absent_control_requests']+=1
                reason=route.get('full_reason')
                if reason and ('gripper' in reason or reason=='pending_gripper'):errors.append('gripper Full')
                if route['path']=='full':
                    if reason=='periodic':
                        counts['periodic_full']+=1
                        if consecutive!=2:errors.append('periodic without exactly2 draft rounds')
                    if reason not in ('startup','periodic','zero_accept'):errors.append('unexpected Full reason '+str(reason))
                    consecutive=0
                else:
                    consecutive+=1;counts['positive_draft_requests']+=1
                    if consecutive>2:errors.append('more than2 actual draft rounds')
                counts['max_consecutive_draft']=max(counts['max_consecutive_draft'],consecutive)
            if errors:raise ValueError(cid+' '+str(episode['case'])+' '+repr(errors))
            cases.append(dict(case=episode['case'],requests=len(rows),passed=True,infer_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        result[cid]=dict(episodes=len(episodes),passed=True,counts=dict(counts),cases=cases)
    atomic(OUT/'E4_protocol_audit.json',dict(time=time.time(),passed=True,conditions=result,
        rules='Only E4 conditions: actual consecutive positive-draft rounds<=2; periodic Full exactly after2; no verify/post gripper flags or gripper Full reasons; Full reasons startup/periodic/zero_accept only.'))
    return result


def full(groups):
    receipt=[]
    for cid,episodes in groups.items():
        for episode in episodes:
            directory=Path(episode['_directory']);case=episode['case'];cfg=episode['config'];execute=10 if cfg['model']=='xiaomi' else 5;H=10 if cfg['model']=='xiaomi' else 50
            rows=[json.loads(l) for l in (directory/'infer.jsonl').read_text().splitlines()];states=[json.loads(l) for l in (directory/'telemetry.jsonl').read_text().splitlines()]
            if len(states)!=episode['actions']+1 or len(rows)!=episode['predictions']:raise ValueError('Count mismatch '+str(directory))
            if [r['step'] for r in states]!=list(range(len(states))):raise ValueError('Telemetry step gap')
            if not episode['success'] and episode['actions']!=LIMITS[case['suite']]:raise ValueError('Failure before native limit')
            arrays=np.load(directory/'arrays.npz',allow_pickle=False);step=0;last_kalman=False;K=0
            for index,r in enumerate(rows):
                n=r['executed_length'];a=arrays[f'actions_{index}'];raw=arrays[f'raw_{index}']
                if r['step']!=step or not 0<n<=min(execute,r['execute_length']) or a.shape!=(H,7) or raw.shape!=(H,32):raise ValueError('Proposal alignment mismatch')
                if not np.isfinite(a).all() or not np.isfinite(raw).all():raise ValueError('Nonfinite action')
                if r['seed']!=seed(case['suite'],case['task'],case['init'],index):raise ValueError('Seed depends on dispatch/order')
                if r['verification_seed']!=verification_seed(r['seed']):raise ValueError('Verification seed identity mismatch')
                if r.get('metadata_revision')=='v33-noise-r2':
                    model_hash=hashlib.sha256(np.random.default_rng(r['seed']).standard_normal((50,32),dtype=np.float32).tobytes()).hexdigest()
                    verify_hash=hashlib.sha256(np.random.default_rng(r['verification_seed']).standard_normal((50,32),dtype=np.float32).tobytes()).hexdigest()
                    route=r['route'];draft=route['path'] in ('pure','v14_verified','periodic_draft') or route.get('rejected_path')=='v14_verified'
                    if r['draft_noise_hash']!=(model_hash if draft else None):raise ValueError('Draft noise provenance mismatch')
                    if r['full_noise_hash']!=(model_hash if route['path']=='full' else None):raise ValueError('Full noise provenance mismatch')
                    if r['verify_noise_hash']!=(verify_hash if 'distance' in route else None):raise ValueError('Verify noise provenance mismatch')
                actual=np.asarray([s['action'] for s in states[step+1:step+n+1]],float)
                if not np.array_equal(actual,a[:n].astype(float)):raise ValueError('Telemetry differs from actual proposed prefix')
                if cfg['experiment']=='E2' and 'distance' in r['route']:
                    commands=[s['action'] for s in states[1:step+1]];quats=[s['eef_quat'] for s in states[1:step+1]]
                    allowed,blocked=gate(commands,quats,cfg.get('angle',20),cfg.get('change',.05),cfg.get('pose',5))
                    if cfg.get('alternate'):allowed=len(commands)>=5
                    expected=allowed and not last_kalman
                    proposed=r['route'].get('rejected_path',r['route']['path'])=='kalman_verified'
                    if proposed!=expected:raise ValueError('E2 causal gate mismatch')
                    if proposed and r.get('draft_noise_hash','missing') is not None:raise ValueError('Kalman incorrectly assigned draft noise')
                    if proposed:K+=1
                last_kalman=r['route']['path']=='kalman_verified';step+=n
            if step!=episode['actions']:raise ValueError('Executed action sum mismatch')
            receipt.append(dict(case=case,condition=cid,actions=step,requests=len(rows),Kalman_attempts=K,passed=True,
                identity_sha256=hashlib.sha256((directory/'identity.json').read_bytes()).hexdigest(),result_sha256=hashlib.sha256((directory/'result.json').read_bytes()).hexdigest()))
    atomic(OUT/'final_execution_audit.json',dict(time=time.time(),passed=True,cases=len(receipt),checks=receipt,
        scope='All completed scientific cases: raw/decoded shapes+finiteness, exact actual execution prefixes, request/telemetry alignment, seed identity, native failure limit, strict-c1 causal E2 routing and Kalman noise identity.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');args=p.parse_args();groups=records();e4(groups);e2(groups)
    if args.full:full(groups)

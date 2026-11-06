"""One-pass CPU audit and runtime summary of COMPLETED continuous episodes.

Gate: --mode gate. Final: --mode full, only after dispatcher passed and all
300 normalized cells are valid. Event envelopes are NOT kernel intersections.
No model, CUDA, simulator or running-case file is opened by this script.
"""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES']=''
R=Path(__file__).resolve().parent
ROOT=R.parents[2]
ARMS=('fresh18_no_encoder','sm16_no_encoder','sm16_encoder')
SUITES=('libero_object','libero_spatial','libero_goal','libero_10')
LIMITS=dict(zip(SUITES,(280,220,300,520)))
spec=importlib.util.spec_from_file_location('continuous_audit_original_client',ROOT/'preparations/idea1_v2_1/runtime_client.py')
client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s]
def require(value,message):
    if not value:raise ValueError(message)
def save(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n')
def csv_write(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()} for row in rows)
def stats(values):
    values=[float(v) for v in values if v is not None]
    return dict(n=len(values),mean=float(np.mean(values)),p50=float(np.percentile(values,50)),p95=float(np.percentile(values,95)),min=min(values),max=max(values)) if values else None


def event_windows(record):
    v=float(record['vlm_ms']);a=float(record['ae_start_after_vlm_start_ms']);end=a+float(record['ae_ms'])
    require(abs(end-(v+record['ae_finish_after_vlm_ms']))<.05,'AE Event endpoint consistency failed')
    result=dict(ae_x_vlm_event_envelope_ms=max(0.,min(end,v)-max(a,0.)),ae_x_next_encoder_event_envelope_ms=None)
    if record['next_encoder']:
        start=v+record['next_encoder_start_after_vlm_ms'];finish=start+record['next_vision_ms']
        require(abs((end-finish)-record['ae_finish_minus_next_encoder_ms'])<.05,'Next Encoder Event endpoint consistency failed')
        result['ae_x_next_encoder_event_envelope_ms']=max(0.,min(end,finish)-max(a,start))
    return result


def load_completed(mode):
    parents=[R/'gate_gpu2_v3',R/'gate_gpu3_v3']
    normalized_path=R/'normalized_cases.json'
    if mode=='full':
        dispatcher=read(R/'dispatch_execution.json')
        require(dispatcher['status']=='passed','Final dispatcher has not passed; no episode scan allowed')
        require(all(c['status']=='passed' and c['parent_exit_code']==0 for c in dispatcher['chunks']),'Final parent exits incomplete')
        parents += [Path(c['output']) for c in dispatcher['chunks']]
        normalized=read(normalized_path)
        require(len(normalized)==300 and all(c['status']=='valid' for c in normalized),'Fixed300 grid incomplete; no episode scan allowed')
    else:normalized=None
    mapping={};receipts=[]
    for parent in parents:
        path=parent/'execution.json';value=read(path)
        require(value['status']=='passed' and value['source_unchanged'] and value['server_exit_code']==0,f'Parent incomplete: {parent}')
        require(value['cleanup']['passed'],'Green Context cleanup failed')
        require(value['source_before']==value['source_after'],'Recorded sources changed')
        for name,expected in value['source_before'].items():require(sha(ROOT/name)==expected,f'Current source changed: {name}')
        receipts.append(dict(path=str(path),sha256=sha(path),gpu_uuid=value['gpu_uuid'],source_hashes=len(value['source_before'])))
        for cell in value['cases']:
            require(cell['status']=='valid','Invalid parent case')
            if cell['source_kind']=='completed_gate':continue
            require(cell['client_exit_code']==0 and 'render_gpu_verified' in cell,'Missing clean client/render proof')
            require(not cell['gpu_before']['graphics_processes'] and not cell['gpu_after']['graphics_processes'],'Renderer did not exit')
            key=cell['case_key'];require(key not in mapping,'Duplicate new episode source')
            mapping[key]={**cell,'gpu_uuid':value['gpu_uuid']}
    if normalized is not None:
        require(len(mapping)==300,'Completed sources do not contain exactly300 distinct cells')
        for row in normalized:
            source=mapping[row['case_key']]
            require(source['directory']==row['directory'] and source['result']['success']==row['success'],'Normalized source/outcome mismatch')
    else:require(len(mapping)==6,'Gate must contain exactly6 cells')
    expected={(arm,s,t,i) for arm in ARMS for s in SUITES for t in range(10) for i in range(3 if t%2==0 else 2)}
    actual={(c['arm'],c['suite'],c['task'],c['init']) for c in mapping.values()}
    require(actual==expected if mode=='full' else actual<=expected,'Cells outside prospective grid')
    return list(mapping.values()),receipts,sha(normalized_path) if normalized is not None else None


def audit_case(cell):
    d=Path(cell['directory']);arm=cell['arm'];identity=read(d/'identity.json');complete=read(d/'complete.json')
    rows=lines(d/'infer.jsonl');telemetry=lines(d/'telemetry.jsonl');consumed=lines(d/'consumed_chunks.jsonl')
    gpu=read(d/'gpu_records.json');reset=read(d/'reset_receipt.json');env=read(d/'environment_protocol.json')
    require(complete['valid'] and complete['status']=='valid' and complete['worker_stopped'],'Worker/case not cleanly complete')
    require(complete['success']==cell['result']['success'],'Parent result differs')
    require((identity['suite'],identity['task'],identity['init'],identity['arm'])==(cell['suite'],cell['task'],cell['init'],arm),'Case identity mismatch')
    require([identity[k] for k in ('D','H','R','env_seed','settling_steps')]==[10,10,5,7,10],'Original model/env protocol differs')
    require(identity['action_limit']==LIMITS[cell['suite']] and complete['steps']<=identity['action_limit'],'Action limit differs')
    require(complete['termination']==('success' if complete['success'] else 'action_limit'),'Termination mismatch')
    if not complete['success']:require(complete['steps']==identity['action_limit'],'Valid failure did not reach action limit')
    require(identity['client_sha256']==sha(R/'client.py') and identity['original_client_sha256']==client.LOADED_CLIENT_SHA256,'Loaded client source mismatch')
    require(env['control_frequency_hz']==20 and env['wall_clock_paced'] and env['no_catchup'],'Control clock differs')
    controls=env['controller']
    for name,values in {'input_min':[-1]*6,'input_max':[1]*6,'output_min':[-.05]*3+[-.5]*3,'output_max':[.05]*3+[.5]*3}.items():
        require(np.allclose(controls[name],values),'Original OSC scale changed')
    require(complete['mailbox_capacity']==1 and complete['max_mailbox_observed']<=1,'Mailbox bound changed')
    require(reset['ok'] and reset['reset'] and reset['bootstrap_pending'] and reset['episode']>gpu['episode'],'Terminal reset missing')
    require(reset==complete['terminal_reset_receipt'] and gpu['ok'],'Drain/reset receipts disagree')
    require(complete['pending_ahead_at_drain']==complete['unused_ahead_at_terminal']==gpu['pending_ahead'],'Unused ahead terminal accounting differs')
    require(len(rows)==complete['predictions'] and len(gpu['records'])==len(rows)-1,'Missing deferred GPU Event records')
    require([r['request_id'] for r in rows]==list(range(len(rows))),'Prediction request IDs skipped')
    require([r['online_request_id'] for r in gpu['records']]==list(range(1,len(rows))),'GPU/request ID mismatch')
    require([t['step'] for t in telemetry]==list(range(complete['steps']+1)),'Telemetry missing actual steps')
    require(abs(telemetry[0]['sim_time_s']-.5)<1e-8,'Original settle10 changed')
    require([r['target_chunk_id'] for r in consumed]==list(range(len(consumed))),'Consumed target sequence skipped')
    require(len(consumed)==complete['consumed_chunks'] and sum(r['executed_length'] for r in consumed)==complete['steps'],'Consumed action count mismatch')
    by_request={r['request_id']:r for r in rows};by_step={t['step']:t for t in telemetry};by_gpu={r['online_request_id']:r for r in gpu['records']}
    used={r['request_id'] for r in consumed};counts=Counter(t['consumed_request_id'] for t in telemetry[1:])
    require({r['request_id'] for r in rows if r['disposition']=='consumed'}==used,'Selected disposition mismatch')
    require(set(counts)==used,'Unselected candidate was executed')
    arrays={};request_output=[];artifact_sha={name:sha(d/name) for name in ('identity.json','complete.json','infer.jsonl','telemetry.jsonl','consumed_chunks.jsonl','gpu_records.json','reset_receipt.json','environment_protocol.json')}
    for i,row in enumerate(rows):
        require(row['seed']==client.model_seed(cell['suite'],cell['task'],cell['init'],row['target_chunk_id']),'Noise differs from target chunk seed')
        require(row['bootstrap']==(i==0) and row['graph_kv_buffers']==2,'Bootstrap/two-slot mismatch')
        require(row['generated_length']==10 and row['execute_length']==5,'D10/H10/R5 output contract mismatch')
        require(row['executed_length']==counts[i] and (i in used or row['executed_length']==0),'Discarded prediction executed actions')
        if i not in used:require(row['disposition'] in ('expired','superseded','terminal'),'Unfinished candidate disposition')
        if i>1:require(row['input_step']>rows[i-1]['input_step'],'Input did not advance to a real later frame')
        elif i==1:require(row['input_step']==0 and row['input_hash']==rows[0]['input_hash'],'First repeated bootstrap input differs')
        t=by_step[row['input_step']]
        require(row['input_hash']==t['real_observation_hash'],'Policy input was not a recorded actual environment frame')
        require(abs(row['input_capture_host']-t['capture_host'])<1e-7 and row['input_capture_host']<=row['request_send_host']<=row['response_host'],'Input timestamp lies in future')
        protocol=row['protocol'];require(protocol['arm']==arm and [protocol[k] for k in ('num_steps','H','R','cache_slots')]==[10,10,5,2],'Runtime arm/model protocol changed')
        require(protocol['next_encoder']==(arm=='sm16_encoder'),'Unexpected Encoder configuration')
        resource=protocol['resource_adapter'];sms=0 if arm=='fresh18_no_encoder' else 16
        require(resource['ae_sm_min']==sms,'Requested SM count changed')
        require(bool(resource['actual_resources']) and all(x['sm_counts']==[16,112] for x in resource['actual_resources']) if sms else not resource['actual_resources'],'Actual resource split changed')
        path=Path(row['prediction_path']);require(sha(path)==row['prediction_sha256'],'Prediction file SHA mismatch')
        with np.load(path,allow_pickle=False) as f:
            action=f['actions'];norm=f['normalized_actions']
            require(int(f['seed'])==row['seed'] and int(f['target_chunk_id'])==row['target_chunk_id'],'Saved target/seed mismatch')
        require(action.shape==(10,7) and norm.shape==(10,32) and np.isfinite(action).all() and np.isfinite(norm).all(),'Invalid output shape/finite')
        require(client.value_digest(action)==row['actions_sha256'] and client.value_digest(norm)==row['normalized_sha256'],'Array digest mismatch')
        require(np.array_equal(action,np.asarray(row['chunk_actions'],dtype=np.float32)),'JSON/NPZ actions disagree')
        arrays[i]=(action,norm)
        out=dict(case_key=cell['case_key'],arm=arm,gpu_uuid=cell['gpu_uuid'],request_id=i,target_chunk_id=row['target_chunk_id'],
            disposition=row['disposition'],selected=i in used,bootstrap=i==0,input_step=row['input_step'],seed=row['seed'],input_sha256=row['input_hash'],
            prediction_sha256=row['prediction_sha256'],capture_to_response_ms=(row['response_host']-row['input_capture_host'])*1000,
            client_roundtrip_ms=row['client_roundtrip_ms'],executed_length=row['executed_length'],graph_setup_ms=row['graph_setup_ms'])
        if i:
            g=by_gpu[i];online=g['online'];f=g['frontiers']
            require(not any(k.startswith('_') for k in g),'Undetached GPU Event handle in diagnostics')
            require(g['full_kv_slots']==2 and g['graph_recaptures']==0 and row['graph_setup_ms']==0,'Slot/recapture invariant violated')
            require(g['target_chunk_id']==row['target_chunk_id'] and g['input_step']==row['input_step'] and g['online']==row['online'],'Event metadata/request mismatch')
            require(g['old_kv_capture']['step']==rows[i-1]['input_step'],'Old KV did not come from previous admitted prediction')
            require(g['old_kv_capture']['host']==rows[i-1]['input_capture_host'],'Old KV capture clock differs from its real source')
            require(g['old_slot']==rows[i-1]['snapshot']['slot'] and g['old_generation']==rows[i-1]['snapshot']['generation'],'Old Lease slot/generation differs')
            require(len(f)==10 and f==sorted(f) and all((18 if sms==0 else 3)<=x<=18 for x in f),'Frontier changed')
            require(len(g['iteration_ms'])==10 and all(math.isfinite(x) and x>=0 for x in g['iteration_ms']),'Iteration Event times invalid')
            if online['has_next']:
                offered=by_step[online['next_capture_step']]
                require(online['next_capture_step']>row['input_step'] and online['next_input_hash']==offered['real_observation_hash'],'Ahead observation invented or not later')
                require(abs(online['next_capture_host']-offered['capture_host'])<1e-7 and online['next_capture_host']<=online['next_observation_received_host']+.001,'Ahead capture timestamp invalid')
                require(row['offered_next_input_step']==online['next_capture_step'] and row['offered_next_input_hash']==online['next_input_hash'],'Accepted next-frame transport mismatch')
                if i+1<len(rows):require(rows[i+1]['input_step']==online['next_capture_step'] and rows[i+1]['input_hash']==online['next_input_hash'] and rows[i+1]['consumed_previous_accepted_observation'],'Pending accepted Prepared/input identity not consumed')
            else:
                require(not row['actual_next_encoder'],'Noahead request encoded an invented future')
                if online['reason']=='not_yet':require(not online['terminal'],'not_yet was classified terminal')
            if i>1:require(row['consumed_previous_accepted_observation']==rows[i-1]['online']['has_next'],'Pending ticket admission chain mismatch')
            require(g['next_encoder']==row['actual_next_encoder']==bool(arm=='sm16_encoder' and online['has_next']),'Actual Encoder count mismatch')
            windows=event_windows(g)
            if g['next_encoder'] and (online['ae_done_when_next_observation_received'] or online['ae_done_after_next_preparation']):
                require(windows['ae_x_next_encoder_event_envelope_ms']<.05,'Event overlap contradicts completed-AE query')
            out.update(windows,actual_next_encoder=g['next_encoder'],ae_done_on_receive=online['ae_done_when_next_observation_received'],
                ae_done_after_prepare=online.get('ae_done_after_next_preparation'),next_reason=online['reason'])
            out.update(old_kv_source_step=g['old_kv_capture']['step'],old_kv_capture_host=g['old_kv_capture']['host'],
                old_kv_rounds_used=sum(x<18 for x in f),
                admitted_input_advance_steps=row['input_step']-rows[i-1]['input_step'],
                accepted_next_frame=online['has_next'],
                accepted_camera_advance_steps=online['next_capture_step']-row['input_step'] if online['has_next'] else None,
                accepted_camera_skipped_steps=online['next_capture_step']-row['input_step']-1 if online['has_next'] else None)
            for key in ('vision_ms','vlm_ms','ae_ms','next_vision_ms','ae_submit_ms','cpu_cpu_ms','exposed_ae_after_vlm_ms','next_encoder_start_after_vlm_ms'):
                out[key]=g.get(key)
            for key in ('observation_wait_host_ms','next_prepare_host_ms'):out[key]=online.get(key)
        request_output.append(out)
    for j,row in enumerate(consumed):
        i=row['request_id'];require(row==by_request[i],'Consumed artifact differs from request record')
        require(row['executed_length']==(min(5,complete['steps']-5*j)),'Consumed chunk is not exactly its R5 block')
        input_path=Path(row['input_path']);require(sha(input_path)==row['input_file_sha256'],'Consumed input file SHA mismatch')
        with np.load(input_path,allow_pickle=False) as f:
            obs={k:f[k] for k in f.files if k not in ('instruction','seed','input_step')}
            require(client.input_digest(obs,str(f['instruction']))==row['input_hash'] and int(f['seed'])==row['seed'] and int(f['input_step'])==row['input_step'],'Consumed input bytes/seed differ')
    applications=[]
    for t in telemetry[1:]:
        row=by_request[t['consumed_request_id']];index=t['action_index'];action=arrays[row['request_id']][0][index]
        require(0<=index<5 and t['chunk_id']==row['target_chunk_id'] and t['seed']==row['seed'],'Applied wrong target/seed/H10 tail')
        require(np.array_equal(action,np.asarray(t['action'],dtype=np.float32)) and client.value_digest(action)==t['action_sha256'],'Applied action differs from selected output')
        require(t['application_host']>=row['response_host'] and t['input_step']==row['input_step'],'Applied prediction before available/from other frame')
        old=by_gpu[row['request_id']]['old_kv_capture'] if row['request_id'] else None
        old_used=any(x<18 for x in by_gpu[row['request_id']]['frontiers']) if old else False
        old_ms=(t['application_host']-old['host'])*1000 if old else None
        old_steps=t['step']-1-old['step'] if old else None
        if old:require(old_ms>=0 and old_steps>=0,'Old KV source lies after action application')
        applications.append(dict(case_key=cell['case_key'],arm=arm,step=t['step'],chunk_id=t['chunk_id'],request_id=t['consumed_request_id'],
            capture_to_actual_application_ms=(t['application_host']-row['input_capture_host'])*1000,observation_age_steps=t['observation_age_steps'],
            deadline_lateness_ms=t['deadline_lateness_ms'],environment_step_ms=t['environment_step_ms'],response_to_application_ms=t['response_to_application_ms'],bootstrap_chunk=t['chunk_id']==0,
            old_kv_source_step=old['step'] if old else None,old_kv_capture_host=old['host'] if old else None,
            old_kv_used_by_prediction=old_used,old_kv_available_source_age_ms=old_ms,old_kv_available_source_age_steps=old_steps,
            used_old_kv_source_age_ms=old_ms if old_used else None,used_old_kv_source_age_steps=old_steps if old_used else None))
    intervals=[(b['application_host']-a['application_host'])*1000 for a,b in zip(telemetry[1:],telemetry[2:])]
    require(all(x>=49.9 for x in intervals),'Actor caught up faster than20Hz')
    require(Counter(r['disposition'] for r in rows)==complete['disposition_counts'],'Disposition accounting differs')
    if len(rows)>1:require(gpu['pending_ahead']==rows[-1]['online']['has_next'],'Terminal unused admission not accounted')
    summary=dict(case_key=cell['case_key'],arm=arm,suite=cell['suite'],task=cell['task'],init=cell['init'],gpu_uuid=cell['gpu_uuid'],
        success=complete['success'],steps=complete['steps'],predictions=len(rows),consumed_chunks=len(consumed),
        dispositions=complete['disposition_counts'],actor_wall_s=complete['actor_wall_s'],actor_hz=complete['steps']/complete['actor_wall_s'],
        prediction_hz=(len(rows)-1)/(rows[-1]['response_host']-rows[1]['request_send_host']) if len(rows)>1 else None,
        online_service_window_s=rows[-1]['response_host']-rows[1]['request_send_host'] if len(rows)>1 else None,
        online_kv_updates=len(rows)-1,
        kv_update_service_hz=(len(rows)-1)/(rows[-1]['response_host']-rows[1]['request_send_host']) if len(rows)>1 else None,
        consumed_chunk_admission_hz=len(consumed)/complete['actor_wall_s'],
        consumed_r5_equivalent_hz=complete['steps']/5/complete['actor_wall_s'],
        unselected_candidate_fraction=sum(r['request_id'] not in used for r in rows[1:])/max(1,len(rows)-1),
        boundary_wait_ms=complete['boundary_wait_ms'],boundary_deadline_gap_ms=complete['boundary_deadline_gap_ms'],
        episode_wall_s=complete['wall_s'],all_work_wall_s=complete['all_work_wall_s'],record_io_s=complete['record_io_s'],
        bootstrap_graph_setup_ms=rows[0]['graph_setup_ms'],bootstrap_roundtrip_ms=rows[0]['client_roundtrip_ms'],
        unused_ahead_at_terminal=gpu['pending_ahead'],reset_episode=reset['episode'],worker_stopped=True,
        action_tick_ms=stats(intervals),control_gap_over50ms_total=sum(max(0.,x-50) for x in intervals),artifact_sha256=artifact_sha)
    bootstrap=dict(case=(cell['suite'],cell['task'],cell['init']),arm=arm,input_sha256=rows[0]['input_hash'],seed=rows[0]['seed'],
        init_vector_sha256=identity['init_vector_hash'],actions=arrays[0][0],normalized=arrays[0][1])
    return summary,request_output,applications,bootstrap


def aggregate(episodes,requests,applications):
    result={}
    for arm in ARMS:
        es=[r for r in episodes if r['arm']==arm];rs=[r for r in requests if r['arm']==arm and not r['bootstrap']];aps=[r for r in applications if r['arm']==arm]
        value=dict(episodes=len(es),successes=sum(e['success'] for e in es),success_rate=sum(e['success'] for e in es)/len(es),
            total_actions=sum(e['steps'] for e in es),requests=len(rs),selected_requests=sum(r['selected'] for r in rs),
            unselected_requests=sum(not r['selected'] for r in rs),unused_ahead_terminal_episodes=sum(e['unused_ahead_at_terminal'] for e in es))
        value['candidate_unselected_fraction']=value['unselected_requests']/value['requests'] if value['requests'] else None
        for key in ('steps','actor_hz','prediction_hz','kv_update_service_hz','consumed_chunk_admission_hz','consumed_r5_equivalent_hz','boundary_wait_ms','boundary_deadline_gap_ms','episode_wall_s','all_work_wall_s','bootstrap_graph_setup_ms','bootstrap_roundtrip_ms','control_gap_over50ms_total'):value[key]=stats(e[key] for e in es)
        for label,selected in [('all_online',rs),('selected',[r for r in rs if r['selected']]),('unselected',[r for r in rs if not r['selected']]),('actual_later_frame',[r for r in rs if r['input_step']>0])]:
            value[label]={key:stats(r.get(key) for r in selected) for key in ('capture_to_response_ms','client_roundtrip_ms','vision_ms','vlm_ms','ae_ms','next_vision_ms','ae_submit_ms','observation_wait_host_ms','next_prepare_host_ms','ae_x_vlm_event_envelope_ms','ae_x_next_encoder_event_envelope_ms','admitted_input_advance_steps','accepted_camera_advance_steps','accepted_camera_skipped_steps','old_kv_rounds_used')}
        for label,selected in [('all_actions',aps),('after_bootstrap_chunk',[r for r in aps if not r['bootstrap_chunk']])]:
            value[label]={key:stats(r[key] for r in selected) for key in ('capture_to_actual_application_ms','observation_age_steps','deadline_lateness_ms','response_to_application_ms','old_kv_available_source_age_ms','old_kv_available_source_age_steps','used_old_kv_source_age_ms','used_old_kv_source_age_steps')}
        enc=[r for r in rs if r.get('actual_next_encoder')]
        value['encoder_window']=dict(jobs=len(enc),positive_event_envelope=sum(r['ae_x_next_encoder_event_envelope_ms']>1e-5 for r in enc),
            ae_not_done_on_receive=sum(not r['ae_done_on_receive'] for r in enc),ae_not_done_after_prepare=sum(not r['ae_done_after_prepare'] for r in enc),
            actual_kernel_intersections_measured_here=False)
        result[arm]=value
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=('gate','full'),default='gate');parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        record=dict(vlm_ms=40.,ae_start_after_vlm_start_ms=5.,ae_ms=50.,ae_finish_after_vlm_ms=15.,
                    next_encoder=True,next_encoder_start_after_vlm_ms=0.,next_vision_ms=12.,ae_finish_minus_next_encoder_ms=3.)
        require(event_windows(record)==dict(ae_x_vlm_event_envelope_ms=35.,ae_x_next_encoder_event_envelope_ms=12.),'Envelope arithmetic failed')
        delayed={**record,'next_encoder_start_after_vlm_ms':20.,'ae_finish_minus_next_encoder_ms':-17.}
        require(event_windows(delayed)['ae_x_next_encoder_event_envelope_ms']==0.,'Late observation produced artificial envelope overlap')
        try:event_windows({**record,'ae_finish_minus_next_encoder_ms':100.})
        except ValueError:pass
        else:raise AssertionError('Inconsistent Event endpoints accepted')
        require('torch' not in __import__('sys').modules,'CPU audit imported Torch')
        print(json.dumps(dict(status='passed',gpu_launched=False,checks=['Event interval arithmetic','late Encoder has zero intersection','inconsistent Event endpoints rejected','no Torch import'])));return
    output=args.output or R/f'{args.mode}_audit'
    output.mkdir(parents=True,exist_ok=False)
    state=dict(status='checking_completed_inputs',mode=args.mode,cpu_only=True,gpu_launched=False,started_utc=datetime.now(timezone.utc).isoformat())
    try:
        cells,receipts,input_sha=load_completed(args.mode);episodes=[];requests=[];applications=[];firsts=defaultdict(dict)
        for cell in cells:
            episode,req,actions,first=audit_case(cell);episodes.append(episode);requests.extend(req);applications.extend(actions);firsts[first['case']][first['arm']]=first
        comparisons=[]
        for case,arms in firsts.items():
            require(set(arms)==set(ARMS),'Incomplete bootstrap arm pairing');ref=arms[ARMS[0]]
            for arm,row in arms.items():
                require((row['input_sha256'],row['seed'],row['init_vector_sha256'])==(ref['input_sha256'],ref['seed'],ref['init_vector_sha256']),'Bootstrap input/seed/environment differ across arms')
                for kind in ('actions','normalized'):
                    delta=np.abs(row[kind].astype(np.float64)-ref[kind].astype(np.float64));mae=float(delta.mean());maximum=float(delta.max())
                    require(maximum<=1e-5 and mae<=1e-6,'Bootstrap numerical tolerance exceeded')
                    comparisons.append(dict(suite=case[0],task=case[1],init=case[2],arm=arm,kind=kind,mae=mae,max_abs=maximum,input_sha256=row['input_sha256']))
        summary=aggregate(episodes,requests,applications)
        state.update(status='passed',finished_utc=datetime.now(timezone.utc).isoformat(),parents=receipts,normalized_cases_input_sha256=input_sha,
            episodes=len(episodes),prediction_requests=len(requests),actual_actions=len(applications),bootstrap_cases=len(firsts),
            numerical_tolerance=dict(max_abs=1e-5,mae=1e-6),bootstrap_max_abs=max(r['max_abs'] for r in comparisons),
            summary=summary,valid_failures_retained=[r['case_key'] for r in episodes if not r['success']],
            mechanism_gate=dict(event_window_exists=summary['sm16_encoder']['encoder_window']['positive_event_envelope']>0,
                required_condition='Actual later-frame source chain plus nonzero AE/nextEncoder CUDA Event envelope. Success/failure does not choose gate cells.',
                kernel_overlap_claim=False,explanation='CUDA Event envelopes include waits/gaps; this audit does not measure CUDA Kernel intersections. Separate nsys mechanism evidence is not a per-episode kernel proof.'),
            limitations=['Prepared Python object identity is enforced by frozen runtime ticket checks; serialized source/digest chaining is cross-checked here, not a new pointer measurement.',
                'Original source init/environment files and loaded client hashes are retained in identity; no simulator or fresh GPU output is generated by this audit.',
                'First online request repeats the pre-bootstrap frame and includes cold capture in observation age. All-online and actual-later-frame age summaries are separated.',
                'Actor is paced at most20Hz; missing candidates cause measured pauses. Actor Hz and control gaps are reported, not silently treated as uninterrupted20Hz.',
                'Old-KV available-source ages refer to the previous admitted prediction source at actual application (actor step minus1). Used-old-KV ages are null when all10 frontiers are18; bootstrap has no old source and is null. Mixed rounds can use both old and new layers.',
                'KV-update/service Hz counts online completed requests over first online send to last response; chunk admission Hz uses actor wall time and counts a final partial chunk once, while R5-equivalent Hz weights it by executed actions/5. Neither rate is camera sampling Hz.',
                'No success-based exclusion, replacement, or retry is performed.'])
        csv_write(output/'episodes.csv',episodes);csv_write(output/'requests.csv',requests);csv_write(output/'applications.csv',applications);csv_write(output/'bootstrap.csv',comparisons)
        save(output/'runtime_summary.json',summary);save(output/'audit.json',state)
        text=['# Completed continuous-runtime audit','',f'Status: passed. {len(episodes)} episodes; {len(requests)} predictions; {len(applications)} actual actions. Bootstrap max error: {state["bootstrap_max_abs"]}.','',
            'All target seeds, R5 consumption, discarded-candidate nonexecution, actual-frame digest chains, pending admissions, deferred Event counts, terminal resets, stopped workers and frozen sources passed. Valid task failures remain included.','',
            '| Arm | Success | Actor Hz mean | Next Encoder jobs | Positive Event windows |','|---|---:|---:|---:|---:|']
        for arm,s in summary.items():text.append(f'| {arm} | {s["successes"]}/{s["episodes"]} | {s["actor_hz"]["mean"]:.3f} | {s["encoder_window"]["jobs"]} | {s["encoder_window"]["positive_event_envelope"]} |')
        text += ['', '**Event envelope overlap is not actual Kernel overlap.** No per-episode Kernel intersection is claimed. Cold bootstrap ages and pauses are retained with separate steady-input summaries.']
        (output/'summary.md').write_text('\n'.join(text)+'\n')
        print(json.dumps({k:state[k] for k in ('status','episodes','prediction_requests','bootstrap_max_abs','mechanism_gate')},indent=2))
    except BaseException as error:
        state.update(status='failed',error=f'{type(error).__name__}: {error}');save(output/'audit.json',state);raise


if __name__=='__main__':main()

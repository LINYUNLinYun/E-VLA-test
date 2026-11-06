"""Expanded corrected-B client. Default is CPU validation, never quality launch.

Service construction/lifecycle/GPU allocation is deliberately outside this
client. A separately authorized B service must advertise and echo this exact
scientific registration, distinct from the Plus service's registration.
"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np

from preparations.idea1_v3_2.transfer_client import rpc,raw_hash,validate_raw
from preparations.idea1_v2_1.runtime_client import input_digest,value_digest
from .transfer_registration import MODES,SCOPE,digest,sha,load_registration


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.%d.temporary'%os.getpid())
    temporary.write_text(json.dumps(value,default=serial,ensure_ascii=False,indent=2,allow_nan=False))
    temporary.replace(path)


def append(path,row):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(row,default=serial,allow_nan=False)+'\n');stream.flush()


def request_seed(case_id,subtask,replan):
    return int(digest(['corrected-B-transfer-v1',case_id,int(subtask),int(replan)])[:8],16)


def counter_namespace(registration):return SCOPE+':'+registration['registration_hash']


def launch_gate(registration,authorization,health,environment,deadline):
    if registration.get('schema')!=2:
        raise PermissionError('Seed-only Bv1 is superseded CPU preparation, not a final quality registration')
    if (authorization.get('scope')!=SCOPE or authorization.get('allow_quality') is not True or
            authorization.get('registration_hash')!=registration['registration_hash']):
        raise PermissionError('Explicit parent B-scoped authorization required; CPU/Plus registration is not permission')
    if tuple(authorization.get('modes',()))!=MODES or environment not in authorization.get('environments',[]):
        raise PermissionError('Authorization does not cover all six modes and this environment')
    if not authorization.get('gpu_uuid') or not isinstance(authorization.get('gpu'),int):
        raise PermissionError('Exact parent-assigned physical GPU/UUID required')
    if not deadline or deadline<=time.time() or deadline>authorization.get('deadline_unix',0):
        raise PermissionError('Future deadline within the explicit authorization required')
    protocol=health['protocol'];identity=authorization['service_manifest_hash']
    if protocol.get('manifest_hash')!=identity or protocol.get('family')!='libero_transfer':
        raise PermissionError('Approved B service identity/family mismatch')
    model_reference=registration['policy_reference']['manifest_hash']
    if authorization.get('model_reference_manifest_hash')!=model_reference:
        raise PermissionError('Authorization binds a different source model reference')
    if protocol.get('model_reference_manifest_hash',identity)!=model_reference:
        raise PermissionError('Service has not proven its referenced LIBERO model identity')
    if protocol.get('gpu_uuid')!=authorization['gpu_uuid'] or not protocol.get('legacy_gate_passed'):
        raise PermissionError('Service GPU identity or legacy real-input gate missing')
    authorized=protocol.get('authorized_scientific_registrations',[])
    if registration['registration_hash'] not in authorized:
        raise PermissionError('Service is Plus-only/unextended: exact B registration capability is absent')
    if protocol.get('counter_namespace')!=counter_namespace(registration):
        raise PermissionError('Dedicated B quality counter namespace required')
    return dict(service_manifest_hash=identity,model_reference_manifest_hash=model_reference,
        registration_hash=registration['registration_hash'],gpu_uuid=authorization['gpu_uuid'],
        gpu=authorization['gpu'],server_pid=protocol.get('server_pid'),counter_namespace=counter_namespace(registration))


def canonical_raw(raw):
    validate_raw(raw)
    return dict(base_rgb=np.ascontiguousarray(raw['base_rgb']),wrist_rgb=np.ascontiguousarray(raw['wrist_rgb']),
                state8=np.asarray(raw['state8'],dtype=np.float32))


def validate_response(response,registration,mode,raw,instruction,seed,service_identity,case=None):
    if (response.get('manifest_hash')!=service_identity or response.get('registration_hash')!=registration['registration_hash'] or
            response.get('counter_namespace')!=counter_namespace(registration)):
        raise ValueError('Response lacks exact B service/registration/counter identity')
    if case is not None:
        attribution=response.get('attribution',{})
        if any(attribution.get(k)!=v for k,v in dict(scope=SCOPE,case_id=case['case_id'],case_hash=digest(case),
                environment=case['environment'],mode=mode,service_manifest_hash=service_identity).items()):
            raise ValueError('Response case attribution differs from the independently registered B case')
    if response.get('mode')!=mode or response.get('execute_length')!=5:
        raise ValueError('Six-mode source H50/R5 contract changed')
    actions=np.asarray(response['actions'],dtype=np.float64);model_raw=np.asarray(response['raw_actions'])
    if actions.shape!=(50,7) or model_raw.shape!=(50,32) or not np.isfinite(actions).all() or not np.isfinite(model_raw).all():
        raise ValueError('Expected finite released source50x7 decoded and50x32 raw actions')
    noise=np.random.default_rng(seed).standard_normal((50,32),dtype=np.float32)
    if response.get('input_hash')!=input_digest(raw,instruction) or response.get('noise_hash')!=value_digest(noise):
        raise ValueError('Service actual input/noise hashes differ from this request')
    draft=mode.startswith('draft_')
    if response.get('noise_used_in_inference') is not (not draft):
        raise ValueError('Draft D=N/A/noise-unused attribution differs')
    expected=dict(vision_encoder_calls=1,vlm_prefix_calls=0 if draft else 1,
                  ae_layer_calls=0 if draft else 180,learned_head_calls=1 if draft else 0,verify_calls=0)
    observed=response.get('observed_calls',{})
    if any(observed.get(k)!=v for k,v in expected.items()):
        raise ValueError('Actual function-route counters violate frozen source mode')
    route=response.get('route',{})
    if (route.get('fresh_prefix') is not True or route.get('fallback') is not False or
            route.get('verify') is not False or route.get('kv_cross_request') is not False):
        raise ValueError('Hidden fallback/verify/cross-request KV route is forbidden')
    return actions


def result_matches(result,registration,case,mode):
    return (result.get('status')=='complete' and result.get('registration_hash')==registration['registration_hash']
        and result.get('case_hash')==digest(case) and result.get('case_id')==case['case_id']
        and result.get('mode')==mode and result.get('family')=='libero_transfer'
        and isinstance(result.get('success'),bool)
        and (not case['video'] or result.get('video',{}).get('full_stream_decode_passed') is True))


def complete_results(root,registration):
    selected={};attempts=[]
    for case in registration['cases']:
        for mode in MODES:
            base=Path(root)/'episodes'/case['environment']/case['case_id']/mode
            matched=[]
            for path in sorted(base.glob('attempt*/result.json')):
                result=json.loads(path.read_text());attempts.append(dict(path=str(path),status=result.get('status'),
                    case_id=case['case_id'],mode=mode,registration_hash=result.get('registration_hash')))
                if result_matches(result,registration,case,mode):
                    required={'initial_raw.npz','telemetry.jsonl','infer.jsonl','tasks.jsonl','requests_sent.jsonl','rpc_responses.jsonl'}
                    artifacts=result.get('artifact_sha256',{})
                    if not required.issubset(artifacts):raise ValueError('Complete case lacks required artifact receipts: '+str(path))
                    for filename,expected in artifacts.items():
                        target=path.parent/filename
                        if Path(filename).name!=filename or not target.is_file() or sha(target)!=expected:
                            raise ValueError('Complete case artifact changed; never silently resume: '+str(target))
                    matched.append(result)
            if len(matched)>1:raise ValueError('Duplicate valid complete case/mode; do not select a preferred outcome')
            if matched:selected[(case['case_id'],mode)]=matched[0]
    return selected,attempts


def next_attempt(base):
    numbers=[]
    for path in Path(base).glob('attempt*'):
        suffix=path.name[len('attempt'):]
        if not suffix.isdigit():raise ValueError('Unexpected attempt namespace')
        numbers.append(int(suffix))
    return Path(base)/('attempt%03d'%(max(numbers,default=-1)+1))


def make_adapter(case,reference,authorization):
    from .corrected_transfer_envs import CorrectedCalvinTransfer,CorrectedGoogleTransfer,CorrectedPushTProjection
    if case['environment']=='calvin':return CorrectedCalvinTransfer(reference)
    if case['environment']=='simpler':
        # The separately approved persistent service already owns this card.
        # Parent orchestrator owns exclusive-device preflight/lifecycle; this
        # client verifies its exact UUID and cannot allocate a different card.
        return CorrectedGoogleTransfer(case['task'],reference,gpu=authorization['gpu'],
            gpu_uuid=authorization['gpu_uuid'],require_idle=False)
    os.environ.setdefault('SDL_VIDEODRIVER','dummy')
    from diffusion_policy.env.pusht.pusht_env import PushTEnv
    return CorrectedPushTProjection(reference,PushTEnv(render_size=256,render_action=False))


def actual_record(env):
    actual=env.telemetry()
    tips=actual.get('actual_fingertip_world_positions',{})
    if len(tips)==2:
        values=list(tips.values());actual['measured_gripper_width_m']=float(np.linalg.norm(np.asarray(values[0])-np.asarray(values[1])))
    native=getattr(env,'native',None);obs=getattr(native,'obs',None)
    engine=getattr(native,'native',None)
    if engine is not None:
        objects=getattr(engine,'episode_objs',None)
        if objects is None:
            single=getattr(engine,'obj',None);objects=[] if single is None else [single]
        actual['object_poses_world']={str(getattr(obj,'name',index)):
            dict(xyz=np.asarray(obj.pose.p).tolist(),quaternion_wxyz=np.asarray(obj.pose.q).tolist())
            for index,obj in enumerate(objects)}
        actual['native_urdf_visual_version']=str(getattr(engine,'urdf_version','not_exposed'))
        for name in ('episode_source_obj','episode_target_obj'):
            obj=getattr(engine,name,None)
            if obj is not None:actual[name]=str(obj.name)
    fields={}
    if isinstance(obs,dict):
        for key in ('robot_obs','scene_obs'):
            if key in obs:fields[key]=np.asarray(obs[key]).tolist()
    elif hasattr(env,'obs'):fields['native_state']=np.asarray(env.obs).tolist()
    return dict(actual=actual,native_observation_fields=fields)


def verify_registered_initialization(case,env):
    """Check native reset honored the official options before any quality RPC."""
    if case['environment']!='simpler':return dict(status='native_nonGoogle_reset')
    engine=env.native.native;expected=case['initialization']
    if case['task']=='google_robot_pick_coke_can':
        actual=engine.obj_init_options
        if (not np.allclose(actual.get('init_xy',[]),expected['official_xy'],atol=1e-12,rtol=0)
                or actual.get('orientation')!=expected['official_orientation']):
            raise ValueError('Native Coke reset did not honor registered officialXY/orientation')
        return dict(status='registered_native_options_honored',requested_xy=expected['official_xy'],
            orientation=expected['official_orientation'],actual_pose_is_separately_measured=True)
    episode=env.native.reset_info.get('episode_id')
    ids=list(engine.episode_model_ids)
    if episode!=expected['official_episode_id'] or ids!=expected['object_ids']:
        raise ValueError('Native MoveNear episode/triplet differs from official registeredcase')
    if (engine.episode_source_obj.name!=ids[expected['source_object_index']] or
            engine.episode_target_obj.name!=ids[expected['target_object_index']]):
        raise ValueError('Native MoveNear source/target role differs from registered officialepisode')
    return dict(status='registered_native_options_honored',official_episode_id=episode,
                object_ids=ids,layout_index=expected['layout_index'],not_50_unique_geometry=True)


def decode_video(path,expected_frames,runner=subprocess.run):
    decoded=runner(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],capture_output=True)
    counted=runner(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries',
        'stream=nb_read_frames','-of','default=nokey=1:noprint_wrappers=1',str(path)],capture_output=True)
    try:frames=int(counted.stdout.decode().strip())
    except (ValueError,AttributeError):frames=None
    passed=decoded.returncode==0 and counted.returncode==0 and frames==expected_frames
    return dict(full_stream_decode_passed=bool(passed),decoded_frames=frames,expected_frames=expected_frames,
                full_decode_returncode=decoded.returncode,framecount_returncode=counted.returncode,
                error=decoded.stderr.decode(errors='replace') if decoded.returncode else None)


def run_attempt(args,registration,case,mode,directory,service,reference,request=rpc,adapter_factory=make_adapter):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();env=None;writer=None
    result=dict(status='running',scope=SCOPE,family='libero_transfer',registration_hash=registration['registration_hash'],
        case_id=case['case_id'],case_hash=digest(case),environment=case['environment'],mode=mode,
        service_manifest_hash=service['service_manifest_hash'],gpu_uuid=service['gpu_uuid'],
        counter_namespace=counter_namespace(registration),case=case,success=None,completed_subtasks=0,
        native_steps=0,requests=0,attempted_quality_requests=0,quality_responses=0,task_results=[],
        denoising=None if mode.startswith('draft_') else 10,H=50,R=5)
    write_json(directory/'result.json',result)
    def deadline():
        if time.time()>=args.deadline_unix:raise TimeoutError('Explicit B execution deadline elapsed')
    def capture(task_index,task_step,audit=None,oracle=None,terminal=False):
        append(directory/'telemetry.jsonl',dict(global_step=result['native_steps'],subtask_index=task_index,
            subtask_step=task_step,sim_time_s=result['native_steps']/case['native_hz'],
            executed_command=audit,oracle_success=oracle,native_terminated_or_truncated=terminal,**actual_record(env)))
        if writer is not None:writer.append_data(canonical_raw(env.raw())['base_rgb'])
    try:
        deadline();env=adapter_factory(case,reference,service)
        if case['environment']=='calvin':env.reset_sequence(case['sequence']['initial_condition'])
        elif case['environment']=='simpler':env.reset(case['seed'],options=case['reset_options'])
        else:env.reset(case['seed'])
        initial=canonical_raw(env.raw());np.savez_compressed(str(directory/'initial_raw.npz'),**initial)
        result.update(initial_raw_hash=raw_hash(initial),initial_actual=actual_record(env),adapter_metadata=env.info)
        result['initialization_audit']=verify_registered_initialization(case,env)
        if case['video']:
            import imageio
            writer=imageio.get_writer(str(directory/'video.mp4'),fps=case['native_hz'],codec='libx264',
                                      pixelformat='yuv420p',macro_block_size=2)
            result['video']=dict(path='video.mp4',fps=case['native_hz'],preregistered=True)
        capture(0,0)
        tasks=case['sequence']['tasks'] if case['environment']=='calvin' else [case.get('task','pusht_projection_stress')]
        for task_index,task in enumerate(tasks):
            request(args.port,dict(op='reset',registration_hash=registration['registration_hash']))
            if case['environment']=='calvin':instruction=env.start_subtask(task)
            elif case['environment']=='simpler':instruction=str(env.native.native.get_language_instruction())
            else:instruction=case['instruction']
            task_result=dict(index=task_index,task=task,instruction=instruction,steps=0,requests=0,success=False)
            result['task_results'].append(task_result)
            first=canonical_raw(env.raw());np.savez_compressed(str(directory/('subtask_%d_initial_raw.npz'%task_index)),**first)
            cap=case.get('max_steps_per_subtask',case.get('max_steps'))
            ended=False
            while task_result['steps']<cap and not task_result['success'] and not ended:
                deadline();raw=canonical_raw(env.raw())
                seed=request_seed(case['case_id'],task_index,task_result['requests'])
                result['attempted_quality_requests']+=1
                append(directory/'requests_sent.jsonl',dict(subtask_index=task_index,replan_index=task_result['requests'],
                    seed=seed,input_hash=input_digest(raw,instruction),registration_hash=registration['registration_hash'],
                    counter_namespace=counter_namespace(registration),category='quality',mode=mode))
                tick=time.perf_counter()
                response=request(args.port,dict(op='predict',family='libero_transfer',environment=case['environment'],mode=mode,
                    registration_hash=registration['registration_hash'],case_id=case['case_id'],
                    raw=raw,instruction=instruction,seed=seed,category='quality'))
                elapsed=time.perf_counter()-tick
                result['quality_responses']+=1
                append(directory/'rpc_responses.jsonl',dict(client_completed_s=elapsed,response=response))
                actions=validate_response(response,registration,mode,raw,instruction,seed,service['service_manifest_hash'],case)
                append(directory/'infer.jsonl',dict(index=result['requests'],subtask_index=task_index,subtask_step=task_result['steps'],
                    seed=seed,input_hash=input_digest(raw,instruction),raw_hash=raw_hash(raw),state8=raw['state8'].tolist(),
                    instruction=instruction,actions=actions.tolist(),client_completed_s=elapsed,
                    response={k:v for k,v in response.items() if k not in ('actions','raw_actions')},raw_actions=response['raw_actions']))
                if getattr(args,'save_observations',False):np.savez_compressed(str(directory/('request_%06d.npz'%result['requests'])),**raw)
                result['requests']+=1;task_result['requests']+=1
                for action in actions[:min(5,cap-task_result['steps'])]:
                    deadline();step_started=time.perf_counter();success,terminal,audit=env.step(action)
                    audit['environment_step_s']=time.perf_counter()-step_started
                    task_result['steps']+=1;result['native_steps']+=1
                    task_result['success']=bool(success);ended=bool(terminal)
                    capture(task_index,task_result['steps'],audit,bool(success),ended)
                    write_json(directory/'result.json',result)
                    if success or ended:break
            append(directory/'tasks.jsonl',task_result)
            if not task_result['success']:
                result['failed_task']=task;break
            result['completed_subtasks']+=1
        result.update(status='complete',success=result['completed_subtasks']==len(tasks))
    except Exception:
        result.update(status='technical_failure',success=None,error=traceback.format_exc())
    finally:
        if writer is not None:
            try:
                writer.close();result['video'].update(decode_video(directory/'video.mp4',result['native_steps']+1))
                if not result['video']['full_stream_decode_passed']:raise RuntimeError('Preregistered video full decode/frame count failed')
            except Exception:
                result.update(status='technical_failure',success=None,video_error=traceback.format_exc())
        if env is not None:
            try:env.close()
            except Exception:result.update(status='technical_failure',success=None,close_error=traceback.format_exc())
        result['artifact_sha256']={p.name:sha(p) for p in directory.iterdir() if p.is_file() and p.name!='result.json' and not p.name.endswith('.temporary')}
        result['wall_s']=time.perf_counter()-started;write_json(directory/'result.json',result)
    return result


def summarize(registration,selected,attempts):
    groups={}
    for case in registration['cases']:
        group='calvin_chains' if case['environment']=='calvin' else case.get('task','pusht_projection_stress')
        groups.setdefault(group,[]).append(case)
    metrics={}
    for group,cases in groups.items():
        metrics[group]={}
        for mode in MODES:
            rows=[selected[(c['case_id'],mode)] for c in cases if (c['case_id'],mode) in selected]
            row=dict(expected=len(cases),valid=len(rows),missing=len(cases)-len(rows),successes=sum(bool(r['success']) for r in rows),
                     success_rate=float(np.mean([r['success'] for r in rows])) if rows else None)
            if group=='calvin_chains':
                completed=[r['completed_subtasks'] for r in rows]
                row.update(mean_completed_chain_length=float(np.mean(completed)) if rows else None,
                    success_at_least={str(i):float(np.mean(np.asarray(completed)>=i)) if rows else None for i in range(1,6)})
            metrics[group][mode]=row
        full=metrics[group]['full_d10']
        metrics[group]['interpretation']='inconclusive: Fullbaseline has zero observed successes' if full['valid'] and full['successes']==0 else 'descriptive paired cases; no causal capacity inference'
    return dict(scope=SCOPE,registration_hash=registration['registration_hash'],expected_conditions=1800,
        valid_complete_conditions=len(selected),all_conditions_valid_complete=len(selected)==1800,
        technical_failure_attempts=sum(r['status']=='technical_failure' for r in attempts),attempt_count=len(attempts),
        metrics=metrics,pooled_capacity_score=None,pusht_is_projection_stress=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',required=True)
    p.add_argument('--environment',choices=('calvin','simpler','pusht'),required=True);p.add_argument('--output',required=True)
    p.add_argument('--authorization');p.add_argument('--port',type=int);p.add_argument('--deadline-unix',type=float)
    p.add_argument('--max-new-cases',type=int,default=1,help='Dispatch shard only; registered scientific target remains300×6')
    p.add_argument('--save-observations',action='store_true');p.add_argument('--launch',action='store_true');a=p.parse_args()
    registration=load_registration(a.registration,verify_sources=True)
    cases=[c for c in registration['cases'] if c['environment']==a.environment]
    if not a.launch:
        print(json.dumps(dict(dry_run=True,scope=SCOPE,registration_hash=registration['registration_hash'],
            environment=a.environment,registered_cases=len(cases),registered_conditions=len(cases)*6,all_conditions=1800,
            dispatch_chunk=a.max_new_cases,gpu_used=False,quality_requests=0)));return
    if not a.authorization or not a.port or a.max_new_cases<1:raise PermissionError('Explicit new B authorization/port/positive shard required')
    authorization=json.loads(Path(a.authorization).read_text())
    health=rpc(a.port,dict(op='health'));service=launch_gate(registration,authorization,health,a.environment,a.deadline_unix)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True);saved=root/'registration.json'
    if saved.exists():
        if json.loads(saved.read_text())!=registration:raise ValueError('Existing output belongs to different scientific registration')
    else:
        with saved.open('x') as stream:json.dump(registration,stream,ensure_ascii=False,indent=2)
    session=root/('session_%s_%d.json'%(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'),os.getpid()))
    write_json(session,dict(status='authorized_B_dispatch',service=service,authorization_sha256=sha(a.authorization),
                            environment=a.environment,deadline_unix=a.deadline_unix))
    reference=json.loads(Path(registration['input_files']['source_reference']['path']).read_text())
    selected,attempts=complete_results(root,registration);new_cases=0
    for case in cases:
        missing=[m for m in MODES if (case['case_id'],m) not in selected]
        if not missing:continue
        if new_cases>=a.max_new_cases:break
        existing=[selected[(case['case_id'],m)] for m in MODES if (case['case_id'],m) in selected]
        if any(r['gpu_uuid']!=service['gpu_uuid'] for r in existing):raise PermissionError('Allsixconditions for a case must stay on same physical GPU')
        new_cases+=1
        for mode in missing:
            directory=next_attempt(root/'episodes'/case['environment']/case['case_id']/mode)
            result=run_attempt(a,registration,case,mode,directory,service,reference)
            selected,attempts=complete_results(root,registration);write_json(root/'progress.json',summarize(registration,selected,attempts))
            if result['status']!='complete':raise RuntimeError('Technical failure preserved; wholecase/mode retry gets a new attempt, not policy0')
    write_json(session,dict(status='dispatch_shard_finished_not_necessarily_science_complete',service=service,
                           new_cases=new_cases,summary=summarize(registration,selected,attempts)))


if __name__=='__main__':main()

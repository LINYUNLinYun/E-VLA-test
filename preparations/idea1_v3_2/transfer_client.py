"""Python3.8-compatible native transfer rollout and CPU adapter preflight."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import socket
import struct
import subprocess
import time
import traceback
import numpy as np
from .transfer_envs import ROOT, MODES, case_list, make_environment, checked_source, sha


def serial(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    raise TypeError(type(value).__name__)


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.%d.temporary'%os.getpid())
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=serial,allow_nan=False))
    temporary.replace(path)


def receive(connection,size):
    data=bytearray()
    while len(data)<size:
        part=connection.recv(size-len(data))
        if not part: raise ConnectionError('Incomplete localhost response')
        data.extend(part)
    return bytes(data)


def rpc(port,request,timeout=180):
    payload=pickle.dumps(request,protocol=4)
    with socket.create_connection(('127.0.0.1',port),timeout=timeout) as connection:
        connection.sendall(struct.pack('!I',len(payload))+payload)
        size=struct.unpack('!I',receive(connection,4))[0]
        if size>128*1024*1024: raise ValueError('Oversized RPC response')
        response=pickle.loads(receive(connection,size))
    if not response.get('ok'): raise RuntimeError(response.get('error',response))
    return response


def raw_hash(raw):
    digest=hashlib.sha256()
    for key in sorted(raw):
        value=np.ascontiguousarray(raw[key])
        digest.update(key.encode());digest.update(str(value.dtype).encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def noise_seed(environment,case,index):
    payload=json.dumps(['idea1-v3.2-E4',environment,int(case),int(index)],separators=(',',':')).encode()
    return int(hashlib.sha256(payload).hexdigest()[:8],16)


def validate_raw(raw):
    if set(raw)!=set(('base_rgb','wrist_rgb','state8')): raise ValueError('Raw schema differs')
    for key in ('base_rgb','wrist_rgb'):
        image=np.asarray(raw[key])
        if image.ndim!=3 or image.shape[-1]!=3 or image.dtype!=np.uint8: raise ValueError('Invalid upright RGB '+key)
    state=np.asarray(raw['state8'])
    if state.shape!=(8,) or not np.isfinite(state).all(): raise ValueError('Invalid state8')


def extract_source(args):
    """Real LIBERO simulator, no cameras/offscreen renderer; never constructs a GPU renderer."""
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    from preparations.idea1_v2_1.runtime_client import official_init_states
    from scipy.spatial.transform import Rotation
    selected=benchmark.get_benchmark_dict()['libero_spatial']()
    task=selected.get_task(0)
    bddl=Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
    env=ControlEnv(str(bddl),use_camera_obs=False,has_renderer=False,has_offscreen_renderer=False,hard_reset=False)
    try:
        env.seed(7);env.reset()
        vector=np.asarray(official_init_states(selected,0)[0])
        obs=env.set_init_state(vector.copy())
        for _ in range(10): obs,_,_,_=env.step([0.]*6+[-1.])
        robot=env.robots[0]
        body=robot.robot_model.root_body
        index=env.sim.model.body_name2id(body)
        base=np.eye(4)
        base[:3,3]=env.sim.data.body_xpos[index]
        base[:3,:3]=env.sim.data.body_xmat[index].reshape(3,3)
        state=np.r_[obs['robot0_eef_pos'],Rotation.from_quat(obs['robot0_eef_quat']).as_rotvec(),obs['robot0_gripper_qpos']]
        controller=robot.controller
        result=dict(status='complete',source='actual camera-free native LIBERO simulator',
            source_case=dict(suite='libero_spatial',task=0,init=0,env_seed=7,settle_steps=10),
            robot_class=type(robot.robot_model).__name__,robot_root_body=body,
            robot_base_world_matrix=base.tolist(),reference_state8=state.tolist(),
            initial_vector_sha256=hashlib.sha256(vector.tobytes()).hexdigest(),
            bddl_path=str(bddl),bddl_sha256=sha(bddl),
            actual_eef_world_xyz=np.asarray(obs['robot0_eef_pos']).tolist(),
            actual_eef_world_quaternion_xyzw=np.asarray(obs['robot0_eef_quat']).tolist(),
            actual_gripper_qpos=np.asarray(obs['robot0_gripper_qpos']).tolist(),
            controller={k:np.asarray(getattr(controller,k)).tolist() for k in ('input_min','input_max','output_min','output_max')},
            camera_obs=False,has_offscreen_renderer=False,gpu_used=False,
            tcp_alignment_assumption='identity TCP offset between embodiments; not estimated from target outcomes')
        checked_source(result)
        target=Path(args.output)
        if target.suffix!='.json': target=target/'source_reference.json'
        write_json(target,result)
        print(json.dumps(dict(status='complete',output=str(target),base=base.tolist(),state8=state.tolist())))
    finally: env.close()


def case_and_source(args):
    reference_path=Path(args.source_reference)
    reference=json.loads(reference_path.read_text())
    checked_source(reference)
    rows=case_list(args.environment)
    if not 0<=args.case<len(rows): raise ValueError('Case outside predefined set')
    return rows[args.case],reference


def preflight(args):
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    case,reference=case_and_source(args)
    receipt=dict(status='running',environment=args.environment,case=args.case,gpu_uuid=args.gpu_uuid,
        source_reference_sha256=sha(args.source_reference),cases=case_list(args.environment),
        source_files={str(Path(__file__).relative_to(ROOT)):sha(__file__),
                      'preparations/idea1_v3_2/transfer_envs.py':sha(ROOT/'preparations/idea1_v3_2/transfer_envs.py')})
    adapter=None
    try:
        adapter=make_environment(args.environment,case,reference,gpu=args.gpu,gpu_uuid=args.gpu_uuid,require_idle=True)
        tested=[]
        selected_cases=case_list(args.environment) if args.all_cases else [case]
        for selected_case in selected_cases:
            adapter.case=selected_case
            raw=adapter.reset();validate_raw(raw)
            before=adapter.telemetry()
            success,terminal,audit=adapter.step(np.zeros(7))
            validate_raw(adapter.raw())
            np.savez_compressed(str(output/('initial_raw_case%02d.npz'%selected_case['case'])),**raw)
            if selected_case['case']==args.case:
                np.savez_compressed(str(output/'initial_raw.npz'),**raw)
            tested.append(dict(case=selected_case['case'],initial_raw_sha256=raw_hash(raw),initial_actual_state=before,
                noop_interface_success=bool(success),noop_native_terminal=bool(terminal),noop_audit=audit))
        row=dict(status='supported',success_oracle=adapter.info.get('oracle'),metadata=adapter.info,
                 tested_cases=tested,all_predeclared_cases_tested=len(tested)==len(case_list(args.environment)),
                 upright_images=True,no_libero_image_flip=True)
        receipt.update(status='complete',environments={args.environment:row})
    except Exception:
        receipt.update(status='complete',environments={args.environment:dict(status='unsupported',
            reason=traceback.format_exc(),success_oracle=None)})
    finally:
        if adapter is not None: adapter.close()
        write_json(output/'preflight.json',receipt)
    print(json.dumps(dict(status=receipt['status'],environment=args.environment,
        support=receipt['environments'][args.environment]['status'],receipt=str(output/'preflight.json'))))


def load_and_check_manifest(args,case):
    if not args.manifest: raise ValueError('Rollout requires immutable --manifest')
    # Local implementation remains compatible with the legacy Python3.8 venv.
    manifest=json.loads(Path(args.manifest).read_text())
    payload={k:v for k,v in manifest.items() if k!='manifest_hash'}
    expected=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if manifest.get('manifest_hash')!=expected: raise ValueError('Manifest hash mismatch')
    selected=[r for r in manifest['cases'] if r['environment']==args.environment and r['case']==args.case]
    if len(selected)!=1 or args.mode not in manifest['modes']: raise ValueError('Case/mode not frozen')
    for field in ('seed','task','instruction','max_steps','native_hz','start_frame','end_frame'):
        if field in case and selected[0].get(field)!=case[field]:
            raise ValueError('Native case field differs from manifest: '+field)
    reference_path=Path(manifest['source_reference_path'])
    if sha(args.source_reference)!=sha(reference_path): raise ValueError('Source reference differs from frozen manifest')
    for filename,value in manifest['source_files'].items():
        if sha(ROOT/filename)!=value: raise ValueError('Frozen source changed: '+filename)
    if args.gpu is None or int(selected[0]['gpu'])!=args.gpu: raise ValueError('Assigned GPU differs from case shard')
    return manifest


def rollout(args):
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'result.json').exists(): raise FileExistsError('Never overwrite a completed/failed attempt result')
    case,reference=case_and_source(args)
    manifest=load_and_check_manifest(args,case)
    result=dict(status='running',environment=args.environment,case=args.case,mode=args.mode,
        manifest_hash=manifest['manifest_hash'],gpu_uuid=args.gpu_uuid,seed=case['seed'],success=None,
        steps=0,telemetry_path=str(output/'telemetry.jsonl'),inference_path=str(output/'infer.jsonl'),
        video_path=str(output/'video.mp4') if args.video else None,source_reference_sha256=sha(args.source_reference),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),process_pid=os.getpid())
    write_json(output/'result.json',result)
    adapter,writer=None,None
    begin=time.perf_counter()
    telemetry,infers=[],[]
    try:
        health=rpc(args.port,dict(op='health'))
        protocol=health.get('protocol',health)
        if protocol.get('manifest_hash') not in (None,manifest['manifest_hash']): raise ValueError('Service manifest mismatch')
        result['service_protocol']=protocol
        tick=time.perf_counter()
        adapter=make_environment(args.environment,case,reference,gpu=args.gpu,gpu_uuid=args.gpu_uuid)
        raw=adapter.reset();validate_raw(raw)
        result['environment_setup_s']=time.perf_counter()-tick
        result['environment_info']=adapter.info
        result['initial_raw_hash']=raw_hash(raw)
        np.savez_compressed(str(output/'initial_raw.npz'),**raw)
        if args.video:
            import imageio
            writer=imageio.get_writer(str(output/'video.mp4'),fps=adapter.hz,codec='libx264',pixelformat='yuv420p',macro_block_size=2)
            result['video_fps']=adapter.hz
            result['video_fps_kind']='environment native control Hz'
        def capture(step,action=None):
            row=dict(step=step,sim_time_s=step/adapter.hz,actual=adapter.telemetry(),executed_command=action)
            telemetry.append(row)
            if writer is not None: writer.append_data(adapter.raw()['base_rgb'])
        capture(0)
        done=False
        cap=case['max_steps']
        while result['steps']<cap and not done:
            if args.deadline_unix and time.time()>=args.deadline_unix: raise TimeoutError('E4 deadline')
            raw=adapter.raw();validate_raw(raw)
            seed=noise_seed(args.environment,args.case,len(infers))
            tick=time.perf_counter()
            response=rpc(args.port,dict(op='predict',mode=args.mode,raw=raw,instruction=case['instruction'],seed=seed,category='quality'))
            elapsed=time.perf_counter()-tick
            actions=np.asarray(response.get('actions'),dtype=np.float64)
            if actions.shape!=(50,7) or not np.isfinite(actions).all(): raise ValueError('Invalid50x7 policy output')
            row=dict(request=len(infers),step=result['steps'],seed=seed,raw_hash=raw_hash(raw),
                client_completed_s=elapsed,response={k:v for k,v in response.items() if k!='actions'},actions=actions.tolist())
            infers.append(row)
            for action in actions[:min(5,cap-result['steps'])]:
                success,terminal,audit=adapter.step(action)
                result['steps']+=1
                capture(result['steps'],audit)
                if success or terminal:
                    result['success']=bool(success)
                    result['termination']='native_success' if success else 'native_terminal_or_truncated'
                    done=True;break
        if not done: result.update(success=False,termination='predeclared_step_cap')
        result.update(status='complete',predictions=len(infers),wall_s=time.perf_counter()-begin)
    except Exception:
        result.update(status='technical_failure',success=None,termination='technical_exception',error=traceback.format_exc(),wall_s=time.perf_counter()-begin)
    finally:
        for filename,rows in (('telemetry.jsonl',telemetry),('infer.jsonl',infers)):
            (output/filename).write_text(''.join(json.dumps(r,default=serial,allow_nan=False)+'\n' for r in rows))
        if writer is not None:
            writer.close()
            decode=subprocess.run(['ffmpeg','-v','error','-i',str(output/'video.mp4'),'-f','null','-'],capture_output=True)
            result['video_decode']=dict(passed=decode.returncode==0,returncode=decode.returncode,error=decode.stderr.decode(errors='replace'))
            if decode.returncode:
                result.update(status='technical_failure',success=None,termination='video_decode_failure')
        if adapter is not None: adapter.close()
        write_json(output/'result.json',result)
    print(json.dumps({k:result.get(k) for k in ('status','environment','case','mode','success','steps','wall_s')}))
    if result['status']!='complete': raise RuntimeError('Transfer attempt technical failure; result preserved')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--environment',choices=('calvin','simpler','pusht'))
    parser.add_argument('--case',type=int,default=0)
    parser.add_argument('--mode',choices=MODES,default='full_d10')
    parser.add_argument('--port',type=int)
    parser.add_argument('--source-reference')
    parser.add_argument('--manifest')
    parser.add_argument('--deadline-unix',type=float)
    parser.add_argument('--output',required=True)
    parser.add_argument('--gpu',type=int)
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--video',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--all-cases',action='store_true')
    parser.add_argument('--extract-source',action='store_true')
    parser.add_argument('--launch',action='store_true')
    args=parser.parse_args()
    if not args.launch:
        print(json.dumps(dict(dry_run=True,**vars(args))));return
    if args.extract_source: extract_source(args)
    elif args.preflight: preflight(args)
    else: rollout(args)


if __name__=='__main__': main()

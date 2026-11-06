"""Causal R5 action execution during the next B1/D10/H10 inference request.

This deliberately changes the synchronous control protocol: after bootstrap,
the previous chunk is applied while the next prediction is in flight. Actions
therefore have a measured one-chunk observation lag. No future frame is invented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from preparations.idea1_v2_1 import runtime_client as original
from results.idea2.online_encoder_20261025 import wire

ARMS=("fresh18_no_encoder","fresh18_encoder","sm16_no_encoder","sm16_encoder")
CLIENT_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def model_observation(obs):
    # MuJoCo can expose views; freeze this actually captured observation before
    # advancing the simulator or handing it to the other process.
    return {k:np.array(v,copy=True) for k,v in obs.items() if isinstance(v,np.ndarray)}


def json_default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,default=json_default)+"\n");temporary.replace(path)


def write_jsonl(path,rows):
    with Path(path).open("w") as stream:
        for row in rows:stream.write(json.dumps(row,default=json_default)+"\n")


def protocol_check(protocol,arm):
    if (protocol.get("num_steps"),protocol.get("H"),protocol.get("R"))!=(10,10,5):
        raise ValueError("Online trial requires original D10/H10/R5")
    if protocol.get("arm")!=arm or protocol.get("execution")!="online_progressive_cuda_graph":
        raise ValueError("Online arm/execution mismatch")
    if protocol.get("observation_lag_control_steps")!=5:
        raise ValueError("Explicit one-chunk lag protocol missing")
    if protocol.get("cache_slots")!=2:raise ValueError("Expected two full KV slots")


def checked_actions(response,request,instruction,arm):
    if not response.get("ok"):raise RuntimeError(str(response))
    if response["input_hash"]!=original.input_digest(request["obs"],instruction):
        raise ValueError("Returned input hash differs from the actual submitted observation")
    if response["seed"]!=request["seed"]:raise ValueError("Noise seed changed")
    if response.get("request_id",response.get("request"))!=request["request_id"]:
        raise ValueError("Prediction request identity changed")
    protocol_check(response["protocol"],arm)
    actions=np.asarray(response["actions"],dtype=np.float32)
    normalized=np.asarray(response["normalized_actions"])
    if actions.shape!=(10,7) or normalized.shape!=(10,32) or not np.isfinite(actions).all() or not np.isfinite(normalized).all():
        raise ValueError("Invalid full action chunk")
    if response.get("graph_kv_buffers")!=2:raise ValueError("Full KV slot count changed")
    if (response.get("generated_length"),response.get("execute_length"))!=(10,5):
        raise ValueError("Returned H10/R5 consumption contract changed")
    return actions


def exchange_round(connection,request,previous_record,previous_actions,obs,step,limit,step_fn,capture_fn):
    """CPU-testable state machine. No action may run before advance arrives."""
    if previous_record["executed_length"] or previous_record.get("terminal_discarded"):
        raise RuntimeError("An action chunk cannot be consumed twice")
    if request["input_step"]!=step:raise ValueError("Current policy input must be captured at the current control step")
    begin=time.perf_counter();wire.send(connection,request)
    advance=wire.receive(connection);advance_host=time.perf_counter()
    if not advance.get("ok") or advance.get("event")!="advance" or advance.get("request_id")!=request["request_id"]:
        raise RuntimeError("Invalid advance handshake; no actions were executed")
    start_step=step;done=False;capture_host=advance_host
    lag=start_step-previous_record["input_step"]
    expected_lag=0 if previous_record["request_id"]==0 else 5
    if lag!=expected_lag:raise ValueError("Previous-chunk observation lag changed")
    previous_record.update(execution_start_step=start_step,execution_start_observation_lag_steps=lag,
        execution_start_observation_lag_nominal_ms=lag*50,consumed_during_request_id=request["request_id"],
        execution_start_host=time.perf_counter())
    try:
        for action_index,action in enumerate(previous_actions[:min(5,limit-step)]):
            executed=np.array(action,dtype=np.float32,copy=True)
            action_begin=time.perf_counter();obs,_,done,_=step_fn(executed.tolist());capture_host=time.perf_counter()
            step+=1;previous_record["executed_length"]+=1
            capture_fn(step,obs,executed,dict(consumed_request_id=previous_record["request_id"],
                consumed_seed=previous_record["seed"],consumed_input_step=previous_record["input_step"],
                action_index=action_index,inflight_request_id=request["request_id"],
                action_sha256=original.value_digest(executed),capture_host=capture_host,
                environment_step_ms=(capture_host-action_begin)*1000,
                observation_lag_steps=step-1-previous_record["input_step"]))
            if done:break
    except BaseException:
        # Release the server's blocking exchange before propagating the local
        # technical failure. Do not continue this episode or retry actions.
        try:wire.send(connection,dict(ok=False,event="observation",request_id=request["request_id"],capture_step=step,done=False,has_next=False))
        except Exception:pass
        raise
    previous_record["execution_end_step"]=step;previous_record["execution_end_host"]=time.perf_counter()
    actual=model_observation(obs);has_next=not done and step<limit
    message=dict(ok=True,event="observation",request_id=request["request_id"],obs=actual,
        instruction=request["instruction"],suite=request["suite"],capture_step=step,done=bool(done),has_next=has_next,
        capture_host=capture_host,input_hash=original.input_digest(actual,request["instruction"]))
    send_host=time.perf_counter();wire.send(connection,message)
    response=wire.receive(connection);return_host=time.perf_counter()
    timing=dict(request_id=request["request_id"],input_step=request["input_step"],advance_host=advance_host,
        capture_step=step,capture_host=capture_host,observation_send_host=send_host,final_response_host=return_host,
        request_to_advance_ms=(advance_host-begin)*1000,advance_to_capture_ms=(capture_host-advance_host)*1000,
        capture_to_send_ms=(send_host-capture_host)*1000,observation_send_to_response_ms=(return_host-send_host)*1000,
        client_roundtrip_ms=(return_host-begin)*1000,has_next=has_next,done=bool(done),
        actual_capture_hash=message["input_hash"],executed_previous_request_id=previous_record["request_id"])
    return obs,step,bool(done),actual,response,timing


def run_episode(args):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Environment client must have empty CUDA_VISIBLE_DEVICES")
    expected=os.environ.get("PROGRESSIVE_EXPECTED_CPUS")
    if expected and set(map(int,expected.split(",")))!=os.sched_getaffinity(0):raise ValueError("Client CPU affinity differs from runner plan")
    selected=benchmark.get_benchmark_dict()[args.suite]();definition=selected.get_task(args.task)
    init_states=original.official_init_states(selected,args.task)
    if not 0<=args.init<len(init_states):raise ValueError("Official init index unavailable")
    bddl=Path(get_libero_path("bddl_files"))/definition.problem_folder/definition.bddl_file
    init_path=Path(get_libero_path("init_states"))/definition.problem_folder/definition.init_states_file
    identity=dict(model="pi05",arm=args.arm,suite=args.suite,task=args.task,init=args.init,env_seed=7,
        init_vector_hash=hashlib.sha256(np.asarray(init_states[args.init]).tobytes()).hexdigest(),
        bddl_sha256=original.sha(bddl),init_file=str(init_path),init_file_sha256=original.sha(init_path),instruction=definition.language,
        D=10,H=10,R=5,settling_steps=10,action_limit=original.LIMITS[args.suite],client_sha256=CLIENT_SHA256,
        original_client_sha256=original.LOADED_CLIENT_SHA256,protocol="causal_previous_R5_during_next_prediction",
        semantic_change="One previously returned R5 chunk executes while the next prediction runs; nonbootstrap actions have5-step/250ms nominal observation lag")
    write_json(output/"identity.json",identity)
    env=None;states=[];infers=[];exchanges=[];executed_vectors=[];steps=0;done=False;io_s=0.;drained=False
    start=time.perf_counter();failure=None;result=None
    try:
        env=OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
        env.seed(7);env.reset();obs=env.set_init_state(init_states[args.init])
        for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
        if float(env.env.control_freq)!=20.:raise ValueError("Expected original20Hz LIBERO control")
        controller=env.env.robots[0].controller
        controller_record={k:np.asarray(getattr(controller,k)).tolist() for k in ("input_min","input_max","output_min","output_max")}
        if not(np.allclose(controller_record["input_min"],-1.) and np.allclose(controller_record["input_max"],1.) and
            np.allclose(controller_record["output_max"],[.05,.05,.05,.5,.5,.5]) and np.allclose(controller_record["output_min"],[-.05,-.05,-.05,-.5,-.5,-.5])):
            raise ValueError("OSC controller scales changed")
        original.rpc(args.port,dict(op="reset"));health=original.rpc(args.port,dict(op="health"))
        protocol_check(health["protocol"],args.arm);write_json(output/"native_protocol.json",health["protocol"])
        write_json(output/"environment_protocol.json",dict(controller=controller_record,control_frequency_hz=20.,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),egl_device=os.environ.get("MUJOCO_EGL_DEVICE_ID"),
            cpu_affinity=sorted(os.sched_getaffinity(0)),raw_health=health,wall_clock_paced=False))
        replay=original.telemetry_helpers();primary,target=replay._goal_object_names(env.env)
        initial={name:replay._object_state(env.env,name).get("pos") for name in primary}
        def capture(step,current_obs,action,metadata):
            tick=time.perf_counter()
            row=replay.snapshot(env,current_obs,step=step,when="initial" if step==0 else "post_action",
                action=None if action is None else np.asarray(action).tolist(),primary_names=primary,target_names=target,initial_positions=initial)
            for name in primary:
                try:
                    gripper,obj=env.env.robots[0].gripper,env.env.get_object(name)
                    row["robot_contacts"][name]=bool(env.env.check_contact(gripper,obj))
                    row["robot_grasps"][name]=bool(env.env._check_grasp(gripper=gripper,object_geoms=obj))
                except Exception:row["robot_contacts"][name]=row["robot_grasps"][name]=None
            arrays={k:np.asarray(v) for k,v in current_obs.items() if isinstance(v,np.ndarray)}
            row.update(metadata,sim_time_s=float(env.env.sim.data.time),wall_timestamp_s=time.time(),
                real_observation_hash=original.input_digest(arrays,definition.language),
                observation_summary={k:v.tolist() for k,v in arrays.items() if v.size<=64},
                telemetry_capture_ms=(time.perf_counter()-tick)*1000)
            states.append(row)
            if action is not None:executed_vectors.append(np.array(action,dtype=np.float32,copy=True))
        capture(0,obs,None,dict(capture_host=time.perf_counter(),consumed_request_id=None,consumed_seed=None))
        def prepare_request(request_id,current_obs,input_step,op):
            nonlocal io_s
            tick=time.perf_counter();arrays=model_observation(current_obs);seed=original.model_seed(args.suite,args.task,args.init,request_id)
            path=output/"inputs"/f"replan{request_id:04d}.npz";path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path,**arrays,instruction=np.asarray(definition.language),seed=np.asarray(seed))
            io_s+=time.perf_counter()-tick
            return dict(op=op,obs=arrays,instruction=definition.language,suite=args.suite,seed=seed,request_id=request_id,input_step=input_step),path
        def record_response(response,request,path,roundtrip,timing=None):
            nonlocal io_s
            actions=checked_actions(response,request,definition.language,args.arm)
            tick=time.perf_counter();prediction=output/f"prediction{request['request_id']:04d}.npz"
            np.savez_compressed(prediction,actions=actions,normalized_actions=np.asarray(response["normalized_actions"]),
                seed=np.asarray(request["seed"]),input_step=np.asarray(request["input_step"]))
            row={k:v for k,v in response.items() if k!="actions"}
            row.update(request_id=request["request_id"],replan=request["request_id"],step=request["input_step"],input_step=request["input_step"],
                seed=request["seed"],input_path=str(path),input_file_sha256=original.sha(path),prediction_path=str(prediction),
                prediction_sha256=original.sha(prediction),actions_sha256=original.value_digest(actions),chunk_actions=actions.tolist(),
                client_roundtrip_ms=roundtrip,executed_length=0,terminal_discarded=False,
                client_exchange=timing,record_io_deferred_until_response=True)
            infers.append(row);io_s+=time.perf_counter()-tick
            return row,actions
        request,path=prepare_request(0,obs,0,"predict");tick=time.perf_counter()
        response=original.rpc(args.port,request);roundtrip=(time.perf_counter()-tick)*1000
        previous,actions=record_response(response,request,path,roundtrip)
        if not previous.get("bootstrap") or previous["frontiers"]!=[18]*10:raise ValueError("Initial prediction is not a fresh bootstrap")
        # action0 is intentionally still unexecuted here.
        while steps<original.LIMITS[args.suite] and not done:
            request_id=len(infers);request,path=prepare_request(request_id,obs,steps,"predict_ahead")
            with socket.create_connection(("127.0.0.1",args.port),timeout=600) as connection:
                obs,steps,done,actual,response,timing=exchange_round(connection,request,previous,actions,obs,steps,
                    original.LIMITS[args.suite],env.step,capture)
            exchanges.append(timing)
            # Only the real end-of-chunk observation is persisted; no full RGB
            # dump per control step, and no compression before the server receives it.
            tick=time.perf_counter();capture_path=output/"lookahead"/f"request{request_id:04d}_step{steps:04d}.npz"
            capture_path.parent.mkdir(exist_ok=True)
            np.savez_compressed(capture_path,**actual,instruction=np.asarray(definition.language),capture_step=np.asarray(steps),
                next_seed=np.asarray(original.model_seed(args.suite,args.task,args.init,request_id+1) if timing["has_next"] else -1))
            timing.update(observation_path=str(capture_path),observation_file_sha256=original.sha(capture_path));io_s+=time.perf_counter()-tick
            previous,actions=record_response(response,request,path,timing["client_roundtrip_ms"],timing)
            if not timing["has_next"]:
                previous.update(terminal_discarded=True,discard_reason="Environment succeeded or reached the original action limit during the previous chunk")
                break
        replay.assign_semantic_phases(states)
        drain_begin=time.perf_counter();diagnostics=original.rpc(args.port,dict(op="drain"));drain_s=time.perf_counter()-drain_begin
        write_json(output/"gpu_records.json",diagnostics);drained=True
        gpu_records=diagnostics.get("records",[])
        if diagnostics.get("pending_ahead") or [r["online_request_id"] for r in gpu_records]!=list(range(1,len(infers))):
            raise RuntimeError("Unconsumed ahead observation or missing deferred GPU diagnostics")
        if sum(r["executed_length"] for r in infers)!=steps or any(r["executed_length"]>5 for r in infers):raise RuntimeError("R5 consumption accounting failed")
        if len(states)!=steps+1 or len(executed_vectors)!=steps:raise RuntimeError("Actual action/telemetry accounting failed")
        if sum(r["terminal_discarded"] for r in infers)!=1 or not infers[-1]["terminal_discarded"] or infers[-1]["executed_length"]:
            raise RuntimeError("Terminal prediction must be received and discarded")
        write_jsonl(output/"infer.jsonl",infers);write_jsonl(output/"telemetry.jsonl",states);write_jsonl(output/"exchanges.jsonl",exchanges)
        result=dict(status="valid",valid=True,success=bool(done),model="pi05",arm=args.arm,suite=args.suite,task=args.task,init=args.init,
            actions=steps,steps=steps,predictions=len(infers),online_requests=len(infers)-1,
            consumed_chunks=sum(r["executed_length"]>0 for r in infers),terminal_discarded_predictions=1,
            termination="success" if done else "action_limit",D=10,H=10,R=5,control_hz=20,video=None,
            wall_s=time.perf_counter()-start,record_io_s=io_s,diagnostic_drain_wall_s=drain_s,
            executed_actions_sha256=original.value_digest(np.asarray(executed_vectors,dtype=np.float32)),
            explicit_nonbootstrap_chunk_lag_control_steps=5,nominal_chunk_lag_ms=250,
            actual_wall_clock_pacing=False,protocol_change="Causal previous-R5 execution concurrent with the next inference, not the original synchronous rollout",
            no_synthetic_future_observations=True,client_sha256=CLIENT_SHA256,source_directory=str(output))
    except BaseException:
        failure=traceback.format_exc()
        result=dict(status="technical_failure",valid=False,success=None,arm=args.arm,suite=args.suite,task=args.task,init=args.init,
            actions=len(executed_vectors),steps=len(executed_vectors),predictions=len(infers),error=failure,client_sha256=CLIENT_SHA256)
        raise
    finally:
        if not drained:
            try:write_json(output/"gpu_records.json",original.rpc(args.port,dict(op="drain")))
            except Exception as error:
                if result is not None:result["diagnostic_drain_error"]=repr(error)
        if env is not None:
            try:env.close()
            except BaseException:
                if result is not None:
                    result.update(status="technical_failure",valid=False,success=None,environment_close_error=traceback.format_exc())
                write_jsonl(output/"infer.jsonl",infers);write_jsonl(output/"telemetry.jsonl",states)
                if result is not None:
                    result["all_work_wall_s"]=time.perf_counter()-start
                    write_json(output/"result.json",result);write_json(output/"complete.json",result)
                raise
        write_jsonl(output/"infer.jsonl",infers);write_jsonl(output/"telemetry.jsonl",states);write_jsonl(output/"exchanges.jsonl",exchanges)
        if result is not None:
            result["all_work_wall_s"]=time.perf_counter()-start
            write_json(output/"result.json",result);write_json(output/"complete.json",result)
    return result


def self_test():
    """Real local socket framing, fake server and fake simulator; no EGL/CUDA."""
    import threading
    a,b=socket.socketpair();a.settimeout(5);b.settimeout(5)
    events=[];errors=[];actions_applied=[]
    def response(request,value):
        return dict(ok=True,request=request["request_id"],actions=np.full((10,7),value,np.float32))
    def server():
        try:
            for request_id in (1,2):
                request=wire.receive(b);assert request["request_id"]==request_id
                assert request["input_step"]==(0 if request_id==1 else 5)
                events.append(("advance",request_id));wire.send(b,dict(ok=True,event="advance",request_id=request_id))
                observation=wire.receive(b)
                assert observation["capture_step"]==(5 if request_id==1 else 7)
                assert observation["has_next"]==(request_id==1)
                assert observation["obs"]["value"].item()==observation["capture_step"]
                events.append(("observation",request_id));wire.send(b,response(request,20 if request_id==1 else 30))
        except BaseException:errors.append(traceback.format_exc())
        finally:b.close()
    thread=threading.Thread(target=server);thread.start();step=0;obs=dict(value=np.asarray([0]))
    def action(value):
        nonlocal step
        assert events and events[-1][0]=="advance"
        actions_applied.append(value[0]);step+=1
        return dict(value=np.asarray([step])),0.,step==7,{}
    previous=dict(request_id=0,seed=original.model_seed("libero_object",0,0,0),input_step=0,executed_length=0)
    request=dict(op="predict_ahead",request_id=1,input_step=0,instruction="test",suite="libero_object",obs=obs)
    obs,count,done,actual,answer,timing=exchange_round(a,request,previous,np.full((10,7),10,np.float32),obs,0,100,action,lambda *args:None)
    assert count==5 and not done and previous["executed_length"]==5 and actions_applied==[10]*5
    previous=dict(request_id=1,seed=original.model_seed("libero_object",0,0,1),input_step=0,executed_length=0)
    request=dict(op="predict_ahead",request_id=2,input_step=5,instruction="test",suite="libero_object",obs=actual)
    obs,count,done,actual,answer,timing=exchange_round(a,request,previous,answer["actions"],obs,5,100,action,lambda *args:None)
    a.close();thread.join(timeout=5)
    assert not thread.is_alive() and not errors,errors
    assert count==7 and done and previous["executed_length"]==2 and previous["execution_start_observation_lag_steps"]==5
    assert actions_applied==[10]*5+[20]*2 and 30 not in actions_applied
    assert not timing["has_next"] and answer["request"]==2
    assert original.model_seed("libero_object",0,0,0)!=original.model_seed("libero_object",0,0,1)
    # A malformed/failed advance must fail closed before touching the simulator.
    c,d=socket.socketpair();c.settimeout(5);d.settimeout(5)
    wire.send(d,dict(ok=False,event="advance",request_id=1))
    before=len(actions_applied)
    try:
        exchange_round(c,dict(request_id=1,input_step=0),dict(request_id=0,input_step=0,executed_length=0),
            np.zeros((10,7),np.float32),{},0,100,action,lambda *args:None)
    except RuntimeError:pass
    else:raise AssertionError("Invalid advance was accepted")
    finally:c.close();d.close()
    assert len(actions_applied)==before
    return dict(status="passed",checks=["no action before advance","bootstrap action0 executed during request1","real obs5 admitted before action1 execution", "action1 executed during request2 with5-step lag","terminal truncates to2 actual actions","terminal response received but never executed","original per-request seed rule","invalid advance executes zero actions"],gpu_or_environment_started=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--port",type=int);p.add_argument("--suite",choices=original.SUITES)
    p.add_argument("--task",type=int);p.add_argument("--init",type=int)
    p.add_argument("--arm",choices=ARMS);p.add_argument("--output")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2));return
    if any(getattr(args,key) is None for key in ("port","suite","task","init","arm","output")):p.error("port/suite/task/init/arm/output are required")
    result=run_episode(args);print(json.dumps(result,indent=2))


if __name__=="__main__":main()

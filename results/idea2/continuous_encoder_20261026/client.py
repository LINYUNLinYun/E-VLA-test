"""One EGL owner at 20 Hz, one network worker, and a single candidate mailbox.

Only complete R5 chunks are admitted at block boundaries. The worker offers
real observations acquired after the current input, rather than waiting R5.
No environment call, image compression, or file write occurs in the worker.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from preparations.idea1_v2_1 import runtime_client as original
from results.idea2.online_encoder_20261025 import wire
from results.idea2.online_encoder_20261025.client import model_observation, write_json, write_jsonl

CLIENT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Frame:
    step: int
    captured: float
    obs: dict
    digest: str


@dataclass
class Candidate:
    record: dict
    actions: np.ndarray
    normalized: np.ndarray
    frame: Frame


class Mailbox:
    """All shared state has one lock; full observation references are bounded.

    Consumed frames retained by the caller are explicit archival data, not a
    producer queue. Replaced candidates immediately lose their raw-frame ref.
    """
    def __init__(self, frame):
        self.condition = threading.Condition()
        self.latest = frame
        self.active_chunk = 0
        self.candidate = None
        self.stopped = False
        self.error = None
        self.first_advance = None
        self.max_mailbox = 0
        self.discards = Counter()
        self.connection = None
        self.aborted = False

    def register_connection(self, connection):
        with self.condition:
            if self.aborted:raise RuntimeError("Network worker was aborted")
            if self.connection is not None:raise RuntimeError("More than one active RPC socket")
            self.connection = connection

    def unregister_connection(self, connection):
        with self.condition:
            if self.connection is connection:self.connection = None

    def abort(self, error):
        """Exceptional cleanup only; normal terminal must receive its response."""
        self.stop(error)
        with self.condition:
            self.aborted = True
            connection = self.connection
        if connection is not None:
            try:connection.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            try:connection.close()
            except OSError:pass

    def _discard(self, candidate, reason):
        candidate.record.update(disposition=reason, discarded_host=time.perf_counter())
        self.discards[reason] += 1

    def publish_frame(self, frame):
        with self.condition:
            if frame.step <= self.latest.step:raise ValueError("Camera steps must increase")
            self.latest = frame
            self.condition.notify_all()

    def offer(self, candidate):
        with self.condition:
            target = candidate.record["target_chunk_id"]
            if self.stopped:self._discard(candidate, "terminal")
            elif target <= self.active_chunk:self._discard(candidate, "expired")
            elif target != self.active_chunk + 1:raise ValueError("Candidate skipped an actor chunk")
            else:
                if self.candidate is not None:self._discard(self.candidate, "superseded")
                candidate.record["disposition"] = "mailbox"
                self.candidate = candidate
                self.max_mailbox = max(self.max_mailbox, 1)
            self.condition.notify_all()

    def take(self, target, timeout=120):
        begin = time.perf_counter()
        with self.condition:
            if target != self.active_chunk + 1:raise ValueError("Actor chunk order changed")
            while self.candidate is None:
                if self.error:raise RuntimeError(self.error)
                if self.stopped:raise RuntimeError("Stopped actor requested another action")
                remaining = timeout - (time.perf_counter() - begin)
                if remaining <= 0:raise TimeoutError("No next R5 result; no extra actions were executed")
                self.condition.wait(min(remaining, 1))
            candidate = self.candidate
            if candidate.record["target_chunk_id"] != target:raise RuntimeError("Expired result reached actor")
            self.candidate = None
            self.active_chunk = target
            candidate.record.update(disposition="consumed", selected_host=time.perf_counter(),
                                    boundary_wait_ms=(time.perf_counter()-begin)*1000)
            self.condition.notify_all()
            return candidate

    def next_input(self, pending, first, previous_step, timeout=120):
        begin = time.perf_counter()
        with self.condition:
            while True:
                if self.error:raise RuntimeError(self.error)
                if self.stopped:return None
                frame = pending if pending is not None else self.latest
                if first or frame.step > previous_step:
                    # The target and the input are observed under the SAME lock.
                    return frame, self.active_chunk + 1
                remaining = timeout - (time.perf_counter()-begin)
                if remaining <= 0:raise TimeoutError("No strictly newer real policy input")
                self.condition.wait(min(remaining, 1))

    def after_advance(self, input_step, timeout=.075):
        begin = time.perf_counter()
        with self.condition:
            if self.first_advance is None:self.first_advance = begin
            self.condition.notify_all()
            while not self.stopped and self.latest.step <= input_step:
                remaining = timeout - (time.perf_counter()-begin)
                if remaining <= 0:break
                self.condition.wait(remaining)
            if self.stopped:return None, True, "terminal"
            if self.latest.step > input_step:return self.latest, False, "new_observation"
            return None, False, "not_yet"

    def stop(self, error=None):
        with self.condition:
            self.stopped = True
            if error:self.error = error
            if self.candidate is not None:
                self._discard(self.candidate, "terminal");self.candidate = None
            self.condition.notify_all()


def validate_protocol(protocol, arm):
    if (protocol.get("num_steps"), protocol.get("H"), protocol.get("R"), protocol.get("cache_slots")) != (10,10,5,2):
        raise ValueError("Expected unchanged D10/H10/R5 and two KV slots")
    if protocol.get("arm") != arm:raise ValueError("Condition changed")
    if protocol.get("execution") != "continuous_progressive_cuda_graph":
        raise ValueError("Expected the continuous actor protocol, not fixed-lag rollout")


def make_candidate(response, request, frame, begin, arm):
    if not response.get("ok"):raise RuntimeError(str(response))
    validate_protocol(response["protocol"], arm)
    if response["input_hash"] != frame.digest or response["seed"] != request["seed"]:
        raise ValueError("Returned observation/noise identity changed")
    if response.get("request_id", response.get("request")) != request["request_id"]:
        raise ValueError("Response request identity changed")
    if response.get("target_chunk_id") != request["target_chunk_id"]:
        raise ValueError("Server did not echo target chunk identity")
    if response.get("input_step",request["input_step"]) != request["input_step"]:
        raise ValueError("Response input step changed")
    actions = np.array(response["actions"], dtype=np.float32, copy=True)
    normalized = np.array(response["normalized_actions"], copy=True)
    if actions.shape != (10,7) or normalized.shape != (10,32) or not np.isfinite(actions).all() or not np.isfinite(normalized).all():
        raise ValueError("Invalid H10 output")
    if response.get("graph_kv_buffers") != 2:raise ValueError("Two-slot constraint changed")
    if (response.get("generated_length"),response.get("execute_length")) != (10,5):
        raise ValueError("H10/R5 output contract changed")
    end = time.perf_counter()
    record = {k:v for k,v in response.items() if k not in ("actions","normalized_actions")}
    record.update(request_id=request["request_id"], target_chunk_id=request["target_chunk_id"],
        input_step=frame.step, step=frame.step, replan=request["target_chunk_id"],
        seed=request["seed"], input_hash=frame.digest, input_capture_host=frame.captured,
        request_send_host=begin, response_host=end, client_roundtrip_ms=(end-begin)*1000,
        input_age_at_response_ms=(end-frame.captured)*1000, executed_length=0,
        actions_sha256=original.value_digest(actions), normalized_sha256=original.value_digest(normalized),
        chunk_actions=actions.tolist(), disposition="unadmitted")
    return Candidate(record, actions, normalized, frame)


def network_loop(args, mailbox, initial_frame, instruction, records, arrays):
    """Only this thread performs streamed RPC; it never calls the simulator."""
    pending = None
    previous_step = -1
    request_id = 1
    try:
        while True:
            selected = mailbox.next_input(pending, request_id == 1, previous_step)
            if selected is None:break
            frame, target = selected
            if request_id == 1:
                # Bootstrap is explicitly the only permitted repeated input.
                if frame is not initial_frame or frame.step != 0:raise ValueError("First online frame must be bootstrap")
            if request_id > 1 and frame.step <= previous_step:raise ValueError("Policy input did not advance")
            consumed_pending = pending is not None
            pending = None
            request = dict(op="predict_ahead", obs=frame.obs, instruction=instruction, suite=args.suite,
                seed=original.model_seed(args.suite,args.task,args.init,target), request_id=request_id,
                task=args.task,init=args.init,target_chunk_id=target,input_step=frame.step,
                input_capture_host=frame.captured)
            begin = time.perf_counter()
            with socket.create_connection(("127.0.0.1",args.port), timeout=5) as connection:
                mailbox.register_connection(connection)
                try:
                    connection.settimeout(180)
                    wire.send(connection, request)
                    advance = wire.receive(connection)
                    advance_host = time.perf_counter()
                    if not advance.get("ok") or advance.get("event") != "advance" or advance.get("request_id") != request_id:
                        raise RuntimeError("Invalid advance handshake")
                    ahead, terminal, reason = mailbox.after_advance(frame.step)
                    packet = dict(ok=True,event="observation",request_id=request_id,
                        target_chunk_id=target,has_next=ahead is not None,terminal=terminal,
                        done=terminal,reason=reason,suite=args.suite,instruction=instruction)
                    with mailbox.condition:
                        packet["capture_step"] = ahead.step if ahead is not None else mailbox.latest.step
                        step_at_send = mailbox.latest.step
                    if ahead is not None:
                        packet.update(obs=ahead.obs,input_hash=ahead.digest,capture_host=ahead.captured)
                    send_host = time.perf_counter();wire.send(connection,packet)
                    response = wire.receive(connection)
                finally:mailbox.unregister_connection(connection)
            candidate = make_candidate(response,request,frame,begin,args.arm)
            candidate.record.update(consumed_previous_accepted_observation=consumed_pending,
                advance_host=advance_host,observation_send_host=send_host,
                advance_to_observation_send_ms=(send_host-advance_host)*1000,
                offered_next_input_step=None if ahead is None else ahead.step,
                offered_next_input_hash=None if ahead is None else ahead.digest,
                next_observation_reason=reason,terminal_exchange=terminal,
                actor_step_at_observation_send=step_at_send)
            records.append(candidate.record)
            arrays.append((candidate.actions,candidate.normalized))
            mailbox.offer(candidate)
            # Retain precisely the raw observation accepted by the server.
            pending = ahead
            previous_step = frame.step
            request_id += 1
            if terminal:break
    except BaseException:
        mailbox.stop(traceback.format_exc())


def execute_actor(mailbox, bootstrap, limit, step_fn, publish_capture, period=.05, chunks=None):
    """Runs only on the environment owner thread; never consumes H10's tail."""
    if chunks is None:chunks=[]
    chunks.append(bootstrap)
    bootstrap.record.update(disposition="consumed",selected_host=time.perf_counter(),boundary_wait_ms=0.)
    with mailbox.condition:
        deadline = time.perf_counter()+120
        while mailbox.first_advance is None:
            if mailbox.error:raise RuntimeError(mailbox.error)
            if time.perf_counter()>deadline:raise TimeoutError("First inference did not advance")
            mailbox.condition.wait(.1)
        next_deadline = mailbox.first_advance
    step = 0;done = False;chunk = bootstrap
    try:
        while step < limit and not done:
            chunk.record.update(execution_start_step=step,
                observation_age_at_selection_steps=step-chunk.frame.step)
            for index in range(min(5,limit-step)):
                with mailbox.condition:
                    if mailbox.error:raise RuntimeError(mailbox.error)
                now = time.perf_counter()
                if next_deadline>now:time.sleep(next_deadline-now)
                action = np.array(chunk.actions[index],dtype=np.float32,copy=True)
                application = time.perf_counter()
                if index == 0:chunk.record["execution_start_host"] = application
                obs,_,done,_ = step_fn(action.tolist())
                captured = time.perf_counter();step += 1
                chunk.record["executed_length"] += 1
                publish_capture(step,obs,action,dict(chunk_id=chunk.record["target_chunk_id"],
                    consumed_request_id=chunk.record["request_id"],seed=chunk.record["seed"],action_index=index,
                    input_step=chunk.frame.step,input_capture_host=chunk.frame.captured,
                    prediction_response_host=chunk.record["response_host"],application_host=application,
                    capture_host=captured,scheduled_application_host=next_deadline,
                    deadline_lateness_ms=max(0.,application-next_deadline)*1000,
                    environment_step_ms=(captured-application)*1000,
                    observation_age_steps=step-1-chunk.frame.step,
                    observation_age_ms=(application-chunk.frame.captured)*1000,
                    response_to_application_ms=(application-chunk.record["response_host"])*1000,
                    action_sha256=original.value_digest(action)))
                # Never catch up by running controls faster than 20 Hz.
                next_deadline = max(next_deadline+period,application+period)
                if done:break
            chunk.record.update(execution_end_step=step,execution_end_host=time.perf_counter())
            if done or step>=limit:break
            original_deadline = next_deadline
            # Select at the NEXT control tick, not immediately after step5's
            # render returns. A newer candidate arriving between those times
            # must remain eligible for the new chunk.
            remaining = original_deadline-time.perf_counter()
            if remaining>0:time.sleep(remaining)
            chunk = mailbox.take(chunk.record["target_chunk_id"]+1)
            selected = time.perf_counter()
            chunk.record["boundary_deadline_gap_ms"] = max(0.,selected-original_deadline)*1000
            chunks.append(chunk)
            # A missing result causes an explicit pause; it never extends R5.
            next_deadline = max(original_deadline,selected)
    finally:
        mailbox.stop()
    return step,bool(done),chunks


def run_episode(args):
    from libero.libero import benchmark,get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    out = Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":raise ValueError("Client must hide CUDA devices; EGL mapping is independent")
    expected = os.environ.get("PROGRESSIVE_EXPECTED_CPUS")
    if expected and set(map(int,expected.split(","))) != os.sched_getaffinity(0):raise ValueError("Client CPU affinity changed")
    suite = benchmark.get_benchmark_dict()[args.suite]()
    definition = suite.get_task(args.task)
    states = original.official_init_states(suite,args.task)
    if not 0<=args.init<len(states):raise ValueError("Invalid official init index")
    bddl = Path(get_libero_path("bddl_files"))/definition.problem_folder/definition.bddl_file
    init_path = Path(get_libero_path("init_states"))/definition.problem_folder/definition.init_states_file
    identity = dict(model="pi05",arm=args.arm,suite=args.suite,task=args.task,init=args.init,env_seed=7,
        instruction=definition.language,bddl_sha256=original.sha(bddl),init_file_sha256=original.sha(init_path),
        init_vector_hash=hashlib.sha256(np.asarray(states[args.init]).tobytes()).hexdigest(),
        D=10,H=10,R=5,settling_steps=10,action_limit=original.LIMITS[args.suite],
        client_sha256=CLIENT_SHA256,original_client_sha256=original.LOADED_CLIENT_SHA256,
        protocol="continuous_actor20Hz_latest_observation_single_mailbox",
        semantics="R5 fixed at selection; noise seed follows consumed chunk ID; latest completed matching-target candidate only")
    write_json(out/"identity.json",identity)
    begin = time.perf_counter();env = None;mailbox = None;worker = None
    records=[];arrays=[];telemetry=[];chunks=[];result=None;diagnostics=None
    try:
        env = OffScreenRenderEnv(bddl_file_name=str(bddl),camera_heights=256,camera_widths=256)
        env.seed(7);env.reset();obs = env.set_init_state(states[args.init])
        for _ in range(10):obs,_,_,_ = env.step([0.]*6+[-1.])
        controller = env.env.robots[0].controller
        controls = {k:np.asarray(getattr(controller,k)).tolist() for k in ("input_min","input_max","output_min","output_max")}
        if float(env.env.control_freq)!=20. or not(np.allclose(controls["input_min"],-1.) and np.allclose(controls["input_max"],1.) and
            np.allclose(controls["output_min"],[-.05,-.05,-.05,-.5,-.5,-.5]) and np.allclose(controls["output_max"],[.05,.05,.05,.5,.5,.5])):
            raise ValueError("Original OSC/20Hz controller changed")
        original.rpc(args.port,dict(op="reset"));health = original.rpc(args.port,dict(op="health"))
        validate_protocol(health["protocol"],args.arm)
        write_json(out/"native_protocol.json",health["protocol"])
        write_json(out/"environment_protocol.json",dict(controller=controls,control_frequency_hz=20,
            wall_clock_paced=True,no_catchup=True,cpu_affinity=sorted(os.sched_getaffinity(0)),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),egl_device=os.environ.get("MUJOCO_EGL_DEVICE_ID")))
        replay = original.telemetry_helpers();primary,target = replay._goal_object_names(env.env)
        initial_positions = {name:replay._object_state(env.env,name).get("pos") for name in primary}
        raw = model_observation(obs)
        initial_frame = Frame(0,time.perf_counter(),raw,original.input_digest(raw,definition.language))
        mailbox = Mailbox(initial_frame)
        owner = threading.get_ident()
        def capture(step,current,action,metadata):
            if threading.get_ident()!=owner:raise RuntimeError("MuJoCo/EGL left the owner thread")
            raw = model_observation(current)
            digest = original.input_digest(raw,definition.language)
            if step:mailbox.publish_frame(Frame(step,metadata["capture_host"],raw,digest))
            row = replay.snapshot(env,current,step=step,when="initial" if step==0 else "post_action",
                action=None if action is None else np.asarray(action).tolist(),primary_names=primary,
                target_names=target,initial_positions=initial_positions)
            for name in primary:
                try:
                    gripper,obj=env.env.robots[0].gripper,env.env.get_object(name)
                    row["robot_contacts"][name]=bool(env.env.check_contact(gripper,obj))
                    row["robot_grasps"][name]=bool(env.env._check_grasp(gripper=gripper,object_geoms=obj))
                except Exception:row["robot_contacts"][name]=row["robot_grasps"][name]=None
            row.update(metadata,real_observation_hash=digest,sim_time_s=float(env.env.sim.data.time),
                wall_timestamp_s=time.time(),observation_summary={k:v.tolist() for k,v in raw.items() if v.size<=64})
            telemetry.append(row)
        capture(0,obs,None,dict(capture_host=initial_frame.captured))
        request = dict(op="predict",obs=initial_frame.obs,instruction=definition.language,suite=args.suite,
            seed=original.model_seed(args.suite,args.task,args.init,0),request_id=0,target_chunk_id=0,input_step=0,
            task=args.task,init=args.init,input_capture_host=initial_frame.captured)
        tick=time.perf_counter();response=original.rpc(args.port,request)
        bootstrap=make_candidate(response,request,initial_frame,tick,args.arm)
        if not bootstrap.record.get("bootstrap") or bootstrap.record["frontiers"] != [18]*10:
            raise ValueError("Bootstrap is not the original full-current-KV action")
        records.append(bootstrap.record);arrays.append((bootstrap.actions,bootstrap.normalized))
        worker=threading.Thread(target=network_loop,args=(args,mailbox,initial_frame,definition.language,records,arrays),
                                name="continuous-network",daemon=True)
        worker.start();actor_begin=time.perf_counter()
        steps,success,chunks=execute_actor(mailbox,bootstrap,original.LIMITS[args.suite],env.step,capture,chunks=chunks)
        actor_end=time.perf_counter();worker.join(timeout=120)
        if worker.is_alive():raise TimeoutError("Network worker did not finish terminal response")
        if mailbox.error:raise RuntimeError(mailbox.error)
        tick=time.perf_counter();diagnostics=original.rpc(args.port,dict(op="drain"));drain_ms=(time.perf_counter()-tick)*1000
        # A ticket can remain when the actor terminates after a nonterminal offer.
        # Draining Events is required; resetting below explicitly discards it.
        record_ids=[r["online_request_id"] for r in diagnostics.get("records",[])]
        if record_ids != [r["request_id"] for r in records[1:]]:raise RuntimeError("Deferred GPU diagnostics missing")
        if sum(c.record["executed_length"] for c in chunks)!=steps or len(telemetry)!=steps+1:raise RuntimeError("Action accounting changed")
        if [c.record["target_chunk_id"] for c in chunks]!=list(range(len(chunks))):raise RuntimeError("Consumed chunk identities skipped")
        for chunk in chunks:
            if chunk.record["seed"]!=original.model_seed(args.suite,args.task,args.init,chunk.record["target_chunk_id"]):raise RuntimeError("Consumed noise seed changed")
            if not 1<=chunk.record["executed_length"]<=5:raise RuntimeError("R5 was extended or unconsumed")
        replay.assign_semantic_phases(telemetry)
        result=dict(status="valid",valid=True,success=success,arm=args.arm,suite=args.suite,task=args.task,init=args.init,
            actions=steps,steps=steps,predictions=len(records),online_requests=len(records)-1,consumed_chunks=len(chunks),
            disposition_counts=dict(Counter(r["disposition"] for r in records)),D=10,H=10,R=5,control_hz=20,
            wall_clock_paced=True,actor_wall_s=actor_end-actor_begin,diagnostic_drain_ms=drain_ms,
            termination="success" if success else "action_limit",mailbox_capacity=1,max_mailbox_observed=mailbox.max_mailbox,
            boundary_wait_ms=sum(c.record.get("boundary_wait_ms",0) for c in chunks),
            boundary_deadline_gap_ms=sum(c.record.get("boundary_deadline_gap_ms",0) for c in chunks),
            pending_ahead_at_drain=diagnostics.get("pending_ahead"),
            unused_ahead_at_terminal=bool(diagnostics.get("pending_ahead")),no_synthetic_future_observations=True,
            client_sha256=CLIENT_SHA256,source_directory=str(out),wall_s=time.perf_counter()-begin)
    except BaseException:
        result=dict(status="technical_failure",valid=False,success=None,arm=args.arm,suite=args.suite,task=args.task,init=args.init,
            steps=max(0,len(telemetry)-1),actions=max(0,len(telemetry)-1),error=traceback.format_exc(),client_sha256=CLIENT_SHA256)
        raise
    finally:
        if mailbox is not None:mailbox.stop()
        if worker is not None and worker.is_alive():
            mailbox.abort("Exceptional episode cleanup interrupted outstanding network RPC")
            worker.join(timeout=7)
        worker_done=worker is None or not worker.is_alive()
        if not worker_done:
            # Do not touch mutable worker records or close the owner's env while
            # an unjoined worker remains. Preserve only an independent receipt;
            # the nonzero process exit delegates cleanup to the launcher.
            emergency=dict(status="technical_failure",valid=False,success=None,arm=args.arm,
                suite=args.suite,task=args.task,init=args.init,worker_stopped=False,
                error="Network worker survived socket abort; archival and explicit env.close skipped",
                client_sha256=CLIENT_SHA256)
            write_json(out/"complete.json",emergency);write_json(out/"result.json",emergency)
            raise RuntimeError(emergency["error"])
        if worker_done:
            if diagnostics is None:
                try:diagnostics=original.rpc(args.port,dict(op="drain"))
                except Exception as error:
                    if result is not None:result["drain_error"]=repr(error)
            try:
                reset_receipt=original.rpc(args.port,dict(op="reset"));write_json(out/"reset_receipt.json",reset_receipt)
                if result is not None:result["terminal_reset_receipt"]=reset_receipt
            except Exception as error:
                if result is not None:result.update(status="technical_failure",valid=False,success=None,reset_error=repr(error))
        if diagnostics is not None:write_json(out/"gpu_records.json",diagnostics)
        close_error=None
        if env is not None:
            try:env.close()
            except Exception:
                close_error=traceback.format_exc()
                if result is not None:result.update(status="technical_failure",valid=False,success=None,environment_close_error=close_error)
        # All image writes are deferred until actor and worker have stopped.
        io_begin=time.perf_counter()
        (out/"inputs").mkdir(exist_ok=True)
        for chunk in chunks:
            row=chunk.record;path=out/"inputs"/f"chunk{row['target_chunk_id']:04d}_request{row['request_id']:04d}.npz"
            np.savez_compressed(path,**chunk.frame.obs,instruction=np.asarray(definition.language),seed=np.asarray(row["seed"]),input_step=np.asarray(chunk.frame.step))
            row.update(input_path=str(path),input_file_sha256=original.sha(path))
        for row,(actions,normalized) in zip(records,arrays):
            path=out/f"prediction{row['request_id']:04d}.npz"
            np.savez_compressed(path,actions=actions,normalized_actions=normalized,seed=np.asarray(row["seed"]),target_chunk_id=np.asarray(row["target_chunk_id"]))
            row.update(prediction_path=str(path),prediction_sha256=original.sha(path))
        write_jsonl(out/"infer.jsonl",records);write_jsonl(out/"telemetry.jsonl",telemetry)
        write_jsonl(out/"consumed_chunks.jsonl",[c.record for c in chunks])
        if result is not None:
            result.update(record_io_s=time.perf_counter()-io_begin,all_work_wall_s=time.perf_counter()-begin,worker_stopped=worker_done)
            write_json(out/"complete.json",result);write_json(out/"result.json",result)
        if close_error:raise RuntimeError(close_error)
    if not result["valid"]:raise RuntimeError("Episode technical cleanup failed")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port",type=int);parser.add_argument("--suite",choices=original.SUITES)
    parser.add_argument("--task",type=int);parser.add_argument("--init",type=int)
    parser.add_argument("--arm");parser.add_argument("--output")
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    if args.self_test:
        from results.idea2.continuous_encoder_20261026.cpu_selftest import run
        print(json.dumps(run(),indent=2));return
    if any(getattr(args,key) is None for key in ("port","suite","task","init","arm","output")):
        parser.error("port/suite/task/init/arm/output required")
    print(json.dumps(run_episode(args),indent=2))


if __name__ == "__main__":main()

"""Bounded CPU state-machine and real framed-socket tests; no CUDA/EGL imports."""
from types import SimpleNamespace
import socket
import threading
import time
import traceback

import numpy as np

from .client import Frame, Candidate, Mailbox, execute_actor, network_loop, original, wire


def frame(step):
    obs={"value":np.asarray([step],dtype=np.int64)}
    return Frame(step,time.perf_counter(),obs,original.input_digest(obs,"test"))


def candidate(target,request,source=None):
    source=source or frame(0)
    row=dict(request_id=request,target_chunk_id=target,input_step=source.step,
        seed=original.model_seed("libero_object",0,0,target),executed_length=0,
        response_host=time.perf_counter(),disposition="unadmitted")
    return Candidate(row,np.full((10,7),target,np.float32),np.zeros((10,32),np.float32),source)


def run():
    checks=[]
    box=Mailbox(frame(0));one=candidate(1,1);two=candidate(1,2)
    box.offer(one);box.offer(two)
    assert one.record["disposition"]=="superseded" and box.max_mailbox==1
    chosen=box.take(1);assert chosen is two
    late=candidate(1,3);box.offer(late)
    assert late.record["disposition"]=="expired" and box.candidate is None
    assert all(c.record["seed"]==original.model_seed("libero_object",0,0,1) for c in (one,two,late))
    checks+= ["single-slot latest matching target mailbox", "expired target is never consumed", "same target has fixed original noise seed"]
    pending=frame(1);box.publish_frame(frame(5))
    selected,target=box.next_input(pending,False,0)
    assert selected is pending and target==2
    checks.append("accepted raw Prepared input wins over newer camera frame")
    box.stop();terminal=candidate(2,4);box.offer(terminal)
    assert terminal.record["disposition"]=="terminal" and box.candidate is None
    checks.append("terminal candidates cannot produce extra actions")
    empty=Mailbox(frame(0));ahead,terminal,reason=empty.after_advance(0,timeout=.002)
    assert ahead is None and not terminal and reason=="not_yet"
    checks.append("missing camera frame is not_yet, not a synthetic future frame")

    # A candidate arriving after step5 returned but BEFORE the next 20Hz tick
    # must replace the earlier candidate before the actor selects its new R5.
    tick_box=Mailbox(frame(0));tick_box.first_advance=time.perf_counter()
    early=candidate(1,1);early.actions.fill(11);tick_box.offer(early)
    fifth=threading.Event();tick_actions=[];tick_count=[0]
    late=candidate(1,2);late.actions.fill(12)
    def late_offer():
        assert fifth.wait(2)
        time.sleep(.004)
        tick_box.offer(late)
    producer=threading.Thread(target=late_offer);producer.start()
    def tick_step(action):
        tick_count[0]+=1;tick_actions.append(action[0])
        return {},0.,tick_count[0]==6,{}
    def tick_capture(index,*unused):
        if index==5:fifth.set()
    _,_,tick_chunks=execute_actor(tick_box,candidate(0,0),100,tick_step,tick_capture,period=.02)
    producer.join(2)
    assert tick_actions==[0]*5+[12] and tick_chunks[1] is late
    assert early.record["disposition"]=="superseded"
    checks.append("next-chunk selection happens at actual next tick and admits intervening newer candidate")

    # Abort a genuinely blocked framed receive, join it, then inspect state.
    abort_box=Mailbox(frame(0));left,right=socket.socketpair();abort_box.register_connection(left)
    receiving=threading.Event();unblocked=threading.Event()
    def blocked_receive():
        receiving.set()
        try:wire.receive(left)
        except (OSError,EOFError,ConnectionError):unblocked.set()
        finally:abort_box.unregister_connection(left)
    blocked=threading.Thread(target=blocked_receive);blocked.start();assert receiving.wait(1)
    abort_box.abort("selftest exceptional cleanup");blocked.join(2);right.close()
    assert not blocked.is_alive() and unblocked.is_set() and abort_box.connection is None
    checks.append("exceptional socket shutdown unblocks receive and worker joins before archival")

    # Exercise the production network loop and actor against actual wire framing.
    listener=socket.socket();listener.bind(("127.0.0.1",0));listener.listen();listener.settimeout(3)
    port=listener.getsockname()[1]
    initial=frame(0);box=Mailbox(initial);records=[];arrays=[];server_errors=[];requests=[];offers=[]
    server_stop=threading.Event();advanced=threading.Event();physical_step=[0];executed=[];captures=[]
    def server():
        pending=None;previous=-1
        try:
            while not server_stop.is_set():
                try:connection,_=listener.accept()
                except socket.timeout:continue
                with connection:
                    connection.settimeout(3);req=wire.receive(connection);requests.append(req)
                    assert req["request_id"]==len(requests)
                    assert req["input_step"]>previous or (len(requests)==1 and req["input_step"]==0)
                    assert req["seed"]==original.model_seed(req["suite"],req["task"],req["init"],req["target_chunk_id"])
                    if pending is not None:
                        assert req["input_step"]==pending["capture_step"]
                        assert original.input_digest(req["obs"],"test")==pending["input_hash"]
                    previous=req["input_step"]
                    advanced.set();wire.send(connection,dict(ok=True,event="advance",request_id=req["request_id"]))
                    packet=wire.receive(connection);offers.append(packet)
                    if packet["has_next"]:
                        assert packet["capture_step"]>req["input_step"]
                        assert packet["obs"]["value"].item()==packet["capture_step"]
                        pending=packet
                    else:pending=None
                    if req["request_id"]==1:
                        # Deliberately miss the actor's next-block deadline.
                        time.sleep(.06)
                        assert physical_step[0]==5,"Actor extended R5 while no result existed"
                    else:time.sleep(.009)
                    target=req["target_chunk_id"]
                    wire.send(connection,dict(ok=True,request=req["request_id"],target_chunk_id=target,input_step=req["input_step"],
                        input_hash=original.input_digest(req["obs"],"test"),seed=req["seed"],
                        actions=np.full((10,7),target,np.float32),normalized_actions=np.zeros((10,32),np.float32),
                        protocol=dict(num_steps=10,H=10,R=5,cache_slots=2,arm="fake",execution="continuous_progressive_cuda_graph"),
                        graph_kv_buffers=2,generated_length=10,execute_length=5))
        except (OSError,EOFError):
            if not server_stop.is_set():server_errors.append(traceback.format_exc())
        except BaseException:server_errors.append(traceback.format_exc())
    server_thread=threading.Thread(target=server,daemon=True);server_thread.start()
    args=SimpleNamespace(port=port,suite="libero_object",task=0,init=0,arm="fake")
    worker=threading.Thread(target=network_loop,args=(args,box,initial,"test",records,arrays),daemon=True)
    worker.start();owner=threading.get_ident()
    def step(action):
        assert threading.get_ident()==owner and advanced.is_set()
        executed.append(action[0]);physical_step[0]+=1
        return {"value":np.asarray([physical_step[0]],dtype=np.int64)},0.,physical_step[0]==12,{}
    def capture(index,obs,action,metadata):
        assert index==physical_step[0]
        captures.append(metadata)
        box.publish_frame(Frame(index,metadata["capture_host"],obs,original.input_digest(obs,"test")))
    bootstrap=candidate(0,0,initial)
    try:
        count,success,chunks=execute_actor(box,bootstrap,100,step,capture,period=.005)
        worker.join(3)
        assert not worker.is_alive() and box.error is None,box.error
        assert count==12 and success and executed==[0]*5+[1]*5+[2]*2
        assert [c.record["executed_length"] for c in chunks]==[5,5,2]
        assert chunks[1].record["boundary_deadline_gap_ms"]>20
        assert box.max_mailbox==1 and box.candidate is None
        assert all(c["action_index"]<5 for c in captures)
        assert all(b["application_host"]-a["application_host"]>=.0049 for a,b in zip(captures,captures[1:]))
        assert len(records)==len(requests) and all(r["disposition"] in ("consumed","expired","superseded","terminal") for r in records)
        checks+= ["real socket advance precedes first actor action", "R5 is never extended or replaced midway", "missing result creates recorded control gap", "terminal truncation stops exactly at real success", "real accepted next observation is reused in subsequent request", "actor does not catch up faster than configured tick"]
    finally:
        box.stop();server_stop.set();listener.close();server_thread.join(4)
    if server_errors:raise AssertionError(server_errors)
    assert "torch" not in __import__("sys").modules and "libero" not in __import__("sys").modules
    return dict(status="passed",checks=checks,gpu_or_environment_started=False,
                fake_actions=count,fake_predictions=len(records)+1,max_mailbox=box.max_mailbox)


if __name__=="__main__":
    import json
    print(json.dumps(run(),indent=2))

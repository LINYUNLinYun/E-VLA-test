"""CPU fakes verify adapter lifecycle/serialization without CUDA initialization."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from preparations.idea2 import progressive_kv_policy as module
from preparations.idea2.openpi_backend import Prepared
from preparations.idea2.scheduler import Observation,Pool


class Event:
    def record(self,*args): pass
    def synchronize(self): pass
    def query(self): return True


class Runtime:
    created=0
    def __init__(self,backend,prepared,noise):
        Runtime.created+=1
        self.graphs=SimpleNamespace(num_layers=18,setup_ms=12.5)
        self.pool=Pool(epoch=1,task="progressive")
        self.pool.acquire_old_kv=self.pool.acquire
        self.records=[]
        self.bootstrap_calls=self.request_calls=self.reset_calls=self.drain_calls=0
        self.next_id=0
    def publish(self,prepared):
        self.pool.offer(Observation(self.next_id,self.pool.epoch,self.pool.task,prepared))
        identity,_=self.pool.reserve()
        self.pool.submitted(identity,SimpleNamespace(prefix=True),Event())
        self.pool.poll()
        self.next_id+=1
        return identity
    def bootstrap(self,prepared):
        self.bootstrap_calls+=1
        self.publish(prepared)
    def reset(self,prepared):
        self.reset_calls+=1
        self.drain()
        self.pool.drain_reset(self.pool.epoch+1,self.pool.task)
        self.next_id=0
        self.bootstrap(prepared)
    def request(self,prepared,noise,**kwargs):
        assert kwargs==dict(lead=module.LEADS[active.mode],collect_events=False,next_encoder=False,lookahead=None)
        self.request_calls+=1
        old=self.pool.acquire_old_kv()
        identity=self.publish(prepared)
        self.pool.release(old,Event())
        frontiers=[min(18,max(kwargs["lead"],i+1)) for i in range(10)]
        row=dict(request=self.next_id-1,epoch=identity.epoch,slot=identity.slot,generation=identity.generation,
                 frontiers=frontiers,next_encoder=False,cpu_cpu_ms=1.0,iteration_ms=[0.1]*10,
                 _events={"not_serializable":object()})
        self.records.append(row)
        return np.ones((10,32),np.float32),row,None
    def drain(self): self.drain_calls+=1
    def close(self): self.pool.drain_reset(self.pool.epoch+1,self.pool.task)


def main():
    global active
    assert not torch.cuda.is_initialized()
    prepared=Prepared(None,[torch.zeros(1,3,4,4)],[torch.ones(1,dtype=torch.bool)],
                      torch.ones(1,8,dtype=torch.long),torch.ones(1,8,dtype=torch.bool),torch.zeros(1,32))
    backend=SimpleNamespace(num_steps=10,model=SimpleNamespace(config=SimpleNamespace(action_horizon=10),pi05=True),
        identity=dict(config="Pi0Config(pi05=True)",num_steps=10),device="cpu",
        prepare=lambda _:prepared,noise=lambda seed:torch.zeros(1,10,32),
        generate=lambda *args:torch.ones(1,10,32),
        policy=SimpleNamespace(_output_transform=lambda d:dict(actions=d["actions"][:,:7])))
    checks=[]
    with patch.object(module,"native_raw",lambda x:x),patch.object(torch.cuda,"Event",Event),patch.object(torch.cuda,"current_stream",lambda *args:None):
        active=module.ProgressivePolicy(backend,"lead3",Runtime)
        first=active.predict({},"task A","libero_spatial",1701)
        engine=active.engine
        assert first["bootstrap"] and first["frontiers"]==[18]*10
        assert engine.bootstrap_calls==1 and engine.request_calls==0
        assert first["actions"].shape==(10,7) and first["normalized_actions"].shape==(10,32)
        checks.append("first response has exactly one bootstrap and fresh original AE, no repeated current Prefill")
        second=active.predict({},"task A","libero_spatial",1702)
        assert not second["bootstrap"] and engine.request_calls==1 and not engine.records
        assert "_events" not in second["progressive"]
        json.dumps(second,default=lambda x:x.tolist())
        checks.append("request disables collect_events/lookahead/next_encoder and serializes no CUDA Event")
        active.reset()
        assert active.engine is engine and active.needs_bootstrap
        reset=active.predict({},"task A","libero_spatial",1701)
        assert reset["bootstrap"] and reset["request"]==0 and engine.reset_calls==1 and Runtime.created==1
        checks.append("reset drains and lazily bootstraps while retaining one Graph engine")
        changed=active.predict({},"task B","libero_spatial",1701)
        assert changed["bootstrap"] and engine.reset_calls==2 and Runtime.created==1
        checks.append("task change resets old observation identity without Graph recapture")
        active.configure("fresh18")
        active.predict({},"task B","libero_spatial",1701)
        fresh=active.predict({},"task B","libero_spatial",1702)
        assert fresh["frontiers"]==[18]*10 and fresh["observation_request_age"]==0
        checks.append("fresh18 uses only current KV after episode bootstrap")
        with patch.object(backend.policy,"_output_transform",side_effect=RuntimeError("decode fault")):
            try:
                active.predict({},"task B","libero_spatial",1703)
            except RuntimeError as error:
                assert str(error)=="decode fault"
            else:
                raise AssertionError("Expected injected decoder exception")
        assert active.needs_bootstrap and active.engine is engine
        checks.append("decoder failure drains and requires fresh bootstrap before next request")
        active.close()
    wrapper_path=Path(__file__).with_name("closedloop.py")
    spec=importlib.util.spec_from_file_location("_progressive_wrapper_test",wrapper_path)
    wrapper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    args=wrapper.parser().parse_args([])
    with patch.object(wrapper.legacy,"dependency_probe",return_value=dict(passed=True)):
        plan=wrapper.make_plan(args)
    assert len(plan["cases"])==40
    for mode in wrapper.LEADS:
        selected=[r for r in plan["cases"] if r["mode"]==mode]
        assert len(selected)==8 and len({(r["suite"],r["task"],r["init"]) for r in selected})==8
    assert plan["next_encoder"] is False and plan["server"]["argv"][3]=="preparations.idea2.progressive_kv_policy"
    assert wrapper.parse_cpus("32-35,40")=={32,33,34,35,40}
    checks.append("default plan has all five conditions and exactly eight original fixed cases each; CPU-mask parser preserved")
    assert not torch.cuda.is_initialized()
    receipt=dict(passed=True,cuda_initialized=False,checks=checks,policy_sha256=wrapper.hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
    Path(__file__).with_name("closedloop_cpu_checks.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()

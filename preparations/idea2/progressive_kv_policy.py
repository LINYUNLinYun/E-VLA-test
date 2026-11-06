"""Closed-loop RPC adapter for the explicitly approximate mixed-KV experiment.

The next environment observation does not exist during an inference request:
lookahead and next-encoder work are therefore always disabled here.
"""
import argparse
import os
import time

import numpy as np
import torch

from .experiments import load_backend, native_raw
from .runtime_graph import layout


LEADS={"lead0":0,"lead1":1,"lead3":3,"lead6":6,"fresh18":18}


class ProgressivePolicy:
    def __init__(self,backend,mode="lead3",runtime_factory=None):
        if backend.num_steps!=10 or backend.model.config.action_horizon!=10 or not backend.model.pi05:
            raise ValueError("Progressive closed-loop experiment requires pi0.5 D10/H10/R5")
        if mode not in LEADS:
            raise ValueError("Unknown progressive condition")
        self.backend,self.mode=backend,mode
        self.runtime_factory=runtime_factory
        self.engine=None
        self.signature=None
        self.task=None
        self.needs_bootstrap=True
        self.request_index=0
        self.episode=0
        self.capture_engines=0

    @property
    def protocol(self):
        return dict(self.backend.identity,mode=self.mode,lead=LEADS[self.mode],H=10,R=5,num_steps=10,
            cache_slots=2,execution="progressive_cuda_graph",interval=1,host_workers=0,diagnostics=True,
            submission_order="prefill_then_ae_kv_ready",next_encoder=False,lookahead=False,
            semantics="fresh_current_KV" if self.mode=="fresh18" else "approximate_per_iteration_mixed_old_new_KV",
            approximation=self.mode!="fresh18",first_response="current full KV with original fresh AE; one bootstrap Prefill",
            graph_reuse_across_episodes=True,full_model=True,real_time=False,
            host_cpu_affinity=sorted(os.sched_getaffinity(0)),
            clock="LIBERO pauses during inference",preparation_core="PreparedModulationCore")

    def configure(self,mode,interval=1,**compatibility):
        if mode not in LEADS or interval!=1:
            raise ValueError("Expected lead0/1/3/6/fresh18, interval=1")
        if compatibility:
            raise ValueError("Unsupported progressive configuration: "+", ".join(sorted(compatibility)))
        self.reset()
        self.mode=mode
        return self.protocol

    def reset(self):
        begin=time.perf_counter()
        if self.engine is not None:
            self.engine.drain()
            self.engine.records.clear()
        self.needs_bootstrap=True
        self.request_index=0
        self.task=None
        self.episode+=1
        return dict(reset=True,episode=self.episode,bootstrap_pending=True,
                    graph_retained=self.engine is not None,drain_ms=(time.perf_counter()-begin)*1000)

    def close(self):
        if self.engine is not None:
            self.engine.close()
            self.engine=None
        self.signature=None

    @torch.no_grad()
    def predict(self,obs,instruction,suite,seed,**unused):
        from preparations.idea1_v2_1.runtime_client import input_digest
        begin=time.perf_counter()
        if self.task is not None and self.task!=(suite,instruction):
            self.reset()
        self.task=(suite,instruction)
        prepared=self.backend.prepare(native_raw(dict(obs=obs,instruction=instruction)))
        noise=self.backend.noise(seed)
        signature=layout(prepared,noise)
        setup_ms=0.0
        recapture_reason=None
        if self.engine is not None and signature!=self.signature:
            self.close()
            self.needs_bootstrap=True
            self.request_index=0
            recapture_reason="input_layout_changed"
        created=self.engine is None
        if created:
            if self.runtime_factory is None:
                from .progressive_kv import ProgressiveRuntime
                factory=ProgressiveRuntime
            else:
                factory=self.runtime_factory
            self.engine=factory(self.backend,prepared,noise)
            if self.engine.graphs.num_layers!=18:
                self.close()
                raise ValueError("This fixed closed-loop matrix requires the original 18-layer VLM")
            self.signature=signature
            self.capture_engines+=1
            setup_ms=self.engine.graphs.setup_ms
        bootstrap=self.needs_bootstrap
        bootstrap_ms=0.0
        try:
            if bootstrap:
                cold_begin=time.perf_counter()
                if created:
                    self.engine.bootstrap(prepared)
                else:
                    self.engine.reset(prepared)
                # Bootstrap already produced the current full Prefix. Reusing
                # it here avoids a second current-frame Prefill.
                lease=self.engine.pool.acquire_old_kv()
                if lease is None:
                    raise RuntimeError("Bootstrap did not publish a full Prefix")
                try:
                    raw=self.backend.generate(lease.value,prepared,noise)
                    normalized=raw[0].detach().cpu().numpy().copy()
                finally:
                    done=torch.cuda.Event()
                    done.record(torch.cuda.current_stream(self.backend.device))
                    done.synchronize()
                    self.engine.pool.release(lease,done)
                identity=self.engine.pool.active
                row=dict(request=0,epoch=identity.epoch,slot=identity.slot,generation=identity.generation,
                    lead=LEADS[self.mode],frontiers=[18]*10,iteration_ms=None,next_encoder=False,
                    force_old=False,graph_recaptures=0,full_kv_slots=2,prefix_done_at_response=True,
                    ae_submit_ms=None,bootstrap=True)
                bootstrap_ms=(time.perf_counter()-cold_begin)*1000
                self.needs_bootstrap=False
            else:
                normalized,row,next_prepared=self.engine.request(prepared,noise,lead=LEADS[self.mode],
                    collect_events=False,next_encoder=False,lookahead=None)
                if next_prepared is not None or row["next_encoder"]:
                    raise RuntimeError("Closed-loop inference cannot invent a future observation")
            normalized=np.asarray(normalized)
            if normalized.shape!=(10,32) or not np.isfinite(normalized).all():
                raise RuntimeError("Invalid normalized progressive action chunk")
            state=prepared.state[0].detach().cpu().numpy()
            actions=self.backend.policy._output_transform(dict(state=state,actions=normalized.copy()))["actions"]
            if np.asarray(actions).shape!=(10,7) or not np.isfinite(actions).all():
                raise RuntimeError("Invalid decoded progressive action chunk")
            public_row={key:value for key,value in row.items() if not key.startswith("_")}
            # The Pool and last_producer own outstanding synchronization Events.
            # This controller does not accumulate microbenchmark event records.
            self.engine.records.clear()
            available=time.perf_counter()
            frontiers=public_row["frontiers"]
            current=self.request_index
            old=current if bootstrap else current-1
            result=dict(actions=actions,normalized_actions=normalized,execute_length=5,generated_length=10,
                submitted_length=5,seed=seed,input_hash=input_digest(obs,instruction),mode=self.mode,lead=LEADS[self.mode],
                protocol=self.protocol,request=current,episode=self.episode,bootstrap=bootstrap,
                bootstrap_ms=bootstrap_ms,graph_setup_ms=setup_ms,capture_engines=self.capture_engines,
                graph_recapture_reason=recapture_reason,graph_kv_buffers=2,execution="progressive_cuda_graph",
                next_encoder=False,lookahead=False,frontiers=frontiers,progressive=public_row,
                current_observation_id=current,old_prefix_observation_id=old,
                new_prefix_observation_id=current,prefix_observation_id=current if min(frontiers)==18 else None,
                observation_request_age=0 if min(frontiers)==18 else None,
                per_iteration_old_layer_counts=[18-f for f in frontiers],
                per_iteration_new_layer_counts=frontiers,suffix_state_id=current,reference_state_id=current,
                snapshot=dict(epoch=public_row["epoch"],slot=public_row["slot"],generation=public_row["generation"]),
                pool=dict(live_snapshots=self.engine.pool.live_snapshots,publications=self.engine.pool.publications,
                          pending_replacements=self.engine.pool.pending_replacements,no_free_slot=self.engine.pool.no_free_slot),
                timing=dict(action_ready_host_ms=(available-begin)*1000,response_host_ms=(time.perf_counter()-begin)*1000,
                            bootstrap_ms=bootstrap_ms,graph_setup_ms=setup_ms,
                            runtime_cpu_cpu_ms=public_row.get("cpu_cpu_ms"),
                            availability_definition="CPU output and decode complete; RPC transport excluded"))
            self.request_index+=1
            return result
        except BaseException:
            self.reset()
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--port",type=int,default=19085)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--mode",choices=list(LEADS),default="lead3")
    args=parser.parse_args()
    expected=os.environ.get("PROGRESSIVE_EXPECTED_CPUS")
    if expected and set(map(int,expected.split(",")))!=os.sched_getaffinity(0):
        raise RuntimeError("Server did not inherit the planned CPU affinity")
    from .server import serve
    backend=load_backend(args.checkpoint,args.device,model_config="pi05_libero",num_steps=10)
    policy=ProgressivePolicy(backend,args.mode)
    try:
        serve(policy,args.port)
    finally:
        policy.close()


if __name__=="__main__":
    main()

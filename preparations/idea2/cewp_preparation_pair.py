"""AB/BA preparation ablations sharing exactly two full KV buffers.

Only experiment dispatch changes: three independently captured AE graphs per
slot read the same old KV. Schedule and modulation graphs never build a prefix.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle
import threading
import time
import traceback
import weakref

import torch

from .cewp_bench import Recorder, write_results
from .cewp_preparation import PreparedModulationCore, PreparedScheduleCore
from .cewp_segmented import SelectiveGraphs
from .experiments import load_backend
from .openpi_backend import cache_tensors
from .policy import PipelinePolicy
from .runtime_graph import RuntimeGraphs


VARIANTS=("baseline","schedule","modulation")


class VariantGraphs:
    """Mixin keeps parent stream/events/lease checks authoritative."""
    def __init__(self,*args,**kwargs):
        self.owner_thread=threading.get_ident()
        self.extra=[]
        self.kv_weakrefs=[]
        begin=time.perf_counter()
        try:
            super().__init__(*args,**kwargs)
            self.kv_weakrefs=[weakref.ref(value) for slot in self.slots
                             for value in (slot.prefix,*cache_tensors(slot.prefix.cache))]
            for slot in self.slots:
                variants={}
                self.extra.append(variants)
                with torch.cuda.stream(self.ae_stream),torch.no_grad():
                    for name,core_class in (("schedule",PreparedScheduleCore),("modulation",PreparedModulationCore)):
                        core=core_class(self.backend,slot.prepared)
                        for _ in range(2):
                            temporary=core.generate(slot.prefix,slot.prepared,slot.noise)
                            self.ae_stream.synchronize()
                            del temporary
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=self.ae_stream):
                            actions=core.generate(slot.prefix,slot.prepared,slot.noise)
                        graph.replay()
                        self.ae_stream.synchronize()
                        variants[name]=dict(core=core,graph=graph,actions=actions)
            pools=[graph.pool() for slot in self.slots for graph in (slot.prefix_graph,slot.ae_graph)]
            pools.extend(item["graph"].pool() for variants in self.extra for item in variants.values())
            vision_pools=[slot.vision_graph.pool() for slot in self.slots if hasattr(slot,"vision_graph")]
            assert len(pools)==8 and len(set(pools+vision_pools))==len(pools)+len(vision_pools)
            actions=[slot.actions.data_ptr() for slot in self.slots]
            actions.extend(item["actions"].data_ptr() for variants in self.extra for item in variants.values())
            assert len(set(actions))==6,"Variant graph outputs alias"
            assert self.addresses==[tuple(t.data_ptr() for t in cache_tensors(slot.prefix.cache)) for slot in self.slots]
            self.audit=dict(kv_snapshots=2,kv_addresses=self.addresses,kv_producer_and_ae_private_pools=8,
                            vision_private_pools=len(vision_pools),total_private_pools=len(pools)+len(vision_pools),
                            pool_handles=pools+vision_pools,action_addresses=actions,
                            captures=8+len(vision_pools),preparation_variants=list(VARIANTS))
            self.setup_ms=(time.perf_counter()-begin)*1000
        except BaseException:
            self.close()
            raise

    @torch.no_grad()
    def generate(self,pool,lease,prepared,noise):
        if threading.get_ident()!=self.owner_thread:
            raise RuntimeError("Variant selection belongs to the Pool owner Host")
        variant=self.backend.preparation_variant
        if variant not in VARIANTS:
            raise ValueError("Unknown preparation variant")
        slot=self.slots[lease.identity.slot]
        if variant=="baseline":
            return super().generate(pool,lease,prepared,noise)
        selected=self.extra[lease.identity.slot][variant]
        previous_graph,previous_actions=slot.ae_graph,slot.actions
        try:
            slot.ae_graph,slot.actions=selected["graph"],selected["actions"]
            return super().generate(pool,lease,prepared,noise)
        finally:
            slot.ae_graph,slot.actions=previous_graph,previous_actions

    def close(self):
        if hasattr(self,"prefix_stream"):
            self.prefix_stream.synchronize()
            self.ae_stream.synchronize()
        self.extra.clear()
        if hasattr(self,"slots"):
            super().close()


class BaseVariantGraphs(VariantGraphs,RuntimeGraphs):
    pass


class VisionVariantGraphs(VariantGraphs,SelectiveGraphs):
    pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--checkpoint",default="shared/checkpoints/openpi/pi05_libero_pytorch_idea2")
    parser.add_argument("--model-config",default="pi05_libero")
    parser.add_argument("--bank",default="results/idea1_v3_1/input_bank.pkl")
    parser.add_argument("--steps",type=int,choices=(1,2,5,10),default=10)
    parser.add_argument("--selective",choices=("base","vision_only"),default="vision_only")
    parser.add_argument("--await-previous-prefix",action="store_true")
    parser.add_argument("--ae-priority",type=int,choices=(0,-1),default=-1)
    parser.add_argument("--warmup",type=int,default=10)
    parser.add_argument("--count",type=int,default=100)
    parser.add_argument("--continuous-count",type=int,default=100)
    parser.add_argument("--reset-every",type=int,default=50)
    parser.add_argument("--instruction-index",type=int,default=0)
    parser.add_argument("--max-error",type=float,default=1e-5)
    parser.add_argument("--mae-error",type=float,default=1e-6)
    parser.add_argument("--smoke",action="store_true")
    args=parser.parse_args()
    if not args.smoke and min(args.count,args.continuous_count)<100:
        parser.error("Formal preparation ablations require >=100 pairs and continuous requests")
    if not args.smoke and args.warmup<10:
        parser.error("Each variant in each group needs >=10 warmup requests")
    dest=Path(args.output)
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"source").mkdir()
    hashes={}
    for path in Path(__file__).parent.glob("*.py"):
        payload=path.read_bytes()
        (dest/"source"/path.name).write_bytes(payload)
        hashes[path.name]=hashlib.sha256(payload).hexdigest()
    bank=Path(args.bank).read_bytes()
    original=pickle.loads(bank)
    instruction=original[args.instruction_index]["instruction"]
    selected=[i for i,row in enumerate(original) if row["instruction"]==instruction]
    records=[original[i] for i in selected]
    if len(records)<2:
        raise ValueError("Need two different observations of one task")
    mode_prefix="graph_ae_high" if args.ae_priority==-1 else "graph_ae_first"
    mode_name=lambda variant:mode_prefix+"__"+variant
    comparisons=(("baseline","schedule"),("schedule","modulation"))
    serialized_args=vars(args)|dict(kind="requests",pairs=[mode_name(a)+":"+mode_name(b) for a,b in comparisons],
                                  continuous=[mode_name(v) for v in VARIANTS])
    result=dict(args=serialized_args,rows=[],groups=[],source_sha256=hashes,
                bank_sha256=hashlib.sha256(bank).hexdigest(),selected_bank_indices=selected,instruction=instruction,
                tolerance=dict(max_abs=args.max_error,mae=args.mae_error),
                experiment="preparation_variant_ABBA",scope="One engine, exactly two full KV buffers, matched old-KV paired preparation variants; all captures excluded from timing")
    write_results(dest,result)
    backend=load_backend(args.checkpoint,model_config=args.model_config,num_steps=args.steps)
    backend.await_previous_prefix=args.await_previous_prefix
    backend.preparation_variant="baseline"
    backend.graph_engine_class=VisionVariantGraphs if args.selective=="vision_only" else BaseVariantGraphs
    backend.selective_overlap="vision_only"
    result["identity"]=backend.identity
    result["gpu_uuid"]=str(torch.cuda.get_device_properties(backend.device).uuid)
    recorder=Recorder(backend)
    policy=PipelinePolicy(backend,diagnostics=True,mode="async_reuse",interval=1,submission_order="ae_first",
                          cuda_graphs=True,ae_priority=args.ae_priority)
    weak=[]
    groups=[("paired",pair) for pair in comparisons]+[("continuous",(variant,)) for variant in VARIANTS]
    try:
        for group_index,(phase,variants) in enumerate(groups):
            print("Preparation group",group_index,phase,variants,flush=True)
            cold=[]
            for variant in variants:
                policy.reset()
                backend.preparation_variant=variant
                for i in range(args.warmup):
                    response=recorder.predict(policy,records,i,900+i,mode=mode_name(variant),phase="warmup",measured=False)
                    if response["graph_setup_ms"]:
                        cold.append(response["graph_setup_ms"])
            if not weak:
                weak=list(policy.graphs.kv_weakrefs)
                result["graph_audit"]=policy.graphs.audit
            start=len(recorder.rows)
            torch.cuda.reset_peak_memory_stats(backend.device)
            if phase=="paired":
                for index in range(args.count):
                    order=variants if index%2==0 else variants[::-1]
                    for ordinal,variant in enumerate(order):
                        policy.reset()
                        backend.preparation_variant=variant
                        recorder.predict(policy,records,2*index,7000+index,mode=mode_name(variant),phase="bootstrap",measured=False)
                        recorder.predict(policy,records,2*index+1,1701+index,mode=mode_name(variant),phase=phase,pair=index,arm_order=ordinal)
                    if index%10==9:
                        recorder.finish_events()
                        print("Preparation pairs",index+1,flush=True)
            else:
                variant=variants[0]
                policy.reset()
                backend.preparation_variant=variant
                for index in range(args.continuous_count):
                    if index and args.reset_every and index%args.reset_every==0:
                        policy.reset()
                    if policy.request==0:
                        recorder.predict(policy,records,index-1,7000+index,mode=mode_name(variant),phase="bootstrap",measured=False)
                    recorder.predict(policy,records,index,1701+index,mode=mode_name(variant),phase=phase)
            recorder.finish_events()
            for row in recorder.rows[start:]:
                row.update(group=group_index,preparation_variant=row["mode"].split("__")[-1],
                           policy_mode="async_reuse",submission_order="ae_first",ae_priority=args.ae_priority)
            result["rows"]=recorder.rows
            result["groups"].append(dict(phase=phase,modes=[mode_name(v) for v in variants],
                preparation_variants=list(variants),capture_engines=1 if cold else 0,
                cold_capture_ms=cold,correct=None))
            write_results(dest,result)
        result["total_engines"]=recorder.capture_engines
        assert recorder.capture_engines==1
        assert all(row["steady_recapture"]==0 for row in recorder.rows)
    except BaseException:
        result["failure"]=traceback.format_exc()
        write_results(dest,result)
        raise
    finally:
        policy.close()
        del policy
        gc.collect()
    alive=sum(ref() is not None for ref in weak)
    result["kv_python_objects_alive_after_close"]=alive
    write_results(dest,result)
    if alive:
        raise RuntimeError(f"Graph close retained {alive} Prefix/cache tensor objects; refusing reference KV allocation")
    print("Verify all preparation outputs",len(recorder.rows),flush=True)
    passed=recorder.verify(backend,records,args.max_error,args.mae_error)
    for group_index,group in enumerate(result["groups"]):
        group["correct"]=all(row["correct"] for row in recorder.rows if row["group"]==group_index)
    result["passed"]=passed
    write_results(dest,result)
    if not passed:
        raise AssertionError("Preparation ablation violated preregistered numerical tolerance")
    print(json.dumps(dict(passed=passed,outputs=len(recorder.rows),engines=recorder.capture_engines,
                          kv_objects_after_close=alive,paired_speedups=result["paired_speedups"]),indent=2),flush=True)


if __name__=="__main__":
    main()

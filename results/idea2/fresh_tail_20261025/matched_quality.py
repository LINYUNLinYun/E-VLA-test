"""Fixed-input fresh-baseline quality study: D10, lead3, SM16/32 x tail0/1/2.

Default invocation only builds a CPU plan and a real same-task input bank.
--execute needs the root scheduler's assigned GPU. This is a quality replay,
not a formal timing benchmark: eight matched-schedule oracles run per variant.
"""
import argparse
import gc
import hashlib
import json
import pickle
import sys
import time
import traceback
import weakref
from pathlib import Path

import numpy as np

from results.idea2.fresh_tail_20261025.common import CONFIGS,HERE,ROOT,delta,save,sha,write_csv


def bank_samples(source,count):
    """Preselected case0; unique current inputs with their actual previous frame."""
    result=[];seen=set()
    for mode in ("fresh18","lead0","lead1","lead3","lead6"):
        leaf=source/"episodes"/mode/"libero_object/task00/init00"
        receipt=json.loads((leaf/"smoke_receipt.json").read_text())
        if receipt["status"]!="passed":raise ValueError("Input trajectory lacks a passed receipt")
        rows=[json.loads(line) for line in (leaf/"infer.jsonl").read_text().splitlines()]
        for previous,current in zip(rows,rows[1:]):
            if current["input_hash"] in seen:continue
            def load(row):
                with np.load(row["input_path"],allow_pickle=False) as payload:
                    return dict(obs={k:payload[k].copy() for k in payload.files if k not in ("instruction","seed")},
                                instruction=str(payload["instruction"].item()))
            old,new=load(previous),load(current)
            if old["instruction"]!=new["instruction"]:raise ValueError("Bank crossed task instruction")
            result.append(dict(current=new,old=old,seed=int(current["seed"]),input_hash=current["input_hash"],
                old_input_hash=previous["input_hash"],source_mode=mode,source_request=current["request"],
                source_input_path=current["input_path"],old_source_input_path=previous["input_path"]))
            seen.add(current["input_hash"])
            if len(result)==count:return result
    raise ValueError(f"Only {len(result)} unique same-task observations available, need {count}")


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execute",action="store_true")
    result.add_argument("--output",required=True)
    result.add_argument("--count",type=int,choices=(12,100),default=100)
    result.add_argument("--source",default=str(ROOT/"results/idea2/progressive_kv_20261025/closedloop_gpu0"))
    result.add_argument("--checkpoint",default=str(ROOT/"shared/checkpoints/openpi/pi05_libero_pytorch_idea2"))
    result.add_argument("--configs",nargs="+",choices=list(CONFIGS),default=list(CONFIGS))
    result.add_argument("--oracle-count",type=int,default=8)
    result.add_argument("--warmup",type=int,default=3)
    result.add_argument("--gpu-uuid")
    result.add_argument("--cpus",help="Optional CPU affinity mask, e.g. 32-47")
    return result


def plan(args):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    bank_path=output/"same_task_bank.pkl"
    if bank_path.exists():
        samples=pickle.loads(bank_path.read_bytes())
        if len(samples)!=args.count:raise ValueError("Existing bank count differs; choose new output")
    else:
        samples=bank_samples(Path(args.source).resolve(),args.count)
        bank_path.write_bytes(pickle.dumps(samples,protocol=5))
    from preparations.idea1_v2_1.runtime_client import input_digest
    for sample in samples:
        for name,key in (("current","input_hash"),("old","old_input_hash")):
            if input_digest(sample[name]["obs"],sample[name]["instruction"])!=sample[key]:
                raise ValueError("Saved bank input hash does not match source")
    if len({r["input_hash"] for r in samples})!=args.count or len({r["current"]["instruction"] for r in samples})!=1:
        raise ValueError("Require distinct same-task observations")
    from results.idea2.progressive_kv_20261025 import closedloop
    hashes=closedloop.source_hashes()
    for file in (HERE/"adapter.py",HERE/"common.py",Path(__file__).resolve(),ROOT/"results/idea2/progressive_kv_20261025/resource_adapter.py"):
        hashes[str(file.relative_to(ROOT))]=sha(file)
    return dict(args=vars(args),status="planned",D=10,H=10,R=5,lead=3,next_encoder=False,
        bank=str(bank_path),bank_sha256=sha(bank_path),unique_observations=len(samples),
        instruction=samples[0]["current"]["instruction"],source_sha256=hashes,
        selection="First 100 unique nonbootstrap observations of fixed case0, ordered fresh18/lead0/lead1/lead3/lead6; first12 is the gate subset; no selection by success or quality difference",
        previous_frame_contract="Every sample retains its real preceding observation. Reset/bootstrap whenever the bank's next old hash differs from previous admitted current hash.",
        noise_contract="Original source model_seed, CUDA FP32 backend.noise generated once; CPU arrays and hashes saved and reused exactly by fresh and all six variants",
        canonical_fresh="Original eager backend builds one complete current Prefix and runs original D10; all reference Prefixes released before variant graphs exist",
        metric="Approximation deltas versus canonical fresh, per-dimension R5/H10. Matched observed-schedule oracle is a distinct implementation gate, never fresh-equality.",
        oracle_count=args.oracle_count,oracle_tolerance=dict(max_abs=1e-5,mae=1e-6),
        ready_semantics="Final K Euler rounds wait for all 18 current K/V pairs; last VLM FFN need not finish",
        cleanup="Close each engine, retain Green Contexts through main return, then finish_contexts; exceptions defer contexts to process exit")


def run(args,manifest):
    import torch
    from preparations.idea2.experiments import load_backend,native_raw
    from preparations.idea2.openpi_backend import cache_tensors
    from results.idea2.fresh_tail_20261025 import adapter
    output=Path(args.output).resolve()
    samples=pickle.loads(Path(manifest["bank"]).read_bytes())
    backend=load_backend(args.checkpoint,model_config="pi05_libero",num_steps=10)
    uuid=str(torch.cuda.get_device_properties(backend.device).uuid)
    if args.gpu_uuid and uuid.removeprefix("GPU-")!=args.gpu_uuid.removeprefix("GPU-"):
        raise ValueError("Assigned GPU UUID mismatch")
    result=dict(manifest=manifest,status="running",gpu_uuid=uuid,identity=backend.identity,baseline=[],variants=[],
                scope="Quality-only fixed replay; oracle and prefix-completion waits invalidate isolated timing interpretations")
    save(output/"result.json",result)
    noises=[];references=[];normalized_references=[]
    prefix=None
    try:
        # No engine exists during canonical fresh generation: only one KV.
        for index,sample in enumerate(samples):
            prepared=backend.prepare(native_raw(sample["current"]))
            noise=backend.noise(sample["seed"])
            noises.append(noise.detach().cpu().numpy().copy())
            prefix=backend.build_prefix(prepared)
            raw=backend.generate(prefix,prepared,noise)
            normalized_references.append(raw[0].detach().cpu().numpy().copy())
            references.append(backend.output(prepared,raw).copy())
            result["baseline"].append(dict(sample=index,input_hash=sample["input_hash"],seed=sample["seed"],
                noise_sha256=hashlib.sha256(noises[-1].tobytes()).hexdigest()))
            del prefix,raw;prefix=None
        np.savez(output/"canonical_fresh.npz",actions=np.stack(references),normalized=np.stack(normalized_references),noise=np.stack(noises))
        del prepared,noise
        gc.collect()
        save(output/"result.json",result)
        for name in args.configs:
            configuration=CONFIGS[name]
            layouts,_=adapter.install(**configuration)
            from preparations.idea2.progressive_kv import ProgressiveRuntime
            engine=None;old_prepared=None;prepared=None;noise=None
            rows=[];resets=0;old_hash=None;weak=[]
            try:
                old_prepared=backend.prepare(native_raw(samples[0]["old"]))
                noise=torch.from_numpy(noises[0]).to(backend.device)
                engine=ProgressiveRuntime(backend,old_prepared,noise)
                addresses=[tuple(t.data_ptr() for t in cache_tensors(slot.prefix.cache)) for slot in engine.graphs.slots]
                weak=[weakref.ref(t) for slot in engine.graphs.slots for t in (slot.prefix,*cache_tensors(slot.prefix.cache))]
                engine.bootstrap(old_prepared)
                for _ in range(args.warmup):
                    prepared=backend.prepare(native_raw(samples[0]["current"]))
                    engine.request(prepared,noise,lead=3,next_encoder=False,collect_events=True,verify=False)
                    engine.reset(old_prepared)
                engine.records.clear()
                old_hash=samples[0]["old_input_hash"]
                forced_reset=args.count//2
                oracle_indices=set(np.linspace(0,args.count-1,args.oracle_count,dtype=int).tolist())
                if forced_reset not in oracle_indices:
                    oracle_indices.remove(sorted(oracle_indices)[len(oracle_indices)//2]);oracle_indices.add(forced_reset)
                if not layouts or any(layout["sm_counts"]!=[configuration["ae_sm_min"],128-configuration["ae_sm_min"]] for layout in layouts):
                    raise RuntimeError("Actual resource partition differs from the selected condition")
                for index,sample in enumerate(samples):
                    if old_hash!=sample["old_input_hash"] or index==forced_reset:
                        old_prepared=backend.prepare(native_raw(sample["old"]))
                        engine.reset(old_prepared);resets+=1
                    prepared=backend.prepare(native_raw(sample["current"]))
                    noise=torch.from_numpy(noises[index]).to(backend.device)
                    normalized,record,nxt=engine.request(prepared,noise,lead=3,next_encoder=False,collect_events=True,verify=index in oracle_indices)
                    if nxt is not None:raise RuntimeError("No future observation is available in this fixed-input quality replay")
                    if configuration["fresh_tail_steps"] and record["frontiers"][-configuration["fresh_tail_steps"]:]!=[18]*configuration["fresh_tail_steps"]:
                        raise RuntimeError("Required fresh tail was not fully current KV")
                    if [tuple(t.data_ptr() for t in cache_tensors(slot.prefix.cache)) for slot in engine.graphs.slots]!=addresses:
                        raise RuntimeError("Graph KV addresses changed")
                    if engine.pool.live_snapshots>2:raise RuntimeError("More than two live KV snapshots")
                    state=prepared.state[0].detach().cpu().numpy()
                    decoded=backend.policy._output_transform(dict(state=state,actions=normalized.copy()))["actions"]
                    oracle=record.get("numerical",{}).get("matched_schedule")
                    if index in oracle_indices and (oracle is None or not oracle["finite"] or oracle["max_abs"]>1e-5 or oracle["mae"]>1e-6):
                        raise RuntimeError("Observed-schedule oracle failed; this is separate from fresh quality")
                    rows.append(dict(configuration=name,sample=index,input_hash=sample["input_hash"],old_input_hash=sample["old_input_hash"],
                        seed=sample["seed"],noise_sha256=result["baseline"][index]["noise_sha256"],frontiers=record["frontiers"],
                        fresh_tail_steps=configuration["fresh_tail_steps"],fresh_tail_wait_ms=record["fresh_tail_wait_ms"],
                        approximation_delta=delta(references[index],decoded),matched_schedule_oracle=oracle,
                        record={k:v for k,v in record.items() if not k.startswith("_")},
                        decoded=decoded.tolist(),normalized=np.asarray(normalized).tolist()))
                    old_hash=sample["input_hash"]
                    engine.records.clear()
                    if (index+1)%10==0:print(name,"quality samples",index+1,flush=True)
                variant=dict(configuration=name,config=configuration,rows=rows,resets_to_actual_old_input=resets,
                    kv_slots=2,kv_addresses=addresses,actual_resources=layouts,graph_setup_ms=engine.graphs.setup_ms,
                    oracle_checks=len(oracle_indices),oracle_indices=sorted(oracle_indices),forced_reset_sample=forced_reset,passed=True)
            finally:
                if engine is not None:engine.close()
                del engine,prepared,old_prepared,noise
                gc.collect()
            if any(ref() is not None for ref in weak):raise RuntimeError("Closed engine retained Prefix/cache Python references")
            variant["kv_python_references_after_close"]=0
            result["variants"].append(variant)
            save(output/"result.json",result)
            write_csv(output/"quality.csv",[r for v in result["variants"] for r in v["rows"]])
        result["status"]="awaiting_cleanup"
        save(output/"result.json",result)
    except BaseException:
        result["status"]="failed";result["traceback"]=traceback.format_exc()
        save(output/"result.json",result)
        raise


def main():
    args=parser().parse_args()
    if not 8<=args.oracle_count<=args.count:raise ValueError("At least eight observed-schedule oracles per variant are required")
    if len(set(args.configs))!=len(args.configs):raise ValueError("Duplicate configurations")
    if args.cpus:
        import os
        from results.idea2.progressive_kv_20261025.closedloop import parse_cpus
        requested=parse_cpus(args.cpus)
        if not requested<=os.sched_getaffinity(0):raise ValueError("CPU affinity outside allowed set")
        os.sched_setaffinity(0,requested)
    candidate=plan(args);output=Path(args.output).resolve();saved=output/"plan.json"
    if saved.exists():
        prior=json.loads(saved.read_text())
        for key in ("bank_sha256","source_sha256","D","H","R","lead"):
            if prior[key]!=candidate[key]:raise ValueError("Frozen plan differs; use a fresh output")
        for key in ("count","configs","oracle_count","warmup","checkpoint","source","gpu_uuid","cpus"):
            if prior["args"][key]!=candidate["args"][key]:raise ValueError("Frozen arguments differ: "+key)
    save(saved,candidate)
    if not args.execute:
        print(json.dumps(dict(status="planned_only",plan=str(saved),count=args.count,configs=args.configs),indent=2));return False
    if (output/"result.json").exists():raise ValueError("Refusing to overwrite existing run")
    run(args,candidate)
    return True


if __name__=="__main__":
    executed=main()
    if executed:
        # main/run frames and their caller-owned GPU inputs are gone here.
        from results.idea2.fresh_tail_20261025 import adapter
        cleanup=adapter.finish_contexts()
        args=parser().parse_args();path=Path(args.output)/"result.json"
        result=json.loads(path.read_text());result["cleanup"]=cleanup
        result["status"]="complete" if cleanup["passed"] else "failed_cleanup";save(path,result)

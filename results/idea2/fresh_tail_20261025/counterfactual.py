"""Fixed-frontier causal quality control; CPU plan by default, exactly 600 AE runs.

For each stored sample/family, replay the observed tail0 schedule and change
only its last one or two frontier values to 18. All early-round choices,
observations and explicit noise stay identical. No Graph or resource split.
"""
import argparse
import gc
import hashlib
import json
import pickle
from pathlib import Path
import time
import traceback
import weakref

import numpy as np

from results.idea2.fresh_tail_20261025.common import HERE,ROOT,DIMENSIONS,save,sha,delta,write_csv


def schedules(frontiers):
    values=list(frontiers)
    if len(values)!=10 or values!=sorted(values) or any(type(f) is not int or not 0<=f<=18 for f in values):
        raise ValueError("Invalid stored 10-round frontier")
    return [values,[*values[:-1],18],[*values[:-2],18,18]]


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",default=str(HERE/"matched100_gpu3"))
    p.add_argument("--output",default=str(HERE/"counterfactual_gpu3"))
    p.add_argument("--execute",action="store_true")
    p.add_argument("--gpu-uuid",default="GPU-555cde36-5e41-bc44-0c69-ea47358fd703")
    p.add_argument("--cpus",default="96-111")
    return p


def plan(args):
    source=Path(args.source).resolve();result=json.loads((source/"result.json").read_text())
    receipt=json.loads((source/"process_receipt.json").read_text())
    if result["status"]!="complete" or not result["cleanup"]["passed"] or receipt["exit_code"]!=0:
        raise ValueError("Require completed explicit-noise source experiment")
    manifest=result["manifest"];bank=Path(manifest["bank"])
    if sha(bank)!=manifest["bank_sha256"] or manifest["args"]["count"]!=100:raise ValueError("Source bank identity/count changed")
    families={v["configuration"]:v["rows"] for v in result["variants"] if v["configuration"] in ("sm16_tail0","sm32_tail0")}
    if set(families)!={"sm16_tail0","sm32_tail0"} or any(len(rows)!=100 for rows in families.values()):raise ValueError("Missing tail0 families")
    for name,rows in families.items():
        for index,row in enumerate(rows):
            if row["sample"]!=index:raise ValueError("Sample order changed")
            if any(row[key]!=result["baseline"][index][key] for key in ("input_hash","seed","noise_sha256")):
                raise ValueError("Stored sample/noise identities disagree")
            schedules(row["frontiers"])
    files=[Path(__file__).resolve(),HERE/"common.py",ROOT/"preparations/idea2/cewp_preparation.py",
        ROOT/"preparations/idea2/runtime_graph.py",ROOT/"preparations/idea2/graph_sampler.py",
        ROOT/"preparations/idea2/openpi_backend.py",ROOT/"preparations/idea2/experiments.py"]
    return dict(status="planned",args=vars(args),source=str(source),source_result_sha256=sha(source/"result.json"),
        bank=str(bank),bank_sha256=sha(bank),canonical_sha256=sha(source/"canonical_fresh.npz"),
        checkpoint=manifest["args"]["checkpoint"],source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files},
        D=10,H=10,R=5,samples=100,families=[16,32],schedules=3,ae_runs=600,prefix_builds=200,
        kv_contract="Exactly two full Prefix caches per sample (actual previous/current). StaticCache only holds references; no copied full KV or Graph. Release all references before next sample.",
        causal_contract="Within each family/sample, all early frontier choices fixed to recorded tail0; only final K entries become18. Same saved explicit noise/current/old inputs. No parallel scheduling jitter.",
        reference="Saved canonical original eager full-current-KV actions, no new third Prefix",
        replay_gate=dict(normalized_max_abs=1e-5,normalized_mae=1e-6,decoded_max_abs=1e-5,decoded_mae=1e-6),
        timing_scope="Serial causal quality ablation, no latency/speedup interpretation")


def replay_sample(backend,sample,noise_array,references,stored,index):
    import torch
    from preparations.idea2.cewp_preparation import PreparedModulationCore
    from preparations.idea2.experiments import native_raw
    from preparations.idea2.openpi_backend import Prefix,cache_tensors
    from preparations.idea2.layer_bench import difference
    prepared=backend.prepare(native_raw(sample["current"]))
    old_prepared=backend.prepare(native_raw(sample["old"]))
    old=backend.build_prefix(old_prepared);new=backend.build_prefix(prepared)
    tensors=(*cache_tensors(old.cache),*cache_tensors(new.cache))
    refs=[weakref.ref(v) for v in (old,new,*tensors)]
    versions=[v._version for v in tensors]
    old_addresses={v.data_ptr() for v in cache_tensors(old.cache)}
    new_addresses={v.data_ptr() for v in cache_tensors(new.cache)}
    if old_addresses&new_addresses:raise RuntimeError("Old/current full KV alias")
    noise=torch.from_numpy(noise_array.copy()).to(backend.device)
    core=PreparedModulationCore(backend,prepared)
    rows=[]
    for sms in (16,32):
        original=stored[f"sm{sms}_tail0"][index]
        family_outputs=[]
        for tail,frontiers in enumerate(schedules(original["frontiers"])):
            class StaticCache:
                step=0
                def __getattr__(self,name):return getattr(new.cache,name)
                def __len__(self):return len(new.cache)
                def __getitem__(self,layer):return new.cache[layer] if layer<frontiers[self.step] else old.cache[layer]
                def update(self,*args,**kwargs):raise RuntimeError("Read-only counterfactual cache")
            cache=StaticCache();selected=Prefix(cache,new.pad_masks,new.position_ids)
            original_velocity=core.velocity;steps=[]
            def velocity(prefix,p,actions,timestep):
                cache.step=len(steps);steps.append(cache.step)
                return original_velocity(selected,p,actions,timestep)
            core.velocity=velocity
            try:
                raw=core.generate(selected,prepared,noise)
                normalized=raw[0].detach().cpu().numpy().copy()
                decoded=backend.output(prepared,raw).copy()
            finally:core.velocity=original_velocity
            if steps!=list(range(10)):raise RuntimeError("Euler round count changed")
            gate=None
            if tail==0:
                gate=dict(normalized=difference(original["normalized"],normalized),decoded=difference(original["decoded"],decoded))
                if any(not value["finite"] or value["max_abs"]>1e-5 or value["mae"]>1e-6 for value in gate.values()):
                    raise RuntimeError("Tail0 exact-schedule replay differs from saved actions: "+str(gate))
            family_outputs.append(decoded)
            rows.append(dict(sample=index,family_ae_sm_min=sms,fresh_tail_steps=tail,input_hash=sample["input_hash"],
                old_input_hash=sample["old_input_hash"],seed=sample["seed"],noise_sha256=original["noise_sha256"],
                observed_tail0_frontiers=original["frontiers"],counterfactual_frontiers=frontiers,
                replay_gate=gate,approximation_delta=delta(references[index],decoded),
                delta_from_fixed_tail0=delta(family_outputs[0],decoded),normalized=normalized.tolist(),decoded=decoded.tolist()))
            del raw,selected,cache
    if [v._version for v in tensors]!=versions:raise RuntimeError("Read-only KV tensor mutated")
    return rows,refs


def summarize(rows):
    table=[];paired=[]
    for sms in (16,32):
        family=[r for r in rows if r["family_ae_sm_min"]==sms]
        tail0={r["sample"]:r for r in family if r["fresh_tail_steps"]==0}
        for tail in (0,1,2):
            selected=[r for r in family if r["fresh_tail_steps"]==tail]
            if not selected:continue
            summary=dict(family_ae_sm_min=sms,fresh_tail_steps=tail,samples=len(selected))
            for h in ("R5","H10"):
                summary[h]=dict(mae=float(np.mean([r["approximation_delta"][h]["mae"] for r in selected])),
                    max_abs=max(r["approximation_delta"][h]["max_abs"] for r in selected),
                    per_dimension={d:dict(mae=float(np.mean([r["approximation_delta"][h]["per_dimension"][d]["mae"] for r in selected])),
                        max_abs=max(r["approximation_delta"][h]["per_dimension"][d]["max_abs"] for r in selected)) for d in DIMENSIONS})
                if tail:
                    changes=np.array([r["approximation_delta"][h]["mae"]-tail0[r["sample"]]["approximation_delta"][h]["mae"] for r in selected])
                    paired.append(dict(family_ae_sm_min=sms,fresh_tail_steps=tail,horizon=h,samples=len(selected),
                        paired_mae_change=float(changes.mean()),improved_inputs=int(np.sum(changes<0)),
                        unchanged_inputs=int(np.sum(changes==0)),worsened_inputs=int(np.sum(changes>0))))
            table.append(summary)
    return table,paired


def run(args,manifest):
    import torch
    from preparations.idea2.experiments import load_backend
    output=Path(args.output).resolve();source=Path(manifest["source"])
    original=json.loads((source/"result.json").read_text())
    stored={v["configuration"]:v["rows"] for v in original["variants"] if v["configuration"].endswith("tail0")}
    samples=pickle.loads(Path(manifest["bank"]).read_bytes())
    with np.load(source/"canonical_fresh.npz",allow_pickle=False) as arrays:
        noises=arrays["noise"].copy();references=arrays["actions"].copy()
    backend=load_backend(manifest["checkpoint"],model_config="pi05_libero",num_steps=10)
    uuid=str(torch.cuda.get_device_properties(backend.device).uuid)
    if uuid.removeprefix("GPU-")!=args.gpu_uuid.removeprefix("GPU-"):raise ValueError("Assigned GPU differs")
    result=dict(status="running",manifest=manifest,gpu_uuid=uuid,rows=[],start_unix=time.time())
    try:
        for index,sample in enumerate(samples):
            if hashlib.sha256(noises[index].tobytes()).hexdigest()!=stored["sm16_tail0"][index]["noise_sha256"]:
                raise ValueError("Explicit noise bytes changed")
            rows,refs=replay_sample(backend,sample,noises[index],references,stored,index)
            gc.collect()
            if any(ref() is not None for ref in refs):raise RuntimeError("Previous sample still retains full KV")
            result["rows"].extend(rows)
            if (index+1)%10==0:
                save(output/"result.json",result);print("counterfactual samples",index+1,flush=True)
        result.update(status="complete",end_unix=time.time(),ae_runs=len(result["rows"]),full_kv_python_references_after_sample=0)
        result["summary"],result["paired_tail_effects"]=summarize(result["rows"])
        save(output/"result.json",result)
        for name,key in (("per_input","rows"),("summary","summary"),("paired_tail_effects","paired_tail_effects")):
            write_csv(output/(name+".csv"),result[key])
    except BaseException:
        result.update(status="failed",traceback=traceback.format_exc(),end_unix=time.time());save(output/"result.json",result);raise


def main():
    args=parser().parse_args();output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    manifest=plan(args);path=output/"plan.json"
    if path.exists():
        previous=json.loads(path.read_text())
        for key in ("source_result_sha256","bank_sha256","canonical_sha256","source_sha256","checkpoint"):
            if previous[key]!=manifest[key]:raise ValueError("Frozen counterfactual plan changed: "+key)
        for key in ("source","gpu_uuid","cpus"):
            if previous["args"][key]!=manifest["args"][key]:raise ValueError("Counterfactual args changed")
    save(path,manifest)
    if not args.execute:
        print(json.dumps(dict(status="planned_only",samples=100,ae_runs=600,plan=str(path))));return
    if (output/"result.json").exists():raise ValueError("Refusing to overwrite results")
    import os
    from results.idea2.progressive_kv_20261025.closedloop import parse_cpus
    affinity=parse_cpus(args.cpus)
    if not affinity<=os.sched_getaffinity(0):raise ValueError("CPU affinity not available")
    os.sched_setaffinity(0,affinity)
    run(args,manifest)


if __name__=="__main__":main()

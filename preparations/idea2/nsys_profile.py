"""Bounded NVTX diagnosis of unchanged eager E1 and real PipelinePolicy replay.

NVTX ranges describe host submission scopes, never GPU busy intervals. Only
Nsight CUDA kernels/events on real streams can establish device overlap.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import functools
import hashlib
import inspect
import json
import os
from pathlib import Path
import pickle

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODES = ("fresh_serial","serial_reuse","async_reuse")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""):
            result.update(block)
    return result.hexdigest()


def json_file(path,value):
    with Path(path).open("x") as stream:
        json.dump(value,stream,indent=2)
        stream.write("\n")


def compare(a,b):
    a,b = np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    return dict(shape=list(a.shape),max_abs=float(np.max(np.abs(a-b))),
        passed=bool(np.allclose(a,b,atol=1e-5,rtol=1e-5) and np.max(np.abs(a-b))<=1e-5))


class NVTX:
    """Instance-forward wrappers cover both module(...) and direct.forward(...).

    No forward hooks (which direct.forward bypasses), tensor transforms, CUDA
    events, new streams or synchronization are introduced by these wrappers.
    """
    def __init__(self,backend,sink,cache_tensors,coarse_only=False):
        self.backend,self.sink,self.cache_tensors = backend,sink,cache_tensors
        self.coarse_only = coarse_only
        self.enabled,self.counts,self.saved = False,Counter(),[]
        self.camera,self.step = 0,0
        self.context,self.outputs,self.kv_checks = {},[],[]
        bundle = backend.model.paligemma_with_expert
        self.bundle = bundle
        self.vision = list(bundle.paligemma.vision_tower.vision_model.encoder.layers)
        self.vlm = list(bundle.paligemma.language_model.layers)
        self.ae = list(bundle.gemma_expert.model.layers)
        self.depths = dict(vision=len(self.vision),vlm=len(self.vlm),action_expert=len(self.ae),denoising_steps=backend.num_steps)
        assert self.depths==dict(vision=27,vlm=18,action_expert=18,denoising_steps=10),self.depths
        self.camera_names = []

    @contextmanager
    def span(self,label):
        if not self.enabled:
            yield
            return
        self.counts[label] += 1
        self.sink.range_push(label)
        try:
            yield
        finally:
            self.sink.range_pop()

    def patch(self,obj,name,callback):
        original = getattr(obj,name)
        owned = name in vars(obj)
        previous = vars(obj).get(name)
        @functools.wraps(original)
        def wrapped(*args,**kwargs):
            if not self.enabled:
                return original(*args,**kwargs)
            return callback(original,*args,**kwargs)
        self.saved.append((obj,name,owned,previous,original))
        setattr(obj,name,wrapped)

    def simple(self,obj,name,label):
        def call(original,*args,**kwargs):
            with self.span(label() if callable(label) else label):
                return original(*args,**kwargs)
        self.patch(obj,name,call)

    def metadata(self,prefix):
        return [dict(object_id=id(t),storage=t.data_ptr(),version=t._version,shape=list(t.shape),dtype=str(t.dtype))
            for t in self.cache_tensors(prefix.cache)]

    def install(self,camera_names):
        self.camera_names = list(camera_names)
        def encode(original,*args,**kwargs):
            previous,self.camera = self.camera,0
            try:
                with self.span("encode"):
                    result = original(*args,**kwargs)
                assert self.camera==len(self.camera_names)
                return result
            finally:
                self.camera = previous
        def image(original,*args,**kwargs):
            self.camera += 1
            assert self.camera<=len(self.camera_names)
            with self.span(f"encode/camera_BRANCH{self.camera:02d}_{self.camera_names[self.camera-1]}"):
                return original(*args,**kwargs)
        def denoise(original,*args,**kwargs):
            self.step += 1
            assert 1<=self.step<=self.backend.num_steps
            with self.span(f"action_expert/denoise_step{self.step:02d}"):
                return original(*args,**kwargs)
        def generate(original,prefix,*args,**kwargs):
            previous,self.step = self.step,0
            before = self.metadata(prefix)
            try:
                with self.span("action_expert"):
                    result = original(prefix,*args,**kwargs)
                assert self.step==self.backend.num_steps
                after = self.metadata(prefix)
                self.kv_checks.append(dict(context=dict(self.context),unchanged=before==after,before=before,after=after))
                assert before==after,"Prefix storage/version changed during generation"
                # Retain controller-owned raw outputs, NOT extra KV snapshots.
                # No clone/D2H/hash/sync occurs here during capture.
                self.outputs.append(dict(context=dict(self.context),raw=result))
                return result
            finally:
                self.step = previous
        self.patch(self.backend.model,"embed_prefix",encode)
        self.patch(self.bundle,"embed_image",image)
        self.patch(self.backend.model,"denoise_step",denoise)
        self.patch(self.backend,"generate",generate)
        self.simple(self.backend,"build_prefix","prefill")
        self.simple(self.backend,"prepare","input_preparation")
        self.simple(self.backend,"noise","noise_preparation")
        self.simple(self.backend,"output","output_copy_and_decode")
        self.simple(self.bundle.paligemma.language_model,"forward","vlm")
        if not self.coarse_only:
            self.simple(self.bundle.paligemma.multi_modal_projector,"forward","encode/projector")
            for i,layer in enumerate(self.vision,1):
                self.simple(layer,"forward",f"vision/L{i:02d}")
            for i,layer in enumerate(self.vlm,1):
                self.simple(layer,"forward",f"vlm/L{i:02d}")
            for i,layer in enumerate(self.ae,1):
                self.simple(layer,"forward",lambda i=i:f"action_expert/denoise_step{self.step:02d}/layer{i:02d}")
        return self

    def attach_policy(self,policy):
        self.simple(policy,"reset","runtime/reset")
        self.simple(policy.pool,"drain_reset","runtime/drain_owned_events")

    def clear(self):
        self.counts.clear()
        self.outputs.clear()
        self.kv_checks.clear()

    def validate(self,expected_generations):
        count = self.counts
        assert count["action_expert"]==expected_generations>0
        prefixes = count["prefill"]
        assert prefixes>0 and count["encode"]==count["vlm"]==prefixes
        for i,name in enumerate(self.camera_names,1):
            assert count[f"encode/camera_BRANCH{i:02d}_{name}"]==prefixes
        for step in range(1,11):
            assert count[f"action_expert/denoise_step{step:02d}"]==expected_generations
        if not self.coarse_only:
            assert count["encode/projector"]==prefixes*len(self.camera_names)
            for i in range(1,28):
                assert count[f"vision/L{i:02d}"]==prefixes*len(self.camera_names)
            for i in range(1,19):
                assert count[f"vlm/L{i:02d}"]==prefixes
                for step in range(1,11):
                    assert count[f"action_expert/denoise_step{step:02d}/layer{i:02d}"]==expected_generations
        assert len(self.outputs)==len(self.kv_checks)==expected_generations and all(r["unchanged"] for r in self.kv_checks)
        return dict(passed=True,depths=self.depths,camera_names=self.camera_names,coarse_only=self.coarse_only,counts=dict(count),
            generated_calls=expected_generations,prefix_calls=prefixes,layer_counts_required=not self.coarse_only)

    def restore(self):
        self.enabled = False
        restored = []
        for obj,name,owned,previous,original in reversed(self.saved):
            if owned:
                setattr(obj,name,previous)
            else:
                delattr(obj,name)
            restored.append(getattr(obj,name)==original)
        self.saved.clear()
        assert all(restored),"Runtime method restoration failed"
        return True


def run(args):
    import gc
    import torch
    from . import experiments
    from .openpi_backend import cache_tensors
    from .policy import PipelinePolicy
    from preparations.idea1_v2_1.runtime_client import input_digest,value_digest
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    payload = Path(args.bank).read_bytes()
    bank = pickle.loads(payload)  # trusted local real observation bank only
    assert len(bank)>=max(2,args.runtime_requests)
    records = bank[:20]
    instruction = records[0]["instruction"]
    assert all(r["instruction"]==instruction and r.get("suite","libero_spatial")=="libero_spatial" for r in records)
    backend = experiments.load_backend(args.checkpoint,args.device)
    prepared = [backend.prepare(experiments.native_raw(r)) for r in records[:args.runtime_requests]]
    current,next_observation = prepared[:2]
    prefix = backend.build_prefix(current)
    noise = backend.noise(args.seed)
    policy = PipelinePolicy(backend, submission_order="prefill_first", host_workers=False)
    source_paths = {Path(__file__),Path(experiments.__file__),ROOT/"preparations/idea2/openpi_backend.py",
        ROOT/"preparations/idea2/policy.py",ROOT/"preparations/idea2/scheduler.py"}
    bundle = backend.model.paligemma_with_expert
    for obj in (backend.model,bundle,bundle.paligemma.vision_tower.vision_model.encoder.layers[0],
        bundle.paligemma.language_model,bundle.paligemma.language_model.layers[0],bundle.gemma_expert.model.layers[0],
        bundle.paligemma.multi_modal_projector):
        source_paths.add(Path(inspect.getfile(type(obj))))
    source_hashes = {str(p.resolve()):sha(p) for p in source_paths}
    identity = dict(checkpoint=str(Path(args.checkpoint).resolve()),weights_sha256=sha(Path(args.checkpoint)/"model.safetensors"),
        bank=str(Path(args.bank).resolve()),bank_sha256=hashlib.sha256(payload).hexdigest(),backend=backend.identity,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),implementation_sha256=source_hashes,args=vars(args),
        frames=[dict(bank_index=i,input_hash=input_digest(r["obs"],instruction),original_record_seed=r.get("seed"),
            diagnostic_noise_seed=args.seed+i,suite="libero_spatial",instruction=instruction) for i,r in enumerate(records[:args.runtime_requests])],
        capture_scope="Original experiments.pair plus original PipelinePolicy.predict/reset real-bank replay, NOT simulation or quality evaluation",
        caveat="NVTX is host submission scope, not GPU busy time. Original pair/policy waits, scalar setup and stream choices are unchanged. Detailed NVTX can perturb host submission; coarse-only is an optional diagnostic contrast.")
    json_file(output/"identity.json",identity)
    trace = None
    result = dict(status="starting",warmup=20,capture_api="cudaProfilerApi",runtime_requests=args.runtime_requests,e1_iterations=args.iterations)
    capture_started = False
    try:
        # All model warmup, input preparation and numerical checks are outside
        # cudaProfilerStart/Stop. pair itself owns all its existing waits.
        for _ in range(20):
            for concurrent in (False,True):
                _,_,fresh = experiments.pair(backend,prefix,current,next_observation,noise,concurrent)
                del fresh
        for mode in MODES:
            policy.configure(mode,1)
            for i in range(20):
                r = records[i % len(records)]
                policy.predict(obs=r["obs"],instruction=instruction,suite="libero_spatial",seed=args.seed+i)
            policy.reset()
        kv_before = [t.detach().cpu().clone() for t in cache_tensors(prefix.cache)]
        base_raw = backend.generate(prefix,current,noise)
        base_decoded = backend.output(current,base_raw)
        native_raw = backend.original(current,noise)
        native_parity = dict(raw=compare(native_raw.detach().cpu(),base_raw.detach().cpu()),
            decoded=compare(backend.output(current,native_raw),base_decoded))
        assert all(c["passed"] for c in native_parity.values()),"Original native/split parity failed"
        camera_names = list(current.observation.images)
        trace = NVTX(backend,torch.cuda.nvtx,cache_tensors,args.coarse_only)
        trace.install(camera_names)
        trace.attach_policy(policy)
        trace.enabled = True
        decorated = []
        for concurrent in (False,True):
            trace.context = dict(section="outside_capture_parity",mode="two_stream" if concurrent else "serial")
            decoded,_,fresh = experiments.pair(backend,prefix,current,next_observation,noise,concurrent)
            del fresh
            raw = trace.outputs[-1]["raw"]
            checks = dict(raw=compare(base_raw.detach().cpu(),raw.detach().cpu()),decoded=compare(base_decoded,decoded))
            assert all(c["passed"] for c in checks.values())
            decorated.append(checks)
        assert all(torch.equal(a,b.detach().cpu()) for a,b in zip(kv_before,cache_tensors(prefix.cache),strict=True))
        trace.clear()
        # One explicit fence before capture only. No added request-to-request
        # synchronization is performed inside runtime replay.
        torch.cuda.synchronize(backend.device)
        torch.cuda.profiler.start()
        capture_started = True
        e1_rows,runtime_rows = [],[]
        with trace.span("section/E1_fixed_KV_causal_reference"):
            for iteration in range(args.iterations):
                for concurrent in (False,True):
                    mode = "two_stream" if concurrent else "serial"
                    trace.context = dict(section="E1",mode=mode,iteration=iteration,seed=args.seed)
                    with trace.span(f"E1/{mode}/iteration_{iteration+1:02d}"):
                        decoded,timing,fresh = experiments.pair(backend,prefix,current,next_observation,noise,concurrent)
                    del fresh
                    e1_rows.append(dict(trace.context,decoded=decoded,timing=timing))
        with trace.span("section/runtime_PipelinePolicy_real_observation_replay"):
            for mode in MODES:
                with trace.span(f"runtime/{mode}/configure_reset"):
                    policy.configure(mode,1)
                for i,r in enumerate(records[:args.runtime_requests]):
                    trace.context = dict(section="runtime",mode=mode,request=i,seed=args.seed+i)
                    with trace.span(f"runtime/{mode}/request_{i:02d}"):
                        response = policy.predict(obs=r["obs"],instruction=instruction,suite="libero_spatial",seed=args.seed+i)
                    runtime_rows.append(dict(response,actions=np.asarray(response["actions"]).copy(),bank_index=i))
                with trace.span(f"runtime/{mode}/final_reset_and_owned_drain"):
                    policy.reset()
        torch.cuda.profiler.stop()
        capture_started = False
        trace.enabled = False
        label_proof = trace.validate(2*args.iterations+3*args.runtime_requests)
        captured_outputs = list(trace.outputs)
        kv_checks = list(trace.kv_checks)
        trace.restore()
        kv_capture_unchanged = all(torch.equal(a,b.detach().cpu()) for a,b in zip(kv_before,cache_tensors(prefix.cache),strict=True))
        assert kv_capture_unchanged
        # Restoration parity, source-version reconstruction and D2H diagnostics
        # happen strictly AFTER cudaProfilerStop, never inside the measured path.
        after_raw = backend.generate(prefix,current,noise)
        after_parity = dict(raw=compare(base_raw.detach().cpu(),after_raw.detach().cpu()),decoded=compare(base_decoded,backend.output(current,after_raw)))
        assert all(c["passed"] for c in after_parity.values())
        e1_raw = [r for r in captured_outputs if r["context"]["section"]=="E1"]
        for row,raw in zip(e1_rows,e1_raw,strict=True):
            row["raw_parity"] = compare(base_raw.detach().cpu(),raw["raw"].detach().cpu())
            row["decoded_parity"] = compare(base_decoded,row.pop("decoded"))
            assert row["raw_parity"]["passed"] and row["decoded_parity"]["passed"]
        runtime_raw = [r for r in captured_outputs if r["context"]["section"]=="runtime"]
        for row,raw in zip(runtime_rows,runtime_raw,strict=True):
            i,source,seed = row["bank_index"],row["prefix_observation_id"],row["seed"]
            assert 0<=source<=i and row["current_observation_id"]==row["suffix_state_id"]==row["reference_state_id"]==i
            assert row["input_hash"]==identity["frames"][i]["input_hash"]
            check_prefix = backend.build_prefix(prepared[source])
            fixed_noise = backend.noise(seed)
            reference = backend.generate(check_prefix,prepared[i],fixed_noise)
            row["raw_same_recorded_source_parity"] = compare(reference.detach().cpu(),raw["raw"].detach().cpu())
            row["decoded_same_recorded_source_parity"] = compare(backend.output(prepared[i],reference),row["actions"])
            assert row["raw_same_recorded_source_parity"]["passed"] and row["decoded_same_recorded_source_parity"]["passed"]
            row["noise_hash"] = value_digest(fixed_noise.detach().cpu().numpy())
            row["actions"] = row["actions"].tolist()
            del check_prefix,reference,fixed_noise
        matching_runtime = []
        for i in range(args.runtime_requests):
            same_request = [r for r in runtime_rows if r["request"]==i]
            for left,right in zip(same_request,same_request[1:]):
                matched = left["prefix_observation_id"]==right["prefix_observation_id"]
                check = compare(left["actions"],right["actions"]) if matched else None
                assert check is None or check["passed"]
                matching_runtime.append(dict(request=i,left=left["mode"],right=right["mode"],same_prefix_source=matched,decoded_check=check))
        assert {str(p.resolve()):sha(p) for p in source_paths}==source_hashes
        result.update(status="completed",native_parity=native_parity,decorated_before_capture_parity=decorated,
            restored_after_capture_parity=after_parity,fixed_kv_bytes_unchanged=kv_capture_unchanged,
            kv_version_checks=kv_checks,label_proof=label_proof,e1=e1_rows,runtime=runtime_rows,runtime_source_matched_checks=matching_runtime,
            methods_restored=True,warmup_distinct_records=len(records),notes=["Original E1 pair and runtime policy paths executed unchanged; no graph variant.",
                "Warmup is 20 requests per mode, cycling the available real bank when it contains fewer than 20 distinct observations.",
                "A fixed E1 prefix remains resident through capture only to verify its full KV bytes after stop; original runtime pool still has two slots. This is a diagnostic, not an uninstrumented performance measurement.",
                "Runtime cache publication/source age may change with profiling overhead. All actual source IDs are retained; each output is checked against its own recorded source, never falsely paired with another age.",
                "Only tensor identity/storage/version metadata is sampled inside generate; complete fixed-prefix KV bytes and all numerical parity are checked outside capture."])
    except BaseException as exc:
        result.update(status="failed",error=repr(exc))
        raise
    finally:
        if capture_started:
            torch.cuda.profiler.stop()
        if trace is not None:
            trace.restore()
        policy.reset()
        json_file(output/"result.json",result)
        if trace is not None:
            trace.clear()
        del policy,prefix,prepared,noise,backend
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",default="shared/checkpoints/openpi/pi0_libero_pytorch_idea2")
    parser.add_argument("--bank",required=True,help="Trusted spatial task00/init07 real_predict_inputs_first25.pkl")
    parser.add_argument("--output",required=True,help="NEW diagnostic data directory; nsys report is generated separately by parent")
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--iterations",type=int,choices=range(3,6),default=3)
    parser.add_argument("--runtime-requests",type=int,choices=range(4,7),default=5)
    parser.add_argument("--seed",type=int,default=1701)
    parser.add_argument("--coarse-only",action="store_true")
    parser.add_argument("--launch",action="store_true")
    args = parser.parse_args()
    if Path(args.output).exists() or Path(args.output).is_symlink():
        parser.error("Output must be NEW; refusing overwrite")
    if not args.launch:
        print(json.dumps(dict(dry_run=True,e1_iterations_per_mode=args.iterations,runtime_requests_per_mode=args.runtime_requests,
            runtime_modes=MODES,warmup_per_mode=20,capture_range="cudaProfilerApi",coarse_only=args.coarse_only)))
        return
    run(args)


if __name__=="__main__":
    main()

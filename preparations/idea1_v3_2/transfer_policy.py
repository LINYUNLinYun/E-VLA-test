"""Zero-shot transfer wrapper around the unchanged released Triton pi0 paths.

Only client-oriented RGB/state input and dynamic CPU prompt lookup are new.
No model construction occurs on import or during CPU metadata preflight.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np

from preparations.idea1_v3.policy import TritonBackend,V14
from .transfer_prompts import ROOT,TOKENIZER,instruction_embedding_from_base,instruction_tokens,sha

MODES = ("full_d10","v14_d10","draft_goal","draft_object","draft_spatial","draft_10")
DRAFT_SUITES = dict(draft_goal="libero_goal",draft_object="libero_object",
                    draft_spatial="libero_spatial",draft_10="libero_10")
ARTIFACTS = dict(draft_goal="draft",draft_object="draft_object",draft_spatial="draft_spatial",draft_10="draft_10")
CONVERTED = ROOT/"shared/checkpoints/realtime-vla-flash/converted"
PROMPT_PROOF = ROOT/"results/idea1_v3_2/e4_prompt_preflight/cpu_native_sp_v1/preflight.json"
OBSERVED_NAMES = ("vision_encoder_calls","vlm_prefix_calls","ae_layer_calls","learned_head_calls",
                  "verify_calls","constructor_full_record_calls","constructor_graph_capture_suppressed")


def observed_counts(counter):
    return {name:int(counter.get(name,0)) for name in OBSERVED_NAMES}


def validate_raw(raw):
    if set(raw)!={"base_rgb","wrist_rgb","state8"}:
        raise ValueError("raw must contain exactly base_rgb,wrist_rgb,state8")
    result = {}
    for key in ("base_rgb","wrist_rgb"):
        image = np.asarray(raw[key])
        if image.ndim!=3 or image.shape[-1]!=3 or image.dtype!=np.uint8:
            raise ValueError("Client-oriented RGB must be HxWx3 uint8: "+key)
        result[key] = np.ascontiguousarray(image)
    state = np.asarray(raw["state8"],np.float32)
    if state.shape!=(8,) or not np.isfinite(state).all():
        raise ValueError("state8 must be finite shape[8]")
    result["state8"] = state
    return result


def json_safe(value):
    if isinstance(value,dict):
        return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,Path):
        return str(value)
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    raise TypeError("RPC response contains unsupported non-Python value: "+str(type(value)))


def prompt_proof():
    proof = json.loads(PROMPT_PROOF.read_text())
    if proof.get("status")!="complete" or not proof.get("passed") or len(proof["released_checks"])<5:
        raise ValueError("Completed released-prompt CPU equivalence proof required")
    for path,expected in proof["source_sha256"].items():
        if sha(path)!=expected:
            raise ValueError("CPU prompt proof source changed: "+path)
    if sha(TOKENIZER)!=proof["tokenizer"]["sha256"]:
        raise ValueError("Tokenizer differs from CPU prompt proof")
    return dict(path=str(PROMPT_PROOF),sha256=sha(PROMPT_PROOF),released_prompts=len(proof["released_checks"]),
                bitwise_equal=all(r["bitwise_equal"] for r in proof["released_checks"]),scope=proof["scope"])


def cpu_preflight():
    sources = [Path(__file__),Path(__file__).with_name("transfer_server.py"),Path(__file__).with_name("transfer_prompts.py"),
        ROOT/"preparations/idea1_v3/policy.py",ROOT/"preparations/idea1_v2_2/pi0_adapter.py",
        ROOT/"policies/realtime-vla-flash/scripts/experiment10/backend.py",
        ROOT/"policies/realtime-vla-flash/scripts/spec/triton/pi0_spec_infer.py",
        ROOT/"policies/realtime-vla-flash/scripts/spec/triton/pi0_infer.py"]
    base = CONVERTED/"base/base_weights.pkl"
    base_hash = sha(base)
    provenance = ROOT/"policies/realtime-vla-flash/results/experiment11_layer_skip_prefix_20261010/provenance/preflight.json"
    old = json.loads(provenance.read_text())
    if base_hash!=old["checkpoint"]["sha256"]:
        raise ValueError("Shared base checkpoint differs from old released protocol")
    return dict(status="complete",cpu_only=True,modes=list(MODES),H=50,R=5,D=10,
        checkpoint_metadata=dict(shared_base=dict(path=str(base),sha256=base_hash),
            original_provenance=dict(path=str(provenance),sha256=sha(provenance)),
            draft_heads={mode:dict(path=str(CONVERTED/artifact/"draft_triton.pkl"),
                sha256=sha(CONVERTED/artifact/"draft_triton.pkl")) for mode,artifact in ARTIFACTS.items()}),
        prompt_proof=prompt_proof(),source_sha256={str(p):sha(p) for p in sources},
        semantics=dict(full="full27 vision, full18 VLM, original18 AE, uniformD10",v14="same but VLM skip1based[3,4,9,11]",
            learned="shared full27 vision and native language preencoder plus learned state path; one fixed released draft head, no VLM/AE/verify/fallback",
            images="RGB already oriented by client, no LIBERO flips",prefix="fresh each request; no cross-request KV reuse",
            graphs="full constructor capture disabled; draft graph disabled to count actual eager route calls"),
        model_constructed=False,gpu_used=False,limitations="CPU prompt and metadata checks do not establish policy or environment/action mapping parity")


@contextmanager
def local_assets_only():
    """Redirect only the known tokenizer URI to its existing local file."""
    from openpi.shared import download
    original = download.maybe_download
    def local(path,*args,**kwargs):
        text = str(path)
        if text=="gs://big_vision/paligemma_tokenizer.model":
            return TOKENIZER
        candidate = Path(text)
        if "://" in text:
            raise RuntimeError("LOCAL_ASSET_REQUIRED; downloads disabled: "+text)
        if not candidate.exists():
            # Preserve original optional norm-stat lookup semantics: its
            # caller catches missing local files, then loads checkpoint assets.
            raise FileNotFoundError(candidate)
        return candidate
    download.maybe_download = local
    try:
        yield
    finally:
        download.maybe_download = original


@contextmanager
def route_observations():
    """Count real function entries, not estimates of low-level GEMM counts."""
    from scripts.spec.triton import pi0_infer as kernels
    from scripts.spec.triton.pi0_spec_infer import Pi0SpecInference
    from scripts.experiment1 import run_intermediate_action_probe as probe
    counts = Counter()
    patches = []
    def patch(owner,name,label,suppress=False):
        original = getattr(owner,name)
        def observed(*args,**kwargs):
            counts[label] += 1
            return None if suppress else original(*args,**kwargs)
        setattr(owner,name,observed)
        patches.append((owner,name,original))
    patch(kernels,"vision_encoder","vision_encoder_calls")
    patch(kernels,"transformer_encoder","vlm_prefix_calls")
    patch(probe,"_decoder_layer","ae_layer_calls")
    patch(Pi0SpecInference,"_run_draft_block","learned_head_calls")
    patch(Pi0SpecInference,"run_verify","verify_calls")
    patch(kernels.Pi0Inference,"record_run","constructor_full_record_calls")
    patch(kernels.Pi0Inference,"record_infer_graph","constructor_graph_capture_suppressed",suppress=True)
    try:
        yield counts
    finally:
        for owner,name,original in reversed(patches):
            setattr(owner,name,original)


class RawTritonBackend(TritonBackend):
    def __init__(self):
        sys.path.insert(0,str(ROOT/"policies/realtime-vla-flash/src"))
        sys.path.insert(0,str(ROOT/"policies/realtime-vla-flash/packages/openpi-client/src"))
        os.environ["SPEC_TRITON_DRAFT_GRAPH"] = "0"
        os.environ.setdefault("JAX_PLATFORMS","cpu")
        with local_assets_only():
            super().__init__()
        self.dynamic_prompt_rows = {}
        self.base_prompt_checks = []
        proof = json.loads(PROMPT_PROOF.read_text())
        for row in proof["released_checks"]:
            prompt = row["instruction"]
            result = instruction_embedding_from_base(prompt,self.adapter.backend.checkpoint)
            suite = self.adapter.prompt_suites[prompt]
            actual = self.adapter.backend.embeddings[(suite,prompt)]
            if not self.torch.equal(result["embeddings"],actual):
                raise ValueError("Existing CPU base table differs from released prompt: "+prompt)
            self.base_prompt_checks.append(dict(instruction=prompt,bitwise_equal=True,shape=list(actual.shape)))

    def ensure_prompt(self,instruction):
        instruction_tokens(instruction)  # Existing-cache prompts also reject truncation.
        adapter = self.adapter
        if instruction not in adapter.prompt_suites:
            result = instruction_embedding_from_base(instruction,adapter.backend.checkpoint)
            # This suite is only a storage key; draft head choice stays explicit
            # in the request mode and never depends on this bookkeeping bucket.
            suite = "libero_goal"
            adapter.backend.embeddings[(suite,instruction)] = result["embeddings"]
            adapter.prompt_suites[instruction] = suite
            self.dynamic_prompt_rows[instruction] = dict(token_ids=result["token_ids"],rows_sha256=result["rows_sha256"])

    def prepare(self,raw,instruction,skip=(),encode=True):
        from scripts.spec.triton import pi0_infer as kernels
        self.ensure_prompt(instruction)
        adapter = self.adapter
        observation = {"observation/image":raw["base_rgb"],"observation/wrist_image":raw["wrist_rgb"],
                       "observation/state":raw["state8"],"prompt":instruction}
        tick = time.perf_counter()
        previous = adapter.backend.runtime
        adapter.backend.prepare(observation,dict(suite=adapter.prompt_suites[instruction],task_description=instruction),
                                np.zeros((50,32),np.float32))
        runtime = adapter.backend.runtime
        if runtime is not previous:
            self.startup_events.append(dict(kind="request_runtime_construction",prompt=instruction,
                seconds=time.perf_counter()-tick,full_AE_calls=0,graph_capture="suppressed by observed wrapper"))
        runtime._infer_module = kernels
        weights,buffers = runtime.weights,runtime.buffers
        buffers["observation_images_normalized"].copy_(adapter.backend.inputs[0])
        buffers["observation_state_normalized"].copy_(adapter.backend.inputs[1])
        buffers["encoder_x"][512:].copy_(weights["language_embeds"])
        timing = dict(input_processing_ms=(time.perf_counter()-tick)*1000)
        _,timing["vision_ms"] = adapter.measure(lambda:kernels.vision_encoder(weights,buffers,2))
        seq = buffers["encoder_x"].shape[0]
        if encode:
            _,timing["vlm_ms"] = adapter.measure(lambda:kernels.transformer_encoder(weights,buffers,seq,
                vlm_layer_indices=tuple(i for i in range(18) if i+1 not in skip)))
        context = dict(runtime=runtime,transformed_state=np.asarray(adapter.backend.transformed["state"]).copy(),
            state=adapter.backend.inputs[1].clone(),seq=seq,startup_events=[])
        if encode:
            context.update(K=buffers["encoder_K"][:,:seq].clone(),V=buffers["encoder_V"][:,:seq].clone())
        return context,timing

    def sample(self,context,seed,steps):
        actions,raw,_,audit,ms,decode = self.adapter.sample(context,seed,
            dict(kind="native",m=None,num_steps=steps),False)
        return actions,raw,dict(ae_ms=ms,decode_ms=decode,nodes=audit["nodes"],actual_NFE=audit["actual_NFE"])


class TransferPolicy:
    def __init__(self,protocol,manifest_hash,backend=None):
        self.backend = backend or RawTritonBackend()
        self.manifest_hash = manifest_hash
        self.legacy_gate_passed = False
        self.counters = {category:{} for category in ("preflight","warmup","quality")}
        self.counters["preflight_reference"] = {}
        self.protocol = dict(self.backend.protocol,manifest_hash=manifest_hash,modes=list(MODES),
            H=50,R=5,D=10,vision_depth=27,ae_depth=18,image_rotation=0,input_images_preoriented=True,
            dynamic_instruction="native CPU SentencePiece full BF16 rows; released10 bitwise proof",
            learned_paths="fixed released head; no VLM, AE, verification or fallback",
            startup_accounting="constructor full graph suppressed; draft graphs disabled; actual route function entries separate from logical requests",
            parent_protocol=protocol)

    def reset(self):
        return dict(request_local=True,reset_scope="no trajectory cache; cumulative counters preserved",counters=self.counters)

    def predict(self,mode,raw,instruction,seed,category="quality"):
        if mode not in MODES or category not in self.counters:
            raise ValueError("Unknown mode or accounting category")
        if category in ("quality","warmup") and not self.legacy_gate_passed:
            raise RuntimeError("LEGACY_GATE_REQUIRED before cross-environment quality")
        if not isinstance(instruction,str) or not instruction.strip() or not isinstance(seed,(int,np.integer)) or int(seed)<0:
            raise ValueError("Nonempty instruction and nonnegative integer seed required")
        raw = validate_raw(raw)
        from preparations.idea1_v2_1.runtime_client import input_digest,value_digest
        tick = time.perf_counter()
        start = len(self.backend.startup_events)
        observed = Counter()
        cumulative = self.counters[category].setdefault(mode,Counter())
        cumulative["attempted_requests"] += 1
        try:
            with route_observations() as observed:
                context,timing = self.backend.prepare(raw,instruction,V14 if mode=="v14_d10" else (),
                                                     encode=mode not in DRAFT_SUITES)
                if mode in DRAFT_SUITES:
                    actions,model_raw,costs = self.backend.learned(context,DRAFT_SUITES[mode])
                    euler_updates = 0
                else:
                    actions,model_raw,costs = self.backend.sample(context,seed,10)
                    euler_updates = int(costs["actual_NFE"])
        except Exception:
            cumulative.update(observed_counts(observed))
            cumulative["failed_requests"] += 1
            raise
        actual = observed_counts(observed)
        actual["ae_euler_updates"] = euler_updates
        cumulative.update(actual)
        if actual.get("constructor_full_record_calls",0) or actual.get("verify_calls",0):
            raise RuntimeError("Unexpected hidden Full AE startup or verification")
        if mode in DRAFT_SUITES and (actual.get("ae_layer_calls",0) or actual.get("vlm_prefix_calls",0)):
            raise RuntimeError("Learned-only route unexpectedly entered AE/VLM")
        expected = dict(vision_encoder_calls=1,vlm_prefix_calls=0 if mode in DRAFT_SUITES else 1,
            ae_layer_calls=0 if mode in DRAFT_SUITES else 180,learned_head_calls=1 if mode in DRAFT_SUITES else 0)
        if any(actual.get(name,0)!=value for name,value in expected.items()):
            raise RuntimeError("Observed real route entries differ from frozen mode: "+str(actual))
        actions = np.asarray(actions,np.float32)
        model_raw = model_raw.detach().float().cpu().numpy().copy()
        if actions.shape!=(50,7) or not np.isfinite(actions).all() or model_raw.shape!=(50,32) or not np.isfinite(model_raw).all():
            raise ValueError("Invalid native decoded action chunk")
        cumulative["logical_requests"] += 1
        noise = np.random.default_rng(int(seed)).standard_normal((50,32),dtype=np.float32)
        return json_safe(dict(actions=actions,raw_actions=model_raw,execute_length=5,mode=mode,
            seed=int(seed),input_hash=input_digest(raw,instruction),noise_hash=value_digest(noise),
            noise_used_in_inference=mode not in DRAFT_SUITES,manifest_hash=self.manifest_hash,protocol_hash=self.manifest_hash,
            timing=dict(timing,**costs,request_wall_ms=(time.perf_counter()-tick)*1000),
            route=dict(mode=mode,fresh_prefix=True,full_vision=True,
                prefix_kind="vision_language_preencoder" if mode in DRAFT_SUITES else "vlm_kv",
                kv_cross_request=False,skip_vlm=list(V14) if mode=="v14_d10" else [],
                fallback=False,verify=False),observed_calls=actual,category=category,counters=self.counters,
            startup_events=self.backend.startup_events[start:]))

    def legacy_preflight(self,bank):
        from preparations.idea1_v2_1.runtime_models import native_observation
        record = pickle.loads(Path(bank).read_bytes())[0]
        images,state = native_observation(record["obs"])
        raw = dict(base_rgb=np.asarray(images[0]),wrist_rgb=np.asarray(images[1]),state8=np.asarray(state,np.float32))
        checks = []
        for mode in MODES:
            skip = V14 if mode=="v14_d10" else ()
            with route_observations() as observed:
                context,_ = TritonBackend.prepare(self.backend,record["obs"],record["instruction"],skip,
                                                  encode=mode not in DRAFT_SUITES)
                if mode in DRAFT_SUITES:
                    expected,expected_raw,_ = self.backend.learned(context,DRAFT_SUITES[mode])
                else:
                    expected,expected_raw,_ = TritonBackend.sample(self.backend,context,int(record["seed"]),10)
            reference_calls = observed_counts(observed)
            self.counters["preflight_reference"][mode] = dict(reference_calls,logical_requests=1)
            candidate = self.predict(mode,raw,record["instruction"],int(record["seed"]),category="preflight")
            native_raw = expected_raw.detach().float().cpu().numpy()
            pair = {}
            for name,a,b in (("raw",native_raw,np.asarray(candidate["raw_actions"],np.float32)),
                             ("decoded",np.asarray(expected,np.float32),np.asarray(candidate["actions"],np.float32))):
                pair[name] = dict(shape=list(a.shape),bitwise_equal=bool(np.array_equal(a,b)),
                    max_abs=float(np.max(np.abs(a-b))),passed=bool(np.allclose(a,b,atol=1e-5,rtol=1e-5)))
            checks.append(dict(mode=mode,comparison=pair,reference_observed_calls=reference_calls,
                candidate_observed_calls=candidate["observed_calls"]))
        self.legacy_gate_passed = all(c["comparison"][name]["passed"] for c in checks for name in ("raw","decoded"))
        self.protocol["legacy_gate_passed"] = self.legacy_gate_passed
        return dict(passed=self.legacy_gate_passed,bank=str(bank),bank_sha256=sha(bank),checks=checks,
                    scope="Same loaded released weights, actual LIBERO observation/noise: original old orientation path vs client-oriented wrapper; no cross-environment/action mapping claim")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-output",required=True)
    args = parser.parse_args()
    output = Path(args.preflight_output)
    if output.exists():
        raise FileExistsError(output)
    result = cpu_preflight()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x") as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(dict(status=result["status"],modes=result["modes"],output=str(output))))


if __name__ == "__main__":
    main()

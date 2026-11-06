"""Guarded same-model E2 parity and synchronized timing on real observations."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from preparations.idea1_v3_1.common import ROOT, SUITES, atomic_json


def compare(a,b):
    a,b = np.asarray(a,np.float32),np.asarray(b,np.float32)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return dict(shape=list(a.shape),other_shape=list(b.shape),max_abs=None,rmse=None,passed=False)
    return dict(shape=list(a.shape),max_abs=float(np.max(np.abs(a-b))),
        rmse=float(np.sqrt(np.mean((a-b)**2))),
        passed=bool(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                    and np.allclose(a,b,atol=1e-5,rtol=1e-5)))


def summary(values):
    return dict(zip(("p50","p95","p99"),np.percentile(values,[50,95,99]).tolist()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--deadline-unix",required=True,type=float)
    parser.add_argument("--checkpoint-manifest",default="results/idea1_v3_1/manifest_v2.json")
    parser.add_argument("--samples",type=int,default=24)
    parser.add_argument("--warmup",type=int,default=3)
    parser.add_argument("--launch",action="store_true")
    args = parser.parse_args()
    if not args.launch:
        print(json.dumps(dict(dry_run=True,bank=args.bank,modes=["native","refresh_all","reuse03","d2"],
            deadline_unix=args.deadline_unix,output=args.output)))
        return
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Experiment receipt is immutable; select a fresh output")
    sources = [Path(__file__)]+[Path(__file__).with_name(name) for name in
        ("residual.py","residual_server.py","residual_rollout.py")]+[
        ROOT/"policies/openpi/src/openpi/models"/name for name in ("pi0.py","gemma.py","siglip.py")]+[
        ROOT/name for name in ("policies/openpi/src/openpi/policies/policy.py",
            "preparations/idea1_v3_1/rollout.py","preparations/idea1_v3_1/common.py",
            "preparations/idea1_v3_1/models.py","preparations/idea1_v2_1/runtime_models.py")]
    report = dict(status="running",started_unix=time.time(),deadline_unix=args.deadline_unix,
        assigned_budget_start_unix=1792642200.,preparation_ready_unix=1792642768.,
        budget_start_interpretation="Parent assigned 04:10 UTC approximate preparation start; absolute deadline includes all preparation and GPU waiting",
        gpu_unavailable_since_first_ready_s=max(0,time.time()-1792642768.),
        waiting_interpretation="GPU unavailable interval after first ready; includes overlapping CPU refinements and tests, not an exclusive idle-time category",
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        failure_code=None,checks=[],rows=[],scope="pi0 full18 AE, full27 vision, full VLM, H50 R5; no pruning",
        boundary="FP32 residual = output of scanned AE layers before final RMSNorm minus BF16 suffix input cast to FP32",
        recomputed="current x/time/state suffix; current final RMSNorm; action head; fresh full prefix every request",
        cache_reset="each sample invocation allocates fresh zero residual; index0 must refresh",
        latency_scope="AE only uses prefilled device KV; completed local request includes transforms/fresh prefix/device synchronization/decode, excludes RPC",
        tolerance=dict(atol=1e-5,rtol=1e-5))
    atomic_json(output,report)
    def checkpoint():
        atomic_json(output,report)
        if time.time() >= args.deadline_unix:
            raise TimeoutError("E2 total wall deadline reached")
    try:
        checkpoint()
        from preparations.idea1_v3_1.common import read_manifest
        original_manifest = read_manifest(args.checkpoint_manifest)
        report["checkpoint"] = original_manifest["checkpoints"]["pi0"]
        report["checkpoint_manifest_hash"] = original_manifest["manifest_hash"]
        for filename,expected in report["checkpoint"]["identity_artifacts"].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:
                raise ValueError("Checkpoint identity artifact mismatch: "+filename)
        from preparations.idea1_v3_1.benchmark import load_banks
        from preparations.idea1_v2_1.runtime_client import input_digest,value_digest
        from preparations.idea1_v3_1.models import raw_observation
        from .residual import ResidualPolicy,require_idle_gpu
        report["gpu_preflight"] = require_idle_gpu()
        checkpoint()
        policy = ResidualPolicy()
        report["devices"] = [dict(id=d.id,kind=d.device_kind,platform=d.platform) for d in policy.jax.devices()]
        records,banks = load_banks([args.bank])
        report["banks"] = banks
        indices = [next(i for i,r in enumerate(records) if r["suite"]==suite) for suite in SUITES]
        indices += [i for i in range(len(records)) if i not in indices][:max(0,args.samples-len(indices))]
        indices = indices[:args.samples]
        if len(indices)<4:
            raise ValueError("Require at least one representative input per suite")
        prepared = []
        for index in indices:
            record = records[index]
            obs = policy.prepare(record["obs"],record["instruction"])
            noise_np = np.random.default_rng(int(record["seed"])).standard_normal(
                (1,policy.model.action_horizon,policy.model.action_dim),dtype=np.float32)
            prepared.append((record,obs,policy.jnp.asarray(noise_np)))
            report["checks"].append(dict(bank_index=index,suite=record["suite"],
                input_hash=input_digest(record["obs"],record["instruction"]),noise_hash=value_digest(noise_np),modes={}))
        checkpoint()
        # One loaded model, same transformed input, same explicit noise.
        for row,(record,obs,noise) in zip(report["checks"],prepared):
            native = np.asarray(policy.functions["native"](obs,noise))
            decoded = policy.decode(native,obs)
            canonical = policy.policy.infer(raw_observation(record["obs"],record["instruction"]),noise=np.asarray(noise)[0])["actions"]
            row["canonical_policy_infer_decoded"] = compare(decoded,canonical)
            for mode in ("refresh_all","reuse03","d2"):
                result = np.asarray(policy.functions[mode](obs,noise))
                row["modes"][mode] = dict(raw=compare(native,result),decoded=compare(decoded,policy.decode(result,obs)))
            d2_native = np.asarray(policy.functions["d2"](obs,noise))
            d2_refresh = np.asarray(policy.functions["d2_refresh"](obs,noise))
            row["d2_native_parity"] = dict(raw=compare(d2_native,d2_refresh),
                decoded=compare(policy.decode(d2_native,obs),policy.decode(d2_refresh,obs)))
            print(json.dumps(dict(parity_input=row["bank_index"],modes=row["modes"])),flush=True)
            checkpoint()
        report["parity_passed"] = all(r["modes"]["refresh_all"][k]["passed"]
            and r["d2_native_parity"][k]["passed"] and r["canonical_policy_infer_decoded"]["passed"]
            for r in report["checks"] for k in ("raw","decoded"))
        if not report["parity_passed"]:
            raise ArithmeticError("Full-refresh custom boundary does not match native")
        from functools import partial
        from types import MethodType
        from openpi.shared import nnx_utils
        from .residual import integrate
        diagnostic = nnx_utils.module_jit(MethodType(partial(integrate,audit=True),policy.model))
        _,first_obs,first_noise = prepared[0]
        ready = policy.prefill(first_obs)
        audited,counts = diagnostic(ready,first_noise,steps=10)
        report["runtime_audit"] = {k:float(np.asarray(v)) for k,v in counts.items()}
        if counts["euler_updates"]!=10 or counts["transformer_refreshes"]!=2:
            raise ArithmeticError("Runtime Euler/Transformer refresh counts differ from D10 / 0,3")
        first_reuse = np.asarray(policy.functions["reuse03"](first_obs,first_noise))
        for _,other_obs,other_noise in prepared[1:3]:
            policy.jax.block_until_ready(policy.functions["reuse03"](other_obs,other_noise))
        repeat_reuse = np.asarray(policy.functions["reuse03"](first_obs,first_noise))
        report["request_cache_isolation"] = compare(first_reuse,repeat_reuse)
        report["audit_output_parity"] = compare(first_reuse,audited)
        if not report["request_cache_isolation"]["passed"] or not report["audit_output_parity"]["passed"]:
            raise ArithmeticError("Request cache isolation/audited output gate failed")
        checkpoint()
        # Warm each compiled shape/mode; initial compilation is excluded.
        for mode in ("native","refresh_all","reuse03","d2"):
            policy.mode = mode
            for record,obs,noise in prepared[:args.warmup]:
                policy.predict(record["obs"],record["instruction"],int(record["seed"]))
                if mode!="native":
                    ready = policy.prefill(obs)
                    policy.jax.block_until_ready(policy.ae[mode](ready,noise))
            checkpoint()
        # Interleave conditions on the same physical GPU to reduce drift.
        for index,(record,obs,noise) in enumerate(prepared):
            for mode in ("native","refresh_all","reuse03","d2"):
                policy.mode = mode
                response = policy.predict(record["obs"],record["instruction"],int(record["seed"]))
                row = dict(index=index,mode=mode,input_hash=response["input_hash"],
                    noise_hash=response["noise_hash"],request_ms=response["timing"]["request_wall_ms"])
                if mode!="native":
                    ready = policy.prefill(obs)
                    policy.jax.block_until_ready(ready)
                    tick = time.perf_counter()
                    policy.jax.block_until_ready(policy.ae[mode](ready,noise))
                    row["ae_ms"] = (time.perf_counter()-tick)*1000
                report["rows"].append(row)
            checkpoint()
        report["timing_summary"] = {mode:{metric:summary([r[metric] for r in report["rows"] if r["mode"]==mode])
            for metric in (("request_ms",) if mode=="native" else ("request_ms","ae_ms"))}
            for mode in ("native","refresh_all","reuse03","d2")}
        report["ae_control_label"] = "refresh_all: new boundary wrapper parity-checked against original native D10; d2: wrapper parity-checked against original native D2"
        report.update(status="complete",completed_unix=time.time(),warmup=args.warmup,samples=len(prepared))
    except Exception as error:
        report.update(status="failed",completed_unix=time.time(),
            failure_code="DEADLINE" if isinstance(error,TimeoutError) else "PARITY" if isinstance(error,ArithmeticError) else "TECHNICAL",
            error=traceback.format_exc())
        raise
    finally:
        atomic_json(output,report)
    print(json.dumps(report["timing_summary"],indent=2),flush=True)


if __name__ == "__main__":
    main()

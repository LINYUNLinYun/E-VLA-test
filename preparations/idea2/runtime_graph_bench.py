"""Dynamic-observation policy replay and same-old-KV numerical verification."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import time

import numpy as np
import torch

from .experiments import load_backend, native_raw
from .graph_bench import cache_digest
from .openpi_backend import error, cache_tensors
from .policy import PipelinePolicy
from .scheduling_bench import span


MODES = {
    "original": {},
    "threaded_prefill_first": dict(submission_order="prefill_first", host_workers=True),
    "graph_serial": dict(cuda_graphs=True, mode="serial_reuse", submission_order="ae_first"),
    "graph_ae_first": dict(cuda_graphs=True, submission_order="ae_first"),
    "graph_prefill_first": dict(cuda_graphs=True, submission_order="prefill_first"),
    "graph_ae_first_high": dict(cuda_graphs=True, submission_order="ae_first", ae_priority=-1),
    "graph_prefill_first_high": dict(cuda_graphs=True, submission_order="prefill_first", ae_priority=-1),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="shared/checkpoints/openpi/pi0_libero_pytorch_idea2")
    parser.add_argument("--bank", default="policies/realtime-vla-flash/results/idea1_v3_supplement/e1_expand_init7_10_20261020T0431Z/episodes/E1/supp_E1_D2_tau0.1_P2_Gconfirmed_disagreement_rejectD2/A/libero_spatial/task00/init07/real_predict_inputs_first25.pkl")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    (dest/"source").mkdir()
    hashes = {}
    for p in Path(__file__).parent.glob("*.py"):
        payload = p.read_bytes()
        (dest/"source"/p.name).write_bytes(payload)
        hashes[p.name] = hashlib.sha256(payload).hexdigest()
    backend = load_backend(args.checkpoint)
    records = pickle.loads(Path(args.bank).read_bytes())
    spec = importlib.util.spec_from_file_location("preparations.idea2._baseline_policy",
        "results/idea2/runtime_graph_20261021/baseline/policy.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    original_output = backend.output
    normalized = []
    def observed_output(prepared, raw):
        # Same state/action transfers and output transform as OpenPIBackend.
        state = prepared.state[0].detach().cpu().numpy()
        actions = raw[0].detach().cpu().numpy()
        normalized.append(actions.copy())
        return backend.policy._output_transform(dict(state=state, actions=actions))["actions"]
    backend.output = observed_output
    rows, cold, audits = [], {}, []
    for mode in args.modes:
        print("Mode", mode, flush=True)
        policy = (baseline.PipelinePolicy(backend, diagnostics=True) if mode == "original" else
                  PipelinePolicy(backend, diagnostics=True, **MODES[mode]))
        captured_addresses = None
        try:
            for i in range(max(1,args.warmup)):
                r = records[i%len(records)]
                response = policy.predict(r["obs"],r["instruction"],"libero_spatial",900+i)
                if i == 0:
                    cold[mode] = dict(response_ms=response["timing"]["response_host_ms"], graph_setup_ms=response.get("graph_setup_ms",0))
                normalized.clear()
            policy.reset()
            if args.profile:
                torch.cuda.profiler.start()
            try:
                for i in range(args.requests):
                    r = records[i%len(records)]
                    normalized.clear()
                    with span(f"runtime/{mode}/{i}"):
                        response = policy.predict(r["obs"],r["instruction"],"libero_spatial",1701+i)
                    rows.append(dict(mode=mode,response=response,normalized=normalized[-1]))
                    assert response["pool"]["live_snapshots"] <= 2
                    if getattr(policy,"graphs",None) is not None:
                        addresses = [tuple(x.data_ptr() for x in cache_tensors(s.prefix.cache)) for s in policy.graphs.slots]
                        assert not set(addresses[0]) & set(addresses[1])
                        if captured_addresses is None:
                            captured_addresses = addresses
                        assert captured_addresses == addresses
                        assert response["graph_setup_ms"] == 0
            finally:
                if args.profile:
                    torch.cuda.profiler.stop()
            # Inspect the last published/in-flight prefixes only after timing.
            for slot in policy.pool.slots:
                if slot.ready is not None and slot.value is not None:
                    slot.ready.synchronize()
                    audits.append(dict(mode=mode,source=slot.observation.observation_id,
                                       digest=cache_digest(slot.value)))
            del slot
        finally:
            policy.close()
            del policy
            gc.collect()
        # Both captured KV slots are gone before eager references are built.
        print("Verify",mode,flush=True)
        for row in (r for r in rows if r["mode"]==mode):
            response = row["response"]
            i,source = response["request"],response["prefix_observation_id"]
            assert 0 <= source <= i
            if i:
                assert source < i
            selected = backend.prepare(native_raw(records[source%len(records)]))
            current = backend.prepare(native_raw(records[i%len(records)]))
            prefix = backend.build_prefix(selected)
            before = cache_digest(prefix)
            raw = backend.generate(prefix,current,backend.noise(response["seed"]))
            row["normalized_error"] = error(raw[0].cpu(),row.pop("normalized"))
            row["decoded_error"] = error(original_output(current,raw),response.pop("actions"))
            assert row["normalized_error"]["max_abs"] <= 1e-5
            assert row["decoded_error"]["max_abs"] <= 1e-5
            assert cache_digest(prefix) == before
            del prefix,raw
        for audit in (a for a in audits if a["mode"]==mode):
            prefix = backend.build_prefix(backend.prepare(native_raw(records[audit["source"]%len(records)])))
            audit["matches_eager"] = cache_digest(prefix)==audit["digest"]
            assert audit["matches_eager"]
            del prefix
        # Persist completed arms, including when a later arm fails.
        (dest/"progress.json").write_text(json.dumps(dict(rows=rows,cold=cold,audits=audits),indent=2))
    summary = {}
    for mode in args.modes:
        selected = [r["response"] for r in rows if r["mode"]==mode and r["response"]["request"]>0]
        summary[mode] = {key:dict(zip(("p50","p95"),np.percentile([r["timing"][key] for r in selected],[50,95]).tolist()))
                         for key in ("action_ready_host_ms","response_host_ms","ae_gpu_ms")}
        summary[mode]["kv_ages"] = sorted({r["observation_request_age"] for r in selected})
        for name in ("ae","prefill"):
            jobs = [r["diagnostics"][name] for r in selected if r["diagnostics"][name] is not None]
            if jobs:
                summary[mode][name+"_cpu_submit_ms"] = float(np.median([j["cpu_submit_ms"] for j in jobs]))
    props = torch.cuda.get_device_properties(backend.device)
    result = dict(args=vars(args),identity=backend.identity,gpu_uuid=str(props.uuid),rows=rows,cold=cold,audits=audits,
                  summary=summary,source_sha256=hashes,bank_sha256=hashlib.sha256(Path(args.bank).read_bytes()).hexdigest(),
                  scope="Dynamic real observations; original backend reference follows each actual old KV source; no profiler timings mixed with unprofiled measurements")
    (dest/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()

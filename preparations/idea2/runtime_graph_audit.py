"""GPU lifecycle/mask/layout audit, separate from performance measurements."""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from .experiments import load_backend, native_raw
from .graph_bench import cache_digest
from .openpi_backend import error, cache_tensors
from .policy import PipelinePolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--bank",default="policies/realtime-vla-flash/results/idea1_v3_supplement/e1_expand_init7_10_20261020T0431Z/episodes/E1/supp_E1_D2_tau0.1_P2_Gconfirmed_disagreement_rejectD2/A/libero_spatial/task00/init07/real_predict_inputs_first25.pkl")
    args = parser.parse_args()
    dest = Path(args.output)
    dest.mkdir(parents=True,exist_ok=False)
    backend = load_backend("shared/checkpoints/openpi/pi0_libero_pytorch_idea2")
    records = pickle.loads(Path(args.bank).read_bytes())
    prepare,output = backend.prepare,backend.output
    variant = dict(missing=False,short=False)
    def prepare_variant(raw):
        value = prepare(raw)
        if variant["missing"]:
            value.image_masks[1].zero_()
            value.token_masks[:,1] = False
        if variant["short"]:
            value.tokens = value.tokens[:,:-1].contiguous()
            value.token_masks = value.token_masks[:,:-1].contiguous()
        return value
    backend.prepare = prepare_variant
    normalized = []
    def record_output(prepared,raw):
        normalized.append(raw[0].cpu().numpy().copy())
        return output(prepared,raw)
    backend.output = record_output
    policy = PipelinePolicy(backend,cuda_graphs=True,submission_order="ae_first",ae_priority=-1,diagnostics=True)
    rows,prefixes,inputs,checks = [],[],{},{}
    cases = []
    for mode,interval,count in (("async_reuse",1,6),("async_reuse",2,6),("async_reuse",4,8),
                                ("serial_reuse",2,4),("fresh_serial",1,3)):
        cases.append((mode,interval,count))
    def step(index,label,instruction=None):
        record = copy.deepcopy(records[index%len(records)])
        if instruction is not None:
            record["instruction"] = instruction
        before = None
        if policy.pool.active is not None:
            old_slot = policy.pool.active.slot
            before = (old_slot,cache_digest(policy.graphs.slots[old_slot].prefix))
        normalized.clear()
        response = policy.predict(record["obs"],record["instruction"],"libero_spatial",2100+len(rows))
        # Explicit drains belong to this correctness audit, not the timed bench.
        policy.background.synchronize()
        if before is not None:
            # A layout/task reset or fresh_serial may legitimately replace the
            # former active slot. Ordinary reuse must leave its old KV intact.
            if label=="reuse":
                assert cache_digest(policy.graphs.slots[before[0]].prefix)==before[1]
        assert policy.pool.live_snapshots<=2
        addresses = [{x.data_ptr() for x in cache_tensors(s.prefix.cache)} for s in policy.graphs.slots]
        assert not addresses[0]&addresses[1]
        key = (response["snapshot"]["epoch"],response["request"])
        inputs[key] = (record,dict(variant))
        rows.append(dict(label=label,key=key,response=response,normalized=normalized[-1]))
        for slot in policy.pool.slots:
            if slot.value is not None and slot.ready is not None:
                slot.ready.synchronize()
                prefixes.append(dict(key=(policy.pool.epoch,slot.observation.observation_id),digest=cache_digest(slot.value)))
        return response
    try:
        for mode,interval,count in cases:
            policy.configure(mode,interval)
            for i in range(count):
                variant.update(missing=bool(i%2),short=False)
                step(i,"fresh" if mode=="fresh_serial" else "reuse")
        policy.configure("async_reuse",1)
        variant.update(missing=False,short=False)
        step(0,"task")
        changed = "pick up the object and move it carefully to the other side"
        task = step(1,"task",changed)
        checks["task_change_reuses_capture_updates_tokens"] = task["request"]==0 and task["graph_setup_ms"]==0
        step(2,"reuse",changed)
        variant["short"] = True
        resized = step(3,"layout",changed)
        checks["layout_change_drains_and_recaptures"] = resized["request"]==0 and resized["graph_setup_ms"]>0
        step(4,"reuse",changed)
        # Inject a failure after a real graph replay has already been enqueued.
        saved = policy.graphs.build_prefix
        def fail_after_replay(*args):
            saved(*args)
            raise RuntimeError("injected failure after prefix replay")
        policy.graphs.build_prefix = fail_after_replay
        try:
            r = records[5%len(records)]
            policy.predict(r["obs"],changed,"libero_spatial",9999)
            raise AssertionError("Expected injected replay failure")
        except RuntimeError as exc:
            assert "injected failure" in str(exc)
            checks["failure_drains_and_releases"] = (policy.pool.live_snapshots==0 and
                all(s.leases==0 and not s.readers for s in policy.pool.slots))
        finally:
            policy.graphs.build_prefix = saved
            del saved,fail_after_replay
        recovered = step(5,"recovery",changed)
        checks["recovery_after_failure"] = recovered["request"]==0 and recovered["graph_setup_ms"]==0
        policy.reset()
        checks["reset_has_no_live_pool_slots"] = policy.pool.live_snapshots==0
    finally:
        policy.close()
        del policy
        gc.collect()
    backend.output = output
    # Captured KV is released before any original reference cache is created.
    print("Verifying dynamic masks, source KV, task/layout transitions",flush=True)
    for row in rows:
        r = row["response"]
        source_key = (r["snapshot"]["epoch"],r["prefix_observation_id"])
        source,source_variant = inputs[source_key]
        current,current_variant = inputs[tuple(row["key"])]
        variant.update(source_variant)
        prefix = backend.build_prefix(backend.prepare(native_raw(source)))
        variant.update(current_variant)
        prepared = backend.prepare(native_raw(current))
        raw = backend.generate(prefix,prepared,backend.noise(r["seed"]))
        row["normalized_error"] = error(raw[0].cpu(),row.pop("normalized"))
        row["decoded_error"] = error(output(prepared,raw),r.pop("actions"))
        assert row["normalized_error"]["max_abs"]<=1e-5
        assert row["decoded_error"]["max_abs"]<=1e-5
        if r["request"]>0 and r["mode"]!="fresh_serial":
            assert r["prefix_observation_id"]<r["request"]
        del prefix,raw
    for snapshot in prefixes:
        record,selected_variant = inputs[tuple(snapshot["key"])]
        variant.update(selected_variant)
        prefix = backend.build_prefix(backend.prepare(native_raw(record)))
        assert cache_digest(prefix)==snapshot["digest"]
        del prefix
    checks.update(numerical_parity=True,all_recorded_prefixes_match_eager=True,
                  old_kv_unchanged_during_reuse=True,two_disjoint_kv_slots=True)
    assert all(checks.values()),checks
    result=dict(checks=checks,rows=rows,prefix_checks=len(prefixes),
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")})
    (dest/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(checks,indent=2),flush=True)


if __name__=="__main__":
    main()

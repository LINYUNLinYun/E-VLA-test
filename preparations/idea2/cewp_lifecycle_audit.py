"""Untimed PI0.5 GPU lifecycle audit for full and selective two-slot graphs.

Every reference KV is built only after the captured slots have been released.
Stream drains here protect deliberate source-buffer mutation and inspection;
they are not part of the production schedule or performance measurements.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from .cewp_preparation import PreparedScheduleCore, PreparedModulationCore
from .cewp_segmented import SelectiveGraphs
from .experiments import load_backend, native_raw
from .graph_bench import cache_digest
from .openpi_backend import Prepared, cache_tensors
from .policy import PipelinePolicy
from .runtime_graph import RuntimeGraphs, input_tensors, layout


CONFIGS = {
    "d10_graph": (10, None, False),
    "d10_split_all": (10, "all", False),
    "d10_vision_only": (10, "vision_only", False),
    "d10_vlm_only": (10, "vlm_only", False),
    "d10_hoist": (10, None, True),
    "d10_vision_hoist": (10, "vision_only", True),
    "d1_vision_hoist": (1, "vision_only", True),
    "d5_vision_hoist": (5, "vision_only", True),
    "d10_vision_hoist_wait": (10, "vision_only", True),
    "d1_vision_wait": (1, "vision_only", False),
    "d5_vision_wait": (5, "vision_only", False),
    "d10_modulation": (10, None, True),
    "d10_vision_modulation_wait": (10, "vision_only", True),
}


def difference(reference, actual):
    delta = np.abs(np.asarray(reference, np.float64)-np.asarray(actual, np.float64))
    return dict(mae=float(delta.mean()), max_abs=float(delta.max()))


def input_digest(prepared):
    digest = hashlib.sha256()
    for tensor in input_tensors(prepared):
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit(backend, records, name, max_error, mae_error):
    steps, selective, hoist = CONFIGS[name]
    backend.num_steps = steps
    backend.await_previous_prefix = name.endswith("_wait")
    modulation = "modulation" in name
    backend.graph_engine_class = SelectiveGraphs if selective else RuntimeGraphs
    if selective:
        backend.selective_overlap = selective
    elif hasattr(backend, "selective_overlap"):
        del backend.selective_overlap
    if hoist:
        backend.graph_core_class = PreparedModulationCore if modulation else PreparedScheduleCore
    elif hasattr(backend, "graph_core_class"):
        del backend.graph_core_class
    prepare, output = backend.prepare, backend.output
    variant = dict(missing=False, short=False)
    inplace = dict(enabled=False, buffer=None, observations=[])
    order = "prefill_first" if selective == "vlm_only" else "ae_first"
    policy = PipelinePolicy(backend, cuda_graphs=True, submission_order=order,
                            ae_priority=-1, diagnostics=True)

    def prepare_variant(raw):
        value = prepare(raw)
        if variant["missing"]:
            value.image_masks[1].zero_()
            value.token_masks[:, 1] = False
        if variant["short"]:
            value.tokens = value.tokens[:, :-1].contiguous()
            value.token_masks = value.token_masks[:, :-1].contiguous()
        if inplace["enabled"]:
            # The caller has already drained before issuing this request; also
            # drain here immediately before mutating the retained input source.
            policy.background.synchronize()
            policy.ae_stream.synchronize()
            if inplace["buffer"] is None:
                inplace["buffer"] = Prepared(None, [x.clone() for x in value.images],
                    [x.clone() for x in value.image_masks], value.tokens.clone(),
                    value.token_masks.clone(), value.state.clone())
            else:
                for target, source in zip(input_tensors(inplace["buffer"]), input_tensors(value), strict=True):
                    if target.shape != source.shape or target.dtype != source.dtype:
                        raise AssertionError("In-place audit must retain source layout")
                    target.copy_(source)
            value = inplace["buffer"]
            inplace["observations"].append(dict(addresses=[x.data_ptr() for x in input_tensors(value)],
                                                 content_sha256=input_digest(value)))
        return value

    normalized = []
    def record_output(prepared, raw):
        normalized.append(raw[0].cpu().numpy().copy())
        return output(prepared, raw)
    backend.prepare, backend.output = prepare_variant, record_output
    rows, prefixes, inputs, checks = [], {}, {}, {}
    capture_ids = []
    parameter_identity = [(n, p.data_ptr(), p._version) for n, p in backend.model.named_parameters()]
    modulation_modules = [m.dense for m in backend.model.paligemma_with_expert.gemma_expert.model.modules()
                          if getattr(m, "cond_dim", None) is not None and getattr(m, "dense", None) is not None]
    forwards = [m.forward for m in modulation_modules]
    def modulation_digest():
        digest = hashlib.sha256()
        for module in modulation_modules:
            for parameter in module.parameters():
                digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()
    original_modulation_digest = modulation_digest() if modulation else None

    def drain():
        policy.background.synchronize()
        policy.ae_stream.synchronize()
        policy.pool.poll()

    def step(index, label, instruction=None):
        drain()
        record = copy.deepcopy(records[index % len(records)])
        if instruction is not None:
            record["instruction"] = instruction
        before = None
        if policy.pool.active is not None:
            old_slot = policy.pool.active.slot
            before = old_slot, cache_digest(policy.graphs.slots[old_slot].prefix)
        normalized.clear()
        response = policy.predict(record["obs"], record["instruction"], record.get("suite", "libero_spatial"), 2100+len(rows))
        drain()
        if before is not None and label in ("reuse", "inplace"):
            assert cache_digest(policy.graphs.slots[before[0]].prefix) == before[1], "Old leased KV changed"
        assert policy.pool.live_snapshots <= 2
        addresses = [{x.data_ptr() for x in cache_tensors(s.prefix.cache)} for s in policy.graphs.slots]
        assert not addresses[0] & addresses[1]
        pools = [g.pool() for s in policy.graphs.slots for g in
                 ([s.vision_graph] if hasattr(s, "vision_graph") else [])+[s.prefix_graph, s.ae_graph]]
        assert len(pools) == len(set(pools)), "Private graph pools alias"
        key = response["snapshot"]["epoch"], response["request"]
        inputs[key] = record, dict(variant)
        rows.append(dict(label=label, key=key, response=response, normalized=normalized[-1]))
        if response["graph_setup_ms"]:
            capture_ids.append(dict(epoch=key[0], label=label, setup_ms=response["graph_setup_ms"]))
        for slot in policy.pool.slots:
            if slot.value is not None and slot.ready is not None:
                slot.ready.synchronize()
                identity = policy.pool.epoch, slot.observation.observation_id
                digest = cache_digest(slot.value)
                if identity in prefixes:
                    assert prefixes[identity] == digest
                prefixes[identity] = digest
        return response

    try:
        for mode, interval, count in (("async_reuse", 1, 6), ("async_reuse", 2, 6), ("async_reuse", 4, 8),
                                      ("serial_reuse", 2, 4), ("fresh_serial", 1, 3)):
            policy.configure(mode, interval)
            for i in range(count):
                variant.update(missing=bool(i % 2), short=False)
                step(i, "fresh" if mode == "fresh_serial" else "reuse")
        policy.configure("async_reuse", 1)
        variant.update(missing=False, short=False)
        step(0, "task")
        changed = "pick up the object and move it carefully to the other side"
        task = step(1, "task", changed)
        checks["task_change_reuses_capture_updates_tokens"] = task["request"] == 0 and task["graph_setup_ms"] == 0
        step(2, "reuse", changed)
        variant["short"] = True
        resized = step(3, "layout", changed)
        checks["layout_change_drains_and_recaptures"] = resized["request"] == 0 and resized["graph_setup_ms"] > 0
        step(4, "reuse", changed)
        saved = policy.graphs.build_prefix
        def fail_after_replay(*args):
            saved(*args)
            raise RuntimeError("injected failure after prefix replay")
        policy.graphs.build_prefix = fail_after_replay
        try:
            record = records[5 % len(records)]
            policy.predict(record["obs"], changed, record.get("suite", "libero_spatial"), 9999)
            raise AssertionError("Expected injected replay failure")
        except RuntimeError as exc:
            assert "injected failure" in str(exc)
            checks["failure_drains_and_releases"] = policy.pool.live_snapshots == 0 and all(
                s.leases == 0 and not s.readers for s in policy.pool.slots)
        finally:
            policy.graphs.build_prefix = saved
            del saved, fail_after_replay
        recovered = step(5, "recovery", changed)
        checks["recovery_after_failure"] = recovered["request"] == 0 and recovered["graph_setup_ms"] == 0
        policy.reset()
        checks["reset_has_no_live_pool_slots"] = policy.pool.live_snapshots == 0
        if selective:
            checks["reset_clears_phase_events"] = not policy.graphs.vision_pending and not policy.graphs.ae_pending
        policy.configure("async_reuse", 1)
        inplace["enabled"] = True
        variant.update(missing=False, short=False)
        for i in range(6):
            variant["missing"] = bool(i % 2)
            # The first step legitimately replaces the short-layout capture.
            step(i, "inplace_start" if i == 0 else "inplace")
        values = inplace["observations"]
        checks["inplace_source_addresses_constant"] = len(values) == 6 and all(
            value["addresses"] == values[0]["addresses"] for value in values)
        checks["inplace_source_contents_change"] = all(a["content_sha256"] != b["content_sha256"]
            for a, b in zip(values, values[1:]))
        checks["inplace_updates_replay_without_recapture"] = all(
            row["response"]["graph_setup_ms"] == 0 for row in rows if row["label"] == "inplace")
    finally:
        policy.close()
        del policy
        inplace["enabled"], inplace["buffer"] = False, None
        gc.collect()
        backend.output = output

    print(name, "verify", len(rows), "requests and", len(prefixes), "prefix identities", flush=True)
    try:
        for row in rows:
            response = row["response"]
            source_key = response["snapshot"]["epoch"], response["prefix_observation_id"]
            source, source_variant = inputs[source_key]
            current, current_variant = inputs[tuple(row["key"])]
            variant.update(source_variant)
            prefix = backend.build_prefix(backend.prepare(native_raw(source)))
            variant.update(current_variant)
            prepared = backend.prepare(native_raw(current))
            raw = backend.generate(prefix, prepared, backend.noise(response["seed"]))
            row["normalized_error"] = difference(raw[0].cpu(), row.pop("normalized"))
            row["decoded_error"] = difference(output(prepared, raw), response.pop("actions"))
            for kind in ("normalized_error", "decoded_error"):
                assert row[kind]["max_abs"] <= max_error and row[kind]["mae"] <= mae_error, row[kind]
            if response["request"] > 0 and response["mode"] != "fresh_serial":
                assert response["prefix_observation_id"] < response["request"]
            del prefix, raw
        for key, digest in prefixes.items():
            record, selected_variant = inputs[key]
            variant.update(selected_variant)
            prefix = backend.build_prefix(backend.prepare(native_raw(record)))
            assert cache_digest(prefix) == digest, "Published full prefix differs from eager"
            del prefix
    finally:
        backend.prepare, backend.output = prepare, output
    checks.update(numerical_parity=True, all_recorded_prefixes_match_eager=True,
                  old_kv_unchanged_during_reuse=True, two_disjoint_kv_slots=True, independent_graph_pools=True)
    checks["all_parameter_addresses_and_versions_unchanged"] = parameter_identity == [
        (n, p.data_ptr(), p._version) for n, p in backend.model.named_parameters()]
    checks["adaptive_dense_forwards_restored"] = all(m.forward == saved for m, saved in zip(modulation_modules, forwards))
    if modulation:
        checks["adaptive_dense_weights_unchanged"] = modulation_digest() == original_modulation_digest
        checks["adaptive_dense_count_37"] = len(modulation_modules) == 37
    assert all(checks.values()), checks
    return dict(config=name, steps=steps, selective=selective, hoist=hoist, modulation=modulation,
                await_previous_prefix=backend.await_previous_prefix, checks=checks, rows=rows,
                prefix_checks=len(prefixes), captures=capture_ids, inplace_observations=inplace["observations"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bank", default="results/idea1_v3_1/input_bank.pkl")
    parser.add_argument("--checkpoint", default="shared/checkpoints/openpi/pi05_libero_pytorch_idea2")
    parser.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--max-error", type=float, default=1e-5)
    parser.add_argument("--mae-error", type=float, default=1e-6)
    args = parser.parse_args()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=False)
    (dest/"source").mkdir()
    sources = {}
    for path in Path(__file__).parent.glob("*.py"):
        content = path.read_bytes()
        (dest/"source"/path.name).write_bytes(content)
        sources[path.name] = hashlib.sha256(content).hexdigest()
    bank_bytes = Path(args.bank).read_bytes()
    bank = pickle.loads(bank_bytes)
    selected = [i for i, r in enumerate(bank) if r["instruction"] == bank[0]["instruction"]]
    records = [bank[i] for i in selected]
    result = dict(args=vars(args), source_sha256=sources, selected_bank_indices=selected,
                  bank_sha256=hashlib.sha256(bank_bytes).hexdigest(), results=[], passed=False,
                  tolerance=dict(max_abs=args.max_error, mae=args.mae_error),
                  scope="Untimed lifecycle and numerical audit; explicit drains permit source-buffer mutation, references allocated after both graph KV slots close")
    def save():
        (dest/"result.json").write_text(json.dumps(result, indent=2))
    save()
    backend = load_backend(args.checkpoint, model_config="pi05_libero")
    result["identity"] = backend.identity
    result["gpu_uuid"] = str(torch.cuda.get_device_properties(backend.device).uuid)
    try:
        for name in args.configs:
            print("Audit", name, flush=True)
            result["results"].append(audit(backend, records, name, args.max_error, args.mae_error))
            save()
        result["passed"] = True
    except BaseException as exc:
        result["failure"] = repr(exc)
        raise
    finally:
        save()
    print(json.dumps(dict(passed=True, configs=len(result["results"]),
                         requests=sum(len(r["rows"]) for r in result["results"])), indent=2), flush=True)


if __name__ == "__main__":
    main()

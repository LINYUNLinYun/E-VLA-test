"""Locate cross-batch numerical drift with original math and at most two KV objects."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import pickle


def run(base, records, batch_size):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from preparations.idea2.batch_backend import BatchOpenPIBackend
    from preparations.idea2.batch_bench import bank_schedule, compare_lanes
    from preparations.idea2.experiments import native_raw
    from preparations.idea2.layer_bench import difference
    from preparations.idea2.openpi_backend import Prefix, cache_tensors
    from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding

    def metric(left, right):
        return difference(left.detach().float().cpu().numpy(), right.detach().float().cpu().numpy())

    schedule = bank_schedule(records, batch_size)
    frame = schedule["frames"][0]
    raws = [native_raw(records[index]) for index in frame["bank_indices"]]
    batched = BatchOpenPIBackend(base.policy, num_steps=10, lane_ids=frame["lane_ids"])
    prepared = batched.prepare_batch(raws)
    noise = batched.noise_batch(frame["seeds"])
    result = dict(batch_size=batch_size, bank_indices=frame["bank_indices"], seeds=frame["seeds"], lanes=[])
    with torch.no_grad():
        embeddings, masks, attention = base.model.embed_prefix(
            prepared.images, prepared.image_masks, prepared.tokens, prepared.token_masks)
        prefix = base.build_prefix(prepared)
        actions = base.generate(prefix, prepared, noise)
        velocity = base.velocity(prefix, prepared, noise, torch.ones(batch_size, device=base.device))
        time = torch.ones(1, device=base.device, dtype=torch.float32)
        def condition(count):
            value = create_sinusoidal_pos_embedding(time.expand(count), base.model.action_in_proj.out_features,
                min_period=4e-3, max_period=4.0, device=base.device).to(time.dtype)
            return F.silu(base.model.time_mlp_out(F.silu(base.model.time_mlp_in(value))))
        cond_batch, cond_one = condition(batch_size), condition(1)
        result["time_condition"] = metric(cond_batch, cond_one.expand_as(cond_batch))
        expert = base.model.paligemma_with_expert.gemma_expert.model
        dense = next(module.dense for module in expert.modules()
                     if getattr(module, "cond_dim", None) is not None and getattr(module, "dense", None) is not None)
        mod_batch, mod_one = dense(cond_batch), dense(cond_one)
        result["first_modulation"] = metric(mod_batch, mod_one.expand_as(mod_batch))
        result["time_weight_dtype"] = str(base.model.time_mlp_in.weight.dtype)
        result["modulation_weight_dtype"] = str(dense.weight.dtype)
        normalized = actions.detach().cpu().numpy().copy()
        decoded = batched.output_batch(prepared, normalized)
        for lane, (raw, seed) in enumerate(zip(raws, frame["seeds"], strict=True)):
            single = base.prepare(raw)
            single_noise = base.noise(seed)
            single_embeddings, single_masks, single_att = base.model.embed_prefix(
                single.images, single.image_masks, single.tokens, single.token_masks)
            single_prefix = base.build_prefix(single)
            single_actions = base.generate(single_prefix, single, single_noise)
            class SliceCache:
                def __getattr__(self, name): return getattr(prefix.cache, name)
                def __len__(self): return len(prefix.cache)
                def __getitem__(self, index):
                    return tuple(value[lane:lane+1] for value in prefix.cache[index])
                def update(self, *args, **kwargs): raise RuntimeError("Diagnostic cache view is read-only")
            shared = Prefix(SliceCache(), prefix.pad_masks[lane:lane+1], prefix.position_ids[lane:lane+1])
            fixed_kv_velocity = base.velocity(shared, single, single_noise, torch.ones(1, device=base.device))
            tensors = lambda p: [*p.images, *p.image_masks, p.tokens, p.token_masks, p.state]
            inputs = [metric(value[lane:lane+1], other)
                      for value, other in zip(tensors(prepared), tensors(single), strict=True)]
            caches = [metric(value[lane:lane+1], other)
                      for value, other in zip(cache_tensors(prefix.cache), cache_tensors(single_prefix.cache), strict=True)]
            result["lanes"].append(dict(lane=lane, prepared_inputs=inputs,
                all_prepared_inputs_bitwise_equal=all(row["max_abs"] == 0 for row in inputs),
                noise=metric(noise[lane:lane+1], single_noise),
                encoder_and_prompt_embeddings=metric(embeddings[lane:lane+1], single_embeddings),
                prefix_masks=metric(prefix.pad_masks[lane:lane+1], single_prefix.pad_masks),
                prefix_positions=metric(prefix.position_ids[lane:lane+1], single_prefix.position_ids),
                layer_kv=caches,
                first_ae_velocity_with_identical_kv=metric(velocity[lane:lane+1], fixed_kv_velocity),
                normalized=metric(actions[lane:lane+1], single_actions),
                decoded=difference(decoded[lane], base.output(single, single_actions))))
            del shared, single_prefix, single, single_noise, single_embeddings, single_masks, single_att, single_actions, fixed_kv_velocity
        result.update(logical_full_kv_objects_max=2, snapshots="One B-lane prefix plus one single-lane prefix; sliced views allocate no KV")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from preparations.idea2.batch_resources import exclusive_gpu
    from preparations.idea2.batch_plan import source_hashes
    state = dict(status="starting", source_before=source_hashes(), results=[])
    def save(): (args.output / "result.json").write_text(json.dumps(state, indent=2) + "\n")
    save()
    try:
        with exclusive_gpu(args.gpu_uuid, args.output):
            from preparations.idea2.experiments import load_backend
            import torch
            records = pickle.loads(Path("results/idea1_v3_1/input_bank.pkl").read_bytes())
            base = load_backend("shared/checkpoints/openpi/pi05_libero_pytorch_idea2", model_config="pi05_libero", num_steps=10)
            state["model_identity"] = base.identity
            try:
                for size in (2, 4):
                    state["results"].append(run(base, records, size))
                    print("DIAGNOSED", size, flush=True)
                    save()
                state["status"] = "passed"
            finally:
                del base
                gc.collect()
                torch.cuda.empty_cache()
        state["source_after"] = source_hashes()
        state["source_unchanged"] = state["source_before"] == state["source_after"]
        if not state["source_unchanged"]: raise RuntimeError("Sources changed during diagnosis")
    except BaseException:
        import traceback
        state.update(status="failed", error=traceback.format_exc())
        raise
    finally: save()


if __name__ == "__main__": main()

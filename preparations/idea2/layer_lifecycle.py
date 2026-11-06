"""Real-GPU lifecycle gate outside performance timing; no additional KV copy."""
import torch

from .experiments import native_raw
from .graph_bench import cache_digest
from .layer_bench import (ae_completion_stream, audit_graph_schedules, difference,
                          input_schedule, record_reader_tail, within_tolerance)
from .runtime_graph import input_tensors
from .scheduler import Observation


@torch.no_grad()
def audit(backend, graphs, pool, records, oracle, *, max_error=1e-5, mae_error=1e-6):
    """Use the existing owner Pool and captured slots, never a second owner.

Reuse one external Prepared object's tensor addresses while changing all input
contents. Also cancel a submitted Encoder-only generation, drain/reset, then
reuse the graphs with new identities. A live consumer lease must reject reset.
    """
    task = pool.task
    schedule_audit = audit_graph_schedules(graphs, oracle)
    if schedule_audit["expected_nfe"] != backend.num_steps:
        raise RuntimeError("Lifecycle oracle does not match the requested number of AE steps")
    pool.drain_reset(pool.epoch + 1, task)
    source = backend.prepare(native_raw(records[0]))
    source_addresses = tuple(x.data_ptr() for x in input_tensors(source))
    rows = []
    observation = 0

    def change_inputs(index):
        replacement = backend.prepare(native_raw(records[index]))
        for target, value in zip(input_tensors(source), input_tensors(replacement), strict=True):
            target.copy_(value)
        if tuple(x.data_ptr() for x in input_tensors(source)) != source_addresses:
            raise RuntimeError("Lifecycle fixture changed its external input addresses")

    def request(index, reject_reset=False):
        nonlocal observation
        _, seed = input_schedule(index, len(records))
        noise = backend.noise(seed)
        ready = torch.cuda.Event()
        ready.record()
        pool.offer(Observation(observation, pool.epoch, task, source))
        observation += 1
        reservation = pool.reserve()
        if reservation is None:
            raise RuntimeError("Lifecycle fixture could not reserve a slot")
        identity = reservation[0]
        metadata = graphs.encode(pool, identity, source, dependency=ready)
        pool.track_preparation(identity, metadata)
        prefix, producer_done, layer_events, metadata = graphs.prefill(pool, identity)
        pool.submitted_layers(identity, prefix, producer_done, layer_events, metadata_ready=metadata)
        lease = pool.acquire_layers(identity)
        try:
            actions, reader_done, _ = graphs.generate(pool, lease, source, noise, input_ready=ready)
            rejected = False
            if reject_reset:
                old_epoch = pool.epoch
                try:
                    pool.drain_reset(old_epoch + 1, task)
                except RuntimeError as error:
                    if "lease" not in str(error).lower():
                        raise
                    rejected = True
                if not rejected or pool.epoch != old_epoch:
                    raise RuntimeError("Reset must reject a live consumer without changing epoch")
            reader_done.synchronize()
            producer_done.synchronize()
            slot = graphs.slots[identity.slot]
            normalized = actions[0].cpu().numpy()
            first = slot.first_actions[0].cpu().numpy()
            decoded = backend.output(source, actions)
            row = dict(frame=index, epoch=identity.epoch, slot=identity.slot,
                       generation=identity.generation, input_addresses_unchanged=True,
                       actual_nfe=slot.actual_nfe, schedule_times=slot.schedule_times,
                       normalized_error=difference(oracle[index]["normalized"], normalized),
                       first_actions_error=difference(oracle[index]["first_actions"], first),
                       decoded_error=difference(oracle[index]["decoded"], decoded),
                       kv_equal=cache_digest(prefix) == oracle[index]["kv_digest"],
                       live_lease_reset_tested=reject_reset, live_lease_reset_rejected=rejected)
            if not row["kv_equal"] or not all(within_tolerance(row[key], max_error, mae_error)
                    for key in ("normalized_error", "first_actions_error", "decoded_error")):
                raise RuntimeError(f"In-place input or lifecycle correctness failed: {row}")
            rows.append(row)
        finally:
            if not lease.released:
                tail = record_reader_tail(torch, graphs)
                pool.release(lease, tail)

    try:
        request(0, reject_reset=True)
        change_inputs(1)
        request(1)
        change_inputs(0)
        request(0)
        pool.drain_reset(pool.epoch + 1, task)

        # Submit a real Encoder, but do not enqueue VLM or any AE consumer.
        change_inputs(1)
        ready = torch.cuda.Event()
        ready.record()
        pool.offer(Observation(observation, pool.epoch, task, source))
        identity = pool.reserve()[0]
        metadata = graphs.encode(pool, identity, source, dependency=ready)
        pool.track_preparation(identity, metadata)
        incomplete_at_reset = not metadata.query()
        old_epoch = pool.epoch
        pool.drain_reset(old_epoch + 1, task)
        if not metadata.query() or any(slot.state != "FREE" for slot in pool.slots):
            raise RuntimeError("Encoder-only reset did not drain its owned work")
        rejected = False
        try:
            pool.acquire_layers(identity)
        except RuntimeError:
            rejected = True
        if not rejected:
            raise RuntimeError("Old-epoch partial KV receipt was accepted after reset")
        change_inputs(0)
        request(0)
        return dict(passed=True, rows=rows, external_input_addresses=source_addresses,
                    num_steps=backend.num_steps, schedule_audit=schedule_audit,
                    encoder_only_pending_at_reset=incomplete_at_reset,
                    encoder_only_reset_drained=True, stale_epoch_rejected=rejected,
                    additional_full_kv_allocations=0,
                    scope="Real GPU tests outside timing; existing owner Pool and two graph KV slots")
    finally:
        vision_stream = getattr(graphs, "vision_stream", graphs.prefix_stream)
        if vision_stream is not graphs.prefix_stream:
            vision_stream.synchronize()
        graphs.prefix_stream.synchronize()
        ae_completion_stream(graphs).synchronize()
        pool.drain_reset(pool.epoch + 1, task)

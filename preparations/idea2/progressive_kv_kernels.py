"""Graph-safe primitives for explicitly experimental progressive KV selection.

The producer updates a complete layer's K and V, then calls ``publish`` on
that SAME CUDA stream. Its device-scope release atomic publishes a contiguous
prefix of completed layers. The AE stream calls ``snapshot`` once per Euler
step: an acquire atomic freezes that step's frontier, with no CPU scalar read.
Every layer in the step uses that frozen value. ``select_layer`` masks memory
loads, so it never reads the unselected (possibly unfinished) new layer.

The caller owns stream ordering and lifetime: zero the frontier on the producer
stream, record a reset Event, and make AE wait for that Event before snapshot.
Do not reset/reuse either slot until all readers finish; do not rewrite a
published layer. Publication must be in layer order with no unfinished gaps.
Reuse ONE layer-sized scratch K/V pair, on the AE stream, only after the previous
layer's attention has consumed it. These functions allocate no tensor storage.
Warm every shape/dtype/kernel before CUDA Graph capture; replay is GPU-only.

This policy may mix old/new *layers*, unlike the complete-old-KV policy. It is
an independently authorized approximation, not a numerical-equivalence claim.
"""
import operator

import torch
import triton
import triton.language as tl


@triton.jit
def _publish(frontier, completed: tl.constexpr):
    tl.atomic_max(frontier, completed, sem="release", scope="gpu")


@triton.jit
def _snapshot(frontier, steps, step: tl.constexpr):
    value = tl.atomic_add(frontier, 0, sem="acquire", scope="gpu")
    tl.store(steps + step, value)


@triton.jit
def _snapshot_override(frontier, steps, override, step: tl.constexpr):
    value = tl.atomic_add(frontier, 0, sem="acquire", scope="gpu")
    fixed = tl.load(override, cache_modifier=".cg")
    tl.store(steps + step, tl.where(fixed >= 0, fixed, value))


@triton.jit
def _select(old_k, old_v, new_k, new_v, scratch_k, scratch_v, steps,
            step: tl.constexpr, layer: tl.constexpr, count: tl.constexpr,
            BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < count
    use_new = layer < tl.load(steps + step, cache_modifier=".cg")
    # Pointer arguments are unsigned views: even NaN payloads and signed zero
    # are copied bitwise, without any floating-point conversion/arithmetic.
    old_keys = tl.load(old_k + offsets, mask=valid & ~use_new, other=0, cache_modifier=".cg")
    old_values = tl.load(old_v + offsets, mask=valid & ~use_new, other=0, cache_modifier=".cg")
    new_keys = tl.load(new_k + offsets, mask=valid & use_new, other=0, cache_modifier=".cg")
    new_values = tl.load(new_v + offsets, mask=valid & use_new, other=0, cache_modifier=".cg")
    tl.store(scratch_k + offsets, tl.where(use_new, new_keys, old_keys), mask=valid)
    tl.store(scratch_v + offsets, tl.where(use_new, new_values, old_values), mask=valid)


def _integer(value, name, minimum=0):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer >= {minimum}") from exc
    if value < minimum or value > 2**31 - 1:
        raise ValueError(f"{name} must be an int32 integer >= {minimum}")
    return value


def _control(tensor, name, scalar=False):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA Tensor")
    if tensor.dtype != torch.int32 or not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous int32")
    if scalar:
        if tensor.numel() != 1:
            raise ValueError(f"{name} must contain one scalar")
    elif tensor.ndim != 1 or not tensor.numel():
        raise ValueError(f"{name} must be a nonempty 1D tensor")


def _step(steps, step):
    _control(steps, "steps")
    step = _integer(step, "step")
    if step >= steps.numel():
        raise ValueError("step exceeds the step-frontier buffer")
    return step


def _disjoint(tensors):
    ranges = [(x.data_ptr(), x.data_ptr() + x.numel() * x.element_size()) for x in tensors]
    for i, (begin, end) in enumerate(ranges):
        if any(begin < other_end and other_begin < end for other_begin, other_end in ranges[:i]):
            raise ValueError("KV and control buffers must have disjoint storage ranges")


@torch.no_grad()
def publish(frontier, completed_layers):
    """Release-publish completed contiguous layers after producer K/V writes."""
    _control(frontier, "frontier", scalar=True)
    completed_layers = _integer(completed_layers, "completed_layers", minimum=1)
    with torch.cuda.device(frontier.device):
        _publish[(1,)](frontier, completed_layers, num_warps=1)


@torch.no_grad()
def snapshot(frontier, steps, step, override=None):
    """Acquire and freeze a step; optional CUDA int32 override freezes >=0.

With override=-1 use the acquired frontier. Override>=0 is only safe if the
caller separately ensures every selected new layer is complete (0 selects
all old layers). The same captured kernel supports dynamic oracle modes.
"""
    _control(frontier, "frontier", scalar=True)
    step = _step(steps, step)
    if frontier.device != steps.device:
        raise ValueError("frontier and steps must use the same CUDA device")
    if override is not None:
        _control(override, "override", scalar=True)
        if override.device != frontier.device:
            raise ValueError("override must use the same CUDA device as frontier")
    _disjoint((frontier, steps) if override is None else (frontier, steps, override))
    with torch.cuda.device(frontier.device):
        if override is None:
            _snapshot[(1,)](frontier, steps, step, num_warps=1)
        else:
            _snapshot_override[(1,)](frontier, steps, override, step, num_warps=1)


@torch.no_grad()
def select_layer(old_pair, new_pair, scratch_pair, steps, step, layer):
    """Copy the chosen layer into caller-owned, single-layer K/V scratch.

All six KV tensors must share shape, dtype, device and contiguous layout, with
disjoint ranges. Supported element widths are 1/2/4/8 bytes; unsigned pointer
reinterpretation preserves all bits. Return the supplied scratch pair.
"""
    step, layer = _step(steps, step), _integer(layer, "layer")
    pairs = (old_pair, new_pair, scratch_pair)
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in pairs):
        raise ValueError("Each KV pair must contain exactly two tensors")
    tensors = [x for pair in pairs for x in pair]
    reference = tensors[0]
    if not isinstance(reference, torch.Tensor):
        raise ValueError("KV entries must be tensors")
    for value in tensors:
        if not isinstance(value, torch.Tensor) or value.device.type != "cuda":
            raise ValueError("KV entries must be CUDA tensors")
        if (value.device != steps.device or value.dtype != reference.dtype
                or value.shape != reference.shape or not value.is_contiguous()):
            raise ValueError("KV tensors must share device, dtype, shape and contiguous layout")
    count = reference.numel()
    raw_type = {1: tl.uint8, 2: tl.uint16, 4: tl.uint32, 8: tl.uint64}.get(reference.element_size())
    if not count or raw_type is None:
        raise ValueError("KV tensors must be nonempty with 1/2/4/8-byte elements")
    _disjoint((*tensors, steps))
    pointers = [triton.reinterpret(value, raw_type) for value in tensors]
    with torch.cuda.device(steps.device):
        _select[(triton.cdiv(count, 256),)](*pointers, steps, step, layer, count, BLOCK=256, num_warps=4)
    return scratch_pair


@torch.no_grad()
def gpu_self_test(device="cuda:0", layers=4, dtype=torch.bfloat16):
    """Explicit opt-in GPU smoke: boundaries, frozen steps, replay and reuse.

Only this diagnostic reads GPU values on CPU. It owns two full test KV sets
and ONE layer scratch pair, never a third full prefix. No model is loaded.
"""
    layers = _integer(layers, "layers", minimum=1)
    device = torch.device(device)
    with torch.cuda.device(device):
        producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
        frontier = torch.zeros((), dtype=torch.int32, device=device)
        steps = torch.empty(10, dtype=torch.int32, device=device)
        override = torch.full((), -1, dtype=torch.int32, device=device)
        shape = (1, 1, 17, 256)
        old = [(torch.full(shape, -i-1., device=device, dtype=dtype),
                torch.full(shape, -i-11., device=device, dtype=dtype)) for i in range(layers)]
        new = [(torch.full(shape, i+1., device=device, dtype=dtype),
                torch.full(shape, i+11., device=device, dtype=dtype)) for i in range(layers)]
        scratch = tuple(torch.empty_like(x) for x in old[0])
        initial = torch.cuda.Event()
        initial.record()
        producer.wait_event(initial)
        consumer.wait_event(initial)
        checks = 0
        addresses = [x.data_ptr() for pair in new for x in pair]
        for revision in range(2):
            # Drained consumer makes same-address producer updates safe.
            consumer.synchronize()
            with torch.cuda.stream(producer):
                if revision:
                    for pair in new:
                        for value in pair:
                            value.add_(32)
            for completed in range(layers + 1):
                with torch.cuda.stream(producer):
                    frontier.zero_()
                    if completed:
                        publish(frontier, completed)
                    reset = torch.cuda.Event()
                    reset.record()
                consumer.wait_event(reset)
                with torch.cuda.stream(consumer):
                    snapshot(frontier, steps, 0)
                    # Later publication must not alter the frozen step choice.
                    publish(frontier, layers)
                    for layer in range(layers):
                        select_layer(old[layer], new[layer], scratch, steps, 0, layer)
                        consumer.synchronize()
                        expected = new[layer] if layer < completed else old[layer]
                        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                                   for a, b in zip(scratch, expected, strict=True))
                        checks += 1
            assert addresses == [x.data_ptr() for pair in new for x in pair]
        # All callsites warmed before capture; graph replays same buffer addresses.
        with torch.cuda.stream(consumer):
            snapshot(frontier, steps, 0, override)
            select_layer(old[0], new[0], scratch, steps, 0, 0)
        consumer.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=consumer):
            snapshot(frontier, steps, 0, override)
            select_layer(old[0], new[0], scratch, steps, 0, 0)
        for completed, fixed in ((0, -1), (layers, 0), (layers, -1), (layers, layers)):
            with torch.cuda.stream(producer):
                frontier.zero_()
                if completed:
                    publish(frontier, completed)
                reset.record()
            consumer.wait_event(reset)
            with torch.cuda.stream(consumer):
                override.fill_(fixed)
                graph.replay()
            consumer.synchronize()
            selected = fixed if fixed >= 0 else completed
            expected = new[0] if selected else old[0]
            assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                       for a, b in zip(scratch, expected, strict=True))
            checks += 1
        return dict(passed=True, checks=checks, layers=layers, dtype=str(dtype),
                    same_address_updates=True, graph_replay=True, dynamic_override=True, scratch_layers=1)

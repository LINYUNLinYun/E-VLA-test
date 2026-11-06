"""Profile-only NVTX labels for the original, unsplit two-stream Graph path.

Run under nsys --cuda-graph-trace=node:nvtx-precapture so capture-time
Vision/VLM/denoising ranges project onto replayed GPU nodes. No graph is split,
no synchronization or CUDA work is added, and no production default is changed.
"""
from contextlib import contextmanager

import torch

from .runtime_graph import DynamicCore, RuntimeGraphs


@contextmanager
def labeled_method(owner, name, label):
    """Temporarily label original calls during single-Host cold capture only."""
    original = getattr(owner, name)
    had_override = name in owner.__dict__
    saved_override = owner.__dict__.get(name)

    def forward(*args, **kwargs):
        with torch.cuda.nvtx.range(label):
            return original(*args, **kwargs)

    setattr(owner, name, forward)
    try:
        yield
    finally:
        if had_override:
            setattr(owner, name, saved_override)
        else:
            delattr(owner, name)


class LabeledCore(DynamicCore):
    @torch.no_grad()
    def build_prefix(self, prepared):
        pair = self.model.paligemma_with_expert
        vision = pair.paligemma.model.vision_tower
        projector = pair.paligemma.model.multi_modal_projector
        with torch.cuda.nvtx.range("Prefill"):
            with labeled_method(pair, "embed_image", "Prefill/VisionEncoder+Projector"), \
                 labeled_method(vision, "forward", "Prefill/VisionEncoder"), \
                 labeled_method(projector, "forward", "Prefill/VisionProjector"), \
                 labeled_method(pair, "embed_language_tokens", "Prefill/TextEmbedding"), \
                 labeled_method(pair, "forward", "Prefill/VLMDecoder"):
                return super().build_prefix(prepared)

    @torch.no_grad()
    def generate(self, prefix, prepared, noise, **kwargs):
        self._profile_step = 0
        with torch.cuda.nvtx.range("AE/Denoising[complete_old_KV]"):
            return super().generate(prefix, prepared, noise, **kwargs)

    @torch.no_grad()
    def velocity(self, prefix, prepared, actions, timestep):
        step = self._profile_step
        self._profile_step += 1
        with torch.cuda.nvtx.range(f"AE/Denoise/step_{step:02d}"):
            return super().velocity(prefix, prepared, actions, timestep)


class LabeledGraphs(RuntimeGraphs):
    @torch.no_grad()
    def build_prefix(self, pool, identity, prepared):
        owner = pool._slot(identity)
        with torch.cuda.nvtx.range(
            f"Prefill/submit[new_KV,slot={identity.slot},obs={owner.observation.observation_id}]"
        ):
            return super().build_prefix(pool, identity, prepared)

    @torch.no_grad()
    def generate(self, pool, lease, prepared, noise):
        with torch.cuda.nvtx.range(
            f"AE/submit[old_KV,slot={lease.identity.slot},obs={lease.observation.observation_id}]"
        ):
            return super().generate(pool, lease, prepared, noise)


def main():
    import sys
    from . import cewp_bench

    forbidden = {"--compile", "--hoist-schedule", "--hoist-modulation", "--selective"}
    if any(arg.split("=", 1)[0] in forbidden for arg in sys.argv[1:]):
        raise ValueError("Label-only runner preserves the original unsplit full-overlap baseline")
    original_loader = cewp_bench.load_backend

    def load(*args, **kwargs):
        backend = original_loader(*args, **kwargs)
        backend.graph_core_class = LabeledCore
        backend.graph_engine_class = LabeledGraphs
        return backend

    cewp_bench.load_backend = load
    try:
        cewp_bench.main()
    finally:
        cewp_bench.load_backend = original_loader


if __name__ == "__main__":
    main()

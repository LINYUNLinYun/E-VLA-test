"""Opt-in compiler ablation, identical neural components in both dispatch modes.

Automatic compiler CUDA graphs are explicitly disabled. RuntimeGraphs owns its
four private pools and performs warm-up before capture. No policy/KV mutation
or sampling change is introduced by this switch.
"""
import torch


def install(backend):
    if hasattr(backend, "compiler_originals"):
        raise ValueError("Compiler already installed")
    backbone = backend.model.paligemma_with_expert
    modules = (backbone.paligemma.model.vision_tower,
               backbone.paligemma.language_model,
               backbone.gemma_expert.model)
    backend.compiler_originals = [(module, module.forward) for module in modules]
    for module, original in backend.compiler_originals:
        module.forward = torch.compile(original, dynamic=False, fullgraph=False,
            options={"triton.cudagraphs": False})
    return dict(scope="vision tower, VLM transformer, AE transformer", dynamic=False,
                fullgraph=False, automatic_cudagraphs=False, precision_changes=False)


def uninstall(backend):
    for module, original in backend.compiler_originals:
        module.forward = original
    del backend.compiler_originals

"""Mask-parameterized Triton VLM; residual capture follows the chosen mask."""
import torch
import triton
from preparations.idea1_v3.policy import TritonBackend
from preparations.idea1_v3_4_v12_compensation.backend import difference, patch_hidden


def encoder(k, weights, buffers, seq, skip=(), capture_layers=(), history=None,
            older=None, ratio=0., reference=False, traced=()):
    residuals, traces = {}, {}
    audit = dict(qkv_layers=[], kv_only_layers=[], hidden_update_layers=[], patched_layers=[],
                 captured_layers=[], grad_enabled=torch.is_grad_enabled(), selector_calls=0)
    k.layer_norm_matmul_n256_1152_2048_bias(buffers["vision_x"], weights["vision_final_norm_w"],
        weights["vision_final_norm_b"], weights["encoder_multi_modal_projector_w"],
        weights["encoder_multi_modal_projector_b"], buffers["encoder_x"], buffers["vision_x_norm"])
    for i in range(18):
        layer, x = i + 1, buffers["encoder_x"]
        before = x.clone() if layer in capture_layers else None
        if layer in traced:
            traces[layer] = dict(input=x.clone())
        if layer not in skip:
            k.rms_matmul_n_2048_2560_qkv_rope(x, weights["encoder_attn_qkv_w"][i], buffers["encoder_rope_weights"],
                buffers["encoder_Q"], buffers["encoder_K"][i, :seq], buffers["encoder_V"][i, :seq], buffers["encoder_x_norm"])
            audit["qkv_layers"].append(layer)
        else:
            k.rms_matmul_n_2048_2560_kv_rope(x, weights["encoder_attn_qkv_w"][i], buffers["encoder_rope_weights"],
                buffers["encoder_K"][i, :seq], buffers["encoder_V"][i, :seq], buffers["encoder_x_norm"])
            audit["kv_only_layers"].append(layer)
        if i < 17 and layer not in skip:
            attn = k.AttnSingleKey(buffers["encoder_Q"], buffers["encoder_K"][i, :seq], buffers["encoder_V"][i, :seq], 256 ** -.5)
            k.matmul_n_2048_2048_res(attn, weights["encoder_attn_o_w"][i], x)
            k.rms_matmul_n_2048_16384_gate(x, weights["encoder_ffn_gate_w"][i], weights["encoder_ffn_up_w"][i],
                buffers["encoder_hidden"], buffers["encoder_x_norm"])
            k.matmul_n_16384_2048_res(buffers["encoder_hidden"], weights["encoder_ffn_down_w"][i], x)
            audit["hidden_update_layers"].append(layer)
        if before is not None:
            residual = torch.empty_like(x, dtype=torch.float32)
            difference[(triton.cdiv(x.numel(), 1024),)](x, before, residual, x.numel(), 1024)
            residuals[layer] = residual
            audit["captured_layers"].append(layer)
        if layer in skip and history is not None:
            residual = history[layer]
            previous = residual if older is None else older[layer]
            if residual.shape != x.shape or previous.shape != x.shape:
                raise ValueError("Residual shape/prompt mismatch")
            if reference:
                estimate = residual if older is None else residual + ratio * (residual - previous)
                x.copy_((x.float() + estimate).to(x.dtype))
            else:
                patch_hidden[(triton.cdiv(x.numel(), 1024),)](x, residual, previous, ratio, x.numel(), older is not None, 1024, enable_fp_fusion=False)
            audit["patched_layers"].append(layer)
        if layer in traced:
            traces[layer]["output"] = x.clone()
    return residuals, traces, audit


class Backend(TritonBackend):
    @torch.no_grad()
    def prepare_residual(self, obs, instruction, skip=(), capture_layers=(), history=None,
                         older=None, ratio=0., reference=False, traced=()):
        from scripts.spec.triton import pi0_infer as k
        if any(i not in range(1, 18) for i in (*skip, *capture_layers)):
            raise ValueError("This protocol only skips/captures layers 1-17")
        if history is not None and set(history) != set(skip):
            raise ValueError("Residual identities differ from the fixed skip mask")
        if older is not None and set(older) != set(skip):
            raise ValueError("Older residual identities differ from fixed skip mask")
        original = k.transformer_encoder
        captured, traces, audit = {}, {}, {}
        def wrapped(weights, buffers, seq, **kwargs):
            res, tr, actual = encoder(k, weights, buffers, seq, skip, capture_layers, history, older, ratio, reference, traced)
            captured.update(res); traces.update(tr); audit.update(actual)
        k.transformer_encoder = wrapped
        try:
            context, timing = super().prepare(obs, instruction, skip)
        finally:
            k.transformer_encoder = original
        context.update(residuals=captured, execution_audit=audit)
        if traced:
            context["traces"] = traces
        return context, timing

    @torch.no_grad()
    def sample(self, *args, **kwargs):
        return super().sample(*args, **kwargs)

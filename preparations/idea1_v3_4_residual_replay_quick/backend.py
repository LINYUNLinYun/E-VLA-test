from preparations.idea1_v3.policy import TritonBackend
from preparations.idea1_v3_4_v12_compensation.backend import difference,patch_hidden
import torch
import triton

def encoder(k,w,b,seq,skip=(),history=None,reference=False,capture_layers=()):
    residuals={}
    k.layer_norm_matmul_n256_1152_2048_bias(b['vision_x'],w['vision_final_norm_w'],w['vision_final_norm_b'],
        w['encoder_multi_modal_projector_w'],w['encoder_multi_modal_projector_b'],b['encoder_x'],b['vision_x_norm'])
    for i in range(18):
        layer=i+1;x=b['encoder_x']
        before=x.clone() if layer in capture_layers else None
        if layer not in skip:
            k.rms_matmul_n_2048_2560_qkv_rope(x,w['encoder_attn_qkv_w'][i],b['encoder_rope_weights'],
                b['encoder_Q'],b['encoder_K'][i,:seq],b['encoder_V'][i,:seq],b['encoder_x_norm'])
        else:
            k.rms_matmul_n_2048_2560_kv_rope(x,w['encoder_attn_qkv_w'][i],b['encoder_rope_weights'],
                b['encoder_K'][i,:seq],b['encoder_V'][i,:seq],b['encoder_x_norm'])
        if i<17 and layer not in skip:
            attn=k.AttnSingleKey(b['encoder_Q'],b['encoder_K'][i,:seq],b['encoder_V'][i,:seq],256**-.5)
            k.matmul_n_2048_2048_res(attn,w['encoder_attn_o_w'][i],x)
            k.rms_matmul_n_2048_16384_gate(x,w['encoder_ffn_gate_w'][i],w['encoder_ffn_up_w'][i],b['encoder_hidden'],b['encoder_x_norm'])
            k.matmul_n_16384_2048_res(b['encoder_hidden'],w['encoder_ffn_down_w'][i],x)
        if before is not None:
            r=torch.empty_like(x,dtype=torch.float32)
            difference[(triton.cdiv(x.numel(),1024),)](x,before,r,x.numel(),1024)
            residuals[layer]=r
        if layer in skip and history is not None:
            r=history[layer]
            if r.shape!=x.shape:raise ValueError('Residual shape/prompt mismatch')
            if reference:
                x.copy_((x.float()+r).to(x.dtype))
            else:
                patch_hidden[(triton.cdiv(x.numel(),1024),)](x,r,r,0.,x.numel(),False,1024,enable_fp_fusion=False)
    return residuals

class Backend(TritonBackend):
    def prepare_replay(self,obs,instruction,skip=(),capture_layers=(),history=None,reference=False):
        from scripts.spec.triton import pi0_infer as k
        original=k.transformer_encoder;captured={}
        def wrapped(w,b,seq,**kwargs):
            captured.update(encoder(k,w,b,seq,skip=skip,history=history,reference=reference,capture_layers=capture_layers))
        k.transformer_encoder=wrapped
        try:c,t=super().prepare(obs,instruction,skip)
        finally:k.transformer_encoder=original
        if capture_layers:c['residuals']=captured
        return c,t

"""Run one view while preserving the released two-view projection arithmetic.

The released batch2 attention-output projection rounds its last128 features
as four BF16 partials. The generic batch1 branch instead uses a single FP32
accumulation. Reuse the original two-part kernels with seq_len256; never edit
their code or select the generic branch. GPU parity remains a required gate.
"""


def project(native,x,weight,bias,res,out,buf):
    if tuple(x.shape)!=(1,256,1152):raise ValueError('Two-part bridge expects one native vision view')
    native.matmul_512x1152x1152_twopart_bias_res[(128,)](
        old_ptr=res,inp_ptr=x,weight_ptr=weight,bias_ptr=bias,out_ptr=out,out2_ptr=buf,
        seq_len=256,features=1152,hidden=1152)
    native.combine_1536_1152_twopart[(128,)](inp_ptr=buf,out_ptr=out,seq_len=256,hidden=1152)


def single_view_vision(native,weights,buffers,num_views):
    if num_views!=1 or tuple(buffers['vision_x'].shape)!=(1,256,1152):raise ValueError('Single-view pipeline requires real batch1 tensors')
    native.conv2d_embed_n256_1152_res(buffers['observation_images_normalized'],weights['vision_patch_embedding_w'],
        weights['vision_patch_embedding_b'],weights['vision_position_embedding'],buffers['vision_x'])
    for i in range(27):
        native.layer_norm_QKV_matmul_n256_1152_3456_bias(buffers['vision_x'],weights['vision_pre_attn_norm_w'][i],
            weights['vision_pre_attn_norm_b'][i],weights['vision_attn_qkv_w'][i],weights['vision_attn_qkv_b'][i],
            buffers['vision_QKV'],buffers['vision_x_norm'])
        attn=native.AttnMultiKey(buffers['vision_QKV'])
        project(native,attn,weights['vision_attn_o_w'][i],weights['vision_attn_o_b'][i],
            buffers['vision_x'],buffers['vision_x'],buffers['vision_x_split_k_buf'])
        native.layer_norm_matmul_n256_1152_4304_bias_gelu(buffers['vision_x'],weights['vision_pre_ffn_norm_w'][i],
            weights['vision_pre_ffn_norm_b'][i],weights['vision_ffn_up_w'][i],weights['vision_ffn_up_b'][i],
            buffers['vision_hidden'],buffers['vision_x_norm'])
        native.matmul_n256_4304_1152_bias_res(buffers['vision_hidden'],weights['vision_ffn_down_w'][i],
            weights['vision_ffn_down_b'][i],buffers['vision_x'],buffers['vision_x'],buffers['vision_x_split_k_buf'])

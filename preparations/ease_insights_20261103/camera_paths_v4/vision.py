"""Selected native B1 vision blocks and a separately computed B2 restore oracle.

No torch/model/CUDA import or startup occurs here. Full native arithmetic and
selected-vs-oracle GPU parity remain required before any scientific rollout.
"""
from preparations.ease_insights_20261103.camera_paths_v3.arithmetic import project
from preparations.ease_insights_20261103.camera_paths_v3.vision_views import camera_buffers,SHAPES
from preparations.ease_insights_20261103.camera_paths_v3.vision_dispatch import checked_raw
from .common import FULL,ALLOCATIONS


def checked_layers(layers):
    layers=tuple(layers)
    if not layers or any(type(i) is not int or not 1<=i<=27 for i in layers) or layers!=tuple(sorted(set(layers))):
        raise ValueError('Need unique sorted original one-based vision layer indices')
    return layers


def allocation(value):
    if not isinstance(value,(tuple,list)) or len(value)!=2:raise ValueError('Need two camera layer sets')
    layers=tuple(checked_layers(v) for v in value)
    if layers not in ((FULL,FULL),*ALLOCATIONS.values()):raise ValueError('Unplanned fresh-camera layer allocation')
    return layers


def selected_view(native,weights,buffers,layers):
    kept=checked_layers(layers)
    if tuple(buffers['vision_x'].shape)!=(1,256,1152) or tuple(buffers['observation_images_normalized'].shape)!=(1,224,224,3):
        raise ValueError('Native selected-view path requires actual batch1 image and vision tensors')
    native.conv2d_embed_n256_1152_res(buffers['observation_images_normalized'],weights['vision_patch_embedding_w'],
        weights['vision_patch_embedding_b'],weights['vision_position_embedding'],buffers['vision_x'])
    visited=[]
    for layer in kept:
        i=layer-1
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
        visited.append(layer)
    if tuple(visited)!=kept:raise RuntimeError('Actual selected vision visits differ')
    return visited


def restore_oracle(native,weights,buffers,layers):
    """Compute every B2 block,then restore omitted-view residuals;reference only."""
    kept=allocation(layers)
    if tuple(buffers['vision_x'].shape)!=SHAPES['vision_x']:raise ValueError('Oracle needs real batch2 buffers')
    native.conv2d_embed_n256_1152_res(buffers['observation_images_normalized'],weights['vision_patch_embedding_w'],
        weights['vision_patch_embedding_b'],weights['vision_position_embedding'],buffers['vision_x'])
    visited=[]
    for i in range(27):
        before={v:buffers['vision_x'][v:v+1].clone() for v in (0,1) if i+1 not in kept[v]}
        native.layer_norm_QKV_matmul_n256_1152_3456_bias(buffers['vision_x'],weights['vision_pre_attn_norm_w'][i],
            weights['vision_pre_attn_norm_b'][i],weights['vision_attn_qkv_w'][i],weights['vision_attn_qkv_b'][i],
            buffers['vision_QKV'],buffers['vision_x_norm'])
        attn=native.AttnMultiKey(buffers['vision_QKV'])
        native.matmul_n256_1152_1152_bias_res(attn,weights['vision_attn_o_w'][i],weights['vision_attn_o_b'][i],
            buffers['vision_x'],buffers['vision_x'],buffers['vision_x_split_k_buf'])
        native.layer_norm_matmul_n256_1152_4304_bias_gelu(buffers['vision_x'],weights['vision_pre_ffn_norm_w'][i],
            weights['vision_pre_ffn_norm_b'][i],weights['vision_ffn_up_w'][i],weights['vision_ffn_up_b'][i],
            buffers['vision_hidden'],buffers['vision_x_norm'])
        native.matmul_n256_4304_1152_bias_res(buffers['vision_hidden'],weights['vision_ffn_down_w'][i],
            weights['vision_ffn_down_b'][i],buffers['vision_x'],buffers['vision_x'],buffers['vision_x_split_k_buf'])
        for view,tensor in before.items():buffers['vision_x'][view:view+1].copy_(tensor)
        visited.append(i+1)
    return visited


def run_vision(native_vision,native,weights,buffers,request):
    kind=request['kind'];reference=False;effective=None;layers=None
    if kind=='native':
        native_vision(weights,buffers,2);batches=[2];actual=[list(FULL),list(FULL)];effective=actual
    elif kind in ('allocated','identity_b1'):
        layers=allocation((FULL,FULL) if kind=='identity_b1' else request['layers'])
        if kind=='allocated' and layers==(FULL,FULL):raise ValueError('Dense controls use the original B2 native path')
        actual=[]
        for camera in (0,1):actual.append(selected_view(native,weights,camera_buffers(buffers,camera),layers[camera]))
        batches=[1,1];effective=[list(v) for v in layers]
    elif kind=='oracle':
        layers=allocation(request['layers']);visited=restore_oracle(native,weights,buffers,layers)
        batches=[2];actual=[list(visited),list(visited)];effective=[list(v) for v in layers];reference=True
    elif kind=='reference':
        raw=request['raw'];checked_raw(raw,buffers);buffers['vision_x'].copy_(raw)
        batches=[];actual=[[],[]];reference=True
    else:raise ValueError('Unknown fresh-vision allocation request')
    audit=dict(kind=kind,updated_views=[0,1] if not reference else [],cached_views=[],vision_pipeline_invocations=len(batches),
        actual_kernel_input_batches=batches,actual_vision_layers_by_camera=actual,
        effective_vision_layers_by_camera=effective,image_layer_evaluations=sum(len(v) for v in actual),
        fresh_images_consumed=not reference,vision_cache='none',numerical_reference_only=reference,
        single_view_projection='original_two_view_BF16_partial_order' if batches==[1,1] else None,
        camera_order=['agentview','wrist'],scope='Observed logical image-layer visits and tensor batches,not GPU kernel-call counts,FLOPs,latency or quality')
    return audit

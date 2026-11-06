"""Expose exactly one batch item to the unmodified native vision kernels.

The native vision_encoder's num_views argument is unused. Actual tensor
batch dimensions must become one; changing the argument alone saves no work.
"""
BATCH_KEYS=('observation_images_normalized','vision_x','vision_x_norm','vision_QKV','vision_hidden')
SHAPES=dict(observation_images_normalized=(2,224,224,3),vision_x=(2,256,1152),
    vision_x_norm=(2,256,1152),vision_QKV=(2,256,3456),vision_hidden=(2,256,4304))
SPLIT_KEY='vision_x_split_k_buf'
SCRATCH_PER_VIEW=256*1152*4


def camera_buffers(buffers,camera):
    if type(camera) is not int or camera not in (0,1):raise ValueError('Camera index must be0 or1')
    for key,shape in SHAPES.items():
        if tuple(buffers[key].shape)!=shape:raise ValueError('Native two-camera buffer shape changed: '+key)
    if tuple(buffers[SPLIT_KEY].shape)!=(2*SCRATCH_PER_VIEW,):raise ValueError('Native vision scratch shape changed')
    result=dict(buffers)
    for key in BATCH_KEYS:result[key]=buffers[key][camera:camera+1]
    # Split-K memory is scratch, not per-camera history. A single-image
    # invocation uses its first correctly sized contiguous workspace.
    result[SPLIT_KEY]=buffers[SPLIT_KEY][:SCRATCH_PER_VIEW]
    return result

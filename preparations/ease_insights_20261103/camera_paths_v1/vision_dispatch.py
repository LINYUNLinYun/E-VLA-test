"""Scoped native vision dispatch and explicit last-refresh camera cache."""
from .vision_views import camera_buffers,SHAPES


def checked_raw(raw,buffers):
    target=buffers['vision_x']
    if (raw is None or tuple(raw.shape)!=SHAPES['vision_x'] or raw.dtype!=target.dtype or raw.device!=target.device):
        raise ValueError('Camera cache/reference must match both native raw vision outputs')


def run_vision(native,weights,buffers,request):
    kind=request['kind'];source=request.get('source_step');age=request.get('age_steps',0)
    if type(age) is not int or age<0:raise ValueError('Camera cache age must be nonnegative steps')
    if kind in ('full','refresh','both'):
        native(weights,buffers,2);cache=buffers['vision_x'].clone() if kind=='refresh' else None
        fresh=[0,1];cached=[];batch=2;calls=1;source=None;age=0
    elif kind=='single':
        camera=request['camera'];selected=camera_buffers(buffers,camera);history=request.get('cache')
        checked_raw(history,buffers)
        if type(source) is not int or source<0:raise ValueError('Cached view requires its own Full source step')
        native(weights,selected,1)
        buffers['vision_x'][1-camera].copy_(history[1-camera])
        cache=None;fresh=[camera];cached=[1-camera];batch=int(selected['observation_images_normalized'].shape[0]);calls=1
    elif kind=='reference':
        raw=request['raw'];checked_raw(raw,buffers);buffers['vision_x'].copy_(raw)
        cache=None;fresh=[];cached=[];batch=0;calls=0;source=None;age=0
    else:raise ValueError('Unknown camera update request')
    audit=dict(kind=kind,updated_views=fresh,cached_views=cached,native_vision_invocations=calls,
        actual_kernel_input_batch=batch,image_layer_evaluations=27*batch,
        cache_source_step=source,cache_age_steps=age,numerical_reference_only=kind=='reference',
        camera_order=['agentview','wrist'],scope='Observed native input batch and feature cache; image-layer evaluations are not kernel-call counts or FLOPs')
    return cache,audit

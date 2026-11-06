"""Environment-only EGL enumeration; CUDA and EGL ordinals differ here."""
import ctypes
import os


def mapping():
    egl=ctypes.CDLL('libEGL.so.1'); egl.eglGetProcAddress.restype=ctypes.c_void_p; egl.eglGetProcAddress.argtypes=[ctypes.c_char_p]
    query=ctypes.CFUNCTYPE(ctypes.c_uint,ctypes.c_int,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int))(egl.eglGetProcAddress(b'eglQueryDevicesEXT'))
    attrib=ctypes.CFUNCTYPE(ctypes.c_uint,ctypes.c_void_p,ctypes.c_int,ctypes.POINTER(ctypes.c_ssize_t))(egl.eglGetProcAddress(b'eglQueryDeviceAttribEXT'))
    devices=(ctypes.c_void_p*16)(); count=ctypes.c_int()
    if not query(16,devices,ctypes.byref(count)): raise RuntimeError('EGL enumeration failed')
    result={}
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    physical=[int(v) for v in visible.split(',')] if visible else None
    for index in range(count.value):
        value=ctypes.c_ssize_t(-1)
        if attrib(devices[index],0x323A,ctypes.byref(value)):
            ordinal=value.value
            result[index]=physical[ordinal] if physical is not None and 0<=ordinal<len(physical) else ordinal
    return result


def worker_environment(env,physical):
    result=dict(env); devices=mapping(); selected=next(index for index,target in devices.items() if target==physical)
    result.update(CUDA_VISIBLE_DEVICES=','.join(map(str,dict.fromkeys((physical,selected)))),MUJOCO_EGL_DEVICE_ID=str(selected),IDEA1_RENDER_PHYSICAL_GPU=str(physical))
    return result

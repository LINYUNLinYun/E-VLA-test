"""Match CUDA UUID to EGL device via the driver, without creating a context."""
import ctypes
import json
import os
import uuid


def enumerate_devices():
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Enumerate in a subprocess without CUDA_VISIBLE_DEVICES")
    cuda = ctypes.CDLL("libcuda.so.1")
    if cuda.cuInit(0):
        raise RuntimeError("CUDA driver initialization failed")
    count = ctypes.c_int()
    if cuda.cuDeviceGetCount(ctypes.byref(count)):
        raise RuntimeError("CUDA device enumeration failed")
    identities = {}
    for ordinal in range(count.value):
        device, raw = ctypes.c_int(), (ctypes.c_ubyte * 16)()
        if cuda.cuDeviceGet(ctypes.byref(device), ordinal) or cuda.cuDeviceGetUuid(ctypes.byref(raw), device):
            raise RuntimeError("CUDA UUID enumeration failed")
        identities[ordinal] = "GPU-" + str(uuid.UUID(bytes=bytes(raw)))
    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.restype = ctypes.c_void_p
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))(egl.eglGetProcAddress(b"eglQueryDevicesEXT"))
    attrib = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t))(egl.eglGetProcAddress(b"eglQueryDeviceAttribEXT"))
    devices, size = (ctypes.c_void_p * 32)(), ctypes.c_int()
    if not query(32, devices, ctypes.byref(size)):
        raise RuntimeError("EGL enumeration failed")
    rows = []
    for index in range(size.value):
        ordinal = ctypes.c_ssize_t(-1)
        if attrib(devices[index], 0x323A, ctypes.byref(ordinal)) and ordinal.value in identities:
            rows.append(dict(egl_index=index, cuda_ordinal=ordinal.value, gpu_uuid=identities[ordinal.value]))
    return rows


if __name__ == "__main__":
    print(json.dumps(enumerate_devices()))

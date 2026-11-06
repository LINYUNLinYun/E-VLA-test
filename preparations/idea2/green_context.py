"""Optional two-stream CUDA Green Context adapter; no model or KV ownership.

The bridge is compiled against local CUDA headers, not a hand-written resource
ABI. Building/importing this module does not initialize CUDA. Creating a pair
does, and must only be done on a coordinated device. GPU compatibility remains
an experimental capability until the accompanying probe passes on that stack.

Usage::

    library = build_library(artifact_directory)  # CPU compiler only
    with GreenContextPair(library, first_sm_min=32, device='cuda:0') as pair:
        ae_stream, vlm_stream = pair.streams
        # Normal torch Stream/Event operations; separate Graph pools per branch.
        # Destroy graphs and stop all stream users BEFORE leaving this scope.

The two resources come from ONE split: a rounded first group and its remainder.
Do not create independent splits for each branch: those may overlap physically.
Requested SM count is a minimum; query actual counts. Priority is not a quota.
The local CUDA 12.4 API gives the remainder weaker symmetry/performance guarantees.
Keep allocation/weights/tensors in the existing primary context. The temporary
converted context is current only while the bridge creates its stream, and is
never destroyed via cuCtxDestroy. One Host thread owns the pair and its lifetime.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import threading


BRIDGE_SOURCE = r'''
#include <cuda.h>
#include <cstdint>
#include <new>
#include <string>

struct Pair {
    CUgreenCtx green[2]{};
    CUcontext converted[2]{};
    CUstream streams[2]{};
    unsigned int sm[2]{};
    unsigned int total = 0;
};
static thread_local std::string error_text;
static int error(CUresult code, const char* site) {
    const char *name=nullptr, *description=nullptr;
    cuGetErrorName(code, &name);
    cuGetErrorString(code, &description);
    error_text=std::string(site)+": "+(name?name:"unknown")+" ("+
        std::to_string(static_cast<int>(code))+"): "+(description?description:"");
    return static_cast<int>(code);
}
static void cleanup_failed_create(Pair* pair) {
    if (!pair) return;
    for (int i=1;i>=0;--i) {
        if (pair->streams[i]) cuStreamDestroy(pair->streams[i]);
        if (pair->green[i]) cuGreenCtxDestroy(pair->green[i]);
    }
    delete pair;
}
extern "C" const char* idea2_green_error() { return error_text.c_str(); }
extern "C" int idea2_green_header_version() { return CUDA_VERSION; }
extern "C" unsigned int idea2_green_resource_size() { return sizeof(CUdevResource); }
extern "C" int idea2_green_create(int device_ordinal, unsigned int first_min,
                                 int priority_a, int priority_b, void** output) {
    *output=nullptr;
    error_text.clear();
    CUresult code;
    CUcontext original=nullptr;
    CUdevice device, current_device;
    CUdevResource whole{}, first{}, remaining{};
    unsigned int groups=1;
    Pair* pair=nullptr;
#define CHECK_CREATE(call) do { code=(call); if (code!=CUDA_SUCCESS) { \
    int result=error(code,#call); cleanup_failed_create(pair); return result; }} while(0)
    // The caller must initialize and select the existing PyTorch primary context.
    CHECK_CREATE(cuCtxGetCurrent(&original));
    if (!original) return error(CUDA_ERROR_INVALID_CONTEXT,"no current primary context");
    CHECK_CREATE(cuDeviceGet(&device,device_ordinal));
    CHECK_CREATE(cuCtxGetDevice(&current_device));
    if (device!=current_device) return error(CUDA_ERROR_INVALID_DEVICE,"current device differs");
    CHECK_CREATE(cuDeviceGetDevResource(device,&whole,CU_DEV_RESOURCE_TYPE_SM));
    if (first_min==0 || first_min>=whole.sm.smCount)
        return error(CUDA_ERROR_INVALID_VALUE,"first_min must leave a nonempty remainder");
    CHECK_CREATE(cuDevSmResourceSplitByCount(&first,&groups,&whole,&remaining,0,first_min));
    if (groups!=1 || first.type!=CU_DEV_RESOURCE_TYPE_SM ||
        remaining.type!=CU_DEV_RESOURCE_TYPE_SM || !remaining.sm.smCount ||
        first.sm.smCount+remaining.sm.smCount!=whole.sm.smCount)
        return error(CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION,"split must cover two nonempty disjoint resources");
    pair=new(std::nothrow) Pair;
    if (!pair) return error(CUDA_ERROR_OUT_OF_MEMORY,"host Pair allocation");
    pair->total=whole.sm.smCount;
    CUdevResource resources[2]={first,remaining};
    int priorities[2]={priority_a,priority_b};
    for (int i=0;i<2;++i) {
        CUdevResourceDesc descriptor=nullptr;
        CHECK_CREATE(cuDevResourceGenerateDesc(&descriptor,&resources[i],1));
        CHECK_CREATE(cuGreenCtxCreate(&pair->green[i],descriptor,device,CU_GREEN_CTX_DEFAULT_STREAM));
        CHECK_CREATE(cuCtxFromGreenCtx(&pair->converted[i],pair->green[i]));
        CHECK_CREATE(cuCtxPushCurrent(pair->converted[i]));
        CUresult stream_code=cuStreamCreateWithPriority(&pair->streams[i],CU_STREAM_NON_BLOCKING,priorities[i]);
        CUcontext popped=nullptr;
        CUresult pop_code=cuCtxPopCurrent(&popped);
        if (pop_code!=CUDA_SUCCESS) {
            int result=error(pop_code,"cuCtxPopCurrent");
            cuCtxSetCurrent(original); // Best-effort restore before error cleanup.
            cleanup_failed_create(pair); return result;
        }
        if (stream_code!=CUDA_SUCCESS) { int result=error(stream_code,"cuStreamCreateWithPriority"); cleanup_failed_create(pair); return result; }
        CUgreenCtx associated=nullptr;
        CHECK_CREATE(cuStreamGetGreenCtx(pair->streams[i],&associated));
        if (associated!=pair->green[i]) { cleanup_failed_create(pair); return error(CUDA_ERROR_INVALID_CONTEXT,"stream Green Context mismatch"); }
        CUdevResource actual{};
        CHECK_CREATE(cuGreenCtxGetDevResource(pair->green[i],&actual,CU_DEV_RESOURCE_TYPE_SM));
        pair->sm[i]=actual.sm.smCount;
        if (pair->sm[i]!=resources[i].sm.smCount) { cleanup_failed_create(pair); return error(CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION,"created SM count mismatch"); }
    }
    CUcontext restored=nullptr;
    CHECK_CREATE(cuCtxGetCurrent(&restored));
    if (restored!=original) { cleanup_failed_create(pair); return error(CUDA_ERROR_INVALID_CONTEXT,"primary context was not restored"); }
    *output=pair;
    return 0;
#undef CHECK_CREATE
}
extern "C" int idea2_green_info(void* opaque, unsigned int* total, unsigned int* counts,
                               uintptr_t* streams, uintptr_t* greens) {
    auto* pair=static_cast<Pair*>(opaque);
    if (!pair) return error(CUDA_ERROR_INVALID_VALUE,"null Pair");
    *total=pair->total;
    for(int i=0;i<2;++i) {
        counts[i]=pair->sm[i];
        streams[i]=reinterpret_cast<uintptr_t>(pair->streams[i]);
        greens[i]=reinterpret_cast<uintptr_t>(pair->green[i]);
    }
    return 0;
}
extern "C" int idea2_green_close(void* opaque) {
    auto* pair=static_cast<Pair*>(opaque);
    if (!pair) return 0;
    for(int i=0;i<2;++i) {
        if (!pair->streams[i]) continue;
        CUstreamCaptureStatus status;
        CUresult code=cuStreamIsCapturing(pair->streams[i],&status);
        if (code!=CUDA_SUCCESS) return error(code,"cuStreamIsCapturing during close");
        if (status!=CU_STREAM_CAPTURE_STATUS_NONE)
            return error(CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,"cannot close during capture");
        code=cuStreamSynchronize(pair->streams[i]);
        if (code!=CUDA_SUCCESS) return error(code,"cuStreamSynchronize during close");
    }
    // Streams must be destroyed before their Green Context; no cuCtxDestroy.
    for(int i=1;i>=0;--i) {
        if(pair->streams[i]) {
            CUresult code=cuStreamDestroy(pair->streams[i]);
            if(code!=CUDA_SUCCESS) return error(code,"cuStreamDestroy");
            pair->streams[i]=nullptr;
        }
        if(pair->green[i]) {
            CUresult code=cuGreenCtxDestroy(pair->green[i]);
            if(code!=CUDA_SUCCESS) return error(code,"cuGreenCtxDestroy");
            pair->green[i]=nullptr;
        }
    }
    delete pair;
    return 0;
}
'''


def build_library(directory, cuda_home='/usr/local/cuda') -> Path:
    """CPU-only g++ compilation; no cuInit, tensor, device probe or GPU work."""
    directory, cuda_home = Path(directory), Path(cuda_home)
    directory.mkdir(parents=True, exist_ok=True)
    header = cuda_home/'include/cuda.h'
    source_hash = hashlib.sha256(BRIDGE_SOURCE.encode()).hexdigest()
    header_hash = hashlib.sha256(header.read_bytes()).hexdigest()
    stem = 'green_bridge_'+source_hash[:12]+'_'+header_hash[:12]
    source, library = directory/(stem+'.cpp'), directory/(stem+'.so')
    command = ['g++', '-std=c++17', '-O2', '-shared', '-fPIC', '-Wall', '-Wextra',
               '-I'+str(cuda_home/'include'), str(source), '-lcuda', '-o', str(library)]
    source.write_text(BRIDGE_SOURCE)
    if not library.exists():
        result = subprocess.run(command, capture_output=True, text=True)
        (directory/(stem+'.build.log')).write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f'CPU Green Context bridge build failed: {result.stderr}')
    (directory/(stem+'.json')).write_text(json.dumps(dict(
        command=command, bridge_sha256=source_hash, header=str(header.resolve()),
        header_sha256=header_hash, library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        initialization='No CUDA driver/device initialization during compilation.'), indent=2))
    return library


class GreenContextPair:
    """Owner of two disjoint resource streams in one PyTorch primary context.

    Explicit close is required. Destroy associated Graphs before close and never
    reuse an ExternalStream after close. All API/lifetime management stays on the
    creating Host thread; this class does not change Pool/Lease or KV behavior.
    """
    def __init__(self, library, first_sm_min, *, device='cuda:0', priorities=(0, 0)):
        import torch
        if not isinstance(first_sm_min, int) or first_sm_min <= 0:
            raise ValueError('first_sm_min must be a positive integer')
        if len(priorities) != 2:
            raise ValueError('exactly two priorities are required')
        self.owner = threading.get_ident()
        self._handle = ctypes.c_void_p()
        self._lib = ctypes.CDLL(str(Path(library).resolve()))
        self._lib.idea2_green_error.restype = ctypes.c_char_p
        self._lib.idea2_green_create.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_int, ctypes.c_int,
                                               ctypes.POINTER(ctypes.c_void_p)]
        self._lib.idea2_green_create.restype = ctypes.c_int
        self._lib.idea2_green_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint),
                                             ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_size_t),
                                             ctypes.POINTER(ctypes.c_size_t)]
        self._lib.idea2_green_info.restype = ctypes.c_int
        self._lib.idea2_green_close.argtypes = [ctypes.c_void_p]
        self._lib.idea2_green_close.restype = ctypes.c_int
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('Green Context requires a CUDA device')
        with torch.cuda.device(self.device):
            torch.cuda.init()
            device_index = torch.cuda.current_device()
            self.device = torch.device('cuda', device_index)
            # Force the existing primary context active without retaining a model.
            torch.empty(0, device=self.device)
            self._check(self._lib.idea2_green_create(device_index, first_sm_min, *priorities,
                                                     ctypes.byref(self._handle)))
            try:
                total, counts = ctypes.c_uint(), (ctypes.c_uint*2)()
                handles, greens = (ctypes.c_size_t*2)(), (ctypes.c_size_t*2)()
                self._check(self._lib.idea2_green_info(self._handle, ctypes.byref(total), counts, handles, greens))
                self.streams = tuple(torch.cuda.ExternalStream(int(ptr), device=self.device) for ptr in handles)
                self.info = dict(requested_first_min=first_sm_min, sm_counts=list(counts),
                                 device_sm_count=total.value, stream_handles=list(handles),
                                 green_handles=list(greens), priorities=list(priorities),
                                 local_header_version=self._lib.idea2_green_header_version(),
                                 resource_abi_size=self._lib.idea2_green_resource_size(),
                                 partition='one rounded group plus disjoint remainder; no priority-based quota')
            except BaseException:
                self.close()
                raise

    def _check(self, code):
        if code:
            raise RuntimeError(self._lib.idea2_green_error().decode())

    def close(self):
        if threading.get_ident() != self.owner:
            raise RuntimeError('Green Context pair must be closed by its creating Host thread')
        if self._handle.value:
            import torch
            with torch.cuda.device(self.device):
                self._check(self._lib.idea2_green_close(self._handle))
            self._handle = ctypes.c_void_p()
            self.streams = ()

    def __enter__(self):
        if not self._handle.value:
            raise RuntimeError('Green Context pair has already been closed')
        return self

    def __exit__(self, *_):
        self.close()

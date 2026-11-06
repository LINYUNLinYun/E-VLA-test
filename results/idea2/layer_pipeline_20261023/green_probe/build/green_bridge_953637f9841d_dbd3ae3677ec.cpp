
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

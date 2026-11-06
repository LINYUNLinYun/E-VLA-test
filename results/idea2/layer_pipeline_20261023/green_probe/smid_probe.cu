#include <cuda_runtime.h>
#include <cstdint>

// Diagnostic only: observe physical SM IDs; no priority-based inference.
__global__ void record_smid(int* counts, unsigned int capacity,
                            unsigned long long cycles) {
    unsigned int sm;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
    if(threadIdx.x==0) atomicAdd(counts+(sm<capacity-1?sm:capacity-1),1);
    unsigned long long start=clock64();
    while(clock64()-start<cycles) { asm volatile("" ::: "memory"); }
}
extern "C" int idea2_launch_smid(void* counts, unsigned int capacity,
                               unsigned long long cycles, unsigned int blocks,
                               void* stream) {
    record_smid<<<blocks,256,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<int*>(counts),capacity,cycles);
    return static_cast<int>(cudaGetLastError());
}

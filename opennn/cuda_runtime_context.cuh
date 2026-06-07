#ifndef CUDA_RUNTIME_CONTEXT_CUH
#define CUDA_RUNTIME_CONTEXT_CUH

#include "kernel.cuh"
#include "types.h"

namespace opennn::device
{

void check_last_error();
void* allocate(Device, Index);
void deallocate(Device, void*, Index);
void set_zero_async(void*, Index, cudaStream_t);
cudaStream_t get_compute_stream();

}

#endif // CUDA_RUNTIME_CONTEXT_CUH

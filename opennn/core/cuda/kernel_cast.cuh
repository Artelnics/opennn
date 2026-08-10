#ifndef KERNEL_CAST_CUH
#define KERNEL_CAST_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

void cast_fp32_to_bf16(const Index n, const float* src, __nv_bfloat16* dst,
                            cudaStream_t stream = nullptr);
void cast_bf16_to_fp32(const Index n, const __nv_bfloat16* src, float* dst);

#endif

#endif

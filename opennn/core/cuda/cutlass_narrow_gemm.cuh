#ifndef CUTLASS_NARROW_GEMM_CUH
#define CUTLASS_NARROW_GEMM_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

// A dense forward whose contraction is narrow, through a CUTLASS kernel
// instantiated for the shape rather than looked up in cuBLASLt's catalogue.
//
// cuBLASLt runs CUTLASS kernels already - the profiles name them - but from a
// fixed set built for sm_80, and for a contraction of 28 it can only promise
// two-element alignment on the input, so it dispatches an `align2` kernel.
// Instantiating directly, with the alignment and the threadblock tile chosen for
// this shape, measured 1.03x to 1.48x faster at 256 to 65,536 rows and
// bit-identical output. `l1_cutlass_probe.cu` is the measurement.
//
// Returns false when the shape is not one this covers, which is the caller's
// signal to keep cuBLASLt. Only bf16 in, bf16 out, row-major, a contraction of
// at most 32 that is a multiple of 4, and an output width that is a multiple of
// 8 - the alignments the instantiation was built with.
bool narrow_k_linear_forward_cutlass(Index rows, Index contraction, Index out_features,
                                     const void* input, const void* weights, const void* bias,
                                     void* output, bool relu, cudaStream_t stream);

#endif

#endif

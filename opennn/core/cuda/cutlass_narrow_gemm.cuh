// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

#ifndef CUTLASS_NARROW_GEMM_CUH
#define CUTLASS_NARROW_GEMM_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

bool narrow_k_linear_forward_cutlass(Index rows, Index contraction, Index out_features,
                                     const void* input, const void* weights, const void* bias,
                                     void* output, bool relu, cudaStream_t stream);

#endif

#endif

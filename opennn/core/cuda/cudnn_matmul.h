// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/opennn_types.h"

namespace opennn::cudnn_matmul
{

// WHY THIS EXISTS. cuBLASLt and cuDNN ship different matmul engine sets, and
// for one shape that matters -- the dense benchmark's 8192x1024x1024 bf16
// hidden layer with bias and ReLU -- cuDNN reaches a kernel cuBLASLt does not
// expose. An exhaustive sweep of all 13,460 cuBLASLt configurations for that
// shape contains nothing under 198 us that draws less than 265 W; cuDNN's
// engine 1 runs it in 189.6 us at 235 W. Sixty cuDNN engine configurations
// were built, executed and verified against a cuBLASLt reference, and
// thirteen of them beat both the time and the energy bound. The measurement
// and the exact graph construction are in cudnn_matmul_probe.cu.
//
// WHAT IS DELIBERATELY NOT HERE. cuDNN can also fuse a second contraction
// into the same graph, which cuBLASLt structurally cannot. That is a much
// larger change -- it removes a layer from the caller, not a kernel from a
// plan -- and it is not attempted here.

// The GEMM stated in cuBLASLt's own terms, exactly as run_lt_matmul_cached
// receives it. Nothing is translated at the call site: the one translation
// into cuDNN's row-major contraction lives inside create(), so there is a
// single place to read when an operand layout is in doubt.
struct Problem
{
    int m = 0;
    int n = 0;
    int k = 0;

    cublasOperation_t  transA   = CUBLAS_OP_N;
    cublasOperation_t  transB   = CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;

    cudaDataType_t dtype_a   = CUDA_R_32F;
    cudaDataType_t dtype_b   = CUDA_R_32F;
    cudaDataType_t out_dtype = CUDA_R_32F;

    // 0 means "derive from m, n and k", the same encoding LtMatmulPlanKey uses.
    int lda = 0;
    int ldb = 0;
    int ldd = 0;

    bool beta_is_zero = true;
};

// Opaque: device_backend.cpp holds a pointer and never a definition, so
// cudnn_frontend.h stays inside this translation unit.
class Plan;

// Returns nullptr whenever cuDNN declines -- which is every problem this file
// has no measured reason to serve, every problem cuDNN itself refuses, and
// every problem asked for while the compute stream is capturing a CUDA graph.
// Never throws; a caller that gets nullptr simply keeps cuBLASLt.
Plan* create(const Problem&) noexcept;

// Builds only the engine configuration at that cuDNN enumeration index -- the
// identity candidate_plan_index() reports -- as a plan with one candidate, so a
// winner recorded by an earlier process comes back without the whole set being
// built again. The index is stable for one cuDNN build on one card.
Plan* create(const Problem&, int64_t plan_index) noexcept;
void  destroy(Plan*) noexcept;

// Candidates are cuDNN engine configurations that built and fit the workspace
// cap. They are NOT ranked: cuDNN's own heuristic puts a 226 us engine first
// where the best is 189.6 us, so the caller must time them.
int    candidate_count(const Plan*) noexcept;
int64_t candidate_plan_index(const Plan*, int candidate) noexcept;
size_t candidate_workspace_bytes(const Plan*, int candidate) noexcept;

// cuDNN's own engine tag ("eng1_k2=3_k5=1"), which is the label that can be
// fed back to cuDNN to identify the kernel that was chosen.
string candidate_name(const Plan*, int candidate);

// a, b, bias and d are the cuBLASLt operands in cuBLASLt's order and meaning;
// the swap into cuDNN's row-major operand order happens inside. Runs on the
// active lane's cuDNN handle, which Backend binds to the lane's compute
// stream, so this lands on the same stream as the cuBLASLt call it replaces.
// Returns false rather than throwing, so a refusal at run time is a fallback
// and not an exception in the middle of a training step.
bool run(Plan*, int candidate,
         const void* a, const void* b, const void* bias, void* d,
         void* workspace) noexcept;

// OPENNN_CUDNN_MATMUL_VERBOSE: report what was offered and what was chosen.
bool verbose() noexcept;

}

#endif

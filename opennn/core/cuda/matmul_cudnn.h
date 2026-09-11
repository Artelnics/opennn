// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2005-2026 Artificial Intelligence, SL.

#pragma once

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/opennn_types.h"

namespace opennn::matmul::cudnn
{

// Optional cuDNN provider for GEMMs that its engine set runs faster than
// cuBLASLt. matmul_backend.cpp owns timing, verification, caching and final
// selection; this provider only builds and executes cuDNN graphs.
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

    // Zero derives the leading dimension from m, n and k.
    int lda = 0;
    int ldb = 0;
    int ldd = 0;

    bool beta_is_zero = true;
};

// Opaque: matmul_backend.cpp holds a pointer and never a definition, so
// cudnn_frontend.h stays inside this translation unit.
class Plan;

// A declined problem returns nullptr and remains on cuBLASLt.
// Rebuild one cached engine without enumerating the complete candidate set.
Plan* create(const Problem&, int64_t plan_index = -1) noexcept;
void  destroy(Plan*) noexcept;

// Candidates are valid engine configurations within the workspace cap.
struct CandidateInfo
{
    int64_t plan_index = -1;
    size_t workspace_bytes = 0;
    string_view name;
};

int candidate_count(const Plan*) noexcept;
CandidateInfo candidate(const Plan*, int) noexcept;

// Uses cuBLASLt operand order and returns false when the caller should fall back.
bool run(Plan*, int candidate,
         const void* a, const void* b, const void* bias, void* d,
         void* workspace) noexcept;

// OPENNN_CUDNN_MATMUL_VERBOSE: report what was offered and what was chosen.
bool verbose() noexcept;

}

#endif

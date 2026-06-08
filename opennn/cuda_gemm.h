//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D A   G E M M   M O D U L E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

#include "tensor_utilities.h"

namespace opennn
{

struct LtMatmulPlan
{
    cublasLtMatmulDesc_t   op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc  = nullptr;
    cublasLtMatrixLayout_t b_desc  = nullptr;
    cublasLtMatrixLayout_t cd_desc = nullptr;
    cublasLtMatmulAlgo_t   algo{};
    bool                   algo_valid = false;
    size_t                 workspace_size = 0;

    LtMatmulPlan() = default;
    LtMatmulPlan(const LtMatmulPlan&) = delete;
    LtMatmulPlan& operator=(const LtMatmulPlan&) = delete;
    LtMatmulPlan(LtMatmulPlan&& other) noexcept { *this = std::move(other); }
    LtMatmulPlan& operator=(LtMatmulPlan&& other) noexcept;
    ~LtMatmulPlan();
};

constexpr size_t cublas_lt_workspace_search_bytes() { return 32ull * 1024 * 1024; }

namespace scratch
{

void* ensure_cublas_lt_workspace(size_t min_bytes = 0);

bfloat16* ensure_bf16_input_scratch(Index n_elements);

bfloat16* ensure_bf16_gradient_scratch(Index n_elements);

float* ensure_fp32_upcast_scratch(Index n_elements);

void* ensure_cudnn_conv_workspace(size_t min_bytes);

}

const void* data_for_gemm_dtype(const TensorView& input, Type target_type);

const LtMatmulPlan& get_lt_gemm_plan(
    int m, int n, int k,
    cublasOperation_t transA,
    cublasOperation_t transB,
    cublasLtEpilogue_t epilogue,
    cudaDataType_t io_dtype  = CUDA_R_32F,
    cudaDataType_t out_dtype = CUDA_R_32F);

void run_lt_matmul(const LtMatmulPlan& plan,
                   const void* a_data, const void* b_data, void* c_data,
                   const void* bias_pointer);

cublasComputeType_t gemm_compute_type(cudaDataType_t a_type, cudaDataType_t b_type = CUDA_R_32F);

void gemm_cuda(cublasOperation_t transa, cublasOperation_t transb,
               int m, int n, int k,
               const void* A, cudaDataType_t Atype, int lda,
               const void* B, cudaDataType_t Btype, int ldb,
               void* C, cudaDataType_t Ctype, int ldc,
               float alpha = 1.0f, float beta = 0.0f);

void gemm_strided_batched_cuda(cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const void* A, int lda, long long stride_a,
                               const void* B, int ldb, long long stride_b,
                               void* C, int ldc, long long stride_c,
                               int batch_count,
                               cudaDataType_t io_dtype = CUDA_R_32F,
                               float alpha = 1.0f, float beta = 0.0f);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

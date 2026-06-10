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

bfloat16* ensure_bf16_gradient_workspace(Index n_elements);

float* ensure_bf16_to_fp32_workspace(Index n_elements);

void* ensure_cudnn_conv_workspace(size_t min_bytes);

const void* data_for_gemm_dtype(const TensorView& input, Type target_type);

void run_lt_matmul_cached(
    int m, int n, int k,
    cublasOperation_t transA,
    cublasOperation_t transB,
    cublasLtEpilogue_t epilogue,
    const void* a_data, const void* b_data, void* c_data,
    const void* bias_pointer,
    cudaDataType_t io_dtype  = CUDA_R_32F,
    cudaDataType_t out_dtype = CUDA_R_32F);

void gemm_cuda(cublasOperation_t transa, cublasOperation_t transb,
               int m, int n, int k,
               const void* A, cudaDataType_t Atype, int lda,
               const void* B, cudaDataType_t Btype, int ldb,
               void* C, cudaDataType_t Ctype, int ldc,
               float alpha = 1.0f, float beta = 0.0f);

void gemm_strided_batched_cuda(cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const void* A, cudaDataType_t Atype, int lda, long long stride_a,
                               const void* B, cudaDataType_t Btype, int ldb, long long stride_b,
                               void* C, cudaDataType_t Ctype, int ldc, long long stride_c,
                               int batch_count,
                               float alpha = 1.0f, float beta = 0.0f);

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

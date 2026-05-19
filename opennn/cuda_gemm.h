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

#ifdef OPENNN_HAS_CUDA

// C and D share one layout descriptor — they always have the same shape and
// dtype in this codebase, and cuBLASLt accepts the same layout for both.
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
    LtMatmulPlan(LtMatmulPlan&& other) noexcept { *this = move(other); }
    LtMatmulPlan& operator=(LtMatmulPlan&& other) noexcept
    {
        std::swap(op_desc, other.op_desc);
        std::swap(a_desc,  other.a_desc);
        std::swap(b_desc,  other.b_desc);
        std::swap(cd_desc, other.cd_desc);
        std::swap(algo,    other.algo);
        std::swap(algo_valid, other.algo_valid);
        std::swap(workspace_size, other.workspace_size);
        return *this;
    }
    ~LtMatmulPlan()
    {
        cublasLtMatrixLayoutDestroy(cd_desc);
        cublasLtMatrixLayoutDestroy(b_desc);
        cublasLtMatrixLayoutDestroy(a_desc);
        cublasLtMatmulDescDestroy(op_desc);
    }
};

struct LtMatmulPlanKey
{
    int m;
    int n;
    int k;
    int transA;
    int transB;
    int epilogue;   // cublasLtEpilogue_t cast to int (e.g. BIAS, RELU_BIAS, BGRADA)
    int io_dtype;   // cudaDataType_t for A and B (inputs)
    int out_dtype;  // cudaDataType_t for C and D (outputs)

    bool operator==(const LtMatmulPlanKey&) const noexcept = default;
};

struct LtMatmulPlanKeyHash
{
    size_t operator()(const LtMatmulPlanKey& key) const noexcept
    {
        return hash_combine(key.m, key.n, key.k,
                            key.transA, key.transB, key.epilogue,
                            key.io_dtype, key.out_dtype);
    }
};

// Upper bound passed to cuBLASLt's heuristic search — limits which algorithms
// are considered. The actual VRAM allocated only grows to the max workspace
// the chosen algorithms reported they need (see ensure_cublas_lt_workspace).
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

inline void run_lt_matmul(const LtMatmulPlan& plan,
                          const void* a_data, const void* b_data, void* c_data,
                          const void* bias_pointer)
{
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer, sizeof(bias_pointer)));

    CHECK_CUBLAS(cublasLtMatmul(Backend::get_cublas_lt_handle(),
                                plan.op_desc,
                                &one,
                                a_data, plan.a_desc,
                                b_data, plan.b_desc,
                                &zero,
                                c_data, plan.cd_desc,
                                c_data, plan.cd_desc,
                                plan.algo_valid ? &plan.algo : nullptr,
                                scratch::ensure_cublas_lt_workspace(plan.workspace_size), plan.workspace_size,
                                Backend::get_compute_stream()));
}

// CUBLAS_COMPUTE_DTYPE (= CUBLAS_COMPUTE_32F_FAST_TF32) is FP32-input only;
// BF16 inputs require plain CUBLAS_COMPUTE_32F.
inline cublasComputeType_t gemm_compute_type(cudaDataType_t a_type, cudaDataType_t b_type = CUDA_R_32F)
{
    return (a_type == CUDA_R_16BF || b_type == CUDA_R_16BF)
        ? CUBLAS_COMPUTE_32F
        : CUBLAS_COMPUTE_DTYPE;
}

inline void gemm_cuda(cublasOperation_t transa, cublasOperation_t transb,
                      int m, int n, int k,
                      const void* A, cudaDataType_t Atype, int lda,
                      const void* B, cudaDataType_t Btype, int ldb,
                      void* C, cudaDataType_t Ctype, int ldc,
                      float alpha = 1.0f, float beta = 0.0f)
{
    const cublasComputeType_t compute = gemm_compute_type(Atype, Btype);
    CHECK_CUBLAS(cublasGemmEx(Backend::get_cublas_handle(),
                              transa, transb,
                              m, n, k,
                              &alpha,
                              A, Atype, lda,
                              B, Btype, ldb,
                              &beta,
                              C, Ctype, ldc,
                              compute,
                              CUBLAS_GEMM_DEFAULT));
}

inline void gemm_strided_batched_cuda(cublasOperation_t transa, cublasOperation_t transb,
                                      int m, int n, int k,
                                      const void* A, int lda, long long stride_a,
                                      const void* B, int ldb, long long stride_b,
                                      void* C, int ldc, long long stride_c,
                                      int batch_count,
                                      cudaDataType_t io_dtype = CUDA_R_32F,
                                      float alpha = 1.0f, float beta = 0.0f)
{
    const cublasComputeType_t compute = gemm_compute_type(io_dtype);
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(Backend::get_cublas_handle(),
                                            transa, transb,
                                            m, n, k,
                                            &alpha,
                                            A, io_dtype, lda, stride_a,
                                            B, io_dtype, ldb, stride_b,
                                            &beta,
                                            C, io_dtype, ldc, stride_c,
                                            batch_count,
                                            compute,
                                            CUBLAS_GEMM_DEFAULT));
}

#endif // OPENNN_HAS_CUDA

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

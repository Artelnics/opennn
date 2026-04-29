//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D A   G E M M   M O D U L E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#pragma once

// All cuBLAS / cuBLASLt GEMM plumbing for OpenNN. Owns:
//   - The cuBLASLt plan cache and its lookup key.
//   - The shared cuBLASLt workspace and BF16 down-cast scratch.
//   - Inline wrappers around cublasGemmEx, cublasGemmStridedBatchedEx, and
//     cublasLtMatmul (with bias / bias+ReLU / bias-grad epilogues).
//
// `Device` owns the handles, streams, and cuDNN op descriptors — i.e. raw
// runtime context. Anything that's specifically about *the matmul layer*
// (plans, workspace, GEMM wrappers) lives here instead.
//
// Dependency direction: this header includes tensor_utilities.h to reach
// Device::get_cublas_lt_handle() etc. tensor_utilities.h does not (and
// should not) include this header.

#include "tensor_utilities.h"

namespace opennn
{

#ifdef OPENNN_WITH_CUDA

// Cached cuBLASLt plan: 1 op desc + 4 layout descs + a heuristic-selected algo.
// Built once per (m, n, k, transA, transB, epilogue, dtypes) shape; bias *pointer*
// is set per call because it varies per layer/iteration.
struct LtMatmulPlan
{
    cublasLtMatmulDesc_t   op_desc = nullptr;
    cublasLtMatrixLayout_t a_desc  = nullptr;
    cublasLtMatrixLayout_t b_desc  = nullptr;
    cublasLtMatrixLayout_t c_desc  = nullptr;
    cublasLtMatrixLayout_t d_desc  = nullptr;
    cublasLtMatmulAlgo_t   algo{};
    bool                   algo_valid = false;

    LtMatmulPlan() = default;
    LtMatmulPlan(const LtMatmulPlan&) = delete;
    LtMatmulPlan& operator=(const LtMatmulPlan&) = delete;
    LtMatmulPlan(LtMatmulPlan&& o) noexcept { *this = std::move(o); }
    LtMatmulPlan& operator=(LtMatmulPlan&& o) noexcept
    {
        std::swap(op_desc, o.op_desc);
        std::swap(a_desc,  o.a_desc);
        std::swap(b_desc,  o.b_desc);
        std::swap(c_desc,  o.c_desc);
        std::swap(d_desc,  o.d_desc);
        std::swap(algo,    o.algo);
        std::swap(algo_valid, o.algo_valid);
        return *this;
    }
    ~LtMatmulPlan()
    {
        cublasLtMatrixLayoutDestroy(d_desc);
        cublasLtMatrixLayoutDestroy(c_desc);
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

    bool operator==(const LtMatmulPlanKey& o) const noexcept
    {
        return m == o.m && n == o.n && k == o.k
            && transA == o.transA && transB == o.transB
            && epilogue == o.epilogue
            && io_dtype == o.io_dtype && out_dtype == o.out_dtype;
    }
};

struct LtMatmulPlanKeyHash
{
    size_t operator()(const LtMatmulPlanKey& k) const noexcept
    {
        size_t h = std::hash<int>{}(k.m);
        const auto mix = [](size_t& acc, int v) {
            acc ^= std::hash<int>{}(v) + 0x9e3779b9 + (acc << 6) + (acc >> 2);
        };
        mix(h, k.n);
        mix(h, k.k);
        mix(h, k.transA);
        mix(h, k.transB);
        mix(h, k.epilogue);
        mix(h, k.io_dtype);
        mix(h, k.out_dtype);
        return h;
    }
};

// =====================================================================
// Workspace + plan accessors. All TU-local, process-lifetime.
// =====================================================================

// 32 MB matches NVIDIA's cuBLASLt sample default. Lazy-allocated on first use,
// shared across all cublasLtMatmul call sites.
constexpr size_t cublas_lt_workspace_bytes() { return 32ull * 1024 * 1024; }

void* get_cublas_lt_workspace();

// BF16 scratch buffer used when a layer (typically the first trainable Dense)
// receives an FP32 input but its weights live in the BF16 working copy.
// cuBLASLt requires A and B to share dtype, so callers down-cast the input
// to BF16 here and feed the scratch into the matmul. Lazy-grown to fit the
// largest request seen so far; callers must not retain the pointer across batches.
__nv_bfloat16* get_bf16_input_scratch(Index n_elements);

// Returns a cached cuBLASLt plan for a GEMM with the requested epilogue.
// Built once per unique (shape, epilogue, dtypes) and reused. The bias pointer
// is set on the returned op_desc by the caller per call — not part of the key.
//
// `io_dtype` is the dtype of the GEMM inputs (A and B); `out_dtype` is the
// dtype of the outputs (C and D). For pure FP32 they're both CUDA_R_32F;
// for fully-bf16 forward (bias_add path) they're both CUDA_R_16BF; for
// mixed-precision backward (BGRADA, weight gradient) inputs are bf16 but
// weight_grad output stays FP32. Defaults preserve legacy behaviour.
//
// Supported epilogues: CUBLASLT_EPILOGUE_BIAS, CUBLASLT_EPILOGUE_RELU_BIAS,
// CUBLASLT_EPILOGUE_BGRADA.
const LtMatmulPlan& get_lt_gemm_plan(
    int m, int n, int k,
    cublasOperation_t transA,
    cublasOperation_t transB,
    cublasLtEpilogue_t epilogue,
    cudaDataType_t io_dtype  = CUDA_R_32F,
    cudaDataType_t out_dtype = CUDA_R_32F);

// =====================================================================
// GEMM wrappers (inline, header-only).
// =====================================================================

inline void gemm_cuda(cublasOperation_t transa, cublasOperation_t transb,
                      int m, int n, int k,
                      const void* A, cudaDataType_t Atype, int lda,
                      const void* B, cudaDataType_t Btype, int ldb,
                      void* C, cudaDataType_t Ctype, int ldc,
                      float alpha = 1.0f, float beta = 0.0f)
{
    // CUBLAS_COMPUTE_32F_FAST_TF32 only triggers TF32 rounding for FP32 inputs;
    // for BF16 inputs cuBLAS rejects it (NOT_SUPPORTED) and we want
    // CUBLAS_COMPUTE_32F (FP32 accumulator over BF16 Tensor Cores).
    const cublasComputeType_t compute = (Atype == CUDA_R_16BF || Btype == CUDA_R_16BF)
                                            ? CUBLAS_COMPUTE_32F
                                            : CUBLAS_COMPUTE_DTYPE;
    CHECK_CUBLAS(cublasGemmEx(Device::get_cublas_handle(),
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

// Strided-batched GEMM (used by MHA's per-head batched matmul). Operands all
// share the same dtype, passed in by the caller (FP32 or BF16 depending on the
// network's activation_dtype).
inline void gemm_strided_batched_cuda(cublasOperation_t transa, cublasOperation_t transb,
                                      int m, int n, int k,
                                      const void* A, int lda, long long stride_a,
                                      const void* B, int ldb, long long stride_b,
                                      void* C, int ldc, long long stride_c,
                                      int batch_count,
                                      cudaDataType_t io_dtype = CUDA_R_32F,
                                      float alpha = 1.0f, float beta = 0.0f)
{
    const cublasComputeType_t compute = (io_dtype == CUDA_R_16BF)
                                            ? CUBLAS_COMPUTE_32F
                                            : CUBLAS_COMPUTE_DTYPE;
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(Device::get_cublas_handle(),
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

// Fused GEMM + (optionally activation +) bias via cuBLASLt epilogue: one launch
// for the whole gemm-bias[-relu] sequence, no intermediate write-then-read of
// the output tensor between stages.
//
// `epilogue` selects the post-matmul fusion:
//   - CUBLASLT_EPILOGUE_BIAS       : D = α(A·B) + bias                (default)
//   - CUBLASLT_EPILOGUE_RELU_BIAS  : D = max(0, α(A·B) + bias)
//
// Layout assumptions (encoded in the cached plan):
//   - All operands use the activation dtype declared on the cached LtMatmulPlan
//     (FP32 or BF16 — the caller chose when get_lt_gemm_plan was first invoked).
//   - Bias is FP32, length = m, broadcast along D's columns
//     (i.e. one bias element per output feature row).
//   - Tightly packed: lda = (transa==N ? m : k), ldb = (transb==N ? k : n), ldc = ldd = m.
//     If a caller needs different strides, it should not use this wrapper.
//
// Not thread-safe: the bias pointer is set on the cached op_desc per call, so
// concurrent invocations on the same shape would race. Matches the rest of this
// codebase's single-stream GPU usage assumption.
inline void gemm_bias_cuda(cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const void* A,
                           const void* B,
                           void* C,
                           const float* bias,
                           cudaDataType_t io_dtype = CUDA_R_32F,
                           cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS,
                           float alpha = 1.0f, float beta = 0.0f)
{
    const LtMatmulPlan& plan = get_lt_gemm_plan(m, n, k, transa, transb,
                                                epilogue, io_dtype, io_dtype);

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    void* workspace = get_cublas_lt_workspace();
    const size_t workspace_size = cublas_lt_workspace_bytes();

    CHECK_CUBLAS(cublasLtMatmul(Device::get_cublas_lt_handle(),
                                plan.op_desc,
                                &alpha,
                                A, plan.a_desc,
                                B, plan.b_desc,
                                &beta,
                                C, plan.c_desc,   // C (read when beta != 0)
                                C, plan.d_desc,   // D (write); aliasing C is supported
                                plan.algo_valid ? &plan.algo : nullptr,
                                workspace, workspace_size,
                                Device::get_compute_stream()));
}

// Fused matmul + bias-gradient via cuBLASLt's BGRADA epilogue. Computes the
// matmul C = α(A·B) + βC and, as a side product, writes the row-wise reduction
// of A (in cuBLAS column-major view) into `bias_grad`. For Dense backward,
// pass A = output_delta with transA = N to get bias_grad = sum_rows(dY) — i.e.
// the bias gradient — for free, replacing a separate sum() reduction kernel.
//
// `bias_grad` length must be m (the matmul's M dim, == D rows in column-major).
// Same threading caveat as gemm_bias_cuda.
inline void gemm_bgrad_cuda(cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const void* A,
                            const void* B,
                            void* C,
                            float* bias_grad,
                            cudaDataType_t io_dtype = CUDA_R_32F,
                            float alpha = 1.0f, float beta = 0.0f)
{
    // Inputs (output_delta, input) follow the activation dtype passed by the
    // caller; the weight gradient (D / C) is always FP32 so Adam can accumulate
    // without precision loss. cuBLASLt mixed-dtype matmul handles bf16 in × fp32
    // out natively.
    const LtMatmulPlan& plan = get_lt_gemm_plan(m, n, k, transa, transb,
                                                CUBLASLT_EPILOGUE_BGRADA,
                                                io_dtype,
                                                CUDA_R_32F);

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_grad, sizeof(bias_grad)));

    void* workspace = get_cublas_lt_workspace();
    const size_t workspace_size = cublas_lt_workspace_bytes();

    CHECK_CUBLAS(cublasLtMatmul(Device::get_cublas_lt_handle(),
                                plan.op_desc,
                                &alpha,
                                A, plan.a_desc,
                                B, plan.b_desc,
                                &beta,
                                C, plan.c_desc,
                                C, plan.d_desc,
                                plan.algo_valid ? &plan.algo : nullptr,
                                workspace, workspace_size,
                                Device::get_compute_stream()));
}

#endif // OPENNN_WITH_CUDA

}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

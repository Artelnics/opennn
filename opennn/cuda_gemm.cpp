//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D A   G E M M   M O D U L E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cuda_gemm.h"

#ifdef OPENNN_WITH_CUDA

namespace opennn
{

// ---------------------------------------------------------------------
// TU-local state (process-lifetime, never freed). Mirrors the pattern used
// for `pooling_scratch_` in kernel_layers.cu and `ones_*` in math_utilities.cpp.
// Visibility is intentionally TU-private; callers go through the public
// accessors below.
// ---------------------------------------------------------------------

namespace
{
    void*  cublas_lt_workspace_ = nullptr;
    void*  bf16_input_scratch_  = nullptr;
    size_t bf16_input_scratch_bytes_ = 0;

    std::unordered_map<LtMatmulPlanKey, LtMatmulPlan, LtMatmulPlanKeyHash> lt_gemm_plans_;
}

void* get_cublas_lt_workspace()
{
    if (!cublas_lt_workspace_)
        CHECK_CUDA(cudaMalloc(&cublas_lt_workspace_, cublas_lt_workspace_bytes()));
    return cublas_lt_workspace_;
}

__nv_bfloat16* get_bf16_input_scratch(Index n_elements)
{
    const size_t needed_bytes = size_t(n_elements) * sizeof(__nv_bfloat16);
    if (needed_bytes > bf16_input_scratch_bytes_)
    {
        if (bf16_input_scratch_) cudaFree(bf16_input_scratch_);
        CHECK_CUDA(cudaMalloc(&bf16_input_scratch_, needed_bytes));
        bf16_input_scratch_bytes_ = needed_bytes;
    }
    return reinterpret_cast<__nv_bfloat16*>(bf16_input_scratch_);
}

const LtMatmulPlan& get_lt_gemm_plan(
    int m, int n, int k,
    cublasOperation_t transA,
    cublasOperation_t transB,
    cublasLtEpilogue_t epilogue,
    cudaDataType_t io_dtype,
    cudaDataType_t out_dtype)
{
    const LtMatmulPlanKey key{m, n, k,
                              static_cast<int>(transA),
                              static_cast<int>(transB),
                              static_cast<int>(epilogue),
                              static_cast<int>(io_dtype),
                              static_cast<int>(out_dtype)};
    auto it = lt_gemm_plans_.find(key);
    if (it != lt_gemm_plans_.end()) return it->second;

    LtMatmulPlan plan;

    // Compute type: FP32 accumulator. For FP32 inputs we use the TF32 fast
    // path; for BF16 inputs we use plain CUBLAS_COMPUTE_32F because the
    // _FAST_TF32 mode is only meaningful when inputs are FP32 (it tells
    // cuBLAS to round FP32 inputs to TF32 before the TC multiply).
    const cublasComputeType_t compute_type = (io_dtype == CUDA_R_16BF)
                                                ? CUBLAS_COMPUTE_32F
                                                : CUBLAS_COMPUTE_32F_FAST_TF32;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&plan.op_desc, compute_type, CUDA_R_32F));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                                &transA, sizeof(transA)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                                &transB, sizeof(transB)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue)));
    // cuBLASLt 12.x requires bias dtype == output dtype for BF16/FP16 outputs.
    // FP32 bias on BF16 output returns 0 algos from the heuristic. So we mirror
    // out_dtype here; storage at the bias pointer must match.
    const cudaDataType_t bias_dtype = out_dtype;
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                                &bias_dtype, sizeof(bias_dtype)));

    // Layouts. Inputs are column-major in the cuBLAS view (the row-major caller
    // achieves row-major semantics by swapping operand roles outside this plan).
    const int a_rows = (transA == CUBLAS_OP_N) ? m : k;
    const int a_cols = (transA == CUBLAS_OP_N) ? k : m;
    const int b_rows = (transB == CUBLAS_OP_N) ? k : n;
    const int b_cols = (transB == CUBLAS_OP_N) ? n : k;

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.a_desc, io_dtype,  a_rows, a_cols, a_rows));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.b_desc, io_dtype,  b_rows, b_cols, b_rows));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.c_desc, out_dtype, m, n, m));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.d_desc, out_dtype, m, n, m));

    // Heuristic: pick one algo that fits within our workspace budget. If none
    // is returned, leave algo_valid=false and the call site will pass nullptr,
    // letting cuBLASLt use its internal default (slower path, but always works).
    cublasLtMatmulPreference_t pref = nullptr;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    const size_t max_workspace = cublas_lt_workspace_bytes();
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, sizeof(max_workspace)));

    cublasLtMatmulHeuristicResult_t heuristic = {};
    int returned_results = 0;
    cublasLtMatmulAlgoGetHeuristic(Device::get_cublas_lt_handle(),
                                   plan.op_desc,
                                   plan.a_desc, plan.b_desc, plan.c_desc, plan.d_desc,
                                   pref, 1, &heuristic, &returned_results);
    cublasLtMatmulPreferenceDestroy(pref);

    if (returned_results > 0)
    {
        plan.algo = heuristic.algo;
        plan.algo_valid = true;
    }

    auto [iter, _] = lt_gemm_plans_.emplace(key, std::move(plan));
    return iter->second;
}

}

#endif // OPENNN_WITH_CUDA

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

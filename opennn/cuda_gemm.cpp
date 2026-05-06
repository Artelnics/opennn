//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D A   G E M M   M O D U L E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cuda_gemm.h"

#ifdef OPENNN_HAS_CUDA

namespace opennn
{

namespace
{
    void* cublas_lt_workspace_ = nullptr;

    Buffer bf16_input_(Device::CUDA);
    Buffer bf16_gradient_(Device::CUDA);
    Buffer fp32_upcast_(Device::CUDA);

    unordered_map<LtMatmulPlanKey, LtMatmulPlan, LtMatmulPlanKeyHash> lt_gemm_plans_;
}

void* ensure_cublas_lt_workspace()
{
    if (!cublas_lt_workspace_)
        CHECK_CUDA(cudaMalloc(&cublas_lt_workspace_, cublas_lt_workspace_bytes()));
    return cublas_lt_workspace_;
}

__nv_bfloat16* ensure_bf16_input_scratch(Index n_elements)
{
    bf16_input_.grow_to(n_elements * Index(sizeof(__nv_bfloat16)));
    return bf16_input_.as<__nv_bfloat16>();
}

__nv_bfloat16* ensure_bf16_gradient_scratch(Index n_elements)
{
    bf16_gradient_.grow_to(n_elements * Index(sizeof(__nv_bfloat16)));
    return bf16_gradient_.as<__nv_bfloat16>();
}

float* ensure_fp32_upcast_scratch(Index n_elements)
{
    fp32_upcast_.grow_to(n_elements * Index(sizeof(float)));
    return fp32_upcast_.as<float>();
}

const void* maybe_cast(const TensorView& input, Type target_type)
{
    if (input.type == target_type) return input.data;

    if (input.type == Type::FP32 && target_type == Type::BF16)
    {
        __nv_bfloat16* scratch = ensure_bf16_input_scratch(input.size());
        cast_fp32_to_bf16_cuda(input.size(), input.as<float>(), scratch);
        return scratch;
    }

    if (input.type == Type::BF16 && target_type == Type::FP32)
    {
        float* scratch = ensure_fp32_upcast_scratch(input.size());
        cast_bf16_to_fp32_cuda(input.size(), input.as<__nv_bfloat16>(), scratch);
        return scratch;
    }

    throw runtime_error("maybe_cast: unsupported type pair");
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

    // _FAST_TF32 is FP32-input only; BF16 needs plain CUBLAS_COMPUTE_32F.
    const cublasComputeType_t compute_type = (io_dtype == CUDA_R_16BF)
                                                ? CUBLAS_COMPUTE_32F
                                                : CUBLAS_COMPUTE_32F_FAST_TF32;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&plan.op_desc, compute_type, CUDA_R_32F));

    auto set_desc = [&](cublasLtMatmulDescAttributes_t attr, const auto& value)
    {
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.op_desc, attr, &value, sizeof(value)));
    };

    set_desc(CUBLASLT_MATMUL_DESC_TRANSA,   transA);
    set_desc(CUBLASLT_MATMUL_DESC_TRANSB,   transB);
    set_desc(CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
    set_desc(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, out_dtype);

    const int a_rows = (transA == CUBLAS_OP_N) ? m : k;
    const int a_cols = (transA == CUBLAS_OP_N) ? k : m;
    const int b_rows = (transB == CUBLAS_OP_N) ? k : n;
    const int b_cols = (transB == CUBLAS_OP_N) ? n : k;

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.a_desc, io_dtype,  a_rows, a_cols, a_rows));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.b_desc, io_dtype,  b_rows, b_cols, b_rows));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.c_desc, out_dtype, m, n, m));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.d_desc, out_dtype, m, n, m));

    cublasLtMatmulPreference_t pref = nullptr;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    const size_t max_workspace = cublas_lt_workspace_bytes();
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, sizeof(max_workspace)));

    cublasLtMatmulHeuristicResult_t heuristic = {};
    int returned_results = 0;
    cublasLtMatmulAlgoGetHeuristic(Backend::get_cublas_lt_handle(),
                                   plan.op_desc,
                                   plan.a_desc, plan.b_desc, plan.c_desc, plan.d_desc,
                                   pref, 1, &heuristic, &returned_results);
    cublasLtMatmulPreferenceDestroy(pref);

    if (returned_results > 0)
    {
        plan.algo = heuristic.algo;
        plan.algo_valid = true;
    }

    auto [iter, _] = lt_gemm_plans_.emplace(key, move(plan));
    return iter->second;
}

}

#endif // OPENNN_HAS_CUDA

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

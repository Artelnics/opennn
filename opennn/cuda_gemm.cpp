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
    Buffer cublas_lt_workspace_(Device::CUDA);

    Buffer bf16_input_(Device::CUDA);
    Buffer bf16_gradient_(Device::CUDA);
    Buffer fp32_upcast_(Device::CUDA);

    Buffer cudnn_conv_workspace_(Device::CUDA);

    unordered_map<LtMatmulPlanKey, LtMatmulPlan, LtMatmulPlanKeyHash> lt_gemm_plans_;
}

namespace scratch
{

void* ensure_cublas_lt_workspace(size_t min_bytes)    { return cublas_lt_workspace_.ensure_bytes(min_bytes); }
bfloat16* ensure_bf16_input_scratch(Index n)     { return bf16_input_.ensure<bfloat16>(n); }
bfloat16* ensure_bf16_gradient_scratch(Index n)  { return bf16_gradient_.ensure<bfloat16>(n); }
float* ensure_fp32_upcast_scratch(Index n)            { return fp32_upcast_.ensure<float>(n); }
void* ensure_cudnn_conv_workspace(size_t min_bytes)   { return cudnn_conv_workspace_.ensure_bytes(min_bytes); }

}

const void* data_for_gemm_dtype(const TensorView& input, Type target_type)
{
    if (input.type == target_type) return input.data;

    if (input.type == Type::FP32 && target_type == Type::BF16)
    {
        bfloat16* dst = scratch::ensure_bf16_input_scratch(input.size());
        cast_fp32_to_bf16_cuda(input.size(), input.as<float>(), dst);
        return dst;
    }

    if (input.type == Type::BF16 && target_type == Type::FP32)
    {
        float* dst = scratch::ensure_fp32_upcast_scratch(input.size());
        cast_bf16_to_fp32_cuda(input.size(), input.as<bfloat16>(), dst);
        return dst;
    }

    throw runtime_error("data_for_gemm_dtype: unsupported type pair");
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
                              int(transA), int(transB), int(epilogue),
                              int(io_dtype), int(out_dtype)};
    auto it = lt_gemm_plans_.find(key);
    if (it != lt_gemm_plans_.end()) return it->second;

    LtMatmulPlan plan;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&plan.op_desc, gemm_compute_type(io_dtype), CUDA_R_32F));

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

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.a_desc,  io_dtype,  a_rows, a_cols, a_rows));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.b_desc,  io_dtype,  b_rows, b_cols, b_rows));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.cd_desc, out_dtype, m, n, m));

    cublasLtMatmulPreference_t pref = nullptr;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    const size_t search_bytes = cublas_lt_workspace_search_bytes();
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &search_bytes, sizeof(search_bytes)));

    cublasLtMatmulHeuristicResult_t heuristic = {};
    int returned_results = 0;
    cublasLtMatmulAlgoGetHeuristic(Backend::get_cublas_lt_handle(),
                                   plan.op_desc,
                                   plan.a_desc, plan.b_desc, plan.cd_desc, plan.cd_desc,
                                   pref, 1, &heuristic, &returned_results);
    cublasLtMatmulPreferenceDestroy(pref);

    if (returned_results > 0)
    {
        plan.algo = heuristic.algo;
        plan.algo_valid = true;
        plan.workspace_size = heuristic.workspaceSize;

        // Grow the global scratch buffer to fit this plan's chosen algorithm.
        scratch::ensure_cublas_lt_workspace(plan.workspace_size);
    }

    return lt_gemm_plans_.emplace(key, move(plan)).first->second;
}

}

#endif // OPENNN_HAS_CUDA

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

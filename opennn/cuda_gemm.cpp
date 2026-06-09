//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D A   G E M M   M O D U L E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cuda_gemm.h"
#include "device_backend.h"
#include "string_utilities.h"

#ifdef OPENNN_HAS_CUDA

namespace opennn
{

namespace
{
    struct LtMatmulPlan
    {
        cublasLtMatmulDesc_t   matmul_descriptor = nullptr;
        cublasLtMatrixLayout_t a_matrix_layout = nullptr;
        cublasLtMatrixLayout_t b_matrix_layout = nullptr;
        cublasLtMatrixLayout_t output_matrix_layout = nullptr;
        cublasLtMatmulAlgo_t   algorithm{};
        bool                   has_algorithm = false;
        size_t                 workspace_bytes = 0;

        LtMatmulPlan() = default;
        LtMatmulPlan(const LtMatmulPlan&) = delete;
        LtMatmulPlan& operator=(const LtMatmulPlan&) = delete;
        LtMatmulPlan(LtMatmulPlan&& other) noexcept { swap_with(other); }
        LtMatmulPlan& operator=(LtMatmulPlan&& other) noexcept { swap_with(other); return *this; }

        void swap_with(LtMatmulPlan& other) noexcept
        {
            swap(matmul_descriptor, other.matmul_descriptor);
            swap(a_matrix_layout, other.a_matrix_layout);
            swap(b_matrix_layout, other.b_matrix_layout);
            swap(output_matrix_layout, other.output_matrix_layout);
            swap(algorithm, other.algorithm);
            swap(has_algorithm, other.has_algorithm);
            swap(workspace_bytes, other.workspace_bytes);
        }

        ~LtMatmulPlan()
        {
            cublasLtMatrixLayoutDestroy(output_matrix_layout);
            cublasLtMatrixLayoutDestroy(b_matrix_layout);
            cublasLtMatrixLayoutDestroy(a_matrix_layout);
            cublasLtMatmulDescDestroy(matmul_descriptor);
        }
    };

    struct LtMatmulPlanKey
    {
        int m;
        int n;
        int k;
        int transA;
        int transB;
        int epilogue;
        int io_dtype;
        int out_dtype;

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

    struct CudaGemmThreadState
    {
        Buffer workspace{Device::CUDA};

        Buffer bf16_input{Device::CUDA};
        Buffer bf16_gradient{Device::CUDA};
        Buffer bf16_to_fp32{Device::CUDA};

        unordered_map<LtMatmulPlanKey, LtMatmulPlan, LtMatmulPlanKeyHash> matmul_plans;
    };

    CudaGemmThreadState& thread_state()
    {
        thread_local CudaGemmThreadState state;
        return state;
    }

    constexpr size_t cublas_lt_workspace_search_bytes = 32ull * 1024 * 1024;

    cublasComputeType_t gemm_compute_type(cudaDataType_t a_type, cudaDataType_t b_type = CUDA_R_32F)
    {
        return (a_type == CUDA_R_16BF || b_type == CUDA_R_16BF)
            ? CUBLAS_COMPUTE_32F
            : CUBLAS_COMPUTE_DTYPE;
    }

    struct LtMatmulPreferenceGuard
    {
        cublasLtMatmulPreference_t pref = nullptr;
        LtMatmulPreferenceGuard() { CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref)); }
        ~LtMatmulPreferenceGuard() { cublasLtMatmulPreferenceDestroy(pref); }
    };

    bool workspace_growth_forbidden() noexcept
    {
        return device::cuda_allocation_growth_forbidden()
            || env_flag_enabled("OPENNN_CUDA_NO_SCRATCH_GROWTH");
    }

    template <typename T>
    T* ensure_workspace(Buffer& workspace_buffer, Index n)
    {
        if (n * Index(sizeof(T)) > workspace_buffer.bytes && workspace_buffer.data)
        {
            throw_if(workspace_growth_forbidden(),
                     "ensure_workspace: workspace allocation growth is forbidden "
                     "(warmup incomplete before CUDA graph capture).");
            device::synchronize(Backend::get_compute_stream());
        }
        return workspace_buffer.ensure<T>(n);
    }

    void* ensure_cublas_lt_workspace(size_t min_bytes)
    {
        return ensure_workspace<uint8_t>(thread_state().workspace, Index(min_bytes));
    }

    bfloat16* ensure_bf16_input_workspace(Index n)
    {
        return ensure_workspace<bfloat16>(thread_state().bf16_input, n);
    }
}

bfloat16* ensure_bf16_gradient_workspace(Index n)
{
    return ensure_workspace<bfloat16>(thread_state().bf16_gradient, n);
}

float* ensure_bf16_to_fp32_workspace(Index n)
{
    return ensure_workspace<float>(thread_state().bf16_to_fp32, n);
}

void* ensure_cudnn_conv_workspace(size_t min_bytes)
{
    return ensure_workspace<uint8_t>(thread_state().workspace, Index(min_bytes));
}

const void* data_for_gemm_dtype(const TensorView& input, Type target_type)
{
    if (input.type == target_type) return input.data;

    if (input.type == Type::FP32 && target_type == Type::BF16)
    {
        bfloat16* dst = ensure_bf16_input_workspace(input.size());
        cast_fp32_to_bf16_cuda(input.size(), input.as<float>(), dst);
        return dst;
    }

    if (input.type == Type::BF16 && target_type == Type::FP32)
    {
        float* dst = ensure_bf16_to_fp32_workspace(input.size());
        cast_bf16_to_fp32_cuda(input.size(), input.as<bfloat16>(), dst);
        return dst;
    }

    throw runtime_error("data_for_gemm_dtype: unsupported type pair");
}

namespace
{
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
    auto& plans = thread_state().matmul_plans;
    auto it = plans.find(key);
    if (it != plans.end()) return it->second;

    throw_if(workspace_growth_forbidden(),
             "get_lt_gemm_plan: new GEMM plan requested while workspace growth is forbidden "
             "(unseen shape during CUDA graph capture; warmup incomplete).");

    LtMatmulPlan plan;

    CHECK_CUBLAS(cublasLtMatmulDescCreate(&plan.matmul_descriptor, gemm_compute_type(io_dtype), CUDA_R_32F));

    auto set_desc = [&](cublasLtMatmulDescAttributes_t attr, const auto& value)
    {
        CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.matmul_descriptor, attr, &value, sizeof(value)));
    };

    set_desc(CUBLASLT_MATMUL_DESC_TRANSA,   transA);
    set_desc(CUBLASLT_MATMUL_DESC_TRANSB,   transB);
    set_desc(CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
    set_desc(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, out_dtype);

    const int a_rows = (transA == CUBLAS_OP_N) ? m : k;
    const int a_cols = (transA == CUBLAS_OP_N) ? k : m;
    const int b_rows = (transB == CUBLAS_OP_N) ? k : n;
    const int b_cols = (transB == CUBLAS_OP_N) ? n : k;

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.a_matrix_layout,  io_dtype,  a_rows, a_cols, a_rows));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.b_matrix_layout,  io_dtype,  b_rows, b_cols, b_rows));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&plan.output_matrix_layout, out_dtype, m, n, m));

    LtMatmulPreferenceGuard pref;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref.pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &cublas_lt_workspace_search_bytes, sizeof(cublas_lt_workspace_search_bytes)));

    cublasLtMatmulHeuristicResult_t heuristic = {};
    int returned_results = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(Backend::get_cublas_lt_handle(),
                                                plan.matmul_descriptor,
                                                plan.a_matrix_layout,
                                                plan.b_matrix_layout,
                                                plan.output_matrix_layout,
                                                plan.output_matrix_layout,
                                                pref.pref, 1, &heuristic, &returned_results));

    if (returned_results > 0)
    {
        plan.algorithm = heuristic.algo;
        plan.has_algorithm = true;
        plan.workspace_bytes = heuristic.workspaceSize;

        ensure_cublas_lt_workspace(plan.workspace_bytes);
    }

    return plans.emplace(key, move(plan)).first->second;
}
}

void run_lt_matmul_cached(
    int m, int n, int k,
    cublasOperation_t transA,
    cublasOperation_t transB,
    cublasLtEpilogue_t epilogue,
    const void* a_data, const void* b_data, void* c_data,
    const void* bias_pointer,
    cudaDataType_t io_dtype,
    cudaDataType_t out_dtype)
{
    const LtMatmulPlan& plan = get_lt_gemm_plan(m, n, k, transA, transB, epilogue, io_dtype, out_dtype);

    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(plan.matmul_descriptor,
        CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer, sizeof(bias_pointer)));

    CHECK_CUBLAS(cublasLtMatmul(Backend::get_cublas_lt_handle(),
                                plan.matmul_descriptor,
                                &one,
                                a_data, plan.a_matrix_layout,
                                b_data, plan.b_matrix_layout,
                                &zero,
                                c_data, plan.output_matrix_layout,
                                c_data, plan.output_matrix_layout,
                                plan.has_algorithm ? &plan.algorithm : nullptr,
                                ensure_cublas_lt_workspace(plan.workspace_bytes), plan.workspace_bytes,
                                Backend::get_compute_stream()));
}

void gemm_cuda(cublasOperation_t transa, cublasOperation_t transb,
               int m, int n, int k,
               const void* A, cudaDataType_t Atype, int lda,
               const void* B, cudaDataType_t Btype, int ldb,
               void* C, cudaDataType_t Ctype, int ldc,
               float alpha, float beta)
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

void gemm_strided_batched_cuda(cublasOperation_t transa, cublasOperation_t transb,
                               int m, int n, int k,
                               const void* A, cudaDataType_t Atype, int lda, long long stride_a,
                               const void* B, cudaDataType_t Btype, int ldb, long long stride_b,
                               void* C, cudaDataType_t Ctype, int ldc, long long stride_c,
                               int batch_count,
                               float alpha, float beta)
{
    const cublasComputeType_t compute = gemm_compute_type(Atype, Btype);
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(Backend::get_cublas_handle(),
                                            transa, transb,
                                            m, n, k,
                                            &alpha,
                                            A, Atype, lda, stride_a,
                                            B, Btype, ldb, stride_b,
                                            &beta,
                                            C, Ctype, ldc, stride_c,
                                            batch_count,
                                            compute,
                                            CUBLAS_GEMM_DEFAULT));
}

}

#else

namespace opennn
{

bfloat16* ensure_bf16_gradient_workspace(Index)
{
    throw runtime_error("ensure_bf16_gradient_workspace requires CUDA support.");
}

float* ensure_bf16_to_fp32_workspace(Index)
{
    throw runtime_error("ensure_bf16_to_fp32_workspace requires CUDA support.");
}

void* ensure_cudnn_conv_workspace(size_t)
{
    throw runtime_error("ensure_cudnn_conv_workspace requires CUDA support.");
}

const void* data_for_gemm_dtype(const TensorView&, Type)
{
    throw runtime_error("data_for_gemm_dtype requires CUDA support.");
}

void run_lt_matmul_cached(int, int, int,
                          cublasOperation_t,
                          cublasOperation_t,
                          cublasLtEpilogue_t,
                          const void*, const void*, void*,
                          const void*,
                          cudaDataType_t,
                          cudaDataType_t)
{
    throw runtime_error("run_lt_matmul_cached requires CUDA support.");
}

void gemm_cuda(cublasOperation_t, cublasOperation_t,
               int, int, int,
               const void*, cudaDataType_t, int,
               const void*, cudaDataType_t, int,
               void*, cudaDataType_t, int,
               float, float)
{
    throw runtime_error("gemm_cuda requires CUDA support.");
}

void gemm_strided_batched_cuda(cublasOperation_t, cublasOperation_t,
                               int, int, int,
                               const void*, cudaDataType_t, int, long long,
                               const void*, cudaDataType_t, int, long long,
                               void*, cudaDataType_t, int, long long,
                               int,
                               float, float)
{
    throw runtime_error("gemm_strided_batched_cuda requires CUDA support.");
}

}

#endif // OPENNN_HAS_CUDA

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

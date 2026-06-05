//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   C U D A   G E M M   M O D U L E
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "cuda_gemm.h"
#include "device_backend.h"

#ifdef OPENNN_HAS_CUDA

namespace opennn
{

namespace
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

    Buffer cublas_lt_workspace_(Device::CUDA);

    Buffer bf16_input_(Device::CUDA);
    Buffer bf16_gradient_(Device::CUDA);
    Buffer fp32_upcast_(Device::CUDA);

    Buffer cudnn_conv_workspace_(Device::CUDA);

    unordered_map<LtMatmulPlanKey, LtMatmulPlan, LtMatmulPlanKeyHash> lt_gemm_plans_;

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

    template <typename T>
    T* ensure_scratch(Buffer& buffer, Index n)
    {
        if (n * Index(sizeof(T)) > buffer.bytes && buffer.data)
            device::synchronize(Backend::get_compute_stream());
        return buffer.ensure<T>(n);
    }
}

void* ensure_cublas_lt_workspace(size_t min_bytes)
{
    return ensure_scratch<uint8_t>(cublas_lt_workspace_, Index(min_bytes));
}

bfloat16* ensure_bf16_input_scratch(Index n)
{
    return ensure_scratch<bfloat16>(bf16_input_, n);
}

bfloat16* ensure_bf16_gradient_scratch(Index n)
{
    return ensure_scratch<bfloat16>(bf16_gradient_, n);
}

float* ensure_fp32_upcast_scratch(Index n)
{
    return ensure_scratch<float>(fp32_upcast_, n);
}

void* ensure_cudnn_conv_workspace(size_t min_bytes)
{
    return ensure_scratch<uint8_t>(cudnn_conv_workspace_, Index(min_bytes));
}

const void* data_for_gemm_dtype(const TensorView& input, Type target_type)
{
    if (input.type == target_type) return input.data;

    if (input.type == Type::FP32 && target_type == Type::BF16)
    {
        bfloat16* dst = ensure_bf16_input_scratch(input.size());
        cast_fp32_to_bf16_cuda(input.size(), input.as<float>(), dst);
        return dst;
    }

    if (input.type == Type::BF16 && target_type == Type::FP32)
    {
        float* dst = ensure_fp32_upcast_scratch(input.size());
        cast_bf16_to_fp32_cuda(input.size(), input.as<bfloat16>(), dst);
        return dst;
    }

    throw runtime_error("data_for_gemm_dtype: unsupported type pair");
}

static const LtMatmulPlan& get_lt_gemm_plan(
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

    LtMatmulPreferenceGuard pref;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref.pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &cublas_lt_workspace_search_bytes, sizeof(cublas_lt_workspace_search_bytes)));

    cublasLtMatmulHeuristicResult_t heuristic = {};
    int returned_results = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(Backend::get_cublas_lt_handle(),
                                                plan.op_desc,
                                                plan.a_desc, plan.b_desc, plan.cd_desc, plan.cd_desc,
                                                pref.pref, 1, &heuristic, &returned_results));

    if (returned_results > 0)
    {
        plan.algo = heuristic.algo;
        plan.algo_valid = true;
        plan.workspace_size = heuristic.workspaceSize;

        // Grow the global scratch buffer to fit this plan's chosen algorithm.
        ensure_cublas_lt_workspace(plan.workspace_size);
    }

    return lt_gemm_plans_.emplace(key, std::move(plan)).first->second;
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
                                ensure_cublas_lt_workspace(plan.workspace_size), plan.workspace_size,
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
                               const void* A, int lda, long long stride_a,
                               const void* B, int ldb, long long stride_b,
                               void* C, int ldc, long long stride_c,
                               int batch_count,
                               cudaDataType_t io_dtype,
                               float alpha, float beta)
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

}

#else

namespace opennn
{

void* ensure_cublas_lt_workspace(size_t)
{
    throw runtime_error("ensure_cublas_lt_workspace requires CUDA support.");
}

bfloat16* ensure_bf16_input_scratch(Index)
{
    throw runtime_error("ensure_bf16_input_scratch requires CUDA support.");
}

bfloat16* ensure_bf16_gradient_scratch(Index)
{
    throw runtime_error("ensure_bf16_gradient_scratch requires CUDA support.");
}

float* ensure_fp32_upcast_scratch(Index)
{
    throw runtime_error("ensure_fp32_upcast_scratch requires CUDA support.");
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
                               const void*, int, long long,
                               const void*, int, long long,
                               void*, int, long long,
                               int,
                               cudaDataType_t,
                               float, float)
{
    throw runtime_error("gemm_strided_batched_cuda requires CUDA support.");
}

}

#endif // OPENNN_HAS_CUDA

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

// Kernel-choice probe for the dense (HIGGS) training step.
//
// The step of the 28-1024-1024-1 classifier is eight GEMMs: three forward, two
// input-gradient, three weight-gradient. This probe times each of them at the
// benchmark's batch sizes through the three paths the library offers -
// cublasGemmEx, cuBLASLt's first heuristic, and the best of cuBLASLt's top-8 -
// so the step's floor is known before any kernel is written. It answers the
// question the end-to-end numbers cannot: how much of the measured step is the
// shape and how much is the library picking badly.
//
// Standalone on purpose (no libopennn): it builds and answers wherever it runs.
//
//   nvcc -O3 -arch=native -o gemm_probe gemm_probe.cu -lcublas -lcublasLt
//   ./gemm_probe [batch ...]
//
// Layout follows the training step. Everything is row-major in memory and
// mapped to cuBLAS's column-major by the usual transpose identity, which turns
// the three kinds into the standard trio: forward NN, input-gradient TN,
// weight-gradient NT (a long-K reduction over the batch).

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        const cudaError_t status = (call);                                     \
        if (status != cudaSuccess) {                                           \
            printf("CUDA error %s at line %d\n", cudaGetErrorString(status),   \
                   __LINE__);                                                  \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CHECK_BLAS(call)                                                       \
    do {                                                                       \
        const cublasStatus_t status = (call);                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            printf("cuBLAS error %d at line %d\n", int(status), __LINE__);     \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

constexpr int WARMUP_ITERATIONS = 20;
constexpr int TIMED_ITERATIONS = 100;
constexpr size_t WORKSPACE_BYTES = 64ull << 20;
constexpr int MAX_CANDIDATES = 8;

enum class Precision { BF16, TF32 };

// One GEMM of the step, already mapped to column-major.
struct Case
{
    std::string name;       // layer and kind, as the profiler names it
    cublasOperation_t op_a;
    cublasOperation_t op_b;
    int m, n, k;
    int lda, ldb, ldc;
    bool fp32_output;       // weight gradients are stored FP32 from BF16 inputs
};

// The step of inputs -> hidden -> hidden -> 1, at batch `rows`.
//
// forward       Y(rows x oc) = X(rows x ic) . W(ic x oc)      -> m=oc n=rows k=ic  NN
// input grad    dX(rows x ic) = dY(rows x oc) . W^T           -> m=ic n=rows k=oc  TN
// weight grad   dW(ic x oc) = X^T . dY(rows x oc)             -> m=oc n=ic k=rows  NT
std::vector<Case> build_cases(int rows, int inputs, int hidden)
{
    std::vector<Case> cases;

    const auto forward = [&](const std::string& name, int ic, int oc) {
        cases.push_back({name + " fwd", CUBLAS_OP_N, CUBLAS_OP_N,
                         oc, rows, ic, oc, ic, oc, false});
    };
    const auto input_gradient = [&](const std::string& name, int ic, int oc) {
        cases.push_back({name + " dx", CUBLAS_OP_T, CUBLAS_OP_N,
                         ic, rows, oc, oc, oc, ic, false});
    };
    const auto weight_gradient = [&](const std::string& name, int ic, int oc) {
        cases.push_back({name + " wgrad", CUBLAS_OP_N, CUBLAS_OP_T,
                         oc, ic, rows, oc, ic, oc, true});
    };

    forward("L1 28x1024", inputs, hidden);
    forward("L2 1024x1024", hidden, hidden);
    forward("L3 1024x1", hidden, 1);

    // L1 needs no input gradient: nothing consumes it.
    input_gradient("L2 1024x1024", hidden, hidden);
    input_gradient("L3 1024x1", hidden, 1);

    weight_gradient("L1 28x1024", inputs, hidden);
    weight_gradient("L2 1024x1024", hidden, hidden);
    weight_gradient("L3 1024x1", hidden, 1);

    // Padding experiments for the awkward K=28 first layer. cuBLAS falls back to
    // an unvectorised align1 kernel there because a 28-element column is 56
    // bytes, so neither operand is 16-byte aligned: (a) pads the leading
    // dimension only, which leaves the arithmetic identical, and (b) pads K to
    // 32 with zero rows, which costs 14% more flops but makes everything
    // regular. If either flips cuBLAS onto an align8 kernel the first layer is
    // a layout problem, not a kernel problem.
    cases.push_back({"L1 fwd pad-ld32", CUBLAS_OP_N, CUBLAS_OP_N,
                     hidden, rows, inputs, hidden, 32, hidden, false});
    cases.push_back({"L1 fwd pad-k32", CUBLAS_OP_N, CUBLAS_OP_N,
                     hidden, rows, 32, hidden, 32, hidden, false});
    cases.push_back({"L1 wgrad pad-n32", CUBLAS_OP_N, CUBLAS_OP_T,
                     hidden, 32, rows, hidden, 32, hidden, true});

    return cases;
}

struct Buffers
{
    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
};

Buffers allocate(size_t elements, size_t input_bytes, size_t output_bytes)
{
    Buffers buffers;
    CHECK_CUDA(cudaMalloc(&buffers.a, elements * input_bytes));
    CHECK_CUDA(cudaMalloc(&buffers.b, elements * input_bytes));
    CHECK_CUDA(cudaMalloc(&buffers.c, elements * output_bytes));
    CHECK_CUDA(cudaMemset(buffers.a, 0x3c, elements * input_bytes));
    CHECK_CUDA(cudaMemset(buffers.b, 0x3c, elements * input_bytes));
    CHECK_CUDA(cudaMemset(buffers.c, 0, elements * output_bytes));
    return buffers;
}

float time_loop(const std::function<void()>& issue)
{
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    for (int i = 0; i < WARMUP_ITERATIONS; ++i) issue();
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < TIMED_ITERATIONS; ++i) issue();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return milliseconds / TIMED_ITERATIONS;
}

int main(int argc, char* argv[])
{
    std::vector<int> batches;
    for (int i = 1; i < argc; ++i) batches.push_back(atoi(argv[i]));
    if (batches.empty()) batches = {7000, 14000};

    const int inputs = 28;
    const int hidden = 1024;

    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDeviceProperties(&properties, 0));
    printf("device: %s (sm_%d%d, %d SMs)\n\n",
           properties.name, properties.major, properties.minor,
           properties.multiProcessorCount);

    cublasHandle_t blas;
    CHECK_BLAS(cublasCreate(&blas));
    cublasLtHandle_t blas_lt;
    CHECK_BLAS(cublasLtCreate(&blas_lt));

    void* workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, WORKSPACE_BYTES));

    // One allocation big enough for every operand of every case.
    size_t elements = 0;
    for (const int batch : batches)
        for (const Case& c : build_cases(batch, inputs, hidden))
            elements = std::max<size_t>(elements,
                                        size_t(std::max(std::max(c.m, c.n), c.k))
                                      * size_t(std::max(std::max(c.m, c.n), c.k)));
    const Buffers buffers = allocate(elements, sizeof(float), sizeof(float));

    for (const Precision precision : {Precision::BF16, Precision::TF32}) {
        const bool bf16 = precision == Precision::BF16;
        const cudaDataType input_type = bf16 ? CUDA_R_16BF : CUDA_R_32F;
        const cublasComputeType_t compute_type =
            bf16 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32;

        for (const int batch : batches) {
            printf("=== %s, batch %d ===\n", bf16 ? "bf16" : "fp32 (TF32)", batch);
            printf("%-20s %10s %10s %10s %10s %9s\n", "gemm", "GemmEx",
                   "Lt top-1", "Lt best-8", "best", "TFLOPS");

            for (const Case& c : build_cases(batch, inputs, hidden)) {
                const cudaDataType output_type =
                    (bf16 && c.fp32_output) ? CUDA_R_32F : input_type;

                const float alpha = 1.0f;
                const float beta = 0.0f;

                // 1. cublasGemmEx, the library's default algorithm choice.
                const float gemm_ex_ms = time_loop([&] {
                    CHECK_BLAS(cublasGemmEx(
                        blas, c.op_a, c.op_b, c.m, c.n, c.k,
                        &alpha, buffers.a, input_type, c.lda,
                        buffers.b, input_type, c.ldb,
                        &beta, buffers.c, output_type, c.ldc,
                        compute_type, CUBLAS_GEMM_DEFAULT));
                });

                // 2/3. cuBLASLt: the first heuristic, then the best of top-8.
                cublasLtMatmulDesc_t operation = nullptr;
                CHECK_BLAS(cublasLtMatmulDescCreate(&operation, compute_type,
                                                    CUDA_R_32F));
                CHECK_BLAS(cublasLtMatmulDescSetAttribute(
                    operation, CUBLASLT_MATMUL_DESC_TRANSA, &c.op_a, sizeof(c.op_a)));
                CHECK_BLAS(cublasLtMatmulDescSetAttribute(
                    operation, CUBLASLT_MATMUL_DESC_TRANSB, &c.op_b, sizeof(c.op_b)));

                cublasLtMatrixLayout_t layout_a = nullptr;
                cublasLtMatrixLayout_t layout_b = nullptr;
                cublasLtMatrixLayout_t layout_c = nullptr;
                CHECK_BLAS(cublasLtMatrixLayoutCreate(
                    &layout_a, input_type,
                    c.op_a == CUBLAS_OP_N ? c.m : c.k,
                    c.op_a == CUBLAS_OP_N ? c.k : c.m, c.lda));
                CHECK_BLAS(cublasLtMatrixLayoutCreate(
                    &layout_b, input_type,
                    c.op_b == CUBLAS_OP_N ? c.k : c.n,
                    c.op_b == CUBLAS_OP_N ? c.n : c.k, c.ldb));
                CHECK_BLAS(cublasLtMatrixLayoutCreate(&layout_c, output_type,
                                                      c.m, c.n, c.ldc));

                cublasLtMatmulPreference_t preference = nullptr;
                CHECK_BLAS(cublasLtMatmulPreferenceCreate(&preference));
                size_t workspace_bytes = WORKSPACE_BYTES;
                CHECK_BLAS(cublasLtMatmulPreferenceSetAttribute(
                    preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                    &workspace_bytes, sizeof(workspace_bytes)));

                cublasLtMatmulHeuristicResult_t candidates[MAX_CANDIDATES]{};
                int found = 0;
                CHECK_BLAS(cublasLtMatmulAlgoGetHeuristic(
                    blas_lt, operation, layout_a, layout_b, layout_c, layout_c,
                    preference, MAX_CANDIDATES, candidates, &found));

                float top_1_ms = -1.0f;
                float best_8_ms = -1.0f;
                for (int candidate = 0; candidate < found; ++candidate) {
                    const float ms = time_loop([&] {
                        CHECK_BLAS(cublasLtMatmul(
                            blas_lt, operation, &alpha,
                            buffers.a, layout_a, buffers.b, layout_b, &beta,
                            buffers.c, layout_c, buffers.c, layout_c,
                            &candidates[candidate].algo, workspace,
                            WORKSPACE_BYTES, nullptr));
                    });
                    if (candidate == 0) top_1_ms = ms;
                    if (best_8_ms < 0.0f || ms < best_8_ms) best_8_ms = ms;
                }

                const double flops = 2.0 * double(c.m) * double(c.n) * double(c.k);
                float best = gemm_ex_ms;
                if (top_1_ms > 0.0f && top_1_ms < best) best = top_1_ms;
                if (best_8_ms > 0.0f && best_8_ms < best) best = best_8_ms;

                printf("%-20s %10.4f %10.4f %10.4f %10.4f %9.1f\n",
                       c.name.c_str(), gemm_ex_ms, top_1_ms, best_8_ms, best,
                       flops / (best * 1e-3) / 1e12);

                cublasLtMatmulPreferenceDestroy(preference);
                cublasLtMatrixLayoutDestroy(layout_c);
                cublasLtMatrixLayoutDestroy(layout_b);
                cublasLtMatrixLayoutDestroy(layout_a);
                cublasLtMatmulDescDestroy(operation);
            }
            printf("\n");
        }
    }

    cudaFree(workspace);
    cudaFree(buffers.a);
    cudaFree(buffers.b);
    cudaFree(buffers.c);
    cublasLtDestroy(blas_lt);
    cublasDestroy(blas);

    return 0;
}

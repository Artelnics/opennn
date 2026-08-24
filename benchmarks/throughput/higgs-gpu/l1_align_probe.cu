// Does padding the contraction dimension unlock a faster cuBLASLt kernel for
// the first dense layer?
//
// The HIGGS classifier's first layer contracts 28 features. OpenNN hands
// cuBLASLt A = weights (m x k, lda = m = 1024) and B = the input (k x n,
// ldb = k = 28). At bf16 an ldb of 28 elements is 56 bytes, so cuBLASLt can
// only promise two-element alignment on B and dispatches an `align2` kernel;
// PyTorch's inductor emits a Triton template that pads K to 32 internally and
// measured 14.7 us against OpenNN's 21.5 at 8192 rows.
//
// This probe times the same matmul three ways, each with the top-8 heuristic
// timed the way OpenNN's autotune does, so the question "would padding help,
// and by how much" is answered before any library change is written:
//
//   k=28          what OpenNN does now
//   k=32          the contraction zero-padded, which costs 14% more MACs
//   k=28 ld=32    only the input's row stride padded, which costs no
//                 arithmetic at all and is a property of the staging buffer
//
// The third is the interesting one: cuBLASLt's alignment promise is about the
// leading dimension and the pointer, not about k, so a 32-element row stride
// should make the same 28-wide contraction eligible for the aligned kernels.
//
//   nvcc -O3 -arch=native -o l1_align_probe l1_align_probe.cu -lcublasLt
//   ./l1_align_probe [rows ...]

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call)                                                            \
    do { const cudaError_t s = (call);                                         \
         if (s != cudaSuccess) { printf("CUDA %s at line %d\n",                \
             cudaGetErrorString(s), __LINE__); exit(1); } } while (0)

#define CHECK_LT(call)                                                         \
    do { const cublasStatus_t s = (call);                                      \
         if (s != CUBLAS_STATUS_SUCCESS) {                                     \
             printf("cuBLASLt status %d at line %d\n", int(s), __LINE__);      \
             exit(1); } } while (0)

constexpr int HIDDEN = 1024;
constexpr int WARMUP = 20;
constexpr int ITERS = 100;
constexpr size_t WORKSPACE = 32u << 20;

__global__ void fill_kernel(__nv_bfloat16* p, size_t n, float value)
{
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = __float2bfloat16(value * (0.001f * float(i % 97) - 0.05f));
}

void fill(__nv_bfloat16* p, size_t n, float value)
{
    fill_kernel<<<unsigned((n + 255) / 256), 256>>>(p, n, value);
    CHECK(cudaGetLastError());
}

// One (m, n, k) timed the way OpenNN times it: ask for the top-8 heuristics,
// run each, keep the best. Returns milliseconds and writes the winning index.
double time_best(cublasLtHandle_t handle, int m, int n, int k, int ldb,
                 const __nv_bfloat16* a, const __nv_bfloat16* b,
                 __nv_bfloat16* c, const __nv_bfloat16* bias,
                 void* workspace, int& winner, int& candidates)
{
    cublasLtMatmulDesc_t desc = nullptr;
    CHECK_LT(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    const cublasOperation_t no_transpose = CUBLAS_OP_N;
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
    const cudaDataType_t bias_type = CUDA_R_16BF;
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &no_transpose, sizeof(no_transpose)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &no_transpose, sizeof(no_transpose)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type, sizeof(bias_type)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    cublasLtMatrixLayout_t layout_a = nullptr, layout_b = nullptr, layout_c = nullptr;
    CHECK_LT(cublasLtMatrixLayoutCreate(&layout_a, CUDA_R_16BF, m, k, m));
    CHECK_LT(cublasLtMatrixLayoutCreate(&layout_b, CUDA_R_16BF, k, n, ldb));
    CHECK_LT(cublasLtMatrixLayoutCreate(&layout_c, CUDA_R_16BF, m, n, m));

    cublasLtMatmulPreference_t pref = nullptr;
    CHECK_LT(cublasLtMatmulPreferenceCreate(&pref));
    const size_t workspace_bytes = WORKSPACE;
    CHECK_LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspace_bytes, sizeof(workspace_bytes)));

    cublasLtMatmulHeuristicResult_t results[8] = {};
    int returned = 0;
    CHECK_LT(cublasLtMatmulAlgoGetHeuristic(handle, desc, layout_a, layout_b, layout_c, layout_c,
                                            pref, 8, results, &returned));
    cublasLtMatmulPreferenceDestroy(pref);
    candidates = returned;

    const float one = 1.0f, zero = 0.0f;
    double best = 1e30;
    winner = -1;

    for (int candidate = 0; candidate < returned; ++candidate)
    {
        if (results[candidate].state != CUBLAS_STATUS_SUCCESS) continue;

        bool usable = true;
        for (int i = 0; i < WARMUP && usable; ++i)
            usable = cublasLtMatmul(handle, desc, &one, a, layout_a, b, layout_b,
                                    &zero, c, layout_c, c, layout_c,
                                    &results[candidate].algo, workspace,
                                    results[candidate].workspaceSize, nullptr)
                     == CUBLAS_STATUS_SUCCESS;
        if (!usable) continue;
        CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < ITERS; ++i)
            cublasLtMatmul(handle, desc, &one, a, layout_a, b, layout_b,
                           &zero, c, layout_c, c, layout_c,
                           &results[candidate].algo, workspace,
                           results[candidate].workspaceSize, nullptr);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));

        float elapsed = 0.0f;
        CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));

        const double ms = double(elapsed) / ITERS;
        if (ms < best) { best = ms; winner = candidate; }
    }

    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(desc);
    return best;
}

int main(int argc, char** argv)
{
    std::vector<int> row_counts;
    for (int i = 1; i < argc; ++i) row_counts.push_back(atoi(argv[i]));
    if (row_counts.empty()) row_counts = {256, 1024, 4096, 8192, 16384, 65536};

    cudaDeviceProp properties{};
    CHECK(cudaGetDeviceProperties(&properties, 0));
    printf("device: %s (%d SMs, %d MB L2)\n\n", properties.name,
           properties.multiProcessorCount, properties.l2CacheSize >> 20);

    cublasLtHandle_t handle = nullptr;
    CHECK_LT(cublasLtCreate(&handle));

    void* workspace = nullptr;
    CHECK(cudaMalloc(&workspace, WORKSPACE));

    const int max_rows = *std::max_element(row_counts.begin(), row_counts.end());

    __nv_bfloat16 *weights = nullptr, *input = nullptr, *output = nullptr, *bias = nullptr;
    CHECK(cudaMalloc(&weights, size_t(HIDDEN) * 32 * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&input,   size_t(max_rows) * 32 * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&output,  size_t(max_rows) * HIDDEN * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&bias,    size_t(HIDDEN) * sizeof(__nv_bfloat16)));

    fill(weights, size_t(HIDDEN) * 32, 1.0f);
    fill(input, size_t(max_rows) * 32, 1.0f);
    fill(bias, HIDDEN, 0.5f);
    CHECK(cudaDeviceSynchronize());

    printf("%8s %13s %13s %15s %12s %12s\n", "rows", "k=28 (ms)", "k=32 (ms)",
           "k=28 ld=32 (ms)", "28/32", "28/ld32");
    for (const int rows : row_counts)
    {
        int winner = 0, candidates = 0;
        const double ms_28 = time_best(handle, HIDDEN, rows, 28, 28, weights, input, output, bias,
                                       workspace, winner, candidates);
        const double ms_32 = time_best(handle, HIDDEN, rows, 32, 32, weights, input, output, bias,
                                       workspace, winner, candidates);
        const double ms_ld = time_best(handle, HIDDEN, rows, 28, 32, weights, input, output, bias,
                                       workspace, winner, candidates);
        printf("%8d %13.4f %13.4f %15.4f %11.3fx %11.3fx\n", rows, ms_28, ms_32, ms_ld,
               ms_28 / ms_32, ms_28 / ms_ld);
    }

    cudaFree(bias); cudaFree(output); cudaFree(input); cudaFree(weights); cudaFree(workspace);
    cublasLtDestroy(handle);
    return 0;
}

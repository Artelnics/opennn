// Does the hidden layer's GEMM lose throughput once its operands stop fitting
// in L2, and does splitting the rows get it back?
//
// The HIGGS classifier's 1024x1024 layer is 86% of an fp32 inference batch and
// 87% of a bf16 one, so it is the only item on the list worth attacking at
// large batch. In the application it measures 44.6 TFLOP/s at 8,192 rows and
// 42.5 at 65,536 - and at 65,536 the activation tensors are 268 MB (fp32)
// against this card's 48 MB of L2, while at 8,192 they are 32 MB and fit.
//
// If that is the cause, then calling cuBLASLt once per chunk of rows - each
// chunk sized to fit L2 - should recover the small-batch rate, at the cost of
// re-reading the 4 MB weight panel per chunk (which stays in L2) and one extra
// launch per chunk. This probe times exactly that, one call against a ladder of
// chunk sizes, in both precisions, with the top-8 heuristics timed the way
// OpenNN's autotune times them.
//
//   nvcc -O3 -arch=native -o gemm_chunk_probe gemm_chunk_probe.cu -lcublasLt
//   ./gemm_chunk_probe [rows ...]

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

constexpr int WIDTH = 1024;                 // both the input and output features
constexpr int WARMUP = 5;
constexpr int ITERS = 30;
constexpr size_t WORKSPACE = 64u << 20;

__global__ void fill_kernel(char* raw, size_t n, int bf16)
{
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = 0.01f * float(i % 61) - 0.3f;
    if (bf16) reinterpret_cast<__nv_bfloat16*>(raw)[i] = __float2bfloat16(v);
    else      reinterpret_cast<float*>(raw)[i] = v;
}

void fill(void* p, size_t n, bool bf16)
{
    fill_kernel<<<unsigned((n + 255) / 256), 256>>>(static_cast<char*>(p), n, bf16 ? 1 : 0);
    CHECK(cudaGetLastError());
}

struct Plan
{
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
    bool has_algo = false;
    size_t workspace = 0;
};

// One plan for one (m, n, k), with the winner of the top-8 heuristics pinned -
// the same selection OpenNN's autotune performs on its first real call.
Plan make_plan(cublasLtHandle_t handle, int m, int n, int k, bool bf16,
               const void* a_data, const void* b_data, void* c_data,
               const void* bias, void* workspace)
{
    const cudaDataType_t io = bf16 ? CUDA_R_16BF : CUDA_R_32F;
    const cublasComputeType_t compute = bf16 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32;

    Plan plan;
    CHECK_LT(cublasLtMatmulDescCreate(&plan.desc, compute, CUDA_R_32F));

    const cublasOperation_t no_transpose = CUBLAS_OP_N;
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
    CHECK_LT(cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_TRANSA, &no_transpose, sizeof(no_transpose)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_TRANSB, &no_transpose, sizeof(no_transpose)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &io, sizeof(io)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(plan.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    CHECK_LT(cublasLtMatrixLayoutCreate(&plan.a, io, m, k, m));
    CHECK_LT(cublasLtMatrixLayoutCreate(&plan.b, io, k, n, k));
    CHECK_LT(cublasLtMatrixLayoutCreate(&plan.c, io, m, n, m));

    cublasLtMatmulPreference_t pref = nullptr;
    CHECK_LT(cublasLtMatmulPreferenceCreate(&pref));
    const size_t workspace_bytes = WORKSPACE;
    CHECK_LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                  &workspace_bytes, sizeof(workspace_bytes)));

    cublasLtMatmulHeuristicResult_t results[8] = {};
    int returned = 0;
    CHECK_LT(cublasLtMatmulAlgoGetHeuristic(handle, plan.desc, plan.a, plan.b, plan.c, plan.c,
                                            pref, 8, results, &returned));
    cublasLtMatmulPreferenceDestroy(pref);

    const float one = 1.0f, zero = 0.0f;
    double best = 1e30;

    for (int candidate = 0; candidate < returned; ++candidate)
    {
        if (results[candidate].state != CUBLAS_STATUS_SUCCESS) continue;

        const auto run = [&] {
            return cublasLtMatmul(handle, plan.desc, &one, a_data, plan.a, b_data, plan.b,
                                  &zero, c_data, plan.c, c_data, plan.c,
                                  &results[candidate].algo, workspace,
                                  results[candidate].workspaceSize, nullptr);
        };
        if (run() != CUBLAS_STATUS_SUCCESS) continue;
        CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        for (int i = 0; i < 3; ++i) run();
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));

        if (double(ms) < best)
        {
            best = double(ms);
            plan.algo = results[candidate].algo;
            plan.has_algo = true;
            plan.workspace = results[candidate].workspaceSize;
        }
    }
    return plan;
}

void destroy(Plan& plan)
{
    cublasLtMatrixLayoutDestroy(plan.c);
    cublasLtMatrixLayoutDestroy(plan.b);
    cublasLtMatrixLayoutDestroy(plan.a);
    cublasLtMatmulDescDestroy(plan.desc);
}

// Time `rows` rows through the plan, in ceil(rows / chunk) calls.
double time_chunked(cublasLtHandle_t handle, Plan& plan, int rows, int chunk, int element_bytes,
                    const void* a_data, const char* b_data, char* c_data, void* workspace)
{
    const float one = 1.0f, zero = 0.0f;
    const size_t b_stride = size_t(chunk) * WIDTH * size_t(element_bytes);
    const size_t c_stride = b_stride;

    const auto pass = [&] {
        for (int start = 0; start < rows; start += chunk)
            cublasLtMatmul(handle, plan.desc, &one,
                           a_data, plan.a,
                           b_data + (size_t(start) / size_t(chunk)) * b_stride, plan.b,
                           &zero,
                           c_data + (size_t(start) / size_t(chunk)) * c_stride, plan.c,
                           c_data + (size_t(start) / size_t(chunk)) * c_stride, plan.c,
                           plan.has_algo ? &plan.algo : nullptr,
                           workspace, plan.workspace, nullptr);
    };

    for (int i = 0; i < WARMUP; ++i) pass();
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) pass();
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    return double(ms) / ITERS;
}

int main(int argc, char** argv)
{
    std::vector<int> row_counts;
    for (int i = 1; i < argc; ++i) row_counts.push_back(atoi(argv[i]));
    if (row_counts.empty()) row_counts = {8192, 16384, 32768, 65536};

    cudaDeviceProp properties{};
    CHECK(cudaGetDeviceProperties(&properties, 0));
    printf("device: %s (%d SMs, %d MB L2)\n", properties.name,
           properties.multiProcessorCount, properties.l2CacheSize >> 20);

    cublasLtHandle_t handle = nullptr;
    CHECK_LT(cublasLtCreate(&handle));
    void* workspace = nullptr;
    CHECK(cudaMalloc(&workspace, WORKSPACE));

    const int max_rows = *std::max_element(row_counts.begin(), row_counts.end());
    const std::vector<int> chunks = {2048, 4096, 8192, 16384, 32768};

    for (const bool bf16 : {true, false})
    {
        const int element_bytes = bf16 ? 2 : 4;
        void *weights = nullptr, *input = nullptr, *output = nullptr, *bias = nullptr;
        CHECK(cudaMalloc(&weights, size_t(WIDTH) * WIDTH * element_bytes));
        CHECK(cudaMalloc(&input, size_t(max_rows) * WIDTH * element_bytes));
        CHECK(cudaMalloc(&output, size_t(max_rows) * WIDTH * element_bytes));
        CHECK(cudaMalloc(&bias, size_t(WIDTH) * element_bytes));
        fill(weights, size_t(WIDTH) * WIDTH, bf16);
        fill(input, size_t(max_rows) * WIDTH, bf16);
        fill(bias, WIDTH, bf16);
        CHECK(cudaDeviceSynchronize());

        printf("\n=== %s, 1024x1024 layer, TFLOP/s (activation tensor MB in brackets) ===\n",
               bf16 ? "bf16" : "fp32 (TF32)");
        printf("%8s %10s", "rows", "one call");
        for (const int chunk : chunks) printf(" %9d", chunk);
        printf("   best\n");

        for (const int rows : row_counts)
        {
            const double flops = 2.0 * double(rows) * WIDTH * WIDTH;
            Plan whole = make_plan(handle, WIDTH, rows, WIDTH, bf16, weights, input, output, bias, workspace);
            const double ms_whole = time_chunked(handle, whole, rows, rows, element_bytes,
                                                 weights, static_cast<const char*>(input),
                                                 static_cast<char*>(output), workspace);
            destroy(whole);

            printf("%5d[%3.0fMB] %10.1f", rows,
                   double(rows) * WIDTH * element_bytes / 1e6, flops / (ms_whole * 1e9));

            double best = flops / (ms_whole * 1e9);
            for (const int chunk : chunks)
            {
                if (chunk >= rows) { printf(" %9s", "-"); continue; }
                Plan plan = make_plan(handle, WIDTH, chunk, WIDTH, bf16, weights, input, output, bias, workspace);
                const double ms = time_chunked(handle, plan, rows, chunk, element_bytes,
                                               weights, static_cast<const char*>(input),
                                               static_cast<char*>(output), workspace);
                destroy(plan);
                const double tflops = flops / (ms * 1e9);
                best = std::max(best, tflops);
                printf(" %9.1f", tflops);
            }
            printf("   %6.1f\n", best);
        }

        cudaFree(bias); cudaFree(output); cudaFree(input); cudaFree(weights);
    }

    cudaFree(workspace);
    cublasLtDestroy(handle);
    return 0;
}

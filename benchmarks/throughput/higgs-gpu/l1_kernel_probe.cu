// A kernel for the dense forward whose contraction is narrow.
//
// The HIGGS classifier's first layer contracts 28 features into 1024. cuBLASLt
// dispatches a general GEMM and reaches 23.8 TFLOP/s on it - the operation is
// really a broadcast of 28 numbers against a 28x1024 panel, and what bounds it
// is writing the output. Measured in the application, bf16: 21.6 us at 8,192
// rows against PyTorch's Triton template at 13.1, and 3.09 us against 1.24 at
// 256 rows. Padding the contraction, padding the leading dimension, and a wider
// heuristic search were all measured and none of them close it, so the
// remaining option is a kernel.
//
// The shape of the kernel follows from the arithmetic. Each output element
// needs k multiply-adds, so per output the cost is k loads of x, k loads of w
// and k fused multiply-adds unless operands are reused from registers. Here k
// is small enough that a thread can hold the whole weight column it owns:
// 28 values, or 14 registers when two adjacent columns are packed. Then x is
// the only thing read per row, from shared memory and at the same address for
// every thread in the block, which is a broadcast rather than a gather.
//
//   nvcc -O3 -arch=native -o l1_kernel_probe l1_kernel_probe.cu -lcublasLt
//   ./l1_kernel_probe [rows ...]

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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

constexpr int FEATURES = 28;
constexpr int HIDDEN = 1024;
constexpr int WARMUP = 20;
constexpr int ITERS = 100;
constexpr size_t WORKSPACE = 32u << 20;

// ---------------------------------------------------------------- the kernel

constexpr int NARROW_MAX_K = 32;      // the contraction the register file holds
constexpr int NARROW_ROWS_TILE = 32;  // rows whose inputs share one staging pass
constexpr int NARROW_THREADS = 128;   // 2 output columns each -> 256 per block

// out[row][col] = act(bias[col] + sum_f in[row][f] * w[f][col]).
//
// One thread owns two adjacent output columns for the life of the kernel and
// keeps their whole weight column in registers - 2k values, loaded once. The
// block stages ROWS_TILE rows of input into shared memory, and then every
// multiply reads x from the same shared address across the whole block, which
// the hardware broadcasts.
//
// Accumulation is fp32 and f ascends, so the result is deterministic and the
// summation order is the same one cuBLAS's own reduction would use.
template<typename T, int K, int ROWS_TILE>
__global__ void narrow_k_forward_kernel(const int rows, const int k, const int out_features,
                                        const T* __restrict__ input,
                                        const T* __restrict__ weights,
                                        const T* __restrict__ bias,
                                        T* __restrict__ output)
{
    __shared__ float staged[ROWS_TILE][K];

    const int col = (blockIdx.x * NARROW_THREADS + threadIdx.x) * 2;
    const int row0 = blockIdx.y * ROWS_TILE;
    const bool has_columns = col + 1 < out_features;

    // K is a compile-time constant so that these unroll into registers. With a
    // runtime bound the compiler puts them in local memory, which is backed by
    // DRAM: measured at 58 GB/s of output against the 1,700 the store floor
    // reaches, a fourteenfold loss and the whole difference between this kernel
    // being worth writing and not.
    float w0[K];
    float w1[K];

    #pragma unroll
    for (int f = 0; f < K; ++f)
    {
        const bool live = f < k && has_columns;
        w0[f] = live ? float(weights[f * out_features + col]) : 0.0f;
        w1[f] = live ? float(weights[f * out_features + col + 1]) : 0.0f;
    }

    const int tile_rows = min(ROWS_TILE, rows - row0);

    // Stage this tile's inputs once for the whole block. The padding columns
    // are zeroed so the unrolled loop can run to K without a branch.
    for (int i = threadIdx.x; i < ROWS_TILE * K; i += NARROW_THREADS)
    {
        const int r = i / K;
        const int f = i - r * K;
        staged[r][f] = (r < tile_rows && f < k)
            ? float(input[size_t(row0 + r) * size_t(k) + f])
            : 0.0f;
    }
    __syncthreads();

    if (!has_columns) return;

    const float bias0 = bias ? float(bias[col]) : 0.0f;
    const float bias1 = bias ? float(bias[col + 1]) : 0.0f;

    for (int r = 0; r < tile_rows; ++r)
    {
        float sum0 = bias0;
        float sum1 = bias1;

        #pragma unroll
        for (int f = 0; f < K; ++f)
        {
            const float x = staged[r][f];       // one address for the whole block
            sum0 += x * w0[f];
            sum1 += x * w1[f];
        }

        sum0 = fmaxf(sum0, 0.0f);
        sum1 = fmaxf(sum1, 0.0f);

        T* out = output + size_t(row0 + r) * size_t(out_features) + col;
        out[0] = T(sum0);
        out[1] = T(sum1);
    }
}

template<typename T, int ROWS_TILE>
float time_narrow(int rows, const T* input, const T* weights, const T* bias, T* output)
{
    const dim3 threads(NARROW_THREADS);
    const dim3 blocks((HIDDEN / 2 + NARROW_THREADS - 1) / NARROW_THREADS,
                      (rows + ROWS_TILE - 1) / ROWS_TILE);

    for (int i = 0; i < WARMUP; ++i)
        narrow_k_forward_kernel<T, NARROW_MAX_K, ROWS_TILE><<<blocks, threads>>>(rows, FEATURES, HIDDEN, input, weights, bias, output);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t began = nullptr, ended = nullptr;
    CHECK(cudaEventCreate(&began)); CHECK(cudaEventCreate(&ended));
    CHECK(cudaEventRecord(began));
    for (int i = 0; i < ITERS; ++i)
        narrow_k_forward_kernel<T, NARROW_MAX_K, ROWS_TILE><<<blocks, threads>>>(rows, FEATURES, HIDDEN, input, weights, bias, output);
    CHECK(cudaEventRecord(ended));
    CHECK(cudaEventSynchronize(ended));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, began, ended));
    CHECK(cudaEventDestroy(began)); CHECK(cudaEventDestroy(ended));
    return ms / ITERS;
}

// ------------------------------------------------------------- the reference

__global__ void fill_kernel(char* raw, size_t n, int bf16, float scale)
{
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = scale * (0.01f * float(i % 61) - 0.3f);
    if (bf16) reinterpret_cast<__nv_bfloat16*>(raw)[i] = __float2bfloat16(v);
    else      reinterpret_cast<float*>(raw)[i] = v;
}

void fill(void* p, size_t n, bool bf16, float scale = 1.0f)
{
    fill_kernel<<<unsigned((n + 255) / 256), 256>>>(static_cast<char*>(p), n, bf16 ? 1 : 0, scale);
    CHECK(cudaGetLastError());
}

// --------------------------------------------------------------- the cuBLAS

double time_cublas(cublasLtHandle_t handle, int rows, bool bf16,
                   const void* a, const void* b, void* c, const void* bias, void* workspace)
{
    const cudaDataType_t io = bf16 ? CUDA_R_16BF : CUDA_R_32F;
    const cublasComputeType_t compute = bf16 ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32;

    cublasLtMatmulDesc_t desc = nullptr;
    CHECK_LT(cublasLtMatmulDescCreate(&desc, compute, CUDA_R_32F));
    const cublasOperation_t n_op = CUBLAS_OP_N;
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &n_op, sizeof(n_op)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &n_op, sizeof(n_op)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &io, sizeof(io)));
    CHECK_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    CHECK_LT(cublasLtMatrixLayoutCreate(&la, io, HIDDEN, FEATURES, HIDDEN));
    CHECK_LT(cublasLtMatrixLayoutCreate(&lb, io, FEATURES, rows, FEATURES));
    CHECK_LT(cublasLtMatrixLayoutCreate(&lc, io, HIDDEN, rows, HIDDEN));

    cublasLtMatmulPreference_t pref = nullptr;
    CHECK_LT(cublasLtMatmulPreferenceCreate(&pref));
    const size_t ws = WORKSPACE;
    CHECK_LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));

    cublasLtMatmulHeuristicResult_t results[8] = {};
    int returned = 0;
    CHECK_LT(cublasLtMatmulAlgoGetHeuristic(handle, desc, la, lb, lc, lc, pref, 8, results, &returned));
    cublasLtMatmulPreferenceDestroy(pref);

    const float one = 1.0f, zero = 0.0f;
    double best = 1e30;
    for (int cand = 0; cand < returned; ++cand)
    {
        if (results[cand].state != CUBLAS_STATUS_SUCCESS) continue;
        const auto run = [&] {
            return cublasLtMatmul(handle, desc, &one, a, la, b, lb, &zero, c, lc, c, lc,
                                  &results[cand].algo, workspace, results[cand].workspaceSize, nullptr);
        };
        if (run() != CUBLAS_STATUS_SUCCESS) continue;
        for (int i = 0; i < WARMUP; ++i) run();
        CHECK(cudaDeviceSynchronize());
        cudaEvent_t began = nullptr, ended = nullptr;
        CHECK(cudaEventCreate(&began)); CHECK(cudaEventCreate(&ended));
        CHECK(cudaEventRecord(began));
        for (int i = 0; i < ITERS; ++i) run();
        CHECK(cudaEventRecord(ended));
        CHECK(cudaEventSynchronize(ended));
        float ms = 0.0f; CHECK(cudaEventElapsedTime(&ms, began, ended));
        CHECK(cudaEventDestroy(began)); CHECK(cudaEventDestroy(ended));
        best = std::min(best, double(ms) / ITERS);
    }

    cublasLtMatrixLayoutDestroy(lc); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(la);
    cublasLtMatmulDescDestroy(desc);
    return best;
}

int main(int argc, char** argv)
{
    std::vector<int> row_counts;
    for (int i = 1; i < argc; ++i) row_counts.push_back(atoi(argv[i]));
    if (row_counts.empty()) row_counts = {256, 1024, 4096, 8192, 16384, 65536};

    cudaDeviceProp props{};
    CHECK(cudaGetDeviceProperties(&props, 0));
    printf("device: %s (%d SMs, %d MB L2)\n", props.name, props.multiProcessorCount, props.l2CacheSize >> 20);

    cublasLtHandle_t handle = nullptr;
    CHECK_LT(cublasLtCreate(&handle));
    void* workspace = nullptr;
    CHECK(cudaMalloc(&workspace, WORKSPACE));

    const int max_rows = *std::max_element(row_counts.begin(), row_counts.end());

    for (const bool bf16 : {true, false})
    {
        const int bytes = bf16 ? 2 : 4;
        void *w = nullptr, *x = nullptr, *out = nullptr, *out_ref = nullptr, *bias = nullptr;
        CHECK(cudaMalloc(&w, size_t(FEATURES) * HIDDEN * bytes));
        CHECK(cudaMalloc(&x, size_t(max_rows) * FEATURES * bytes));
        CHECK(cudaMalloc(&out, size_t(max_rows) * HIDDEN * bytes));
        CHECK(cudaMalloc(&out_ref, size_t(max_rows) * HIDDEN * bytes));
        CHECK(cudaMalloc(&bias, size_t(HIDDEN) * bytes));
        fill(w, size_t(FEATURES) * HIDDEN, bf16, 0.3f);
        fill(x, size_t(max_rows) * FEATURES, bf16, 1.0f);
        fill(bias, HIDDEN, bf16, 0.1f);
        CHECK(cudaDeviceSynchronize());

        printf("\n=== %s, 28 -> 1024 forward with bias and ReLU, milliseconds ===\n", bf16 ? "bf16" : "fp32 (TF32)");
        printf("%8s %14s %14s %10s %12s\n", "rows", "cuBLASLt best8", "narrow-k", "speedup", "out GB/s");

        for (const int rows : row_counts)
        {
            const double lt = time_cublas(handle, rows, bf16, w, x, out_ref, bias, workspace);
            float mine = 1e30f;
            int best_tile = 0;
            const auto consider = [&](float ms, int tile) { if (ms < mine) { mine = ms; best_tile = tile; } };
            if (bf16)
            {
                consider(time_narrow<__nv_bfloat16, 4>(rows, (const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (const __nv_bfloat16*)bias, (__nv_bfloat16*)out), 4);
                consider(time_narrow<__nv_bfloat16, 8>(rows, (const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (const __nv_bfloat16*)bias, (__nv_bfloat16*)out), 8);
                consider(time_narrow<__nv_bfloat16, 16>(rows, (const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (const __nv_bfloat16*)bias, (__nv_bfloat16*)out), 16);
                consider(time_narrow<__nv_bfloat16, 32>(rows, (const __nv_bfloat16*)x, (const __nv_bfloat16*)w, (const __nv_bfloat16*)bias, (__nv_bfloat16*)out), 32);
            }
            else
            {
                consider(time_narrow<float, 4>(rows, (const float*)x, (const float*)w, (const float*)bias, (float*)out), 4);
                consider(time_narrow<float, 8>(rows, (const float*)x, (const float*)w, (const float*)bias, (float*)out), 8);
                consider(time_narrow<float, 16>(rows, (const float*)x, (const float*)w, (const float*)bias, (float*)out), 16);
                consider(time_narrow<float, 32>(rows, (const float*)x, (const float*)w, (const float*)bias, (float*)out), 32);
            }

            // Agreement with cuBLASLt's own result, which is the reference here.
            std::vector<char> a(size_t(rows) * HIDDEN * bytes), b(size_t(rows) * HIDDEN * bytes);
            CHECK(cudaMemcpy(a.data(), out, a.size(), cudaMemcpyDeviceToHost));
            CHECK(cudaMemcpy(b.data(), out_ref, b.size(), cudaMemcpyDeviceToHost));
            double worst = 0.0;
            for (size_t i = 0; i < size_t(rows) * HIDDEN; ++i)
            {
                const float va = bf16 ? float(((__nv_bfloat16*)a.data())[i]) : ((float*)a.data())[i];
                const float vb = bf16 ? float(((__nv_bfloat16*)b.data())[i]) : ((float*)b.data())[i];
                worst = std::max(worst, double(fabsf(va - vb)));
            }

            const double gb = double(rows) * HIDDEN * bytes / 1e9;
            printf("%8d %14.4f %14.4f %9.3fx %12.0f   tile %2d  max|diff| %.3g\n",
                   rows, lt, double(mine), lt / double(mine), gb / (double(mine) / 1e3), best_tile, worst);
        }

        cudaFree(bias); cudaFree(out_ref); cudaFree(out); cudaFree(x); cudaFree(w);
    }

    cudaFree(workspace);
    cublasLtDestroy(handle);
    return 0;
}

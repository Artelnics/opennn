// The narrow first layer through CUTLASS rather than cuBLASLt.
//
// cuBLASLt already runs CUTLASS-generated kernels - the profiles name them,
// `cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_...` - but from a fixed
// catalogue built for sm_80 and reused here. For the 28 -> 1024 layer it
// dispatches an `align2` kernel, because the input's leading dimension is 28
// elements, and reaches 23.8 TFLOP/s where PyTorch's Triton template, generated
// for this exact shape, reaches 39.
//
// So this instantiates the kernel directly, with the alignment and the tile
// chosen for the shape instead of looked up: A is the input (rows x 28,
// row-major, so alignment 4 divides 28 where 8 does not), B is the weight panel
// (28 x 1024, row-major, alignment 8), and the bias arrives as a C operand
// whose row stride is zero, which broadcasts one vector down the whole output
// and lets the ReLU ride in the same epilogue.
//
// Several threadblock tiles are timed because there is no reason to expect the
// catalogue's choice to be right for a contraction this short.
//
//   nvcc -O3 -arch=native -std=c++17 -I <cutlass>/include --expt-relaxed-constexpr \
//        -o l1_cutlass_probe l1_cutlass_probe.cu -lcublasLt
//   ./l1_cutlass_probe [rows ...]

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/numeric_types.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call)                                                            \
    do { const cudaError_t status_ = (call);                                   \
         if (status_ != cudaSuccess) { printf("CUDA %s at line %d\n",          \
             cudaGetErrorString(status_), __LINE__); exit(1); } } while (0)

#define CHECK_LT(call)                                                         \
    do { const cublasStatus_t status_ = (call);                                \
         if (status_ != CUBLAS_STATUS_SUCCESS) {                               \
             printf("cuBLASLt status %d at line %d\n", int(status_), __LINE__);\
             exit(1); } } while (0)

constexpr int FEATURES = 28;
constexpr int HIDDEN = 1024;
constexpr int WARMUP = 20;
constexpr int ITERS = 100;
constexpr size_t WORKSPACE = 32u << 20;

using Element = cutlass::bfloat16_t;
using Accumulator = float;

// A = input (rows x K) row-major, B = weights (K x N) row-major,
// C = bias broadcast, D = output (rows x N) row-major.
template<typename ThreadblockShape, typename WarpShape, int Stages>
struct NarrowGemm
{
    using Epilogue = cutlass::epilogue::thread::LinearCombinationRelu<
        Element, 8, Accumulator, Accumulator>;

    using Gemm = cutlass::gemm::device::Gemm<
        Element, cutlass::layout::RowMajor,
        Element, cutlass::layout::RowMajor,
        Element, cutlass::layout::RowMajor,
        Accumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape,
        cutlass::gemm::GemmShape<16, 8, 16>,
        Epilogue,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        Stages,
        4,                          // alignment of A: 28 is divisible by 4, not 8
        8                           // alignment of B
    >;

    static double run(int rows, const Element* x, const Element* w,
                      const Element* bias, Element* out)
    {
        typename Gemm::Arguments args(
            {rows, HIDDEN, FEATURES},
            {const_cast<Element*>(x), FEATURES},
            {const_cast<Element*>(w), HIDDEN},
            {const_cast<Element*>(bias), 0},        // stride 0 broadcasts the bias
            {out, HIDDEN},
            {Accumulator(1), Accumulator(1)});      // alpha = 1, beta = 1 (add bias)

        Gemm gemm;
        if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -1.0;
        if (gemm.initialize(args) != cutlass::Status::kSuccess) return -2.0;

        for (int i = 0; i < WARMUP; ++i)
            if (gemm() != cutlass::Status::kSuccess) return -3.0;
        CHECK(cudaDeviceSynchronize());

        cudaEvent_t began = nullptr, ended = nullptr;
        CHECK(cudaEventCreate(&began)); CHECK(cudaEventCreate(&ended));
        CHECK(cudaEventRecord(began));
        for (int i = 0; i < ITERS; ++i) gemm();
        CHECK(cudaEventRecord(ended));
        CHECK(cudaEventSynchronize(ended));
        float ms = 0.0f;
        CHECK(cudaEventElapsedTime(&ms, began, ended));
        CHECK(cudaEventDestroy(began)); CHECK(cudaEventDestroy(ended));
        return double(ms) / ITERS;
    }
};

__global__ void fill_kernel(__nv_bfloat16* p, size_t n, float scale)
{
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) p[i] = __float2bfloat16(scale * (0.01f * float(i % 61) - 0.3f));
}

void fill(void* p, size_t n, float scale)
{
    fill_kernel<<<unsigned((n + 255) / 256), 256>>>(static_cast<__nv_bfloat16*>(p), n, scale);
    CHECK(cudaGetLastError());
}

double time_cublas(cublasLtHandle_t handle, int rows,
                   const void* a, const void* b, void* c, const void* bias, void* workspace)
{
    cublasLtMatmulDesc_t desc = nullptr;
    CHECK_LT(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t n_op = CUBLAS_OP_N;
    const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_RELU_BIAS;
    const cudaDataType_t io = CUDA_R_16BF;
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
    printf("device: %s (%d SMs, %d MB L2)\n\n", props.name, props.multiProcessorCount, props.l2CacheSize >> 20);

    cublasLtHandle_t handle = nullptr;
    CHECK_LT(cublasLtCreate(&handle));
    void* workspace = nullptr;
    CHECK(cudaMalloc(&workspace, WORKSPACE));

    const int max_rows = *std::max_element(row_counts.begin(), row_counts.end());
    Element *w = nullptr, *x = nullptr, *out = nullptr, *ref = nullptr, *bias = nullptr;
    CHECK(cudaMalloc(&w, size_t(FEATURES) * HIDDEN * sizeof(Element)));
    CHECK(cudaMalloc(&x, size_t(max_rows) * FEATURES * sizeof(Element)));
    CHECK(cudaMalloc(&out, size_t(max_rows) * HIDDEN * sizeof(Element)));
    CHECK(cudaMalloc(&ref, size_t(max_rows) * HIDDEN * sizeof(Element)));
    CHECK(cudaMalloc(&bias, size_t(HIDDEN) * sizeof(Element)));
    fill(w, size_t(FEATURES) * HIDDEN, 0.3f);
    fill(x, size_t(max_rows) * FEATURES, 1.0f);
    fill(bias, HIDDEN, 0.1f);
    CHECK(cudaDeviceSynchronize());

    using T1 = cutlass::gemm::GemmShape<128, 128, 32>;
    using T2 = cutlass::gemm::GemmShape<128, 64, 32>;
    using T3 = cutlass::gemm::GemmShape<64, 128, 32>;
    using T4 = cutlass::gemm::GemmShape<64, 64, 32>;
    using T5 = cutlass::gemm::GemmShape<256, 128, 32>;
    using W1 = cutlass::gemm::GemmShape<64, 64, 32>;
    using W2 = cutlass::gemm::GemmShape<32, 64, 32>;
    using W3 = cutlass::gemm::GemmShape<64, 32, 32>;

    const char* names[] = {"128x128/64x64 s3", "128x64/64x32 s4", "64x128/32x64 s4",
                           "64x64/32x64 s6", "256x128/64x64 s3"};

    printf("%8s %12s %10s %10s %10s %10s %10s   %s\n", "rows", "cuBLASLt",
           names[0], names[1], names[2], names[3], names[4], "best");
    for (const int rows : row_counts)
    {
        const double lt = time_cublas(handle, rows, w, x, ref, bias, workspace);

        double ms[5];
        ms[0] = NarrowGemm<T1, W1, 3>::run(rows, x, w, bias, out);
        ms[1] = NarrowGemm<T2, W3, 4>::run(rows, x, w, bias, out);
        ms[2] = NarrowGemm<T3, W2, 4>::run(rows, x, w, bias, out);
        ms[3] = NarrowGemm<T4, W2, 6>::run(rows, x, w, bias, out);
        ms[4] = NarrowGemm<T5, W1, 3>::run(rows, x, w, bias, out);

        double best = 1e30;
        int best_index = -1;
        for (int i = 0; i < 5; ++i) if (ms[i] > 0 && ms[i] < best) { best = ms[i]; best_index = i; }

        // Agreement with cuBLASLt's own result, for whichever tile ran last.
        std::vector<__nv_bfloat16> a(size_t(rows) * HIDDEN), b(size_t(rows) * HIDDEN);
        CHECK(cudaMemcpy(a.data(), out, a.size() * 2, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(b.data(), ref, b.size() * 2, cudaMemcpyDeviceToHost));
        double worst = 0.0;
        for (size_t i = 0; i < a.size(); ++i)
            worst = std::max(worst, double(fabsf(float(a[i]) - float(b[i]))));

        printf("%8d %12.4f", rows, lt);
        for (int i = 0; i < 5; ++i)
        {
            if (ms[i] > 0) printf(" %10.4f", ms[i]);
            else           printf(" %10s", ms[i] == -1.0 ? "n/a" : "fail");
        }
        if (best_index >= 0)
            printf("   %.4f %s %.3fx  max|diff| %.3g\n", best, names[best_index], lt / best, worst);
        else
            printf("   none usable\n");
    }

    cudaFree(bias); cudaFree(ref); cudaFree(out); cudaFree(x); cudaFree(w); cudaFree(workspace);
    cublasLtDestroy(handle);
    return 0;
}

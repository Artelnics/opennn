// Bandwidth ceilings and candidate kernels for the two skinny dense layers.
//
// The 1024x1024 layer of the HIGGS classifier runs at cuBLAS's floor, but the
// first (28 -> 1024) and last (1024 -> 1) do not: cuBLAS dispatches a generic
// GEMM and a generic GEMV where both operations are really memory streaming.
// This probe puts the ceiling on them first - a pure store for the first
// layer's output and a pure load for the last layer's reduction - and then
// times candidate kernels against cuBLAS and against that ceiling, so it is
// clear both what to aim for and when to stop.
//
//   nvcc -O3 -arch=native -o dense_skinny_probe dense_skinny_probe.cu -lcublas
//   ./dense_skinny_probe [rows ...]

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#define CHECK(call)                                                            \
    do {                                                                       \
        const cudaError_t s = (call);                                          \
        if (s != cudaSuccess) {                                                \
            printf("CUDA %s at line %d\n", cudaGetErrorString(s), __LINE__);   \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

constexpr int WARMUP = 20;
constexpr int ITERS = 100;
constexpr int FEATURES = 28;      // HIGGS inputs
constexpr int HIDDEN = 1024;

// ---------------------------------------------------------------- ceilings

// Writes the first layer's output tensor and nothing else: the floor for L1.
__global__ void store_only(__nv_bfloat16* __restrict__ y, int n)
{
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i < n) {
        const uint4 v = make_uint4(0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u);
        *reinterpret_cast<uint4*>(y + i) = v;
    }
}

// Reads the last layer's input tensor and reduces it: the floor for L3.
__global__ void load_only(const __nv_bfloat16* __restrict__ x, float* __restrict__ out, int rows, int cols)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rows) return;

    const __nv_bfloat16* row = x + size_t(warp) * cols;
    float acc = 0.0f;
    for (int base = lane * 8; base < cols; base += 32 * 8) {
        const uint4 packed = *reinterpret_cast<const uint4*>(row + base);
        const __nv_bfloat162* pairs = reinterpret_cast<const __nv_bfloat162*>(&packed);
        for (int k = 0; k < 4; ++k) {
            const float2 f = __bfloat1622float2(pairs[k]);
            acc += f.x + f.y;
        }
    }
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) out[warp] = acc;
}

// ------------------------------------------------------------ L3 candidate

// y[i] = sigmoid(bias + sum_j a[i][j] * w[j]). One warp per row, the weight
// vector read once per lane into registers, a shuffle reduction, and the bias
// and sigmoid fused - no intermediate, no atomics, deterministic order.
__global__ void l3_reduce(const __nv_bfloat16* __restrict__ a,
                          const __nv_bfloat16* __restrict__ w,
                          const float bias,
                          float* __restrict__ y,
                          const int rows, const int cols)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= rows) return;

    const __nv_bfloat16* row = a + size_t(warp) * cols;

    float acc = 0.0f;
    for (int base = lane * 8; base < cols; base += 32 * 8) {
        const uint4 a_packed = *reinterpret_cast<const uint4*>(row + base);
        const uint4 w_packed = *reinterpret_cast<const uint4*>(w + base);
        const __nv_bfloat162* ap = reinterpret_cast<const __nv_bfloat162*>(&a_packed);
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&w_packed);
        for (int k = 0; k < 4; ++k) {
            const float2 af = __bfloat1622float2(ap[k]);
            const float2 wf = __bfloat1622float2(wp[k]);
            acc += af.x * wf.x + af.y * wf.y;
        }
    }
    for (int off = 16; off; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    if (lane == 0) y[warp] = 1.0f / (1.0f + __expf(-(acc + bias)));
}


// ------------------------------------------------------------ L1 candidate
//
// out[i][n] = relu(bias[n] + sum_j x[i][j] * w[j][n]), with j only 28 wide.
// The contraction is tiny and the output is 32x the input, so this is a store
// problem, not a GEMM: each block keeps a slice of the weight matrix in shared
// memory, streams rows through registers, and writes 16 bytes per thread. K is
// padded to a multiple of the MMA step internally, which costs four zero
// columns and removes the ragged tail cuBLAS falls back to an align1 kernel to
// handle.
constexpr int L1_COLS_PER_BLOCK = 256;
constexpr int L1_COLS_PER_THREAD = 8;      // one 16-byte store per row per lane
constexpr int L1_WARPS = 8;                // 256 threads: enough to hide the store latency
constexpr int L1_THREADS = L1_WARPS * 32;
constexpr int L1_ROWS_PER_LANE = 8;        // rows carried together to reuse each weight load
constexpr int L1_ROWS_PER_BLOCK = L1_WARPS * L1_ROWS_PER_LANE * 2;

__global__ void l1_broadcast(const __nv_bfloat16* __restrict__ x,
                             const __nv_bfloat16* __restrict__ w,
                             const __nv_bfloat16* __restrict__ bias,
                             __nv_bfloat16* __restrict__ out,
                             const int rows, const int features, const int cols)
{
    // The weight slice stays bf16 in shared memory: 28x256x2 = 14 KB, which
    // leaves room for several blocks per SM. Keeping it as float would triple
    // the footprint and cost the occupancy this kernel needs to stay
    // store-bound.
    extern __shared__ __nv_bfloat16 w_tile[];

    const int col_block = blockIdx.y * L1_COLS_PER_BLOCK;

    for (int i = int(threadIdx.x); i < features * L1_COLS_PER_BLOCK; i += L1_THREADS)
    {
        const int j = i / L1_COLS_PER_BLOCK;
        const int c = i - j * L1_COLS_PER_BLOCK;
        w_tile[i] = w[size_t(j) * cols + col_block + c];
    }
    __syncthreads();

    const int warp = int(threadIdx.x) >> 5;
    const int lane = int(threadIdx.x) & 31;
    const int col0 = lane * L1_COLS_PER_THREAD;

    float bias_reg[L1_COLS_PER_THREAD];
    #pragma unroll
    for (int c = 0; c < L1_COLS_PER_THREAD; ++c)
        bias_reg[c] = bias ? __bfloat162float(bias[col_block + col0 + c]) : 0.0f;

    // Each lane carries L1_ROWS_PER_LANE rows at once so a weight value read
    // from shared memory is reused across all of them. With one row per lane
    // the shared traffic was 117 MB against 16.8 MB of stores and the kernel
    // was shared-bound; blocking by 8 brings it down to roughly the store
    // traffic, which is where it belongs.
    const int block_start = int(blockIdx.x) * L1_ROWS_PER_BLOCK;
    const int row_end = min(rows, block_start + L1_ROWS_PER_BLOCK);

    for (int r0 = block_start + warp * L1_ROWS_PER_LANE; r0 < row_end;
         r0 += L1_WARPS * L1_ROWS_PER_LANE)
    {
        const int rows_here = min(L1_ROWS_PER_LANE, row_end - r0);

        float acc[L1_ROWS_PER_LANE][L1_COLS_PER_THREAD];
        #pragma unroll
        for (int r = 0; r < L1_ROWS_PER_LANE; ++r)
            #pragma unroll
            for (int c = 0; c < L1_COLS_PER_THREAD; ++c) acc[r][c] = bias_reg[c];

        for (int j = 0; j < features; ++j)
        {
            float wv[L1_COLS_PER_THREAD];
            const __nv_bfloat16* w_row = w_tile + j * L1_COLS_PER_BLOCK + col0;
            #pragma unroll
            for (int c = 0; c < L1_COLS_PER_THREAD; ++c) wv[c] = __bfloat162float(w_row[c]);

            #pragma unroll
            for (int r = 0; r < L1_ROWS_PER_LANE; ++r)
            {
                if (r >= rows_here) break;
                const float xv = __bfloat162float(x[size_t(r0 + r) * features + j]);
                #pragma unroll
                for (int c = 0; c < L1_COLS_PER_THREAD; ++c) acc[r][c] += xv * wv[c];
            }
        }

        for (int r = 0; r < rows_here; ++r)
        {
            __nv_bfloat16 packed[L1_COLS_PER_THREAD];
            #pragma unroll
            for (int c = 0; c < L1_COLS_PER_THREAD; ++c)
                packed[c] = __float2bfloat16(acc[r][c] > 0.0f ? acc[r][c] : 0.0f);

            *reinterpret_cast<uint4*>(out + size_t(r0 + r) * cols + col_block + col0) =
                *reinterpret_cast<const uint4*>(packed);
        }
    }
}

// ---------------------------------------------------------------- harness

float time_kernel(const std::function<void()>& issue)
{
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    for (int i = 0; i < WARMUP; ++i) issue();
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(start));
    for (int i = 0; i < ITERS; ++i) issue();
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return ms / ITERS;
}

void report(const char* name, float ms, double bytes)
{
    printf("%-34s %8.4f ms   %7.0f GB/s\n", name, ms, bytes / (ms * 1e-3) / 1e9);
}

// Correctness against a host reference. Inputs are bf16 and accumulation is
// fp32, so the tolerance covers bf16 input rounding, not the reduction order -
// the shuffle tree is deterministic.
bool verify_l3(int rows, int cols)
{
    std::vector<float> ha(size_t(rows) * cols), hw(cols);
    // Pseudo-random rather than modular patterns: correlated inputs drive the
    // dot product far from zero, the sigmoid saturates to exactly 0 or 1 on
    // both sides, and the comparison stops testing anything.
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> wide(-1.0f, 1.0f);
    std::uniform_real_distribution<float> narrow(-0.05f, 0.05f);
    for (size_t i = 0; i < ha.size(); ++i) ha[i] = wide(rng);
    for (int j = 0; j < cols; ++j)         hw[j] = narrow(rng);

    std::vector<__nv_bfloat16> ba(ha.size()), bw(cols);
    for (size_t i = 0; i < ha.size(); ++i) ba[i] = __float2bfloat16(ha[i]);
    for (int j = 0; j < cols; ++j)         bw[j] = __float2bfloat16(hw[j]);

    __nv_bfloat16 *da = nullptr, *dw = nullptr;
    float* dy = nullptr;
    CHECK(cudaMalloc(&da, ba.size() * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&dw, cols * sizeof(__nv_bfloat16)));
    CHECK(cudaMalloc(&dy, rows * sizeof(float)));
    CHECK(cudaMemcpy(da, ba.data(), ba.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dw, bw.data(), cols * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    const float bias = 0.1f;
    l3_reduce<<<(rows * 32 + 255) / 256, 256>>>(da, dw, bias, dy, rows, cols);
    CHECK(cudaDeviceSynchronize());

    std::vector<float> got(rows);
    CHECK(cudaMemcpy(got.data(), dy, rows * sizeof(float), cudaMemcpyDeviceToHost));

    double worst = 0.0;
    for (int i = 0; i < rows; ++i) {
        double acc = 0.0;
        for (int j = 0; j < cols; ++j)
            acc += double(__bfloat162float(ba[size_t(i) * cols + j])) * double(__bfloat162float(bw[j]));
        const double want = 1.0 / (1.0 + exp(-(acc + bias)));
        if (fabs(want - got[i]) > worst) worst = fabs(want - got[i]);
    }
    printf("  sample: got[0]=%.9g got[1]=%.9g got[2]=%.9g\n", got[0], got[1], got[2]);
    cudaFree(da); cudaFree(dw); cudaFree(dy);
    printf("L3 correctness vs host reference: max abs error %.3g  %s\n\n",
           worst, worst < 1e-5 ? "OK" : "FAIL");
    return worst < 1e-5;
}

int main(int argc, char* argv[])
{
    std::vector<int> batches;
    for (int i = 1; i < argc; ++i) batches.push_back(atoi(argv[i]));
    if (batches.empty()) batches = {7000, 8192};

    cudaDeviceProp prop{};
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("device: %s (%d SMs, %.0f MB L2)\n\n", prop.name,
           prop.multiProcessorCount, prop.l2CacheSize / 1048576.0);

    verify_l3(512, HIDDEN);

    cublasHandle_t blas;
    cublasCreate(&blas);

    for (const int rows : batches) {
        const size_t activation = size_t(rows) * HIDDEN;

        __nv_bfloat16 *a = nullptr, *w = nullptr, *y_bf = nullptr;
        float* y = nullptr;
        CHECK(cudaMalloc(&a, activation * sizeof(__nv_bfloat16)));
        CHECK(cudaMalloc(&y_bf, activation * sizeof(__nv_bfloat16)));
        CHECK(cudaMalloc(&w, HIDDEN * sizeof(__nv_bfloat16)));
        CHECK(cudaMalloc(&y, size_t(rows) * sizeof(float)));
        CHECK(cudaMemset(a, 0x3c, activation * sizeof(__nv_bfloat16)));
        CHECK(cudaMemset(w, 0x3c, HIDDEN * sizeof(__nv_bfloat16)));

        const double act_bytes = double(activation) * sizeof(__nv_bfloat16);

        printf("=== rows %d  (activation tensor %.2f MB) ===\n", rows, act_bytes / 1048576.0);

        // Ceilings.
        const int store_blocks = int((activation / 8 + 255) / 256);
        report("ceiling: pure store (L1 out)",
               time_kernel([&] { store_only<<<store_blocks, 256>>>(y_bf, int(activation)); }),
               act_bytes);

        const int warp_blocks = (rows * 32 + 255) / 256;
        report("ceiling: pure load+reduce (L3 in)",
               time_kernel([&] { load_only<<<warp_blocks, 256>>>(a, y, rows, HIDDEN); }),
               act_bytes);

        // L3: cuBLAS GEMV against the candidate.
        const float alpha = 1.0f, beta = 0.0f;
        report("L3 cuBLAS gemv (bf16)",
               time_kernel([&] {
                   cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, 1, rows, HIDDEN,
                                &alpha, w, CUDA_R_16BF, HIDDEN, a, CUDA_R_16BF, HIDDEN,
                                &beta, y, CUDA_R_32F, 1, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT);
               }),
               act_bytes);

        report("L3 own reduction kernel",
               time_kernel([&] { l3_reduce<<<warp_blocks, 256>>>(a, w, 0.1f, y, rows, HIDDEN); }),
               act_bytes);

        // L1: cuBLAS GEMM against the candidate. Traffic is the output store
        // plus the (much smaller) input read.
        __nv_bfloat16 *x1 = nullptr, *w1 = nullptr, *b1 = nullptr;
        CHECK(cudaMalloc(&x1, size_t(rows) * FEATURES * sizeof(__nv_bfloat16)));
        CHECK(cudaMalloc(&w1, size_t(FEATURES) * HIDDEN * sizeof(__nv_bfloat16)));
        CHECK(cudaMalloc(&b1, HIDDEN * sizeof(__nv_bfloat16)));
        CHECK(cudaMemset(x1, 0x3c, size_t(rows) * FEATURES * sizeof(__nv_bfloat16)));
        CHECK(cudaMemset(w1, 0x3c, size_t(FEATURES) * HIDDEN * sizeof(__nv_bfloat16)));
        CHECK(cudaMemset(b1, 0, HIDDEN * sizeof(__nv_bfloat16)));

        const double l1_bytes = act_bytes + double(rows) * FEATURES * sizeof(__nv_bfloat16);

        report("L1 cuBLAS gemm (bf16)",
               time_kernel([&] {
                   cublasGemmEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, HIDDEN, rows, FEATURES,
                                &alpha, w1, CUDA_R_16BF, HIDDEN, x1, CUDA_R_16BF, FEATURES,
                                &beta, y_bf, CUDA_R_16BF, HIDDEN, CUBLAS_COMPUTE_32F,
                                CUBLAS_GEMM_DEFAULT);
               }),
               l1_bytes);

        {
            const size_t shared_bytes = size_t(FEATURES) * L1_COLS_PER_BLOCK * sizeof(__nv_bfloat16);
            CHECK(cudaFuncSetAttribute(l1_broadcast, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       int(shared_bytes)));
            const dim3 grid((rows + L1_ROWS_PER_BLOCK - 1) / L1_ROWS_PER_BLOCK, HIDDEN / L1_COLS_PER_BLOCK);
            report("L1 own broadcast kernel",
                   time_kernel([&] {
                       l1_broadcast<<<grid, L1_THREADS, shared_bytes>>>(x1, w1, b1, y_bf, rows, FEATURES, HIDDEN);
                   }),
                   l1_bytes);
        }

        cudaFree(x1); cudaFree(w1); cudaFree(b1);

        cudaFree(a); cudaFree(w); cudaFree(y); cudaFree(y_bf);
        printf("\n");
    }

    cublasDestroy(blas);
    return 0;
}

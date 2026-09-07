//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S M A L L   K   L I N E A R   K E R N E L
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// A first layer over tabular data contracts a handful of features into
// hundreds or thousands of outputs: 8192 x 28 -> 1024 in bf16 does 0.5
// GFLOP and writes 16.8 MB. The kernel is the output write, and what
// matters is keeping every store a full sector while the tensor cores idle.
// cuBLASLt cannot promise better than two-element alignment on a 28-wide
// input and picks an `align2` kernel that reads it in pieces: 22 us for that
// shape on an RTX 5070 Ti, against a 10 us memset of the output. This kernel
// runs it in 13 us.
//
// Each warp owns 16 rows of a 64-row by 64-column tile and runs the
// contraction, padded to 32, as sixteen mma.sync m16n8k16 instructions.
// The A fragments come straight from global memory as 4-byte loads: the
// fragment layout wants pairs of consecutive k for one row, which is what a
// row-major input holds, so the 28-wide rows never touch shared memory.
// The weights tile is loaded once per block and held in registers. The
// accumulators are packed to bf16 and staged through shared memory so that
// every store instruction writes four whole 128-byte rows.

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/core/cuda/kernel_small_k_linear.cuh"
#include "opennn/core/device_backend.h"

#include <algorithm>
#include <limits>

namespace opennn
{
bool env_flag_enabled(const char*, bool default_value) noexcept;
}

namespace
{

constexpr int tile_rows = 64;
constexpr int tile_columns = 64;
constexpr int padded_contraction = 32;
constexpr int threads = 128;
constexpr int warps = threads / 32;

// Weights tile row stride: 144 bytes keeps the uint4 fills aligned and the
// fragment gathers (k = 2t, n = g) conflict-free.
constexpr int weights_stride = tile_columns + 8;

// Staged output row stride in 32-bit words: 36 spreads the scattered
// fragment stores over all 32 banks and keeps the row reads 16-byte aligned.
constexpr int stage_stride = 36;

__device__ __forceinline__ uint32_t pack_bf16x2(float low, float high)
{
    const __nv_bfloat162 pair = __floats2bfloat162_rn(low, high);
    return *reinterpret_cast<const uint32_t*>(&pair);
}

__device__ __forceinline__ void mma_bf16_16x8x16(float* c, const uint32_t* a, uint32_t b0, uint32_t b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void)c; (void)a; (void)b0; (void)b1;
#endif
}

template<typename BiasT, bool Relu>
__global__ void __launch_bounds__(threads)
small_k_linear_kernel(const int rows, const int contraction, const int columns, const int row_tiles,
                      const __nv_bfloat16* __restrict__ input,
                      const __nv_bfloat16* __restrict__ weights,
                      const BiasT* __restrict__ bias,
                      __nv_bfloat16* __restrict__ output)
{
    __shared__ __align__(16) __nv_bfloat16 weights_tile[padded_contraction * weights_stride];
    __shared__ __align__(16) uint32_t stage[warps * 16 * stage_stride];
    __shared__ float bias_tile[tile_columns];

    const int column_tiles = columns / tile_columns;
    const int column_0 = int(blockIdx.x % column_tiles) * tile_columns;
    const int row_tile_stride = int(gridDim.x) / column_tiles;

    const int thread = int(threadIdx.x);
    const int warp = thread >> 5;
    const int lane = thread & 31;
    const int group = lane >> 2;      // mma fragment row (and n) index
    const int member = lane & 3;      // mma fragment k (and n pair) index

    // Weights tile: contraction x 64, rows past the contraction zero.
    {
        const int k = thread >> 2;
        const int c = (thread & 3) * 16;
        uint4 values[2] = {make_uint4(0u, 0u, 0u, 0u), make_uint4(0u, 0u, 0u, 0u)};
        if (k < contraction)
        {
            const uint4* source = reinterpret_cast<const uint4*>(weights + size_t(k) * columns + column_0 + c);
            values[0] = source[0];
            values[1] = source[1];
        }
        *reinterpret_cast<uint4*>(&weights_tile[k * weights_stride + c]) = values[0];
        *reinterpret_cast<uint4*>(&weights_tile[k * weights_stride + c + 8]) = values[1];

        if (thread < tile_columns)
            bias_tile[thread] = bias ? float(bias[column_0 + thread]) : 0.0f;
    }
    __syncthreads();

    // B fragments for 8 column groups x 2 contraction steps: {k, k+1} at
    // column g, and {k+8, k+9}, with k = 16 step + 2 member.
    uint32_t b[2][8][2];
    {
        const uint16_t* tile = reinterpret_cast<const uint16_t*>(weights_tile);
        #pragma unroll
        for (int step = 0; step < 2; ++step)
            #pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                const int k = step * 16 + 2 * member;
                const int n = j * 8 + group;
                b[step][j][0] = uint32_t(tile[k * weights_stride + n])
                              | (uint32_t(tile[(k + 1) * weights_stride + n]) << 16);
                b[step][j][1] = uint32_t(tile[(k + 8) * weights_stride + n])
                              | (uint32_t(tile[(k + 9) * weights_stride + n]) << 16);
            }
    }

    float bias_pair[8][2];
    #pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        bias_pair[j][0] = bias_tile[8 * j + 2 * member];
        bias_pair[j][1] = bias_tile[8 * j + 2 * member + 1];
    }

    // A fragments straight from global memory: row g and row g+8 of the
    // warp's 16 rows, consecutive k pairs, zero past the contraction.
    const auto load_a = [&](int row_tile, uint32_t (&a)[2][4], int& row_g)
    {
        row_g = row_tile * tile_rows + warp * 16 + group;
        const int row_h = row_g + 8;
        const __nv_bfloat16* a_g = input + size_t(row_g) * contraction;
        const __nv_bfloat16* a_h = input + size_t(row_h) * contraction;
        #pragma unroll
        for (int step = 0; step < 2; ++step)
        {
            const int k0 = step * 16 + 2 * member;
            const int k1 = k0 + 8;
            a[step][0] = (row_g < rows && k0 < contraction) ? *reinterpret_cast<const uint32_t*>(a_g + k0) : 0u;
            a[step][1] = (row_h < rows && k0 < contraction) ? *reinterpret_cast<const uint32_t*>(a_h + k0) : 0u;
            a[step][2] = (row_g < rows && k1 < contraction) ? *reinterpret_cast<const uint32_t*>(a_g + k1) : 0u;
            a[step][3] = (row_h < rows && k1 < contraction) ? *reinterpret_cast<const uint32_t*>(a_h + k1) : 0u;
        }
    };

    int row_tile = int(blockIdx.x) / column_tiles;
    uint32_t a_current[2][4];
    int row_g = 0;
    if (row_tile < row_tiles) load_a(row_tile, a_current, row_g);

    uint32_t* warp_stage = stage + warp * (16 * stage_stride);

    for (; row_tile < row_tiles; row_tile += row_tile_stride)
    {
        // The next tile's loads overlap this tile's epilogue.
        uint32_t a_next[2][4];
        int row_g_next = 0;
        if (row_tile + row_tile_stride < row_tiles) load_a(row_tile + row_tile_stride, a_next, row_g_next);

        float c[8][4];
        #pragma unroll
        for (int j = 0; j < 8; ++j) c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.0f;

        #pragma unroll
        for (int step = 0; step < 2; ++step)
            #pragma unroll
            for (int j = 0; j < 8; ++j) mma_bf16_16x8x16(c[j], a_current[step], b[step][j][0], b[step][j][1]);

        // c[j] holds (row g, columns 8j+2m, +1) and (row g+8, the same columns).
        #pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            float y0 = c[j][0] + bias_pair[j][0];
            float y1 = c[j][1] + bias_pair[j][1];
            float y2 = c[j][2] + bias_pair[j][0];
            float y3 = c[j][3] + bias_pair[j][1];
            if (Relu)
            {
                y0 = fmaxf(y0, 0.0f);
                y1 = fmaxf(y1, 0.0f);
                y2 = fmaxf(y2, 0.0f);
                y3 = fmaxf(y3, 0.0f);
            }
            warp_stage[group * stage_stride + 4 * j + member] = pack_bf16x2(y0, y1);
            warp_stage[(group + 8) * stage_stride + 4 * j + member] = pack_bf16x2(y2, y3);
        }
        __syncwarp();

        // Read back by rows: eight lanes per 128-byte row, four rows per store.
        {
            const int stage_row = lane >> 3;
            const int stage_word = (lane & 7) * 4;
            const int tile_row_0 = row_g - group;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int r = 4 * i + stage_row;
                const int row = tile_row_0 + r;
                const uint4 values = *reinterpret_cast<const uint4*>(warp_stage + r * stage_stride + stage_word);
                if (row < rows)
                    *reinterpret_cast<uint4*>(output + size_t(row) * columns + column_0 + 2 * stage_word) = values;
            }
        }
        __syncwarp();

        #pragma unroll
        for (int step = 0; step < 2; ++step)
            #pragma unroll
            for (int i = 0; i < 4; ++i) a_current[step][i] = a_next[step][i];
        row_g = row_g_next;
    }
}

template<typename BiasT, bool Relu>
int resident_blocks()
{
    static const int blocks = []
    {
        int device = 0;
        int multiprocessors = 0;
        int per_multiprocessor = 0;
        if (cudaGetDevice(&device) != cudaSuccess
            || cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device) != cudaSuccess
            || cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_multiprocessor,
                                                             small_k_linear_kernel<BiasT, Relu>,
                                                             threads, 0) != cudaSuccess)
        {
            cudaGetLastError();
            return 0;
        }
        return multiprocessors * std::max(per_multiprocessor, 1);
    }();
    return blocks;
}

// Whole columns of tiles per block, so a block's weights tile is loaded once,
// and as few blocks as keep every SM full, so the tiles split evenly.
int grid_size(int row_tiles, int column_tiles, int resident)
{
    const long long tiles = (long long)row_tiles * column_tiles;
    const long long per_block = std::max(1LL, (tiles + resident - 1) / resident);
    long long blocks = (tiles + per_block - 1) / per_block;
    blocks = ((blocks + column_tiles - 1) / column_tiles) * column_tiles;
    return int(std::min(blocks, tiles));
}

template<typename BiasT, bool Relu>
bool launch(int rows, int contraction, int columns,
            const __nv_bfloat16* input, const __nv_bfloat16* weights, const BiasT* bias,
            __nv_bfloat16* output, cudaStream_t stream)
{
    const int resident = resident_blocks<BiasT, Relu>();
    if (resident == 0) return false;

    const int row_tiles = (rows + tile_rows - 1) / tile_rows;
    const int column_tiles = columns / tile_columns;
    const int grid = grid_size(row_tiles, column_tiles, resident);

    OPENNN_CUDA_LAUNCH((small_k_linear_kernel<BiasT, Relu><<<grid, threads, 0, stream>>>(
        rows, contraction, columns, row_tiles, input, weights, bias, output)));
    return true;
}

bool aligned(const void* pointer, size_t bytes)
{
    return reinterpret_cast<uintptr_t>(pointer) % bytes == 0;
}

}

bool small_k_linear_forward_cuda(Index rows, Index contraction, Index out_features,
                                 const void* input, const void* weights,
                                 const void* bias, bool bias_fp32,
                                 void* output, bool relu, cudaStream_t stream)
{
    static const bool enabled = opennn::env_flag_enabled("OPENNN_SMALL_K_LINEAR", true)
                                && opennn::device::cuda_compute_capability() >= 80;
    if (!enabled) return false;

    // The 4-byte A loads need an even contraction; the 16-byte weight and
    // output vectors need 64-column tiles and aligned rows.
    if (contraction < 2 || contraction > padded_contraction || contraction % 2 != 0) return false;
    if (out_features <= 0 || out_features % tile_columns != 0) return false;
    if (rows <= 0 || rows > Index(std::numeric_limits<int>::max())) return false;
    if (!aligned(input, 4) || !aligned(weights, 16) || !aligned(output, 16)) return false;
    if (bias && !aligned(bias, bias_fp32 ? 4 : 2)) return false;

    if (stream == nullptr) stream = opennn::device::get_compute_stream();

    const int rows_int = int(rows);
    const int contraction_int = int(contraction);
    const int columns = int(out_features);
    const __nv_bfloat16* a = static_cast<const __nv_bfloat16*>(input);
    const __nv_bfloat16* w = static_cast<const __nv_bfloat16*>(weights);
    __nv_bfloat16* y = static_cast<__nv_bfloat16*>(output);

    if (bias_fp32)
    {
        const float* b = static_cast<const float*>(bias);
        return relu ? launch<float, true>(rows_int, contraction_int, columns, a, w, b, y, stream)
                    : launch<float, false>(rows_int, contraction_int, columns, a, w, b, y, stream);
    }

    const __nv_bfloat16* b = static_cast<const __nv_bfloat16*>(bias);
    return relu ? launch<__nv_bfloat16, true>(rows_int, contraction_int, columns, a, w, b, y, stream)
                : launch<__nv_bfloat16, false>(rows_int, contraction_int, columns, a, w, b, y, stream);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

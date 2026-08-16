// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/core/cuda/kernel_pooling.cuh"

namespace
{

struct PoolGeometry
{
    int height, width, channels;
    int out_height, out_width;
    int pool_height, pool_width;
    int stride_h, stride_w;
    int pad_h, pad_w;
};

// One thread per (n, ho, wo) and VEC adjacent channels; channel groups are the
// fastest index, so a warp writes consecutive channels of one output row.
template<typename T, int VEC>
__global__ void max_pooling_forward_kernel(const Index groups, const PoolGeometry g,
                                           const T* __restrict__ x,
                                           T* __restrict__ y,
                                           uint8_t* __restrict__ mask)
{
    const int channel_groups = g.channels / VEC;
    for (Index gi = Index(blockIdx.x) * blockDim.x + threadIdx.x; gi < groups;
         gi += Index(gridDim.x) * blockDim.x)
    {
        const int c0 = int(gi % channel_groups) * VEC;
        Index rest = gi / channel_groups;
        const int wo = int(rest % g.out_width); rest /= g.out_width;
        const int ho = int(rest % g.out_height);
        const Index n = rest / g.out_height;

        float best[VEC];
        uint8_t arg[VEC];
        #pragma unroll
        for (int k = 0; k < VEC; ++k) { best[k] = -INFINITY; arg[k] = 0; }

        const int hi0 = ho * g.stride_h - g.pad_h;
        const int wi0 = wo * g.stride_w - g.pad_w;
        for (int i = 0; i < g.pool_height; ++i)
        {
            const int hi = hi0 + i;
            if (hi < 0 || hi >= g.height) continue;
            for (int j = 0; j < g.pool_width; ++j)
            {
                const int wi = wi0 + j;
                if (wi < 0 || wi >= g.width) continue;
                float v[VEC];
                VecIO<T, VEC>::load_float(x + ((n * g.height + hi) * Index(g.width) + wi) * g.channels + c0, v);
                const uint8_t position = uint8_t(i * g.pool_width + j);
                #pragma unroll
                for (int k = 0; k < VEC; ++k)
                    if (v[k] > best[k]) { best[k] = v[k]; arg[k] = position; }
            }
        }

        const Index o = ((n * g.out_height + ho) * Index(g.out_width) + wo) * g.channels + c0;
        VecIO<T, VEC>::store_float(y + o, best);
        if (mask) VecIO<uint8_t, VEC>::store(mask + o, arg);
    }
}

// One thread per (n, hi, wi) and VEC adjacent channels: gathers dY from the
// outputs whose window covers (hi, wi) and whose argmax lands on it. With
// pool 3 / stride 2 those are at most four outputs.
template<typename T, int VEC>
__global__ void max_pooling_backward_kernel(const Index groups, const PoolGeometry g,
                                            const T* __restrict__ dy,
                                            const uint8_t* __restrict__ mask,
                                            T* __restrict__ dx)
{
    const int channel_groups = g.channels / VEC;
    for (Index gi = Index(blockIdx.x) * blockDim.x + threadIdx.x; gi < groups;
         gi += Index(gridDim.x) * blockDim.x)
    {
        const int c0 = int(gi % channel_groups) * VEC;
        Index rest = gi / channel_groups;
        const int wi = int(rest % g.width); rest /= g.width;
        const int hi = int(rest % g.height);
        const Index n = rest / g.height;

        // Outputs whose window contains (hi, wi): ho * stride - pad <= hi <
        // ho * stride - pad + pool.
        const int ho_begin = max(0, (hi + g.pad_h - g.pool_height + g.stride_h) / g.stride_h);
        const int ho_end   = min(g.out_height - 1, (hi + g.pad_h) / g.stride_h);
        const int wo_begin = max(0, (wi + g.pad_w - g.pool_width + g.stride_w) / g.stride_w);
        const int wo_end   = min(g.out_width - 1, (wi + g.pad_w) / g.stride_w);

        float sum[VEC];
        #pragma unroll
        for (int k = 0; k < VEC; ++k) sum[k] = 0.0f;

        for (int ho = ho_begin; ho <= ho_end; ++ho)
        {
            const int i = hi - (ho * g.stride_h - g.pad_h);
            for (int wo = wo_begin; wo <= wo_end; ++wo)
            {
                const int j = wi - (wo * g.stride_w - g.pad_w);
                const uint8_t here = uint8_t(i * g.pool_width + j);
                const Index o = ((n * g.out_height + ho) * Index(g.out_width) + wo) * g.channels + c0;
                uint8_t arg[VEC];
                VecIO<uint8_t, VEC>::load(mask + o, arg);
                bool any = false;
                #pragma unroll
                for (int k = 0; k < VEC; ++k) any |= arg[k] == here;
                if (!any) continue;
                float v[VEC];
                VecIO<T, VEC>::load_float(dy + o, v);
                #pragma unroll
                for (int k = 0; k < VEC; ++k)
                    if (arg[k] == here) sum[k] += v[k];
            }
        }

        VecIO<T, VEC>::store_float(dx + ((n * g.height + hi) * Index(g.width) + wi) * g.channels + c0, sum);
    }
}

template<typename T>
int pooling_vector_width(Index channels)
{
    // 16-byte groups where the channel count allows it.
    const int wide = 16 / int(sizeof(T));
    return channels % wide == 0 ? wide : 1;
}

PoolGeometry make_geometry(Index height, Index width, Index channels, Index out_height, Index out_width,
                           int pool_height, int pool_width, int stride_h, int stride_w, int pad_h, int pad_w)
{
    return {checked_int(height), checked_int(width), checked_int(channels),
            checked_int(out_height), checked_int(out_width),
            pool_height, pool_width, stride_h, stride_w, pad_h, pad_w};
}

}

template<typename T>
void max_pooling_forward_cuda(const T* x, T* y, uint8_t* mask,
                              Index batch, Index height, Index width, Index channels,
                              Index out_height, Index out_width,
                              int pool_height, int pool_width,
                              int stride_h, int stride_w,
                              int pad_h, int pad_w)
{
    if (batch == 0 || channels == 0) return;
    if (pool_height * pool_width > 255)
        throw std::runtime_error("max_pooling_forward_cuda: pool window too large for a one-byte argmax.");
    const PoolGeometry g = make_geometry(height, width, channels, out_height, out_width,
                                         pool_height, pool_width, stride_h, stride_w, pad_h, pad_w);
    const int vec = pooling_vector_width<T>(channels);
    const Index groups = batch * out_height * out_width * (channels / vec);
    if (vec == 8)      launch_elementwise_strided(groups, max_pooling_forward_kernel<T, 8>, g, x, y, mask);
    else if (vec == 4) launch_elementwise_strided(groups, max_pooling_forward_kernel<T, 4>, g, x, y, mask);
    else               launch_elementwise_strided(groups, max_pooling_forward_kernel<T, 1>, g, x, y, mask);
}

template<typename T>
void max_pooling_backward_cuda(const T* dy, const uint8_t* mask, T* dx,
                               Index batch, Index height, Index width, Index channels,
                               Index out_height, Index out_width,
                               int pool_height, int pool_width,
                               int stride_h, int stride_w,
                               int pad_h, int pad_w)
{
    if (batch == 0 || channels == 0) return;
    const PoolGeometry g = make_geometry(height, width, channels, out_height, out_width,
                                         pool_height, pool_width, stride_h, stride_w, pad_h, pad_w);
    const int vec = pooling_vector_width<T>(channels);
    const Index groups = batch * height * width * (channels / vec);
    if (vec == 8)      launch_elementwise_strided(groups, max_pooling_backward_kernel<T, 8>, g, dy, mask, dx);
    else if (vec == 4) launch_elementwise_strided(groups, max_pooling_backward_kernel<T, 4>, g, dy, mask, dx);
    else               launch_elementwise_strided(groups, max_pooling_backward_kernel<T, 1>, g, dy, mask, dx);
}

#define INSTANTIATE(T) \
    template void max_pooling_forward_cuda<T>(const T*, T*, uint8_t*, Index, Index, Index, Index, Index, Index, int, int, int, int, int, int); \
    template void max_pooling_backward_cuda<T>(const T*, const uint8_t*, T*, Index, Index, Index, Index, Index, Index, int, int, int, int, int, int);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE

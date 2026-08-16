// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence Techniques, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/core/cuda/kernel_pooling.cuh"

namespace
{

// One thread per output pixel and VEC channels: the window maximum and its
// position.
template<typename T, int VEC>
__global__ void max_pooling_forward_kernel(const Index groups, const MaxPoolGeometry s,
                                           const T* __restrict__ x,
                                           T* __restrict__ y,
                                           uint8_t* __restrict__ mask)
{
    for (Index gi = Index(blockIdx.x) * blockDim.x + threadIdx.x; gi < groups;
         gi += Index(gridDim.x) * blockDim.x)
    {
        Index n; int ho, wo, c0;
        s.decompose(gi, s.out_height, s.out_width, VEC, n, ho, wo, c0);

        float best[VEC];
        uint8_t arg[VEC];
        #pragma unroll
        for (int k = 0; k < VEC; ++k) { best[k] = -INFINITY; arg[k] = 0; }

        for (int i = 0; i < s.pool_height; ++i)
        {
            const int hi = ho * s.stride_h - s.pad_h + i;
            if (hi < 0 || hi >= s.height) continue;
            for (int j = 0; j < s.pool_width; ++j)
            {
                const int wi = wo * s.stride_w - s.pad_w + j;
                if (wi < 0 || wi >= s.width) continue;
                float v[VEC];
                VecIO<T, VEC>::load_float(x + s.input_offset(n, hi, wi, c0), v);
                const uint8_t position = uint8_t(i * s.pool_width + j);
                #pragma unroll
                for (int k = 0; k < VEC; ++k)
                    if (v[k] > best[k]) { best[k] = v[k]; arg[k] = position; }
            }
        }

        const Index o = s.output_offset(n, ho, wo, c0);
        VecIO<T, VEC>::store_float(y + o, best);
        if (mask) VecIO<uint8_t, VEC>::store(mask + o, arg);
    }
}

// One thread per input pixel and VEC channels: gathers dY from the outputs
// whose window covers (hi, wi) - at most ceil(pool / stride)^2 of them - and
// whose argmax lands on it.
template<typename T, int VEC>
__global__ void max_pooling_backward_kernel(const Index groups, const MaxPoolGeometry s,
                                            const T* __restrict__ dy,
                                            const uint8_t* __restrict__ mask,
                                            T* __restrict__ dx)
{
    for (Index gi = Index(blockIdx.x) * blockDim.x + threadIdx.x; gi < groups;
         gi += Index(gridDim.x) * blockDim.x)
    {
        Index n; int hi, wi, c0;
        s.decompose(gi, s.height, s.width, VEC, n, hi, wi, c0);

        // ho * stride - pad <= hi < ho * stride - pad + pool, clipped to the output.
        const int ho_begin = max(0, (hi + s.pad_h - s.pool_height + s.stride_h) / s.stride_h);
        const int ho_end   = min(s.out_height - 1, (hi + s.pad_h) / s.stride_h);
        const int wo_begin = max(0, (wi + s.pad_w - s.pool_width + s.stride_w) / s.stride_w);
        const int wo_end   = min(s.out_width - 1, (wi + s.pad_w) / s.stride_w);

        float sum[VEC];
        #pragma unroll
        for (int k = 0; k < VEC; ++k) sum[k] = 0.0f;

        for (int ho = ho_begin; ho <= ho_end; ++ho)
        {
            const int i = hi - (ho * s.stride_h - s.pad_h);
            for (int wo = wo_begin; wo <= wo_end; ++wo)
            {
                const int j = wi - (wo * s.stride_w - s.pad_w);
                const uint8_t here = uint8_t(i * s.pool_width + j);
                const Index o = s.output_offset(n, ho, wo, c0);
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

        VecIO<T, VEC>::store_float(dx + s.input_offset(n, hi, wi, c0), sum);
    }
}

// 16-byte channel groups where the channel count allows it, else scalar.
template<typename T, typename F>
void with_vector_width(Index channels, F&& launch)
{
    constexpr int wide = 16 / int(sizeof(T));
    if (channels % wide != 0) launch.template operator()<1>();
    else if constexpr (wide == 8) launch.template operator()<8>();
    else launch.template operator()<4>();
}

}

template<typename T>
void max_pooling_forward_cuda(const T* x, T* y, uint8_t* mask, const MaxPoolGeometry& g)
{
    if (g.batch == 0 || g.channels == 0) return;
    if (g.pool_height * g.pool_width > 255)
        throw std::runtime_error("max_pooling_forward_cuda: pool window too large for a one-byte argmax.");
    with_vector_width<T>(g.channels, [&]<int VEC>()
    {
        launch_elementwise_strided(g.batch * g.out_height * g.out_width * (g.channels / VEC),
                                   max_pooling_forward_kernel<T, VEC>, g, x, y, mask);
    });
}

template<typename T>
void max_pooling_backward_cuda(const T* dy, const uint8_t* mask, T* dx, const MaxPoolGeometry& g)
{
    if (g.batch == 0 || g.channels == 0) return;
    with_vector_width<T>(g.channels, [&]<int VEC>()
    {
        launch_elementwise_strided(g.batch * g.height * g.width * (g.channels / VEC),
                                   max_pooling_backward_kernel<T, VEC>, g, dy, mask, dx);
    });
}

#define INSTANTIATE(T) \
    template void max_pooling_forward_cuda<T>(const T*, T*, uint8_t*, const MaxPoolGeometry&); \
    template void max_pooling_backward_cuda<T>(const T*, const uint8_t*, T*, const MaxPoolGeometry&);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE

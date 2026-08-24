//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S C A L I N G   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cfloat>

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/neural_network/layers/kernel_scaling.cuh"

template<typename TIn, typename TOut>
__global__ void clamping_kernel(const int n, const int features,
                                const TIn* __restrict__ input,
                                const float* __restrict__ lower,
                                const float* __restrict__ upper,
                                TOut* __restrict__ output)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const int f = i % features;
        const float x = static_cast<float>(input[i]);
        output[i] = static_cast<TOut>(fminf(fmaxf(x, lower[f]), upper[f]));
    }
}

template<typename TIn, typename TOut>
void clamping_cuda(const Index n, const int features,
                   const TIn* input, const float* lower, const float* upper,
                   TOut* output)
{
    launch_elementwise_strided(n, clamping_kernel<TIn, TOut>, features, input, lower, upper, output);
}

template<typename TIn, typename TOut, bool Inverse>
__global__ void scale_kernel(const int n, const int features,
                             const TIn* __restrict__ input,
                             const float* __restrict__ minimums,
                             const float* __restrict__ maximums,
                             const float* __restrict__ means,
                             const float* __restrict__ stds,
                             const float* __restrict__ scalers,
                             const float min_range, const float max_range,
                             TOut* __restrict__ output)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const int f = i % features;
        const int code = static_cast<int>(scalers[f]);
        const float x = static_cast<float>(input[i]);
        float y = x;

        switch (code)
        {
        case 1:
            if constexpr (Inverse)
                y = (max_range - min_range < FLT_EPSILON)
                    ? minimums[f]
                    : (x - min_range) / (max_range - min_range)
                        * (maximums[f] - minimums[f]) + minimums[f];
            else
            {
                const float range = maximums[f] - minimums[f];
                y = (range < FLT_EPSILON) ? 0.0f
                  : (x - minimums[f]) / range * (max_range - min_range) + min_range;
            }
            break;
        case 2:
            if constexpr (Inverse)
                y = means[f] + x * stds[f];
            else
                y = (stds[f] > FLT_EPSILON) ? (x - means[f]) / stds[f] : 0.0f;
            break;
        case 3:
            if constexpr (Inverse)
                y = (stds[f] > FLT_EPSILON) ? x * stds[f] : means[f];
            else
                y = (stds[f] > FLT_EPSILON) ? x / stds[f] : 0.0f;
            break;
        case 4:
            if constexpr (Inverse)
                y = expf(x);
            else
                y = logf(fmaxf(x, FLT_EPSILON));
            break;
        case 5:
            if constexpr (Inverse)
                y = x * 255.0f;
            else
                y = x / 255.0f;
            break;
        default:
            break;
        }

        output[i] = static_cast<TOut>(y);
    }
}

template<typename TIn, typename TOut>
void scale_cuda(const Index n, const int features,
                const TIn* input,
                const float* minimums, const float* maximums,
                const float* means, const float* stds,
                const float* scalers,
                const float min_range, const float max_range,
                TOut* output,
                const bool inverse)
{
    if (inverse)
        launch_elementwise_strided(n, scale_kernel<TIn, TOut, true>, features,
                                   input, minimums, maximums, means, stds, scalers,
                                   min_range, max_range, output);
    else
        launch_elementwise_strided(n, scale_kernel<TIn, TOut, false>, features,
                                   input, minimums, maximums, means, stds, scalers,
                                   min_range, max_range, output);
}

#define INSTANTIATE(TIn, TOut) \
    template void clamping_cuda<TIn, TOut>(const Index, const int, const TIn*, const float*, const float*, TOut*); \
    template void scale_cuda<TIn, TOut>(const Index, const int, const TIn*, const float*, const float*, const float*, const float*, const float*, float, float, TOut*, bool);

OPENNN_INSTANTIATE_FLOAT_BF16_2(INSTANTIATE)
#undef INSTANTIATE

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

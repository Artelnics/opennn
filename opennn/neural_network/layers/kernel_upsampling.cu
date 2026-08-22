//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   U P S A M P L I N G   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/neural_network/layers/kernel_upsampling.cuh"

__global__ void upsampling_forward_kernel(
    const int n,
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int in_h, const int in_w,
    const int out_h, const int out_w,
    const int channels, const int scale)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        Index b; int oh, ow, c;
        nhwc_decompose(i, channels, out_w, out_h, b, oh, ow, c);

        const int iw = ow / scale;
        const int ih = oh / scale;
        dst[i] = src[((b * in_h + ih) * in_w + iw) * channels + c];
    }
}

__global__ void upsampling_backward_kernel(
    const int n,
    const float* __restrict__ out_delta,
    float* __restrict__ in_delta,
    const int in_h, const int in_w,
    const int out_h, const int out_w,
    const int channels, const int scale)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        Index b; int ih, iw, c;
        nhwc_decompose(i, channels, in_w, in_h, b, ih, iw, c);

        float acc = 0.0f;
        for (int dh = 0; dh < scale; ++dh)
            for (int dw = 0; dw < scale; ++dw)
            {
                const int oh = ih * scale + dh;
                const int ow = iw * scale + dw;
                acc += out_delta[((b * out_h + oh) * out_w + ow) * channels + c];
            }
        in_delta[i] = acc;
    }
}

void upsampling_forward_cuda(const int batch, const int in_h, const int in_w, const int channels, const int scale,
                             const float* src, float* dst)
{
    // Widened before multiplying, then checked: the product is the whole
    // upsampled activation and overflowed int well before the tensors did.
    const int n = checked_int(Index(batch) * Index(in_h * scale)
                            * Index(in_w * scale) * Index(channels));
    launch_elementwise_strided(n, upsampling_forward_kernel,
                       src, dst, in_h, in_w, in_h * scale, in_w * scale, channels, scale);
}

void upsampling_backward_cuda(const int batch, const int in_h, const int in_w, const int channels, const int scale,
                              const float* out_delta, float* in_delta)
{
    const int n = checked_int(Index(batch) * Index(in_h) * Index(in_w) * Index(channels));
    // No pre-zeroing: the kernel assigns in_delta[i] for every i below n.
    launch_elementwise_strided(n, upsampling_backward_kernel,
                       out_delta, in_delta, in_h, in_w, in_h * scale, in_w * scale, channels, scale);
}

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

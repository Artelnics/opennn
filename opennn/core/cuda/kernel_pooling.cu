//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L I N G   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// pooling, first-token reduction, upsampling and concatenation

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/core/cuda/kernel_pooling.cuh"

struct PoolingScratch
{
    void* data = nullptr;
    Index bytes = 0;

    float* ensure(Index floats_needed)
    {
        const Index new_bytes = floats_needed * Index(sizeof(float));
        if (new_bytes <= bytes) return static_cast<float*>(data);

        if (data) opennn::device::deallocate(opennn::Device::CUDA, data, bytes);
        data = opennn::device::allocate(opennn::Device::CUDA, new_bytes);
        bytes = new_bytes;
        return static_cast<float*>(data);
    }
};

template<typename T>
__global__ void max_pooling_3d_forward_kernel(const int n, const T* __restrict__ in, T* __restrict__ out, float* __restrict__ indices, const int S, const int F)
{
    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += Index(blockDim.x) * gridDim.x)
    {
        const int f = idx % F;
        const int b = idx / F;

        float max_val = -1e20f;
        int max_index = 0;

        for (int s = 0; s < S; ++s)
        {
            const float val = static_cast<float>(in[(int64_t(b) * S + s) * F + f]);
            if (val > max_val) { max_val = val; max_index = s; }
        }

        out[idx] = static_cast<T>(max_val);
        if (indices != nullptr) indices[idx] = static_cast<float>(max_index);
    }
}

template<typename T>
void max_pooling_3d_forward_cuda(const Index n, const T* in, T* out, float* indices, const int S, const int F)
{
    launch_elementwise_strided(n, max_pooling_3d_forward_kernel<T>, in, out, indices, S, F);
}

template<typename T>
__global__ void max_pooling_3d_backward_kernel(const int n, const T* __restrict__ delta, T* __restrict__ in_gradient, const float* __restrict__ indices, const int S, const int F)
{
    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += Index(blockDim.x) * gridDim.x)
    {
        const int f = idx % F;
        const int b = idx / F;
        const int max_s = static_cast<int>(indices[idx]);

        in_gradient[(int64_t(b) * S + max_s) * F + f] = delta[idx];
    }
}

template<typename T>
void max_pooling_3d_backward_cuda(const Index n, const T* delta, T* in_gradient, const float* indices, const int S, const int F)
{
    launch_elementwise_strided(n, max_pooling_3d_backward_kernel<T>, delta, in_gradient, indices, S, F);
}

static float* get_pooling_scratch(size_t floats_needed)
{
    checked_host_condition(
        floats_needed > static_cast<size_t>(std::numeric_limits<Index>::max()),
        "pooling scratch size exceeds Index range.");

    static PoolingScratch& scratch = *new PoolingScratch();
    return scratch.ensure(Index(floats_needed));
}

template<typename T>
__global__ void pooling_3d_valid_mask_kernel(const int BS, const int S, const int F,
                                             const T* __restrict__ in,
                                             float* __restrict__ valid_mask,
                                             float* __restrict__ counts)
{
    const int bs = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs >= BS) return;

    const T* token = in + int64_t(bs) * F;
    const bool valid = !token_is_padding(token, F);

    valid_mask[bs] = valid ? 1.0f : 0.0f;
    if (valid) atomicAdd(&counts[bs / S], 1.0f);
}

template<typename T>
__global__ void average_pooling_3d_forward_kernel(const int n, const T* __restrict__ in, T* __restrict__ out,
                                                  const int S, const int F,
                                                  const float* __restrict__ valid_mask,
                                                  const float* __restrict__ counts)
{
    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += Index(blockDim.x) * gridDim.x)
    {
        const int f = idx % F;
        const int b = idx / F;

        const float count = counts[b];
        if (count == 0.0f) { out[idx] = static_cast<T>(0.0f); continue; }

        float sum = 0.0f;
        for (int s = 0; s < S; ++s)
        {
            const int64_t bs = int64_t(b) * S + s;
            sum += valid_mask[bs] * static_cast<float>(in[bs * F + f]);
        }

        out[idx] = static_cast<T>(sum / count);
    }
}

template<typename T>
static void prepare_pooling_valid_mask(const int B, const int S, const int F, const T* in,
                                       float*& valid_mask, float*& counts)
{
    const int BS = checked_int(Index(B) * S);
    const cudaStream_t stream = opennn::device::get_compute_stream();

    float* const scratch = get_pooling_scratch(static_cast<size_t>(BS) + B);
    valid_mask = scratch;
    counts     = scratch + BS;
    opennn::device::set_zero_async(counts, Index(B) * Index(sizeof(float)), stream);

    launch_elementwise(BS, pooling_3d_valid_mask_kernel<T>, S, F, in, valid_mask, counts);
}

template<typename T>
void average_pooling_3d_forward_cuda(const Index n, const T* in, T* out, const int S, const int F)
{
    if (n == 0) return;

    const int total = checked_int(n);
    const int B = total / F;

    float* valid_mask = nullptr;
    float* counts     = nullptr;
    prepare_pooling_valid_mask(B, S, F, in, valid_mask, counts);

    launch_elementwise_strided(n, average_pooling_3d_forward_kernel<T>, in, out, S, F, valid_mask, counts);
}

template<typename T>
__global__ void average_pooling_3d_backward_kernel(const int n, const T* __restrict__ delta, T* __restrict__ in_gradient,
                                                   const int S, const int F,
                                                   const float* __restrict__ valid_mask,
                                                   const float* __restrict__ counts)
{
    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += Index(blockDim.x) * gridDim.x)
    {
        const int f = idx % F;
        const int b = idx / F;

        // A fully padded row has nothing to divide by. Zeroing the gradient
        // instead of skipping the row keeps every element of in_gradient
        // written here, so the caller does not have to pre-zero the tensor.
        const float count = counts[b];
        const float gradient_val = count > 0.0f
                                 ? static_cast<float>(delta[idx]) / count
                                 : 0.0f;

        for (int s = 0; s < S; ++s)
        {
            const int64_t bs = int64_t(b) * S + s;
            in_gradient[bs * F + f] = static_cast<T>(valid_mask[bs] * gradient_val);
        }
    }
}

template<typename T>
void average_pooling_3d_backward_cuda(const Index n, const T* in, const T* delta, T* in_gradient, const int S, const int F)
{
    if (n == 0) return;

    const int total = checked_int(n);
    const int B = total / F;

    float* valid_mask = nullptr;
    float* counts     = nullptr;
    prepare_pooling_valid_mask(B, S, F, in, valid_mask, counts);

    launch_elementwise_strided(n, average_pooling_3d_backward_kernel<T>, delta, in_gradient, S, F, valid_mask, counts);
}

template<typename T, bool Gather>
__global__ void first_token_3d_kernel(const int n, const int S, const int F, const T* __restrict__ src, T* __restrict__ dst)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int b = i / F;
        const int h = i % F;
        const int strided = b * S * F + h;
        if constexpr (Gather) dst[i] = src[strided];
        else                  dst[strided] = src[i];
    }
}

template<typename T>
void first_token_3d_forward_cuda(const int B, const int S, const int F, const T* in, T* out)
{
    launch_elementwise_strided(Index(B) * F, first_token_3d_kernel<T, true>, S, F, in, out);
}

template<typename T>
void first_token_3d_backward_cuda(const int B, const int S, const int F, const T* delta, T* in_gradient)
{
    launch_elementwise_strided(Index(B) * F, first_token_3d_kernel<T, false>, S, F, delta, in_gradient);
}

__global__ void upsample_forward_kernel(
    const int n,
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int in_h, const int in_w,
    const int out_h, const int out_w,
    const int channels, const int scale)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int c  = i % channels;
        const int ow = (i / channels) % out_w;
        const int oh = (i / channels / out_w) % out_h;
        const int b  =  i / channels / out_w / out_h;

        const int iw = ow / scale;
        const int ih = oh / scale;
        dst[i] = src[((b * in_h + ih) * in_w + iw) * channels + c];
    }
}

__global__ void upsample_backward_kernel(
    const int n,
    const float* __restrict__ out_delta,
    float* __restrict__ in_delta,
    const int in_h, const int in_w,
    const int out_h, const int out_w,
    const int channels, const int scale)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int c  = i % channels;
        const int iw = (i / channels) % in_w;
        const int ih = (i / channels / in_w) % in_h;
        const int b  =  i / channels / in_w / in_h;

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

void upsample_forward_cuda(const int batch, const int in_h, const int in_w, const int channels, const int scale,
                           const float* src, float* dst)
{
    const int n = batch * (in_h * scale) * (in_w * scale) * channels;
    launch_elementwise_strided(n, upsample_forward_kernel,
                       src, dst, in_h, in_w, in_h * scale, in_w * scale, channels, scale);
}

void upsample_backward_cuda(const int batch, const int in_h, const int in_w, const int channels, const int scale,
                            const float* out_delta, float* in_delta)
{
    const int n = batch * in_h * in_w * channels;
    if (n == 0) return;
    // No pre-zeroing: the kernel assigns in_delta[i] for every i below n.
    launch_elementwise_strided(n, upsample_backward_kernel,
                       out_delta, in_delta, in_h, in_w, in_h * scale, in_w * scale, channels, scale);
}

template<bool Scatter>
__global__ void concat_slice_kernel(
    const int n,
    const float* __restrict__ src,
    float* __restrict__ dst,
    const int H, const int W,
    const int slice_ch, const int total_ch, const int ch_offset)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int c  = i % slice_ch;
        const int w  = (i / slice_ch) % W;
        const int h  = (i / slice_ch / W) % H;
        const int b  =  i / slice_ch / W / H;
        const int strided = ((b * H + h) * W + w) * total_ch + ch_offset + c;
        if constexpr (Scatter) dst[strided] = src[i];
        else                   dst[i] = src[strided];
    }
}

void concat_forward_slice_cuda(const int batch, const int H, const int W,
                               const int slice_ch, const int total_ch, const int ch_offset,
                               const float* src, float* dst)
{
    launch_elementwise_strided(Index(batch) * H * W * slice_ch, concat_slice_kernel<true>,
                       src, dst, H, W, slice_ch, total_ch, ch_offset);
}

void concat_backward_slice_cuda(const int batch, const int H, const int W,
                                const int slice_ch, const int total_ch, const int ch_offset,
                                const float* out_delta, float* in_delta)
{
    launch_elementwise_strided(Index(batch) * H * W * slice_ch, concat_slice_kernel<false>,
                       out_delta, in_delta, H, W, slice_ch, total_ch, ch_offset);
}

#define INSTANTIATE(T) \
    template void max_pooling_3d_forward_cuda<T>(const Index, const T*, T*, float*, const int, const int); \
    template void max_pooling_3d_backward_cuda<T>(const Index, const T*, T*, const float*, const int, const int); \
    template void average_pooling_3d_forward_cuda<T>(const Index, const T*, T*, const int, const int); \
    template void average_pooling_3d_backward_cuda<T>(const Index, const T*, const T*, T*, const int, const int); \
    template void first_token_3d_forward_cuda<T>(const int, const int, const int, const T*, T*); \
    template void first_token_3d_backward_cuda<T>(const int, const int, const int, const T*, T*);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

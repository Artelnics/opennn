//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   P O O L 3 D   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/neural_network/operators/kernel_pool3d.cuh"

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

// Where a sequence ends, clamped to what this tensor actually holds. A null
// lengths pointer means no record was exported, and every row counts.
__device__ inline int clamped_length(const int* __restrict__ lengths, const int b, const int S)
{
    if (lengths == nullptr) return S;
    const int length = lengths[b];
    if (length < 0) return 0;
    return length < S ? length : S;
}

template<typename T>
__global__ void max_pooling_3d_forward_kernel(const int n, const T* __restrict__ in, T* __restrict__ out, float* __restrict__ indices, const int S, const int F,
                                              const int* __restrict__ lengths)
{
    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += Index(blockDim.x) * gridDim.x)
    {
        const int f = idx % F;
        const int b = idx / F;

        const int steps = clamped_length(lengths, b, S);

        // Nothing to take a maximum over. Zero matches what the average does
        // with a fully padded sequence, and keeps the recorded index in range
        // for the backward pass.
        if (steps == 0)
        {
            out[idx] = static_cast<T>(0.0f);
            if (indices != nullptr) indices[idx] = 0.0f;
            continue;
        }

        float max_val = -1e20f;
        int max_index = 0;

        for (int s = 0; s < steps; ++s)
        {
            const float val = static_cast<float>(in[(int64_t(b) * S + s) * F + f]);
            if (val > max_val) { max_val = val; max_index = s; }
        }

        out[idx] = static_cast<T>(max_val);
        if (indices != nullptr) indices[idx] = static_cast<float>(max_index);
    }
}

template<typename T>
void max_pooling_3d_forward_cuda(const Index n, const T* in, T* out, float* indices, const int S, const int F,
                                 const int* valid_lengths)
{
    launch_elementwise_strided(n, max_pooling_3d_forward_kernel<T>, in, out, indices, S, F, valid_lengths);
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

// Exact lengths make the mask a prefix, so it can be written straight out
// rather than reduced into: one thread per row decides that row, and the thread
// holding the first row of a sequence writes the count the whole sequence
// divides by.
__global__ void pooling_3d_length_mask_kernel(const int BS, const int S,
                                              const int* __restrict__ lengths,
                                              float* __restrict__ valid_mask,
                                              float* __restrict__ counts)
{
    const int bs = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs >= BS) return;

    const int b = bs / S;
    const int s = bs - b * S;
    const int length = clamped_length(lengths, b, S);

    valid_mask[bs] = s < length ? 1.0f : 0.0f;
    if (s == 0) counts[b] = static_cast<float>(length);
}

template<typename T>
static void prepare_pooling_valid_mask(const int B, const int S, const int F, const T* in,
                                       const int* device_lengths,
                                       float*& valid_mask, float*& counts)
{
    const int BS = checked_int(Index(B) * S);
    const cudaStream_t stream = opennn::device::get_compute_stream();

    float* const scratch = get_pooling_scratch(static_cast<size_t>(BS) + B);
    valid_mask = scratch;
    counts     = scratch + BS;

    if (device_lengths != nullptr)
    {
        launch_elementwise(BS, pooling_3d_length_mask_kernel, S, device_lengths, valid_mask, counts);
        return;
    }

    opennn::device::set_zero_async(counts, Index(B) * Index(sizeof(float)), stream);

    launch_elementwise(BS, pooling_3d_valid_mask_kernel<T>, S, F, in, valid_mask, counts);
}

template<typename T>
void average_pooling_3d_forward_cuda(const Index n, const T* in, T* out, const int S, const int F,
                                     const int* valid_lengths)
{
    if (n == 0) return;

    const int total = checked_int(n);
    const int B = total / F;

    float* valid_mask = nullptr;
    float* counts     = nullptr;
    prepare_pooling_valid_mask(B, S, F, in, valid_lengths, valid_mask, counts);

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
void average_pooling_3d_backward_cuda(const Index n, const T* in, const T* delta, T* in_gradient, const int S, const int F,
                                      const int* valid_lengths)
{
    if (n == 0) return;

    const int total = checked_int(n);
    const int B = total / F;

    float* valid_mask = nullptr;
    float* counts     = nullptr;
    prepare_pooling_valid_mask(B, S, F, in, valid_lengths, valid_mask, counts);

    launch_elementwise_strided(n, average_pooling_3d_backward_kernel<T>, delta, in_gradient, S, F, valid_mask, counts);
}

#define INSTANTIATE(T) \
    template void max_pooling_3d_forward_cuda<T>(const Index, const T*, T*, float*, const int, const int, const int*); \
    template void max_pooling_3d_backward_cuda<T>(const Index, const T*, T*, const float*, const int, const int); \
    template void average_pooling_3d_forward_cuda<T>(const Index, const T*, T*, const int, const int, const int*); \
    template void average_pooling_3d_backward_cuda<T>(const Index, const T*, const T*, T*, const int, const int, const int*);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

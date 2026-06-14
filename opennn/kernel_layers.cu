#include "kernel_common.cuh"
#include <curand_kernel.h>

template<typename TIn, typename TOut>
__global__ void bounding_kernel(const int n, const int features,
                                const TIn* __restrict__ input,
                                const float* __restrict__ lower,
                                const float* __restrict__ upper,
                                TOut* __restrict__ output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int f = i % features;
        const float x = static_cast<float>(input[i]);
        output[i] = static_cast<TOut>(fminf(fmaxf(x, lower[f]), upper[f]));
    }
}

template<typename TIn, typename TOut>
void bounding_cuda(const Index n, const int features,
                   const TIn* input, const float* lower, const float* upper,
                   TOut* output)
{
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(bounding_kernel<TIn, TOut><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, features, input, lower, upper, output));
}

template void bounding_cuda<float,         float>        (const Index, const int, const float*,         const float*, const float*, float*);
template void bounding_cuda<float,         __nv_bfloat16>(const Index, const int, const float*,         const float*, const float*, __nv_bfloat16*);
template void bounding_cuda<__nv_bfloat16, float>        (const Index, const int, const __nv_bfloat16*, const float*, const float*, float*);
template void bounding_cuda<__nv_bfloat16, __nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, const float*, __nv_bfloat16*);

template<typename TIn, typename TOut>
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
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int f = i % features;
        const int code = static_cast<int>(scalers[f]);
        const float x = static_cast<float>(input[i]);
        float y = x;

        switch (code)
        {
        case 1:
        {
            const float range = maximums[f] - minimums[f];
            y = (range < FLT_EPSILON) ? 0.0f
              : (x - minimums[f]) / range * (max_range - min_range) + min_range;
            break;
        }
        case 2:
            y = (stds[f] > FLT_EPSILON) ? (x - means[f]) / stds[f] : 0.0f;
            break;
        case 3:
            y = (stds[f] > FLT_EPSILON) ? x / stds[f] : 0.0f;
            break;
        case 4:
            y = logf(x);
            break;
        case 5:
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
                float min_range, float max_range,
                TOut* output)
{
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(scale_kernel<TIn, TOut><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, features,
                                                                   input, minimums, maximums, means, stds, scalers,
                                                                   min_range, max_range, output));
}

template void scale_cuda<float,         float>        (const Index, const int, const float*,         const float*, const float*, const float*, const float*, const float*, float, float, float*);
template void scale_cuda<float,         __nv_bfloat16>(const Index, const int, const float*,         const float*, const float*, const float*, const float*, const float*, float, float, __nv_bfloat16*);
template void scale_cuda<__nv_bfloat16, float>        (const Index, const int, const __nv_bfloat16*, const float*, const float*, const float*, const float*, const float*, float, float, float*);
template void scale_cuda<__nv_bfloat16, __nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, const float*, const float*, const float*, const float*, float, float, __nv_bfloat16*);

template<typename TIn, typename TOut>
__global__ void unscale_kernel(const int n, const int features,
                               const TIn* __restrict__ input,
                               const float* __restrict__ minimums,
                               const float* __restrict__ maximums,
                               const float* __restrict__ means,
                               const float* __restrict__ stds,
                               const float* __restrict__ scalers,
                               const float min_range, const float max_range,
                               TOut* __restrict__ output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int f = i % features;
        const int code = static_cast<int>(scalers[f]);
        const float x = static_cast<float>(input[i]);
        float y = x;

        switch (code)
        {
        case 1:
            y = (x - min_range) / (max_range - min_range)
                * (maximums[f] - minimums[f]) + minimums[f];
            break;
        case 2:
            y = means[f] + x * stds[f];
            break;
        case 3:
            y = x * stds[f];
            break;
        case 4:
            y = expf(x);
            break;
        case 5:
            y = x * 255.0f;
            break;
        default:
            break;
        }

        output[i] = static_cast<TOut>(y);
    }
}

template<typename TIn, typename TOut>
void unscale_cuda(const Index n, const int features,
                  const TIn* input,
                  const float* minimums, const float* maximums,
                  const float* means, const float* stds,
                  const float* scalers,
                  float min_range, float max_range,
                  TOut* output)
{
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(unscale_kernel<TIn, TOut><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, features,
                                                                    input, minimums, maximums, means, stds, scalers,
                                                                    min_range, max_range, output));
}

template void unscale_cuda<float,         float>        (const Index, const int, const float*,         const float*, const float*, const float*, const float*, const float*, float, float, float*);
template void unscale_cuda<float,         __nv_bfloat16>(const Index, const int, const float*,         const float*, const float*, const float*, const float*, const float*, float, float, __nv_bfloat16*);
template void unscale_cuda<__nv_bfloat16, float>        (const Index, const int, const __nv_bfloat16*, const float*, const float*, const float*, const float*, const float*, float, float, float*);
template void unscale_cuda<__nv_bfloat16, __nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, const float*, const float*, const float*, const float*, float, float, __nv_bfloat16*);

template<typename TIn>
__global__ void diff_to_fp32_kernel(const int n,
                                    const TIn* __restrict__ input,
                                    const float* __restrict__ target,
                                    float* __restrict__ output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        output[i] = static_cast<float>(input[i]) - target[i];
}

template<typename TIn>
void diff_to_fp32_cuda(const Index n, const TIn* input, const float* target, float* output)
{
    if (n == 0) return;
    const int total = checked_int(n);
    OPENNN_CUDA_LAUNCH(diff_to_fp32_kernel<TIn><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, input, target, output));
}

template void diff_to_fp32_cuda<float>        (const Index, const float*,         const float*, float*);
template void diff_to_fp32_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, const float*, float*);

template<typename TIn, typename TOut>
__global__ void scaled_diff_kernel(const int n,
                                   const TIn* __restrict__ input,
                                   const float* __restrict__ target,
                                   const float scale,
                                   TOut* __restrict__ output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const float d = static_cast<float>(input[i]) - target[i];
        output[i] = static_cast<TOut>(scale * d);
    }
}

template<typename TIn, typename TOut>
void scaled_diff_cuda_typed(const Index n, const TIn* input, const float* target,
                            float scale, TOut* output)
{
    if (n == 0) return;
    const int total = checked_int(n);
    OPENNN_CUDA_LAUNCH(scaled_diff_kernel<TIn, TOut><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, input, target, scale, output));
}

template void scaled_diff_cuda_typed<float,         float>        (const Index, const float*,         const float*, float, float*);
template void scaled_diff_cuda_typed<float,         __nv_bfloat16>(const Index, const float*,         const float*, float, __nv_bfloat16*);
template void scaled_diff_cuda_typed<__nv_bfloat16, float>        (const Index, const __nv_bfloat16*, const float*, float, float*);
template void scaled_diff_cuda_typed<__nv_bfloat16, __nv_bfloat16>(const Index, const __nv_bfloat16*, const float*, float, __nv_bfloat16*);

template<typename T>
__global__ void embedding_forward_kernel(const int n, const float* __restrict__ inputs, const float* __restrict__ weights, const float* __restrict__ positional_encoding, T* __restrict__ outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    const float scale = scale_embedding ? sqrtf(static_cast<float>(embedding_dimension)) : 1.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int token_index = i / embedding_dimension;
        const int dim_index = i % embedding_dimension;
        const int token_id = static_cast<int>(inputs[token_index]);

        float val = (token_id > 0 && token_id < vocabulary_size)
            ? scale * weights[token_id * embedding_dimension + dim_index]
            : 0.0f;

        if (positional_encoding != nullptr && token_id > 0)
        {
            const int seq_index = token_index % sequence_length;
            val += positional_encoding[seq_index * embedding_dimension + dim_index];
        }

        outputs[i] = static_cast<T>(val);
    }
}

template<typename T>
void embedding_forward_cuda(const Index n, const float* inputs, const float* weights, const float* positional_encoding, T* outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(embedding_forward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(
        total, inputs, weights, positional_encoding, outputs,
        sequence_length, embedding_dimension, vocabulary_size, scale_embedding));
}

template void embedding_forward_cuda<float>        (const Index, const float*, const float*, const float*, float*,         const int, const int, const int, const bool);
template void embedding_forward_cuda<__nv_bfloat16>(const Index, const float*, const float*, const float*, __nv_bfloat16*, const int, const int, const int, const bool);

template<typename T>
__global__ void embedding_backward_kernel(const int n, const float* __restrict__ inputs, const T* __restrict__ output_deltas, float* __restrict__ weight_gradients, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    const float scale = scale_embedding ? sqrtf(static_cast<float>(embedding_dimension)) : 1.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int token_index = i / embedding_dimension;
        const int dim_index = i % embedding_dimension;
        const int token_id = static_cast<int>(inputs[token_index]);

        if (token_id <= 0 || token_id >= vocabulary_size) continue;

        atomicAdd(&weight_gradients[token_id * embedding_dimension + dim_index], scale * static_cast<float>(output_deltas[i]));
    }
}

template<typename T>
void embedding_backward_cuda(const Index n, const float* inputs, const T* output_deltas, float* weight_gradients, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(embedding_backward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(
        total, inputs, output_deltas, weight_gradients,
        embedding_dimension, vocabulary_size, scale_embedding));
}

template void embedding_backward_cuda<float>        (const Index, const float*, const float*,         float*, const int, const int, const bool);
template void embedding_backward_cuda<__nv_bfloat16>(const Index, const float*, const __nv_bfloat16*, float*, const int, const int, const bool);

// Both head reshapes are the same permutation [B, P, Q, D] -> [B, Q, P, D]:
// split_heads uses (P=S, Q=H) and merge_heads (P=H, Q=S).
template<typename T>
__global__ void swap_heads_scalar_kernel(const int n, const T* __restrict__ in, T* __restrict__ out, const int P, const int Q, const int D)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int d = i % D;
        const int q = (i / D) % Q;
        const int p = (i / (D * Q)) % P;
        const int b = i / (D * Q * P);

        out[((int64_t(b) * Q + q) * P + p) * D + d] = in[i];
    }
}

template<typename T>
__global__ void swap_heads_vec_kernel(const int n_vec, const T* __restrict__ in, T* __restrict__ out, const int P, const int Q, const int D_vec)
{
    const float4* const in_v  = reinterpret_cast<const float4*>(in);
    float4* const       out_v = reinterpret_cast<float4*>(out);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += blockDim.x * gridDim.x)
    {
        const int d = i % D_vec;
        const int q = (i / D_vec) % Q;
        const int p = (i / (D_vec * Q)) % P;
        const int b = i / (D_vec * Q * P);

        out_v[((int64_t(b) * Q + q) * P + p) * D_vec + d] = in_v[i];
    }
}

template<typename T>
void split_heads_cuda(const Index n, const T* in, T* out, const int S, const int H, const int D)
{
    if (n == 0) return;

    if ((static_cast<size_t>(D) * sizeof(T)) % 16 == 0 && are_float4_aligned(in, out))
    {
        const int vec_width = static_cast<int>(16 / sizeof(T));
        const int D_vec     = D / vec_width;
        const int n_vec     = checked_int(n / vec_width);
        OPENNN_CUDA_LAUNCH(swap_heads_vec_kernel<T><<<grid_size_for(n_vec), block_size, 0, opennn::device::get_compute_stream()>>>(n_vec, in, out, S, H, D_vec));
    }
    else
    {
        const int total = checked_int(n);
        OPENNN_CUDA_LAUNCH(swap_heads_scalar_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, in, out, S, H, D));
    }
}

template void split_heads_cuda<float>        (const Index, const float*,         float*,         const int, const int, const int);
template void split_heads_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const int, const int, const int);

template<typename T>
void merge_heads_cuda(const Index n, const T* in, T* out, const int S, const int H, const int D)
{
    if (n == 0) return;

    if ((static_cast<size_t>(D) * sizeof(T)) % 16 == 0 && are_float4_aligned(in, out))
    {
        const int vec_width = static_cast<int>(16 / sizeof(T));
        const int D_vec     = D / vec_width;
        const int n_vec     = checked_int(n / vec_width);
        OPENNN_CUDA_LAUNCH(swap_heads_vec_kernel<T><<<grid_size_for(n_vec), block_size, 0, opennn::device::get_compute_stream()>>>(n_vec, in, out, H, S, D_vec));
    }
    else
    {
        const int total = checked_int(n);
        OPENNN_CUDA_LAUNCH(swap_heads_scalar_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, in, out, H, S, D));
    }
}

template void merge_heads_cuda<float>        (const Index, const float*,         float*,         const int, const int, const int);
template void merge_heads_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const int, const int, const int);

template<typename T>
__global__ void padding_mask_kernel(const int num_tokens, const T* __restrict__ source_input, T* __restrict__ padding_mask, const int embedding_dimension)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_tokens; i += blockDim.x * gridDim.x)
    {
        const T* token = source_input + i * embedding_dimension;
        bool is_pad = true;
        for (int e = 0; e < embedding_dimension; ++e)
            if (fabsf(static_cast<float>(token[e])) > 1e-7f) { is_pad = false; break; }
        padding_mask[i] = static_cast<T>(is_pad ? 1.0f : 0.0f);
    }
}

template<typename T>
__global__ void fused_masks_kernel(const int n, T* __restrict__ attention_weights, const T* __restrict__ padding_mask,
                                         const int heads_number, const int query_sequence_length,
                                         const int source_sequence_length, const bool use_causal_mask)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int sk = i % source_sequence_length;
        const int sq = (i / source_sequence_length) % query_sequence_length;
        const int b  = i / (source_sequence_length * query_sequence_length * heads_number);

        if ((use_causal_mask && sk > sq) || static_cast<float>(padding_mask[b * source_sequence_length + sk]) > 0.5f)
            attention_weights[i] = static_cast<T>(-1e9f);
    }
}

template<typename T>
void attention_masks_cuda(const int batch_size, const int heads_number,
                          const int query_sequence_length, const int source_sequence_length,
                          const int embedding_dimension, const T* source_input,
                          T* attention_weights, T* padding_mask, const bool use_causal_mask)
{
    const int num_tokens = batch_size * source_sequence_length;
    if (num_tokens > 0)
        OPENNN_CUDA_LAUNCH(padding_mask_kernel<T><<<grid_size_for(num_tokens), block_size, 0, opennn::device::get_compute_stream()>>>(
            num_tokens, source_input, padding_mask, embedding_dimension));

    const int n = batch_size * heads_number * query_sequence_length * source_sequence_length;
    if (n > 0)
        OPENNN_CUDA_LAUNCH(fused_masks_kernel<T><<<grid_size_for(n), block_size, 0, opennn::device::get_compute_stream()>>>(
            n, attention_weights, padding_mask, heads_number,
            query_sequence_length, source_sequence_length, use_causal_mask));
}

template void attention_masks_cuda<float>        (int, int, int, int, int, const float*,         float*,         float*,         bool);
template void attention_masks_cuda<__nv_bfloat16>(int, int, int, int, int, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, bool);

template<typename T>
__global__ void attention_sequence_lengths_kernel(const int batch_size,
                                                  const int query_sequence_length,
                                                  const int source_sequence_length,
                                                  const int embedding_dimension,
                                                  const T* __restrict__ source_input,
                                                  int32_t* __restrict__ query_lengths,
                                                  int32_t* __restrict__ source_lengths)
{
    const int batch = blockIdx.x;
    if (batch >= batch_size) return;

    __shared__ int stop;
    if (threadIdx.x == 0)
    {
        stop = 0;
        query_lengths[batch] = query_sequence_length;
        source_lengths[batch] = 1;
    }
    __syncthreads();

    const T* sequence = source_input + batch * source_sequence_length * embedding_dimension;

    for (int s = 0; s < source_sequence_length; ++s)
    {
        bool nonzero = false;
        const T* token = sequence + s * embedding_dimension;
        for (int e = threadIdx.x; e < embedding_dimension; e += blockDim.x)
            if (fabsf(static_cast<float>(token[e])) > 1e-7f) { nonzero = true; break; }

        const int token_is_valid = __syncthreads_or(nonzero);

        if (threadIdx.x == 0)
        {
            if (token_is_valid) source_lengths[batch] = s + 1;
            else stop = 1;
        }
        __syncthreads();
        if (stop) break;
    }
}

template<typename T>
void attention_sequence_lengths_cuda(const int batch_size,
                                     const int query_sequence_length,
                                     const int source_sequence_length,
                                     const int embedding_dimension,
                                     const T* source_input,
                                     int32_t* query_lengths,
                                     int32_t* source_lengths)
{
    if (batch_size > 0)
        OPENNN_CUDA_LAUNCH(attention_sequence_lengths_kernel<T><<<batch_size, block_size, 0, opennn::device::get_compute_stream()>>>(
            batch_size,
            query_sequence_length,
            source_sequence_length,
            embedding_dimension,
            source_input,
            query_lengths,
            source_lengths));
}

template void attention_sequence_lengths_cuda<float>        (int, int, int, int, const float*,         int32_t*, int32_t*);
template void attention_sequence_lengths_cuda<__nv_bfloat16>(int, int, int, int, const __nv_bfloat16*, int32_t*, int32_t*);

template<typename T>
__global__ void max_pooling_3d_forward_kernel(const int n, const T* __restrict__ in, T* __restrict__ out, float* __restrict__ indices, const int S, const int F)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
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
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(max_pooling_3d_forward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, in, out, indices, S, F));
}

template void max_pooling_3d_forward_cuda<float>        (const Index, const float*,         float*,         float*, const int, const int);
template void max_pooling_3d_forward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, float*, const int, const int);

template<typename T>
__global__ void max_pooling_3d_backward_kernel(const int n, const T* __restrict__ delta, T* __restrict__ in_gradient, const float* __restrict__ indices, const int S, const int F)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
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
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(max_pooling_3d_backward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, delta, in_gradient, indices, S, F));
}

template void max_pooling_3d_backward_cuda<float>        (const Index, const float*,         float*,         const float*, const int, const int);
template void max_pooling_3d_backward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const float*, const int, const int);

namespace
{

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

    ~PoolingScratch()
    {
        if (data) opennn::device::deallocate(opennn::Device::CUDA, data, bytes);
    }
};

PoolingScratch pooling_scratch_;

}

static float* get_pooling_scratch(size_t floats_needed)
{
    checked_host_condition(
        floats_needed > static_cast<size_t>(std::numeric_limits<Index>::max()),
        "pooling scratch size exceeds Index range.");
    return pooling_scratch_.ensure(Index(floats_needed));
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
    bool valid = false;
    for (int f = 0; f < F; ++f)
        if (fabsf(static_cast<float>(token[f])) > 1e-7f) { valid = true; break; }

    valid_mask[bs] = valid ? 1.0f : 0.0f;
    if (valid) atomicAdd(&counts[bs / S], 1.0f);
}

template<typename T>
__global__ void average_pooling_3d_forward_kernel(const int n, const T* __restrict__ in, T* __restrict__ out,
                                                  const int S, const int F,
                                                  const float* __restrict__ valid_mask,
                                                  const float* __restrict__ counts)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
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
static void prepare_pooling_valid_mask(int B, int S, int F, const T* in,
                                       float*& valid_mask, float*& counts)
{
    const int BS = checked_int(Index(B) * S);
    cudaStream_t stream = opennn::device::get_compute_stream();

    float* scratch = get_pooling_scratch(static_cast<size_t>(BS) + B);
    valid_mask = scratch;
    counts     = scratch + BS;
    opennn::device::set_zero_async(counts, Index(B) * Index(sizeof(float)), stream);

    OPENNN_CUDA_LAUNCH(pooling_3d_valid_mask_kernel<T><<<grid_size_for(BS), block_size, 0, stream>>>(BS, S, F, in, valid_mask, counts));
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

    OPENNN_CUDA_LAUNCH(average_pooling_3d_forward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, in, out, S, F, valid_mask, counts));
}

template void average_pooling_3d_forward_cuda<float>        (const Index, const float*,         float*,         const int, const int);
template void average_pooling_3d_forward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const int, const int);

template<typename T>
__global__ void average_pooling_3d_backward_kernel(const int n, const T* __restrict__ delta, T* __restrict__ in_gradient,
                                                   const int S, const int F,
                                                   const float* __restrict__ valid_mask,
                                                   const float* __restrict__ counts)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const int f = idx % F;
        const int b = idx / F;

        const float count = counts[b];
        if (count == 0.0f) continue;

        const float gradient_val = static_cast<float>(delta[idx]) / count;
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

    OPENNN_CUDA_LAUNCH(average_pooling_3d_backward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, delta, in_gradient, S, F, valid_mask, counts));
}

template void average_pooling_3d_backward_cuda<float>        (const Index, const float*,         const float*,         float*,         const int, const int);
template void average_pooling_3d_backward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, const int, const int);

__device__ __forceinline__ void warp_reduce_sum2(float& a, float& b)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        a += __shfl_down_sync(0xffffffff, a, offset);
        b += __shfl_down_sync(0xffffffff, b, offset);
    }
}

template<typename T>
__global__ void layernorm_forward_kernel(const int N, const int D, const T* __restrict__ X, T* __restrict__ Y, float* __restrict__ means, float* __restrict__ inv_vars, const float* __restrict__ gamma, const float* __restrict__ beta, const float eps)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const T* x_row = X + idx * D;
    T* y_row = Y + idx * D;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float x = static_cast<float>(x_row[i]);
        local_sum    += x;
        local_sum_sq += x * x;
    }

    warp_reduce_sum2(local_sum, local_sum_sq);

    __shared__ float warp_sum[32];
    __shared__ float warp_sum_sq[32];
    __shared__ float s_mean;
    __shared__ float s_inv_var;

    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0)
    {
        warp_sum[warp_id]    = local_sum;
        warp_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + 31) >> 5;
    if (warp_id == 0)
    {
        float s    = (threadIdx.x < num_warps) ? warp_sum[threadIdx.x]    : 0.0f;
        float s_sq = (threadIdx.x < num_warps) ? warp_sum_sq[threadIdx.x] : 0.0f;
        warp_reduce_sum2(s, s_sq);

        if (threadIdx.x == 0)
        {
            const float inv_D = 1.0f / static_cast<float>(D);
            const float mean = s * inv_D;
            // Clamp variance to >= 0: E[x^2] - E[x]^2 can go slightly negative from
            // catastrophic cancellation at large activations, which would make
            // rsqrtf return inf/nan (matches the CPU layer-norm fix).
            const float variance = fmaxf(s_sq * inv_D - mean * mean, 0.0f);
            const float inv_var = rsqrtf(variance + eps);
            s_mean    = mean;
            s_inv_var = inv_var;
            means[idx]    = mean;
            inv_vars[idx] = inv_var;
        }
    }
    __syncthreads();

    const float mean    = s_mean;
    const float inv_var = s_inv_var;

    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float x_hat = (static_cast<float>(x_row[i]) - mean) * inv_var;
        y_row[i] = static_cast<T>(fmaf(gamma[i], x_hat, beta[i]));
    }
}

static inline int layernorm_threads(int D)
{
    if (D <= 32) return 32;
    if (D <= 64) return 64;
    if (D <= 128) return 128;
    return 256;
}

template<typename T>
void layernorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps)
{
    if (N == 0 || D == 0) return;

    OPENNN_CUDA_LAUNCH(layernorm_forward_kernel<T><<<N, layernorm_threads(D), 0, opennn::device::get_compute_stream()>>>(N, D, X, Y, means, inv_vars, gamma, beta, eps));
}

template void layernorm_forward_cuda<float>        (const int, const int, const float*,         float*,         float*, float*, const float*, const float*, const float);
template void layernorm_forward_cuda<__nv_bfloat16>(const int, const int, const __nv_bfloat16*, __nv_bfloat16*, float*, float*, const float*, const float*, const float);

template<typename T>
__global__ void layernorm_backward_kernel(const int N, const int D, const T* __restrict__ dY, const T* __restrict__ X, const float* __restrict__ means, const float* __restrict__ inv_vars, const float* __restrict__ gamma, T* __restrict__ dX)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const T* dy_row = dY + idx * D;
    const T* x_row = X + idx * D;
    T* dx_row = dX + idx * D;

    const float mean = means[idx];
    const float inv_var = inv_vars[idx];

    float local_sum_D      = 0.0f;
    float local_sum_D_xhat = 0.0f;

    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float d     = static_cast<float>(dy_row[i]) * gamma[i];
        const float x_hat = (static_cast<float>(x_row[i]) - mean) * inv_var;
        local_sum_D      += d;
        local_sum_D_xhat += d * x_hat;
    }

    warp_reduce_sum2(local_sum_D, local_sum_D_xhat);

    __shared__ float warp_sum_D[32];
    __shared__ float warp_sum_D_xhat[32];
    __shared__ float s_mean_D;
    __shared__ float s_mean_D_xhat;

    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0)
    {
        warp_sum_D[warp_id]      = local_sum_D;
        warp_sum_D_xhat[warp_id] = local_sum_D_xhat;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + 31) >> 5;
    if (warp_id == 0)
    {
        float s  = (threadIdx.x < num_warps) ? warp_sum_D[threadIdx.x]      : 0.0f;
        float sx = (threadIdx.x < num_warps) ? warp_sum_D_xhat[threadIdx.x] : 0.0f;
        warp_reduce_sum2(s, sx);

        if (threadIdx.x == 0)
        {
            const float inv_D = 1.0f / static_cast<float>(D);
            s_mean_D      = s  * inv_D;
            s_mean_D_xhat = sx * inv_D;
        }
    }
    __syncthreads();

    const float mean_D      = s_mean_D;
    const float mean_D_xhat = s_mean_D_xhat;

    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float d     = static_cast<float>(dy_row[i]) * gamma[i];
        const float x_hat = (static_cast<float>(x_row[i]) - mean) * inv_var;
        dx_row[i] = static_cast<T>((d - mean_D - x_hat * mean_D_xhat) * inv_var);
    }
}

template<typename T, int NUM_WARPS>
__global__ void layernorm_gamma_beta_gradient_coalesced_kernel(const int N, const int D,
                                                               const T* __restrict__ dY,
                                                               const T* __restrict__ X,
                                                               const float* __restrict__ means,
                                                               const float* __restrict__ inv_vars,
                                                               float* __restrict__ dGamma,
                                                               float* __restrict__ dBeta)
{
    const int lane    = threadIdx.x; 
    const int warp_id = threadIdx.y;
    const int d       = blockIdx.x * 32 + lane;
    const bool active = (d < D);

    float local_gamma = 0.0f;
    float local_beta  = 0.0f;

    if (active)
    {
        for (int n = warp_id; n < N; n += NUM_WARPS)
        {
            const float dy    = static_cast<float>(dY[n * D + d]);
            const float x_hat = (static_cast<float>(X[n * D + d]) - means[n]) * inv_vars[n];
            local_gamma += dy * x_hat;
            local_beta  += dy;
        }
    }

    __shared__ float partial_gamma[NUM_WARPS][32];
    __shared__ float partial_beta [NUM_WARPS][32];

    partial_gamma[warp_id][lane] = local_gamma;
    partial_beta [warp_id][lane] = local_beta;
    __syncthreads();

    if (warp_id == 0 && active)
    {
        float g = 0.0f;
        float b = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w)
        {
            g += partial_gamma[w][lane];
            b += partial_beta [w][lane];
        }
        dGamma[d] = g;
        dBeta [d] = b;
    }
}

template<typename T>
void layernorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* means, const float* inv_vars, const float* gamma, T* dX, float* dGamma, float* dBeta)
{
    if (N == 0 || D == 0) return;

    if (dX)
        OPENNN_CUDA_LAUNCH(layernorm_backward_kernel<T><<<N, layernorm_threads(D), 0, opennn::device::get_compute_stream()>>>(N, D, dY, X, means, inv_vars, gamma, dX));

    constexpr int NUM_WARPS = 8;
    const dim3 block(32, NUM_WARPS);
    const int grid_x = (D + 31) / 32;
    layernorm_gamma_beta_gradient_coalesced_kernel<T, NUM_WARPS><<<grid_x, block, 0,
        opennn::device::get_compute_stream()>>>(N, D, dY, X, means, inv_vars, dGamma, dBeta);
    opennn::device::check_last_error();
}

template void layernorm_backward_cuda<float>        (const int, const int, const float*,         const float*,         const float*, const float*, const float*, float*,         float*, float*);
template void layernorm_backward_cuda<__nv_bfloat16>(const int, const int, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, const float*, __nv_bfloat16*, float*, float*);

template<typename T>
__global__ void activation_forward_kernel(const int n, T* __restrict__ data, const int function)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const float x = static_cast<float>(data[idx]);
        float y = x;

        if (function == 1)
            y = 1.0f / (1.0f + expf(-x));
        else if (function == 2)
            y = tanhf(x);
        else if (function == 3)
            y = fmaxf(x, 0.0f);

        data[idx] = static_cast<T>(y);
    }
}

template<typename T>
void activation_forward_cuda(const Index n, T* data, const int function)
{
    if (n == 0) return;

    const int total = checked_int(n);
    OPENNN_CUDA_LAUNCH(activation_forward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, data, function));
}

template void activation_forward_cuda<float>        (const Index, float*,         const int);
template void activation_forward_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const int);

template<typename T>
__global__ void activation_backward_kernel(const int n, const T* __restrict__ outputs, T* __restrict__ delta, const int function)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const float y = static_cast<float>(outputs[idx]);
        const float d = static_cast<float>(delta[idx]);
        float out = d;

        if (function == 1)
            out = d * y * (1.0f - y);
        else if (function == 2)
            out = d * (1.0f - y * y);
        else if (function == 3)
            out = y > 0.0f ? d : 0.0f;

        delta[idx] = static_cast<T>(out);
    }
}

template<typename T>
void activation_backward_cuda(const Index n, const T* outputs, T* delta, const int function)
{
    if (n == 0) return;

    const int total = checked_int(n);
    OPENNN_CUDA_LAUNCH(activation_backward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, outputs, delta, function));
}

template void activation_backward_cuda<float>        (const Index, const float*,         float*,         const int);
template void activation_backward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const int);

template<typename T>
__global__ void dropout_forward_kernel(const int n, T* __restrict__ output, uint8_t* __restrict__ mask, const float scale, const float rate, const unsigned long long seed)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);
    const float r = curand_uniform(&state);

    const uint8_t keep = (r >= rate) ? uint8_t(1) : uint8_t(0);
    mask[idx] = keep;
    output[idx] = static_cast<T>(static_cast<float>(output[idx]) * (keep * scale));
}

template<typename T>
void dropout_forward_cuda(const Index n, T* output, uint8_t* mask, const float rate, const unsigned long long seed)
{
    if (n == 0) return;

    const int total = checked_int(n);
    const float scale = 1.0f / (1.0f - rate);

    OPENNN_CUDA_LAUNCH(dropout_forward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, output, mask, scale, rate, seed));
}

template void dropout_forward_cuda<float>        (const Index, float*,         uint8_t*, const float, const unsigned long long);
template void dropout_forward_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, uint8_t*, const float, const unsigned long long);

template<typename T>
__global__ void dropout_backward_kernel(const int n, const T* __restrict__ output_delta, T* __restrict__ input_delta, const uint8_t* __restrict__ mask, const float scale)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float dy = static_cast<float>(output_delta[idx]);
    const float m  = static_cast<float>(mask[idx]) * scale;
    input_delta[idx] = static_cast<T>(dy * m);
}

template<typename T>
void dropout_backward_cuda(const Index n, const T* output_delta, T* input_delta, const uint8_t* mask, const float rate)
{
    if (n == 0) return;

    const int total = checked_int(n);
    const float scale = 1.0f / (1.0f - rate);

    OPENNN_CUDA_LAUNCH(dropout_backward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, output_delta, input_delta, mask, scale));
}

template void dropout_backward_cuda<float>        (const Index, const float*,         float*,         const uint8_t*, const float);
template void dropout_backward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const uint8_t*, const float);

template<typename T>
__global__ void gather_time_slice_kernel(const int batch,
                                         const int time_steps,
                                         const int features,
                                         const int t,
                                         const T* __restrict__ src,
                                         T* __restrict__ dst)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * features;
    if (idx >= total) return;

    const int b = idx / features;
    const int f = idx - b * features;
    dst[idx] = src[(b * time_steps + t) * features + f];
}

template<typename T>
void gather_time_slice_cuda(const Index batch,
                            const Index time_steps,
                            const Index features,
                            const Index t,
                            const T* src,
                            T* dst)
{
    if (batch == 0 || features == 0) return;
    const int total = checked_int(batch * features);
    OPENNN_CUDA_LAUNCH(gather_time_slice_kernel<T><<<grid_size_for(total), block_size, 0,
                                  opennn::device::get_compute_stream()>>>(
        checked_int(batch),
        checked_int(time_steps),
        checked_int(features),
        checked_int(t),
        src, dst));
}

template void gather_time_slice_cuda<float>        (const Index, const Index, const Index, const Index, const float*,         float*);
template void gather_time_slice_cuda<__nv_bfloat16>(const Index, const Index, const Index, const Index, const __nv_bfloat16*, __nv_bfloat16*);

template<typename T>
__global__ void scatter_time_slice_kernel(const int batch,
                                          const int time_steps,
                                          const int features,
                                          const int t,
                                          const T* __restrict__ src,
                                          T* __restrict__ dst)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * features;
    if (idx >= total) return;

    const int b = idx / features;
    const int f = idx - b * features;
    dst[(b * time_steps + t) * features + f] = src[idx];
}

template<typename T>
void scatter_time_slice_cuda(const Index batch,
                             const Index time_steps,
                             const Index features,
                             const Index t,
                             const T* src,
                             T* dst)
{
    if (batch == 0 || features == 0) return;
    const int total = checked_int(batch * features);
    OPENNN_CUDA_LAUNCH(scatter_time_slice_kernel<T><<<grid_size_for(total), block_size, 0,
                                   opennn::device::get_compute_stream()>>>(
        checked_int(batch),
        checked_int(time_steps),
        checked_int(features),
        checked_int(t),
        src, dst));
}

template void scatter_time_slice_cuda<float>        (const Index, const Index, const Index, const Index, const float*,         float*);
template void scatter_time_slice_cuda<__nv_bfloat16>(const Index, const Index, const Index, const Index, const __nv_bfloat16*, __nv_bfloat16*);

template<typename T>
__global__ void transpose_2d_kernel(const int rows,
                                    const int cols,
                                    const T* __restrict__ src,
                                    T* __restrict__ dst)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * cols;
    if (idx >= total) return;

    const int r = idx / cols;
    const int c = idx - r * cols;
    dst[c * rows + r] = src[r * cols + c];
}

template<typename T>
void transpose_2d_cuda(const Index rows,
                       const Index cols,
                       const T* src,
                       T* dst)
{
    if (rows == 0 || cols == 0) return;
    const int total = checked_int(rows * cols);
    OPENNN_CUDA_LAUNCH(transpose_2d_kernel<T><<<grid_size_for(total), block_size, 0,
                             opennn::device::get_compute_stream()>>>(
        checked_int(rows),
        checked_int(cols),
        src, dst));
}

template void transpose_2d_cuda<float>        (const Index, const Index, const float*,         float*);
template void transpose_2d_cuda<__nv_bfloat16>(const Index, const Index, const __nv_bfloat16*, __nv_bfloat16*);

template<typename T>
__global__ void rnn_step_fused_forward_kernel(const int batch,
                                              const int in_features,
                                              const int out_features,
                                              const T* __restrict__ step_input,
                                              const T* __restrict__ prev_hidden,
                                              const T* __restrict__ W_in,
                                              const T* __restrict__ W_rec,
                                              const T* __restrict__ bias,
                                              T* __restrict__ step_hidden,
                                              T* derivs,
                                              const int activation_id)
{
    extern __shared__ float smem[];
    float* sX = smem;
    float* sH = smem + in_features;

    const int b = blockIdx.x;
    const int j = threadIdx.x;

    for (int i = j; i < in_features; i += blockDim.x)
        sX[i] = static_cast<float>(step_input[b * in_features + i]);

    if (prev_hidden)
        for (int k = j; k < out_features; k += blockDim.x)
            sH[k] = static_cast<float>(prev_hidden[b * out_features + k]);

    __syncthreads();

    if (j >= out_features) return;

    float z = static_cast<float>(bias[j]);

    for (int i = 0; i < in_features; ++i)
        z += sX[i] * static_cast<float>(W_in[i * out_features + j]);

    if (prev_hidden)
        for (int k = 0; k < out_features; ++k)
            z += sH[k] * static_cast<float>(W_rec[k * out_features + j]);

    float h_out;
    float dh_out;
    rnn_activation(activation_id, z, h_out, dh_out);

    step_hidden[b * out_features + j] = static_cast<T>(h_out);
    if (derivs) derivs[b * out_features + j] = static_cast<T>(dh_out);
}

template<typename T>
void rnn_step_fused_forward_cuda(const Index batch,
                                 const Index in_features,
                                 const Index out_features,
                                 const T* step_input,
                                 const T* prev_hidden,
                                 const T* W_in,
                                 const T* W_rec,
                                 const T* bias,
                                 T* step_hidden,
                                 T* derivs_or_null,
                                 const int activation_id)
{
    if (batch == 0 || out_features == 0) return;

    const int block_size = checked_int(out_features);
    const int grid_size  = checked_int(batch);
    checked_host_condition(block_size > 1024,
                           "rnn_step_fused_forward_cuda: out_features exceeds CUDA max threads per block.");
    const Index shmem_floats = in_features + (prev_hidden ? out_features : Index(0));
    const size_t shmem_bytes = static_cast<size_t>(shmem_floats) * sizeof(float);

    OPENNN_CUDA_LAUNCH(rnn_step_fused_forward_kernel<T><<<grid_size, block_size, shmem_bytes,
                                       opennn::device::get_compute_stream()>>>(
        checked_int(batch),
        checked_int(in_features),
        checked_int(out_features),
        step_input, prev_hidden, W_in, W_rec, bias,
        step_hidden, derivs_or_null, activation_id));
}

template void rnn_step_fused_forward_cuda<float>        (const Index, const Index, const Index, const float*,         const float*,         const float*,         const float*,         const float*,         float*,         float*,         const int);
template void rnn_step_fused_forward_cuda<__nv_bfloat16>(const Index, const Index, const Index, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, const int);

template<typename T>
__global__ void rnn_elementwise_multiply_kernel(const int n,
                                                T* __restrict__ dst,
                                                const T* __restrict__ a)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = static_cast<float>(dst[idx]) * static_cast<float>(a[idx]);
    dst[idx] = static_cast<T>(v);
}

template<typename T>
void rnn_elementwise_multiply_cuda(const Index n, T* dst, const T* a)
{
    if (n == 0) return;
    const int total = checked_int(n);
    OPENNN_CUDA_LAUNCH(rnn_elementwise_multiply_kernel<T><<<grid_size_for(total), block_size, 0,
                                         opennn::device::get_compute_stream()>>>(
        total, dst, a));
}

template void rnn_elementwise_multiply_cuda<float>        (const Index, float*,         const float*);
template void rnn_elementwise_multiply_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const __nv_bfloat16*);

template<typename T>
__global__ void rnn_accumulate_bias_grad_kernel(const int batch,
                                                const int features,
                                                const T* __restrict__ delta,
                                                float* __restrict__ bias_grad)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= features) return;

    float acc = 0.0f;
    for (int b = 0; b < batch; ++b)
        acc += static_cast<float>(delta[b * features + f]);

    atomicAdd(bias_grad + f, acc);
}

template<typename T>
void rnn_accumulate_bias_grad_cuda(const Index batch,
                                   const Index features,
                                   const T* delta,
                                   float* bias_grad)
{
    if (batch == 0 || features == 0) return;
    const int total = checked_int(features);
    OPENNN_CUDA_LAUNCH(rnn_accumulate_bias_grad_kernel<T><<<grid_size_for(total), block_size, 0,
                                         opennn::device::get_compute_stream()>>>(
        checked_int(batch),
        checked_int(features),
        delta, bias_grad));
}

template void rnn_accumulate_bias_grad_cuda<float>        (const Index, const Index, const float*,         float*);
template void rnn_accumulate_bias_grad_cuda<__nv_bfloat16>(const Index, const Index, const __nv_bfloat16*, float*);

template<typename T>
__global__ void rnn_step_fused_backward_pre_kernel(const int batch,
                                                   const int out_features,
                                                   const int time_steps,
                                                   const int t,
                                                   const bool first_iter,
                                                   const T* __restrict__ output_delta,
                                                   const T* __restrict__ next_carry,
                                                   const T* __restrict__ activation_derivatives,
                                                   T* __restrict__ delta)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch * out_features;
    if (idx >= total) return;

    const int b = idx / out_features;
    const int j = idx - b * out_features;

    const float dh = first_iter
        ? static_cast<float>(output_delta[idx])
        : static_cast<float>(next_carry[idx]);

    const float sigma_prime = static_cast<float>(
        activation_derivatives[(b * time_steps + t) * out_features + j]);

    const float dz = dh * sigma_prime;

    delta[idx] = static_cast<T>(dz);
}

template<typename T>
void rnn_step_fused_backward_pre_cuda(const Index batch,
                                      const Index out_features,
                                      const Index time_steps,
                                      const Index t,
                                      const bool first_iter,
                                      const T* output_delta,
                                      const T* next_carry,
                                      const T* activation_derivatives,
                                      T* delta)
{
    if (batch == 0 || out_features == 0) return;

    checked_host_condition(t < 0 || t >= time_steps,
                           "rnn_step_fused_backward_pre_cuda: time step out of range.");

    const int total = checked_int(batch * out_features);
    OPENNN_CUDA_LAUNCH(rnn_step_fused_backward_pre_kernel<T><<<grid_size_for(total), block_size, 0,
                                            opennn::device::get_compute_stream()>>>(
        checked_int(batch),
        checked_int(out_features),
        checked_int(time_steps),
        checked_int(t),
        first_iter,
        output_delta, next_carry,
        activation_derivatives,
        delta));
}

template void rnn_step_fused_backward_pre_cuda<float>        (const Index, const Index, const Index, const Index, const bool, const float*,         const float*,         const float*,         float*);
template void rnn_step_fused_backward_pre_cuda<__nv_bfloat16>(const Index, const Index, const Index, const Index, const bool, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);

#include "kernel_common.cuh"
#include <curand_kernel.h>
#include <cub/block/block_reduce.cuh>

template<typename TIn, typename TOut>
__global__ void bounding_kernel(const int n, const int features,
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
void bounding_cuda(const Index n, const int features,
                   const TIn* input, const float* lower, const float* upper,
                   TOut* output)
{
    launch_elementwise_strided(n, bounding_kernel<TIn, TOut>, features, input, lower, upper, output);
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
                y = x * stds[f];
            else
                y = (stds[f] > FLT_EPSILON) ? x / stds[f] : 0.0f;
            break;
        case 4:
            y = Inverse ? expf(x) : logf(fmaxf(x, FLT_EPSILON));
            break;
        case 5:
            y = Inverse ? x * 255.0f : x / 255.0f;
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
                TOut* output)
{
    launch_elementwise_strided(n, scale_kernel<TIn, TOut, false>, features,
                       input, minimums, maximums, means, stds, scalers,
                       min_range, max_range, output);
}

template<typename TIn, typename TOut>
void unscale_cuda(const Index n, const int features,
                  const TIn* input,
                  const float* minimums, const float* maximums,
                  const float* means, const float* stds,
                  const float* scalers,
                  const float min_range, const float max_range,
                  TOut* output)
{
    launch_elementwise_strided(n, scale_kernel<TIn, TOut, true>, features,
                       input, minimums, maximums, means, stds, scalers,
                       min_range, max_range, output);
}

template<typename TIn, typename TOut>
__global__ void scaled_diff_kernel(const int n,
                                   const TIn* __restrict__ input,
                                   const float* __restrict__ target,
                                   const float scale,
                                   TOut* __restrict__ output)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const float d = static_cast<float>(input[i]) - target[i];
        output[i] = static_cast<TOut>(scale * d);
    }
}

template<typename TIn, typename TOut>
void scaled_diff_cuda_typed(const Index n, const TIn* input, const float* target,
                            const float scale, TOut* output)
{
    launch_elementwise_strided(n, scaled_diff_kernel<TIn, TOut>, input, target, scale, output);
}

template<typename TW, typename T>
__global__ void embedding_forward_kernel(const int n, const float* __restrict__ inputs, const TW* __restrict__ weights, const float* __restrict__ positional_encoding, T* __restrict__ outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    const float scale = scale_embedding ? sqrtf(static_cast<float>(embedding_dimension)) : 1.0f;

    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const int token_index = i / embedding_dimension;
        const int dim_index = i % embedding_dimension;
        const int token_id = static_cast<int>(inputs[token_index]);

        float val = (token_id > 0 && token_id < vocabulary_size)
            ? scale * static_cast<float>(weights[token_id * embedding_dimension + dim_index])
            : 0.0f;

        if (positional_encoding != nullptr && token_id > 0)
        {
            const int seq_index = token_index % sequence_length;
            val += positional_encoding[seq_index * embedding_dimension + dim_index];
        }

        outputs[i] = static_cast<T>(val);
    }
}

template<typename TW, typename T>
void embedding_forward_cuda(const Index n, const float* inputs, const TW* weights, const float* positional_encoding, T* outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    launch_elementwise_strided(n, embedding_forward_kernel<TW, T>, inputs, weights, positional_encoding, outputs,
                       sequence_length, embedding_dimension, vocabulary_size, scale_embedding);
}

template<typename T>
__global__ void embedding_backward_kernel(const int n, const float* __restrict__ inputs, const T* __restrict__ output_deltas, float* __restrict__ weight_gradients, float* __restrict__ positional_gradients, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    const float scale = scale_embedding ? sqrtf(static_cast<float>(embedding_dimension)) : 1.0f;

    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const int token_index = i / embedding_dimension;
        const int dim_index = i % embedding_dimension;
        const int token_id = static_cast<int>(inputs[token_index]);

        if (token_id <= 0 || token_id >= vocabulary_size) continue;

        const float delta = static_cast<float>(output_deltas[i]);
        atomicAdd(&weight_gradients[token_id * embedding_dimension + dim_index], scale * delta);

        if (positional_gradients != nullptr)
        {
            const int seq_index = token_index % sequence_length;
            atomicAdd(&positional_gradients[seq_index * embedding_dimension + dim_index], delta);
        }
    }
}

template<typename T>
void embedding_backward_cuda(const Index n, const float* inputs, const T* output_deltas, float* weight_gradients, float* positional_gradients, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    launch_elementwise(n, embedding_backward_kernel<T>, inputs, output_deltas, weight_gradients, positional_gradients,
                       sequence_length, embedding_dimension, vocabulary_size, scale_embedding);
}

template<typename T>
__global__ void swap_heads_scalar_kernel(const int n, const T* __restrict__ in, T* __restrict__ out, const int P, const int Q, const int D)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
    {
        const int d = i % D;
        const int q = (i / D) % Q;
        const int p = (i / (D * Q)) % P;
        const int b = i / (D * Q * P);

        out[((int64_t(b) * Q + q) * P + p) * D + d] = in[i];
    }
}

template<typename T>
void split_heads_cuda(const Index n, const T* in, T* out, const int S, const int H, const int D)
{
    if (n == 0) return;

    if ((static_cast<size_t>(D) * sizeof(T)) % 16 == 0 && are_float4_aligned(in, out))
    {
        const int vec_width = static_cast<int>(16 / sizeof(T));
        launch_elementwise_strided(n / vec_width, swap_heads_scalar_kernel<float4>,
                                   reinterpret_cast<const float4*>(in), reinterpret_cast<float4*>(out),
                                   S, H, D / vec_width);
    }
    else
        launch_elementwise_strided(n, swap_heads_scalar_kernel<T>, in, out, S, H, D);
}

template<typename T>
void merge_heads_cuda(const Index n, const T* in, T* out, const int S, const int H, const int D)
{
    split_heads_cuda(n, in, out, H, S, D);
}

template<typename T>
__global__ void padding_mask_kernel(const int num_tokens, const T* __restrict__ source_input, T* __restrict__ padding_mask, const int embedding_dimension)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < num_tokens; i += Index(blockDim.x) * gridDim.x)
    {
        const T* token = source_input + i * embedding_dimension;
        padding_mask[i] = static_cast<T>(token_is_padding(token, embedding_dimension) ? 1.0f : 0.0f);
    }
}

template<typename T, int MAX_ELEMS>
__global__ void masked_softmax_rows_kernel(const int rows, const int source_sequence_length,
                                           const int heads_number, const int query_sequence_length,
                                           T* __restrict__ attention_weights,
                                           const T* __restrict__ padding_mask,
                                           const int use_causal_mask,
                                           const int zero_padded_queries)
{
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + (int(threadIdx.x) >> 5);
    if (row >= rows) return;

    const int lane = threadIdx.x & 31;
    const int sq = row % query_sequence_length;
    const int b  = row / (query_sequence_length * heads_number);

    T* row_values = attention_weights + Index(row) * source_sequence_length;
    const T* pad_row = padding_mask + Index(b) * source_sequence_length;

    if (zero_padded_queries && sq < source_sequence_length
        && static_cast<float>(pad_row[sq]) > 0.5f)
    {
        #pragma unroll
        for (int e = 0; e < MAX_ELEMS; ++e)
        {
            const int sk = lane + e * 32;
            if (sk < source_sequence_length)
                row_values[sk] = static_cast<T>(0.0f);
        }
        return;
    }

    float values[MAX_ELEMS];
    float row_max = -1e30f;

    #pragma unroll
    for (int e = 0; e < MAX_ELEMS; ++e)
    {
        const int sk = lane + e * 32;
        float value = -INFINITY;
        if (sk < source_sequence_length)
        {
            const bool masked = (use_causal_mask && sk > sq)
                             || static_cast<float>(pad_row[sk]) > 0.5f;
            value = masked ? -1e9f : static_cast<float>(row_values[sk]);
        }
        values[e] = value;
        row_max = fmaxf(row_max, value);
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        row_max = fmaxf(row_max, __shfl_xor_sync(0xffffffff, row_max, offset));

    float row_sum = 0.0f;
    #pragma unroll
    for (int e = 0; e < MAX_ELEMS; ++e)
    {
        values[e] = expf(values[e] - row_max);
        row_sum += values[e];
    }

    for (int offset = 16; offset > 0; offset >>= 1)
        row_sum += __shfl_xor_sync(0xffffffff, row_sum, offset);

    const float inv_row_sum = 1.0f / row_sum;

    #pragma unroll
    for (int e = 0; e < MAX_ELEMS; ++e)
    {
        const int sk = lane + e * 32;
        if (sk < source_sequence_length)
            row_values[sk] = static_cast<T>(values[e] * inv_row_sum);
    }
}

template<typename T>
static void launch_masked_softmax_rows(const int batch_size, const int heads_number,
                                       const int query_sequence_length, const int source_sequence_length,
                                       T* attention_weights, const T* padding_mask,
                                       const bool use_causal_mask, const bool zero_padded_queries,
                                       cudaStream_t stream)
{
    const int rows = batch_size * heads_number * query_sequence_length;
    if (rows <= 0 || source_sequence_length <= 0) return;

    constexpr int threads = 128;
    constexpr int warps_per_block = threads / 32;
    const int blocks = (rows + warps_per_block - 1) / warps_per_block;
    const int causal = use_causal_mask ? 1 : 0;
    const int zero_queries = zero_padded_queries ? 1 : 0;

    const auto launch = [&](auto elems_tag)
    {
        constexpr int ELEMS = decltype(elems_tag)::value;
        OPENNN_CUDA_LAUNCH(masked_softmax_rows_kernel<T, ELEMS><<<blocks, threads, 0, stream>>>(
            rows, source_sequence_length, heads_number, query_sequence_length,
            attention_weights, padding_mask, causal, zero_queries));
    };

    const int elems = (source_sequence_length + 31) / 32;
    if      (elems <= 4)  launch(std::integral_constant<int, 4>{});
    else if (elems <= 8)  launch(std::integral_constant<int, 8>{});
    else if (elems <= 16) launch(std::integral_constant<int, 16>{});
    else if (elems <= 32) launch(std::integral_constant<int, 32>{});
    else if (elems <= 64) launch(std::integral_constant<int, 64>{});
    else
        throw std::runtime_error("masked softmax: source sequence length above 2048 is not supported.");
}

template<typename T>
void attention_masked_softmax_cuda(const int batch_size, const int heads_number,
                          const int query_sequence_length, const int source_sequence_length,
                          const int embedding_dimension, const T* source_input,
                          T* attention_weights, T* padding_mask, const bool use_causal_mask,
                          const bool zero_padded_queries)
{
    launch_elementwise_strided(Index(batch_size) * source_sequence_length, padding_mask_kernel<T>,
                       source_input, padding_mask, embedding_dimension);

    launch_masked_softmax_rows<T>(batch_size, heads_number,
                                  query_sequence_length, source_sequence_length,
                                  attention_weights, padding_mask, use_causal_mask,
                                  zero_padded_queries,
                                  opennn::device::get_compute_stream());
}

template<typename T>
__global__ void length_to_padding_mask_kernel(const int n, const int source_sequence_length,
                                              const int* __restrict__ lengths, T* __restrict__ padding_mask)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int b = i / source_sequence_length;
        const int s = i % source_sequence_length;
        padding_mask[i] = static_cast<T>(s >= lengths[b] ? 1.0f : 0.0f);
    }
}

template<typename T>
void attention_length_masked_softmax_cuda(const int batch_size, const int heads_number,
                                const int query_sequence_length, const int source_sequence_length,
                                const int* device_lengths, T* attention_weights, T* padding_mask,
                                const bool use_causal_mask, const bool zero_padded_queries)
{
    if (batch_size == 0) return;

    launch_elementwise_strided(Index(batch_size) * source_sequence_length, length_to_padding_mask_kernel<T>,
                       source_sequence_length, device_lengths, padding_mask);

    launch_masked_softmax_rows<T>(batch_size, heads_number,
                                  query_sequence_length, source_sequence_length,
                                  attention_weights, padding_mask, use_causal_mask,
                                  zero_padded_queries, opennn::device::get_compute_stream());
}

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
            if (fabsf(static_cast<float>(token[e])) > padding_epsilon) { nonzero = true; break; }

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
};

}

static float* get_pooling_scratch(size_t floats_needed)
{
    checked_host_condition(
        floats_needed > static_cast<size_t>(std::numeric_limits<Index>::max()),
        "pooling scratch size exceeds Index range.");
    // Immortal on purpose: a static destructor would free device memory after
    // the CUDA context may already be gone; the driver reclaims it at exit.
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

    launch_elementwise_strided(n, average_pooling_3d_backward_kernel<T>, delta, in_gradient, S, F, valid_mask, counts);
}

// Forward gathers each sample's first-token features; backward scatters the
// delta back into the first token's slot.
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

__device__ __forceinline__ void warp_reduce_sum2(float& a, float& b)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        a += __shfl_down_sync(0xffffffff, a, offset);
        b += __shfl_down_sync(0xffffffff, b, offset);
    }
}

__device__ __forceinline__ bool block_reduce_sum2(float& a, float& b)
{
    warp_reduce_sum2(a, b);

    __shared__ float warp_a[32];
    __shared__ float warp_b[32];

    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0)
    {
        warp_a[warp_id] = a;
        warp_b[warp_id] = b;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + 31) >> 5;
    if (warp_id == 0)
    {
        a = (threadIdx.x < num_warps) ? warp_a[threadIdx.x] : 0.0f;
        b = (threadIdx.x < num_warps) ? warp_b[threadIdx.x] : 0.0f;
        warp_reduce_sum2(a, b);
    }
    return threadIdx.x == 0;
}

template<typename T, bool FuseResidual, bool HasMean>
__global__ void norm_forward_kernel(const int N, const int D, const T* __restrict__ X, const T* __restrict__ R, T* __restrict__ sum, T* __restrict__ Y, float* __restrict__ means, float* __restrict__ inv_vars, const float* __restrict__ gamma, const float* __restrict__ beta, const float eps)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const T* x_row = X + idx * D;
    T* y_row = Y + idx * D;
    T* s_row = FuseResidual ? sum + idx * D : nullptr;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        float x;
        if constexpr (FuseResidual)
        {
            x = static_cast<float>(x_row[i]) + static_cast<float>(R[idx * D + i]);
            s_row[i] = static_cast<T>(x);
        }
        else
            x = static_cast<float>(x_row[i]);

        if constexpr (HasMean) local_sum += x;
        local_sum_sq += x * x;
    }

    __shared__ float s_mean;
    __shared__ float s_inv_var;

    if (block_reduce_sum2(local_sum, local_sum_sq))
    {
        const float inv_D = 1.0f / static_cast<float>(D);
        if constexpr (HasMean)
        {
            const float mean = local_sum * inv_D;

            const float variance = fmaxf(local_sum_sq * inv_D - mean * mean, 0.0f);
            const float inv_var = rsqrtf(variance + eps);
            s_mean    = mean;
            s_inv_var = inv_var;
            means[idx]    = mean;
            inv_vars[idx] = inv_var;
        }
        else
        {
            const float inv_var = rsqrtf(local_sum_sq * inv_D + eps);
            s_inv_var = inv_var;
            if (inv_vars) inv_vars[idx] = inv_var;
        }
    }
    __syncthreads();

    const float inv_var = s_inv_var;
    float mean = 0.0f;
    if constexpr (HasMean) mean = s_mean;

    const T* src_row = FuseResidual ? s_row : x_row;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        if constexpr (HasMean)
        {
            const float x_hat = (static_cast<float>(src_row[i]) - mean) * inv_var;
            y_row[i] = static_cast<T>(fmaf(gamma[i], x_hat, beta[i]));
        }
        else
        {
            const float x_hat = static_cast<float>(src_row[i]) * inv_var;
            y_row[i] = static_cast<T>(gamma[i] * x_hat);
        }
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
__global__ void batchnorm_inference_kernel(const Index total, const int channels,
                                           const T* __restrict__ x,
                                           const T* __restrict__ residual,
                                           const float* __restrict__ gamma,
                                           const float* __restrict__ beta,
                                           const float* __restrict__ mean,
                                           const float* __restrict__ variance,
                                           const float epsilon,
                                           const int apply_relu,
                                           T* __restrict__ y)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += Index(blockDim.x) * gridDim.x)
    {
        const int c = int(i % channels);
        const float scale = gamma[c] * rsqrtf(variance[c] + epsilon);
        float value = (static_cast<float>(x[i]) - mean[c]) * scale + beta[c];
        if (residual) value += static_cast<float>(residual[i]);
        if (apply_relu) value = fmaxf(value, 0.0f);
        y[i] = static_cast<T>(value);
    }
}

template<typename T>
void batchnorm_inference_cuda(const Index total, const Index channels,
                              const T* x, const T* residual,
                              const float* gamma, const float* beta,
                              const float* mean, const float* variance,
                              const float epsilon, const bool apply_relu, T* y)
{
    if (channels == 0) return;
    launch_elementwise_strided(total, batchnorm_inference_kernel<T>, checked_int(channels),
                       x, residual, gamma, beta, mean, variance,
                       epsilon, apply_relu ? 1 : 0, y);
}

// Folds BN into pointwise (1x1) conv weights, transposing to {kernel_size, kernels} GEMM layout.
__global__ void conv_bn_fold_kernel(const Index total, const int kernel_size, const int kernels,
                                    const float* __restrict__ weights,
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ beta,
                                    const float* __restrict__ mean,
                                    const float* __restrict__ variance,
                                    const float epsilon,
                                    float* __restrict__ folded_weights,
                                    float* __restrict__ folded_bias)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += Index(blockDim.x) * gridDim.x)
    {
        const int k = int(i / kernel_size);
        const int r = int(i % kernel_size);
        const float scale = gamma[k] * rsqrtf(variance[k] + epsilon);
        folded_weights[Index(r) * kernels + k] = weights[i] * scale;
        if (r == 0)
            folded_bias[k] = beta[k] - mean[k] * scale;
    }
}

void conv_bn_fold_cuda(const Index kernels, const Index kernel_size,
                       const float* weights,
                       const float* gamma, const float* beta,
                       const float* mean, const float* variance,
                       const float epsilon,
                       float* folded_weights, float* folded_bias)
{
    launch_elementwise_strided(kernels * kernel_size, conv_bn_fold_kernel,
                       checked_int(kernel_size), checked_int(kernels), weights,
                       gamma, beta, mean, variance, epsilon,
                       folded_weights, folded_bias);
}

__global__ void add_relu_kernel(const Index total,
                                const float* __restrict__ a,
                                const float* __restrict__ b,
                                const int apply_relu,
                                float* __restrict__ y)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += Index(blockDim.x) * gridDim.x)
    {
        const float value = a[i] + b[i];
        y[i] = apply_relu ? fmaxf(value, 0.0f) : value;
    }
}

void add_relu_cuda(const Index total, const float* a, const float* b,
                   const bool apply_relu, float* y)
{
    launch_elementwise_strided(total, add_relu_kernel, a, b, apply_relu ? 1 : 0, y);
}

template<typename T>
void layernorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps)
{
    if (N == 0 || D == 0) return;

    OPENNN_CUDA_LAUNCH((norm_forward_kernel<T, false, true><<<N, layernorm_threads(D), 0, opennn::device::get_compute_stream()>>>(N, D, X, nullptr, nullptr, Y, means, inv_vars, gamma, beta, eps)));
}

template<typename T>
void layernorm_add_forward_cuda(const int N, const int D, const T* X, const T* R, T* sum, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps)
{
    if (N == 0 || D == 0) return;

    OPENNN_CUDA_LAUNCH((norm_forward_kernel<T, true, true><<<N, layernorm_threads(D), 0, opennn::device::get_compute_stream()>>>(N, D, X, R, sum, Y, means, inv_vars, gamma, beta, eps)));
}

template<typename T, bool HasMean>
__global__ void norm_backward_kernel(const int N, const int D, const T* __restrict__ dY, const T* __restrict__ X, const float* __restrict__ means, const float* __restrict__ inv_vars, const float* __restrict__ gamma, T* __restrict__ dX)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const T* dy_row = dY + idx * D;
    const T* x_row = X + idx * D;
    T* dx_row = dX + idx * D;

    float mean = 0.0f;
    if constexpr (HasMean) mean = means[idx];
    const float inv_var = inv_vars[idx];

    float local_sum_D      = 0.0f;
    float local_sum_D_xhat = 0.0f;

    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float d     = static_cast<float>(dy_row[i]) * gamma[i];
        float x_hat;
        if constexpr (HasMean) x_hat = (static_cast<float>(x_row[i]) - mean) * inv_var;
        else                   x_hat = static_cast<float>(x_row[i]) * inv_var;
        if constexpr (HasMean) local_sum_D += d;
        local_sum_D_xhat += d * x_hat;
    }

    __shared__ float s_mean_D;
    __shared__ float s_mean_D_xhat;

    if (block_reduce_sum2(local_sum_D, local_sum_D_xhat))
    {
        const float inv_D = 1.0f / static_cast<float>(D);
        if constexpr (HasMean) s_mean_D = local_sum_D * inv_D;
        s_mean_D_xhat = local_sum_D_xhat * inv_D;
    }
    __syncthreads();

    float mean_D = 0.0f;
    if constexpr (HasMean) mean_D = s_mean_D;
    const float mean_D_xhat = s_mean_D_xhat;

    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float d     = static_cast<float>(dy_row[i]) * gamma[i];
        float x_hat;
        if constexpr (HasMean) x_hat = (static_cast<float>(x_row[i]) - mean) * inv_var;
        else                   x_hat = static_cast<float>(x_row[i]) * inv_var;
        if constexpr (HasMean)
            dx_row[i] = static_cast<T>((d - mean_D - x_hat * mean_D_xhat) * inv_var);
        else
            dx_row[i] = static_cast<T>((d - x_hat * mean_D_xhat) * inv_var);
    }
}

template<typename T, int NUM_WARPS, bool HasMean>
__global__ void norm_weight_gradient_coalesced_kernel(const int N, const int D,
                                                      const int chunk,
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
    const int n0      = blockIdx.y * chunk;
    const int n1      = min(N, n0 + chunk);

    float local_gamma = 0.0f;
    float local_beta  = 0.0f;

    if (active)
    {
        for (int n = n0 + warp_id; n < n1; n += NUM_WARPS)
        {
            const float dy    = static_cast<float>(dY[n * D + d]);
            float x_hat;
            if constexpr (HasMean) x_hat = (static_cast<float>(X[n * D + d]) - means[n]) * inv_vars[n];
            else                   x_hat = static_cast<float>(X[n * D + d]) * inv_vars[n];
            local_gamma += dy * x_hat;
            if constexpr (HasMean) local_beta += dy;
        }
    }

    __shared__ float partial_gamma[NUM_WARPS][32];
    __shared__ float partial_beta [HasMean ? NUM_WARPS : 1][32];

    partial_gamma[warp_id][lane] = local_gamma;
    if constexpr (HasMean) partial_beta[warp_id][lane] = local_beta;
    __syncthreads();

    if (warp_id == 0 && active)
    {
        float g = 0.0f;
        float b = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w)
        {
            g += partial_gamma[w][lane];
            if constexpr (HasMean) b += partial_beta[w][lane];
        }
        if (gridDim.y == 1)
        {
            dGamma[d] = g;
            if constexpr (HasMean) dBeta[d] = b;
        }
        else
        {
            atomicAdd(dGamma + d, g);
            if constexpr (HasMean) atomicAdd(dBeta + d, b);
        }
    }
}

template<typename T, bool HasMean>
static void norm_backward_launch(const int N, const int D, const T* dY, const T* X, const float* means, const float* inv_vars, const float* gamma, T* dX, float* dGamma, float* dBeta)
{
    if (dX)
        OPENNN_CUDA_LAUNCH((norm_backward_kernel<T, HasMean><<<N, layernorm_threads(D), 0, opennn::device::get_compute_stream()>>>(N, D, dY, X, means, inv_vars, gamma, dX)));

    constexpr int NUM_WARPS = 8;
    const dim3 block(32, NUM_WARPS);
    const int grid_x = (D + 31) / 32;

    const int desired_chunks = grid_x < 192 ? 192 / grid_x : 1;
    int chunk = ceil_div(N, desired_chunks);
    if (chunk < NUM_WARPS * 8) chunk = NUM_WARPS * 8;
    const int grid_y = ceil_div(N, chunk);
    if (grid_y > 1)
    {
        const cudaStream_t stream = opennn::device::get_compute_stream();
        cudaMemsetAsync(dGamma, 0, size_t(D) * sizeof(float), stream);
        if constexpr (HasMean) cudaMemsetAsync(dBeta, 0, size_t(D) * sizeof(float), stream);
    }
    norm_weight_gradient_coalesced_kernel<T, NUM_WARPS, HasMean><<<dim3(grid_x, grid_y), block, 0,
        opennn::device::get_compute_stream()>>>(N, D, chunk, dY, X, means, inv_vars, dGamma, dBeta);
    opennn::device::check_last_error();
}

template<typename T>
void layernorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* means, const float* inv_vars, const float* gamma, T* dX, float* dGamma, float* dBeta)
{
    if (N == 0 || D == 0) return;

    norm_backward_launch<T, true>(N, D, dY, X, means, inv_vars, gamma, dX, dGamma, dBeta);
}

template<typename T>
void rmsnorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* inv_rms, const float* weight, const float eps)
{
    if (N == 0 || D == 0) return;

    OPENNN_CUDA_LAUNCH((norm_forward_kernel<T, false, false><<<N, layernorm_threads(D), 0, opennn::device::get_compute_stream()>>>(N, D, X, nullptr, nullptr, Y, nullptr, inv_rms, weight, nullptr, eps)));
}

template<typename T>
void rmsnorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* inv_rms, const float* weight, T* dX, float* dWeight)
{
    if (N == 0 || D == 0) return;

    norm_backward_launch<T, false>(N, D, dY, X, nullptr, inv_rms, weight, dX, dWeight, nullptr);
}

template<typename T, int SIGN>
__global__ void rope_apply_kernel(const int rows, const int seq, const int model_dim, const int head_dim, const int rotary_dim, const int offset, const T* __restrict__ in, T* __restrict__ out, const float* __restrict__ cos, const float* __restrict__ sin)
{
    const int row = blockIdx.x;
    if (row >= rows) return;

    const int pos  = (row % seq) + offset;
    const int half = rotary_dim >> 1;
    const float* cr = cos + pos * rotary_dim;
    const float* sr = sin + pos * rotary_dim;

    const int row_base = row * model_dim;

    for (int e = threadIdx.x; e < model_dim; e += blockDim.x)
    {
        const int d = e % head_dim;
        const int base_e = row_base + e;

        if (d < rotary_dim)
        {
            const int head_start = base_e - d;
            const float partner = (d < half)
                ? -static_cast<float>(in[head_start + d + half])
                :  static_cast<float>(in[head_start + d - half]);
            out[base_e] = static_cast<T>(static_cast<float>(in[base_e]) * cr[d] + SIGN * partner * sr[d]);
        }
        else
        {
            out[base_e] = in[base_e];
        }
    }
}

static inline int rope_threads(int model_dim)
{
    if (model_dim <= 64)  return 64;
    if (model_dim <= 128) return 128;
    return 256;
}

template<typename T>
void rope_forward_cuda(const int rows, const int seq, const int model_dim, const int head_dim, const int rotary_dim, const int offset, const T* in, T* out, const float* cos, const float* sin)
{
    if (rows == 0 || model_dim == 0) return;

    OPENNN_CUDA_LAUNCH((rope_apply_kernel<T, 1><<<rows, rope_threads(model_dim), 0, opennn::device::get_compute_stream()>>>(rows, seq, model_dim, head_dim, rotary_dim, offset, in, out, cos, sin)));
}

template<typename T>
void rope_backward_cuda(const int rows, const int seq, const int model_dim, const int head_dim, const int rotary_dim, const int offset, const T* dout, T* din, const float* cos, const float* sin)
{
    if (rows == 0 || model_dim == 0) return;

    OPENNN_CUDA_LAUNCH((rope_apply_kernel<T, -1><<<rows, rope_threads(model_dim), 0, opennn::device::get_compute_stream()>>>(rows, seq, model_dim, head_dim, rotary_dim, offset, dout, din, cos, sin)));
}

template<typename T>
__global__ void qk_rope_cache_append_kernel(const int n_q_heads, const int n_kv_heads, const int head_dim,
                                            const float eps, const int* __restrict__ position,
                                            const T* __restrict__ qkv,
                                            const float* __restrict__ q_norm_w, const float* __restrict__ k_norm_w,
                                            const float* __restrict__ cos_table, const float* __restrict__ sin_table,
                                            T* __restrict__ q_out, T* __restrict__ k_cache, T* __restrict__ v_cache)
{
    const int h      = blockIdx.x;
    const int pos    = *position;
    const int kv_dim = n_kv_heads * head_dim;
    const T* src     = qkv + size_t(h) * head_dim;

    if (h >= n_q_heads + n_kv_heads)
    {
        T* dst = v_cache + size_t(pos) * kv_dim + size_t(h - n_q_heads - n_kv_heads) * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) dst[d] = src[d];
        return;
    }

    const bool is_q = h < n_q_heads;
    const float* norm_w = is_q ? q_norm_w : k_norm_w;
    T* dst = is_q ? q_out + size_t(h) * head_dim
                  : k_cache + size_t(pos) * kv_dim + size_t(h - n_q_heads) * head_dim;

    extern __shared__ float vals[];

    float local_sum_sq = 0.0f;
    float ignore = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        const float x = static_cast<float>(src[d]);
        local_sum_sq += x * x;
    }

    __shared__ float s_inv_rms;
    if (block_reduce_sum2(local_sum_sq, ignore))
        s_inv_rms = rsqrtf(local_sum_sq / static_cast<float>(head_dim) + eps);
    __syncthreads();

    const float inv = s_inv_rms;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
        vals[d] = static_cast<float>(src[d]) * inv * (norm_w ? norm_w[d] : 1.0f);
    __syncthreads();

    const int half   = head_dim >> 1;
    const float* cr  = cos_table + size_t(pos) * head_dim;
    const float* sr  = sin_table + size_t(pos) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x)
    {
        const float partner = d < half ? -vals[d + half] : vals[d - half];
        dst[d] = static_cast<T>(vals[d] * cr[d] + partner * sr[d]);
    }
}

template<typename T>
void qk_rope_cache_append_cuda(const int n_q_heads, const int n_kv_heads, const int head_dim,
                               const float eps, const int* position,
                               const T* qkv, const float* q_norm_w, const float* k_norm_w,
                               const float* cos_table, const float* sin_table,
                               T* q_out, T* k_cache, T* v_cache)
{
    const int blocks = n_q_heads + 2 * n_kv_heads;
    const int threads = rope_threads(head_dim);
    const int smem = head_dim * int(sizeof(float));
    OPENNN_CUDA_LAUNCH((qk_rope_cache_append_kernel<T><<<blocks, threads, smem, opennn::device::get_compute_stream()>>>(
        n_q_heads, n_kv_heads, head_dim, eps, position, qkv, q_norm_w, k_norm_w,
        cos_table, sin_table, q_out, k_cache, v_cache)));
}

template<typename T>
__global__ void swiglu_forward_kernel(const int n, const T* __restrict__ gate, const T* __restrict__ up, T* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float g = static_cast<float>(gate[i]);
    const float silu = g / (1.0f + expf(-g));
    out[i] = static_cast<T>(silu * static_cast<float>(up[i]));
}

template<typename T>
__global__ void swiglu_backward_kernel(const int n, const T* __restrict__ dout, const T* __restrict__ gate, const T* __restrict__ up, T* __restrict__ dgate, T* __restrict__ dup)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float d   = static_cast<float>(dout[i]);
    const float g   = static_cast<float>(gate[i]);
    const float sig = 1.0f / (1.0f + expf(-g));
    const float silu = g * sig;

    if (dup)   dup[i]   = static_cast<T>(d * silu);
    if (dgate) dgate[i] = static_cast<T>(d * static_cast<float>(up[i]) * sig * (1.0f + g * (1.0f - sig)));
}

template<typename T>
void swiglu_forward_cuda(const int n, const T* gate, const T* up, T* out)
{
    launch_elementwise(n, swiglu_forward_kernel<T>, gate, up, out);
}

template<typename T>
void swiglu_backward_cuda(const int n, const T* dout, const T* gate, const T* up, T* dgate, T* dup)
{
    launch_elementwise(n, swiglu_backward_kernel<T>, dout, gate, up, dgate, dup);
}

template<typename T>
__global__ void grouped_attention_kernel(const int total_queries, const int query_seq, const int key_seq,
                                          const int n_query_heads, const int n_kv_heads, const int head_dim,
                                          const int group, const float scale, const int qoffset, const int causal,
                                          const T* __restrict__ Q, const T* __restrict__ K,
                                          const T* __restrict__ V, T* __restrict__ O)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_queries) return;

    const int i   = idx % query_seq;
    const int hq  = (idx / query_seq) % n_query_heads;
    const int b   = idx / (query_seq * n_query_heads);
    const int hkv = hq / group;
    const int valid = causal ? min(qoffset + i + 1, key_seq) : key_seq;

    const T* q_vec = Q + ((size_t(b) * query_seq + i) * n_query_heads + hq) * head_dim;
    T*       o_vec = O + ((size_t(b) * query_seq + i) * n_query_heads + hq) * head_dim;

    float acc[256];
    for (int d = 0; d < head_dim; ++d) acc[d] = 0.0f;
    float m = -1e30f, l = 0.0f;

    for (int j = 0; j < valid; ++j) {
        const T* k_vec = K + ((size_t(b) * key_seq + j) * n_kv_heads + hkv) * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; ++d) s += static_cast<float>(q_vec[d]) * static_cast<float>(k_vec[d]);
        s *= scale;

        const float m_new = fmaxf(m, s);
        const float corr  = __expf(m - m_new);
        const float p     = __expf(s - m_new);
        l = l * corr + p;

        const T* v_vec = V + ((size_t(b) * key_seq + j) * n_kv_heads + hkv) * head_dim;
        for (int d = 0; d < head_dim; ++d) acc[d] = acc[d] * corr + p * static_cast<float>(v_vec[d]);
        m = m_new;
    }

    const float inv_l = 1.0f / l;
    for (int d = 0; d < head_dim; ++d) o_vec[d] = static_cast<T>(acc[d] * inv_l);
}

template<typename T>
__global__ void grouped_attention_softmax_kernel(const int rows, const int query_seq, const int key_seq,
                                                 const int qoffset, const int causal,
                                                 const float* __restrict__ scores, T* __restrict__ probs)
{
    const int warps_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * warps_per_block + (int(threadIdx.x) >> 5);
    if (row >= rows) return;

    const int lane = threadIdx.x & 31;
    const int i = row % query_seq;
    const int valid = causal ? min(qoffset + i + 1, key_seq) : key_seq;

    const float* s_row = scores + size_t(row) * key_seq;
    T* p_row = probs + size_t(row) * key_seq;

    float m = -1e30f;
    for (int j = lane; j < valid; j += 32) m = fmaxf(m, s_row[j]);
    for (int offset = 16; offset > 0; offset >>= 1)
        m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, offset));

    float l = 0.0f;
    for (int j = lane; j < valid; j += 32) l += __expf(s_row[j] - m);
    for (int offset = 16; offset > 0; offset >>= 1)
        l += __shfl_xor_sync(0xffffffffu, l, offset);

    const float inv_l = 1.0f / l;
    for (int j = lane; j < key_seq; j += 32)
        p_row[j] = static_cast<T>(j < valid ? __expf(s_row[j] - m) * inv_l : 0.0f);
}

template<typename T>
void grouped_attention_softmax_cuda(const int rows, const int query_seq, const int key_seq,
                                    const int query_position_offset, const bool causal,
                                    const float* scores, T* probs)
{
    if (rows <= 0 || key_seq <= 0) return;

    constexpr int threads = 128;
    const int blocks = (rows + (threads / 32) - 1) / (threads / 32);
    OPENNN_CUDA_LAUNCH((grouped_attention_softmax_kernel<T><<<blocks, threads, 0, opennn::device::get_compute_stream()>>>(
        rows, query_seq, key_seq, query_position_offset, causal ? 1 : 0, scores, probs)));
}

template<typename T, int N>
__device__ __forceinline__ void load_head_fragment(const T* __restrict__ p, float* out)
{
    if constexpr (std::is_same_v<T, __nv_bfloat16> && N % 2 == 0)
    {
        const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(p);
        #pragma unroll
        for (int i = 0; i < N / 2; ++i)
        {
            const float2 f = __bfloat1622float2(p2[i]);
            out[2 * i] = f.x; out[2 * i + 1] = f.y;
        }
    }
    else
    {
        #pragma unroll
        for (int i = 0; i < N; ++i) out[i] = static_cast<float>(p[i]);
    }
}

template<typename T, int HEAD_DIM, int GROUP>
__global__ void grouped_attention_decode_kernel(const int n_kv_heads, const float scale,
                                                const int* __restrict__ position_device, const int kv_length_host,
                                                const T* __restrict__ Q, const T* __restrict__ K,
                                                const T* __restrict__ V, float* __restrict__ partials)
{
    constexpr int FRAG = HEAD_DIM / 32;

    const int hkv      = blockIdx.x;
    const int lane     = threadIdx.x & 31;
    const int warp     = threadIdx.x >> 5;
    const int warps    = blockDim.x >> 5;
    const int split    = blockIdx.y * warps + warp;
    const int n_splits = gridDim.y * warps;
    const int valid    = position_device ? *position_device + 1 : kv_length_host;

    float q[GROUP][FRAG];
    #pragma unroll
    for (int g = 0; g < GROUP; ++g)
        load_head_fragment<T, FRAG>(Q + (size_t(hkv) * GROUP + g) * HEAD_DIM + lane * FRAG, q[g]);

    float m[GROUP], l[GROUP], acc[GROUP][FRAG];
    #pragma unroll
    for (int g = 0; g < GROUP; ++g)
    {
        m[g] = -1e30f; l[g] = 0.0f;
        #pragma unroll
        for (int f = 0; f < FRAG; ++f) acc[g][f] = 0.0f;
    }

    for (int j = split; j < valid; j += n_splits)
    {
        const size_t row = (size_t(j) * n_kv_heads + hkv) * HEAD_DIM + size_t(lane) * FRAG;

        float k_frag[FRAG];
        load_head_fragment<T, FRAG>(K + row, k_frag);

        float s[GROUP];
        #pragma unroll
        for (int g = 0; g < GROUP; ++g)
        {
            float dot = 0.0f;
            #pragma unroll
            for (int f = 0; f < FRAG; ++f) dot += q[g][f] * k_frag[f];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, offset);
            s[g] = dot * scale;
        }

        float v_frag[FRAG];
        load_head_fragment<T, FRAG>(V + row, v_frag);

        #pragma unroll
        for (int g = 0; g < GROUP; ++g)
        {
            const float m_new = fmaxf(m[g], s[g]);
            const float corr  = __expf(m[g] - m_new);
            const float p     = __expf(s[g] - m_new);
            l[g] = l[g] * corr + p;
            m[g] = m_new;
            #pragma unroll
            for (int f = 0; f < FRAG; ++f) acc[g][f] = acc[g][f] * corr + p * v_frag[f];
        }
    }

    #pragma unroll
    for (int g = 0; g < GROUP; ++g)
    {
        float* slot = partials + ((size_t(hkv) * n_splits + split) * GROUP + g) * (HEAD_DIM + 2);
        #pragma unroll
        for (int f = 0; f < FRAG; ++f) slot[lane * FRAG + f] = acc[g][f];
        if (lane == 0) { slot[HEAD_DIM] = m[g]; slot[HEAD_DIM + 1] = l[g]; }
    }
}

template<typename T>
__global__ void grouped_attention_decode_combine_kernel(const int group, const int head_dim, const int n_splits,
                                                        const float* __restrict__ partials, T* __restrict__ O)
{
    const int hq  = blockIdx.x;
    const int hkv = hq / group;
    const int g   = hq % group;
    const int d   = threadIdx.x;

    extern __shared__ float sm[];
    float* sm_m = sm;
    float* sm_l = sm + n_splits;

    const float* base = partials + (size_t(hkv) * n_splits * group + g) * (head_dim + 2);
    const size_t stride = size_t(group) * (head_dim + 2);

    for (int s = threadIdx.x; s < n_splits; s += blockDim.x)
    {
        sm_m[s] = base[s * stride + head_dim];
        sm_l[s] = base[s * stride + head_dim + 1];
    }
    __syncthreads();

    float M = -1e30f;
    for (int s = 0; s < n_splits; ++s) if (sm_l[s] > 0.0f) M = fmaxf(M, sm_m[s]);

    float L = 0.0f, out = 0.0f;
    for (int s = 0; s < n_splits; ++s)
    {
        if (sm_l[s] <= 0.0f) continue;
        const float e = __expf(sm_m[s] - M);
        L += sm_l[s] * e;
        out += e * base[s * stride + d];
    }

    O[size_t(hq) * head_dim + d] = static_cast<T>(out / L);
}

// Both sampling kernels launch with exactly SAMPLING_BLOCK_THREADS threads.
constexpr int SAMPLING_BLOCK_THREADS = 256;
using BlockArgMaxReduce = cub::BlockReduce<cub::KeyValuePair<int, float>, SAMPLING_BLOCK_THREADS>;

// cub::ArgMax resolves ties toward the lower index; broadcast the winner to all threads.
__device__ __forceinline__ void block_argmax(float& v, int& i,
                                             typename BlockArgMaxReduce::TempStorage& temp,
                                             cub::KeyValuePair<int, float>& winner)
{
    const auto best = BlockArgMaxReduce(temp).Reduce(cub::KeyValuePair<int, float>(i, v), cub::ArgMax());
    if (threadIdx.x == 0) winner = best;
    __syncthreads();
    v = winner.value; i = winner.key;
    __syncthreads();
}

template<typename T, int SLOTS>
__global__ void logits_top_candidates_kernel(const int n, const int k, const T* __restrict__ logits,
                                             float2* __restrict__ out)
{
    const int stride = gridDim.x * blockDim.x;

    float v[SLOTS]; int vi[SLOTS];
    #pragma unroll
    for (int j = 0; j < SLOTS; ++j) { v[j] = -1e30f; vi[j] = 0x7fffffff; }

    int cnt = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n && cnt < SLOTS; i += stride)
    {
        if (i == 0) continue;
        v[cnt] = static_cast<float>(logits[i]); vi[cnt] = i; ++cnt;
    }

    __shared__ typename BlockArgMaxReduce::TempStorage sm_argmax;
    __shared__ cub::KeyValuePair<int, float> sm_winner;

    for (int round = 0; round < k; ++round)
    {
        float best = -1e30f; int besti = 0x7fffffff, slot = -1;
        #pragma unroll
        for (int j = 0; j < SLOTS; ++j)
            if (v[j] > best || (v[j] == best && vi[j] < besti)) { best = v[j]; besti = vi[j]; slot = j; }

        float wv = best; int wi = besti;
        block_argmax(wv, wi, sm_argmax, sm_winner);

        if (threadIdx.x == 0) out[blockIdx.x * k + round] = make_float2(wv, __int_as_float(wi));
        if (slot >= 0 && besti == wi) v[slot] = -1e30f;
        __syncthreads();
    }
}

template<int SLOTS>
__global__ void sample_from_candidates_kernel(const int m, const int k,
                                              const float temperature, const float top_p,
                                              const unsigned long long seed, const unsigned long long step,
                                              const float2* __restrict__ candidates,
                                              int* __restrict__ id_out, float* __restrict__ token_out)
{
    float v[SLOTS]; int vi[SLOTS];
    #pragma unroll
    for (int j = 0; j < SLOTS; ++j) { v[j] = -1e30f; vi[j] = 0x7fffffff; }

    int cnt = 0;
    for (int i = threadIdx.x; i < m && cnt < SLOTS; i += blockDim.x)
    {
        const float2 c = candidates[i];
        v[cnt] = c.x; vi[cnt] = __float_as_int(c.y); ++cnt;
    }

    __shared__ typename BlockArgMaxReduce::TempStorage sm_argmax;
    __shared__ cub::KeyValuePair<int, float> sm_winner;
    __shared__ float top_v[32];
    __shared__ int   top_i[32];

    for (int round = 0; round < k; ++round)
    {
        float best = -1e30f; int besti = 0x7fffffff, slot = -1;
        #pragma unroll
        for (int j = 0; j < SLOTS; ++j)
            if (v[j] > best || (v[j] == best && vi[j] < besti)) { best = v[j]; besti = vi[j]; slot = j; }

        float wv = best; int wi = besti;
        block_argmax(wv, wi, sm_argmax, sm_winner);

        if (threadIdx.x == 0) { top_v[round] = wv; top_i[round] = wi; }
        if (slot >= 0 && besti == wi) v[slot] = -1e30f;
        __syncthreads();
    }

    if (threadIdx.x != 0) return;

    int pick = top_i[0];

    if (temperature > 0.0f)
    {
        float p[32];
        float sum = 0.0f;
        for (int j = 0; j < k; ++j)
        {
            p[j] = __expf((top_v[j] - top_v[0]) / temperature);
            sum += p[j];
        }
        for (int j = 0; j < k; ++j) p[j] /= sum;

        float kept = 1.0f;
        int keep = k;
        if (top_p > 0.0f && top_p < 1.0f)
        {
            float cumulative = 0.0f;
            for (int j = 0; j < k; ++j) { cumulative += p[j]; keep = j + 1; if (cumulative >= top_p) break; }
            kept = cumulative;
        }

        curandStatePhilox4_32_10_t state;
        curand_init(seed, 0, step, &state);
        const float u = curand_uniform(&state) * kept;

        float cumulative = 0.0f;
        pick = top_i[keep - 1];
        for (int j = 0; j < keep; ++j) { cumulative += p[j]; if (u <= cumulative) { pick = top_i[j]; break; } }
    }

    *id_out = pick;
    if (token_out) *token_out = float(pick);
}

template<typename T>
void sample_logits_row_cuda(const int n, const float temperature, const int top_k, const float top_p,
                            const unsigned long long seed, const unsigned long long step,
                            const T* logits, float2* candidates_scratch, int* id_out, float* token_out)
{
    cudaStream_t stream = opennn::device::get_compute_stream();
    const int k = temperature <= 0.0f ? 1 : std::max(1, std::min(top_k, 32));
    const int blocks = LOGITS_SAMPLE_BLOCKS;

    OPENNN_CUDA_LAUNCH((logits_top_candidates_kernel<T, 8><<<blocks, 256, 0, stream>>>(
        n, k, logits, candidates_scratch)));
    OPENNN_CUDA_LAUNCH((sample_from_candidates_kernel<16><<<1, 256, 0, stream>>>(
        blocks * k, k, temperature, top_p, seed, step, candidates_scratch, id_out, token_out)));
}

template<typename T>
void grouped_attention_cuda(const int batch, const int query_seq, const int key_seq,
                            const int n_query_heads, const int n_kv_heads, const int head_dim,
                            const float scale, const int query_position_offset, const bool causal,
                            const int* position_device, float* decode_partials,
                            const T* Q, const T* K, const T* V, T* O)
{
    const int total = batch * n_query_heads * query_seq;
    if (total == 0) return;
    const int group = n_query_heads / n_kv_heads;
    cudaStream_t stream = opennn::device::get_compute_stream();

    if (batch == 1 && query_seq == 1 && causal && decode_partials
        && grouped_attention_decode_supported(head_dim, group))
    {
        constexpr int warps = 8;
        const int split_blocks = GROUPED_ATTENTION_DECODE_SPLITS / warps;
        const dim3 grid(n_kv_heads, split_blocks);
        const int valid = std::min(query_position_offset + 1, key_seq);

        #define OPENNN_DECODE_ATTENTION_CASE(HD, G) \
            if (head_dim == HD && group == G) \
                OPENNN_CUDA_LAUNCH((grouped_attention_decode_kernel<T, HD, G><<<grid, warps * 32, 0, stream>>>( \
                    n_kv_heads, scale, position_device, valid, Q, K, V, decode_partials)));

        OPENNN_DECODE_ATTENTION_CASE(64, 1)  OPENNN_DECODE_ATTENTION_CASE(64, 2)
        OPENNN_DECODE_ATTENTION_CASE(64, 4)  OPENNN_DECODE_ATTENTION_CASE(64, 8)
        OPENNN_DECODE_ATTENTION_CASE(128, 1) OPENNN_DECODE_ATTENTION_CASE(128, 2)
        OPENNN_DECODE_ATTENTION_CASE(128, 4) OPENNN_DECODE_ATTENTION_CASE(128, 8)
        OPENNN_DECODE_ATTENTION_CASE(256, 1) OPENNN_DECODE_ATTENTION_CASE(256, 2)
        OPENNN_DECODE_ATTENTION_CASE(256, 4)
        #undef OPENNN_DECODE_ATTENTION_CASE

        const int smem = 2 * GROUPED_ATTENTION_DECODE_SPLITS * int(sizeof(float));
        OPENNN_CUDA_LAUNCH((grouped_attention_decode_combine_kernel<T><<<n_query_heads, head_dim, smem, stream>>>(
            group, head_dim, GROUPED_ATTENTION_DECODE_SPLITS, decode_partials, O)));
        return;
    }

    const int block = 128;
    const int grid = (total + block - 1) / block;
    OPENNN_CUDA_LAUNCH((grouped_attention_kernel<T><<<grid, block, 0, stream>>>(
        total, query_seq, key_seq, n_query_heads, n_kv_heads, head_dim, group, scale,
        query_position_offset, causal ? 1 : 0, Q, K, V, O)));
}

__device__ __forceinline__ float opennn_activation_value(float x, int function)
{
    if (function == activation_sigmoid)    return 1.0f / (1.0f + expf(-x));
    if (function == activation_tanh)       return tanhf(x);
    if (function == activation_relu)       return fmaxf(x, 0.0f);
    if (function == activation_leaky_relu) return x >= 0.0f ? x : leaky_relu_slope * x;
    if (function == activation_gelu)       return 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
    if (function == activation_gelu_tanh)
    {
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + 0.044715f * x * x * x)));
    }
    if (function == activation_silu)       return x / (1.0f + expf(-x));
    return x;
}

__device__ __forceinline__ float opennn_activation_grad(float y, float d, int function)
{
    if (function == activation_sigmoid)    return d * y * (1.0f - y);
    if (function == activation_tanh)       return d * (1.0f - y * y);
    if (function == activation_relu)       return y > 0.0f ? d : 0.0f;
    if (function == activation_leaky_relu) return y >= 0.0f ? d : leaky_relu_slope * d;
    if (function == activation_gelu)
    {
        const float cdf = 0.5f * (1.0f + erff(y * 0.70710678118654752440f));
        const float pdf = 0.39894228040143267794f * expf(-0.5f * y * y);
        return d * (cdf + y * pdf);
    }
    if (function == activation_gelu_tanh)
    {
        constexpr float sqrt_2_over_pi = 0.7978845608028654f;
        const float y2 = y * y;
        const float u = sqrt_2_over_pi * (y + 0.044715f * y * y2);
        const float t = tanhf(u);
        const float du = sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * y2);
        return d * (0.5f * (1.0f + t) + 0.5f * y * (1.0f - t * t) * du);
    }
    if (function == activation_silu)
    {

        const float s = 1.0f / (1.0f + expf(-y));
        return d * s * (1.0f + y * (1.0f - s));
    }
    return d;
}

template<typename T>
__global__ void activation_forward_kernel(const int n, T* __restrict__ data, const int function)
{
    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += Index(blockDim.x) * gridDim.x)
        data[idx] = static_cast<T>(opennn_activation_value(static_cast<float>(data[idx]), function));
}

__global__ void activation_forward_kernel_bf162(const int n2, __nv_bfloat162* __restrict__ data, const int function)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n2; idx += blockDim.x * gridDim.x)
    {
        const float2 f = __bfloat1622float2(data[idx]);
        data[idx] = __floats2bfloat162_rn(opennn_activation_value(f.x, function),
                                          opennn_activation_value(f.y, function));
    }
}

__global__ void activation_forward_kernel_f4(const int n_vec, const int n, float* __restrict__ data, const int function)
{
    const Index tid = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = Index(blockDim.x) * gridDim.x;

    float4* __restrict__ const d4 = reinterpret_cast<float4*>(data);
    for (Index i = tid; i < n_vec; i += stride)
    {
        float4 v = d4[i];
        v.x = opennn_activation_value(v.x, function);
        v.y = opennn_activation_value(v.y, function);
        v.z = opennn_activation_value(v.z, function);
        v.w = opennn_activation_value(v.w, function);
        d4[i] = v;
    }

    const int tail_start = n_vec * 4;
    for (Index i = tail_start + tid; i < n; i += stride)
        data[i] = opennn_activation_value(data[i], function);
}

template<typename T>
void activation_forward_cuda(const Index n, T* data, const int function)
{
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
        if ((n & 1) == 0)
        {
            launch_elementwise_strided(n / 2, activation_forward_kernel_bf162, reinterpret_cast<__nv_bfloat162*>(data), function);
            return;
        }

    if constexpr (std::is_same_v<T, float>)
    {
        launch_vec4(n, are_float4_aligned(data), activation_forward_kernel_f4, data, function);
        return;
    }

    launch_elementwise_strided(n, activation_forward_kernel<T>, data, function);
}

template<typename T>
__global__ void activation_backward_kernel(const int n, const T* __restrict__ outputs, T* __restrict__ delta, const int function)
{
    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += Index(blockDim.x) * gridDim.x)
        delta[idx] = static_cast<T>(opennn_activation_grad(static_cast<float>(outputs[idx]),
                                                           static_cast<float>(delta[idx]), function));
}

__global__ void activation_backward_kernel_bf162(const int n2, const __nv_bfloat162* __restrict__ outputs,
                                                 __nv_bfloat162* __restrict__ delta, const int function)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n2; idx += blockDim.x * gridDim.x)
    {
        const float2 y = __bfloat1622float2(outputs[idx]);
        const float2 d = __bfloat1622float2(delta[idx]);
        delta[idx] = __floats2bfloat162_rn(opennn_activation_grad(y.x, d.x, function),
                                           opennn_activation_grad(y.y, d.y, function));
    }
}

__global__ void activation_backward_kernel_f4(const int n_vec, const int n,
                                              const float* __restrict__ outputs,
                                              float* __restrict__ delta, const int function)
{
    const Index tid = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = Index(blockDim.x) * gridDim.x;

    const float4* __restrict__ const y4 = reinterpret_cast<const float4*>(outputs);
    float4* __restrict__ const d4 = reinterpret_cast<float4*>(delta);
    for (Index i = tid; i < n_vec; i += stride)
    {
        const float4 y = y4[i];
        float4 d = d4[i];
        d.x = opennn_activation_grad(y.x, d.x, function);
        d.y = opennn_activation_grad(y.y, d.y, function);
        d.z = opennn_activation_grad(y.z, d.z, function);
        d.w = opennn_activation_grad(y.w, d.w, function);
        d4[i] = d;
    }

    const int tail_start = n_vec * 4;
    for (Index i = tail_start + tid; i < n; i += stride)
        delta[i] = opennn_activation_grad(outputs[i], delta[i], function);
}

template<typename T>
void activation_backward_cuda(const Index n, const T* outputs, T* delta, const int function)
{
    if constexpr (std::is_same_v<T, __nv_bfloat16>)
        if ((n & 1) == 0)
        {
            launch_elementwise_strided(n / 2, activation_backward_kernel_bf162,
                               reinterpret_cast<const __nv_bfloat162*>(outputs),
                               reinterpret_cast<__nv_bfloat162*>(delta), function);
            return;
        }

    if constexpr (std::is_same_v<T, float>)
    {
        launch_vec4(n, are_float4_aligned(outputs, delta), activation_backward_kernel_f4,
                    outputs, delta, function);
        return;
    }

    launch_elementwise_strided(n, activation_backward_kernel<T>, outputs, delta, function);
}

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
    launch_elementwise(n, dropout_forward_kernel<T>, output, mask, 1.0f / (1.0f - rate), rate, seed);
}

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
    launch_elementwise(n, dropout_backward_kernel<T>, output_delta, input_delta, mask, 1.0f / (1.0f - rate));
}

// Gather: dst[b,f] = src[b,t,f]. Scatter: dst[b,t,f] = src[b,f].
template<typename T, bool Gather>
__global__ void time_slice_kernel(const int n,
                                  const int time_steps,
                                  const int features,
                                  const int t,
                                  const T* __restrict__ src,
                                  T* __restrict__ dst)
{
    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += Index(blockDim.x) * gridDim.x)
    {
        const int b = int(idx) / features;
        const int f = int(idx) - b * features;
        const Index strided = (Index(b) * time_steps + t) * features + f;
        if constexpr (Gather) dst[idx] = src[strided];
        else                  dst[strided] = src[idx];
    }
}

template<typename T>
void gather_time_slice_cuda(const Index batch,
                            const Index time_steps,
                            const Index features,
                            const Index t,
                            const T* src,
                            T* dst)
{
    launch_elementwise(batch * features, time_slice_kernel<T, true>,
                       checked_int(time_steps), checked_int(features), checked_int(t), src, dst);
}

template<typename T>
void scatter_time_slice_cuda(const Index batch,
                             const Index time_steps,
                             const Index features,
                             const Index t,
                             const T* src,
                             T* dst)
{
    launch_elementwise(batch * features, time_slice_kernel<T, false>,
                       checked_int(time_steps), checked_int(features), checked_int(t), src, dst);
}

__global__ void scatter_time_slice_fill_kernel(const int batch,
                                               const int time_steps,
                                               const int features,
                                               const int t,
                                               const float* __restrict__ src,
                                               float* __restrict__ dst)
{
    const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)batch * time_steps * features;
    if (idx >= total) return;

    const int f  = int(idx % features);
    const long long bt = idx / features;
    const int ts = int(bt % time_steps);
    const int b  = int(bt / time_steps);

    dst[idx] = (ts == t) ? src[b * features + f] : 0.0f;
}

void scatter_time_slice_fill_cuda(const Index batch,
                                  const Index time_steps,
                                  const Index features,
                                  const Index t,
                                  const float* src,
                                  float* dst)
{
    if (batch == 0 || time_steps == 0 || features == 0) return;
    const int total = checked_int(batch * time_steps * features);
    OPENNN_CUDA_LAUNCH(scatter_time_slice_fill_kernel<<<grid_size_for(total), block_size, 0,
                                   opennn::device::get_compute_stream()>>>(
        checked_int(batch),
        checked_int(time_steps),
        checked_int(features),
        checked_int(t),
        src, dst));
}

struct RnnCopyParams
{
    RnnCopySpec specs[RNN_COPY_MAX_REGIONS];
    int count;
};

__global__ void rnn_copy_regions_kernel(const RnnCopyParams params)
{
    const int region = blockIdx.y;
    if (region >= params.count) return;

    const RnnCopySpec spec = params.specs[region];
    const int total = spec.rows * spec.cols;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += gridDim.x * blockDim.x)
    {
        if (spec.transpose)
        {
            const int r = idx / spec.cols;
            const int c = idx - r * spec.cols;
            spec.dst[c * spec.rows + r] = spec.src[idx];
        }
        else
            spec.dst[idx] = spec.src[idx];
    }
}

void rnn_copy_regions_cuda(const RnnCopySpec* specs, int count,
                           cudaStream_t stream)
{
    if (count <= 0) return;
    if (stream == nullptr) stream = opennn::device::get_compute_stream();

    RnnCopyParams params;
    int max_total = 0;
    for (int i = 0; i < count && i < RNN_COPY_MAX_REGIONS; ++i)
    {
        params.specs[i] = specs[i];
        max_total = max(max_total, specs[i].rows * specs[i].cols);
    }
    params.count = min(count, RNN_COPY_MAX_REGIONS);

    const dim3 grid(grid_size_for(max_total), params.count);
    OPENNN_CUDA_LAUNCH(rnn_copy_regions_kernel<<<grid, block_size, 0, stream>>>(params));
}

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

template<typename T>
__global__ void bias_grad_sum_kernel(const int batch, const int features, const int chunk,
                                     const T* __restrict__ delta, float* __restrict__ bias_grad)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= features) return;
    const long long b0 = (long long)blockIdx.y * chunk;
    const long long b1 = min((long long)batch, b0 + chunk);
    float acc = 0.0f;
    for (long long b = b0; b < b1; ++b)
        acc += static_cast<float>(delta[b * features + f]);
    atomicAdd(bias_grad + f, acc);
}

template<typename T>
void bias_grad_sum_cuda(const Index batch, const Index features, const T* delta, float* bias_grad)
{
    if (batch == 0 || features == 0) return;
    const int f = checked_int(features);

    const int f_blocks = ceil_div(f, block_size);
    const int desired_chunks = f_blocks < 256 ? 256 / f_blocks : 1;
    int chunk = checked_int((batch + desired_chunks - 1) / desired_chunks);
    if (chunk < 64) chunk = 64;
    const int n_chunks = int((batch + chunk - 1) / chunk);
    const dim3 grid(f_blocks, n_chunks);
    OPENNN_CUDA_LAUNCH(bias_grad_sum_kernel<T><<<grid, block_size, 0,
                                         opennn::device::get_compute_stream()>>>(
        checked_int(batch), f, chunk, delta, bias_grad));
}

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

__global__ void detection_forward_kernel(const int batch_size,
                                         const int grid_size,
                                         const int boxes_per_cell,
                                         const int classes_number,
                                         const int channels,
                                         const int class_activation,
                                         const float* __restrict__ anchors,
                                         const float* __restrict__ src,
                                         float* __restrict__ dst)
{
    const int values_per_box = 5 + classes_number;
    const int total = batch_size * grid_size * grid_size * boxes_per_cell;

    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += Index(blockDim.x) * gridDim.x)
    {
        const int box = idx % boxes_per_cell;
        const int t   = idx / boxes_per_cell;
        const int col = t % grid_size;
        const int t2  = t / grid_size;
        const int row = t2 % grid_size;
        const int b   = t2 / grid_size;

        const int cell = ((b * grid_size + row) * grid_size + col) * channels;
        const int base = cell + box * values_per_box;

        const float aw = anchors[box * 2 + 0];
        const float ah = anchors[box * 2 + 1];

        dst[base + 0] = sigmoid_f(src[base + 0]);
        dst[base + 1] = sigmoid_f(src[base + 1]);
        dst[base + 2] = __expf(fminf(fmaxf(src[base + 2], -4.0f), 4.0f)) * aw;
        dst[base + 3] = __expf(fminf(fmaxf(src[base + 3], -4.0f), 4.0f)) * ah;
        dst[base + 4] = sigmoid_f(src[base + 4]);

        if (class_activation == class_activation_sigmoid)
        {
            for (int c = 0; c < classes_number; ++c)
                dst[base + 5 + c] = sigmoid_f(src[base + 5 + c]);
        }
        else
        {
            float max_logit = src[base + 5];
            for (int c = 1; c < classes_number; ++c)
            {
                const float v = src[base + 5 + c];
                if (v > max_logit) max_logit = v;
            }
            float sum = 0.0f;
            for (int c = 0; c < classes_number; ++c)
            {
                const float e = __expf(src[base + 5 + c] - max_logit);
                dst[base + 5 + c] = e;
                sum += e;
            }
            const float inv_sum = 1.0f / (sum + 1e-7f);
            for (int c = 0; c < classes_number; ++c)
                dst[base + 5 + c] *= inv_sum;
        }
    }
}

void detection_forward_cuda(const Index batch_size,
                            const Index grid_size,
                            const Index boxes_per_cell,
                            const Index classes_number,
                            const Index channels,
                            const int class_activation,
                            const float* anchors,
                            const float* input,
                            float* output)
{
    if (batch_size == 0 || grid_size == 0 || boxes_per_cell == 0) return;

    const int total = checked_int(batch_size * grid_size * grid_size * boxes_per_cell);
    OPENNN_CUDA_LAUNCH(detection_forward_kernel<<<grid_size_strided_for(total), block_size, 0,
                               opennn::device::get_compute_stream()>>>(
        checked_int(batch_size),
        checked_int(grid_size),
        checked_int(boxes_per_cell),
        checked_int(classes_number),
        checked_int(channels),
        class_activation,
        anchors, input, output));
}

__global__ void detection_backward_kernel(const int batch_size,
                                          const int grid_size,
                                          const int boxes_per_cell,
                                          const int classes_number,
                                          const int channels,
                                          const int class_activation,
                                          const float* __restrict__ out,
                                          const float* __restrict__ delta,
                                          float* __restrict__ in_delta)
{
    const int values_per_box = 5 + classes_number;
    const int total = batch_size * grid_size * grid_size * boxes_per_cell;

    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += Index(blockDim.x) * gridDim.x)
    {
        const int box = idx % boxes_per_cell;
        const int t   = idx / boxes_per_cell;
        const int col = t % grid_size;
        const int t2  = t / grid_size;
        const int row = t2 % grid_size;
        const int b   = t2 / grid_size;

        const int cell = ((b * grid_size + row) * grid_size + col) * channels;
        const int base = cell + box * values_per_box;

        const float ox = out[base + 0];
        const float oy = out[base + 1];
        const float oo = out[base + 4];

        in_delta[base + 0] = delta[base + 0] * ox * (1.0f - ox);
        in_delta[base + 1] = delta[base + 1] * oy * (1.0f - oy);

        in_delta[base + 2] = delta[base + 2] * out[base + 2];
        in_delta[base + 3] = delta[base + 3] * out[base + 3];
        in_delta[base + 4] = delta[base + 4] * oo * (1.0f - oo);

        if (class_activation == class_activation_sigmoid)
        {
            for (int c = 0; c < classes_number; ++c)
            {
                const float s = out[base + 5 + c];
                in_delta[base + 5 + c] = delta[base + 5 + c] * s * (1.0f - s);
            }
        }
        else
        {
            float dot = 0.0f;
            for (int c = 0; c < classes_number; ++c)
                dot += delta[base + 5 + c] * out[base + 5 + c];

            for (int c = 0; c < classes_number; ++c)
            {
                const float s = out[base + 5 + c];
                in_delta[base + 5 + c] = s * (delta[base + 5 + c] - dot);
            }
        }
    }
}

void detection_backward_cuda(const Index batch_size,
                             const Index grid_size,
                             const Index boxes_per_cell,
                             const Index classes_number,
                             const Index channels,
                             const int class_activation,
                             const float* output,
                             const float* output_delta,
                             float* input_delta)
{
    if (batch_size == 0 || grid_size == 0 || boxes_per_cell == 0) return;

    const int total = checked_int(batch_size * grid_size * grid_size * boxes_per_cell);
    OPENNN_CUDA_LAUNCH(detection_backward_kernel<<<grid_size_strided_for(total), block_size, 0,
                                opennn::device::get_compute_stream()>>>(
        checked_int(batch_size),
        checked_int(grid_size),
        checked_int(boxes_per_cell),
        checked_int(classes_number),
        checked_int(channels),
        class_activation,
        output, output_delta, input_delta));
}

__global__ void detection_v8_forward_kernel(const int batch_size,
                                            const int grid_size,
                                            const int grid_width,
                                            const int channels,  // = 4*reg_max + classes_number
                                            const int box_ch,    // = 4*reg_max (pass-through when >4)
                                            const float* __restrict__ src,
                                            float* __restrict__ dst)
{
    const int total = batch_size * grid_size * grid_width;

    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += Index(blockDim.x) * gridDim.x)
    {
        const int col = idx % grid_width;
        const int t   = idx / grid_width;
        const int row = t % grid_size;
        const int b   = t / grid_size;

        const int base = ((b * grid_size + row) * grid_width + col) * channels;

        for (int ch = 0; ch < box_ch; ++ch)
            dst[base + ch] = (box_ch == 4) ? sigmoid_f(src[base + ch]) : src[base + ch];
        for (int ch = box_ch; ch < channels; ++ch)
            dst[base + ch] = sigmoid_f(src[base + ch]);
    }
}

void detection_v8_forward_cuda(const Index batch_size,
                               const Index grid_size,
                               const Index grid_width,
                               const Index classes_number,
                               const Index reg_max,
                               const float* input,
                               float* output)
{
    if (batch_size == 0 || grid_size == 0) return;

    const int box_ch   = checked_int(4 * max(reg_max, Index(1)));
    const int total    = checked_int(batch_size * grid_size * grid_width);
    const int channels = checked_int(box_ch + classes_number);
    OPENNN_CUDA_LAUNCH(detection_v8_forward_kernel<<<grid_size_for(total), block_size, 0,
                               opennn::device::get_compute_stream()>>>(
        checked_int(batch_size), checked_int(grid_size), checked_int(grid_width),
        channels, box_ch, input, output));
}

__global__ void detection_v8_backward_kernel(const int batch_size,
                                             const int grid_size,
                                             const int grid_width,
                                             const int channels,
                                             const int box_ch,
                                             const float* __restrict__ out,
                                             const float* __restrict__ delta,
                                             float* __restrict__ in_delta)
{
    const int total = batch_size * grid_size * grid_width;

    for (Index idx = Index(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += Index(blockDim.x) * gridDim.x)
    {
        const int col = idx % grid_width;
        const int t   = idx / grid_width;
        const int row = t % grid_size;
        const int b   = t / grid_size;

        const int base = ((b * grid_size + row) * grid_width + col) * channels;

        for (int ch = 0; ch < box_ch; ++ch)
        {
            if (box_ch == 4)
            {
                const float s = out[base + ch];
                in_delta[base + ch] = delta[base + ch] * s * (1.0f - s);
            }
            else
            {
                in_delta[base + ch] = delta[base + ch];  // DFL: identity
            }
        }
        for (int ch = box_ch; ch < channels; ++ch)
        {
            const float s = out[base + ch];
            in_delta[base + ch] = delta[base + ch] * s * (1.0f - s);
        }
    }
}

void detection_v8_backward_cuda(const Index batch_size,
                                const Index grid_size,
                                const Index grid_width,
                                const Index classes_number,
                                const Index reg_max,
                                const float* output,
                                const float* output_delta,
                                float* input_delta)
{
    if (batch_size == 0 || grid_size == 0) return;

    const int box_ch   = checked_int(4 * max(reg_max, Index(1)));
    const int total    = checked_int(batch_size * grid_size * grid_width);
    const int channels = checked_int(box_ch + classes_number);
    OPENNN_CUDA_LAUNCH(detection_v8_backward_kernel<<<grid_size_for(total), block_size, 0,
                                opennn::device::get_compute_stream()>>>(
        checked_int(batch_size), checked_int(grid_size), checked_int(grid_width),
        channels, box_ch, output, output_delta, input_delta));
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
    cudaMemsetAsync(in_delta, 0, size_t(n) * sizeof(float), opennn::device::get_compute_stream());
    launch_elementwise_strided(n, upsample_backward_kernel,
                       out_delta, in_delta, in_h, in_w, in_h * scale, in_w * scale, channels, scale);
}

// Forward scatters a slice's channels into the concatenated tensor; backward
// gathers the slice's delta back out of it.
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

// WARPS_PER_ROW warps cooperate on one output row; a 256-thread block covers
// 8 / WARPS_PER_ROW rows. One warp per row keeps the most rows in flight and
// needs no block-wide reduction; wide outputs (lm_head) stream better when a
// whole block walks a single contiguous row.
template<typename T, int WARPS_PER_ROW>
__global__ void w8a16_linear_out_major_kernel(
    const int m, const int in_features, const int out_features,
    const T* __restrict__ x, const int8_t* __restrict__ w,
    const float* __restrict__ scales, const T* __restrict__ bias,
    T* __restrict__ y)
{
    constexpr int ROWS_PER_BLOCK = 8 / WARPS_PER_ROW;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row_in_block = warp / WARPS_PER_ROW;
    const int part = warp % WARPS_PER_ROW;
    const int j = blockIdx.x * ROWS_PER_BLOCK + row_in_block;
    const bool active = j < out_features;

    float acc[W8A16_MAX_M];
    for (int r = 0; r < m; ++r) acc[r] = 0.0f;

    if (active)
    {
        const int8_t* __restrict__ row = w + size_t(j) * in_features;
        const int stride = 32 * WARPS_PER_ROW;

        if ((in_features & 3) == 0)
        {
            const char4* __restrict__ row4 = reinterpret_cast<const char4*>(row);
            const int k4 = in_features >> 2;
            for (int k = lane + part * 32; k < k4; k += stride)
            {
                const char4 wv = row4[k];
                const int kk = k << 2;
                for (int r = 0; r < m; ++r)
                {
                    const T* __restrict__ xr = x + size_t(r) * in_features + kk;
                    acc[r] += float(wv.x) * static_cast<float>(xr[0])
                            + float(wv.y) * static_cast<float>(xr[1])
                            + float(wv.z) * static_cast<float>(xr[2])
                            + float(wv.w) * static_cast<float>(xr[3]);
                }
            }
        }
        else
            for (int k = lane + part * 32; k < in_features; k += stride)
            {
                const float wv = float(row[k]);
                for (int r = 0; r < m; ++r)
                    acc[r] += wv * static_cast<float>(x[size_t(r) * in_features + k]);
            }
    }

    for (int r = 0; r < m; ++r)
        for (int offset = 16; offset > 0; offset >>= 1)
            acc[r] += __shfl_down_sync(0xffffffffu, acc[r], offset);

    const auto store = [&](const int r, const float sum)
    {
        y[size_t(r) * out_features + j] = static_cast<T>(
            sum * scales[j] + (bias ? static_cast<float>(bias[j]) : 0.0f));
    };

    if constexpr (WARPS_PER_ROW == 1)
    {
        if (!active || lane != 0) return;
        for (int r = 0; r < m; ++r) store(r, acc[r]);
    }
    else
    {
        __shared__ float partials[8][W8A16_MAX_M];
        if (lane == 0)
            for (int r = 0; r < m; ++r) partials[warp][r] = acc[r];
        __syncthreads();
        if (!active || part != 0 || lane != 0) return;
        for (int r = 0; r < m; ++r)
        {
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < WARPS_PER_ROW; ++i)
                sum += partials[row_in_block * WARPS_PER_ROW + i][r];
            store(r, sum);
        }
    }
}

template<typename T>
__global__ void w8a16_linear_in_major_kernel(
    const int m, const int in_features, const int out_features,
    const T* __restrict__ x, const int8_t* __restrict__ w,
    const float* __restrict__ scales, const T* __restrict__ bias,
    T* __restrict__ y)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= out_features) return;

    float acc[W8A16_MAX_M];
    for (int r = 0; r < m; ++r) acc[r] = 0.0f;

    for (int k = 0; k < in_features; ++k)
    {
        const float wv = float(w[size_t(k) * out_features + j]);
        for (int r = 0; r < m; ++r)
            acc[r] += wv * static_cast<float>(x[size_t(r) * in_features + k]);
    }

    const float scale = scales[j];
    const float bias_value = bias ? static_cast<float>(bias[j]) : 0.0f;
    for (int r = 0; r < m; ++r)
        y[size_t(r) * out_features + j] = static_cast<T>(acc[r] * scale + bias_value);
}

template<typename T>
void w8a16_linear_cuda(const int m, const int in_features, const int out_features,
                       const bool weights_out_major,
                       const T* x, const int8_t* w, const float* scales,
                       const T* bias, T* y)
{
    checked_host_condition(m <= 0 || m > W8A16_MAX_M,
                           "w8a16_linear_cuda: m out of range.");
    if (out_features == 0) return;
    cudaStream_t stream = opennn::device::get_compute_stream();

    if (!weights_out_major)
    {
        OPENNN_CUDA_LAUNCH(w8a16_linear_in_major_kernel<T>
            <<<grid_size_for(out_features), block_size, 0, stream>>>(
                m, in_features, out_features, x, w, scales, bias, y));
        return;
    }

    if (w8a16_out_major_warps(out_features) == 8)
        OPENNN_CUDA_LAUNCH((w8a16_linear_out_major_kernel<T, 8>
            <<<out_features, block_size, 0, stream>>>(
                m, in_features, out_features, x, w, scales, bias, y)));
    else
        OPENNN_CUDA_LAUNCH((w8a16_linear_out_major_kernel<T, 1>
            <<<ceil_div(out_features, 8), block_size, 0, stream>>>(
                m, in_features, out_features, x, w, scales, bias, y)));
}

// The row index comes from blockIdx.y, so the scale lookup needs no integer
// division: one per element dominated this kernel before.
template<typename T, bool SCALE_BY_ROW>
__global__ void w8_dequant_kernel(const int row_length,
                                  const int8_t* __restrict__ q,
                                  const float* __restrict__ scales,
                                  T* __restrict__ out)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= row_length) return;

    const Index i = Index(blockIdx.y) * row_length + column;
    const float scale = SCALE_BY_ROW ? scales[blockIdx.y] : scales[column];
    out[i] = static_cast<T>(float(q[i]) * scale);
}

template<typename T>
void w8_dequant_cuda(const Index rows, const Index row_length, const bool scale_by_row,
                     const int8_t* q, const float* scales, T* out)
{
    if (rows == 0 || row_length == 0) return;
    const dim3 grid(unsigned(grid_size_for(checked_int(row_length))), unsigned(checked_int(rows)));
    cudaStream_t stream = opennn::device::get_compute_stream();
    if (scale_by_row)
        OPENNN_CUDA_LAUNCH((w8_dequant_kernel<T, true>
            <<<grid, block_size, 0, stream>>>(checked_int(row_length), q, scales, out)));
    else
        OPENNN_CUDA_LAUNCH((w8_dequant_kernel<T, false>
            <<<grid, block_size, 0, stream>>>(checked_int(row_length), q, scales, out)));
}

template<typename T>
__global__ void embedding_forward_w8_kernel(
    const int n, const float* __restrict__ inputs, const int8_t* __restrict__ weights,
    const float* __restrict__ weight_scales, const float* __restrict__ positional_encoding,
    T* __restrict__ outputs, const int sequence_length, const int embedding_dimension,
    const int vocabulary_size, const bool scale_embedding)
{
    const float scale = scale_embedding ? sqrtf(static_cast<float>(embedding_dimension)) : 1.0f;

    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += Index(blockDim.x) * gridDim.x)
    {
        const int token_index = i / embedding_dimension;
        const int dim_index = i % embedding_dimension;
        const int token_id = static_cast<int>(inputs[token_index]);

        float val = (token_id > 0 && token_id < vocabulary_size)
            ? scale * weight_scales[token_id]
                * float(weights[size_t(token_id) * embedding_dimension + dim_index])
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
void embedding_forward_w8_cuda(const Index n, const float* inputs, const int8_t* weights,
                               const float* weight_scales, const float* positional_encoding,
                               T* outputs, const int sequence_length,
                               const int embedding_dimension, const int vocabulary_size,
                               const bool scale_embedding)
{
    launch_elementwise_strided(n, embedding_forward_w8_kernel<T>, inputs, weights,
                       weight_scales, positional_encoding, outputs,
                       sequence_length, embedding_dimension, vocabulary_size, scale_embedding);
}

#define INSTANTIATE(T) \
    template void embedding_backward_cuda<T>(const Index, const float*, const T*, float*, float*, const int, const int, const int, const bool); \
    template void split_heads_cuda<T>(const Index, const T*, T*, const int, const int, const int); \
    template void merge_heads_cuda<T>(const Index, const T*, T*, const int, const int, const int); \
    template void attention_masked_softmax_cuda<T>(int, int, int, int, int, const T*, T*, T*, bool, bool); \
    template void attention_length_masked_softmax_cuda<T>(int, int, int, int, const int*, T*, T*, bool, bool); \
    template void attention_sequence_lengths_cuda<T>(int, int, int, int, const T*, int32_t*, int32_t*); \
    template void max_pooling_3d_forward_cuda<T>(const Index, const T*, T*, float*, const int, const int); \
    template void max_pooling_3d_backward_cuda<T>(const Index, const T*, T*, const float*, const int, const int); \
    template void average_pooling_3d_forward_cuda<T>(const Index, const T*, T*, const int, const int); \
    template void average_pooling_3d_backward_cuda<T>(const Index, const T*, const T*, T*, const int, const int); \
    template void first_token_3d_forward_cuda<T>(const int, const int, const int, const T*, T*); \
    template void first_token_3d_backward_cuda<T>(const int, const int, const int, const T*, T*); \
    template void batchnorm_inference_cuda<T>(const Index, const Index, const T*, const T*, const float*, const float*, const float*, const float*, const float, const bool, T*); \
    template void layernorm_forward_cuda<T>(const int, const int, const T*, T*, float*, float*, const float*, const float*, const float); \
    template void layernorm_add_forward_cuda<T>(const int, const int, const T*, const T*, T*, T*, float*, float*, const float*, const float*, const float); \
    template void layernorm_backward_cuda<T>(const int, const int, const T*, const T*, const float*, const float*, const float*, T*, float*, float*); \
    template void rmsnorm_forward_cuda<T>(const int, const int, const T*, T*, float*, const float*, const float); \
    template void rmsnorm_backward_cuda<T>(const int, const int, const T*, const T*, const float*, const float*, T*, float*); \
    template void rope_forward_cuda<T>(const int, const int, const int, const int, const int, const int, const T*, T*, const float*, const float*); \
    template void rope_backward_cuda<T>(const int, const int, const int, const int, const int, const int, const T*, T*, const float*, const float*); \
    template void qk_rope_cache_append_cuda<T>(const int, const int, const int, const float, const int*, const T*, const float*, const float*, const float*, const float*, T*, T*, T*); \
    template void swiglu_forward_cuda<T>(const int, const T*, const T*, T*); \
    template void swiglu_backward_cuda<T>(const int, const T*, const T*, const T*, T*, T*); \
    template void sample_logits_row_cuda<T>(const int, const float, const int, const float, const unsigned long long, const unsigned long long, const T*, float2*, int*, float*); \
    template void grouped_attention_cuda<T>(const int, const int, const int, const int, const int, const int, const float, const int, const bool, const int*, float*, const T*, const T*, const T*, T*); \
    template void grouped_attention_softmax_cuda<T>(const int, const int, const int, const int, const bool, const float*, T*); \
    template void activation_forward_cuda<T>(const Index, T*, const int); \
    template void activation_backward_cuda<T>(const Index, const T*, T*, const int); \
    template void dropout_forward_cuda<T>(const Index, T*, uint8_t*, const float, const unsigned long long); \
    template void dropout_backward_cuda<T>(const Index, const T*, T*, const uint8_t*, const float); \
    template void gather_time_slice_cuda<T>(const Index, const Index, const Index, const Index, const T*, T*); \
    template void scatter_time_slice_cuda<T>(const Index, const Index, const Index, const Index, const T*, T*); \
    template void transpose_2d_cuda<T>(const Index, const Index, const T*, T*); \
    template void rnn_step_fused_forward_cuda<T>(const Index, const Index, const Index, const T*, const T*, const T*, const T*, const T*, T*, T*, const int); \
    template void bias_grad_sum_cuda<T>(const Index, const Index, const T*, float*); \
    template void rnn_step_fused_backward_pre_cuda<T>(const Index, const Index, const Index, const Index, const bool, const T*, const T*, const T*, T*); \
    template void w8a16_linear_cuda<T>(const int, const int, const int, const bool, const T*, const int8_t*, const float*, const T*, T*); \
    template void w8_dequant_cuda<T>(const Index, const Index, const bool, const int8_t*, const float*, T*); \
    template void embedding_forward_w8_cuda<T>(const Index, const float*, const int8_t*, const float*, const float*, T*, const int, const int, const int, const bool);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE

// Quantized weights are transposed once at upload so decode always runs the
// out-major GEMV.
template void transpose_2d_cuda<int8_t>(const Index, const Index, const int8_t*, int8_t*);

#define INSTANTIATE(TIn, TOut) \
    template void bounding_cuda<TIn, TOut>(const Index, const int, const TIn*, const float*, const float*, TOut*); \
    template void scale_cuda<TIn, TOut>(const Index, const int, const TIn*, const float*, const float*, const float*, const float*, const float*, float, float, TOut*); \
    template void unscale_cuda<TIn, TOut>(const Index, const int, const TIn*, const float*, const float*, const float*, const float*, const float*, float, float, TOut*); \
    template void scaled_diff_cuda_typed<TIn, TOut>(const Index, const TIn*, const float*, float, TOut*); \
    template void embedding_forward_cuda<TIn, TOut>(const Index, const float*, const TIn*, const float*, TOut*, const int, const int, const int, const bool);

OPENNN_INSTANTIATE_FLOAT_BF16_2(INSTANTIATE)
#undef INSTANTIATE

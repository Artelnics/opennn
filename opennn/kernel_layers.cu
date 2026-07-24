#include "kernel_common.cuh"
#include <curand_kernel.h>

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
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(bounding_kernel<TIn, TOut><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, features, input, lower, upper, output));
}

template void bounding_cuda<float,         float>        (const Index, const int, const float*,         const float*, const float*, float*);
template void bounding_cuda<float,         __nv_bfloat16>(const Index, const int, const float*,         const float*, const float*, __nv_bfloat16*);
template void bounding_cuda<__nv_bfloat16, float>        (const Index, const int, const __nv_bfloat16*, const float*, const float*, float*);
template void bounding_cuda<__nv_bfloat16, __nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, const float*, __nv_bfloat16*);

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
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(scale_kernel<TIn, TOut, false><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, features,
                                                                   input, minimums, maximums, means, stds, scalers,
                                                                   min_range, max_range, output));
}

template void scale_cuda<float,         float>        (const Index, const int, const float*,         const float*, const float*, const float*, const float*, const float*, float, float, float*);
template void scale_cuda<float,         __nv_bfloat16>(const Index, const int, const float*,         const float*, const float*, const float*, const float*, const float*, float, float, __nv_bfloat16*);
template void scale_cuda<__nv_bfloat16, float>        (const Index, const int, const __nv_bfloat16*, const float*, const float*, const float*, const float*, const float*, float, float, float*);
template void scale_cuda<__nv_bfloat16, __nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, const float*, const float*, const float*, const float*, float, float, __nv_bfloat16*);

template<typename TIn, typename TOut>
void unscale_cuda(const Index n, const int features,
                  const TIn* input,
                  const float* minimums, const float* maximums,
                  const float* means, const float* stds,
                  const float* scalers,
                  const float min_range, const float max_range,
                  TOut* output)
{
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(scale_kernel<TIn, TOut, true><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, features,
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
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
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

    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += Index(blockDim.x) * gridDim.x)
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
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(embedding_backward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(
        total, inputs, output_deltas, weight_gradients, positional_gradients,
        sequence_length, embedding_dimension, vocabulary_size, scale_embedding));
}

template void embedding_backward_cuda<float>        (const Index, const float*, const float*,         float*, float*, const int, const int, const int, const bool);
template void embedding_backward_cuda<__nv_bfloat16>(const Index, const float*, const __nv_bfloat16*, float*, float*, const int, const int, const int, const bool);

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
__global__ void swap_heads_vec_kernel(const int n_vec, const T* __restrict__ in, T* __restrict__ out, const int P, const int Q, const int D_vec)
{
    const float4* const in_v  = reinterpret_cast<const float4*>(in);
    float4* const       out_v = reinterpret_cast<float4*>(out);

    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < n_vec; i += Index(blockDim.x) * gridDim.x)
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
    split_heads_cuda(n, in, out, H, S, D);
}

template void merge_heads_cuda<float>        (const Index, const float*,         float*,         const int, const int, const int);
template void merge_heads_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const int, const int, const int);

template<typename T>
__global__ void padding_mask_kernel(const int num_tokens, const T* __restrict__ source_input, T* __restrict__ padding_mask, const int embedding_dimension)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < num_tokens; i += Index(blockDim.x) * gridDim.x)
    {
        const T* token = source_input + i * embedding_dimension;
        bool is_pad = true;
        for (int e = 0; e < embedding_dimension; ++e)
            if (fabsf(static_cast<float>(token[e])) > 1e-7f) { is_pad = false; break; }
        padding_mask[i] = static_cast<T>(is_pad ? 1.0f : 0.0f);
    }
}

static inline int layernorm_threads(int D);

// One warp per row, the row slice cached in registers, shuffle-only reductions.
// Masked positions behave like a -1e9 penalty, so a fully masked row still
// softmaxes to uniform.
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
    const int num_tokens = batch_size * source_sequence_length;
    if (num_tokens > 0)
        OPENNN_CUDA_LAUNCH(padding_mask_kernel<T><<<grid_size_for(num_tokens), block_size, 0, opennn::device::get_compute_stream()>>>(
            num_tokens, source_input, padding_mask, embedding_dimension));

    launch_masked_softmax_rows<T>(batch_size, heads_number,
                                  query_sequence_length, source_sequence_length,
                                  attention_weights, padding_mask, use_causal_mask,
                                  zero_padded_queries,
                                  opennn::device::get_compute_stream());
}

template void attention_masked_softmax_cuda<float>        (int, int, int, int, int, const float*,         float*,         float*,         bool, bool);
template void attention_masked_softmax_cuda<__nv_bfloat16>(int, int, int, int, int, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, bool, bool);

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
                                const int* host_lengths, T* attention_weights, T* padding_mask,
                                const bool use_causal_mask, const bool zero_padded_queries)
{
    if (batch_size == 0) return;

    cudaStream_t stream = opennn::device::get_compute_stream();

    int* device_lengths = nullptr;
    cudaMallocAsync(&device_lengths, size_t(batch_size) * sizeof(int), stream);
    cudaMemcpyAsync(device_lengths, host_lengths, size_t(batch_size) * sizeof(int),
                    cudaMemcpyHostToDevice, stream);

    const int m = batch_size * source_sequence_length;
    OPENNN_CUDA_LAUNCH(length_to_padding_mask_kernel<T><<<grid_size_for(m), block_size, 0, stream>>>(
        m, source_sequence_length, device_lengths, padding_mask));

    launch_masked_softmax_rows<T>(batch_size, heads_number,
                                  query_sequence_length, source_sequence_length,
                                  attention_weights, padding_mask, use_causal_mask,
                                  zero_padded_queries, stream);

    cudaFreeAsync(device_lengths, stream);
}

template void attention_length_masked_softmax_cuda<float>        (int, int, int, int, const int*, float*,         float*,         bool, bool);
template void attention_length_masked_softmax_cuda<__nv_bfloat16>(int, int, int, int, const int*, __nv_bfloat16*, __nv_bfloat16*, bool, bool);

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
    if (n == 0) return;

    const int total = checked_int(n);

    OPENNN_CUDA_LAUNCH(max_pooling_3d_forward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, in, out, indices, S, F));
}

template void max_pooling_3d_forward_cuda<float>        (const Index, const float*,         float*,         float*, const int, const int);
template void max_pooling_3d_forward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, float*, const int, const int);

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

    OPENNN_CUDA_LAUNCH(average_pooling_3d_backward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, delta, in_gradient, S, F, valid_mask, counts));
}

template void average_pooling_3d_backward_cuda<float>        (const Index, const float*,         const float*,         float*,         const int, const int);
template void average_pooling_3d_backward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, const int, const int);

template<typename T>
__global__ void first_token_3d_forward_kernel(const int n, const int S, const int F, const T* __restrict__ in, T* __restrict__ out)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int b = i / F;
        const int h = i % F;
        out[i] = in[b * S * F + h];
    }
}

template<typename T>
void first_token_3d_forward_cuda(const int B, const int S, const int F, const T* in, T* out)
{
    if (B == 0 || F == 0) return;
    const int total = B * F;
    OPENNN_CUDA_LAUNCH(first_token_3d_forward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, S, F, in, out));
}

template void first_token_3d_forward_cuda<float>        (const int, const int, const int, const float*,         float*);
template void first_token_3d_forward_cuda<__nv_bfloat16>(const int, const int, const int, const __nv_bfloat16*, __nv_bfloat16*);

template<typename T>
__global__ void first_token_3d_backward_kernel(const int n, const int S, const int F, const T* __restrict__ delta, T* __restrict__ in_gradient)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int b = i / F;
        const int h = i % F;
        in_gradient[b * S * F + h] = delta[i];
    }
}

template<typename T>
void first_token_3d_backward_cuda(const int B, const int S, const int F, const T* delta, T* in_gradient)
{
    if (B == 0 || F == 0) return;
    const int total = B * F;
    OPENNN_CUDA_LAUNCH(first_token_3d_backward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, S, F, delta, in_gradient));
}

template void first_token_3d_backward_cuda<float>        (const int, const int, const int, const float*,         float*);
template void first_token_3d_backward_cuda<__nv_bfloat16>(const int, const int, const int, const __nv_bfloat16*, __nv_bfloat16*);

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
            // Clamp variance to >= 0: E[x^2] - E[x]^2 can go slightly negative
            // from catastrophic cancellation, making rsqrtf return inf/nan.
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

// Fused residual-add + layernorm: computes S = X + R once, writes S to `sum`
// (the residual-stream tensor the backward needs), and writes LayerNorm(S) to Y.
template<typename T>
__global__ void layernorm_add_forward_kernel(const int N, const int D, const T* __restrict__ X, const T* __restrict__ R, T* __restrict__ sum, T* __restrict__ Y, float* __restrict__ means, float* __restrict__ inv_vars, const float* __restrict__ gamma, const float* __restrict__ beta, const float eps)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const T* x_row   = X   + idx * D;
    const T* r_row   = R   + idx * D;
    T*       s_row   = sum + idx * D;
    T*       y_row   = Y   + idx * D;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float s = static_cast<float>(x_row[i]) + static_cast<float>(r_row[i]);
        s_row[i]      = static_cast<T>(s);   // store the residual-stream sum
        local_sum    += s;
        local_sum_sq += s * s;
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
        const float x_hat = (static_cast<float>(s_row[i]) - mean) * inv_var;
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

// NHWC batchnorm inference with the residual add and ReLU fused in (cuDNN's
// legacy inference call has no NHWC kernel for several ResNet shapes).
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
    if (total == 0 || channels == 0) return;
    const int n = checked_int(total);
    OPENNN_CUDA_LAUNCH(batchnorm_inference_kernel<T><<<grid_size_for(n), block_size, 0,
                                         opennn::device::get_compute_stream()>>>(
        total, checked_int(channels), x, residual, gamma, beta, mean, variance,
        epsilon, apply_relu ? 1 : 0, y));
}

template void batchnorm_inference_cuda<float>        (const Index, const Index, const float*,         const float*,         const float*, const float*, const float*, const float*, const float, const bool, float*);
template void batchnorm_inference_cuda<__nv_bfloat16>(const Index, const Index, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, const float*, const float*, const float, const bool, __nv_bfloat16*);

// Inference-time batchnorm folding: the BN affine collapses into the
// convolution as W'[k,...] = W[k,...] * gamma[k]/sqrt(var[k]+eps) and
// b'[k] = beta[k] - mean[k] * gamma[k]/sqrt(var[k]+eps). transpose flips the
// folded weights from KRSC to [RSC, kernels], the GEMM layout 1x1 convs need.
__global__ void conv_bn_fold_kernel(const Index total, const int kernel_size, const int kernels,
                                    const float* __restrict__ weights,
                                    const float* __restrict__ gamma,
                                    const float* __restrict__ beta,
                                    const float* __restrict__ mean,
                                    const float* __restrict__ variance,
                                    const float epsilon,
                                    const int transpose,
                                    float* __restrict__ folded_weights,
                                    float* __restrict__ folded_bias)
{
    for (Index i = Index(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += Index(blockDim.x) * gridDim.x)
    {
        const int k = int(i / kernel_size);
        const int r = int(i % kernel_size);
        const float scale = gamma[k] * rsqrtf(variance[k] + epsilon);
        const float value = weights[i] * scale;
        if (transpose)
            folded_weights[Index(r) * kernels + k] = value;
        else
            folded_weights[i] = value;
        if (r == 0)
            folded_bias[k] = beta[k] - mean[k] * scale;
    }
}

void conv_bn_fold_cuda(const Index kernels, const Index kernel_size,
                       const float* weights,
                       const float* gamma, const float* beta,
                       const float* mean, const float* variance,
                       const float epsilon, const bool transpose,
                       float* folded_weights, float* folded_bias)
{
    const Index total = kernels * kernel_size;
    if (total == 0) return;
    const int n = checked_int(total);
    OPENNN_CUDA_LAUNCH(conv_bn_fold_kernel<<<grid_size_for(n), block_size, 0,
                                         opennn::device::get_compute_stream()>>>(
        total, checked_int(kernel_size), checked_int(kernels), weights,
        gamma, beta, mean, variance, epsilon, transpose ? 1 : 0,
        folded_weights, folded_bias));
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
    if (total == 0) return;
    const int n = checked_int(total);
    OPENNN_CUDA_LAUNCH(add_relu_kernel<<<grid_size_for(n), block_size, 0,
                                         opennn::device::get_compute_stream()>>>(
        total, a, b, apply_relu ? 1 : 0, y));
}

template<typename T>
void layernorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps)
{
    if (N == 0 || D == 0) return;

    OPENNN_CUDA_LAUNCH(layernorm_forward_kernel<T><<<N, layernorm_threads(D), 0, opennn::device::get_compute_stream()>>>(N, D, X, Y, means, inv_vars, gamma, beta, eps));
}

template<typename T>
void layernorm_add_forward_cuda(const int N, const int D, const T* X, const T* R, T* sum, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps)
{
    if (N == 0 || D == 0) return;

    OPENNN_CUDA_LAUNCH(layernorm_add_forward_kernel<T><<<N, layernorm_threads(D), 0, opennn::device::get_compute_stream()>>>(N, D, X, R, sum, Y, means, inv_vars, gamma, beta, eps));
}

template void layernorm_forward_cuda<float>        (const int, const int, const float*,         float*,         float*, float*, const float*, const float*, const float);
template void layernorm_forward_cuda<__nv_bfloat16>(const int, const int, const __nv_bfloat16*, __nv_bfloat16*, float*, float*, const float*, const float*, const float);
template void layernorm_add_forward_cuda<float>        (const int, const int, const float*,         const float*,         float*,         float*,         float*, float*, const float*, const float*, const float);
template void layernorm_add_forward_cuda<__nv_bfloat16>(const int, const int, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, float*, float*, const float*, const float*, const float);

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
        if (gridDim.y == 1)
        {
            dGamma[d] = g;
            dBeta [d] = b;
        }
        else
        {
            atomicAdd(dGamma + d, g);
            atomicAdd(dBeta  + d, b);
        }
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
    // Narrow D gives few blocks in x (d_model 256 -> 8), so split the row
    // range across grid.y until the GPU is covered; multi-chunk grids
    // accumulate with atomics into zeroed buffers.
    const int desired_chunks = grid_x < 192 ? 192 / grid_x : 1;
    int chunk = ceil_div(N, desired_chunks);
    if (chunk < NUM_WARPS * 8) chunk = NUM_WARPS * 8;
    const int grid_y = ceil_div(N, chunk);
    if (grid_y > 1)
    {
        cudaStream_t stream = opennn::device::get_compute_stream();
        cudaMemsetAsync(dGamma, 0, size_t(D) * sizeof(float), stream);
        cudaMemsetAsync(dBeta,  0, size_t(D) * sizeof(float), stream);
    }
    layernorm_gamma_beta_gradient_coalesced_kernel<T, NUM_WARPS><<<dim3(grid_x, grid_y), block, 0,
        opennn::device::get_compute_stream()>>>(N, D, chunk, dY, X, means, inv_vars, dGamma, dBeta);
    opennn::device::check_last_error();
}

template void layernorm_backward_cuda<float>        (const int, const int, const float*,         const float*,         const float*, const float*, const float*, float*,         float*, float*);
template void layernorm_backward_cuda<__nv_bfloat16>(const int, const int, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, const float*, __nv_bfloat16*, float*, float*);

// RMSNorm: Y = weight * X / sqrt(mean(X^2) + eps), no mean subtraction, no
// bias. `inv_rms` stores the per-row 1/rms for the backward pass.
template<typename T>
__global__ void rmsnorm_forward_kernel(const int N, const int D, const T* __restrict__ X, T* __restrict__ Y, float* __restrict__ inv_rms, const float* __restrict__ weight, const float eps)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const T* x_row = X + idx * D;
    T* y_row = Y + idx * D;

    float local_sum_sq = 0.0f;
    float ignore = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float x = static_cast<float>(x_row[i]);
        local_sum_sq += x * x;
    }

    warp_reduce_sum2(local_sum_sq, ignore);

    __shared__ float warp_sum_sq[32];
    __shared__ float s_inv_rms;

    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0)
        warp_sum_sq[warp_id] = local_sum_sq;
    __syncthreads();

    const int num_warps = (blockDim.x + 31) >> 5;
    if (warp_id == 0)
    {
        float s_sq = (threadIdx.x < num_warps) ? warp_sum_sq[threadIdx.x] : 0.0f;
        float d = 0.0f;
        warp_reduce_sum2(s_sq, d);

        if (threadIdx.x == 0)
        {
            const float inv_D = 1.0f / static_cast<float>(D);
            const float inverse = rsqrtf(s_sq * inv_D + eps);
            s_inv_rms    = inverse;
            inv_rms[idx] = inverse;
        }
    }
    __syncthreads();

    const float inverse = s_inv_rms;

    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float x_hat = static_cast<float>(x_row[i]) * inverse;
        y_row[i] = static_cast<T>(weight[i] * x_hat);
    }
}

template<typename T>
void rmsnorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* inv_rms, const float* weight, const float eps)
{
    if (N == 0 || D == 0) return;

    OPENNN_CUDA_LAUNCH(rmsnorm_forward_kernel<T><<<N, layernorm_threads(D), 0, opennn::device::get_compute_stream()>>>(N, D, X, Y, inv_rms, weight, eps));
}

template void rmsnorm_forward_cuda<float>        (const int, const int, const float*,         float*,         float*, const float*, const float);
template void rmsnorm_forward_cuda<__nv_bfloat16>(const int, const int, const __nv_bfloat16*, __nv_bfloat16*, float*, const float*, const float);

// dX_i = (d_i - x_hat_i * mean(d . x_hat)) * inv_rms,  d = dY * weight,
// x_hat = X * inv_rms. Same shape as layer norm's dX but without the -mean(d)
// centring term.
template<typename T>
__global__ void rmsnorm_backward_kernel(const int N, const int D, const T* __restrict__ dY, const T* __restrict__ X, const float* __restrict__ inv_rms, const float* __restrict__ weight, T* __restrict__ dX)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const T* dy_row = dY + idx * D;
    const T* x_row = X + idx * D;
    T* dx_row = dX + idx * D;

    const float inverse = inv_rms[idx];

    float local_sum_d_xhat = 0.0f;
    float ignore = 0.0f;

    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float d     = static_cast<float>(dy_row[i]) * weight[i];
        const float x_hat = static_cast<float>(x_row[i]) * inverse;
        local_sum_d_xhat += d * x_hat;
    }

    warp_reduce_sum2(local_sum_d_xhat, ignore);

    __shared__ float warp_sum[32];
    __shared__ float s_mean_d_xhat;

    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0)
        warp_sum[warp_id] = local_sum_d_xhat;
    __syncthreads();

    const int num_warps = (blockDim.x + 31) >> 5;
    if (warp_id == 0)
    {
        float s = (threadIdx.x < num_warps) ? warp_sum[threadIdx.x] : 0.0f;
        float d = 0.0f;
        warp_reduce_sum2(s, d);
        if (threadIdx.x == 0)
        {
            const float inv_D = 1.0f / static_cast<float>(D);
            s_mean_d_xhat = s * inv_D;
        }
    }
    __syncthreads();

    const float mean_d_xhat = s_mean_d_xhat;

    for (int i = threadIdx.x; i < D; i += blockDim.x)
    {
        const float d     = static_cast<float>(dy_row[i]) * weight[i];
        const float x_hat = static_cast<float>(x_row[i]) * inverse;
        dx_row[i] = static_cast<T>((d - x_hat * mean_d_xhat) * inverse);
    }
}

template<typename T, int NUM_WARPS>
__global__ void rmsnorm_weight_gradient_coalesced_kernel(const int N, const int D,
                                                         const int chunk,
                                                         const T* __restrict__ dY,
                                                         const T* __restrict__ X,
                                                         const float* __restrict__ inv_rms,
                                                         float* __restrict__ dWeight)
{
    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int d       = blockIdx.x * 32 + lane;
    const bool active = (d < D);
    const int n0      = blockIdx.y * chunk;
    const int n1      = min(N, n0 + chunk);

    float local_weight = 0.0f;

    if (active)
    {
        for (int n = n0 + warp_id; n < n1; n += NUM_WARPS)
        {
            const float dy    = static_cast<float>(dY[n * D + d]);
            const float x_hat = static_cast<float>(X[n * D + d]) * inv_rms[n];
            local_weight += dy * x_hat;
        }
    }

    __shared__ float partial_weight[NUM_WARPS][32];
    partial_weight[warp_id][lane] = local_weight;
    __syncthreads();

    if (warp_id == 0 && active)
    {
        float g = 0.0f;
        #pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w)
            g += partial_weight[w][lane];
        if (gridDim.y == 1)
            dWeight[d] = g;
        else
            atomicAdd(dWeight + d, g);
    }
}

template<typename T>
void rmsnorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* inv_rms, const float* weight, T* dX, float* dWeight)
{
    if (N == 0 || D == 0) return;

    if (dX)
        OPENNN_CUDA_LAUNCH(rmsnorm_backward_kernel<T><<<N, layernorm_threads(D), 0, opennn::device::get_compute_stream()>>>(N, D, dY, X, inv_rms, weight, dX));

    constexpr int NUM_WARPS = 8;
    const dim3 block(32, NUM_WARPS);
    const int grid_x = (D + 31) / 32;
    const int desired_chunks = grid_x < 192 ? 192 / grid_x : 1;
    int chunk = ceil_div(N, desired_chunks);
    if (chunk < NUM_WARPS * 8) chunk = NUM_WARPS * 8;
    const int grid_y = ceil_div(N, chunk);
    if (grid_y > 1)
    {
        cudaStream_t stream = opennn::device::get_compute_stream();
        cudaMemsetAsync(dWeight, 0, size_t(D) * sizeof(float), stream);
    }
    rmsnorm_weight_gradient_coalesced_kernel<T, NUM_WARPS><<<dim3(grid_x, grid_y), block, 0,
        opennn::device::get_compute_stream()>>>(N, D, chunk, dY, X, inv_rms, dWeight);
    opennn::device::check_last_error();
}

template void rmsnorm_backward_cuda<float>        (const int, const int, const float*,         const float*,         const float*, const float*, float*,         float*);
template void rmsnorm_backward_cuda<__nv_bfloat16>(const int, const int, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, __nv_bfloat16*, float*);

// RoPE (rotate_half): one block per row, threads stride over the model_dim
// channels. For channel d of head h, pair (d, d+half) within the head is rotated
// by the row's sequence position. Channels >= rotary_dim pass through. Forward
// uses +sin; backward (SIGN = -1) applies the transpose (inverse) rotation.
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
        const int d = e % head_dim;              // channel within the head
        const int base_e = row_base + e;

        if (d < rotary_dim)
        {
            const int head_start = base_e - d;   // row_base + h*head_dim
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

template void rope_forward_cuda<float>        (const int, const int, const int, const int, const int, const int, const float*,         float*,         const float*, const float*);
template void rope_forward_cuda<__nv_bfloat16>(const int, const int, const int, const int, const int, const int, const __nv_bfloat16*, __nv_bfloat16*, const float*, const float*);
template void rope_backward_cuda<float>        (const int, const int, const int, const int, const int, const int, const float*,         float*,         const float*, const float*);
template void rope_backward_cuda<__nv_bfloat16>(const int, const int, const int, const int, const int, const int, const __nv_bfloat16*, __nv_bfloat16*, const float*, const float*);

// SwiGLU: out = silu(gate) * up (element-wise). silu(g) = g * sigmoid(g).
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
    if (n == 0) return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    OPENNN_CUDA_LAUNCH(swiglu_forward_kernel<T><<<grid, block, 0, opennn::device::get_compute_stream()>>>(n, gate, up, out));
}

template<typename T>
void swiglu_backward_cuda(const int n, const T* dout, const T* gate, const T* up, T* dgate, T* dup)
{
    if (n == 0) return;
    const int block = 256;
    const int grid = (n + block - 1) / block;
    OPENNN_CUDA_LAUNCH(swiglu_backward_kernel<T><<<grid, block, 0, opennn::device::get_compute_stream()>>>(n, dout, gate, up, dgate, dup));
}

template void swiglu_forward_cuda<float>        (const int, const float*,         const float*,         float*);
template void swiglu_forward_cuda<__nv_bfloat16>(const int, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);
template void swiglu_backward_cuda<float>        (const int, const float*,         const float*,         const float*,         float*,         float*);
template void swiglu_backward_cuda<__nv_bfloat16>(const int, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*);

// Grouped-query causal attention. One thread per query (b, hq, i); flash-style
// online softmax so scores are not materialized. Q head hq uses KV head
// hq/(n_query_heads/n_kv_heads). Causal: query i (abs pos offset+i) attends keys
// 0..(offset+i). head_dim capped at 256 (Qwen3=128, Qwen3.5=256).
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
void grouped_attention_cuda(const int batch, const int query_seq, const int key_seq,
                            const int n_query_heads, const int n_kv_heads, const int head_dim,
                            const float scale, const int query_position_offset, const bool causal,
                            const T* Q, const T* K, const T* V, T* O)
{
    const int total = batch * n_query_heads * query_seq;
    if (total == 0) return;
    const int group = n_query_heads / n_kv_heads;
    const int block = 128;
    const int grid = (total + block - 1) / block;
    OPENNN_CUDA_LAUNCH((grouped_attention_kernel<T><<<grid, block, 0, opennn::device::get_compute_stream()>>>(
        total, query_seq, key_seq, n_query_heads, n_kv_heads, head_dim, group, scale,
        query_position_offset, causal ? 1 : 0, Q, K, V, O)));
}

template void grouped_attention_cuda<float>        (const int, const int, const int, const int, const int, const int, const float, const int, const bool, const float*,         const float*,         const float*,         float*);
template void grouped_attention_cuda<__nv_bfloat16>(const int, const int, const int, const int, const int, const int, const float, const int, const bool, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*);

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
        // `y` is the pre-activation input for needs_input activations.
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

template<typename T>
void activation_forward_cuda(const Index n, T* data, const int function)
{
    if (n == 0) return;

    if constexpr (std::is_same_v<T, __nv_bfloat16>)
        if ((n & 1) == 0)
        {
            const int n2 = checked_int(n / 2);
            OPENNN_CUDA_LAUNCH(activation_forward_kernel_bf162<<<grid_size_for(n2), block_size, 0,
                opennn::device::get_compute_stream()>>>(n2, reinterpret_cast<__nv_bfloat162*>(data), function));
            return;
        }

    const int total = checked_int(n);
    OPENNN_CUDA_LAUNCH(activation_forward_kernel<T><<<grid_size_for(total), block_size, 0, opennn::device::get_compute_stream()>>>(total, data, function));
}

template void activation_forward_cuda<float>        (const Index, float*,         const int);
template void activation_forward_cuda<__nv_bfloat16>(const Index, __nv_bfloat16*, const int);

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

template<typename T>
void activation_backward_cuda(const Index n, const T* outputs, T* delta, const int function)
{
    if (n == 0) return;

    if constexpr (std::is_same_v<T, __nv_bfloat16>)
        if ((n & 1) == 0)
        {
            const int n2 = checked_int(n / 2);
            OPENNN_CUDA_LAUNCH(activation_backward_kernel_bf162<<<grid_size_for(n2), block_size, 0,
                opennn::device::get_compute_stream()>>>(n2, reinterpret_cast<const __nv_bfloat162*>(outputs),
                                                        reinterpret_cast<__nv_bfloat162*>(delta), function));
            return;
        }

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

// Column-sum of a (batch x features) delta into an fp32 bias gradient.
// The caller must zero bias_grad first (this atomicAdds).
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
    // Shrink the batch chunk until the grid covers the whole GPU (narrow
    // feature counts give few blocks in x).
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

template void bias_grad_sum_cuda<float>        (const Index, const Index, const float*,         float*);
template void bias_grad_sum_cuda<__nv_bfloat16>(const Index, const Index, const __nv_bfloat16*, float*);

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


// -----------------------------------------------------------------------------
// YOLO DetectionOperator
// -----------------------------------------------------------------------------
// CPU reference: operators.cpp:DetectionOperator::apply / apply_delta.
// Thread layout: one thread per box. Tile = (batch, grid, grid, boxes_per_cell);
// each thread owns a contiguous span of (5 + classes_number) floats in NHWC
// layout (channels-last), matching the CPU loop's `base` index arithmetic.

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
        else  // Softmax
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
    OPENNN_CUDA_LAUNCH(detection_forward_kernel<<<grid_size_for(total), block_size, 0,
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
        // d/dx exp(x)*a evaluated through the output o = exp(x)*a is just o (the anchor cancels).
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
        else  // Softmax: ∂L/∂x_i = s_i * (g_i - Σ_j g_j s_j)
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
    OPENNN_CUDA_LAUNCH(detection_backward_kernel<<<grid_size_for(total), block_size, 0,
                                opennn::device::get_compute_stream()>>>(
        checked_int(batch_size),
        checked_int(grid_size),
        checked_int(boxes_per_cell),
        checked_int(classes_number),
        checked_int(channels),
        class_activation,
        output, output_delta, input_delta));
}

// ── YOLOv8 anchor-free DetectionV8Operator ────────────────────────────────────
// All 4+C channels are sigmoid-gated. One "box" per cell, no anchor parameters.

__global__ void detection_v8_forward_kernel(const int batch_size,
                                            const int grid_size,
                                            const int grid_width,
                                            const int channels,  // = 4 + classes_number
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

        for (int ch = 0; ch < channels; ++ch)
            dst[base + ch] = sigmoid_f(src[base + ch]);
    }
}

void detection_v8_forward_cuda(const Index batch_size,
                               const Index grid_size,
                               const Index grid_width,
                               const Index classes_number,
                               const float* input,
                               float* output)
{
    if (batch_size == 0 || grid_size == 0) return;

    const int total   = checked_int(batch_size * grid_size * grid_width);
    const int channels = checked_int(4 + classes_number);
    OPENNN_CUDA_LAUNCH(detection_v8_forward_kernel<<<grid_size_for(total), block_size, 0,
                               opennn::device::get_compute_stream()>>>(
        checked_int(batch_size), checked_int(grid_size), checked_int(grid_width),
        channels, input, output));
}

__global__ void detection_v8_backward_kernel(const int batch_size,
                                             const int grid_size,
                                             const int grid_width,
                                             const int channels,
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

        for (int ch = 0; ch < channels; ++ch)
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
                                const float* output,
                                const float* output_delta,
                                float* input_delta)
{
    if (batch_size == 0 || grid_size == 0) return;

    const int total   = checked_int(batch_size * grid_size * grid_width);
    const int channels = checked_int(4 + classes_number);
    OPENNN_CUDA_LAUNCH(detection_v8_backward_kernel<<<grid_size_for(total), block_size, 0,
                                opennn::device::get_compute_stream()>>>(
        checked_int(batch_size), checked_int(grid_size), checked_int(grid_width),
        channels, output, output_delta, input_delta));
}

// ── Nearest-neighbor upsample (NHWC layout) ───────────────────────────────────

__global__ void upsample_forward_kernel(
    const int n,               // total output elements = batch * out_h * out_w * channels
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
    const int n,               // total input elements = batch * in_h * in_w * channels
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
    if (n == 0) return;
    OPENNN_CUDA_LAUNCH(upsample_forward_kernel<<<grid_size_for(n), block_size, 0,
        opennn::device::get_compute_stream()>>>(
        n, src, dst, in_h, in_w, in_h * scale, in_w * scale, channels, scale));
}

void upsample_backward_cuda(const int batch, const int in_h, const int in_w, const int channels, const int scale,
                            const float* out_delta, float* in_delta)
{
    const int n = batch * in_h * in_w * channels;
    if (n == 0) return;
    cudaMemsetAsync(in_delta, 0, size_t(n) * sizeof(float), opennn::device::get_compute_stream());
    OPENNN_CUDA_LAUNCH(upsample_backward_kernel<<<grid_size_for(n), block_size, 0,
        opennn::device::get_compute_stream()>>>(
        n, out_delta, in_delta, in_h, in_w, in_h * scale, in_w * scale, channels, scale));
}

// ── Channel concatenation (NHWC) ─────────────────────────────────────────────
// One call per input slice; each thread handles one element of that slice.

__global__ void concat_forward_slice_kernel(
    const int n,                        // batch * H * W * slice_ch
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
        dst[((b * H + h) * W + w) * total_ch + ch_offset + c] = src[i];
    }
}

__global__ void concat_backward_slice_kernel(
    const int n,
    const float* __restrict__ out_delta,
    float* __restrict__ in_delta,
    const int H, const int W,
    const int slice_ch, const int total_ch, const int ch_offset)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int c  = i % slice_ch;
        const int w  = (i / slice_ch) % W;
        const int h  = (i / slice_ch / W) % H;
        const int b  =  i / slice_ch / W / H;
        in_delta[i] = out_delta[((b * H + h) * W + w) * total_ch + ch_offset + c];
    }
}

void concat_forward_slice_cuda(const int batch, const int H, const int W,
                               const int slice_ch, const int total_ch, const int ch_offset,
                               const float* src, float* dst)
{
    const int n = batch * H * W * slice_ch;
    if (n == 0) return;
    OPENNN_CUDA_LAUNCH(concat_forward_slice_kernel<<<grid_size_for(n), block_size, 0,
        opennn::device::get_compute_stream()>>>(
        n, src, dst, H, W, slice_ch, total_ch, ch_offset));
}

void concat_backward_slice_cuda(const int batch, const int H, const int W,
                                const int slice_ch, const int total_ch, const int ch_offset,
                                const float* out_delta, float* in_delta)
{
    const int n = batch * H * W * slice_ch;
    if (n == 0) return;
    OPENNN_CUDA_LAUNCH(concat_backward_slice_kernel<<<grid_size_for(n), block_size, 0,
        opennn::device::get_compute_stream()>>>(
        n, out_delta, in_delta, H, W, slice_ch, total_ch, ch_offset));
}

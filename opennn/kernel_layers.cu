// Layer-op kernels: bounding, scaling/unscaling, embedding, multi-head
// attention helpers (split/merge heads, padding mask, fused mask), 3D pooling
// (max/avg, forward/backward), and layer normalisation. All host wrappers are
// templated over T = float / __nv_bfloat16.

#include "kernel_common.cuh"

// Per-feature clip: output[i] = clamp(input[i], lower[f], upper[f]) where f = i % features.
template<typename T>
__global__ void bounding_kernel(const int n, const int features,
                                const T* __restrict__ input,
                                const float* __restrict__ lower,
                                const float* __restrict__ upper,
                                T* __restrict__ output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int f = i % features;
        const float x = static_cast<float>(input[i]);
        output[i] = static_cast<T>(fminf(fmaxf(x, lower[f]), upper[f]));
    }
}

template<typename T>
void bounding_cuda(const Index n, const int features,
                   const T* input, const float* lower, const float* upper,
                   T* output)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    bounding_kernel<T><<<grid_size_for(total), block_size>>>(total, features, input, lower, upper, output);
}

template void bounding_cuda<float>        (const Index, const int, const float*,         const float*, const float*, float*);
template void bounding_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, const float*, __nv_bfloat16*);

#define SCALER_EPSILON 1e-7f

// Per-feature input scaling. Method per feature is encoded in `scalers` (cast from
// ScalerMethod enum stored as float): MinMax, MeanStd, StdDev, Logarithm, ImageMinMax.
template<typename T>
__global__ void scale_kernel(const int n, const int features,
                             const T* __restrict__ input,
                             const float* __restrict__ minimums,
                             const float* __restrict__ maximums,
                             const float* __restrict__ means,
                             const float* __restrict__ stds,
                             const float* __restrict__ scalers,
                             const float min_range, const float max_range,
                             T* __restrict__ output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int f = i % features;
        const int code = static_cast<int>(scalers[f]);
        const float x = static_cast<float>(input[i]);
        float y = x;

        switch(code)
        {
        case 1: // MinimumMaximum
            y = (x - minimums[f]) / ((maximums[f] - minimums[f]) + SCALER_EPSILON)
                * (max_range - min_range) + min_range;
            break;
        case 2: // MeanStandardDeviation
            y = (x - means[f]) / (stds[f] + SCALER_EPSILON);
            break;
        case 3: // StandardDeviation
            y = x / (stds[f] + SCALER_EPSILON);
            break;
        case 4: // Logarithm
            y = logf(x);
            break;
        case 5: // ImageMinMax
            y = x / 255.0f;
            break;
        default:
            break;
        }

        output[i] = static_cast<T>(y);
    }
}

template<typename T>
void scale_cuda(const Index n, const int features,
                const T* input,
                const float* minimums, const float* maximums,
                const float* means, const float* stds,
                const float* scalers,
                float min_range, float max_range,
                T* output)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    scale_kernel<T><<<grid_size_for(total), block_size>>>(total, features,
                                                          input, minimums, maximums, means, stds, scalers,
                                                          min_range, max_range, output);
}

template void scale_cuda<float>        (const Index, const int, const float*,         const float*, const float*, const float*, const float*, const float*, float, float, float*);
template void scale_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, const float*, const float*, const float*, const float*, float, float, __nv_bfloat16*);

// Inverse of scale_kernel: per-feature output unscaling.
template<typename T>
__global__ void unscale_kernel(const int n, const int features,
                               const T* __restrict__ input,
                               const float* __restrict__ minimums,
                               const float* __restrict__ maximums,
                               const float* __restrict__ means,
                               const float* __restrict__ stds,
                               const float* __restrict__ scalers,
                               const float min_range, const float max_range,
                               T* __restrict__ output)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int f = i % features;
        const int code = static_cast<int>(scalers[f]);
        const float x = static_cast<float>(input[i]);
        float y = x;

        switch(code)
        {
        case 1: // MinimumMaximum
            y = (x - min_range) / (max_range - min_range)
                * (maximums[f] - minimums[f]) + minimums[f];
            break;
        case 2: // MeanStandardDeviation
            y = means[f] + x * stds[f];
            break;
        case 3: // StandardDeviation
            y = x * stds[f];
            break;
        case 4: // Logarithm
            y = expf(x);
            break;
        case 5: // ImageMinMax
            y = x * 255.0f;
            break;
        default:
            break;
        }

        output[i] = static_cast<T>(y);
    }
}

template<typename T>
void unscale_cuda(const Index n, const int features,
                  const T* input,
                  const float* minimums, const float* maximums,
                  const float* means, const float* stds,
                  const float* scalers,
                  float min_range, float max_range,
                  T* output)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    unscale_kernel<T><<<grid_size_for(total), block_size>>>(total, features,
                                                            input, minimums, maximums, means, stds, scalers,
                                                            min_range, max_range, output);
}

template void unscale_cuda<float>        (const Index, const int, const float*,         const float*, const float*, const float*, const float*, const float*, float, float, float*);
template void unscale_cuda<__nv_bfloat16>(const Index, const int, const __nv_bfloat16*, const float*, const float*, const float*, const float*, const float*, float, float, __nv_bfloat16*);

// Token embedding lookup: outputs[i] = scale * weights[token_id, dim] + positional_encoding.
// `inputs` carries integer token ids stored as float. Out-of-vocab tokens get 0.
// `scale_embedding` applies sqrt(D) scaling (Transformer convention).
template<typename T>
__global__ void embedding_forward_kernel(const int n, const float* __restrict__ inputs, const float* __restrict__ weights, const float* __restrict__ positional_encoding, T* __restrict__ outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding, const bool add_positional_encoding)
{
    const float scale = scale_embedding ? sqrtf(static_cast<float>(embedding_dimension)) : 1.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int token_idx = i / embedding_dimension;
        const int dim_idx = i % embedding_dimension;
        const int token_id = static_cast<int>(inputs[token_idx]);

        float val = (token_id >= 0 && token_id < vocabulary_size)
            ? scale * weights[token_id * embedding_dimension + dim_idx]
            : 0.0f;

        if (add_positional_encoding && positional_encoding != nullptr && token_id > 0)
        {
            const int seq_idx = token_idx % sequence_length;
            val += positional_encoding[seq_idx * embedding_dimension + dim_idx];
        }

        outputs[i] = static_cast<T>(val);
    }
}

template<typename T>
void embedding_forward_cuda(const Index n, const float* inputs, const float* weights, const float* positional_encoding, T* outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding, const bool add_positional_encoding)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    embedding_forward_kernel<T><<<grid_size_for(total), block_size>>>(
        total, inputs, weights, positional_encoding, outputs,
        sequence_length, embedding_dimension, vocabulary_size, scale_embedding, add_positional_encoding);
}

template void embedding_forward_cuda<float>        (const Index, const float*, const float*, const float*, float*,         const int, const int, const int, const bool, const bool);
template void embedding_forward_cuda<__nv_bfloat16>(const Index, const float*, const float*, const float*, __nv_bfloat16*, const int, const int, const int, const bool, const bool);

// Embedding weight gradient via atomicAdd into the vocabulary table.
// Padding tokens (id 0 or out-of-range) contribute nothing.
template<typename T>
__global__ void embedding_backward_kernel(const int n, const float* __restrict__ inputs, const T* __restrict__ output_deltas, float* __restrict__ weight_gradients, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    const float scale = scale_embedding ? sqrtf(static_cast<float>(embedding_dimension)) : 1.0f;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int token_idx = i / embedding_dimension;
        const int dim_idx = i % embedding_dimension;
        const int token_id = static_cast<int>(inputs[token_idx]);

        if (token_id <= 0 || token_id >= vocabulary_size) continue;

        atomicAdd(&weight_gradients[token_id * embedding_dimension + dim_idx], scale * static_cast<float>(output_deltas[i]));
    }
}

template<typename T>
void embedding_backward_cuda(const Index n, const float* inputs, const T* output_deltas, float* weight_gradients, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    embedding_backward_kernel<T><<<grid_size_for(total), block_size>>>(
        total, inputs, output_deltas, weight_gradients,
        embedding_dimension, vocabulary_size, scale_embedding);
}

template void embedding_backward_cuda<float>        (const Index, const float*, const float*,         float*, const int, const int, const bool);
template void embedding_backward_cuda<__nv_bfloat16>(const Index, const float*, const __nv_bfloat16*, float*, const int, const int, const bool);

// [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim], scalar path.
template<typename T>
__global__ void split_heads_scalar_kernel(const int n, const T* __restrict__ in, T* __restrict__ out, const int S, const int H, const int D)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int d = i % D;
        const int h = (i / D) % H;
        const int s = (i / (D * H)) % S;
        const int b = i / (D * H * S);

        out[((b * H + h) * S + s) * D + d] = in[i];
    }
}

// split_heads using float4 lanes when head_dim*sizeof(T) is a multiple of 16 bytes.
// D_vec = head_dim / vec_width.
template<typename T>
__global__ void split_heads_vec_kernel(const int n_vec, const T* __restrict__ in, T* __restrict__ out, const int S, const int H, const int D_vec)
{
    const float4* const in_v  = reinterpret_cast<const float4*>(in);
    float4* const       out_v = reinterpret_cast<float4*>(out);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += blockDim.x * gridDim.x)
    {
        const int d = i % D_vec;
        const int h = (i / D_vec) % H;
        const int s = (i / (D_vec * H)) % S;
        const int b = i / (D_vec * H * S);

        out_v[((b * H + h) * S + s) * D_vec + d] = in_v[i];
    }
}

template<typename T>
void split_heads_cuda(const Index n, const T* in, T* out, const int S, const int H, const int D)
{
    if (n == 0) return;

    if ((static_cast<size_t>(D) * sizeof(T)) % 16 == 0)
    {
        const int vec_width = static_cast<int>(16 / sizeof(T));
        const int D_vec     = D / vec_width;
        const int n_vec     = static_cast<int>(n) / vec_width;
        split_heads_vec_kernel<T><<<grid_size_for(n_vec), block_size>>>(n_vec, in, out, S, H, D_vec);
    }
    else
    {
        const int total = static_cast<int>(n);
        split_heads_scalar_kernel<T><<<grid_size_for(total), block_size>>>(total, in, out, S, H, D);
    }
}

template void split_heads_cuda<float>        (const Index, const float*,         float*,         const int, const int, const int);
template void split_heads_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const int, const int, const int);

// Inverse of split_heads: [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim].
template<typename T>
__global__ void merge_heads_scalar_kernel(const int n, const T* __restrict__ in, T* __restrict__ out, const int S, const int H, const int D)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        const int d = i % D;
        const int s = (i / D) % S;
        const int h = (i / (D * S)) % H;
        const int b = i / (D * S * H);

        out[((b * S + s) * H + h) * D + d] = in[i];
    }
}

// merge_heads using float4 lanes when head_dim*sizeof(T) is a multiple of 16 bytes.
template<typename T>
__global__ void merge_heads_vec_kernel(const int n_vec, const T* __restrict__ in, T* __restrict__ out, const int S, const int H, const int D_vec)
{
    const float4* const in_v  = reinterpret_cast<const float4*>(in);
    float4* const       out_v = reinterpret_cast<float4*>(out);

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += blockDim.x * gridDim.x)
    {
        const int d = i % D_vec;
        const int s = (i / D_vec) % S;
        const int h = (i / (D_vec * S)) % H;
        const int b = i / (D_vec * S * H);

        out_v[((b * S + s) * H + h) * D_vec + d] = in_v[i];
    }
}

template<typename T>
void merge_heads_cuda(const Index n, const T* in, T* out, const int S, const int H, const int D)
{
    if (n == 0) return;

    if ((static_cast<size_t>(D) * sizeof(T)) % 16 == 0)
    {
        const int vec_width = static_cast<int>(16 / sizeof(T));
        const int D_vec     = D / vec_width;
        const int n_vec     = static_cast<int>(n) / vec_width;
        merge_heads_vec_kernel<T><<<grid_size_for(n_vec), block_size>>>(n_vec, in, out, S, H, D_vec);
    }
    else
    {
        const int total = static_cast<int>(n);
        merge_heads_scalar_kernel<T><<<grid_size_for(total), block_size>>>(total, in, out, S, H, D);
    }
}

template void merge_heads_cuda<float>        (const Index, const float*,         float*,         const int, const int, const int);
template void merge_heads_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const int, const int, const int);

// Per-token validity for attention: padding_mask[i] = 1 if all embedding dims of
// token i are ~0 (treated as PAD), else 0.
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

// Apply padding + optional causal mask to pre-softmax attention weights by
// setting masked positions to -1e9 (so softmax produces ~0). Operates on
// [batch, heads, seq_q, seq_k] flattened to n.
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

        if((use_causal_mask && sk > sq) || static_cast<float>(padding_mask[b * source_sequence_length + sk]) > 0.5f)
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
    if(num_tokens > 0)
        padding_mask_kernel<T><<<grid_size_for(num_tokens), block_size>>>(
            num_tokens, source_input, padding_mask, embedding_dimension);

    const int n = batch_size * heads_number * query_sequence_length * source_sequence_length;
    if(n > 0)
        fused_masks_kernel<T><<<grid_size_for(n), block_size>>>(
            n, attention_weights, padding_mask, heads_number,
            query_sequence_length, source_sequence_length, use_causal_mask);
}

template void attention_masks_cuda<float>        (int, int, int, int, int, const float*,         float*,         float*,         bool);
template void attention_masks_cuda<__nv_bfloat16>(int, int, int, int, int, const __nv_bfloat16*, __nv_bfloat16*, __nv_bfloat16*, bool);

// Max pooling over the seq axis of [batch, seq, features]. Saves the argmax
// position per (batch, feature) into `indices` for the backward pass.
template<typename T>
__global__ void max_pooling_3d_forward_kernel(const int n, const T* __restrict__ in, T* __restrict__ out, float* __restrict__ indices, const int S, const int F)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const int f = idx % F;
        const int b = idx / F;

        float max_val = -1e20f;
        int max_idx = 0;

        for (int s = 0; s < S; ++s)
        {
            const float val = static_cast<float>(in[(b * S + s) * F + f]);
            if (val > max_val) { max_val = val; max_idx = s; }
        }

        out[idx] = static_cast<T>(max_val);
        if (indices != nullptr) indices[idx] = static_cast<float>(max_idx);
    }
}

template<typename T>
void max_pooling_3d_forward_cuda(const Index n, const T* in, T* out, float* indices, const int S, const int F)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    max_pooling_3d_forward_kernel<T><<<grid_size_for(total), block_size>>>(total, in, out, indices, S, F);
}

template void max_pooling_3d_forward_cuda<float>        (const Index, const float*,         float*,         float*, const int, const int);
template void max_pooling_3d_forward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, float*, const int, const int);

// Scatter delta back to argmax positions saved by max_pooling_3d_forward.
// Caller must zero-initialise in_gradient first (host does cudaMemset).
template<typename T>
__global__ void max_pooling_3d_backward_kernel(const int n, const T* __restrict__ delta, T* __restrict__ in_gradient, const float* __restrict__ indices, const int S, const int F)
{
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        const int f = idx % F;
        const int b = idx / F;
        const int max_s = static_cast<int>(indices[idx]);

        in_gradient[(b * S + max_s) * F + f] = delta[idx];
    }
}

template<typename T>
void max_pooling_3d_backward_cuda(const Index n, const T* delta, T* in_gradient, const float* indices, const int S, const int F)
{
    if (n == 0) return;

    const int total = static_cast<int>(n);

    max_pooling_3d_backward_kernel<T><<<grid_size_for(total), block_size>>>(total, delta, in_gradient, indices, S, F);
}

template void max_pooling_3d_backward_cuda<float>        (const Index, const float*,         float*,         const float*, const int, const int);
template void max_pooling_3d_backward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const float*, const int, const int);

// Per-TU pooling scratch. Sized lazily, never freed (process-lifetime allocation).
namespace { float* pooling_scratch_ = nullptr; size_t pooling_scratch_size_ = 0; }

static float* get_pooling_scratch(size_t floats_needed)
{
    if (floats_needed > pooling_scratch_size_)
    {
        if (pooling_scratch_) cudaFree(pooling_scratch_);
        cudaMalloc(&pooling_scratch_, floats_needed * sizeof(float));
        pooling_scratch_size_ = floats_needed;
    }
    return pooling_scratch_;
}

// Helper for average pooling: writes per-token validity (1 if any feature is
// ~nonzero, else 0) and accumulates per-batch valid counts via atomicAdd.
template<typename T>
__global__ void pooling_3d_valid_mask_kernel(const int BS, const int S, const int F,
                                             const T* __restrict__ in,
                                             float* __restrict__ valid_mask,
                                             float* __restrict__ counts)
{
    const int bs = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs >= BS) return;

    const T* token = in + bs * F;
    bool valid = false;
    for (int f = 0; f < F; ++f)
        if (fabsf(static_cast<float>(token[f])) > 1e-7f) { valid = true; break; }

    valid_mask[bs] = valid ? 1.0f : 0.0f;
    if (valid) atomicAdd(&counts[bs / S], 1.0f);
}

// Average pooling over the seq axis, masking invalid (~zero) tokens. Output
// is sum-of-valid-tokens / valid_count; zero rows when no valid tokens.
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
            const int bs = b * S + s;
            sum += valid_mask[bs] * static_cast<float>(in[bs * F + f]);
        }

        out[idx] = static_cast<T>(sum / count);
    }
}

template<typename T>
void average_pooling_3d_forward_cuda(const Index n, const T* in, T* out, const int S, const int F)
{
    if (n == 0) return;

    const int B  = static_cast<int>(n) / F;
    const int BS = B * S;

    float* scratch    = get_pooling_scratch(static_cast<size_t>(BS) + B);
    float* valid_mask = scratch;
    float* counts     = scratch + BS;
    cudaMemset(counts, 0, B * sizeof(float));

    pooling_3d_valid_mask_kernel<T><<<grid_size_for(BS), block_size>>>(BS, S, F, in, valid_mask, counts);

    const int total = static_cast<int>(n);
    average_pooling_3d_forward_kernel<T><<<grid_size_for(total), block_size>>>(total, in, out, S, F, valid_mask, counts);
}

template void average_pooling_3d_forward_cuda<float>        (const Index, const float*,         float*,         const int, const int);
template void average_pooling_3d_forward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, __nv_bfloat16*, const int, const int);

// Distribute output delta uniformly across the batch's valid tokens
// (delta_per_token = delta / valid_count, zero where invalid).
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
            const int bs = b * S + s;
            in_gradient[bs * F + f] = static_cast<T>(valid_mask[bs] * gradient_val);
        }
    }
}

template<typename T>
void average_pooling_3d_backward_cuda(const Index n, const T* in, const T* delta, T* in_gradient, const int S, const int F)
{
    if (n == 0) return;

    const int B  = static_cast<int>(n) / F;
    const int BS = B * S;

    float* scratch    = get_pooling_scratch(static_cast<size_t>(BS) + B);
    float* valid_mask = scratch;
    float* counts     = scratch + BS;
    cudaMemset(counts, 0, B * sizeof(float));

    pooling_3d_valid_mask_kernel<T><<<grid_size_for(BS), block_size>>>(BS, S, F, in, valid_mask, counts);

    const int total = static_cast<int>(n);
    average_pooling_3d_backward_kernel<T><<<grid_size_for(total), block_size>>>(total, delta, in_gradient, S, F, valid_mask, counts);
}

template void average_pooling_3d_backward_cuda<float>        (const Index, const float*,         const float*,         float*,         const int, const int);
template void average_pooling_3d_backward_cuda<__nv_bfloat16>(const Index, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, const int, const int);

// Two-element warp-stride reduction. Used by layernorm forward/backward to fold
// (sum, sum_sq) and (sum_D, sum_D_xhat) pairs in a single shuffle pass.
__device__ __forceinline__ void warp_reduce_sum2(float& a, float& b)
{
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        a += __shfl_down_sync(0xffffffff, a, offset);
        b += __shfl_down_sync(0xffffffff, b, offset);
    }
}

// Per-row layer norm, one block per row of [N, D]: y = gamma*(x-mean)/sqrt(var+eps) + beta.
// Saves mean and 1/sqrt(var+eps) for the backward pass. Variance computed from the
// online moments E[X], E[X^2] reduced via warp_reduce_sum2.
template<typename T>
__global__ void layernorm_forward_kernel(const int N, const int D, const T* __restrict__ X, T* __restrict__ Y, float* __restrict__ means, float* __restrict__ inv_vars, const float* __restrict__ gamma, const float* __restrict__ beta, const float eps)
{
    const int idx = blockIdx.x;
    if (idx >= N) return;

    const T* x_row = X + idx * D;
    T* y_row = Y + idx * D;

    // Per-thread accumulation. Variance derived as E[X^2] - E[X]^2 after the block reduction.
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
            const float variance = s_sq * inv_D - mean * mean;
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

template<typename T>
void layernorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps)
{
    if (N == 0 || D == 0) return;

    int threads = 256;
    if (D <= 32) threads = 32;
    else if (D <= 64) threads = 64;
    else if (D <= 128) threads = 128;

    layernorm_forward_kernel<T><<<N, threads>>>(N, D, X, Y, means, inv_vars, gamma, beta, eps);
}

template void layernorm_forward_cuda<float>        (const int, const int, const float*,         float*,         float*, float*, const float*, const float*, const float);
template void layernorm_forward_cuda<__nv_bfloat16>(const int, const int, const __nv_bfloat16*, __nv_bfloat16*, float*, float*, const float*, const float*, const float);

// Per-row dX for layer norm, one block per row of [N, D]. Uses cached mean and
// inv_var saved by layernorm_forward_kernel.
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

// Per-feature dGamma/dBeta accumulation for layer norm, one block per feature
// (column of [N, D]). Reduces across rows via warp shuffle.
template<typename T>
__global__ void layernorm_gamma_beta_gradient_kernel(const int N, const int D, const T* __restrict__ dY, const T* __restrict__ X, const float* __restrict__ means, const float* __restrict__ inv_vars, float* __restrict__ dGamma, float* __restrict__ dBeta)
{
    const int d = blockIdx.x;
    if (d >= D) return;

    float local_gamma = 0.0f;
    float local_beta  = 0.0f;

    for (int n = threadIdx.x; n < N; n += blockDim.x)
    {
        const float dy    = static_cast<float>(dY[n * D + d]);
        const float x_hat = (static_cast<float>(X[n * D + d]) - means[n]) * inv_vars[n];
        local_gamma += dy * x_hat;
        local_beta  += dy;
    }

    warp_reduce_sum2(local_gamma, local_beta);

    __shared__ float warp_gamma[32];
    __shared__ float warp_beta[32];

    const int lane    = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0)
    {
        warp_gamma[warp_id] = local_gamma;
        warp_beta[warp_id]  = local_beta;
    }
    __syncthreads();

    const int num_warps = (blockDim.x + 31) >> 5;
    if (warp_id == 0)
    {
        float g = (threadIdx.x < num_warps) ? warp_gamma[threadIdx.x] : 0.0f;
        float b = (threadIdx.x < num_warps) ? warp_beta[threadIdx.x]  : 0.0f;
        warp_reduce_sum2(g, b);

        if (threadIdx.x == 0)
        {
            dGamma[d] = g;
            dBeta[d]  = b;
        }
    }
}

template<typename T>
void layernorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* means, const float* inv_vars, const float* gamma, T* dX, float* dGamma, float* dBeta)
{
    if (N == 0 || D == 0) return;

    int dx_threads = 256;
    if (D <= 32) dx_threads = 32;
    else if (D <= 64) dx_threads = 64;
    else if (D <= 128) dx_threads = 128;

    layernorm_backward_kernel<T><<<N, dx_threads>>>(N, D, dY, X, means, inv_vars, gamma, dX);

    const int gb_threads = 256;
    layernorm_gamma_beta_gradient_kernel<T><<<D, gb_threads>>>(N, D, dY, X, means, inv_vars, dGamma, dBeta);
}

template void layernorm_backward_cuda<float>        (const int, const int, const float*,         const float*,         const float*, const float*, const float*, float*,         float*, float*);
template void layernorm_backward_cuda<__nv_bfloat16>(const int, const int, const __nv_bfloat16*, const __nv_bfloat16*, const float*, const float*, const float*, __nv_bfloat16*, float*, float*);

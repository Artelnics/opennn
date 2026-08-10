//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A T T E N T I O N   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// attention: heads, masks, softmax, rotary and decode sampling

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/core/cuda/kernel_attention.cuh"
#include <curand_kernel.h>
#include <cub/block/block_reduce.cuh>

constexpr int SAMPLING_BLOCK_THREADS = 256;
using BlockArgMaxReduce = cub::BlockReduce<cub::KeyValuePair<int, float>, SAMPLING_BLOCK_THREADS>;

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

#define INSTANTIATE(T) \
    template void split_heads_cuda<T>(const Index, const T*, T*, const int, const int, const int); \
    template void merge_heads_cuda<T>(const Index, const T*, T*, const int, const int, const int); \
    template void attention_masked_softmax_cuda<T>(int, int, int, int, int, const T*, T*, T*, bool, bool); \
    template void attention_length_masked_softmax_cuda<T>(int, int, int, int, const int*, T*, T*, bool, bool); \
    template void attention_sequence_lengths_cuda<T>(int, int, int, int, const T*, int32_t*, int32_t*); \
    template void rope_forward_cuda<T>(const int, const int, const int, const int, const int, const int, const T*, T*, const float*, const float*); \
    template void rope_backward_cuda<T>(const int, const int, const int, const int, const int, const int, const T*, T*, const float*, const float*); \
    template void qk_rope_cache_append_cuda<T>(const int, const int, const int, const float, const int*, const T*, const float*, const float*, const float*, const float*, T*, T*, T*); \
    template void sample_logits_row_cuda<T>(const int, const float, const int, const float, const unsigned long long, const unsigned long long, const T*, float2*, int*, float*); \
    template void grouped_attention_cuda<T>(const int, const int, const int, const int, const int, const int, const float, const int, const bool, const int*, float*, const T*, const T*, const T*, T*); \
    template void grouped_attention_softmax_cuda<T>(const int, const int, const int, const int, const bool, const float*, T*);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.

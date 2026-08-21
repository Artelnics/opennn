#ifndef KERNEL_ATTENTION_CUH
#define KERNEL_ATTENTION_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

template<typename T>
void split_heads_cuda(const Index n, const T* in, T* out, const int S, const int H, const int D);

template<typename T>
void concatenate_heads_cuda(const Index n, const T* in, T* out, const int S, const int H, const int D);

template<typename T>
void attention_masked_softmax_cuda(const int batch_size, const int heads_number, const int query_sequence_length,
                          const int source_sequence_length, const int embedding_dimension,
                          const T* source_input, T* attention_weights, T* padding_mask,
                          const bool use_causal_mask, const bool zero_padded_queries);

template<typename T>
void attention_length_masked_softmax_cuda(const int batch_size, const int heads_number, const int query_sequence_length,
                                const int source_sequence_length, const int* device_lengths,
                                T* attention_weights, T* padding_mask, const bool use_causal_mask,
                                const bool zero_padded_queries);

template<typename T>
void attention_sequence_lengths_cuda(const int batch_size,
                                     const int query_sequence_length,
                                     const int source_sequence_length,
                                     const int embedding_dimension,
                                     const T* source_input,
                                     int32_t* query_lengths,
                                     int32_t* source_lengths);

// SDPA length tensors from an exported record (see kernel).
void attention_sdpa_lengths_cuda(const int batch_size, const int query_sequence_length,
                                 const int source_sequence_length, const int* record,
                                 int32_t* query_lengths, int32_t* source_lengths);

template<typename T>
void rope_forward_cuda(const int rows, const int seq, const int model_dim, const int head_dim, const int rotary_dim, const int offset, const T* in, T* out, const float* cos, const float* sin);

template<typename T>
void rope_backward_cuda(const int rows, const int seq, const int model_dim, const int head_dim, const int rotary_dim, const int offset, const T* dout, T* din, const float* cos, const float* sin);

inline constexpr int GROUPED_ATTENTION_DECODE_SPLITS = 128;

constexpr bool grouped_attention_decode_supported(const int head_dim, const int group)
{
    const bool dim_ok = head_dim == 64 || head_dim == 128 || head_dim == 256;
    const bool group_ok = group == 1 || group == 2 || group == 4 || group == 8;
    return dim_ok && group_ok && group * head_dim <= 1024;
}

template<typename T>
void grouped_attention_cuda(const int batch, const int query_seq, const int key_seq,
                            const int n_query_heads, const int n_kv_heads, const int head_dim,
                            const float scale, const int query_position_offset, const bool causal,
                            const int* position_device, float* decode_partials,
                            const T* Q, const T* K, const T* V, T* O);

template<typename T>
void grouped_attention_softmax_cuda(const int rows, const int query_seq, const int key_seq,
                                    const int query_position_offset, const bool causal,
                                    const float* scores, T* probs);

template<typename T>
void qk_rope_cache_append_cuda(const int n_q_heads, const int n_kv_heads, const int head_dim,
                               const float eps, const int* position,
                               const T* qkv, const float* q_norm_w, const float* k_norm_w,
                               const float* cos_table, const float* sin_table,
                               T* q_out, T* k_cache, T* v_cache);

inline constexpr int LOGITS_SAMPLE_BLOCKS = 128;

template<typename T>
void sample_logits_row_cuda(const int n, const float temperature, const int top_k, const float top_p,
                            const unsigned long long seed, const unsigned long long step,
                            const T* logits, float2* candidates_scratch, int* id_out, float* token_out);

#endif

#endif

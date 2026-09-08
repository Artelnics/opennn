#ifndef KERNEL_EMBEDDING_CUH
#define KERNEL_EMBEDDING_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

template<typename TW, typename T>
void embedding_forward_cuda(const Index n, const float* inputs, const TW* weights, const float* positional_encoding, T* outputs, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding);

template<typename T>
void embedding_backward_cuda(const Index n, const float* inputs, const T* output_deltas, float* weight_gradients, float* positional_gradients, const int sequence_length, const int embedding_dimension, const int vocabulary_size, const bool scale_embedding);

void token_valid_lengths_cuda(const Index batch_size, const Index sequence_length,
                              const float* token_ids, int* lengths, cudaStream_t stream);

#endif

#endif

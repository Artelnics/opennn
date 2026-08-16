#ifndef KERNEL_QUANTIZATION_CUH
#define KERNEL_QUANTIZATION_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

inline constexpr int W8A16_MAX_M = 16;

template<typename T>
void w8a16_linear_cuda(const int m, const int in_features, const int out_features,
                       const bool weights_out_major,
                       const T* x, const int8_t* w, const float* scales,
                       const T* bias, T* y);

template<typename T>
void w8_dequant_cuda(const Index rows, const Index row_length, const bool scale_by_row,
                     const int8_t* q, const float* scales, T* out);

template<typename T>
void embedding_forward_w8_cuda(const Index n, const float* inputs, const int8_t* weights,
                               const float* weight_scales, const float* positional_encoding,
                               T* outputs, const int sequence_length,
                               const int embedding_dimension, const int vocabulary_size,
                               const bool scale_embedding);

#endif

#endif

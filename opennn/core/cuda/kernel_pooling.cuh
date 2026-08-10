#ifndef KERNEL_POOLING_CUH
#define KERNEL_POOLING_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

template<typename T>
void max_pooling_3d_forward_cuda(const Index n, const T* in, T* out, float* indices, const int S, const int F);

template<typename T>
void max_pooling_3d_backward_cuda(const Index n, const T* delta, T* in_grad, const float* indices, const int S, const int F);

template<typename T>
void average_pooling_3d_forward_cuda(const Index n, const T* in, T* out, const int S, const int F);

template<typename T>
void average_pooling_3d_backward_cuda(const Index n, const T* in, const T* delta, T* in_grad, const int S, const int F);

template<typename T>
void first_token_3d_forward_cuda(const int B, const int S, const int F, const T* in, T* out);

template<typename T>
void first_token_3d_backward_cuda(const int B, const int S, const int F, const T* delta, T* in_gradient);

void upsample_forward_cuda(int batch, int in_h, int in_w, int channels, int scale,
                           const float* src, float* dst);
void upsample_backward_cuda(int batch, int in_h, int in_w, int channels, int scale,
                            const float* out_delta, float* in_delta);

void concat_forward_slice_cuda(int batch, int H, int W,
                               int slice_ch, int total_ch, int ch_offset,
                               const float* src, float* dst);
void concat_backward_slice_cuda(int batch, int H, int W,
                                int slice_ch, int total_ch, int ch_offset,
                                const float* out_delta, float* in_delta);

#endif

#endif

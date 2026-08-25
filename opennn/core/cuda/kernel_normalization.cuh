#ifndef KERNEL_NORMALIZATION_CUH
#define KERNEL_NORMALIZATION_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

template<typename T>
void batchnorm_inference_cuda(const Index total, const Index channels,
                              const T* x, const T* residual,
                              const float* gamma, const float* beta,
                              const float* mean, const float* variance,
                              const float epsilon, const bool apply_relu, T* y);

Index batchnorm_partial_rows(const Index rows);

template<typename T>
void batchnorm_forward_fused_cuda(const Index rows, const Index channels,
                                  const T* x, const T* residual,
                                  const float* gamma, const float* beta,
                                  const float epsilon, const float momentum,
                                  T* y, float* mean, float* inv_var,
                                  float* running_mean, float* running_var,
                                  const bool relu, uint8_t* mask,
                                  float* partials);

template<typename T>
void batchnorm_backward_fused_cuda(const Index rows, const Index channels,
                                   const T* x, T* dy_dx, const T* y, const uint8_t* mask,
                                   const float* gamma, const float* beta,
                                   const float* mean, const float* inv_var,
                                   const bool xhat_from_y,
                                   T* dpre, float* dgamma, float* dbeta,
                                   float* partials);

void conv_bn_fold_cuda(const Index kernels, const Index kernel_size,
                       const float* weights,
                       const float* gamma, const float* beta,
                       const float* mean, const float* variance,
                       const float epsilon,
                       float* folded_weights, float* folded_bias);

void add_relu_cuda(const Index total, const float* a, const float* b,
                   const bool apply_relu, float* y);

template<typename T>
void layernorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps);

template<typename T>
void layernorm_add_forward_cuda(const int N, const int D, const T* X, const T* R, T* sum, T* Y, float* means, float* inv_vars, const float* gamma, const float* beta, const float eps);

template<typename T>
void layernorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* means, const float* inv_vars, const float* gamma, T* dX, T* dX2, float* dGamma, float* dBeta);

template<typename T>
void rmsnorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* inv_rms, const float* weight, const float eps);

template<typename T>
void rmsnorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* inv_rms, const float* weight, T* dX, float* dWeight);

#endif

#endif

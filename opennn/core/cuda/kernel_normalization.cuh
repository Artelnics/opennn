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

// Fused batch-norm backward for NHWC activations, FP32 math on BF16 or FP32
// tensors: the ReLU mask taken from the saved output Y (skipped when y is
// null), the two per-channel reductions, dX written in place over dY, and -
// when dpre is given - the residual fork dPre = masked dY for the skip branch.
// Two launches over the tensor instead of the separate dReLU kernel, delta
// copy and staged casts a shape without a fused cuDNN engine used to pay.
// `partials` is scratch for 2 * batchnorm_backward_partial_rows(rows) *
// channels floats.
Index batchnorm_backward_partial_rows(const Index rows);

template<typename T>
void batchnorm_backward_fused_cuda(const Index rows, const Index channels,
                                   const T* x, T* dy_dx, const T* y,
                                   const float* gamma, const float* mean, const float* inv_var,
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
void layernorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* means, const float* inv_vars, const float* gamma, T* dX, float* dGamma, float* dBeta);

template<typename T>
void rmsnorm_forward_cuda(const int N, const int D, const T* X, T* Y, float* inv_rms, const float* weight, const float eps);

template<typename T>
void rmsnorm_backward_cuda(const int N, const int D, const T* dY, const T* X, const float* inv_rms, const float* weight, T* dX, float* dWeight);

#endif

#endif

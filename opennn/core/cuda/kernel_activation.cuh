#ifndef KERNEL_ACTIVATION_CUH
#define KERNEL_ACTIVATION_CUH

#ifdef OPENNN_HAS_CUDA

#include "opennn/core/cuda/kernel_prelude.cuh"

template<typename T>
void dropout_forward_cuda(const Index n, T* output, uint8_t* mask, const float rate, const unsigned long long* seed_state);

void advance_dropout_seed_cuda(unsigned long long* seed_state);

template<typename T>
void dropout_backward_cuda(const Index n, const T* output_delta, T* input_delta, const uint8_t* mask, const float rate);

template<typename T>
void activation_forward_cuda(const Index n, T* data, const int function);

template<typename T>
void activation_backward_cuda(const Index n, const T* outputs, T* delta, const int function);

template<typename T>
void swiglu_forward_cuda(const int n, const T* gate, const T* up, T* out);

template<typename T>
void swiglu_backward_cuda(const int n, const T* dout, const T* gate, const T* up, T* dgate, T* dup);

#endif

#endif

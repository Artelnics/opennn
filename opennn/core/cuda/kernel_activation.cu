//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A C T I V A T I O N   K E R N E L S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "opennn/core/cuda/kernel_common.cuh"
#include "opennn/core/cuda/kernel_activation.cuh"
#include <curand_kernel.h>

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
        const float s = sigmoid_f(y);
        return d * s * (1.0f + y * (1.0f - s));
    }
    return d;
}

template<typename T>
__global__ void swiglu_forward_kernel(const int n, const T* __restrict__ gate, const T* __restrict__ up, T* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float g = static_cast<float>(gate[i]);
    out[i] = static_cast<T>(opennn_activation_value(g, activation_silu) * static_cast<float>(up[i]));
}

template<typename T>
__global__ void swiglu_backward_kernel(const int n, const T* __restrict__ dout, const T* __restrict__ gate, const T* __restrict__ up, T* __restrict__ dgate, T* __restrict__ dup)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const float d = static_cast<float>(dout[i]);
    const float g = static_cast<float>(gate[i]);

    if (dup)   dup[i]   = static_cast<T>(d * opennn_activation_value(g, activation_silu));
    if (dgate) dgate[i] = static_cast<T>(opennn_activation_grad(g, d * static_cast<float>(up[i]), activation_silu));
}

template<typename T>
void swiglu_forward_cuda(const int n, const T* gate, const T* up, T* out)
{
    launch_elementwise(n, swiglu_forward_kernel<T>, gate, up, out);
}

template<typename T>
void swiglu_backward_cuda(const int n, const T* dout, const T* gate, const T* up, T* dgate, T* dup)
{
    launch_elementwise(n, swiglu_backward_kernel<T>, dout, gate, up, dgate, dup);
}

template<typename T, int VEC>
__global__ void activation_forward_kernel(const int n_vec, const int n,
                                          T* __restrict__ data, const int function)
{
    const Index tid = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = Index(blockDim.x) * gridDim.x;

    for (Index i = tid; i < n_vec; i += stride)
    {
        float v[VEC];
        VecIO<T, VEC>::load_float(data + i * VEC, v);

        #pragma unroll
        for (int k = 0; k < VEC; ++k) v[k] = opennn_activation_value(v[k], function);

        VecIO<T, VEC>::store_float(data + i * VEC, v);
    }

    for (Index i = Index(n_vec) * VEC + tid; i < n; i += stride)
        data[i] = static_cast<T>(opennn_activation_value(static_cast<float>(data[i]), function));
}

template<typename T>
void activation_forward_cuda(const Index n, T* data, const int function)
{
    launch_vec_on<vec16<T>>(opennn::device::get_compute_stream(), n,
                            are_aligned<16>(data),
                            activation_forward_kernel<T, vec16<T>>, data, function);
}

template<typename T, int VEC>
__global__ void activation_backward_kernel(const int n_vec, const int n,
                                           const T* __restrict__ outputs,
                                           T* __restrict__ delta, const int function)
{
    const Index tid = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    const Index stride = Index(blockDim.x) * gridDim.x;

    for (Index i = tid; i < n_vec; i += stride)
    {
        float y[VEC];
        float d[VEC];
        VecIO<T, VEC>::load_float(outputs + i * VEC, y);
        VecIO<T, VEC>::load_float(delta + i * VEC, d);

        #pragma unroll
        for (int k = 0; k < VEC; ++k) d[k] = opennn_activation_grad(y[k], d[k], function);

        VecIO<T, VEC>::store_float(delta + i * VEC, d);
    }

    for (Index i = Index(n_vec) * VEC + tid; i < n; i += stride)
        delta[i] = static_cast<T>(opennn_activation_grad(static_cast<float>(outputs[i]),
                                                         static_cast<float>(delta[i]), function));
}

template<typename T>
void activation_backward_cuda(const Index n, const T* outputs, T* delta, const int function)
{
    launch_vec_on<vec16<T>>(opennn::device::get_compute_stream(), n,
                            are_aligned<16>(outputs, delta),
                            activation_backward_kernel<T, vec16<T>>,
                            outputs, delta, function);
}

template<typename T>
__global__ void dropout_forward_kernel(
    int n, T* __restrict__ output, uint8_t* __restrict__ mask,
    float scale, float rate, const unsigned long long* __restrict__ seed_state)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    curandStatePhilox4_32_10_t state;
    curand_init(seed_state[0], idx, 0, &state);

    const uint8_t keep = static_cast<uint8_t>(curand_uniform(&state) >= rate);

    mask[idx] = keep;
    output[idx] = static_cast<T>(static_cast<float>(output[idx]) * keep * scale);
}

__global__ void advance_dropout_seed_kernel(unsigned long long* seed_state)
{
    seed_state[0] += 0x9E3779B97F4A7C15ull;
}

void advance_dropout_seed_cuda(unsigned long long* seed_state)
{
    launch_single(nullptr, advance_dropout_seed_kernel, seed_state);
}

template<typename T>
void dropout_forward_cuda(const Index n, T* output, uint8_t* mask, const float rate, const unsigned long long* seed_state)
{
    launch_elementwise(n, dropout_forward_kernel<T>, output, mask, 1.0f / (1.0f - rate), rate, seed_state);
}

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
    launch_elementwise(n, dropout_backward_kernel<T>, output_delta, input_delta, mask, 1.0f / (1.0f - rate));
}

#define INSTANTIATE(T) \
    template void swiglu_forward_cuda<T>(const int, const T*, const T*, T*); \
    template void swiglu_backward_cuda<T>(const int, const T*, const T*, const T*, T*, T*); \
    template void activation_forward_cuda<T>(const Index, T*, const int); \
    template void activation_backward_cuda<T>(const Index, const T*, T*, const int); \
    template void dropout_forward_cuda<T>(const Index, T*, uint8_t*, const float, const unsigned long long*); \
    template void dropout_backward_cuda<T>(const Index, const T*, T*, const uint8_t*, const float);

OPENNN_INSTANTIATE_FLOAT_BF16(INSTANTIATE)
#undef INSTANTIATE

// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2026 Artificial Intelligence, SL.
// Licensed under the GNU Lesser General Public License v2.1 or later.
